import asyncio
import os
import re
import subprocess
import time
import glob
from datetime import datetime
from typing import Any

from repodoc.config import Config
from repodoc.graph import DependencyGraphBuilder
from repodoc.graph.builder import build_graph_from_analysis, component_dependencies
from repodoc.agent.agent import DocAgent
from repodoc.utils import (
    content_hash,
    file_manager,
    log_operation,
    summarize_markdown,
    validate_and_fix_links,
)


def flatten_module_tree_leaf_first(
    tree: dict[str, Any], parent_path: str = ""
) -> list[tuple[str, dict[str, Any], str]]:
    """Flatten module tree into leaf-first ordered list.

    Returns list of (module_name, module_info, full_path) tuples.
    Leaves are returned before their parents.
    """
    result = []

    def _collect(
        nodes: dict[str, Any], path: str
    ) -> list[tuple[str, dict[str, Any], str]]:
        collected = []
        for name, info in nodes.items():
            full_path = f"{path}/{name}" if path else name
            children = info.get("children", {})
            if children:
                collected.extend(_collect(children, full_path))
            collected.append((name, info, full_path))
        return collected

    return _collect(tree, parent_path)


def group_modules_by_depth(
    tree: dict[str, Any], parent_path: str = ""
) -> dict[int, list[tuple[str, dict[str, Any], str]]]:
    """Group modules by depth level for parallel processing.

    Returns dict of depth -> list of (module_name, module_info, full_path) tuples.
    Depth 0 = leaf modules (no children), higher depths = parent modules.
    """
    depth_groups = {}

    def _collect(nodes: dict[str, Any], path: str, depth: int):
        for name, info in nodes.items():
            full_path = f"{path}/{name}" if path else name
            children = info.get("children", {})
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append((name, info, full_path))
            if children:
                _collect(children, full_path, depth + 1)

    _collect(tree, parent_path, 0)
    return depth_groups


def get_module_depth(
    tree: dict[str, Any], module_name: str, current_depth: int = 0
) -> int:
    """Get the depth of a module in the tree (0 = top-level)."""
    for name, info in tree.items():
        if name == module_name:
            return current_depth
        children = info.get("children", {})
        if children:
            depth = get_module_depth(children, module_name, current_depth + 1)
            if depth >= 0:
                return depth
    return -1


def _collect_component_modules(
    tree: dict[str, Any],
) -> dict[str, list[str]]:
    """Map each component id to its DEEPEST owning module chain in the tree.

    A component may appear in a parent module's full member list AND in a
    child module's sub-division; the deepest module is its true home, so its
    doc lands in the most specific directory."""
    best: dict[str, list[str]] = {}

    def walk(nodes: dict[str, Any], path: list[str]) -> None:
        for name, info in nodes.items():
            cur = path + [name]
            for cid in info.get("components", []):
                if cid not in best or len(cur) > len(best[cid]):
                    best[cid] = cur
            children = info.get("children")
            if isinstance(children, dict):
                walk(children, cur)

    walk(tree, [])
    return best


def _build_module_dependencies(
    components: dict[str, Any],
    module_tree: dict[str, Any],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Aggregate cross-module call edges.

    Returns (module -> modules it depends on, module -> modules that depend
    on it), keyed by the FULL module path ("core/services"), using each
    component's deepest module home. Callers documenting a specific module
    look it up by its full path; the README/overview uses
    _top_level_module_dependencies to fold everything onto top-level modules."""
    comp_modules = _collect_component_modules(module_tree)

    def module_of(cid: str) -> str | None:
        chain = comp_modules.get(cid)
        return "/".join(chain) if chain else None

    deps: dict[str, set[str]] = {}
    dependents: dict[str, set[str]] = {}
    for cid, node in components.items():
        m = module_of(cid)
        if m is None:
            continue
        for dep_id in component_dependencies(node):
            dm = module_of(dep_id)
            if dm and dm != m:
                deps.setdefault(m, set()).add(dm)
                dependents.setdefault(dm, set()).add(m)
    return deps, dependents


def _subtree_module_dependencies(
    deps: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Aggregate full-path module dependencies over each module's subtree.

    A parent module inherits the dependencies of every descendant module
    ("core" inherits "core/services" -> "net"); edges inside the same subtree
    are dropped (they are internal cooperation, not external dependency).
    Every ancestor of a known module is included, so a parent whose members
    all live in sub-modules still gets its subtree's dependencies.
    Used by module docs, which describe the whole subtree."""

    def _prefixes(module_id: str) -> list[str]:
        parts = module_id.split("/")
        return ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]

    modules: set[str] = set()
    for module, targets in deps.items():
        modules.update(_prefixes(module))
        for target in targets:
            modules.update(_prefixes(target))
    out: dict[str, set[str]] = {}
    for module in modules:
        prefix = module + "/"
        merged: set[str] = set()
        for owner, targets in deps.items():
            if owner != module and not owner.startswith(prefix):
                continue
            for target in targets:
                if target != module and not target.startswith(prefix):
                    merged.add(target)
        out[module] = merged
    return out


def _top_level_module_dependencies(
    deps: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Fold full-path module dependencies onto top-level modules.

    Used by the README overview, which documents top-level modules: both
    sides of every edge are reduced to their first path segment and
    self-edges are dropped."""
    out: dict[str, set[str]] = {}
    for module, targets in deps.items():
        top = module.split("/")[0]
        merged = out.setdefault(top, set())
        for target in targets:
            target_top = target.split("/")[0]
            if target_top != top:
                merged.add(target_top)
    return out


_MERMAID_BLOCK_RE = re.compile(
    r"```mermaid[ \t]*\r?\n.*?```", re.DOTALL
)

# Total character budget for all member/child doc summaries embedded in one
# parent prompt. A fixed per-doc budget (2000 chars x N members) overflowed
# the 60K prompt cap on real repos and the truncated tail made the module doc
# fail on every retry (gson: 30+ member docs).
DOCS_SUMMARY_TOTAL_BUDGET = 40_000
CHILDREN_SUMMARY_TOTAL_BUDGET = 12_000


def summarize_docs_for_prompt(
    contents: dict[str, str],
    total_budget: int = DOCS_SUMMARY_TOTAL_BUDGET,
    per_doc_budget: int = 2000,
) -> dict[str, str]:
    """Summarize each doc so the TOTAL stays under budget.

    The per-doc budget shrinks as more docs share the total, so a 60-member
    module still fits the prompt instead of being truncated."""
    if not contents:
        return {}
    per = min(per_doc_budget, max(200, total_budget // len(contents)))
    return {
        name: summarize_markdown(content, budget=per)
        for name, content in contents.items()
    }


def build_module_mermaid(
    module_deps: dict[str, set[str]],
    module_names: list[str] | None = None,
) -> str:
    """Deterministic mermaid flowchart of the module dependency map.

    The README diagram must reflect the actual knowledge graph; letting the
    LLM draw it produced invented modules/edges. Nodes are the given module
    names plus every endpoint of the dependency edges; labels keep the real
    names, mermaid ids are sanitized."""
    node_set: set[str] = set(module_names or [])
    node_set.update(module_deps.keys())
    for targets in module_deps.values():
        node_set.update(targets)
    if not node_set:
        return ""

    node_ids: dict[str, str] = {}

    def nid(name: str) -> str:
        if name not in node_ids:
            safe = re.sub(r"[^0-9A-Za-z_]", "_", name)
            node_ids[name] = f"m_{safe}" if safe else f"m_n{len(node_ids)}"
        return node_ids[name]

    lines = ["flowchart TD"]
    for name in sorted(node_set):
        label = name.replace('"', "'")
        lines.append(f'    {nid(name)}["{label}"]')
    edges = {
        (source, target)
        for source, targets in module_deps.items()
        for target in targets
        if target in node_set and target != source
    }
    for source, target in sorted(edges):
        lines.append(f"    {nid(source)} --> {nid(target)}")
    return "\n".join(lines)


def _enforce_overview_diagram(result, diagram_code: str) -> None:
    """Replace/insert the README's module diagram with the canonical one.

    Prompt instructions alone cannot guarantee a verbatim diagram; the first
    mermaid block is overwritten with the graph-derived one (or appended when
    missing). Runs BEFORE the DocNode snapshot so manual-edit detection stays
    accurate."""
    if not (
        result is not None
        and getattr(result, "success", False)
        and getattr(result, "output", None)
    ):
        return
    if not diagram_code:
        return
    output = result.output
    path = output.get("path")
    content = output.get("content", "")
    if not path or not content:
        return
    block = f"```mermaid\n{diagram_code}\n```"
    if _MERMAID_BLOCK_RE.search(content):
        updated = _MERMAID_BLOCK_RE.sub(
            lambda _m: block, content, count=1
        )
    else:
        updated = content.rstrip() + "\n\n## Architecture\n\n" + block + "\n"
    if updated == content:
        return
    try:
        file_manager.save_text(updated, path)
    except OSError as e:
        print(f"Warning: could not enforce the README diagram: {e}")
        return
    output["content"] = updated


def refresh_doc_snapshots(graph, paths: list[str]) -> int:
    """Adopt on-disk content for the given doc files into their graph DocNodes.

    validate_and_fix_links rewrites markdown files (broken links, invalid
    mermaid diagrams) AFTER the DocNode snapshot was taken. Without this
    refresh the next update compares the file against the pre-fix snapshot,
    concludes a human edited it, and skips that doc forever (stale) unless
    --force is used. Only the files passed in are adopted: a doc the validator
    did not touch keeps its snapshot, so genuine manual edits still surface."""
    if graph is None or not paths:
        return 0
    from repodoc.graph.models import DocNode

    targets = {os.path.abspath(p) for p in paths}
    refreshed = 0
    for nid, attrs in list(graph.nodes(data=True)):
        if attrs.get("node_type") != "doc":
            continue
        d = attrs.get("data", {})
        file_path = d.get("file_path")
        if not file_path or os.path.abspath(file_path) not in targets:
            continue
        if not os.path.exists(file_path):
            continue
        try:
            content = file_manager.load_text(file_path)
        except Exception as e:
            print(f"Warning: could not re-read {file_path}: {e}")
            continue
        recorded = d.get("content_hash") or ""
        actual = content_hash(content)
        # Legacy nodes (full content, no hash) are compacted to a hash here.
        if recorded == actual and not d.get("content"):
            continue
        graph.add_doc_node(
            DocNode(
                id=d.get("id", nid),
                content_hash=actual,
                format=d.get("format", "markdown"),
                version=d.get("version", 1),
                file_path=file_path,
                stale=bool(d.get("stale")),
            )
        )
        refreshed += 1
    return refreshed


class DocPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.agent = None
        self.components = {}
        self.leaf_nodes = []
        self.graph = None
        self.module_tree = {}
        self._start_time = None
        self._generated_files = []
        # True only when this run parsed the repository. Restoring an existing
        # graph must NOT snapshot the current tree: the tree may have changed
        # since the graph was built, and a snapshot would hide that from update.
        self._analyzed_working_tree = False

    def get_data_dir(self) -> str:
        return os.path.join(self.config.output_dir, "data")

    def get_project_name(self) -> str:
        repo_name = os.path.basename(os.path.normpath(self.config.repo_path))
        return "".join(c if c.isalnum() else "_" for c in repo_name)

    def run_analysis(self):
        """Analyze, or restore from the persisted knowledge graph.

        The knowledge graph is the single required input for doc generation:
        when {data}/{project}_knowledge_graph.json exists (produced by
        `repodoc analyze`), components and the module tree are restored from
        the graph alone — no re-parsing of the repository, no other input
        file. Otherwise the repo is analyzed and the graph persisted."""
        if self.load_from_graph():
            return self.components, self.leaf_nodes

        builder = DependencyGraphBuilder(self.config)
        self.components, self.leaf_nodes = builder.build_dependency_graph()
        self._analyzed_working_tree = True
        self.graph = build_graph_from_analysis(self.components)
        # Attach the clustering blueprint (module tree) when it already
        # exists, so the graph carries the full architecture from the start.
        from repodoc.graph.builder import attach_module_tree_to_graph

        attach_module_tree_to_graph(
            self.graph, self.load_module_tree(), self.components
        )
        # Persist immediately: the knowledge graph is the pipeline's backbone,
        # not a post-doc-generation summary — it must exist on disk right
        # after analysis (code nodes + calls edges + module structure).
        self._save_graph()
        return self.components, self.leaf_nodes

    def load_from_graph(self) -> bool:
        """Restore pipeline state (graph, components, module tree) from the
        persisted knowledge graph. Returns True when the graph existed."""
        project_name = self.get_project_name()
        kg_path = os.path.join(
            self.get_data_dir(), f"{project_name}_knowledge_graph.json"
        )
        if not os.path.exists(kg_path):
            return False
        from repodoc.graph.graph import HeterogeneousGraph
        from repodoc.graph.builder import graph_to_components, graph_to_module_tree

        self.graph = HeterogeneousGraph.load(kg_path)
        self.components = graph_to_components(self.graph)
        self.module_tree = graph_to_module_tree(self.graph)
        # leaf_nodes is an analysis-internal value (clustering input only);
        # it is not recoverable from the graph and not needed for docs.
        self.leaf_nodes = []
        return True

    def _save_graph(self) -> None:
        """Persist the knowledge graph to {data}/{project}_knowledge_graph.json."""
        if not self.graph:
            return
        project_name = self.get_project_name()
        data_dir = self.get_data_dir()
        kg_path = os.path.join(data_dir, f"{project_name}_knowledge_graph.json")
        self.graph.save(kg_path)

    def _save_graph_safe(self) -> None:
        try:
            self._save_graph()
        except Exception as e:
            print(f"Warning: Knowledge graph save failed: {e}")

    def _doc_id_from_path(self, fpath: str) -> str:
        """Map a generated doc file to its graph DocNode id.

        overview README -> "overview"; anything else -> "doc/{rel_dir}/{stem}"
        (rel_dir is the doc's directory relative to docs_dir)."""
        try:
            rel = os.path.relpath(fpath, self.config.docs_dir)
        except ValueError:
            rel = os.path.basename(fpath)
        rel = rel.replace("\\", "/")
        if rel == "README.md":
            return "overview"
        rel_dir, fname = os.path.split(rel)
        stem = os.path.splitext(fname)[0]
        return f"doc/{rel_dir}/{stem}" if rel_dir else f"doc/{stem}"

    def _add_doc_to_graph(self, result, target_id: str = None) -> None:
        """Incrementally attach a generated doc to the knowledge graph:
        add the DocNode now (not by scanning files afterwards) and, when the
        doc describes a known graph node (component or module), the precise
        describes edge to it."""
        if not self.graph:
            return
        if not getattr(result, "success", False):
            return  # never attach failed generations
        output = getattr(result, "output", None)
        if not output or not output.get("path"):
            return
        from repodoc.graph.models import DocNode, REL_DESCRIBES

        fpath = output["path"]
        doc_id = self._doc_id_from_path(fpath)
        self.graph.add_doc_node(
            DocNode(
                id=doc_id,
                content_hash=content_hash(output.get("content", "")),
                format="markdown",
                file_path=fpath,
            )
        )
        if target_id and self.graph.has_node(target_id):
            self.graph.add_edge(doc_id, target_id, REL_DESCRIBES, 1.0)

    def load_module_tree(self):
        """Load the clustering blueprint: from the knowledge graph when it
        carries module nodes (single-file input), else from module_tree.json
        (legacy/compat), else empty."""
        if self.graph is not None:
            from repodoc.graph.builder import graph_to_module_tree

            restored = graph_to_module_tree(self.graph)
            if restored:
                self.module_tree = restored
                return self.module_tree

        project_name = self.get_project_name()
        data_dir = self.get_data_dir()
        new_path = os.path.join(data_dir, f"{project_name}_module_tree.json")
        old_docs_path = os.path.join(self.config.docs_dir, "module_tree.json")

        if os.path.exists(new_path):
            self.module_tree = file_manager.load_json(new_path) or {}
        elif os.path.exists(old_docs_path):
            self.module_tree = file_manager.load_json(old_docs_path) or {}
        else:
            self.module_tree = {}
        return self.module_tree

    def initialize_agent(self):
        self.agent = DocAgent(self.config)
        return self.agent

    def get_code_statistics(self) -> dict[str, Any]:
        """Extract code analysis statistics from components."""
        stats = {
            "total_components": len(self.components),
            "component_types": {},
            "language_counts": {},
            "top_dependencies": [],
            "total_files": set(),
            "modules_with_components": {},
        }

        for comp_id, node in self.components.items():
            ctype = node.component_type or "unknown"
            stats["component_types"][ctype] = stats["component_types"].get(ctype, 0) + 1

            if hasattr(node, "file_path") and node.file_path:
                ext = os.path.splitext(node.file_path)[1]
                if ext:
                    stats["language_counts"][ext] = (
                        stats["language_counts"].get(ext, 0) + 1
                    )
                stats["total_files"].add(node.file_path)

            if hasattr(node, "depends_on") and node.depends_on:
                deps_list = list(node.depends_on)[:3]
                for dep in deps_list:
                    if dep not in [d[0] for d in stats["top_dependencies"]]:
                        stats["top_dependencies"].append((dep, 1))
                    else:
                        for i, (d, c) in enumerate(stats["top_dependencies"]):
                            if d == dep:
                                stats["top_dependencies"][i] = (d, c + 1)
                                break

        stats["top_dependencies"] = sorted(
            stats["top_dependencies"], key=lambda x: x[1], reverse=True
        )[:10]
        stats["total_files"] = len(stats["total_files"])

        return stats

    async def generate_overview(self, output_path: str = None):
        """Layer 3: overview README, based on the actual module docs (layer 2)
        and the module dependency map."""
        if not self.agent:
            self.initialize_agent()

        module_structure = self.load_module_tree()
        # The overview documents top-level modules, so dependencies are
        # folded from full paths onto their top-level names.
        module_deps, _ = _build_module_dependencies(
            self.components, module_structure
        )
        module_deps = _top_level_module_dependencies(module_deps)
        module_diagram = build_module_mermaid(
            module_deps, list(module_structure.keys())
        )

        # Module doc summaries (top-level module docs generated in layer 2)
        raw_module_docs: dict[str, str] = {}
        docs_dir = self.config.docs_dir
        for name in module_structure.keys():
            md_path = os.path.join(docs_dir, f"{name}.md")
            if os.path.exists(md_path):
                try:
                    with open(md_path, "r") as f:
                        raw_module_docs[name] = f.read()
                except Exception:
                    pass
        module_docs_summary = summarize_docs_for_prompt(
            raw_module_docs, DOCS_SUMMARY_TOTAL_BUDGET, 2000
        )

        module_count = len(module_structure)
        result = await self.agent.execute(
            {
                "type": "write_doc",
                "config": self.config,
                "input": {
                    "type": "overview",
                    "repo_name": os.path.basename(self.config.repo_path),
                    "component_count": len(self.components),
                    "module_count": module_count,
                    "module_dependencies": {
                        m: sorted(ds) for m, ds in module_deps.items()
                    },
                    "module_diagram": module_diagram,
                    "module_docs_summary": module_docs_summary,
                    "output_path": output_path
                    or os.path.join(self.config.docs_dir, "README.md"),
                },
            }
        )
        # The diagram is graph-derived: enforce it even if the LLM altered it.
        _enforce_overview_diagram(result, module_diagram)
        return result

    async def generate_module_docs(
        self,
        module_tree: dict = None,
        parent_dir: str = None,
        children_docs: dict[str, str] = None,
    ):
        """Layer 2: generate module docs for every tree node, BOTTOM-UP.

        Modules at the same depth are processed in parallel; deeper (leaf)
        modules are generated before their parents, so a parent module can
        reference its children's actual docs. Component docs (layer 1) must
        already exist.
        """
        if not self.agent:
            self.initialize_agent()

        if module_tree is None:
            module_tree = self.load_module_tree()

        if parent_dir is None:
            parent_dir = self.config.docs_dir

        if children_docs is None:
            children_docs = {}

        depth_groups = group_modules_by_depth(module_tree)
        module_deps, module_dependents = _build_module_dependencies(
            self.components, module_tree
        )
        module_deps = _subtree_module_dependencies(module_deps)
        module_dependents = _subtree_module_dependencies(module_dependents)

        results = []

        # Bottom-up: deepest modules first (reverse depth order)
        for depth in sorted(depth_groups.keys(), reverse=True):
            modules_at_depth = depth_groups[depth]

            def make_task(module_name, module_info, full_path):
                return self._generate_single_module_doc(
                    module_name,
                    module_info,
                    full_path,
                    module_tree,
                    module_deps,
                    module_dependents,
                )

            jobs = [(m, i, p) for m, i, p in modules_at_depth]
            depth_results = await asyncio.gather(
                *(make_task(*j) for j in jobs), return_exceptions=True
            )

            # Failure-driven retry loop: lower concurrency explicitly on each
            # failed round until all module docs succeed (doc completeness
            # is the hard goal).
            from repodoc.llm import _LIMITER

            for _round in range(4):
                failed = [
                    j
                    for j, r in zip(jobs, depth_results)
                    if isinstance(r, Exception)
                    or (r is not None and not getattr(r, "success", False))
                ]
                if not failed:
                    break
                if _LIMITER.permits > _LIMITER.min_permits:
                    _LIMITER.force_decrease()
                print(
                    f"Concurrency {_LIMITER.permits}, retrying "
                    f"{len(failed)} failed module docs (round {_round + 1})..."
                )
                retry = await asyncio.gather(
                    *(make_task(*j) for j in failed), return_exceptions=True
                )
                # slot retry results back at the failed positions
                retry_iter = iter(retry)
                depth_results = [
                    next(retry_iter)
                    if (
                        isinstance(r, Exception)
                        or (r is not None and not getattr(r, "success", False))
                    )
                    else r
                    for r in depth_results
                ]
                if _LIMITER.permits <= _LIMITER.min_permits:
                    break  # concurrency bottomed out: last try done

            results.extend(
                [r for r in depth_results if not isinstance(r, Exception)]
            )

        return results

    async def _generate_single_module_doc(
        self,
        module_name: str,
        module_info: dict,
        full_path: str,
        module_tree: dict,
        module_deps: dict[str, set[str]] = None,
        module_dependents: dict[str, set[str]] = None,
    ):
        """Generate documentation for a single module (architecture narrative)."""
        components = module_info.get("components", [])
        children = module_info.get("children", {})
        path_parts = full_path.split("/")
        current_parent_dir = self.config.docs_dir

        for part in path_parts[:-1]:
            current_parent_dir = os.path.join(current_parent_dir, part)

        module_dir = os.path.join(current_parent_dir, module_name)
        if children:
            file_manager.ensure_directory(module_dir)

        # Child module docs (leaf modules generated first, so they exist)
        raw_child_docs: dict[str, str] = {}
        for child_name in children.keys():
            child_doc_path = os.path.join(module_dir, f"{child_name}.md")
            if os.path.exists(child_doc_path):
                try:
                    with open(child_doc_path, "r") as f:
                        raw_child_docs[child_name] = f.read()
                except Exception:
                    pass
        child_docs_content = summarize_docs_for_prompt(
            raw_child_docs, CHILDREN_SUMMARY_TOTAL_BUDGET, 1500
        )

        # Member component docs (layer 1, generated before module docs).
        # Each component's doc lives in its deepest module dir.
        raw_component_docs: dict[str, str] = {}
        comp_modules = _collect_component_modules(module_tree)
        for comp_id in components:
            node = self.components.get(comp_id)
            if not (node and node.name):
                continue
            chain = comp_modules.get(comp_id, [module_name])
            comp_doc_path = os.path.join(self.config.docs_dir, *chain, f"{node.name}.md")
            if os.path.exists(comp_doc_path):
                try:
                    with open(comp_doc_path, "r") as f:
                        raw_component_docs[node.name] = f.read()
                except Exception:
                    pass
        component_docs_content = summarize_docs_for_prompt(
            raw_component_docs, DOCS_SUMMARY_TOTAL_BUDGET, 2000
        )

        # Sibling / parent context from the tree
        parent_module = ""
        sibling_modules = []
        if path_parts:
            parent_parts = path_parts[:-1]
            parent_module = parent_parts[-1] if parent_parts else ""
            if parent_parts:
                parent_info = module_tree
                for p in parent_parts:
                    parent_info = parent_info.get(p, {}).get("children", {})
                sibling_modules = [k for k in parent_info.keys() if k != module_name]
            else:
                sibling_modules = [k for k in module_tree.keys() if k != module_name]

        result = await self.agent.execute(
            {
                "type": "write_doc",
                "config": self.config,
                "input": {
                    "type": "module",
                    "module_name": module_name,
                    "module_path": module_info.get("path", module_name),
                    "parent_module": parent_module,
                    "sibling_modules": sibling_modules,
                    "module_dependencies": sorted(
                        module_deps.get(full_path, set())
                    ),
                    "dependent_modules": sorted(
                        module_dependents.get(full_path, set())
                    ),
                    "component_docs": component_docs_content,
                    "sub_modules": list(children.keys()),
                    "children_docs": child_docs_content,
                    "output_path": os.path.join(
                        current_parent_dir, f"{module_name}.md"
                    ),
                },
            }
        )
        if result is not None and getattr(result, "success", False):
            # Module docs describe the module node itself (when it exists).
            self._add_doc_to_graph(result, target_id=full_path)
        return result

    async def _generate_single_component(
        self, node_id: str, output_dir: str, module_chain: list[str]
    ) -> dict:
        node = self.components.get(node_id)
        if not node:
            return None

        # Reverse dependencies: which components call this one or derive from /
        # reference it (its role in flows and the type hierarchy).
        predecessors = [
            cid
            for cid, n in self.components.items()
            if node_id in component_dependencies(n)
        ]

        component = {
            "name": node.name,
            "component_type": node.component_type,
            "file_path": node.file_path,
            "relative_path": node.relative_path,
            "docstring": node.docstring,
            "source_code": node.source_code or "",
            "depends_on": sorted(component_dependencies(node)),
        }

        result = await self.agent.execute(
            {
                "type": "write_doc",
                "config": self.config,
                "input": {
                    "type": "component",
                    "component": component,
                    "predecessors": predecessors,
                    "module_name": ".".join(module_chain),
                    "output_path": os.path.join(output_dir, f"{node.name}.md"),
                },
            }
        )
        if result is not None and getattr(result, "success", False):
            # Precise describes edge at generation time (no filename matching).
            self._add_doc_to_graph(result, target_id=node_id)
        return result

    async def generate_component_docs(
        self, docs_dir: str = None, module_tree: dict = None
    ):
        """Layer 1: generate docs for ALL components in the tree (including
        nested modules). Each component doc lands in its deepest module's
        directory. Runs in parallel (Semaphore 16)."""
        if not self.agent:
            self.initialize_agent()

        docs_dir = docs_dir or self.config.docs_dir

        if not module_tree:
            module_tree = self.load_module_tree()

        comp_modules = _collect_component_modules(module_tree)

        sem = asyncio.Semaphore(16)

        async def _run(comp_id: str, module_chain: list[str]):
            async with sem:
                module_dir = os.path.join(docs_dir, *module_chain)
                file_manager.ensure_directory(module_dir)
                return await self._generate_single_component(
                    comp_id, module_dir, module_chain
                )

        jobs = [
            (comp_id, chain)
            for comp_id, chain in sorted(comp_modules.items())
            if comp_id in self.components
        ]
        results = await asyncio.gather(
            *(_run(*j) for j in jobs), return_exceptions=True
        )

        # Failure-driven retry loop: after each round of failures, explicitly
        # lower the concurrency (visible in the terminal) and retry the
        # failed docs — doc completeness is the hard goal, so we keep
        # lowering until everything succeeds or concurrency bottoms out.
        from repodoc.llm import _LIMITER

        for _round in range(4):
            failed = [
                j
                for j, r in zip(jobs, results)
                if isinstance(r, Exception)
                or (r is not None and not getattr(r, "success", False))
            ]
            if not failed:
                break
            if _LIMITER.permits > _LIMITER.min_permits:
                _LIMITER.force_decrease()
            print(
                f"Concurrency {_LIMITER.permits}, retrying "
                f"{len(failed)} failed component docs (round {_round + 1})..."
            )
            retry = await asyncio.gather(
                *(_run(*j) for j in failed), return_exceptions=True
            )
            # slot retry results back at the failed positions
            retry_iter = iter(retry)
            results = [
                next(retry_iter)
                if (
                    isinstance(r, Exception)
                    or (r is not None and not getattr(r, "success", False))
                )
                else r
                for r in results
            ]
            if _LIMITER.permits <= _LIMITER.min_permits:
                break  # concurrency bottomed out: this round was the last try

        still_failed = [
            j
            for j, r in zip(jobs, results)
            if isinstance(r, Exception)
            or (r is not None and not getattr(r, "success", False))
        ]
        if still_failed:
            print(
                f"WARNING: {len(still_failed)} component docs failed permanently: "
                + ", ".join(j[0] for j in still_failed[:5])
                + ("..." if len(still_failed) > 5 else "")
            )

        return [
            r for r in results if r is not None and not isinstance(r, Exception)
        ]


    def _collect_generated_files(self, docs_dir: str, data_dir: str) -> list[str]:
        files = []
        for ext in ["*.md", "*.json", "*.mmd"]:
            for f in glob.glob(os.path.join(docs_dir, "**", ext), recursive=True):
                rel = os.path.relpath(f, docs_dir)
                files.append(rel)
            for f in glob.glob(os.path.join(data_dir, "**", ext), recursive=True):
                rel = os.path.relpath(f, data_dir)
                if rel not in files:
                    files.append(rel)
        return sorted(files)

    async def _generate_module_docs_safe(self):
        try:
            return await self.generate_module_docs()
        except Exception as e:
            print(f"Warning: Module docs generation failed: {e}")
            return []

    async def _generate_overview_safe(self, output_path: str):
        try:
            return await self.generate_overview(output_path)
        except Exception as e:
            print(f"Warning: Overview generation failed: {e}")
            return None

    async def generate_all(self, docs_dir: str = None, data_dir: str = None):
        """Bottom-up layered generation:
        layer 1 component docs -> layer 2 module docs -> layer 3 overview."""
        docs_dir = docs_dir or self.config.docs_dir
        data_dir = data_dir or self.get_data_dir()
        file_manager.ensure_directory(docs_dir)
        file_manager.ensure_directory(data_dir)

        self._start_time = time.time()
        self._generated_files = []

        module_tree = self.load_module_tree()

        # Layer 1: all component docs (parallel)
        component_results = await self.generate_component_docs(
            docs_dir=docs_dir, module_tree=module_tree
        )
        print(f"Layer 1: {len(component_results)} component docs")
        self._save_graph_safe()  # graph now carries code + component docs

        # Layer 2: module docs, bottom-up (leaf modules first)
        module_results = await self._generate_module_docs_safe()
        print(f"Layer 2: {len(module_results)} module docs")
        self._save_graph_safe()  # + module docs

        # Layer 3: overview README
        overview_result = await self._generate_overview_safe(
            os.path.join(docs_dir, "README.md")
        )
        if overview_result is not None and getattr(overview_result, "success", False):
            self._add_doc_to_graph(overview_result)  # overview doc node
        print("Layer 3: overview README")

        self._save_graph_safe()

        link_results = validate_and_fix_links(docs_dir)
        # The validator rewrites files AFTER the snapshots were taken; adopt
        # those rewrites so the next update does not mistake them for manual
        # edits (see refresh_doc_snapshots).
        refreshed = refresh_doc_snapshots(
            self.graph,
            [os.path.join(docs_dir, rel) for rel in link_results.get("fixed", [])],
        )
        if refreshed:
            print(f"  Refreshed {refreshed} doc snapshot(s) after link/mermaid fixes")
        self._save_graph_safe()
        all_files = self._collect_generated_files(docs_dir, data_dir)
        total_duration = time.time() - self._start_time if self._start_time else 0

        token_usage = self._collect_token_usage()
        git_commit = self._get_git_commit()

        def _failed(results) -> int:
            return sum(1 for r in results if not getattr(r, "success", False))

        generation_quality = {
            "component_docs_done": len(component_results),
            "component_docs_failed": _failed(component_results),
            "module_docs_done": len(module_results),
            "module_docs_failed": _failed(module_results),
            "overview_failed": not (
                overview_result is not None
                and getattr(overview_result, "success", False)
            ),
        }
        if generation_quality["component_docs_failed"] or generation_quality[
            "module_docs_failed"
        ]:
            print(
                "WARNING: some docs failed permanently: "
                f"{generation_quality['component_docs_failed']} component, "
                f"{generation_quality['module_docs_failed']} module "
                f"(see metadata doc_generation stats)"
            )

        self._save_and_log_metadata(
            data_dir,
            all_files,
            total_duration,
            token_usage,
            git_commit,
            link_results,
            generation_quality,
        )

        return {
            "overview": overview_result,
            "modules": module_results,
            "components": component_results,
            "total_components": len(self.components),
            "docs_directory": docs_dir,
        }

    def _collect_token_usage(self) -> dict[str, Any]:
        """Collect token usage from the shared LLM client (covers clustering
        and doc generation)."""
        from repodoc.llm import total_token_usage

        usage = total_token_usage(self.config)
        return {
            "history": usage["history"],
            "total": usage["total"],
            "prompt": usage["prompt"],
            "completion": usage["completion"],
        }

    def _get_git_commit(self) -> str | None:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.config.repo_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _save_and_log_metadata(
        self,
        data_dir: str,
        all_files: list[str],
        total_duration: float,
        token_usage: dict[str, Any],
        git_commit: str | None,
        link_results: dict[str, Any],
        generation_quality: dict[str, Any] | None = None,
    ) -> None:
        """Save metadata to JSON and log operation."""
        metadata = {
            "generation_info": {
                "timestamp": datetime.now().isoformat(),
                "total_duration_seconds": round(total_duration, 2),
                "main_model": self.config.main_model,
                "generator_version": "0.1.0",
                "repo_path": self.config.repo_path,
                "git_commit": git_commit,
            },
            "statistics": {
                "total_components": len(self.components),
                # When the graph was restored (no fresh analysis) leaf_nodes is
                # empty; report the components actually assigned to modules.
                "leaf_nodes": len(self.leaf_nodes)
                or len(_collect_component_modules(self.module_tree)),
                "total_files": len(all_files),
            },
            "token_usage": {
                "total_tokens": token_usage["total"],
                "prompt_tokens": token_usage["prompt"],
                "completion_tokens": token_usage["completion"],
                "llm_calls": len(token_usage["history"]),
                "calls": token_usage["history"],
            },
            "files_generated": all_files,
            "link_validation": {
                "files_fixed": len(link_results.get("fixed", [])),
                "broken_links_removed": len(link_results.get("removed", [])),
            },
            "doc_generation": generation_quality or {},
        }
        # Record the analyzed working tree (including uncommitted changes) so
        # the next update does not re-process or misread this state. When the
        # graph was restored, preserve the snapshot of the tree it was built
        # from — the working tree may already differ from it.
        from repodoc.worktree import record_worktree_snapshot

        project_name = self.get_project_name()
        metadata_path = os.path.join(
            data_dir, f"{project_name}_metadata.json"
        )
        if self._analyzed_working_tree:
            record_worktree_snapshot(metadata, self.config)
        else:
            previous = file_manager.load_json(metadata_path) or {}
            if previous.get("working_tree"):
                metadata["working_tree"] = previous["working_tree"]
            if previous.get("generation_info", {}).get("git_commit"):
                metadata.setdefault("generation_info", {})["git_commit"] = (
                    previous["generation_info"]["git_commit"]
                )
        file_manager.save_json(metadata, metadata_path)

        log_operation(
            output_dir=self.config.output_dir,
            operation_type="full_generation",
            repo_path=self.config.repo_path,
            git_commit=git_commit,
            duration_seconds=total_duration,
            total_tokens=token_usage["total"],
            prompt_tokens=token_usage["prompt"],
            completion_tokens=token_usage["completion"],
            llm_calls=len(token_usage["history"]),
            components_processed=len(self.components),
            files_generated=len(all_files),
            status="success",
        )


def run_pipeline(config: Config, generate_docs: bool = True):
    pipeline = DocPipeline(config)
    components, leaf_nodes = pipeline.run_analysis()
    if generate_docs:
        results = asyncio.run(pipeline.generate_all())
        return results
    return {"components": components, "leaf_nodes": leaf_nodes}
