"""
Repository dependency graph builder: file tree -> parse -> components -> leaf nodes.

Merges the former analysis pipeline (repo analyzer, call graph analyzer,
component assembly, topological leaf selection) into one pass over the
repository, dispatching files to the language analyzers via the registry.
"""

import fnmatch
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from repodoc.config import Config
from repodoc.graph.analyzers import EXTENSION_MAP, analyze_file
from repodoc.graph.graph import HeterogeneousGraph
from repodoc.graph.models import REL_CALLS, CallRelationship, CodeNode, Node
from repodoc.utils import file_manager, safe_open_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File tree walking (include/exclude patterns)
# ---------------------------------------------------------------------------


def _should_exclude_path(path: str, filename: str, exclude_patterns: Set[str]) -> bool:
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(filename, pattern):
            return True
        if pattern.endswith("/") and path.startswith(pattern.rstrip("/")):
            return True
        if path.startswith(pattern + "/") or path == pattern:
            return True
        if pattern in path.split("/"):
            return True
    return False


def _should_include_file(path: str, filename: str, include_patterns: List[str]) -> bool:
    if not include_patterns:
        return True
    for pattern in include_patterns:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(filename, pattern):
            return True
    return False


def _collect_code_files(
    repo_dir: str,
    include_patterns: List[str],
    exclude_patterns: Set[str],
) -> List[Dict[str, str]]:
    """Walk the repository and return code files as {path, language} dicts."""
    from repodoc.graph.analyzers import DEFAULT_IGNORE_PATTERNS, DEFAULT_INCLUDE_PATTERNS

    include = include_patterns if include_patterns is not None else DEFAULT_INCLUDE_PATTERNS
    exclude = (
        set(DEFAULT_IGNORE_PATTERNS) | set(exclude_patterns)
        if exclude_patterns is not None
        else set(DEFAULT_IGNORE_PATTERNS)
    )
    # Keep the path unresolved (macOS /tmp is a symlink to /private/tmp; git
    # reports the unresolved form, and update() matches file_path against it)
    base = Path(repo_dir)
    code_files: List[Dict[str, str]] = []

    for root, dirs, files in os.walk(base):
        root_path = Path(root)
        rel_root = str(root_path.relative_to(base))
        # Prune excluded directories
        dirs[:] = [
            d
            for d in dirs
            if not _should_exclude_path(
                f"{rel_root}/{d}" if rel_root != "." else d, d, exclude
            )
        ]
        for fname in files:
            if os.path.islink(root_path / fname):
                continue
            rel_path = os.path.relpath(str(root_path / fname), base)
            if _should_exclude_path(rel_path, fname, exclude):
                continue
            if not _should_include_file(rel_path, fname, include):
                continue
            ext = Path(fname).suffix.lower()
            if ext not in EXTENSION_MAP:
                continue
            code_files.append({"path": rel_path, "language": EXTENSION_MAP[ext]})
    return code_files


# ---------------------------------------------------------------------------
# Relationship resolution (analyzer output -> component depends_on)
# ---------------------------------------------------------------------------


def _assemble_components(
    nodes: List[Node], relationships: List[CallRelationship]
) -> Dict[str, Node]:
    """Turn analyzer nodes + call relationships into an id-keyed component dict
    with depends_on edges merged in (mirrors the former ast_parser pipeline)."""
    # Canonicalize relationship callees against all known nodes (id/name/last segment)
    func_lookup: Dict[str, str] = {}
    for node in nodes:
        fid = node.id or f"{node.file_path}:{node.name}"
        func_lookup[fid] = fid
        func_lookup[node.name] = fid
        method_name = fid.split(".")[-1]
        if method_name not in func_lookup:
            func_lookup[method_name] = fid

    components: Dict[str, Node] = {}
    id_mapping: Dict[str, str] = {}
    for node in nodes:
        if not node.id:
            continue
        components[node.id] = node
        id_mapping[node.id] = node.id
        legacy_id = f"{node.file_path}:{node.name}"
        if legacy_id and legacy_id != node.id:
            id_mapping[legacy_id] = node.id

    for rel in relationships:
        callee = rel.callee
        if callee in func_lookup:
            callee = func_lookup[callee]
        elif "." in callee:
            method_name = callee.split(".")[-1]
            if method_name in func_lookup:
                callee = func_lookup[method_name]
        caller_id = id_mapping.get(rel.caller)
        callee_id = id_mapping.get(callee)
        if not callee_id:
            for comp in components.values():
                if comp.name == callee:
                    callee_id = comp.id
                    break
        if caller_id and caller_id in components and callee_id:
            components[caller_id].depends_on.add(callee_id)
    return components


# ---------------------------------------------------------------------------
# Leaf node selection (cycle handling + type/name filtering)
# ---------------------------------------------------------------------------

def _detect_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """Detect cycles using Tarjan's algorithm (strongly connected components)."""
    index_counter = [0]
    index: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    onstack: Set[str] = set()
    stack: List[str] = []
    result: List[List[str]] = []

    def strongconnect(node: str) -> None:
        index[node] = index_counter[0]
        lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        onstack.add(node)
        for successor in graph.get(node, set()):
            if successor not in index:
                strongconnect(successor)
                lowlink[node] = min(lowlink[node], lowlink[successor])
            elif successor in onstack:
                lowlink[node] = min(lowlink[node], index[successor])
        if lowlink[node] == index[node]:
            scc = []
            while True:
                successor = stack.pop()
                onstack.discard(successor)
                scc.append(successor)
                if successor == node:
                    break
            if len(scc) > 1:
                result.append(scc)

    for node in graph:
        if node not in index:
            strongconnect(node)
    return result


def _resolve_cycles(graph: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Resolve cycles by breaking one edge per cycle."""
    cycles = _detect_cycles(graph)
    if not cycles:
        return graph
    new_graph = {node: deps.copy() for node, deps in graph.items()}
    for cycle in cycles:
        for j in range(len(cycle) - 1):
            current, next_node = cycle[j], cycle[j + 1]
            if next_node in new_graph.get(current, set()):
                new_graph[current].discard(next_node)
                break
    return new_graph


def _build_graph_from_components(components: Dict[str, Node]) -> Dict[str, Set[str]]:
    """Build adjacency graph from components (A -> B if A depends on B)."""
    graph: Dict[str, Set[str]] = {}
    for comp_id, component in components.items():
        graph.setdefault(comp_id, set())
        for dep_id in getattr(component, "depends_on", []):
            if dep_id in components:
                graph[comp_id].add(dep_id)
    return graph


def _get_valid_component_types(components: Dict[str, Node]) -> Set[str]:
    """Component types eligible as leaf nodes. Classes always qualify;
    functions only when the repo is function-heavy (a single class would
    otherwise hide all function docs — typical for JS/Go/Rust/C repos).

    struct is NOT counted as a class: in C it is a data declaration, not a
    type with behavior — libevent's 1439 functions vs 763 struct/class would
    otherwise lose all function docs (1439 < 763*2)."""
    type_counts = Counter(c.component_type for c in components.values())
    # enum/annotation are first-class doc targets in Java; "method" is
    # deliberately excluded (methods belong to their class's doc)
    valid = {"class", "interface", "struct", "enum", "annotation"}
    classes = type_counts.get("class", 0) + type_counts.get("interface", 0)
    funcs = type_counts.get("function", 0)
    if not classes or funcs > classes * 2:
        valid.add("function")
    return valid


def _get_leaf_nodes(
    graph: Dict[str, Set[str]], components: Dict[str, Node]
) -> List[str]:
    """Leaf nodes: no other node depends on them. Filter by valid types."""
    acyclic = _resolve_cycles(graph)
    leaf_nodes = set(acyclic.keys())
    valid_types = _get_valid_component_types(components)

    def concise_node(ln: Set[str]) -> List[str]:
        concise = set()
        for n in ln:
            concise.add(n.replace(".__init__", "") if n.endswith("__init__") else n)
        valid = valid_types
        keep = []
        for n in concise:
            if not isinstance(n, str) or not n.strip():
                continue
            if n in components and components[n].component_type in valid:
                keep.append(n)
        return keep

    out = concise_node(leaf_nodes)
    # Only shrink to true leaves (no dependents) when functions are in play:
    # function counts can explode on big C/JS repos. Class/interface/enum
    # counts are naturally bounded (Java projects stay ~400), and shrinking
    # them is discontinuous — mybatis jumped 357 -> 422 -> 118 when enum/
    # annotation leaves pushed it over the 400 mark.
    if "function" in valid_types and len(out) >= 400:
        for node, deps in acyclic.items():
            for dep in deps:
                leaf_nodes.discard(dep)
        out = concise_node(leaf_nodes)
    if not leaf_nodes:
        return []
    return out


# ---------------------------------------------------------------------------
# Knowledge graph assembly (code nodes + calls edges)
# ---------------------------------------------------------------------------


def build_graph_from_analysis(components: Dict[str, Node]) -> HeterogeneousGraph:
    """Populate a heterogeneous graph with CodeNodes and calls edges from components."""
    graph = HeterogeneousGraph()
    for nid, node in components.items():
        code_node = CodeNode(
            id=node.id,
            type=node.component_type,
            component_type=node.component_type,
            file_path=node.file_path,
            source_code=node.source_code,
            name=node.name,
            relative_path=node.relative_path,
            start_line=node.start_line,
            end_line=node.end_line,
            docstring=node.docstring or "",
            depends_on=sorted(node.depends_on),
            parameters=node.parameters,
            receiver=node.receiver,
        )
        graph.add_code_node(code_node)
    for nid, node in components.items():
        for callee_id in node.depends_on:
            if callee_id in components:
                graph.add_edge(nid, callee_id, REL_CALLS, 1.0)
    return graph


def attach_module_tree_to_graph(
    graph: HeterogeneousGraph,
    module_tree: dict,
    components: Dict[str, Node],
) -> HeterogeneousGraph:
    """Attach the clustering result (module tree) to the knowledge graph.

    Adds a ModuleNode per module (id = module path in the tree), contains
    edges for module -> component and module -> child module, and depends_on
    edges for cross-module call aggregation (component calls that cross
    module boundaries, using each component's deepest module home).

    Returns the graph for chaining. Components/modules missing from the
    graph are skipped silently (e.g. modules over components the tree
    excludes from documentation)."""
    from repodoc.graph.models import REL_CONTAINS, REL_DEPENDS_ON, ModuleNode

    def deepest_module(nodes: dict, path: str, best: dict) -> None:
        """Map each component id to its DEEPEST owning module path."""
        for name, info in nodes.items():
            cur = f"{path}/{name}" if path else name
            for cid in info.get("components", []):
                if cid not in best or len(cur) > len(best[cid]):
                    best[cid] = cur
            children = info.get("children")
            if isinstance(children, dict):
                deepest_module(children, cur, best)

    def add_module_nodes(nodes: dict, parent: str | None) -> None:
        for name, info in nodes.items():
            cur = f"{parent}/{name}" if parent else name
            graph.add_module_node(
                ModuleNode(
                    id=cur,
                    name=name,
                    path=info.get("path", ""),
                    parent=parent,
                    components=list(info.get("components", [])),
                )
            )
            if parent is not None:
                graph.add_edge(parent, cur, REL_CONTAINS, 1.0)
            for cid in info.get("components", []):
                if graph.has_node(cid):
                    graph.add_edge(cur, cid, REL_CONTAINS, 1.0)
            children = info.get("children")
            if isinstance(children, dict):
                add_module_nodes(children, cur)

    add_module_nodes(module_tree, None)

    # Cross-module dependency edges: component calls that cross module
    # boundaries (module A depends on module B).
    comp_module: dict[str, str] = {}
    deepest_module(module_tree, "", comp_module)
    for cid, node in components.items():
        m = comp_module.get(cid)
        if m is None or not graph.has_node(m):
            continue
        for dep_id in getattr(node, "depends_on", []):
            dm = comp_module.get(dep_id)
            if dm and dm != m and graph.has_node(dm):
                graph.add_edge(m, dm, REL_DEPENDS_ON, 1.0)

    return graph


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


class DependencyGraphBuilder:
    """Build and save dependency graph; return components and leaf nodes."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def build_dependency_graph(
        self,
        build_kg: bool = False,
        kg_output_path: str | None = None,
    ) -> Tuple[Dict[str, Node], List[str]]:
        file_manager.ensure_directory(self.config.dependency_graph_dir)
        repo_name = os.path.basename(os.path.normpath(self.config.repo_path))
        sanitized = "".join(c if c.isalnum() else "_" for c in repo_name)
        dependency_graph_path = os.path.join(
            self.config.dependency_graph_dir,
            f"{sanitized}_dependency_graph.json",
        )

        code_files = _collect_code_files(
            self.config.repo_path, include_patterns=None, exclude_patterns=None
        )

        all_nodes: List[Node] = []
        all_relationships: List[CallRelationship] = []
        base = Path(self.config.repo_path)
        for file_info in code_files:
            file_path = base / file_info["path"]
            try:
                content = safe_open_text(base, file_path)
            except PermissionError as e:
                logger.debug("Skipping %s: %s", file_path, e)
                continue
            # Skip ANTLR-generated files which can cause recursion errors
            if "ANTLR GENERATED CODE" in content[:500]:
                logger.debug("Skipping ANTLR generated file: %s", file_path)
                continue
            nodes, relationships = analyze_file(
                str(file_path), content, str(base)
            )
            all_nodes.extend(nodes)
            all_relationships.extend(relationships)

        components = _assemble_components(all_nodes, all_relationships)

        # Save dependency graph (JSON, {component_id: node dump})
        result = {}
        for cid, component in components.items():
            d = component.model_dump()
            if isinstance(d.get("depends_on"), set):
                d["depends_on"] = list(d["depends_on"])
            result[cid] = d
        os.makedirs(os.path.dirname(dependency_graph_path), exist_ok=True)
        with open(dependency_graph_path, "w", encoding="utf-8") as f:
            import json

            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.debug(
            "Dependency graph saved: %s (%d components)",
            dependency_graph_path,
            len(components),
        )

        if build_kg and kg_output_path:
            kg = build_graph_from_analysis(components)
            kg_dir = os.path.dirname(kg_output_path)
            if kg_dir:
                file_manager.ensure_directory(kg_dir)
            kg.save(kg_output_path)
            logger.debug(
                "Knowledge graph saved: %s (%d nodes, %d edges)",
                kg_output_path,
                kg.node_count(),
                kg.edge_count(),
            )

        graph = _build_graph_from_components(components)
        leaf_nodes = _get_leaf_nodes(graph, components)

        valid_types = _get_valid_component_types(components)

        keep_leaf_nodes = []
        for ln in leaf_nodes:
            if not isinstance(ln, str) or not ln.strip():
                continue
            if ln in components and components[ln].component_type in valid_types:
                keep_leaf_nodes.append(ln)
            elif ln not in components:
                logger.warning("Leaf node %s not found in components", ln)

        return components, keep_leaf_nodes
