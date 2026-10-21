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

from repodoc.config import MAX_SOURCE_FILE_BYTES, Config
from repodoc.graph.analyzers import EXTENSION_MAP, analyze_file
from repodoc.graph.graph import HeterogeneousGraph
from repodoc.graph.imports import (
    ImportIndex,
    candidate_from_node,
    resolve_with_imports,
)
from repodoc.graph.models import (
    REL_CALLS,
    REL_IMPLEMENTS,
    REL_IMPORTS,
    REL_INSTANTIATES,
    REL_USES,
    CallRelationship,
    CodeNode,
    Node,
)
from repodoc.utils import file_manager, safe_open_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File tree walking (include/exclude patterns)
# ---------------------------------------------------------------------------


def _should_exclude_path(path: str, filename: str, exclude_patterns: Set[str]) -> bool:
    parts = path.split("/") if path else []
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(filename, pattern):
            return True
        if pattern.endswith("/"):
            # Directory pattern: match a whole path segment ("bin/" excludes
            # bin/x.py but not bindings/x.py).
            segment = pattern.rstrip("/")
            if segment and segment in parts:
                return True
            continue
        if path == pattern or pattern in parts:
            return True
        if path.startswith(pattern + "/"):
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
    max_file_size: int = MAX_SOURCE_FILE_BYTES,
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
            try:
                if (root_path / fname).stat().st_size > max_file_size:
                    logger.debug(
                        "Skipping oversized file (%d bytes): %s",
                        (root_path / fname).stat().st_size,
                        rel_path,
                    )
                    continue
            except OSError:
                continue
            code_files.append({"path": rel_path, "language": EXTENSION_MAP[ext]})
    return code_files


# ---------------------------------------------------------------------------
# Relationship resolution (analyzer output -> component depends_on)
# ---------------------------------------------------------------------------


# Relationship type -> Node field. Everything not listed is a call.
_RELATION_FIELDS = {
    REL_IMPLEMENTS: "implements",
    REL_USES: "uses",
    REL_INSTANTIATES: "instantiates",
    REL_IMPORTS: "imports",
}


def _assemble_components(
    nodes: List[Node],
    relationships: List[CallRelationship],
    imports_by_file: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Node]:
    """Turn analyzer nodes + relationships into an id-keyed component dict.

    Reference resolution is scope-aware: the caller's own file wins, then the
    caller file's import map (disambiguates same-named components across
    modules), then a globally unique name; ambiguous references are dropped
    instead of guessed (the former bare-name lookup created wrong cross-file
    edges on every name collision). Relations are stored in typed fields
    (calls / implements / uses / instantiates / imports) so the graph keeps
    their semantics."""
    components: Dict[str, Node] = {}
    id_mapping: Dict[str, str] = {}
    by_name: Dict[str, List[str]] = {}
    by_file_name: Dict[tuple[str, str], List[str]] = {}

    for node in nodes:
        if not node.id:
            continue
        components[node.id] = node
        id_mapping[node.id] = node.id
        legacy_id = f"{node.file_path}:{node.name}"
        if legacy_id and legacy_id != node.id:
            id_mapping[legacy_id] = node.id
        rel = (node.relative_path or node.file_path).replace(os.sep, "/")
        for name in dict.fromkeys((node.name, node.id.split(".")[-1])):
            if not name:
                continue
            by_name.setdefault(name, []).append(node.id)
            by_file_name.setdefault((rel, name), []).append(node.id)

    imports_by_file = imports_by_file or {}
    import_index = ImportIndex(
        [candidate_from_node(n) for n in components.values()]
    )

    def _resolve(ref: str, caller_id: str) -> Optional[str]:
        if not ref:
            return None
        if ref in components:
            return ref
        mapped = id_mapping.get(ref)
        if mapped:
            return mapped
        last = ref.split(".")[-1]
        caller = components.get(caller_id)
        caller_file = (
            (caller.relative_path or caller.file_path).replace(os.sep, "/")
            if caller
            else ""
        )
        # 1. Same file wins: a bare name is most likely local.
        for key in dict.fromkeys((ref, last)):
            cands = by_file_name.get((caller_file, key), [])
            if len(cands) == 1:
                return cands[0]
        # 2. Import context: the caller file's import map resolves names that
        #    collide across files (exact id, id suffix, module membership).
        import_match = resolve_with_imports(
            ref, caller_file, imports_by_file, import_index
        )
        if import_match:
            return import_match
        # 3. Globally unique name.
        for key in dict.fromkeys((ref, last)):
            cands = by_name.get(key, [])
            if len(cands) == 1:
                return cands[0]
        # 4. Unique dotted-suffix match (e.g. "mod.Class.method").
        suffix = "." + ref
        suffix_cands = [cid for cid in components if cid.endswith(suffix)]
        if len(suffix_cands) == 1:
            return suffix_cands[0]
        logger.debug(
            "Dropping ambiguous/unresolved reference %r from %s", ref, caller_id
        )
        return None

    for rel in relationships:
        caller_id = id_mapping.get(rel.caller, rel.caller)
        if caller_id not in components:
            continue
        relation = getattr(rel, "relation_type", REL_CALLS)
        callee_id = _resolve(rel.callee, caller_id)
        if not callee_id:
            # Keep the reference on the node: a later update that introduces
            # the missing definition reconciles it into an edge.
            components[caller_id].unresolved.setdefault(relation, set()).add(
                rel.callee
            )
            continue
        field = _RELATION_FIELDS.get(relation, "depends_on")
        getattr(components[caller_id], field).add(callee_id)
    return components


def infer_go_implements(components: Dict[str, Node]) -> Dict[str, Set[str]]:
    """Go structural typing: map struct id -> interface ids it implements.

    Go interfaces are satisfied implicitly — there is no declaration to
    record. A struct implements an interface when its method set (Go methods
    in the same directory whose receiver is the struct) covers every
    interface method signature ("Name/arity"). Conservative: full-signature
    matches only, and the struct must have at least one method."""
    by_dir_methods: Dict[Tuple[str, str], Set[Tuple[str, int]]] = {}
    structs: Dict[str, Tuple[str, str]] = {}
    interfaces: Dict[str, Set[Tuple[str, int]]] = {}
    for cid, node in components.items():
        rel = (node.relative_path or "").replace(os.sep, "/")
        if not rel.endswith(".go"):
            continue
        if node.component_type == "method" and node.receiver:
            receiver = node.receiver.lstrip("*")
            key = (os.path.dirname(rel), receiver)
            by_dir_methods.setdefault(key, set()).add(
                (node.name, len(node.parameters or []))
            )
        elif node.component_type == "struct":
            structs[cid] = (rel, node.name)
        elif node.component_type == "interface" and node.parameters:
            sigs: Set[Tuple[str, int]] = set()
            for sig in node.parameters:
                name, _, arity = str(sig).partition("/")
                if name:
                    sigs.add((name, int(arity) if arity.isdigit() else -1))
            if sigs:
                interfaces[cid] = sigs

    out: Dict[str, Set[str]] = {}
    if not structs or not interfaces:
        return out
    for cid, (rel, name) in structs.items():
        methods = by_dir_methods.get((os.path.dirname(rel), name))
        if not methods:
            continue
        wanted = {
            interface_id
            for interface_id, sigs in interfaces.items()
            if sigs <= methods
        }
        if wanted:
            out[cid] = wanted
    return out


def component_dependencies(node: Node) -> Set[str]:
    """All components a node references: calls, inheritance, type use,
    instantiation and imports."""
    out: Set[str] = set()
    for field in ("depends_on", "implements", "uses", "instantiates", "imports"):
        out |= set(getattr(node, field, ()) or ())
    return out


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
        for dep_id in component_dependencies(component):
            if dep_id in components:
                graph[comp_id].add(dep_id)
    return graph


# Type-like components that always qualify as doc targets. The set mirrors
# what the language analyzers actually emit: Java/C# produce "abstract class",
# "static class" and "record"; Rust/PHP produce "trait".
_CLASS_LIKE_TYPES = frozenset(
    {"class", "abstract class", "static class", "interface", "record", "trait"}
)
_ALWAYS_VALID_TYPES = _CLASS_LIKE_TYPES | {"struct", "enum", "annotation"}

# Importance ordering for _rank_by_importance (lower = more important).
_TYPE_RANK = {
    "class": 0,
    "abstract class": 0,
    "static class": 0,
    "interface": 1,
    "trait": 1,
    "record": 2,
    "enum": 2,
    "annotation": 3,
    "struct": 4,
    "function": 5,
}


def _get_valid_component_types(
    components: Dict[str, Node], config: Optional[Config] = None
) -> Set[str]:
    """Component types eligible as leaf nodes. Type-like components always
    qualify; functions only when the repo is function-heavy (types would
    otherwise hide all function docs — typical for JS/Go/Rust/C repos).

    struct is NOT counted as a type-like component for the heuristic: in C it
    is a data declaration, not a type with behavior — libevent's 1439
    functions vs 763 struct/class would otherwise lose all function docs
    (1439 < 763*2). It still qualifies as a doc target.

    config.doc_types overrides the automatic selection entirely."""
    if config is not None and config.doc_types:
        return {t.strip().lower() for t in config.doc_types if t.strip()}
    type_counts = Counter(c.component_type for c in components.values())
    # enum/annotation are first-class doc targets in Java; "method" is
    # deliberately excluded (methods belong to their class's doc)
    valid = set(_ALWAYS_VALID_TYPES)
    types = sum(type_counts.get(t, 0) for t in _CLASS_LIKE_TYPES)
    funcs = type_counts.get("function", 0)
    if not types or funcs > types * 2:
        valid.add("function")
    return valid


def _rank_by_importance(
    node_ids: List[str],
    graph: Dict[str, Set[str]],
    components: Dict[str, Node],
    limit: int,
) -> List[str]:
    """Keep the `limit` most important nodes: most depended upon first, then
    by type (core types before helpers), then deterministically by id."""
    if not limit or len(node_ids) <= limit:
        return node_ids
    dependents: Counter[str] = Counter()
    for _src, deps in graph.items():
        for dep in deps:
            dependents[dep] += 1
    type_rank = _TYPE_RANK
    ranked = sorted(
        node_ids,
        key=lambda cid: (
            -dependents.get(cid, 0),
            type_rank.get(components[cid].component_type, 9),
            cid,
        ),
    )
    return ranked[:limit]


def _get_leaf_nodes(
    graph: Dict[str, Set[str]],
    components: Dict[str, Node],
    config: Optional[Config] = None,
) -> List[str]:
    """Leaf nodes: no other node depends on them. Filter by valid types."""
    acyclic = _resolve_cycles(graph)
    leaf_nodes = set(acyclic.keys())
    valid_types = _get_valid_component_types(components, config)

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
    if config is not None and config.max_docs:
        out = _rank_by_importance(out, acyclic, components, config.max_docs)
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
            implements=sorted(node.implements),
            uses=sorted(node.uses),
            instantiates=sorted(node.instantiates),
            imports=sorted(node.imports),
            unresolved={
                relation: sorted(refs)
                for relation, refs in (node.unresolved or {}).items()
                if refs
            },
            parameters=node.parameters,
            receiver=node.receiver,
            owner=node.owner,
        )
        graph.add_code_node(code_node)
    relation_fields = (
        (REL_CALLS, "depends_on"),
        (REL_IMPLEMENTS, "implements"),
        (REL_USES, "uses"),
        (REL_INSTANTIATES, "instantiates"),
        (REL_IMPORTS, "imports"),
    )
    for nid, node in components.items():
        for relation, field in relation_fields:
            for target in getattr(node, field, ()) or ():
                if target in components:
                    graph.add_edge(nid, target, relation, 1.0)
    return graph


def graph_to_components(graph: HeterogeneousGraph) -> Dict[str, Node]:
    """Restore analysis Nodes from graph code nodes.

    The knowledge graph carries every analysis field on its CodeNodes, so
    doc generation/update can rely on the graph ALONE (no re-parsing, no
    dependency_graph.json). Reverse of build_graph_from_analysis."""
    components: Dict[str, Node] = {}
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "code":
            continue
        d = attrs.get("data", {})
        components[node_id] = Node(
            id=d.get("id", node_id),
            name=d.get("name", ""),
            component_type=d.get("component_type") or d.get("type", "unknown"),
            file_path=d.get("file_path", ""),
            relative_path=d.get("relative_path", ""),
            depends_on=set(d.get("depends_on", [])),
            implements=set(d.get("implements", [])),
            uses=set(d.get("uses", [])),
            instantiates=set(d.get("instantiates", [])),
            imports=set(d.get("imports", [])),
            unresolved={
                key: set(refs)
                for key, refs in (d.get("unresolved") or {}).items()
            },
            source_code=d.get("source_code"),
            start_line=d.get("start_line", 0),
            end_line=d.get("end_line", 0),
            docstring=d.get("docstring", ""),
            parameters=d.get("parameters"),
            receiver=d.get("receiver"),
            owner=d.get("owner"),
        )
    return components


def graph_to_module_tree(graph: HeterogeneousGraph) -> dict:
    """Restore the nested module tree dict from graph module nodes.

    ModuleNodes carry name/path/components and their id is the tree path
    ("net/http"), so the full hierarchy (children included) is rebuilt from
    the graph alone. Empty dict when the graph has no module nodes."""
    tree: dict = {}
    modules = [
        attrs.get("data", {})
        for _nid, attrs in graph.nodes(data=True)
        if attrs.get("node_type") == "module"
    ]
    for d in sorted(modules, key=lambda m: m.get("id", "")):
        mid = d.get("id", "")
        parts = mid.split("/")
        if not parts or not parts[-1]:
            continue
        target = tree
        for part in parts[:-1]:
            target = target[part]["children"]
        target[parts[-1]] = {
            "path": d.get("path", ""),
            "components": list(d.get("components", [])),
            "children": {},
        }
    return tree


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
        for dep_id in component_dependencies(node):
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
            self.config.repo_path,
            include_patterns=self.config.include_patterns or None,
            exclude_patterns=set(self.config.exclude_patterns) or None,
            max_file_size=self.config.max_file_size,
        )

        all_nodes: List[Node] = []
        all_relationships: List[CallRelationship] = []
        imports_by_file: Dict[str, Dict[str, str]] = {}
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
            nodes, relationships, imports = analyze_file(
                str(file_path), content, str(base)
            )
            all_nodes.extend(nodes)
            all_relationships.extend(relationships)
            if imports:
                rel = os.path.relpath(str(file_path), str(base)).replace(
                    os.sep, "/"
                )
                imports_by_file[rel] = imports

        components = _assemble_components(
            all_nodes, all_relationships, imports_by_file
        )

        # Go has no explicit `implements` clause: infer structural conformance
        # so the graph carries the same implementor edges as other languages.
        for struct_id, interface_ids in infer_go_implements(components).items():
            components[struct_id].implements |= interface_ids

        # Save dependency graph (JSON, {component_id: node dump})
        result = {}
        for cid, component in components.items():
            d = component.model_dump()
            for field_name in (
                "depends_on",
                "implements",
                "uses",
                "instantiates",
                "imports",
            ):
                if isinstance(d.get(field_name), set):
                    d[field_name] = sorted(d[field_name])
            unresolved = d.get("unresolved")
            if isinstance(unresolved, dict):
                d["unresolved"] = {
                    key: sorted(refs) for key, refs in unresolved.items()
                }
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
        leaf_nodes = _get_leaf_nodes(graph, components, self.config)

        valid_types = _get_valid_component_types(components, self.config)

        keep_leaf_nodes = []
        for ln in leaf_nodes:
            if not isinstance(ln, str) or not ln.strip():
                continue
            if ln in components and components[ln].component_type in valid_types:
                keep_leaf_nodes.append(ln)
            elif ln not in components:
                logger.warning("Leaf node %s not found in components", ln)

        return components, keep_leaf_nodes
