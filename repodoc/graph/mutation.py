"""Incremental knowledge graph mutation: apply code changes to the graph.

The knowledge graph is the single source of truth for the repository's
current code state. For every changed file the old (base commit) and new
(working tree) versions are re-analyzed, and the graph is mutated in place:

- NEW component      -> add CodeNode + calls edges + heuristic module placement
- MODIFIED component -> overwrite node data + diff calls edges
- REMOVED component  -> remove node + all incident edges + module membership

Cross-module (REL_DEPENDS_ON) edges are always rebuilt from scratch after
mutation, so module-level dependencies never go stale.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from repodoc.graph.analyzers import analyze_file
from repodoc.graph.graph import HeterogeneousGraph
from repodoc.graph.imports import (
    Candidate,
    ImportIndex,
    candidate_from_node,
    module_path_from_relative,
    resolve_with_imports,
)
from repodoc.graph.models import (
    CODE_DEPENDENCY_RELATIONS,
    REL_CALLS,
    REL_CONTAINS,
    REL_DEPENDS_ON,
    REL_IMPLEMENTS,
    RELATION_NODE_FIELDS,
    CodeNode,
    ModuleNode,
    Node,
)

logger = logging.getLogger(__name__)

# (Node field, graph edge relation) pairs for every code-level dependency.
_RELATION_FIELD_EDGES: Tuple[Tuple[str, str], ...] = tuple(
    (field, relation) for relation, field in RELATION_NODE_FIELDS.items()
)
_FIELD_BY_RELATION = dict(RELATION_NODE_FIELDS)


@dataclass
class GraphMutationResult:
    """Summary of one incremental graph mutation run."""

    added: List[str] = field(default_factory=list)
    updated: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    added_edges: List[Tuple[str, str]] = field(default_factory=list)
    removed_edges: List[Tuple[str, str]] = field(default_factory=list)
    modules_added: List[str] = field(default_factory=list)

    @property
    def changed(self) -> List[str]:
        """All component ids touched by the mutation (added + updated)."""
        return sorted(set(self.added) | set(self.updated))


# ---------------------------------------------------------------------------
# Dependency resolution (mirrors _assemble_components in builder.py, but
# resolves against the EXISTING graph + the file's new nodes)
# ---------------------------------------------------------------------------


def _build_resolution_index(
    graph: HeterogeneousGraph, new_nodes: List[Node]
) -> tuple[
    Dict[str, Set[str]], Dict[tuple[str, str], Set[str]], Set[str]
]:
    """Ambiguity-aware name indexes for update-time reference resolution.

    Unlike a first-wins mapping, colliding names keep all candidates so an
    ambiguous reference is left unresolved instead of pointing at an arbitrary
    component (the full builder drops those too). New-file nodes replace the
    graph's old version of the same id."""
    by_name: Dict[str, Set[str]] = {}
    by_file_name: Dict[tuple[str, str], Set[str]] = {}
    all_ids: Set[str] = set()
    new_ids = {n.id for n in new_nodes}

    def add(cid: str, name: str, rel_path: str) -> None:
        keys = {name, cid.split(".")[-1]}
        for key in keys:
            if not key:
                continue
            by_name.setdefault(key, set()).add(cid)
            if rel_path:
                by_file_name.setdefault((rel_path, key), set()).add(cid)

    for node in new_nodes:
        rel = (node.relative_path or node.file_path or "").replace(os.sep, "/")
        all_ids.add(node.id)
        add(node.id, node.name, rel)

    for nid, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "code":
            continue
        d = attrs.get("data", {})
        cid = d.get("id", nid)
        if cid in new_ids:
            continue
        rel = (d.get("relative_path") or d.get("file_path") or "").replace(
            os.sep, "/"
        )
        all_ids.add(cid)
        add(cid, d.get("name") or "", rel)

    return by_name, by_file_name, all_ids


def _collect_candidates(
    graph: HeterogeneousGraph, new_nodes: List[Node]
) -> List[Candidate]:
    """Import-resolution candidates: the graph's code nodes + the new nodes."""
    out = [candidate_from_node(n) for n in new_nodes if n.id]
    for nid, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "code":
            continue
        d = attrs.get("data", {})
        rel = (d.get("relative_path") or d.get("file_path") or "").replace(
            os.sep, "/"
        )
        out.append(
            Candidate(
                id=d.get("id", nid),
                module=module_path_from_relative(rel),
                name=d.get("name") or "",
                file=rel,
            )
        )
    return out


def _resolve_dependencies(
    graph: HeterogeneousGraph,
    new_nodes: List[Node],
    relationships: list,
    imports: Optional[Dict[str, str]] = None,
    caller_file: str = "",
) -> Dict[str, Dict[str, Set[str]]]:
    """Resolve relationships into {node field: {node_id: targets}}.

    Only targets that exist in the graph (or among the new nodes) become
    edges — unresolved external references are dropped (there is no node to
    point at). Resolution order: exact id, the changed file's import map
    (disambiguates names that collide across modules), then the name lookup.
    Each relation keeps its own field/edge type (calls, implements, uses,
    instantiates, imports)."""
    new_by_id = {n.id: n for n in new_nodes}
    by_name, by_file_name, all_ids = _build_resolution_index(graph, new_nodes)
    import_index = ImportIndex(_collect_candidates(graph, new_nodes))
    imports_by_file = {caller_file: imports} if imports and caller_file else {}
    resolved: Dict[str, Dict[str, Set[str]]] = {
        field: {n.id: set() for n in new_nodes}
        for field, _relation in _RELATION_FIELD_EDGES
    }

    def resolve(ref: str, caller: str) -> Optional[str]:
        if not ref:
            return None
        if ref in new_by_id or ref in all_ids:
            return ref
        last = ref.split(".")[-1]
        caller_node = new_by_id[caller]
        file_of_caller = (
            caller_node.relative_path or caller_node.file_path or ""
        ).replace(os.sep, "/")
        # 1. Same file wins (a bare name is most likely local).
        for key in dict.fromkeys((ref, last)):
            candidates = by_file_name.get((file_of_caller, key), set())
            if len(candidates) == 1:
                return next(iter(candidates))
        # 2. Import-aware resolution.
        if imports_by_file:
            matched = resolve_with_imports(
                ref, file_of_caller, imports_by_file, import_index
            )
            if matched:
                return matched
        # 3. Globally unique name, then unique dotted suffix.
        for key in dict.fromkeys((ref, last)):
            candidates = by_name.get(key, set())
            if len(candidates) == 1:
                return next(iter(candidates))
        suffix = "." + ref
        suffix_matches = {cid for cid in all_ids if cid.endswith(suffix)}
        if len(suffix_matches) == 1:
            return next(iter(suffix_matches))
        return None

    for rel in relationships:
        if rel.caller in new_by_id:
            caller = rel.caller
        else:
            candidates = by_name.get(rel.caller, set()) | by_name.get(
                rel.caller.split(".")[-1], set()
            )
            if len(candidates) != 1:
                continue
            caller = next(iter(candidates))
            if caller not in new_by_id:
                continue
        relation = getattr(rel, "relation_type", REL_CALLS)
        callee = resolve(rel.callee, caller)
        if callee is None:
            # Keep the dangling reference so a later update can wire it.
            new_by_id[caller].unresolved.setdefault(relation, set()).add(
                rel.callee
            )
            continue
        field_name = _FIELD_BY_RELATION.get(relation, "depends_on")
        resolved[field_name][caller].add(callee)
    return resolved


# ---------------------------------------------------------------------------
# Module membership helpers
# ---------------------------------------------------------------------------


def _deepest_module_map(graph: HeterogeneousGraph) -> Dict[str, str]:
    """Map each component id to its DEEPEST owning module id ("" if none)."""
    best: Dict[str, str] = {}
    for nid, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "module":
            continue
        d = attrs.get("data", {})
        mid = d.get("id", nid)
        for cid in d.get("components", []):
            if cid not in best or len(mid) > len(best[cid]):
                best[cid] = mid
    return best


def component_module_ids(graph: HeterogeneousGraph, node_id: str) -> List[str]:
    """Module ids that currently contain the component (deepest last)."""
    out: List[str] = []
    for nid, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "module":
            continue
        d = attrs.get("data", {})
        if node_id in d.get("components", []):
            out.append(d.get("id", nid))
    out.sort(key=len)
    return out


def _find_existing_module(graph: HeterogeneousGraph, node: Node) -> Optional[str]:
    """Best existing module home for a component.

    Preference order:
    1. A module that already holds a component from the SAME FILE — several
       modules can share one directory path (semantic clustering), so path
       matching alone would place new classes arbitrarily.
    2. The deepest module whose path matches the component's location:
       either the member directory ('src/atcher/core') or a single-file
       module's path without extension ('src/atcher/config' for config.py).
    Returns None when nothing matches."""
    rel = (node.relative_path or "").replace(os.sep, "/")
    rel_dir = os.path.dirname(rel)
    rel_noext = rel.rsplit(".", 1)[0] if "." in os.path.basename(rel) else rel
    if not rel_dir and not rel_noext:
        return None

    rel_by_cid: Dict[str, str] = {}
    module_nodes: List[Tuple[str, Dict]] = []
    for nid, attrs in graph.nodes(data=True):
        node_type = attrs.get("node_type")
        d = attrs.get("data", {})
        if node_type == "code":
            rel_by_cid[d.get("id", nid)] = (
                d.get("relative_path") or ""
            ).replace(os.sep, "/")
        elif node_type == "module":
            module_nodes.append((d.get("id", nid), d))

    same_file_modules = {
        mid
        for mid, d in module_nodes
        if any(rel_by_cid.get(cid) == rel for cid in d.get("components", []))
    }
    if len(same_file_modules) == 1:
        return next(iter(same_file_modules))

    best_id: Optional[str] = None
    best_len = -1
    for mid, d in module_nodes:
        mpath = str(d.get("path") or "").strip().removeprefix("./").rstrip("/")
        if not mpath or mpath == ".":
            continue
        matched = (
            rel_noext == mpath
            or (rel_dir and (rel_dir == mpath or rel_dir.startswith(mpath + "/")))
        )
        if matched and len(mpath) > best_len:
            best_id = mid
            best_len = len(mpath)
    return best_id


def _place_new_component(graph: HeterogeneousGraph, node: Node) -> Optional[str]:
    """Heuristically place a new component into the module tree.

    Prefers an existing module whose path matches the component's location
    (keeps new members in their semantically correct module); falls back to
    the first directory segment (root-level files group by file stem).
    Creates the top-level module node when none exists. Returns the module id
    (None when the component has no path information)."""
    rel = node.relative_path or ""
    parts = [p for p in rel.split("/") if p]
    if not parts:
        return None
    rel_dir = os.path.dirname(rel)
    module_name = _find_existing_module(graph, node)
    if module_name is None:
        if rel_dir:
            module_name = parts[0]
            module_path = parts[0]
        else:
            module_name = os.path.splitext(parts[0])[0] or parts[0]
            module_path = "."

    if not graph.has_node(module_name):
        graph.add_module_node(
            ModuleNode(
                id=module_name,
                name=module_name,
                path=module_path,
                parent=None,
                components=[],
            )
        )
    else:
        existing = graph.get_node(module_name)
        if isinstance(existing, ModuleNode):
            graph.add_module_node(
                ModuleNode(
                    id=existing.id,
                    name=existing.name,
                    path=existing.path or module_path,
                    parent=existing.parent,
                    components=existing.components,
                )
            )

    if not graph.has_edge(module_name, node.id):
        graph.add_edge(module_name, node.id, REL_CONTAINS, 1.0)
    existing = graph.get_node(module_name)
    if isinstance(existing, ModuleNode) and node.id not in existing.components:
        graph.add_module_node(
            ModuleNode(
                id=existing.id,
                name=existing.name,
                path=existing.path,
                parent=existing.parent,
                components=existing.components + [node.id],
            )
        )
    return module_name


def _detach_references(graph: HeterogeneousGraph, node_id: str) -> List[str]:
    """Drop a going-away node from every referencing node's stored lists.

    Removing the node drops its incident edges, but the referencing nodes
    (usually in files that were not re-analyzed) keep it in depends_on /
    implements / uses / ... — a ghost reference that would otherwise leak
    into their prompts and docs. Returns the referrer ids rewritten."""
    referrers = {
        s
        for s, t, rt, _w in graph.edges()
        if t == node_id and s != node_id and rt in CODE_DEPENDENCY_RELATIONS
    }
    removed = graph.get_node(node_id)
    aliases = {node_id, node_id.split(".")[-1]}
    if isinstance(removed, CodeNode) and removed.name:
        aliases.add(removed.name)
    fields = [field for field, _relation in _RELATION_FIELD_EDGES]
    for cid in sorted(referrers):
        node = graph.get_node(cid)
        if not isinstance(node, CodeNode):
            continue
        data = node.model_dump()
        changed = False
        for field in fields:
            values = list(data.get(field) or [])
            kept = [v for v in values if v != node_id]
            if len(kept) != len(values):
                data[field] = kept
                changed = True
        unresolved = data.get("unresolved") or {}
        if unresolved:
            cleaned: Dict[str, List[str]] = {}
            for relation, refs in unresolved.items():
                kept_refs = [ref for ref in refs if ref not in aliases]
                if len(kept_refs) != len(refs):
                    changed = True
                if kept_refs:
                    cleaned[relation] = kept_refs
            if cleaned != unresolved:
                data["unresolved"] = cleaned
        if changed:
            graph.add_code_node(CodeNode.model_validate(data))
    return sorted(referrers)


def reconcile_unresolved_references(graph: HeterogeneousGraph) -> int:
    """Wire stored unresolved references to components that now exist.

    Called after graph mutations: a definition added by one changed file can
    make references in unchanged files resolvable, and removing a duplicate
    can make an ambiguous name unique. References that match exactly one
    component become edges and leave the unresolved list; ambiguous ones stay.
    Returns the number of edges added."""
    by_key: Dict[str, Set[str]] = {}
    for nid, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "code":
            continue
        data = attrs.get("data", {})
        cid = data.get("id", nid)
        for key in {cid, cid.split(".")[-1], data.get("name") or ""}:
            if key:
                by_key.setdefault(key, set()).add(cid)

    added = 0
    for nid, attrs in list(graph.nodes(data=True)):
        if attrs.get("node_type") != "code":
            continue
        data = dict(attrs.get("data", {}))
        unresolved = data.get("unresolved") or {}
        if not unresolved:
            continue
        changed = False
        remaining: Dict[str, List[str]] = {}
        for relation, refs in unresolved.items():
            field = _FIELD_BY_RELATION.get(relation, "depends_on")
            resolved_values = set(data.get(field) or [])
            kept: List[str] = []
            for ref in refs:
                candidates = by_key.get(ref)
                if not candidates:
                    candidates = by_key.get(str(ref).split(".")[-1])
                candidates = candidates or set()
                if len(candidates) == 1:
                    target = next(iter(candidates))
                    changed = True
                    if target != nid:
                        resolved_values.add(target)
                        if graph.has_node(target) and not graph.has_edge(
                            nid, target
                        ):
                            graph.add_edge(nid, target, relation, 1.0)
                            added += 1
                else:
                    kept.append(ref)
            if kept:
                remaining[relation] = kept
            data[field] = sorted(resolved_values)
        if changed:
            data["unresolved"] = remaining
            graph.add_code_node(CodeNode.model_validate(data))
    return added


def _remove_component_from_modules(graph: HeterogeneousGraph, node_id: str) -> None:
    """Remove a component from every module node's membership + contains edges."""
    for nid, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "module":
            continue
        d = attrs.get("data", {})
        comps = d.get("components", [])
        if node_id not in comps:
            continue
        graph.remove_edge(nid, node_id)
        kept = [c for c in comps if c != node_id]
        graph.add_module_node(
            ModuleNode(
                id=d.get("id", nid),
                name=d.get("name", ""),
                path=d.get("path", ""),
                parent=d.get("parent"),
                components=kept,
            )
        )


def infer_go_implements_in_graph(graph: HeterogeneousGraph) -> int:
    """Refresh Go struct -> interface implements edges on the live graph.

    Recomputes structural conformance for the whole graph (cheap: structs x
    interfaces) so method/interface additions and removals converge. Only
    edges to Go interfaces are touched. Returns the number of edges added."""
    from repodoc.graph.builder import graph_to_components, infer_go_implements

    components = graph_to_components(graph)
    desired = infer_go_implements(components)
    go_interface_ids = {
        cid
        for cid, node in components.items()
        if node.component_type == "interface"
        and (node.relative_path or "").replace(os.sep, "/").endswith(".go")
    }
    added = 0
    for cid, node in components.items():
        rel = (node.relative_path or "").replace(os.sep, "/")
        if node.component_type != "struct" or not rel.endswith(".go"):
            continue
        graph_node = graph.get_node(cid)
        if not isinstance(graph_node, CodeNode):
            continue
        wanted = desired.get(cid, set())
        existing = {
            t
            for t, rt, _w in graph.get_out_edges(cid)
            if rt == REL_IMPLEMENTS
        }
        for target in sorted(existing & go_interface_ids):
            if target not in wanted:
                graph.remove_edge(cid, target)
        for target in sorted(wanted):
            if graph.has_node(target) and not graph.has_edge(cid, target):
                graph.add_edge(cid, target, REL_IMPLEMENTS, 1.0)
                added += 1
        field = sorted(
            (set(graph_node.implements or []) - go_interface_ids) | wanted
        )
        if field != sorted(graph_node.implements or []):
            data = graph_node.model_dump()
            data["implements"] = field
            graph.add_code_node(CodeNode.model_validate(data))
    return added


def rebuild_cross_module_edges(graph: HeterogeneousGraph) -> int:
    """Rebuild all REL_DEPENDS_ON edges from component calls edges.

    Removes every existing depends_on edge and recomputes them from the
    graph's own code-node depends_on lists + deepest module map, so module
    dependencies always mirror the current component-level calls edges.
    Returns the number of depends_on edges written."""
    for s, t, rt, _w in graph.edges(REL_DEPENDS_ON):
        graph.remove_edge(s, t)

    comp_module = _deepest_module_map(graph)
    count = 0
    for nid, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "code":
            continue
        m = comp_module.get(nid)
        if m is None or not graph.has_node(m):
            continue
        data = attrs.get("data", {})
        targets: Set[str] = set()
        for field, _relation in _RELATION_FIELD_EDGES:
            targets |= set(data.get(field, []) or [])
        for dep_id in targets:
            dm = comp_module.get(dep_id)
            if dm and dm != m and graph.has_node(dm) and not graph.has_edge(m, dm):
                graph.add_edge(m, dm, REL_DEPENDS_ON, 1.0)
                count += 1
    return count


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply_file_changes_to_graph(
    graph: HeterogeneousGraph,
    file_path: str,
    new_content: Optional[str],
    repo_path: str,
    old_file_path: Optional[str] = None,
    rebuild_modules: bool = True,
) -> GraphMutationResult:
    """Reconcile the graph with the working-tree version of one file.

    The GRAPH itself is the reference for what existed before (not the base
    commit): every graph code node whose relative path is this file (or its
    pre-rename path) is diffed against a fresh analysis of the working-tree
    content. This converges the graph to the actual code even when git
    changes cancel out between updates (e.g. a component added and then
    removed without a commit in between).

    Args:
        graph: The knowledge graph to mutate (in place).
        file_path: Repo-relative (or absolute) path of the changed file.
        new_content: Working-tree file content; None means the file was
            deleted (its graph nodes are removed). "" is an existing empty
            file — never treated as a deletion.
        repo_path: Repository root (used for analyzer relative paths).
        old_file_path: Pre-rename path of the file (renames): graph nodes of
            the old path are also in scope and get removed; the old content
            is NOT analyzed — the graph holds the previous state.
        rebuild_modules: Rebuild cross-module depends_on edges at the end of
            this call. Batch callers pass False and rebuild ONCE after the
            whole batch (per-file rebuilds are O(files x graph)).

    Returns a GraphMutationResult summarizing added/updated/removed nodes
    and edges. Callers must persist the graph afterwards; when processing a
    batch of files, run reconcile_unresolved_references(graph) once after the
    batch so references that the new definitions resolved become edges."""
    result = GraphMutationResult()

    # Analyzers derive module paths from (file_path, repo_path); pass the
    # absolute path so relative_path/ids match the full-repo analysis.
    analyze_path = file_path
    if not os.path.isabs(file_path) and repo_path:
        analyze_path = os.path.join(repo_path, file_path)

    new_nodes: List[Node] = []
    new_rels: list = []
    new_imports: Dict[str, str] = {}
    if new_content is not None:
        if not new_content.strip():
            # Empty/whitespace-only file: never a deletion. The analysis has
            # nothing to reconcile, and removing the components would turn a
            # transient truncation into permanent graph loss.
            logger.warning(
                "Skipping graph reconciliation for %s: file is empty", file_path
            )
            return result
        try:
            new_nodes, new_rels, new_imports = analyze_file(
                analyze_path, new_content, repo_path
            )
        except Exception as e:
            logger.warning("Failed to analyze %s: %s", file_path, e)
            new_nodes, new_rels, new_imports = [], [], {}
        if not new_nodes:
            # A non-empty file that yields zero components is almost always a
            # parse failure — never nuke the graph over an unparseable file.
            logger.warning(
                "Skipping graph reconciliation for %s: analysis produced no "
                "components (parse failure?)",
                file_path,
            )
            return result

    # Graph's current nodes for this file (and the pre-rename path)
    paths = {file_path}
    if old_file_path:
        paths.add(old_file_path)
    graph_nodes: dict[str, dict] = {}
    for nid, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "code":
            continue
        d = attrs.get("data", {})
        if d.get("relative_path") in paths:
            graph_nodes[nid] = d

    new_ids = {n.id for n in new_nodes if n.id}

    # Resolve dependencies ONCE against the whole graph: analyzer nodes carry
    # empty relation fields (builder fills them repo-wide), so the resolved
    # maps are what gets persisted on the mutated CodeNodes. The changed
    # file's import map disambiguates names that collide across modules.
    caller_file = ""
    if new_nodes:
        caller_file = (
            new_nodes[0].relative_path or new_nodes[0].file_path or ""
        ).replace(os.sep, "/")
    resolved = _resolve_dependencies(
        graph, new_nodes, new_rels, new_imports, caller_file
    )

    # --- Removed: in the graph for this file, not in the new analysis ---
    for nid in sorted(set(graph_nodes) - new_ids):
        for s, t, rt, _w in list(graph.edges()):
            if rt in CODE_DEPENDENCY_RELATIONS:
                if s == nid or t == nid:
                    result.removed_edges.append((s, t))
        _detach_references(graph, nid)
        _remove_component_from_modules(graph, nid)
        if graph.remove_node(nid):
            result.removed.append(nid)
            logger.info("Removed component %s", nid)

    # --- Added: in the new analysis, not in the graph ---
    for node in new_nodes:
        if node.id in graph_nodes:
            continue
        for field, _relation in _RELATION_FIELD_EDGES:
            setattr(node, field, resolved[field].get(node.id, set()))
        _add_component_node(graph, node)
        for field, relation in _RELATION_FIELD_EDGES:
            for target in sorted(getattr(node, field) - {node.id}):
                if graph.has_node(target):
                    graph.add_edge(node.id, target, relation, 1.0)
                    result.added_edges.append((node.id, target))
        result.added.append(node.id)
        logger.info("Added component %s", node.id)

    # --- Updated: present in both, content OR resolved edges differ ---
    for node in new_nodes:
        if node.id not in graph_nodes:
            continue
        # Resolve first: a new/changed definition elsewhere in the same file
        # can change this node's edges even when its own source is identical.
        for field, _relation in _RELATION_FIELD_EDGES:
            setattr(node, field, resolved[field].get(node.id, set()))
        existing = graph.get_node(node.id)
        if isinstance(existing, CodeNode) and _same_component(existing, node):
            continue  # re-analysis is identical to the graph: nothing changed
        _add_component_node(graph, node)
        result.updated.append(node.id)
        # Edge diff per relation type: remove stale edges, add missing ones.
        for field, relation in _RELATION_FIELD_EDGES:
            wanted = getattr(node, field) - {node.id}
            existing_edges = {
                t for t, rt, _w in graph.get_out_edges(node.id) if rt == relation
            }
            for t in sorted(existing_edges - wanted):
                if graph.remove_edge(node.id, t):
                    result.removed_edges.append((node.id, t))
            for t in sorted(wanted - existing_edges):
                if graph.has_node(t):
                    graph.add_edge(node.id, t, relation, 1.0)
                    result.added_edges.append((node.id, t))
        logger.debug("Updated component %s", node.id)

    # --- Module placement for new components ---
    for node in new_nodes:
        if node.id in graph_nodes:
            continue
        module_id = _place_new_component(graph, node)
        if module_id and module_id not in result.modules_added:
            result.modules_added.append(module_id)

    # Only rebuild module edges for single-file callers; batch callers pass
    # rebuild_modules=False and rebuild once after the whole batch.
    if rebuild_modules:
        rebuild_cross_module_edges(graph)
    return result


def _add_component_node(graph: HeterogeneousGraph, node: Node) -> None:
    """Add or overwrite a CodeNode in the graph (mirrors build_graph_from_analysis)."""
    graph.add_code_node(
        CodeNode(
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
    )


def _same_component(existing: CodeNode, node: Node) -> bool:
    """True when the graph node already carries the same component content.

    Line numbers are deliberately excluded: a line shift alone is not a
    content change (and would mark every moved component as updated)."""
    if (
        existing.source_code != node.source_code
        or existing.docstring != (node.docstring or "")
        or existing.parameters != node.parameters
        or existing.receiver != node.receiver
        or existing.owner != node.owner
        or existing.relative_path != node.relative_path
    ):
        return False
    for field, _relation in _RELATION_FIELD_EDGES:
        if sorted(getattr(existing, field, None) or []) != sorted(
            getattr(node, field, None) or set()
        ):
            return False
    existing_unresolved = {
        key: sorted(refs)
        for key, refs in (existing.unresolved or {}).items()
        if refs
    }
    node_unresolved = {
        key: sorted(refs) for key, refs in (node.unresolved or {}).items() if refs
    }
    if existing_unresolved != node_unresolved:
        return False
    return True