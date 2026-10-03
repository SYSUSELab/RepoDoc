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
from repodoc.graph.models import (
    REL_CALLS,
    REL_CONTAINS,
    REL_DEPENDS_ON,
    REL_IMPLEMENTS,
    CodeNode,
    ModuleNode,
    Node,
)

logger = logging.getLogger(__name__)


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


def _build_lookup(
    graph: HeterogeneousGraph, new_nodes: List[Node]
) -> Dict[str, str]:
    """Inverted index: component id/name/last-segment -> canonical graph id.

    New-file nodes win over graph nodes on name collisions (they are the
    freshest analysis of the changed code)."""
    lookup: Dict[str, str] = {}
    for node in new_nodes:
        lookup.setdefault(node.id, node.id)
        lookup.setdefault(node.name, node.id)
        lookup.setdefault(node.id.split(".")[-1], node.id)
    for nid, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "code":
            continue
        d = attrs.get("data", {})
        cid = d.get("id", nid)
        lookup.setdefault(cid, cid)
        name = d.get("name")
        if name:
            lookup.setdefault(name, cid)
            lookup.setdefault(cid.split(".")[-1], cid)
    return lookup


def _resolve_dependencies(
    graph: HeterogeneousGraph, new_nodes: List[Node], relationships: list
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Resolve relationships into ({node_id: call targets}, {node_id: base ids}).

    Only targets that exist in the graph (or among the new nodes) become
    edges — unresolved external references are dropped (there is no node to
    point at). Inheritance/implements relations are kept in a separate map
    so they become REL_IMPLEMENTS edges instead of REL_CALLS."""
    lookup = _build_lookup(graph, new_nodes)
    deps: Dict[str, Set[str]] = {n.id: set() for n in new_nodes}
    impls: Dict[str, Set[str]] = {n.id: set() for n in new_nodes}
    for rel in relationships:
        caller = lookup.get(rel.caller)
        if caller is None or caller not in deps:
            continue
        callee = lookup.get(rel.callee)
        if callee is None and "." in rel.callee:
            callee = lookup.get(rel.callee.split(".")[-1])
        if callee is None or not (callee in deps or graph.has_node(callee)):
            continue
        if getattr(rel, "relation_type", REL_CALLS) == REL_IMPLEMENTS:
            impls[caller].add(callee)
        else:
            deps[caller].add(callee)
    return deps, impls


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
    """Deepest existing module whose path matches the component's location.

    Module paths are real locations — either the member directory
    ('src/atcher/core') or a single-file module's path without extension
    ('src/atcher/config' for config.py). A component matches when its file
    path without extension equals the module path, or its directory equals
    (or nests under) it. Returns None when nothing matches."""
    rel = node.relative_path or ""
    rel_dir = os.path.dirname(rel)
    rel_noext = rel.rsplit(".", 1)[0] if "." in os.path.basename(rel) else rel
    if not rel_dir and not rel_noext:
        return None
    best_id: Optional[str] = None
    best_len = -1
    for nid, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "module":
            continue
        d = attrs.get("data", {})
        mpath = str(d.get("path") or "").strip().removeprefix("./").rstrip("/")
        if not mpath or mpath == ".":
            continue
        matched = (
            rel_noext == mpath
            or (rel_dir and (rel_dir == mpath or rel_dir.startswith(mpath + "/")))
        )
        if matched and len(mpath) > best_len:
            best_id = d.get("id", nid)
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
    """Drop a going-away node from its callers'/implementors' stored lists.

    Removing the node drops its incident edges, but the referencing nodes
    (usually in files that were not re-analyzed) keep it in depends_on or
    implements — a ghost reference that would otherwise leak into their
    prompts and docs. Returns the referrer ids that were rewritten."""
    callers = {
        s
        for s, t, rt, _w in graph.edges()
        if t == node_id and s != node_id and rt in (REL_CALLS, REL_IMPLEMENTS)
    }
    for cid in sorted(callers):
        node = graph.get_node(cid)
        if not isinstance(node, CodeNode):
            continue
        deps = node.depends_on or []
        impls = node.implements or []
        kept_deps = [d for d in deps if d != node_id]
        kept_impls = [b for b in impls if b != node_id]
        if len(kept_deps) != len(deps) or len(kept_impls) != len(impls):
            data = node.model_dump()
            data["depends_on"] = kept_deps
            data["implements"] = kept_impls
            graph.add_code_node(CodeNode.model_validate(data))
    return sorted(callers)


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
        targets = set(data.get("depends_on", [])) | set(data.get("implements", []))
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

    Returns a GraphMutationResult summarizing added/updated/removed nodes
    and edges. Callers must persist the graph afterwards."""
    result = GraphMutationResult()

    # Analyzers derive module paths from (file_path, repo_path); pass the
    # absolute path so relative_path/ids match the full-repo analysis.
    analyze_path = file_path
    if not os.path.isabs(file_path) and repo_path:
        analyze_path = os.path.join(repo_path, file_path)

    new_nodes: List[Node] = []
    new_rels: list = []
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
            new_nodes, new_rels = analyze_file(analyze_path, new_content, repo_path)
        except Exception as e:
            logger.warning("Failed to analyze %s: %s", file_path, e)
            new_nodes, new_rels = [], []
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
    # empty depends_on (builder fills them repo-wide), so the resolved maps
    # are what gets persisted on the mutated CodeNodes.
    deps, impls = _resolve_dependencies(graph, new_nodes, new_rels)

    # --- Removed: in the graph for this file, not in the new analysis ---
    for nid in sorted(set(graph_nodes) - new_ids):
        for s, t, rt, _w in list(graph.edges()):
            if rt in (REL_CALLS, REL_IMPLEMENTS):
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
        node.depends_on = deps.get(node.id, set())
        node.implements = impls.get(node.id, set())
        _add_component_node(graph, node)
        for callee in sorted(node.depends_on - {node.id}):
            if graph.has_node(callee):
                graph.add_edge(node.id, callee, REL_CALLS, 1.0)
                result.added_edges.append((node.id, callee))
        for base in sorted(node.implements - {node.id}):
            if graph.has_node(base):
                graph.add_edge(node.id, base, REL_IMPLEMENTS, 1.0)
                result.added_edges.append((node.id, base))
        result.added.append(node.id)
        logger.info("Added component %s", node.id)

    # --- Updated: present in both, content OR resolved edges differ ---
    for node in new_nodes:
        if node.id not in graph_nodes:
            continue
        # Resolve first: a new/changed definition elsewhere in the same file
        # can change this node's edges even when its own source is identical.
        node.depends_on = deps.get(node.id, set())
        node.implements = impls.get(node.id, set())
        existing = graph.get_node(node.id)
        if isinstance(existing, CodeNode) and _same_component(existing, node):
            continue  # re-analysis is identical to the graph: nothing changed
        _add_component_node(graph, node)
        result.updated.append(node.id)
        # Edge diff per relation type: remove stale edges, add missing ones.
        for relation, wanted_set in (
            (REL_CALLS, deps.get(node.id, set())),
            (REL_IMPLEMENTS, impls.get(node.id, set())),
        ):
            wanted = wanted_set - {node.id}
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
            parameters=node.parameters,
            receiver=node.receiver,
        )
    )


def _same_component(existing: CodeNode, node: Node) -> bool:
    """True when the graph node already carries the same component content.

    Line numbers are deliberately excluded: a line shift alone is not a
    content change (and would mark every moved component as updated)."""
    return (
        existing.source_code == node.source_code
        and existing.docstring == (node.docstring or "")
        and existing.parameters == node.parameters
        and existing.receiver == node.receiver
        and existing.relative_path == node.relative_path
        and sorted(existing.implements or []) == sorted(node.implements or set())
        and sorted(existing.depends_on or []) == sorted(node.depends_on or set())
    )