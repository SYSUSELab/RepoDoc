"""Incremental graph/module-tree merge helpers for ``repodoc update``.

Split out of update.py. This module owns the data-level merge steps:
- extracting NEW components from the working tree (with cross-file
  depends_on resolution against the full component set)
- refreshing changed files' existing components and their calls edges
- removing DELETED components from the module tree and the graph, and
  marking their docs deprecated (file-level, no LLM)

Module assignment for new components lives in graph.cluster
(assign_new_components), orchestrated by update.run_update.
"""

import logging
import os
from pathlib import Path
from typing import Any

from repodoc.agent.diff import ChangeType, ComponentChange
from repodoc.agent.generate import _collect_component_modules
from repodoc.config import Config
from repodoc.graph.graph import HeterogeneousGraph
from repodoc.graph.models import REL_CALLS, CodeNode, Node

logger = logging.getLogger(__name__)


def _resolve_cross_file_depends(
    nodes: list[Node],
    rels: list[Any],
    known_components: dict[str, Node],
) -> None:
    """Merge depends_on into `nodes` from call relationships resolved against
    the FULL component set.

    _assemble_components only resolves callees present in the same file;
    cross-file calls (e.g. a new component calling an existing component)
    need the whole component map, so those edges are resolved here."""
    lookup: dict[str, str] = {}
    for cid, n in known_components.items():
        lookup[cid] = cid
        lookup[n.name] = cid
    for n in nodes:
        lookup[n.id] = n.id
        lookup[n.name] = n.id

    for rel in rels:
        caller_id = None
        for n in nodes:
            if (
                n.id == rel.caller
                or rel.caller == n.name
                or rel.caller.startswith(n.id)
            ):
                caller_id = n.id
                break
        if caller_id is None:
            continue
        callee_id = lookup.get(rel.callee)
        if not callee_id and "." in rel.callee:
            callee_id = lookup.get(rel.callee.split(".")[-1])
        if callee_id:
            for n in nodes:
                if n.id == caller_id:
                    n.depends_on.add(callee_id)
                    break


def _mark_docs_deprecated(
    docs_dir: str,
    module_tree: dict,
    components: dict[str, Node],
    removed_ids: list[str],
) -> list[str]:
    """Prepend a deprecation notice to the deleted components' doc files.

    File-level (no LLM call) — the component is gone, so a full regeneration
    buys nothing. Returns the touched file paths."""
    comp_modules = _collect_component_modules(module_tree)
    touched: list[str] = []
    for cid in removed_ids:
        node = components.get(cid)
        if not (node and node.name):
            continue
        chain = comp_modules.get(cid)
        if not chain:
            continue
        path = os.path.join(docs_dir, *chain, f"{node.name}.md")
        if not os.path.exists(path):
            continue
        try:
            content = Path(path).read_text(encoding="utf-8")
            if content.startswith("> ⚠️ **DEPRECATED**"):
                continue
            notice = (
                f"> ⚠️ **DEPRECATED**: `{cid}` was removed from the code base.\n"
                f"> This document is kept for historical reference.\n\n"
            )
            Path(path).write_text(notice + content, encoding="utf-8")
            touched.append(path)
        except Exception:
            continue
    return touched


def _remove_components_from_tree(
    graph: HeterogeneousGraph,
    module_tree: dict,
    components: dict[str, Node],
    removed_ids: list[str],
) -> set[str]:
    """Drop deleted components from the module tree and the graph (code
    nodes and every edge touching them). The graph reflects the CURRENT code
    state. Returns module ids whose membership changed (their docs should be
    regenerated)."""
    from repodoc.graph.models import ModuleNode

    removed = set(removed_ids)
    affected_modules: set[str] = set()

    def walk(nodes: dict, path: str) -> None:
        for name, info in nodes.items():
            cur = f"{path}/{name}" if path else name
            members = info.get("components", [])
            kept = [c for c in members if c not in removed]
            if len(kept) != len(members):
                info["components"] = kept
                affected_modules.add(cur)
            children = info.get("children")
            if isinstance(children, dict):
                walk(children, cur)

    walk(module_tree, "")

    for cid in removed:
        graph.remove_node(cid)  # node + all its edges
        components.pop(cid, None)

    # Mirror the updated membership into the graph's ModuleNodes.
    def find_info(nodes: dict, path: str, target: str) -> dict | None:
        for name, info in nodes.items():
            cur = f"{path}/{name}" if path else name
            if cur == target:
                return info
            children = info.get("children")
            if isinstance(children, dict):
                found = find_info(children, cur, target)
                if found is not None:
                    return found
        return None

    for mod_id in affected_modules:
        info = find_info(module_tree, "", mod_id)
        node = graph.get_node(mod_id)
        if info is not None and isinstance(node, ModuleNode):
            node.components = list(info.get("components", []))
            graph.add_module_node(node)

    return affected_modules


def _extract_new_component_nodes(
    change_map: dict[str, ComponentChange],
    config: Config,
    known_components: dict[str, Node],
) -> dict[str, Node]:
    """Parse the working tree to build full Nodes for NEW components.

    Uses the language analyzers (all supported languages — unlike
    GitCodeExtractor, which is Python-only). depends_on edges are resolved
    against the full component set (the new component may call existing
    components). Returns {component_id: Node}, keyed by the change_map id
    (the knowledge-graph namespace)."""
    from repodoc.graph.builder import _assemble_components, analyze_file

    new_nodes: dict[str, Node] = {}
    base = Path(config.repo_path)
    for cid, change in change_map.items():
        if change.change_type != ChangeType.NEW_COMPONENT:
            continue
        full_path = base / change.file_path
        try:
            content = full_path.read_text(encoding="utf-8")
        except Exception:
            continue
        try:
            nodes, rels = analyze_file(str(full_path), content, str(base))
        except Exception:
            continue
        # depends_on is merged in by _assemble_components (analyzer nodes
        # carry no dependency edges yet); cross-file calls resolved against
        # the full component set.
        assembled = _assemble_components(nodes, rels)
        _resolve_cross_file_depends(
            list(assembled.values()), rels, known_components
        )
        for n in assembled.values():
            if n.name == change.name:
                n.id = cid  # keep the diff's id (graph namespace)
                new_nodes[cid] = n
                break
    return new_nodes


def _refresh_changed_components(
    graph: HeterogeneousGraph,
    components: dict[str, Node],
    config: Config,
    changed_files: list[str],
) -> None:
    """Re-parse changed files and refresh existing components' fields and
    calls edges, so the graph reflects new code (new dependencies, changed
    source) without a full re-analysis. Components dict is mutated in place;
    graph code nodes and calls edges are refreshed."""
    from repodoc.graph.builder import _assemble_components, analyze_file

    for f in changed_files:
        full = f if f.startswith("/") else os.path.join(config.repo_path, f)
        if not os.path.exists(full):
            continue
        try:
            content = Path(full).read_text(encoding="utf-8")
        except Exception:
            continue
        try:
            nodes, rels = analyze_file(full, content, config.repo_path)
        except Exception:
            continue
        # depends_on edges are merged by _assemble_components, not the raw
        # analyzer output; cross-file calls resolve against the full set.
        assembled = list(_assemble_components(nodes, rels).values())
        _resolve_cross_file_depends(assembled, rels, components)
        fresh = {n.name: n for n in assembled}
        full_resolved = str(Path(full).resolve())
        for node_id, node in list(components.items()):
            try:
                in_file = str(Path(node.file_path or "").resolve()) == full_resolved
            except Exception:
                in_file = node.file_path == f
            if not in_file:
                continue
            new_node = fresh.get(node.name)
            if new_node is None:
                continue
            node.depends_on = new_node.depends_on
            node.source_code = new_node.source_code
            node.docstring = new_node.docstring or node.docstring
            node.parameters = new_node.parameters
            node.receiver = new_node.receiver
            # refresh graph node + calls edges
            graph.remove_out_edges(node_id, REL_CALLS)
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
                    parameters=node.parameters,
                    receiver=node.receiver,
                )
            )
            for callee in node.depends_on:
                if callee in components:
                    graph.add_edge(node_id, callee, REL_CALLS, 1.0)


def _merge_new_nodes_into_graph(
    graph: HeterogeneousGraph,
    new_nodes: dict[str, Node],
    components: dict[str, Node],
) -> None:
    """Add NEW components as code nodes with calls edges into the graph."""
    for nid, node in new_nodes.items():
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
                parameters=node.parameters,
                receiver=node.receiver,
            )
        )
        for callee in node.depends_on:
            if callee in components:
                graph.add_edge(nid, callee, REL_CALLS, 1.0)
