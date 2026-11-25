"""Code-layer graph diff used by incremental update."""

from __future__ import annotations

from typing import Any

from repodoc.graph.store import GraphStore


def _code_nodes(graph: GraphStore) -> dict[str, dict[str, Any]]:
    return {
        node_id: node
        for node_id, node in graph.nodes.items()
        if node["layer"] == "code"
    }


def _code_edges(graph: GraphStore, code_ids: set[str]) -> dict[str, dict[str, Any]]:
    return {
        edge_id: edge
        for edge_id, edge in graph.edges.items()
        if edge["src"] in code_ids and edge["dst"] in code_ids
    }


def diff_code_graphs(old: GraphStore, new: GraphStore) -> dict[str, Any]:
    old_nodes = _code_nodes(old)
    new_nodes = _code_nodes(new)
    old_ids = set(old_nodes)
    new_ids = set(new_nodes)
    modified = sorted(
        node_id
        for node_id in old_ids & new_ids
        if old_nodes[node_id].get("attrs") != new_nodes[node_id].get("attrs")
    )
    old_edges = _code_edges(old, old_ids)
    new_edges = _code_edges(new, new_ids)
    old_edge_ids = set(old_edges)
    new_edge_ids = set(new_edges)
    modified_edges = sorted(
        edge_id
        for edge_id in old_edge_ids & new_edge_ids
        if old_edges[edge_id].get("attrs") != new_edges[edge_id].get("attrs")
    )
    # A changed edge is represented as remove+add for the delta contract.
    edges_removed = sorted((old_edge_ids - new_edge_ids) | set(modified_edges))
    edges_added = sorted((new_edge_ids - old_edge_ids) | set(modified_edges))
    return {
        "nodes_added": sorted(new_ids - old_ids),
        "nodes_removed": sorted(old_ids - new_ids),
        "nodes_modified": modified,
        "edges_added": edges_added,
        "edges_removed": edges_removed,
    }


def changed_paths(delta: dict[str, Any], old: GraphStore, new: GraphStore) -> set[str]:
    paths: set[str] = set()
    for key in ("nodes_added", "nodes_removed", "nodes_modified"):
        for node_id in delta.get(key, []):
            node = new.nodes.get(node_id) or old.nodes.get(node_id)
            if node is not None and node["attrs"].get("path"):
                paths.add(node["attrs"]["path"])
    return paths
