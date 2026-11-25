"""Graph store backed by NetworkX in memory and JSON/JSONL on disk."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from repodoc.schema import (
    EDGE_TYPES,
    NODE_TYPES,
    SCHEMA_VERSION,
    edge_id,
    layer_for_node,
    stable_hash,
    utc_now,
)


@dataclass
class GraphStore:
    repo: str
    repo_path: str
    commit: str | None = None
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: dict[str, dict[str, Any]] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add_node(
        self,
        node_type: str,
        node_id: str,
        attrs: dict[str, Any] | None = None,
        *,
        layer: str | None = None,
    ) -> str:
        if node_type not in NODE_TYPES:
            raise ValueError(f"unknown node type: {node_type}")
        attrs = dict(attrs or {})
        existing = self.nodes.get(node_id)
        if existing is not None:
            if existing["type"] != node_type:
                raise ValueError(
                    f"node id reused with different types: {node_id} "
                    f"({existing['type']} vs {node_type})"
                )
            existing["attrs"].update(attrs)
            return node_id
        self.nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "layer": layer or layer_for_node(node_type),
            "attrs": attrs,
        }
        return node_id

    def add_edge(
        self,
        edge_type: str,
        src: str,
        dst: str,
        attrs: dict[str, Any] | None = None,
    ) -> str:
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"unknown edge type: {edge_type}")
        eid = edge_id(edge_type, src, dst)
        attrs = dict(attrs or {})
        locations = list(attrs.pop("locations", []) or [])
        existing = self.edges.get(eid)
        if existing is not None:
            if locations:
                existing["attrs"].setdefault("locations", []).extend(locations)
                existing["attrs"]["count"] = len(existing["attrs"]["locations"])
            existing["attrs"].update(attrs)
            return eid
        if locations:
            attrs["locations"] = locations
            attrs["count"] = len(locations)
        self.edges[eid] = {
            "id": eid,
            "type": edge_type,
            "src": src,
            "dst": dst,
            "attrs": attrs,
        }
        return eid

    def remove_edge(self, edge_id: str) -> None:
        self.edges.pop(edge_id, None)

    def remove_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            return
        del self.nodes[node_id]
        for key in [
            key
            for key, edge in self.edges.items()
            if edge["src"] == node_id or edge["dst"] == node_id
        ]:
            del self.edges[key]

    # ------------------------------------------------------------------
    # Versions and serialization
    # ------------------------------------------------------------------
    @property
    def graph_version(self) -> str:
        payload = {
            "nodes": [self.nodes[key] for key in sorted(self.nodes)],
            "edges": [self.edges[key] for key in sorted(self.edges)],
        }
        return "g_" + stable_hash(payload)

    @property
    def code_graph_version(self) -> str:
        payload = {
            "nodes": [
                self.nodes[key]
                for key in sorted(self.nodes)
                if self.nodes[key]["layer"] == "code"
            ],
            "edges": [
                self.edges[key]
                for key in sorted(self.edges)
                if self.nodes.get(self.edges[key]["src"], {}).get("layer") == "code"
                and self.nodes.get(self.edges[key]["dst"], {}).get("layer") == "code"
            ],
        }
        return "g_" + stable_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "repo": self.repo,
            "repo_path": self.repo_path,
            "commit": self.commit,
            "graph_version": self.graph_version,
            "created_at": self.created_at,
            "nodes": [self.nodes[key] for key in sorted(self.nodes)],
            "edges": [self.edges[key] for key in sorted(self.edges)],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphStore":
        store = cls(
            repo=data["repo"],
            repo_path=data["repo_path"],
            commit=data.get("commit"),
            created_at=data.get("created_at") or utc_now(),
        )
        for node in data.get("nodes", []):
            store.nodes[node["id"]] = node
        for edge in data.get("edges", []):
            store.edges[edge["id"]] = edge
        return store

    def save(self, state_dir: Path) -> dict[str, str]:
        graph_dir = Path(state_dir) / "graph"
        graph_dir.mkdir(parents=True, exist_ok=True)
        nodes_path = graph_dir / "nodes.jsonl"
        edges_path = graph_dir / "edges.jsonl"
        graph_path = graph_dir / "graph.json"

        with nodes_path.open("w", encoding="utf-8") as f:
            for key in sorted(self.nodes):
                f.write(json.dumps(self.nodes[key], ensure_ascii=False) + "\n")
        with edges_path.open("w", encoding="utf-8") as f:
            for key in sorted(self.edges):
                f.write(json.dumps(self.edges[key], ensure_ascii=False) + "\n")
        graph_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return {
            "graph": str(graph_path),
            "nodes": str(nodes_path),
            "edges": str(edges_path),
        }

    @classmethod
    def load(cls, state_dir: Path) -> "GraphStore":
        graph_path = Path(state_dir) / "graph" / "graph.json"
        if not graph_path.is_file():
            raise FileNotFoundError(f"graph snapshot not found: {graph_path}")
        return cls.from_dict(json.loads(graph_path.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------
    # Views and queries
    # ------------------------------------------------------------------
    def to_networkx(self, *, include_doc_nodes: bool = True) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        for node in self.nodes.values():
            if not include_doc_nodes and node["layer"] == "document":
                continue
            graph.add_node(node["id"], **node)
        for edge in self.edges.values():
            if edge["src"] not in graph or edge["dst"] not in graph:
                continue
            graph.add_edge(edge["src"], edge["dst"], key=edge["id"], **edge)
        return graph

    def nodes_of_type(self, node_type: str) -> list[dict[str, Any]]:
        return [node for node in self.nodes.values() if node["type"] == node_type]

    def edges_of_type(self, edge_type: str) -> list[dict[str, Any]]:
        return [edge for edge in self.edges.values() if edge["type"] == edge_type]

    def outgoing(
        self, node_id: str, edge_types: Iterable[str] | None = None
    ) -> list[dict[str, Any]]:
        allowed = set(edge_types) if edge_types else None
        return [
            edge
            for edge in self.edges.values()
            if edge["src"] == node_id and (allowed is None or edge["type"] in allowed)
        ]

    def incoming(
        self, node_id: str, edge_types: Iterable[str] | None = None
    ) -> list[dict[str, Any]]:
        allowed = set(edge_types) if edge_types else None
        return [
            edge
            for edge in self.edges.values()
            if edge["dst"] == node_id and (allowed is None or edge["type"] in allowed)
        ]

    def summary(self) -> dict[str, Any]:
        node_types = Counter(node["type"] for node in self.nodes.values())
        edge_types = Counter(edge["type"] for edge in self.edges.values())
        return {
            "repo": self.repo,
            "graph_version": self.graph_version,
            "code_graph_version": self.code_graph_version,
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "node_types": dict(sorted(node_types.items())),
            "edge_types": dict(sorted(edge_types.items())),
        }
