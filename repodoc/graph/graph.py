"""
Heterogeneous knowledge graph: CodeNode, DocNode, ModuleNode with typed edges.

Backed by NetworkX DiGraph; supports add/query nodes and edges, and JSON save/load.
"""

import json
from pathlib import Path
from typing import Any, Literal

import networkx as nx

from repodoc.graph.models import (
    RELATION_TYPES,
    CodeNode,
    DocNode,
    ModuleNode,
)

NodeType = Literal["code", "doc", "module"]
NodeLike = CodeNode | DocNode | ModuleNode


class HeterogeneousGraph:
    """
    Heterogeneous graph over code, doc, and module nodes with typed, weighted edges.

    Node id is a global string. Edges carry relation_type and weight for propagation.
    """

    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()

    def add_code_node(self, node: CodeNode) -> None:
        """Add or overwrite a CodeNode by id."""
        self._g.add_node(
            node.id,
            node_type="code",
            data=node.model_dump(),
        )

    def add_doc_node(self, node: DocNode) -> None:
        """Add or overwrite a DocNode by id."""
        self._g.add_node(
            node.id,
            node_type="doc",
            data=node.model_dump(),
        )

    def add_module_node(self, node: ModuleNode) -> None:
        """Add or overwrite a ModuleNode by id."""
        self._g.add_node(
            node.id,
            node_type="module",
            data=node.model_dump(),
        )

    def get_node(self, node_id: str) -> CodeNode | DocNode | ModuleNode | None:
        """Return the node as CodeNode, DocNode, or ModuleNode; None if missing."""
        if not self._g.has_node(node_id):
            return None
        attrs = self._g.nodes[node_id]
        node_type = attrs.get("node_type")
        data = attrs.get("data", {})
        if node_type == "code":
            return CodeNode.model_validate(data)
        if node_type == "doc":
            return DocNode.model_validate(data)
        if node_type == "module":
            return ModuleNode.model_validate(data)
        return None

    def has_node(self, node_id: str) -> bool:
        """Return True if the graph contains this node id."""
        return self._g.has_node(node_id)

    def get_nodes_by_type(
        self,
        node_type: NodeType,
    ) -> list[CodeNode | DocNode | ModuleNode]:
        """Return all nodes of the given type (code, doc, or module)."""
        out: list[CodeNode | DocNode | ModuleNode] = []
        for nid, attrs in self._g.nodes(data=True):
            if attrs.get("node_type") != node_type:
                continue
            node = self.get_node(nid)
            if node is not None:
                out.append(node)
        return out

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> None:
        """Add a directed edge. Source and target nodes must exist. Overwrites if edge exists."""
        if relation_type not in RELATION_TYPES:
            raise ValueError(
                f"relation_type must be one of {RELATION_TYPES}, got {relation_type!r}"
            )
        if not self._g.has_node(source_id):
            raise ValueError(f"source node not found: {source_id}")
        if not self._g.has_node(target_id):
            raise ValueError(f"target node not found: {target_id}")
        self._g.add_edge(
            source_id, target_id, relation_type=relation_type, weight=weight
        )

    def get_out_edges(self, node_id: str) -> list[tuple[str, str, float]]:
        """
        Return all out-edges from the node as (target_id, relation_type, weight).
        Required for the semantic impact propagation algorithm.
        """
        if not self._g.has_node(node_id):
            return []
        out: list[tuple[str, str, float]] = []
        for _s, t, attrs in self._g.out_edges(node_id, data=True):
            rt = attrs.get("relation_type", "calls")
            w = attrs.get("weight", 1.0)
            out.append((t, rt, w))
        return out

    def node_count(self) -> int:
        """Total number of nodes."""
        return self._g.number_of_nodes()

    def edge_count(self) -> int:
        """Total number of edges."""
        return self._g.number_of_edges()

    def successors(self, node_id: str) -> list[str]:
        """Return list of successor node IDs (outgoing edges)."""
        if not self._g.has_node(node_id):
            return []
        return list(self._g.successors(node_id))

    def predecessors(self, node_id: str) -> list[str]:
        """Return list of predecessor node IDs (incoming edges)."""
        if not self._g.has_node(node_id):
            return []
        return list(self._g.predecessors(node_id))

    def nodes(self, data: bool = False):
        """Return nodes as iterator. If data=True, yields (node_id, attrs) pairs."""
        return self._g.nodes(data=data)

    def get_edge_data(self, source_id: str, target_id: str) -> dict | None:
        """Return edge attributes as dict, or None if edge doesn't exist."""
        if not self._g.has_edge(source_id, target_id):
            return None
        return dict(self._g.edges[source_id, target_id])

    def has_edge(self, source_id: str, target_id: str) -> bool:
        """Return True if a directed edge exists between the two nodes."""
        return self._g.has_edge(source_id, target_id)

    def edges(self, relation_type: str | None = None) -> list[tuple[str, str, str, float]]:
        """Return all edges as (source_id, target_id, relation_type, weight).

        When relation_type is given, only edges of that type are returned.
        Used by the incremental updater to diff/rebuild typed edge sets."""
        out: list[tuple[str, str, str, float]] = []
        for s, t, attrs in self._g.edges(data=True):
            rt = attrs.get("relation_type", "calls")
            if relation_type is not None and rt != relation_type:
                continue
            out.append((s, t, rt, attrs.get("weight", 1.0)))
        return out

    def remove_edge(self, source_id: str, target_id: str) -> bool:
        """Remove a single edge. Returns True when it existed and was removed."""
        if not self._g.has_edge(source_id, target_id):
            return False
        self._g.remove_edge(source_id, target_id)
        return True

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and ALL incident edges (incoming and outgoing).

        Returns True when the node existed and was removed. Used by the
        incremental updater to delete removed components (and their calls /
        describes / contains edges) from the knowledge graph."""
        if not self._g.has_node(node_id):
            return False
        self._g.remove_node(node_id)
        return True

    def save(self, path: str | Path) -> None:
        """Persist graph to a JSON file (nodes and edges)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nodes_payload: list[dict[str, Any]] = []
        for nid, attrs in self._g.nodes(data=True):
            nodes_payload.append(
                {
                    "id": nid,
                    "node_type": attrs.get("node_type", "code"),
                    "data": attrs.get("data", {}),
                }
            )
        edges_payload: list[dict[str, Any]] = []
        for s, t, attrs in self._g.edges(data=True):
            edges_payload.append(
                {
                    "source_id": s,
                    "target_id": t,
                    "relation_type": attrs.get("relation_type", "calls"),
                    "weight": attrs.get("weight", 1.0),
                }
            )
        doc = {"nodes": nodes_payload, "edges": edges_payload}
        with path.open("w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "HeterogeneousGraph":
        """Load graph from a JSON file produced by save()."""
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            doc = json.load(f)
        g = cls()
        for item in doc.get("nodes", []):
            nid = item["id"]
            node_type = item.get("node_type", "code")
            data = item.get("data", {})
            g._g.add_node(nid, node_type=node_type, data=data)
        for item in doc.get("edges", []):
            s = item["source_id"]
            t = item["target_id"]
            rt = item.get("relation_type", "calls")
            w = item.get("weight", 1.0)
            if g._g.has_node(s) and g._g.has_node(t):
                g._g.add_edge(s, t, relation_type=rt, weight=w)
        return g
