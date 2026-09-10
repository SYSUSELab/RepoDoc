"""
Repository knowledge graph: models, heterogeneous graph, and graph building.

Public API: Node/CallRelationship (analysis), CodeNode/DocNode/ModuleNode and
relation constants (graph), HeterogeneousGraph, build_graph_from_analysis,
DependencyGraphBuilder.

NOTE: analyzers/ is imported lazily on first use (tree-sitter syntax packages
are heavy); builder.py dispatches through its registry.
"""

from repodoc.graph.builder import (
    DependencyGraphBuilder,
    build_graph_from_analysis,
)
from repodoc.graph.graph import HeterogeneousGraph
from repodoc.graph.models import (
    REL_CALLS,
    REL_DESCRIBES,
    REL_IMPLEMENTS,
    RELATION_TYPES,
    CallRelationship,
    CodeNode,
    DocNode,
    Node,
)

__all__ = [
    "DependencyGraphBuilder",
    "build_graph_from_analysis",
    "HeterogeneousGraph",
    "CodeNode",
    "DocNode",
    "Node",
    "CallRelationship",
    "REL_CALLS",
    "REL_DESCRIBES",
    "REL_IMPLEMENTS",
    "RELATION_TYPES",
]
