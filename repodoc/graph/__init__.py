"""
Repository knowledge graph: models, heterogeneous graph, and graph building.

Public API: Node/CallRelationship (analysis), CodeNode/DocNode/ConceptNode and
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
    REL_SEMANTIC_IMPACT,
    RELATION_TYPES,
    CallRelationship,
    CodeNode,
    ConceptNode,
    DocNode,
    Node,
)

__all__ = [
    "DependencyGraphBuilder",
    "build_graph_from_analysis",
    "HeterogeneousGraph",
    "CodeNode",
    "ConceptNode",
    "DocNode",
    "Node",
    "CallRelationship",
    "REL_CALLS",
    "REL_DESCRIBES",
    "REL_IMPLEMENTS",
    "REL_SEMANTIC_IMPACT",
    "RELATION_TYPES",
]
