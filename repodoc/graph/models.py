"""
Data models for the repository knowledge graph.

Two families live here:
- Analysis models: Node (code component), CallRelationship (parser output).
- Graph models: CodeNode / DocNode / ConceptNode and relation constants
  (nodes and typed edges stored in the HeterogeneousGraph).

Note: CodeNode.type (KG) and Node.component_type (analysis) carry the same
kind of information ("function"/"class"/...) but are kept as separate fields —
the mapping happens in graph/builder.py.
"""

from typing import List, Optional, Set

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Analysis models
# ---------------------------------------------------------------------------


class Node(BaseModel):
    """A code component (function/class/struct/...) extracted by an analyzer."""

    id: str
    name: str
    component_type: str
    file_path: str
    relative_path: str
    depends_on: Set[str] = set()
    source_code: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    docstring: str = ""
    parameters: Optional[List[str]] = None
    # Go/Rust method receiver type (e.g. "User" in func (u *User) Name()); None for others
    receiver: Optional[str] = None


class CallRelationship(BaseModel):
    caller: str
    callee: str
    call_line: Optional[int] = None
    is_resolved: bool = False


# ---------------------------------------------------------------------------
# Knowledge graph models
# ---------------------------------------------------------------------------

# Relation type constants for graph edges (design: calls, implements, describes, semantic_impact).
REL_CALLS = "calls"
REL_IMPLEMENTS = "implements"
REL_DESCRIBES = "describes"
REL_SEMANTIC_IMPACT = "semantic_impact"

RELATION_TYPES = (REL_CALLS, REL_IMPLEMENTS, REL_DESCRIBES, REL_SEMANTIC_IMPACT)


class CodeNode(BaseModel):
    """Code entity: file, class, or function. Maps from analysis Node in Step 2."""

    id: str
    type: str  # e.g. "function", "class", "file"
    file_path: str
    source_code: Optional[str] = None
    name: Optional[str] = None
    relative_path: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    docstring: str = ""


class DocNode(BaseModel):
    """Document fragment: module doc, API doc, or architecture diagram."""

    id: str
    content: str
    format: str  # e.g. "markdown", "mermaid"
    version: int = 1
    file_path: Optional[str] = None


class ConceptNode(BaseModel):
    """Business concept extracted from comments, naming, or commit messages."""

    id: str
    concept: str
    context: str = ""
    confidence: float = 1.0  # 0.0 to 1.0
