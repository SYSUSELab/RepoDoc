"""
Data models for the repository knowledge graph.

Two families live here:
- Analysis models: Node (code component), CallRelationship (parser output).
- Graph models: CodeNode / DocNode / ModuleNode and relation constants
  (nodes and typed edges stored in the HeterogeneousGraph).

Note: CodeNode.type (KG) and Node.component_type (analysis) carry the same
kind of information ("function"/"class"/...) but are kept as separate fields —
the mapping happens in graph/builder.py.
"""

from typing import Dict, List, Optional, Set

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Relation type constants for graph edges
# (design: calls, implements, describes + module structure).
# ---------------------------------------------------------------------------
REL_CALLS = "calls"
REL_IMPLEMENTS = "implements"
# Type/reference relations (kept apart from calls so impact propagation and
# prompts can distinguish behavior from type usage).
REL_USES = "uses"  # field/property/parameter/type-annotation references
REL_INSTANTIATES = "instantiates"  # constructor invocation (new X())
REL_IMPORTS = "imports"  # import/use statements
REL_DESCRIBES = "describes"
# Module clustering edges: contains (module -> component / child module),
# depends_on (module -> module, cross-module call aggregation).
REL_CONTAINS = "contains"
REL_DEPENDS_ON = "depends_on"

# Relations that make the target affected by / relevant to the source (used by
# impact propagation and rename follow-through).
CODE_DEPENDENCY_RELATIONS = (
    REL_CALLS,
    REL_IMPLEMENTS,
    REL_USES,
    REL_INSTANTIATES,
)

RELATION_TYPES = (
    REL_CALLS,
    REL_IMPLEMENTS,
    REL_USES,
    REL_INSTANTIATES,
    REL_IMPORTS,
    REL_DESCRIBES,
    REL_CONTAINS,
    REL_DEPENDS_ON,
)

# Graph relation -> Node/CodeNode field that stores the resolved targets.
RELATION_NODE_FIELDS = {
    REL_CALLS: "depends_on",
    REL_IMPLEMENTS: "implements",
    REL_USES: "uses",
    REL_INSTANTIATES: "instantiates",
    REL_IMPORTS: "imports",
}


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
    # Base classes / implemented interfaces (resolved component ids). Kept
    # separate from depends_on so the graph can carry REL_IMPLEMENTS edges.
    implements: Set[str] = set()
    # Type references / instantiations / imports, resolved to component ids.
    uses: Set[str] = set()
    instantiates: Set[str] = set()
    imports: Set[str] = set()
    # References that could not be resolved when the file was analyzed,
    # keyed by relation type ("calls", "uses", ...). They stay on the graph
    # so a later update that introduces the missing definition can wire the
    # edge instead of losing it forever.
    unresolved: Dict[str, Set[str]] = {}
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
    # "calls" (default) or "implements" (extends/implements clauses).
    relation_type: str = REL_CALLS


# ---------------------------------------------------------------------------
# Knowledge graph models
# ---------------------------------------------------------------------------


class CodeNode(BaseModel):
    """Code entity: file, class, or function. Maps from analysis Node in Step 2.

    component_type mirrors type from the analysis model, and depends_on /
    parameters / receiver carry the FULL analysis fields, so the graph JSON
    is self-contained: it covers everything dependency_graph.json holds."""

    id: str
    type: str  # e.g. "function", "class", "file"
    component_type: str = ""  # same value as type (analysis model's field name)
    file_path: str
    source_code: Optional[str] = None
    name: Optional[str] = None
    relative_path: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    docstring: str = ""
    depends_on: List[str] = []  # call targets (resolved)
    implements: List[str] = []  # resolved base-class / interface component ids
    uses: List[str] = []  # resolved type references
    instantiates: List[str] = []  # resolved constructor targets
    imports: List[str] = []  # resolved import targets
    # Unresolved references by relation type; reconciled when the missing
    # definition appears in a later update.
    unresolved: Dict[str, List[str]] = {}
    parameters: Optional[List[str]] = None
    receiver: Optional[str] = None  # Go/Rust method receiver type


class DocNode(BaseModel):
    """Document fragment: module doc, API doc, or architecture diagram.

    The graph keeps only a SHA-256 of the doc content (content_hash) instead
    of the full text: the files on disk are the documents, the hash is all
    that manual-edit detection needs. `content` is legacy (graphs written
    before hashing); new/refreshed nodes leave it empty."""

    id: str
    content: str = ""
    content_hash: str = ""
    format: str  # e.g. "markdown", "mermaid"
    version: int = 1
    file_path: Optional[str] = None
    # True when the last update skipped this doc (manual edit detected):
    # subsequent updates keep retrying it until it is refreshed.
    stale: bool = False


class ModuleNode(BaseModel):
    """Clustered module from the LLM/heuristic clustering result (module tree).

    Member components are ALSO expressed as contains edges (the authoritative
    graph relation); the components field keeps the node self-contained so
    the graph JSON alone carries the full clustering result."""

    id: str  # module path in the tree, e.g. "core" or "net/http"
    name: str
    path: str = ""  # source directory, e.g. "src/atcher/core"
    parent: Optional[str] = None  # parent module id, None for top-level modules
    components: List[str] = []  # member component ids (mirrors module_tree)
