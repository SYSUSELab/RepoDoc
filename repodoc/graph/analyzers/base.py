"""Shared base class for all language analyzers.

Contract (unified across the 10 analyzers):
- Construction only stores state; parsing happens in `analyze()`.
  (Diff analysis constructs two analyzers for old/new versions and
  calls analyze() on each, so construction must be side-effect free.)
- `nodes` and `call_relationships` are the only outputs.
- Component ids use the dotted module path: `<relative/module/path>.<name>`
  (optionally `<path>.<parent>.<name>` for methods); `file_path` is the
  absolute path, `relative_path` is repo-relative.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from repodoc.graph.models import REL_CALLS, CallRelationship, Node


class BaseAnalyzer:
    #: File extensions that are stripped when computing the module path.
    EXTENSIONS: tuple[str, ...] = ()

    def __init__(self, file_path: str, content: str, repo_path: Optional[str] = None):
        self.file_path = Path(file_path)
        self.content = content
        self.repo_path = Path(repo_path) if repo_path else None
        self.nodes: List[Node] = []
        self.call_relationships: List[CallRelationship] = []
        #: Local import alias -> canonical dotted target. Filled by analyzers
        #: that parse import statements; used for import-aware reference
        #: resolution (ambiguous bare names become correct edges).
        self.imports: Dict[str, str] = {}
        self.top_level_nodes: dict[str, Node] = {}
        self._seen_relationships: set[tuple] = set()
        self._content_bytes = content.encode("utf8")

    def analyze(self) -> None:
        """Parse the content and populate nodes / call_relationships."""
        raise NotImplementedError

    # -- shared path helpers -------------------------------------------------

    def _get_relative_path(self) -> str:
        """Repo-relative path; falls back to the absolute path outside the repo."""
        if self.repo_path:
            try:
                return os.path.relpath(str(self.file_path), str(self.repo_path))
            except ValueError:
                return str(self.file_path)
        return str(self.file_path)

    def _get_module_path(self) -> str:
        """Dotted module path: relative path minus extension, / -> ."""
        rel = self._get_relative_path()
        for ext in self.EXTENSIONS:
            if rel.endswith(ext):
                rel = rel[: -len(ext)]
                break
        return rel.replace("/", ".").replace("\\", ".")

    def _get_component_id(self, name: str, parent: Optional[str] = None) -> str:
        module_path = self._get_module_path()
        if parent:
            return f"{module_path}.{parent}.{name}"
        return f"{module_path}.{name}" if module_path else name

    # -- shared tree-sitter helpers ------------------------------------------

    def _get_node_text(self, node) -> str:
        """Extract node text via byte slice (content is encoded once)."""
        return self._content_bytes[node.start_byte : node.end_byte].decode("utf8")

    def _find_child_by_type(self, node, node_type: str):
        """Find the first direct child of a given type."""
        for child in node.children:
            if child.type == node_type:
                return child
        return None

    def _add_relationship(
        self,
        caller: str,
        callee: str,
        call_line: Optional[int] = None,
        is_resolved: bool = False,
        relation_type: str = REL_CALLS,
    ) -> bool:
        """Append a relationship, deduplicating on
        (caller, callee, call_line, relation_type)."""
        key = (caller, callee, call_line, relation_type)
        if key in self._seen_relationships:
            return False
        self._seen_relationships.add(key)
        self.call_relationships.append(
            CallRelationship(
                caller=caller,
                callee=callee,
                call_line=call_line,
                is_resolved=is_resolved,
                relation_type=relation_type,
            )
        )
        return True
