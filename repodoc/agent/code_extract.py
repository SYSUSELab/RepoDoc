"""Extract code components from specific git commits.

Split out of update.py. NOTE: extraction uses the stdlib ``ast`` module,
so it only supports Python files; other languages fall back to the
working tree at regeneration time.
"""

import logging
import os
import subprocess
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Git code extraction (Python-only by design)
# ---------------------------------------------------------------------------


class GitCodeExtractor:
    """Extract code components from specific git commits.

    NOTE: extraction uses the stdlib `ast` module, so it only supports Python
    files. For other languages, regeneration falls back to the working tree.
    """

    def __init__(self, repo_path: str) -> None:
        self.repo_path = os.path.abspath(repo_path)

    def get_file_content_at_commit(self, file_path: str, commit: str) -> Optional[str]:
        """Get file content at a specific commit (None if not present)."""
        try:
            if os.path.isabs(file_path):
                file_path = os.path.relpath(file_path, self.repo_path)
            result = subprocess.run(
                ["git", "show", f"{commit}:{file_path}"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout if result.returncode == 0 else None
        except Exception as e:
            logger.warning(f"Failed to get file content at {commit}: {e}")
            return None

    def extract_component_at_commit(
        self, file_path: str, component_name: str, commit: str
    ) -> Optional[dict[str, Any]]:
        """Extract a Python component's code at a given commit."""
        content = self.get_file_content_at_commit(file_path, commit)
        if not content:
            return None

        try:
            import ast

            tree = ast.parse(content)
        except SyntaxError:
            logger.warning(f"Failed to parse {file_path} at {commit}")
            return None

        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name != component_name:
                    continue
                source_code = ast.get_source_segment(content, node)
                if source_code is None:
                    lines = content.split("\n")
                    source_code = "\n".join(
                        lines[node.lineno - 1 : node.end_lineno]
                    )
                params = []
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if getattr(node, "args", None):
                        params = [arg.arg for arg in node.args.args]
                        params.extend(arg.arg for arg in node.args.kwonlyargs)
                return {
                    "source_code": source_code or "",
                    "docstring": ast.get_docstring(node) or "",
                    "parameters": params,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "component_type": (
                        "class" if isinstance(node, ast.ClassDef) else "function"
                    ),
                }
        return None


