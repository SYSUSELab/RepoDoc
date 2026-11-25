"""Tree-sitter graph builder with pluggable language analyzers.

The builder owns graph IDs, file records, import/call edges and comment
assignment. Language-specific node names and import rules live behind
``LanguageAnalyzer`` so new languages do not fork the pipeline.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Iterable

from tree_sitter import Parser

from repodoc.graph.languages.base import FileInfo, LanguageAnalyzer
from repodoc.graph.languages.python import PythonAnalyzer
from repodoc.graph.store import GraphStore
from repodoc.schema import file_node_id, sha256_file

_SKIP_PARTS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".repodoc"}


class CodeGraphBuilder:
    """Build a repository graph using one language analyzer.

    ``PythonGraphBuilder`` remains the compatibility name for the default
    Python analyzer. New languages call this class with their own analyzer.
    """

    skip_parts = set(_SKIP_PARTS)

    def __init__(
        self,
        repo_path: str | Path,
        *,
        repo_name: str | None = None,
        exclude_dirs: Iterable[str | Path] = (),
        analyzer: LanguageAnalyzer | None = None,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.repo_name = repo_name or self.repo_path.name
        self.exclude_dirs = [Path(p).resolve() for p in exclude_dirs]
        self.analyzer = analyzer or PythonAnalyzer()
        self.graph = GraphStore(
            repo=self.repo_name,
            repo_path=str(self.repo_path),
            commit=git_head(self.repo_path),
        )
        self.file_infos: dict[str, FileInfo] = {}
        self.parser = Parser(self.analyzer.language)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def build(self) -> GraphStore:
        for path in self.analyzer.iter_files(
            self.repo_path, self.skip_parts, self.exclude_dirs
        ):
            self._parse_file(path)
        for info in self.file_infos.values():
            self._resolve_imports(info)
        for info in self.file_infos.values():
            self._resolve_calls(info)
        self._assign_comments()
        return self.graph

    # ------------------------------------------------------------------
    # File discovery and parsing
    # ------------------------------------------------------------------
    def _parse_file(self, path: Path) -> None:
        rel_path = path.relative_to(self.repo_path).as_posix()
        source_bytes = path.read_bytes()
        source = source_bytes.decode("utf-8", errors="replace")
        tree = self.parser.parse(source_bytes)
        root = tree.root_node

        file_id = file_node_id(rel_path)
        comments = self.analyzer.comments(source_bytes, root, rel_path)
        file_attrs = {
            "path": rel_path,
            "language": self.analyzer.name,
            "hash": sha256_file(path),
            "loc": len(source.splitlines()),
            "comments": comments,
            **self.analyzer.file_metadata(source_bytes, root, rel_path),
        }
        self.graph.add_node("file", file_id, file_attrs)
        info = FileInfo(
            rel_path=rel_path,
            node_id=file_id,
            path=path,
            language=self.analyzer.name,
            comments=comments,
        )
        self.file_infos[rel_path] = info
        info.imports = self.analyzer.parse_imports(source_bytes, root, rel_path)
        self.analyzer.visit_definitions(
            self,
            root,
            source_bytes,
            rel_path,
            file_id,
            [],
            [],
            info,
        )

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------
    def _resolve_imports(self, info: FileInfo) -> None:
        for spec in info.imports:
            target = self.analyzer.resolve_module(self, info.rel_path, spec.module)
            if target is None:
                continue
            self.graph.add_edge(
                "imports",
                info.node_id,
                self.file_infos[target].node_id,
                {
                    "module": spec.module,
                    "names": [spec.name] if spec.name else [],
                    "locations": [spec.location],
                },
            )
            if spec.name:
                alias = spec.alias or spec.name
                info.import_aliases[alias] = (target, spec.name)
            else:
                alias = spec.alias or spec.module.split(".")[0].split("/")[-1]
                info.import_aliases[alias] = (target, None)

    # ------------------------------------------------------------------
    # Call resolution
    # ------------------------------------------------------------------
    def _find_symbol(
        self, rel_path: str, name: str, prefer_kind: str | None = None
    ) -> str | None:
        info = self.file_infos.get(rel_path)
        if info is None:
            return None
        candidates = info.symbols.get(name, [])
        if not candidates:
            return None
        if prefer_kind:
            for candidate in candidates:
                if self.graph.nodes[candidate]["attrs"].get("kind") == prefer_kind:
                    return candidate
        return candidates[0]

    def _resolve_calls(self, info: FileInfo) -> None:
        for call in info.calls:
            target_id = self.analyzer.resolve_callee(self, info, call)
            if target_id is None:
                continue
            self.graph.add_edge(
                "calls",
                call.source_symbol_id,
                target_id,
                {"locations": [call.location]},
            )

    # ------------------------------------------------------------------
    # Comments
    # ------------------------------------------------------------------
    def _assign_comments(self) -> None:
        comments_by_file: dict[str, list[dict[str, Any]]] = {
            rel: info.comments for rel, info in self.file_infos.items()
        }
        for node in self.graph.nodes.values():
            if node["type"] != "symbol":
                continue
            attrs = node["attrs"]
            path = attrs.get("path")
            start = attrs.get("start", {}).get("line")
            end = attrs.get("end", {}).get("line")
            if not path or start is None or end is None:
                continue
            attrs["comments"] = [
                comment
                for comment in comments_by_file.get(path, [])
                if start <= comment["start_line"] and comment["end_line"] <= end
            ]


class PythonGraphBuilder(CodeGraphBuilder):
    """Backward-compatible Python builder used by existing pipelines/tests."""

    def __init__(
        self,
        repo_path: str | Path,
        *,
        repo_name: str | None = None,
        exclude_dirs: Iterable[str | Path] = (),
    ) -> None:
        super().__init__(
            repo_path,
            repo_name=repo_name,
            exclude_dirs=exclude_dirs,
            analyzer=PythonAnalyzer(),
        )


def git_head(repo_path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None
