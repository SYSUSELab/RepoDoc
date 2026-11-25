"""Python tree-sitter analyzer."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Iterable

import tree_sitter_python
from tree_sitter import Language, Node

from repodoc.graph.languages.base import (
    CallSpec,
    FileInfo,
    ImportSpec,
    location_of,
    span_of,
    text_of,
    walk_nodes,
)
from repodoc.schema import api_node_id, symbol_node_id

_IMPORT_FROM_RE = re.compile(r"^from\s+([A-Za-z0-9_.]+)\s+import\s+(.+)$", re.S)
_IMPORT_RE = re.compile(r"^import\s+(.+)$", re.S)


def _docstring(source: bytes, body: Node | None) -> str | None:
    if body is None:
        return None
    for child in body.named_children:
        if child.type != "expression_statement":
            return None
        for sub in child.named_children:
            if sub.type != "string":
                continue
            raw = text_of(source, sub)
            try:
                value = ast.literal_eval(raw)
                return value if isinstance(value, str) else None
            except (ValueError, SyntaxError):
                return raw.strip()
        return None
    return None


def _signature(source: bytes, node: Node, body: Node | None) -> str:
    end = body.start_byte if body is not None else node.end_byte
    raw = source[node.start_byte : end].decode("utf-8", errors="replace").strip()
    lines: list[str] = []
    for line in raw.splitlines():
        line = re.split(r"\s+#", line, maxsplit=1)[0].rstrip()
        lines.append(line)
        if line.endswith(":"):
            break
    return "\n".join(lines).rstrip(":").strip()


def _clean_import_names(raw: str) -> list[tuple[str, str | None]]:
    raw = raw.strip()
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1]
    out: list[tuple[str, str | None]] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if " as " in piece:
            name, alias = piece.split(" as ", 1)
            out.append((name.strip(), alias.strip()))
        else:
            out.append((piece, None))
    return out


class PythonAnalyzer:
    name = "python"
    extensions = (".py",)
    language = Language(tree_sitter_python.language())

    def iter_files(
        self,
        repo_path: Path,
        skip_parts: set[str],
        exclude_dirs: Iterable[Path],
    ) -> list[Path]:
        excluded = [Path(item).resolve() for item in exclude_dirs]
        files: list[Path] = []
        for path in sorted(repo_path.rglob("*.py")):
            if not path.is_file():
                continue
            rel_parts = path.relative_to(repo_path).parts
            if any(part in skip_parts for part in rel_parts):
                continue
            resolved = path.resolve()
            if any(resolved.is_relative_to(item) for item in excluded):
                continue
            files.append(path)
        return files

    def comments(
        self, source: bytes, root: Node, rel_path: str
    ) -> list[dict[str, Any]]:
        return [
            {
                "text": text_of(source, node).strip(),
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
            }
            for node in walk_nodes(root)
            if node.type == "comment"
        ]

    def file_metadata(self, source: bytes, root: Node, rel_path: str) -> dict[str, Any]:
        return {"docstring": _docstring(source, root)}

    def parse_imports(
        self, source: bytes, root: Node, rel_path: str
    ) -> list[ImportSpec]:
        specs: list[ImportSpec] = []
        for node in walk_nodes(root):
            if node.type not in {"import_statement", "import_from_statement"}:
                continue
            text = text_of(source, node).strip()
            location = location_of(rel_path, node)
            from_match = _IMPORT_FROM_RE.match(text)
            if from_match:
                module = from_match.group(1)
                for name, alias in _clean_import_names(from_match.group(2)):
                    specs.append(ImportSpec(module, name, alias, location))
                continue
            import_match = _IMPORT_RE.match(text)
            if import_match:
                for name, alias in _clean_import_names(import_match.group(1)):
                    specs.append(ImportSpec(name, None, alias, location))
        return specs

    def visit_definitions(
        self,
        builder: Any,
        node: Node,
        source: bytes,
        rel_path: str,
        parent_id: str,
        class_stack: list[str],
        function_stack: list[str],
        info: FileInfo,
        decorators: list[str] | None = None,
    ) -> None:
        decorators = decorators or []
        if node.type == "decorated_definition":
            definition = node.child_by_field_name("definition")
            if definition is None:
                definition = next(
                    (
                        child
                        for child in node.named_children
                        if child.type.endswith("definition")
                    ),
                    None,
                )
            if definition is not None:
                decorator_texts = [
                    text_of(source, child).strip()
                    for child in node.named_children
                    if child.type == "decorator"
                ]
                self.visit_definitions(
                    builder,
                    definition,
                    source,
                    rel_path,
                    parent_id,
                    class_stack,
                    function_stack,
                    info,
                    decorator_texts,
                )
            return

        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            body = node.child_by_field_name("body")
            if name_node is None:
                return
            name = text_of(source, name_node)
            qualified = ".".join([*class_stack, *function_stack, name])
            symbol_id = symbol_node_id(rel_path, qualified, "class")
            builder.graph.add_node(
                "symbol",
                symbol_id,
                {
                    "name": name,
                    "qualified_name": qualified,
                    "kind": "class",
                    "path": rel_path,
                    "language": self.name,
                    "signature": _signature(source, node, body),
                    "source": text_of(source, node),
                    "docstring": _docstring(source, body),
                    "decorators": decorators,
                    "visibility": "public" if not name.startswith("_") else "private",
                    **span_of(node),
                },
            )
            builder.graph.add_edge("contains", parent_id, symbol_id)
            info.symbols.setdefault(name, []).append(symbol_id)
            if body is not None:
                for child in body.named_children:
                    self.visit_definitions(
                        builder,
                        child,
                        source,
                        rel_path,
                        symbol_id,
                        [*class_stack, name],
                        [],
                        info,
                    )
            return

        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            body = node.child_by_field_name("body")
            if name_node is None:
                return
            name = text_of(source, name_node)
            qualified = ".".join([*class_stack, *function_stack, name])
            kind = "method" if class_stack and not function_stack else "function"
            symbol_id = symbol_node_id(rel_path, qualified, kind)
            source_text = text_of(source, node)
            builder.graph.add_node(
                "symbol",
                symbol_id,
                {
                    "name": name,
                    "qualified_name": qualified,
                    "kind": kind,
                    "path": rel_path,
                    "language": self.name,
                    "signature": _signature(source, node, body),
                    "source": source_text,
                    "docstring": _docstring(source, body),
                    "decorators": decorators,
                    "is_async": source_text.lstrip().startswith("async "),
                    "visibility": "public" if not name.startswith("_") else "private",
                    **span_of(node),
                },
            )
            builder.graph.add_edge("contains", parent_id, symbol_id)
            info.symbols.setdefault(name, []).append(symbol_id)

            if not name.startswith("_") and not (
                name.startswith("__") and name.endswith("__")
            ):
                api_id = api_node_id(rel_path, qualified)
                builder.graph.add_node(
                    "api",
                    api_id,
                    {
                        "name": name,
                        "qualified_name": qualified,
                        "kind": kind,
                        "language": self.name,
                        "signature": _signature(source, node, body),
                        "symbol_id": symbol_id,
                        "path": rel_path,
                        "visibility": "public",
                    },
                )
                builder.graph.add_edge("exposes_api", info.node_id, api_id)
                builder.graph.add_edge("handles", api_id, symbol_id)

            if body is not None:
                self._collect_calls(builder, body, source, rel_path, symbol_id, info)
                for child in body.named_children:
                    self.visit_definitions(
                        builder,
                        child,
                        source,
                        rel_path,
                        symbol_id,
                        class_stack,
                        [*function_stack, name],
                        info,
                    )
            return

        for child in node.named_children:
            self.visit_definitions(
                builder,
                child,
                source,
                rel_path,
                parent_id,
                class_stack,
                function_stack,
                info,
            )

    def _collect_calls(
        self,
        builder: Any,
        node: Node,
        source: bytes,
        rel_path: str,
        source_symbol_id: str,
        info: FileInfo,
    ) -> None:
        if node.type in {
            "function_definition",
            "class_definition",
            "decorated_definition",
        }:
            return
        if node.type == "call":
            function = node.child_by_field_name("function")
            if function is not None:
                info.calls.append(
                    CallSpec(
                        source_symbol_id=source_symbol_id,
                        callee=text_of(source, function).strip(),
                        location=location_of(rel_path, node),
                    )
                )
        for child in node.named_children:
            self._collect_calls(
                builder, child, source, rel_path, source_symbol_id, info
            )

    def resolve_module(self, builder: Any, importer: str, module: str) -> str | None:
        importer_path = Path(importer)
        candidates = [
            importer_path.parent / (module.replace(".", "/") + ".py"),
            importer_path.parent / (module.split(".")[-1] + ".py"),
            Path(module.replace(".", "/") + ".py"),
            Path(module.split(".")[-1] + ".py"),
        ]
        for candidate in candidates:
            rel = candidate.as_posix().lstrip("./")
            if rel in builder.file_infos:
                return rel
        return None

    def resolve_callee(
        self, builder: Any, info: FileInfo, call: CallSpec
    ) -> str | None:
        callee = call.callee
        if callee.startswith(("self.", "cls.")):
            name = callee.split(".")[-1]
            return builder._find_symbol(info.rel_path, name, prefer_kind="method")

        if "." in callee:
            head, tail = callee.split(".", 1)
            if head in info.import_aliases:
                target_rel, imported = info.import_aliases[head]
                target_name = imported or tail.split(".")[-1]
                return (
                    builder._find_symbol(target_rel, target_name)
                    if target_rel
                    else None
                )
            return builder._find_symbol(info.rel_path, tail.split(".")[-1])

        local = builder._find_symbol(info.rel_path, callee)
        if local is not None:
            return local
        if callee in info.import_aliases:
            target_rel, imported = info.import_aliases[callee]
            return (
                builder._find_symbol(target_rel, imported or callee)
                if target_rel
                else None
            )
        return None
