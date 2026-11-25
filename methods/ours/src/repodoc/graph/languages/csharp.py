"""C# analyzer."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import tree_sitter_c_sharp
from tree_sitter import Language, Node

from repodoc.graph.languages.base import (
    FileInfo,
    ImportSpec,
    collect_calls,
    comment_nodes,
    emit_symbol,
    iter_source_files,
    location_of,
    resolve_callee,
    short_signature,
    text_of,
    walk_nodes,
)

_EXTRA_SKIP = {"bin", "obj", "packages", ".vs"}
_TYPES = {
    "class_declaration": "class",
    "interface_declaration": "interface",
    "struct_declaration": "struct",
    "enum_declaration": "enum",
    "record_declaration": "record",
    "record_struct_declaration": "record",
}


def _modifiers(source: bytes, node: Node) -> set[str]:
    values: set[str] = set()
    for child in node.children:
        if child.type == "modifier":
            values.update(text_of(source, child).split())
    return values


def _visibility(source: bytes, node: Node, default: str = "internal") -> str:
    values = _modifiers(source, node)
    for value in ("public", "protected", "internal", "private"):
        if value in values:
            return value
    return default


class CSharpAnalyzer:
    name = "csharp"
    extensions = (".cs",)
    language = Language(tree_sitter_c_sharp.language())

    def iter_files(
        self,
        repo_path: Path,
        skip_parts: set[str],
        exclude_dirs: Iterable[Path],
    ) -> list[Path]:
        return iter_source_files(
            repo_path, self.extensions, skip_parts, exclude_dirs, _EXTRA_SKIP
        )

    def comments(
        self, source: bytes, root: Node, rel_path: str
    ) -> list[dict[str, Any]]:
        return [
            {
                "text": text_of(source, node).strip(),
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
            }
            for node in comment_nodes(root)
        ]

    def file_metadata(self, source: bytes, root: Node, rel_path: str) -> dict[str, Any]:
        comments = self.comments(source, root, rel_path)
        return {"docstring": comments[0]["text"] if comments else None}

    def parse_imports(
        self, source: bytes, root: Node, rel_path: str
    ) -> list[ImportSpec]:
        specs: list[ImportSpec] = []
        for node in walk_nodes(root):
            if node.type != "using_directive":
                continue
            raw = text_of(source, node).strip()
            raw = re.sub(r"^global\s+", "", raw)
            raw = re.sub(r"^using\s+", "", raw).rstrip(";").strip()
            if raw.startswith("static "):
                raw = raw[len("static ") :].strip()
            alias = None
            if "=" in raw:
                alias, raw = [item.strip() for item in raw.split("=", 1)]
            parts = [part for part in raw.split(".") if part]
            if not parts:
                continue
            name = None if alias else parts[-1]
            specs.append(
                ImportSpec(
                    raw,
                    name,
                    alias or (None if len(parts) == 1 else parts[-1]),
                    location_of(rel_path, node),
                )
            )
        return specs

    def resolve_module(self, builder: Any, importer: str, module: str) -> str | None:
        raw = module.replace(".", "/").strip("/")
        candidates = [raw + ".cs", raw]
        for candidate in candidates:
            rel = candidate.lstrip("./")
            if rel in builder.file_infos:
                return rel
        matches = [
            rel
            for rel in builder.file_infos
            if rel.endswith(f"/{raw}.cs")
            or f"/{raw}/" in f"/{rel}"
            or Path(rel).parent.as_posix().endswith(raw)
        ]
        if matches:
            return sorted(matches)[0]
        basename = Path(raw).name + ".cs"
        matches = [rel for rel in builder.file_infos if Path(rel).name == basename]
        return matches[0] if len(matches) == 1 else None

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
        if node.type in _TYPES:
            self._add_type(
                builder, node, source, rel_path, parent_id, class_stack, info
            )
            return

        if node.type in {"method_declaration", "constructor_declaration"}:
            self._add_method(
                builder, node, source, rel_path, parent_id, class_stack, info
            )
            return

        if node.type in {"property_declaration", "field_declaration"}:
            self._add_member(
                builder, node, source, rel_path, parent_id, class_stack, info
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
                decorators,
            )

    def _is_interface(
        self, builder: Any, info: FileInfo, class_stack: list[str]
    ) -> bool:
        if not class_stack:
            return False
        return any(
            builder.graph.nodes.get(symbol_id, {}).get("attrs", {}).get("kind")
            == "interface"
            for symbol_id in info.symbols.get(class_stack[-1], [])
        )

    def _add_type(
        self,
        builder: Any,
        node: Node,
        source: bytes,
        rel_path: str,
        parent_id: str,
        class_stack: list[str],
        info: FileInfo,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = text_of(source, name_node)
        qualified = ".".join([*class_stack, name])
        visibility = _visibility(source, node)
        public = visibility == "public"
        kind = _TYPES[node.type]
        symbol_id = emit_symbol(
            builder,
            info,
            rel_path,
            name,
            qualified,
            kind,
            node,
            source,
            parent_id,
            exported=public,
            language=self.name,
            visibility=visibility,
            signature=short_signature(source, node, body_field="body"),
        )
        body = node.child_by_field_name("body")
        if body is None:
            return
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

    def _add_method(
        self,
        builder: Any,
        node: Node,
        source: bytes,
        rel_path: str,
        parent_id: str,
        class_stack: list[str],
        info: FileInfo,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = text_of(source, name_node)
        visibility = _visibility(source, node, default="private")
        implicit_public = self._is_interface(builder, info, class_stack)
        public = visibility == "public" or implicit_public
        kind = "constructor" if node.type == "constructor_declaration" else "method"
        symbol_id = emit_symbol(
            builder,
            info,
            rel_path,
            name,
            ".".join([*class_stack, name]),
            kind,
            node,
            source,
            parent_id,
            exported=public,
            language=self.name,
            visibility="public" if public else visibility,
            signature=short_signature(source, node),
        )
        body = node.child_by_field_name("body")
        if body is not None:
            collect_calls(
                body,
                source,
                rel_path,
                symbol_id,
                info,
                lambda child: self._callee(child, source),
                skip_types={
                    "method_declaration",
                    "constructor_declaration",
                    "class_declaration",
                },
            )

    def _add_member(
        self,
        builder: Any,
        node: Node,
        source: bytes,
        rel_path: str,
        parent_id: str,
        class_stack: list[str],
        info: FileInfo,
    ) -> None:
        visibility = _visibility(source, node, default="private")
        if visibility != "public":
            return
        if node.type == "property_declaration":
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return
            name = text_of(source, name_node)
            emit_symbol(
                builder,
                info,
                rel_path,
                name,
                ".".join([*class_stack, name]),
                "property",
                node,
                source,
                parent_id,
                exported=True,
                language=self.name,
                visibility="public",
            )
            return
        for child in node.named_children:
            if child.type != "variable_declaration":
                continue
            for declarator in child.named_children:
                if declarator.type != "variable_declarator":
                    continue
                name_node = declarator.child_by_field_name("name")
                if name_node is None:
                    continue
                name = text_of(source, name_node)
                emit_symbol(
                    builder,
                    info,
                    rel_path,
                    name,
                    ".".join([*class_stack, name]),
                    "field",
                    declarator,
                    source,
                    parent_id,
                    exported=True,
                    language=self.name,
                    visibility="public",
                )

    def _callee(self, node: Node, source: bytes) -> str | None:
        if node.type == "invocation_expression":
            function = node.child_by_field_name("function")
            return text_of(source, function) if function is not None else None
        if node.type == "object_creation_expression":
            type_node = node.child_by_field_name("type")
            return text_of(source, type_node) if type_node is not None else None
        return None

    def resolve_callee(self, builder: Any, info: FileInfo, call: Any) -> str | None:
        return resolve_callee(
            builder,
            info,
            call.callee,
            separators=("."),
            prefixes=("this.", "base.", "self."),
        )
