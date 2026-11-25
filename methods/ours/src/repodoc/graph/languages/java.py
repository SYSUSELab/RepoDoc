"""Java analyzer."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import tree_sitter_java
from tree_sitter import Language, Node

from repodoc.graph.languages.base import (
    FileInfo,
    ImportSpec,
    collect_calls,
    comment_nodes,
    emit_symbol,
    iter_source_files,
    location_of,
    modifier_text,
    resolve_callee,
    short_signature,
    text_of,
    walk_nodes,
)

_EXTRA_SKIP = {"target", "out", ".gradle", ".mvn"}
_TYPE_NODES = {
    "class_declaration": "class",
    "interface_declaration": "interface",
    "enum_declaration": "enum",
    "record_declaration": "record",
    "annotation_type_declaration": "annotation",
}


class JavaAnalyzer:
    name = "java"
    extensions = (".java",)
    language = Language(tree_sitter_java.language())

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
            if node.type != "import_declaration":
                continue
            raw = text_of(source, node).strip()
            raw = re.sub(r"^import\s+", "", raw).rstrip(";").strip()
            is_static = raw.startswith("static ")
            if is_static:
                raw = raw[len("static ") :].strip()
            wildcard = raw.endswith(".*")
            if wildcard:
                raw = raw[:-2]
            parts = [part for part in raw.split(".") if part]
            if not parts:
                continue
            if is_static:
                module = raw
                name = None if wildcard else parts[-1]
                alias = None if wildcard else name
            elif wildcard:
                module = raw
                name = None
                alias = None
            else:
                module = raw
                name = parts[-1]
                alias = name
            specs.append(
                ImportSpec(
                    module,
                    name,
                    alias,
                    location_of(rel_path, node),
                )
            )
        return specs

    def resolve_module(self, builder: Any, importer: str, module: str) -> str | None:
        raw = module.replace(".", "/").strip("/")
        candidates = [raw + ".java", raw]
        for candidate in candidates:
            rel = candidate.lstrip("./")
            if rel in builder.file_infos:
                return rel
        # A wildcard/package import points at a directory. Choose a stable
        # representative source file from the package.
        matches = [
            rel
            for rel in builder.file_infos
            if rel.endswith(f"/{raw}.java")
            or rel.startswith(f"{raw}/")
            or f"/{raw}/" in f"/{rel}"
        ]
        if matches:
            return sorted(matches)[0]
        return None

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
        if node.type in _TYPE_NODES:
            self._add_type(
                builder, node, source, rel_path, parent_id, class_stack, info
            )
            return

        if node.type in {"method_declaration", "constructor_declaration"}:
            block = node.child_by_field_name("body")
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return
            name = text_of(source, name_node)
            qualified = ".".join([*class_stack, name])
            modifiers = modifier_text(source, node)
            is_interface = bool(
                class_stack
                and class_stack[-1]
                and any(
                    builder.graph.nodes.get(symbol_id, {}).get("attrs", {}).get("kind")
                    == "interface"
                    for symbol_id in info.symbols.get(class_stack[-1], [])
                )
            )
            public = "public" in modifiers.split() or is_interface
            kind = "constructor" if node.type == "constructor_declaration" else "method"
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
                visibility="public" if public else "private",
                signature=short_signature(source, node),
            )
            if block is not None:
                collect_calls(
                    block,
                    source,
                    rel_path,
                    symbol_id,
                    info,
                    lambda child: self._callee(child, source),
                    skip_types={
                        "method_declaration",
                        "constructor_declaration",
                        "class_declaration",
                        "interface_declaration",
                    },
                )
            return

        if node.type == "field_declaration":
            modifiers = modifier_text(source, node)
            public = "public" in modifiers.split()
            if public:
                for declarator in node.named_children:
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
        modifiers = modifier_text(source, node)
        public = "public" in modifiers.split()
        kind = _TYPE_NODES[node.type]
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
            visibility="public" if public else "package",
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

    def _callee(self, node: Node, source: bytes) -> str | None:
        if node.type == "method_invocation":
            name = node.child_by_field_name("name")
            if name is None:
                return None
            obj = node.child_by_field_name("object")
            if obj is not None:
                return f"{text_of(source, obj)}.{text_of(source, name)}"
            return text_of(source, name)
        if node.type == "object_creation_expression":
            type_node = node.child_by_field_name("type")
            if type_node is not None:
                return text_of(source, type_node)
        return None

    def resolve_callee(self, builder: Any, info: FileInfo, call: Any) -> str | None:
        return resolve_callee(builder, info, call.callee, separators=("."))
