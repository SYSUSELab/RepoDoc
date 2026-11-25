"""PHP analyzer."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import tree_sitter_php
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

_EXTRA_SKIP = {"vendor", "storage", "cache", "var"}


def _visibility(source: bytes, node: Node) -> str:
    for child in node.children:
        if child.type == "visibility_modifier":
            raw = text_of(source, child).strip().lower()
            if raw in {"public", "protected", "private"}:
                return raw
    return "public"


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] in {'"', "'"} and value[-1] == value[0]:
        return value[1:-1]
    return value


def _use_texts(source: bytes, node: Node) -> list[str]:
    """Return class/function/const names from a namespace use declaration."""

    raw = text_of(source, node).strip()
    raw = re.sub(r"^use\s+", "", raw).rstrip(";").strip()
    prefix = ""
    if raw.startswith("function "):
        prefix = "function "
        raw = raw[len("function ") :].strip()
    elif raw.startswith("const "):
        prefix = "const "
        raw = raw[len("const ") :].strip()
    if "{" in raw and raw.endswith("}"):
        head, body = raw.split("{", 1)
        head = head.rstrip("\\")
        items = [
            f"{head}\\{item.strip()}" for item in body[:-1].split(",") if item.strip()
        ]
    else:
        items = [item.strip() for item in raw.split(",") if item.strip()]
    return [prefix + item for item in items]


class PhpAnalyzer:
    name = "php"
    extensions = (".php", ".phtml")
    language = Language(tree_sitter_php.language_php())

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
            location = location_of(rel_path, node)
            if node.type == "namespace_use_declaration":
                for raw in _use_texts(source, node):
                    kind = ""
                    for prefix in ("function ", "const "):
                        if raw.startswith(prefix):
                            kind = prefix.strip()
                            raw = raw[len(prefix) :].strip()
                            break
                    alias = None
                    if re.search(r"\s+as\s+", raw, re.I):
                        raw, alias = re.split(r"\s+as\s+", raw, maxsplit=1, flags=re.I)
                        alias = alias.strip()
                    raw = raw.strip().lstrip("\\")
                    if not raw:
                        continue
                    parts = raw.split("\\")
                    name = parts[-1]
                    specs.append(
                        ImportSpec(
                            raw,
                            name if kind != "const" else name,
                            alias or name,
                            location,
                        )
                    )
                continue
            if node.type in {
                "require_expression",
                "require_once_expression",
                "include_expression",
                "include_once_expression",
            }:
                # __DIR__ . '/relative.php' or a plain string literal.
                strings = re.findall(r"(['\"])(.*?)\1", text_of(source, node))
                if strings:
                    include_path = _strip_quotes(strings[-1][1]).lstrip("/")
                    specs.append(ImportSpec(include_path, None, None, location))
        return specs

    def resolve_module(self, builder: Any, importer: str, module: str) -> str | None:
        raw = _strip_quotes(module).replace("\\", "/").strip()
        if not raw:
            return None
        importer_parent = Path(importer).parent
        candidates = []
        if raw.endswith(".php") or "/" in raw:
            candidates.extend(
                [
                    (importer_parent / raw).as_posix(),
                    raw.lstrip("./"),
                ]
            )
        namespace_path = raw if raw.endswith(".php") else raw + ".php"
        candidates.extend(
            [
                namespace_path.lstrip("./"),
                (Path("src") / namespace_path).as_posix(),
                (importer_parent / namespace_path).as_posix(),
            ]
        )
        for candidate in candidates:
            rel = Path(candidate).as_posix().lstrip("./")
            if rel in builder.file_infos:
                return rel
        suffix = namespace_path
        matches = [
            rel
            for rel in builder.file_infos
            if rel.endswith(f"/{suffix}") or rel == suffix
        ]
        if len(matches) == 1:
            return matches[0]
        # PSR-4 roots are often absent from a small fixture (or not recorded
        # in composer.json). Fall back to a unique basename match.
        basename = Path(namespace_path).name
        matches = [rel for rel in builder.file_infos if Path(rel).name == basename]
        if len(matches) == 1:
            return matches[0]
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
        if node.type in {
            "class_declaration",
            "interface_declaration",
            "trait_declaration",
            "enum_declaration",
        }:
            self._add_type(
                builder, node, source, rel_path, parent_id, class_stack, info
            )
            return

        if node.type in {"method_declaration", "function_definition"}:
            self._add_function(
                builder,
                node,
                source,
                rel_path,
                parent_id,
                class_stack,
                info,
                force_kind="method"
                if node.type == "method_declaration"
                else "function",
            )
            return

        if node.type == "namespace_definition":
            body = node.child_by_field_name("body")
            if body is None:
                return
            for child in body.named_children:
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
        kind = {
            "class_declaration": "class",
            "interface_declaration": "interface",
            "trait_declaration": "trait",
            "enum_declaration": "enum",
        }[node.type]
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
            exported=True,
            language=self.name,
            visibility="public",
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

    def _add_function(
        self,
        builder: Any,
        node: Node,
        source: bytes,
        rel_path: str,
        parent_id: str,
        class_stack: list[str],
        info: FileInfo,
        *,
        force_kind: str,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = text_of(source, name_node)
        visibility = _visibility(source, node)
        public = visibility == "public"
        kind = (
            "constructor"
            if force_kind == "method" and name.lower() == "__construct"
            else force_kind
        )
        qualified = ".".join([*class_stack, name])
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
                    "function_definition",
                    "method_declaration",
                    "class_declaration",
                },
            )

    def _callee(self, node: Node, source: bytes) -> str | None:
        if node.type == "function_call_expression":
            function = node.child_by_field_name("function")
            return text_of(source, function) if function is not None else None
        if node.type in {"member_call_expression", "nullsafe_member_call_expression"}:
            obj = node.child_by_field_name("object")
            name = node.child_by_field_name("name")
            if name is not None:
                return (
                    f"{text_of(source, obj)}->{text_of(source, name)}"
                    if obj
                    else text_of(source, name)
                )
        if node.type == "scoped_call_expression":
            scope = node.child_by_field_name("scope")
            name = node.child_by_field_name("name")
            if name is not None:
                return (
                    f"{text_of(source, scope)}::{text_of(source, name)}"
                    if scope
                    else text_of(source, name)
                )
        if node.type == "object_creation_expression":
            for child in node.named_children:
                if child.type in {"name", "qualified_name"}:
                    return text_of(source, child)
        return None

    def resolve_callee(self, builder: Any, info: FileInfo, call: Any) -> str | None:
        return resolve_callee(builder, info, call.callee, separators=("->", "::", "."))
