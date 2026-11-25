"""Rust analyzer."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import tree_sitter_rust
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

_EXTRA_SKIP = {"target", ".cargo", ".rustup"}
_TYPE_NODES = {"struct_item", "enum_item", "trait_item", "union_item", "type_item"}
_IMPL_NODES = {"impl_item"}


def _rust_visibility(source: bytes, node: Node) -> bool:
    return any(
        text_of(source, child).strip().startswith("pub")
        for child in node.children
        if child.type == "visibility_modifier"
    )


def _strip_generics(name: str) -> str:
    return re.sub(r"<.*>$", "", name.strip())


def _use_parts(source: bytes, node: Node) -> list[str]:
    """Flatten a Rust use-tree into complete path strings."""

    if node.type in {"identifier", "crate", "self", "super", "metavariable"}:
        return [text_of(source, node)]
    if node.type == "scoped_identifier":
        path = node.child_by_field_name("path")
        name = node.child_by_field_name("name")
        if path is None or name is None:
            return []
        return [f"{part}::{text_of(source, name)}" for part in _use_parts(source, path)]
    if node.type == "scoped_use_list":
        path = node.child_by_field_name("path")
        items = node.child_by_field_name("list")
        prefixes = _use_parts(source, path) if path is not None else [""]
        results: list[str] = []
        if items is not None:
            for item in items.named_children:
                for suffix in _use_parts(source, item):
                    for prefix in prefixes:
                        results.append(f"{prefix}::{suffix}".strip(":"))
        return results
    if node.type == "use_list":
        results = []
        for item in node.named_children:
            results.extend(_use_parts(source, item))
        return results
    if node.type == "use_as_clause":
        path = node.child_by_field_name("path")
        return _use_parts(source, path) if path is not None else []
    if node.type == "use_wildcard":
        path = node.child_by_field_name("path")
        return _use_parts(source, path) if path is not None else []
    return []


class RustAnalyzer:
    name = "rust"
    extensions = (".rs",)
    language = Language(tree_sitter_rust.language())

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
            if node.type == "mod_item":
                name = node.child_by_field_name("name")
                if name is not None:
                    specs.append(
                        ImportSpec(text_of(source, name), None, None, location)
                    )
                continue
            if node.type != "use_declaration":
                continue
            argument = node.child_by_field_name("argument")
            if argument is None:
                continue
            alias = None
            target = argument
            if argument.type == "use_as_clause":
                target = argument.child_by_field_name("path") or argument
                alias_node = argument.child_by_field_name("alias")
                alias = text_of(source, alias_node) if alias_node is not None else None
            for used in _use_parts(source, target):
                parts = [part for part in used.split("::") if part]
                if not parts:
                    continue
                # `use crate::foo` and `use super::foo` name a module. Keep
                # the complete path so the resolver can find foo.rs.
                if len(parts) == 2 and parts[0] in {"crate", "self", "super"}:
                    module, name = used, None
                else:
                    module, name = "::".join(parts[:-1]), parts[-1]
                specs.append(ImportSpec(module or used, name, alias, location))
        return specs

    def resolve_module(self, builder: Any, importer: str, module: str) -> str | None:
        raw = module.replace("::", "/").strip("/")
        importer_parent = Path(importer).parent
        if raw.startswith("crate/"):
            bases = [Path(raw[6:]), Path("src") / raw[6:]]
        elif raw.startswith("self/"):
            bases = [importer_parent / raw[5:]]
        elif raw.startswith("super/"):
            bases = [importer_parent.parent / raw[6:]]
        else:
            bases = [Path(raw), importer_parent / raw]
        candidates: list[str] = []
        for base in bases:
            candidates.extend(
                [
                    base.as_posix() + ".rs",
                    (base / "mod.rs").as_posix(),
                ]
            )
        for candidate in candidates:
            rel = candidate.lstrip("./")
            if rel in builder.file_infos:
                return rel
        # Rust projects commonly keep modules under src/. Fall back to a
        # unique suffix match rather than creating a false edge.
        suffix = raw.replace("crate/", "").replace("self/", "")
        matches = [
            rel
            for rel in builder.file_infos
            if rel.endswith(f"/{suffix}.rs") or rel == f"{suffix}.rs"
        ]
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
        if node.type == "impl_item":
            type_node = node.child_by_field_name("type")
            body = node.child_by_field_name("body")
            if type_node is None or body is None:
                return
            type_name = _strip_generics(text_of(source, type_node))
            ids = info.symbols.get(type_name, [])
            impl_parent = next(
                (
                    item
                    for item in ids
                    if builder.graph.nodes[item]["attrs"].get("kind")
                    in {"struct", "enum", "trait", "union"}
                ),
                parent_id,
            )
            for child in body.named_children:
                self.visit_definitions(
                    builder,
                    child,
                    source,
                    rel_path,
                    impl_parent,
                    [*class_stack, type_name],
                    [],
                    info,
                )
            return

        if node.type == "trait_item":
            name_node = node.child_by_field_name("name")
            body = node.child_by_field_name("body")
            if name_node is None:
                return
            name = text_of(source, name_node)
            qualified = ".".join([*class_stack, *function_stack, name])
            is_public = _rust_visibility(source, node)
            symbol_id = emit_symbol(
                builder,
                info,
                rel_path,
                name,
                qualified,
                "trait",
                node,
                source,
                parent_id,
                exported=is_public,
                language=self.name,
                visibility="public" if is_public else "private",
            )
            if body is not None:
                for child in body.named_children:
                    if child.type == "function_signature_item":
                        self._add_function(
                            builder,
                            child,
                            source,
                            rel_path,
                            symbol_id,
                            [*class_stack, name],
                            info,
                            exported=is_public,
                            force_kind="method",
                        )
            return

        if node.type in _TYPE_NODES:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return
            name = text_of(source, name_node)
            kind = {
                "struct_item": "struct",
                "enum_item": "enum",
                "union_item": "union",
                "type_item": "type",
            }.get(node.type, "type")
            qualified = ".".join([*class_stack, *function_stack, name])
            is_public = _rust_visibility(source, node)
            emit_symbol(
                builder,
                info,
                rel_path,
                name,
                qualified,
                kind,
                node,
                source,
                parent_id,
                exported=is_public,
                language=self.name,
                visibility="public" if is_public else "private",
            )
            return

        if node.type in {"const_item", "static_item", "macro_definition"}:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return
            name = text_of(source, name_node)
            kind = {
                "const_item": "const",
                "static_item": "static",
                "macro_definition": "macro",
            }[node.type]
            qualified = ".".join([*class_stack, *function_stack, name])
            emit_symbol(
                builder,
                info,
                rel_path,
                name,
                qualified,
                kind,
                node,
                source,
                parent_id,
                exported=_rust_visibility(source, node),
                language=self.name,
                visibility="public" if _rust_visibility(source, node) else "private",
            )
            return

        if node.type in {"function_item", "function_signature_item"}:
            self._add_function(
                builder,
                node,
                source,
                rel_path,
                parent_id,
                class_stack,
                info,
                exported=_rust_visibility(source, node),
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
        exported: bool,
        force_kind: str | None = None,
    ) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = text_of(source, name_node)
        kind = force_kind or ("method" if class_stack else "function")
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
            exported=exported,
            language=self.name,
            visibility="public" if exported else "private",
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
                skip_types={"function_item", "impl_item", "trait_item"},
            )

    def _callee(self, node: Node, source: bytes) -> str | None:
        if node.type == "call_expression":
            function = node.child_by_field_name("function")
            return text_of(source, function) if function is not None else None
        return None

    def resolve_callee(self, builder: Any, info: FileInfo, call: Any) -> str | None:
        return resolve_callee(builder, info, call.callee, separators=("::", "."))
