"""Go analyzer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import tree_sitter_go
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

_EXTRA_SKIP = {"vendor", "testdata"}


def _exported(name: str) -> bool:
    return bool(name) and name[0].isupper()


def _unquote(source: bytes, node: Node | None) -> str | None:
    if node is None:
        return None
    raw = text_of(source, node).strip()
    if len(raw) >= 2 and raw[0] in {'"', "'", "`"} and raw[-1] == raw[0]:
        return raw[1:-1]
    return raw or None


def _receiver_name(source: bytes, node: Node) -> str | None:
    receiver = node.child_by_field_name("receiver")
    if receiver is None:
        return None
    for child in receiver.named_children:
        stack = [child]
        while stack:
            item = stack.pop()
            if item.type == "type_identifier":
                return text_of(source, item)
            stack.extend(item.named_children)
    return None


class GoAnalyzer:
    name = "go"
    extensions = (".go",)
    language = Language(tree_sitter_go.language())

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
            if node.type != "import_spec":
                continue
            path = _unquote(source, node.child_by_field_name("path"))
            if not path:
                continue
            name_node = node.child_by_field_name("name")
            name = text_of(source, name_node) if name_node is not None else None
            if name in {".", "_"}:
                alias = None
            else:
                alias = name or Path(path).name
            specs.append(ImportSpec(path, name, alias, location_of(rel_path, node)))
        return specs

    def resolve_module(self, builder: Any, importer: str, module: str) -> str | None:
        raw = module.strip().strip('"').replace("\\", "/")
        if raw.endswith(".go") and raw.lstrip("./") in builder.file_infos:
            return raw.lstrip("./")
        parts = [part for part in raw.split("/") if part]
        if not parts:
            return None
        # With a go.mod module prefix, the useful suffix is the package
        # directory. Try longest suffix first and pick a stable representative
        # file for the package.
        for start in range(len(parts)):
            suffix = "/".join(parts[start:])
            matches = [
                rel
                for rel in builder.file_infos
                if rel == suffix
                or rel.endswith(f"/{suffix}")
                or Path(rel).parent.as_posix().endswith(suffix)
            ]
            if not matches:
                continue
            files = sorted(matches)
            for preferred in ("doc.go", "package.go"):
                for rel in files:
                    if Path(rel).name == preferred:
                        return rel
            return files[0]
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
        if node.type == "type_declaration":
            for child in node.named_children:
                if child.type in {"type_spec", "type_alias"}:
                    self._add_type(builder, child, source, rel_path, parent_id, info)
            return

        if node.type == "function_declaration":
            self._add_function(
                builder,
                node,
                source,
                rel_path,
                parent_id,
                class_stack,
                info,
                force_kind="function",
            )
            return

        if node.type == "method_declaration":
            receiver = _receiver_name(source, node)
            type_parent = parent_id
            if receiver:
                ids = info.symbols.get(receiver, [])
                type_parent = next(
                    (
                        item
                        for item in ids
                        if builder.graph.nodes[item]["attrs"].get("kind")
                        in {"struct", "interface", "type"}
                    ),
                    parent_id,
                )
            self._add_function(
                builder,
                node,
                source,
                rel_path,
                type_parent,
                [receiver] if receiver else class_stack,
                info,
                force_kind="method",
            )
            return

        if node.type in {"const_declaration", "var_declaration"}:
            kind = "const" if node.type == "const_declaration" else "var"
            for child in node.named_children:
                if child.type not in {"const_spec", "var_spec"}:
                    continue
                names = child.child_by_field_name("name")
                if names is None:
                    continue
                name = text_of(source, names)
                if not name:
                    continue
                exported = _exported(name)
                emit_symbol(
                    builder,
                    info,
                    rel_path,
                    name,
                    name,
                    kind,
                    child,
                    source,
                    parent_id,
                    exported=exported,
                    language=self.name,
                    visibility="public" if exported else "private",
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
        info: FileInfo,
    ) -> None:
        name_node = node.child_by_field_name("name")
        type_node = node.child_by_field_name("type")
        if name_node is None:
            return
        name = text_of(source, name_node)
        kind = "type"
        if type_node is not None:
            kind = {
                "struct_type": "struct",
                "interface_type": "interface",
            }.get(type_node.type, "type")
        exported = _exported(name)
        symbol_id = emit_symbol(
            builder,
            info,
            rel_path,
            name,
            name,
            kind,
            node,
            source,
            parent_id,
            exported=exported,
            language=self.name,
            visibility="public" if exported else "private",
        )
        if type_node is None or type_node.type != "interface_type":
            return
        # Methods declared in an interface are part of the public contract
        # even though they have no implementation body.
        for child in type_node.named_children:
            if child.type != "method_elem":
                continue
            method = child.child_by_field_name("name")
            if method is None:
                continue
            method_name = text_of(source, method)
            emit_symbol(
                builder,
                info,
                rel_path,
                method_name,
                f"{name}.{method_name}",
                "method",
                child,
                source,
                symbol_id,
                exported=exported,
                language=self.name,
                visibility="public" if exported else "private",
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
        qualified = ".".join([*class_stack, name])
        exported = _exported(name)
        symbol_id = emit_symbol(
            builder,
            info,
            rel_path,
            name,
            qualified,
            force_kind,
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
                skip_types={"func_literal"},
            )

    def _callee(self, node: Node, source: bytes) -> str | None:
        if node.type == "call_expression":
            function = node.child_by_field_name("function")
            return text_of(source, function) if function is not None else None
        return None

    def resolve_callee(self, builder: Any, info: FileInfo, call: Any) -> str | None:
        return resolve_callee(builder, info, call.callee, separators=("."))
