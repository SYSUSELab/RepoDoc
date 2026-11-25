"""C and C++ analyzers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import tree_sitter_c
import tree_sitter_cpp
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

_EXTRA_SKIP = {"cmake-build-debug", "cmake-build-release", "third_party", "3rdparty"}
_TYPE_SPECIFIERS = {
    "class_specifier",
    "struct_specifier",
    "union_specifier",
    "enum_specifier",
}


def _string_value(source: bytes, node: Node | None) -> str | None:
    if node is None:
        return None
    raw = text_of(source, node).strip()
    if raw.startswith("<") and raw.endswith(">"):
        return None
    if len(raw) >= 2 and raw[0] in {'"', "'"} and raw[-1] == raw[0]:
        return raw[1:-1]
    return raw or None


def _declarator_name(source: bytes, node: Node | None) -> str | None:
    if node is None:
        return None
    if node.type in {
        "identifier",
        "field_identifier",
        "type_identifier",
        "operator_name",
        "destructor_name",
    }:
        return text_of(source, node)
    if node.type == "qualified_identifier":
        return text_of(source, node).split("::")[-1]
    declarator = node.child_by_field_name("declarator")
    if declarator is not None:
        return _declarator_name(source, declarator)
    for child in node.named_children:
        found = _declarator_name(source, child)
        if found:
            return found
    return None


def _function_name(source: bytes, node: Node) -> str | None:
    declarator = node.child_by_field_name("declarator")
    return _declarator_name(source, declarator)


def _declarator_scope(source: bytes, node: Node | None) -> str | None:
    if node is None:
        return None
    if node.type == "qualified_identifier":
        raw = text_of(source, node)
        return raw.split("::")[-2] if "::" in raw else None
    declarator = node.child_by_field_name("declarator")
    if declarator is not None:
        return _declarator_scope(source, declarator)
    for child in node.named_children:
        found = _declarator_scope(source, child)
        if found:
            return found
    return None


def _contains_type(node: Node, wanted: str) -> bool:
    stack = [node]
    while stack:
        item = stack.pop()
        if item.type == wanted:
            return True
        stack.extend(item.named_children)
    return False


class CFamilyAnalyzer:
    """Shared tree-sitter logic for C and C++."""

    name = "c"
    extensions = (".c", ".h")
    language = Language(tree_sitter_c.language())
    cpp = False

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
            if node.type != "preproc_include":
                continue
            module = _string_value(source, node.child_by_field_name("path"))
            if module:
                specs.append(
                    ImportSpec(module, None, None, location_of(rel_path, node))
                )
        return specs

    def resolve_module(self, builder: Any, importer: str, module: str) -> str | None:
        raw = module.replace("\\", "/").strip().strip('"')
        if not raw:
            return None
        importer_parent = Path(importer).parent
        candidates = [
            (importer_parent / raw).as_posix(),
            raw.lstrip("./"),
            (Path("include") / raw).as_posix(),
        ]
        for candidate in candidates:
            rel = Path(candidate).as_posix().lstrip("./")
            if rel in builder.file_infos:
                return rel
        basename = Path(raw).name
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
        if node.type in {"namespace_definition"} and self.cpp:
            body = node.child_by_field_name("body")
            if body is not None:
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

        if node.type in {"template_declaration"} and self.cpp:
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
            return

        if node.type == "linkage_specification":
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
            return

        if node.type in _TYPE_SPECIFIERS:
            self._add_type(
                builder, node, source, rel_path, parent_id, class_stack, info
            )
            return

        if node.type == "type_definition":
            self._add_typedef(
                builder, node, source, rel_path, parent_id, class_stack, info
            )
            return

        if node.type in {"function_definition", "declaration"}:
            self._add_function_declaration(
                builder, node, source, rel_path, parent_id, class_stack, info
            )
            return

        if node.type == "preproc_def" and node.child_by_field_name("name"):
            name = text_of(source, node.child_by_field_name("name"))
            emit_symbol(
                builder,
                info,
                rel_path,
                name,
                name,
                "macro",
                node,
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
        kind = {
            "class_specifier": "class",
            "struct_specifier": "struct",
            "union_specifier": "union",
            "enum_specifier": "enum",
        }[node.type]
        qualified = ".".join([*class_stack, name])
        visibility = "public"
        if self.cpp and class_stack:
            visibility = "private"
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
            visibility=visibility,
            signature=short_signature(source, node, body_field="body"),
        )
        body = node.child_by_field_name("body")
        if body is not None:
            self._visit_class_body(
                builder,
                body,
                source,
                rel_path,
                symbol_id,
                [*class_stack, name],
                info,
                default_visibility="public"
                if kind in {"struct", "union"}
                else "private",
            )

    def _visit_class_body(
        self,
        builder: Any,
        body: Node,
        source: bytes,
        rel_path: str,
        parent_id: str,
        class_stack: list[str],
        info: FileInfo,
        *,
        default_visibility: str,
    ) -> None:
        access = default_visibility
        for child in body.named_children:
            if child.type == "access_specifier":
                access = text_of(source, child).strip().rstrip(":")
                continue
            if child.type in _TYPE_SPECIFIERS:
                self._add_type(
                    builder,
                    child,
                    source,
                    rel_path,
                    parent_id,
                    class_stack,
                    info,
                )
                continue
            if child.type in {
                "function_definition",
                "field_declaration",
                "declaration",
            }:
                self._add_function_declaration(
                    builder,
                    child,
                    source,
                    rel_path,
                    parent_id,
                    class_stack,
                    info,
                    visibility=access,
                )
                continue

    def _add_typedef(
        self,
        builder: Any,
        node: Node,
        source: bytes,
        rel_path: str,
        parent_id: str,
        class_stack: list[str],
        info: FileInfo,
    ) -> None:
        name_node = node.child_by_field_name("declarator")
        name = _declarator_name(source, name_node)
        if not name:
            return
        # `typedef struct Point Point;` already has the struct symbol.
        if name in info.symbols:
            return
        type_node = node.child_by_field_name("type")
        if type_node is not None and type_node.type in _TYPE_SPECIFIERS:
            kind = {
                "class_specifier": "class",
                "struct_specifier": "struct",
                "union_specifier": "union",
                "enum_specifier": "enum",
            }[type_node.type]
        else:
            kind = "typedef"
        emit_symbol(
            builder,
            info,
            rel_path,
            name,
            ".".join([*class_stack, name]),
            kind,
            node,
            source,
            parent_id,
            exported=True,
            language=self.name,
            visibility="public",
        )

    def _add_function_declaration(
        self,
        builder: Any,
        node: Node,
        source: bytes,
        rel_path: str,
        parent_id: str,
        class_stack: list[str],
        info: FileInfo,
        *,
        visibility: str = "public",
    ) -> None:
        if node.type == "field_declaration" and not self.cpp:
            return
        name = _function_name(source, node)
        if not name:
            return
        # Skip ordinary variables.
        declarator = node.child_by_field_name("declarator")
        if declarator is None or not _contains_type(declarator, "function_declarator"):
            return
        scope = _declarator_scope(source, declarator)
        effective_stack = list(class_stack)
        if scope and not effective_stack:
            # A definition may appear before the class declaration (the .cpp
            # file sorts before the .hpp file). Uppercase scope names are a
            # conservative signal for a class/struct, not a namespace.
            if (
                scope[:1].isupper()
                or self._find_type_symbol(builder, scope) is not None
            ):
                effective_stack = [scope]
        parent = parent_id
        if effective_stack and not class_stack:
            found = self._find_type_symbol(builder, effective_stack[-1])
            if found:
                parent = found
        static = "static" in text_of(source, node).split()
        public = visibility == "public" and not static
        kind = "method" if effective_stack else "function"
        if effective_stack and name == effective_stack[-1]:
            kind = "constructor"
        qualified = ".".join([*effective_stack, name])
        symbol_id = emit_symbol(
            builder,
            info,
            rel_path,
            name,
            qualified,
            kind,
            node,
            source,
            parent,
            exported=public,
            language=self.name,
            visibility=visibility
            if effective_stack
            else ("private" if static else "public"),
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
                    "class_specifier",
                    "struct_specifier",
                },
            )

    def _find_type_symbol(self, builder: Any, name: str) -> str | None:
        for info in builder.file_infos.values():
            for symbol_id in info.symbols.get(name, []):
                if builder.graph.nodes[symbol_id]["attrs"].get("kind") in {
                    "class",
                    "struct",
                    "union",
                    "enum",
                }:
                    return symbol_id
        return None

    def _callee(self, node: Node, source: bytes) -> str | None:
        if node.type == "call_expression":
            function = node.child_by_field_name("function")
            return text_of(source, function) if function is not None else None
        if node.type == "new_expression" and self.cpp:
            type_node = node.child_by_field_name("type")
            return text_of(source, type_node) if type_node is not None else None
        return None

    def resolve_callee(self, builder: Any, info: FileInfo, call: Any) -> str | None:
        found = resolve_callee(builder, info, call.callee, separators=("::", "->", "."))
        if found:
            return found
        # Included headers are parsed as separate files. If a call is not
        # local, look through the imported files' symbols before giving up.
        simple = call.callee.replace("::", ".").replace("->", ".").split(".")[-1]
        for target_rel, _ in info.import_aliases.values():
            if not target_rel:
                continue
            candidate = builder._find_symbol(target_rel, simple)
            if candidate:
                return candidate
        return None


class CAnalyzer(CFamilyAnalyzer):
    name = "c"
    extensions = (".c", ".h")
    language = Language(tree_sitter_c.language())
    cpp = False


class CppAnalyzer(CFamilyAnalyzer):
    name = "cpp"
    extensions = (".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh", ".hxx", ".h")
    language = Language(tree_sitter_cpp.language())
    cpp = True
