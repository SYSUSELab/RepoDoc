"""JavaScript / TypeScript analyzers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import tree_sitter_javascript
import tree_sitter_typescript
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

_SOURCE_EXTENSIONS = (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts")
_REQUIRE_RE = re.compile(
    r"(?:const|let|var)\s+([A-Za-z_$][\w$]*|\{[^}]+\})\s*=\s*require\(\s*['\"]([^'\"]+)['\"]\s*\)"
)
_SKIP_PARTS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "coverage",
    "fixtures",
    "__snapshots__",
    ".cache",
    ".vitepress",
    ".next",
    ".nuxt",
    ".repodoc",
}


def _string_value(source: bytes, node: Node | None) -> str | None:
    if node is None:
        return None
    raw = text_of(source, node).strip()
    if len(raw) >= 2 and raw[0] in {'"', "'", "`"} and raw[-1] == raw[0]:
        return raw[1:-1]
    return None


def _short_signature(source: bytes, node: Node) -> str:
    raw = text_of(source, node).strip()
    body = node.child_by_field_name("body")
    if body is not None:
        raw = source[node.start_byte : body.start_byte].decode(
            "utf-8", errors="replace"
        )
    raw = re.split(r"\n\s*", raw.strip(), maxsplit=1)[0].strip()
    return raw[:500]


def _is_exported(node: Node) -> bool:
    parent = node.parent
    if parent is None:
        return False
    if parent.type == "export_statement":
        return True
    if parent.type in {"lexical_declaration", "variable_declaration"}:
        return parent.parent is not None and parent.parent.type == "export_statement"
    return False


class ECMAScriptAnalyzer:
    """Shared tree-sitter logic for JavaScript and TypeScript."""

    name = "javascript"
    extensions = (".js", ".jsx", ".mjs", ".cjs")
    language = Language(tree_sitter_javascript.language())

    def iter_files(
        self,
        repo_path: Path,
        skip_parts: set[str],
        exclude_dirs: Iterable[Path],
    ) -> list[Path]:
        excluded = [Path(item).resolve() for item in exclude_dirs]
        all_skip = set(skip_parts) | _SKIP_PARTS
        files: list[Path] = []
        for path in sorted(repo_path.rglob("*")):
            if not path.is_file() or path.suffix not in self.extensions:
                continue
            rel_parts = path.relative_to(repo_path).parts
            if any(part in all_skip for part in rel_parts):
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
        for node in root.named_children:
            if node.type == "comment":
                return {"docstring": text_of(source, node).strip()}
            break
        return {"docstring": None}

    def parse_imports(
        self, source: bytes, root: Node, rel_path: str
    ) -> list[ImportSpec]:
        specs: list[ImportSpec] = []
        for node in walk_nodes(root):
            if node.type not in {"import_statement", "export_statement"}:
                continue
            if (
                node.type == "export_statement"
                and node.child_by_field_name("source") is None
            ):
                continue
            module = _string_value(source, node.child_by_field_name("source"))
            if not module:
                continue
            location = location_of(rel_path, node)
            if node.type == "export_statement":
                specs.append(ImportSpec(module, None, None, location))
                continue
            clause = next(
                (
                    child
                    for child in node.named_children
                    if child.type == "import_clause"
                ),
                None,
            )
            if clause is None:
                # Side-effect import: import "./setup"
                specs.append(ImportSpec(module, None, None, location))
                continue
            for child in clause.named_children:
                if child.type == "identifier":
                    specs.append(
                        ImportSpec(module, "default", text_of(source, child), location)
                    )
                elif child.type == "namespace_import":
                    identifiers = [
                        item
                        for item in child.named_children
                        if item.type == "identifier"
                    ]
                    if identifiers:
                        specs.append(
                            ImportSpec(
                                module, None, text_of(source, identifiers[-1]), location
                            )
                        )
                elif child.type == "named_imports":
                    for spec in child.named_children:
                        if spec.type != "import_specifier":
                            continue
                        name = spec.child_by_field_name("name")
                        alias = spec.child_by_field_name("alias")
                        if name is None:
                            continue
                        specs.append(
                            ImportSpec(
                                module,
                                text_of(source, name),
                                text_of(source, alias) if alias else None,
                                location,
                            )
                        )
        # CommonJS require() declarations.
        for node in walk_nodes(root):
            if node.type != "variable_declarator":
                continue
            value = node.child_by_field_name("value")
            if value is None or value.type != "call_expression":
                continue
            function = value.child_by_field_name("function")
            if function is None or text_of(source, function) != "require":
                continue
            arguments = value.child_by_field_name("arguments")
            if arguments is None:
                continue
            string = next(
                (child for child in arguments.named_children if child.type == "string"),
                None,
            )
            module = _string_value(source, string)
            if not module:
                continue
            location = location_of(rel_path, node)
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue
            if name_node.type == "identifier":
                specs.append(
                    ImportSpec(module, "default", text_of(source, name_node), location)
                )
            elif name_node.type == "object_pattern":
                for item in name_node.named_children:
                    if item.type in {
                        "shorthand_property_identifier_pattern",
                        "identifier",
                    }:
                        specs.append(
                            ImportSpec(module, text_of(source, item), None, location)
                        )
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
        marks = set(decorators or [])

        if node.type in {"export_statement", "ambient_declaration"}:
            declaration = node.child_by_field_name("declaration")
            if declaration is not None:
                self.visit_definitions(
                    builder,
                    declaration,
                    source,
                    rel_path,
                    parent_id,
                    class_stack,
                    function_stack,
                    info,
                    [*marks, "export"],
                )
            return

        if node.type in {
            "function_declaration",
            "function_expression",
            "generator_function_declaration",
            "generator_function",
        }:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return
            self._add_function(
                builder,
                node,
                name_node,
                source,
                rel_path,
                parent_id,
                class_stack,
                function_stack,
                info,
                exported="export" in marks or _is_exported(node),
            )
            return

        if node.type == "method_definition":
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return
            self._add_function(
                builder,
                node,
                name_node,
                source,
                rel_path,
                parent_id,
                class_stack,
                function_stack,
                info,
                exported="export" in marks,
                force_kind="method",
            )
            return

        if node.type in {"class_declaration", "class_expression"}:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return
            name = text_of(source, name_node)
            qualified = ".".join([*class_stack, *function_stack, name])
            symbol_id = symbol_node_id(rel_path, qualified, "class")
            exported = "export" in marks or _is_exported(node)
            builder.graph.add_node(
                "symbol",
                symbol_id,
                {
                    "name": name,
                    "qualified_name": qualified,
                    "kind": "class",
                    "path": rel_path,
                    "language": self.name,
                    "signature": _short_signature(source, node),
                    "source": text_of(source, node),
                    "docstring": None,
                    "decorators": sorted(marks),
                    "exported": exported,
                    "visibility": "public" if exported else "private",
                    **span_of(node),
                },
            )
            builder.graph.add_edge("contains", parent_id, symbol_id)
            info.symbols.setdefault(name, []).append(symbol_id)
            if exported:
                api_id = api_node_id(rel_path, qualified)
                builder.graph.add_node(
                    "api",
                    api_id,
                    {
                        "name": name,
                        "qualified_name": qualified,
                        "kind": "class",
                        "language": self.name,
                        "signature": _short_signature(source, node),
                        "symbol_id": symbol_id,
                        "path": rel_path,
                        "visibility": "public",
                    },
                )
                builder.graph.add_edge("exposes_api", info.node_id, api_id)
                builder.graph.add_edge("handles", api_id, symbol_id)
            body = node.child_by_field_name("body")
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
                        [*marks, "export"] if exported else [],
                    )
            return

        if node.type in {
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
        }:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return
            name = text_of(source, name_node)
            qualified = ".".join([*class_stack, *function_stack, name])
            kind = node.type.replace("_declaration", "").replace("_alias", "")
            symbol_id = symbol_node_id(rel_path, qualified, kind)
            exported = "export" in marks or _is_exported(node)
            builder.graph.add_node(
                "symbol",
                symbol_id,
                {
                    "name": name,
                    "qualified_name": qualified,
                    "kind": kind,
                    "path": rel_path,
                    "language": self.name,
                    "signature": _short_signature(source, node),
                    "source": text_of(source, node),
                    "docstring": None,
                    "decorators": sorted(marks),
                    "exported": exported,
                    "visibility": "public" if exported else "private",
                    **span_of(node),
                },
            )
            builder.graph.add_edge("contains", parent_id, symbol_id)
            info.symbols.setdefault(name, []).append(symbol_id)
            return

        if node.type in {"lexical_declaration", "variable_declaration"}:
            exported = "export" in marks or _is_exported(node)
            for declarator in node.named_children:
                if declarator.type != "variable_declarator":
                    continue
                name_node = declarator.child_by_field_name("name")
                value = declarator.child_by_field_name("value")
                if (
                    name_node is None
                    or name_node.type != "identifier"
                    or value is None
                    or value.type not in {"arrow_function", "function_expression"}
                ):
                    continue
                self._add_function(
                    builder,
                    value,
                    name_node,
                    source,
                    rel_path,
                    parent_id,
                    class_stack,
                    function_stack,
                    info,
                    exported=exported,
                    signature_node=declarator,
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
        name_node: Node,
        source: bytes,
        rel_path: str,
        parent_id: str,
        class_stack: list[str],
        function_stack: list[str],
        info: FileInfo,
        *,
        exported: bool,
        force_kind: str | None = None,
        signature_node: Node | None = None,
    ) -> None:
        name = text_of(source, name_node).strip("\"'")
        if not name:
            return
        qualified = ".".join([*class_stack, *function_stack, name])
        kind = force_kind or ("method" if class_stack else "function")
        symbol_id = symbol_node_id(rel_path, qualified, kind)
        signature = _short_signature(source, signature_node or node)
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
                "signature": signature,
                "source": source_text,
                "docstring": None,
                "decorators": ["export"] if exported else [],
                "exported": exported,
                "is_async": source_text.lstrip().startswith("async "),
                "visibility": "public"
                if exported and not name.startswith("_")
                else "private",
                **span_of(node),
            },
        )
        builder.graph.add_edge("contains", parent_id, symbol_id)
        info.symbols.setdefault(name, []).append(symbol_id)

        if exported and not name.startswith("_") and not name.startswith("#"):
            api_id = api_node_id(rel_path, qualified)
            builder.graph.add_node(
                "api",
                api_id,
                {
                    "name": name,
                    "qualified_name": qualified,
                    "kind": kind,
                    "language": self.name,
                    "signature": signature,
                    "symbol_id": symbol_id,
                    "path": rel_path,
                    "visibility": "public",
                },
            )
            builder.graph.add_edge("exposes_api", info.node_id, api_id)
            builder.graph.add_edge("handles", api_id, symbol_id)

        body = node.child_by_field_name("body")
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
            "function_declaration",
            "function_expression",
            "arrow_function",
            "class_declaration",
            "class_expression",
            "method_definition",
        }:
            return
        if node.type in {"call_expression", "new_expression"}:
            function = node.child_by_field_name("function") or node.child_by_field_name(
                "constructor"
            )
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
        if not module.startswith("."):
            return None
        base = (Path(importer).parent / module).as_posix()
        candidates = [base]
        for suffix in _SOURCE_EXTENSIONS:
            candidates.append(base + suffix)
            candidates.append((Path(base) / f"index{suffix}").as_posix())
        for candidate in candidates:
            rel = Path(candidate).as_posix().lstrip("./")
            if rel in builder.file_infos:
                return rel
        return None

    def resolve_callee(
        self, builder: Any, info: FileInfo, call: CallSpec
    ) -> str | None:
        callee = call.callee
        if callee.startswith(("this.", "self.", "super.")):
            return builder._find_symbol(
                info.rel_path, callee.split(".")[-1], prefer_kind="method"
            )
        if "." in callee:
            head, tail = callee.split(".", 1)
            if head in info.import_aliases:
                target_rel, imported = info.import_aliases[head]
                if target_rel:
                    target_name = (
                        imported
                        if imported not in {None, "default"}
                        else tail.split(".")[-1]
                    )
                    found = builder._find_symbol(target_rel, target_name)
                    if found:
                        return found
            return builder._find_symbol(info.rel_path, tail.split(".")[-1])
        local = builder._find_symbol(info.rel_path, callee)
        if local is not None:
            return local
        if callee in info.import_aliases:
            target_rel, imported = info.import_aliases[callee]
            if target_rel:
                if imported in {None, "default"}:
                    found = builder._find_symbol(target_rel, callee)
                    if found:
                        return found
                    return self._find_exported(builder, target_rel)
                return builder._find_symbol(target_rel, imported)
        return None

    def _find_exported(self, builder: Any, rel_path: str) -> str | None:
        info = builder.file_infos.get(rel_path)
        if info is None:
            return None
        for symbol_ids in info.symbols.values():
            for symbol_id in symbol_ids:
                if builder.graph.nodes[symbol_id]["attrs"].get("exported"):
                    return symbol_id
        return None


class JavaScriptAnalyzer(ECMAScriptAnalyzer):
    name = "javascript"
    extensions = (".js", ".jsx", ".mjs", ".cjs")
    language = Language(tree_sitter_javascript.language())


class TypeScriptAnalyzer(ECMAScriptAnalyzer):
    name = "typescript"
    extensions = (".ts", ".mts", ".cts")
    language = Language(tree_sitter_typescript.language_typescript())


class TsxAnalyzer(TypeScriptAnalyzer):
    name = "typescript"
    # TSX is a superset of TS and is used when a repository contains any
    # .tsx file, so the analyzer can cover mixed .ts/.tsx trees.
    extensions = (".tsx", ".ts", ".mts", ".cts")
    language = Language(tree_sitter_typescript.language_tsx())
