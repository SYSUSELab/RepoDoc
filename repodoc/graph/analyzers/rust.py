"""Rust analyzer using tree-sitter-rust."""

import logging
import traceback
from typing import List, Optional, Tuple

from tree_sitter import Parser, Language
import tree_sitter_rust

from repodoc.graph.analyzers.base import BaseAnalyzer
from repodoc.graph.models import CallRelationship, Node

logger = logging.getLogger(__name__)


class RustASTAnalyzer(BaseAnalyzer):
    EXTENSIONS = (".rs",)

    def __init__(self, file_path: str, content: str, repo_path: str = None):
        super().__init__(file_path, content, repo_path)
        self.current_impl = None
        self.current_trait = None

        try:
            language_capsule = tree_sitter_rust.language()
            self.rust_language = Language(language_capsule)
            self.parser = Parser(self.rust_language)
        except Exception as e:
            logger.error(f"Failed to initialize Rust parser: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.parser = None
            self.rust_language = None

    def analyze(self) -> None:
        if self.parser is None:
            logger.warning(f"Skipping {self.file_path} - parser initialization failed")
            return

        try:
            tree = self.parser.parse(self._content_bytes)
            root_node = tree.root_node

            self._extract_use_imports(root_node)
            self._extract_items(root_node)
            self._extract_calls(root_node)
        except Exception as e:
            logger.error(
                f"Error analyzing Rust file {self.file_path}: {e}", exc_info=True
            )

    def _extract_use_imports(self, root_node) -> None:
        """Record `use` bindings for import-aware reference resolution.

        `crate::` paths are stripped to repo-relative form (a component id
        carries its file prefix, so suffix matching finds it); `self::` and
        `super::` are resolved against the caller's module path."""
        current_module = self._get_module_path()

        def visit(node) -> None:
            if node.type == "use_declaration":
                for child in node.children:
                    if child.type in (
                        "scoped_identifier",
                        "identifier",
                        "use_as_clause",
                        "scoped_use_list",
                        "use_wildcard",
                    ):
                        self._register_use_tree(child, "", current_module)
            for child in node.children:
                visit(child)

        visit(root_node)

    def _resolve_rust_segments(self, segments, current_module: str) -> str:
        """Turn `crate::/self::/super::` use segments into a dotted path."""
        segments = [s for s in segments if s]
        if not segments:
            return ""
        head = segments[0]
        if head == "crate":
            return ".".join(segments[1:])
        if head == "self":
            return ".".join(
                current_module.split(".") + segments[1:]
            )
        if head == "super":
            base = current_module.split(".")
            rest = segments
            while rest and rest[0] == "super":
                base = base[:-1]
                rest = rest[1:]
            return ".".join(base + rest)
        return ".".join(segments)

    def _use_node_target(self, node, current_module: str) -> str:
        text = self._get_node_text(node).replace(" ", "")
        return self._resolve_rust_segments(text.split("::"), current_module)

    def _register_use_tree(self, node, prefix: str, current_module: str) -> None:
        from repodoc.graph.imports import register_namespace

        if node.type == "identifier":
            name = self._get_node_text(node)
            self.imports[name] = f"{prefix}.{name}" if prefix else name
        elif node.type == "scoped_identifier":
            target = self._use_node_target(node, current_module)
            if target:
                self.imports[target.split(".")[-1]] = target
        elif node.type == "use_as_clause":
            inner = next(
                (
                    c
                    for c in node.children
                    if c.type in ("scoped_identifier", "identifier")
                ),
                None,
            )
            alias = next(
                (
                    self._get_node_text(c)
                    for c in node.children
                    if c.type == "identifier"
                ),
                None,
            )
            if inner is None or not alias:
                return
            if inner.type == "identifier":
                name = self._get_node_text(inner)
                target = f"{prefix}.{name}" if prefix else name
            else:
                target = self._use_node_target(inner, current_module)
            if target:
                self.imports[alias] = target
        elif node.type == "use_wildcard":
            text = self._get_node_text(node).replace(" ", "")
            path = text.rstrip("*").rstrip(":")
            target = self._resolve_rust_segments(path.split("::"), current_module)
            if target:
                register_namespace(self.imports, target)
        elif node.type == "scoped_use_list":
            prefix_node = next(
                (c for c in node.children if c.type == "scoped_identifier"),
                None,
            )
            base = (
                self._use_node_target(prefix_node, current_module)
                if prefix_node is not None
                else prefix
            )
            for item in node.children:
                if item.type != "use_list":
                    continue
                for entry in item.children:
                    if entry.type == "identifier":
                        name = self._get_node_text(entry)
                        self.imports[name] = f"{base}.{name}" if base else name
                    elif entry.type in (
                        "use_as_clause",
                        "scoped_identifier",
                        "use_wildcard",
                    ):
                        self._register_use_tree(entry, base, current_module)
                    elif entry.type == "self":
                        # `use a::{self, b}` also imports module `a` itself.
                        if base:
                            self.imports[base.split(".")[-1]] = base

    def _extract_items(self, root_node) -> None:
        """Extract top-level items (functions, structs, impls, traits, etc.)."""
        self._visit_item(root_node)

    def _visit_item(self, node) -> None:
        """Recursively visit nodes to find items."""
        if node.type == "function_item":
            self._extract_function(node)
        elif node.type == "struct_item":
            self._extract_struct(node)
        elif node.type == "impl_item":
            self._extract_impl(node)
        elif node.type == "trait_item":
            self._extract_trait(node)
        elif node.type == "enum_item":
            self._extract_enum(node)

        for child in node.children:
            self._visit_item(child)

    def _extract_function(self, node) -> None:
        """Extract a function declaration."""
        name = None
        params = []

        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_text(child)
            elif child.type == "parameters":
                params = self._extract_params(child)

        if not name:
            return

        component_id = self._get_component_id(name)

        node_obj = Node(
            id=component_id,
            name=name,
            component_type="function",
            file_path=str(self.file_path),
            relative_path=self._get_relative_path(),
            docstring=self._extract_docstring(node),
            source_code=self._get_node_text(node),
            parameters=params,
        )

        self.nodes.append(node_obj)
        self.top_level_nodes[component_id] = node_obj

    def _extract_struct(self, node) -> None:
        """Extract a struct declaration."""
        name = None

        for child in node.children:
            if child.type == "type_identifier":
                name = self._get_node_text(child)
                break
            elif child.type == "field_declaration":
                for fc in child.children:
                    if fc.type == "field_identifier":
                        name = self._get_node_text(fc)
                        break

        if not name:
            return

        component_id = self._get_component_id(name)

        node_obj = Node(
            id=component_id,
            name=name,
            component_type="struct",
            file_path=str(self.file_path),
            relative_path=self._get_relative_path(),
            docstring=self._extract_docstring(node),
            source_code=self._get_node_text(node),
        )

        self.nodes.append(node_obj)
        self.top_level_nodes[component_id] = node_obj

    def _extract_impl(self, node) -> None:
        """Extract an impl block."""
        impl_type = None

        for child in node.children:
            if child.type == "type_identifier":
                impl_type = self._get_node_text(child)
                break

        if not impl_type:
            return

        old_impl = self.current_impl
        self.current_impl = impl_type

        for child in node.children:
            if child.type == "declaration_list":
                for decl in child.children:
                    if decl.type == "function_item":
                        self._extract_impl_method(decl, impl_type)

        self.current_impl = old_impl

    def _extract_impl_method(self, node, impl_type: str) -> None:
        """Extract a method within an impl block."""
        name = None
        params = []

        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_text(child)
            elif child.type == "parameters":
                params = self._extract_params(child)

        if not name:
            return

        component_id = self._get_component_id(name, impl_type)

        node_obj = Node(
            id=component_id,
            name=name,
            component_type="method",
            file_path=str(self.file_path),
            relative_path=self._get_relative_path(),
            docstring=self._extract_docstring(node),
            source_code=self._get_node_text(node),
            parameters=params,
            receiver=impl_type,
        )

        self.nodes.append(node_obj)
        self.top_level_nodes[component_id] = node_obj

    def _extract_trait(self, node) -> None:
        """Extract a trait declaration."""
        name = None

        for child in node.children:
            if child.type == "type_identifier":
                name = self._get_node_text(child)
                break

        if not name:
            return

        component_id = self._get_component_id(name)

        node_obj = Node(
            id=component_id,
            name=name,
            component_type="trait",
            file_path=str(self.file_path),
            relative_path=self._get_relative_path(),
            docstring=self._extract_docstring(node),
            source_code=self._get_node_text(node),
        )

        self.nodes.append(node_obj)
        self.top_level_nodes[component_id] = node_obj
        self.current_trait = name

    def _extract_enum(self, node) -> None:
        """Extract an enum declaration."""
        name = None

        for child in node.children:
            if child.type == "type_identifier":
                name = self._get_node_text(child)
                break

        if not name:
            return

        component_id = self._get_component_id(name)

        node_obj = Node(
            id=component_id,
            name=name,
            component_type="enum",
            file_path=str(self.file_path),
            relative_path=self._get_relative_path(),
            docstring=self._extract_docstring(node),
            source_code=self._get_node_text(node),
        )

        self.nodes.append(node_obj)
        self.top_level_nodes[component_id] = node_obj

    def _extract_params(self, node) -> List[str]:
        """Extract function parameters."""
        params = []
        for child in node.children:
            if child.type == "parameter":
                param_text = ""
                for pc in child.children:
                    if pc.type in (
                        "identifier",
                        "type_identifier",
                        "mutable_specifier",
                        "reference_expression",
                    ):
                        if param_text:
                            param_text += " "
                        param_text += self._get_node_text(pc)
                if param_text:
                    params.append(param_text)
            elif child.type == "self_parameter":
                params.append("self")
        return params

    def _extract_docstring(self, node) -> str:
        """Extract docstring from comments before a node."""
        prev_sibling = node.prev_named_sibling
        if prev_sibling and prev_sibling.type == "comment":
            return prev_sibling.text.decode()
        return ""

    def _get_component_id(self, name: str, impl_type: str = None) -> str:
        """Get unique component ID (kept as <module>::<name> for compatibility)."""
        module_path = self._get_module_path()

        if impl_type:
            return f"{module_path}.{impl_type}::{name}"
        return f"{module_path}::{name}"

    def _extract_calls(self, root_node) -> None:
        """Extract function call relationships."""
        self._find_calls(root_node)

    def _find_calls(self, node) -> None:
        """Recursively find all call expressions."""
        if node.type in ("call_expression", "method_call_expression"):
            self._extract_call(node)

        for child in node.children:
            self._find_calls(child)

    def _extract_call(self, node) -> None:
        """Extract a function call."""
        func_name = None
        line = node.start_point[0] + 1

        if node.type == "call_expression":
            if node.children:
                first_child = node.children[0]
                if first_child.type == "identifier":
                    func_name = self._get_node_text(first_child)
                elif first_child.type == "field_expression":
                    func_name = self._get_node_text(first_child)
                elif first_child.type == "scoped_identifier":
                    func_name = self._get_node_text(first_child)
        elif node.type == "method_call_expression":
            for child in node.children:
                if child.type == "field_identifier":
                    func_name = self._get_node_text(child)
                    break

        if not func_name:
            return

        caller_id = self._find_current_function(node)

        if caller_id:
            self._add_relationship(caller=caller_id, callee=func_name, call_line=line)

    def _find_current_function(self, node) -> str:
        """Find the current function/method containing this node."""
        parent = node.parent
        while parent:
            if parent.type == "function_item":
                for child in parent.children:
                    if child.type == "identifier":
                        func_name = self._get_node_text(child)
                        return self._get_component_id(func_name)
            elif parent.type == "impl_item":
                impl_type = None
                for child in parent.children:
                    if child.type == "type_identifier":
                        impl_type = self._get_node_text(child)
                        break
                if impl_type:
                    for child in parent.children:
                        if child.type == "declaration_list":
                            for decl in child.children:
                                if decl.type == "function_item":
                                    for dc in decl.children:
                                        if dc.type == "identifier":
                                            method_name = self._get_node_text(dc)
                                            return self._get_component_id(
                                                method_name, impl_type
                                            )
            parent = parent.parent
        return None


def analyze_rust_file(
    file_path: str, content: str, repo_path: str = None
) -> tuple[List[Node], List[CallRelationship]]:
    """Analyze a Rust source file and return nodes and call relationships."""
    analyzer = RustASTAnalyzer(file_path, content, repo_path)
    analyzer.analyze()
    logger.info(
        f"Found {len(analyzer.nodes)} top-level nodes, {len(analyzer.call_relationships)} calls"
    )
    return analyzer.nodes, analyzer.call_relationships
