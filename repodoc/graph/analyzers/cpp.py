"""C++ analyzer using tree-sitter-cpp."""

import logging
from typing import List, Tuple

from tree_sitter import Parser, Language
import tree_sitter_cpp

from repodoc.graph.analyzers.base import BaseAnalyzer
from repodoc.graph.models import REL_IMPLEMENTS, CallRelationship, Node

logger = logging.getLogger(__name__)


class TreeSitterCppAnalyzer(BaseAnalyzer):
    EXTENSIONS = (".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".h")

    def analyze(self) -> None:
        language_capsule = tree_sitter_cpp.language()
        cpp_language = Language(language_capsule)
        parser = Parser(cpp_language)
        tree = parser.parse(self._content_bytes)
        root = tree.root_node
        lines = self.content.splitlines()

        self._extract_nodes(root, lines)
        self._extract_relationships(root)

    def _extract_nodes(self, node, lines):
        """Recursively extract top-level nodes (classes, functions, global variables)."""
        node_type = None
        node_name = None
        containing_class = None

        if node.type == "class_specifier":
            node_type = "class"
            for child in node.children:
                if child.type == "type_identifier":
                    node_name = child.text.decode()
                    break
        elif node.type == "struct_specifier":
            node_type = "struct"
            for child in node.children:
                if child.type == "type_identifier":
                    node_name = child.text.decode()
                    break
        elif node.type == "function_definition":
            containing_class = self._find_containing_class_for_method(node)
            node_type = "method" if containing_class else "function"

            declarator = next(
                (c for c in node.children if c.type == "function_declarator"), None
            )
            if declarator:
                for child in declarator.children:
                    if child.type == "identifier":
                        node_name = child.text.decode()
                        break
                    elif child.type == "field_identifier":
                        node_name = child.text.decode()
                        break
                    elif child.type == "qualified_identifier":
                        identifiers = [
                            c for c in child.children if c.type == "identifier"
                        ]
                        if identifiers:
                            node_name = identifiers[-1].text.decode()
                            break
        elif node.type == "declaration":
            if self._is_global_variable(node):
                node_type = "variable"
                for child in node.children:
                    if child.type == "init_declarator":
                        identifier = next(
                            (c for c in child.children if c.type == "identifier"), None
                        )
                        if identifier:
                            node_name = identifier.text.decode()
                            break
                    elif child.type == "identifier":
                        node_name = child.text.decode()
                        break
        elif node.type == "namespace_definition":
            node_type = "namespace"
            found_namespace_keyword = False
            for child in node.children:
                if child.type == "namespace":
                    found_namespace_keyword = True
                elif found_namespace_keyword and child.type == "identifier":
                    node_name = child.text.decode()
                    break

        if node_type and node_name:
            if node_type == "method":
                component_id = self._get_component_id(node_name, containing_class)
                top_level_key = component_id
            else:
                component_id = self._get_component_id(node_name)
                top_level_key = node_name

            node_obj = Node(
                id=component_id,
                name=node_name,
                component_type=node_type,
                file_path=str(self.file_path),
                relative_path=self._get_relative_path(),
                source_code="\n".join(
                    lines[node.start_point[0] : node.end_point[0] + 1]
                ),
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                docstring="",
            )

            self.top_level_nodes[top_level_key] = node_obj

            if node_type in ["class", "struct", "function"]:
                self.nodes.append(node_obj)

        for child in node.children:
            self._extract_nodes(child, lines)

    def _is_global_variable(self, node) -> bool:
        """Check if a declaration node is a global variable."""
        parent = node.parent
        while parent:
            if parent.type in [
                "function_definition",
                "class_specifier",
                "struct_specifier",
            ]:
                return False
            parent = parent.parent
        return True

    def _find_containing_class_for_method(self, node):
        """Find the class that contains this method definition."""
        current = node.parent
        while current:
            if current.type == "class_specifier":
                for child in current.children:
                    if child.type == "type_identifier":
                        return child.text.decode()
            elif current.type == "struct_specifier":
                for child in current.children:
                    if child.type == "type_identifier":
                        return child.text.decode()
            current = current.parent
        return None

    def _extract_relationships(self, node):
        if node.type == "call_expression":
            containing_function = self._find_containing_function_or_method(node)
            if containing_function:
                containing_function_id = self._get_component_id_for_function(
                    containing_function
                )

                called_function = None
                for child in node.children:
                    if child.type == "identifier":
                        called_function = child.text.decode()
                        break
                    elif child.type == "field_expression":
                        method_name = None
                        for field_child in child.children:
                            if field_child.type == "field_identifier":
                                method_name = field_child.text.decode()
                                break
                        if method_name:
                            called_function = method_name
                            break

                if called_function and not self._is_system_function(called_function):
                    target_class = self._find_class_containing_method(called_function)

                    if target_class:
                        target_class_id = self._get_component_id(target_class)
                        self._add_relationship(
                            caller=containing_function_id,
                            callee=target_class_id,
                            call_line=node.start_point[0] + 1,
                        )
                    elif called_function in self.top_level_nodes:
                        called_function_id = self._get_component_id(called_function)
                        self._add_relationship(
                            caller=containing_function_id,
                            callee=called_function_id,
                            call_line=node.start_point[0] + 1,
                        )

        elif node.type == "base_class_clause":
            containing_class = self._find_containing_class(node)
            if containing_class:
                for child in node.children:
                    if child.type == "type_identifier":
                        base_class = child.text.decode()
                        containing_class_id = self._get_component_id(containing_class)
                        self._add_relationship(
                            caller=containing_class_id,
                            callee=base_class,
                            call_line=node.start_point[0] + 1,
                            relation_type=REL_IMPLEMENTS,
                        )

        elif node.type == "new_expression":
            containing_function = self._find_containing_function_or_method(node)
            if containing_function:
                containing_function_id = self._get_component_id_for_function(
                    containing_function
                )

                for child in node.children:
                    if child.type == "type_identifier":
                        class_name = child.text.decode()
                        if class_name in self.top_level_nodes:
                            class_id = self._get_component_id(class_name)
                            self._add_relationship(
                                caller=containing_function_id,
                                callee=class_id,
                                call_line=node.start_point[0] + 1,
                            )
                        break

        elif node.type == "identifier":
            parent = node.parent
            if parent and parent.type not in [
                "function_definition",
                "class_specifier",
                "declaration",
                "function_declarator",
            ]:
                var_name = node.text.decode()
                if (
                    var_name in self.top_level_nodes
                    and self.top_level_nodes[var_name].component_type == "variable"
                ):
                    containing_function = self._find_containing_function_or_method(
                        node
                    )
                    if containing_function and containing_function != var_name:
                        containing_function_id = self._get_component_id_for_function(
                            containing_function
                        )
                        self._add_relationship(
                            caller=containing_function_id,
                            callee=var_name,
                            call_line=node.start_point[0] + 1,
                        )

        for child in node.children:
            self._extract_relationships(child)

    def _find_containing_function_or_method(self, node):
        """Find the function or method that contains this node."""
        current = node.parent
        while current:
            if current.type == "function_definition":
                declarator = next(
                    (c for c in current.children if c.type == "function_declarator"),
                    None,
                )
                if declarator:
                    identifier = next(
                        (c for c in declarator.children if c.type == "identifier"), None
                    )
                    if identifier:
                        return identifier.text.decode()
            current = current.parent
        return None

    def _get_component_id_for_function(self, func_name):
        if func_name in self.top_level_nodes:
            # Node.id already carries the class prefix for methods
            return self.top_level_nodes[func_name].id
        return self._get_component_id(func_name)

    def _find_containing_class(self, node):
        """Find the class that contains this node."""
        current = node.parent
        while current:
            if current.type == "class_specifier":
                for child in current.children:
                    if child.type == "type_identifier":
                        return child.text.decode()
            current = current.parent
        return None

    def _is_system_function(self, func_name: str) -> bool:
        """Check if function is a system/library function."""
        system_functions = {
            "printf", "scanf", "malloc", "free", "strlen", "strcpy", "strcmp",
            "cout", "cin", "endl", "std", "new", "delete",
        }
        return func_name in system_functions

    def _find_class_containing_method(self, method_name):
        for node_name, node_obj in self.top_level_nodes.items():
            if node_obj.component_type in ["class", "struct"]:
                if self._class_has_method(node_obj, method_name):
                    return node_name
        return None

    def _class_has_method(self, class_node, method_name):
        lines = class_node.source_code.split("\n")
        for line in lines:
            if f"{method_name}(" in line and (
                "void" in line
                or "int" in line
                or "bool" in line
                or class_node.name in line
            ):
                return True
        return False


def analyze_cpp_file(
    file_path: str, content: str, repo_path: str = None
) -> Tuple[List[Node], List[CallRelationship]]:
    analyzer = TreeSitterCppAnalyzer(file_path, content, repo_path)
    analyzer.analyze()
    return analyzer.nodes, analyzer.call_relationships
