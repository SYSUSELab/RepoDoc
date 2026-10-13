"""C analyzer using tree-sitter-c."""

import logging
from typing import List, Tuple

from tree_sitter import Parser, Language
import tree_sitter_c

from repodoc.graph.analyzers.base import BaseAnalyzer
from repodoc.graph.models import CallRelationship, Node

logger = logging.getLogger(__name__)


class TreeSitterCAnalyzer(BaseAnalyzer):
    EXTENSIONS = (".c", ".h")

    def analyze(self) -> None:
        language_capsule = tree_sitter_c.language()
        c_language = Language(language_capsule)
        parser = Parser(c_language)
        tree = parser.parse(self._content_bytes)
        root = tree.root_node
        lines = self.content.splitlines()

        self._extract_includes(root)
        self._extract_nodes(root, lines)
        self._extract_relationships(root)

    def _extract_includes(self, root) -> None:
        """Record local `#include "..."` targets as wildcard namespaces.

        Component ids are file-scoped, so the included header's module path
        lets ambiguous bare function names resolve to the right file."""
        from repodoc.graph.imports import (
            register_namespace,
            resolve_include_target,
        )

        rel = self._get_relative_path()
        for node in root.children:
            if node.type != "preproc_include":
                continue
            for child in node.children:
                if child.type != "string_literal":
                    continue
                include = None
                for c in child.children:
                    if c.type == "string_content":
                        include = self._get_node_text(c)
                target = resolve_include_target(rel, include or "")
                if target:
                    register_namespace(self.imports, target)

    def _extract_nodes(self, node, lines):
        """Recursively extract top-level nodes (functions, structs, and global variables)."""
        node_type = None
        node_name = None

        if node.type == "function_definition":
            node_type = "function"
            # look for function_declarator
            declarator = next(
                (c for c in node.children if c.type == "function_declarator"), None
            )
            if declarator:
                identifier = next(
                    (c for c in declarator.children if c.type == "identifier"), None
                )
                if identifier:
                    node_name = identifier.text.decode()
        elif node.type == "struct_specifier":
            # Extract struct definitions: struct Name { ... }
            node_type = "struct"
            for child in node.children:
                if child.type == "type_identifier":
                    node_name = child.text.decode()
                    break
        elif node.type == "type_definition":
            # Handle typedef struct definitions: typedef struct { ... } Name;
            struct_spec = next(
                (c for c in node.children if c.type == "struct_specifier"), None
            )
            if struct_spec:
                node_type = "struct"
                type_declarator = next(
                    (c for c in node.children if c.type == "type_identifier"), None
                )
                if type_declarator:
                    node_name = type_declarator.text.decode()
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
                        pointer_declarator = next(
                            (
                                c
                                for c in child.children
                                if c.type == "pointer_declarator"
                            ),
                            None,
                        )
                        if pointer_declarator:
                            identifier = next(
                                (
                                    c
                                    for c in pointer_declarator.children
                                    if c.type == "identifier"
                                ),
                                None,
                            )
                            if identifier:
                                node_name = identifier.text.decode()
                                break
                    elif child.type == "identifier":
                        node_name = child.text.decode()
                        break

        if node_type and node_name:
            component_id = self._get_component_id(node_name)
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

            if node_type in ["function", "struct"]:
                self.nodes.append(node_obj)
            self.top_level_nodes[node_name] = node_obj

        for child in node.children:
            self._extract_nodes(child, lines)

    def _is_global_variable(self, node) -> bool:
        parent = node.parent
        while parent:
            if parent.type == "function_definition":
                return False
            parent = parent.parent
        return True

    def _extract_relationships(self, node):
        """Extract various types of relationships between top-level nodes."""

        # 1. function calls other functions
        if node.type == "call_expression":
            containing_function = self._find_containing_function(node)
            if containing_function:
                containing_function_id = self._get_component_id(containing_function)

                function_node = next(
                    (c for c in node.children if c.type == "identifier"), None
                )
                if function_node:
                    called_function = function_node.text.decode()
                    if not self._is_system_function(called_function):
                        self._add_relationship(
                            caller=containing_function_id,
                            callee=called_function,  # Simple name: cross-file resolution
                            call_line=node.start_point[0] + 1,
                        )

        # 2. function uses global variables
        if node.type == "identifier":
            containing_function = self._find_containing_function(node)
            if containing_function:
                var_name = node.text.decode()
                if (
                    var_name in self.top_level_nodes
                    and self.top_level_nodes[var_name].component_type == "variable"
                ):
                    containing_function_id = self._get_component_id(containing_function)
                    var_component_id = self._get_component_id(var_name)
                    self._add_relationship(
                        caller=containing_function_id,
                        callee=var_component_id,
                        call_line=node.start_point[0] + 1,
                        is_resolved=True,  # Local file relationship
                    )

        for child in node.children:
            self._extract_relationships(child)

    def _find_containing_function(self, node):
        """Find the function that contains this node."""
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
                        func_name = identifier.text.decode()
                        if func_name in self.top_level_nodes:
                            return func_name
            current = current.parent
        return None

    def _is_system_function(self, func_name: str) -> bool:
        """Check if function is a system/library function."""
        system_functions = {
            "printf", "scanf", "malloc", "free", "strlen", "strcpy", "strcmp",
            "memcpy", "memset", "exit", "abort", "fopen", "fclose", "fread",
            "fwrite", "SDL_Init", "SDL_CreateWindow", "SDL_Log", "SDL_GetError",
            "SDL_Quit",
        }
        return func_name in system_functions


def analyze_c_file(
    file_path: str, content: str, repo_path: str = None
) -> Tuple[List[Node], List[CallRelationship]]:
    analyzer = TreeSitterCAnalyzer(file_path, content, repo_path)
    analyzer.analyze()
    return analyzer.nodes, analyzer.call_relationships
