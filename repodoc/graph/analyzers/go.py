"""Go analyzer using tree-sitter-go."""

import logging
import traceback
from typing import List, Optional, Tuple

from tree_sitter import Parser, Language
import tree_sitter_go

from repodoc.graph.analyzers.base import BaseAnalyzer
from repodoc.graph.models import CallRelationship, Node

logger = logging.getLogger(__name__)

_GO_LANGUAGE = None
_GO_PARSER = None


def _get_go_parser():
    """Process-wide Go parser (parsing is single-threaded; creating a
    Language/Parser per file is pure overhead)."""
    global _GO_LANGUAGE, _GO_PARSER
    if _GO_PARSER is None:
        _GO_LANGUAGE = Language(tree_sitter_go.language())
        _GO_PARSER = Parser(_GO_LANGUAGE)
    return _GO_PARSER


class GoASTAnalyzer(BaseAnalyzer):
    EXTENSIONS = (".go",)

    def __init__(self, file_path: str, content: str, repo_path: str = None):
        super().__init__(file_path, content, repo_path)
        self.current_package = "main"
        self.current_struct = None

        try:
            self.parser = _get_go_parser()
            self.go_language = _GO_LANGUAGE
        except Exception as e:
            logger.error(f"Failed to initialize Go parser: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.parser = None
            self.go_language = None

    def analyze(self) -> None:
        if self.parser is None:
            logger.warning(f"Skipping {self.file_path} - parser initialization failed")
            return

        try:
            tree = self.parser.parse(self._content_bytes)
            root_node = tree.root_node

            self._extract_package(root_node)
            self._extract_functions_and_methods(root_node)
            self._extract_calls(root_node)
        except Exception as e:
            logger.error(
                f"Error analyzing Go file {self.file_path}: {e}", exc_info=True
            )

    def _extract_package(self, root_node) -> None:
        """Extract package name from the AST."""
        for child in root_node.children:
            if child.type == "package_clause":
                for pkg_child in child.children:
                    if pkg_child.type == "package_identifier":
                        self.current_package = self._get_node_text(pkg_child)
                        break
                break

    def _extract_functions_and_methods(self, root_node) -> None:
        """Extract function and method declarations."""
        self._visit_node(root_node)

    def _visit_node(self, node) -> None:
        """Recursively visit nodes to find functions and methods."""
        if node.type in ("function_declaration", "method_declaration"):
            self._extract_function_or_method(node)
        elif node.type == "struct_type":
            self._extract_struct(node)

        for child in node.children:
            self._visit_node(child)

    def _extract_function_or_method(self, node) -> None:
        """Extract a function or method declaration."""
        name = None
        params = []
        is_method = False
        receiver = None
        seen_param_list = False

        for child in node.children:
            # Method names are field_identifier in current tree-sitter-go grammar
            if child.type in ("identifier", "field_identifier"):
                name = self._get_node_text(child)
            elif child.type == "parameter_list":
                if not seen_param_list:
                    # First parameter_list of a method is the receiver
                    seen_param_list = True
                    receiver = self._extract_receiver(child)
                    if receiver:
                        is_method = True
                else:
                    params = self._extract_params(child)
            elif child.type == "parameter_declaration":
                params.append(self._get_node_text(child))
            elif child.type == "receiver":
                # Legacy grammar: dedicated receiver node
                receiver = self._extract_receiver(child)
                is_method = True

        if not name:
            return

        component_id = self._get_component_id(name, receiver)

        node_obj = Node(
            id=component_id,
            name=name,
            component_type="function" if not is_method else "method",
            file_path=str(self.file_path),
            relative_path=self._get_relative_path(),
            docstring=self._extract_docstring(node),
            source_code=self._get_node_text(node),
            parameters=params,
            receiver=receiver or None,
        )

        self.nodes.append(node_obj)
        self.top_level_nodes[component_id] = node_obj

    def _extract_receiver(self, node) -> str:
        """Extract the receiver TYPE (not the variable name) from a method.

        Handles both grammars: a dedicated "receiver" node (older tree-sitter-go)
        and a leading parameter_list (current grammar).
        """
        if node.type == "parameter_list":
            decl = next(
                (c for c in node.children if c.type == "parameter_declaration"), None
            )
            if decl:
                pointer = next(
                    (c for c in decl.children if c.type == "pointer_type"), None
                )
                if pointer:
                    type_node = next(
                        (c for c in pointer.children if c.type == "type_identifier"),
                        None,
                    )
                    return f"*{type_node.text.decode()}" if type_node else pointer.text.decode()
                type_node = next(
                    (c for c in decl.children if c.type == "type_identifier"), None
                )
                if type_node:
                    return type_node.text.decode()
            return ""
        # Legacy "receiver" node: children include identifier and/or pointer
        for child in node.children:
            if child.type == "pointer":
                return self._get_node_text(child)
            if child.type == "type_identifier":
                return self._get_node_text(child)
        return ""

    def _extract_struct(self, node) -> None:
        """Extract struct declaration."""
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
        self.current_struct = name

    def _extract_params(self, node) -> List[str]:
        """Extract function parameters."""
        params = []
        for child in node.children:
            if child.type == "parameter_declaration":
                param_text = ""
                for pc in child.children:
                    if pc.type in ("identifier", "type_identifier", "pointer"):
                        if param_text:
                            param_text += " "
                        param_text += self._get_node_text(pc)
                if param_text:
                    params.append(param_text)
        return params

    def _extract_docstring(self, node) -> str:
        """Extract docstring from comments before a node."""
        prev_sibling = node.prev_named_sibling
        if prev_sibling and prev_sibling.type == "comment":
            return prev_sibling.text.decode()
        return ""

    def _get_component_id(self, name: str, receiver: str = None) -> str:
        """Get unique component ID."""
        return super()._get_component_id(name, receiver)

    def _extract_calls(self, root_node) -> None:
        """Extract function call relationships."""
        self._find_calls(root_node)

    def _find_calls(self, node) -> None:
        """Recursively find all call expressions."""
        if node.type == "call_expression":
            self._extract_call(node)
        elif node.type == "selector_expression":
            self._extract_selector_call(node)

        for child in node.children:
            self._find_calls(child)

    def _extract_call(self, node) -> None:
        """Extract a function call."""
        func_name = None
        line = node.start_point[0] + 1

        if node.children:
            first_child = node.children[0]
            if first_child.type == "identifier":
                func_name = self._get_node_text(first_child)
            elif first_child.type == "selector_expression":
                func_name = self._get_node_text(first_child)
            elif first_child.type == "qualified_identifier":
                func_name = self._get_node_text(first_child)

        if not func_name:
            return

        caller_id = self._find_current_function(node)

        if caller_id:
            self._add_relationship(caller=caller_id, callee=func_name, call_line=line)

    def _extract_selector_call(self, node) -> None:
        """Extract a selector expression call (e.g., obj.Method())."""
        if len(node.children) < 2:
            return

        field_identifier = None
        for child in reversed(node.children):
            if child.type == "field_identifier":
                field_identifier = self._get_node_text(child)
                break

        if not field_identifier:
            return

        line = node.start_point[0] + 1
        caller_id = self._find_current_function(node)

        if caller_id:
            receiver_text = ""
            for child in node.children:
                if child.type not in ("field_identifier",):
                    receiver_text = self._get_node_text(child)
                    break

            full_callee = (
                f"{receiver_text}.{field_identifier}"
                if receiver_text
                else field_identifier
            )

            self._add_relationship(caller=caller_id, callee=full_callee, call_line=line)

    def _find_current_function(self, node) -> str:
        """Find the current function/method containing this node."""
        parent = node.parent
        while parent:
            if parent.type in ("function_declaration", "method_declaration"):
                for child in parent.children:
                    if child.type in ("identifier", "field_identifier"):
                        func_name = self._get_node_text(child)
                        receiver = None
                        for c in parent.children:
                            if c.type == "receiver":
                                receiver = self._extract_receiver(c)
                            elif (
                                c.type == "parameter_list"
                                and parent.type == "method_declaration"
                                and receiver is None
                            ):
                                receiver = self._extract_receiver(c)
                        return self._get_component_id(func_name, receiver)
            parent = parent.parent
        return None


def analyze_go_file(
    file_path: str, content: str, repo_path: str = None
) -> tuple[List[Node], List[CallRelationship]]:
    """Analyze a Go source file and return nodes and call relationships."""
    analyzer = GoASTAnalyzer(file_path, content, repo_path)
    analyzer.analyze()
    logger.info(
        f"Found {len(analyzer.nodes)} top-level nodes, {len(analyzer.call_relationships)} calls"
    )
    return analyzer.nodes, analyzer.call_relationships
