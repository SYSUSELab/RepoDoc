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
            self._extract_imports(root_node)
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

    def _extract_imports(self, root_node) -> None:
        """Record import bindings: `import "path"` binds the package name,
        `import alias "path"` binds the alias, dot imports contribute a
        wildcard namespace, blank imports contribute nothing."""
        for child in root_node.children:
            if child.type != "import_declaration":
                continue
            for spec in child.children:
                if spec.type == "import_spec":
                    self._register_import_spec(spec)
                elif spec.type == "import_spec_list":
                    for item in spec.children:
                        if item.type == "import_spec":
                            self._register_import_spec(item)

    def _register_import_spec(self, spec) -> None:
        alias = None
        path = ""
        dot_import = False
        for child in spec.children:
            if child.type == "package_identifier":
                alias = self._get_node_text(child)
            elif child.type == "blank_identifier":
                return
            elif child.type == "dot":
                dot_import = True
            elif child.type in (
                "interpreted_string_literal",
                "raw_string_literal",
            ):
                path = self._literal_content(child)
        if not path:
            return
        target = path.replace("/", ".").strip(".")
        if dot_import:
            from repodoc.graph.imports import register_namespace

            register_namespace(self.imports, target)
            return
        local = alias or path.rstrip("/").split("/")[-1]
        if local:
            self.imports[local] = target

    def _literal_content(self, node) -> str:
        """Unquoted content of a Go string literal node."""
        for child in node.children:
            if child.type.endswith("_content"):
                return self._get_node_text(child)
        return self._get_node_text(node).strip('"`')

    def _extract_functions_and_methods(self, root_node) -> None:
        """Extract function and method declarations."""
        self._visit_node(root_node)

    def _visit_node(self, node) -> None:
        """Recursively visit nodes to find type declarations, functions and methods."""
        if node.type in ("function_declaration", "method_declaration"):
            self._extract_function_or_method(node)
        elif node.type == "type_declaration":
            self._extract_type_declaration(node)

        for child in node.children:
            self._visit_node(child)

    def _extract_function_or_method(self, node) -> None:
        """Extract a function or method declaration.

        The FIRST parameter_list after the name is always the parameter list;
        parenthesized results (`(int, error)`) also parse as parameter_list
        and must never be mistaken for parameters. For methods, the receiver
        is the (only) parameter_list before the name."""
        name = None
        name_index = None
        params = []
        is_method = False
        receiver = None

        for index, child in enumerate(node.children):
            # Method names are field_identifier in current tree-sitter-go grammar
            if child.type in ("identifier", "field_identifier"):
                name = self._get_node_text(child)
                name_index = index
                break

        # Legacy grammar: a dedicated receiver node before the name.
        receiver_node = next(
            (c for c in node.children if c.type == "receiver"), None
        )
        if receiver_node is not None:
            receiver = self._extract_receiver(receiver_node)
            is_method = bool(receiver)
        elif node.type == "method_declaration" and name_index is not None:
            receiver_lists = [
                c
                for c in node.children[:name_index]
                if c.type == "parameter_list"
            ]
            if receiver_lists:
                receiver = self._extract_receiver(receiver_lists[0])
                is_method = bool(receiver)

        if name_index is not None:
            after_name = [
                c
                for c in node.children[name_index + 1 :]
                if c.type == "parameter_list"
            ]
            if after_name:
                params = self._extract_params(after_name[0])

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

    def _extract_type_declaration(self, node) -> None:
        """Extract named type declarations: structs and interfaces.

        The declaration wrapper is `type_declaration`; it holds one
        `type_spec` per name (grouped declarations hold several). Only struct
        and interface types become components — aliases and basic types are
        not documentation targets."""
        docstring = self._extract_docstring(node)
        for spec in node.children:
            if spec.type != "type_spec":
                continue
            name = ""
            type_node = None
            for child in spec.children:
                if child.type == "type_identifier":
                    name = self._get_node_text(child)
                elif child.type in ("struct_type", "interface_type"):
                    type_node = child
            if not name or type_node is None:
                continue

            is_interface = type_node.type == "interface_type"
            self._add_type_node(
                name=name,
                component_type="interface" if is_interface else "struct",
                docstring=docstring,
                source_code=self._get_node_text(spec),
                # Interface method signatures double as the method set: used
                # by the doc prompts and by struct-implements inference.
                parameters=(
                    self._interface_method_signatures(type_node)
                    if is_interface
                    else None
                ),
            )
            if not is_interface:
                self.current_struct = name

    def _add_type_node(
        self,
        name: str,
        component_type: str,
        docstring: str,
        source_code: str,
        parameters: Optional[List[str]] = None,
    ) -> None:
        """Append a named-type component (mirrors the struct/interface shape)."""
        component_id = self._get_component_id(name)
        node_obj = Node(
            id=component_id,
            name=name,
            component_type=component_type,
            file_path=str(self.file_path),
            relative_path=self._get_relative_path(),
            docstring=docstring,
            source_code=source_code,
            parameters=parameters,
        )
        self.nodes.append(node_obj)
        self.top_level_nodes[component_id] = node_obj

    def _interface_method_signatures(self, interface_node) -> List[str]:
        """Method signatures ("Name/arity") declared by an interface.

        Arity is the parameter count of the first parameter list (the second,
        optional one holds results). Embedded interfaces are skipped. Used by
        the struct-implements inference, which needs more than bare names."""
        signatures: List[str] = []
        for child in interface_node.children:
            if child.type != "method_elem":
                continue
            name = ""
            arity = 0
            first_params = True
            for method_child in child.children:
                if method_child.type == "field_identifier" and not name:
                    name = self._get_node_text(method_child)
                elif method_child.type == "parameter_list" and first_params:
                    arity = sum(
                        1
                        for c in method_child.children
                        if c.type == "parameter_declaration"
                    )
                    first_params = False
            if name:
                signatures.append(f"{name}/{arity}")
        return signatures

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
