"""C# analyzer using tree-sitter-c_sharp."""

import logging
from typing import List, Tuple

from tree_sitter import Parser, Language
import tree_sitter_c_sharp

from repodoc.graph.analyzers.base import BaseAnalyzer
from repodoc.graph.models import (
    REL_IMPLEMENTS,
    REL_INSTANTIATES,
    REL_USES,
    CallRelationship,
    Node,
)

logger = logging.getLogger(__name__)


class TreeSitterCSharpAnalyzer(BaseAnalyzer):
    EXTENSIONS = (".cs",)

    def analyze(self) -> None:
        language_capsule = tree_sitter_c_sharp.language()
        cs_language = Language(language_capsule)
        parser = Parser(cs_language)
        tree = parser.parse(self._content_bytes)
        root = tree.root_node
        lines = self.content.splitlines()

        self._extract_imports(root)
        self._extract_nodes(root, lines)
        self._extract_relationships(root)

    def _extract_imports(self, root) -> None:
        """Record `using` directives.

        Plain/static usings contribute a wildcard namespace (C# exposes the
        namespace's simple names directly); aliased usings (`using X = A.B;`)
        bind the alias."""
        from repodoc.graph.imports import register_namespace

        for node in root.children:
            if node.type != "using_directive":
                continue
            target_node = next(
                (
                    c
                    for c in node.children
                    if c.type in ("qualified_name", "identifier")
                ),
                None,
            )
            if target_node is None:
                continue
            target = self._get_node_text(target_node).replace(" ", "")
            alias_node = None
            seen_equals = False
            for child in node.children:
                if child.type == "=":
                    seen_equals = True
                elif seen_equals and child.type == "identifier":
                    alias_node = child
                    break
            if alias_node is not None:
                self.imports[self._get_node_text(alias_node)] = target
            else:
                register_namespace(self.imports, target)

    def _extract_nodes(self, node, lines):
        node_type = None
        node_name = None
        parameters_for_node = None

        if node.type == "method_declaration":
            containing_class = self._find_containing_class(node)
            method_name = self._method_name(node)
            if containing_class and method_name:
                node_type = "method"
                # Qualified name mirrors the Java analyzer: methods belong to
                # their class's component (and are excluded from leaf docs).
                node_name = f"{containing_class}.{method_name}"
                parameters_for_node = self._extract_method_parameters(node)

        if node.type == "class_declaration":
            is_abstract = any(
                c.type == "modifier" and "abstract" in c.text.decode()
                for c in node.children
            )
            is_static = any(
                c.type == "modifier" and "static" in c.text.decode()
                for c in node.children
            )
            if is_static:
                node_type = "static class"
            elif is_abstract:
                node_type = "abstract class"
            else:
                node_type = "class"
            found_class_keyword = False
            for child in node.children:
                if child.type == "class":
                    found_class_keyword = True
                elif found_class_keyword and child.type == "identifier":
                    node_name = child.text.decode()
                    break
        elif node.type == "interface_declaration":
            node_type = "interface"
            found_interface_keyword = False
            for child in node.children:
                if child.type == "interface":
                    found_interface_keyword = True
                elif found_interface_keyword and child.type == "identifier":
                    node_name = child.text.decode()
                    break
        elif node.type == "struct_declaration":
            node_type = "struct"
            found_struct_keyword = False
            for child in node.children:
                if child.type == "struct":
                    found_struct_keyword = True
                elif found_struct_keyword and child.type == "identifier":
                    node_name = child.text.decode()
                    break
        elif node.type == "enum_declaration":
            node_type = "enum"
            found_enum_keyword = False
            for child in node.children:
                if child.type == "enum":
                    found_enum_keyword = True
                elif found_enum_keyword and child.type == "identifier":
                    node_name = child.text.decode()
                    break
        elif node.type == "record_declaration":
            node_type = "record"
            found_record_keyword = False
            for child in node.children:
                if child.type == "record":
                    found_record_keyword = True
                elif found_record_keyword and child.type == "identifier":
                    node_name = child.text.decode()
                    break
        elif node.type == "delegate_declaration":
            node_type = "delegate"
            for child in node.children:
                if child.type == "identifier":
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
                parameters=parameters_for_node,
            )
            self.nodes.append(node_obj)
            self.top_level_nodes[node_name] = node_obj

        for child in node.children:
            self._extract_nodes(child, lines)

    def _extract_relationships(self, node):
        containing_class = self._find_containing_class(node)

        if node.type == "class_declaration":
            class_name = self._get_identifier_name_cs(node)
            if class_name:
                class_component_id = self._get_component_id(class_name)

                base_list = next(
                    (c for c in node.children if c.type == "base_list"), None
                )
                if base_list:
                    for child in base_list.children:
                        if child.type == "identifier":
                            base_name = child.text.decode()
                            if base_name in [n.name for n in self.top_level_nodes.values()]:
                                base_component_id = self._get_component_id(base_name)
                                self._add_relationship(
                                    caller=class_component_id,
                                    callee=base_component_id,
                                    call_line=node.start_point[0] + 1,
                                    is_resolved=True,
                                    relation_type=REL_IMPLEMENTS,
                                )

        elif node.type == "property_declaration":
            if containing_class:
                containing_class_id = self._get_component_id(containing_class)
                type_identifiers = [c for c in node.children if c.type == "identifier"]
                if len(type_identifiers) >= 2:
                    property_type = type_identifiers[0].text.decode()
                    if property_type and not self._is_primitive_type(property_type):
                        self._add_relationship(
                            caller=containing_class_id,
                            callee=property_type,
                            call_line=node.start_point[0] + 1,
                            relation_type=REL_USES,
                        )

        elif node.type == "field_declaration":
            if containing_class:
                containing_class_id = self._get_component_id(containing_class)
                type_node = next(
                    (c for c in node.children if c.type == "identifier"), None
                )
                if type_node:
                    field_type = type_node.text.decode()
                    if field_type and not self._is_primitive_type(field_type):
                        self._add_relationship(
                            caller=containing_class_id,
                            callee=field_type,
                            call_line=node.start_point[0] + 1,
                            relation_type=REL_USES,
                        )

        elif node.type == "method_declaration":
            if containing_class:
                containing_class_id = self._get_component_id(containing_class)
                param_list = next(
                    (c for c in node.children if c.type == "parameter_list"), None
                )
                if param_list:
                    for child in param_list.children:
                        if child.type == "parameter":
                            type_node = next(
                                (c for c in child.children if c.type == "identifier"),
                                None,
                            )
                            if type_node:
                                param_type = type_node.text.decode()
                                if param_type and not self._is_primitive_type(
                                    param_type
                                ):
                                    self._add_relationship(
                                        caller=containing_class_id,
                                        callee=param_type,
                                        call_line=node.start_point[0] + 1,
                                        relation_type=REL_USES,
                                    )

        elif node.type == "object_creation_expression":
            # `new User()` — the constructor target of the containing method.
            caller_id = self._relation_caller(node, containing_class)
            if caller_id:
                type_name = self._creation_type_name(node)
                if type_name and not self._is_primitive_type(type_name):
                    self._add_relationship(
                        caller=caller_id,
                        callee=type_name,
                        call_line=node.start_point[0] + 1,
                        relation_type=REL_INSTANTIATES,
                    )

        elif node.type == "invocation_expression":
            # `obj.Method()` / `Method()` — record the bare method name so
            # import-aware resolution can place it.
            caller_id = self._relation_caller(node, containing_class)
            if caller_id:
                method_name = self._invocation_name(node)
                if method_name:
                    self._add_relationship(
                        caller=caller_id,
                        callee=method_name,
                        call_line=node.start_point[0] + 1,
                    )

        for child in node.children:
            self._extract_relationships(child)

    def _creation_type_name(self, node) -> str:
        """Type name of an object_creation_expression (`new X<T>(...)` -> X)."""
        for child in node.children:
            if child.type == "identifier":
                return self._get_node_text(child)
            if child.type in ("qualified_name", "alias_qualified_name"):
                return self._get_node_text(child).replace(" ", "")
            if child.type == "generic_name":
                name_node = next(
                    (c for c in child.children if c.type == "identifier"), None
                )
                if name_node:
                    return self._get_node_text(name_node)
        return ""

    def _invocation_name(self, node) -> str:
        """Method name of an invocation_expression (`a.B.M()` -> M)."""
        if not node.children:
            return ""
        target = node.children[0]
        if target.type == "identifier":
            return self._get_node_text(target)
        if target.type in ("member_access_expression", "generic_name"):
            names = [
                c
                for c in target.children
                if c.type in ("identifier", "generic_name")
            ]
            if names:
                last = names[-1]
                if last.type == "generic_name":
                    ident = next(
                        (c for c in last.children if c.type == "identifier"),
                        None,
                    )
                    return self._get_node_text(ident) if ident else ""
                return self._get_node_text(last)
        return ""

    def _is_primitive_type(self, type_name: str) -> bool:
        """Check if type is a C# primitive or common built-in type."""
        primitives = {
            "bool", "byte", "sbyte", "char", "decimal", "double", "float",
            "int", "uint", "long", "ulong", "short", "ushort", "string",
            "object", "void", "Boolean", "Byte", "SByte", "Char", "Decimal",
            "Double", "Single", "Int32", "UInt32", "Int64", "UInt64", "Int16",
            "UInt16", "String", "Object", "Void", "List", "Dictionary", "IList",
            "IDictionary", "IEnumerable", "ICollection", "Task",
            "CancellationToken", "DateTime", "TimeSpan", "Guid",
        }
        return type_name in primitives

    def _get_identifier_name_cs(self, node):
        if node.type == "class_declaration":
            found_class_keyword = False
            for child in node.children:
                if child.type == "class":
                    found_class_keyword = True
                elif found_class_keyword and child.type == "identifier":
                    return child.text.decode()
        elif node.type == "interface_declaration":
            found_interface_keyword = False
            for child in node.children:
                if child.type == "interface":
                    found_interface_keyword = True
                elif found_interface_keyword and child.type == "identifier":
                    return child.text.decode()
        elif node.type == "struct_declaration":
            found_struct_keyword = False
            for child in node.children:
                if child.type == "struct":
                    found_struct_keyword = True
                elif found_struct_keyword and child.type == "identifier":
                    return child.text.decode()
        name_node = next((c for c in node.children if c.type == "identifier"), None)
        return name_node.text.decode() if name_node else None

    def _find_containing_class(self, node):
        current = node.parent
        while current:
            if current.type in [
                "class_declaration",
                "interface_declaration",
                "struct_declaration",
                "enum_declaration",
                "record_declaration",
                "delegate_declaration",
            ]:
                class_name = self._get_identifier_name_cs(current)
                if class_name and class_name in self.top_level_nodes:
                    return class_name
            current = current.parent
        return None

    def _method_name(self, node) -> str:
        """Method name: the last identifier before the parameter list (the
        return type, when written as an identifier, comes first)."""
        name = ""
        for child in node.children:
            if child.type == "parameter_list":
                break
            if child.type == "identifier":
                name = self._get_node_text(child)
        return name

    def _extract_method_parameters(self, node) -> List[str]:
        params: List[str] = []
        param_list = next(
            (c for c in node.children if c.type == "parameter_list"), None
        )
        if param_list:
            for child in param_list.children:
                if child.type == "parameter":
                    params.append(self._get_node_text(child).replace("\n", " "))
        return params

    def _find_containing_method(self, node) -> str:
        """Qualified name (`Class.Method`) of the enclosing method, if any."""
        current = node.parent
        while current:
            if current.type == "method_declaration":
                cls = self._find_containing_class(current)
                method = self._method_name(current)
                if cls and method:
                    return f"{cls}.{method}"
                return ""
            if current.type in [
                "class_declaration",
                "interface_declaration",
                "struct_declaration",
                "enum_declaration",
                "record_declaration",
            ]:
                return ""
            current = current.parent
        return ""

    def _relation_caller(self, node, containing_class) -> str:
        """Prefer the enclosing method component; fall back to the class."""
        method = self._find_containing_method(node)
        if method:
            return self._get_component_id(method)
        if containing_class:
            return self._get_component_id(containing_class)
        return ""


def analyze_csharp_file(
    file_path: str, content: str, repo_path: str = None
) -> Tuple[List[Node], List[CallRelationship]]:
    analyzer = TreeSitterCSharpAnalyzer(file_path, content, repo_path)
    analyzer.analyze()
    return analyzer.nodes, analyzer.call_relationships
