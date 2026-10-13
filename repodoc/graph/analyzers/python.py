"""Python analyzer using the standard library `ast` module."""

import ast
import logging
import warnings
from typing import List, Optional, Tuple

from repodoc.graph.analyzers.base import BaseAnalyzer
from repodoc.graph.models import REL_IMPLEMENTS, CallRelationship, Node

logger = logging.getLogger(__name__)

# Builtin names never treated as dependencies
PYTHON_BUILTINS = {
    "print", "len", "str", "int", "float", "bool", "list", "dict", "tuple",
    "set", "range", "enumerate", "zip", "isinstance", "hasattr", "getattr",
    "setattr", "open", "super", "__import__", "type", "object", "Exception",
    "ValueError", "TypeError", "KeyError", "IndexError", "AttributeError",
    "ImportError", "max", "min", "sum", "abs", "round", "sorted", "reversed",
    "filter", "map", "any", "all", "next", "iter", "callable", "repr",
    "format", "exec", "eval",
}


class PythonASTAnalyzer(BaseAnalyzer, ast.NodeVisitor):
    EXTENSIONS = (".py",)

    def __init__(self, file_path: str, content: str, repo_path: Optional[str] = None):
        super().__init__(file_path, content, repo_path)
        self.lines = content.splitlines()
        self.current_class_name: str | None = None
        self.current_function_name: str | None = None

    def analyze(self) -> None:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=SyntaxWarning)
                tree = ast.parse(self.content)
            self.visit(tree)
        except SyntaxError as e:
            # e.g. Python 2 files (print statements) in otherwise py3 repos
            logger.debug("Could not parse %s: %s", self.file_path, e)
        except Exception as e:
            logger.error("Error analyzing %s: %s", self.file_path, e, exc_info=True)

    def visit_ClassDef(self, node: ast.ClassDef):
        base_classes = [self._extract_base_class_name(base) for base in node.bases]
        base_classes = [name for name in base_classes if name is not None]
        component_id = f"{self._get_module_path()}.{node.name}"
        class_node = Node(
            id=component_id,
            name=node.name,
            component_type="class",
            file_path=str(self.file_path),
            relative_path=self._get_relative_path(),
            source_code="\n".join(
                self.lines[node.lineno - 1 : node.end_lineno or node.lineno]
            ),
            start_line=node.lineno,
            end_line=node.end_lineno,
            docstring=ast.get_docstring(node) or "",
        )
        self.nodes.append(class_node)
        self.top_level_nodes[node.name] = class_node
        for base_name in base_classes:
            if base_name in self.top_level_nodes:
                self._add_relationship(
                    caller=component_id,
                    callee=f"{self._get_module_path()}.{base_name}",
                    call_line=node.lineno,
                    is_resolved=True,
                    relation_type=REL_IMPLEMENTS,
                )
        self.current_class_name = node.name
        self.generic_visit(node)
        self.current_class_name = None

    def _extract_base_class_name(self, base):
        if isinstance(base, ast.Name):
            return base.id
        if isinstance(base, ast.Attribute):
            parts = []
            n = base
            while isinstance(n, ast.Attribute):
                parts.append(n.attr)
                n = n.value
            if isinstance(n, ast.Name):
                parts.append(n.id)
            return ".".join(reversed(parts))
        return None

    def _process_function_node(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        if not self.current_class_name:
            component_id = f"{self._get_module_path()}.{node.name}"
            func_node = Node(
                id=component_id,
                name=node.name,
                component_type="function",
                file_path=str(self.file_path),
                relative_path=self._get_relative_path(),
                source_code="\n".join(
                    self.lines[node.lineno - 1 : node.end_lineno or node.lineno]
                ),
                start_line=node.lineno,
                end_line=node.end_lineno,
                docstring=ast.get_docstring(node) or "",
                parameters=[arg.arg for arg in node.args.args],
            )
            if self._should_include_function(func_node):
                self.nodes.append(func_node)
                self.top_level_nodes[node.name] = func_node
        self.current_function_name = node.name
        self.generic_visit(node)
        self.current_function_name = None

    def _should_include_function(self, func: Node) -> bool:
        if func.name.startswith("test_") or func.name.endswith("_test"):
            return False
        return True

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._process_function_node(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._process_function_node(node)

    def visit_Import(self, node: ast.Import):
        """Record `import a.b.c` / `import a.b.c as x` bindings.

        A plain `import a.b.c` binds only the ROOT package name ("a"), so
        that is what gets registered; `as` binds the alias to the full path."""
        for alias in node.names:
            if alias.asname:
                self.imports[alias.asname] = alias.name
            else:
                root = alias.name.split(".")[0]
                self.imports[root] = root
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Record `from pkg.mod import Name [as Alias]` bindings.

        Relative imports are resolved against the caller's package first, so
        the recorded target is a repo-relative dotted path."""
        module = node.module or ""
        base = self._resolve_relative_module(node.level, module)
        for alias in node.names:
            if alias.name == "*":
                continue
            target = f"{base}.{alias.name}" if base else alias.name
            self.imports[alias.asname or alias.name] = target
        self.generic_visit(node)

    def _resolve_relative_module(self, level: int, module: str) -> str:
        """Resolve a Python relative import to a repo-relative dotted path.

        level 1 = the current package (file's directory), level 2 = its
        parent, and so on (PEP 328 semantics)."""
        if not level:
            return module
        package = self._get_module_path().split(".")[:-1]
        up = level - 1
        if up:
            package = package[: len(package) - up] if up <= len(package) else []
        parts = package + ([module] if module else [])
        return ".".join(parts)

    def visit_Call(self, node: ast.Call):
        if self.current_class_name or (
            self.current_function_name and not self.current_class_name
        ):
            call_name = self._get_call_name(node.func)
            if call_name:
                if self.current_class_name:
                    caller_id = f"{self._get_module_path()}.{self.current_class_name}"
                else:
                    caller_id = (
                        f"{self._get_module_path()}.{self.current_function_name}"
                    )
                callee_id = (
                    f"{self._get_module_path()}.{call_name}"
                    if call_name in self.top_level_nodes
                    else call_name
                )
                self._add_relationship(
                    caller=caller_id,
                    callee=callee_id,
                    call_line=node.lineno,
                    is_resolved=call_name in self.top_level_nodes,
                )
        self.generic_visit(node)

    def _get_call_name(self, node) -> str | None:
        if isinstance(node, ast.Name):
            return None if node.id in PYTHON_BUILTINS else node.id
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                return (
                    None
                    if node.value.id in PYTHON_BUILTINS
                    else f"{node.value.id}.{node.attr}"
                )
            if isinstance(node.value, ast.Attribute):
                base_name = self._get_call_name(node.value)
                return f"{base_name}.{node.attr}" if base_name else node.attr
            return node.attr
        return None


def analyze_python_file(
    file_path: str, content: str, repo_path: Optional[str] = None
) -> Tuple[List[Node], List[CallRelationship]]:
    analyzer = PythonASTAnalyzer(file_path, content, repo_path)
    analyzer.analyze()
    return analyzer.nodes, analyzer.call_relationships
