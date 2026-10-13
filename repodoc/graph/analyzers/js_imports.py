"""Import extraction shared by the JavaScript and TypeScript analyzers.

Both grammars expose the same `import_statement` shape:

    import Def, { User, Post as P } from './models';
    import * as M from './utils';

Relative specifiers are resolved to repo-relative dotted module paths; bare
specifiers (npm packages, path aliases) are kept as-is — they simply will
not match repository components.
"""

import os
from typing import Callable, Dict


def resolve_module_specifier(relative_path: str, spec: str) -> str:
    """Repo-relative dotted target for an import specifier.

    `./models` from `src/views.ts` -> `src.models`; `../core/base` ->
    `core.base`; bare packages stay unchanged."""
    spec = (spec or "").strip()
    if not spec:
        return ""
    if spec.startswith("."):
        rel = (relative_path or "").replace("\\", "/")
        base_dir = os.path.dirname(rel)
        joined = os.path.normpath(os.path.join(base_dir, spec)).replace(
            os.sep, "/"
        )
        return joined.replace("/", ".")
    return spec.replace("/", ".")


def _string_fragment(node, get_text: Callable) -> str:
    """Unquoted text of a string node (prefers the string_fragment child)."""
    for child in node.children:
        if child.type == "string_fragment":
            return get_text(child)
    return get_text(node).strip("'\"`")


def collect_js_imports(
    root_node,
    get_text: Callable,
    relative_path: str,
) -> Dict[str, str]:
    """Walk a JS/TS syntax tree and return local alias -> dotted target."""
    imports: Dict[str, str] = {}

    def _handle_statement(node) -> None:
        clause = None
        spec = ""
        for child in node.children:
            if child.type == "import_clause":
                clause = child
            elif child.type == "string":
                spec = _string_fragment(child, get_text)
        target = resolve_module_specifier(relative_path, spec)
        if clause is None or not target:
            return  # side-effect import: no local binding

        for child in clause.children:
            if child.type == "identifier":
                # Default import: bind the local name to a same-named member
                # of the target module (falls back to module membership).
                local = get_text(child)
                imports[local] = f"{target}.{local}"
            elif child.type == "namespace_import":
                for c in child.children:
                    if c.type == "identifier":
                        imports[get_text(c)] = target
            elif child.type == "named_imports":
                for specifier in child.children:
                    if specifier.type != "import_specifier":
                        continue
                    names = [
                        get_text(c)
                        for c in specifier.children
                        if c.type == "identifier"
                    ]
                    if not names:
                        continue
                    original = names[0]
                    local = names[1] if len(names) > 1 else original
                    imports[local] = f"{target}.{original}"

    def _walk(node) -> None:
        if node.type == "import_statement":
            _handle_statement(node)
        for child in node.children:
            _walk(child)

    _walk(root_node)
    return imports
