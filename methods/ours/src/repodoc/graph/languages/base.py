"""Language-agnostic graph builder contracts and shared data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from repodoc.schema import api_node_id, symbol_node_id


@dataclass
class ImportSpec:
    module: str
    name: str | None
    alias: str | None
    location: dict[str, Any]


@dataclass
class CallSpec:
    source_symbol_id: str
    callee: str
    location: dict[str, Any]


@dataclass
class FileInfo:
    rel_path: str
    node_id: str
    path: Path
    language: str = "python"
    comments: list[dict[str, Any]] = field(default_factory=list)
    imports: list[ImportSpec] = field(default_factory=list)
    calls: list[CallSpec] = field(default_factory=list)
    symbols: dict[str, list[str]] = field(default_factory=dict)
    import_aliases: dict[str, tuple[str | None, str | None]] = field(
        default_factory=dict
    )


class LanguageAnalyzer(Protocol):
    """Language-specific parsing hooks used by ``CodeGraphBuilder``.

    Implementations own tree-sitter node names, import syntax, public API rules
    and call/name resolution. The builder owns graph IDs, edges and persistence.
    """

    name: str
    extensions: tuple[str, ...]
    language: Any

    def iter_files(
        self,
        repo_path: Path,
        skip_parts: set[str],
        exclude_dirs: Iterable[Path],
    ) -> list[Path]: ...

    def comments(
        self, source: bytes, root: Any, rel_path: str
    ) -> list[dict[str, Any]]: ...

    def file_metadata(
        self, source: bytes, root: Any, rel_path: str
    ) -> dict[str, Any]: ...

    def parse_imports(
        self, source: bytes, root: Any, rel_path: str
    ) -> list[ImportSpec]: ...

    def visit_definitions(
        self,
        builder: Any,
        node: Any,
        source: bytes,
        rel_path: str,
        parent_id: str,
        class_stack: list[str],
        function_stack: list[str],
        info: FileInfo,
        decorators: list[str] | None = None,
    ) -> None: ...

    def resolve_module(
        self, builder: Any, importer: str, module: str
    ) -> str | None: ...

    def resolve_callee(
        self, builder: Any, info: FileInfo, call: CallSpec
    ) -> str | None: ...


def text_of(source: bytes, node: Any) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def span_of(node: Any) -> dict[str, dict[str, int]]:
    def point(end: bool = False) -> dict[str, int]:
        point_value = node.end_point if end else node.start_point
        return {
            "line": point_value[0] + 1,
            "column": point_value[1] + 1,
            "byte": node.end_byte if end else node.start_byte,
        }

    return {"start": point(), "end": point(end=True)}


def location_of(path: str, node: Any) -> dict[str, Any]:
    return {"path": path, **span_of(node)}


def walk_nodes(root: Any) -> Iterable[Any]:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


DEFAULT_SKIP_PARTS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "coverage",
    "target",
    "vendor",
    "obj",
    "bin",
    "__pycache__",
    ".repodoc",
}


def iter_source_files(
    repo_path: Path,
    extensions: tuple[str, ...],
    skip_parts: set[str],
    exclude_dirs: Iterable[Path],
    extra_skip: Iterable[str] = (),
) -> list[Path]:
    """Discover source files while keeping generated/vendor trees out."""

    excluded = [Path(item).resolve() for item in exclude_dirs]
    all_skip = set(skip_parts) | DEFAULT_SKIP_PARTS | set(extra_skip)
    files: list[Path] = []
    for path in sorted(repo_path.rglob("*")):
        if not path.is_file() or path.suffix not in extensions:
            continue
        rel_parts = path.relative_to(repo_path).parts
        if any(part in all_skip for part in rel_parts):
            continue
        resolved = path.resolve()
        if any(resolved.is_relative_to(item) for item in excluded):
            continue
        files.append(path)
    return files


def short_signature(
    source: bytes, node: Any, *, body_field: str = "body", limit: int = 500
) -> str:
    """Return a stable, body-free signature for a definition node."""

    body = node.child_by_field_name(body_field)
    end = body.start_byte if body is not None else node.end_byte
    raw = source[node.start_byte : end].decode("utf-8", errors="replace")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return " ".join(lines)[:limit]


def comment_nodes(root: Any) -> Iterable[Any]:
    for node in walk_nodes(root):
        if "comment" in node.type:
            yield node


def is_public_modifier(
    source: bytes,
    node: Any,
    *,
    modifier_types: tuple[str, ...] = ("modifier", "modifiers", "visibility_modifier"),
    public_values: tuple[str, ...] = ("public", "pub"),
) -> bool:
    for child in node.children:
        if child.type not in modifier_types:
            continue
        raw = text_of(source, child).strip()
        if raw in public_values or raw.startswith("pub("):
            return True
    return False


def modifier_text(
    source: bytes,
    node: Any,
    modifier_types: tuple[str, ...] = ("modifier", "modifiers", "visibility_modifier"),
) -> str:
    for child in node.children:
        if child.type in modifier_types:
            return text_of(source, child).strip()
    return ""


def _unique_symbol_id(
    nodes: dict[str, Any], path: str, qualified: str, kind: str, node: Any
) -> str:
    value = symbol_node_id(path, qualified, kind)
    if value not in nodes:
        return value
    # Overloads and nested definitions can share a name. Keep IDs stable by
    # anchoring the alternate IDs to the source byte offset.
    return symbol_node_id(path, f"{qualified}@{node.start_byte}", kind)


def _unique_api_id(nodes: dict[str, Any], path: str, qualified: str, node: Any) -> str:
    value = api_node_id(path, qualified)
    if value not in nodes:
        return value
    return api_node_id(path, f"{qualified}@{node.start_byte}")


def emit_symbol(
    builder: Any,
    info: FileInfo,
    rel_path: str,
    name: str,
    qualified: str,
    kind: str,
    node: Any,
    source: bytes,
    parent_id: str,
    *,
    exported: bool,
    language: str,
    visibility: str | None = None,
    signature: str | None = None,
    docstring: str | None = None,
    decorators: Iterable[str] = (),
    extra: dict[str, Any] | None = None,
) -> str:
    """Add a symbol/API pair using the shared graph conventions."""

    symbol_id = _unique_symbol_id(builder.graph.nodes, rel_path, qualified, kind, node)
    attrs = {
        "name": name,
        "qualified_name": qualified,
        "kind": kind,
        "path": rel_path,
        "language": language,
        "signature": signature
        if signature is not None
        else short_signature(source, node),
        "source": text_of(source, node),
        "docstring": docstring,
        "decorators": list(decorators),
        "exported": exported,
        "visibility": visibility or ("public" if exported else "private"),
        **span_of(node),
        **(extra or {}),
    }
    builder.graph.add_node("symbol", symbol_id, attrs)
    builder.graph.add_edge("contains", parent_id, symbol_id)
    info.symbols.setdefault(name, []).append(symbol_id)

    if exported:
        api_id = _unique_api_id(builder.graph.nodes, rel_path, qualified, node)
        builder.graph.add_node(
            "api",
            api_id,
            {
                "name": name,
                "qualified_name": qualified,
                "kind": kind,
                "language": language,
                "signature": attrs["signature"],
                "symbol_id": symbol_id,
                "path": rel_path,
                "visibility": "public"
                if attrs["visibility"] == "public"
                else attrs["visibility"],
            },
        )
        builder.graph.add_edge("exposes_api", info.node_id, api_id)
        builder.graph.add_edge("handles", api_id, symbol_id)
    return symbol_id


def collect_calls(
    node: Any,
    source: bytes,
    rel_path: str,
    source_symbol_id: str,
    info: FileInfo,
    callee_for: Any,
    *,
    skip_types: Iterable[str] = (),
) -> None:
    """Walk a body and append call specs without descending into definitions."""

    skipped = set(skip_types)
    if node.type in skipped:
        return
    callee = callee_for(node)
    if callee:
        info.calls.append(
            CallSpec(
                source_symbol_id=source_symbol_id,
                callee=callee,
                location=location_of(rel_path, node),
            )
        )
    for child in node.named_children:
        collect_calls(
            child,
            source,
            rel_path,
            source_symbol_id,
            info,
            callee_for,
            skip_types=skipped,
        )


def resolve_callee(
    builder: Any,
    info: FileInfo,
    callee: str,
    *,
    separators: tuple[str, ...] = (".", "::", "->"),
    prefixes: tuple[str, ...] = ("self.", "this.", "super.", "cls.", "$this->"),
) -> str | None:
    """Conservative symbol-level call resolution shared by all analyzers."""

    raw = callee.strip().strip("()")
    for prefix in prefixes:
        if raw.startswith(prefix):
            name = raw[len(prefix) :].split("(")[0].strip()
            return builder._find_symbol(
                info.rel_path, name.split(".")[-1], prefer_kind="method"
            )
    normalized = raw
    for separator in separators:
        normalized = normalized.replace(separator, ".")
    normalized = normalized.strip("$")
    parts = [part for part in normalized.split(".") if part]
    if not parts:
        return None
    if len(parts) > 1:
        head, tail = parts[0], parts[-1]
        if head in info.import_aliases:
            target_rel, imported = info.import_aliases[head]
            if target_rel:
                target_name = tail if imported in {None, "default", head} else imported
                found = builder._find_symbol(target_rel, target_name)
                if found:
                    # `use Foo\Bar as Baz; Baz::method()` imports a class,
                    # while static imports import the method itself. If the
                    # imported symbol is a type, resolve the trailing method.
                    if target_name != tail and builder.graph.nodes[found]["attrs"].get(
                        "kind"
                    ) in {
                        "class",
                        "interface",
                        "struct",
                        "union",
                        "trait",
                        "enum",
                        "record",
                    }:
                        method = builder._find_symbol(target_rel, tail)
                        if method:
                            return method
                    return found
        local = builder._find_symbol(info.rel_path, tail)
        if local is not None:
            return local
        return None
    local = builder._find_symbol(info.rel_path, parts[0])
    if local is not None:
        return local
    if parts[0] in info.import_aliases:
        target_rel, imported = info.import_aliases[parts[0]]
        if target_rel:
            return builder._find_symbol(target_rel, imported or parts[0])
    return None
