"""Import-aware reference resolution.

Analyzers expose an ``imports`` map (local alias -> canonical dotted target)
collected from the file's import/use/include statements. Targets are either:

- module targets: dotted module paths, either repo-relative ("pkg.models")
  or language-qualified ("com.example.models.User", "github.com/x/y/models");
- symbol targets: dotted references ending in the symbol name.

When a bare or short reference is ambiguous (several components share the
name), the caller file's import map disambiguates it: the alias is replaced
by its target and matched against candidate components by exact id, id
suffix, or module membership. Only unambiguous matches are returned — a
wrong edge (and a wrong impact propagation) is worse than a missing one.
"""

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence

#: Reserved import-map key holding wildcard namespace targets (Java `import
#: a.b.*`, C# `using a.b;`, Rust `use a::b::*;`): a U+001F-separated list of
#: dotted namespaces to try for otherwise-unresolvable bare references.
WILDCARD_NAMESPACES = "*"
_NS_SEP = "\x1f"

#: Module stems that represent their package/directory (JS `./components`
#: resolves index.ts, Python `pkg` resolves `pkg/__init__.py`).
_INDEX_STEMS = (".index", ".__init__", ".mod", ".main")


def register_namespace(imports: Dict[str, str], namespace: str) -> None:
    """Append a wildcard namespace target to an import map."""
    namespace = (namespace or "").strip(".")
    if not namespace:
        return
    existing = imports.get(WILDCARD_NAMESPACES, "")
    targets = existing.split(_NS_SEP) if existing else []
    if namespace not in targets:
        targets.append(namespace)
    imports[WILDCARD_NAMESPACES] = _NS_SEP.join(targets)


@dataclass(frozen=True)
class Candidate:
    """A component eligible as an import-resolution target."""

    id: str
    module: str  # file-derived dotted module path (relative path minus ext)
    name: str
    file: str  # repo-relative file path


def module_path_from_relative(relative_path: str) -> str:
    """Dotted module path for a repo-relative file path (minus extension)."""
    rel = (relative_path or "").replace("\\", "/")
    base = os.path.basename(rel)
    if "." in base:
        rel = rel[: rel.rindex(".")]
    return rel.replace("/", ".")


def candidate_from_node(node) -> Candidate:
    """Build a Candidate from an analysis Node."""
    rel = (node.relative_path or node.file_path or "").replace("\\", "/")
    return Candidate(
        id=node.id,
        module=module_path_from_relative(rel),
        name=node.name or "",
        file=rel,
    )


def _norm(reference: str) -> str:
    """Normalize a reference: Rust uses `::` where other languages use `.`."""
    return (reference or "").replace("::", ".")


def _canonical_modules(module: str) -> Iterator[str]:
    """Module spellings for matching: itself plus its package form when the
    stem is a package entry point (index / __init__ / mod / main)."""
    module = _norm(module)
    yield module
    for stem in _INDEX_STEMS:
        if module.endswith(stem):
            yield module[: -len(stem)]


def _module_matches(component_module: str, target: str) -> bool:
    """True when a component's module path corresponds to an import target.

    Handles both directions: repo-relative modules ("models") matching a
    longer language path ("github.com.me.proj.models"), and imported modules
    nested under a longer component module path."""
    if not component_module or not target:
        return False
    for cm in _canonical_modules(component_module):
        for tm in _canonical_modules(target):
            if (
                cm == tm
                or cm.endswith("." + tm)
                or tm.endswith("." + cm)
            ):
                return True
    return False


class ImportIndex:
    """Lookup index over candidates so resolution never scans the whole repo.

    Built once per analysis/update run; keys are normalized dotted names."""

    def __init__(self, candidates: Sequence[Candidate]) -> None:
        self._by_id: Dict[str, list[Candidate]] = {}
        self._by_module: Dict[str, list[Candidate]] = {}
        self._by_name: Dict[str, list[Candidate]] = {}
        self._by_name_last: Dict[str, list[Candidate]] = {}
        for c in candidates:
            self._by_id.setdefault(_norm(c.id), []).append(c)
            for module in _canonical_modules(c.module):
                self._by_module.setdefault(module, []).append(c)
            if c.name:
                self._by_name.setdefault(_norm(c.name), []).append(c)
                self._by_name_last.setdefault(
                    _norm(c.name).rsplit(".", 1)[-1], []
                ).append(c)

    @staticmethod
    def _unique(cands: Iterable[Candidate]) -> Optional[str]:
        ids = {c.id for c in cands}
        return next(iter(ids)) if len(ids) == 1 else None

    def match(self, fqn: str) -> Optional[str]:
        """Match a fully-qualified reference (unique only)."""
        fqn = _norm(fqn)
        if not fqn:
            return None

        exact = self._unique(self._by_id.get(fqn, ()))
        if exact:
            return exact

        # Symbol-based match: the target carries the symbol name at the end
        # ("github.com/x/y.Client" -> "Client" in module "y"; "com.shop.User"
        # -> class User). Methods may carry a qualified name ("User.save"),
        # so the name's last segment is looked up too. This runs BEFORE
        # module membership: a Java/C# class file's module path ends with the
        # class name, so module-only matching would also hit its methods.
        head, _, symbol = fqn.rpartition(".")
        if symbol:
            pool = list(self._by_name.get(symbol, ()))
            pool += self._by_name_last.get(symbol, ())

            def _symbol_matches(c: Candidate) -> bool:
                name = _norm(c.name)
                base = (
                    _module_matches(c.module, head)
                    or _module_matches(c.module, f"{head}.{symbol}")
                )
                if not name.endswith("." + symbol):
                    return name == symbol and base
                # Qualified method name ("User.save"): the class part must
                # match the namespace/module, so wildcard usings resolve
                # simple method calls to the right class.
                qualifier = name[: -len(symbol) - 1]
                return base or _module_matches(
                    c.module, f"{head}.{qualifier}"
                )

            matched = self._unique(c for c in pool if _symbol_matches(c))
            if matched:
                return matched

        # The target may name a file/module (e.g. a Python module import or a
        # C include): match components whose own module path is the target.
        for module in _canonical_modules(fqn):
            matched = self._unique(self._by_module.get(module, ()))
            if matched:
                return matched
        return None


def _as_index(candidates) -> ImportIndex:
    """Accept an ImportIndex or a raw candidate sequence."""
    if isinstance(candidates, ImportIndex):
        return candidates
    return ImportIndex(candidates)


def resolve_with_imports(
    ref: str,
    caller_file: str,
    imports_by_file: Mapping[str, Mapping[str, str]],
    candidates,
) -> Optional[str]:
    """Resolve ``ref`` using the caller file's import map. None when no
    (unambiguous) match exists, so callers can fall back to name matching.

    ``candidates`` is an ImportIndex (preferred) or a raw candidate list."""
    import_map = imports_by_file.get(caller_file) if imports_by_file else None
    if not import_map or not ref:
        return None
    index = _as_index(candidates)
    parts = [p for p in _norm(ref).split(".") if p]
    # Longest local-alias prefix wins ("M.User" over "M"); once an alias
    # matches, shorter aliases must not be considered.
    for i in range(len(parts), 0, -1):
        alias = ".".join(parts[:i])
        target = import_map.get(alias)
        if target is None:
            continue
        tail = parts[i:]
        fqn = target + ("." + ".".join(tail) if tail else "")
        return index.match(fqn)
    # Wildcard namespace imports (using/import *): try each namespace as the
    # reference's prefix (Java/C# simple-name access).
    namespaces = import_map.get(WILDCARD_NAMESPACES, "")
    if namespaces:
        matches = set()
        for namespace in namespaces.split(_NS_SEP):
            if not namespace:
                continue
            matched = index.match(f"{namespace}.{ref}")
            if matched:
                matches.add(matched)
        if len(matches) == 1:
            return next(iter(matches))
    return None


def resolve_include_target(relative_path: str, include: str) -> str:
    """Dotted module path for a C/C++ `#include "..."` from the caller's dir.

    Returns "" for absolute/escaping includes; the caller may additionally
    register the bare header stem as a fallback namespace (include search
    paths are not known)."""
    include = (include or "").strip().strip('"')
    if not include or include.startswith(("/", "\\")):
        return ""
    rel = (relative_path or "").replace("\\", "/")
    base_dir = os.path.dirname(rel)
    joined = os.path.normpath(os.path.join(base_dir, include)).replace(
        os.sep, "/"
    )
    if joined.startswith(".."):
        return ""
    return module_path_from_relative(joined)


def merge_import_maps(maps: Iterable[Mapping[str, str]]) -> Dict[str, str]:
    """Merge alias maps (later entries win); small helper for analyzers."""
    merged: Dict[str, str] = {}
    for m in maps:
        merged.update(m)
    return merged
