"""Language detection and analyzer registry."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from repodoc.graph.languages.base import LanguageAnalyzer
from repodoc.graph.languages.c_family import CAnalyzer, CppAnalyzer
from repodoc.graph.languages.csharp import CSharpAnalyzer
from repodoc.graph.languages.ecmascript import (
    JavaScriptAnalyzer,
    TsxAnalyzer,
    TypeScriptAnalyzer,
)
from repodoc.graph.languages.go import GoAnalyzer
from repodoc.graph.languages.java import JavaAnalyzer
from repodoc.graph.languages.php import PhpAnalyzer
from repodoc.graph.languages.python import PythonAnalyzer
from repodoc.graph.languages.rust import RustAnalyzer

_SKIP = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    "coverage",
    "target",
    "vendor",
    "bin",
    "obj",
    ".gradle",
    ".mvn",
    "__pycache__",
    ".repodoc",
}

_SUFFIX_TO_NAME = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".php": "php",
    ".phtml": "php",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".c++": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".cs": "csharp",
}


def analyzers() -> list[LanguageAnalyzer]:
    return [
        PythonAnalyzer(),
        JavaScriptAnalyzer(),
        TypeScriptAnalyzer(),
        TsxAnalyzer(),
        RustAnalyzer(),
        GoAnalyzer(),
        JavaAnalyzer(),
        PhpAnalyzer(),
        CAnalyzer(),
        CppAnalyzer(),
        CSharpAnalyzer(),
    ]


def analyzer_by_name(name: str) -> LanguageAnalyzer:
    lowered = name.lower()
    aliases = {
        "js": "javascript",
        "jsx": "javascript",
        "ts": "typescript",
        "tsx": "tsx",
        "py": "python",
        "rs": "rust",
        "golang": "go",
        "c++": "cpp",
        "cplusplus": "cpp",
        "cs": "csharp",
        "c#": "csharp",
        "c_sharp": "csharp",
        "dotnet": "csharp",
    }
    wanted = aliases.get(lowered, lowered)
    for analyzer in analyzers():
        if analyzer.name == wanted:
            return analyzer
    raise ValueError(f"unknown language analyzer: {name}")


def detect_analyzer(repo_path: str | Path) -> LanguageAnalyzer:
    """Pick the dominant supported language by source-file count.

    Mixed-language repositories are intentionally deterministic: the language
    with the most source files wins. Per-file multi-parser support can be added
    later without changing the graph contract.
    """
    root = Path(repo_path)
    counts: Counter[str] = Counter()
    header_count = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part in _SKIP for part in parts):
            continue
        suffix = path.suffix.lower()
        if suffix == ".h":
            header_count += 1
            continue
        language = _SUFFIX_TO_NAME.get(suffix)
        if language:
            counts[language] += 1
    if counts.get("cpp", 0):
        counts["cpp"] += header_count
    elif counts.get("c", 0):
        counts["c"] += header_count
    elif header_count:
        counts["c"] = header_count
    if not counts:
        return PythonAnalyzer()
    if counts.get("tsx", 0):
        # The TSX grammar also handles plain TypeScript and JSX in a mixed
        # repository, whereas the TS grammar cannot parse .tsx files.
        return TsxAnalyzer()
    dominant = counts.most_common(1)[0][0]
    return analyzer_by_name(dominant)
