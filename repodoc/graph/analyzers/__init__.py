"""
Analyzer registry: file extension -> language analyzer.

`analyze_file()` is the single entry point used by the graph builder.
Analyzer classes are imported lazily on first use so importing this package
does not load all tree-sitter syntax packages.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from repodoc.graph.models import CallRelationship, Node

logger = logging.getLogger(__name__)

# File extension -> analyzer name. Note: .h files are routed to the C++
# analyzer (C++ headers are far more common than standalone C headers).
EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".cs": "csharp",
    ".c": "c",
    ".h": "cpp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".php": "php",
    ".phtml": "php",
    ".inc": "php",
    ".go": "go",
    ".rs": "rust",
}

# Directories/files always excluded from analysis
DEFAULT_IGNORE_PATTERNS = {
    ".github", ".git", ".gitignore", ".gitmodules", ".gitattributes",
    ".vscode", ".idea", ".vs",
    "examples",
    "*.pyc", "*.pyo", "*.pyd", "__pycache__", ".pytest_cache", ".coverage",
    ".tox", ".nox", ".mypy_cache", ".ruff_cache", ".hypothesis",
    "poetry.lock", "Pipfile.lock", "package-lock.json", "yarn.lock",
    ".npm", ".yarn", ".pnpm-store", "bun.lock", "bun.lockb",
    "*.class", "*.jar", "*.war", "*.ear", "*.nar",
    ".gradle/", ".settings/", ".classpath", "gradle-app.setting", "*.gradle",
    ".project",
    "*.o", "*.obj", "*.dll", "*.dylib", "*.exe", "*.lib", "*.out", "*.a",
    "*.pdb", "*.suo", "*.user", "*.userosscache", "*.sln.docstates",
    "*.nupkg", "bin/",
    ".svn", ".hg",
    "*.svg", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico", "*.pdf",
    "*.mov", "*.mp4", "*.mp3", "*.wav",
    "venv", ".venv", "env", ".env", "virtualenv",
    "*.swo", "*.swn", "*.sublime-*", "*.log", "*.bak", "*.swp", "*.tmp",
    "*.temp", ".cache", ".sass-cache", ".DS_Store", "Thumbs.db",
    "desktop.ini", "*.egg-info", "*.egg", "*.whl", "*.so",
    ".docusaurus", "*.min.js", "*.min.css", "*.map",
    ".terraform", "*.tfstate*", "digest.txt", "*.ini",
    "tests", "test", "Tests", "Test", "Examples",
}

# File extensions considered code (subset of EXTENSION_MAP keys)
DEFAULT_INCLUDE_PATTERNS = [
    "*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.java", "*.cpp", "*.c",
    "*.h", "*.cs", "*.go", "*.rs", "*.php",
]

_ANALYZER_REGISTRY: Optional[dict[str, type]] = None


def _build_registry() -> dict[str, type]:
    """Import all analyzer classes (lazy, to avoid loading every syntax package)."""
    from repodoc.graph.analyzers.c import TreeSitterCAnalyzer
    from repodoc.graph.analyzers.cpp import TreeSitterCppAnalyzer
    from repodoc.graph.analyzers.csharp import TreeSitterCSharpAnalyzer
    from repodoc.graph.analyzers.go import GoASTAnalyzer
    from repodoc.graph.analyzers.java import TreeSitterJavaAnalyzer
    from repodoc.graph.analyzers.javascript import TreeSitterJSAnalyzer
    from repodoc.graph.analyzers.php import TreeSitterPHPAnalyzer
    from repodoc.graph.analyzers.python import PythonASTAnalyzer
    from repodoc.graph.analyzers.rust import RustASTAnalyzer
    from repodoc.graph.analyzers.typescript import TreeSitterTSAnalyzer

    return {
        "python": PythonASTAnalyzer,
        "javascript": TreeSitterJSAnalyzer,
        "typescript": TreeSitterTSAnalyzer,
        "java": TreeSitterJavaAnalyzer,
        "csharp": TreeSitterCSharpAnalyzer,
        "c": TreeSitterCAnalyzer,
        "cpp": TreeSitterCppAnalyzer,
        "php": TreeSitterPHPAnalyzer,
        "go": GoASTAnalyzer,
        "rust": RustASTAnalyzer,
    }


def get_analyzer(language: str) -> Optional[type]:
    """Return the analyzer class for a language, or None if unsupported."""
    global _ANALYZER_REGISTRY
    if _ANALYZER_REGISTRY is None:
        _ANALYZER_REGISTRY = _build_registry()
    return _ANALYZER_REGISTRY.get(language)


def analyze_file(
    file_path: str, content: str, repo_path: Optional[str] = None
) -> Tuple[List[Node], List[CallRelationship]]:
    """Analyze a single file via the registry. Returns (nodes, relationships)."""
    ext = Path(file_path).suffix.lower()
    language = EXTENSION_MAP.get(ext)
    if language is None:
        return [], []
    analyzer_cls = get_analyzer(language)
    if analyzer_cls is None:
        return [], []
    try:
        analyzer = analyzer_cls(file_path, content, repo_path)
        analyzer.analyze()
        return analyzer.nodes, analyzer.call_relationships
    except Exception as e:
        logger.error("Error analyzing %s (%s): %s", file_path, language, e)
        return [], []
