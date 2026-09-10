"""Git change analysis: per-language file diffs -> component-level changes.

Split out of update.py. Per-language behavior is genericized via DiffSpec
(see its docstring); build_change_map bridges analyzer names to the
knowledge-graph id namespace.
"""

import logging
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from repodoc.graph.analyzers import get_analyzer

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Change analysis
# ---------------------------------------------------------------------------


class ChangeType(Enum):
    API_SIGNATURE_CHANGED = "api_signature_changed"
    NEW_COMPONENT = "new_component"
    REMOVED_COMPONENT = "removed_component"
    DOCSTRING_CHANGED = "docstring_changed"
    CODE_BODY_CHANGED = "code_body_changed"


@dataclass
class ComponentChange:
    name: str
    change_type: ChangeType
    file_path: str
    line_number: int = 0
    old_signature: Optional[str] = None
    new_signature: Optional[str] = None
    details: str = ""


@dataclass
class FileChangeAnalysis:
    file_path: str
    language: str
    changes: list[ComponentChange] = field(default_factory=list)
    has_important_changes: bool = False


IMPORTANT_TYPES = frozenset(
    {
        ChangeType.API_SIGNATURE_CHANGED,
        ChangeType.NEW_COMPONENT,
        ChangeType.REMOVED_COMPONENT,
        ChangeType.CODE_BODY_CHANGED,
    }
)


@dataclass(frozen=True)
class DiffSpec:
    """Per-language diff behavior.

    - supports_docstring: whether docstrings can be compared (C analyzers
      never extract them, so skip the branch).
    - compare_source: compare whole source_code instead of signature/docstring
      (Go/Rust historically; their analyzers do not reliably produce either).
    - important_types: change types that make a file "important" (all four
      for every language — fixes the old C/Rust bug that never treated body
      changes as important).
    """

    language: str
    supports_docstring: bool = True
    compare_source: bool = False
    important_types: frozenset[ChangeType] = IMPORTANT_TYPES


LANGUAGE_DIFF_TABLE: dict[str, DiffSpec] = {
    ".py": DiffSpec("python"),
    ".js": DiffSpec("javascript"),
    ".jsx": DiffSpec("javascript"),
    ".mjs": DiffSpec("javascript"),
    ".cjs": DiffSpec("javascript"),
    ".ts": DiffSpec("typescript"),
    ".tsx": DiffSpec("typescript"),
    ".java": DiffSpec("java"),
    ".cs": DiffSpec("csharp"),
    ".c": DiffSpec("c", supports_docstring=False),
    ".h": DiffSpec("cpp", supports_docstring=False),
    ".cpp": DiffSpec("cpp", supports_docstring=False),
    ".cc": DiffSpec("cpp", supports_docstring=False),
    ".cxx": DiffSpec("cpp", supports_docstring=False),
    ".hpp": DiffSpec("cpp", supports_docstring=False),
    ".php": DiffSpec("php"),
    ".go": DiffSpec("go", compare_source=True),
    ".rs": DiffSpec("rust", compare_source=True),
}


def get_diff_spec(file_path: str) -> DiffSpec | None:
    ext = Path(file_path).suffix.lower()
    return LANGUAGE_DIFF_TABLE.get(ext)


def _is_test_file(file_path: str) -> bool:
    """Check if a file is a test file (exact name patterns, no substring matches)."""
    path = Path(file_path)
    name = path.name.lower()
    # Directories only (exclude the filename itself from dir checks)
    dirs = [p.lower() for p in path.parts[:-1]]
    return (
        name.startswith("test_")
        or name.endswith("_test")
        or name.endswith(".test")
        or name in ("test", "tests")
        or "tests" in dirs
        or "test" in dirs
    )


def analyze_diff(
    file_path: str, old_content: str, new_content: str, spec: DiffSpec
) -> FileChangeAnalysis:
    """Compare two versions of a file via its language analyzer (generic)."""
    analysis = FileChangeAnalysis(file_path=file_path, language=spec.language)

    if old_content == new_content:
        return analysis

    analyzer_cls = get_analyzer(spec.language)
    if analyzer_cls is None:
        analysis.has_important_changes = old_content != new_content
        if old_content != new_content:
            analysis.changes.append(
                ComponentChange(
                    name="<file_changed>",
                    change_type=ChangeType.API_SIGNATURE_CHANGED,
                    file_path=file_path,
                    details=f"{spec.language} analyzer not available",
                )
            )
        return analysis

    old_analyzer = analyzer_cls(file_path, old_content)
    new_analyzer = analyzer_cls(file_path, new_content)

    try:
        old_analyzer.analyze()
    except Exception:
        pass
    try:
        new_analyzer.analyze()
    except Exception:
        pass

    old_components = {n.name: n for n in old_analyzer.nodes if n.name}
    new_components = {n.name: n for n in new_analyzer.nodes if n.name}

    old_names = set(old_components.keys())
    new_names = set(new_components.keys())

    for name in sorted(new_names - old_names):
        node = new_components[name]
        analysis.changes.append(
            ComponentChange(
                name=name,
                change_type=ChangeType.NEW_COMPONENT,
                file_path=file_path,
                details=f"New {node.component_type} added",
            )
        )

    for name in sorted(old_names - new_names):
        node = old_components[name]
        analysis.changes.append(
            ComponentChange(
                name=name,
                change_type=ChangeType.REMOVED_COMPONENT,
                file_path=file_path,
                details=f"{node.component_type} removed",
            )
        )

    for name in sorted(old_names & new_names):
        old_node = old_components[name]
        new_node = new_components[name]

        if spec.compare_source:
            if old_node.source_code != new_node.source_code:
                analysis.changes.append(
                    ComponentChange(
                        name=name,
                        change_type=ChangeType.CODE_BODY_CHANGED,
                        file_path=file_path,
                        details="Source code changed",
                    )
                )
            continue

        old_params = old_node.parameters or []
        new_params = new_node.parameters or []

        if old_params != new_params:
            analysis.changes.append(
                ComponentChange(
                    name=name,
                    change_type=ChangeType.API_SIGNATURE_CHANGED,
                    file_path=file_path,
                    old_signature=str(old_params),
                    new_signature=str(new_params),
                    details="Parameters changed",
                )
            )
        elif (
            spec.supports_docstring
            and old_node.docstring != new_node.docstring
        ):
            if old_node.docstring and new_node.docstring:
                analysis.changes.append(
                    ComponentChange(
                        name=name,
                        change_type=ChangeType.DOCSTRING_CHANGED,
                        file_path=file_path,
                        details="Only docstring changed",
                    )
                )
            else:
                analysis.changes.append(
                    ComponentChange(
                        name=name,
                        change_type=ChangeType.CODE_BODY_CHANGED,
                        file_path=file_path,
                        details="Code body or docstring changed",
                    )
                )
        else:
            analysis.changes.append(
                ComponentChange(
                    name=name,
                    change_type=ChangeType.CODE_BODY_CHANGED,
                    file_path=file_path,
                    details="Only internal code changed",
                )
            )

    analysis.has_important_changes = any(
        c.change_type in spec.important_types for c in analysis.changes
    )
    return analysis


def analyze_git_diff(
    repo_path: str,
    base_commit: Optional[str] = None,
    include_tests: bool = False,
) -> dict[str, FileChangeAnalysis]:
    """Analyze git diff to find important code changes.

    Args:
        repo_path: Path to the repository
        base_commit: Base commit to compare against
        include_tests: Whether to include test files in analysis (default: False)
    """
    cmd = (
        ["git", "diff", "--name-only", base_commit]
        if base_commit
        else ["git", "diff", "--name-only", "HEAD"]
    )
    result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("git diff failed: %s", result.stderr.strip())
        return {}
    changed_files = [f for f in result.stdout.strip().split("\n") if f]

    if not include_tests:
        changed_files = [f for f in changed_files if not _is_test_file(f)]

    analyses: dict[str, FileChangeAnalysis] = {}

    for file_path in changed_files:
        full_path = Path(repo_path) / file_path

        spec = get_diff_spec(file_path)
        if spec is None:
            # Non-code file (md, json, ...) — does not affect API docs
            continue

        # Old content at the base commit (needed for both modified and
        # deleted files).
        show_cmd = ["git", "show", f"{base_commit or 'HEAD'}:{file_path}"]
        show_result = subprocess.run(
            show_cmd, cwd=repo_path, capture_output=True, text=True
        )
        old_content = show_result.stdout if show_result.returncode == 0 else ""

        if not full_path.exists():
            # Deleted file: every component it defined at the base commit is
            # REMOVED. Compare the old content against empty content — the
            # analyzer reports all old components as removed.
            if old_content:
                analyses[file_path] = analyze_diff(file_path, old_content, "", spec)
            continue

        try:
            new_content = full_path.read_text()
        except Exception:
            continue

        analyses[file_path] = analyze_diff(file_path, old_content, new_content, spec)

    return analyses


def filter_important_changes(
    analyses: dict[str, FileChangeAnalysis],
) -> dict[str, FileChangeAnalysis]:
    """Filter to only files with important API-affecting changes."""
    return {
        file_path: analysis
        for file_path, analysis in analyses.items()
        if analysis.has_important_changes
    }


def build_change_map(
    analyses: dict[str, FileChangeAnalysis],
) -> dict[str, ComponentChange]:
    """Map knowledge-graph component ids -> ComponentChange for important
    changes (module.path.name style ids, mirroring analyzer ids)."""
    change_map: dict[str, ComponentChange] = {}
    for file_path, analysis in analyses.items():
        for change in analysis.changes:
            if change.change_type not in (
                ChangeType.NEW_COMPONENT,
                ChangeType.API_SIGNATURE_CHANGED,
                ChangeType.REMOVED_COMPONENT,
            ):
                continue
            module_path = change.file_path
            for ext in (
                ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".cs",
                ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".php",
            ):
                if module_path.endswith(ext):
                    module_path = module_path[: -len(ext)]
                    break
            component_id = f"{module_path.replace('/', '.')}.{change.name}"
            change_map[component_id] = change
    return change_map


