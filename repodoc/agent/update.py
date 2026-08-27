"""Incremental documentation update: git diff analysis -> impact -> regenerate.

Merges the former diff_analysis / incremental_updater / code_extractor into
one module. Per-language diff logic is genericized via DiffSpec; impact
propagation direction is aligned with the graph edge convention
(caller -> callee, so downstream = callers = predecessors).
"""

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from repodoc.agent.agent import DocAgent
from repodoc.agent.generate import DocPipeline
from repodoc.config import Config
from repodoc.graph.analyzers import get_analyzer
from repodoc.graph.graph import HeterogeneousGraph
from repodoc.graph.models import REL_CALLS, Node
from repodoc.utils import file_manager, log_operation

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
        if not full_path.exists():
            # Deleted files: the component removal cannot be detected without
            # the old content — skip (documented limitation)
            continue

        try:
            new_content = full_path.read_text()
        except Exception:
            continue

        show_cmd = ["git", "show", f"{base_commit or 'HEAD'}:{file_path}"]
        result = subprocess.run(
            show_cmd, cwd=repo_path, capture_output=True, text=True
        )
        old_content = result.stdout if result.returncode == 0 else ""

        spec = get_diff_spec(file_path)
        if spec is None:
            # Non-code file (md, json, ...) — does not affect API docs
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


# ---------------------------------------------------------------------------
# Impact computation
# ---------------------------------------------------------------------------


class IncrementalUpdater:
    """Finds affected components for changed files via the knowledge graph.

    Graph edge convention: caller -> callee (REL_CALLS). A change to X
    affects the components that CALL X, i.e. X's predecessors.
    """

    def __init__(self, graph: HeterogeneousGraph, config: Any):
        self.graph = graph
        self.config = config
        self._impact_cache: dict[str, set[str]] = {}

    def _find_components_in_file(self, file_path: str) -> set[str]:
        """Find all components defined in a file (by graph node file_path)."""
        components: set[str] = set()
        path = Path(file_path).as_posix()

        for node_id, node_attrs in self.graph.nodes(data=True):
            node_data = node_attrs.get("data", {})
            source_file = node_data.get("file_path", "")
            if source_file and self._paths_match(source_file, path):
                components.add(node_id)

        if not components:
            components = self._find_components_by_relative_path(file_path)
        return components

    def _find_components_by_relative_path(self, file_path: str) -> set[str]:
        """Match by relative path structure (filename + at least one dir)."""
        components: set[str] = set()
        try:
            current_path = Path(file_path)
            current_name = current_path.name
            current_parts = current_path.parts
            current_len = len(current_parts)

            for node_id, node_attrs in self.graph.nodes(data=True):
                node_data = node_attrs.get("data", {})
                source_file = node_data.get("file_path", "")
                if not source_file:
                    continue

                source_path = Path(source_file)
                source_parts = source_path.parts
                source_len = len(source_parts)

                if source_path.name != current_name:
                    continue

                # Need at least 2 matching parts (filename + 1 directory)
                min_match = 2
                match_count = 0
                src_idx, tgt_idx = source_len - 1, current_len - 1
                while (
                    src_idx >= 0 and tgt_idx >= 0 and match_count < min_match
                ):
                    if source_parts[src_idx] == current_parts[tgt_idx]:
                        match_count += 1
                    else:
                        break
                    src_idx -= 1
                    tgt_idx -= 1

                if match_count >= min_match:
                    components.add(node_id)
        except Exception as e:
            logger.debug(f"Error in relative path matching: {e}")
        return components

    def _paths_match(self, source: str, target: str) -> bool:
        try:
            return Path(source).resolve() == Path(target).resolve()
        except Exception:
            return source == target

    def _get_downstream_components(self, components: set[str]) -> set[str]:
        """Components affected by a change to the given ones = their CALLERS
        (predecessors in the caller -> callee graph)."""
        downstream: set[str] = set()
        to_process = list(components)
        visited: set[str] = set()

        while to_process:
            component = to_process.pop()
            if component in visited:
                continue
            visited.add(component)

            for pred in self.graph.predecessors(component):
                edge_data = self.graph.get_edge_data(pred, component)
                if edge_data and edge_data.get("relation_type") == REL_CALLS:
                    downstream.add(pred)
                    to_process.append(pred)
        return downstream

    def get_upstream_dependencies(self, component: str) -> set[str]:
        """Components this component depends on = its callees (successors)."""
        if component in self._impact_cache:
            return self._impact_cache[component]

        dependencies: set[str] = set()
        to_process = [component]
        visited: set[str] = set()

        while to_process:
            comp = to_process.pop()
            if comp in visited:
                continue
            visited.add(comp)

            for succ in self.graph.successors(comp):
                edge_data = self.graph.get_edge_data(comp, succ)
                if edge_data and edge_data.get("relation_type") == REL_CALLS:
                    dependencies.add(succ)
                    to_process.append(succ)

        self._impact_cache[component] = dependencies
        return dependencies

    def compute_impact_graph(
        self,
        changed_files: list[str],
        module_tree: Optional[dict[str, Any]] = None,
        change_map: Optional[dict[str, ComponentChange]] = None,
    ) -> dict[str, Any]:
        """Compute a complete impact analysis for changed files.

        Returns a structured result with:
        - changed_files / changed_components / affected_components
        - new_components: components detected as NEW via diff analysis that
          are not in the knowledge graph
        - not_in_module_tree / impact_order / total_affected
        """
        changed_components: set[str] = set()
        for f in changed_files:
            changed_components.update(self._find_components_in_file(f))

        all_affected = self.get_affected_components(changed_files)

        module_tree_components: set[str] = set()
        if module_tree:
            for mod_info in module_tree.values():
                module_tree_components.update(mod_info.get("components", []))

        graph_components: set[str] = set()
        for node_id, _attrs in self.graph.nodes(data=True):
            if isinstance(node_id, str):
                graph_components.add(node_id)

        if module_tree:
            filtered_affected = all_affected & module_tree_components
            not_in_module_tree = all_affected - module_tree_components
        else:
            filtered_affected = all_affected
            not_in_module_tree = set()

        # New components: derived from diff analysis (NEW_COMPONENT changes
        # whose ids are absent from the knowledge graph). The old
        # "changed - graph" computation was always empty.
        new_components: set[str] = set()
        if change_map:
            new_components = {
                cid
                for cid, change in change_map.items()
                if change.change_type == ChangeType.NEW_COMPONENT
                and cid not in graph_components
            }

        impact_order = self._compute_topological_order(filtered_affected)

        return {
            "changed_files": changed_files,
            "changed_components": list(changed_components),
            "affected_components": list(filtered_affected),
            "new_components": list(new_components),
            "not_in_module_tree": list(not_in_module_tree),
            "impact_order": impact_order,
            "total_affected": len(filtered_affected),
        }

    def get_affected_components(self, changed_files: list[str]) -> set[str]:
        """All components affected by the given file changes (changed + downstream)."""
        affected: set[str] = set()
        for file_path in changed_files:
            components = self._find_components_in_file(file_path)
            affected.update(components)
            affected.update(self._get_downstream_components(components))
        return affected

    def _compute_topological_order(self, components: set[str]) -> list[str]:
        """Processing order: dependencies first (deps = callees/successors)."""
        order: list[str] = []
        remaining = set(components)
        visited: set[str] = set()

        while remaining:
            ready = []
            for comp in remaining:
                deps = self.get_upstream_dependencies(comp)
                if deps.issubset(visited):
                    ready.append(comp)

            if not ready:
                # Cycle or dependencies outside the affected set — pick one
                ready = [remaining.pop()]
            else:
                for r in ready:
                    remaining.discard(r)

            order.extend(ready)
            visited.update(ready)

        return order


# ---------------------------------------------------------------------------
# Git code extraction (Python-only by design)
# ---------------------------------------------------------------------------


class GitCodeExtractor:
    """Extract code components from specific git commits.

    NOTE: extraction uses the stdlib `ast` module, so it only supports Python
    files. For other languages, regeneration falls back to the working tree.
    """

    def __init__(self, repo_path: str) -> None:
        self.repo_path = os.path.abspath(repo_path)

    def get_file_content_at_commit(self, file_path: str, commit: str) -> Optional[str]:
        """Get file content at a specific commit (None if not present)."""
        try:
            if os.path.isabs(file_path):
                file_path = os.path.relpath(file_path, self.repo_path)
            result = subprocess.run(
                ["git", "show", f"{commit}:{file_path}"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout if result.returncode == 0 else None
        except Exception as e:
            logger.warning(f"Failed to get file content at {commit}: {e}")
            return None

    def extract_component_at_commit(
        self, file_path: str, component_name: str, commit: str
    ) -> Optional[dict[str, Any]]:
        """Extract a Python component's code at a given commit."""
        content = self.get_file_content_at_commit(file_path, commit)
        if not content:
            return None

        try:
            import ast

            tree = ast.parse(content)
        except SyntaxError:
            logger.warning(f"Failed to parse {file_path} at {commit}")
            return None

        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name != component_name:
                    continue
                source_code = ast.get_source_segment(content, node)
                if source_code is None:
                    lines = content.split("\n")
                    source_code = "\n".join(
                        lines[node.lineno - 1 : node.end_lineno]
                    )
                params = []
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if getattr(node, "args", None):
                        params = [arg.arg for arg in node.args.args]
                        params.extend(arg.arg for arg in node.args.kwonlyargs)
                return {
                    "source_code": source_code or "",
                    "docstring": ast.get_docstring(node) or "",
                    "parameters": params,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "component_type": (
                        "class" if isinstance(node, ast.ClassDef) else "function"
                    ),
                }
        return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _get_git_commit(repo_path: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _load_graph_and_assets(
    config: Config, project_name: str
) -> tuple[HeterogeneousGraph, dict]:
    """Load the knowledge graph and module tree from previous output.

    The knowledge graph is the single required input: the module tree is
    restored from its module nodes. module_tree.json is only a fallback for
    graphs produced before module nodes existed."""
    graph_path = os.path.join(
        config.output_dir, "data", f"{project_name}_knowledge_graph.json"
    )
    if not os.path.exists(graph_path):
        raise FileNotFoundError(
            f"Knowledge graph not found: {graph_path}\n"
            "Run 'repodoc generate' first to generate initial documentation"
        )
    graph = HeterogeneousGraph.load(graph_path)

    from repodoc.graph.builder import graph_to_module_tree

    module_tree = graph_to_module_tree(graph)
    if not module_tree:
        module_tree_path = os.path.join(
            config.output_dir, "data", f"{project_name}_module_tree.json"
        )
        if os.path.exists(module_tree_path):
            module_tree = file_manager.load_json(module_tree_path) or {}
    return graph, module_tree


def _build_pipeline_from_graph(graph: HeterogeneousGraph, config: Config) -> DocPipeline:
    """Reconstruct the DocPipeline component map from graph code nodes."""
    from repodoc.graph.builder import graph_to_components

    pipeline = DocPipeline(config)
    pipeline.components = graph_to_components(graph)
    pipeline.load_module_tree()
    pipeline.initialize_agent()
    return pipeline


def _find_module_name(module_tree: dict, node_id: str) -> Optional[str]:
    """Find the module a component belongs to (recursive)."""
    for mod_name, mod_info in module_tree.items():
        if node_id in mod_info.get("components", []):
            return mod_name
        children = mod_info.get("children", {})
        if children:
            found = _find_module_name(children, node_id)
            if found:
                return found
    return None


async def run_update(
    config: Config,
    changed_files: Optional[tuple[str, ...]] = None,
    base_commit: Optional[str] = None,
    analyze_changes: bool = True,
    include_tests: bool = False,
    confirm: Optional[Callable[[str], bool]] = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Update documentation incrementally for changed files.

    Args:
        config: RepoDoc config (repo_path, output_dir, ...).
        changed_files: Explicit list of changed files (auto-detected via git
            if not provided).
        base_commit: Base commit to compare against (auto-detected from
            metadata if not provided).
        analyze_changes: Analyze changes to filter unimportant ones.
        include_tests: Include test files in analysis.
        confirm: Optional callable(message) -> bool; called before regenerating.
        verbose: Enable debug logging.

    Returns:
        Dict with status, impact analysis, and result counts.
    """
    start_time = time.time()

    project_name = "".join(
        c if c.isalnum() else "_"
        for c in os.path.basename(os.path.normpath(config.repo_path))
    )
    data_dir = os.path.join(config.output_dir, "data")
    metadata_path = os.path.join(data_dir, f"{project_name}_metadata.json")

    # Auto-detect base_commit from metadata if not provided
    stored_commit = None
    if os.path.exists(metadata_path):
        try:
            metadata = file_manager.load_json(metadata_path)
            stored_commit = (metadata or {}).get("generation_info", {}).get(
                "git_commit"
            )
            if stored_commit:
                print(f"Found previous commit from metadata: {stored_commit[:8]}")
        except Exception:
            pass

    if not base_commit and stored_commit:
        base_commit = stored_commit
        print(f"Using base commit: {base_commit[:8]}")

    current_commit = _get_git_commit(config.repo_path)
    if current_commit:
        print(f"Current commit: {current_commit[:8]}")

    # --- Change analysis ---
    analyses: Optional[dict[str, FileChangeAnalysis]] = None
    if analyze_changes and not changed_files:
        print("\nAnalyzing git changes...")
        analyses = analyze_git_diff(config.repo_path, base_commit, include_tests)
        if not analyses:
            print("No changes detected")
            return {"status": "no_changes", "impact": None}

        important = filter_important_changes(analyses)
        if not important:
            print(
                "\nNo important API changes detected. Documentation update not needed."
            )
            return {"status": "no_important_changes", "impact": None}

        changed_files = tuple(important.keys())
        print(f"\nFiles with important changes: {len(changed_files)}")

    if not changed_files:
        print("Detecting changed files via git...")
        try:
            diff_base = base_commit or "HEAD"
            result = subprocess.run(
                ["git", "diff", "--name-only", diff_base],
                cwd=config.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            changed_files = tuple(f for f in result.stdout.strip().split("\n") if f)
        except Exception as e:
            print(f"Error detecting changes: {e}")
            return {"status": "error", "impact": None}

    if not changed_files:
        print("No changed files to process")
        return {"status": "no_changes", "impact": None}

    # --- Load knowledge graph + module tree ---
    try:
        graph, module_tree = _load_graph_and_assets(config, project_name)
    except FileNotFoundError as e:
        print(e)
        return {"status": "error", "impact": None}
    print(f"Loaded knowledge graph: {graph.node_count()} nodes, {graph.edge_count()} edges")

    # Recompute analyses for change mapping if not already available
    if analyses is None:
        try:
            analyses = analyze_git_diff(config.repo_path, base_commit, include_tests)
        except Exception:
            analyses = {}

    change_map = build_change_map(analyses)

    # --- Impact ---
    updater = IncrementalUpdater(graph, config)
    normalized_files = [
        f if f.startswith("/") else os.path.abspath(os.path.join(config.repo_path, f))
        for f in changed_files
    ]
    print(f"\nAnalyzing impact of {len(normalized_files)} changed files...")
    impact = updater.compute_impact_graph(normalized_files, module_tree, change_map)

    print(f"  Changed files: {impact['changed_files'] and len(normalized_files)}")
    print(f"  Changed components: {len(impact['changed_components'])}")
    print(f"  Total affected (in module_tree): {impact['total_affected']}")

    if impact.get("new_components"):
        print("\n  New components (not in knowledge graph):")
        for comp in impact["new_components"][:10]:
            print(f"    + {comp}")
        print("  ⚠️  Run 'repodoc generate' to include them in the full build.")

    if impact["total_affected"] == 0:
        print("\nNo documentation updates needed")
        return {"status": "no_updates_needed", "impact": impact}

    if confirm and not confirm(
        f"\nThis will update documentation for {impact['total_affected']} component(s). Continue?"
    ):
        print("Aborted")
        return {"status": "aborted", "impact": impact}

    # --- Regenerate affected components ---
    print("\nRegenerating documentation for affected components...")
    pipeline = _build_pipeline_from_graph(graph, config)
    module_tree = pipeline.module_tree
    docs_dir = config.docs_dir
    code_extractor = GitCodeExtractor(config.repo_path)

    async def regenerate_component(node_id: str):
        node = pipeline.components.get(node_id)
        if not node:
            return None
        module_name = _find_module_name(module_tree, node_id)
        if not module_name:
            return None

        module_dir = os.path.join(docs_dir, module_name)
        file_manager.ensure_directory(module_dir)
        output_path = os.path.join(module_dir, f"{node.name}.md")

        source_code = node.source_code or ""
        docstring = node.docstring or ""
        parameters = node.parameters or []

        change_info = change_map.get(node_id)
        if change_info and current_commit:
            if change_info.change_type in (
                ChangeType.NEW_COMPONENT,
                ChangeType.API_SIGNATURE_CHANGED,
            ):
                extracted = code_extractor.extract_component_at_commit(
                    change_info.file_path, change_info.name, current_commit
                )
                if extracted:
                    source_code = extracted.get("source_code", source_code)
                    docstring = extracted.get("docstring", docstring)
                    parameters = extracted.get("parameters", parameters)
            elif change_info.change_type == ChangeType.REMOVED_COMPONENT:
                docstring = f"⚠️ DEPRECATED: This component was removed.\n\n{docstring}"
        else:
            # Downstream component — use current working tree
            if node.file_path:
                full_path = os.path.join(config.repo_path, node.file_path)
                if os.path.exists(full_path):
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            source_code = f.read()
                    except Exception:
                        pass

        component = {
            "name": node.name,
            "component_type": node.component_type,
            "file_path": node.file_path,
            "relative_path": node.relative_path,
            "docstring": docstring,
            "source_code": source_code or "",
            "depends_on": list(node.depends_on) if node.depends_on else [],
        }
        return await pipeline.agent.execute(
            {
                "type": "write_doc",
                "config": config,
                "input": {
                    "type": "component",
                    "component": component,
                    "output_path": output_path,
                },
            }
        )

    sem = asyncio.Semaphore(16)

    async def _run(node_id: str):
        async with sem:
            return await regenerate_component(node_id)

    results = await asyncio.gather(
        *(_run(cid) for cid in impact["affected_components"]),
        return_exceptions=True,
    )

    def is_successful_result(r) -> bool:
        return not isinstance(r, Exception) and bool(r) and getattr(r, "success", False)

    success_count = sum(1 for r in results if is_successful_result(r))
    print(
        f"\nSuccessfully regenerated {success_count}/{len(results)} component(s)"
    )

    duration = time.time() - start_time

    # --- Logging + metadata ---
    from repodoc.llm import total_token_usage

    usage = total_token_usage(config)
    log_operation(
        output_dir=config.output_dir,
        operation_type="incremental_update",
        repo_path=config.repo_path,
        git_commit=current_commit,
        base_commit=base_commit,
        duration_seconds=duration,
        total_tokens=usage["total"],
        prompt_tokens=usage["prompt"],
        completion_tokens=usage["completion"],
        llm_calls=usage["llm_calls"],
        components_processed=len(impact["affected_components"]),
        components_updated=success_count,
        status="success",
    )

    if current_commit and os.path.exists(metadata_path):
        try:
            metadata = file_manager.load_json(metadata_path)
            if metadata:
                metadata["generation_info"]["git_commit"] = current_commit
                if "update_info" not in metadata:
                    metadata["update_info"] = {}
                metadata["update_info"]["last_update_commit"] = current_commit
                metadata["update_info"]["last_base_commit"] = base_commit
                metadata["update_info"]["components_updated"] = success_count
                metadata["update_info"]["affected_components"] = impact[
                    "affected_components"
                ]
                metadata["generation_info"]["timestamp"] = datetime.now().isoformat()
                file_manager.save_json(metadata, metadata_path)
        except Exception as e:
            print(f"Warning: Could not update metadata: {e}")

    print("\nDocumentation update complete!")
    return {
        "status": "success",
        "impact": impact,
        "updated": success_count,
        "total": len(results),
    }
