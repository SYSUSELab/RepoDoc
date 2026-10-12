"""Incremental documentation update: git diff analysis -> graph mutation -> impact -> docs.

The knowledge graph is the single source of truth for the repository's
current code state. `update` works in four stages:

1. CHANGE ANALYSIS — git diff (old content @ base commit vs working tree) is
   re-analyzed per file; component-level changes are classified by severity
   (signature/new/removed are "severe", body/docstring changes are not).
2. GRAPH MUTATION — the knowledge graph is mutated in place: new components
   are added (with heuristic module placement), modified components are
   overwritten (with calls-edge diffs), removed components are deleted with
   all incident edges; cross-module edges are rebuilt.
3. LAYERED IMPACT — propagation from the changed components over REL_CALLS
   predecessors (callers), bounded by --max-hops; modules whose members
   changed get their module docs refreshed; the overview follows.
4. DOC UPDATE — Plan A minimal-edit regeneration: the existing document is
   embedded as the format anchor and the LLM only edits affected sections,
   so repeated updates do not cause format drift. Full regeneration is used
   for new components, missing docs, or large-change/high-coverage cases.
"""

import asyncio
import difflib
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from repodoc.agent.generate import (
    DocPipeline,
    _build_module_dependencies,
    _collect_component_modules,
    refresh_doc_snapshots,
)
from repodoc.config import MAX_SOURCE_FILE_BYTES, Config
from repodoc.graph.analyzers import get_analyzer
from repodoc.graph.builder import component_dependencies
from repodoc.graph.graph import HeterogeneousGraph
from repodoc.graph.models import (
    CODE_DEPENDENCY_RELATIONS,
    REL_CALLS,
    REL_DESCRIBES,
    REL_IMPLEMENTS,
    RELATION_NODE_FIELDS,
    CodeNode,
    DocNode,
    ModuleNode,
    Node,
)
from repodoc.graph.mutation import (
    GraphMutationResult,
    apply_file_changes_to_graph,
    component_module_ids,
)
from repodoc.utils import (
    file_manager,
    log_operation,
    summarize_markdown,
    validate_and_fix_links,
)

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
    # Analyzer component id (exact graph node id, incl. methods); "" when the
    # analyzer could not supply one (then callers fall back to name-based id).
    component_id: str = ""


@dataclass
class FileChangeAnalysis:
    file_path: str
    language: str
    changes: list[ComponentChange] = field(default_factory=list)
    has_important_changes: bool = False


@dataclass
class FileDiff:
    """Old/new contents of one changed file plus its change analysis.

    None content means the file does not exist on that side (added/deleted);
    "" is an existing but empty file."""

    file_path: str
    old_content: Optional[str]
    new_content: Optional[str]
    spec: Optional["DiffSpec"] = None
    analysis: Optional[FileChangeAnalysis] = None


IMPORTANT_TYPES = frozenset(
    {
        ChangeType.API_SIGNATURE_CHANGED,
        ChangeType.NEW_COMPONENT,
        ChangeType.REMOVED_COMPONENT,
        ChangeType.CODE_BODY_CHANGED,
    }
)

SEVERE_TYPES = frozenset(
    {
        ChangeType.API_SIGNATURE_CHANGED,
        ChangeType.NEW_COMPONENT,
        ChangeType.REMOVED_COMPONENT,
    }
)


@dataclass(frozen=True)
class DiffSpec:
    """Per-language diff behavior.

    - supports_docstring: whether docstrings can be compared (C analyzers
      never extract them, so skip the branch).
    - compare_source: force whole-source comparison instead of
      signature/docstring comparison (off for every bundled language; Go/Rust
      now expose parameters so they no longer need it).
    - important_types: change types that make a file "important" (all four
      for every language).
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
    ".hxx": DiffSpec("cpp", supports_docstring=False),
    ".php": DiffSpec("php"),
    ".phtml": DiffSpec("php"),
    ".inc": DiffSpec("php"),
    # Go/Rust analyzers extract parameters + docstrings, so the generic
    # signature comparison applies; compare_source made every edit a body
    # change, which disabled --propagate sig for both languages.
    ".go": DiffSpec("go"),
    ".rs": DiffSpec("rust"),
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


def _component_id(file_path: str, name: str) -> str:
    """Fallback component id: <module.path>.<name>.

    Mirrors analyzer ids (module path = repo-relative path minus the file
    extension, separators become dots). Only used when a change carries no
    analyzer id (e.g. analyzer unavailable); analyze_diff sets component_id
    from the analyzer itself for every normal change."""
    suffix = Path(file_path).suffix
    module_path = file_path[: -len(suffix)] if suffix else file_path
    return f"{module_path.replace(os.sep, '.').replace('/', '.')}.{name}"


def analyze_diff(
    file_path: str,
    old_content: Optional[str],
    new_content: Optional[str],
    spec: DiffSpec,
) -> FileChangeAnalysis:
    """Compare two versions of a file via its language analyzer (generic).

    None content means the file did not exist on that side; it is compared as
    an empty document (new file -> all components NEW, deletion -> all
    REMOVED)."""
    analysis = FileChangeAnalysis(file_path=file_path, language=spec.language)
    old_content = old_content or ""
    new_content = new_content or ""

    if old_content == new_content:
        return analysis

    analyzer_cls = get_analyzer(spec.language)
    if analyzer_cls is None:
        # No analyzer: the file content changed but no component can be
        # attributed, so mark it important without a (bogus) severe id —
        # propagation has nothing concrete to seed from.
        analysis.has_important_changes = old_content != new_content
        if old_content != new_content:
            analysis.changes.append(
                ComponentChange(
                    name="<file_changed>",
                    change_type=ChangeType.CODE_BODY_CHANGED,
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

    # Key by analyzer component ID, not name: name keying silently collapses
    # same-named methods/overloads from different classes into one entry.
    old_components = {n.id: n for n in old_analyzer.nodes if n.id}
    new_components = {n.id: n for n in new_analyzer.nodes if n.id}

    def _impl_map(analyzer) -> dict[str, set[str]]:
        """Base classes/interfaces per component id, from the raw analyzer
        relationships (Node.implements is only filled during graph assembly)."""
        out: dict[str, set[str]] = {}
        for rel in analyzer.call_relationships:
            if getattr(rel, "relation_type", REL_CALLS) == REL_IMPLEMENTS:
                out.setdefault(rel.caller, set()).add(rel.callee)
        return out

    old_impls = _impl_map(old_analyzer)
    new_impls = _impl_map(new_analyzer)

    old_ids = set(old_components.keys())
    new_ids = set(new_components.keys())

    for cid in sorted(new_ids - old_ids):
        node = new_components[cid]
        analysis.changes.append(
            ComponentChange(
                name=node.name,
                component_id=cid,
                change_type=ChangeType.NEW_COMPONENT,
                file_path=file_path,
                details=f"New {node.component_type} added",
                line_number=node.start_line,
            )
        )

    for cid in sorted(old_ids - new_ids):
        node = old_components[cid]
        analysis.changes.append(
            ComponentChange(
                name=node.name,
                component_id=cid,
                change_type=ChangeType.REMOVED_COMPONENT,
                file_path=file_path,
                details=f"{node.component_type} removed",
            )
        )

    for cid in sorted(old_ids & new_ids):
        old_node = old_components[cid]
        new_node = new_components[cid]

        if spec.compare_source:
            if old_node.source_code != new_node.source_code:
                analysis.changes.append(
                    ComponentChange(
                        name=new_node.name,
                        component_id=cid,
                        change_type=ChangeType.CODE_BODY_CHANGED,
                        file_path=file_path,
                        details="Source code changed",
                    )
                )
            continue

        old_bases = old_impls.get(old_node.id, set())
        new_bases = new_impls.get(new_node.id, set())
        if old_bases != new_bases:
            analysis.changes.append(
                ComponentChange(
                    name=new_node.name,
                    component_id=cid,
                    change_type=ChangeType.API_SIGNATURE_CHANGED,
                    file_path=file_path,
                    old_signature=str(sorted(old_bases)),
                    new_signature=str(sorted(new_bases)),
                    details="Base classes / implemented interfaces changed",
                )
            )
            continue

        old_params = old_node.parameters or []
        new_params = new_node.parameters or []

        if old_params != new_params:
            analysis.changes.append(
                ComponentChange(
                    name=new_node.name,
                    component_id=cid,
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
                        name=new_node.name,
                        component_id=cid,
                        change_type=ChangeType.DOCSTRING_CHANGED,
                        file_path=file_path,
                        details="Only docstring changed",
                    )
                )
            else:
                analysis.changes.append(
                    ComponentChange(
                        name=new_node.name,
                        component_id=cid,
                        change_type=ChangeType.CODE_BODY_CHANGED,
                        file_path=file_path,
                        details="Code body or docstring changed",
                    )
                )
        else:
            analysis.changes.append(
                ComponentChange(
                    name=new_node.name,
                    component_id=cid,
                    change_type=ChangeType.CODE_BODY_CHANGED,
                    file_path=file_path,
                    details="Only internal code changed",
                )
            )

    analysis.has_important_changes = any(
        c.change_type in spec.important_types for c in analysis.changes
    )
    return analysis


def _detect_renames(
    repo_path: str, base_commit: Optional[str] = None
) -> dict[str, str]:
    """Detect renamed files (new path -> old path) from git."""
    cmd = (
        ["git", "diff", "--name-status", "--find-renames", base_commit]
        if base_commit
        else ["git", "diff", "--name-status", "--find-renames", "HEAD"]
    )
    try:
        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)
    except Exception as e:
        logger.debug("git rename detection failed: %s", e)
        return {}
    if result.returncode != 0:
        return {}
    renames: dict[str, str] = {}
    for line in result.stdout.strip().split("\n"):
        if not line or "\t" not in line:
            continue
        fields = line.split("\t")
        if fields[0].startswith("R") and len(fields) >= 3:
            renames[fields[2]] = fields[1]
    return renames


class GitError(RuntimeError):
    """A git command failed for reasons other than 'path does not exist'."""


def _git_show(repo_path: str, file_path: str, commit: str | None) -> Optional[str]:
    """File content at a commit; None when the file did not exist there.

    Raises GitError when git itself failed (bad ref, broken repo) so a failed
    read is never misread as a deletion."""
    ref = commit or "HEAD"
    try:
        result = subprocess.run(
            ["git", "show", f"{ref}:{file_path}"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as e:
        raise GitError(f"git show {ref}:{file_path} failed: {e}") from e
    if result.returncode == 0:
        return result.stdout
    # Non-zero: either the path is absent at this ref (normal for new files)
    # or the ref/repo is broken. Ask git directly to tell the two apart.
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{ref}:{file_path}"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if exists.returncode != 0:
        ref_ok = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", ref],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if ref_ok.returncode != 0:
            raise GitError(
                f"base ref {ref!r} is not a valid commit; "
                "run update with --base-commit or regenerate docs"
            )
        return None  # path absent at this ref
    raise GitError(
        f"git show {ref}:{file_path} failed: {result.stderr.strip()}"
    )


def _read_working_tree(repo_path: str, file_path: str) -> Optional[str]:
    """Working-tree content; None when the file does not exist (deleted).

    Read errors propagate: a transient permission/lock failure must never be
    mistaken for a deletion. "" is a real empty file. Oversized files are
    skipped (raise) for the same reason generated files are skipped during
    full analysis."""
    full_path = Path(repo_path) / file_path
    if not full_path.exists():
        return None
    size = full_path.stat().st_size
    if size > MAX_SOURCE_FILE_BYTES:
        raise OSError(f"file too large to analyze ({size} bytes)")
    return full_path.read_text(encoding="utf-8", errors="replace")


def analyze_git_diff(
    repo_path: str,
    base_commit: Optional[str] = None,
    include_tests: bool = False,
) -> tuple[dict[str, FileDiff], dict[str, str]]:
    """Analyze git diff to find code changes (old vs working tree per file).

    Deleted files ARE included (old content comes from `git show`), so
    component removals are detected instead of being skipped. Untracked new
    files are scanned explicitly (`git diff` is blind to them).

    Returns (diffs, renames): diffs is file_path -> FileDiff; renames maps
    new path -> old path for detected renames (old content of a renamed file
    is read from the OLD path at the base commit)."""
    cmd = (
        ["git", "diff", "--name-status", "--find-renames", base_commit]
        if base_commit
        else ["git", "diff", "--name-status", "--find-renames", "HEAD"]
    )
    result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("git diff failed: %s", result.stderr.strip())
        return {}, {}
    changed_files: list[str] = []
    renames: dict[str, str] = {}
    for line in result.stdout.strip().split("\n"):
        if not line or "\t" not in line:
            continue
        fields = line.split("\t")
        status = fields[0]
        if status.startswith("R") and len(fields) >= 3:
            renames[fields[2]] = fields[1]
            changed_files.append(fields[2])
        elif status.startswith("R"):
            changed_files.append(fields[-1])
        else:
            changed_files.append(fields[1])

    # Untracked (new) files are invisible to `git diff`; scan them explicitly
    # so brand-new source files are detected as additions.
    try:
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if untracked.returncode == 0:
            changed_files.extend(
                f for f in untracked.stdout.strip().split("\n") if f
            )
    except Exception as e:
        logger.debug("git ls-files failed: %s", e)

    if not include_tests:
        changed_files = [f for f in changed_files if not _is_test_file(f)]

    diffs: dict[str, FileDiff] = {}

    # Generation walks only the include patterns and honours the ignore
    # patterns; update must apply the SAME rules, otherwise a changed file in
    # examples/, docs_src/, scripts/... enters the graph through the back door.
    from repodoc.graph.analyzers import (
        DEFAULT_IGNORE_PATTERNS,
        DEFAULT_INCLUDE_PATTERNS,
    )
    from repodoc.graph.builder import _should_exclude_path, _should_include_file

    ignore_patterns = set(DEFAULT_IGNORE_PATTERNS)

    for file_path in changed_files:
        spec = get_diff_spec(file_path)
        if spec is None:
            # Non-code file (md, json, ...) — does not affect API docs
            continue
        fname = Path(file_path).name
        if _should_exclude_path(file_path, fname, ignore_patterns):
            logger.debug("Skipping ignored path: %s", file_path)
            continue
        if not _should_include_file(file_path, fname, DEFAULT_INCLUDE_PATTERNS):
            logger.debug("Skipping non-included path: %s", file_path)
            continue

        old_path = renames.get(file_path, file_path)
        try:
            old_content = _git_show(repo_path, old_path, base_commit)
            new_content = _read_working_tree(repo_path, file_path)
        except (GitError, OSError) as e:
            # Never reconcile a file whose old/new content could not be read:
            # a transient read failure must not be treated as a deletion.
            logger.warning("Skipping %s: %s", file_path, e)
            continue

        # A pure rename (identical content) must still be processed: its
        # components keep their identity but change ids, so the graph and the
        # docs have to be migrated (skipping it would leave the old path's
        # nodes and docs in place forever).
        if old_content == new_content and file_path not in renames:
            continue

        diffs[file_path] = FileDiff(
            file_path=file_path,
            old_content=old_content,
            new_content=new_content,
            spec=spec,
            analysis=analyze_diff(file_path, old_content, new_content, spec),
        )

    return diffs, renames


def filter_important_changes(
    diffs: dict[str, FileDiff],
) -> dict[str, FileDiff]:
    """Filter to only files with important API-affecting changes."""
    return {
        file_path: fd
        for file_path, fd in diffs.items()
        if fd.analysis and fd.analysis.has_important_changes
    }


def _code_diff(old_content: Optional[str], new_content: Optional[str], file_path: str) -> str:
    """Unified diff of old vs new content (pure difflib, no git dependency)."""
    old_content = old_content or ""
    new_content = new_content or ""
    if not old_content and not new_content:
        return ""
    diff = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )
    return "".join(diff)


# ---------------------------------------------------------------------------
# Layered impact computation
# ---------------------------------------------------------------------------


class IncrementalUpdater:
    """Layered impact analysis over the knowledge graph.

    Graph edge convention: caller -> callee (REL_CALLS). A change to X
    affects the components that CALL X, i.e. X's predecessors. Propagation
    is bounded by max_hops and gated by change severity:
    - layer 0: the changed components themselves (their docs always update)
    - layer 1: transitive callers of SEVERE changes (signature/new/removed),
      up to max_hops; body/docstring changes do not propagate outward
    - layer 2: modules containing any layer-0/1 component (module docs)
    - layer 3: the overview README (when any module doc is affected)
    """

    def __init__(self, graph: HeterogeneousGraph, config: Any):
        self.graph = graph
        self.config = config

    def _bounded_callers(self, seeds: set[str], max_hops: int) -> set[str]:
        """Dependents reachable from seeds within max_hops via REL_CALLS or
        REL_IMPLEMENTS predecessors (excludes the seeds themselves).

        Both edges mean "this component depends on the changed one": callers
        must follow API changes, and subclasses/implementors must follow base
        class / interface changes."""
        affected: set[str] = set()
        frontier: list[tuple[str, int]] = [(s, 0) for s in seeds]
        visited: set[str] = set(seeds)

        while frontier:
            component, depth = frontier.pop(0)
            if depth >= max_hops:
                continue
            for pred in self.graph.predecessors(component):
                if pred in visited:
                    continue
                edge_data = self.graph.get_edge_data(pred, component)
                if (
                    not edge_data
                    or edge_data.get("relation_type")
                    not in CODE_DEPENDENCY_RELATIONS
                ):
                    continue
                visited.add(pred)
                affected.add(pred)
                frontier.append((pred, depth + 1))
        return affected

    def compute_layered_impact(
        self,
        changed_components: list[str],
        severe_components: set[str],
        propagate: str = "sig",
        max_hops: int = 3,
        caller_seeds: set[str] | None = None,
    ) -> dict[str, Any]:
        """Compute the layered update scope for the given changed components.

        Args:
            changed_components: ids mutated in this update (added + updated).
            severe_components: ids whose change is signature/new/removed.
            propagate: "sig" (only severe changes propagate to callers) or
                "all" (every changed component propagates).
            max_hops: max caller-propagation depth (0 disables propagation).
            caller_seeds: components treated as changed on top of the seeds
                (e.g. callers of REMOVED components, snapshotted before the
                removal — their docs reference a component that is gone).

        Returns:
            Dict with components_to_doc, modules_to_doc, update_overview,
            severe_components, total_components and the layer breakdown.
        """
        layer0 = [c for c in changed_components if self.graph.has_node(c)]

        seeds = (
            layer0
            if propagate == "all"
            else [c for c in layer0 if c in severe_components]
        )
        # Callers of REMOVED components: the component itself is gone from the
        # graph, so these arrive as explicit extra seeds.
        extra_seeds = set(caller_seeds or set())
        seeds = list(set(seeds) | extra_seeds)
        layer1 = self._bounded_callers(set(seeds), max_hops) if seeds else set()
        # ...and they are affected themselves: their docs describe a component
        # that no longer exists.
        layer1 |= {c for c in extra_seeds if self.graph.has_node(c)}
        layer1 -= set(layer0)

        components_to_doc = sorted(set(layer0) | layer1)
        modules_to_doc = sorted(
            {
                m
                for c in components_to_doc
                for m in component_module_ids(self.graph, c)
            }
        )
        update_overview = bool(modules_to_doc)

        return {
            "changed_components": layer0,
            "severe_components": sorted(severe_components),
            "caller_components": sorted(layer1),
            "components_to_doc": components_to_doc,
            "modules_to_doc": modules_to_doc,
            "update_overview": update_overview,
            "total_components": len(components_to_doc),
        }


# ---------------------------------------------------------------------------
# Graph + asset loading
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


def _graph_save_path(config: Config, project_name: str) -> str:
    return os.path.join(
        config.output_dir, "data", f"{project_name}_knowledge_graph.json"
    )


# ---------------------------------------------------------------------------
# Doc path helpers
# ---------------------------------------------------------------------------


def _module_doc_path(config: Config, module_id: str) -> str:
    """Module doc file for a module node id ('net/http' -> docs/net/http.md)."""
    return os.path.join(config.docs_dir, f"{module_id}.md")


def _component_doc_path(docs_dir: str, chain: list[str], name: str) -> str:
    return os.path.join(docs_dir, *chain, f"{name}.md")


def _sync_doc_node(
    graph: HeterogeneousGraph,
    pipeline: DocPipeline,
    fpath: str,
    content: str,
    target_id: Optional[str] = None,
    stale: bool = False,
) -> None:
    """Attach/refresh the DocNode for a written doc (keeps the graph current)."""
    doc_id = pipeline._doc_id_from_path(fpath)
    graph.add_doc_node(
        DocNode(
            id=doc_id,
            content=content,
            format="markdown",
            file_path=fpath,
            stale=stale,
        )
    )
    if target_id and graph.has_node(target_id) and not graph.has_edge(doc_id, target_id):
        graph.add_edge(doc_id, target_id, REL_DESCRIBES, 1.0)


def _mark_doc_stale(
    graph: HeterogeneousGraph, pipeline: DocPipeline, doc_path: str, stale: bool = True
) -> None:
    """Set/clear the stale flag on a doc node (skipped manual edits retry)."""
    doc_id = pipeline._doc_id_from_path(doc_path)
    node = graph.get_node(doc_id)
    if isinstance(node, DocNode):
        graph.add_doc_node(
            DocNode(
                id=node.id,
                content=node.content,
                format=node.format,
                file_path=node.file_path,
                version=node.version,
                stale=stale,
            )
        )


def _stale_doc_targets(
    graph: HeterogeneousGraph,
) -> tuple[set[str], set[str], bool]:
    """Components/modules/overview whose docs were skipped (stale flag set).

    These stay on the update list across runs until their docs are refreshed
    (or deleted with the component), so a manually edited doc is retried on
    every update instead of silently falling off the schedule."""
    components: set[str] = set()
    modules: set[str] = set()
    overview = False
    for s, t, rt, _w in graph.edges(REL_DESCRIBES):
        d = graph.get_node(s)
        if not (isinstance(d, DocNode) and d.stale):
            continue
        target = graph.get_node(t)
        if isinstance(target, ModuleNode):
            modules.add(t)
        else:
            components.add(t)
    ov = graph.get_node("overview")
    if isinstance(ov, DocNode) and ov.stale:
        overview = True
    return components, modules, overview


# ---------------------------------------------------------------------------
# Doc update strategies (Plan A: minimal-edit; full regen fallback)
# ---------------------------------------------------------------------------


def should_partial_update(existing_doc: str, code_diff: str) -> bool:
    """Decide between minimal-edit update and full regeneration.

    Full regeneration when: no existing doc, or the doc is short AND the
    change is large (embedding the old doc would cost more than rewriting,
    and a big diff leaves few sections untouched to preserve)."""
    if not existing_doc:
        return False
    if len(existing_doc) < 1000 and len(code_diff) > 2000:
        return False
    return True


def _safe_read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def _user_edited_doc(
    graph: HeterogeneousGraph, pipeline: DocPipeline, doc_path: str, file_content: str
) -> bool:
    """True when the doc file differs from the graph's DocNode content.

    The graph DocNode records the last content WE wrote; a mismatch means a
    human (or another tool) modified the file — never overwrite it silently."""
    doc_id = pipeline._doc_id_from_path(doc_path)
    node = graph.get_node(doc_id)
    if not isinstance(node, DocNode):
        return False
    return (node.content or "") != file_content


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def _run_jobs_with_retry(jobs: list, run_fn: Callable, label: str) -> list:
    """Run doc jobs with a shared semaphore and failure-driven retries that
    explicitly lower concurrency each round (mirrors generate.py)."""
    from repodoc.llm import _LIMITER

    sem = asyncio.Semaphore(16)

    async def _one(job):
        async with sem:
            return await run_fn(job)

    results = await asyncio.gather(*(_one(j) for j in jobs), return_exceptions=True)
    for _round in range(4):
        failed = [
            j
            for j, r in zip(jobs, results)
            if isinstance(r, Exception)
            or (r is not None and not getattr(r, "success", False))
        ]
        if not failed:
            break
        if _LIMITER.permits > _LIMITER.min_permits:
            _LIMITER.force_decrease()
        print(
            f"Concurrency {_LIMITER.permits}, retrying {len(failed)} {label} "
            f"(round {_round + 1})..."
        )
        retry = await asyncio.gather(*(_one(j) for j in failed), return_exceptions=True)
        retry_iter = iter(retry)
        results = [
            next(retry_iter)
            if (
                isinstance(r, Exception)
                or (r is not None and not getattr(r, "success", False))
            )
            else r
            for r in results
        ]
        if _LIMITER.permits <= _LIMITER.min_permits:
            break
    return results


def _remove_component_docs(
    graph: HeterogeneousGraph,
    pipeline: DocPipeline,
    removed: list[str],
    pre_mutation_modules: dict[str, list[str]],
) -> int:
    """Delete doc files + doc nodes for removed components. Returns count."""
    removed_count = 0
    for node_id in removed:
        chain = pre_mutation_modules.get(node_id, [])
        node = pipeline.components.get(node_id)
        name = node.name if node else node_id.split(".")[-1]
        doc_path = _component_doc_path(pipeline.config.docs_dir, chain, name)
        if os.path.exists(doc_path):
            try:
                os.remove(doc_path)
                removed_count += 1
                print(f"  Removed doc: {os.path.relpath(doc_path, pipeline.config.docs_dir)}")
            except OSError as e:
                logger.warning("Failed to remove doc %s: %s", doc_path, e)
        doc_id = pipeline._doc_id_from_path(doc_path)
        if graph.has_node(doc_id):
            graph.remove_node(doc_id)  # drops describes edges too
    return removed_count


def _migrate_renamed_docs(
    graph: HeterogeneousGraph,
    pipeline: DocPipeline,
    renamed_pairs: list[tuple[str, str]],
    pre_mutation_modules: dict[str, list[str]],
    post_mutation_modules: dict[str, list[str]],
) -> int:
    """Move docs + transfer doc nodes/describes edges for renamed components.

    A rename deletes the old component id and adds a new one; the existing
    doc (file + DocNode + describes edge) is migrated to the new id so the
    next update only edits the doc instead of losing it. Returns count."""
    migrated = 0
    docs_dir = pipeline.config.docs_dir
    for old_id, new_id in renamed_pairs:
        name = new_id.split(".")[-1]
        old_chain = pre_mutation_modules.get(old_id, [])
        new_chain = post_mutation_modules.get(new_id, [])
        # Tolerate a single module-id string in place of a chain
        if isinstance(old_chain, str):
            old_chain = [old_chain]
        if isinstance(new_chain, str):
            new_chain = [new_chain]
        old_doc_path = _component_doc_path(docs_dir, old_chain, name)
        new_doc_path = _component_doc_path(docs_dir, new_chain, name)

        moved = False
        if os.path.exists(old_doc_path):
            try:
                os.makedirs(os.path.dirname(new_doc_path), exist_ok=True)
                shutil.move(old_doc_path, new_doc_path)
                moved = True
            except OSError as e:
                logger.warning("Failed to move doc %s -> %s: %s", old_doc_path, new_doc_path, e)

        old_doc_id = pipeline._doc_id_from_path(old_doc_path)
        new_doc_id = pipeline._doc_id_from_path(new_doc_path)
        if graph.has_node(old_doc_id):
            old_doc = graph.get_node(old_doc_id)
            graph.remove_node(old_doc_id)
            graph.add_doc_node(
                DocNode(
                    id=new_doc_id,
                    content=old_doc.content if old_doc else "",
                    format="markdown",
                    file_path=new_doc_path,
                )
            )
            if graph.has_node(new_id):
                graph.add_edge(new_doc_id, new_id, REL_DESCRIBES, 1.0)
            migrated += 1
            print(
                f"  Migrated doc: {os.path.relpath(old_doc_path, docs_dir)} -> "
                f"{os.path.relpath(new_doc_path, docs_dir)}"
            )
        elif moved:
            migrated += 1
    return migrated


def _cleanup_empty_dirs(docs_dir: str) -> int:
    """Remove empty directories under docs_dir (bottom-up). Returns count."""
    removed = 0
    for root, _dirs, _files in os.walk(docs_dir, topdown=False):
        if root == docs_dir:
            continue
        try:
            if not os.listdir(root):
                os.rmdir(root)
                removed += 1
        except OSError:
            pass
    return removed


# ---------------------------------------------------------------------------
# Pending-update journal (crash / failure recovery)
# ---------------------------------------------------------------------------


def _pending_updates_path(config: Config, project_name: str) -> str:
    return os.path.join(
        config.output_dir, "data", f"{project_name}_pending_updates.json"
    )


def _load_pending_updates(config: Config, project_name: str) -> dict[str, Any]:
    """Docs left unfinished by an interrupted or partially failed run.

    The graph is mutated before docs are written, so a crash or an LLM
    failure can leave code nodes updated while their docs still describe the
    old code. This journal keeps those docs scheduled until they succeed (or
    their component disappears)."""
    data = file_manager.load_json(_pending_updates_path(config, project_name))
    if not isinstance(data, dict):
        return {"components": [], "modules": [], "overview": False}
    return {
        "components": [c for c in data.get("components", []) if isinstance(c, str)],
        "modules": [m for m in data.get("modules", []) if isinstance(m, str)],
        "overview": bool(data.get("overview")),
    }


def _save_pending_updates(
    config: Config, project_name: str, pending: dict[str, Any]
) -> None:
    """Persist (or clear) the pending-update journal atomically."""
    path = _pending_updates_path(config, project_name)
    if not (
        pending.get("components")
        or pending.get("modules")
        or pending.get("overview")
    ):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        return
    pending = dict(pending)
    pending["updated_at"] = datetime.now().isoformat()
    file_manager.save_json(pending, path)


def _pending_is_empty(pending: dict[str, Any]) -> bool:
    return not (
        pending.get("components")
        or pending.get("modules")
        or pending.get("overview")
    )


def _job_failed(result: Any) -> bool:
    """True when a doc job raised or the agent reported failure. A None
    result is a deliberate skip (manual edit) and is not a failure."""
    return isinstance(result, Exception) or (
        result is not None and not getattr(result, "success", False)
    )


async def run_update(
    config: Config,
    changed_files: Optional[tuple[str, ...]] = None,
    base_commit: Optional[str] = None,
    analyze_changes: bool = True,
    include_tests: bool = False,
    confirm: Optional[Callable[[str], bool]] = None,
    verbose: bool = False,
    propagate: str = "sig",
    max_hops: int = 3,
    force: bool = False,
    dry_run: bool = False,
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
        propagate: "sig" (only signature/new/removed changes propagate to
            callers) or "all" (every changed component propagates).
        max_hops: Max caller-propagation depth (0 disables propagation).
        force: Overwrite docs that were manually edited since the last update.
        dry_run: Compute and report the update scope without touching the
            graph, docs, journal or metadata.

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

    # --- Stage 1: change analysis ---
    diffs: dict[str, FileDiff] = {}
    renames: dict[str, str] = {}
    doc_diffs: dict[str, FileDiff] = {}
    if analyze_changes and not changed_files:
        print("\nAnalyzing git changes...")
        diffs, renames = analyze_git_diff(config.repo_path, base_commit, include_tests)
        if not diffs:
            print("No changes detected")
            return {"status": "no_changes", "impact": None}

        doc_diffs = filter_important_changes(diffs)
        if doc_diffs:
            print(f"\nFiles with important changes: {len(doc_diffs)}")
    elif changed_files:
        # Explicit file list: rename info is not available from auto-detection,
        # so query git for the rename pairs explicitly.
        renames = _detect_renames(config.repo_path, base_commit)
        for f in changed_files:
            spec = get_diff_spec(f)
            if spec is None:
                continue
            old_path = renames.get(f, f)
            try:
                old_content = _git_show(config.repo_path, old_path, base_commit)
                new_content = _read_working_tree(config.repo_path, f)
            except (GitError, OSError) as e:
                logger.warning("Skipping %s: %s", f, e)
                continue
            if old_content == new_content and f not in renames:
                continue
            diffs[f] = FileDiff(
                file_path=f,
                old_content=old_content,
                new_content=new_content,
                spec=spec,
                analysis=analyze_diff(f, old_content, new_content, spec),
            )
        if not diffs:
            print("No changed files to process")
            return {"status": "no_changes", "impact": None}
        doc_diffs = diffs
    else:
        print("Detecting changed files via git...")
        diffs, renames = analyze_git_diff(
            config.repo_path, base_commit, include_tests
        )
        if not diffs:
            print("No changed files to process")
            return {"status": "no_changes", "impact": None}
        doc_diffs = diffs

    # --- Stage 2: load graph + mutate it ---
    try:
        graph, module_tree = _load_graph_and_assets(config, project_name)
    except FileNotFoundError as e:
        print(e)
        return {"status": "error", "impact": None}
    print(
        f"Loaded knowledge graph: {graph.node_count()} nodes, {graph.edge_count()} edges"
    )

    # Pre-mutation components: needed to pair renamed components (old id -> new id).
    from repodoc.graph.builder import graph_to_components, graph_to_module_tree

    pre_components = graph_to_components(graph)

    # Docs left unfinished by a previous (crashed/partial) run stay on the
    # schedule even when this run sees no new important changes.
    pending_updates = _load_pending_updates(config, project_name)
    if not _pending_is_empty(pending_updates):
        print(
            f"Resuming {len(pending_updates['components'])} component + "
            f"{len(pending_updates['modules'])} module pending doc update(s) "
            "from a previous run"
        )

    pre_mutation_modules = _collect_component_modules(
        graph_to_module_tree(graph)
    )

    # Pre-mutation dependent snapshot: REMOVED components lose their incoming
    # edges during mutation, so their callers/implementors must be snapshotted
    # BEFORE the removal to remain reachable for impact propagation. The edge
    # relation is recorded per (target, pred) pair so the rename follow-through
    # can re-point the correct list (depends_on vs implements).
    pre_callers: dict[str, set[str]] = {}
    pre_ref_types: dict[tuple[str, str], str] = {}
    for nid, attrs in graph.nodes(data=True):
        if attrs.get("node_type") != "code":
            continue
        callers = set()
        for pred in graph.predecessors(nid):
            ed = graph.get_edge_data(pred, nid)
            if ed and ed.get("relation_type") in CODE_DEPENDENCY_RELATIONS:
                callers.add(pred)
                pre_ref_types[(nid, pred)] = ed["relation_type"]
        pre_callers[nid] = callers

    print(f"\nMutating knowledge graph ({len(diffs)} file(s))...")
    mutation = GraphMutationResult()
    per_file_diffs: dict[str, str] = {}
    for fd in doc_diffs.values():
        per_file_diffs[fd.file_path] = _code_diff(
            fd.old_content, fd.new_content, fd.file_path
        )
    for fd in diffs.values():
        r = apply_file_changes_to_graph(
            graph,
            fd.file_path,
            fd.new_content,
            config.repo_path,
            old_file_path=renames.get(fd.file_path),
        )
        mutation.added.extend(r.added)
        mutation.updated.extend(r.updated)
        mutation.removed.extend(r.removed)
        mutation.added_edges.extend(r.added_edges)
        mutation.removed_edges.extend(r.removed_edges)
        mutation.modules_added.extend(r.modules_added)
    mutation.added = sorted(set(mutation.added))
    mutation.updated = sorted(set(mutation.updated))
    mutation.removed = sorted(set(mutation.removed))

    if not dry_run:
        graph.save(_graph_save_path(config, project_name))
    print(
        f"  Added: {len(mutation.added)}, updated: {len(mutation.updated)}, "
        f"removed: {len(mutation.removed)} component(s); "
        f"edges +{len(mutation.added_edges)}/-{len(mutation.removed_edges)}"
    )

    # Rename pairs (old component id -> new component id), matched by the
    # file rename plus the component name. Resolved BEFORE the impact stage:
    # a renamed component keeps its identity under a new id, so its callers
    # (whose files were not re-analyzed) must follow it, and its old id must
    # not count as a removal — that would delete the doc we want to migrate.
    renamed_pairs: list[tuple[str, str]] = []
    for new_path, old_path in renames.items():
        old_ids = {
            cid
            for cid, n in pre_components.items()
            if n.relative_path == old_path
        }
        old_by_name = {pre_components[cid].name: cid for cid in old_ids}
        for cid in mutation.added:
            if not graph.has_node(cid):
                continue
            node = graph.get_node(cid)
            if (
                getattr(node, "relative_path", None) == new_path
                and node.name in old_by_name
            ):
                renamed_pairs.append((old_by_name[node.name], cid))
    renamed_old_ids = {old for old, _new in renamed_pairs}
    if renamed_old_ids:
        mutation.removed = [r for r in mutation.removed if r not in renamed_old_ids]

    # Callers/implementors follow the rename: re-point their edges and stored
    # depends_on / implements entries from the old id to the new one
    # (pre_ref_types was snapshotted before the mutation, when the old id
    # still existed and its incoming relation was visible).
    followed = 0
    for old_id, new_id in renamed_pairs:
        for pred in sorted(pre_callers.get(old_id, set())):
            if not (graph.has_node(pred) and graph.has_node(new_id)):
                continue
            pred_node = graph.get_node(pred)
            relation = pre_ref_types.get((old_id, pred), REL_CALLS)
            field = RELATION_NODE_FIELDS.get(relation, "depends_on")
            values = set(getattr(pred_node, field, None) or [])
            # The removal pass already detached the old id; the new id must be
            # attached either way (that is the rename following through), so
            # compare against the desired set, not just presence.
            new_values = (values - {old_id}) | {new_id}
            if new_values != values:
                data = pred_node.model_dump()
                data[field] = sorted(new_values)
                graph.add_code_node(CodeNode.model_validate(data))
            if not graph.has_edge(pred, new_id):
                graph.add_edge(pred, new_id, relation, 1.0)
                followed += 1
    if followed:
        print(f"  Re-pointed {followed} caller edge(s) to renamed component id(s)")
        if not dry_run:
            graph.save(_graph_save_path(config, project_name))

    if (
        not doc_diffs
        and not renamed_pairs
        and _pending_is_empty(pending_updates)
    ):
        print(
            "\nNo important API changes detected — documentation update not "
            "needed (knowledge graph synced to the working tree)."
        )
        return {"status": "no_important_changes", "impact": None}

    # Severe ids: signature / new / removed changes drive caller propagation
    severe_components: set[str] = set()
    for fd in doc_diffs.values():
        for change in (fd.analysis.changes if fd.analysis else []):
            if change.change_type in SEVERE_TYPES:
                severe_components.add(
                    change.component_id
                    or _component_id(fd.file_path, change.name)
                )
    severe_components &= set(mutation.changed) | set(mutation.removed)

    # Callers of REMOVED components (snapshotted pre-mutation): their docs
    # reference the now-deleted component and need refreshing.
    removed_caller_seeds = {
        c
        for nid in mutation.removed
        for c in pre_callers.get(nid, set())
    }

    # Modules that lost members: their docs must drop the removed components.
    removed_module_ids = sorted(
        {
            "/".join(pre_mutation_modules.get(nid, []))
            for nid in mutation.removed
            if pre_mutation_modules.get(nid)
        }
    )

    # --- Stage 3: layered impact ---
    module_tree = graph_to_module_tree(graph)
    pipeline = _build_pipeline_from_graph(graph, config)
    updater = IncrementalUpdater(graph, config)
    # Doc scope: only components whose file has an IMPORTANT change get their
    # docs refreshed (the graph itself was already fully reconciled above).
    doc_file_paths = {fd.file_path for fd in doc_diffs.values()}
    doc_scope_components = [
        cid
        for cid in mutation.changed
        if graph.has_node(cid)
        and graph.get_node(cid).relative_path in doc_file_paths
    ]
    if not doc_scope_components and doc_file_paths:
        # The graph may already be converged (a previous run synced it before
        # crashing, or the changes cancel out): the code is current and only
        # the docs are behind. Re-document the components of the changed files
        # instead of concluding there is nothing to do.
        doc_scope_components = [
            nid
            for nid, attrs in graph.nodes(data=True)
            if attrs.get("node_type") == "code"
            and attrs.get("data", {}).get("relative_path") in doc_file_paths
        ]
    impact = updater.compute_layered_impact(
        doc_scope_components,
        severe_components,
        propagate=propagate,
        max_hops=max_hops,
        caller_seeds=removed_caller_seeds,
    )
    impact["modules_to_doc"] = sorted(
        set(impact["modules_to_doc"]) | set(removed_module_ids)
    )
    if removed_module_ids:
        impact["update_overview"] = True

    # Docs skipped by manual-edit detection stay on the schedule (stale flag)
    stale_components, stale_modules, stale_overview = _stale_doc_targets(graph)
    impact["components_to_doc"] = sorted(
        set(impact["components_to_doc"]) | stale_components
    )
    # A retried component doc is rewritten in the context of its module, so the
    # modules owning stale components must be refreshed as well.
    stale_component_modules = {
        m for cid in stale_components for m in component_module_ids(graph, cid)
    }
    impact["modules_to_doc"] = sorted(
        set(impact["modules_to_doc"]) | stale_modules | stale_component_modules
    )
    if stale_overview or stale_modules or stale_component_modules:
        impact["update_overview"] = True

    # Unfinished docs from a previous run: resume them (skip ids whose
    # component no longer exists — its doc was already cleaned up).
    pending_components = [
        c for c in pending_updates["components"] if graph.has_node(c)
    ]
    pending_modules = [
        m for m in pending_updates["modules"] if graph.has_node(m)
    ]
    impact["components_to_doc"] = sorted(
        set(impact["components_to_doc"]) | set(pending_components)
    )
    impact["modules_to_doc"] = sorted(
        set(impact["modules_to_doc"]) | set(pending_modules)
    )
    if pending_updates["overview"]:
        impact["update_overview"] = True
    impact["total_components"] = len(impact["components_to_doc"])

    print(f"\nImpact analysis (propagate={propagate}, max_hops={max_hops}):")
    print(f"  Changed components: {len(impact['changed_components'])}")
    if impact["caller_components"]:
        print(f"  Affected callers: {len(impact['caller_components'])}")
    print(f"  Components to document: {len(impact['components_to_doc'])}")
    print(f"  Modules to update: {len(impact['modules_to_doc'])}")
    if impact["update_overview"]:
        print("  README overview: update")

    if (
        not impact["components_to_doc"]
        and not impact["modules_to_doc"]
        and not mutation.removed
        and not renamed_pairs
    ):
        print("\nNo documentation updates needed")
        return {"status": "no_updates_needed", "impact": impact}

    if dry_run:
        estimated_calls = (
            len(impact["components_to_doc"])
            + len(impact["modules_to_doc"])
            + (1 if impact["update_overview"] else 0)
        )
        print("\n[dry-run] no graph, docs or metadata were modified")
        return {
            "status": "dry_run",
            "impact": impact,
            "estimated_llm_calls": estimated_calls,
            "files_changed": len(diffs),
            "components_removed": len(mutation.removed),
        }

    docs_dir = config.docs_dir
    comp_modules = _collect_component_modules(module_tree)

    # --- Rehome module-less components (generate only clusters leaf nodes:
    # functions in class-heavy repos never got a module home). Place them
    # heuristically so their docs land in a proper module directory, and
    # migrate any existing root-level docs into the new home. ---
    from repodoc.graph.mutation import _place_new_component

    rehome_pairs: list[tuple[str, str]] = []
    newly_placed: set[str] = set()
    for cid in impact["components_to_doc"]:
        if component_module_ids(graph, cid):
            continue
        node = pipeline.components.get(cid)
        if not node:
            continue
        _place_new_component(graph, node)
        newly_placed.add(cid)
        root_doc = _component_doc_path(docs_dir, [], node.name)
        if os.path.exists(root_doc):
            rehome_pairs.append((cid, cid))
    if rehome_pairs:
        print(
            f"\nRehoming {len(rehome_pairs)} component doc(s) into module "
            "directories..."
        )
        _migrate_renamed_docs(
            graph,
            pipeline,
            rehome_pairs,
            {c: [] for c, _new in rehome_pairs},
            {c: component_module_ids(graph, c)[0] for c, _new in rehome_pairs},
        )
    if newly_placed:
        # refresh tree/maps so the new homes are visible downstream
        module_tree = graph_to_module_tree(graph)
        comp_modules = _collect_component_modules(module_tree)
        new_modules = {
            component_module_ids(graph, cid)[0] for cid in newly_placed
        }
        impact["modules_to_doc"] = sorted(
            set(impact["modules_to_doc"]) | new_modules
        )
        impact["update_overview"] = True
        print(
            f"  Placed {len(newly_placed)} module-less component(s) into "
            f"{sorted(new_modules)}"
        )

    # --- Rename migration: move the existing docs to the new ids. The pairs
    # were computed before the impact stage (the old ids are already excluded
    # from mutation.removed, so this must run BEFORE removal cleanup below) ---
    migrated_docs = 0
    if renamed_pairs:
        print(f"\nMigrating docs for {len(renamed_pairs)} renamed component(s)...")
        migrated_docs = _migrate_renamed_docs(
            graph,
            pipeline,
            renamed_pairs,
            pre_mutation_modules,
            comp_modules,
        )

    removed_docs = 0
    if mutation.removed:
        print(f"\nCleaning up docs for {len(mutation.removed)} removed component(s)...")
        removed_docs = _remove_component_docs(
            graph, pipeline, mutation.removed, pre_mutation_modules
        )

    cleaned_dirs = _cleanup_empty_dirs(docs_dir)
    if cleaned_dirs:
        print(f"  Removed {cleaned_dirs} empty doc directorie(s)")

    if confirm and not confirm(
        f"\nThis will update {impact['total_components']} component doc(s), "
        f"{len(impact['modules_to_doc'])} module doc(s)"
        + (" and the README" if impact["update_overview"] else "")
        + f" ({removed_docs} removed doc(s) cleaned). Continue?"
    ):
        return {"status": "aborted", "impact": impact}

    # Module dependency map: computed once from the FINAL module tree (the
    # rehome/rename steps above may have moved components) and shared by the
    # module docs and the README regeneration below — the README path reads it
    # even when no module doc is due, so it must not live inside that branch.
    module_deps, module_dependents = _build_module_dependencies(
        pipeline.components, module_tree
    )

    # --- Stage 4: regenerate docs, bottom-up (components -> modules -> README) ---
    print(f"\nUpdating {len(impact['components_to_doc'])} component doc(s)...")
    component_results = []

    def _component_jobs() -> list[tuple[str, str, str]]:
        jobs = []
        for node_id in impact["components_to_doc"]:
            node = pipeline.components.get(node_id)
            if not node:
                continue
            chain = comp_modules.get(node_id, [])
            doc_path = _component_doc_path(docs_dir, chain, node.name)
            existing_doc = _safe_read(doc_path)
            # Diff of the file this component lives in ('' when none)
            code_diff = per_file_diffs.get(node.relative_path, "")
            jobs.append((node_id, doc_path, existing_doc))
        return jobs

    component_jobs = _component_jobs()

    # Journal the full intended scope BEFORE writing any doc: if the process
    # dies mid-run, the next run resumes exactly these docs (at-least-once).
    _save_pending_updates(
        config,
        project_name,
        {
            "components": [job[0] for job in component_jobs],
            "modules": list(impact["modules_to_doc"]),
            "overview": bool(impact["update_overview"]),
            "base_commit": base_commit,
        },
    )

    async def _update_component(job):
        node_id, doc_path, existing_doc = job
        node = pipeline.components.get(node_id)
        if not node:
            return None
        if existing_doc and not force and _user_edited_doc(
            graph, pipeline, doc_path, existing_doc
        ):
            print(
                f"  SKIP {os.path.relpath(doc_path, docs_dir)}: "
                "manually edited since last update "
                "(use --force to overwrite; retried on next update)"
            )
            _mark_doc_stale(graph, pipeline, doc_path)
            return None
        chain = comp_modules.get(node_id, [])
        module_name = ".".join(chain)
        code_diff = per_file_diffs.get(node.relative_path, "")

        predecessors = [
            cid
            for cid, n in pipeline.components.items()
            if node_id in component_dependencies(n)
        ]
        component = {
            "name": node.name,
            "component_type": node.component_type,
            "file_path": node.file_path,
            "relative_path": node.relative_path,
            "docstring": node.docstring,
            "source_code": node.source_code or "",
            "depends_on": sorted(component_dependencies(node)),
        }

        if should_partial_update(existing_doc, code_diff):
            return await pipeline.agent.execute(
                {
                    "type": "write_doc",
                    "config": config,
                    "input": {
                        "type": "update_component",
                        "component": component,
                        "predecessors": predecessors,
                        "module_name": module_name,
                        "existing_doc": existing_doc,
                        "code_diff": code_diff,
                        "output_path": doc_path,
                    },
                }
            )
        # Full regeneration (new doc, missing doc, or large change)
        return await pipeline.agent.execute(
            {
                "type": "write_doc",
                "config": config,
                "input": {
                    "type": "component",
                    "component": component,
                    "predecessors": predecessors,
                    "module_name": module_name,
                    "output_path": doc_path,
                },
            }
        )

    component_results = await _run_jobs_with_retry(
        component_jobs, _update_component, "component docs"
    )

    # Module docs (layer 2): refresh modules whose members changed
    module_results = []
    module_jobs: list[tuple[str, str, dict, str]] = []
    if impact["modules_to_doc"]:
        print(f"\nUpdating {len(impact['modules_to_doc'])} module doc(s)...")

        def _module_info(module_id: str) -> tuple[str, dict]:
            parts = module_id.split("/")
            info: Any = module_tree
            for i, p in enumerate(parts):
                if not isinstance(info, dict) or info.get(p) is None or not isinstance(info[p], dict):
                    return parts[-1], {"components": [], "children": {}, "path": ""}
                info = info[p]
                if i < len(parts) - 1:
                    info = info.get("children", {})
            return parts[-1], info

        for module_id in impact["modules_to_doc"]:
            module_name, module_info = _module_info(module_id)
            module_dir = _module_doc_path(config, module_id)
            module_jobs.append((module_id, module_name, module_info, module_dir))

        async def _update_module(job):
            module_id, module_name, module_info, doc_path = job
            existing_doc = _safe_read(doc_path)
            if existing_doc and not force and _user_edited_doc(
                graph, pipeline, doc_path, existing_doc
            ):
                print(
                    f"  SKIP {os.path.relpath(doc_path, docs_dir)}: "
                    "manually edited since last update "
                    "(use --force to overwrite; retried on next update)"
                )
                _mark_doc_stale(graph, pipeline, doc_path)
                return None

            component_docs_content = {}
            for comp_id in module_info.get("components", []):
                node = pipeline.components.get(comp_id)
                if not (node and node.name):
                    continue
                chain = comp_modules.get(comp_id, [module_name])
                comp_doc_path = _component_doc_path(docs_dir, chain, node.name)
                content = _safe_read(comp_doc_path)
                if content:
                    component_docs_content[node.name] = summarize_markdown(
                        content, budget=2000
                    )

            child_docs_content = {}
            children = module_info.get("children", {})
            for child_name in children.keys():
                child_doc_path = os.path.join(
                    docs_dir, module_id, f"{child_name}.md"
                )
                content = _safe_read(child_doc_path)
                if content:
                    child_docs_content[child_name] = summarize_markdown(
                        content, budget=1500
                    )

            path_parts = module_id.split("/")
            parent_module = path_parts[-2] if len(path_parts) > 1 else ""

            if should_partial_update(existing_doc, ""):
                return await pipeline.agent.execute(
                    {
                        "type": "write_doc",
                        "config": config,
                        "input": {
                            "type": "update_module",
                            "module_name": module_name,
                            "module_path": module_info.get("path", module_id),
                            "parent_module": parent_module,
                            "sibling_modules": [],
                            "module_dependencies": sorted(
                                module_deps.get(module_id, set())
                            ),
                            "dependent_modules": sorted(
                                module_dependents.get(module_id, set())
                            ),
                            "sub_modules": list(children.keys()),
                            "component_docs": component_docs_content,
                            "children_docs": child_docs_content,
                            "existing_doc": existing_doc,
                            "output_path": doc_path,
                        },
                    }
                )
            return await pipeline.agent.execute(
                {
                    "type": "write_doc",
                    "config": config,
                    "input": {
                        "type": "module",
                        "module_name": module_name,
                        "module_path": module_info.get("path", module_id),
                        "parent_module": parent_module,
                        "sibling_modules": [],
                        "module_dependencies": sorted(
                            module_deps.get(module_id, set())
                        ),
                        "dependent_modules": sorted(
                            module_dependents.get(module_id, set())
                        ),
                        "sub_modules": list(children.keys()),
                        "component_docs": component_docs_content,
                        "children_docs": child_docs_content,
                        "output_path": doc_path,
                    },
                }
            )

        module_results = await _run_jobs_with_retry(
            module_jobs, _update_module, "module docs"
        )

    # Overview (layer 3)
    overview_result = None
    if impact["update_overview"]:
        print("\nUpdating README overview...")
        readme_path = os.path.join(docs_dir, "README.md")
        existing_doc = _safe_read(readme_path)
        if existing_doc and not force and _user_edited_doc(
            graph, pipeline, readme_path, existing_doc
        ):
            print(
                f"  SKIP {os.path.relpath(readme_path, docs_dir)}: "
                "manually edited since last update "
                "(use --force to overwrite; retried on next update)"
            )
            _mark_doc_stale(graph, pipeline, readme_path)
        else:
            module_docs_summary = {}
            for name in module_tree.keys():
                content = _safe_read(os.path.join(docs_dir, f"{name}.md"))
                if content:
                    module_docs_summary[name] = summarize_markdown(
                        content, budget=2000
                    )

            overview_input = {
                "type": "update_overview",
                "repo_name": os.path.basename(config.repo_path),
                "component_count": len(pipeline.components),
                "module_count": len(module_tree),
                "module_dependencies": {
                    m: sorted(ds) for m, ds in module_deps.items()
                },
                "module_docs_summary": module_docs_summary,
                "existing_doc": existing_doc,
                "output_path": readme_path,
            }
            try:
                if should_partial_update(existing_doc, ""):
                    overview_result = await pipeline.agent.execute(
                        {"type": "write_doc", "config": config, "input": overview_input}
                    )
                else:
                    overview_result = await pipeline.agent.execute(
                        {
                            "type": "write_doc",
                            "config": config,
                            "input": {
                                "type": "overview",
                                "repo_name": overview_input["repo_name"],
                                "component_count": overview_input["component_count"],
                                "module_count": overview_input["module_count"],
                                "module_dependencies": overview_input["module_dependencies"],
                                "module_docs_summary": overview_input["module_docs_summary"],
                                "output_path": readme_path,
                            },
                        }
                    )
            except Exception as e:
                # Same failure contract as component/module docs: journal it,
                # never abort the whole run over the README.
                print(f"Warning: README update failed: {e}")
                overview_result = False

    # --- Sync graph doc nodes + persist ---
    for job, result in zip(component_jobs, component_results):
        node_id, doc_path, _existing = job
        if result is not None and getattr(result, "success", False) and result.output:
            _sync_doc_node(graph, pipeline, doc_path, result.output.get("content", ""), target_id=node_id)
    for job, result in zip(module_jobs, module_results):
        module_id, _name, _info, doc_path = job
        if result is not None and getattr(result, "success", False) and result.output:
            _sync_doc_node(graph, pipeline, doc_path, result.output.get("content", ""), target_id=module_id)
    if overview_result is not None and getattr(overview_result, "success", False) and overview_result.output:
        _sync_doc_node(graph, pipeline, readme_path, overview_result.output.get("content", ""))

    # --- Link / mermaid validation ---
    print("\nValidating links and mermaid diagrams...")
    link_results = validate_and_fix_links(docs_dir)
    # The validator rewrites files AFTER the snapshots above were taken; adopt
    # those rewrites so the next run does not mistake them for manual edits.
    refreshed = refresh_doc_snapshots(
        graph,
        [os.path.join(docs_dir, rel) for rel in link_results.get("fixed", [])],
    )
    if refreshed:
        print(f"  Refreshed {refreshed} doc snapshot(s) after link/mermaid fixes")
    graph.save(_graph_save_path(config, project_name))

    def is_successful_result(r) -> bool:
        return not isinstance(r, Exception) and bool(r) and getattr(r, "success", False)

    success_count = sum(1 for r in component_results if is_successful_result(r))
    module_success = sum(1 for r in module_results if is_successful_result(r))
    overview_ok = is_successful_result(overview_result)

    # --- Failure recovery: journal what did not make it ---
    failed_component_ids = sorted(
        {
            job[0]
            for job, r in zip(component_jobs, component_results)
            if _job_failed(r)
        }
    )
    failed_module_ids = sorted(
        {
            job[0]
            for job, r in zip(module_jobs, module_results)
            if _job_failed(r)
        }
    )
    overview_failed = bool(impact["update_overview"]) and _job_failed(
        overview_result
    )
    # Flag failed docs stale so their state is visible; the pending journal is
    # the authoritative retry queue (a never-generated doc has no DocNode).
    for job, r in zip(component_jobs, component_results):
        if _job_failed(r):
            _mark_doc_stale(graph, pipeline, job[1])
    for job, r in zip(module_jobs, module_results):
        if _job_failed(r):
            _mark_doc_stale(graph, pipeline, job[3])
    if overview_failed:
        _mark_doc_stale(graph, pipeline, readme_path)

    remaining_pending = {
        "components": [c for c in failed_component_ids if graph.has_node(c)],
        "modules": failed_module_ids,
        "overview": overview_failed,
    }
    _save_pending_updates(config, project_name, remaining_pending)
    pending_after = not _pending_is_empty(remaining_pending)
    if pending_after:
        # Persist the stale flags set above.
        graph.save(_graph_save_path(config, project_name))

    print(
        f"\nSuccessfully updated {success_count}/{len(component_results)} "
        f"component doc(s), {module_success}/{len(module_results)} module doc(s)"
        + (", README" if overview_ok else "")
    )
    if pending_after:
        print(
            f"WARNING: {len(failed_component_ids)} component doc(s) and "
            f"{len(failed_module_ids)} module doc(s) failed"
            + (" (README included)" if overview_failed else "")
            + "; they stay pending and the next `repodoc update` will retry them."
        )

    duration = time.time() - start_time
    status = "partial" if pending_after else "success"

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
        components_processed=len(impact["components_to_doc"]),
        components_updated=success_count,
        status=status,
        metadata={
            "graph_mutation": {
                "added": len(mutation.added),
                "updated": len(mutation.updated),
                "removed": len(mutation.removed),
                "edges_added": len(mutation.added_edges),
                "edges_removed": len(mutation.removed_edges),
            },
            "modules_updated": module_success,
            "overview_updated": overview_ok,
            "docs_removed": removed_docs,
            "docs_migrated": migrated_docs,
            "failed_components": failed_component_ids,
            "failed_modules": failed_module_ids,
            "pending": pending_after,
        },
    )

    if os.path.exists(metadata_path):
        try:
            metadata = file_manager.load_json(metadata_path)
            if metadata:
                if "generation_info" not in metadata:
                    metadata["generation_info"] = {}
                if "update_info" not in metadata:
                    metadata["update_info"] = {}
                # The watermark advances ONLY when no doc is left pending:
                # otherwise the next run must resync from the old base.
                if current_commit and not pending_after:
                    metadata["generation_info"]["git_commit"] = current_commit
                    metadata["update_info"]["last_update_commit"] = current_commit
                metadata["update_info"]["last_attempt_commit"] = current_commit
                metadata["update_info"]["last_base_commit"] = base_commit
                metadata["update_info"]["components_updated"] = success_count
                metadata["update_info"]["affected_components"] = impact[
                    "components_to_doc"
                ]
                metadata["update_info"]["pending_docs"] = (
                    remaining_pending if pending_after else {}
                )
                metadata["generation_info"]["timestamp"] = datetime.now().isoformat()
                file_manager.save_json(metadata, metadata_path)
        except Exception as e:
            print(f"Warning: Could not update metadata: {e}")

    print("\nDocumentation update complete!")
    return {
        "status": status,
        "impact": impact,
        "updated": success_count,
        "total": len(component_results),
        "modules_updated": module_success,
        "removed_docs": removed_docs,
        "pending": pending_after,
    }