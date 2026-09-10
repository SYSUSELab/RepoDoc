"""Incremental documentation update: git diff -> impact -> regenerate.

This module is the orchestrator; the heavy lifting lives in siblings:
    agent.diff               - change analysis (per-language diffs)
    agent.impact             - affected components via the knowledge graph
    agent.code_extract       - component source at a specific commit
    agent.incremental_merge  - graph/module-tree merge helpers
"""

import asyncio
import logging
import os
import subprocess
import time
from datetime import datetime
from typing import Any, Callable, Optional

from repodoc.agent.code_extract import GitCodeExtractor
from repodoc.agent.diff import (
    ChangeType,
    ComponentChange,
    FileChangeAnalysis,
    analyze_git_diff,
    build_change_map,
    filter_important_changes,
)
from repodoc.agent.generate import (
    DocPipeline,
    _build_module_dependencies,
    _collect_component_modules,
)
from repodoc.agent.impact import IncrementalUpdater
from repodoc.agent.incremental_merge import (
    _extract_new_component_nodes,
    _mark_docs_deprecated,
    _merge_new_nodes_into_graph,
    _refresh_changed_components,
    _remove_components_from_tree,
)
from repodoc.agent.retry import run_with_retry
from repodoc.config import Config
from repodoc.graph.cluster import assign_new_components
from repodoc.graph.graph import HeterogeneousGraph
from repodoc.graph.models import Node
from repodoc.utils import (
    file_manager,
    log_operation,
    project_name as project_name_of,
)

logger = logging.getLogger(__name__)


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
    pipeline.graph = graph  # required: doc<->graph links + tree restore use it
    pipeline.components = graph_to_components(graph)
    pipeline.load_module_tree()
    pipeline.initialize_agent()
    return pipeline



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

    project_name = project_name_of(config.repo_path)
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

    # --- Skip when this exact update was already applied ---
    # Same base commit as the last update AND the code state is unchanged
    # (same HEAD, clean working tree) means re-running regenerates every
    # affected doc again for the identical base->current diff — wasted LLM
    # calls. The working-tree check keeps real (uncommitted) changes safe.
    if (
        base_commit
        and current_commit
        and os.path.exists(metadata_path)
        and not changed_files
    ):
        try:
            meta = file_manager.load_json(metadata_path) or {}
            ui = meta.get("update_info") or {}
            if (
                base_commit == ui.get("last_base_commit")
                and current_commit == ui.get("last_update_commit")
            ):
                dirty = (
                    subprocess.run(
                        ["git", "diff", "--quiet", "HEAD"],
                        cwd=config.repo_path,
                        capture_output=True,
                    ).returncode
                    != 0
                )
                if not dirty:
                    print(
                        f"Already up to date: update {base_commit[:8]} -> "
                        f"{current_commit[:8]} was applied in the last update."
                    )
                    return {"status": "no_changes", "impact": None}
        except Exception:
            pass

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

    # New components: extracted from the working tree (after the pipeline is
    # built — depends_on resolution needs the full component map) and merged
    # into the graph + module tree below (incremental clustering — no full
    # re-cluster).
    new_ids = list(impact.get("new_components") or [])
    new_nodes: dict[str, Node] = {}
    if new_ids:
        print("\n  New components (not in knowledge graph):")
        for comp in new_ids[:10]:
            print(f"    + {comp}")

    if impact["total_affected"] == 0 and not new_ids:
        print("\nNo documentation updates needed")
        return {"status": "no_updates_needed", "impact": impact}

    if confirm and not confirm(
        f"\nThis will update documentation for "
        f"{impact['total_affected'] + len(new_ids)} component(s). Continue?"
    ):
        print("Aborted")
        return {"status": "aborted", "impact": impact}

    # --- Regenerate affected components + merge NEW components into the graph ---
    print("\nRegenerating documentation for affected components...")
    pipeline = _build_pipeline_from_graph(graph, config)
    module_tree = pipeline.module_tree
    docs_dir = config.docs_dir
    code_extractor = GitCodeExtractor(config.repo_path)

    original_top_level = set(module_tree.keys())

    # Refresh changed files' existing components (new deps/source) in the
    # graph, then merge NEW components in and assign them to modules
    # (incremental clustering — existing module boundaries are untouched).
    _refresh_changed_components(
        graph, pipeline.components, config, list(changed_files)
    )
    if new_ids:
        new_nodes = _extract_new_component_nodes(
            change_map, config, pipeline.components
        )
        print(
            f"  Extracted {len(new_nodes)} new component node(s) from working tree"
        )
    components_all = pipeline.components
    assignment: dict[str, str] = {}
    if new_nodes:
        components_all = {**pipeline.components, **new_nodes}
        _merge_new_nodes_into_graph(graph, new_nodes, components_all)
        module_tree, assignment = await asyncio.to_thread(
            assign_new_components, module_tree, new_nodes, config
        )
        pipeline.module_tree = module_tree
        print(
            f"  Assigned {len(assignment)} new component(s) into "
            f"{len(set(assignment.values()))} module(s)"
        )
        from repodoc.graph.builder import attach_module_tree_to_graph

        attach_module_tree_to_graph(graph, module_tree, components_all)
        graph.save(
            os.path.join(
                config.output_dir, "data", f"{project_name}_knowledge_graph.json"
            )
        )

    # Deleted components: mark their docs deprecated (file-level, no LLM),
    # then drop them from the module tree and the graph. Modules that lost a
    # member get their docs regenerated.
    removed_ids = [
        cid
        for cid, ch in (change_map or {}).items()
        if ch.change_type == ChangeType.REMOVED_COMPONENT
    ]
    removed_present = [cid for cid in removed_ids if cid in components_all]
    removed_modules: set[str] = set()
    if removed_present:
        print(f"\n  Components deleted from the code ({len(removed_present)}):")
        for cid in removed_present[:10]:
            print(f"    - {cid}")
        touched = _mark_docs_deprecated(
            docs_dir, module_tree, components_all, removed_present
        )
        if touched:
            print(f"    Marked {len(touched)} doc(s) as deprecated")
        removed_modules = _remove_components_from_tree(
            graph, module_tree, components_all, removed_present
        )
        pipeline.module_tree = module_tree

    comp_modules = _collect_component_modules(module_tree)

    # Affected set: impact callers + new components + existing components the
    # new ones call (their docs should mention the new caller). Deleted
    # components are excluded — their docs were handled above.
    affected = set(impact["affected_components"]) - set(removed_present)
    if new_nodes:
        affected.update(new_nodes.keys())
        for nid, node in new_nodes.items():
            affected.update(dep for dep in node.depends_on if dep in components_all)

    async def regenerate_component(node_id: str):
        node = pipeline.components.get(node_id) or new_nodes.get(node_id)
        if not node:
            return None
        chain = comp_modules.get(node_id)
        if not chain:
            return None

        module_dir = os.path.join(docs_dir, *chain)
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

        predecessors = [
            cid
            for cid, n in components_all.items()
            if node_id in getattr(n, "depends_on", set())
        ]

        component = {
            "name": node.name,
            "component_type": node.component_type,
            "file_path": node.file_path,
            "relative_path": node.relative_path,
            "docstring": docstring,
            "source_code": source_code or "",
            "depends_on": list(node.depends_on) if node.depends_on else [],
        }
        result = await pipeline.agent.execute(
            {
                "type": "write_doc",
                "config": config,
                "input": {
                    "type": "component",
                    "component": component,
                    "predecessors": predecessors,
                    "module_name": ".".join(chain),
                    "output_path": output_path,
                },
            }
        )
        if result is not None and getattr(result, "success", False):
            # B2: docs written during update go into the graph too, so the
            # knowledge graph stays the single source of truth.
            pipeline._add_doc_to_graph(result, target_id=node_id)
        return result

    sem = asyncio.Semaphore(16)

    async def _run(node_id: str):
        async with sem:
            return await regenerate_component(node_id)

    jobs = sorted(affected)
    results, still_failed = await run_with_retry(jobs, _run, "component docs")
    if still_failed:
        print(
            f"WARNING: {len(still_failed)} component docs failed permanently: "
            + ", ".join(still_failed[:5])
            + ("..." if len(still_failed) > 5 else "")
        )

    # --- Module docs for modules whose membership changed ---
    # (gained new components and/or lost deleted ones)
    affected_modules = {
        mid for cid, mid in assignment.items() if cid in new_nodes
    } | removed_modules
    if affected_modules:
        print(
            f"Regenerating {len(affected_modules)} module doc(s) with changed membership..."
        )
        module_deps, module_dependents = _build_module_dependencies(
            components_all, module_tree
        )
        mod_jobs = sorted(affected_modules)

        async def _regenerate_module(mod_id: str):
            parts = mod_id.split("/")
            info = module_tree
            for p in parts:
                info = info[p]
            return await pipeline._generate_single_module_doc(
                parts[-1],
                info,
                mod_id,
                module_tree,
                module_deps,
                module_dependents,
            )

        mod_results, mod_failed = await run_with_retry(
            mod_jobs, _regenerate_module, "module docs"
        )
        if mod_failed:
            print(f"WARNING: {len(mod_failed)} module doc(s) failed: {mod_failed}")

        # Overview: only when the top-level module set changed (avoids churn)
        if set(module_tree.keys()) != original_top_level:
            print("Top-level modules changed; regenerating overview README...")
            overview = await pipeline.generate_overview(
                os.path.join(docs_dir, "README.md")
            )
            if overview is not None and getattr(overview, "success", False):
                pipeline._add_doc_to_graph(overview)

    # Persist the graph: refreshed nodes, new components, updated docs.
    pipeline._save_graph_safe()

    def is_successful_result(r) -> bool:
        return not isinstance(r, Exception) and bool(r) and getattr(r, "success", False)

    success_count = sum(1 for r in results if is_successful_result(r))
    print(
        f"\nSuccessfully regenerated {success_count}/{len(results)} component(s)"
    )
    if new_nodes:
        print(f"  New components merged into the knowledge graph: {len(new_nodes)}")

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
                metadata["update_info"]["new_components"] = list(new_nodes.keys())
                metadata["update_info"]["new_component_modules"] = sorted(
                    set(assignment.values())
                )
                metadata["update_info"]["removed_components"] = removed_present
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
