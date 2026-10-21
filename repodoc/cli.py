import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import logging

from dotenv import load_dotenv

load_dotenv()

from repodoc.config import MAX_SOURCE_FILE_BYTES, Config
from repodoc.graph import DependencyGraphBuilder
from repodoc.graph.cluster import cluster_modules, module_tree_stats
from repodoc.agent.generate import run_pipeline
from repodoc.utils import FileLock, file_manager


def _scope_options(f):
    """Shared analysis/doc-scope options for analyze/docs/generate."""
    f = click.option(
        "-I",
        "--include",
        "include_patterns",
        multiple=True,
        help="Only analyze files matching this glob (repeatable, e.g. 'src/*')",
    )(f)
    f = click.option(
        "-X",
        "--exclude",
        "exclude_patterns",
        multiple=True,
        help="Skip files/dirs matching this glob or name (repeatable)",
    )(f)
    f = click.option(
        "--max-file-size",
        type=int,
        default=MAX_SOURCE_FILE_BYTES,
        show_default=True,
        help="Skip source files larger than this (bytes)",
    )(f)
    f = click.option(
        "--doc-types",
        default="",
        help="Comma-separated component types to document (default: auto)",
    )(f)
    f = click.option(
        "--max-docs",
        type=int,
        default=0,
        show_default=True,
        help="Cap the number of documented components (0 = no cap)",
    )(f)
    return f


def make_config(
    repo_path: str,
    output_dir: str = "output",
    cluster_strategy: str = "llm",
    doc_strategy: str = "agent",
    include_patterns: tuple = (),
    exclude_patterns: tuple = (),
    max_file_size: int | None = None,
    doc_types: str = "",
    max_docs: int = 0,
) -> Config:
    try:
        max_impact_docs = int(os.getenv("REPODOC_MAX_IMPACT_DOCS", "60") or 60)
    except ValueError:
        max_impact_docs = 60
    extra_headers: dict[str, str] = {}
    headers_env = os.getenv("LLM_EXTRA_HEADERS", "").strip()
    if headers_env:
        import json

        try:
            parsed = json.loads(headers_env)
            if isinstance(parsed, dict):
                extra_headers = {str(k): str(v) for k, v in parsed.items()}
        except Exception:
            pass
    return Config(
        repo_path=repo_path,
        output_dir=output_dir,
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        llm_api_key=os.getenv("LLM_API_KEY", "sk-1234"),
        main_model=os.getenv("MAIN_MODEL", "deepseek-chat"),
        cluster_model=os.getenv("CLUSTER_MODEL")
        or os.getenv("MAIN_MODEL", "deepseek-chat"),
        cluster_strategy=cluster_strategy,
        doc_strategy=doc_strategy,
        llm_headers=extra_headers,
        include_patterns=list(include_patterns),
        exclude_patterns=list(exclude_patterns),
        max_file_size=max_file_size or MAX_SOURCE_FILE_BYTES,
        doc_types=[t.strip() for t in doc_types.split(",") if t.strip()],
        max_docs=max_docs,
        max_impact_docs=max_impact_docs,
    )


def get_project_name(repo_path: str) -> str:
    """Extract safe project name from repo path."""
    repo_name = os.path.basename(os.path.normpath(repo_path))
    return "".join(c if c.isalnum() else "_" for c in repo_name)


def _build_and_save_graph(
    components: dict,
    module_tree: dict,
    data_dir: str,
    project_name: str,
) -> None:
    """Build the heterogeneous knowledge graph (code nodes + calls edges +
    module structure) and persist it as the single input for doc generation."""
    from repodoc.graph.builder import (
        attach_module_tree_to_graph,
        build_graph_from_analysis,
    )

    graph = build_graph_from_analysis(components)
    attach_module_tree_to_graph(graph, module_tree, components)
    kg_path = os.path.join(data_dir, f"{project_name}_knowledge_graph.json")
    graph.save(kg_path)


def _rescope_graph(config: Config, kg_path: str, tree_path: str) -> bool:
    """Re-cluster an existing graph under the current doc scope.

    `--max-docs` / `--doc-types` are clustering-scope options; with an
    existing graph they used to be silently ignored. Rebuilding the module
    tree from the in-graph components makes them effective without re-parsing
    the repository. Returns True when the scope actually changed."""
    from repodoc.graph.builder import (
        _build_graph_from_components,
        _get_leaf_nodes,
        attach_module_tree_to_graph,
        graph_to_components,
        graph_to_module_tree,
    )
    from repodoc.graph.graph import HeterogeneousGraph
    from repodoc.agent.generate import _collect_component_modules

    graph = HeterogeneousGraph.load(kg_path)
    components = graph_to_components(graph)
    dep_graph = _build_graph_from_components(components)
    leaf_nodes = _get_leaf_nodes(dep_graph, components, config)
    wanted = {
        cid
        for cid in leaf_nodes
        if cid in components and not components[cid].name.startswith("_")
    }
    existing = set(_collect_component_modules(graph_to_module_tree(graph)))
    if existing == wanted:
        return False

    # Module docs recorded via describes edges: they are regenerated under the
    # new tree, so remove the managed files (only when unedited) to avoid
    # orphan module docs from the old scope.
    from repodoc.graph.models import REL_DESCRIBES, DocNode, ModuleNode
    from repodoc.utils import content_hash

    old_module_docs = {}
    for source, target, relation, _w in graph.edges(REL_DESCRIBES):
        if relation != REL_DESCRIBES:
            continue
        target_node = graph.get_node(target)
        doc = graph.get_node(source)
        if (
            isinstance(target_node, ModuleNode)
            and isinstance(doc, DocNode)
            and doc.file_path
        ):
            old_module_docs[doc.file_path] = doc

    # Drop the previous module layer (module nodes + their edges); code nodes
    # and their edges stay untouched.
    for nid, attrs in list(graph.nodes(data=True)):
        if attrs.get("node_type") == "module":
            graph.remove_node(nid)

    for path, doc in old_module_docs.items():
        if not os.path.exists(path):
            if graph.has_node(doc.id):
                graph.remove_node(doc.id)
            continue
        try:
            content = file_manager.load_text(path)
        except OSError:
            continue
        if doc.content_hash:
            if content_hash(content) != doc.content_hash:
                continue  # user-edited: keep it
        elif (doc.content or "") != content:
            continue
        try:
            os.remove(path)
        except OSError:
            continue
        if graph.has_node(doc.id):
            graph.remove_node(doc.id)
    module_tree = cluster_modules(leaf_nodes, components, config)
    file_manager.save_json(module_tree, tree_path)
    attach_module_tree_to_graph(graph, module_tree, components)
    graph.save(kg_path)
    stats = module_tree_stats(module_tree)
    click.echo(
        f"  Re-scoped documentation: {len(wanted)} components, "
        f"{stats['modules']} modules (max-docs={config.max_docs or '∞'}, "
        f"doc-types={','.join(config.doc_types) or 'auto'})"
    )
    return True


def _record_analysis_snapshot(
    config: Config, data_dir: str, project_name: str
) -> None:
    """Record the analyzed working tree so the next update is a no-op until
    something actually changes (uncommitted changes included)."""
    from repodoc.worktree import get_head_commit, record_worktree_snapshot

    path = os.path.join(data_dir, f"{project_name}_metadata.json")
    metadata = file_manager.load_json(path) or {}
    metadata.setdefault("generation_info", {})
    commit = get_head_commit(config.repo_path)
    if commit:
        metadata["generation_info"]["git_commit"] = commit
    record_worktree_snapshot(metadata, config)
    file_manager.save_json(metadata, path)


def _ensure_knowledge_graph(
    config: Config,
    data_dir: str,
    project_name: str,
    rescope: bool = False,
) -> bool:
    """Prepare the analysis inputs for doc generation.

    Returns True when the knowledge graph already exists (doc generation
    restores components + module tree from it, skipping analysis/clustering).
    Otherwise runs analysis + clustering, persists the module tree and the
    knowledge graph, and returns False.

    `rescope=True` (explicit --max-docs/--doc-types) rebuilds the module tree
    of an existing graph so the options are not silently ignored."""
    kg_path = os.path.join(data_dir, f"{project_name}_knowledge_graph.json")
    if os.path.exists(kg_path):
        if rescope:
            _rescope_graph(
                config,
                kg_path,
                os.path.join(data_dir, f"{project_name}_module_tree.json"),
            )
        return True

    file_manager.ensure_directory(data_dir)
    click.echo("=" * 40)
    click.echo("Phase 1: Analyzing dependencies...")
    builder = DependencyGraphBuilder(config)
    components, leaf_nodes = builder.build_dependency_graph()
    click.echo(f"  Components: {len(components)}")
    click.echo(f"  Leaf nodes: {len(leaf_nodes)}")

    click.echo("=" * 40)
    click.echo("Phase 2: Clustering modules...")
    module_tree = cluster_modules(leaf_nodes, components, config)
    tree_path = os.path.join(data_dir, f"{project_name}_module_tree.json")
    file_manager.save_json(module_tree, tree_path)
    stats = module_tree_stats(module_tree)
    click.echo(f"  Modules: {len(module_tree)} top-level")
    click.echo(
        f"  Cluster stats: {stats['modules']} total, depth {stats['max_depth']}, "
        f"{stats['leaf_components']} leaf components, "
        f"{stats['empty_modules']} empty, largest {stats['largest_module_components']}"
    )

    _build_and_save_graph(components, module_tree, data_dir, project_name)
    click.echo(f"  Knowledge graph saved: {kg_path}")
    click.echo("=" * 40)
    return False


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """RepoDoc: Automated documentation via Repository Knowledge Graph."""
    pass


@cli.command(short_help="Build knowledge graph: dependency graph + module tree")
@click.argument("repo_path", default=".")
@click.option("-o", "--output-dir", default="output", help="Output directory")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option(
    "--no-cluster",
    is_flag=True,
    help="Skip module clustering (dependency graph only, saves LLM tokens)",
)
@click.option(
    "--cluster-strategy",
    type=click.Choice(["llm", "heuristic"], case_sensitive=False),
    default="llm",
    help="Clustering strategy (ablation): llm | heuristic (default: llm)",
)
@_scope_options
def analyze(
    repo_path: str,
    output_dir: str,
    verbose: bool,
    no_cluster: bool,
    cluster_strategy: str,
    include_patterns: tuple,
    exclude_patterns: tuple,
    max_file_size: int,
    doc_types: str,
    max_docs: int,
):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    config = make_config(
        repo_path,
        output_dir,
        cluster_strategy=cluster_strategy,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        max_file_size=max_file_size,
        doc_types=doc_types,
        max_docs=max_docs,
    )
    project_name = get_project_name(repo_path)
    data_dir = os.path.join(output_dir, "data")
    builder = DependencyGraphBuilder(config)
    try:
        components, leaf_nodes = builder.build_dependency_graph()
        click.echo(f"Components: {len(components)}")
        click.echo(f"Leaf nodes: {len(leaf_nodes)}")
        click.echo(f"Dependency graph saved: {config.dependency_graph_dir}")

        # Persist the knowledge graph right after analysis: it is the
        # pipeline's backbone, not a post-doc-generation summary.
        _build_and_save_graph(components, {}, data_dir, project_name)
        click.echo(
            f"Knowledge graph saved: "
            f"{os.path.join(data_dir, project_name + '_knowledge_graph.json')}"
        )

        if not no_cluster:
            file_manager.ensure_directory(data_dir)
            module_tree = cluster_modules(leaf_nodes, components, config)
            tree_path = os.path.join(data_dir, f"{project_name}_module_tree.json")
            file_manager.save_json(module_tree, tree_path)
            stats = module_tree_stats(module_tree)
            click.echo(
                f"Module tree saved: {tree_path} ({len(module_tree)} top-level; "
                f"{stats['modules']} total, depth {stats['max_depth']}, "
                f"{stats['empty_modules']} empty)"
            )
            # Merge the clustering blueprint into the knowledge graph and
            # persist: the graph is the single integrated asset.
            _build_and_save_graph(components, module_tree, data_dir, project_name)
            kg_path = os.path.join(data_dir, f"{project_name}_knowledge_graph.json")
            click.echo(f"Knowledge graph updated with module structure: {kg_path}")
        # Snapshot the analyzed tree (uncommitted changes included): the next
        # `update` compares against it instead of re-processing git state.
        _record_analysis_snapshot(config, data_dir, project_name)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command(short_help="Generate documentation using AI agents")
@click.argument("repo_path", default=".")
@click.option("-o", "--output-dir", default="output", help="Output directory")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option(
    "--cluster-strategy",
    type=click.Choice(["llm", "heuristic"], case_sensitive=False),
    default="llm",
    help="Clustering strategy (ablation): llm | heuristic (default: llm)",
)
@click.option(
    "--doc-strategy",
    type=click.Choice(["agent", "direct"], case_sensitive=False),
    default="agent",
    help="Doc generation strategy (ablation): agent | direct (default: agent)",
)
@_scope_options
def docs(
    repo_path: str,
    output_dir: str,
    verbose: bool,
    cluster_strategy: str,
    doc_strategy: str,
    include_patterns: tuple,
    exclude_patterns: tuple,
    max_file_size: int,
    doc_types: str,
    max_docs: int,
):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    config = make_config(
        repo_path,
        output_dir,
        cluster_strategy=cluster_strategy,
        doc_strategy=doc_strategy,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        max_file_size=max_file_size,
        doc_types=doc_types,
        max_docs=max_docs,
    )
    project_name = get_project_name(repo_path)
    data_dir = os.path.join(output_dir, "data")
    try:
        # The knowledge graph is the single required input: when it exists
        # (produced by `repodoc analyze`), doc generation restores everything
        # from the graph alone and skips re-analysis/clustering entirely.
        # Explicit doc-scope options re-cluster the graph (they would
        # otherwise be silently ignored).
        _ensure_knowledge_graph(
            config,
            data_dir,
            project_name,
            rescope=bool(doc_types or max_docs),
        )

        results = run_pipeline(config, generate_docs=True)
        click.echo(f"Documentation generated: {results['docs_directory']}")
        click.echo(f"Total components: {results['total_components']}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command(short_help="Run all phases: analyze -> cluster -> docs")
@click.argument("repo_path", default=".")
@click.option("-o", "--output-dir", default="output", help="Output directory")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option(
    "--cluster-strategy",
    type=click.Choice(["llm", "heuristic"], case_sensitive=False),
    default="llm",
    help="Clustering strategy (ablation): llm | heuristic (default: llm)",
)
@click.option(
    "--doc-strategy",
    type=click.Choice(["agent", "direct"], case_sensitive=False),
    default="agent",
    help="Doc generation strategy (ablation): agent | direct (default: agent)",
)
@_scope_options
def generate(
    repo_path: str,
    output_dir: str,
    verbose: bool,
    cluster_strategy: str,
    doc_strategy: str,
    include_patterns: tuple,
    exclude_patterns: tuple,
    max_file_size: int,
    doc_types: str,
    max_docs: int,
):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    config = make_config(
        repo_path,
        output_dir,
        cluster_strategy=cluster_strategy,
        doc_strategy=doc_strategy,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        max_file_size=max_file_size,
        doc_types=doc_types,
        max_docs=max_docs,
    )
    project_name = get_project_name(repo_path)
    data_dir = os.path.join(output_dir, "data")
    try:
        # Identical to `analyze` + `docs`: when the knowledge graph already
        # exists this skips analysis/clustering entirely (no wasted LLM calls),
        # unless explicit doc-scope options ask for a different scope.
        graph_existed = _ensure_knowledge_graph(
            config,
            data_dir,
            project_name,
            rescope=bool(doc_types or max_docs),
        )
        if graph_existed:
            click.echo("Knowledge graph found; skipping analysis/clustering.")

        click.echo("=" * 40)
        click.echo("Phase 3: Generating documentation...")
        results = run_pipeline(config, generate_docs=True)

        click.echo("=" * 40)
        click.echo(f"Done! Documentation: {results['docs_directory']}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.group()
def config():
    """Manage RepoDoc configuration."""
    pass


@config.command("show")
def config_show():
    """Show current configuration."""
    click.echo("Current RepoDoc Configuration:")
    click.echo("-" * 40)

    config_data = {
        "LLM Base URL": os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        "LLM API Key": (
            "***" + os.getenv("LLM_API_KEY", "")[-4:]
            if os.getenv("LLM_API_KEY")
            else "not set"
        ),
        "Main Model": os.getenv("MAIN_MODEL", "deepseek-chat"),
        "Cluster Model": os.getenv(
            "CLUSTER_MODEL", os.getenv("MAIN_MODEL", "deepseek-chat")
        ),
    }

    for key, value in config_data.items():
        click.echo(f"  {key}: {value}")


VALID_CONFIG_KEYS = ["LLM_BASE_URL", "LLM_API_KEY", "MAIN_MODEL", "CLUSTER_MODEL"]


@config.command("set", short_help="Set a config value in .env")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str):
    """Set a configuration value in the project .env file.

    Valid keys:

    \b
      LLM_BASE_URL (LLM API base URL)
      LLM_API_KEY (API key)
      MAIN_MODEL (model for documentation)
      CLUSTER_MODEL (model for clustering, defaults to MAIN_MODEL)

    Examples:

    \b
      repodoc config set MAIN_MODEL deepseek-chat
      repodoc config set LLM_BASE_URL http://localhost:4000/
      repodoc config set LLM_API_KEY sk-your-api-key
    """
    key_upper = key.upper()
    if key_upper not in VALID_CONFIG_KEYS:
        click.echo(f"Error: Invalid key '{key}'. Valid keys are:")
        for k in VALID_CONFIG_KEYS:
            click.echo(f"  - {k}")
        raise click.Abort()

    env_path = Path(".env")
    env_vars = {}

    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and "=" in line:
                    k, v = line.split("=", 1)
                    env_vars[k] = v

    env_vars[key_upper] = value

    with open(env_path, "w") as f:
        for k, v in env_vars.items():
            f.write(f"{k}={v}\n")

    click.echo(f"Set {key_upper}={value} in .env")


@config.command("validate")
def config_validate():
    """Validate configuration."""
    click.echo("Validating configuration...")

    errors = []
    warnings = []

    if not os.getenv("LLM_API_KEY"):
        errors.append("LLM_API_KEY is not set")

    if not os.getenv("LLM_BASE_URL"):
        warnings.append("LLM_BASE_URL not set, using default")

    if errors:
        click.echo("Errors:", err=True)
        for e in errors:
            click.echo(f"  - {e}", err=True)
    else:
        click.echo("Configuration is valid!")

    if warnings:
        click.echo("Warnings:")
        for w in warnings:
            click.echo(f"  - {w}")


@cli.command()
@click.argument("repo_path", default=".")
@click.argument("changed_files", nargs=-1, required=False)
@click.option("-o", "--output-dir", default="output", help="Output directory")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option(
    "--base-commit",
    default=None,
    help="Base commit to compare against (auto-detect from metadata if not provided)",
)
@click.option(
    "--analyze-changes/--no-analyze-changes",
    default=True,
    help="Analyze changes to filter unimportant ones (default: enabled)",
)
@click.option(
    "--include-tests",
    is_flag=True,
    help="Include test files in analysis (default: exclude test files)",
)
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    help="Skip confirmation prompt and automatically proceed",
)
@click.option(
    "--propagate",
    type=click.Choice(["sig", "all"], case_sensitive=False),
    default="sig",
    help="Impact propagation: sig (only signature/new/removed changes spread "
    "to callers) | all (every changed component spreads) (default: sig)",
)
@click.option(
    "--max-hops",
    type=int,
    default=3,
    show_default=True,
    help="Max caller-propagation depth in the knowledge graph (0 disables)",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite docs that were manually edited since the last update",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview the update scope (files, docs, estimated LLM calls) "
    "without changing the graph, docs or metadata",
)
def update(
    repo_path: str,
    changed_files: tuple,
    output_dir: str,
    verbose: bool,
    base_commit: str,
    analyze_changes: bool,
    include_tests: bool,
    yes: bool,
    propagate: str,
    max_hops: int,
    force: bool,
    dry_run: bool,
):
    """Update documentation incrementally for changed files.

    REPO_PATH: Path to the repository (default: current directory)
    CHANGED_FILES: List of changed files (optional, auto-detects via git if not provided)

    Example:
        repodoc update /path/to/flask src/flask/app.py src/flask/ctx.py
        repodoc update /path/to/flask --base-commit HEAD~1
        repodoc update /path/to/flask --analyze-changes
        repodoc update /path/to/flask --propagate all --max-hops 5
    """
    import asyncio

    from repodoc.agent.update import run_update

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    config = make_config(repo_path, output_dir)

    async def _run():
        return await run_update(
            config,
            changed_files=changed_files,
            base_commit=base_commit,
            analyze_changes=analyze_changes,
            include_tests=include_tests,
            confirm=None if yes else (lambda msg: click.confirm(msg)),
            propagate=propagate,
            max_hops=max_hops,
            force=force,
            dry_run=dry_run,
        )

    # Single-writer guard: concurrent updates would interleave graph mutation
    # and doc writes (last writer wins, docs and graph diverge).
    project_name = get_project_name(repo_path)
    lock_path = os.path.join(output_dir, "data", f"{project_name}_update.lock")
    file_manager.ensure_directory(os.path.dirname(lock_path))
    try:
        with FileLock(lock_path):
            result = asyncio.run(_run())
    except TimeoutError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
    if result.get("status") == "dry_run":
        impact = result["impact"]
        click.echo(
            f"\nDry run: would update {len(impact['components_to_doc'])} "
            f"component doc(s), {len(impact['modules_to_doc'])} module doc(s)"
            + (", and the README" if impact["update_overview"] else "")
        )
        click.echo(f"Estimated LLM calls: {result['estimated_llm_calls']}")
        click.echo("No changes were made.")
    if result.get("status") == "error":
        raise click.Abort()


if __name__ == "__main__":
    cli()
