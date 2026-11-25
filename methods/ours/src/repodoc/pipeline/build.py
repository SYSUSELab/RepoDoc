"""Full pipeline: code graph -> component tree -> doc tree -> markdown/graph."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repodoc.document.decompose import decompose_large_components
from repodoc.document.generator import generate_documents, write_doc_indexes
from repodoc.document.planner import (
    build_component_tree,
    build_doc_plan,
    write_component_tree_to_graph,
)
from repodoc.graph.builder import CodeGraphBuilder
from repodoc.graph.languages import detect_analyzer
from repodoc.pipeline.manifest import (
    artifact_path,
    create_manifest,
    read_manifest,
    source_hash,
    write_manifest,
)
from repodoc.schema import stable_hash, validate


class StateExistsError(RuntimeError):
    """Raised when a state directory already exists and --force was not used."""


@dataclass
class BuildResult:
    repo_path: Path
    state_dir: Path
    docs_dir: Path
    graph_version: str
    code_graph_version: str
    component_version: str
    doc_version: str
    pages: int
    graph_nodes: int
    graph_edges: int
    cluster_mode: str = "toy"
    llm_usage: dict[str, Any] | None = None
    llm_budget: dict[str, Any] | None = None
    split_components: int = 0


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def build_project(
    repo_path: str | Path,
    state_dir: str | Path,
    *,
    docs_dir: str | Path | None = None,
    force: bool = False,
    repo_name: str | None = None,
    cluster_mode: str = "toy",
    llm_client: Any | None = None,
    max_cluster_nodes: int = 200,
    cluster_chunk_files: int = 80,
    cluster_chunk_chars: int = 80_000,
    cluster_merge_mode: str = "llm",
    progress: Any | None = None,
    llm_max_tokens: int | None = None,
    llm_temperature: float = 0.1,
    llm_model: str | None = None,
    model_limits: Any | None = None,
    cluster_context_ratio: float | None = None,
    cluster_output_ratio: float | None = None,
    max_component_nodes: int = 300,
    max_component_children: int = 8,
    min_component_child_nodes: int = 80,
    component_tree_path: str | Path | None = None,
) -> BuildResult:
    repo = Path(repo_path).resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"repository not found: {repo}")
    if cluster_mode not in {"toy", "llm", "hierarchical", "auto"}:
        raise ValueError("cluster_mode must be 'toy', 'llm', 'hierarchical', or 'auto'")
    if cluster_merge_mode not in {"directory", "llm"}:
        raise ValueError("cluster_merge_mode must be 'directory' or 'llm'")
    state = Path(state_dir).resolve()
    docs = Path(docs_dir).resolve() if docs_dir else state / "docs" / "markdown"
    effective_repo_name = repo_name or repo.name

    if state.exists() and read_manifest(state) is not None and not force:
        raise StateExistsError(
            f"state already exists: {state}; pass --force to rebuild from scratch"
        )
    if state.exists() and force:
        if state == repo or repo.is_relative_to(state):
            raise ValueError("refusing to delete the repository or a parent directory")
        shutil.rmtree(state)
    state.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)

    analyzer = detect_analyzer(repo)
    if progress is not None:
        progress.stage(f"Parsing {analyzer.name} code graph")
    builder = CodeGraphBuilder(
        repo,
        repo_name=effective_repo_name,
        exclude_dirs=[state, docs],
        analyzer=analyzer,
    )
    graph = builder.build()
    code_graph_version = graph.code_graph_version
    if progress is not None:
        progress.info(
            f"Graph ready: {len(graph.nodes_of_type('file'))} files, "
            f"{sum(1 for n in graph.nodes.values() if n['layer'] == 'code')} code nodes"
        )
    if cluster_mode == "auto":
        file_count = len(graph.nodes_of_type("file"))
        cluster_mode = "hierarchical" if file_count > cluster_chunk_files else "llm"

    llm_usage: dict[str, Any] | None = None
    llm_budget: dict[str, Any] | None = None
    if progress is not None:
        progress.stage("Preparing component tree")
    if component_tree_path is not None:
        component_tree = json.loads(
            Path(component_tree_path).expanduser().read_text(encoding="utf-8")
        )
        component_tree["repo"] = effective_repo_name
        component_tree["repo_path"] = str(repo)
        component_tree["commit"] = graph.commit
        component_tree["source_graph_version"] = graph.code_graph_version
        write_component_tree_to_graph(graph, component_tree)
        cluster_mode = "reuse"
    elif cluster_mode == "toy":
        component_tree = build_component_tree(graph, repo_name=effective_repo_name)
    else:
        from repodoc.document.llm_cluster import LLMClusterAgent
        from repodoc.llm import (
            OpenCodeGoClient,
            OpenCodeGoConfig,
            model_limits_or_fallback,
        )
        from repodoc.llm.budget import compute_budget

        if progress is not None:
            progress.stage(f"LLM clustering ({cluster_mode})")
        config = OpenCodeGoConfig.from_env()
        limits = model_limits or model_limits_or_fallback(config)
        if cluster_mode == "hierarchical":
            cluster_context_ratio = (
                cluster_context_ratio if cluster_context_ratio is not None else 0.12
            )
            cluster_output_ratio = (
                cluster_output_ratio if cluster_output_ratio is not None else 0.05
            )
        budget = compute_budget(
            limits,
            "cluster",
            context_ratio=cluster_context_ratio,
            output_ratio=cluster_output_ratio,
        )
        own_client = llm_client is None
        client = llm_client or OpenCodeGoClient(config)
        try:
            agent = LLMClusterAgent(
                client,
                model=llm_model,
                max_nodes=max_cluster_nodes,
                max_tokens=llm_max_tokens,
                temperature=llm_temperature,
                budget=budget,
                progress=progress,
            )
            if cluster_mode == "hierarchical":
                component_tree, trace = agent.cluster_hierarchical(
                    graph,
                    max_chunk_files=cluster_chunk_files,
                    max_chunk_chars=cluster_chunk_chars,
                    merge_mode=cluster_merge_mode,
                )
            else:
                component_tree, trace = agent.cluster(graph)
            llm_usage = (component_tree.get("cluster_meta") or {}).get("usage")
            llm_budget = budget.as_dict()
            _write_json(state / "components" / "llm_cluster_trace.json", trace)
        finally:
            if own_client:
                client.close()
    if progress is not None:
        progress.stage("Splitting oversized components")
    split_components = decompose_large_components(
        graph,
        component_tree,
        max_nodes=max_component_nodes,
        max_children=max_component_children,
        min_child_nodes=min_component_child_nodes,
    )
    component_version = "c_" + stable_hash(component_tree)
    if progress is not None:
        progress.stage("Planning document tree")
    doc_tree = build_doc_plan(component_tree, graph, repo_name=effective_repo_name)
    if progress is not None:
        progress.stage("Rendering deterministic document pages")
    doc_tree, pages, sections, doc_version = generate_documents(
        graph,
        component_tree,
        doc_tree,
        docs_dir=docs,
    )

    if progress is not None:
        progress.stage("Saving graph, document tree and manifest")
    component_path = state / "components" / "component_tree.json"
    doc_tree_path = state / "docs" / "doc_tree.json"
    _write_json(component_path, component_tree)
    _write_json(doc_tree_path, doc_tree)
    pages_path, sections_path = write_doc_indexes(state, pages, sections)
    graph_artifacts = graph.save(state)

    validate("graph", graph.to_dict())
    validate("component_tree", component_tree)
    validate("doc_tree", doc_tree)

    source_hash_value = source_hash(repo, exclude_dirs=[state, docs])
    artifacts = {
        **{
            key: artifact_path(Path(value), state)
            for key, value in graph_artifacts.items()
        },
        "component_tree": artifact_path(component_path, state),
        "doc_tree": artifact_path(doc_tree_path, state),
        "pages": artifact_path(pages_path, state),
        "sections": artifact_path(sections_path, state),
        "markdown": artifact_path(docs, state),
    }
    manifest = create_manifest(
        repo_path=repo,
        state_dir=state,
        source_hash_value=source_hash_value,
        code_graph_version=code_graph_version,
        graph_version=doc_tree["graph_version"],
        component_version=component_version,
        doc_version=doc_version,
        artifacts=artifacts,
        run_id=uuid.uuid4().hex[:12],
        repo_name=effective_repo_name,
    )
    validate("manifest", manifest)
    write_manifest(state, manifest)

    return BuildResult(
        repo_path=repo,
        state_dir=state,
        docs_dir=docs,
        graph_version=manifest["graph_version"],
        code_graph_version=code_graph_version,
        component_version=component_version,
        doc_version=doc_version,
        pages=len(pages),
        graph_nodes=len(graph.nodes),
        graph_edges=len(graph.edges),
        cluster_mode=cluster_mode,
        llm_usage=llm_usage,
        llm_budget=llm_budget,
        split_components=split_components,
    )
