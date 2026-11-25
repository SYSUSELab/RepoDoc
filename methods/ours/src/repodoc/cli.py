"""`repodoc` command line interface for the knowledge-graph prototype."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from repodoc import __version__
from repodoc.graph.store import GraphStore
from repodoc.llm import (
    LLMError,
    ledger,
    OpenCodeGoClient,
    OpenCodeGoConfig,
    compute_budget,
    mask_key,
    model_limits_or_fallback,
)
from repodoc.pipeline import (
    StateExistsError,
    apply_update,
    build_project,
    plan_update,
    status_info,
)
from repodoc.pipeline.manifest import read_manifest
from repodoc.progress import Progress
from repodoc.usage import collect_usage, write_usage_metadata


def _default_state(repo: Path) -> Path:
    return Path.cwd() / ".repodoc" / repo.name


def _state_from_args(args: argparse.Namespace) -> Path:
    repo = Path(args.repo).expanduser().resolve()
    if args.state_dir:
        return Path(args.state_dir).expanduser().resolve()
    return _default_state(repo)


def _doc_page_entries(state: Path) -> list[tuple[int, str]]:
    path = state / "docs" / "doc_tree.json"
    if not path.is_file():
        return []
    tree = json.loads(path.read_text(encoding="utf-8"))
    pages: list[tuple[int, str]] = []

    def visit(page: dict[str, Any]) -> None:
        pages.append((int(page["level"]), str(page["path"])))
        for child in page.get("children", []):
            visit(child)

    visit(tree["root"])
    return sorted(pages, key=lambda item: (item[0], item[1]))


def _doc_page_paths(state: Path) -> list[str]:
    return [path for _level, path in _doc_page_entries(state)]


def _page_levels(state: Path) -> dict[str, int]:
    return {path: level for level, path in _doc_page_entries(state)}


def _completed_page_stems(state: Path) -> set[str]:
    completed: set[str] = set()
    trace_dir = state / "docs" / "llm_traces"
    if not trace_dir.is_dir():
        return completed
    for path in trace_dir.glob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not row.get("failed") and row.get("content"):
            completed.add(path.stem)
    return completed


def _generate_llm_batches(
    state: Path,
    page_paths: list[str],
    args: argparse.Namespace,
    progress: Progress,
) -> list[dict[str, Any]]:
    from repodoc.document.llm_docs import generate_llm_pages

    failed: list[dict[str, Any]] = []
    completed = _completed_page_stems(state)

    def run(batch: list[str]) -> None:
        if not batch:
            return
        try:
            result = generate_llm_pages(
                state,
                batch,
                max_pages=len(batch),
                output_ratio=args.docs_output_ratio,
                context_ratio=args.docs_context_ratio,
                model=args.llm_model,
                max_tokens=args.llm_max_tokens,
                progress=progress,
            )
            for item in result["pages"]:
                progress.info(f"{item['page']}: {item.get('sections', '?')} sections")
            return
        except Exception as exc:  # noqa: BLE001 - split and preserve successes
            completed_now = _completed_page_stems(state)
            completed.update(completed_now)
            remaining = [path for path in batch if Path(path).stem not in completed]
            if len(remaining) > 1:
                progress.info(
                    f"Batch failed; splitting {len(remaining)} remaining pages: {exc}"
                )
                middle = len(remaining) // 2
                run(remaining[:middle])
                run(remaining[middle:])
                return
            failed.append({"pages": remaining or batch, "error": str(exc)})
            print(f"  FAILED page: {exc}", file=sys.stderr)

    workers = max(1, int(getattr(args, "workers", 1) or 1))
    if workers > 1:
        levels = _page_levels(state)
        by_level: dict[int, list[str]] = {}
        for path in page_paths:
            if Path(path).stem in completed:
                continue
            by_level.setdefault(levels.get(path, 0), []).append(path)
        serial_pages: list[str] = []
        for level in sorted(by_level):
            paths = by_level[level]
            for start in range(0, len(paths), max(1, args.batch_size)):
                batch = paths[start : start + max(1, args.batch_size)]
                if len(batch) == 1:
                    serial_pages.append(batch[0])
                    continue
                progress.stage(
                    f"Parallel docs batch level={level} "
                    f"{start + 1}-{start + len(batch)}/{len(paths)}"
                )
                temp_root = Path(
                    tempfile.mkdtemp(prefix="repodoc-doc-workers-", dir=state.parent)
                )
                jobs: list[tuple[str, Path]] = []
                try:
                    for index, page in enumerate(batch):
                        worker_state = temp_root / f"worker-{index}"
                        shutil.copytree(state, worker_state)
                        jobs.append((page, worker_state))

                    def worker(item: tuple[str, Path]) -> tuple[str, Path, str | None]:
                        page, worker_state = item
                        try:
                            generate_llm_pages(
                                worker_state,
                                [page],
                                max_pages=1,
                                output_ratio=args.docs_output_ratio,
                                context_ratio=args.docs_context_ratio,
                                model=args.llm_model,
                                max_tokens=args.llm_max_tokens,
                            )
                            return page, worker_state, None
                        except Exception as exc:  # noqa: BLE001
                            return page, worker_state, str(exc)

                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        futures = [pool.submit(worker, item) for item in jobs]
                        for future in as_completed(futures):
                            page, worker_state, error = future.result()
                            trace = (
                                worker_state
                                / "docs"
                                / "llm_traces"
                                / f"{Path(page).stem}.json"
                            )
                            if trace.is_file():
                                target = state / "docs" / "llm_traces" / trace.name
                                target.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(trace, target)
                            if error is not None:
                                failed.append({"pages": [page], "error": error})
                                progress.info(f"FAILED page {page}: {error}")
                                continue
                            try:
                                generate_llm_pages(
                                    state,
                                    [page],
                                    max_pages=1,
                                    reuse_traces=True,
                                    progress=progress,
                                )
                            except Exception as exc:  # noqa: BLE001
                                failed.append({"pages": [page], "error": str(exc)})
                                progress.info(f"FAILED apply {page}: {exc}")
                finally:
                    shutil.rmtree(temp_root, ignore_errors=True)
        for page in serial_pages:
            run([page])
        return failed

    for start in range(0, len(page_paths), max(1, args.batch_size)):
        batch = page_paths[start : start + max(1, args.batch_size)]
        progress.stage(
            f"Generating docs batch {start + 1}-{start + len(batch)}/{len(page_paths)}"
        )
        run(batch)
    return failed


def _write_usage_metadata(state: Path) -> Path:
    manifest = read_manifest(state) or {}
    markdown = (manifest.get("artifacts") or {}).get("markdown", "docs/markdown")
    docs_dir = Path(markdown)
    if not docs_dir.is_absolute():
        docs_dir = state / docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    path = docs_dir / "metadata.json"
    write_usage_metadata(state, path)
    metadata = collect_usage(state)
    totals = (metadata.get("usage") or {}).get("totals") or {}
    print(
        "Usage: "
        f"tokens={totals.get('total_tokens', 'unknown')} "
        f"cost_usd={totals.get('cost_usd', 'unknown')} "
        f"status={(metadata.get('usage') or {}).get('cost_status', 'unknown')} "
        f"-> {path}"
    )
    return path


def cmd_generate(args: argparse.Namespace) -> int:
    repo = Path(args.repo).expanduser().resolve()
    state = _state_from_args(args)
    if not repo.is_dir():
        raise FileNotFoundError(f"repository not found: {repo}")

    ledger.reset()
    progress = Progress()
    if args.mode == "update":
        if read_manifest(state) is None:
            raise RuntimeError(f"update requires an existing base state: {state}")
        progress.stage("Planning and applying incremental update")
        result = apply_update(
            repo,
            state,
            use_planner=not args.no_llm,
            direct_only=args.direct_only,
            llm_model=args.llm_model,
            llm_max_tokens=args.llm_max_tokens,
        )
        if result["status"] == "noop":
            print(f"Up to date: {result['reason']}")
            _write_usage_metadata(state)
            return 0
        pages = list(result.get("regenerated_pages") or [])
        progress.info(f"Update applied: {result.get('path', '')}")
        progress.info(f"Selected pages: {pages}")
        if args.no_llm or not pages:
            _write_usage_metadata(state)
            return 0
        failed = _generate_llm_batches(state, pages, args, progress)
        if failed:
            _write_usage_metadata(state)
            print(f"{len(failed)} LLM batch(es) failed", file=sys.stderr)
            return 1
        print("Update documentation complete.")
        _write_usage_metadata(state)
        return 0

    manifest = read_manifest(state)
    if manifest is None or args.force:
        progress.stage("Building full documentation state")
        build_project(
            repo,
            state,
            force=args.force,
            repo_name=args.repo_name,
            cluster_mode=args.cluster_mode,
            cluster_chunk_files=args.cluster_chunk_files,
            cluster_chunk_chars=args.cluster_chunk_chars,
            cluster_merge_mode=args.cluster_merge_mode,
            llm_model=args.llm_model,
            llm_max_tokens=args.llm_max_tokens,
            cluster_context_ratio=args.cluster_context_ratio,
            cluster_output_ratio=args.cluster_output_ratio,
            progress=progress,
        )
    else:
        progress.info(f"Reusing state: {state}")

    pages = _doc_page_paths(state)
    if args.no_llm:
        progress.info(f"Document tree ready: {len(pages)} pages (LLM disabled).")
        _write_usage_metadata(state)
        return 0
    completed = _completed_page_stems(state)
    pages = [path for path in pages if Path(path).stem not in completed]
    if not pages:
        progress.info("All pages already have successful LLM traces.")
        _write_usage_metadata(state)
        return 0
    progress.stage(f"Generating LLM documentation for {len(pages)} pages")
    failed = _generate_llm_batches(state, pages, args, progress)
    if failed:
        report = state / "docs" / "failed_pages.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(failed, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _write_usage_metadata(state)
        print(
            f"{len(failed)} LLM page(s) failed; see {report}",
            file=sys.stderr,
        )
        return 1
    print("Documentation generation complete.")
    _write_usage_metadata(state)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    repo = Path(args.repo).expanduser().resolve()
    state = _state_from_args(args)
    ledger.reset()
    progress = Progress()
    docs = Path(args.docs_dir).expanduser().resolve() if args.docs_dir else None
    if args.cluster_mode == "llm":
        print(
            f"LLM clustering enabled: one request, max_nodes={args.max_cluster_nodes}, "
            f"max_tokens={args.llm_max_tokens or '<dynamic from model limits>'}, "
            f"model={args.llm_model or '<from env>'}"
        )
    result = build_project(
        repo,
        state,
        docs_dir=docs,
        force=args.force,
        repo_name=args.repo_name,
        cluster_mode=args.cluster_mode,
        max_cluster_nodes=args.max_cluster_nodes,
        cluster_chunk_files=args.cluster_chunk_files,
        cluster_chunk_chars=args.cluster_chunk_chars,
        cluster_merge_mode=args.cluster_merge_mode,
        llm_max_tokens=args.llm_max_tokens,
        llm_temperature=args.llm_temperature,
        llm_model=args.llm_model,
        cluster_context_ratio=args.cluster_context_ratio,
        cluster_output_ratio=args.cluster_output_ratio,
        max_component_nodes=args.max_component_nodes,
        max_component_children=args.max_component_children,
        min_component_child_nodes=args.min_component_child_nodes,
        component_tree_path=args.component_tree,
        progress=progress,
    )
    print(f"Built state: {result.state_dir}")
    print(f"  cluster mode:      {result.cluster_mode}")
    print(f"  split components:  {result.split_components}")
    print(f"  docs:              {result.docs_dir}")
    print(f"  pages:             {result.pages}")
    print(f"  graph nodes/edges: {result.graph_nodes}/{result.graph_edges}")
    print(f"  code graph:        {result.code_graph_version}")
    print(f"  graph:             {result.graph_version}")
    print(f"  components:        {result.component_version}")
    print(f"  docs:              {result.doc_version}")
    if result.llm_usage:
        print(
            f"  llm usage:         {json.dumps(result.llm_usage, ensure_ascii=False)}"
        )
    _write_usage_metadata(state)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    repo = Path(args.repo).expanduser().resolve()
    state = _state_from_args(args)
    info = status_info(repo, state, exclude_dirs=[state])
    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False))
    else:
        print(f"repo:        {repo}")
        print(f"state:       {state}")
        print(f"status:      {info['state']}")
        print(
            f"update:      {'yes' if info['needs_update'] else 'no'} ({info['reason']})"
        )
        manifest = info.get("manifest")
        if manifest:
            print(f"last commit: {manifest.get('last_commit')}")
            print(f"graph:       {manifest.get('graph_version')}")
            print(f"docs:        {manifest.get('doc_version')}")
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    repo = Path(args.repo).expanduser().resolve()
    state = _state_from_args(args)
    ledger.reset()
    if args.dry_run:
        result = plan_update(repo, state, write=True, direct_only=args.direct_only)
        if result["status"] == "noop":
            print(f"Up to date: {result['reason']}")
            return 0
        change_set = result["change_set"] or {}
        impact = change_set.get("impact") or {}
        print(f"Change detected: {result['reason']}")
        print(f"Planned pages: {impact.get('affected_pages', [])}")
        print(f"Change set: {result['path']}")
        return 0

    result = apply_update(
        repo,
        state,
        use_llm=args.llm_update,
        use_patch=args.llm_patch,
        use_planner=args.llm_plan,
        direct_only=args.direct_only,
        llm_model=args.llm_model,
        llm_max_tokens=args.llm_max_tokens,
    )
    if result["status"] == "noop":
        print(f"Up to date: {result['reason']}")
        _write_usage_metadata(state)
        return 0
    print(f"Applied change set: {result['path']}")
    print(f"Regenerated pages: {result['regenerated_pages']}")
    print(f"Unassigned new nodes: {len(result['unassigned_nodes'])}")
    if result.get("llm_decisions"):
        print(f"LLM triage: {json.dumps(result['llm_decisions'], ensure_ascii=False)}")
    if result.get("patched_pages"):
        print(f"LLM patched pages: {result['patched_pages']}")
    if result.get("patch_fallbacks"):
        print(
            f"Patch fallbacks: {json.dumps(result['patch_fallbacks'], ensure_ascii=False)}"
        )
    print(f"graph: {result['graph_version']}")
    print(f"docs:  {result['doc_version']}")
    _write_usage_metadata(state)
    return 0


def cmd_llm_docs(args: argparse.Namespace) -> int:
    from repodoc.document.llm_docs import generate_llm_pages

    state = Path(args.state_dir).expanduser().resolve()
    ledger.reset()
    progress = Progress()
    pages = (
        []
        if args.all_pages
        else [item.strip() for item in args.pages.split(",") if item.strip()]
    )
    result = generate_llm_pages(
        state,
        pages,
        all_pages=args.all_pages,
        max_pages=args.max_pages,
        max_source_chars=args.max_source_chars,
        max_tokens=args.llm_max_tokens,
        temperature=args.llm_temperature,
        model=args.llm_model,
        dry_run=args.dry_run,
        reuse_traces=args.reuse_traces,
        context_ratio=args.docs_context_ratio,
        output_ratio=args.docs_output_ratio,
        progress=progress,
    )
    if result["dry_run"]:
        print("Dry run: no API request sent")
        print(f"budget: {json.dumps(result.get('budget'), ensure_ascii=False)}")
        for item in result["pages"]:
            print(
                f"  {item['page']}: prompt_chars={item['prompt_chars']} "
                f"context_chars={item['context_chars']}"
            )
        return 0
    print(f"Updated pages: {[item['page'] for item in result['pages']]}")
    for item in result["pages"]:
        print(
            f"  {item['page']}: sections={item['sections']} "
            f"prompt_chars={item['prompt_chars']} trace={item['trace']}"
        )
        usage = item.get("usage") or {}
        if usage:
            print(f"    usage: {json.dumps(usage, ensure_ascii=False)}")
    print(f"budget: {json.dumps(result.get('budget'), ensure_ascii=False)}")
    print(f"graph: {result['graph_version']}")
    print(f"docs:  {result['doc_version']}")
    _write_usage_metadata(state)
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    from repodoc.web.server import run_server

    run_server(
        host=args.host,
        port=args.port,
        data_dir=args.data_dir,
    )
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    state = Path(args.state_dir).expanduser().resolve()
    if args.out:
        path = write_usage_metadata(state, args.out, scope=args.run_scope)
        print(f"Wrote usage metadata: {path}")
    else:
        print(
            json.dumps(
                collect_usage(state, scope=args.run_scope),
                indent=2,
                ensure_ascii=False,
            )
        )
    return 0


def cmd_llm_check(args: argparse.Namespace) -> int:
    config = OpenCodeGoConfig.from_env()
    print("OpenCode Go configuration")
    print(f"  base_url:   {config.base_url}")
    print(f"  model:      {config.model}")
    print(f"  session:    {config.session_id}")
    print(f"  user_agent: {config.user_agent}")
    print(f"  api_key:    {mask_key(config.api_key)}")
    print(f"  timeout:    {config.timeout}s")
    limits = model_limits_or_fallback(config)
    print(
        f"  model limits: context={limits.context:,} output={limits.output:,} "
        f"({limits.source})"
    )
    for task in ("cluster", "docs"):
        budget = compute_budget(limits, task)
        print(
            f"  {task} budget: input_tokens={budget.input_tokens:,} "
            f"output_tokens={budget.output_tokens:,} context_chars={budget.context_chars:,}"
        )
    if not config.api_key:
        print(
            "\nOPENCODE_GO_API_KEY is not set. Export it or put it in .env:\n"
            "  OPENCODE_GO_API_KEY=oc_sk_...",
            file=sys.stderr,
        )
        return 2
    if args.no_call:
        print("\nConfiguration looks complete; skipped the test request (--no-call).")
        return 0
    try:
        with OpenCodeGoClient(config) as client:
            response = client.chat(
                [{"role": "user", "content": args.prompt}],
                max_tokens=args.max_tokens,
            )
        message = response.get("choices", [{}])[0].get("message", {})
        print("\nTest response:")
        print(message.get("content") or "<empty content>")
        if message.get("reasoning_content"):
            print(
                "\nReasoning content present:",
                str(message["reasoning_content"])[:500],
            )
        usage = response.get("usage")
        if usage:
            print("\nUsage:")
            print(json.dumps(usage, indent=2, ensure_ascii=False))
        return 0
    except LLMError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        if exc.body:
            print(exc.body, file=sys.stderr)
        return 2


def _print_component(component: dict[str, Any], indent: int = 0) -> None:
    print(
        f"{'  ' * indent}- {component['name']} "
        f"({component['kind']}, depth={component['depth']}, members={len(component['members'])})"
    )
    for child in component.get("children", []):
        _print_component(child, indent + 1)


def _print_doc_page(page: dict[str, Any], indent: int = 0) -> None:
    print(
        f"{'  ' * indent}- {page['path']} "
        f"({page['doc_type']}, L{page['level']}, sections={len(page.get('sections', []))})"
    )
    for child in page.get("children", []):
        _print_doc_page(child, indent + 1)


def cmd_inspect(args: argparse.Namespace) -> int:
    state = Path(args.state_dir).expanduser().resolve()
    if args.what == "graph":
        graph = GraphStore.load(state)
        data = graph.summary()
        if args.json:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(f"graph version: {data['graph_version']}")
            print(f"code graph:    {data['code_graph_version']}")
            print(f"nodes/edges:   {data['nodes']}/{data['edges']}")
            print(f"node types:    {data['node_types']}")
            print(f"edge types:    {data['edge_types']}")
        return 0

    if args.what == "components":
        path = state / "components" / "component_tree.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if args.json:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            _print_component(data["root"])
        return 0

    if args.what == "docs":
        path = state / "docs" / "doc_tree.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if args.json:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            _print_doc_page(data["root"])
        return 0

    raise ValueError(f"unknown inspect target: {args.what}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repodoc",
        description="Repository knowledge graph and hierarchical document prototype.",
        epilog=(
            "Common: generate, web, stats, llm-check. "
            "Advanced: build, update, llm-docs, inspect."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "build", help="build graph, component tree, doc tree and markdown"
    )
    p.add_argument("--repo", required=True, help="repository path")
    p.add_argument(
        "--repo-name", help="stable repository name (default: checkout basename)"
    )
    p.add_argument("--state-dir", help="state directory (default: ./.repodoc/<repo>)")
    p.add_argument(
        "--docs-dir", help="markdown output directory (default: <state>/docs/markdown)"
    )
    p.add_argument(
        "--cluster-mode",
        choices=["toy", "llm", "hierarchical", "auto"],
        default="toy",
        help="toy deterministic grouping, one LLM request, or bounded hierarchical LLM clustering",
    )
    p.add_argument(
        "--max-cluster-nodes",
        type=int,
        default=200,
        help="refuse one-shot LLM clustering above this many file nodes (default: 200)",
    )
    p.add_argument(
        "--cluster-chunk-files",
        type=int,
        default=80,
        help="max files per hierarchical clustering chunk (default: 80)",
    )
    p.add_argument(
        "--cluster-chunk-chars",
        type=int,
        default=80_000,
        help="max prompt chars per hierarchical clustering chunk (default: 80000)",
    )
    p.add_argument(
        "--cluster-merge-mode",
        choices=["directory", "llm"],
        default="llm",
        help="merge hierarchical chunks by directory prefix or a final LLM merge (default: llm)",
    )
    p.add_argument(
        "--llm-max-tokens",
        type=int,
        default=None,
        help="override output tokens; default is dynamic from model limits",
    )
    p.add_argument(
        "--cluster-context-ratio",
        type=float,
        default=None,
        help="override cluster context ratio of model context",
    )
    p.add_argument(
        "--cluster-output-ratio",
        type=float,
        default=None,
        help="override cluster output ratio of model max output",
    )
    p.add_argument(
        "--llm-temperature",
        type=float,
        default=0.1,
        help="temperature for the LLM clustering request (default: 0.1)",
    )
    p.add_argument("--llm-model", help="override OPENCODE_GO_MODEL for this run")
    p.add_argument(
        "--component-tree",
        help="reuse an existing component_tree.json instead of running clustering",
    )
    p.add_argument(
        "--max-component-nodes",
        type=int,
        default=300,
        help="split components larger than this into child components/pages (default: 300)",
    )
    p.add_argument(
        "--max-component-children",
        type=int,
        default=8,
        help="target number of children for an oversized component (default: 8)",
    )
    p.add_argument(
        "--min-component-child-nodes",
        type=int,
        default=80,
        help="minimum node count for a decomposed child (default: 80)",
    )
    p.add_argument(
        "--force", action="store_true", help="delete and rebuild an existing state"
    )
    p.set_defaults(func=cmd_build)

    p = sub.add_parser(
        "generate",
        help="one-step graph/clustering/LLM documentation generation (or update)",
    )
    p.add_argument("--repo", required=True, help="repository path")
    p.add_argument("--state-dir", help="state directory (default: ./.repodoc/<repo>)")
    p.add_argument("--repo-name", help="stable repository name")
    p.add_argument(
        "--mode",
        choices=["full", "update"],
        default="full",
        help="full generation (default) or update an existing state",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="LLM pages per batch (default: 8)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="parallel LLM workers per same-level batch (default: 4)",
    )
    p.add_argument(
        "--cluster-mode",
        choices=["auto", "llm", "hierarchical", "toy"],
        default="auto",
        help="auto chooses one-shot for small repos and hierarchical for large repos",
    )
    p.add_argument("--cluster-chunk-files", type=int, default=80)
    p.add_argument("--cluster-chunk-chars", type=int, default=80_000)
    p.add_argument(
        "--cluster-merge-mode",
        choices=["directory", "llm"],
        default="llm",
    )
    p.add_argument("--docs-output-ratio", type=float, default=0.1)
    p.add_argument("--docs-context-ratio", type=float)
    p.add_argument("--cluster-context-ratio", type=float)
    p.add_argument("--cluster-output-ratio", type=float)
    p.add_argument("--llm-model")
    p.add_argument("--llm-max-tokens", type=int)
    p.add_argument("--direct-only", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="skip LLM document/update planning; clustering may still use LLM",
    )
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("status", help="show state and whether an update is needed")
    p.add_argument("--repo", required=True, help="repository path")
    p.add_argument("--state-dir", help="state directory (default: ./.repodoc/<repo>)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser(
        "update", help="detect git changes and incrementally update graph/docs"
    )
    p.add_argument("--repo", required=True, help="repository path")
    p.add_argument("--state-dir", help="state directory (default: ./.repodoc/<repo>)")
    p.add_argument("--dry-run", action="store_true", help="only plan the change set")
    p.add_argument(
        "--llm-update",
        action="store_true",
        help="use one LLM triage request to choose affected pages",
    )
    p.add_argument(
        "--llm-patch",
        action="store_true",
        help="use section-level LLM content patches for affected pages",
    )
    p.add_argument(
        "--llm-plan",
        action="store_true",
        help="use an LLM planner to choose page actions and parent/overview escalation",
    )
    p.add_argument(
        "--direct-only",
        action="store_true",
        help="update only pages that directly cite changed code; expand to parents only for new/deleted files",
    )
    p.add_argument("--llm-model", help="override OPENCODE_GO_MODEL for triage")
    p.add_argument(
        "--llm-max-tokens", type=int, help="override triage output token cap"
    )
    p.set_defaults(func=cmd_update)

    p = sub.add_parser(
        "llm-docs",
        help="re-generate selected doc pages with LLM (bounded, no automatic retry)",
    )
    p.add_argument("--state-dir", required=True)
    page_group = p.add_mutually_exclusive_group(required=True)
    page_group.add_argument(
        "--pages",
        help="comma-separated page paths/titles, e.g. app.md,overview",
    )
    page_group.add_argument(
        "--all-pages",
        action="store_true",
        help="regenerate every page in the doc tree",
    )
    p.add_argument("--max-pages", type=int, default=4)
    p.add_argument(
        "--max-source-chars",
        type=int,
        default=None,
        help="override context chars; default is dynamic from model limits",
    )
    p.add_argument(
        "--llm-max-tokens",
        type=int,
        default=None,
        help="override output tokens; default is dynamic from model limits",
    )
    p.add_argument("--llm-temperature", type=float, default=0.2)
    p.add_argument("--llm-model")
    p.add_argument(
        "--docs-context-ratio",
        type=float,
        default=None,
        help="override docs context ratio of model context",
    )
    p.add_argument(
        "--docs-output-ratio",
        type=float,
        default=None,
        help="override docs output ratio of model max output",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--reuse-traces",
        action="store_true",
        help="re-apply saved llm_traces content without calling the API",
    )
    p.set_defaults(func=cmd_llm_docs)

    p = sub.add_parser(
        "llm-check", help="show OpenCode Go config and optionally send a test request"
    )
    p.add_argument("--no-call", action="store_true", help="only print configuration")
    p.add_argument("--prompt", default="Say hi in one short sentence.")
    p.add_argument("--max-tokens", type=int, default=64)
    p.set_defaults(func=cmd_llm_check)

    p = sub.add_parser("web", help="serve the local RepoDoc web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--data-dir",
        help="web data directory (default: <project>/web-data)",
    )
    p.set_defaults(func=cmd_web)

    p = sub.add_parser(
        "stats", help="summarize token, cost and LLM-time usage for a state"
    )
    p.add_argument("--state-dir", required=True, help="state directory")
    p.add_argument(
        "--out",
        help="write CodeWiki-compatible metadata.json (default: print JSON)",
    )
    p.add_argument(
        "--run-scope",
        choices=["state", "latest-run"],
        default="state",
        help=(
            "state: every call retained in this state; latest-run: newest "
            "per-command ledger (use for incremental update cost)"
        ),
    )
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("inspect", help="inspect graph/components/docs state")
    p.add_argument("--state-dir", required=True, help="state directory")
    p.add_argument("--what", choices=["graph", "components", "docs"], default="graph")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_inspect)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (StateExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
