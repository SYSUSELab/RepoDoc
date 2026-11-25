"""Run generation, parsing and judging for one system x repository pair."""

from __future__ import annotations

import os
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from . import benchmark, experiments
from .config import Config
from .metrics import (
    docs_stats,
    extract_results_metrics,
    generation_stats,
    normalize_tokens,
)
from .util import (
    HarnessError,
    file_sha256,
    fingerprint_markdown,
    git_head,
    now_iso,
    read_json,
    run_logged,
    safe_component,
    stamp,
    write_json,
)


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def docs(self) -> Path:
        return self.root / "docs"

    @property
    def parsed(self) -> Path:
        return self.root / "parsed"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def meta(self) -> Path:
        return self.root / "meta.json"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics.json"


def run_paths(cfg: Config, experiment: str, system: str, repo: str) -> RunPaths:
    experiments.validate_experiment_name(experiment)
    return RunPaths(
        cfg.runs_dir
        / safe_component(experiment)
        / "systems"
        / safe_component(system)
        / safe_component(repo)
    )


def _ensure_not_inside(path: Path, other: Path) -> None:
    try:
        path.resolve().relative_to(other.resolve())
    except ValueError:
        return
    raise HarnessError(f"refusing to use {path} because it is inside {other}")


def _backup_dir(path: Path) -> Path:
    target = path.with_name(f"{path.name}.prev-{stamp()}")
    counter = 1
    while target.exists():
        target = path.with_name(f"{path.name}.prev-{stamp()}-{counter}")
        counter += 1
    path.rename(target)
    return target


def _load_meta(
    paths: RunPaths, cfg: Config, experiment: str, system: str, repo: str
) -> dict[str, Any]:
    meta = read_json(paths.meta)
    if not isinstance(meta, dict):
        meta = {
            "experiment": experiment,
            "system": system,
            "repo": repo,
            "status": "new",
            "created_at": now_iso(),
            "stages": {},
            "versions": {},
            "config": {},
        }
    for key in ("stages", "versions", "config"):
        if not isinstance(meta.get(key), dict):
            meta[key] = {}
    return meta


def _save_meta(paths: RunPaths, meta: dict[str, Any]) -> None:
    meta["updated_at"] = now_iso()
    write_json(paths.meta, meta)


def _stage_start(meta: dict[str, Any], name: str) -> None:
    meta["stages"][name] = {"status": "running", "started_at": now_iso()}


def _stage_finish(
    meta: dict[str, Any], name: str, result: dict[str, Any], status: str = "ok"
) -> None:
    stage = meta["stages"].setdefault(name, {})
    stage.update(result)
    stage["status"] = status
    stage["finished_at"] = now_iso()


def _system_provenance(
    cfg: Config, spec: dict[str, Any], checkout: Path
) -> dict[str, Any]:
    source = spec.get("source_path")
    adapter = spec.get("adapter_path")
    return {
        "system": spec["name"],
        "system_kind": spec.get("kind"),
        "system_source": str(source) if source else None,
        "system_commit": git_head(source) if source and source.is_dir() else None,
        "adapter": str(adapter) if adapter else None,
        "adapter_sha256": file_sha256(adapter)
        if adapter and adapter.is_file()
        else None,
        "benchmark_commit": git_head(cfg.benchmark_source),
        "repo_checkout": str(checkout),
        "repo_commit": git_head(checkout),
    }


def resolve_checkout(
    cfg: Config,
    repo: str,
    ref: str = "base",
    explicit: str | None = None,
) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise HarnessError(f"checkout path does not exist: {path}")
        return path

    if ref == "update" and not cfg.repo_commit(repo, "update"):
        raise HarnessError(
            f"no update commit configured for {repo}. Set `update_commit` in "
            "configs/repos.yaml or data/locks/repos.json, or pass --checkout."
        )

    candidate = cfg.checkout_dir(repo, ref)
    if candidate.is_dir():
        if not any(child.name != ".git" for child in candidate.iterdir()):
            raise HarnessError(
                f"{candidate} has no checked-out files.\n"
                f"Run `python -m harness prepare --repo {repo}` first."
            )
        return candidate

    clone = cfg.clone_dir(repo)
    if clone.is_dir():
        commit = cfg.repo_commit(repo, ref)
        head = git_head(clone)
        if commit and not head:
            raise HarnessError(
                f"{clone} has no checked-out working tree.\n"
                f"Run `python -m harness prepare --repo {repo}` first."
            )
        if commit and head and head != commit:
            raise HarnessError(
                f"{clone} is at {head[:12]}, but {ref} expects {commit[:12]}.\n"
                f"Run `python -m harness prepare --repo {repo}` first."
            )
        if not any(child.name != ".git" for child in clone.iterdir()):
            raise HarnessError(
                f"{clone} has no checked-out files.\n"
                f"Run `python -m harness prepare --repo {repo}` first."
            )
        return clone

    raise HarnessError(
        f"no checkout found for {repo} ({ref}).\n"
        f"Run `python -m harness prepare --repo {repo} --from-hf` first."
    )


def generate(
    cfg: Config,
    system: str,
    repo: str,
    experiment: str,
    *,
    ref: str = "base",
    checkout: str | None = None,
    mode: str = "full",
    base_docs: str | None = None,
    force: bool = False,
    extra_args: Sequence[str] | None = None,
) -> tuple[RunPaths, dict[str, Any], bool]:
    """Run one system adapter.

    Returns (paths, meta, skipped). `skipped` is True only when a previous
    successful generation is being reused.
    """
    spec = cfg.system(system)
    paths = run_paths(cfg, experiment, system, repo)
    meta = _load_meta(paths, cfg, experiment, system, repo)

    source = spec["source_path"]
    if not source.exists():
        raise HarnessError(f"system source directory not found: {source}")
    adapter = spec["adapter_path"]
    if not adapter.is_file():
        raise HarnessError(f"system adapter not found: {adapter}")

    checkout_path = resolve_checkout(cfg, repo, ref, checkout)
    base_path = Path(base_docs).expanduser().resolve() if base_docs else None
    if base_path is not None:
        if not base_path.is_dir():
            raise HarnessError(f"--base-docs does not exist: {base_path}")
        if not benchmark.list_markdown(base_path)[0]:
            raise HarnessError(f"--base-docs has no top-level markdown: {base_path}")
        if base_path == paths.docs.resolve():
            raise HarnessError(
                "--base-docs must be a different directory from the run docs"
            )
        _ensure_not_inside(paths.docs, base_path)
        _ensure_not_inside(base_path, paths.docs)
    base_fp = fingerprint_markdown(base_path) if base_path else None
    paths.logs.mkdir(parents=True, exist_ok=True)
    paths.work.mkdir(parents=True, exist_ok=True)
    previous_versions = dict(meta.get("versions") or {})
    current_versions = _system_provenance(cfg, spec, checkout_path)
    meta["versions"] = {**previous_versions, **current_versions}

    docs_nonempty = paths.docs.is_dir() and any(paths.docs.iterdir())
    previous_generate_ok = (meta.get("stages", {}).get("generate", {}) or {}).get(
        "status"
    ) == "ok"
    current_fp = fingerprint_markdown(paths.docs) if docs_nonempty else None
    stored_fp = previous_versions.get("docs_fingerprint")
    docs_changed = bool(stored_fp and current_fp and stored_fp != current_fp)
    repo_changed = bool(
        previous_versions.get("repo_commit")
        and current_versions.get("repo_commit")
        and previous_versions["repo_commit"] != current_versions["repo_commit"]
    )
    system_changed = bool(
        previous_versions.get("adapter_sha256")
        and current_versions.get("adapter_sha256")
        and previous_versions["adapter_sha256"] != current_versions["adapter_sha256"]
    )
    system_source_changed = bool(
        previous_versions.get("system_commit")
        and current_versions.get("system_commit")
        and previous_versions["system_commit"] != current_versions["system_commit"]
    )
    stored_base_fp = previous_versions.get("base_docs_fingerprint")
    base_changed = bool(stored_base_fp and base_fp and stored_base_fp != base_fp)
    inputs_changed = (
        docs_changed
        or repo_changed
        or system_changed
        or system_source_changed
        or base_changed
    )
    if docs_nonempty and not force and previous_generate_ok:
        if inputs_changed:
            reasons = []
            if docs_changed:
                reasons.append("docs")
            if repo_changed:
                reasons.append("repo commit")
            if system_changed:
                reasons.append("system adapter")
            if system_source_changed:
                reasons.append("system source")
            if base_changed:
                reasons.append("base docs")
            print(f"{', '.join(reasons).capitalize()} changed; regenerating.")
        else:
            _stage_finish(
                meta,
                "generate",
                {
                    "status": "skipped",
                    "reason": "existing successful generation reused",
                },
                status="skipped",
            )
            _save_meta(paths, meta)
            experiments.write_manifest(cfg, experiment)
            print(f"Reusing existing docs: {paths.docs}")
            return paths, meta, True

    if docs_nonempty and (force or not previous_generate_ok or inputs_changed):
        backup = _backup_dir(paths.docs)
        print(f"Moved previous docs to: {backup}")

    if mode == "update":
        if base_path is not None:
            shutil.copytree(base_path, paths.docs, dirs_exist_ok=True)
        else:
            if not paths.docs.is_dir() or not any(paths.docs.iterdir()):
                raise HarnessError(
                    "update mode needs an existing docs directory; pass --base-docs "
                    "or run a full generation first"
                )
    else:
        paths.docs.mkdir(parents=True, exist_ok=True)

    cmd = [
        "bash",
        str(adapter),
        "--repo",
        str(checkout_path),
        "--out",
        str(paths.docs),
        "--workdir",
        str(paths.work),
        "--mode",
        mode,
    ]
    configured_extras = spec.get("extra_args") or []
    if isinstance(configured_extras, str):
        configured_extras = shlex.split(configured_extras)
    extras = list(configured_extras) + list(extra_args or [])
    if extras:
        cmd += ["--", *[str(item) for item in extras]]

    env = dict(os.environ)
    for key, value in cfg.system_env(spec).items():
        # The YAML values are defaults; an explicit environment variable wins.
        env.setdefault(key, value)
    env.update(
        {
            "REPODOC_REPO": str(checkout_path),
            "REPODOC_OUT": str(paths.docs),
            "REPODOC_WORKDIR": str(paths.work),
            "REPODOC_MODE": mode,
            "REPODOC_REPO_NAME": repo,
            "REPODOC_EXPERIMENT": experiment,
            "REPODOC_SYSTEM": system,
        }
    )

    meta["config"]["generate"] = {
        "mode": mode,
        "ref": ref,
        "extra_args": [str(item) for item in extras],
    }
    # Do not merge stale metadata if the adapter does not write new metadata.
    (paths.work / "system_metadata.json").unlink(missing_ok=True)
    _stage_start(meta, "generate")
    meta["status"] = "generating"
    _save_meta(paths, meta)
    log = paths.logs / "generate.log"
    rc, duration = run_logged(cmd, log, cwd=checkout_path, env=env)
    result = {
        "command": cmd,
        "log": str(log),
        "duration_sec": round(duration, 1),
        "returncode": rc,
        "mode": mode,
        "ref": ref,
        "checkout": str(checkout_path),
    }
    if rc != 0:
        _stage_finish(meta, "generate", result, status="error")
        meta["status"] = "failed"
        _save_meta(paths, meta)
        raise HarnessError(f"documentation generation failed (exit {rc}). See {log}")

    top, nested = benchmark.list_markdown(paths.docs)
    if not top:
        _stage_finish(meta, "generate", result, status="error")
        meta["status"] = "failed"
        _save_meta(paths, meta)
        if nested:
            raise HarnessError(
                "generator produced only nested markdown, which CodeWikiBench cannot parse. "
                f"See {paths.docs}"
            )
        raise HarnessError(
            f"generator produced no markdown at the docs root: {paths.docs}"
        )
    if nested:
        _stage_finish(meta, "generate", result, status="error")
        meta["status"] = "failed"
        _save_meta(paths, meta)
        raise HarnessError(
            "generator produced nested markdown, which CodeWikiBench cannot parse. "
            f"See {paths.docs}"
        )

    result["docs_pages"] = len(top)
    result["docs_bytes"] = docs_stats(paths.docs).get("bytes")
    meta.setdefault("versions", {})["docs_fingerprint"] = fingerprint_markdown(
        paths.docs
    )
    if base_fp:
        meta["versions"]["base_docs_fingerprint"] = base_fp
    else:
        meta["versions"].pop("base_docs_fingerprint", None)
    _stage_finish(meta, "generate", result)
    meta["status"] = "generated"
    _save_meta(paths, meta)
    experiments.write_manifest(cfg, experiment)
    print(f"Generated {len(top)} pages -> {paths.docs}")
    return paths, meta, False


def _normalize_generation(
    generation: dict[str, Any], gen_stage: dict[str, Any]
) -> dict[str, Any]:
    """Return one comparable generation accounting record.

    Wall-clock time always comes from the harness stage. LLM time, token counts
    and cost come from the system adapter/metadata and may be unavailable.
    """
    tokens = normalize_tokens(generation.get("tokens"))
    generation["tokens"] = tokens
    duration = gen_stage.get("duration_sec")
    generation["duration_sec"] = duration
    llm_duration = generation.get("llm_duration_sec")
    if llm_duration is None:
        llm_duration = generation.get("llm_seconds")
    generation["llm_duration_sec"] = llm_duration
    generation.pop("llm_seconds", None)
    cost_status = generation.get("cost_status")
    if generation.get("cost_usd") is None:
        generation["cost_status"] = cost_status or "unavailable"
    elif cost_status in (None, "unavailable"):
        generation["cost_status"] = "unknown"
    else:
        generation["cost_status"] = cost_status
    if generation.get("usage_status") is None:
        generation["usage_status"] = generation["cost_status"]
    pages = generation.get("docs_pages") or generation.get("pages")
    if pages:
        pages = int(pages)
        if tokens.get("total") is not None:
            generation["tokens_per_page"] = round(tokens["total"] / pages, 1)
        if generation.get("cost_usd") is not None:
            generation["cost_per_page_usd"] = round(
                float(generation["cost_usd"]) / pages, 6
            )
        if duration:
            generation["seconds_per_page"] = round(float(duration) / pages, 1)
        if llm_duration:
            generation["llm_seconds_per_page"] = round(float(llm_duration) / pages, 1)
    return generation


def evaluate(
    cfg: Config,
    system: str,
    repo: str,
    experiment: str,
    *,
    force: bool = False,
    rubrics: str | None = None,
    judge_models: Sequence[str] | None = None,
    batch_size: int | None = None,
    method: str | None = None,
) -> tuple[RunPaths, dict[str, Any]]:
    cfg.system(system)  # validate name/enabled
    paths = run_paths(cfg, experiment, system, repo)
    if not paths.docs.is_dir():
        raise HarnessError(
            f"docs directory not found: {paths.docs}; run generation first"
        )
    meta = _load_meta(paths, cfg, experiment, system, repo)
    paths.logs.mkdir(parents=True, exist_ok=True)
    paths.results.mkdir(parents=True, exist_ok=True)

    current_fp = fingerprint_markdown(paths.docs)
    stored_fp = (meta.get("versions") or {}).get("docs_fingerprint")
    if not force and stored_fp and current_fp and stored_fp != current_fp:
        print("Docs changed since the previous run; reparsing and re-judging.")
        force = True

    rubric_path = (
        Path(rubrics).expanduser().resolve() if rubrics else cfg.rubric_path(repo)
    )
    if not rubric_path.is_file():
        raise HarnessError(
            f"frozen rubrics not found: {rubric_path}\n"
            f"Download benchmark data with `python -m harness prepare --repo {repo} --from-hf`, "
            f"or generate them with `python -m harness rubrics --repo {repo}`."
        )

    # ---- parse ---------------------------------------------------------
    _stage_start(meta, "parse")
    meta["status"] = "parsing"
    _save_meta(paths, meta)
    try:
        parse_result = benchmark.parse_docs(
            cfg,
            paths.docs,
            paths.parsed,
            project_name=repo,
            log_path=paths.logs / "parse.log",
            force=force,
        )
    except Exception as exc:
        _stage_finish(meta, "parse", {"error": str(exc)}, status="error")
        meta["status"] = "failed"
        _save_meta(paths, meta)
        raise
    _stage_finish(
        meta,
        "parse",
        parse_result,
        status="skipped" if parse_result.get("skipped") else "ok",
    )
    _save_meta(paths, meta)

    # ---- judge ---------------------------------------------------------
    judge_spec = cfg.evaluation.get("judge", {})
    actual_models = list(judge_models or judge_spec.get("models") or [])
    if not actual_models:
        raise HarnessError("no judge models configured; set evaluation.judge.models")
    actual_batch = (
        batch_size if batch_size is not None else judge_spec.get("batch_size", 4)
    )
    actual_method = method or judge_spec.get("combination_method", "average")
    combined_results = paths.results / "combined_results.json"
    judge_config = {
        "models": actual_models,
        "batch_size": actual_batch,
        "method": actual_method,
        "use_tools": bool(judge_spec.get("use_tools", True)),
        "enable_retry": bool(judge_spec.get("enable_retry", True)),
    }
    rubric_sha = file_sha256(rubric_path)
    previous_judge = meta.get("config", {}).get("judge")
    previous_rubric = meta.get("config", {}).get("rubrics") or {}
    judge_force = force
    if previous_judge and previous_judge != judge_config:
        print("Judge configuration changed since the previous run; re-judging.")
        judge_force = True
    if previous_rubric.get("sha256") and previous_rubric.get("sha256") != rubric_sha:
        print("Rubric file changed since the previous run; re-judging.")
        judge_force = True

    _stage_start(meta, "judge")
    meta["status"] = "judging"
    meta["config"]["rubrics"] = {"path": str(rubric_path), "sha256": rubric_sha}
    meta["config"]["judge"] = judge_config
    _save_meta(paths, meta)
    try:
        judge_result = benchmark.judge_docs(
            cfg,
            rubric_path,
            paths.parsed,
            combined_results,
            paths.logs / "judge.log",
            models=actual_models,
            force=judge_force,
            batch_size=actual_batch,
            method=actual_method,
        )
    except Exception as exc:
        _stage_finish(meta, "judge", {"error": str(exc)}, status="error")
        meta["status"] = "failed"
        _save_meta(paths, meta)
        raise
    _stage_finish(meta, "judge", judge_result)
    _save_meta(paths, meta)

    # ---- reports and metrics ------------------------------------------
    results = read_json(combined_results)
    if not isinstance(results, dict):
        raise HarnessError(
            f"judge did not produce a valid results file: {combined_results}"
        )
    report = benchmark.write_reports(cfg, combined_results, paths.results)

    architecture_spec = cfg.evaluation.get("architecture") or {}
    result_metrics = extract_results_metrics(
        results,
        architecture_keywords=architecture_spec.get("keywords"),
        architecture_include_topics=architecture_spec.get("include_topics"),
    )
    gen_stage = meta.get("stages", {}).get("generate", {}) or {}
    parse_stage = meta.get("stages", {}).get("parse", {}) or {}
    judge_stage = meta.get("stages", {}).get("judge", {}) or {}
    adapter_stats = read_json(paths.work / "system_metadata.json", default={}) or {}
    if not isinstance(adapter_stats, dict):
        raise HarnessError(
            f"adapter metadata must be a JSON object: {paths.work / 'system_metadata.json'}"
        )
    generation = {
        "mode": gen_stage.get("mode"),
        "ref": gen_stage.get("ref"),
        "checkout": gen_stage.get("checkout"),
        "docs_pages": gen_stage.get("docs_pages"),
        **generation_stats(paths.docs),
        **adapter_stats,
    }
    generation = _normalize_generation(generation, gen_stage)
    metrics = {
        "experiment": experiment,
        "system": system,
        "repo": repo,
        "status": "ok",
        "score": result_metrics["score"],
        "score_100": result_metrics["score_100"],
        "num_leaves": result_metrics["num_leaves"],
        "num_errors": result_metrics["num_errors"],
        "num_failed": report.get("num_failed"),
        "topics": result_metrics["topics"],
        "architecture": result_metrics["architecture"],
        "generation": generation,
        "judge": {
            "models": actual_models,
            "batch_size": actual_batch,
            "method": actual_method,
            "duration_sec": judge_stage.get("duration_sec"),
            "tokens": result_metrics["judge_tokens"],
        },
        "artifacts": {
            "docs": docs_stats(paths.docs),
            "docs_dir": str(paths.docs),
            "parsed_dir": str(paths.parsed),
            "results": str(combined_results),
            "report_markdown": str(paths.results / "report.md"),
        },
        "versions": meta.get("versions", {}),
        "timestamps": {
            "generated_at": gen_stage.get("finished_at"),
            "judged_at": judge_stage.get("finished_at"),
        },
        "stage_durations": {
            "generate_sec": gen_stage.get("duration_sec"),
            "parse_sec": parse_stage.get("duration_sec"),
            "judge_sec": judge_stage.get("duration_sec"),
        },
    }
    write_json(paths.metrics, metrics)
    meta.setdefault("versions", {})["docs_fingerprint"] = current_fp
    meta["status"] = "evaluated"
    meta["metrics"] = {
        "score": metrics["score"],
        "score_100": metrics["score_100"],
        "num_leaves": metrics["num_leaves"],
        "num_errors": metrics["num_errors"],
    }
    _save_meta(paths, meta)
    experiments.write_manifest(cfg, experiment)
    print(
        f"Score: {metrics['score_100']:.2f}/100 "
        f"({metrics['num_leaves']} leaves, {metrics['num_errors']} errors) -> {paths.metrics}"
    )
    return paths, metrics


def run(
    cfg: Config,
    system: str,
    repo: str,
    experiment: str,
    *,
    ref: str = "base",
    checkout: str | None = None,
    force: bool = False,
    extra_args: Sequence[str] | None = None,
    rubrics: str | None = None,
    judge_models: Sequence[str] | None = None,
    batch_size: int | None = None,
    method: str | None = None,
) -> tuple[RunPaths, dict[str, Any]]:
    paths, _meta, skipped = generate(
        cfg,
        system,
        repo,
        experiment,
        ref=ref,
        checkout=checkout,
        mode="full",
        force=force,
        extra_args=extra_args,
    )
    return evaluate(
        cfg,
        system,
        repo,
        experiment,
        force=force or not skipped,
        rubrics=rubrics,
        judge_models=judge_models,
        batch_size=batch_size,
        method=method,
    )


def update(
    cfg: Config,
    system: str,
    repo: str,
    experiment: str,
    *,
    base_docs: str,
    ref: str = "update",
    checkout: str | None = None,
    force: bool = False,
    extra_args: Sequence[str] | None = None,
    rubrics: str | None = None,
    judge_models: Sequence[str] | None = None,
    batch_size: int | None = None,
    method: str | None = None,
) -> tuple[RunPaths, dict[str, Any]]:
    spec = cfg.system(system)
    if not spec.get("supports_update", False):
        raise HarnessError(f"system '{system}' is not marked as supporting update mode")
    paths, _meta, skipped = generate(
        cfg,
        system,
        repo,
        experiment,
        ref=ref,
        checkout=checkout,
        mode="update",
        base_docs=base_docs,
        force=force,
        extra_args=extra_args,
    )
    return evaluate(
        cfg,
        system,
        repo,
        experiment,
        force=force or not skipped,
        rubrics=rubrics,
        judge_models=judge_models,
        batch_size=batch_size,
        method=method,
    )
