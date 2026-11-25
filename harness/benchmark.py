"""Thin wrappers around the CodeWikiBench CLI.

We deliberately call the upstream CLI instead of importing its modules so the
benchmark checkout stays pristine and can be upgraded independently.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from .config import Config
from .util import (
    HarnessError,
    read_json,
    run_capture,
    run_logged,
    write_json,
)


def _bench_cmd(cfg: Config, *args: str) -> list[str]:
    # CodeWikiBench exposes a console script (`codewikibench`), but this branch
    # has no codewikibench/__main__.py, so `python -m codewikibench` fails.
    # Calling cli:main keeps the harness independent of PATH/venv layout.
    launcher = "import sys; from codewikibench.cli import main; sys.exit(main())"
    return [cfg.bench_python, "-c", launcher, *args]


def _bench_cwd(cfg: Config) -> Path:
    return cfg.benchmark_source if cfg.benchmark_source.is_dir() else cfg.root


def _run(
    cfg: Config,
    args: Sequence[str],
    log_path: Path,
    *,
    check: bool = True,
) -> dict[str, Any]:
    cmd = _bench_cmd(cfg, *args)
    rc, duration = run_logged(cmd, log_path, cwd=_bench_cwd(cfg), env=cfg.bench_env())
    result = {"command": cmd, "returncode": rc, "duration_sec": round(duration, 1)}
    if check and rc != 0:
        raise HarnessError(
            f"codewikibench command failed (exit {rc}): {' '.join(args)}\n"
            f"See log: {log_path}"
        )
    return result


def list_markdown(docs_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return top-level and nested markdown files.

    CodeWikiBench's parser currently visits only the docs root, so nested pages
    would silently disappear. We fail early with an actionable message.
    """
    if not docs_dir.is_dir():
        raise HarnessError(f"docs directory does not exist: {docs_dir}")
    top = sorted(
        p for p in docs_dir.iterdir() if p.is_file() and p.suffix.lower() == ".md"
    )
    nested = sorted(
        p
        for p in docs_dir.rglob("*.md")
        if p.is_file() and p.parent.resolve() != docs_dir.resolve()
    )
    return top, nested


def _looks_like_deepwiki_numbering(top: list[Path]) -> bool:
    """True when every page starts with a `1.2 - Title` style heading."""
    if not top:
        return False
    for page in top:
        first = next(
            (
                line.strip()
                for line in page.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ),
            "",
        )
        if not re.match(r"^\d+(?:\.\d+)*\s*-\s*\S", first):
            return False
    return True


def normalize_missing_module_tree(docs_dir: Path) -> bool:
    """Create a flat module_tree.json from filenames when a system omits it.

    CodeWikiBench accepts a module_tree.json to define section order. Without
    one it expects DeepWiki-style `N - Title` headings. Synthesizing a flat tree
    lets any generator that emits plain markdown be parsed via its filenames.

    DeepWiki numbering is detected and left alone: in that format the heading,
    not the filename, carries the hierarchy.
    """
    top, _ = list_markdown(docs_dir)
    if not top:
        raise HarnessError(f"no .md files found in docs directory: {docs_dir}")
    tree_path = docs_dir / "module_tree.json"
    if tree_path.exists():
        try:
            tree_data = read_json(tree_path)
        except (json.JSONDecodeError, OSError) as exc:
            raise HarnessError(
                f"invalid module_tree.json in {docs_dir}: {exc}"
            ) from exc
        if not isinstance(tree_data, dict):
            raise HarnessError(
                f"invalid module_tree.json in {docs_dir}: expected a JSON object"
            )
        return False
    if _looks_like_deepwiki_numbering(top):
        print(
            "DeepWiki-style numbered headings detected; not synthesizing module_tree.json"
        )
        return False

    tree: dict[str, Any] = {}
    for page in top:
        tree[page.stem] = {}
    write_json(tree_path, tree)
    return True


def parse_docs(
    cfg: Config,
    docs_dir: Path,
    parsed_dir: Path,
    project_name: str,
    log_path: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    docs_dir = docs_dir.resolve()
    parsed_dir = parsed_dir.resolve()
    top, nested = list_markdown(docs_dir)
    if not top:
        raise HarnessError(f"no .md files found in docs directory: {docs_dir}")
    if nested:
        shown = "\n".join(f"  - {p.relative_to(docs_dir)}" for p in nested[:10])
        raise HarnessError(
            "CodeWikiBench only parses markdown at the docs root, but found nested pages:\n"
            f"{shown}\n"
            "Make the system adapter emit a flat docs directory (or add a flattening step)."
        )

    if (parsed_dir / "docs_tree.json").is_file() and (
        parsed_dir / "structured_docs.json"
    ).is_file():
        if not force:
            try:
                tree_data = read_json(parsed_dir / "docs_tree.json")
                structured_data = read_json(parsed_dir / "structured_docs.json")
            except (json.JSONDecodeError, OSError) as exc:
                print(
                    f"warning: parsed docs are invalid ({exc}); reparsing {parsed_dir}"
                )
            else:
                if isinstance(tree_data, dict) and isinstance(structured_data, dict):
                    return {"skipped": True, "reason": "parsed docs already exist"}
                print(
                    f"warning: parsed docs have unexpected JSON types; reparsing {parsed_dir}"
                )

    normalized = False
    if cfg.evaluation.get("run", {}).get("normalize_missing_module_tree", True):
        normalized = normalize_missing_module_tree(docs_dir)

    parsed_dir.mkdir(parents=True, exist_ok=True)
    result = _run(
        cfg,
        [
            "parse",
            "--docs-dir",
            str(docs_dir),
            "--out",
            str(parsed_dir),
            "--project-name",
            project_name,
        ],
        log_path,
    )
    result["normalized_module_tree"] = normalized
    result["docs_pages"] = len(top)
    return result


def generate_rubrics(
    cfg: Config,
    parsed_reference_dir: Path,
    out_path: Path,
    log_path: Path,
    *,
    models: Sequence[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    parsed_reference_dir = parsed_reference_dir.resolve()
    out_path = out_path.resolve()
    spec = cfg.evaluation.get("rubrics", {}).get("generation", {})
    models = list(models or spec.get("models") or [])
    if not models:
        raise HarnessError("no rubric-generation models configured")

    args = [
        "rubrics",
        "--docs",
        str(parsed_reference_dir),
        "--out",
        str(out_path),
    ]
    for model in models:
        args += ["--model", model]
    if spec.get("use_tools", True):
        args.append("--use-tools")
    if spec.get("language_agnostic", True):
        args.append("--language-agnostic")
    else:
        args.append("--no-language-agnostic")
    if spec.get("combine_model"):
        args += ["--combine-model", str(spec["combine_model"])]
    if spec.get("temperature") is not None:
        args += ["--temperature", str(spec["temperature"])]
    if spec.get("max_retries") is not None:
        args += ["--max-retries", str(spec["max_retries"])]
    if force:
        args.append("--force")

    return _run(cfg, args, log_path)


def judge_docs(
    cfg: Config,
    rubrics_path: Path,
    parsed_dir: Path,
    out_path: Path,
    log_path: Path,
    *,
    models: Sequence[str] | None = None,
    force: bool = False,
    batch_size: int | None = None,
    method: str | None = None,
) -> dict[str, Any]:
    rubrics_path = rubrics_path.resolve()
    parsed_dir = parsed_dir.resolve()
    out_path = out_path.resolve()
    spec = cfg.evaluation.get("judge", {})
    models = list(models or spec.get("models") or [])
    if not models:
        raise HarnessError("no judge models configured")

    if not rubrics_path.is_file():
        raise HarnessError(
            f"frozen rubrics not found: {rubrics_path}\n"
            "Run `python -m harness prepare --repo <name> --from-hf` "
            "or `python -m harness rubrics --repo <name>` first."
        )

    args = [
        "judge",
        "--rubrics",
        str(rubrics_path),
        "--docs",
        str(parsed_dir),
        "--out",
        str(out_path),
    ]
    for model in models:
        args += ["--model", model]
    args += ["--batch-size", str(batch_size or spec.get("batch_size", 4))]
    args += ["--method", method or spec.get("combination_method", "average")]
    if spec.get("weights"):
        weights = spec["weights"]
        if isinstance(weights, (list, tuple)):
            weights = ",".join(str(w) for w in weights)
        args += ["--weights", str(weights)]
    if spec.get("use_tools", True):
        args.append("--use-tools")
    if spec.get("enable_retry", True):
        args.append("--enable-retry")
    if spec.get("max_retries") is not None:
        args += ["--max-retries", str(spec["max_retries"])]
    if force:
        args.append("--force")

    return _run(cfg, args, log_path)


def _report(
    cfg: Config, results_path: Path, fmt: str, max_score: float | None = None
) -> str:
    results_path = results_path.resolve()
    args = ["report", "--results", str(results_path), "--format", fmt]
    if max_score is not None:
        args += ["--max-score", str(max_score)]
    proc = run_capture(
        _bench_cmd(cfg, *args),
        cwd=_bench_cwd(cfg),
        env=cfg.bench_env(),
    )
    if proc.returncode != 0:
        raise HarnessError(
            f"codewikibench report failed for {results_path}:\n{proc.stderr.strip()}"
        )
    return proc.stdout


def write_reports(
    cfg: Config,
    results_path: Path,
    results_dir: Path,
    *,
    max_score: float | None = None,
) -> dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    max_score = (
        max_score
        if max_score is not None
        else cfg.evaluation.get("report", {}).get("max_score", 0.99)
    )
    markdown = _report(cfg, results_path, "markdown", max_score)
    (results_dir / "report.md").write_text(markdown, encoding="utf-8")

    report_json = json.loads(_report(cfg, results_path, "json", max_score))
    write_json(results_dir / "report.json", report_json)

    summary = _report(cfg, results_path, "summary", max_score)
    (results_dir / "report.txt").write_text(summary + "\n", encoding="utf-8")
    return report_json
