"""Command line interface: `python -m harness <command>`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__, aggregate, architecture, prepare, runner
from .config import Config, ConfigError
from .doctor import run_doctor
from .util import (
    ROOT,
    HarnessError,
    load_dotenv_simple,
    now_iso,
    read_json,
    safe_component,
    write_json,
)

METHODS = ["average", "majority_vote", "weighted_average", "max", "min"]


def _add_target_args(
    parser: argparse.ArgumentParser,
    *,
    ref_default: str = "base",
    include_checkout: bool = True,
) -> None:
    parser.add_argument(
        "--system", required=True, help="system name from configs/systems.yaml"
    )
    parser.add_argument(
        "--repo", required=True, help="repository name from configs/repos.yaml"
    )
    parser.add_argument(
        "--experiment",
        default="pilot",
        help=(
            "output experiment label, preferably "
            "<language>-<repo>-<scope>-v<N> (default: pilot)"
        ),
    )
    if include_checkout:
        parser.add_argument(
            "--ref",
            default=ref_default,
            choices=["base", "update"],
            help=f"which prepared checkout/commit to use (default: {ref_default})",
        )
        parser.add_argument(
            "--checkout",
            help="use an explicit repository checkout instead of data/checkouts/...",
        )
        parser.add_argument(
            "--system-arg",
            action="append",
            default=[],
            metavar="ARG",
            help="extra adapter argument; repeat for multiple arguments (use --system-arg=--flag for flags)",
        )
    parser.add_argument(
        "--force", action="store_true", help="redo stages even if outputs exist"
    )


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rubrics", help="override the frozen rubrics JSON path")
    parser.add_argument(
        "--judge-model",
        action="append",
        default=[],
        help="judge model; repeat to override configs/evaluation.yaml",
    )
    parser.add_argument("--batch-size", type=int, help="judge concurrency override")
    parser.add_argument(
        "--method", choices=METHODS, help="judge-combination method override"
    )


def _add_run_args(
    parser: argparse.ArgumentParser, *, ref_default: str = "base"
) -> None:
    _add_target_args(parser, ref_default=ref_default)
    _add_eval_args(parser)


def _select_repos(cfg: Config, args: argparse.Namespace) -> list[str]:
    selected: list[str] = []
    if getattr(args, "all", False):
        selected.extend(cfg.repo_names())
    if getattr(args, "split", None):
        selected.extend(cfg.repo_names(args.split))
    selected.extend(getattr(args, "repo", []) or [])
    if not selected:
        raise HarnessError(
            "choose repositories with --repo NAME, --split core, or --all"
        )
    # Preserve config order and remove duplicates.
    wanted = set(selected)
    return [name for name in cfg.repo_names() if name in wanted]


def cmd_generate(cfg: Config, args: argparse.Namespace) -> int:
    paths, _meta, skipped = runner.generate(
        cfg,
        args.system,
        args.repo,
        args.experiment,
        ref=args.ref,
        checkout=args.checkout,
        mode="full",
        force=args.force,
        extra_args=args.system_arg,
    )
    print(f"{'Reused' if skipped else 'Generated'}: {paths.docs}")
    return 0


def cmd_evaluate(cfg: Config, args: argparse.Namespace) -> int:
    paths, _metrics = runner.evaluate(
        cfg,
        args.system,
        args.repo,
        args.experiment,
        force=args.force,
        rubrics=args.rubrics,
        judge_models=args.judge_model or None,
        batch_size=args.batch_size,
        method=args.method,
    )
    print(f"Results: {paths.results}")
    return 0


def cmd_run(cfg: Config, args: argparse.Namespace) -> int:
    _, metrics = runner.run(
        cfg,
        args.system,
        args.repo,
        args.experiment,
        ref=args.ref,
        checkout=args.checkout,
        force=args.force,
        extra_args=args.system_arg,
        rubrics=args.rubrics,
        judge_models=args.judge_model or None,
        batch_size=args.batch_size,
        method=args.method,
    )
    print(f"Done: {metrics['score_100']:.2f}/100")
    return 0


def cmd_update(cfg: Config, args: argparse.Namespace) -> int:
    _, metrics = runner.update(
        cfg,
        args.system,
        args.repo,
        args.experiment,
        base_docs=args.base_docs,
        ref=args.ref,
        checkout=args.checkout,
        force=args.force,
        extra_args=args.system_arg,
        rubrics=args.rubrics,
        judge_models=args.judge_model or None,
        batch_size=args.batch_size,
        method=args.method,
    )
    print(f"Update run: {metrics['score_100']:.2f}/100")
    return 0


def cmd_batch(cfg: Config, args: argparse.Namespace) -> int:
    suite = cfg.suite(args.suite)
    experiment = args.experiment or args.suite
    systems = [
        s for s in suite["systems"] if not args.only_system or s in args.only_system
    ]
    repos = [r for r in suite["repos"] if not args.only_repo or r in args.only_repo]
    if not systems or not repos:
        raise HarnessError(
            "suite selection is empty after --only-system/--only-repo filters"
        )

    started_at = now_iso()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for system in systems:
        try:
            spec = cfg.system(system, allow_disabled=True)
        except ConfigError as exc:
            failures.append({"system": system, "repo": None, "error": str(exc)})
            if not args.continue_on_error:
                raise
            continue
        if spec.get("enabled", True) is False:
            print(f"Skipping disabled system: {system}")
            continue
        for repo in repos:
            print(f"\n=== {system} / {repo} ===")
            try:
                _, metrics = runner.run(
                    cfg,
                    system,
                    repo,
                    experiment,
                    ref=args.ref,
                    force=args.force,
                    extra_args=args.system_arg,
                    rubrics=args.rubrics,
                    judge_models=args.judge_model or None,
                    batch_size=args.batch_size,
                    method=args.method,
                )
                results.append(
                    {
                        "system": system,
                        "repo": repo,
                        "score_100": metrics.get("score_100"),
                        "metrics": str(
                            runner.run_paths(cfg, experiment, system, repo).metrics
                        ),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - batch should report all failures
                failures.append({"system": system, "repo": repo, "error": str(exc)})
                print(f"FAILED: {system} / {repo}: {exc}", file=sys.stderr)
                if not args.continue_on_error:
                    raise

    out = cfg.runs_dir / safe_component(experiment)
    write_json(
        out / "batch.json",
        {
            "suite": args.suite,
            "experiment": experiment,
            "started_at": started_at,
            "finished_at": now_iso(),
            "systems": systems,
            "repos": repos,
            "results": results,
            "failures": failures,
        },
    )
    if results:
        payload = aggregate.summarize(cfg, experiment)
        print(f"\nWrote {out / 'summary.md'} ({len(payload['records'])} runs)")

    if failures:
        print(
            f"\n{len(failures)} run(s) failed; see {out / 'batch.json'}",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_prepare(cfg: Config, args: argparse.Namespace) -> int:
    if args.skip_checkouts and not args.from_hf:
        raise HarnessError("--skip-checkouts only makes sense together with --from-hf")
    names = _select_repos(cfg, args)
    print(f"Preparing {len(names)} repositories: {', '.join(names)}")
    if args.from_hf:
        prepare.fetch_hf(cfg, names, force=args.force)
    if not args.skip_checkouts:
        prepare.prepare_checkouts(
            cfg, names, allow_head=args.allow_head, force=args.force
        )
    return 0


def cmd_rubrics(cfg: Config, args: argparse.Namespace) -> int:
    names = _select_repos(cfg, args)
    failures = 0
    for repo in names:
        print(f"Rubrics for {repo}")
        try:
            out = prepare.frozen_rubrics(
                cfg, repo, models=args.model or None, force=args.force
            )
            print(f"  -> {out}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAILED: {exc}", file=sys.stderr)
    return 1 if failures else 0


def cmd_summary(cfg: Config, args: argparse.Namespace) -> int:
    payload = aggregate.summarize(cfg, args.experiment)
    out = cfg.runs_dir / safe_component(args.experiment)
    print(f"Wrote {out / 'summary.md'}")
    print(f"Wrote {out / 'summary.csv'}")
    print(f"Wrote {out / 'summary.json'}")
    if not payload["records"]:
        print("No metrics found for this experiment.")
        return 0
    if args.format == "json":
        import json

        print(
            json.dumps(
                {k: payload[k] for k in ("averages", "topic_averages")}, indent=2
            )
        )
    elif args.format == "csv":
        print((out / "summary.csv").read_text(encoding="utf-8"))
    else:
        print((out / "summary.md").read_text(encoding="utf-8"))
    return 0


def cmd_architecture(cfg: Config, args: argparse.Namespace) -> int:
    results = read_json(Path(args.results))
    if not isinstance(results, dict):
        raise HarnessError(f"results file is not a JSON object: {args.results}")
    spec = cfg.evaluation.get("architecture") or {}
    keywords = args.keyword or spec.get("keywords")
    metrics = architecture.architecture_metrics(
        results,
        keywords=keywords,
        include_topics=spec.get("include_topics"),
    )
    docs_stats = (
        architecture.docs_architecture_stats(Path(args.docs)) if args.docs else None
    )
    payload = {"metrics": metrics, "docs": docs_stats}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out.with_suffix(".json"), payload)
        out.with_suffix(".md").write_text(
            architecture.render_markdown(metrics, docs_stats), encoding="utf-8"
        )
    if args.format == "json":
        import json

        print(json.dumps(payload, indent=2))
    else:
        print(architecture.render_markdown(metrics, docs_stats))
    return 0


def cmd_architecture_rubrics(cfg: Config, args: argparse.Namespace) -> int:
    results = read_json(Path(args.rubrics))
    if not isinstance(results, dict):
        raise HarnessError(f"rubrics file is not a JSON object: {args.rubrics}")
    spec = cfg.evaluation.get("architecture") or {}
    keywords = args.keyword or spec.get("keywords")
    filtered = architecture.filter_architecture_rubrics(
        results,
        keywords=keywords,
        include_topics=spec.get("include_topics"),
    )
    write_json(Path(args.out), filtered)
    print(
        f"Wrote {args.out} ({len(filtered['rubrics'])} top-level topics, "
        f"{len(list(architecture._walk_leaves(filtered['rubrics'])))} leaves)"
    )
    return 0


def cmd_list(cfg: Config, args: argparse.Namespace) -> int:
    import json

    data = {
        "systems": {
            name: {
                "kind": cfg.system(name, allow_disabled=True).get("kind"),
                "enabled": cfg.system(name, allow_disabled=True).get("enabled", True)
                is not False,
                "adapter": str(cfg.system(name, allow_disabled=True)["adapter_path"]),
            }
            for name in cfg.system_names(include_disabled=True)
        },
        "repos": {
            name: {
                "language": cfg.repo(name).get("language"),
                "split": cfg.repo(name).get("split"),
                "commit": cfg.repo_commit(name, "base"),
                "update_commit": cfg.repo_commit(name, "update"),
            }
            for name in cfg.repo_names()
        },
        "suites": {name: cfg.suite(name) for name in cfg.suite_names()},
    }
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0

    print("Systems:")
    for name, spec in data["systems"].items():
        flag = "" if spec["enabled"] else " (disabled)"
        print(f"  {name:<16} {spec['kind']}{flag}  {spec['adapter']}")
    print("\nRepositories:")
    for name, spec in data["repos"].items():
        print(
            f"  {name:<30} {str(spec['language']):<12} split={spec['split']:<5} "
            f"commit={spec['commit'] or '-'}"
        )
    print("\nSuites:")
    for name, spec in data["suites"].items():
        print(f"  {name:<16} systems={spec['systems']} repos={len(spec['repos'])}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m harness",
        description="Run documentation-generation systems against CodeWikiBench.",
        epilog=(
            "Typical: doctor -> prepare -> run -> summary. "
            "Advanced: generate, evaluate, update, batch, architecture."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="list systems, repositories and suites")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("doctor", help="check environments, adapters and prepared data")
    p.add_argument("--strict", action="store_true", help="treat warnings as failures")
    p.set_defaults(func=lambda cfg, args: run_doctor(cfg, strict=args.strict))

    p = sub.add_parser(
        "prepare", help="download benchmark data and create pinned checkouts"
    )
    p.add_argument(
        "--repo", action="append", default=[], help="repository name; repeatable"
    )
    p.add_argument(
        "--split", choices=["core", "full"], help="select all repos in a split"
    )
    p.add_argument("--all", action="store_true", help="select all 22 repositories")
    p.add_argument(
        "--from-hf",
        action="store_true",
        help="download reference docs/rubrics/commits from HuggingFace",
    )
    p.add_argument(
        "--allow-head",
        action="store_true",
        help="fall back to each remote's HEAD when no commit is configured",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="overwrite reference/rubrics and recreate worktrees",
    )
    p.add_argument(
        "--skip-checkouts",
        action="store_true",
        help="only download data; do not clone repositories or create worktrees",
    )
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser(
        "rubrics", help="generate and freeze rubrics from reference documentation"
    )
    p.add_argument(
        "--repo", action="append", default=[], help="repository name; repeatable"
    )
    p.add_argument("--split", choices=["core", "full"])
    p.add_argument("--all", action="store_true")
    p.add_argument(
        "--model",
        action="append",
        default=[],
        help="rubric-generation model; repeatable",
    )
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_rubrics)

    p = sub.add_parser("generate", help="run a system adapter and produce docs only")
    _add_target_args(p)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("evaluate", help="parse and judge existing docs")
    _add_target_args(p, include_checkout=False)
    _add_eval_args(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("run", help="generate docs, parse, judge and write metrics")
    _add_run_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser(
        "update", help="incrementally update docs from a previous full run"
    )
    _add_run_args(p, ref_default="update")
    p.add_argument(
        "--base-docs",
        required=True,
        help="docs directory from the base full generation",
    )
    p.set_defaults(func=cmd_update)

    p = sub.add_parser("batch", help="run every system x repository in a suite")
    p.add_argument(
        "--suite", required=True, help="suite name from configs/experiments.yaml"
    )
    p.add_argument(
        "--experiment",
        help=(
            "output label, preferably <language>-<repo>-<scope>-v<N> "
            "(default: suite name)"
        ),
    )
    p.add_argument("--ref", default="base", choices=["base", "update"])
    p.add_argument("--force", action="store_true")
    p.add_argument("--system-arg", action="append", default=[], metavar="ARG")
    p.add_argument("--rubrics", help="override rubrics path for every run")
    p.add_argument("--judge-model", action="append", default=[])
    p.add_argument("--batch-size", type=int)
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--only-system", action="append", default=[])
    p.add_argument("--only-repo", action="append", default=[])
    p.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep going after a failed run (default: true)",
    )
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser(
        "architecture",
        help="architecture-focused view over an existing CodeWikiBench results file",
    )
    p.add_argument("--results", required=True, help="combined results JSON")
    p.add_argument("--docs", help="optional flat docs directory for structural stats")
    p.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="override architecture keyword; repeatable",
    )
    p.add_argument("--format", choices=["md", "json"], default="md")
    p.add_argument("--out", help="write <out>.json and <out>.md")
    p.set_defaults(func=cmd_architecture)

    p = sub.add_parser(
        "architecture-rubrics",
        help="export the architecture-focused subset of a frozen rubric tree",
    )
    p.add_argument("--rubrics", required=True, help="combined rubrics JSON")
    p.add_argument("--out", required=True, help="filtered rubrics JSON")
    p.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="override architecture keyword; repeatable",
    )
    p.set_defaults(func=cmd_architecture_rubrics)

    p = sub.add_parser("summary", help="aggregate metrics for an experiment")
    p.add_argument("--experiment", default="pilot")
    p.add_argument("--format", choices=["md", "csv", "json"], default="md")
    p.set_defaults(func=cmd_summary)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv_simple(Path.cwd() / ".env")
    load_dotenv_simple(ROOT / ".env")

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = Config.load()
        return int(args.func(cfg, args) or 0)
    except (ConfigError, HarnessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
