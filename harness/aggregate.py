"""Aggregate per-run metrics into experiment-level tables."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from typing import Any, Iterable

from .config import Config
from .util import now_iso, read_json, safe_component, write_json


def _mean(values: Iterable[float]) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _sum_or_none(values: Iterable[Any]) -> float | None:
    values = [float(value) for value in values if value is not None]
    return sum(values) if values else None


def _generation_tokens(record: dict[str, Any]) -> dict[str, Any]:
    generation = record.get("generation") or {}
    tokens = generation.get("tokens")
    return tokens if isinstance(tokens, dict) else {}


def _load_records(
    cfg: Config, experiment: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = cfg.runs_dir / safe_component(experiment)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for metrics_path in sorted(root.glob("systems/*/*/metrics.json")):
        meta = read_json(metrics_path.with_name("meta.json"))
        if isinstance(meta, dict) and meta.get("status") == "failed":
            # Keep the failure record, but do not report stale metrics from an
            # earlier successful run in the same directory.
            continue
        data = read_json(metrics_path)
        if isinstance(data, dict):
            data["_path"] = str(metrics_path)
            records.append(data)
    for meta_path in sorted(root.glob("systems/*/*/meta.json")):
        data = read_json(meta_path)
        if isinstance(data, dict) and data.get("status") == "failed":
            failures.append(
                {
                    "system": data.get("system"),
                    "repo": data.get("repo"),
                    "status": data.get("status"),
                    "stage": next(
                        (
                            name
                            for name, stage in (data.get("stages") or {}).items()
                            if stage.get("status") == "error"
                        ),
                        None,
                    ),
                    "meta": str(meta_path),
                }
            )
    return records, failures


def _topic_averages(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        system = record.get("system", "?")
        for topic in record.get("topics") or []:
            name = topic.get("name") or "unnamed"
            grouped[system][name].append(float(topic.get("score", 0) or 0) * 100)
    return {
        system: {name: _mean(values) for name, values in topics.items()}
        for system, topics in grouped.items()
    }


def _system_averages(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record.get("system", "?")].append(record)
    rows = []
    for system, items in sorted(grouped.items()):
        scores = [float(item.get("score_100", 0) or 0) for item in items]
        rows.append(
            {
                "system": system,
                "repos": len(items),
                "score_mean": _mean(scores),
                "score_min": min(scores) if scores else None,
                "score_max": max(scores) if scores else None,
                "architecture_mean": _mean(
                    (item.get("architecture") or {}).get("score_100") for item in items
                ),
                "errors": sum(int(item.get("num_errors", 0) or 0) for item in items),
                "cost_usd": _mean(
                    (item.get("generation") or {}).get("cost_usd") for item in items
                ),
                "cost_total_usd": _sum_or_none(
                    (item.get("generation") or {}).get("cost_usd") for item in items
                ),
                "duration_mean_sec": _mean(
                    (item.get("generation") or {}).get("duration_sec") for item in items
                ),
                "llm_duration_mean_sec": _mean(
                    (item.get("generation") or {}).get("llm_duration_sec")
                    for item in items
                ),
                "tokens_total": _sum_or_none(
                    _generation_tokens(item).get("total") for item in items
                ),
                "input_tokens_total": _sum_or_none(
                    _generation_tokens(item).get("input") for item in items
                ),
                "output_tokens_total": _sum_or_none(
                    _generation_tokens(item).get("output") for item in items
                ),
                "reasoning_tokens_total": _sum_or_none(
                    _generation_tokens(item).get("reasoning") for item in items
                ),
            }
        )
    return rows


def _render_markdown(
    experiment: str,
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    averages: list[dict[str, Any]],
    topics: dict[str, dict[str, float | None]],
) -> str:
    lines = [
        f"# Experiment `{experiment}`",
        "",
        f"Generated: {now_iso()}  |  runs: {len(records)}  |  failed: {len(failures)}",
        "",
        "Scores are weighted CodeWikiBench rubric scores (0-100), recomputed from the scored rubric tree.",
        "",
        "## By system",
        "",
        "| System | Repos | Mean | Min | Max | Architecture | Leaf errors | Mean gen time | Mean LLM time | Total tokens | Mean cost (USD) | Total cost (USD) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in averages:
        lines.append(
            f"| {row['system']} | {row['repos']} | {_fmt(row['score_mean'])} | "
            f"{_fmt(row['score_min'])} | {_fmt(row['score_max'])} | "
            f"{_fmt(row.get('architecture_mean'))} | {row['errors']} | "
            f"{_fmt(row.get('duration_mean_sec'), 0)}s | "
            f"{_fmt(row.get('llm_duration_mean_sec'), 0)}s | "
            f"{_fmt(row.get('tokens_total'), 0)} | "
            f"{_fmt(row['cost_usd'], 4)} | {_fmt(row.get('cost_total_usd'), 4)} |"
        )

    lines += [
        "",
        "## Per repository",
        "",
        "| System | Repository | Score | Architecture | Leaves | Errors | Gen model | Gen time | Gen cost | Pages | Mermaid | Flowchart | Sequence | Links | Judge models |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in sorted(
        records, key=lambda r: (str(r.get("system")), str(r.get("repo")))
    ):
        generation = record.get("generation") or {}
        judge = record.get("judge") or {}
        artifacts = record.get("artifacts") or {}
        docs = artifacts.get("docs") or {}
        models = ", ".join(judge.get("models") or [])
        docs_arch = docs.get("architecture") or {}
        diagram_types = docs_arch.get("diagram_types") or {}
        tokens = _generation_tokens(record)
        token_text = (
            f"{_fmt(tokens.get('input'), 0)} / {_fmt(tokens.get('output'), 0)} / "
            f"{_fmt(tokens.get('total'), 0)}"
        )
        judge_tokens = judge.get("tokens") or {}
        judge_token_text = (
            f"{_fmt(judge_tokens.get('input'), 0)} / "
            f"{_fmt(judge_tokens.get('output'), 0)}"
        )
        cost = generation.get("cost_usd")
        cost_text = _fmt(cost, 4)
        if generation.get("cost_status") in {"partial", "unknown", "unavailable"}:
            cost_text = f"{cost_text} ({generation.get('cost_status')})"
        lines.append(
            f"| {record.get('system')} | {record.get('repo')} | "
            f"{_fmt(record.get('score_100'))} | "
            f"{_fmt((record.get('architecture') or {}).get('score_100'))} | "
            f"{_fmt(record.get('num_leaves'), 0)} | "
            f"{_fmt(record.get('num_errors'), 0)} | {_fmt(generation.get('model'))} | "
            f"{_fmt(generation.get('duration_sec'), 0)}s | "
            f"{_fmt(generation.get('llm_duration_sec'), 0)}s | "
            f"{token_text} | {cost_text} | {_fmt(docs.get('pages'), 0)} | "
            f"{_fmt(judge.get('duration_sec'), 0)}s | {judge_token_text} | "
            f"{_fmt(docs_arch.get('mermaid_blocks'), 0)} | "
            f"{_fmt(diagram_types.get('flowchart'), 0)} | "
            f"{_fmt(diagram_types.get('sequenceDiagram'), 0)} | "
            f"{_fmt(docs_arch.get('internal_markdown_links'), 0)} | {models} |"
        )

    if topics:
        systems = sorted(topics)
        all_topics = sorted({name for values in topics.values() for name in values})
        lines += ["", "## Topic averages (0-100)", ""]
        lines.append("| Topic | " + " | ".join(systems) + " |")
        lines.append("|---|" + "|".join("---:" for _ in systems) + "|")
        for topic in all_topics:
            lines.append(
                f"| {topic} | "
                + " | ".join(_fmt(topics[system].get(topic)) for system in systems)
                + " |"
            )

    if failures:
        lines += [
            "",
            "## Failed runs",
            "",
            "| System | Repository | Stage | Meta |",
            "|---|---|---|---|",
        ]
        for failure in failures:
            lines.append(
                f"| {failure.get('system')} | {failure.get('repo')} | {failure.get('stage')} | "
                f"{failure.get('meta')} |"
            )

    lines.append("")
    return "\n".join(lines)


def summarize(cfg: Config, experiment: str) -> dict[str, Any]:
    records, failures = _load_records(cfg, experiment)
    averages = _system_averages(records)
    topics = _topic_averages(records)
    payload = {
        "experiment": experiment,
        "generated_at": now_iso(),
        "records": records,
        "failures": failures,
        "averages": averages,
        "topic_averages": topics,
    }

    out_dir = cfg.runs_dir / safe_component(experiment)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary.json", payload)
    (out_dir / "summary.md").write_text(
        _render_markdown(experiment, records, failures, averages, topics),
        encoding="utf-8",
    )

    csv_path = out_dir / "summary.csv"
    fields = [
        "system",
        "repo",
        "score_100",
        "architecture_score_100",
        "num_leaves",
        "num_errors",
        "num_failed",
        "gen_model",
        "gen_duration_sec",
        "gen_llm_duration_sec",
        "gen_input_tokens",
        "gen_output_tokens",
        "gen_reasoning_tokens",
        "gen_total_tokens",
        "gen_tokens_per_page",
        "gen_cost_usd",
        "gen_cost_status",
        "gen_cost_per_page_usd",
        "docs_pages",
        "judge_duration_sec",
        "judge_input_tokens",
        "judge_output_tokens",
        "mermaid_blocks",
        "flowcharts",
        "sequence_diagrams",
        "internal_links",
        "judge_models",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            generation = record.get("generation") or {}
            judge = record.get("judge") or {}
            judge_tokens = judge.get("tokens") or {}
            docs = (record.get("artifacts") or {}).get("docs") or {}
            tokens = _generation_tokens(record)
            writer.writerow(
                {
                    "system": record.get("system"),
                    "repo": record.get("repo"),
                    "score_100": record.get("score_100"),
                    "architecture_score_100": (record.get("architecture") or {}).get(
                        "score_100"
                    ),
                    "num_leaves": record.get("num_leaves"),
                    "num_errors": record.get("num_errors"),
                    "num_failed": record.get("num_failed"),
                    "gen_model": generation.get("model"),
                    "gen_duration_sec": generation.get("duration_sec"),
                    "gen_llm_duration_sec": generation.get("llm_duration_sec"),
                    "gen_input_tokens": tokens.get("input"),
                    "gen_output_tokens": tokens.get("output"),
                    "gen_reasoning_tokens": tokens.get("reasoning"),
                    "gen_total_tokens": tokens.get("total"),
                    "gen_tokens_per_page": generation.get("tokens_per_page"),
                    "gen_cost_usd": generation.get("cost_usd"),
                    "gen_cost_status": generation.get("cost_status"),
                    "gen_cost_per_page_usd": generation.get("cost_per_page_usd"),
                    "docs_pages": docs.get("pages"),
                    "judge_duration_sec": judge.get("duration_sec"),
                    "judge_input_tokens": judge_tokens.get("input"),
                    "judge_output_tokens": judge_tokens.get("output"),
                    "mermaid_blocks": (docs.get("architecture") or {}).get(
                        "mermaid_blocks"
                    ),
                    "flowcharts": (
                        (docs.get("architecture") or {}).get("diagram_types") or {}
                    ).get("flowchart"),
                    "sequence_diagrams": (
                        (docs.get("architecture") or {}).get("diagram_types") or {}
                    ).get("sequenceDiagram"),
                    "internal_links": (docs.get("architecture") or {}).get(
                        "internal_markdown_links"
                    ),
                    "judge_models": ";".join(judge.get("models") or []),
                }
            )
    return payload
