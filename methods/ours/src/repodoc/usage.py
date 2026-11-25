"""Token, cost and LLM-time accounting for a RepoDoc state directory.

The on-disk format intentionally matches CodeWiki's ``metadata.json`` usage
ledger so the harness and experiment tables can treat both systems the same.
Counts are reconstructed from the actual API responses stored in traces; missing
usage is reported as unknown rather than silently counted as zero.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from repodoc import __version__
from repodoc.llm import ledger
from repodoc.llm.ledger import _number
from repodoc.llm.ledger import usage_counts as _usage_counts
from repodoc.pipeline.manifest import read_manifest
from repodoc.schema import utc_now


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _add_call(
    calls: list[dict[str, Any]],
    *,
    phase: str,
    target: str,
    kind: str,
    usage: Any,
    model: str | None = None,
    requests: int = 1,
    seconds: float | None = None,
    failed: bool = False,
) -> None:
    counts = _usage_counts(usage)
    calls.append(
        {
            "phase": phase,
            "kind": kind,
            "target": target,
            "model": model or "",
            "requests": max(1, int(requests)),
            "seconds": round(float(seconds), 2) if seconds is not None else None,
            "usage_reported": counts is not None,
            "failed": failed,
            **(counts or {}),
        }
    )


def _trace_calls(state: Path) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    trace_dir = state / "docs" / "llm_traces"
    if trace_dir.is_dir():
        for path in sorted(trace_dir.rglob("*.json")):
            row = _read_json(path)
            if not row:
                continue
            response = row.get("response")
            usage = response.get("usage") if isinstance(response, dict) else None
            _add_call(
                calls,
                phase="docs",
                target=str(row.get("page") or path.stem),
                kind=str(row.get("kind") or "page"),
                usage=usage,
                model=str(response.get("model") or "")
                if isinstance(response, dict)
                else None,
                seconds=_number(row.get("elapsed_sec")),
                failed=bool(row.get("failed")),
            )

    changes = state / "changes"
    if changes.is_dir():
        for path in sorted(changes.glob("*.json")):
            name = path.name
            if name.endswith(".llm_plan.json"):
                phase, kind = "planning", "update_plan"
            elif ".patch." in name:
                phase, kind = "patching", "page_patch"
            else:
                continue
            row = _read_json(path)
            if not row:
                continue
            response = row.get("response")
            usage = response.get("usage") if isinstance(response, dict) else None
            _add_call(
                calls,
                phase=phase,
                target=str(
                    row.get("page") or name.split(".patch.")[-1].removesuffix(".json")
                ),
                kind=kind,
                usage=usage,
                model=str(response.get("model") or "")
                if isinstance(response, dict)
                else None,
                seconds=_number(row.get("elapsed_sec")),
                failed=bool(row.get("failed")),
            )
    return calls


def _cluster_call(state: Path) -> dict[str, Any] | None:
    tree = _read_json(state / "components" / "component_tree.json")
    if not tree:
        return None
    meta = tree.get("cluster_meta")
    if not isinstance(meta, dict) or not isinstance(meta.get("usage"), dict):
        return None
    mode = str(meta.get("mode") or "")
    if mode == "llm-hierarchical":
        requests = int(meta.get("chunk_count") or 1)
        if meta.get("merge_mode") == "llm" and not meta.get("merge_fallback"):
            requests += 1
    else:
        requests = 1
    counts = _usage_counts(meta.get("usage")) or {}
    return {
        "phase": "clustering",
        "kind": str(meta.get("mode") or "cluster"),
        "target": "component_tree",
        "model": str(meta.get("model") or ""),
        "requests": max(1, requests),
        "seconds": _number(meta.get("llm_seconds")),
        "usage_reported": bool(counts),
        "failed": False,
        **counts,
    }


def _summarize(calls: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {}
    by_phase: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}
    unpriced: set[str] = set()
    unknown_usage = 0
    for call in calls:
        requests = int(call.get("requests") or 1)
        phase_totals = by_phase.setdefault(
            str(call.get("phase") or "unknown"), {"calls": 0}
        )
        model_totals = by_model.setdefault(
            str(call.get("model") or "<default>"), {"calls": 0}
        )
        phase_totals["calls"] += requests
        model_totals["calls"] += requests
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "total_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
        ):
            value = _number(call.get(key))
            if value is None:
                continue
            totals[key] = totals.get(key, 0) + int(value)
            phase_totals[key] = phase_totals.get(key, 0) + int(value)
            model_totals[key] = model_totals.get(key, 0) + int(value)
        totals["requests"] = totals.get("requests", 0) + requests
        phase_totals["requests"] = phase_totals.get("requests", 0) + requests
        model_totals["requests"] = model_totals.get("requests", 0) + requests
        if not call.get("usage_reported", True):
            unknown_usage += 1
        cost = _number(call.get("cost_usd"))
        if cost is None:
            unpriced.add(str(call.get("model") or "<default>"))
            for group in (phase_totals, model_totals):
                group["unpriced_calls"] = group.get("unpriced_calls", 0) + 1
            continue
        totals["cost_usd"] = totals.get("cost_usd", 0.0) + float(cost)
        phase_totals["cost_usd"] = phase_totals.get("cost_usd", 0.0) + float(cost)
        model_totals["cost_usd"] = model_totals.get("cost_usd", 0.0) + float(cost)

    if totals.get("cost_usd") is not None:
        totals["cost_usd"] = round(float(totals["cost_usd"]), 8)
    for group in (by_phase, by_model):
        for values in group.values():
            if values.get("cost_usd") is not None:
                values["cost_usd"] = round(float(values["cost_usd"]), 8)
    if not calls:
        cost_status = "unknown"
    elif len(unpriced) == 0 and unknown_usage == 0:
        cost_status = "complete"
    elif totals.get("cost_usd") is not None:
        cost_status = "partial"
    else:
        cost_status = "unknown"
    return {
        "schema_version": 2,
        "scope": "this state only (pages resumed from disk are not re-billed)",
        "currency": "USD",
        "cost_status": cost_status,
        "incomplete": unknown_usage > 0,
        "unknown_usage_calls": unknown_usage,
        "n_calls": totals.get("requests", 0),
        "totals": totals,
        "by_phase": by_phase,
        "by_model": by_model,
        "unpriced_models": sorted(unpriced),
        "calls": calls,
    }


def _current_run_usage() -> dict[str, Any] | None:
    entries = ledger.entries()
    if not entries:
        return None
    summary = _summarize(entries)
    summary["run_id"] = ledger.run_id()
    summary["started_at"] = ledger.started_at()
    summary["finished_at"] = utc_now()
    summary["elapsed_sec"] = ledger.elapsed_seconds()
    return summary


def _component_stats(tree: dict[str, Any] | None) -> dict[str, Any]:
    total = 0
    leaves = 0
    max_depth = 0

    def visit(component: dict[str, Any], depth: int) -> None:
        nonlocal total, leaves, max_depth
        total += 1
        max_depth = max(max_depth, depth)
        children = component.get("children") or []
        if children:
            for child in children:
                if isinstance(child, dict):
                    visit(child, depth + 1)
        else:
            leaves += 1

    root = (tree or {}).get("root")
    if isinstance(root, dict):
        visit(root, 0)
    return {"total_components": total, "leaf_nodes": leaves, "max_depth": max_depth}


def _doc_page_count(tree: dict[str, Any] | None) -> int:
    count = 0

    def visit(page: dict[str, Any]) -> None:
        nonlocal count
        count += 1
        for child in page.get("children") or []:
            if isinstance(child, dict):
                visit(child)

    root = (tree or {}).get("root")
    if isinstance(root, dict):
        visit(root)
    return count


def _latest_run_usage(state: Path) -> dict[str, Any] | None:
    ledger_dir = state / "usage" / "runs"
    if not ledger_dir.is_dir():
        return None
    records: list[tuple[str, dict[str, Any]]] = []
    for path in ledger_dir.glob("*.json"):
        data = _read_json(path)
        if isinstance(data, dict) and data.get("totals") is not None:
            records.append((str(data.get("started_at") or ""), data))
    if not records:
        return None
    records.sort(key=lambda item: item[0])
    return records[-1][1]


def collect_usage(state_dir: str | Path, *, scope: str = "state") -> dict[str, Any]:
    """Build a CodeWiki-compatible metadata document for one RepoDoc state.

    ``scope="state"`` reports every call retained in the state (full runs).
    ``scope="latest-run"`` reports the newest per-command ledger, which is what
    an incremental update should use for its own token/cost.
    """
    if scope not in {"state", "latest-run"}:
        raise ValueError(f"unknown usage scope: {scope}")
    state = Path(state_dir).expanduser().resolve()
    manifest = read_manifest(state) or {}
    calls: list[dict[str, Any]] = []
    cluster = _cluster_call(state)
    if cluster:
        calls.append(cluster)
    calls.extend(_trace_calls(state))
    usage = _summarize(calls)
    current_run = _current_run_usage()
    if current_run:
        usage["last_run"] = current_run

    tree = _read_json(state / "components" / "component_tree.json")
    doc_tree = _read_json(state / "docs" / "doc_tree.json")
    docs_dir = state / "docs" / "markdown"
    files_generated = (
        [
            path.relative_to(docs_dir).as_posix()
            for path in sorted(docs_dir.glob("*.md"))
            if path.is_file()
        ]
        if docs_dir.is_dir()
        else []
    )
    model = next(
        (
            str(call.get("model"))
            for call in calls
            if call.get("model") and call.get("model") != "<default>"
        ),
        None,
    )
    statistics = _component_stats(tree)
    statistics["pages"] = _doc_page_count(doc_tree) or len(files_generated)
    if scope == "latest-run":
        latest = _latest_run_usage(state)
        if latest is None:
            raise FileNotFoundError(
                f"no per-run usage ledger under {state / 'usage' / 'runs'}"
            )
        usage = latest
    return {
        "generation_info": {
            "timestamp": utc_now(),
            "provider": "opencode-go",
            "main_model": model,
            "generator_version": __version__,
            "repo_path": manifest.get("repo_path"),
            "commit_id": manifest.get("last_commit"),
            "language": "python",
            "layout": "flat",
        },
        "statistics": statistics,
        "files_generated": files_generated,
        "usage": usage,
    }


def write_usage_metadata(
    state_dir: str | Path, out_path: str | Path, *, scope: str = "state"
) -> Path:
    state = Path(state_dir).expanduser().resolve()
    metadata = collect_usage(state, scope=scope)
    path = Path(out_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    current_run = (metadata.get("usage") or {}).get("last_run")
    if scope == "state" and isinstance(current_run, dict) and current_run.get("run_id"):
        ledger_dir = state / "usage" / "runs"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = ledger_dir / f"{current_run['run_id']}.json"
        ledger_path.write_text(
            json.dumps(current_run, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return path
