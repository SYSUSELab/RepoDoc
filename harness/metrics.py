"""Score extraction and run metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .architecture import architecture_metrics, docs_architecture_stats
from .util import read_json, read_text


def is_leaf(item: dict[str, Any]) -> bool:
    return not item.get("sub_tasks")


def walk_leaves(
    items: Iterable[dict[str, Any]], prefix: str = ""
) -> list[tuple[str, dict[str, Any]]]:
    leaves: list[tuple[str, dict[str, Any]]] = []
    for index, item in enumerate(items):
        path = f"{prefix}.{index}" if prefix else str(index)
        children = item.get("sub_tasks") or []
        if children:
            leaves.extend(walk_leaves(children, path))
        else:
            leaves.append((path, item))
    return leaves


def node_score(item: dict[str, Any]) -> float:
    """Use the recorded score, or recompute a parent from its children."""
    children = item.get("sub_tasks") or []
    if children:
        return weighted_score(children)
    return float(item.get("score", 0) or 0)


def weighted_score(items: Iterable[dict[str, Any]]) -> float:
    total_weight = 0.0
    total_score = 0.0
    for item in items:
        weight = float(item.get("weight", 1) or 0)
        total_weight += weight
        total_score += node_score(item) * weight
    return total_score / total_weight if total_weight else 0.0


def extract_results_metrics(
    results: dict[str, Any],
    *,
    architecture_keywords: list[str] | None = None,
    architecture_include_topics: list[str] | None = None,
) -> dict[str, Any]:
    rubrics = results.get("rubrics", []) if isinstance(results, dict) else []
    if not isinstance(rubrics, list):
        rubrics = []
    leaves = walk_leaves(rubrics)
    errors = 0
    token_input = 0
    token_output = 0
    for _, leaf in leaves:
        evaluation = leaf.get("evaluation") or {}
        if (
            evaluation.get("error")
            or "[evaluation error]" in str(evaluation.get("reasoning", "")).lower()
        ):
            errors += 1
        tokens = evaluation.get("tokens") or {}
        token_input += int(tokens.get("input", 0) or 0)
        token_output += int(tokens.get("output", 0) or 0)

    topics = []
    for item in rubrics:
        topics.append(
            {
                "name": item.get("requirements") or item.get("name") or "unnamed",
                "weight": item.get("weight", 1),
                "score": node_score(item),
            }
        )

    return {
        "score": weighted_score(rubrics),
        "score_100": round(weighted_score(rubrics) * 100, 2),
        "num_leaves": len(leaves),
        "num_errors": errors,
        "topics": topics,
        "architecture": architecture_metrics(results, keywords=architecture_keywords),
        "judge_tokens": {"input": token_input, "output": token_output},
        "combination_metadata": results.get("combination_metadata", {}),
    }


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def normalize_tokens(raw: Any) -> dict[str, int]:
    """Normalize provider/CodeWiki/adapter token aliases to one schema.

    Canonical keys are ``input``, ``output``, ``total`` and ``reasoning``.
    Reasoning tokens are a subset of output tokens and are never added again.
    """
    if not isinstance(raw, dict):
        return {}
    input_tokens = _number(
        _first(raw, "input", "input_tokens", "prompt", "prompt_tokens")
    )
    output_tokens = _number(
        _first(raw, "output", "output_tokens", "completion", "completion_tokens")
    )
    reasoning = _number(_first(raw, "reasoning", "reasoning_tokens"))
    if reasoning is None:
        details = raw.get("completion_tokens_details")
        if isinstance(details, dict):
            reasoning = _number(details.get("reasoning_tokens"))
    opencode = raw.get("opencode")
    if isinstance(opencode, dict) and reasoning is None:
        detail_tokens = opencode.get("tokens")
        if isinstance(detail_tokens, dict):
            reasoning = _number(detail_tokens.get("reasoning"))

    result: dict[str, int] = {}
    if input_tokens is not None:
        result["input"] = int(input_tokens)
    if output_tokens is not None:
        result["output"] = int(output_tokens)
    if reasoning is not None:
        result["reasoning"] = int(reasoning)
    total = _number(_first(raw, "total", "total_tokens"))
    if total is None and input_tokens is not None and output_tokens is not None:
        total = input_tokens + output_tokens
    if total is not None:
        result["total"] = int(total)
    requests = _number(_first(raw, "requests", "n_calls", "calls"))
    if requests is not None:
        result["requests"] = int(requests)
    for key in ("cache_read_tokens", "cache_write_tokens"):
        value = _number(raw.get(key))
        if value is not None:
            result[key] = int(value)
    return result


def usage_stats(metadata: dict[str, Any]) -> dict[str, Any]:
    """Extract a comparable usage block from CodeWiki/Repodoc metadata."""
    usage = metadata.get("usage")
    if not isinstance(usage, dict):
        return {}
    totals = usage.get("totals")
    if not isinstance(totals, dict):
        # Adapters may write flat usage fields without a `totals` wrapper.
        totals = usage
    tokens = normalize_tokens(totals)
    calls = usage.get("calls")
    llm_seconds: float | None = None
    if isinstance(calls, list):
        seconds = 0.0
        found = False
        for call in calls:
            if not isinstance(call, dict):
                continue
            value = _number(call.get("seconds"))
            if value is None:
                value = _number(call.get("elapsed_sec"))
            if value is not None:
                seconds += float(value)
                found = True
        if found:
            llm_seconds = round(seconds, 2)
    if llm_seconds is None:
        llm_seconds = _number(usage.get("llm_seconds"))

    cost = _number(totals.get("cost_usd"))
    if cost is None:
        cost = _number(usage.get("cost_usd"))
    cost_status = usage.get("cost_status")
    requests = tokens.get("requests")
    if requests is None:
        requests = _number(usage.get("n_calls"))
    if cost_status is None:
        if requests:
            cost_status = "complete" if cost is not None else "unknown"
        else:
            cost_status = "unavailable"

    return {
        "usage_schema_version": usage.get("schema_version"),
        "currency": usage.get("currency"),
        "usage_status": usage.get("cost_status") or cost_status,
        "cost_status": cost_status,
        "usage_incomplete": usage.get("incomplete"),
        "n_calls": usage.get("n_calls") or requests,
        "unknown_usage_calls": usage.get("unknown_usage_calls"),
        "tokens": tokens,
        "cost_usd": cost,
        "llm_duration_sec": llm_seconds,
        "by_phase": usage.get("by_phase"),
        "by_model": usage.get("by_model"),
        "unpriced_models": usage.get("unpriced_models"),
    }


def docs_stats(docs_dir: Path) -> dict[str, Any]:
    if not docs_dir.is_dir():
        return {}
    files = sorted(p for p in docs_dir.glob("*.md") if p.is_file())
    text = "\n".join(read_text(p) for p in files)
    return {
        "pages": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "words": len(text.split()),
        "architecture": docs_architecture_stats(docs_dir),
    }


def generation_stats(docs_dir: Path) -> dict[str, Any]:
    """Pull comparable model/token/cost/time provenance from metadata.json."""
    metadata = read_json(docs_dir / "metadata.json")
    if not isinstance(metadata, dict):
        return {
            "usage_status": "unavailable",
            "cost_status": "unavailable",
            "usage_source": "metadata.json missing",
        }
    generation_info = metadata.get("generation_info") or {}
    stats = {
        "model": generation_info.get("main_model"),
        "generator_version": generation_info.get("generator_version"),
        "generation_commit": generation_info.get("commit_id"),
        "language": generation_info.get("language"),
        "usage_source": "docs/metadata.json",
    }
    stats.update(usage_stats(metadata))
    return stats
