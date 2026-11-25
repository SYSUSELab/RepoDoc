"""Process-local LLM usage recorder.

The state-level ledger in ``repodoc.usage`` is reconstructed from traces and is
therefore cumulative. This recorder captures exactly the calls made by the
current CLI invocation, which is what an incremental update should report.
"""

from __future__ import annotations

import threading
import time
import uuid
from datetime import UTC, datetime
from typing import Any


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def usage_counts(raw: Any) -> dict[str, Any] | None:
    """Normalize one OpenAI-compatible usage object."""
    if not isinstance(raw, dict):
        return None
    prompt = _number(raw.get("prompt_tokens"))
    completion = _number(raw.get("completion_tokens"))
    if prompt is None and completion is None:
        return None
    prompt = int(prompt or 0)
    completion = int(completion or 0)
    opencode = raw.get("opencode") if isinstance(raw.get("opencode"), dict) else {}
    detail_tokens = (
        opencode.get("tokens") if isinstance(opencode.get("tokens"), dict) else {}
    )
    details = (
        raw.get("completion_tokens_details")
        if isinstance(raw.get("completion_tokens_details"), dict)
        else {}
    )
    reasoning = _number(details.get("reasoning_tokens"))
    if reasoning is None:
        reasoning = _number(detail_tokens.get("reasoning"))
    cost = _number(opencode.get("list_cost"))
    if cost is None:
        cost = _number(raw.get("cost_usd"))
    cache_read = _number(detail_tokens.get("cache_read"))
    if cache_read is None:
        prompt_details = (
            raw.get("prompt_tokens_details")
            if isinstance(raw.get("prompt_tokens_details"), dict)
            else {}
        )
        cache_read = _number(prompt_details.get("cached_tokens"))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "reasoning_tokens": int(reasoning or 0),
        "total_tokens": int(_number(raw.get("total_tokens")) or prompt + completion),
        "cache_read_tokens": int(cache_read or 0),
        "cache_write_tokens": int(_number(detail_tokens.get("cache_write_5m")) or 0),
        "cost_usd": cost,
    }


_LOCK = threading.Lock()
_ENTRIES: list[dict[str, Any]] = []
_RUN_ID = ""
_STARTED_AT = ""


def reset() -> None:
    """Start a new per-command ledger."""
    global _RUN_ID, _STARTED_AT
    with _LOCK:
        _ENTRIES.clear()
        _RUN_ID = (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        )
        _STARTED_AT = datetime.now(UTC).isoformat()


def _append(
    *,
    counts: dict[str, Any] | None,
    model: str,
    seconds: float,
    error: str | None,
) -> None:
    entry = {
        "phase": "llm",
        "kind": "chat",
        "target": "",
        "model": model,
        "requests": 1,
        "seconds": round(max(0.0, seconds), 3),
        "usage_reported": counts is not None,
        "failed": bool(error),
        **(counts or {}),
    }
    if error:
        entry["error"] = error
    with _LOCK:
        _ENTRIES.append(entry)


def record_response(response: dict[str, Any], *, seconds: float, model: str) -> None:
    usage = response.get("usage") if isinstance(response, dict) else None
    _append(counts=usage_counts(usage), model=model, seconds=seconds, error=None)


def record_error(*, seconds: float, model: str, error: str) -> None:
    _append(counts=None, model=model, seconds=seconds, error=error)


def entries() -> list[dict[str, Any]]:
    with _LOCK:
        return [dict(entry) for entry in _ENTRIES]


def run_id() -> str:
    return _RUN_ID


def started_at() -> str:
    return _STARTED_AT


def elapsed_seconds() -> float:
    if not _STARTED_AT:
        return 0.0
    try:
        start = datetime.fromisoformat(_STARTED_AT)
    except ValueError:
        return 0.0
    return round((datetime.now(UTC) - start).total_seconds(), 3)


def monotonic() -> float:
    return time.monotonic()
