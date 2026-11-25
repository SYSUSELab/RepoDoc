"""LLM triage agent for incremental update decisions.

One bounded request reviews the change set and affected page list, then returns
per-page actions. The MVP applies `noop` or `regenerate`; `patch`/`delete` are
accepted but currently fall back to the deterministic update path.
"""

from __future__ import annotations

import json
import re
from typing import Any

SYSTEM_PROMPT = """You are a documentation maintenance triage agent.
Given a code change set, affected components, and affected documentation pages,
decide which pages should be regenerated. Return ONLY valid JSON.

JSON shape:
{"decisions":[{"page":"path/to/page.md","action":"regenerate|noop|patch|delete","reason":"..."}]}

Rules:
- Use only page paths listed in the input.
- `regenerate` when code facts, APIs, behavior, dependencies, or component structure may have changed.
- `noop` only when the page cannot be affected by this change.
- Prefer a small, precise set of pages; do not regenerate everything by default.
- Do not invent paths or code IDs.
"""


class LLMUpdateError(RuntimeError):
    pass


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise LLMUpdateError("LLM update triage returned empty content")
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
        if match:
            text = match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise LLMUpdateError(f"LLM update triage has no JSON object: {text[:500]}")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMUpdateError(f"LLM update triage JSON is invalid: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMUpdateError("LLM update triage response must be a JSON object")
    return data


class UpdateTriageAgent:
    def __init__(
        self,
        client: Any,
        *,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def build_prompt(self, change_set: dict[str, Any], page_paths: list[str]) -> str:
        delta = change_set.get("graph_delta") or {}
        files = change_set.get("files") or []
        units = change_set.get("change_units") or []
        parts = [
            f"Change {change_set.get('base_commit', '')[:12]} -> "
            f"{change_set.get('target_commit', '')[:12]}",
            "CHANGED FILES:",
            *[f"- {item.get('status')}: {item.get('path')}" for item in files],
            "",
            "GRAPH DELTA COUNTS:",
            *[
                f"- {key}: {len(delta.get(key) or [])}"
                for key in (
                    "nodes_added",
                    "nodes_removed",
                    "nodes_modified",
                    "edges_added",
                    "edges_removed",
                )
            ],
            "",
            "CHANGE UNITS:",
            *[
                f"- {unit.get('classification')} {unit.get('path')} "
                f"targets={len(unit.get('targets') or [])}"
                for unit in units[:20]
            ],
            "",
            "AFFECTED PAGES:",
            *[f"- {path}" for path in page_paths],
            "",
            "Return the decisions JSON now.",
        ]
        return "\n".join(parts)

    def decide(
        self, change_set: dict[str, Any], page_paths: list[str]
    ) -> tuple[dict[str, str], dict[str, Any]]:
        prompt = self.build_prompt(change_set, page_paths)
        response = self.client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        try:
            content = response["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUpdateError("LLM update triage response has no message") from exc
        data = _extract_json(content)
        decisions_raw = data.get("decisions")
        if not isinstance(decisions_raw, list):
            raise LLMUpdateError("LLM update triage response has no `decisions` list")
        allowed = set(page_paths)
        decisions: dict[str, str] = {}
        reasons: dict[str, str] = {}
        for item in decisions_raw:
            if not isinstance(item, dict):
                continue
            page = str(item.get("page") or "")
            action = str(item.get("action") or "")
            if page not in allowed or action not in {
                "regenerate",
                "noop",
                "patch",
                "delete",
            }:
                continue
            decisions[page] = action
            reasons[page] = str(item.get("reason") or "")
        if not decisions:
            raise LLMUpdateError("LLM update triage produced no valid decisions")
        return decisions, {
            "prompt": prompt,
            "response": response,
            "content": content,
            "reasons": reasons,
        }
