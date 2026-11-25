"""LLM update impact planner.

The deterministic graph diff owns the facts: changed files, changed symbols,
and the candidate documentation pages. This planner only chooses documentation
actions inside that universe. It can escalate a local change to a parent or
overview when the supplied evidence shows architectural impact.
"""

from __future__ import annotations

import json
import re
from typing import Any

SYSTEM_PROMPT = """You are a repository documentation update planner.
The deterministic code graph has already identified changed code and a candidate
set of documentation pages. Decide, for each candidate page, whether it must be
left alone, patched only in affected sections, or regenerated.

Return ONLY valid JSON:
{
  "change_scope": "local|module|architecture",
  "decisions": [
    {
      "page": "path/to/page.md",
      "action": "noop|patch|regenerate",
      "sections": ["Heading"],
      "reason": "short evidence-based reason"
    }
  ]
}

Rules:
- Use only page paths listed in CANDIDATE PAGES. Never invent paths.
- PAGES MARKED MANDATORY must not be noop; choose patch or regenerate.
- Prefer noop for pages whose documented responsibilities and interfaces did
  not change.
- Prefer patch when only a few sections need new facts; use regenerate when the
  page's structure, purpose, architecture, or multiple sections changed.
- A local internal implementation or default-value change usually affects the
  directly citing page only. A public API, entrypoint, cross-component
  dependency, lifecycle, or component-boundary change may affect parent pages.
  New/deleted components, changed subsystem boundaries, or changed end-to-end
  flows may affect overview.
- Keep the update focused. Do not regenerate everything.
- Use only evidence in CONTEXT; state uncertainty instead of inventing facts.
- Include a decision for every candidate page you believe should change. Omitted
  non-mandatory pages are treated as noop.
"""


class UpdatePlannerError(RuntimeError):
    pass


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise UpdatePlannerError("LLM update planner returned empty content")
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
        if match:
            text = match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise UpdatePlannerError(f"LLM update planner has no JSON object: {text[:500]}")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise UpdatePlannerError(f"LLM update planner JSON is invalid: {exc}") from exc
    if not isinstance(data, dict):
        raise UpdatePlannerError("LLM update planner response must be a JSON object")
    return data


class UpdatePlannerAgent:
    def __init__(
        self,
        client: Any,
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.last_prompt: str | None = None
        self.last_response: dict[str, Any] | None = None

    def build_prompt(
        self,
        change_set: dict[str, Any],
        candidates: list[dict[str, Any]],
        mandatory_pages: list[str],
        *,
        change_context: str = "",
        max_context_chars: int = 8_000,
    ) -> str:
        delta = change_set.get("graph_delta") or {}
        files = change_set.get("files") or []
        units = change_set.get("change_units") or []
        impact = change_set.get("impact") or {}
        page_lines: list[str] = []
        for page in candidates:
            sections = ", ".join(page.get("sections") or []) or "(none)"
            page_lines.append(
                f"- page={page['path']} level={page.get('level')} "
                f"type={page.get('doc_type')} direct={page.get('direct', False)} "
                f"structural={page.get('structural', False)} "
                f"component={page.get('component', '')}\n"
                f"  sections: {sections}"
            )
        context = (change_context or "")[:max_context_chars]
        parts = [
            f"Change {change_set.get('base_commit', '')[:12]} -> "
            f"{change_set.get('target_commit', '')[:12]}",
            f"CHANGE SCOPE HINT: {impact.get('scope', 'unknown')}",
            f"STRUCTURAL CHANGE: {bool(impact.get('structural', False))}",
            f"ARCHITECTURE SIGNAL: {bool(impact.get('architecture_signal', False))}",
            "",
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
                f"impacted_docs={unit.get('impacted_docs', [])}"
                for unit in units[:30]
            ],
            "",
            "MANDATORY PAGES:",
            *([f"- {path}" for path in mandatory_pages] or ["- none"]),
            "",
            "CANDIDATE PAGES:",
            *(page_lines or ["- none"]),
            "",
            "CHANGED CODE CONTEXT:",
            context or "(no code context)",
            "",
            "Return the planner JSON now.",
        ]
        return "\n".join(parts)

    def plan(
        self,
        change_set: dict[str, Any],
        candidates: list[dict[str, Any]],
        mandatory_pages: list[str],
        *,
        change_context: str = "",
    ) -> tuple[dict[str, str], dict[str, Any]]:
        prompt = self.build_prompt(
            change_set, candidates, mandatory_pages, change_context=change_context
        )
        response = self.client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.last_prompt = prompt
        self.last_response = response
        try:
            content = response["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise UpdatePlannerError(
                "LLM update planner response has no message"
            ) from exc
        data = _extract_json(content)
        raw = data.get("decisions")
        if not isinstance(raw, list):
            raise UpdatePlannerError(
                "LLM update planner response has no `decisions` list"
            )
        allowed = {item["path"] for item in candidates}
        decisions: dict[str, str] = {}
        reasons: dict[str, str] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            page = str(item.get("page") or "")
            action = str(item.get("action") or "")
            if page not in allowed or action not in {"noop", "patch", "regenerate"}:
                continue
            decisions[page] = action
            reasons[page] = str(item.get("reason") or "")
        if not decisions:
            raise UpdatePlannerError(
                "LLM update planner produced no valid page decisions"
            )
        return decisions, {
            "prompt": prompt,
            "response": response,
            "content": content,
            "decisions": decisions,
            "reasons": reasons,
            "change_scope": str(data.get("change_scope") or ""),
        }
