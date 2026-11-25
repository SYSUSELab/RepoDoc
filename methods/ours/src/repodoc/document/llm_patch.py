"""Section-level LLM content patching for incremental updates.

The model receives one page's sections plus the code-change summary and returns
structured replacements for only the affected sections. The caller validates
section IDs and old content hashes before applying anything.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from repodoc.graph.store import GraphStore
from repodoc.schema import sha256_text

SYSTEM_PROMPT = """You are a documentation patch editor.
Update only the documentation sections that are affected by the supplied code change.
Return ONLY one valid JSON object. Never wrap the entire response in a markdown
code fence. `new_content` is a JSON string; escape newlines as \n. Fenced
```mermaid blocks are allowed inside `new_content`; preserve them when the
section contains architecture diagrams.

JSON shape:
{"patches":[{"section_id":"section:...","old_content_hash":"...","new_content":"...","reason":"..."}]}

Rules:
- Use only section IDs listed in the input.
- Copy `old_content_hash` exactly from the section being replaced.
- `new_content` is the complete replacement markdown for that one section.
- Preserve existing facts, links, and citations unless the code change requires a change.
- Return an empty `patches` list only when no section is affected.
- If changed code affects a section, return at least one patch for it.
- If unsure, return {"patches":[]} rather than free-form text or code.
- Do not invent APIs, files, or line references.
"""


class LLMPatchError(RuntimeError):
    pass


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise LLMPatchError("LLM patch returned empty content")

    def parse(candidate: str) -> dict[str, Any] | None:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    # Do not strip an inner Mermaid fence when the response itself is JSON.
    data = parse(text)
    if data is not None:
        return data
    if text.startswith("```"):
        outer = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.I)
        outer = re.sub(r"\s*```\s*$", "", outer, count=1)
        data = parse(outer.strip())
        if data is not None:
            return data
        text = outer
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise LLMPatchError(f"LLM patch has no JSON object: {text[:500]}")
    data = parse(text[start : end + 1])
    if data is None:
        try:
            json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMPatchError(f"LLM patch JSON is invalid: {exc}") from exc
        raise LLMPatchError("LLM patch response must be a JSON object")
    return data


def build_patch_prompt(
    page: dict[str, Any],
    section_rows: list[dict[str, Any]],
    change_set: dict[str, Any],
    *,
    max_section_chars: int = 12_000,
    section_ids: list[str] | None = None,
) -> str:
    units = change_set.get("change_units") or []
    delta = change_set.get("graph_delta") or {}
    parts = [
        f"PAGE: {page['title']} ({page['path']})",
        f"LEVEL: {page['level']}  DOC_TYPE: {page['doc_type']}",
        "",
        "CODE CHANGE SUMMARY:",
        *[
            f"- {unit.get('classification')} {unit.get('path')} "
            f"targets={len(unit.get('targets') or [])}"
            for unit in units[:20]
        ],
        "",
        "CHANGED CODE:",
        str(change_set.get("code_context") or "(no code snippet available)"),
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
        "AFFECTED SECTIONS (patch only these):",
        *[f"- {section_id}" for section_id in (section_ids or [])],
        "",
        "CURRENT SECTIONS:",
    ]
    allowed = set(section_ids or [])
    for row in section_rows:
        if allowed and row["id"] not in allowed:
            continue
        content = str(row.get("content") or "")
        if len(content) > max_section_chars:
            content = content[:max_section_chars] + "\n... [truncated]"
        parts.extend(
            [
                "",
                f"### SECTION {row['id']}",
                f"heading: {row.get('heading', '')}",
                f"content_hash: {row.get('content_hash', '')}",
                "content:",
                content,
            ]
        )
    parts.extend(["", "Return the patches JSON now."])
    return "\n".join(parts)


class LLMContentPatchAgent:
    def __init__(
        self,
        client: Any,
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        max_section_chars: int = 12_000,
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_section_chars = max_section_chars
        self.last_prompt: str | None = None
        self.last_response: dict[str, Any] | None = None
        self.last_content: str | None = None

    def propose(
        self,
        page: dict[str, Any],
        section_rows: list[dict[str, Any]],
        change_set: dict[str, Any],
        *,
        section_ids: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prompt = build_patch_prompt(
            page,
            section_rows,
            change_set,
            max_section_chars=self.max_section_chars,
            section_ids=section_ids,
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
            raise LLMPatchError("LLM patch response has no message") from exc
        self.last_content = content
        data = _extract_json(content)
        patches = data.get("patches")
        if not isinstance(patches, list):
            raise LLMPatchError("LLM patch response has no `patches` list")
        return patches, {
            "prompt": prompt,
            "response": response,
            "content": content,
            "page": page["path"],
        }


def apply_section_patches(
    graph: GraphStore,
    page: dict[str, Any],
    section_rows: list[dict[str, Any]],
    patches: list[dict[str, Any]],
    *,
    docs_dir: Path,
    source_graph_version: str,
) -> tuple[int, list[str]]:
    """Apply validated replacements; return (applied_count, warnings)."""
    rows_by_id = {row["id"]: row for row in section_rows}
    page_sections = {section["id"]: section for section in page.get("sections", [])}
    applied = 0
    warnings: list[str] = []
    for patch in patches:
        if not isinstance(patch, dict):
            warnings.append("ignored non-object patch")
            continue
        section_id = str(patch.get("section_id") or "")
        row = rows_by_id.get(section_id)
        meta = page_sections.get(section_id)
        if row is None:
            warnings.append(f"missing section row: {section_id}")
            continue
        if meta is None:
            warnings.append(f"missing page section metadata: {section_id}")
            continue
        if str(patch.get("old_content_hash") or "") != str(
            row.get("content_hash") or ""
        ):
            warnings.append(f"stale content hash for section: {section_id}")
            continue
        new_content = str(patch.get("new_content") or "").strip()
        if not new_content:
            warnings.append(f"empty replacement for section: {section_id}")
            continue
        new_content += "\n"
        new_hash = sha256_text(new_content)
        row["content"] = new_content
        row["content_hash"] = new_hash
        meta["content_hash"] = new_hash
        node = graph.nodes.get(section_id)
        if node is not None:
            node["attrs"]["content"] = new_content
            node["attrs"]["content_hash"] = new_hash
            node["attrs"]["source_graph_version"] = source_graph_version
        applied += 1

    if applied:
        lines = [f"# {page['title']}", ""]
        for section in page.get("sections", []):
            row = rows_by_id.get(section["id"])
            content = row.get("content", "") if row else ""
            lines.extend([f"## {section['heading']}", "", content.strip(), ""])
        markdown = "\n".join(lines).rstrip() + "\n"
        (docs_dir / page["path"]).write_text(markdown, encoding="utf-8")
        page["content_hash"] = sha256_text(markdown)
        page["status"] = "generated"
        graph.add_node(
            "doc_page",
            page["id"],
            {
                "path": page["path"],
                "title": page["title"],
                "level": page["level"],
                "doc_type": page["doc_type"],
                "component_id": page["component_id"],
                "status": page["status"],
                "content_hash": page["content_hash"],
                "source_graph_version": source_graph_version,
                "section_count": len(page.get("sections", [])),
            },
        )
    return applied, warnings
