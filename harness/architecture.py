"""Architecture-focused views over CodeWikiBench rubrics and generated docs.

CodeWikiBench's rubric tree is hierarchical. The benchmark does not store a
single fixed "Architecture" topic label, so this module provides an explicit,
reproducible view: select leaf requirements whose text describes structural
or interaction concerns (architecture, components, flows, lifecycle, ...),
then recompute a weighted score over those leaves. The full benchmark score is
left untouched.

The module also reports reference-free structural signals (Mermaid diagrams,
internal links, architecture headings) so a high rubric score cannot hide a
top-level document that is only an API inventory.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Iterable

DEFAULT_KEYWORDS = [
    "architecture",
    "architectural",
    "component",
    "module",
    "interface",
    "pipeline",
    "workflow",
    "flow",
    "lifecycle",
    "layer",
    "boundary",
    "depend",
    "event",
    "protocol",
    "orchestrat",
    "system",
    "framework",
    "runtime",
    "integration",
    "sandbox",
    "backend",
    "frontend",
    "service",
    "design",
]

_ARCHITECTURE_HEADING = re.compile(
    r"architect|component|module|interface|pipeline|flow|lifecycle|design|"
    r"boundary|system|overview|runtime",
    re.I,
)
_MERMAID_BLOCK = re.compile(r"```mermaid[ \t]*\r?\n(.*?)```", re.S | re.I)
_INTERNAL_MD_LINK = re.compile(r"\[[^\]]*\]\(([^)]+\.md)(?:#[^)]*)?\)", re.I)
_HEADING = re.compile(r"^#{1,4}\s+(.+?)\s*$", re.M)


def _topic_match(name: str, topics: Iterable[str] | None) -> bool:
    if not topics:
        return False
    lowered = name.casefold()
    return any(
        str(topic).casefold() in lowered for topic in topics if str(topic).strip()
    )


def _keyword_pattern(keywords: Iterable[str] | None) -> re.Pattern[str] | None:
    if keywords is None:
        keywords = DEFAULT_KEYWORDS
    terms = [str(term).strip() for term in keywords if str(term).strip()]
    if not terms:
        return None
    return re.compile("|".join(re.escape(term) for term in terms), re.I)


def _walk_leaves(
    items: Iterable[dict[str, Any]],
    path: tuple[int, ...] = (),
    ancestors: tuple[str, ...] = (),
) -> Iterable[tuple[tuple[int, ...], dict[str, Any], tuple[str, ...]]]:
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        item_path = (*path, index)
        name = str(item.get("requirements") or item.get("name") or "")
        children = item.get("sub_tasks") or []
        if children:
            yield from _walk_leaves(children, item_path, (*ancestors, name))
        else:
            yield item_path, item, ancestors


def _weight(item: dict[str, Any]) -> float:
    return float(item.get("weight", 1) or 0)


def _score(item: dict[str, Any]) -> float:
    return float(item.get("score", 0) or 0)


def _weighted_average(items: Iterable[dict[str, Any]]) -> float:
    items = list(items)
    total = sum(_weight(item) for item in items)
    if not total:
        return 0.0
    return sum(_score(item) * _weight(item) for item in items) / total


def architecture_metrics(
    results: dict[str, Any],
    *,
    keywords: Iterable[str] | None = None,
    include_topics: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compute an architecture-focused score from scored CodeWikiBench results.

    A leaf is selected when its requirement text matches one of the keywords,
    or when its top-level topic is listed in ``include_topics``. Scores are
    recomputed recursively so each rubric level keeps its original weight.
    """
    rubrics = results.get("rubrics", []) if isinstance(results, dict) else []
    if not isinstance(rubrics, list):
        rubrics = []
    pattern = _keyword_pattern(keywords)

    def leaf_selected(leaf: dict[str, Any], force: bool) -> bool:
        if force or pattern is None:
            return True
        text = str(leaf.get("requirements") or leaf.get("name") or "")
        return bool(pattern.search(text))

    def selected_score(node: dict[str, Any], force: bool) -> float | None:
        children = node.get("sub_tasks") or []
        if not children:
            return _score(node) if leaf_selected(node, force) else None
        selected: list[tuple[dict[str, Any], float]] = []
        for child in children:
            score = selected_score(child, force)
            if score is not None:
                selected.append((child, score))
        if not selected:
            return None
        total_weight = sum(_weight(child) for child, _score_value in selected)
        if not total_weight:
            return 0.0
        return sum(score * _weight(child) for child, score in selected) / total_weight

    def count_selected(node: dict[str, Any], force: bool) -> int:
        children = node.get("sub_tasks") or []
        if not children:
            return 1 if leaf_selected(node, force) else 0
        return sum(count_selected(child, force) for child in children)

    topics: list[dict[str, Any]] = []
    total_leaves = 0
    for topic in rubrics:
        if not isinstance(topic, dict):
            continue
        topic_leaves = len(list(_walk_leaves([topic])))
        total_leaves += topic_leaves
        name = str(
            topic.get("requirements") or topic.get("name") or f"topic-{len(topics)}"
        )
        force = _topic_match(name, include_topics)
        score = selected_score(topic, force)
        selected_count = count_selected(topic, force)
        if score is None or selected_count == 0:
            continue
        topics.append(
            {
                "name": name,
                "weight": _weight(topic),
                "score": score,
                "score_100": round(score * 100, 2),
                "selected_leaves": selected_count,
                "total_leaves": topic_leaves,
            }
        )
    topics.sort(key=lambda item: (-item["score"], item["name"]))
    selected_count = sum(item["selected_leaves"] for item in topics)
    total_topic_weight = sum(item["weight"] for item in topics)
    score = (
        sum(item["score"] * item["weight"] for item in topics) / total_topic_weight
        if total_topic_weight
        else 0.0
    )
    return {
        "score": score,
        "score_100": round(score * 100, 2),
        "num_leaves": selected_count,
        "total_leaves": total_leaves,
        "coverage": round(selected_count / total_leaves, 4) if total_leaves else 0.0,
        "topics": topics,
        "keywords": list(DEFAULT_KEYWORDS if keywords is None else keywords),
        "include_topics": list(include_topics or []),
        "method": "keyword_filter+topics",
    }


def filter_architecture_rubrics(
    results: dict[str, Any],
    *,
    keywords: Iterable[str] | None = None,
    include_topics: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return a rubric tree containing only selected architecture leaves."""
    pattern = _keyword_pattern(keywords)
    source = results.get("rubrics", []) if isinstance(results, dict) else []
    if not isinstance(source, list):
        source = []

    def prune(node: dict[str, Any], force: bool = False) -> dict[str, Any] | None:
        children = node.get("sub_tasks") or []
        if not children:
            text = str(node.get("requirements") or node.get("name") or "")
            return (
                copy.deepcopy(node)
                if force or pattern is None or pattern.search(text)
                else None
            )
        kept = [child for item in children if (child := prune(item, force)) is not None]
        if not kept:
            return None
        result = copy.deepcopy(node)
        result["sub_tasks"] = kept
        return result

    rubrics = [
        item
        for node in source
        if (
            item := prune(
                node,
                _topic_match(
                    str(node.get("requirements") or node.get("name") or ""),
                    include_topics,
                ),
            )
        )
        is not None
    ]
    metadata = copy.deepcopy(
        results.get("combination_metadata", {}) if isinstance(results, dict) else {}
    )
    metadata["architecture_filter"] = {
        "keywords": list(DEFAULT_KEYWORDS if keywords is None else keywords),
        "include_topics": list(include_topics or []),
        "method": "keyword_filter+topics",
    }
    return {"rubrics": rubrics, "combination_metadata": metadata}


def _diagram_type(block: str) -> str:
    first = next((line.strip() for line in block.splitlines() if line.strip()), "")
    if re.match(r"^(?:flowchart|graph)\b", first):
        return "flowchart"
    if first.startswith("sequenceDiagram"):
        return "sequenceDiagram"
    if first.startswith("classDiagram"):
        return "classDiagram"
    if first.startswith("stateDiagram"):
        return "stateDiagram"
    if first.startswith("erDiagram"):
        return "erDiagram"
    return "other"


def docs_architecture_stats(docs_dir: str | Path) -> dict[str, Any]:
    """Reference-free structural signals for a flat documentation directory."""
    docs = Path(docs_dir)
    files = sorted(p for p in docs.glob("*.md") if p.is_file())
    diagram_types: dict[str, int] = {}
    pages_with_diagrams = 0
    internal_links = 0
    architecture_headings = 0
    total_headings = 0
    total_words = 0
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        total_words += len(text.split())
        blocks = _MERMAID_BLOCK.findall(text)
        if blocks:
            pages_with_diagrams += 1
        for block in blocks:
            kind = _diagram_type(block)
            diagram_types[kind] = diagram_types.get(kind, 0) + 1
        internal_links += len(_INTERNAL_MD_LINK.findall(text))
        headings = _HEADING.findall(text)
        total_headings += len(headings)
        architecture_headings += sum(
            1 for heading in headings if _ARCHITECTURE_HEADING.search(heading)
        )
    return {
        "pages": len(files),
        "words": total_words,
        "mermaid_blocks": sum(diagram_types.values()),
        "diagram_types": dict(sorted(diagram_types.items())),
        "pages_with_diagrams": pages_with_diagrams,
        "diagram_page_ratio": round(pages_with_diagrams / len(files), 4)
        if files
        else 0.0,
        "internal_markdown_links": internal_links,
        "headings": total_headings,
        "architecture_headings": architecture_headings,
        "architecture_heading_ratio": round(architecture_headings / total_headings, 4)
        if total_headings
        else 0.0,
    }


def render_markdown(
    metrics: dict[str, Any], docs_stats: dict[str, Any] | None = None
) -> str:
    lines = [
        "# Architecture-focused evaluation",
        "",
        "Selection: architecture keywords plus explicitly included topics.",
        "",
        f"- Architecture score: **{metrics.get('score_100', 0):.2f}/100**",
        f"- Selected leaves: {metrics.get('num_leaves', 0)} / {metrics.get('total_leaves', 0)}"
        f" (coverage {float(metrics.get('coverage', 0)) * 100:.1f}%)",
        "",
        "## Architecture topics",
        "",
        "| Topic | Weight | Selected / Total | Score |",
        "|---|---:|---:|---:|",
    ]
    for topic in metrics.get("topics") or []:
        lines.append(
            f"| {topic.get('name')} | {topic.get('weight')} | "
            f"{topic.get('selected_leaves')} / {topic.get('total_leaves')} | "
            f"{float(topic.get('score_100', 0)):.2f} |"
        )
    if docs_stats:
        lines += [
            "",
            "## Documentation structure",
            "",
            f"- Pages: {docs_stats.get('pages', 0)}",
            f"- Mermaid blocks: {docs_stats.get('mermaid_blocks', 0)}"
            f" ({docs_stats.get('pages_with_diagrams', 0)} pages)",
            f"- Diagram types: `{docs_stats.get('diagram_types', {})}`",
            f"- Internal markdown links: {docs_stats.get('internal_markdown_links', 0)}",
            f"- Architecture headings: {docs_stats.get('architecture_headings', 0)}"
            f" / {docs_stats.get('headings', 0)}",
        ]
    lines.append("")
    return "\n".join(lines)
