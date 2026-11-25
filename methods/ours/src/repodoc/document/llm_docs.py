"""Stage-three LLM document generation for selected pages.

The rest of the document tree keeps its deterministic content. This module
re-generates one page at a time with a bounded context and writes the result
back to markdown, the doc tree, and the document nodes in the graph.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from repodoc.graph.store import GraphStore
from repodoc.pipeline.manifest import read_manifest, write_manifest
from repodoc.llm.budget import ComputedBudget, compute_budget
from repodoc.schema import (
    doc_section_id,
    sha256_text,
    stable_hash,
    utc_now,
    validate,
)

SYSTEM_PROMPT = """You are a senior software documentation writer.
Write documentation grounded strictly in the supplied repository context.
Return ONLY valid JSON. Do not include markdown fences or prose outside JSON.

JSON shape:
{
  "sections": [
    {
      "heading": "Overview",
      "level": 2,
      "content": "markdown text for this section; may contain fenced ```mermaid blocks```",
      "covers": ["code:..."],
      "citations": ["path/to/file.py:12-30"]
    }
  ]
}

Rules:
- For `covers`, copy exact IDs shown in the context. Never use bare names or path:line strings.
- `citations` must be file:line-line references grounded in the context.
- Do not invent APIs, classes, behavior, source locations, or links.
- Low-level pages explain responsibility, public API, implementation, and dependencies.
- High-level pages are architecture documents, not API inventories. They explain the
  purpose and boundaries of the system, how components interact, how control and data
  move through the system, why the design is shaped that way, and the main trade-offs
  and cross-cutting concerns. Mention only the entry points needed to explain a flow.
- High-level pages must include at least one fenced ```mermaid diagram in a section's
  `content`. A repository overview must contain both an end-to-end `flowchart` and a
  `sequenceDiagram` for a critical flow. Subsystem and parent-module pages must contain
  a `flowchart`, and should also contain a `sequenceDiagram` when the evidence supports
  a runtime flow. Do not add diagrams to low-level implementation pages merely for
  decoration.
- Do not copy child documentation verbatim. Synthesize it and link to child pages
  using the paths shown in the context.
- Mermaid syntax: use simple alphanumeric node IDs, quote labels that contain
  punctuation, avoid semicolons in sequence messages, and keep diagrams focused.
- Keep headings concise. Use 2-4 sections for low-level pages and 4-5 sections for
  high-level pages.
"""


class LLMDocError(RuntimeError):
    pass


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise LLMDocError("LLM returned empty document content")

    def parse(candidate: str) -> dict[str, Any] | None:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    # A JSON response may itself contain fenced Mermaid blocks. Only strip an
    # outer Markdown fence; never treat the first inner ``` block as the
    # response envelope.
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
        raise LLMDocError(
            f"LLM document response contains no JSON object: {text[:500]}"
        )
    data = parse(text[start : end + 1])
    if data is None:
        try:
            json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMDocError(f"LLM document JSON is invalid: {exc}") from exc
        raise LLMDocError("LLM document response must be a JSON object")
    return data


def _citation(graph: GraphStore, node_id: str) -> str | None:
    node = graph.nodes.get(node_id)
    if node is None:
        return None
    attrs = node["attrs"]
    if node["type"] == "symbol":
        return f"{attrs['path']}:{attrs['start']['line']}-{attrs['end']['line']}"
    if node["type"] == "file":
        return attrs.get("path")
    if node["type"] == "api":
        for edge in graph.outgoing(node_id, {"handles"}):
            return _citation(graph, edge["dst"])
    return None


def _component_map(component_tree: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}

    def visit(component: dict[str, Any]) -> None:
        mapping[component["id"]] = component
        for child in component.get("children", []):
            visit(child)

    visit(component_tree["root"])
    return mapping


def _page_map(doc_tree: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}

    def visit(page: dict[str, Any]) -> None:
        keys = {
            page["id"],
            page["path"],
            Path(page["path"]).stem,
            page["title"],
            page["title"].lower(),
        }
        for key in keys:
            mapping.setdefault(str(key), page)
        for child in page.get("children", []):
            visit(child)

    visit(doc_tree["root"])
    return mapping


def _docs_dir(state_dir: Path, manifest: dict[str, Any]) -> Path:
    value = (manifest.get("artifacts") or {}).get("markdown")
    if not value:
        return state_dir / "docs" / "markdown"
    path = Path(value)
    return path if path.is_absolute() else state_dir / path


def _read_doc(docs_dir: Path, page: dict[str, Any], limit: int) -> str:
    path = docs_dir / page["path"]
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    return text if len(text) <= limit else text[:limit] + "\n... [truncated]"


def _is_high_level(page: dict[str, Any]) -> bool:
    """Top-level components and parent pages are architecture pages.

    Level 2 is the component/subsystem level even when a component has no
    children (for example a cohesive runtime subsystem). Those pages need the
    architecture treatment too; otherwise they degrade into API inventories.
    """
    return (
        page["level"] >= 2
        or bool(page.get("children"))
        or page["doc_type"] in {"architecture", "overview", "flow"}
    )


def _page_kind(page: dict[str, Any]) -> str:
    if page["level"] >= 3 or page["path"] == "overview.md":
        return "overview"
    if page.get("children"):
        return "parent"
    if page["level"] >= 2:
        return "subsystem"
    return "component"


def _finish_context(lines: list[str], max_chars: int) -> str:
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    marker = "\n\n[CONTEXT TRUNCATED BY BUDGET]"
    return text[: max(0, max_chars - len(marker))] + marker


def _walk_components(component: dict[str, Any]):
    yield component
    for child in component.get("children", []):
        yield from _walk_components(child)


def _component_hierarchy(
    component_tree: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str | None], dict[str, str]]:
    """Return components, parent ids, and top-level ancestor ids."""
    root = component_tree["root"]
    components = _component_map(component_tree)
    parent: dict[str, str | None] = {root["id"]: None}
    top: dict[str, str] = {root["id"]: root["id"]}

    def visit(component: dict[str, Any], top_id: str) -> None:
        for child in component.get("children", []):
            parent[child["id"]] = component["id"]
            top[child["id"]] = top_id
            visit(child, top_id)

    for child in root.get("children", []):
        parent[child["id"]] = root["id"]
        top[child["id"]] = child["id"]
        visit(child, child["id"])
    return components, parent, top


def _component_page_map(doc_tree: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}

    def visit(page: dict[str, Any]) -> None:
        mapping.setdefault(page["component_id"], page)
        for child in page.get("children", []):
            visit(child)

    visit(doc_tree["root"])
    return mapping


def _descendant_ids(component: dict[str, Any]) -> set[str]:
    ids = {component["id"]}
    for child in component.get("children", []):
        ids.update(_descendant_ids(child))
    return ids


def _component_label(
    component: dict[str, Any], page_map: dict[str, dict[str, Any]]
) -> str:
    page = page_map.get(component["id"])
    link = f", page={page['path']}" if page else ""
    return (
        f"{component['name']} (id={component['id']}, depth={component['depth']}, "
        f"path={component['path']}, members={len(component.get('members', []))}{link})"
    )


def _read_readme(graph: GraphStore, max_chars: int = 3_500) -> str:
    root = Path(graph.repo_path)
    for name in ("README.md", "README.rst", "README.txt"):
        path = root / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            return text[:max_chars] + (
                "\n... [truncated]" if len(text) > max_chars else ""
            )
    return ""


def _repo_facts(graph: GraphStore) -> list[str]:
    file_count = sum(1 for node in graph.nodes.values() if node["type"] == "file")
    code_nodes = sum(1 for node in graph.nodes.values() if node["layer"] == "code")
    api_count = sum(1 for node in graph.nodes.values() if node["type"] == "api")
    return [
        f"REPOSITORY: {graph.repo}",
        f"COMMIT: {graph.commit or 'unknown'}",
        f"CODE FACTS: files={file_count} code_nodes={code_nodes} public_apis={api_count}",
    ]


def _owner_map(
    graph: GraphStore, components: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Map every code node to its deepest owning component."""
    owner: dict[str, dict[str, Any]] = {}

    def consider(node_id: str, component: dict[str, Any]) -> None:
        if component.get("depth", 0) == 0:
            return
        current = owner.get(node_id)
        if current is None or component["depth"] > current["depth"]:
            owner[node_id] = component

    for component in components.values():
        for member_id in component.get("members", []):
            consider(member_id, component)
    for edge in graph.edges.values():
        if edge["type"] != "part_of_component":
            continue
        component = components.get(edge["dst"])
        if component is not None:
            consider(edge["src"], component)
    return owner


def _node_name(graph: GraphStore, node_id: str) -> str:
    node = graph.nodes.get(node_id)
    if node is None:
        return node_id
    attrs = node["attrs"]
    return str(
        attrs.get("qualified_name") or attrs.get("name") or attrs.get("path") or node_id
    )


def _node_location(graph: GraphStore, node_id: str) -> str:
    node = graph.nodes.get(node_id)
    if node is None:
        return ""
    attrs = node["attrs"]
    path = attrs.get("path")
    if not path:
        return ""
    start = (attrs.get("start") or {}).get("line")
    return f"{path}:{start}" if start else str(path)


def _append_component_context(
    lines: list[str],
    *,
    focus: dict[str, Any],
    components: dict[str, dict[str, Any]],
    top: dict[str, str],
    page_map: dict[str, dict[str, Any]],
    overview: bool,
) -> None:
    lines.extend(["", "COMPONENT MAP:"])
    if overview:
        lines.append("TOP-LEVEL COMPONENTS AND THEIR CHILDREN:")
        for child in focus.get("children", []):
            lines.append(f"- {_component_label(child, page_map)}")
            for grandchild in child.get("children", []):
                lines.append(f"  - {_component_label(grandchild, page_map)}")
        return

    lines.append(f"FOCUS: {_component_label(focus, page_map)}")
    lines.append(f"FOCUS SUMMARY: {focus.get('summary', '')}")
    children = focus.get("children", [])
    if children:
        lines.append("CHILD COMPONENTS:")
        for child in children:
            lines.append(f"- {_component_label(child, page_map)}")
    focus_top = top.get(focus["id"], focus["id"])
    peers = [
        component
        for component in components.values()
        if component.get("depth") == 1 and component["id"] != focus_top
    ]
    if peers:
        lines.append("PEER TOP-LEVEL COMPONENTS:")
        for component in peers:
            lines.append(f"- {_component_label(component, page_map)}")


def _pair_label(
    key: str,
    components: dict[str, dict[str, Any]],
    page_map: dict[str, dict[str, Any]],
) -> str:
    if key.startswith("file:"):
        return key[5:]
    component = components.get(key)
    return _component_label(component, page_map) if component else key


def _append_relationship_context(
    lines: list[str],
    graph: GraphStore,
    *,
    focus: dict[str, Any],
    components: dict[str, dict[str, Any]],
    top: dict[str, str],
    owner: dict[str, dict[str, Any]],
    page_map: dict[str, dict[str, Any]],
    overview: bool,
) -> None:
    """Summarize cross-component imports/calls gathered from the code graph.

    This is deliberately an aggregate: architecture prose needs to know which
    subsystems depend on which, not a dump of every symbol. Representative call
    edges give the model concrete flow evidence for sequence diagrams.
    """
    focus_desc = _descendant_ids(focus) if not overview else set(components)
    leaf_focus = not focus.get("children")
    internal: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    external: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    samples: list[tuple[str, str, str]] = []
    seen_samples: set[tuple[str, str]] = set()

    for edge in graph.edges.values():
        if edge["type"] not in {"imports", "calls"}:
            continue
        src_node = graph.nodes.get(edge["src"])
        dst_node = graph.nodes.get(edge["dst"])
        if src_node is None or dst_node is None:
            continue
        src_component = owner.get(edge["src"])
        dst_component = owner.get(edge["dst"])
        if src_component is None or dst_component is None:
            continue
        src_top = top.get(src_component["id"], src_component["id"])
        dst_top = top.get(dst_component["id"], dst_component["id"])

        if overview:
            if src_top == dst_top:
                continue
            internal[(src_top, dst_top)][edge["type"]] += 1
        else:
            src_in = src_component["id"] in focus_desc
            dst_in = dst_component["id"] in focus_desc
            if src_in and dst_in:
                src_key = src_component["id"]
                dst_key = dst_component["id"]
                if src_component["id"] == dst_component["id"]:
                    if not leaf_focus:
                        continue
                    src_key = "file:" + str(src_node["attrs"].get("path", "<unknown>"))
                    dst_key = "file:" + str(dst_node["attrs"].get("path", "<unknown>"))
                    if src_key == dst_key:
                        continue
                internal[(src_key, dst_key)][edge["type"]] += 1
            elif src_in != dst_in:
                if src_top == dst_top:
                    continue
                external[(src_top, dst_top)][edge["type"]] += 1
            else:
                continue

        if edge["type"] == "calls" and len(samples) < 12:
            key = (edge["src"], edge["dst"])
            if key not in seen_samples:
                seen_samples.add(key)
                samples.append((edge["src"], edge["dst"], edge["type"]))

    def emit(title: str, buckets: dict[tuple[str, str], Counter[str]]) -> None:
        lines.extend(["", title])
        if not buckets:
            lines.append(
                "- No cross-boundary import/call edges were observed in the graph."
            )
            return
        ranked = sorted(buckets.items(), key=lambda item: -sum(item[1].values()))[:15]
        for (src_key, dst_key), counts in ranked:
            parts = [f"{kind}={count}" for kind, count in counts.items() if count]
            lines.append(
                f"- {_pair_label(src_key, components, page_map)} -> "
                f"{_pair_label(dst_key, components, page_map)}: {', '.join(parts)}"
            )

    if overview:
        emit("TOP-LEVEL COMPONENT RELATIONSHIPS (imports/calls):", internal)
    else:
        label = "FILE" if leaf_focus else "CHILD COMPONENT"
        emit(f"INTERNAL {label} RELATIONSHIPS (imports/calls):", internal)
        emit("EXTERNAL COMPONENT RELATIONSHIPS (imports/calls):", external)

    if samples:
        lines.extend(["", "REPRESENTATIVE CALL EDGES (flow evidence):"])
        for src, dst, _kind in samples:
            lines.append(
                f"- `{_node_name(graph, src)}` ({_node_location(graph, src)}) -> "
                f"`{_node_name(graph, dst)}` ({_node_location(graph, dst)})"
            )


def _key_entrypoints(
    graph: GraphStore,
    components: dict[str, dict[str, Any]],
    owner: dict[str, dict[str, Any]],
    top: dict[str, str],
    focus: dict[str, Any],
    *,
    overview: bool,
    limit: int = 18,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    focus_ids = set(components) if overview else _descendant_ids(focus)
    degree: Counter[str] = Counter()
    for edge in graph.edges.values():
        if edge["type"] == "calls":
            degree[edge["src"]] += 1
            degree[edge["dst"]] += 1
    handlers = {
        edge["src"]: edge["dst"]
        for edge in graph.edges.values()
        if edge["type"] == "handles"
    }
    candidates: list[tuple[int, str, dict[str, Any], dict[str, Any]]] = []
    for node in graph.nodes.values():
        if node["type"] != "api":
            continue
        component = owner.get(node["id"])
        if component is None or component["id"] not in focus_ids:
            continue
        name = str(
            node["attrs"].get("qualified_name") or node["attrs"].get("name") or ""
        )
        short_name = name.rsplit(".", 1)[-1]
        if not name or short_name.startswith("_"):
            continue
        symbol_id = handlers.get(node["id"])
        score = degree.get(symbol_id, 0) if symbol_id else 0
        candidates.append((score, name, node, component))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    if not overview:
        return [
            (node, component) for _score, _name, node, component in candidates[:limit]
        ]

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for _score, _name, node, component in candidates:
        top_id = top.get(component["id"], component["id"])
        if len(grouped[top_id]) < 3:
            grouped[top_id].append((node, component))
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for child in focus.get("children", []):
        selected.extend(grouped.get(child["id"], [])[:3])
    return selected[:limit]


def _append_entrypoint_context(
    lines: list[str],
    graph: GraphStore,
    entrypoints: list[tuple[dict[str, Any], dict[str, Any]]],
    page_map: dict[str, dict[str, Any]],
) -> None:
    lines.extend(["", "KEY ENTRY POINTS (evidence, not an exhaustive API inventory):"])
    if not entrypoints:
        lines.append("- No public API nodes were found for this page.")
        return
    handlers = {
        edge["src"]: edge["dst"]
        for edge in graph.edges.values()
        if edge["type"] == "handles"
    }
    for node, component in entrypoints:
        attrs = node["attrs"]
        signature = str(attrs.get("signature") or "").splitlines()[0][:180]
        citation = _citation(graph, node["id"]) or ""
        lines.append(
            f"- `{attrs.get('qualified_name') or attrs.get('name')}` "
            f"component={component['name']} signature={signature} "
            f"id={node['id']} source={citation}"
        )
        lines[-1] += f" handler={handlers.get(node['id'], '')}"


def _append_source_excerpts(
    lines: list[str],
    graph: GraphStore,
    entrypoints: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    max_excerpts: int = 8,
    max_chars: int = 1_200,
) -> None:
    handlers = {
        edge["src"]: edge["dst"]
        for edge in graph.edges.values()
        if edge["type"] == "handles"
    }
    shown = 0
    lines.extend(["", "SOURCE EXCERPTS FOR KEY ENTRY POINTS:"])
    for node, _component in entrypoints:
        if shown >= max_excerpts:
            break
        symbol_id = handlers.get(node["id"])
        symbol = graph.nodes.get(symbol_id or "")
        if symbol is None:
            continue
        source = str(symbol["attrs"].get("source") or "")
        if not source:
            continue
        citation = _citation(graph, symbol_id) or _node_location(graph, symbol_id)
        excerpt = source[:max_chars]
        if len(source) > max_chars:
            excerpt += "\n... [truncated]"
        lines.extend(
            [
                "",
                f"### `{_node_name(graph, symbol_id)}` ({citation})",
                "```python",
                excerpt,
                "```",
            ]
        )
        shown += 1
    if shown == 0:
        lines.append("- No source excerpts available.")


def _append_file_roles(
    lines: list[str],
    graph: GraphStore,
    focus: dict[str, Any],
    owner: dict[str, dict[str, Any]],
) -> None:
    focus_ids = _descendant_ids(focus)
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in graph.nodes.values():
        if node["type"] not in {"file", "symbol", "api"}:
            continue
        component = owner.get(node["id"])
        if component is None or component["id"] not in focus_ids:
            continue
        path = node["attrs"].get("path")
        if path:
            by_path[str(path)].append(node)
    lines.extend(["", "FILE AND MODULE RESPONSIBILITIES:"])
    if not by_path:
        lines.append("- No file nodes were found for this component.")
        return
    ranked = sorted(by_path.items(), key=lambda item: -len(item[1]))[:30]
    for path, nodes in ranked:
        file_node = next((node for node in nodes if node["type"] == "file"), None)
        docstring = str(
            (file_node or {}).get("attrs", {}).get("docstring") or ""
        ).strip()
        docstring = docstring.splitlines()[0][:180] if docstring else ""
        names = [
            str(node["attrs"].get("qualified_name") or node["attrs"].get("name") or "")
            for node in nodes
            if node["type"] == "symbol"
        ]
        names = [
            name
            for name in names
            if name and not name.rsplit(".", 1)[-1].startswith("_")
        ]
        suffix = f"; key symbols: {', '.join(names[:6])}" if names else ""
        lines.append(
            f"- {path} (nodes={len(nodes)}, symbols={sum(1 for n in nodes if n['type'] == 'symbol')}, "
            f"apis={sum(1 for n in nodes if n['type'] == 'api')})"
            f"{': ' + docstring if docstring else ''}{suffix}"
        )


def _append_child_docs(
    lines: list[str],
    page: dict[str, Any],
    docs_dir: Path,
    *,
    per_child: int,
    total_budget: int,
) -> None:
    lines.extend(
        [
            "",
            "CHILD DOCUMENTATION EXCERPTS (use for responsibilities and flows; do not copy API inventories):",
        ]
    )
    total = 0
    for child_page in page.get("children", []):
        if total >= total_budget:
            break
        content = _read_doc(docs_dir, child_page, per_child)
        if not content:
            continue
        take = min(len(content), total_budget - total)
        lines.extend(
            [
                "",
                f"### {child_page['title']} ({child_page['path']})",
                content[:take],
            ]
        )
        total += take
    if total == 0:
        lines.append("- No child documentation was available.")


def build_page_context(
    graph: GraphStore,
    component_tree: dict[str, Any],
    doc_tree: dict[str, Any],
    page: dict[str, Any],
    *,
    docs_dir: Path,
    max_source_chars: int = 12_000,
    child_doc_chars: int = 4_000,
) -> str:
    components, _parent, top = _component_hierarchy(component_tree)
    component = components[page["component_id"]]
    page_map = _component_page_map(doc_tree)
    lines: list[str] = [
        f"PAGE: {page['title']} ({page['path']})",
        f"LEVEL: {page['level']}  DOC_TYPE: {page['doc_type']}  KIND: {_page_kind(page)}",
        f"COMPONENT: {component['name']} path={component['path']}",
        f"COMPONENT SUMMARY: {component['summary']}",
        "",
    ]
    lines.extend(_repo_facts(graph))
    lines.append("")

    if _is_high_level(page):
        overview = _page_kind(page) == "overview"
        owner = _owner_map(graph, components)
        _append_component_context(
            lines,
            focus=component,
            components=components,
            top=top,
            page_map=page_map,
            overview=overview,
        )
        _append_relationship_context(
            lines,
            graph,
            focus=component,
            components=components,
            top=top,
            owner=owner,
            page_map=page_map,
            overview=overview,
        )
        entrypoints = _key_entrypoints(
            graph,
            components,
            owner,
            top,
            component,
            overview=overview,
        )
        _append_entrypoint_context(lines, graph, entrypoints, page_map)

        if overview:
            readme = _read_readme(graph)
            if readme:
                lines.extend(["", "PROJECT README EXCERPT:", readme])
            child_budget = min(55_000, max(4_000, max_source_chars // 2))
        else:
            child_budget = min(50_000, max(4_000, max_source_chars // 2))

        child_pages = page.get("children", [])
        if child_pages:
            _append_child_docs(
                lines,
                page,
                docs_dir,
                per_child=max(1_200, child_doc_chars),
                total_budget=child_budget,
            )
        else:
            _append_file_roles(lines, graph, component, owner)
            _append_source_excerpts(lines, graph, entrypoints)
        return _finish_context(lines, max_source_chars)

    members = [
        graph.nodes[node_id]
        for node_id in component.get("members", [])
        if node_id in graph.nodes
    ]
    files = sorted(
        [n for n in members if n["type"] == "file"],
        key=lambda n: n["attrs"].get("path", ""),
    )
    symbols = sorted(
        [n for n in members if n["type"] == "symbol"],
        key=lambda n: (
            n["attrs"].get("path", ""),
            n["attrs"].get("start", {}).get("line", 0),
        ),
    )
    apis = sorted(
        [n for n in members if n["type"] == "api"],
        key=lambda n: n["attrs"].get("qualified_name", ""),
    )
    lines.append("FILES:")
    for file_node in files:
        lines.append(f"- {file_node['attrs'].get('path', '')}")
    if files and files[0]["attrs"].get("docstring"):
        lines.append(f"MODULE DOCSTRING: {files[0]['attrs']['docstring']}")
    lines.extend(["", "PUBLIC APIs:"])
    for api in apis:
        attrs = api["attrs"]
        citation = _citation(graph, api["id"])
        lines.append(
            f"- {attrs.get('qualified_name', '')} id={api['id']} "
            f"signature={attrs.get('signature', '')} source={citation or ''}"
        )
    lines.extend(["", "SYMBOLS:"])
    for symbol in symbols:
        attrs = symbol["attrs"]
        citation = _citation(graph, symbol["id"])
        lines.append(
            f"- {attrs.get('qualified_name', '')} id={symbol['id']} "
            f"kind={attrs.get('kind', '')} signature={attrs.get('signature', '')} "
            f"source={citation or ''}"
        )
        if attrs.get("docstring"):
            lines.append(f"  docstring: {str(attrs['docstring'])[:500]}")
        source = attrs.get("source")
        if source:
            lines.append("  code: " + str(source).replace("\n", " ")[:600])
        if len("\n".join(lines)) >= max_source_chars:
            break
    return _finish_context(lines, max_source_chars)


def _diagram_requirement(page: dict[str, Any]) -> str:
    kind = _page_kind(page)
    if kind == "overview":
        return (
            "DIAGRAM REQUIREMENT: include an end-to-end ```mermaid flowchart and at least "
            "one ```mermaid sequenceDiagram for a critical runtime or data flow. Both must "
            "appear inside section content."
        )
    if kind in {"subsystem", "parent"}:
        return (
            "DIAGRAM REQUIREMENT: include at least one ```mermaid flowchart that shows "
            "internal structure and external boundaries. If the evidence describes a "
            "runtime or data flow, also include a sequenceDiagram."
        )
    return "DIAGRAMS: optional; use them only when they clarify a flow."


def _architecture_task(page: dict[str, Any]) -> str:
    kind = _page_kind(page)
    if kind == "overview":
        return (
            "Write a repository overview as an architecture document, not an API reference. "
            "Use 4-5 sections: (1) Purpose and system context; (2) End-to-end architecture "
            "with a Mermaid flowchart showing actors, major subsystems, and data/control "
            "handoffs; (3) One or more critical flows with a Mermaid sequenceDiagram, "
            "including what is transformed or persisted; (4) Design decisions, boundaries, "
            "failure isolation, concurrency/configuration concerns, and trade-offs; "
            "(5) A compact module map that links to the child pages. Explain why components "
            "interact, not just that they exist. Do not list signatures or copy child docs."
        )
    if kind == "parent":
        return (
            "Write a parent-module architecture document that synthesizes its child pages. "
            "Use 4-5 sections: (1) Purpose and boundary of this module; (2) Internal "
            "architecture with a Mermaid flowchart; (3) Key interactions/flows across the "
            "children, including data or control handoffs and a sequenceDiagram when useful; "
            "(4) Design decisions and cross-cutting concerns; (5) Child module map with links. "
            "Do not merely concatenate child summaries or reproduce API inventories."
        )
    return (
        "Write a subsystem architecture document for this top-level component. Use 4-5 "
        "sections: (1) Purpose and boundaries; (2) Internal architecture with a Mermaid "
        "flowchart showing the main files/objects and their dependencies; (3) Key runtime "
        "or data flows, with a sequenceDiagram when the code evidence supports one; "
        "(4) Design decisions, invariants, error/context handling, and extension points; "
        "(5) Related modules and links. Explain the logic and interactions; do not turn "
        "the page into a list of public APIs or docstrings."
    )


class LLMDocumentAgent:
    def __init__(
        self,
        client: Any,
        *,
        model: str | None = None,
        max_source_chars: int | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
        budget: ComputedBudget | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.budget = budget
        self.max_source_chars = (
            max_source_chars
            if max_source_chars is not None
            else (budget.context_chars if budget is not None else 12_000)
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else (budget.output_tokens if budget is not None else 8_192)
        )
        self.temperature = temperature
        self.last_prompt: str | None = None
        self.last_context: str | None = None
        self.last_response: dict[str, Any] | None = None
        self.last_elapsed_sec: float | None = None

    def build_prompt(
        self,
        graph: GraphStore,
        component_tree: dict[str, Any],
        doc_tree: dict[str, Any],
        page: dict[str, Any],
        *,
        docs_dir: Path,
    ) -> tuple[str, str]:
        context = build_page_context(
            graph,
            component_tree,
            doc_tree,
            page,
            docs_dir=docs_dir,
            max_source_chars=self.max_source_chars,
        )
        high_level = _is_high_level(page)
        if high_level:
            mode = f"ARCHITECTURE_{_page_kind(page).upper()}"
            task = _architecture_task(page)
            requirements = (
                f"{_diagram_requirement(page)}\n"
                "Use only evidence in CONTEXT. State uncertainty instead of inventing a "
                "relationship. Link to child pages using their relative paths. A short "
                "entry-point mention is fine when it explains a flow, but do not include "
                "an exhaustive API/signature table."
            )
        else:
            mode = "LOW_LEVEL_IMPLEMENTATION"
            task = (
                "Write a low-level component page grounded in the context. Cover responsibility, "
                "public API, implementation, and dependencies. Cite exact file:line-line spans."
            )
            requirements = (
                "Use 2-4 sections. Do not add diagrams unless they materially help."
            )
        user_prompt = (
            f"MODE: {mode}\n\n"
            f"TASK:\n{task}\n\n"
            f"{requirements}\n\n"
            f"CONTEXT:\n{context}\n\n"
            "Return the JSON sections now."
        )
        return user_prompt, context

    def generate(
        self,
        graph: GraphStore,
        component_tree: dict[str, Any],
        doc_tree: dict[str, Any],
        page: dict[str, Any],
        *,
        docs_dir: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        user_prompt, context = self.build_prompt(
            graph,
            component_tree,
            doc_tree,
            page,
            docs_dir=docs_dir,
        )
        started = time.monotonic()
        response = self.client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.last_elapsed_sec = round(time.monotonic() - started, 3)
        self.last_prompt = user_prompt
        self.last_context = context
        self.last_response = response
        try:
            message = response["choices"][0]["message"]
            content = message.get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMDocError("LLM response has no choices[0].message") from exc
        if not content:
            reasoning = str(message.get("reasoning_content") or "")
            usage = response.get("usage") or {}
            raise LLMDocError(
                f"LLM returned empty document content (reasoning_chars={len(reasoning)}, "
                f"usage={usage})"
            )
        data = _extract_json(content)
        raw_sections = data.get("sections")
        if not isinstance(raw_sections, list) or not raw_sections:
            raise LLMDocError("LLM document JSON has no non-empty `sections` list")
        sections = normalize_sections(raw_sections, page, graph)
        diagrams = _validate_diagrams(page, sections)
        trace = {
            "page": page["path"],
            "mode": "high_level" if _is_high_level(page) else "low_level",
            "kind": _page_kind(page),
            "created_at": utc_now(),
            "elapsed_sec": self.last_elapsed_sec,
            "prompt": user_prompt,
            "context_chars": len(context),
            "context_truncated": "[CONTEXT TRUNCATED BY BUDGET]" in context,
            "budget": self.budget.as_dict() if self.budget else None,
            "diagrams": diagrams,
            "response": response,
            "content": content,
        }
        return sections, trace


def _resolve_covers(raw_covers: Iterable[Any], graph: GraphStore) -> list[str]:
    resolved: set[str] = set()
    for raw in raw_covers:
        item = str(raw).strip()
        if not item:
            continue
        if item in graph.nodes:
            resolved.add(item)
            continue
        shorthand = item[5:] if item.startswith("code:") else item
        for node in graph.nodes.values():
            if node["layer"] not in {"code", "semantic"}:
                continue
            attrs = node["attrs"]
            candidates = {
                attrs.get("qualified_name"),
                attrs.get("name"),
                attrs.get("path"),
            }
            if shorthand in candidates or shorthand == node["id"]:
                resolved.add(node["id"])
    return sorted(resolved)


def _resolve_citation_covers(citations: Iterable[Any], graph: GraphStore) -> list[str]:
    """Best-effort fallback: map path:line-line citations to overlapping code nodes."""
    import re

    resolved: set[str] = set()
    for raw in citations:
        citation = str(raw).strip()
        match = re.match(r"^(.*):(\d+)-(\d+)$", citation)
        if not match:
            continue
        path, start, end = match.group(1), int(match.group(2)), int(match.group(3))
        for node in graph.nodes.values():
            if node["layer"] != "code":
                continue
            attrs = node["attrs"]
            if attrs.get("path") != path:
                continue
            if node["type"] == "file":
                resolved.add(node["id"])
                continue
            node_start = (attrs.get("start") or {}).get("line")
            node_end = (attrs.get("end") or {}).get("line")
            if node_start is None or node_end is None:
                continue
            if start <= node_end and end >= node_start:
                resolved.add(node["id"])
    return sorted(resolved)


def normalize_sections(
    raw_sections: Iterable[dict[str, Any]],
    page: dict[str, Any],
    graph: GraphStore,
) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for order, raw in enumerate(raw_sections):
        if not isinstance(raw, dict):
            continue
        heading = str(raw.get("heading") or "").strip()
        content = str(raw.get("content") or "").strip()
        if not heading or not content:
            continue
        level = raw.get("level", 2)
        try:
            level = int(level)
        except (TypeError, ValueError):
            level = 2
        level = min(4, max(1, level))
        citations = [str(item) for item in (raw.get("citations") or []) if item]
        covers = _resolve_covers(raw.get("covers") or [], graph)
        if not covers:
            covers = _resolve_citation_covers(citations, graph)
        if not covers:
            component_id = page.get("component_id")
            component = graph.nodes.get(component_id) if component_id else None
            if component is not None:
                covers = [component_id]
                if not _is_high_level(page):
                    covers.extend(component["attrs"].get("members") or [])
                covers = sorted({item for item in covers if item in graph.nodes})
        sections.append(
            {
                "id": doc_section_id(page["id"], heading, order),
                "heading": heading,
                "level": level,
                "order": order,
                "content": content.strip() + "\n",
                "content_hash": sha256_text(content.strip() + "\n"),
                "covers": sorted(set(covers)),
                "citations": sorted(set(citations)),
                "_content": content.strip() + "\n",
            }
        )
    if not sections:
        raise LLMDocError("all LLM sections were empty or malformed")
    return sections


def _diagram_info(sections: list[dict[str, Any]]) -> dict[str, Any]:
    blocks: list[str] = []
    for section in sections:
        content = str(section.get("_content") or section.get("content") or "")
        blocks.extend(
            match.group(1)
            for match in re.finditer(
                r"```mermaid[ \t]*\r?\n(.*?)```", content, re.S | re.I
            )
        )
    types: list[str] = []
    for block in blocks:
        first = next((line.strip() for line in block.splitlines() if line.strip()), "")
        if re.match(r"^(?:flowchart|graph)\b", first):
            types.append("flowchart")
        elif first.startswith("sequenceDiagram"):
            types.append("sequenceDiagram")
        elif first.startswith("classDiagram"):
            types.append("classDiagram")
        elif first.startswith("stateDiagram"):
            types.append("stateDiagram")
        elif first.startswith("erDiagram"):
            types.append("erDiagram")
        else:
            types.append("other")
    return {"count": len(blocks), "types": sorted(set(types))}


def _validate_diagrams(
    page: dict[str, Any], sections: list[dict[str, Any]]
) -> dict[str, Any]:
    info = _diagram_info(sections)
    if not _is_high_level(page):
        return info
    if info["count"] == 0:
        raise LLMDocError(
            "high-level page produced no Mermaid diagram; architecture pages must "
            "include at least one flowchart"
        )
    if _page_kind(page) == "overview":
        missing = [
            kind
            for kind in ("flowchart", "sequenceDiagram")
            if kind not in info["types"]
        ]
        if missing:
            raise LLMDocError(
                "repository overview is missing required Mermaid diagram(s): "
                + ", ".join(missing)
            )
    return info


def _render_markdown(page: dict[str, Any], sections: list[dict[str, Any]]) -> str:
    lines = [f"# {page['title']}", ""]
    for section in sections:
        lines.extend([f"## {section['heading']}", "", section["_content"].strip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def _replace_page_sections(
    graph: GraphStore,
    page: dict[str, Any],
    source_graph_version: str,
    sections_with_content: list[dict[str, Any]],
) -> None:
    for node in list(graph.nodes.values()):
        if node["type"] == "doc_section" and node["attrs"].get("page_id") == page["id"]:
            graph.remove_node(node["id"])

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
    graph.add_edge("documents", page["id"], page["component_id"])
    for section in sections_with_content:
        graph.add_node(
            "doc_section",
            section["id"],
            {
                "page_id": page["id"],
                "heading": section["heading"],
                "level": section["level"],
                "order": section["order"],
                "content": section.get("_content", "").strip() + "\n",
                "content_hash": section["content_hash"],
                "covers": section["covers"],
                "citations": section["citations"],
                "source_graph_version": source_graph_version,
            },
        )
        graph.add_edge("child_of", section["id"], page["id"])
        for covered in section["covers"]:
            graph.add_edge("cites", section["id"], covered)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def generate_llm_pages(
    state_dir: str | Path,
    page_keys: Iterable[str],
    *,
    all_pages: bool = False,
    client: Any | None = None,
    max_pages: int = 4,
    max_source_chars: int | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.2,
    model: str | None = None,
    dry_run: bool = False,
    reuse_traces: bool = False,
    limits: Any | None = None,
    budget: ComputedBudget | None = None,
    context_ratio: float | None = None,
    output_ratio: float | None = None,
    progress: Any | None = None,
) -> dict[str, Any]:
    state = Path(state_dir).resolve()
    manifest = read_manifest(state)
    if manifest is None:
        raise LLMDocError(f"no manifest in {state}; run `repodoc build` first")
    graph = GraphStore.load(state)
    component_tree = json.loads(
        (state / "components" / "component_tree.json").read_text(encoding="utf-8")
    )
    doc_tree = json.loads(
        (state / "docs" / "doc_tree.json").read_text(encoding="utf-8")
    )
    docs_dir = _docs_dir(state, manifest)
    pages_path = state / (manifest.get("artifacts") or {}).get(
        "pages", "docs/pages.jsonl"
    )
    sections_path = state / (manifest.get("artifacts") or {}).get(
        "sections", "docs/sections.jsonl"
    )
    pages = _load_jsonl(pages_path)
    sections = _load_jsonl(sections_path)

    lookup = _page_map(doc_tree)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    if all_pages:
        page_keys = sorted({page["path"] for page in _page_map(doc_tree).values()})
    for key in page_keys:
        page = lookup.get(str(key).strip())
        if page is None:
            available = sorted({p["path"] for p in _page_map(doc_tree).values()})
            raise LLMDocError(
                f"unknown page '{key}'; available paths: {', '.join(available)}"
            )
        if page["id"] not in seen:
            seen.add(page["id"])
            selected.append(page)
    if not selected:
        raise LLMDocError("no pages selected")
    if len(selected) > max_pages:
        raise LLMDocError(
            f"selected {len(selected)} pages, above --max-pages={max_pages}; "
            "refusing to send a large documentation batch"
        )
    selected.sort(key=lambda page: (page["level"], page["path"]))

    source_graph_version = component_tree["source_graph_version"]
    from repodoc.llm import OpenCodeGoConfig, model_limits_or_fallback

    config = OpenCodeGoConfig.from_env()
    limits = limits or model_limits_or_fallback(config)
    budget = budget or compute_budget(
        limits,
        "docs",
        context_ratio=context_ratio,
        output_ratio=output_ratio,
    )
    results: list[dict[str, Any]] = []
    own_client = False
    if not dry_run and not reuse_traces and client is None:
        from repodoc.llm import OpenCodeGoClient

        client = OpenCodeGoClient()
        own_client = True
    try:
        for page_index, page in enumerate(selected, 1):
            if progress is not None:
                progress.step(
                    "Generating page",
                    page_index,
                    len(selected),
                    detail=f"{page['path']} (L{page['level']})",
                )
            agent = LLMDocumentAgent(
                client,
                model=model,
                max_source_chars=max_source_chars,
                max_tokens=max_tokens,
                temperature=temperature,
                budget=budget,
            )
            if dry_run:
                user_prompt, context = agent.build_prompt(
                    graph, component_tree, doc_tree, page, docs_dir=docs_dir
                )
                results.append(
                    {
                        "page": page["path"],
                        "prompt_chars": len(user_prompt),
                        "context_chars": len(context),
                        "context_truncated": "[CONTEXT TRUNCATED BY BUDGET]" in context,
                        "dry_run": True,
                    }
                )
                continue

            if reuse_traces:
                trace_path = (
                    state / "docs" / "llm_traces" / f"{Path(page['path']).stem}.json"
                )
                if not trace_path.is_file():
                    raise LLMDocError(
                        f"trace not found for page {page['path']}: {trace_path}"
                    )
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                content = trace.get("content") or ""
                data = _extract_json(content)
                raw_sections = normalize_sections(
                    data.get("sections") or [], page, graph
                )
            else:
                try:
                    raw_sections, trace = agent.generate(
                        graph, component_tree, doc_tree, page, docs_dir=docs_dir
                    )
                except Exception:
                    if agent.last_response is not None:
                        failure_trace = (
                            state
                            / "docs"
                            / "llm_traces"
                            / "failed"
                            / f"{Path(page['path']).stem}.{int(time.time() * 1000)}.json"
                        )
                        failure_trace.parent.mkdir(parents=True, exist_ok=True)
                        failure_trace.write_text(
                            json.dumps(
                                {
                                    "page": page["path"],
                                    "created_at": utc_now(),
                                    "elapsed_sec": agent.last_elapsed_sec,
                                    "prompt": agent.last_prompt,
                                    "context": agent.last_context,
                                    "response": agent.last_response,
                                    "failed": True,
                                },
                                indent=2,
                                ensure_ascii=False,
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                    raise
            markdown = _render_markdown(page, raw_sections)
            (docs_dir / page["path"]).write_text(markdown, encoding="utf-8")
            page["content_hash"] = sha256_text(markdown)
            page["status"] = "generated"
            page["sections"] = [
                {
                    key: value
                    for key, value in section.items()
                    if key not in {"_content", "content"}
                }
                for section in raw_sections
            ]
            _replace_page_sections(graph, page, source_graph_version, raw_sections)

            for old in [row for row in pages if row.get("id") == page["id"]]:
                pages.remove(old)
            pages.append(
                {
                    "id": page["id"],
                    "path": page["path"],
                    "title": page["title"],
                    "level": page["level"],
                    "doc_type": page["doc_type"],
                    "component_id": page["component_id"],
                    "status": page["status"],
                    "content_hash": page["content_hash"],
                    "section_ids": [section["id"] for section in page["sections"]],
                }
            )
            sections = [row for row in sections if row.get("page_path") != page["path"]]
            for section in raw_sections:
                sections.append(
                    {
                        **{
                            key: value
                            for key, value in section.items()
                            if key != "_content"
                        },
                        "page_path": page["path"],
                        "content": section["_content"],
                    }
                )
            trace_path = (
                state / "docs" / "llm_traces" / f"{Path(page['path']).stem}.json"
            )
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(
                json.dumps(trace, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            if progress is not None:
                progress.done(
                    "Generated page",
                    page_index,
                    len(selected),
                    detail=f"{page['path']} ({len(raw_sections)} sections)",
                )
            results.append(
                {
                    "page": page["path"],
                    "prompt_chars": trace["context_chars"],
                    "context_truncated": trace.get("context_truncated", False),
                    "sections": len(raw_sections),
                    "diagrams": trace.get("diagrams"),
                    "usage": (trace.get("response") or {}).get("usage"),
                    "trace": str(trace_path),
                    "reused": reuse_traces,
                }
            )
    finally:
        if own_client and client is not None:
            client.close()

    if dry_run:
        return {"dry_run": True, "pages": results, "budget": budget.as_dict()}

    doc_tree["graph_version"] = graph.graph_version
    markdown = {
        path.relative_to(docs_dir).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(docs_dir.rglob("*.md"))
        if path.is_file()
    }
    doc_tree["doc_version"] = "d_" + stable_hash(
        {"tree": doc_tree, "markdown": markdown}
    )
    graph.save(state)
    (state / "docs" / "doc_tree.json").write_text(
        json.dumps(doc_tree, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_jsonl(pages_path, pages)
    _write_jsonl(sections_path, sections)
    manifest["graph_version"] = doc_tree["graph_version"]
    manifest["doc_version"] = doc_tree["doc_version"]
    manifest["updated_at"] = utc_now()
    manifest["last_run"] = {
        "run_id": manifest.get("last_run", {}).get("run_id", "llm-docs"),
        "status": "llm-docs",
        "finished_at": utc_now(),
        "pages": [page["path"] for page in selected],
    }
    validate("doc_tree", doc_tree)
    validate("manifest", manifest)
    write_manifest(state, manifest)
    return {
        "dry_run": False,
        "pages": results,
        "graph_version": doc_tree["graph_version"],
        "doc_version": doc_tree["doc_version"],
        "budget": budget.as_dict(),
    }
