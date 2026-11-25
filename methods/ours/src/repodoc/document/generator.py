"""Deterministic document generation for the toy knowledge-graph prototype."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from repodoc.graph.store import GraphStore
from repodoc.schema import (
    doc_section_id,
    sha256_text,
    source_ref,
    stable_hash,
)


def _node(graph: GraphStore, node_id: str) -> dict[str, Any] | None:
    return graph.nodes.get(node_id)


def _citation(graph: GraphStore, node_id: str) -> str | None:
    node = _node(graph, node_id)
    if node is None:
        return None
    attrs = node["attrs"]
    if node["type"] == "symbol":
        return source_ref(attrs["path"], attrs["start"], attrs["end"])
    if node["type"] == "file":
        return attrs.get("path")
    if node["type"] == "api":
        for edge in graph.outgoing(node_id, {"handles"}):
            return _citation(graph, edge["dst"])
    return None


def _docstring(graph: GraphStore, node_id: str) -> str | None:
    node = _node(graph, node_id)
    if node is None:
        return None
    attrs = node["attrs"]
    if attrs.get("docstring"):
        return attrs["docstring"]
    if node["type"] == "api" and attrs.get("symbol_id"):
        return _docstring(graph, attrs["symbol_id"])
    return None


def _members(
    graph: GraphStore, component: dict[str, Any], node_type: str
) -> list[dict[str, Any]]:
    nodes = [
        graph.nodes[node_id]
        for node_id in component.get("members", [])
        if node_id in graph.nodes and graph.nodes[node_id]["type"] == node_type
    ]
    return sorted(
        nodes,
        key=lambda node: (
            node["attrs"].get("path", ""),
            node["attrs"].get("start", {}).get("line", 0),
            node["attrs"].get("name", ""),
        ),
    )


def _component_by_id(component_tree: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}

    def visit(component: dict[str, Any]) -> None:
        mapping[component["id"]] = component
        for child in component.get("children", []):
            visit(child)

    visit(component_tree["root"])
    return mapping


def _section(
    page: dict[str, Any],
    heading: str,
    content: str,
    *,
    level: int,
    order: int,
    covers: Iterable[str] = (),
    citations: Iterable[str] = (),
) -> dict[str, Any]:
    text = content.strip() + "\n"
    return {
        "id": doc_section_id(page["id"], heading, order),
        "heading": heading,
        "level": level,
        "order": order,
        "content": text,
        "content_hash": sha256_text(text),
        "covers": sorted(set(covers)),
        "citations": sorted(set(c for c in citations if c)),
    }


def _api_block(graph: GraphStore, api: dict[str, Any]) -> str:
    attrs = api["attrs"]
    lines = [f"### `{attrs.get('qualified_name', attrs.get('name', api['id']))}`"]
    signature = attrs.get("signature")
    if signature:
        lines.extend(["", "```python", signature, "```"])
    docstring = _docstring(graph, api["id"])
    if docstring:
        lines.extend(["", docstring.strip()])
    citation = _citation(graph, api["id"])
    if citation:
        lines.extend(["", f"Source: `{citation}`"])
    return "\n".join(lines)


def _symbol_block(graph: GraphStore, symbol: dict[str, Any]) -> str:
    attrs = symbol["attrs"]
    lines = [f"### `{attrs.get('qualified_name', attrs.get('name', symbol['id']))}`"]
    lines.extend(["", f"- kind: `{attrs.get('kind', 'symbol')}`"])
    citation = _citation(graph, symbol["id"])
    if citation:
        lines.append(f"- source: `{citation}`")
    signature = attrs.get("signature")
    if signature:
        lines.extend(["", "```python", signature, "```"])
    if attrs.get("docstring"):
        lines.extend(["", attrs["docstring"].strip()])
    if attrs.get("comments"):
        lines.extend(["", "Comments:"])
        lines.extend(f"- {comment['text']}" for comment in attrs["comments"][:5])
    calls = []
    for edge in graph.outgoing(symbol["id"], {"calls"}):
        target = graph.nodes.get(edge["dst"])
        if target is not None:
            calls.append(target["attrs"].get("qualified_name", target["id"]))
    if calls:
        lines.extend(
            ["", "Calls: " + ", ".join(f"`{name}`" for name in sorted(set(calls)))]
        )
    return "\n".join(lines)


def _root_sections(
    graph: GraphStore,
    component_tree: dict[str, Any],
    page: dict[str, Any],
) -> list[dict[str, Any]]:
    root = component_tree["root"]
    children = root.get("children", [])
    overview = (
        f"Repository `{component_tree['repo']}` at commit "
        f"`{component_tree.get('commit') or 'working-tree'}`.\n\n"
        f"This toy snapshot contains {len(root['members'])} code nodes grouped into "
        f"{len(children)} component(s)."
    )
    component_rows = [
        "| Component | Path | Files | Symbols | APIs |",
        "|---|---|---:|---:|---:|",
    ]
    for child in children:
        files = _members(graph, child, "file")
        symbols = _members(graph, child, "symbol")
        apis = _members(graph, child, "api")
        link = next(
            (
                candidate["path"]
                for candidate in page.get("children", [])
                if candidate["component_id"] == child["id"]
            ),
            "",
        )
        component_rows.append(
            f"| [{child['name']}]({link}) | `{child['path']}` | {len(files)} | "
            f"{len(symbols)} | {len(apis)} |"
        )
    sections = [
        _section(page, "Summary", overview, level=2, order=0, covers=[root["id"]]),
        _section(
            page,
            "Components",
            "\n".join(component_rows),
            level=2,
            order=1,
            covers=[child["id"] for child in children],
        ),
    ]
    api_nodes = sorted(
        graph.nodes_of_type("api"),
        key=lambda node: (
            node["attrs"].get("path", ""),
            node["attrs"].get("qualified_name", ""),
        ),
    )
    api_content = "\n\n".join(_api_block(graph, api) for api in api_nodes)
    sections.append(
        _section(
            page,
            "Public API",
            api_content or "No public API nodes detected.",
            level=2,
            order=2,
            covers=[api["id"] for api in api_nodes],
            citations=[_citation(graph, api["id"]) or "" for api in api_nodes],
        )
    )
    return sections


def _parent_component_sections(
    graph: GraphStore,
    component_tree: dict[str, Any],
    component: dict[str, Any],
    page: dict[str, Any],
) -> list[dict[str, Any]]:
    children = component.get("children", [])
    total_members = sum(len(child.get("members", [])) for child in children)
    overview = (
        f"{component.get('summary', '')}\n\n"
        f"This component is split into {len(children)} child component(s) "
        f"covering {total_members} code node(s)."
    )
    child_links = {
        child["component_id"]: child["path"] for child in page.get("children", [])
    }
    rows = ["| Child component | Path | Nodes |", "|---|---|---:|"]
    for child in children:
        link = child_links.get(child["id"])
        label = f"[{child['name']}]({link})" if link else child["name"]
        rows.append(
            f"| {label} | `{child.get('path', '')}` | {len(child.get('members', []))} |"
        )
    sections = [
        _section(
            page,
            "Overview",
            overview,
            level=2,
            order=0,
            covers=[component["id"], *[child["id"] for child in children]],
        ),
        _section(
            page,
            "Child Components",
            "\n".join(rows),
            level=2,
            order=1,
            covers=[child["id"] for child in children],
        ),
    ]
    return sections


def _component_sections(
    graph: GraphStore,
    component_tree: dict[str, Any],
    component: dict[str, Any],
    page: dict[str, Any],
) -> list[dict[str, Any]]:
    if component.get("children"):
        return _parent_component_sections(graph, component_tree, component, page)
    members = component.get("members", [])
    files = _members(graph, component, "file")
    symbols = _members(graph, component, "symbol")
    apis = _members(graph, component, "api")
    overview = (
        f"{component.get('summary', '')}\n\n"
        f"- path: `{component.get('path', '')}`\n"
        f"- files: {len(files)}\n"
        f"- symbols: {len(symbols)}\n"
        f"- public APIs: {len(apis)}"
    )
    sections = [
        _section(
            page,
            "Overview",
            overview,
            level=2,
            order=0,
            covers=[component["id"]],
        )
    ]

    api_content = "\n\n".join(_api_block(graph, api) for api in apis)
    sections.append(
        _section(
            page,
            "Public API",
            api_content or "No public API nodes detected in this component.",
            level=2,
            order=1,
            covers=[api["id"] for api in apis],
            citations=[_citation(graph, api["id"]) or "" for api in apis],
        )
    )

    implementation = "\n\n".join(_symbol_block(graph, symbol) for symbol in symbols)
    sections.append(
        _section(
            page,
            "Implementation",
            implementation or "No symbols detected in this component.",
            level=2,
            order=2,
            covers=[symbol["id"] for symbol in symbols],
            citations=[_citation(graph, symbol["id"]) or "" for symbol in symbols],
        )
    )

    member_ids = set(members)
    dependency_lines: list[str] = []
    dependency_covers: set[str] = set()
    dependency_citations: set[str] = set()
    for member_id in sorted(members):
        for edge in graph.outgoing(member_id, {"imports", "calls"}):
            target = edge["dst"]
            if target in member_ids:
                continue
            target_node = graph.nodes.get(target)
            if target_node is None:
                continue
            target_attrs = target_node["attrs"]
            label = (
                target_attrs.get("qualified_name") or target_attrs.get("path") or target
            )
            citation = _citation(graph, target)
            dependency_lines.append(
                f"- `{edge['type']}` → `{label}`"
                + (f" ({citation})" if citation else "")
            )
            dependency_covers.add(target)
            if citation:
                dependency_citations.add(citation)
    sections.append(
        _section(
            page,
            "Dependencies",
            "\n".join(sorted(set(dependency_lines)))
            or "No external dependencies detected.",
            level=2,
            order=3,
            covers=dependency_covers,
            citations=dependency_citations,
        )
    )
    return sections


def _render_markdown(page: dict[str, Any], sections: list[dict[str, Any]]) -> str:
    lines = [f"# {page['title']}", ""]
    for section in sections:
        lines.extend(
            [f"## {section['heading']}", "", section.get("_content", "").strip(), ""]
        )
    return "\n".join(lines).rstrip() + "\n"


def _add_doc_nodes(
    graph: GraphStore,
    component_tree: dict[str, Any],
    page: dict[str, Any],
    *,
    parent_id: str | None,
    source_graph_version: str,
) -> None:
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
    if parent_id:
        graph.add_edge("child_of", page["id"], parent_id)
        graph.add_edge("derived_from", parent_id, page["id"])
    for section in page.get("sections", []):
        graph.add_node(
            "doc_section",
            section["id"],
            {
                "page_id": page["id"],
                "heading": section["heading"],
                "level": section["level"],
                "order": section["order"],
                "content": section.get("_content", ""),
                "content_hash": section["content_hash"],
                "covers": section["covers"],
                "citations": section["citations"],
                "source_graph_version": source_graph_version,
            },
        )
        graph.add_edge("child_of", section["id"], page["id"])
        for covered in section["covers"]:
            if covered in graph.nodes:
                graph.add_edge("cites", section["id"], covered)
    for child in page.get("children", []):
        _add_doc_nodes(
            graph,
            component_tree,
            child,
            parent_id=page["id"],
            source_graph_version=source_graph_version,
        )


def generate_documents(
    graph: GraphStore,
    component_tree: dict[str, Any],
    doc_tree: dict[str, Any],
    *,
    docs_dir: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], str]:
    """Generate markdown, add document nodes to the graph, return updated tree.

    Returns (doc_tree, pages, sections, doc_version).
    """
    docs_path = Path(docs_dir)
    docs_path.mkdir(parents=True, exist_ok=True)
    components = _component_by_id(component_tree)
    source_graph_version = doc_tree["source_graph_version"]
    pages: list[dict[str, Any]] = []
    sections: list[dict[str, Any]] = []

    def visit(page: dict[str, Any], *, is_root: bool) -> None:
        component = components[page["component_id"]]
        if is_root:
            page_sections = _root_sections(graph, component_tree, page)
        else:
            page_sections = _component_sections(graph, component_tree, component, page)

        for section in page_sections:
            section["_content"] = section.pop("content")
        page["sections"] = []
        for section in page_sections:
            public_section = {
                key: value for key, value in section.items() if key != "_content"
            }
            page["sections"].append(public_section)
            sections.append(
                {
                    **public_section,
                    "page_path": page["path"],
                    "content": section["_content"],
                }
            )
        # Keep content on the in-memory page until document nodes are written,
        # then strip it before validating/serializing the doc tree.
        for section, with_content in zip(page["sections"], page_sections):
            section["_content"] = with_content["_content"]
        markdown = _render_markdown(page, page_sections)
        (docs_path / page["path"]).write_text(markdown, encoding="utf-8")
        page["content_hash"] = sha256_text(markdown)
        page["status"] = "generated"
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
        for child in page.get("children", []):
            visit(child, is_root=False)

    visit(doc_tree["root"], is_root=True)
    _add_doc_nodes(
        graph,
        component_tree,
        doc_tree["root"],
        parent_id=None,
        source_graph_version=source_graph_version,
    )

    for page in _walk_pages(doc_tree["root"]):
        page.pop("_content", None)
        for section in page.get("sections", []):
            section.pop("_content", None)

    final_graph_version = graph.graph_version
    doc_tree["graph_version"] = final_graph_version
    markdown = {
        page["path"]: (docs_path / page["path"]).read_text(encoding="utf-8")
        for page in _walk_pages(doc_tree["root"])
    }
    doc_version = "d_" + stable_hash({"tree": doc_tree, "markdown": markdown})
    doc_tree["doc_version"] = doc_version
    return doc_tree, pages, sections, doc_version


def _walk_pages(page: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield page
    for child in page.get("children", []):
        yield from _walk_pages(child)


def regenerate_page(
    graph: GraphStore,
    component_tree: dict[str, Any],
    doc_tree: dict[str, Any],
    page: dict[str, Any],
    *,
    docs_dir: str | Path,
    source_graph_version: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Regenerate one deterministic page and replace its doc nodes."""
    components = _component_by_id(component_tree)
    if page["component_id"] == component_tree["root"]["id"]:
        page_sections = _root_sections(graph, component_tree, page)
    else:
        page_sections = _component_sections(
            graph, component_tree, components[page["component_id"]], page
        )
    for section in page_sections:
        section["_content"] = section.pop("content")
    page["sections"] = [
        {key: value for key, value in section.items() if key != "_content"}
        for section in page_sections
    ]
    markdown = _render_markdown(page, page_sections)
    docs_path = Path(docs_dir)
    docs_path.mkdir(parents=True, exist_ok=True)
    (docs_path / page["path"]).write_text(markdown, encoding="utf-8")
    page["content_hash"] = sha256_text(markdown)
    page["status"] = "generated"

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
            "section_count": len(page["sections"]),
        },
    )
    graph.add_edge("documents", page["id"], page["component_id"])
    section_rows: list[dict[str, Any]] = []
    for section in page_sections:
        graph.add_node(
            "doc_section",
            section["id"],
            {
                "page_id": page["id"],
                "heading": section["heading"],
                "level": section["level"],
                "order": section["order"],
                "content": section.get("_content", ""),
                "content_hash": section["content_hash"],
                "covers": section["covers"],
                "citations": section["citations"],
                "source_graph_version": source_graph_version,
            },
        )
        graph.add_edge("child_of", section["id"], page["id"])
        for covered in section["covers"]:
            if covered in graph.nodes:
                graph.add_edge("cites", section["id"], covered)
        section_rows.append(
            {
                **{key: value for key, value in section.items() if key != "_content"},
                "page_path": page["path"],
                "content": section["_content"],
            }
        )
    return page, section_rows


def write_doc_indexes(
    state_dir: str | Path,
    pages: list[dict[str, Any]],
    sections: list[dict[str, Any]],
) -> tuple[Path, Path]:
    docs_state = Path(state_dir) / "docs"
    docs_state.mkdir(parents=True, exist_ok=True)
    pages_path = docs_state / "pages.jsonl"
    sections_path = docs_state / "sections.jsonl"
    with pages_path.open("w", encoding="utf-8") as f:
        for page in pages:
            f.write(json.dumps(page, ensure_ascii=False) + "\n")
    with sections_path.open("w", encoding="utf-8") as f:
        for section in sections:
            f.write(json.dumps(section, ensure_ascii=False) + "\n")
    return pages_path, sections_path
