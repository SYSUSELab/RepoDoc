"""Toy component clustering and document planning.

The first prototype groups files into components by directory (or file stem at
the repository root). A later stage replaces this with an LLM clusterer, but the
tree shape and graph write-back are already the real contract.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from repodoc.graph.store import GraphStore
from repodoc.schema import (
    COMPONENT_KINDS,
    SCHEMA_VERSION,
    component_node_id,
    doc_page_id,
    slugify,
    stable_hash,
    utc_now,
)


def _component(
    *,
    repo: str,
    name: str,
    kind: str,
    path: str,
    parent_path: str,
    depth: int,
    members: list[str],
    children: list[dict[str, Any]],
    summary: str,
    signals: list[str],
    rationale: str,
    confidence: float = 1.0,
) -> dict[str, Any]:
    if kind not in COMPONENT_KINDS:
        raise ValueError(f"unknown component kind: {kind}")
    return {
        "id": component_node_id(repo, parent_path, name),
        "name": name,
        "kind": kind,
        "path": path,
        "summary": summary,
        "depth": depth,
        "members": sorted(set(members)),
        "children": children,
        "confidence": confidence,
        "evidence": {"signals": signals, "rationale": rationale},
        "coverage": {
            "assigned_nodes": len(set(members)),
            "unassigned_nodes": 0,
            "unassigned": [],
        },
    }


def _file_members(graph: GraphStore, file_id: str) -> list[str]:
    members: set[str] = {file_id}
    stack = [file_id]
    while stack:
        current = stack.pop()
        for edge in graph.outgoing(current, {"contains", "exposes_api"}):
            if edge["dst"] not in members:
                members.add(edge["dst"])
                stack.append(edge["dst"])
    return sorted(members)


def build_component_tree(
    graph: GraphStore,
    *,
    repo_name: str | None = None,
) -> dict[str, Any]:
    """Build the toy component tree and write component nodes/edges to the graph."""
    repo = repo_name or graph.repo
    file_nodes = graph.nodes_of_type("file")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for file_node in file_nodes:
        rel_path = file_node["attrs"]["path"]
        parent = Path(rel_path).parent.as_posix()
        key = parent if parent != "." else Path(rel_path).stem
        groups[key].append(file_node)

    child_components: list[dict[str, Any]] = []
    all_code_node_ids = {
        node["id"] for node in graph.nodes.values() if node["layer"] == "code"
    }
    assigned: set[str] = set()

    for key in sorted(groups):
        files = sorted(groups[key], key=lambda node: node["attrs"]["path"])
        members: list[str] = []
        for file_node in files:
            members.extend(_file_members(graph, file_node["id"]))
        members = sorted(set(members))
        assigned.update(members)
        name = key.replace("/", "_")
        path = key if "/" in key else files[0]["attrs"]["path"]
        component = _component(
            repo=repo,
            name=name,
            kind="module",
            path=path,
            parent_path="",
            depth=1,
            members=members,
            children=[],
            summary=(
                f"Toy component grouping {len(files)} file(s): "
                + ", ".join(file["attrs"]["path"] for file in files)
            ),
            signals=["toy:file-grouping"],
            rationale="Grouped by directory, or by file stem for root-level files.",
        )
        child_components.append(component)

    root_members = sorted(all_code_node_ids)
    root = _component(
        repo=repo,
        name=repo,
        kind="repository",
        path=".",
        parent_path="",
        depth=0,
        members=root_members,
        children=child_components,
        summary=f"Toy repository {repo} with {len(child_components)} component(s).",
        signals=["toy:repository-root"],
        rationale="Single repository root for the prototype.",
    )
    unassigned = sorted(all_code_node_ids - assigned)
    root["coverage"]["unassigned"] = unassigned
    root["coverage"]["unassigned_nodes"] = len(unassigned)

    tree = {
        "schema_version": SCHEMA_VERSION,
        "repo": repo,
        "repo_path": graph.repo_path,
        "commit": graph.commit,
        "source_graph_version": graph.code_graph_version,
        "cluster_version": "",
        "created_at": utc_now(),
        "root": root,
        "cluster_meta": {
            "mode": "toy-file-grouping",
            "model": None,
            "prompt_version": "toy-v1",
            "note": "Deterministic placeholder for the LLM clustering stage.",
        },
    }
    tree["cluster_version"] = "c_" + stable_hash(tree)

    _write_components_to_graph(graph, root)
    return tree


def _write_components_to_graph(graph: GraphStore, component: dict[str, Any]) -> None:
    attrs = {
        "name": component["name"],
        "kind": component["kind"],
        "path": component["path"],
        "summary": component["summary"],
        "depth": component["depth"],
        "confidence": component["confidence"],
        "members": component["members"],
        "evidence": component["evidence"],
        "coverage": component["coverage"],
    }
    graph.add_node("component", component["id"], attrs)
    # The repository root is an ancestor, not the direct owner of every node.
    # Leaf components carry the part_of_component edges; the root is reachable
    # through has_child_component edges.
    if component["kind"] != "repository":
        for member_id in component["members"]:
            if member_id in graph.nodes:
                graph.add_edge("part_of_component", member_id, component["id"])
    for child in component["children"]:
        _write_components_to_graph(graph, child)
        graph.add_edge("has_child_component", component["id"], child["id"])


def build_component_tree_from_groups(
    graph: GraphStore,
    *,
    groups: list[dict[str, Any]],
    repo_name: str | None = None,
    cluster_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a component tree from an LLM-provided group specification.

    `groups` is a nested list of:
    {"name", "path", "summary", "members", "children"}. Every member must be a
    code node id; unknown ids are dropped and unassigned nodes are collected.
    """
    repo = repo_name or graph.repo
    valid_ids = {node["id"] for node in graph.nodes.values() if node["layer"] == "code"}
    assigned: set[str] = set()

    def make(
        spec: dict[str, Any], depth: int, parent_path: str
    ) -> dict[str, Any] | None:
        name = str(spec.get("name") or f"component_{depth}").strip()
        if not name:
            return None
        members: list[str] = []
        for member in spec.get("members") or []:
            member_id = str(member)
            if member_id in valid_ids and member_id not in assigned:
                assigned.add(member_id)
                members.append(member_id)
        child_specs = spec.get("children") or []
        children = [
            child
            for child_spec in child_specs
            if isinstance(child_spec, dict)
            for child in [make(child_spec, depth + 1, f"{parent_path}/{name}")]
            if child is not None
        ]
        if not members and not children:
            return None
        path = str(spec.get("path") or "")
        summary = str(
            spec.get("summary")
            or f"LLM-proposed component {name} with {len(members)} code node(s)."
        )
        return _component(
            repo=repo,
            name=name,
            kind="component" if depth == 1 else "module",
            path=path,
            parent_path=parent_path,
            depth=depth,
            members=members,
            children=children,
            summary=summary,
            signals=["llm:clustering"],
            rationale=str(
                spec.get("rationale") or "LLM grouped related code components."
            ),
        )

    child_components = [
        component
        for spec in groups
        if isinstance(spec, dict)
        for component in [make(spec, 1, "")]
        if component is not None
    ]
    if not child_components:
        raise ValueError("LLM clustering produced no non-empty components")

    unassigned = sorted(valid_ids - assigned)
    if unassigned:
        child_components.append(
            _component(
                repo=repo,
                name="unassigned",
                kind="module",
                path="",
                parent_path="",
                depth=1,
                members=unassigned,
                children=[],
                summary="Code nodes the clustering response did not assign.",
                signals=["fallback:unassigned"],
                rationale="Safety net: every code node must belong to some component.",
            )
        )
        assigned.update(unassigned)

    root = _component(
        repo=repo,
        name=repo,
        kind="repository",
        path=".",
        parent_path="",
        depth=0,
        members=sorted(valid_ids),
        children=child_components,
        summary=f"Repository {repo} with {len(child_components)} component(s).",
        signals=["repository-root"],
        rationale="Single repository root.",
    )
    root["coverage"]["unassigned"] = unassigned
    root["coverage"]["unassigned_nodes"] = len(unassigned)

    meta = {
        "mode": "llm",
        "model": None,
        "prompt_version": "cluster-v1",
        **(cluster_meta or {}),
    }
    tree = {
        "schema_version": SCHEMA_VERSION,
        "repo": repo,
        "repo_path": graph.repo_path,
        "commit": graph.commit,
        "source_graph_version": graph.code_graph_version,
        "cluster_version": "",
        "created_at": utc_now(),
        "root": root,
        "cluster_meta": meta,
    }
    tree["cluster_version"] = "c_" + stable_hash(tree)
    _write_components_to_graph(graph, root)
    return tree


def _doc_type_for_depth(depth: int) -> str:
    if depth == 0:
        return "architecture"
    if depth == 1:
        return "component"
    if depth == 2:
        return "module"
    return "implementation"


def _page_slug(value: str, fallback: str = "component") -> str:
    return slugify(value, fallback=fallback).replace("_", "-")


def _unique_page_path(
    component: dict[str, Any],
    used: set[str],
    *,
    parent_name: str = "",
) -> str:
    """Choose a readable, deterministic page filename.

    Numeric suffixes such as ``-2`` lose the component's context. Prefer a
    path-derived qualifier when two components share a display name.
    """

    name = _page_slug(str(component.get("name") or "component"))
    path = str(component.get("path") or "").strip("./")
    path_parts = [part for part in path.split("/") if part]
    path_name = _page_slug(Path(path).name) if path_parts else ""
    path_context = (
        "-".join(_page_slug(part) for part in path_parts[-2:]) if path_parts else ""
    )
    parent = _page_slug(parent_name) if parent_name else ""

    candidates = [name]
    for candidate in (
        path_name,
        path_context,
        f"{parent}-{name}" if parent else "",
        f"{path_name}-{name}" if path_name else "",
        f"{name}-{stable_hash(str(component.get('id') or name))[:6]}",
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        page = f"{candidate}.md"
        if page not in used:
            used.add(page)
            return page
    # The hash candidate is deterministic; this is only a total-function
    # fallback for an already-used hash.
    page = f"{name}-{stable_hash(str(component.get('id') or name))[:8]}.md"
    used.add(page)
    return page


def _page_for_component(
    component: dict[str, Any],
    *,
    source_graph_version: str,
    used_paths: set[str],
    parent_name: str = "",
) -> dict[str, Any]:
    page_path = _unique_page_path(component, used_paths, parent_name=parent_name)
    page_id = doc_page_id(page_path)
    children = [
        _page_for_component(
            child,
            source_graph_version=source_graph_version,
            used_paths=used_paths,
            parent_name=str(component.get("name") or ""),
        )
        for child in component["children"]
    ]
    return {
        "id": page_id,
        "path": page_path,
        "title": component["name"],
        "level": max(0, 3 - component["depth"]),
        "doc_type": _doc_type_for_depth(component["depth"]),
        "component_id": component["id"],
        "status": "planned",
        "content_hash": "",
        "generated_from": {
            "graph_version": source_graph_version,
            "child_doc_ids": [child["id"] for child in children],
        },
        "children": children,
        "sections": [],
    }


def write_component_tree_to_graph(
    graph: GraphStore, component_tree: dict[str, Any]
) -> None:
    """Write an existing component tree (e.g. reused from a prior run) to graph."""
    _write_components_to_graph(graph, component_tree["root"])


def page_for_component(
    component: dict[str, Any],
    *,
    source_graph_version: str,
    used_paths: set[str],
) -> dict[str, Any]:
    """Public helper for creating one page spec (used by incremental update)."""
    return _page_for_component(
        component,
        source_graph_version=source_graph_version,
        used_paths=used_paths,
    )


def build_doc_plan(
    component_tree: dict[str, Any],
    graph: GraphStore,
    *,
    repo_name: str | None = None,
) -> dict[str, Any]:
    """Map the component tree to a document tree.

    The current mapping is one overview page plus one page per component.
    A future planner can merge/split components, add cross-cutting pages, or
    skip levels based on repository size.
    """
    repo = repo_name or component_tree["repo"]
    source_graph_version = component_tree["source_graph_version"]
    root_component = component_tree["root"]
    used_paths = {"overview.md"}
    child_pages = [
        _page_for_component(
            child,
            source_graph_version=source_graph_version,
            used_paths=used_paths,
        )
        for child in root_component["children"]
    ]
    root_page = {
        "id": doc_page_id("overview.md"),
        "path": "overview.md",
        "title": "Overview",
        "level": 3,
        "doc_type": "architecture",
        "component_id": root_component["id"],
        "status": "planned",
        "content_hash": "",
        "generated_from": {
            "graph_version": source_graph_version,
            "child_doc_ids": [page["id"] for page in child_pages],
        },
        "children": child_pages,
        "sections": [],
    }
    tree = {
        "schema_version": SCHEMA_VERSION,
        "repo": repo,
        "repo_path": graph.repo_path,
        "commit": graph.commit,
        "source_graph_version": source_graph_version,
        "graph_version": source_graph_version,
        "doc_plan_version": "",
        "doc_version": "",
        "created_at": utc_now(),
        "root": root_page,
    }
    tree["doc_plan_version"] = "d_" + stable_hash(tree)
    return tree
