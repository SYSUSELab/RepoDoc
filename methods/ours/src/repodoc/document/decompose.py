"""Deterministic decomposition of oversized components.

LLM clustering often produces a very large component (e.g. a test suite with
thousands of nodes). A single page then cannot fit the model context. This
module splits such components by source-file groups and leaves the parent page
to synthesize the child pages.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from repodoc.graph.store import GraphStore
from repodoc.schema import component_node_id, slugify, stable_hash, utc_now

MAX_COMPONENT_NODES = 300
MAX_CHILDREN = 8
MIN_CHILD_NODES = 80


def _component_map(root: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}

    def visit(component: dict[str, Any]) -> None:
        mapping[component["id"]] = component
        for child in component.get("children", []):
            visit(child)

    visit(root)
    return mapping


def _file_groups(
    graph: GraphStore, component: dict[str, Any]
) -> list[tuple[str, list[str]]]:
    by_path: dict[str, list[str]] = defaultdict(list)
    for node_id in component.get("members", []):
        node = graph.nodes.get(node_id)
        if node is None:
            continue
        path = node["attrs"].get("path") or "<unknown>"
        by_path[path].append(node_id)
    return [(path, sorted(ids)) for path, ids in sorted(by_path.items())]


def _chunk_groups(groups: list[tuple[str, list[str]]], target: int) -> list[list[str]]:
    # A single source file can itself be larger than the target. Split its
    # member list (file + symbols/APIs) into target-sized slices first.
    expanded: list[tuple[str, list[str]]] = []
    for path, node_ids in groups:
        if len(node_ids) > target:
            parts = (len(node_ids) + target - 1) // target
            size = (len(node_ids) + parts - 1) // parts
            for start in range(0, len(node_ids), size):
                expanded.append((path, node_ids[start : start + size]))
        else:
            expanded.append((path, node_ids))

    chunks: list[list[str]] = []
    current: list[str] = []
    for _path, node_ids in expanded:
        if current and len(current) + len(node_ids) > target:
            chunks.append(current)
            current = []
        current.extend(node_ids)
    if current:
        chunks.append(current)
    return chunks


def _common_path(paths: list[str]) -> str:
    if not paths:
        return ""
    parts = [Path(path).parts for path in paths if path]
    if not parts:
        return ""
    common = []
    for column in zip(*parts):
        if all(part == column[0] for part in column):
            common.append(column[0])
        else:
            break
    return "/".join(common)


def _child_component(
    graph: GraphStore,
    parent: dict[str, Any],
    name: str,
    members: list[str],
) -> dict[str, Any]:
    paths = sorted(
        {
            graph.nodes[node_id]["attrs"].get("path")
            for node_id in members
            if node_id in graph.nodes and graph.nodes[node_id]["attrs"].get("path")
        }
    )
    base = _common_path(paths) or (paths[0] if paths else parent["path"])
    component_id = component_node_id(graph.repo, parent["id"], name)
    return {
        "id": component_id,
        "name": name,
        "kind": "module",
        "path": base,
        "summary": f"Subcomponent {name} of {parent['name']} ({len(members)} nodes).",
        "depth": parent["depth"] + 1,
        "members": sorted(set(members)),
        "children": [],
        "confidence": 0.75,
        "evidence": {
            "signals": ["decompose:large-component"],
            "rationale": "Split an oversized component by source-file groups.",
        },
        "coverage": {
            "assigned_nodes": len(set(members)),
            "unassigned_nodes": 0,
            "unassigned": [],
        },
    }


_GENERIC_SYMBOLS = {
    "__init__",
    "__call__",
    "index",
    "get",
    "post",
    "put",
    "delete",
    "head",
    "options",
    "run",
    "main",
    "setup",
    "teardown",
    "setUp",
    "tearDown",
    "helper",
    "default",
}


def _chunk_paths(graph: GraphStore, members: list[str]) -> list[str]:
    return sorted(
        {
            graph.nodes[node_id]["attrs"].get("path")
            for node_id in members
            if node_id in graph.nodes and graph.nodes[node_id]["attrs"].get("path")
        }
    )


def _name_slug(value: str, fallback: str = "module") -> str:
    return slugify(value, fallback=fallback).replace("_", "-")


def _file_label(path: str) -> str:
    stem = Path(path).stem
    if stem in {"__init__", "init", "mod"}:
        parent = Path(path).parent.name
        return _name_slug(parent or stem)
    return _name_slug(stem)


def _chunk_topics(graph: GraphStore, members: list[str]) -> list[str]:
    """Find source-level names that describe a chunk's contents."""

    topics: list[str] = []
    for node_id in sorted(members):
        node = graph.nodes.get(node_id)
        if node is None or node["type"] not in {"symbol", "api"}:
            continue
        attrs = node["attrs"]
        raw = attrs.get("name") or attrs.get("qualified_name") or ""
        raw = str(raw).split(".")[-1]
        if not raw or raw in _GENERIC_SYMBOLS or raw.startswith("_"):
            continue
        topic = _name_slug(raw, fallback="")
        if topic and topic not in topics:
            topics.append(topic)
    return topics


def _topic_label(stem: str, topic: str) -> str:
    if topic == stem or topic.startswith(f"{stem}-"):
        return topic
    return f"{stem}-{topic}"


def _chunk_name(
    graph: GraphStore,
    parent: dict[str, Any],
    members: list[str],
    *,
    path_chunk_counts: Counter[str],
    used_names: set[str],
) -> str:
    """Build a semantic, stable name for a decomposed child.

    Numeric ``part_N`` suffixes are deliberately avoided: a page name should
    tell the reader which files or source concepts it covers. The file stem is
    the base; when a file or stem is split across chunks, the first meaningful
    symbol/API name disambiguates it. A short content hash is only a last-resort
    fallback when two chunks genuinely have no semantic distinction.
    """

    paths = _chunk_paths(graph, members)
    topics = _chunk_topics(graph, members)
    labels = [_file_label(path) for path in paths]
    labels = list(dict.fromkeys(labels))
    single_file_split = bool(
        paths and len(paths) == 1 and path_chunk_counts.get(paths[0], 0) > 1
    )
    stem = labels[0] if labels else "component"
    if paths and len(paths) == 1:
        base = _topic_label(stem, topics[0]) if single_file_split and topics else stem
    elif labels:
        base = "-".join(labels[:3])
    else:
        base = _file_label(parent["path"]) if parent.get("path") else "component"
    base = base[:72].strip("-") or "component"

    candidates = [base]
    for topic in topics:
        # Do not repeat the topic already used as the base qualifier.
        candidate = (
            _topic_label(stem, topic) if single_file_split else f"{base}-{topic}"
        )
        if candidate not in candidates:
            candidates.append(candidate)
    for path in paths:
        parent_label = _name_slug(Path(path).parent.name, fallback="")
        candidate = f"{parent_label}-{base}" if parent_label else base
        if candidate not in candidates:
            candidates.append(candidate)
    start = min(
        (
            graph.nodes[node_id]["attrs"].get("start", {}).get("line", 0)
            for node_id in members
            if node_id in graph.nodes and graph.nodes[node_id]["attrs"].get("start")
        ),
        default=0,
    )
    end = max(
        (
            graph.nodes[node_id]["attrs"].get("end", {}).get("line", 0)
            for node_id in members
            if node_id in graph.nodes and graph.nodes[node_id]["attrs"].get("end")
        ),
        default=0,
    )
    if start or end:
        candidates.append(f"{base}-lines-{start}-{end}")
    candidates.append(f"{base}-{stable_hash(sorted(members))[:6]}")

    for candidate in candidates:
        candidate = candidate[:96].strip("-")
        if candidate and candidate not in used_names:
            return candidate
    # A used hash collision is astronomically unlikely, but keep the operation
    # total and deterministic.
    return f"{base}-{stable_hash(sorted(members))[:8]}"


def _decompose_one(
    graph: GraphStore,
    component: dict[str, Any],
    *,
    max_nodes: int,
    max_children: int,
    min_child_nodes: int,
) -> bool:
    for child in component.get("children", []):
        _decompose_one(
            graph,
            child,
            max_nodes=max_nodes,
            max_children=max_children,
            min_child_nodes=min_child_nodes,
        )
    if component.get("children") or len(component.get("members", [])) <= max_nodes:
        return False
    groups = _file_groups(graph, component)
    if len(groups) <= 1:
        return False
    total = sum(len(node_ids) for _, node_ids in groups)
    target = max(min_child_nodes, (total + max_children - 1) // max_children)
    chunks = _chunk_groups(groups, target)
    if len(chunks) <= 1:
        return False

    path_chunk_counts: Counter[str] = Counter()
    for members in chunks:
        for path in _chunk_paths(graph, members):
            path_chunk_counts[path] += 1

    used_names = {child["name"] for child in component.get("children", [])}
    children: list[dict[str, Any]] = []
    for members in chunks:
        name = _chunk_name(
            graph,
            component,
            members,
            path_chunk_counts=path_chunk_counts,
            used_names=used_names,
        )
        used_names.add(name)
        children.append(_child_component(graph, component, name, members))

    component["members"] = []
    component["coverage"] = {
        "assigned_nodes": 0,
        "unassigned_nodes": 0,
        "unassigned": [],
    }
    component["children"] = children
    component["evidence"] = {
        **component.get("evidence", {}),
        "signals": list(component.get("evidence", {}).get("signals", []))
        + ["decompose:large-component"],
        "rationale": (
            component.get("evidence", {}).get("rationale", "")
            + " Oversized component split into child pages."
        ).strip(),
    }
    graph.add_node(
        "component",
        component["id"],
        {
            "name": component["name"],
            "kind": component["kind"],
            "path": component["path"],
            "summary": component["summary"],
            "depth": component["depth"],
            "confidence": component["confidence"],
            "members": [],
            "evidence": component["evidence"],
            "coverage": component["coverage"],
        },
    )
    for child in children:
        graph.add_node("component", child["id"], child)
        graph.add_edge("has_child_component", component["id"], child["id"])
        _decompose_one(
            graph,
            child,
            max_nodes=max_nodes,
            max_children=max_children,
            min_child_nodes=min_child_nodes,
        )
    return True


def decompose_large_components(
    graph: GraphStore,
    component_tree: dict[str, Any],
    *,
    max_nodes: int = MAX_COMPONENT_NODES,
    max_children: int = MAX_CHILDREN,
    min_child_nodes: int = MIN_CHILD_NODES,
) -> int:
    changed = 0
    for child in component_tree["root"].get("children", []):
        if _decompose_one(
            graph,
            child,
            max_nodes=max_nodes,
            max_children=max_children,
            min_child_nodes=min_child_nodes,
        ):
            changed += 1

    if changed:
        # Rebuild membership edges after moving members to children.
        for edge_id, edge in list(graph.edges.items()):
            if edge["type"] == "part_of_component":
                graph.remove_edge(edge_id)
        mapping = _component_map(component_tree["root"])
        for component in mapping.values():
            if component.get("kind") == "repository":
                continue
            for member_id in component.get("members", []):
                if member_id in graph.nodes:
                    graph.add_edge("part_of_component", member_id, component["id"])
        component_tree["cluster_meta"]["decompose"] = {
            "mode": "file-group-split",
            "max_nodes": max_nodes,
            "max_children": max_children,
            "updated_at": utc_now(),
        }
        component_tree["cluster_version"] = "c_" + stable_hash(component_tree)
    return changed
