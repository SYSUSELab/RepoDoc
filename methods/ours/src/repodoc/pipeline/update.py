"""Graph-driven incremental update for the prototype.

MVP behavior:
- build the new code graph from the target checkout;
- diff the code layer;
- repair component membership by stable file paths;
- regenerate only affected doc pages deterministically;
- write graph/doc/manifest versions and an applied change set.

New files get an `unassigned` component; full re-clustering remains a later step.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from repodoc.change.gitdiff import git_changed_files, git_head
from repodoc.change.graph_diff import diff_code_graphs
from repodoc.document.generator import regenerate_page
from repodoc.document.llm_patch import apply_section_patches
from repodoc.document.planner import page_for_component
from repodoc.graph.builder import CodeGraphBuilder
from repodoc.graph.languages import detect_analyzer
from repodoc.graph.store import GraphStore
from repodoc.pipeline.manifest import (
    read_manifest,
    source_hash,
    status_info,
    write_manifest,
)
from repodoc.schema import (
    SCHEMA_VERSION,
    component_node_id,
    stable_hash,
    utc_now,
    validate,
)


def _docs_dir(state: Path, manifest: dict[str, Any]) -> Path:
    value = (manifest.get("artifacts") or {}).get("markdown")
    if not value:
        return state / "docs" / "markdown"
    path = Path(value)
    return path if path.is_absolute() else state / path


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
        mapping[page["id"]] = page
        for child in page.get("children", []):
            visit(child)

    visit(doc_tree["root"])
    return mapping


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _old_path_to_component(
    graph: GraphStore, component_tree: dict[str, Any]
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for component in _component_map(component_tree).values():
        if component.get("kind") == "repository":
            continue
        for member_id in component.get("members", []):
            node = graph.nodes.get(member_id)
            if node is None:
                continue
            path = node["attrs"].get("path")
            if path:
                mapping.setdefault(path, component["id"])
    return mapping


def _old_dir_to_component(
    graph: GraphStore, component_tree: dict[str, Any]
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for component in _component_map(component_tree).values():
        if component.get("kind") == "repository":
            continue
        for member_id in component.get("members", []):
            node = graph.nodes.get(member_id)
            if node is None:
                continue
            path = node["attrs"].get("path")
            if path:
                mapping.setdefault(Path(path).parent.as_posix(), component["id"])
    return mapping


def _semantic_changed_ids(
    old: GraphStore, new: GraphStore, delta: dict[str, Any]
) -> set[str]:
    changed = set(delta.get("nodes_added") or []) | set(
        delta.get("nodes_removed") or []
    )
    for node_id in delta.get("nodes_modified") or []:
        old_node = old.nodes.get(node_id)
        new_node = new.nodes.get(node_id)
        if old_node is None or new_node is None:
            changed.add(node_id)
            continue
        old_attrs = old_node.get("attrs") or {}
        new_attrs = new_node.get("attrs") or {}
        keys = (
            ("hash", "loc", "docstring")
            if old_node["type"] == "file"
            else ("source", "signature", "docstring", "name", "kind")
        )
        if any(old_attrs.get(key) != new_attrs.get(key) for key in keys):
            changed.add(node_id)
    return changed


def _affected_section_ids(
    graph: GraphStore, page_id: str, changed_ids: set[str]
) -> set[str]:
    affected: set[str] = set()
    for section in graph.nodes_of_type("doc_section"):
        if section["attrs"].get("page_id") != page_id:
            continue
        for edge in graph.outgoing(section["id"], {"cites"}):
            if edge["dst"] in changed_ids:
                affected.add(section["id"])
                break
    return affected


def _remove_page_from_graph(graph: GraphStore, page_id: str) -> None:
    for node in list(graph.nodes.values()):
        if node["type"] == "doc_section" and node["attrs"].get("page_id") == page_id:
            graph.remove_node(node["id"])
    graph.remove_node(page_id)


def _remove_page_from_tree(page: dict[str, Any], page_id: str) -> bool:
    children = page.get("children", [])
    for index, child in enumerate(list(children)):
        if child["id"] == page_id:
            children.pop(index)
            return True
        if _remove_page_from_tree(child, page_id):
            return True
    return False


def _impact_scope(
    old_graph: GraphStore,
    new_graph: GraphStore,
    delta: dict[str, Any],
    component_tree: dict[str, Any],
    doc_tree: dict[str, Any],
) -> dict[str, Any]:
    """Build the deterministic candidate universe for update planning.

    ``direct_pages`` are the only pages whose sections cite a changed code
    node. ``candidates`` is deliberately larger so an LLM planner can escalate
    to a parent or overview when the change has architectural impact.
    """
    changed_ids = {
        node_id
        for key in ("nodes_added", "nodes_removed", "nodes_modified")
        for node_id in delta.get(key, [])
    }
    path_to_component = _old_path_to_component(old_graph, component_tree)
    directory_to_component = _old_dir_to_component(old_graph, component_tree)
    affected_components: set[str] = set()
    for node_id in changed_ids:
        node = old_graph.nodes.get(node_id) or new_graph.nodes.get(node_id)
        if node is None:
            continue
        path = node["attrs"].get("path")
        component_id = path_to_component.get(path)
        if component_id is None and path:
            component_id = directory_to_component.get(Path(path).parent.as_posix())
        if component_id:
            affected_components.add(component_id)
        for edge in old_graph.outgoing(node_id, {"part_of_component"}):
            affected_components.add(edge["dst"])

    pages = _page_map(doc_tree)
    direct_pages: set[str] = set()
    for section in old_graph.nodes_of_type("doc_section"):
        for edge in old_graph.outgoing(section["id"], {"cites"}):
            if edge["dst"] in changed_ids:
                direct_pages.add(section["attrs"]["page_id"])

    structural = any(
        node_id in delta.get(key, [])
        and (old_graph.nodes.get(node_id) or new_graph.nodes.get(node_id))
        and (old_graph.nodes.get(node_id) or new_graph.nodes.get(node_id))["type"]
        == "file"
        for node_id in changed_ids
        for key in ("nodes_added", "nodes_removed")
    )

    def node_component(graph: GraphStore, node_id: str) -> str | None:
        node = graph.nodes.get(node_id)
        if node is None:
            return None
        return path_to_component.get(node["attrs"].get("path"))

    cross_component_edge = False
    for edge_id in set(delta.get("edges_added") or []) | set(
        delta.get("edges_removed") or []
    ):
        # Modified edges appear in both add/remove sides of the graph delta;
        # only genuinely new or removed cross-component edges are architecture
        # signals.
        if edge_id in old_graph.edges and edge_id in new_graph.edges:
            continue
        edge = old_graph.edges.get(edge_id) or new_graph.edges.get(edge_id)
        if edge is None:
            continue
        src_component = node_component(old_graph, edge["src"]) or node_component(
            new_graph, edge["src"]
        )
        dst_component = node_component(old_graph, edge["dst"]) or node_component(
            new_graph, edge["dst"]
        )
        if src_component and dst_component and src_component != dst_component:
            cross_component_edge = True
            break
    architecture_signal = structural or cross_component_edge

    component_pages = {
        page["id"]
        for page in pages.values()
        if page["component_id"] in affected_components
    }
    seed_pages = set(direct_pages) | component_pages
    ancestor_pages: set[str] = set()
    stack = list(seed_pages)
    while stack:
        page_id = stack.pop()
        for edge in old_graph.outgoing(page_id, {"child_of"}):
            parent = edge["dst"]
            if parent in pages and parent not in ancestor_pages:
                ancestor_pages.add(parent)
                stack.append(parent)
    root = doc_tree["root"]["id"]
    candidates = set(seed_pages) | ancestor_pages | {root}
    if structural:
        mandatory = set(component_pages) | ancestor_pages | {root}
    elif cross_component_edge:
        # A cross-component dependency is an architecture signal. Keep parent
        # pages optional, but never let the overview silently no-op it.
        mandatory = {root}
    else:
        mandatory = set()
    default_pages = set(direct_pages) if not structural and direct_pages else candidates
    return {
        "affected_components": affected_components,
        "direct_pages": direct_pages,
        "component_pages": component_pages,
        "ancestor_pages": ancestor_pages,
        "candidate_pages": candidates,
        "mandatory_pages": mandatory,
        "default_pages": default_pages,
        "structural": structural,
        "architecture_signal": architecture_signal,
        "cross_component_edge": cross_component_edge,
    }


def _change_units(
    old_graph: GraphStore,
    new_graph: GraphStore,
    delta: dict[str, Any],
    component_tree: dict[str, Any],
    affected_pages: set[str],
) -> list[dict[str, Any]]:
    path_to_component = _old_path_to_component(old_graph, component_tree)
    node_to_page: dict[str, set[str]] = defaultdict(set)
    for section in old_graph.nodes_of_type("doc_section"):
        page_id = section["attrs"].get("page_id")
        if page_id:
            for edge in old_graph.outgoing(section["id"], {"cites"}):
                node_to_page[edge["dst"]].add(page_id)

    by_path: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"added": [], "removed": [], "modified": []}
    )
    for key, op in (
        ("nodes_added", "added"),
        ("nodes_removed", "removed"),
        ("nodes_modified", "modified"),
    ):
        for node_id in delta.get(key, []):
            node = new_graph.nodes.get(node_id) or old_graph.nodes.get(node_id)
            if node is None:
                continue
            path = node["attrs"].get("path") or node_id
            by_path[path][op].append(node_id)

    units: list[dict[str, Any]] = []
    for index, (path, ops) in enumerate(sorted(by_path.items())):
        targets = ops["added"] + ops["removed"] + ops["modified"]
        if ops["added"] and not ops["removed"] and not ops["modified"]:
            op = "add"
        elif ops["removed"] and not ops["added"] and not ops["modified"]:
            op = "delete"
        else:
            op = "modify"
        node_types = {
            (new_graph.nodes.get(node_id) or old_graph.nodes.get(node_id))["type"]
            for node_id in targets
        }
        if "file" in node_types and op == "add":
            classification = "new_file"
        elif "file" in node_types and op == "delete":
            classification = "deleted_file"
        elif "symbol" in node_types or "api" in node_types:
            classification = "behavior_change"
        else:
            classification = "metadata_change"
        impacted_components = sorted(
            {
                path_to_component[path]
                for node_id in targets
                if path_to_component.get(path)
            }
        )
        impacted_docs = sorted(
            {
                page
                for node_id in targets
                for page in node_to_page.get(node_id, set())
                if page in affected_pages
            }
        )
        units.append(
            {
                "id": f"change:{index}",
                "op": op,
                "targets": sorted(set(targets)),
                "old_targets": sorted(set(ops["removed"])),
                "classification": classification,
                "impacted_components": impacted_components,
                "impacted_docs": impacted_docs,
                "recommended_actions": [
                    "recluster_local"
                    if classification in {"new_file", "deleted_file"}
                    else "regenerate_page"
                ],
                "path": path,
            }
        )
    return units


def _repair_components(
    graph: GraphStore,
    component_tree: dict[str, Any],
    new_graph: GraphStore,
    old_path_to_component: dict[str, str],
    old_dir_to_component: dict[str, str],
) -> dict[str, Any]:
    new_code_nodes = {
        node_id: node
        for node_id, node in new_graph.nodes.items()
        if node["layer"] == "code"
    }
    members_by_component: dict[str, list[str]] = defaultdict(list)
    unassigned: list[str] = []
    for node_id, node in new_code_nodes.items():
        path = node["attrs"].get("path")
        component_id = old_path_to_component.get(path)
        if not component_id and path:
            component_id = old_dir_to_component.get(Path(path).parent.as_posix())
        if component_id:
            members_by_component[component_id].append(node_id)
        else:
            unassigned.append(node_id)

    mapping = _component_map(component_tree)
    for component_id, component in mapping.items():
        if component.get("kind") == "repository":
            continue
        members = sorted(set(members_by_component.get(component_id, [])))
        component["members"] = members
        component["coverage"] = {
            "assigned_nodes": len(members),
            "unassigned_nodes": 0,
            "unassigned": [],
        }
        graph.add_node(
            "component",
            component_id,
            {
                "name": component["name"],
                "kind": component["kind"],
                "path": component["path"],
                "summary": component["summary"],
                "depth": component["depth"],
                "confidence": component["confidence"],
                "members": members,
                "evidence": component["evidence"],
                "coverage": component["coverage"],
            },
        )

    root = component_tree["root"]
    removed_components: list[str] = []

    def prune(component: dict[str, Any], is_root: bool = False) -> bool:
        component["children"] = [
            child for child in component.get("children", []) if prune(child)
        ]
        if (
            not is_root
            and not component.get("members")
            and not component.get("children")
        ):
            removed_components.append(component["id"])
            return False
        return True

    root["children"] = [child for child in root["children"] if prune(child)]
    for component_id in removed_components:
        graph.remove_node(component_id)

    new_groups: dict[str, list[str]] = defaultdict(list)
    for node_id in unassigned:
        path = new_code_nodes[node_id]["attrs"].get("path") or node_id
        parent = Path(path).parent.as_posix()
        key = parent if parent != "." else Path(path).stem
        new_groups[key].append(node_id)

    new_components: list[dict[str, Any]] = []
    for key, node_ids in sorted(new_groups.items()):
        name = key.replace("/", "_")
        component_id = component_node_id(graph.repo, "", name)
        if component_id in mapping:
            name = f"{name}_new"
            component_id = component_node_id(graph.repo, "", name)
        first_path = new_code_nodes[node_ids[0]]["attrs"].get("path", "")
        component = {
            "id": component_id,
            "name": name,
            "kind": "module",
            "path": key if "/" in key else first_path,
            "summary": f"New component created during update from {len(node_ids)} node(s).",
            "depth": 1,
            "members": sorted(node_ids),
            "children": [],
            "confidence": 0.5,
            "evidence": {
                "signals": ["update:new_component"],
                "rationale": "New directory/file group created by local component repair.",
            },
            "coverage": {
                "assigned_nodes": len(node_ids),
                "unassigned_nodes": 0,
                "unassigned": [],
            },
        }
        root["children"].append(component)
        mapping[component_id] = component
        new_components.append(component)
        members_by_component[component_id] = sorted(node_ids)
        graph.add_node("component", component_id, component)
        graph.add_edge("has_child_component", root["id"], component_id)

    root["members"] = sorted(new_code_nodes)
    root["coverage"] = {
        "assigned_nodes": len(new_code_nodes),
        "unassigned_nodes": 0,
        "unassigned": [],
    }

    for edge_id, edge in list(graph.edges.items()):
        if edge["type"] == "part_of_component":
            graph.remove_edge(edge_id)
    for component_id, members in members_by_component.items():
        for member_id in members:
            if member_id in graph.nodes:
                graph.add_edge("part_of_component", member_id, component_id)

    code_graph_version = graph.code_graph_version
    component_tree["source_graph_version"] = code_graph_version
    component_tree["cluster_meta"]["update"] = {
        "mode": "membership-repair",
        "updated_at": utc_now(),
        "new_components": [component["id"] for component in new_components],
        "removed_components": removed_components,
    }
    component_tree["cluster_version"] = "c_" + stable_hash(component_tree)
    return {
        "new_components": new_components,
        "removed_components": removed_components,
        "unassigned_nodes": [],
    }


def _changed_code_context(
    old_graph: GraphStore,
    new_graph: GraphStore,
    delta: dict[str, Any],
    *,
    max_items: int = 20,
    max_snippet: int = 800,
    only_ids: set[str] | None = None,
) -> str:
    lines: list[str] = []
    for key, op in (
        ("nodes_added", "added"),
        ("nodes_removed", "removed"),
        ("nodes_modified", "modified"),
    ):
        for node_id in delta.get(key, [])[:max_items]:
            if only_ids is not None and node_id not in only_ids:
                continue
            node = new_graph.nodes.get(node_id) or old_graph.nodes.get(node_id)
            if node is None or node["type"] == "file":
                continue
            attrs = node["attrs"]
            lines.append(
                f"- {op}: {node_id} path={attrs.get('path', '')} "
                f"qualified_name={attrs.get('qualified_name', '')} "
                f"kind={attrs.get('kind', '')}"
            )
            new_source = str(attrs.get("source") or "").strip()
            old_node = old_graph.nodes.get(node_id)
            old_source = (
                str((old_node or {}).get("attrs", {}).get("source") or "").strip()
                if op == "modified"
                else ""
            )
            if old_source:
                lines.append(
                    "  old source: " + old_source[:max_snippet].replace("\n", "\n  ")
                )
            if new_source:
                lines.append(
                    "  new source: " + new_source[:max_snippet].replace("\n", "\n  ")
                )
    return "\n".join(lines)


def _build_plan(
    repo: Path,
    state: Path,
    *,
    direct_only: bool = False,
    plan_llm: bool = False,
) -> dict[str, Any]:
    status = status_info(repo, state, exclude_dirs=[state])
    if not status["needs_update"]:
        return {
            "status": "noop",
            "reason": status["reason"],
            "change_set": None,
            "path": None,
        }
    manifest = status.get("manifest") or read_manifest(state)
    if manifest is None:
        raise RuntimeError("no manifest; run `repodoc build` first")
    if manifest.get("last_commit") is None:
        raise RuntimeError(
            "update requires a git repository and a commit-based manifest"
        )
    target = git_head(repo)
    if not target:
        raise RuntimeError("cannot resolve git HEAD")
    base = manifest["last_commit"]
    files = git_changed_files(repo, base, target)

    old_graph = GraphStore.load(state)
    component_tree = json.loads(
        (state / "components" / "component_tree.json").read_text(encoding="utf-8")
    )
    doc_tree = json.loads(
        (state / "docs" / "doc_tree.json").read_text(encoding="utf-8")
    )
    docs_dir = _docs_dir(state, manifest)
    new_graph = CodeGraphBuilder(
        repo,
        repo_name=old_graph.repo,
        exclude_dirs=[state, docs_dir],
        analyzer=detect_analyzer(repo),
    ).build()

    delta = diff_code_graphs(old_graph, new_graph)
    semantic_changed = _semantic_changed_ids(old_graph, new_graph, delta)
    change_context = _changed_code_context(
        old_graph, new_graph, delta, only_ids=semantic_changed
    )
    old_path_map = _old_path_to_component(old_graph, component_tree)
    old_dir_map = _old_dir_to_component(old_graph, component_tree)
    scope = _impact_scope(old_graph, new_graph, delta, component_tree, doc_tree)
    affected_components = scope["affected_components"]
    affected_pages = set(scope["candidate_pages"])
    if direct_only and not scope["structural"] and scope["direct_pages"]:
        affected_pages = set(scope["direct_pages"])
    if plan_llm:
        # Keep the candidate set available to the planner; execution will
        # replace affected_pages with the validated decisions.
        affected_pages = set(scope["candidate_pages"])
    change_units = _change_units(
        old_graph, new_graph, delta, component_tree, affected_pages
    )
    new_paths = {
        node["attrs"].get("path")
        for node in new_graph.nodes.values()
        if node["layer"] == "code" and node["type"] == "file"
    }
    requires_recluster = any(
        path
        and path not in old_path_map
        and Path(path).parent.as_posix() not in old_dir_map
        for path in new_paths
    )
    if any(unit["classification"] == "deleted_file" for unit in change_units):
        requires_recluster = True

    pages = _page_map(doc_tree)
    affected_page_paths = sorted(
        pages[page_id]["path"] for page_id in affected_pages if page_id in pages
    )
    change_set = {
        "schema_version": SCHEMA_VERSION,
        "repo": repo.name,
        "base_commit": base,
        "target_commit": target,
        "status": "planned",
        "files": files,
        "graph_delta": delta,
        "change_units": change_units,
        "impact": {
            "requires_recluster": requires_recluster,
            "scope": "global"
            if requires_recluster
            else ("local" if affected_pages else "none"),
            "reason": (
                "new/deleted files or unassigned nodes; local membership repair only in this MVP"
                if requires_recluster
                else "affected components and docs derived from graph delta"
            ),
            "affected_components": sorted(affected_components),
            "affected_pages": affected_page_paths,
            "candidate_pages": sorted(
                pages[page_id]["path"]
                for page_id in scope["candidate_pages"]
                if page_id in pages
            ),
            "mandatory_pages": sorted(
                pages[page_id]["path"]
                for page_id in scope["mandatory_pages"]
                if page_id in pages
            ),
            "direct_pages": sorted(
                pages[page_id]["path"]
                for page_id in scope["direct_pages"]
                if page_id in pages
            ),
            "structural": scope["structural"],
            "architecture_signal": scope["architecture_signal"],
            "cross_component_edge": scope["cross_component_edge"],
        },
        "created_at": utc_now(),
    }
    validate("change_set", change_set)
    return {
        "status": "planned",
        "reason": status["reason"],
        "change_set": change_set,
        "path": None,
        "manifest": manifest,
        "old_graph": old_graph,
        "new_graph": new_graph,
        "component_tree": component_tree,
        "doc_tree": doc_tree,
        "docs_dir": docs_dir,
        "affected_pages": affected_pages,
        "affected_components": affected_components,
        "scope": scope,
        "old_path_map": old_path_map,
        "old_dir_map": old_dir_map,
        "semantic_changed": semantic_changed,
        "change_context": change_context,
        "target": target,
        "repo": repo,
        "state": state,
    }


def plan_update(
    repo_path: str | Path,
    state_dir: str | Path,
    *,
    write: bool = True,
    direct_only: bool = False,
    plan_llm: bool = False,
) -> dict[str, Any]:
    repo = Path(repo_path).resolve()
    state = Path(state_dir).resolve()
    plan = _build_plan(repo, state, direct_only=direct_only, plan_llm=plan_llm)
    if plan["status"] == "noop":
        return plan
    change_set = plan["change_set"]
    path = state / "changes" / f"{plan['target']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if write:
        path.write_text(
            json.dumps(change_set, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return {
        "status": "planned",
        "reason": plan["reason"],
        "change_set": change_set,
        "path": str(path),
    }


def apply_update(
    repo_path: str | Path,
    state_dir: str | Path,
    *,
    use_llm: bool = False,
    use_patch: bool = False,
    direct_only: bool = False,
    use_planner: bool = False,
    llm_client: Any | None = None,
    model_limits: Any | None = None,
    llm_model: str | None = None,
    llm_max_tokens: int | None = None,
) -> dict[str, Any]:
    repo = Path(repo_path).resolve()
    state = Path(state_dir).resolve()
    plan = _build_plan(repo, state, direct_only=direct_only, plan_llm=use_planner)
    if plan["status"] == "noop":
        return plan
    old_graph: GraphStore = plan["old_graph"]
    new_graph: GraphStore = plan["new_graph"]
    component_tree: dict[str, Any] = plan["component_tree"]
    doc_tree: dict[str, Any] = plan["doc_tree"]
    docs_dir: Path = plan["docs_dir"]
    affected_pages: set[str] = plan["affected_pages"]
    target: str = plan["target"]
    manifest: dict[str, Any] = plan["manifest"]

    old_path_map = plan["old_path_map"]
    old_dir_map = plan["old_dir_map"]
    for node in list(old_graph.nodes.values()):
        if node["layer"] == "code":
            old_graph.remove_node(node["id"])
    for node in new_graph.nodes.values():
        if node["layer"] == "code":
            old_graph.add_node(node["type"], node["id"], node["attrs"])
    for edge in new_graph.edges.values():
        src = new_graph.nodes.get(edge["src"])
        dst = new_graph.nodes.get(edge["dst"])
        if src and dst and src["layer"] == "code" and dst["layer"] == "code":
            old_graph.add_edge(edge["type"], edge["src"], edge["dst"], edge["attrs"])
    old_graph.commit = target
    old_graph.repo_path = str(repo)
    # Replacing code nodes removes edges that untouched doc sections had to
    # those nodes. Restore citations from their persisted covers; affected
    # sections are removed/rebuilt by regenerate_page below.
    for section in old_graph.nodes_of_type("doc_section"):
        for covered in section["attrs"].get("covers", []):
            if covered in old_graph.nodes:
                old_graph.add_edge("cites", section["id"], covered)
    component_tree["commit"] = target
    component_tree["repo_path"] = str(repo)
    doc_tree["commit"] = target
    doc_tree["repo_path"] = str(repo)

    repair = _repair_components(
        old_graph, component_tree, new_graph, old_path_map, old_dir_map
    )
    unassigned = repair["unassigned_nodes"]
    source_graph_version = old_graph.code_graph_version

    page_map = _page_map(doc_tree)
    used_paths = {page["path"] for page in page_map.values()}
    new_pages: list[dict[str, Any]] = []
    root_page = doc_tree["root"]
    for component in repair["new_components"]:
        new_page = page_for_component(
            component,
            source_graph_version=source_graph_version,
            used_paths=used_paths,
        )
        root_page["children"].append(new_page)
        root_page["generated_from"]["child_doc_ids"].append(new_page["id"])
        affected_pages.add(new_page["id"])
        new_pages.append(new_page)

    removed_page_ids: set[str] = set()
    deleted_page_paths: list[str] = []
    for component_id in repair["removed_components"]:
        page = next(
            (
                item
                for item in page_map.values()
                if item["component_id"] == component_id
            ),
            None,
        )
        if page is None:
            continue
        _remove_page_from_tree(root_page, page["id"])
        if page["id"] in root_page["generated_from"].get("child_doc_ids", []):
            root_page["generated_from"]["child_doc_ids"].remove(page["id"])
        _remove_page_from_graph(old_graph, page["id"])
        (docs_dir / page["path"]).unlink(missing_ok=True)
        removed_page_ids.add(page["id"])
        deleted_page_paths.append(page["path"])
        affected_pages.discard(page["id"])
    if new_pages or removed_page_ids:
        affected_pages.add(root_page["id"])
    page_map = _page_map(doc_tree)

    pages = _load_jsonl(
        state / (manifest.get("artifacts") or {}).get("pages", "docs/pages.jsonl")
    )
    sections = _load_jsonl(
        state / (manifest.get("artifacts") or {}).get("sections", "docs/sections.jsonl")
    )
    if removed_page_ids:
        pages = [row for row in pages if row.get("id") not in removed_page_ids]
        deleted_paths = set(deleted_page_paths)
        sections = [
            row for row in sections if row.get("page_path") not in deleted_paths
        ]
    page_map = _page_map(doc_tree)

    decisions: dict[str, str] = {}
    llm_meta: dict[str, Any] | None = None
    planner_meta: dict[str, Any] | None = None
    planner_fallback = False
    patch_agent: Any | None = None
    budget = None
    client = None
    own_client = False
    llm_enabled = use_llm or use_patch or use_planner
    if llm_enabled:
        from repodoc.change.update_planner import UpdatePlannerAgent
        from repodoc.change.llm_update import UpdateTriageAgent
        from repodoc.document.llm_patch import LLMContentPatchAgent
        from repodoc.llm import (
            OpenCodeGoClient,
            OpenCodeGoConfig,
            compute_budget,
            model_limits_or_fallback,
        )

        config = OpenCodeGoConfig.from_env()
        limits = model_limits or model_limits_or_fallback(config)
        budget = compute_budget(limits, "docs")
        own_client = llm_client is None
        client = llm_client or OpenCodeGoClient(config)
        if use_patch:
            patch_agent = LLMContentPatchAgent(
                client,
                model=llm_model,
                max_tokens=llm_max_tokens or budget.output_tokens,
            )
        if use_planner:
            scope = plan["scope"]
            component_map = _component_map(component_tree)
            candidate_ids = [
                page_id
                for page_id in sorted(scope["candidate_pages"])
                if page_id in page_map
            ]
            candidate_rows: list[dict[str, Any]] = []
            for page_id in candidate_ids:
                page = page_map[page_id]
                page_sections = [
                    row for row in sections if row.get("page_path") == page["path"]
                ]
                component = component_map.get(page["component_id"], {})
                candidate_rows.append(
                    {
                        "path": page["path"],
                        "level": page["level"],
                        "doc_type": page["doc_type"],
                        "direct": page_id in scope["direct_pages"],
                        "structural": page_id in scope["mandatory_pages"],
                        "component": component.get("name", ""),
                        "sections": [
                            str(row.get("heading") or "")
                            for row in page_sections
                            if row.get("heading")
                        ],
                    }
                )
            mandatory_paths = sorted(
                page_map[page_id]["path"]
                for page_id in scope["mandatory_pages"]
                if page_id in page_map
            )
            planner = UpdatePlannerAgent(
                client,
                model=llm_model,
                max_tokens=llm_max_tokens or min(8_192, budget.output_tokens),
            )
            try:
                planner_decisions, planner_meta = planner.plan(
                    plan["change_set"],
                    candidate_rows,
                    mandatory_paths,
                    change_context=plan["change_context"],
                )
            except Exception as exc:  # noqa: BLE001 - fall back to rule policy
                planner_fallback = True
                planner_meta = {
                    "failed": True,
                    "error": str(exc),
                    "prompt": planner.last_prompt,
                    "response": planner.last_response,
                }
                planner_decisions = {}
            trace_path = state / "changes" / f"{target}.llm_plan.json"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(
                json.dumps(planner_meta, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            if planner_fallback:
                affected_pages = set(scope["default_pages"])
            else:
                selected: set[str] = set()
                for page_id in candidate_ids:
                    page_path = page_map[page_id]["path"]
                    action = planner_decisions.get(page_path)
                    if action is None:
                        action = (
                            "regenerate"
                            if page_id in scope["direct_pages"]
                            or page_id in scope["mandatory_pages"]
                            else "noop"
                        )
                    # Local behavior/config changes must not rewrite a parent
                    # or overview unless the deterministic graph found a
                    # structural or cross-component architecture signal.
                    if (
                        not scope["architecture_signal"]
                        and page_id not in scope["direct_pages"]
                        and action != "noop"
                    ):
                        action = "noop"
                    # For a local behavior change, full-page regeneration of
                    # the directly citing page was more stable than section
                    # patching in the GraphRAG pilot.
                    if (
                        not scope["architecture_signal"]
                        and page_id in scope["direct_pages"]
                        and action == "patch"
                    ):
                        action = "regenerate"
                    if page_id in scope["mandatory_pages"] and action == "noop":
                        action = "regenerate"
                    if action != "noop":
                        selected.add(page_id)
                    decisions[page_path] = action
                affected_pages = selected
            # New pages are always executed even if they were not part of the
            # pre-repair candidate universe.
            affected_pages.update(page["id"] for page in new_pages)
            if decisions:
                llm_meta = {
                    "decisions": decisions,
                    "reasons": (planner_meta or {}).get("reasons", {}),
                    "change_scope": (planner_meta or {}).get("change_scope", ""),
                    "fallback": planner_fallback,
                    "architecture_signal": scope["architecture_signal"],
                    "direct_pages": sorted(
                        page_map[page_id]["path"]
                        for page_id in scope["direct_pages"]
                        if page_id in page_map
                    ),
                }
        elif use_llm:
            page_paths = sorted(
                page_map[page_id]["path"]
                for page_id in affected_pages
                if page_id in page_map
            )
            triage_agent = UpdateTriageAgent(
                client,
                model=llm_model,
                max_tokens=llm_max_tokens or budget.output_tokens,
            )
            decisions, llm_meta = triage_agent.decide(plan["change_set"], page_paths)
            triage_path = state / "changes" / f"{target}.llm_triage.json"
            triage_path.parent.mkdir(parents=True, exist_ok=True)
            triage_path.write_text(
                json.dumps(llm_meta, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    regenerated: list[str] = []
    patched_pages: list[str] = []
    patch_fallbacks: dict[str, Any] = {}
    patch_traces: dict[str, Any] = {}
    for page_id in sorted(
        (page_map[page_id] for page_id in affected_pages if page_id in page_map),
        key=lambda page: (page["level"], page["path"]),
    ):
        if use_patch and not use_llm:
            default_action = (
                "regenerate"
                if page_id["level"] >= 3
                or page_id["doc_type"] in {"architecture", "overview", "flow"}
                else "patch"
            )
        else:
            default_action = "regenerate"
        action = decisions.get(page_id["path"], default_action)
        if action == "noop":
            continue
        if action == "delete":
            action = "regenerate"
        page_id = page_id["id"]
        page = page_map[page_id]
        if use_patch and action == "patch" and patch_agent is not None:
            page_sections = [
                row for row in sections if row.get("page_path") == page["path"]
            ]
            affected_section_ids = _affected_section_ids(
                old_graph, page_id, plan["semantic_changed"]
            )
            if affected_section_ids:
                page_sections = [
                    row for row in page_sections if row["id"] in affected_section_ids
                ]
            patch_change_set = dict(plan["change_set"])
            patch_change_set["code_context"] = plan["change_context"]
            try:
                patches, patch_meta = patch_agent.propose(
                    page,
                    page_sections,
                    patch_change_set,
                    section_ids=sorted(affected_section_ids),
                )
                applied, warnings = apply_section_patches(
                    old_graph,
                    page,
                    page_sections,
                    patches,
                    docs_dir=docs_dir,
                    source_graph_version=source_graph_version,
                )
            except Exception as exc:
                applied = 0
                warnings = [str(exc)]
                patch_meta = None
                if patch_agent.last_response is not None:
                    patch_meta = {
                        "prompt": patch_agent.last_prompt,
                        "response": patch_agent.last_response,
                        "content": patch_agent.last_content,
                        "failed": True,
                    }
            if patch_meta is not None:
                trace_path = (
                    state / "changes" / f"{target}.patch.{Path(page['path']).stem}.json"
                )
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                trace_path.write_text(
                    json.dumps(patch_meta, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            if applied:
                patched_pages.append(page["path"])
                patch_traces[page["path"]] = warnings
                pages = [row for row in pages if row.get("id") != page["id"]]
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
                continue
            patch_fallbacks[page["path"]] = warnings or ["no section patches"]
        page["generated_from"]["graph_version"] = source_graph_version
        page, section_rows = regenerate_page(
            old_graph,
            component_tree,
            doc_tree,
            page,
            docs_dir=docs_dir,
            source_graph_version=source_graph_version,
        )
        regenerated.append(page["path"])
        pages = [row for row in pages if row.get("id") != page["id"]]
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
        sections.extend(section_rows)

    if own_client and client is not None:
        client.close()

    doc_tree["source_graph_version"] = source_graph_version
    doc_tree["graph_version"] = old_graph.graph_version
    markdown = {
        path.relative_to(docs_dir).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(docs_dir.rglob("*.md"))
        if path.is_file()
    }
    doc_tree["doc_version"] = "d_" + stable_hash(
        {"tree": doc_tree, "markdown": markdown}
    )
    component_version = "c_" + stable_hash(component_tree)

    graph_paths = old_graph.save(state)
    (state / "components" / "component_tree.json").write_text(
        json.dumps(component_tree, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (state / "docs" / "doc_tree.json").write_text(
        json.dumps(doc_tree, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_jsonl(
        state / (manifest.get("artifacts") or {}).get("pages", "docs/pages.jsonl"),
        pages,
    )
    _write_jsonl(
        state
        / (manifest.get("artifacts") or {}).get("sections", "docs/sections.jsonl"),
        sections,
    )

    change_set = plan["change_set"]
    change_set["status"] = "applied"
    change_set["impact"]["unassigned_nodes"] = unassigned
    change_set["impact"]["regenerated_pages"] = regenerated
    change_set["impact"]["new_components"] = [
        component["name"] for component in repair["new_components"]
    ]
    change_set["impact"]["removed_components"] = repair["removed_components"]
    change_set["impact"]["new_pages"] = [page["path"] for page in new_pages]
    change_set["impact"]["deleted_pages"] = deleted_page_paths
    change_set["impact"]["affected_pages"] = sorted(
        page_map[page_id]["path"] for page_id in affected_pages if page_id in page_map
    )
    if decisions:
        change_set["impact"]["llm_decisions"] = decisions
        change_set["impact"]["llm_reasons"] = (llm_meta or {}).get("reasons", {})
    if planner_meta is not None:
        change_set["impact"]["llm_plan"] = {
            "change_scope": planner_meta.get("change_scope", ""),
            "fallback": planner_fallback,
            "architecture_signal": scope["architecture_signal"],
            "decisions": decisions,
        }
    if patched_pages:
        change_set["impact"]["patched_pages"] = patched_pages
    if patch_fallbacks:
        change_set["impact"]["patch_fallbacks"] = patch_fallbacks
    change_set_path = state / "changes" / f"{target}.json"
    change_set_path.parent.mkdir(parents=True, exist_ok=True)
    change_set_path.write_text(
        json.dumps(change_set, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    manifest["repo"] = old_graph.repo
    manifest["repo_path"] = str(repo)
    manifest["last_commit"] = target
    manifest["source_hash"] = source_hash(repo, exclude_dirs=[state, docs_dir])
    manifest["code_graph_version"] = source_graph_version
    manifest["graph_version"] = doc_tree["graph_version"]
    manifest["component_version"] = component_version
    manifest["doc_version"] = doc_tree["doc_version"]
    manifest["updated_at"] = utc_now()
    manifest["last_run"] = {
        "run_id": manifest.get("last_run", {}).get("run_id", "update"),
        "status": "updated",
        "finished_at": utc_now(),
        "pages": regenerated,
    }
    for key, value in {
        "graph": graph_paths["graph"],
        "nodes": graph_paths["nodes"],
        "edges": graph_paths["edges"],
        "component_tree": str(state / "components" / "component_tree.json"),
        "doc_tree": str(state / "docs" / "doc_tree.json"),
        "pages": str(
            state / (manifest.get("artifacts") or {}).get("pages", "docs/pages.jsonl")
        ),
        "sections": str(
            state
            / (manifest.get("artifacts") or {}).get("sections", "docs/sections.jsonl")
        ),
        "markdown": str(docs_dir),
    }.items():
        try:
            manifest["artifacts"][key] = (
                Path(value).resolve().relative_to(state).as_posix()
            )
        except ValueError:
            manifest["artifacts"][key] = str(Path(value).resolve())
    validate("component_tree", component_tree)
    validate("doc_tree", doc_tree)
    validate("change_set", change_set)
    validate("manifest", manifest)
    write_manifest(state, manifest)
    return {
        "status": "applied",
        "change_set": change_set,
        "path": str(change_set_path),
        "regenerated_pages": regenerated,
        "unassigned_nodes": unassigned,
        "new_components": [component["name"] for component in repair["new_components"]],
        "removed_components": repair["removed_components"],
        "new_pages": [page["path"] for page in new_pages],
        "deleted_pages": deleted_page_paths,
        "llm_decisions": decisions,
        "patched_pages": patched_pages,
        "patch_fallbacks": patch_fallbacks,
        "graph_version": manifest["graph_version"],
        "doc_version": manifest["doc_version"],
    }
