"""LLM clustering agent for stage two.

One small request is sent to OpenCode Go: a compact code-node/edge summary goes
in, a component group JSON comes out. The deterministic document pipeline then
consumes the resulting component tree.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from repodoc.document.planner import build_component_tree_from_groups
from repodoc.llm.budget import ComputedBudget
from repodoc.graph.store import GraphStore

PROMPT_VERSION = "cluster-v1"
SYSTEM_PROMPT = """You are a software architecture clustering agent.
Group the provided FILE nodes into a small hierarchy of functional components.
Return ONLY valid JSON. Do not include markdown fences or prose.

JSON shape:
{
  "components": [
    {
      "name": "component-name",
      "path": "relative/path",
      "summary": "one sentence",
      "members": ["code:file:..."],
      "children": []
    }
  ]
}

Rules:
- Assign EVERY provided file node ID exactly once, across all components.
- Members must be file node IDs. Do not list symbols/APIs; they inherit their file's component.
- Do not invent or modify node IDs.
- Use 2-8 top-level components for a small repository; nested children are optional.
- Use functional/architectural names, not only directory names.
- Keep the hierarchy understandable for documentation readers.
"""


CHUNK_SYSTEM_PROMPT = (
    SYSTEM_PROMPT + "\n\nCHUNK MODE: This is one bounded slice of a larger repository. "
    "Use 1-4 components for this chunk, not 2-8 repository-wide modules. "
    "Do not invent files outside the provided chunk."
)


MERGE_SYSTEM_PROMPT = """You are a software architecture merge agent.
You receive local component groups produced from bounded file chunks. Merge them
into a small, coherent top-level architecture. Return ONLY valid JSON.

JSON shape:
{
  "components": [
    {
      "name": "component-name",
      "path": "relative/path",
      "summary": "one sentence",
      "children": [0, 2, 5]
    }
  ]
}

Rules:
- `children` are LOCAL COMPONENT indices shown in the input.
- Assign every local component index exactly once.
- Use 2-8 top-level components for a normal repository; fewer for tiny repos.
- Do not invent indices, files, or component names.
- Prefer semantic/functional boundaries over blindly following directories.
- Use the relationship counts to keep tightly coupled local components together
  when that does not create an oversized module.
"""


class LLMClusterError(RuntimeError):
    pass


def _code_nodes(graph: GraphStore) -> list[dict[str, Any]]:
    return [node for node in graph.nodes.values() if node["layer"] == "code"]


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


def _node_file_map(graph: GraphStore) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for node in graph.nodes.values():
        if node["layer"] != "code":
            continue
        if node["type"] == "file":
            mapping[node["id"]] = node["id"]
        else:
            path = node["attrs"].get("path")
            if path:
                mapping[node["id"]] = f"code:file:{path}"
    return mapping


def build_cluster_prompt(
    graph: GraphStore,
    *,
    max_nodes: int = 200,
    max_chars: int = 120_000,
    max_edges: int = 300,
    file_ids: set[str] | None = None,
) -> str:
    """Build a file-level clustering prompt.

    `max_nodes` limits the number of FILE nodes, not every symbol/API; symbols
    and APIs inherit their file's component deterministically.
    """
    files = sorted(
        graph.nodes_of_type("file"), key=lambda node: node["attrs"].get("path", "")
    )
    if file_ids is not None:
        files = [node for node in files if node["id"] in file_ids]
    if len(files) > max_nodes:
        raise LLMClusterError(
            f"graph has {len(files)} files, above --max-cluster-nodes={max_nodes}; "
            "refusing to send a potentially expensive clustering request"
        )
    lines = [
        f"Repository: {graph.repo}",
        "",
        "FILE NODES (assign every ID exactly once):",
    ]
    for file_node in files:
        attrs = file_node["attrs"]
        members = _file_members(graph, file_node["id"])
        symbols = []
        apis = []
        for member_id in members:
            member = graph.nodes.get(member_id)
            if member is None:
                continue
            member_attrs = member["attrs"]
            if member["type"] == "symbol":
                symbols.append(
                    member_attrs.get("qualified_name") or member_attrs.get("name", "")
                )
            elif member["type"] == "api":
                apis.append(
                    member_attrs.get("qualified_name") or member_attrs.get("name", "")
                )
        lines.append(f"[file] {file_node['id']} path={attrs.get('path', '')}")
        if symbols:
            lines.append("  symbols: " + ", ".join(sorted(set(symbols))[:30]))
        if apis:
            lines.append("  apis: " + ", ".join(sorted(set(apis))[:30]))

    node_to_file = _node_file_map(graph)
    relationships: set[tuple[str, str, str]] = set()
    for edge in graph.edges.values():
        if edge["type"] == "imports":
            src = edge["src"] if edge["src"] in graph.nodes else None
            dst = edge["dst"] if edge["dst"] in graph.nodes else None
        elif edge["type"] == "calls":
            src = node_to_file.get(edge["src"])
            dst = node_to_file.get(edge["dst"])
        else:
            continue
        if src and dst and src != dst:
            relationships.add((edge["type"], src, dst))
    if relationships:
        lines.append("")
        lines.append("FILE-LEVEL RELATIONSHIPS:")
        for edge_type, src, dst in sorted(relationships)[:max_edges]:
            if file_ids is not None and (src not in file_ids or dst not in file_ids):
                continue
            lines.append(f"{edge_type}: {src} -> {dst}")

    prompt = "\n".join(lines)
    if len(prompt) > max_chars:
        raise LLMClusterError(
            f"cluster prompt is {len(prompt)} chars, above max_chars={max_chars}; "
            "refusing to send a potentially expensive request"
        )
    return prompt


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise LLMClusterError("LLM returned empty clustering content")
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
        if match:
            text = match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise LLMClusterError(f"LLM response contains no JSON object: {text[:500]}")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMClusterError(f"LLM clustering JSON is invalid: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMClusterError("LLM clustering response must be a JSON object")
    return data


def _expand_file_members(
    graph: GraphStore, groups: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Expand file IDs in LLM output to file + contained symbols/APIs."""
    file_ids = {node["id"] for node in graph.nodes_of_type("file")}
    valid_ids = {node["id"] for node in graph.nodes.values() if node["layer"] == "code"}

    def expand(spec: dict[str, Any]) -> dict[str, Any]:
        members: list[str] = []
        seen: set[str] = set()
        for raw_member in spec.get("members") or []:
            member = str(raw_member)
            expanded = _file_members(graph, member) if member in file_ids else [member]
            for item in expanded:
                if item in valid_ids and item not in seen:
                    seen.add(item)
                    members.append(item)
        return {
            **spec,
            "members": members,
            "children": [
                expand(child)
                for child in (spec.get("children") or [])
                if isinstance(child, dict)
            ],
        }

    return [expand(group) for group in groups if isinstance(group, dict)]


def _chunk_key(path: str) -> str:
    parts = Path(path).parts
    if len(parts) >= 2:
        return "/".join(parts[:2])
    if parts:
        return Path(parts[0]).stem
    return "root"


def _parent_key(paths: list[str]) -> str:
    parts = [Path(path).parts for path in paths if path]
    if not parts:
        return "root"
    common: list[str] = []
    for column in zip(*parts):
        if all(part == column[0] for part in column):
            common.append(column[0])
        else:
            break
    if common:
        return "/".join(common)
    return Path(paths[0]).stem if paths else "root"


def _chunk_file_sets(
    graph: GraphStore,
    *,
    max_files: int,
    max_chars: int,
) -> list[list[str]]:
    files = sorted(
        graph.nodes_of_type("file"), key=lambda node: node["attrs"].get("path", "")
    )
    by_key: dict[str, list[str]] = {}
    for file_node in files:
        path = file_node["attrs"].get("path", "")
        top_key = Path(path).parts[0] if Path(path).parts else Path(path).stem
        by_key.setdefault(top_key, []).append(file_node["id"])

    def fits(ids: list[str]) -> bool:
        try:
            build_cluster_prompt(
                graph,
                max_nodes=max_files,
                max_chars=max_chars,
                file_ids=set(ids),
            )
        except LLMClusterError:
            return False
        return True

    def split_oversized(ids: list[str]) -> list[list[str]]:
        if fits(ids):
            return [ids]
        if len(ids) <= 1:
            raise LLMClusterError(
                f"one file exceeds hierarchical chunk budget ({max_chars} chars)"
            )
        middle = len(ids) // 2
        return split_oversized(ids[:middle]) + split_oversized(ids[middle:])

    chunks: list[list[str]] = []
    current: list[str] = []
    for _key, ids in sorted(by_key.items()):
        for start in range(0, len(ids), max_files):
            candidate = split_oversized(ids[start : start + max_files])
            for part in candidate:
                if current:
                    if len(current) + len(part) > max_files or not fits(current + part):
                        chunks.append(current)
                        current = []
                current.extend(part)
    if current:
        chunks.append(current)
    return chunks


def _merge_usage(usages: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    if not usages:
        return None
    totals: dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "opencode": {"list_cost": 0.0, "tokens": {}},
    }
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            totals[key] += int(usage.get(key) or 0)
        opencode = usage.get("opencode") or {}
        totals["opencode"]["list_cost"] += float(opencode.get("list_cost") or 0)
        for key, value in (opencode.get("tokens") or {}).items():
            if isinstance(value, (int, float)):
                totals["opencode"]["tokens"][key] = (
                    totals["opencode"]["tokens"].get(key, 0) + value
                )
    totals["opencode"]["list_cost"] = round(totals["opencode"]["list_cost"], 8)
    return totals


def _local_relationships(
    graph: GraphStore, local_specs: list[dict[str, Any]]
) -> tuple[dict[str, int], list[str]]:
    owner: dict[str, int] = {}
    for index, spec in enumerate(local_specs):
        for member_id in spec.get("members") or []:
            node = graph.nodes.get(member_id)
            if node is not None and node["type"] == "file":
                owner[member_id] = index
    node_to_file = _node_file_map(graph)
    counts: dict[str, int] = {}
    for edge in graph.edges.values():
        if edge["type"] == "imports":
            src_file = edge["src"]
            dst_file = edge["dst"]
        elif edge["type"] == "calls":
            src_file = node_to_file.get(edge["src"])
            dst_file = node_to_file.get(edge["dst"])
        else:
            continue
        src = owner.get(src_file)
        dst = owner.get(dst_file)
        if src is None or dst is None or src == dst:
            continue
        key = f"local {src} -> {dst}: {edge['type']}"
        counts[key] = counts.get(key, 0) + 1
    lines = [
        f"- {key} (count={count})"
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            :40
        ]
    ]
    return counts, lines


def _merge_prompt(
    graph: GraphStore,
    local_specs: list[dict[str, Any]],
    relationships: list[str],
) -> str:
    lines = [
        f"Repository: {graph.repo}",
        "",
        "LOCAL COMPONENTS:",
    ]
    for index, spec in enumerate(local_specs):
        paths = [
            graph.nodes[member_id]["attrs"].get("path", "")
            for member_id in spec.get("members", [])
            if member_id in graph.nodes and graph.nodes[member_id]["type"] == "file"
        ]
        sample = ", ".join(sorted(paths)[:8])
        lines.append(
            f"- local={index} name={spec.get('name', '')} "
            f"path={spec.get('path', '')} files={len(paths)} "
            f"summary={spec.get('summary', '')}"
        )
        if sample:
            lines.append(f"  files: {sample}")
    if relationships:
        lines.extend(["", "LOCAL RELATIONSHIPS:", *relationships])
    lines.extend(["", "Return the merged components JSON now."])
    return "\n".join(lines)


class LLMClusterAgent:
    def __init__(
        self,
        client: Any,
        *,
        model: str | None = None,
        max_nodes: int = 200,
        max_chars: int | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.1,
        budget: ComputedBudget | None = None,
        progress: Any | None = None,
    ) -> None:
        self.client = client
        self.progress = progress
        self.model = model
        self.max_nodes = max_nodes
        self.budget = budget
        self.max_chars = (
            max_chars
            if max_chars is not None
            else (budget.context_chars if budget is not None else 120_000)
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else (budget.output_tokens if budget is not None else 4_096)
        )
        self.temperature = temperature

    def cluster_hierarchical(
        self,
        graph: GraphStore,
        *,
        max_chunk_files: int = 80,
        max_chunk_chars: int | None = None,
        merge_mode: str = "directory",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Cluster bounded file chunks, then merge into top-level components."""
        chunk_chars = max(4_096, min(max_chunk_chars or 80_000, self.max_chars))
        chunks = _chunk_file_sets(
            graph, max_files=max_chunk_files, max_chars=chunk_chars
        )
        if not chunks:
            raise LLMClusterError("hierarchical clustering found no file chunks")
        if self.progress is not None:
            self.progress.stage(
                f"Clustering {len(chunks)} bounded file chunks "
                f"(max {max_chunk_files} files / {chunk_chars} chars each)"
            )

        parents: dict[str, dict[str, Any]] = {}
        local_specs: list[dict[str, Any]] = []
        chunk_traces: list[dict[str, Any]] = []
        usages: list[dict[str, Any]] = []
        total_prompt_chars = 0
        total_response_chars = 0
        total_llm_seconds = 0.0
        for chunk_index, chunk_files in enumerate(chunks, 1):
            if self.progress is not None:
                self.progress.step(
                    "Clustering chunk",
                    chunk_index,
                    len(chunks),
                    detail=f"{len(chunk_files)} files",
                )
            prompt = build_cluster_prompt(
                graph,
                max_nodes=max_chunk_files,
                max_chars=chunk_chars,
                file_ids=set(chunk_files),
            )
            chunk_started = time.monotonic()
            response = self.client.chat(
                [
                    {"role": "system", "content": CHUNK_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            chunk_seconds = round(time.monotonic() - chunk_started, 3)
            total_llm_seconds += chunk_seconds
            try:
                message = response["choices"][0]["message"]
            except (KeyError, IndexError, TypeError) as exc:
                raise LLMClusterError(
                    "hierarchical clustering response has no choices[0].message"
                ) from exc
            content = message.get("content") or ""
            data = _extract_json(content)
            groups = data.get("components")
            if not isinstance(groups, list):
                raise LLMClusterError(
                    "hierarchical clustering response has no `components` list"
                )
            expanded = _expand_file_members(graph, groups)
            covered_files = {
                member_id
                for group in expanded
                for member_id in group.get("members", [])
                if graph.nodes.get(member_id, {}).get("type") == "file"
            }
            missing = [
                file_id for file_id in chunk_files if file_id not in covered_files
            ]
            missing_members: list[str] = []
            for file_id in missing:
                missing_members.extend(_file_members(graph, file_id))
            if missing_members:
                expanded.append(
                    {
                        "name": "chunk_unassigned",
                        "path": "",
                        "summary": "Files omitted by the chunk clustering response.",
                        "members": sorted(set(missing_members)),
                        "children": [],
                    }
                )
            paths = [
                graph.nodes[file_id]["attrs"].get("path", "")
                for file_id in chunk_files
                if file_id in graph.nodes
            ]
            parent_key = _parent_key(paths)
            parent = parents.setdefault(
                parent_key,
                {
                    "name": parent_key.replace("/", "_"),
                    "path": parent_key,
                    "summary": f"Hierarchical clustering group {parent_key}.",
                    "members": [],
                    "children": [],
                },
            )
            parent["children"].extend(expanded)
            local_specs.extend(expanded)
            total_prompt_chars += len(prompt)
            total_response_chars += len(content)
            chunk_traces.append(
                {
                    "files": len(chunk_files),
                    "parent": parent_key,
                    "elapsed_sec": chunk_seconds,
                    "prompt": prompt,
                    "response": response,
                    "content": content,
                }
            )
            usages.append(response.get("usage"))
            if self.progress is not None:
                self.progress.done(
                    "Clustering chunk",
                    chunk_index,
                    len(chunks),
                    detail=f"{len(expanded)} local components",
                )

        groups = list(parents.values())
        merge_trace: dict[str, Any] | None = None
        merge_fallback = False
        if merge_mode == "llm" and len(local_specs) > 1:
            if self.progress is not None:
                self.progress.stage(
                    f"Merging {len(local_specs)} local components with LLM"
                )
            _relationship_counts, relationship_lines = _local_relationships(
                graph, local_specs
            )
            merge_prompt = _merge_prompt(graph, local_specs, relationship_lines)
            try:
                merge_started = time.monotonic()
                merge_response = self.client.chat(
                    [
                        {"role": "system", "content": MERGE_SYSTEM_PROMPT},
                        {"role": "user", "content": merge_prompt},
                    ],
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=min(8_192, self.max_tokens),
                )
                merge_seconds = round(time.monotonic() - merge_started, 3)
                total_llm_seconds += merge_seconds
                message = merge_response["choices"][0]["message"]
                merge_content = message.get("content") or ""
                merge_data = _extract_json(merge_content)
                raw_groups = merge_data.get("components")
                if not isinstance(raw_groups, list):
                    raise LLMClusterError("merge response has no `components` list")
                merged: list[dict[str, Any]] = []
                assigned_local: set[int] = set()
                for raw_group in raw_groups:
                    if not isinstance(raw_group, dict):
                        continue
                    indices: list[int] = []
                    for raw_index in raw_group.get("children") or []:
                        try:
                            index = int(raw_index)
                        except (TypeError, ValueError):
                            continue
                        if (
                            0 <= index < len(local_specs)
                            and index not in assigned_local
                        ):
                            indices.append(index)
                            assigned_local.add(index)
                    if not indices:
                        continue
                    children = [local_specs[index] for index in indices]
                    merged.append(
                        {
                            "name": str(
                                raw_group.get("name") or f"merged-{len(merged)}"
                            ),
                            "path": str(raw_group.get("path") or ""),
                            "summary": str(
                                raw_group.get("summary")
                                or "Merged hierarchical component."
                            ),
                            "members": [],
                            "children": children,
                        }
                    )
                if not merged:
                    raise LLMClusterError("merge response produced no non-empty groups")
                remaining = [
                    local_specs[index]
                    for index in range(len(local_specs))
                    if index not in assigned_local
                ]
                if remaining:
                    merged.append(
                        {
                            "name": "merged_unassigned",
                            "path": "",
                            "summary": "Local components omitted by the merge response.",
                            "members": [],
                            "children": remaining,
                        }
                    )
                groups = merged
                total_prompt_chars += len(merge_prompt)
                total_response_chars += len(merge_content)
                usages.append(merge_response.get("usage"))
                merge_trace = {
                    "elapsed_sec": merge_seconds,
                    "prompt": merge_prompt,
                    "response": merge_response,
                    "content": merge_content,
                }
                if self.progress is not None:
                    self.progress.info(
                        f"LLM merge produced {len(groups)} top-level components"
                    )
            except Exception as exc:  # noqa: BLE001 - fallback to directory merge
                merge_fallback = True
                merge_trace = {"failed": True, "error": str(exc)}
                if self.progress is not None:
                    self.progress.info(
                        f"LLM merge failed, falling back to directory merge: {exc}"
                    )
        model = self.model or getattr(
            getattr(self.client, "config", None), "model", None
        )
        session = getattr(getattr(self.client, "config", None), "session_id", None)
        tree = build_component_tree_from_groups(
            graph,
            groups=groups,
            cluster_meta={
                "mode": "llm-hierarchical",
                "model": model,
                "prompt_version": "cluster-hierarchical-v1",
                "merge_mode": merge_mode,
                "merge_fallback": merge_fallback,
                "session_id": session,
                "prompt_chars": total_prompt_chars,
                "response_chars": total_response_chars,
                "file_count": len(graph.nodes_of_type("file")),
                "code_node_count": sum(
                    1 for node in graph.nodes.values() if node["layer"] == "code"
                ),
                "chunk_count": len(chunks),
                "chunk_files": [len(chunk) for chunk in chunks],
                "max_chunk_files": max_chunk_files,
                "max_chunk_chars": chunk_chars,
                "budget": self.budget.as_dict() if self.budget else None,
                "llm_seconds": round(total_llm_seconds, 3),
                "usage": _merge_usage(usages),
            },
        )
        return tree, {
            "mode": "llm-hierarchical",
            "chunks": chunk_traces,
            "merge": merge_trace,
            "merge_mode": merge_mode,
            "merge_fallback": merge_fallback,
            "prompt_chars": total_prompt_chars,
            "response_chars": total_response_chars,
        }

    def cluster(self, graph: GraphStore) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.progress is not None:
            self.progress.stage("Clustering all files in one LLM request")
        prompt = build_cluster_prompt(
            graph, max_nodes=self.max_nodes, max_chars=self.max_chars
        )
        started = time.monotonic()
        response = self.client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        elapsed = round(time.monotonic() - started, 3)
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClusterError("LLM response has no choices[0].message") from exc
        content = message.get("content") or ""
        data = _extract_json(content)
        groups = data.get("components")
        if not isinstance(groups, list):
            raise LLMClusterError("LLM clustering JSON has no `components` list")
        groups = _expand_file_members(graph, groups)

        model = self.model or getattr(
            getattr(self.client, "config", None), "model", None
        )
        session = getattr(getattr(self.client, "config", None), "session_id", None)
        tree = build_component_tree_from_groups(
            graph,
            groups=groups,
            cluster_meta={
                "mode": "llm",
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "session_id": session,
                "prompt_chars": len(prompt),
                "response_chars": len(content),
                "file_count": len(graph.nodes_of_type("file")),
                "code_node_count": sum(
                    1 for node in graph.nodes.values() if node["layer"] == "code"
                ),
                "budget": self.budget.as_dict() if self.budget else None,
                "llm_seconds": elapsed,
                "usage": response.get("usage"),
            },
        )
        trace = {
            "elapsed_sec": elapsed,
            "prompt": prompt,
            "response": response,
            "content": content,
        }
        return tree, trace
