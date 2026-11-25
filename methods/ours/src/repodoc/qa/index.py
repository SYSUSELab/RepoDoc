"""Lightweight GraphRAG index over RepoDoc documents and code.

This intentionally avoids a vector database. BM25 is strong for exact
symbol/file questions; graph expansion adds the component/doc relationships that
plain text retrieval misses. Dense retrieval can be added later behind the same
``search`` interface.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from repodoc.graph.store import GraphStore

_TOKEN_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*|\d+")


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        pass
    return rows


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text or ""):
        lower = raw.lower()
        out.append(lower)
        if "_" in lower:
            out.extend(part for part in lower.split("_") if len(part) > 1)
        # camelCase -> camel, case
        camel = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", raw)
        if len(camel) > 1:
            out.extend(part.lower() for part in camel if len(part) > 1)
    return out


def _clip(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


@dataclass
class Chunk:
    id: str
    kind: str
    title: str
    text: str
    path: str | None = None
    start: int | None = None
    end: int | None = None
    page_path: str | None = None
    heading: str | None = None
    component_id: str | None = None
    component_name: str | None = None
    covers: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    node_id: str | None = None

    def searchable(self) -> str:
        parts = [self.title, self.text]
        if self.path:
            parts.append(self.path)
            parts.append(Path(self.path).stem)
        if self.component_name:
            parts.append(self.component_name)
        return "\n".join(parts)


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = [_tokens(chunk.searchable()) for chunk in chunks]
        self.lengths = [len(tokens) for tokens in self.docs]
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        self.freqs = [Counter(tokens) for tokens in self.docs]
        df: Counter[str] = Counter()
        for freq in self.freqs:
            df.update(freq.keys())
        n = max(1, len(self.docs))
        self.idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def scores(self, query: str) -> list[float]:
        terms = _tokens(query)
        result = [0.0] * len(self.docs)
        for i, freq in enumerate(self.freqs):
            length = self.lengths[i] or 1
            score = 0.0
            for term in terms:
                tf = freq.get(term, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (
                    1 - self.b + self.b * length / max(1, self.avgdl)
                )
                score += self.idf.get(term, 0) * tf * (self.k1 + 1) / denom
            result[i] = score
        return result


class RepoIndex:
    def __init__(self, state: Path, *, dense: bool | None = None):
        self.state = state.resolve()
        self.manifest = _read_json(self.state / "manifest.json", default={}) or {}
        self._chunks: list[Chunk] = []
        self._by_id: dict[str, Chunk] = {}
        self._by_node: dict[str, Chunk] = {}
        self._by_path: dict[str, list[Chunk]] = defaultdict(list)
        self._sections_by_cover: dict[str, list[Chunk]] = defaultdict(list)
        self._dense_enabled = (
            dense
            if dense is not None
            else os.environ.get("REPODOC_DENSE", "1").lower()
            not in {"0", "false", "off", "no"}
        )
        self._dense_index: Any = None
        self._dense_status = (
            "disabled" if not self._dense_enabled else "not_initialized"
        )
        self._dense_lock = threading.Lock()
        self._build()
        self._bm25 = BM25(self._chunks)
        self._chunk_position = {
            chunk.id: index for index, chunk in enumerate(self._chunks)
        }

    @property
    def dense_status(self) -> str:
        return self._dense_status

    def _get_dense(self) -> Any:
        if not self._dense_enabled:
            return None
        with self._dense_lock:
            if self._dense_index is not None:
                return (
                    None if self._dense_status == "unavailable" else self._dense_index
                )
            from repodoc.qa.dense import DenseIndex

            model = os.environ.get("REPODOC_EMBED_MODEL") or None
            index = DenseIndex(
                self._chunks,
                cache_dir=self.state.parent / "qa" / "index",
                model_name=model,
            )
            self._dense_index = index
            if index.ensure():
                self._dense_status = "ready"
                return index
            self._dense_status = "unavailable"
            return None

    def _add(self, chunk: Chunk) -> None:
        if chunk.id in self._by_id:
            return
        self._chunks.append(chunk)
        self._by_id[chunk.id] = chunk
        if chunk.node_id:
            self._by_node.setdefault(chunk.node_id, chunk)
        if chunk.path:
            self._by_path[chunk.path].append(chunk)
        for covered in chunk.covers:
            self._sections_by_cover[covered].append(chunk)

    def _build(self) -> None:
        pages_path = self.state / (
            (self.manifest.get("artifacts") or {}).get("pages", "docs/pages.jsonl")
        )
        sections_path = self.state / (
            (self.manifest.get("artifacts") or {}).get(
                "sections", "docs/sections.jsonl"
            )
        )
        components_path = self.state / (
            (self.manifest.get("artifacts") or {}).get(
                "component_tree", "components/component_tree.json"
            )
        )
        pages = _read_jsonl(pages_path)
        page_by_path = {str(page.get("path")): page for page in pages}
        component_page: dict[str, str] = {}
        for page in pages:
            if page.get("component_id") and page.get("path"):
                component_page[str(page["component_id"])] = str(page["path"])

        for section in _read_jsonl(sections_path):
            page_path = str(section.get("page_path") or "")
            page = page_by_path.get(page_path, {})
            heading = str(section.get("heading") or "Documentation")
            content = str(section.get("content") or "").strip()
            if not content:
                continue
            title = f"{page.get('title') or page_path} / {heading}"
            self._add(
                Chunk(
                    id=str(section.get("id") or f"section:{page_path}:{heading}"),
                    kind="doc",
                    title=title,
                    text=f"{heading}\n{content}",
                    page_path=page_path,
                    heading=heading,
                    component_id=page.get("component_id"),
                    covers=[str(item) for item in (section.get("covers") or [])],
                    citations=[str(item) for item in (section.get("citations") or [])],
                )
            )

        component_tree = _read_json(components_path, default={}) or {}
        component_nodes: dict[str, str] = {}
        component_names: dict[str, str] = {}

        def add_component(node: dict[str, Any], parent: str | None = None) -> None:
            cid = str(node.get("id") or "")
            if not cid:
                return
            name = str(node.get("name") or cid)
            summary = str(node.get("summary") or "")
            children = node.get("children") or []
            page_path = component_page.get(cid)
            text = (
                f"{name}\n{summary}\n"
                f"path: {node.get('path') or ''}\n"
                f"kind: {node.get('kind') or 'component'}\n"
                f"children: {', '.join(str(child.get('name')) for child in children)}\n"
                f"members: {len(node.get('members') or [])}"
            )
            self._add(
                Chunk(
                    id=cid,
                    kind="component",
                    title=name,
                    text=text,
                    component_id=cid,
                    component_name=name,
                    page_path=page_path,
                    node_id=cid,
                )
            )
            component_names[cid] = name
            for member in node.get("members") or []:
                component_nodes.setdefault(str(member), cid)
            for child in children:
                if isinstance(child, dict):
                    add_component(child, cid)

        if isinstance(component_tree.get("root"), dict):
            add_component(component_tree["root"])

        graph_path = self.state / (
            (self.manifest.get("artifacts") or {}).get("graph", "graph/graph.json")
        )
        if graph_path.is_file():
            try:
                graph = GraphStore.load(self.state)
            except (OSError, json.JSONDecodeError):
                graph = None
            if graph is not None:
                for node in graph.nodes.values():
                    if node.get("type") != "symbol":
                        continue
                    attrs = node.get("attrs") or {}
                    qualified = str(
                        attrs.get("qualified_name") or attrs.get("name") or node["id"]
                    )
                    path = str(attrs.get("path") or "")
                    start = attrs.get("start", {}).get("line")
                    end = attrs.get("end", {}).get("line")
                    source = str(attrs.get("source") or "")
                    signature = str(attrs.get("signature") or "")
                    docstring = str(attrs.get("docstring") or "")
                    cid = component_nodes.get(node["id"])
                    component_name = component_names.get(cid) if cid else None
                    citation = (
                        f"{path}:{start}-{end}"
                        if path and start is not None and end is not None
                        else path
                    )
                    text = "\n".join(
                        item
                        for item in (
                            f"{qualified}",
                            f"kind: {attrs.get('kind') or ''}",
                            f"signature: {signature}" if signature else "",
                            f"docstring: {docstring}" if docstring else "",
                            source,
                        )
                        if item
                    )
                    self._add(
                        Chunk(
                            id=f"symbol:{node['id']}",
                            kind="code",
                            title=qualified,
                            text=_clip(text, 5_000),
                            path=path or None,
                            start=int(start) if start is not None else None,
                            end=int(end) if end is not None else None,
                            component_id=cid,
                            component_name=component_name,
                            citations=[citation] if citation else [],
                            node_id=str(node["id"]),
                        )
                    )

    def _expand(self, selected: list[Chunk], question: str, limit: int) -> list[Chunk]:
        result = list(selected)
        seen = {chunk.id for chunk in result}
        query_lower = question.lower()
        architecture = any(
            word in query_lower
            for word in (
                "architecture",
                "architectural",
                "overview",
                "structure",
                "component",
                "module",
                "flow",
                "design",
                "pipeline",
            )
        )
        for chunk in selected[:5]:
            if chunk.kind == "doc":
                # The section explicitly points to code nodes it covers.
                for node_id in chunk.covers[:4]:
                    target = self._by_node.get(node_id)
                    if target and target.id not in seen:
                        result.append(target)
                        seen.add(target.id)
                for component_id in (chunk.component_id,):
                    target = (
                        self._by_id.get(str(component_id)) if component_id else None
                    )
                    if target and target.id not in seen:
                        result.append(target)
                        seen.add(target.id)
                for citation in chunk.citations[:2]:
                    path = citation.split(":", 1)[0]
                    for target in self._by_path.get(path, [])[:2]:
                        if target.id not in seen:
                            result.append(target)
                            seen.add(target.id)
            elif chunk.kind == "code":
                by_cover = self._sections_by_cover.get(chunk.node_id or "", [])[:2]
                for target in by_cover:
                    if target.id not in seen:
                        result.append(target)
                        seen.add(target.id)
                if chunk.component_id:
                    target = self._by_id.get(chunk.component_id)
                    if target and target.id not in seen:
                        result.append(target)
                        seen.add(target.id)
        if architecture:
            for chunk in self._chunks:
                if len(result) >= limit:
                    break
                if chunk.kind == "component" and chunk.id not in seen:
                    result.append(chunk)
                    seen.add(chunk.id)
        return result[:limit]

    def _lexical_scores(self, query: str) -> list[float]:
        scores = self._bm25.scores(query)
        lower = query.lower()
        architecture_query = any(
            word in lower
            for word in (
                "architecture",
                "architectural",
                "overview",
                "structure",
                "component",
                "module",
                "flow",
                "design",
            )
        )
        for index, chunk in enumerate(self._chunks):
            title = chunk.title.lower()
            path = (chunk.path or "").lower()
            if architecture_query and chunk.kind == "component":
                scores[index] += 7.0
            if (
                architecture_query
                and chunk.kind == "doc"
                and ("architecture" in title or "overview" in title)
            ):
                scores[index] += 3.0
            if title == lower or title in lower or lower in title:
                scores[index] += 8.0
            if path and (path in lower or Path(path).name.lower() in lower):
                scores[index] += 5.0
            if chunk.kind == "code" and any(
                token in title for token in _tokens(query) if len(token) > 3
            ):
                scores[index] += 1.5
        return scores

    @staticmethod
    def _rrf_fuse(
        lexical: list[tuple[int, float]],
        dense: list[tuple[int, float]],
        lexical_weight: float = 0.58,
    ) -> list[int]:
        scores: dict[int, float] = {}
        for rank, (index, _) in enumerate(lexical, start=1):
            scores[index] = scores.get(index, 0.0) + lexical_weight / (60 + rank)
        for rank, (index, _) in enumerate(dense, start=1):
            scores[index] = scores.get(index, 0.0) + (1 - lexical_weight) / (60 + rank)
        return [
            index
            for index, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        ]

    def search(self, query: str, limit: int = 10) -> list[Chunk]:
        if not query.strip():
            return []
        scores = self._lexical_scores(query)
        lexical = sorted(
            ((index, score) for index, score in enumerate(scores) if score > 0),
            key=lambda item: (-item[1], self._chunks[item[0]].id),
        )
        dense_index = self._get_dense()
        dense = []
        if dense_index is not None:
            dense = [
                (self._chunk_position[chunk.id], score)
                for chunk, score in dense_index.search(query, limit=max(limit * 4, 40))
                if chunk.id in self._chunk_position
            ]
            # DenseIndex already sorts; retain its rank order.
        if dense:
            ordered = self._rrf_fuse(lexical, dense)
            selected = [self._chunks[index] for index in ordered[:limit]]
        else:
            selected = [self._chunks[index] for index, _ in lexical[:limit]]
        return self._expand(selected, query, max(limit, 12))


def build_index(state: str | Path) -> RepoIndex:
    return RepoIndex(Path(state))
