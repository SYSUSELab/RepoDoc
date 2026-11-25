"""Optional local dense retrieval for RepoDoc Q&A.

The dense backend is intentionally optional: BM25 remains the zero-dependency
baseline, while ``fastembed`` provides small ONNX models without pulling in
torch or a vector database. Embeddings are cached per source chunk, so a graph
or document update only embeds chunks whose searchable text changed.
"""

from __future__ import annotations

import json
import math
import os
import threading
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:
    from repodoc.qa.index import Chunk

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_MAX_CHUNKS = 6000
_CACHE_VERSION = 1


class DenseUnavailable(RuntimeError):
    """Raised when the optional dense backend cannot be initialized."""


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _normalize(vector: Iterable[float]) -> list[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        return values
    return [value / norm for value in values]


def _dot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _text_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()[:20]


def _priority(chunk: Chunk) -> tuple[int, str]:
    # Documents/components are cheap and usually the most valuable semantic
    # units. Cap code chunks only when a repository is exceptionally large.
    return ({"doc": 0, "component": 1, "code": 2}.get(chunk.kind, 3), chunk.id)


class _FastEmbedBackend:
    def __init__(self, model_name: str) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise DenseUnavailable(
                "fastembed is not installed; install methods/ours[dense]"
            ) from exc
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [list(vector) for vector in self._model.embed(texts, batch_size=64)]

    def query(self, text: str) -> list[float]:
        try:
            return list(self._model.query_embed([text]))[0]
        except (AttributeError, IndexError):
            return self.embed([text])[0]


class DenseIndex:
    """Cosine-similarity index with a per-chunk persistent embedding cache."""

    def __init__(
        self,
        chunks: list["Chunk"],
        *,
        cache_dir: Path | None = None,
        model_name: str | None = None,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        query_fn: Callable[[str], list[float]] | None = None,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
    ) -> None:
        self.chunks = sorted(chunks, key=_priority)[: max(1, max_chunks)]
        self.by_id = {chunk.id: chunk for chunk in self.chunks}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model_name = model_name or os.environ.get(
            "REPODOC_EMBED_MODEL", DEFAULT_MODEL
        )
        self._embed_fn = embed_fn
        self._query_fn = query_fn
        self._backend: _FastEmbedBackend | None = None
        self._vectors: dict[str, list[float]] = {}
        self._dim = 0
        self._ready = False
        self._status = "not_initialized"
        self._lock = threading.Lock()

    @property
    def status(self) -> str:
        return self._status

    @property
    def size(self) -> int:
        return len(self._vectors)

    def _cache_path(self) -> Path | None:
        return self.cache_dir / "embeddings.json" if self.cache_dir else None

    def _load_cache(self) -> dict[str, Any]:
        path = self._cache_path()
        if path is None:
            return {}
        data = _read_json(path, default={}) or {}
        if not isinstance(data, dict):
            return {}
        if data.get("version") != _CACHE_VERSION:
            return {}
        if data.get("model") != self.model_name:
            return {}
        entries = data.get("entries")
        return entries if isinstance(entries, dict) else {}

    def _ensure_backend(self) -> None:
        if self._embed_fn is not None or self._query_fn is not None:
            return
        if self._backend is None:
            self._backend = _FastEmbedBackend(self.model_name)

    def _embed_missing(self, pending: list["Chunk"], entries: dict[str, Any]) -> None:
        if not pending:
            return
        if self._embed_fn is not None:
            vectors = self._embed_fn([chunk.searchable() for chunk in pending])
        else:
            self._ensure_backend()
            assert self._backend is not None
            vectors = self._backend.embed([chunk.searchable() for chunk in pending])
        if len(vectors) != len(pending):
            raise DenseUnavailable("embedding backend returned the wrong batch size")
        for chunk, vector in zip(pending, vectors):
            normalized = _normalize(vector)
            if not normalized:
                continue
            self._vectors[chunk.id] = normalized
            self._dim = len(normalized)
            entries[chunk.id] = {
                "hash": _text_hash(chunk.searchable()),
                "vector": normalized,
            }

    def _save_cache(self, entries: dict[str, Any]) -> None:
        path = self._cache_path()
        if path is None:
            return
        active = {chunk.id for chunk in self.chunks}
        entries = {key: value for key, value in entries.items() if key in active}
        _write_json(
            path,
            {
                "version": _CACHE_VERSION,
                "model": self.model_name,
                "dim": self._dim,
                "entries": entries,
            },
        )

    def ensure(self) -> bool:
        with self._lock:
            if self._ready:
                return True
            if self._status == "unavailable":
                return False
            entries = self._load_cache()
            pending: list["Chunk"] = []
            for chunk in self.chunks:
                entry = entries.get(chunk.id)
                expected = _text_hash(chunk.searchable())
                if (
                    isinstance(entry, dict)
                    and entry.get("hash") == expected
                    and isinstance(entry.get("vector"), list)
                ):
                    vector = _normalize(entry["vector"])
                    if vector:
                        self._vectors[chunk.id] = vector
                        self._dim = self._dim or len(vector)
                        continue
                pending.append(chunk)
            try:
                self._embed_missing(pending, entries)
            except Exception:
                self._status = "unavailable"
                self._vectors = {}
                return False
            self._save_cache(entries)
            self._ready = True
            self._status = "ready"
            return True

    def _query_vector(self, query: str) -> list[float]:
        if self._query_fn is not None:
            return _normalize(self._query_fn(query))
        if self._embed_fn is not None:
            return _normalize(self._embed_fn([query])[0])
        self._ensure_backend()
        assert self._backend is not None
        return _normalize(self._backend.query(query))

    def search(self, query: str, limit: int = 10) -> list[tuple[Chunk, float]]:
        if not query.strip() or not self.ensure():
            return []
        query_vector = self._query_vector(query)
        if not query_vector:
            return []
        scored = [
            (self.by_id[chunk_id], _dot(query_vector, vector))
            for chunk_id, vector in self._vectors.items()
            if chunk_id in self.by_id
        ]
        scored.sort(key=lambda item: (-item[1], item[0].id))
        return scored[: max(1, limit)]
