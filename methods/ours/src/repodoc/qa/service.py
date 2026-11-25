"""Answer questions about a RepoDoc state with cited GraphRAG evidence."""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from repodoc.llm import LLMError, OpenCodeGoClient
from repodoc.qa.index import Chunk, RepoIndex, build_index

_INDEX_LOCK = threading.Lock()
_INDEX_CACHE: dict[Path, tuple[str, RepoIndex]] = {}
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")

_REFUSAL = (
    "I could not find enough evidence in this repository's documentation and "
    "code to answer the question. Try naming a specific file, module, class, "
    "or function."
)

_SYSTEM_PROMPT = """You are RepoDoc, a repository documentation assistant.

Answer the user's question using ONLY the provided evidence.
- Cite evidence inline as [1], [2], ... matching its number.
- Prefer concise, technical answers.
- If the evidence does not contain enough information, say so directly.
- Do not use general world knowledge to fill gaps.
- Answer in the same language as the user's question.
- Use Markdown; code references should include path and line range when available.
"""


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def _index_key(state: Path) -> str:
    manifest = _read_json(state / "manifest.json", default={}) or {}
    return "|".join(
        str(manifest.get(key) or "")
        for key in ("doc_version", "graph_version", "component_version")
    )


def get_index(state: str | Path) -> RepoIndex:
    path = Path(state).resolve()
    key = _index_key(path)
    with _INDEX_LOCK:
        cached = _INDEX_CACHE.get(path)
        if cached and cached[0] == key:
            return cached[1]
        index = build_index(path)
        _INDEX_CACHE[path] = (key, index)
        return index


def _session_path(state: Path, session_id: str) -> Path:
    return state.parent / "qa" / "sessions" / f"{session_id}.json"


def _load_session(state: Path, session_id: str) -> dict[str, Any]:
    path = _session_path(state, session_id)
    data = _read_json(path, default={}) or {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("session_id", session_id)
    data.setdefault("messages", [])
    return data


def _citation(chunk: Chunk, number: int) -> dict[str, Any]:
    if chunk.kind == "doc":
        return {
            "number": number,
            "type": "doc",
            "label": chunk.title,
            "page": chunk.page_path,
            "heading": chunk.heading,
            "id": chunk.id,
        }
    if chunk.kind == "code":
        return {
            "number": number,
            "type": "code",
            "label": (
                f"{chunk.path}:{chunk.start}-{chunk.end}"
                if chunk.path and chunk.start and chunk.end
                else chunk.title
            ),
            "path": chunk.path,
            "start": chunk.start,
            "end": chunk.end,
            "component": chunk.component_name,
            "id": chunk.id,
        }
    return {
        "number": number,
        "type": "component",
        "label": chunk.title,
        "page": chunk.page_path,
        "component_id": chunk.component_id,
        "id": chunk.id,
    }


def _evidence_block(number: int, chunk: Chunk, limit: int = 5000) -> str:
    if chunk.kind == "doc":
        header = (
            f"[{number}] DOC {chunk.page_path or ''} / {chunk.heading or chunk.title}"
        )
    elif chunk.kind == "code":
        line = ""
        if chunk.path and chunk.start and chunk.end:
            line = f"{chunk.path}:{chunk.start}-{chunk.end}"
        header = f"[{number}] CODE {line or chunk.title}"
        if chunk.component_name:
            header += f" (component: {chunk.component_name})"
    else:
        header = f"[{number}] COMPONENT {chunk.title}"
    text = chunk.text
    if len(text) > limit:
        text = text[:limit] + "\n...[truncated]"
    return f"{header}\n{text}"


@dataclass
class _PreparedQuestion:
    state: Path
    session: dict[str, Any]
    session_id: str
    mode: str
    chunks: list[Chunk]
    citations: list[dict[str, Any]]
    user_prompt: str
    client: OpenCodeGoClient | None
    rewrite_usage: dict[str, Any] | None


class QAService:
    def _prepare(
        self,
        state: str | Path,
        question: str,
        session_id: str | None,
        mode: str,
    ) -> _PreparedQuestion:
        state_path = Path(state).resolve()
        question = question.strip()
        if not question:
            raise ValueError("question is required")
        if not session_id or not _SESSION_ID_RE.fullmatch(session_id):
            session_id = uuid.uuid4().hex
        mode = "deep" if mode == "deep" else "quick"
        session = _load_session(state_path, session_id)
        index = get_index(state_path)
        client = OpenCodeGoClient()
        rewrite_usage = None
        queries = [question]
        if mode == "deep":
            queries, rewrite_usage = self._rewrite_queries(
                client, state_path, question, session
            )
        chunks = self._search_many(index, queries, limit=10)
        if not chunks:
            client.close()
            return _PreparedQuestion(
                state=state_path,
                session=session,
                session_id=session_id,
                mode=mode,
                chunks=[],
                citations=[],
                user_prompt="",
                client=None,
                rewrite_usage=rewrite_usage,
            )

        evidence = "\n\n".join(
            _evidence_block(number, chunk)
            for number, chunk in enumerate(chunks, start=1)
        )
        history = self._history(session)
        user_prompt = (
            f"Repository: {self._repo_name(state_path)}\n\n"
            f"Conversation history:\n{history or '(none)'}\n\n"
            f"Evidence:\n---BEGIN EVIDENCE---\n{evidence}\n---END EVIDENCE---\n\n"
            f"Question: {question}\n\n"
            "Answer with inline citations like [1]. "
            "If the evidence is insufficient, say exactly what is missing."
        )
        citations = [
            _citation(chunk, number) for number, chunk in enumerate(chunks, start=1)
        ]
        return _PreparedQuestion(
            state=state_path,
            session=session,
            session_id=session_id,
            mode=mode,
            chunks=chunks,
            citations=citations,
            user_prompt=user_prompt,
            client=client,
            rewrite_usage=rewrite_usage,
        )

    def _refusal_result(
        self, prepared: _PreparedQuestion, answer: str = _REFUSAL
    ) -> dict[str, Any]:
        result = {
            "answer": answer,
            "citations": [],
            "session_id": prepared.session_id,
            "mode": prepared.mode,
            "usage": (
                {"rewrite": prepared.rewrite_usage} if prepared.rewrite_usage else None
            ),
        }
        self._save_turn(prepared.state, prepared.session, answer, result)
        return result

    def ask(
        self,
        state: str | Path,
        question: str,
        session_id: str | None = None,
        mode: str = "quick",
    ) -> dict[str, Any]:
        prepared = self._prepare(state, question, session_id, mode)
        if not prepared.chunks:
            return self._refusal_result(prepared)
        assert prepared.client is not None
        try:
            response = prepared.client.chat(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prepared.user_prompt},
                ],
                temperature=0.1,
            )
        except LLMError as exc:
            raise RuntimeError(f"answer model failed: {exc}") from exc
        finally:
            prepared.client.close()
        try:
            content = response["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("answer model returned no content") from exc
        if not content.strip():
            raise RuntimeError("answer model returned empty content")

        result = {
            "answer": content.strip(),
            "citations": prepared.citations,
            "session_id": prepared.session_id,
            "mode": prepared.mode,
            "usage": {
                "answer": response.get("usage"),
                **(
                    {"rewrite": prepared.rewrite_usage}
                    if prepared.rewrite_usage
                    else {}
                ),
            },
        }
        self._save_turn(prepared.state, prepared.session, question.strip(), result)
        return result

    def ask_stream(
        self,
        state: str | Path,
        question: str,
        session_id: str | None = None,
        mode: str = "quick",
    ) -> Iterator[dict[str, Any]]:
        """Yield start/delta/done/error events for a streaming answer."""

        prepared = self._prepare(state, question, session_id, mode)
        if not prepared.chunks:
            start_event = {
                "type": "start",
                "session_id": prepared.session_id,
                "mode": prepared.mode,
                "citations": [],
                "usage": (
                    {"rewrite": prepared.rewrite_usage}
                    if prepared.rewrite_usage
                    else None
                ),
            }
            result = self._refusal_result(prepared)
            yield start_event
            yield {"type": "delta", "content": _REFUSAL}
            yield {
                "type": "done",
                "answer": _REFUSAL,
                "citations": [],
                "session_id": result["session_id"],
                "mode": result["mode"],
                "usage": result.get("usage"),
            }
            return

        assert prepared.client is not None
        parts: list[str] = []
        answer_usage: dict[str, Any] | None = None
        try:
            yield {
                "type": "start",
                "session_id": prepared.session_id,
                "mode": prepared.mode,
                "citations": prepared.citations,
                "usage": (
                    {"rewrite": prepared.rewrite_usage}
                    if prepared.rewrite_usage
                    else None
                ),
            }
            for chunk in prepared.client.stream_chat(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prepared.user_prompt},
                ],
                temperature=0.1,
            ):
                if isinstance(chunk.get("usage"), dict):
                    answer_usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    parts.append(content)
                    yield {"type": "delta", "content": content}
            answer = "".join(parts).strip()
            if not answer:
                raise RuntimeError("answer model returned empty content")
            result = {
                "answer": answer,
                "citations": prepared.citations,
                "session_id": prepared.session_id,
                "mode": prepared.mode,
                "usage": {
                    "answer": answer_usage,
                    **(
                        {"rewrite": prepared.rewrite_usage}
                        if prepared.rewrite_usage
                        else {}
                    ),
                },
            }
            self._save_turn(prepared.state, prepared.session, question.strip(), result)
            yield {
                "type": "done",
                "answer": answer,
                "citations": prepared.citations,
                "session_id": prepared.session_id,
                "mode": prepared.mode,
                "usage": result["usage"],
            }
        except Exception as exc:
            yield {
                "type": "error",
                "message": f"answer model failed: {exc}",
            }
        finally:
            prepared.client.close()

    def _rewrite_queries(
        self,
        client: OpenCodeGoClient,
        state: Path,
        question: str,
        session: dict[str, Any],
    ) -> tuple[list[str], dict[str, Any] | None]:
        prompt = (
            f"Repository: {self._repo_name(state)}\n"
            f"Conversation history:\n{self._history(session, limit=4) or '(none)'}\n\n"
            f"User question: {question}\n\n"
            "Write 2-3 short technical search queries for finding evidence in "
            "repository documentation and code. Use exact class, function, file "
            "and component names when possible. One query per line. No bullets, "
            "no explanations."
        )
        try:
            response = client.chat(
                [
                    {
                        "role": "system",
                        "content": "You rewrite repository questions into precise retrieval queries.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            content = str(response["choices"][0]["message"].get("content") or "")
            queries = [
                line.strip(" -\t") for line in content.splitlines() if line.strip()
            ][:3]
            queries = [query for query in queries if len(query) >= 2]
            return [question, *queries], response.get("usage")
        except Exception:
            return [question], None

    @staticmethod
    def _search_many(index: RepoIndex, queries: list[str], limit: int) -> list[Chunk]:
        if len(queries) == 1:
            return index.search(queries[0], limit=limit)
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for query in queries:
            for rank, chunk in enumerate(index.search(query, limit=limit), start=1):
                chunks[chunk.id] = chunk
                scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (40 + rank)
        ordered = sorted(chunks.values(), key=lambda chunk: -scores.get(chunk.id, 0.0))
        return ordered[: max(limit, 12)]

    @staticmethod
    def _repo_name(state: Path) -> str:
        manifest = _read_json(state / "manifest.json", default={}) or {}
        return str(manifest.get("repo") or state.parent.name)

    @staticmethod
    def _history(session: dict[str, Any], limit: int = 6) -> str:
        messages = session.get("messages") or []
        rows = []
        for message in messages[-limit:]:
            role = "User" if message.get("role") == "user" else "Assistant"
            content = str(message.get("content") or "")
            if len(content) > 3000:
                content = content[:3000] + "\n...[truncated]"
            rows.append(f"{role}: {content}")
        return "\n".join(rows)

    def _save_turn(
        self,
        state: Path,
        session: dict[str, Any],
        question: str,
        result: dict[str, Any],
    ) -> None:
        now = datetime.now(UTC).isoformat()
        session["updated_at"] = now
        session.setdefault("messages", []).append(
            {"role": "user", "content": question, "created_at": now}
        )
        session["messages"].append(
            {
                "role": "assistant",
                "content": result.get("answer") or "",
                "citations": result.get("citations") or [],
                "usage": result.get("usage"),
                "created_at": now,
            }
        )
        _write_json(_session_path(state, str(session["session_id"])), session)
