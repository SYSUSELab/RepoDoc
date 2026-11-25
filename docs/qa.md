# RepoDoc Q&A (GraphRAG MVP)

## Goals

Allow questions about a completed RepoDoc project. Answers must:

- be grounded in the repository's docs, code, and component graph;
- include clickable document/code citations;
- explicitly refuse when evidence is insufficient;
- never mix in CodeWiki or general world knowledge.

## Implementation

A lightweight GraphRAG pipeline without Chroma or LangChain:

```text
question
→ BM25 over:
  - doc sections
  - code symbols
  - component summaries
→ optional dense retrieval:
  - BAAI/bge-small-en-v1.5 (fastembed / ONNX)
  - cosine similarity
→ BM25 + dense RRF fusion
→ exact symbol/file match bonus
→ graph expansion:
  - doc_section.covers → code symbols
  - doc_section.citations → source files
  - code symbol → component
  - component → document page
→ top evidence (≤12 chunks)
→ Quick: one cited streaming LLM answer
→ Deep: LLM query rewrite → multi-query hybrid retrieval → RRF → streaming answer
```

Dense retrieval is optional:

```bash
# Enabled by default; falls back to BM25 when fastembed or the model is unavailable
export REPODOC_DENSE=1
export REPODOC_EMBED_MODEL=BAAI/bge-small-en-v1.5

# Disable dense retrieval
export REPODOC_DENSE=0
```

The first use downloads the bge-small ONNX model through fastembed. Embeddings
are cached by chunk content hash:

```text
<run>/qa/index/embeddings.json
```

After a document or graph version change, only changed chunks are re-embedded.

### Index Units

| Type | Source | Contents |
|---|---|---|
| `doc` | `docs/sections.jsonl` | page title, heading, body, covers, citations |
| `code` | graph `symbol` nodes | qualified name, signature, docstring, source span |
| `component` | `component_tree.json` | name, summary, path, children, member count |

The index is built on first question and cached; it is invalidated when
`doc_version` / `graph_version` changes. The BM25 index is rebuilt in memory;
dense vectors use a persistent per-chunk cache.

### Sessions

Sessions live at:

```text
web-data/runs/<run_id>/qa/sessions/<session_id>.json
```

Each session stores questions, answers, citations, and usage. Summary-based
memory is a future extension.

### API

```text
POST /api/projects/{id}/ask
{
  "question": "How does redirect handling work?",
  "session_id": "optional-stable-id",
  "mode": "quick"
}
```

`mode=deep` adds query rewriting and multi-query fusion at the cost of one cheap
LLM call.

Streaming endpoint:

```text
POST /api/projects/{id}/ask/stream
Content-Type: application/json

{"question": "...", "session_id": "...", "mode": "quick"}
```

It returns `text/event-stream` with:

```text
start  → session_id / citations / mode
delta  → incremental answer text
done   → complete answer / citations / usage
error  → message
```

Source endpoint:

```text
GET /api/projects/{id}/source?path=src/foo.py&start=10&end=40
```

`ask` response:

```json
{
  "answer": "... [1] ...",
  "citations": [
    {"type": "code", "path": "src/foo.py", "start": 10, "end": 40},
    {"type": "doc", "page": "overview.md", "heading": "..."}
  ],
  "session_id": "...",
  "usage": {}
}
```

## UI

- Dedicated `/project/:id/ask` page.
- Fixed two-column layout:
  - left: conversation and bottom composer;
  - right: Evidence panel.
- Each answer has `Sources (n) →`.
- The right panel has two states:
  - `Sources`: citation list;
  - `Source`: one code/document evidence view.
- Clicking a citation switches the right panel to Source without interrupting the
  left conversation.
- The Evidence header is fixed; its body is the only scroll container.
- Back to Sources stays on the Ask page.
- Mobile switches between Chat and Evidence tabs.
- No nested code scrollers, colored cards, or heavy shadows.

## Known Limitations

- The dense model is optional and requires a first-use download.
- Code retrieval still depends on names and symbol text; semantic aliasing is limited.
- BM25 is in-memory and rebuilt on startup; dense vectors are persisted.
- Deep mode is query rewrite plus multi-query fusion.
- `calls/imports` expansion is one conservative hop; full impact analysis is future work.
- There is no QA evaluation set or quantitative metric yet.

## Next Steps

1. Build a QA evaluation set (not part of the current implementation phase).
2. Refresh document/code chunks incrementally from graph deltas.
3. Add stricter citation precision and groundedness checks.
4. Add multi-turn context compression and retrieval feedback.
