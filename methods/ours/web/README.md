# RepoDoc Web UI

Local browser UI for RepoDoc. It accepts a GitHub URL or local checkout, runs
RepoDoc in the background, and reads the generated state directly. CodeWiki is
not part of this UI.

## Run

```bash
# first run builds the Vue app automatically
make web
# open http://127.0.0.1:8765
```

If you only want to rebuild the frontend:

```bash
make web-setup
```

For frontend development:

```bash
# terminal 1: API only
.venv-ours/bin/repodoc web --port 8000

# terminal 2: Vite with /api proxy
cd methods/ours/web
pnpm dev
# open http://127.0.0.1:5173
```

## Data layout

Web data is deliberately separated from local experiment outputs:

```text
web-data/
├── repos/
│   └── <owner>__<repo>/       # GitHub clones
└── runs/
    └── <run_id>/
        ├── run.json
        ├── generate.log
        └── state/              # normal RepoDoc state
```

Local experiments remain under `runs/`; the Web UI does not list them.

## Input

- `https://github.com/owner/repo` or `owner/repo` → clone into
  `web-data/repos/<owner>__<repo>`, then generate;
- local absolute/relative path → use it in place;
- typing a partial repository name opens GitHub search candidates plus local
  clones. Set `GITHUB_TOKEN` to raise GitHub API rate limits.

## What it shows

- a dedicated generation page with stage-specific animation and progress;
- logs hidden by default with a Show log toggle;
- a separate history page for completed Web projects;
- delete generated docs from a history card (the cloned repository is kept);
- document tree and Markdown pages;
- Mermaid diagrams with click-to-zoom;
- graph, token and cost summary in the top strip;
- LLM call/time accounting in run state;
- **Ask RepoDoc**: left conversation + right Evidence panel, bottom composer,
  Quick/Deep search, streaming answers, hybrid BM25 + dense retrieval, and
  `Sources (n)` citation navigation.

## API

```text
GET  /api/repos
GET  /api/github/search?q=...
GET  /api/projects
GET  /api/projects/{id}
GET  /api/projects/{id}/page?path=overview.md
GET  /api/projects/{id}/usage
POST /api/projects/{id}/ask         {"question": "...", "session_id": "...", "mode": "quick|deep"}
POST /api/projects/{id}/ask/stream  same body; text/event-stream
GET  /api/projects/{id}/source?path=...&start=...&end=...
DELETE /api/projects/{id}
POST /api/runs                 {"source": "https://github.com/owner/repo"}
GET  /api/runs/{run_id}
POST /api/runs/{run_id}/cancel
```

The UI intentionally reads `docs/doc_tree.json`, `components/component_tree.json`,
`graph/graph.json` and `docs/markdown/metadata.json` directly, so CLI and Web
views cannot drift apart.

## UI notes

The visual language is adapted from the local `repo-agent` frontend: quiet
light/dark palette, centered entry page, typewriter subtitle, document sidebar,
serif reading column, Mermaid zoom, and a bottom progress surface. Data shapes,
navigation and generation workflow are RepoDoc-specific.
