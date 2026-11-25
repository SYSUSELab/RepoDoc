# RepoDoc

A research repository for **repository-level project documentation generation and
updates**. It currently includes:

- **Baseline**: CodeWiki 2.0 (`baselines/codewiki`)
- **Benchmark**: CodeWikiBench (`benchmarks/codewikibench`)
- **Our method**: `methods/ours` (code knowledge graph + hierarchical documents + Web UI)
- **Shared experiment harness**: `harness/`, `configs/`, `adapters/`

Upstream checkouts keep their own `.git` histories and are never modified. Our
adapters, configuration, data, and run outputs live in separate directories so
that new baselines, models, ablations, and update experiments can be added
without forking upstream code.

`baselines/codewiki` and `benchmarks/codewikibench` are independent clones. The
top-level repository only records adapters, harness code, method code, and
experiment configuration.

## Layout

```text
RepoDoc/
├── baselines/                 # Third-party baselines (upstream clones)
│   └── codewiki/              # CodeWiki 2.0, upstream 7fcf935
├── methods/
│   └── ours/                  # Our method: multi-language graph + docs + Web UI
├── benchmarks/
│   └── codewikibench/         # CodeWikiBench, upstream 00f56b6
├── adapters/                  # System adapters with a shared generation interface
│   ├── codewiki.sh
│   ├── ours.sh
│   └── _template.sh
├── configs/
│   ├── systems.yaml           # Systems: source directory, adapter, environment
│   ├── repos.yaml             # Benchmark repositories, language, split, commit
│   ├── evaluation.yaml        # Rubric / judge / report configuration
│   └── experiments.yaml       # Experiment suites (pilot/core/full)
├── harness/                   # Experiment CLI: prepare/generate/evaluate/run/batch/summary
├── scripts/                   # Offline smoke-test scripts
├── data/                      # Cloned repositories, references, rubrics (gitignored)
├── runs/                      # Local experiment outputs (gitignored)
├── web-data/                  # Web UI clones and runs, separate from runs/ (gitignored)
├── docs/                      # Protocol, reproducibility, extension, and Q&A docs
├── tests/                     # Harness unit tests
├── Makefile
├── pyproject.toml
└── .env.example
```

## Quick Start

Requirements: Python 3.12, `uv`, Git, and a working LLM endpoint for generation
and judging.

```bash
# 1. Install all environments
make setup

# 2. Configure the model endpoint
cp .env.example .env
$EDITOR .env

# 3. Check the environment
make doctor

# 4. Start the local Web UI (the frontend is built on first use)
make web
# http://127.0.0.1:8765
```

### Common Commands

| Command | Purpose |
|---|---|
| `make generate REPO=graphrag` | Run a full RepoDoc generation |
| `make generate REPO=data/repos/graphrag` | Generate from a local path |
| `make update REPO=/path/to/target STATE=runs/ours/graphrag/state` | Incremental RepoDoc update |
| `make run REPO=graphrag` | Benchmark generation + parse + judge (ours by default) |
| `make run REPO=graphrag SYSTEM=codewiki` | Run the same benchmark with CodeWiki |
| `make evaluate REPO=graphrag` | Judge existing docs |
| `make summary EXPERIMENT=ours-graphrag-v1` | Summarize an experiment |
| `make web` | Start the Web UI |
| `make test` / `make lint` | Tests / static checks |
| `make demo` | Offline smoke tests without an LLM |
| `make help` | Common command help |
| `make help-all` | Advanced and single-stage commands |

`make generate` writes state to `runs/ours/<repo>/state` by default; the Web UI
does not list those local runs. The default experiment name is
`<system>-<repo>-v1`, and `EXPERIMENT=...` overrides it.

### Direct CLI Access

The Makefile is only a shortcut layer. The underlying CLIs remain available:

```bash
.venv-ours/bin/repodoc --help
.venv/bin/python -m harness --help
```

### CodeWiki Baseline Configuration

CodeWiki uses its own `~/.codewiki` configuration and does not read RepoDoc's
`.env`:

```bash
.venv-codewiki/bin/codewiki config set \
  --provider openai-compatible \
  --api-key "$API_KEY" \
  --base-url "$BASE_URL" \
  --main-model claude-sonnet-4 \
  --cluster-model claude-sonnet-4
```

To isolate multiple model configurations, set `CODEWIKI_HOME=/path/to/home`.
The adapter uses it as the process HOME.

### Prepare Benchmark Data

```bash
make prepare REPO=graphrag
make prepare REPO=rasa
```

This downloads CodeWikiBench reference docs and rubrics and creates pinned
checkouts. Subsequent `make run` and `make evaluate` calls use those artifacts.

## Experiment Pipeline

Every `system × repo` run has four stages. Artifacts are stored under
`runs/<experiment>/systems/<system>/<repo>/`:

```text
docs/        # Flat markdown produced by the generator
parsed/      # CodeWikiBench docs_tree.json + structured_docs.json
results/     # Per-judge JSON, combined_results.json, report.md/json
logs/        # generate.log / parse.log / judge.log
meta.json    # Status, commands, versions, repository commit, rubric hash
metrics.json # Scores, topic breakdown, tokens/cost, timings, document size
```

`metrics.json.generation` is the shared cost-efficiency interface:
`tokens` (input/output/reasoning/total), `cost_usd`, `cost_status`,
`duration_sec` (wall clock), `llm_duration_sec` (LLM call time), and per-page
token/cost/time. CodeWiki and ours must both provide a `metadata.json` usage
ledger. Missing usage is marked `unknown`/`unavailable`, never zero. See
[`runs/README.md`](runs/README.md) for naming and layout rules.

Key rules:

- **Freeze rubrics once**: `data/rubrics/<repo>/combined_rubrics.json`. Every
  system uses the same rubric and judge set so results remain comparable.
- **Generated docs must be flat**: CodeWikiBench currently parses only `.md`
  files at the docs root. The CodeWiki adapter therefore always uses `--flat`.
  When `module_tree.json` is absent, the harness synthesizes a flat module tree
  from filenames.
- **Use a dedicated CodeWiki venv**: its `coding-agent-wrapper` dependency must
  not conflict with CodeWikiBench's optional `caw` dependency.
- **Incremental updates**: `harness update` implements the base commit -> target
  commit workflow: generate base docs, check out the target, call the system's
  update mode, then judge with the same rubric/judge.
- **Our prototype**: `methods/ours` implements multi-language code graphs, LLM
  file clustering, large-component decomposition, hierarchical LLM docs,
  build/status/inspect, and an incremental update MVP. Python quantitative
  evaluation is complete; other languages are the next vertical slices.

## Web UI

RepoDoc includes a local Web UI that serves RepoDoc only and does not adapt
CodeWiki:

```bash
make web
# Builds the frontend on first use, then serves http://127.0.0.1:8765
```

- GitHub URLs and `owner/repo` inputs are cloned into `web-data/repos/`;
- local paths are used in place;
- partial repository names suggest GitHub candidates (set `GITHUB_TOKEN` to
  raise API limits);
- generation opens a dedicated progress page that survives page refreshes;
- completed runs open the generated documentation;
- Web run state and logs live under `web-data/runs/`;
- the project page provides **Ask RepoDoc** with documented, code, and component
  evidence, BM25 + optional dense RRF, and streaming answers;
- answers include clickable document/code citations and refuse when evidence is
  insufficient;
- local experiment directories under `runs/` never appear in the Web project list.

Implementation: `methods/ours/web` (Vue 3 + Vite) and
`methods/ours/src/repodoc/web` (FastAPI). The visual language is based on the
local `repo-agent` frontend, but the data model and API are RepoDoc-specific.
See [`methods/ours/web/README.md`](methods/ours/web/README.md).

## Adding a Baseline or Method

1. Clone a baseline into `baselines/<name>/`, or put a method in `methods/<name>/`.
2. Copy `adapters/_template.sh` to `adapters/<name>.sh` and implement the contract.
3. Register the system in `configs/systems.yaml`.
4. Smoke-test with `harness generate` on `OpenHands`, then use `run`/`batch`.
5. If the system supports incremental updates, set `supports_update: true` and
   handle `--mode update`.

Full checklist: [`docs/adding-a-system.md`](docs/adding-a-system.md).

## Experiment Design

- **Main experiment**: full documentation generation on 22 repositories (or the
  paper's 7-repository `core` split).
- **Update experiment**: base and target commits of the same repository, comparing
  incremental updates against full rebuilds on quality, cost, time, and regression.
- **Judging**: CodeWikiBench hierarchical rubric, binary leaf coverage, weighted
  bottom-up aggregation, combined judges, and per-run score/topic/token/error reports.
- **Statistics**: for stochastic methods, repeat each configuration at least
  three times, report mean ± std, and keep rubric/judge fixed.

Protocol: [`docs/experiment-protocol.md`](docs/experiment-protocol.md).
Reproducibility: [`docs/reproducibility.md`](docs/reproducibility.md).
Language support: [`docs/language-extension.md`](docs/language-extension.md).
Q&A design: [`docs/qa.md`](docs/qa.md).

## Current Status

| Component | Status |
|---|---|
| CodeWiki baseline | Checked out and adapter works; `make demo` passes offline |
| CodeWikiBench | Checked out; parse/judge/report work |
| Shared harness | Works for full generation, evaluation, batch, and summary |
| Our method | Stages 1/2/3 work; one-step generate, progress, retry, same-level concurrency, dynamic budgets; stage 4 update MVP |
| Frozen rubrics | Download with `prepare --from-hf` or generate with `harness rubrics` |
| Incremental update | Graph delta, component repair, added/deleted files, LLM triage; section patches implemented but still need real-API hardening |
| Languages | Python/JS/TS plus Rust/Go/Java/PHP/C/C++/C# analyzers; Chart.js JS checkout ready; full doc experiments still need one vertical slice per language |
| Cost/time accounting | CodeWiki and ours record tokens, cost, wall time, and LLM time in `metrics.json` / `summary.csv` |
| Web UI | Vue 3 + Vite + FastAPI; GitHub import, search, background generation, docs, usage, streaming Q&A |
| CodeWikiBench evaluation | GraphRAG and Rasa have full ours runs; Rasa used the official CodeWiki demo: architecture 92.62 vs 65.70, full 87.42 vs 66.16 (Qwen+GLM). Same-model CodeWiki GraphRAG was paused in favor of the official demo |
