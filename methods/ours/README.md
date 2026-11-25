# Our Method: Knowledge Graph + Hierarchical Documents

Stages 1–3 are implemented: code graph, LLM file clustering, and hierarchical LLM
documentation. The graph supports Python, JavaScript/TypeScript, Rust, Go, Java,
PHP, C, C++, and C#. Top-level and parent pages are architecture documents with
Mermaid diagrams; lower-level pages keep API and implementation detail. Stage 4
is an incremental-update MVP, and the project includes a local Web UI.

## Implemented

```text
methods/ours/
├── pyproject.toml
├── src/repodoc/
│   ├── graph/           # tree-sitter analyzers -> code graph
│   ├── document/        # component tree -> doc tree -> markdown
│   ├── change/          # git diff -> planned change set
│   ├── pipeline/        # build / manifest / status / update
│   ├── schemas/         # JSON Schema contracts
│   └── cli.py
└── tests/
```

Capabilities:

- Parse Python, JavaScript, TypeScript, Rust, Go, Java, PHP, C, C++, and C#
  repositories into file/symbol/API nodes and `contains`, `imports`, `calls`,
  `exposes_api`, and `handles` edges.
- Record source spans, docstrings, comments, signatures, and paths.
- Build a deterministic **toy component tree** from directories/files and write
  `component` nodes plus `has_child_component`/`part_of_component` edges.
- Build a **doc tree** (overview + component pages); pages/sections are written
  back to the graph through `documents`, `cites`, `child_of`, and `derived_from`
  edges.
- Emit `graph.json`, `nodes.jsonl`, `edges.jsonl`, `component_tree.json`,
  `doc_tree.json`, `pages.jsonl`, `sections.jsonl`, and markdown.
- `status` decides whether an update is needed from the Git commit or source hashes.
- `update` performs graph delta, component membership repair, affected-page
  discovery, and incremental document regeneration.
- Stage 2 supports `--cluster-mode llm`: the LLM assigns **file nodes** only;
  symbols/APIs inherit their file assignment. The default remains deterministic.
- Stage 3 docs: L0/L1 leaf pages cover APIs/implementation; L2 component pages and
  the overview cover boundaries, component relationships, data/control flow,
  design decisions, and tradeoffs, with mandatory Mermaid diagrams.
- High-level context is assembled from the component tree, cross-component
  import/call edges, representative calls, entry points, and child docs instead
  of simply concatenating APIs.

## Language Support

| Language | Extensions | Definitions | Public API rule | Imports |
|---|---|---|---|---|
| Python | `.py` | module / class / function | non-underscore names | `import` / `from` |
| JavaScript | `.js/.jsx/.mjs/.cjs` | function / class / method / arrow | `export` | ESM / CommonJS |
| TypeScript / TSX | `.ts/.tsx/.mts/.cts` | JS plus interface/type/enum | `export` | ESM |
| Rust | `.rs` | struct/enum/trait/union/type/fn/method | `pub` | `use` / `mod` |
| Go | `.go` | type/function/method/const/var | leading uppercase | `import` package |
| Java | `.java` | class/interface/enum/record/method/constructor/field | `public` | `import` / static import |
| PHP | `.php/.phtml` | class/interface/trait/enum/function/method | `public` | `use` / `require` / `include` |
| C | `.c/.h` | struct/union/enum/typedef/function/macro | non-`static` | `#include` |
| C++ | `.cpp/.cc/.cxx/.hpp/.hh/.h` | class/struct/function/method/constructor | `public` / non-`static` | `#include` |
| C# | `.cs` | class/interface/struct/record/enum/method/property/field | `public` | `using` |

Language detection chooses the dominant source language by file count. Mixed
repositories currently use one primary analyzer; per-file multi-parser graph
building is a later extension.

## Schema Contracts

```text
src/repodoc/schemas/
├── graph.schema.json
├── component_tree.schema.json
├── doc_tree.schema.json
├── manifest.schema.json
└── change_set.schema.json
```

Design rules:

- Component Tree and Doc Tree are separate; components link to doc pages through
  `documents` edges.
- The document unit is **page + section**.
- Sections store `covers` and `citations`, which support incremental updates.
- The graph is the source of truth; JSON and Markdown are projections.

## One-Step Generate

Users do not need to compose `build` and `llm-docs` manually:

```bash
# From the repository root: choose clustering, build, and generate docs
make generate REPO=data/repos/graphrag

# Start the Web UI
make web
```

The low-level CLI remains available:

```bash
# Full: choose clustering, build, and generate all LLM docs
.venv-ours/bin/repodoc generate --repo /path/to/repo

# Update: plan and apply changes from an existing state
.venv-ours/bin/repodoc generate \
  --repo /path/to/target \
  --state-dir /path/to/base-state \
  --mode update
```

Defaults:

```text
cluster-mode: auto
  small repositories  -> one LLM clustering request
  large repositories  -> hierarchical + LLM merge
cluster-chunk-files: 80
cluster-chunk-chars: 80000
docs batch size:     8
docs workers:        4
docs output ratio:   0.1
LLM request timeout: 300s
```

Failed batches are recursively split and only failed pages are retried. Final
failures are written to `docs/failed_pages.json`; rerunning the same command
resumes. Failures in high-level docs are never silently treated as complete.

Advanced experiments can still use `build`, `llm-docs`, and `update` directly
with arbitrary overrides.

## Document Naming

Page filenames are derived from component names and source paths. Numeric
suffixes such as `_part_1` or `-2` are not used:

- Unsplit components use the component name.
- Multi-file chunks from a large component use one to three source-file stems,
  for example `blueprints-ctx-debughelpers.md`.
- A large single file split into chunks uses the file stem plus the first
  meaningful symbol/API name, for example `test-basic-request-context.md`.
- Duplicate component names use path context, for example `pkg2-utils.md`
  instead of `utils-2.md`.
- A stable content hash is only a last-resort fallback when no semantic name
  exists.

Existing generated states are not renamed automatically. Rerun
`repodoc build` / `generate` to apply the new naming.

## Token, Cost, and Time Accounting

`repodoc` rebuilds usage for all real LLM calls from state and writes a
CodeWiki-compatible `metadata.json`:

```bash
.venv-ours/bin/repodoc stats --state-dir /path/to/state \
  --out /path/to/state/docs/markdown/metadata.json
```

The ledger covers:

- clustering (bounded-chunk calls and merge);
- all document generation calls in `docs/llm_traces/`, including failed attempts;
- update planner and patch calls in `changes/`;
- `usage.totals` with input/output/reasoning/total tokens, cache, requests, and cost;
- `usage.by_phase`, `usage.by_model`, `usage.calls`, and `cost_status`
  (`complete`/`partial`/`unknown`);
- exact calls from the current CLI process in `usage/runs/<run_id>.json`, linked
  as `usage.last_run`;
- `created_at` and `elapsed_sec` on new traces. Old traces without these fields
  remain `null`; they are not fabricated.

Harness experiments call adapters and merge usage into
`metrics.json.generation`; wall time is recorded by the harness, not by adapters.

`stats --run-scope state` is the default and summarizes the whole state for full
generation. `stats --run-scope latest-run` reads the latest
`usage/runs/<run_id>.json` for one incremental update. When `build` and
`llm-docs` run as separate processes their ledgers are separate; one-step
`generate` combines clustering and docs in one ledger.

## Web UI

`methods/ours/web` is a local Vue 3 + Vite UI that reads RepoDoc state directly
and does not introduce a second data model or adapt CodeWiki:

```bash
make web         # First run: pnpm install + vite build, then http://127.0.0.1:8765
```

It supports GitHub URL / `owner/repo` imports, repository search, local paths,
background generation, document tree, Markdown/Mermaid, graph and usage views,
and Ask RepoDoc over doc sections, code symbols, and component graph evidence
(with citations, BM25 + optional dense RRF, and streaming answers). Web data
lives under `web-data/repos` and `web-data/runs`, separate from local `runs/`.
See [`docs/qa.md`](../../docs/qa.md) and
[`methods/ours/web/README.md`](web/README.md).

## LLM: OpenCode Go / deepseek-v4.1-flash

Ours uses an independent LLM configuration and never reads CodeWiki's
`~/.codewiki` or CodeWikiBench's `BASE_URL/API_KEY`.

Request shape:

```text
POST https://opencode.ai/inference/go/openai/v1/chat/completions
Authorization: Bearer oc_sk_...
Content-Type: application/json
User-Agent: repodoc-ours/0.1
x-opencode-session: <stable conversation/run id>

{"model": "deepseek-v4.1-flash", "messages": [...]}
```

Environment:

```bash
export OPENCODE_GO_API_KEY=oc_sk_...
# Optional; defaults point at Go chat-completions
# export OPENCODE_GO_BASE_URL=https://opencode.ai/inference/go/openai/v1/
# export OPENCODE_GO_MODEL=deepseek-v4.1-flash
# export OPENCODE_GO_SESSION=repodoc-local
```

A project-root `.env` is also read from the current working directory without
overriding existing environment variables.

Validate configuration without sending a request:

```bash
.venv-ours/bin/repodoc llm-check --no-call
```

Send a minimal request after setting the key:

```bash
.venv-ours/bin/repodoc llm-check
```

`--no-call` prints the base URL, model, session, user agent, and masked key. A
real request prints the response and usage, including OpenCode's
`usage.opencode` cost/allowance fields.

## Running

`methods/ours` uses its own `.venv-ours` and does not depend on CodeWiki's venv
or configuration:

```bash
make setup
.venv-ours/bin/repodoc --version
```

Examples:

```bash
.venv-ours/bin/repodoc build \
  --repo /path/to/toy-repo \
  --state-dir /tmp/repodoc-state \
  --docs-dir /tmp/repodoc-state/docs/markdown \
  --force

.venv-ours/bin/repodoc status \
  --repo /path/to/toy-repo --state-dir /tmp/repodoc-state

.venv-ours/bin/repodoc inspect \
  --state-dir /tmp/repodoc-state --what graph
.venv-ours/bin/repodoc inspect \
  --state-dir /tmp/repodoc-state --what components
.venv-ours/bin/repodoc inspect \
  --state-dir /tmp/repodoc-state --what docs
```

Test:

```bash
make test
```

### Dynamic Token/Context Budgets

Budgets are not hard-coded at `12000 chars / 8192 tokens`. `repodoc llm-check`
reads the model catalog and computes task budgets by ratio:

```text
deepseek-v4.1-flash:
  model context=1,000,000  output=384,000

cluster budget:
  input_tokens=20,000  output_tokens=7,680  context_chars=70,000

docs budget:
  input_tokens=40,000  output_tokens=19,200 context_chars=140,000
```

Policy:

```text
cluster: 2% of context, 2% of max output
docs:    4% of context, 5% of max output
```

Ratios can be overridden with `REPODOC_CLUSTER_*_RATIO` /
`REPODOC_DOC_*_RATIO` or CLI flags such as `--cluster-context-ratio` and
`--docs-context-ratio`. Fixed minima are safety floors for small models, not the
primary budget.

### Large-Component Decomposition

After clustering, `build` splits components above the configured thresholds:

```bash
.venv-ours/bin/repodoc build ... \
  --max-component-nodes 300 \
  --max-component-children 8 \
  --min-component-child-nodes 80
```

Strategy:

- Group members by source file.
- Split a single oversized file into symbol/API chunks.
- Clear the parent's members and create child components/pages.
- The parent becomes a high-level synthesis page; children contain implementation.
- Rebuild `part_of_component` / `has_child_component` edges.

Flask full run (commit `49943b5`):

```text
final: 34 pages
  L0 implementation: 16
  L1 module:         10
  L2 component:       7
  L3 overview:        1
all page contexts <= 140k docs budget
no context truncation in final traces
```

The decomposed `flask-application-core` (four child pages) and
`test-suite -> main-tests -> test_helpers` both keep parent and child pages.
The parent is synthesized from child docs and high-level relationship evidence.

### LLM Clustering (Stage 2)

The default `--cluster-mode toy` never calls an API. Enable LLM clustering
explicitly after checking the key:

```bash
.venv-ours/bin/repodoc build \
  --repo /path/to/small-repo \
  --state-dir /tmp/repodoc-llm-state \
  --cluster-mode llm \
  --max-cluster-nodes 200 \
  --force
```

Guards:

- `--max-cluster-nodes` limits **file nodes**, not all symbols/APIs.
- Prompts above `max_chars` are rejected without a request.
- `llm` mode sends one clustering request per build.
- `hierarchical` mode sends bounded chunks and a merge request.
- The LLM only assigns components to files; symbols/APIs inherit assignments and
  unassigned nodes are reported.
- Failed responses are not retried automatically, preventing unbounded API use.

Verified small cases:

- toy repo: 11 code nodes / 2 files -> 2 components;
- MarkupSafe: 191 code nodes / 12 files -> 6 components, all 191 assigned;
- usage is stored in `component_tree.json.cluster_meta.usage`; the trace is
  `components/llm_cluster_trace.json`.

### Hierarchical Clustering

Large repositories are not sent in one clustering request:

```bash
.venv-ours/bin/repodoc build \
  --repo /path/to/repo \
  --state-dir /tmp/state-hierarchical \
  --cluster-mode hierarchical \
  --cluster-chunk-files 80 \
  --cluster-chunk-chars 80000 \
  --cluster-merge-mode llm \
  --max-component-nodes 300 \
  --force
```

How it works:

1. Group files by top-level directory.
2. Greedily pack groups into chunks under file and prompt limits.
3. Run one LLM clustering request per chunk.
4. `--cluster-merge-mode llm` sends chunk component summaries and cross-component
   relationships to one merge call and produces 2–8 top-level components;
   `directory` mode merges by common path prefix.
5. Missing files are reported in `chunk_unassigned`.

GraphRAG (439 files):

```text
chunks:                 7
chunk files:            [80, 80, 80, 80, 14, 79, 26]
chunk prompt cap:       80,000 chars
top-level components:   8
pages:                  39
cluster cost:           $0.02913
```

Comparison:

| Variant | Pages | Qwen | GLM | Mean |
|---|---:|---:|---:|---:|
| single-call | 39 | 84.82 | 75.60 | 80.21 |
| hierarchical + directory | 42 | 80.33 | 78.82 | 79.58 |
| hierarchical + LLM merge | 39 | **85.25** | **79.89** | **82.57** |

LLM merge beat single-call under both judges in this run (Qwen +0.43, GLM +4.29)
while keeping prompts bounded. Repeats still show variance: page counts were
`[39, 57, 41]`, means `[82.57, 81.10, 80.07]`. The current conclusion is that
**chunk + LLM merge is better on average but not stable enough to treat one
39-page run as a guarantee**.

### LLM Hierarchical Documents (Stage 3)

Reuse the existing graph/component_tree/doc_tree and rewrite selected pages
without reclustering. Generation can be batched; `--max-pages` limits requests:

```bash
# 1) Copy baseline state to preserve deterministic docs
cp -a /path/to/state /path/to/state-llm-docs

# 2) Preview page prompt/context sizes without API calls
.venv-ours/bin/repodoc llm-docs \
  --state-dir /path/to/state-llm-docs \
  --pages core-markup-api,overview \
  --max-pages 2 --dry-run

# 3) Generate the pages (low-level first, overview last; budgets from model limits)
.venv-ours/bin/repodoc llm-docs \
  --state-dir /path/to/state-llm-docs \
  --pages core-markup-api,overview \
  --max-pages 2
```

Properties:

- At most one request per page; `--max-pages` defaults to 4.
- Context/output budgets come from the model catalog's `limit.context` and
  `limit.output`.
- Low-level pages use real API/symbol/source-span context.
- L2 component pages use component boundaries, file responsibilities,
  cross-file/cross-component import+call aggregates, representative calls, key
  entry points, and source excerpts.
- Parent/overview pages use child-doc summaries, top-level component relations,
  entry points, and README content.
- High-level prompts require
  `Purpose -> Architecture (flowchart) -> Key flows (sequenceDiagram) -> Design decisions -> Module map`
  and at least one Mermaid diagram. The overview requires both flowchart and
  sequenceDiagram.
- High-level pages must not dump complete API lists or copy child docs.
- Docs are written to markdown, `doc_tree.json`, `pages.jsonl`, `sections.jsonl`,
  and graph `doc_page`/`doc_section` nodes.
- Traces go to `docs/llm_traces/<page>.json`.
- `--reuse-traces` reapplies saved responses without API calls (useful for
  correcting cover mappings).

After the architecture upgrade, all 10 high-level Flask pages include Mermaid;
L2 top-level pages have five sections, and the overview has five sections and
three diagrams. Verified MarkupSafe selected-page results:

- `core-markup-api`: 4 sections, usage 3985/6957, list cost $0.00477;
- `native-escaping-backend`: 3 sections, usage 478/1347, list cost $0.00088;
- `test-suite`: 4 sections, usage 2055/3567, list cost $0.00245;
- `packaging-build`: 4 sections, usage 1102/2041, list cost $0.00135;
- `performance-benchmarks`: 3 sections, usage 351/717, list cost $0.00045;
- `documentation-config`: 3 sections, usage 345/879, list cost $0.00058;
- `overview`: 4 sections, regenerated after children, usage 3867/3747, list cost $0.00283.

All seven MarkupSafe pages are now LLM-generated; docs, doc_tree, doc sections,
and cover/citation edges are written back.

Reasoning models may spend the entire `max_tokens` budget on reasoning. The
first `test-suite` attempt with 12k context / 8192 tokens returned empty content;
reducing to 6k context succeeded. Failed responses remain in
`docs/llm_traces/<page>.json` for diagnosis.

### Incremental Update (Stage 4 MVP)

```bash
# Build the base commit once
.venv-ours/bin/repodoc build --repo /path/to/base --repo-name markupsafe \
  --state-dir /path/to/state-base --force

# Switch to the target commit and update in place
.venv-ours/bin/repodoc update --repo /path/to/update --state-dir /path/to/state-base
```

Implemented:

- code-layer graph delta (added/removed/modified nodes and edges);
- stable file-path component membership repair;
- affected-page discovery from `cites`, component membership, and parent pages;
- regeneration of only affected pages (including overview ancestors);
- graph/doc/manifest version write-back;
- `change_set` application records and `regenerated_pages`.

MarkupSafe `3.0.0 -> 3.0.1` verification:

```text
changed files: 4
modified code nodes: 55
affected pages: src_markupsafe.md, overview.md
unassigned new nodes: 0
incremental docs vs full rebuild: content identical
code graph version: identical
```

Added/deleted files:

- New files in existing directories join existing components.
- New directories/file groups create local components and doc pages.
- Deleting the last member removes the component, page, and markdown.
- `--direct-only` updates only pages that directly cite changed code; only
  structural additions/deletions propagate to parents and overview.
- `--llm-plan` asks an LLM to choose `noop/patch/regenerate` per deterministic
  candidate page. A guard prevents local changes from rewriting parents/overview
  without structural or true cross-component signals.
- `--llm-update` is the legacy triage path (`noop/regenerate` only).
- `--llm-patch` supports section-level patches: validate section ID and
  `old_content_hash`, apply local replacements, and fall back to full-page
  regeneration on failure.
- Content patching remains experimental: MarkupSafe real-API trials produced
  empty patches, non-JSON, and timeouts, all of which fell back. Prompt and model
  tuning are still needed.

Limitation: new components use local directory grouping, not LLM reclustering;
large structural changes should still use a full rebuild.

### LLM Update Triage

```bash
.venv-ours/bin/repodoc update \
  --repo /path/to/update \
  --state-dir /path/to/state-base \
  --llm-update
```

One request reviews the change set, graph delta, and affected pages and returns
`regenerate|noop|patch|delete` per page. On MarkupSafe `3.0.0 -> 3.0.1` the model
returned:

```json
{"overview.md": "regenerate", "src_markupsafe.md": "regenerate"}
```

Usage was 339/512, list cost $0.00036; the final docs matched a full rebuild.

### Section-Level LLM Content Patch

```bash
.venv-ours/bin/repodoc update \
  --repo /path/to/update \
  --state-dir /path/to/state-base \
  --llm-patch
```

Only affected `doc_section` records are sent to the LLM, which returns:

```json
{"patches":[{"section_id":"...","old_content_hash":"...","new_content":"..."}]}
```

The program validates section IDs and hashes and replaces only approved sections;
failure falls back to full-page regeneration. Real MarkupSafe trials have not yet
succeeded reliably (empty patches, non-JSON, timeouts), but fake-client tests
cover successful patches, hash validation, fallback, and graph write-back.

## State Layout

```text
<state-dir>/
├── manifest.json
├── graph/{graph.json,nodes.jsonl,edges.jsonl}
├── components/component_tree.json
├── docs/{doc_tree.json,pages.jsonl,sections.jsonl,markdown/*.md}
└── changes/<commit>.json
```

## Next Steps

See [`docs/language-extension.md`](../../docs/language-extension.md).

1. Languages: analyzers cover Python, JavaScript, TypeScript, Rust, Go, Java,
   PHP, C, C++, and C#. Run one documentation vertical slice per language:
   - start with Chart.js for JS;
   - then Java/C#, C/C++, and Rust/Go/PHP;
   - use one medium-sized repository per language before expanding.
2. Validate and repair Mermaid output and include it in CodeWikiBench quality reporting.
3. Continue hardening update graph delta, impact sets, reclustering, and high-level patches.
4. Extend `adapters/ours.sh` to more benchmark repositories (GraphRAG already has a full pilot).
5. Extend CodeWikiBench-Update experiments.

## CodeWikiBench Architecture Pilot

The harness can export an architecture subset from a frozen CodeWikiBench rubric:

```bash
.venv/bin/python -m harness architecture-rubrics ...
```

It can also recompute architecture scores and Mermaid/link metrics from existing
judge results. Full-granularity GraphRAG at `a398cc38...` with the same judge and
all 56 leaves:

```text
ours      78.97 (39 pages, 19 Mermaid)
CodeWiki  65.53 (37 pages, 149 Mermaid)
```

This is not a page-compressed pilot: ours has 8 top-level components, 30 child
pages, and an overview. See `runs/architecture-graphrag/README.md`.

### GraphRAG Update Pilot

On the small diff from base `a398cc38...` to target `6c66b7c...`, only 5 of 35
pages were affected and 30 remained byte-identical. Update documentation cost was
`$0.04354`; full rebuild was `$0.26976`.

Two judges (`qwen3.8-flash`, `glm-5.3-flash`), 56-leaf scores:

```text
                 qwen   glm    mean
base             74.19  74.25  74.22
update-regen     75.10  67.25  71.17
update-patch     69.48  72.76  71.12
update-direct    75.53  78.05  76.79
full             78.05  65.82  71.94
```

`--direct-only` updates only pages that directly cite changed code and does not
rewrite ancestors or the overview. It is the only update strategy that beat base
under both judges (+1.34 / +3.80), regenerating one page at `$0.01082`.
`--llm-plan` implements candidate planning with a deterministic guard; on the
local diff it converges to direct-only, while structural changes still allow
parent/overview expansion. See `runs/update-graphrag/README.md`.

Do not modify `baselines/` or `benchmarks/`; all integration happens through
adapters and configuration.
