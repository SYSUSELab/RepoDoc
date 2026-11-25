# Runs

Experiment output directory, ignored by `.gitignore`. Every experiment that
enters a comparison must live here with a canonical experiment name and the
`systems/<system>/<repo>/` layout.

## Experiment Naming

Format:

```text
<language>-<repo>-<scope>-v<N>
```

- Lowercase; only letters, digits, `.`, `-`, and `_`.
- `language`: `python`, `javascript`, `typescript`, `java`, `csharp`, `cpp`, `c`, ...
- `repo`: lowercase repository name (`graphrag`, `rasa`, `chartjs`, ...).
- `scope`: `full`, `update`, `cluster`, `planner`, `patch`, ...
- `v<N>`: version for the same settings; increment when rubric, judge, model, or
  key parameters change.

Examples:

```text
runs/python-rasa-full-v1/
runs/python-graphrag-update-v1/
runs/javascript-chartjs-full-v1/
runs/typescript-puppeteer-full-v1/
```

Cross-repository suites may use a stable suite name (`core-v1`), but must still
use `systems/<system>/<repo>/`. The harness rejects experiment names containing
`../` or other invalid characters.

## Layout

```text
runs/<experiment>/
├── experiment.json                  # system/repo/language/tokens/cost index
├── batch.json                       # suite and success/failure list (optional)
├── summary.md / .csv / .json        # summaries
└── systems/<system>/<repo>/
    ├── docs/                        # flat markdown
    ├── parsed/                      # CodeWikiBench parse output
    ├── results/
    │   ├── combined_results.json
    │   ├── <judge-model>.json
    │   ├── report.md
    │   ├── report.json
    │   └── report.txt
    ├── logs/{generate,parse,judge}.log
    ├── work/                        # adapter work dir, state, config snapshots
    │   └── system_metadata.json     # normalized token/cost/LLM time
    ├── meta.json                    # versions, commands, status, stage timings
    └── metrics.json                 # final metrics read by summary scripts
```

`metrics.json.generation` is the only cost-efficiency interface:

```json
{
  "model": "...",
  "duration_sec": 123.4,
  "llm_duration_sec": 98.7,
  "tokens": {
    "input": 1000,
    "output": 200,
    "reasoning": 80,
    "total": 1200,
    "requests": 12
  },
  "cost_usd": 0.12,
  "cost_status": "complete",
  "tokens_per_page": 120.0,
  "cost_per_page_usd": 0.012,
  "seconds_per_page": 12.3
}
```

- `duration_sec` is harness wall time, including adapter, retries, and writes.
- `llm_duration_sec` is adapter/system LLM time (may be `null`).
- `cost_status=complete` means every call is priced; `partial`/`unknown`/
  `unavailable` must be reported as-is, never treated as zero.
- CodeWiki and ours both read the same `usage` fields from `docs/metadata.json`.
- Judge tokens/time are in `judge.tokens` / `judge.duration_sec`.
- Ours per-run ledger is `state/usage/runs/<run_id>.json`. Use state scope for
  full runs and `--run-scope latest-run` for incremental updates; do not mix them.

## Web UI Data

Web data is separate from local experiments:

```text
web-data/
├── repos/              # GitHub clones
└── runs/<run_id>/
    ├── run.json
    ├── generate.log
    └── state/          # normal RepoDoc state
```

The Web project list scans only `web-data/runs/`; local `runs/` experiments never
appear.

## Recorded Python Costs

| Experiment | System | Pages | Tokens (in/out/total) | Cost (USD) | Wall time |
|---|---|---:|---|---:|---:|
| GraphRAG base pilot | ours | 35 | 411,315 / 334,065 / 745,380 | 0.26136 | not recorded |
| GraphRAG full pilot | ours | 57 | 541,900 / 467,080 / 1,008,980 | 0.35973 | ~66 min |
| Rasa full | ours | 188 | 2,368,628 / 1,638,668 / 4,007,296 | 1.74669 | ~131 min |
| GraphRAG canonical | ours | 57 | 541,900 / 467,080 / 1,008,980 | 0.35973 | ~66 min |
| GraphRAG demo | CodeWiki | 37 | unavailable | unavailable | unavailable |
| Rasa demo | CodeWiki | 49 | unavailable | unavailable | unavailable |

CodeWiki demo metadata contains model/version/commit/file list only; no usage or
timing. A formal comparison must regenerate CodeWiki 2.0 locally with the same
model and record `usage.calls[].seconds`.

## Historical Directories

These one-off pilots predate the naming convention. They remain for provenance
and are excluded from new `summary`/batch aggregation:

| Directory | Experiment type |
|---|---|
| `runs/rasa/` | Python / Rasa / full + judge |
| `runs/generate-graphrag/` | Python / GraphRAG / full |
| `runs/architecture-graphrag/` | Python / GraphRAG / architecture judge |
| `runs/clustering-graphrag/` | Python / GraphRAG / clustering ablation |
| `runs/update-graphrag/` | Python / GraphRAG / update + planner |
| `runs/markupsafe-generate/` | Python / MarkupSafe / full smoke |
| `runs/markupsafe-update/` | Python / MarkupSafe / update smoke |

New experiments must use `<language>-<repo>-<scope>-v<N>` and
`runs/<experiment>/systems/<system>/<repo>/`; do not add new top-level directories
of the old style.

## Architecture Artifacts

```text
runs/<experiment>/
├── architecture-rubrics.json       # architecture leaf subset from frozen rubric
├── <system>.json / .md             # harness architecture single-system report
└── judge-<model>/                  # filtered rubric results per system
```

`harness architecture` recomputes scores from existing judge results without API
calls. `scripts/architecture_judge.py` is a local CodeWikiBench judge wrapper that
uses the OpenCode Go endpoint and adds `x-opencode-session`; it does not modify the
upstream checkout.

## Offline Smoke Artifacts

`make demo` / `make smoke-codewiki` produce:

```text
runs/smoke-codewiki/{whole,cluster}/   # CodeWiki fake-backend smoke
runs/smoke-codewiki-harness/           # harness -> adapter -> fake CLI smoke
```

These directories are gitignored and need no manual cleanup.

## Conventions

- An experiment is a comparable set of runs. After changing rubric, judge, or
  generation model, use a new experiment name; do not overwrite old results.
- Start with **one medium-sized repository per language** for system comparisons.
  Python has GraphRAG/Rasa; JS/TS, Java/C#, and C/C++ each get a new experiment
  before expanding repository count.
- Do not compress full hierarchical docs into a few pages to save cost; control
  cost by reducing repositories/repeats.
- `--force` renames old docs to `docs.prev-<timestamp>` rather than deleting them.
- `metrics.json.score_100` is recomputed from the scored rubric tree; `score` is
  the 0–1 version.
- Failed generation writes no `metrics.json`, but keeps `meta.json` and logs;
  `batch` records failures in `batch.json`.
- Summary command:

```bash
.venv/bin/python -m harness summary --experiment pilot --format md
```

## Recommended Update Layout

Use three experiments:

```text
runs/python-graphrag-base-v1/              # full at base commit
runs/python-graphrag-full-v1/              # full rebuild at update commit
runs/python-graphrag-update-v1/            # incremental from base docs
```

All use `systems/<system>/graphrag/`. The incremental run points at base docs:

```bash
.venv/bin/python -m harness update \
  --system ours --repo graphrag --experiment python-graphrag-update-v1 \
  --ref update --base-docs runs/python-graphrag-base-v1/systems/ours/graphrag/docs
```

`harness update --base-docs ...` copies base docs into the target run's `docs/`
and invokes the system with `--mode update`. There is no dedicated update-quality
metric, so rerun the same rubric/judge at the update commit and report cost, time,
and changes relative to base.
