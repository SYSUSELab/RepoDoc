# Reproducibility and Records

## 1. Pinned Versions

Current upstream versions:

| Component | Version/commit |
|---|---|
| CodeWiki | `7fcf9354f70a5468bc739d849055f8cbab9d46aa` (2.0.0) |
| CodeWikiBench | `00f56b656b9fb8f23b87aecbfdcf19510e3e4f36` |

Every run records:

- `meta.json.versions.system_commit`: baseline source commit;
- `meta.json.versions.adapter_sha256`: adapter file hash;
- `meta.json.versions.benchmark_commit`: CodeWikiBench commit;
- `meta.json.versions.repo_commit`: documented repository commit;
- `meta.json.config.rubrics.sha256`: frozen rubric hash;
- `meta.json.config.judge`: judge model and parameters;
- `metrics.json.generation`: tokens, cost, `cost_status`, wall time, LLM time;
- `work/system_metadata.json`: normalized adapter usage.

Papers/reports should also record:

- generator model, provider, and date (API models may change silently); ours
  defaults to `OPENCODE_GO_MODEL=deepseek-v4.1-flash`, with
  `OPENCODE_GO_SESSION` and user agent recorded;
- judge model, provider, temperature, and tool settings;
- key CLI flags (`--flat`, `--no-artifacts`, `--with-prose`, `--focus`, ...);
- repeat count and random seed for stochastic methods.

## 2. Environment

```bash
# Version information
.venv/bin/python --version
uv pip freeze --python .venv/bin/python > runs/<experiment>/pip-freeze.txt
uv pip freeze --python .venv-ours/bin/python > runs/<experiment>/pip-freeze-ours.txt
uv pip freeze --python .venv-codewiki/bin/python > runs/<experiment>/pip-freeze-codewiki.txt
.venv-codewiki/bin/codewiki --version
git -C baselines/codewiki rev-parse HEAD
git -C benchmarks/codewikibench rev-parse HEAD
```

Archive `pip-freeze.txt`, `nvidia-smi` (for local models), `uname -a`, and a
snapshot of `configs/` with the experiment artifacts.

## 3. Data

- `data/locks/repos.json` records the actual commit/URL per repository;
- `data/reference/<repo>/parsed/` is the only input for rubric generation;
- `data/rubrics/<repo>/combined_rubrics.json` must not change after freezing;
- if data is downloaded again from HuggingFace, record the dataset revision/date;
  datasets can change.

Suggested archive:

```text
runs/<experiment>/
├── experiment.json           # system/repo/language/token/cost index
├── summary.md / .csv / .json
├── configs/                  # configuration snapshot
├── locks/repos.json
├── rubrics/                  # only the combined_rubrics.json files used
└── pip-freeze.txt
```

Do not keep only final JSON; `logs/` contains judge errors, retries, truncation,
and other information needed to explain failures.

## 4. Score Accounting

`harness/metrics.py` recomputes scores from the scored rubric tree:

```text
score = Σ(top-level score × weight) / Σ(weight)
      = Σ(leaf score × leaf weight) / Σ(leaf weight)   # bottom-up equivalent
```

- `score`: 0–1;
- `score_100`: 0–100, used in tables;
- parent scores are weighted child averages, so the total equals the weighted
  average of all leaves;
- `num_errors`: leaves with `evaluation.error=true` or `[evaluation error]` in
  reasoning;
- failed leaves in benchmark reports use `score <= max_score`.

Known limitation: CodeWikiBench API judges do not record token usage, so
`judge.tokens` may be 0. Estimate judge cost from provider billing or call counts.

## 5. Randomness and Repeats

- If generation temperature > 0 or sampling is enabled, repeat each
  `(system, repo)` at least three times.
- Use independent experiment names (`core-r1`, `core-r2`, `core-r3`); do not
  overwrite.
- Main tables report mean ± std and retain per-repo results.
- Judges are stochastic too. Fix judge configuration within an experiment and use
  the same rubric across repeats; document any judge repeats separately.

## 6. Minimal Reproduction

```bash
make setup
cp .env.example .env && $EDITOR .env
make doctor

make prepare REPO=graphrag
make run REPO=graphrag SYSTEM=codewiki EXPERIMENT=python-graphrag-v1
make run REPO=graphrag SYSTEM=ours EXPERIMENT=python-graphrag-v1
make summary EXPERIMENT=python-graphrag-v1

# Batch
make batch SUITE=core EXPERIMENT=core-v1

# Web UI
make web
```

To validate CodeWiki without an LLM:

```bash
make demo
```

It uses CodeWiki's real dependency analysis and `DocumentationGenerator`, injects
a fake backend, covers whole-repository and clustered modes, and runs
`harness -> adapter -> docs` through a fake CLI. Artifacts are under
`runs/smoke-codewiki/` and `runs/smoke-codewiki-harness/`.

To validate the benchmark without an LLM:

```bash
make demo-report
```
