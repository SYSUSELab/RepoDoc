# Adding a System (Baseline or Ours)

Goal: integrate a system by changing only its source directory, adapter, and
registry entry; the harness main flow stays unchanged.

## 0. Choose Location and Type

- Third-party baseline: `baselines/<name>/`
- Our method/ablation: `methods/<name>/`
- Externally hosted system (e.g. DeepWiki): do not put it in `baselines/`; export
  docs to `data/external/<name>/<repo>/docs/` and evaluate with `evaluate` or the
  benchmark CLI.

Keep only system code and its own README in the directory. Do not store data, run
outputs, or API keys there.

## 1. Prepare the System

For an upstream clone:

```bash
git clone <upstream-url> baselines/<name>
git -C baselines/<name> checkout <pinned-commit>
```

Record in `baselines/<name>/README.md` or `EXPERIMENT.md`:

- upstream URL and commit;
- license;
- installation (Python version, system dependencies, Node/Java, ...);
- model/provider configuration;
- run commands and known limitations.

For our method:

```text
methods/ours/
├── README.md
├── scripts/generate.py
├── src/
└── tests/
```

## 2. Create an Adapter

```bash
cp adapters/_template.sh adapters/<name>.sh
chmod +x adapters/<name>.sh
```

Follow `adapters/README.md`:

```bash
bash adapters/<name>.sh \
  --repo /abs/checkout \
  --out /abs/run/docs \
  --workdir /abs/run/work \
  --mode full
```

Minimal example:

```bash
cd "$REPO"
if [[ "$MODE" == "update" ]]; then
  my-docgen --repo "$REPO" --output "$OUT" --update
else
  my-docgen --repo "$REPO" --output "$OUT"
fi
```

Rules:

- `--out` must contain flat markdown;
- use only `--workdir` for temporary files;
- return non-zero on failure;
- never write secrets to logs;
- write `$OUT/metadata.json` (CodeWiki-compatible) and
  `$WORKDIR/system_metadata.json` (normalized copy) with tokens, cost, and
  `cost_status`; use `unknown`/`unavailable` when data is missing, never zero.
  Format: `adapters/README.md`.

## 3. Register in `configs/systems.yaml`

```yaml
systems:
  <name>:
    kind: baseline          # baseline | proposed
    description: "..."
    upstream: "https://..."
    source: baselines/<name>
    adapter: adapters/<name>.sh
    supports_update: false
    enabled: true
    env: {}
    extra_args: []
```

For a dedicated venv or binary:

```yaml
    env:
      MYDOCGEN_BIN: "{root}/.venv-<name>/bin/mydocgen"
```

`{root}` is replaced with the repository root.

## 4. Smoke Test

Check the environment:

```bash
.venv/bin/python -m harness list
.venv/bin/python -m harness doctor
```

Run generation only and inspect flat docs:

```bash
.venv/bin/python -m harness generate \
  --system <name> --repo OpenHands --experiment smoke-<name>
find runs/smoke-<name>/systems/<name>/OpenHands/docs -maxdepth 1 -name '*.md' | head
```

If docs contain subdirectories, fix the adapter; do not change the benchmark
parser to accommodate one system.

Run the full evaluation:

```bash
.venv/bin/python -m harness run \
  --system <name> --repo OpenHands --experiment smoke-<name> --force
cat runs/smoke-<name>/systems/<name>/OpenHands/results/report.txt
```

## 5. Add to an Experiment Suite

Edit `configs/experiments.yaml`:

```yaml
suites:
  core-v1:
    systems: [codewiki, <name>]
    repos:
      - OpenHands
      - svelte
      # ...
```

Run:

```bash
.venv/bin/python -m harness batch --suite core-v1 --experiment core-v1
.venv/bin/python -m harness summary --experiment core-v1
```

## 6. Incremental Update Support

If the system can update existing docs:

1. Set `supports_update: true` in `configs/systems.yaml`.
2. In `--mode update`, update `--out` in place; do not clear it.
3. Document input requirements (metadata, caches, commit info) in the system README.
4. Use the base/full/incr layout in `docs/experiment-protocol.md`.

The CodeWiki adapter demonstrates how to map `--mode update` to `--update`.

## 7. Completion Checklist

- [ ] System code/version is traceable (commit or version).
- [ ] Adapter runs independently and follows the contract.
- [ ] Output is flat `.md` with at least `overview.md` or several pages.
- [ ] Failures return non-zero and logs are diagnosable.
- [ ] Registered in `configs/systems.yaml` with `enabled: true`.
- [ ] `OpenHands` smoke test passes and `metrics.json` contains a score.
- [ ] Model, flags, and token/cost/LLM-time accounting are recorded.
- [ ] `generation.tokens`, `cost_usd`, and `duration_sec` are readable; failed or
      unpriced calls are marked `partial`/`unknown`.
- [ ] Main-table entries are in a suite and repeated at least three times if
      stochastic.
- [ ] README documents fairness settings (reference docs, prose, etc.).
