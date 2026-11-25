# Baselines

Third-party documentation systems live here, one directory per system. **Do not
modify upstream source for experiments.** Shared input/output adaptation belongs
in `adapters/`; system registration belongs in `configs/systems.yaml`.

## Included

| System | Directory | Upstream commit | Notes |
|---|---|---|---|
| CodeWiki 2.0 | `baselines/codewiki` | `7fcf9354f70a5468bc739d849055f8cbab9d46aa` | dependency graph + hierarchical agents; supports `--update` |
| DeepWiki | not run locally | - | Hosted service; exported docs can be evaluated as an external system via `harness evaluate` |

CodeWiki details:

- Install: `make setup-codewiki` (dedicated `.venv-codewiki`)
- Config: `.venv-codewiki/bin/codewiki config set ...`, stored in `~/.codewiki`
- Adapter: `adapters/codewiki.sh`
- Output: flat markdown + `module_tree.json` + `metadata.json`; usage is merged
  into `metrics.json`
- Offline smoke: `make smoke-codewiki` (real dependency analysis +
  `DocumentationGenerator`, fake backend, no `~/.codewiki`, no API calls)

## Shortest Path for a New Baseline

```bash
git clone <url> baselines/<name>
cp adapters/_template.sh adapters/<name>.sh
# edit adapters/<name>.sh
# add <name> to configs/systems.yaml
.venv/bin/python -m harness doctor
.venv/bin/python -m harness run --system <name> --repo OpenHands --experiment smoke
```

Record license, version, run commands, and model configuration in
`baselines/<name>/README.md`. For an upstream clone you may add `EXPERIMENT.md`;
do not edit the upstream README.

## External / Hosted Systems

For systems such as DeepWiki:

1. Export markdown with CodeWikiBench's `crawl_deepwiki_docs` or an official export.
2. Put markdown under `data/external/<system>/<repo>/docs/`.
3. Treat it as an existing-docs run before `harness evaluate`, or call
   `python -m codewikibench parse` + `judge` directly.
4. Label the source (crawl date, site version) in result tables; do not conflate
   it with locally reproducible systems.
