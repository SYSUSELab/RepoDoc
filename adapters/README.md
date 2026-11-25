# Adapters

Adapters expose different documentation systems through a common interface that
the harness can call. System implementations live under `baselines/` or
`methods/`; adapters only translate arguments, prepare environments, and invoke
the system.

## Contract

```bash
bash adapters/<system>.sh \
  --repo   /abs/path/to/checkout \
  --out    /abs/path/to/docs \
  --workdir /abs/path/to/work \
  --mode   full|update \
  [-- <system-specific args>]
```

Requirements:

1. Write **flat** `.md` files directly under `--out`; subdirectories are not
   allowed because CodeWikiBench currently parses only the docs root.
2. Return 0 on success and non-zero on failure, with errors on stderr when possible.
3. `--mode update` means base docs already exist in `--out` and should be updated
   in place. Systems without update support set `supports_update: false` in
   `configs/systems.yaml`; the harness will reject them.
4. `--repo` is read-only input. Write caches, logs, and temporary files to
   `--workdir`.
5. Write CodeWiki-compatible generation metadata to `--out/metadata.json` and a
   normalized copy to `--workdir/system_metadata.json`. The harness prefers the
   latter. Both should include tokens, cost, and optional LLM call time:

```json
{
  "generation_info": {
    "timestamp": "2026-11-23T12:00:00+00:00",
    "provider": "opencode-go",
    "main_model": "deepseek-v4.1-flash",
    "generator_version": "0.1.0",
    "language": "python"
  },
  "usage": {
    "schema_version": 2,
    "currency": "USD",
    "cost_status": "complete",
    "n_calls": 12,
    "totals": {
      "input_tokens": 1000,
      "output_tokens": 2000,
      "reasoning_tokens": 800,
      "total_tokens": 3000,
      "requests": 12,
      "cost_usd": 0.12
    },
    "calls": [
      {"phase": "docs", "seconds": 1.2, "cost_usd": 0.01}
    ]
  }
}
```

Canonical `system_metadata.json` fields:

```json
{
  "model": "deepseek-v4.1-flash",
  "generator_version": "0.1.0",
  "tokens": {"input": 1000, "output": 2000, "reasoning": 800, "total": 3000},
  "cost_usd": 0.12,
  "cost_status": "complete",
  "llm_duration_sec": 12.3,
  "n_calls": 12,
  "by_phase": {"docs": {"calls": 12, "cost_usd": 0.12}},
  "notes": "anything useful for the run table"
}
```

When usage is unavailable, set `cost_status: "unknown"` /
`usage_status: "unavailable"`; **never write 0**. The harness records wall time in
`meta.json.stages.generate` and `metrics.json.generation.duration_sec`; adapters
must not duplicate timing.

## Environment

The harness sets:

| Variable | Value |
|---|---|
| `REPODOC_REPO` | Absolute checkout path |
| `REPODOC_OUT` | Absolute docs path |
| `REPODOC_WORKDIR` | Absolute work directory |
| `REPODOC_MODE` | `full` / `update` |
| `REPODOC_REPO_NAME` | Repository name from config |
| `REPODOC_EXPERIMENT` | Experiment name |
| `REPODOC_SYSTEM` | System name |

`env` in `configs/systems.yaml` provides defaults; existing process environment
variables win (for example, a temporary `CODEWIKI_BIN` override).

## Existing Adapters

- `codewiki.sh`: `cd` into the repository and call
  `codewiki generate --output <out> --flat`; add `--update` for update mode; save
  a masked `codewiki config show --json` snapshot. Set `CODEWIKI_HOME` to isolate
  `~/.codewiki` across model configurations.
- `ours.sh`: call `repodoc build` + `repodoc llm-docs`, then `repodoc stats` to
  write `docs/metadata.json`. Token/cost accounting includes clustering and all
  LLM doc calls, including failed/retried attempts.
- `_template.sh`: template for new systems.

## Common Pitfalls

- Writing into subdirectories of `--out` breaks parsing; output must be flat.
- Do not depend on the current working directory; use absolute `--repo`/`--out`.
- Never put API keys in logs or `system_metadata.json`.
- If a system emits HTML/PDF, convert it to markdown in the adapter or extend the
  benchmark parser.
