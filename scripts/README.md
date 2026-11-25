# Scripts

## `smoke_codewiki_fake.py`

Offline end-to-end smoke test for CodeWiki's core generation path. It needs no LLM
API and does not read `~/.codewiki`.

```bash
make smoke-codewiki
# or
.venv-codewiki/bin/python scripts/smoke_codewiki_fake.py
```

It uses CodeWiki's real:

- tree-sitter dependency analysis / dependency graph;
- `DocumentationGenerator` orchestration;
- `module_tree.json`, `first_module_tree.json`, and `metadata.json` output;
- whole-repository and clustered generation paths.

Only the LLM backend is replaced with `FakeBackend`. Artifacts:

```text
runs/smoke-codewiki/
├── whole/          # whole-repository docs mode
├── cluster/        # forced clustering, multiple modules + overview
└── smoke-result.json
```

Checks:

- `overview.md` is non-empty;
- whole mode writes `module_tree.json == {}`;
- cluster mode has at least two modules, each with its own `.md`;
- the dependency graph exists;
- fake-backend usage is written to `metadata.json`.

## `fake_codewiki_cli.py` / `fake_codewiki_cli.sh`

A fake CodeWiki CLI implementing only `--version`, `config show --json`, and
`generate`. It uses real CodeWiki dependency analysis and
`DocumentationGenerator` with a fake backend. It validates:

```text
harness -> adapters/codewiki.sh -> fake codewiki CLI -> docs
```

## `smoke_harness_codewiki.sh`

Runs the harness generation path with the toy repo and fake CLI; no benchmark,
rubric, or judge calls:

```bash
make smoke-codewiki-harness
```

Artifacts go to `runs/smoke-codewiki-harness/`. `make smoke-codewiki` runs both
the core smoke and this integration smoke.

## `smoke_ours_toy.sh`

Smoke test for the toy `methods/ours` knowledge-graph + hierarchical-document
pipeline without LLM calls:

```bash
make smoke-ours
```

It creates a toy Python repository under `runs/smoke-ours/` and runs:

```text
tree-sitter code graph -> toy component tree -> doc tree -> overview/component markdown
```

Then it runs `repodoc status/inspect` and checks graph/component/doc/manifest
artifacts.

These scripts validate CodeWiki, ours, and integration wiring only. Real APIs,
benchmark rubrics, and judges are not included.
