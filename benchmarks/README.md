# Benchmarks

## CodeWikiBench

`benchmarks/codewikibench/` is a complete clone of upstream CodeWikiBench
(commit `00f56b656b9fb8f23b87aecbfdcf19510e3e4f36`). Do not implement experiment
logic here.

- Paper: <https://arxiv.org/abs/2510.24428>
- Dataset: <https://huggingface.co/datasets/anhnh2002/codewikibench>
- Upstream README: `benchmarks/codewikibench/README.md`

### Role in This Project

| Stage | Command | Output |
|---|---|---|
| parse | `codewikibench parse` | `docs_tree.json` + `structured_docs.json` |
| rubrics | `codewikibench rubrics` | frozen `combined_rubrics.json` per repo |
| judge | `codewikibench judge` | per-judge results + `combined_results.json` |
| report | `codewikibench report` | missed leaves, topics, Markdown report |

The harness wraps these commands with unified paths and logs without modifying
the upstream package.

> This branch has no `codewikibench/__main__.py`, so `python -m codewikibench ...`
> fails. Use the installed `codewikibench` console script. The harness invokes
> `python -c "from codewikibench.cli import main; ..."` instead of relying on PATH.

If judge/rubric models use `caw:<provider>:<model>`, install the optional
dependency:

```bash
make setup-bench-caw
```

### Data Conventions

- Metadata, `docs_tree`, `structured_docs`, and rubrics for all 22 repositories
  are downloaded from HuggingFace into `data/reference/`, `data/rubrics/`, and
  `data/locks/`; **large files are not committed**.
- `configs/repos.yaml` stores repository name, URL, language, and split only. The
  real commit comes from `data/locks/repos.json`, written by
  `harness prepare --from-hf`.
- The seven repositories used in the paper tables are marked `split: core`.

### Important Limitations

1. **Flat docs only**: the upstream parser reads `.md` files at the docs root.
   Generators with subdirectories need adapter flattening or a parser extension.
2. **Without `module_tree.json`**, the parser expects DeepWiki-style `N-Title`
   first lines. The harness checks whether all pages follow that format; if so it
   preserves numbering without synthesizing a module tree. Otherwise it builds a
   flat `module_tree.json` from filenames so any markdown generator can be parsed.
   This is evaluation normalization, not special treatment.
3. **API judge token counts are 0**: upstream `judge.py` does not backfill token
   usage for non-caw judges, so `judge.tokens` may be 0. Estimate judge cost from
   call volume separately.
4. Scores are weighted binary leaf coverage; `metrics.json` recomputes totals
   from the scored tree to avoid inconsistency with `combination_metadata`.

### Quick Self-Check (No LLM)

```bash
.venv/bin/python -c 'import sys; from codewikibench.cli import main; sys.exit(main())' report \
  --results benchmarks/codewikibench/examples/OpenHands/codewiki/evaluation_results/combined_evaluation_results.json \
  --format summary
```
