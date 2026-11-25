# Data

Experiment inputs and frozen assets live here. Everything except this README is
ignored by `.gitignore`.

```text
data/
├── repos/<repo>/                         # blobless git clone cache
├── checkouts/<repo>/<ref>-<commit12>/    # pinned worktree
├── reference/<repo>/
│   ├── metadata.json                     # dataset metadata (repo_url/commit)
│   └── parsed/
│       ├── docs_tree.json
│       └── structured_docs.json
├── rubrics/<repo>/
│   ├── combined_rubrics.json             # frozen rubric shared by systems
│   └── rubrics.log
├── locks/repos.json                      # resolved commits/URLs
├── external/<system>/<repo>/docs/        # hosted/manual markdown
└── logs/                                 # clone / worktree logs
```

## Fetching Data

```bash
# Install optional dependencies (datasets/huggingface_hub)
make setup-data

# Download reference docs + rubrics + commit, then create a worktree
make prepare REPO=OpenHands

# Download the seven core repositories
make prepare-core SPLIT=core
```

Equivalent CLI:

```bash
.venv/bin/python -m harness prepare --repo OpenHands --from-hf
.venv/bin/python -m harness prepare --split core --from-hf
.venv/bin/python -m harness prepare --all --from-hf

# Download only references/rubrics/commits, without cloning (external docs)
.venv/bin/python -m harness prepare --repo OpenHands --from-hf --skip-checkouts
```

`prepare` does:

1. Read a row from `anhnh2002/codewikibench`; `REPODOC_DATASET_REVISION` pins the
   HuggingFace revision.
2. Write `structured_docs` / `docs_tree` to `reference/<repo>/parsed/`.
3. Write rubrics to `rubrics/<repo>/combined_rubrics.json`.
4. Write `commit_id` to `locks/repos.json`.
5. `git clone --filter=blob:none --no-checkout` into `repos/<repo>`.
6. Create `checkouts/<repo>/base-<commit12>`.

## Offline / Manual Placement

If HuggingFace is unavailable, place files manually:

- `reference/<repo>/parsed/` must contain `docs_tree.json` and
  `structured_docs.json`;
- `rubrics/<repo>/combined_rubrics.json` may be `{"rubrics": [...]}` or a
  top-level rubric list;
- example `locks/repos.json`:

```json
{
  "OpenHands": {
    "url": "https://github.com/All-Hands-AI/OpenHands",
    "commit": "<40-char sha>",
    "source": "manual"
  }
}
```

Then run `python -m harness prepare --repo OpenHands` without `--from-hf` to
create the checkout.

## Fairness Constraints

- **Freeze rubrics before running systems.** Never regenerate a rubric for one system.
- Reference docs are only for rubric generation; judges must not see them.
- Changing `combined_rubrics.json` invalidates the `sha256` in `meta.json`; create
  a new experiment instead of overwriting old results.
- `data/` is not in git. For long-term preservation, copy final rubrics and
  `locks/repos.json` to artifact storage and publish checksums in the paper/README.
