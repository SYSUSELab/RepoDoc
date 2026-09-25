# RepoDoc

RepoDoc: Automated Documentation Generation and Maintenance via Repository Knowledge Graph

## Usage

```bash
repodoc analyze <repo_path>    # Build knowledge graph: dependency graph + module tree
                               #   --no-cluster      skip LLM clustering (saves tokens)
                               #   --cluster-strategy llm|heuristic (ablation)
repodoc docs <repo_path>       # Generate documentation (auto-runs analysis if needed)
repodoc generate <repo_path>   # Full pipeline: analyze -> cluster -> docs
repodoc update <repo_path>     # Incrementally update docs for changed files
                               #   --propagate sig|all   caller propagation scope
                               #   --max-hops N          max propagation depth (0 disables)
                               #   --force               overwrite manually edited docs
repodoc config show            # Show current configuration
```

`repodoc update` works in four stages:

1. **Change analysis** — git diff (old commit vs working tree, incl. deleted and
   untracked files) is re-analyzed per file; changes are classified by severity.
2. **Graph mutation** — the knowledge graph is updated in place: new components
   are added (with module placement), modified components overwritten (with
   calls-edge diffs), removed components deleted with all incident edges.
3. **Layered impact** — changes propagate over calls edges to affected callers
   (bounded by `--max-hops`, gated by severity), then to module docs and the
   README overview.
4. **Doc update** — existing docs are updated minimally (old doc embedded as
   the format anchor; only affected sections change). Docs you edited by hand
   are skipped with a warning (`--force` overwrites them); removed components
   have their docs cleaned up; renamed files migrate their docs.

## Configuration

Create `.env`:

```
LLM_BASE_URL=<endpoint>
LLM_API_KEY=<key>
MAIN_MODEL=<model>
```
