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
repodoc config show            # Show current configuration
```

## Configuration

Create `.env`:

```
LLM_BASE_URL=<endpoint>
LLM_API_KEY=<key>
MAIN_MODEL=<model>
```
