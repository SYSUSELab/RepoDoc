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
                               #   --dry-run             preview scope/cost, change nothing
repodoc config show            # Show current configuration
```

`analyze`, `docs` and `generate` share the analysis/documentation scope
options:

```bash
  -I, --include GLOB     only analyze matching files (repeatable, e.g. 'src/*')
  -X, --exclude GLOB     skip matching files/dirs (repeatable)
  --max-file-size N      skip source files larger than N bytes
  --doc-types T1,T2      component types to document (default: automatic)
  --max-docs N           cap the number of documented components (ranked by
                         dependents, then type; 0 = no cap)
```

When the knowledge graph already exists, explicit `--max-docs`/`--doc-types`
re-cluster it from the in-graph components (rebuilding the module tree)
instead of being silently ignored.

`repodoc update` works in four stages:

1. **Change analysis** — the working tree is compared against a snapshot
   recorded by the last successful analyze/generate/update (per-file hashes
   plus the text of dirty files); git diff remains the fallback for explicit
   `--base-commit`/file lists and for rename detection. Re-running on an
   unchanged tree is a no-op even with uncommitted changes, and reverting a
   change syncs the graph back. Changes are re-analyzed per file and
   classified by severity
   and keyed by exact component id (methods included). Files outside the
   generation include/ignore rules are skipped, and an unreadable or empty file
   is never mistaken for a deletion. Reference resolution is scope- and
   import-aware: a callee resolves in the caller's own file first, then
   through the caller file's import map (aliases and wildcard namespaces
   collected from Python/JS/TS/Go/Rust/Java/C#/PHP/C includes; same-named
   components in different modules stay distinct), then by a globally unique
   name; ambiguous references are dropped instead of guessed and kept on the
   node as unresolved references. After the batch, unresolved references that
   the new definitions made resolvable are wired into edges automatically, so
   the call graph can grow as well as shrink.
2. **Graph mutation** — the knowledge graph is updated in place: new components
   are added (with module placement), modified components overwritten (with
   typed edge diffs), removed components deleted with all incident edges.
   Relations keep their semantics: `calls`, `implements` (extends/interface),
   `uses` (field/property/parameter/type references), `instantiates`
   (constructor calls) and `imports`. Go struct→interface conformance (which
   the language expresses implicitly) is recomputed structurally on every
   run, and a brand-new module that collected 13+ members is clustered once
   into semantic sub-modules — confined to the new components, so nothing
   existing moves and no doc is migrated.
3. **Layered impact** — changes propagate over calls/implements/uses/
   instantiates edges to affected callers, subclasses and type users (bounded
   by `--max-hops`, gated by severity), then to module docs and the README
   overview. Propagated caller docs are capped by a default budget of 60
   (ranked by distance, then fan-in; `REPODOC_MAX_IMPACT_DOCS=0` disables the
   cap); skipped callers are reported instead of being silently dropped.
4. **Doc update** — existing docs are updated minimally (old doc embedded as
   the format anchor; only affected sections change). Docs you edited by hand
   are skipped with a warning (`--force` overwrites them); removed components
   have their docs cleaned up; renamed files migrate their docs.

Updates are crash-safe: the knowledge graph is written atomically, a lock
prevents concurrent runs, and docs that fail (or that a crash left unfinished)
are journaled in `{data}/{project}_pending_updates.json`. The git watermark in
the metadata only advances once nothing is pending, so the next
`repodoc update` resumes exactly those docs.

Generation is also guarded: empty or truncated LLM outputs (unbalanced code
fences) are rejected instead of written as zero-byte docs, all doc writes are
confined to the docs directory, the README's module diagram is generated from
the knowledge graph and enforced verbatim, doc snapshots in the graph store a
content hash instead of the full text (manual-edit detection works on the
hash), and generated stats land in `metadata.doc_generation`. Link validation resolves targets relative to the
linking file, never rewrites fenced code blocks, and never edits Mermaid
diagrams (invalid ones are reported only). Dependency/build trees such as
`node_modules`, `vendor`, `dist` and `target` are excluded, and oversized
source files are skipped.

Prompting and clustering are reproducible and budget-aware: instructions go
in the system role (with an explicit untrusted-data guard), each layer
receives structural summaries (title, opening paragraph, section outline) of
its children's docs instead of raw character cuts, and clustering responses
are cached by prompt + model in `{data}/clustering_cache.json` so reruns and
ablations reuse the same module tree. The clustering prompt itself carries a
semantic signal per component (type, docstring first line, parameters), falling
back to name-only lists when the budget is exceeded; both strategies exclude
private components. Cluster quality stats (module count, depth, empty modules,
largest module) are printed after clustering.

## Configuration

Create `.env`:

```
LLM_BASE_URL=<endpoint>
LLM_API_KEY=<key>
MAIN_MODEL=<model>
# optional: cap propagated caller docs per update (default 60, 0 = unlimited)
REPODOC_MAX_IMPACT_DOCS=60
```
