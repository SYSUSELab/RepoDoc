#!/usr/bin/env bash
# Adapter for the RepoDoc method.
#
# Pipeline: tree-sitter graph -> LLM file clustering -> oversized-component
# splitting -> hierarchical LLM docs -> flat markdown for CodeWikiBench.
# Every network call belongs to methods/ours; this adapter only orchestrates it.

set -euo pipefail

REPO=""
OUT=""
WORKDIR=""
MODE="full"
EXTRA=()

usage() {
  cat <<'EOF'
Usage: ours.sh --repo PATH --out PATH --workdir PATH [--mode full|update] [-- ARGS...]

Environment:
  OURS_BIN                 repodoc executable (default: {root}/.venv-ours/bin/repodoc)
  OURS_CLUSTER_MODE        toy|llm (default: llm)
  OURS_MAX_CLUSTER_NODES   file nodes allowed in the clustering request (default: 200)
  OURS_MAX_PAGES           maximum pages sent to llm-docs (default: 40)
  OURS_MAX_COMPONENT_NODES component split threshold (default: 300)
  OURS_CLUSTER_CONTEXT_RATIO cluster context ratio override
  OURS_CLUSTER_OUTPUT_RATIO  cluster output ratio override
  OURS_DOCS_CONTEXT_RATIO  docs context ratio override
  OURS_DOCS_OUTPUT_RATIO   docs output ratio override
  OURS_MODEL               OpenCode Go model override
  OURS_FORCE               set to 1 to rebuild an existing state
  OURS_USAGE_SCOPE         state (default) or latest-run for update accounting
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)    REPO="${2:?--repo needs a value}"; shift 2 ;;
    --out)     OUT="${2:?--out needs a value}"; shift 2 ;;
    --workdir) WORKDIR="${2:?--workdir needs a value}"; shift 2 ;;
    --mode)    MODE="${2:?--mode needs a value}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --)        shift; EXTRA+=("$@"); break ;;
    *)         echo "ours adapter: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$REPO" || -z "$OUT" || -z "$WORKDIR" ]]; then
  echo "ours adapter: --repo, --out and --workdir are required" >&2
  usage >&2
  exit 2
fi
if [[ ! -d "$REPO" ]]; then
  echo "ours adapter: repository does not exist: $REPO" >&2
  exit 2
fi
if [[ "$MODE" != "full" ]]; then
  echo "ours adapter: only --mode full is implemented (got: $MODE)" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OURS_BIN="${OURS_BIN:-$ROOT/.venv-ours/bin/repodoc}"
CLUSTER_MODE="${OURS_CLUSTER_MODE:-llm}"
MAX_CLUSTER_NODES="${OURS_MAX_CLUSTER_NODES:-200}"
MAX_PAGES="${OURS_MAX_PAGES:-40}"
MAX_COMPONENT_NODES="${OURS_MAX_COMPONENT_NODES:-300}"
STATE="$WORKDIR/repodoc-state"

if [[ ! -x "$OURS_BIN" ]]; then
  echo "ours adapter: repodoc not found: $OURS_BIN" >&2
  exit 127
fi

mkdir -p "$OUT" "$WORKDIR"
CONFIG_ARGS=()
if [[ -n "${OURS_DOCS_CONTEXT_RATIO:-}" ]]; then
  CONFIG_ARGS+=(--docs-context-ratio "$OURS_DOCS_CONTEXT_RATIO")
fi
if [[ -n "${OURS_DOCS_OUTPUT_RATIO:-}" ]]; then
  CONFIG_ARGS+=(--docs-output-ratio "$OURS_DOCS_OUTPUT_RATIO")
fi
if [[ -n "${OURS_MODEL:-}" ]]; then
  CONFIG_ARGS+=(--llm-model "$OURS_MODEL")
fi

BUILD_ARGS=(build --repo "$REPO" --state-dir "$STATE"
  --cluster-mode "$CLUSTER_MODE"
  --max-cluster-nodes "$MAX_CLUSTER_NODES"
  --max-component-nodes "$MAX_COMPONENT_NODES")
if [[ -n "${OURS_CLUSTER_CONTEXT_RATIO:-}" ]]; then
  BUILD_ARGS+=(--cluster-context-ratio "$OURS_CLUSTER_CONTEXT_RATIO")
fi
if [[ -n "${OURS_CLUSTER_OUTPUT_RATIO:-}" ]]; then
  BUILD_ARGS+=(--cluster-output-ratio "$OURS_CLUSTER_OUTPUT_RATIO")
fi
if [[ "${OURS_FORCE:-0}" == "1" ]]; then
  BUILD_ARGS+=(--force)
fi
# Extra arguments are appended to build so callers can tune clustering/splitting.
BUILD_ARGS+=("${EXTRA[@]}")

printf '%q ' "$OURS_BIN" "${BUILD_ARGS[@]}" >"$WORKDIR/adapter-build-command.txt"
printf '\n' >>"$WORKDIR/adapter-build-command.txt"
"$OURS_BIN" "${BUILD_ARGS[@]}"

DOC_ARGS=(llm-docs --state-dir "$STATE" --all-pages --max-pages "$MAX_PAGES")
DOC_ARGS+=("${CONFIG_ARGS[@]}")
printf '%q ' "$OURS_BIN" "${DOC_ARGS[@]}" >"$WORKDIR/adapter-docs-command.txt"
printf '\n' >>"$WORKDIR/adapter-docs-command.txt"
set +e
"$OURS_BIN" "${DOC_ARGS[@]}"
DOCS_RC=$?
set -e

shopt -s nullglob
pages=("$STATE"/docs/markdown/*.md)
if (( ${#pages[@]} > 0 )); then
  cp "${pages[@]}" "$OUT"/
fi

# Write CodeWiki-compatible usage metadata and archive the normalized copy in
# work/ so token, cost and time provenance survives docs cleanup. This also runs
# after a failed page or batch, because failed calls still consume tokens.
STATS_ARGS=(stats --state-dir "$STATE" --out "$OUT/metadata.json"
  --run-scope "${OURS_USAGE_SCOPE:-state}")
printf '%q ' "$OURS_BIN" "${STATS_ARGS[@]}" >"$WORKDIR/adapter-stats-command.txt"
printf '\n' >>"$WORKDIR/adapter-stats-command.txt"
"$OURS_BIN" "${STATS_ARGS[@]}"

META_PY="${REPODOC_HARNESS_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$META_PY" ]]; then
  META_PY="$(command -v python3 || true)"
fi
if [[ -n "$META_PY" ]]; then
  "$META_PY" "$ROOT/scripts/system_metadata.py" \
    --docs "$OUT" \
    --out "$WORKDIR/system_metadata.json" \
    --system ours \
    || echo "ours adapter: warning: could not summarize usage metadata" >&2
fi

if (( DOCS_RC != 0 )); then
  echo "ours adapter: llm-docs failed (exit $DOCS_RC); docs and usage kept" >&2
  exit "$DOCS_RC"
fi
if (( ${#pages[@]} == 0 )); then
  echo "ours adapter: no markdown pages produced" >&2
  exit 1
fi

echo "ours adapter: wrote ${#pages[@]} flat pages to $OUT"
