#!/usr/bin/env bash
# Template adapter for a new documentation-generation baseline or our method.
#
# Copy to adapters/<system>.sh, register it in configs/systems.yaml, and replace
# the TODO below. The harness provides:
#
#   --repo     Working tree checked out at the benchmark commit (read it only).
#   --out      Directory where the system must write flat .md files.
#   --workdir  Scratch directory for logs/caches; safe to modify.
#   --mode     `full` or `update`; `update` means base docs already exist in --out.
#
# Environment variables with the same values are also set by the harness:
#   REPODOC_REPO, REPODOC_OUT, REPODOC_WORKDIR, REPODOC_MODE,
#   REPODOC_REPO_NAME, REPODOC_EXPERIMENT, REPODOC_SYSTEM.

set -euo pipefail

REPO=""
OUT=""
WORKDIR=""
MODE="full"
EXTRA=()

usage() {
  echo "Usage: $0 --repo PATH --out PATH --workdir PATH [--mode full|update] [-- ARGS...]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)    REPO="${2:?--repo needs a value}"; shift 2 ;;
    --out)     OUT="${2:?--out needs a value}"; shift 2 ;;
    --workdir) WORKDIR="${2:?--workdir needs a value}"; shift 2 ;;
    --mode)    MODE="${2:?--mode needs a value}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --)        shift; EXTRA+=("$@"); break ;;
    *)         echo "$0: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$REPO" && -n "$OUT" && -n "$WORKDIR" ]] || { usage >&2; exit 2; }
[[ -d "$REPO" ]] || { echo "repository not found: $REPO" >&2; exit 2; }
[[ "$MODE" == "full" || "$MODE" == "update" ]] || { echo "bad mode: $MODE" >&2; exit 2; }

mkdir -p "$OUT" "$WORKDIR"

# TODO: invoke the system. Examples:
#
#   cd "$REPO"
#   if [[ "$MODE" == "update" ]]; then
#     my-doc-generator --output "$OUT" --update "${EXTRA[@]}"
#   else
#     my-doc-generator --output "$OUT" "${EXTRA[@]}"
#   fi
#
# For an update-capable system, update the existing docs already present in
# "$OUT" in place; do not delete and recreate them.

echo "adapter is a template; edit $0 before running" >&2
exit 2
