#!/usr/bin/env bash
# CodeWiki 2.0 adapter.
#
# Contract: see adapters/_template.sh and docs/adding-a-system.md.
# CodeWiki writes markdown pages plus module_tree.json/metadata.json. We pass
# --flat because CodeWikiBench's parser only reads .md files at the docs root.

set -euo pipefail

REPO=""
OUT=""
WORKDIR=""
MODE="full"
EXTRA=()

usage() {
  cat <<'EOF'
Usage: codewiki.sh --repo PATH --out PATH --workdir PATH [--mode full|update] [-- ARGS...]

Environment:
  CODEWIKI_BIN          codewiki executable (default: codewiki)
  CODEWIKI_HOME         optional isolated HOME for ~/.codewiki config
  CODEWIKI_EXTRA_ARGS   additional shell-separated flags for `codewiki generate`
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
    *)         echo "codewiki adapter: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$REPO" || -z "$OUT" || -z "$WORKDIR" ]]; then
  echo "codewiki adapter: --repo, --out and --workdir are required" >&2
  usage >&2
  exit 2
fi
if [[ ! -d "$REPO" ]]; then
  echo "codewiki adapter: repository does not exist: $REPO" >&2
  exit 2
fi
if [[ "$MODE" != "full" && "$MODE" != "update" ]]; then
  echo "codewiki adapter: --mode must be full or update (got: $MODE)" >&2
  exit 2
fi

CODEWIKI_BIN="${CODEWIKI_BIN:-codewiki}"
if ! command -v "$CODEWIKI_BIN" >/dev/null 2>&1; then
  echo "codewiki adapter: executable not found: $CODEWIKI_BIN" >&2
  echo "Install with 'make setup-codewiki' or set CODEWIKI_BIN." >&2
  exit 127
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${CODEWIKI_HOME:-}" ]]; then
  mkdir -p "$CODEWIKI_HOME"
  export HOME="$CODEWIKI_HOME"
fi
META_PY="${REPODOC_HARNESS_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$META_PY" ]]; then
  META_PY="$(command -v python3 || true)"
fi

mkdir -p "$OUT" "$WORKDIR"

# Keep enough provenance to reproduce a run without logging API keys.
"$CODEWIKI_BIN" --version >"$WORKDIR/codewiki-version.txt" 2>&1 || true
"$CODEWIKI_BIN" config show --json >"$WORKDIR/codewiki-config.json" 2>/dev/null || true

ARGS=(generate --output "$OUT" --flat)
if [[ "$MODE" == "update" ]]; then
  ARGS+=(--update)
fi

if [[ -n "${CODEWIKI_EXTRA_ARGS:-}" ]]; then
  read -r -a ENV_EXTRA <<<"$CODEWIKI_EXTRA_ARGS"
  ARGS+=("${ENV_EXTRA[@]}")
fi
ARGS+=("${EXTRA[@]}")

printf '%q ' "$CODEWIKI_BIN" "${ARGS[@]}" >"$WORKDIR/adapter-command.txt"
printf '\n' >>"$WORKDIR/adapter-command.txt"

# Run in a subshell instead of `exec` so metadata can be archived even when
# generation returns non-zero (failed runs still consume tokens).
set +e
( cd "$REPO" && "$CODEWIKI_BIN" "${ARGS[@]}" )
rc=$?
set -e

if [[ -f "$OUT/metadata.json" ]]; then
  cp "$OUT/metadata.json" "$WORKDIR/codewiki-metadata.json"
fi
if [[ -n "$META_PY" ]]; then
  "$META_PY" "$ROOT/scripts/system_metadata.py" \
    --docs "$OUT" \
    --out "$WORKDIR/system_metadata.json" \
    --system codewiki \
    || echo "codewiki adapter: warning: could not summarize usage metadata" >&2
fi

exit "$rc"
