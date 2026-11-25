#!/usr/bin/env bash
# Offline integration smoke: harness -> adapters/codewiki.sh -> fake CodeWiki CLI.
#
# Uses the toy repo created by smoke_codewiki_fake.py and does not call an LLM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv-codewiki/bin/python"
TOY="$ROOT/runs/smoke-codewiki/whole/repo"
OUT="$ROOT/runs/smoke-codewiki-harness"

if [[ ! -x "$PY" ]]; then
  echo "missing $PY; run 'make setup-codewiki' first" >&2
  exit 1
fi
if [[ ! -d "$TOY" ]]; then
  echo "creating toy repo via scripts/smoke_codewiki_fake.py"
  "$PY" "$ROOT/scripts/smoke_codewiki_fake.py"
fi

# Validate the real CLI is installed and its click command tree loads. This
# does not call an LLM; generation below uses the fake CLI.
"$ROOT/.venv-codewiki/bin/codewiki" --version
"$ROOT/.venv-codewiki/bin/codewiki" generate --help >/dev/null

rm -rf "$OUT"
cd "$ROOT"
CODEWIKI_BIN="$ROOT/scripts/fake_codewiki_cli.sh" \
  "$PY" -m harness generate \
    --system codewiki \
    --repo OpenHands \
    --experiment smoke-codewiki-harness \
    --checkout "$TOY" \
    --force

DOCS="$OUT/systems/codewiki/OpenHands/docs"
test -s "$DOCS/overview.md"
test -s "$DOCS/calculator.md"
test -s "$DOCS/app.md"
test -s "$DOCS/metadata.json"
test -f "$OUT/systems/codewiki/OpenHands/meta.json"
echo
echo "OK: harness + adapter + fake CodeWiki CLI smoke passed."
echo "Artifacts: $OUT"
