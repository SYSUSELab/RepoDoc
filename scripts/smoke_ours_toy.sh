#!/usr/bin/env bash
# Offline smoke for methods/ours: toy Python repo -> graph -> component/doc trees -> markdown.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv-ours/bin/python"
OUT="$ROOT/runs/smoke-ours"
REPO="$OUT/repo"
STATE="$OUT/state"

if [[ ! -x "$PY" ]]; then
  echo "missing $PY; run 'make setup-ours' first" >&2
  exit 1
fi

rm -rf "$OUT"
mkdir -p "$REPO"
cat > "$REPO/calculator.py" <<'PY'
"""Tiny calculator."""

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
PY
cat > "$REPO/app.py" <<'PY'
"""Tiny app."""

from calculator import add, multiply

def run() -> int:
    """Run one calculation."""
    return multiply(add(1, 2), 3)

class App:
    def compute(self):
        # use the module function
        return run()
PY

"$PY" -m repodoc build --repo "$REPO" --state-dir "$STATE" --force
"$PY" -m repodoc status --repo "$REPO" --state-dir "$STATE"
"$PY" -m repodoc inspect --state-dir "$STATE" --what graph
"$PY" -m repodoc inspect --state-dir "$STATE" --what components
"$PY" -m repodoc inspect --state-dir "$STATE" --what docs

test -s "$STATE/docs/markdown/overview.md"
test -s "$STATE/docs/markdown/app.md"
test -s "$STATE/docs/markdown/calculator.md"
test -s "$STATE/graph/graph.json"
echo
echo "OK: methods/ours toy pipeline smoke passed."
echo "Artifacts: $STATE"
