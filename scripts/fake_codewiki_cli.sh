#!/usr/bin/env bash
# Offline `codewiki` CLI wrapper for harness integration smoke tests.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/.venv-codewiki/bin/python" "$ROOT/scripts/fake_codewiki_cli.py" "$@"
