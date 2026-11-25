#!/usr/bin/env python3
"""Copy comparable generation accounting from docs/metadata.json to work/.

CodeWiki writes ``docs/metadata.json`` natively. RepoDoc writes the same shape
via ``repodoc stats``. Keeping this adapter-side copy means a later cache/cleanup
of ``docs/`` cannot silently remove the token/cost provenance from a run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.metrics import generation_stats
from harness.util import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True, help="generated docs directory")
    parser.add_argument("--out", required=True, help="system_metadata.json path")
    parser.add_argument("--system", help="system name, for provenance")
    parser.add_argument("--notes", help="optional human-readable note")
    args = parser.parse_args(argv)

    docs = Path(args.docs).expanduser().resolve()
    stats = generation_stats(docs)
    if not stats.get("tokens"):
        stats.setdefault("usage_status", "unavailable")
        stats.setdefault("cost_status", "unavailable")
        stats.setdefault("tokens", {})
    stats["source"] = "docs/metadata.json"
    if args.system:
        stats["system"] = args.system
    if args.notes:
        stats["notes"] = args.notes
    write_json(Path(args.out).expanduser().resolve(), stats)
    print(
        json.dumps({"wrote": str(args.out), "usage_status": stats.get("usage_status")})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
