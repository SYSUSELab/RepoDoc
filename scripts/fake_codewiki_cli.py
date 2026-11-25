#!/usr/bin/env python3
"""A tiny offline stand-in for the `codewiki` CLI.

Only the commands used by `adapters/codewiki.sh` are implemented:

    codewiki --version
    codewiki config show --json
    codewiki generate --output DIR --flat ...

Generation uses the real CodeWiki DependencyGraphBuilder/DocumentationGenerator
with the deterministic FakeBackend from `smoke_codewiki_fake.py`. No API key,
network call, or ~/.codewiki config is used.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from codewiki.src.be.documentation_generator import DocumentationGenerator
from codewiki.src.config import Config
from smoke_codewiki_fake import FakeBackend

FAKE_CONFIG = {
    "provider": "openai-compatible",
    "base_url": "http://127.0.0.1:9/v1",
    "api_key": "offline-smoke-no-key",
    "main_model": "fake-codewiki-model",
    "cluster_model": "fake-codewiki-model",
    "fallback_model": "fake-codewiki-model",
}


def generate(output: Path, *, cluster: bool) -> None:
    repo = Path.cwd()
    docs = output.resolve()
    config = Config(
        repo_path=str(repo),
        output_dir=str(docs / "temp"),
        dependency_graph_dir=str(docs / "temp" / "dependency_graphs"),
        docs_dir=str(docs),
        max_depth=2,
        llm_base_url=str(FAKE_CONFIG["base_url"]),
        llm_api_key=str(FAKE_CONFIG["api_key"]),
        main_model=str(FAKE_CONFIG["main_model"]),
        cluster_model=str(FAKE_CONFIG["cluster_model"]),
        fallback_model=str(FAKE_CONFIG["fallback_model"]),
        provider="openai-compatible",
        # cluster=True forces the clustered path; otherwise document the repo whole.
        max_token_per_module=1 if cluster else 10_000_000,
        min_token_per_module=1,
        artifacts_enabled=False,
        with_prose=False,
        use_gitignore=False,
        layout="flat",
        max_tokens=8192,
    )
    asyncio.run(DocumentationGenerator(config, backend=FakeBackend()).run())


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if argv and argv[0] in ("--version", "-V"):
        print("CodeWiki CLI, version 2.0.0-offline-fake")
        return 0
    if len(argv) >= 2 and argv[0] == "config" and argv[1] == "show":
        print(json.dumps(FAKE_CONFIG, indent=2))
        return 0
    if not argv or argv[0] != "generate":
        print(
            "fake codewiki CLI: only --version, config show, and generate are supported",
            file=os.sys.stderr,
        )
        return 2

    parser = argparse.ArgumentParser(prog="codewiki generate")
    parser.add_argument("--output", "-o", default="docs")
    parser.add_argument("--flat", action="store_true")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--no-artifacts", action="store_true")
    parser.add_argument("--with-prose", action="store_true")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args, unknown = parser.parse_known_args(argv[1:])
    if unknown:
        print(
            f"fake codewiki CLI: ignoring unsupported flags: {unknown}",
            file=os.sys.stderr,
        )

    cluster = os.environ.get("FAKE_CODEWIKI_CLUSTER", "1") not in ("0", "false", "no")
    generate(Path(args.output), cluster=cluster)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
