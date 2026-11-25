#!/usr/bin/env python
"""Run the CodeWikiBench judge with the OpenCode Go compatible endpoint.

CodeWikiBench expects its judge endpoint in BASE_URL/API_KEY. The project's
OpenCode Go endpoint additionally requires ``x-opencode-session`` and a
``User-Agent`` header. We inject those headers through pydantic-ai's
OpenAIProvider without modifying the benchmark checkout.

Example:
    python scripts/architecture_judge.py \
      --rubrics /tmp/openhands-architecture-rubrics.json \
      --docs benchmarks/codewikibench/examples/OpenHands/codewiki \
      --out runs/architecture-openhands/judge-deepseek/codewiki.json \
      --model deepseek-v4.1-flash --batch-size 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks" / "codewikibench" / "src"))

from harness.util import load_dotenv_simple  # noqa: E402


def _configure_endpoint() -> None:
    load_dotenv_simple(ROOT / ".env")
    base = os.environ.get("BASE_URL", "").strip()
    if not base or "example.com" in base:
        os.environ["BASE_URL"] = os.environ["OPENCODE_GO_BASE_URL"]
        os.environ["API_KEY"] = os.environ["OPENCODE_GO_API_KEY"]
    os.environ.setdefault("OPENCODE_GO_SESSION", "repodoc-architecture-judge")


def _patch_llm() -> None:
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider

    from codewikibench import config
    import codewikibench.utils as utils

    def get_llm(model: str | None = None):
        model_name = model or config.MODEL
        client = AsyncOpenAI(
            base_url=config.BASE_URL,
            api_key=config.API_KEY,
            default_headers={
                "x-opencode-session": os.environ.get(
                    "OPENCODE_GO_SESSION", "repodoc-architecture-judge"
                ),
                "User-Agent": os.environ.get(
                    "OPENCODE_GO_USER_AGENT", "repodoc-ours/0.1"
                ),
            },
        )
        return OpenAIChatModel(
            model_name=model_name,
            provider=OpenAIProvider(openai_client=client),
            settings=OpenAIChatModelSettings(max_tokens=36_000, timeout=300),
        )

    utils.get_llm = get_llm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rubrics", required=True)
    parser.add_argument("--docs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="deepseek-v4.1-flash")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--no-tools", action="store_true")
    args = parser.parse_args(argv)

    _configure_endpoint()
    _patch_llm()

    from codewikibench.cli import main as bench_main

    bench_args = [
        "judge",
        "--rubrics",
        args.rubrics,
        "--docs",
        args.docs,
        "--out",
        args.out,
        "--model",
        args.model,
        "--batch-size",
        str(args.batch_size),
        "--method",
        "average",
        "--max-retries",
        str(args.max_retries),
    ]
    if not args.no_tools:
        bench_args.append("--use-tools")
    if args.max_retries >= 0:
        bench_args.append("--enable-retry")
    return int(bench_main(bench_args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
