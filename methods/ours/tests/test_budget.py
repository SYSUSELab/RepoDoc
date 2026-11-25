from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import httpx  # noqa: E402

from repodoc.llm import ModelLimits, OpenCodeGoConfig  # noqa: E402
from repodoc.llm.budget import compute_budget  # noqa: E402
from repodoc.llm.models import fetch_model_limits  # noqa: E402


class BudgetTests(unittest.TestCase):
    def test_ratios_from_model_limits(self) -> None:
        limits = ModelLimits("deepseek-v4.1-flash", 1_000_000, 384_000)
        cluster = compute_budget(limits, "cluster")
        docs = compute_budget(limits, "docs")
        self.assertEqual(cluster.input_tokens, 20_000)
        self.assertEqual(cluster.output_tokens, 7_680)
        self.assertEqual(cluster.context_chars, 70_000)
        self.assertEqual(docs.input_tokens, 40_000)
        self.assertEqual(docs.output_tokens, 19_200)
        self.assertEqual(docs.context_chars, 140_000)

    def test_small_model_floors_are_guardrails(self) -> None:
        limits = ModelLimits("small", 128_000, 8_000)
        docs = compute_budget(limits, "docs")
        self.assertEqual(docs.input_tokens, 5_120)
        self.assertEqual(docs.output_tokens, 4_096)
        self.assertLessEqual(docs.input_tokens + docs.output_tokens, limits.context)

    def test_catalog_limits(self) -> None:
        config = OpenCodeGoConfig(
            api_key="oc_sk_test",
            base_url="https://example.invalid/v1/",
            model="deepseek-v4.1-flash",
            session_id="test",
        )

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/v1/models")
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "deepseek-v4.1-flash",
                            "limit": {"context": 1_000_000, "output": 384_000},
                        }
                    ]
                },
            )

        limits = fetch_model_limits(config, transport=httpx.MockTransport(handler))
        self.assertEqual(limits.context, 1_000_000)
        self.assertEqual(limits.output, 384_000)


if __name__ == "__main__":
    unittest.main()
