from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import httpx  # noqa: E402

from repodoc.llm import LLMError, OpenCodeGoClient, OpenCodeGoConfig, mask_key  # noqa: E402


def test_config() -> OpenCodeGoConfig:
    return OpenCodeGoConfig(
        api_key="oc_sk_test_123456789",
        base_url="https://opencode.ai/inference/go/openai/v1/",
        model="deepseek-v4.1-flash",
        session_id="session-test",
        user_agent="repodoc-ours/test",
    )


class OpenCodeGoClientTests(unittest.TestCase):
    def test_request_format_and_headers(self) -> None:
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(
                {
                    "url": str(request.url),
                    "headers": dict(request.headers),
                    "body": json.loads(request.content),
                }
            )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "hi", "reasoning_content": "short"}}
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 2,
                        "opencode": {"billing_source": "go", "cost": 0},
                    },
                },
            )

        transport = httpx.MockTransport(handler)
        with OpenCodeGoClient(test_config(), transport=transport) as client:
            response = client.chat(
                [{"role": "user", "content": "Say hi"}],
                session_id="conversation-42",
            )
            content = client.complete("Say hi")

        first = seen[0]
        self.assertEqual(
            first["url"],
            "https://opencode.ai/inference/go/openai/v1/chat/completions",
        )
        self.assertEqual(
            first["headers"]["authorization"], "Bearer oc_sk_test_123456789"
        )
        self.assertEqual(first["headers"]["x-opencode-session"], "conversation-42")
        self.assertEqual(first["headers"]["user-agent"], "repodoc-ours/test")
        self.assertTrue(first["headers"]["content-type"].startswith("application/json"))
        self.assertEqual(first["body"]["model"], "deepseek-v4.1-flash")
        self.assertEqual(
            first["body"]["messages"], [{"role": "user", "content": "Say hi"}]
        )
        self.assertEqual(response["usage"]["opencode"]["billing_source"], "go")
        self.assertEqual(content, "hi")

    def test_stream_chat_parses_sse(self) -> None:
        seen: list[dict] = []
        body = (
            b'data: {"choices":[{"delta":{"content":"hel"}}],"usage":null}\n\n'
            b'data: {"choices":[{"delta":{"content":"lo"}}],"usage":null}\n\n'
            b'data: {"choices":[],"usage":{"total_tokens":7}}\n\n'
            b"data: [DONE]\n\n"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(
                200,
                content=body,
                headers={"content-type": "text/event-stream"},
            )

        with OpenCodeGoClient(
            test_config(), transport=httpx.MockTransport(handler)
        ) as client:
            chunks = list(client.stream_chat([{"role": "user", "content": "hi"}]))

        self.assertTrue(seen[0]["stream"])
        self.assertEqual(
            "".join(
                chunk["choices"][0]["delta"].get("content", "")
                for chunk in chunks
                if chunk.get("choices")
            ),
            "hello",
        )
        self.assertEqual(chunks[-1]["usage"]["total_tokens"], 7)

    def test_missing_key(self) -> None:
        config = OpenCodeGoConfig(api_key=None)
        with self.assertRaises(LLMError):
            OpenCodeGoClient(config)

    def test_mask_key(self) -> None:
        self.assertEqual(mask_key(None), "<not set>")
        self.assertNotIn("123456789", mask_key("oc_sk_test_123456789"))

    def test_http_error_includes_body(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, text='{"error":"forbidden"}')

        with OpenCodeGoClient(
            test_config(), transport=httpx.MockTransport(handler)
        ) as client:
            with self.assertRaises(LLMError) as ctx:
                client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("forbidden", ctx.exception.body or "")


if __name__ == "__main__":
    unittest.main()
