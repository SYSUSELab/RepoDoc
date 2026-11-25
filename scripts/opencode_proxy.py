#!/usr/bin/env python3
"""Minimal OpenAI-compatible proxy that injects OpenCode Go headers.

CodeWiki constructs its OpenAI client without ``x-opencode-session``. The Go
endpoint rejects such requests. Run this proxy locally and point CodeWiki's
``base_url`` at it; Authorization and request bodies pass through unchanged.

Example:
    python scripts/opencode_proxy.py --port 8766 \
      --upstream https://opencode.ai/inference/go/openai/v1
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HOP_HEADERS = {
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class ProxyHandler(BaseHTTPRequestHandler):
    upstream = ""
    session = "repodoc-codewiki"
    user_agent = "repodoc-codewiki-proxy/0.1"
    timeout = 300.0

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        print(f"proxy: {self.address_string()} {fmt % args}", flush=True)

    def _target_url(self) -> str:
        incoming = self.path
        if incoming.startswith("/v1"):
            incoming = incoming[3:]
        if not incoming.startswith("/"):
            incoming = "/" + incoming
        return self.upstream.rstrip("/") + incoming

    def _forward(self, method: str, body: bytes | None = None) -> None:
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in _HOP_HEADERS
        }
        headers["x-opencode-session"] = self.session
        headers["Accept-Encoding"] = "identity"
        headers.setdefault("User-Agent", self.user_agent)
        request = urllib.request.Request(
            self._target_url(),
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read()
                status = response.status
                content_type = response.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            status = exc.code
            content_type = (
                exc.headers.get("Content-Type", "application/json")
                if exc.headers
                else "application/json"
            )
        except Exception as exc:  # noqa: BLE001 - proxy must return a useful error
            payload = json.dumps(
                {
                    "error": {
                        "message": f"proxy upstream error: {type(exc).__name__}: {exc}"
                    }
                }
            ).encode()
            status = 502
            content_type = "application/json"

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        self._forward("GET")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length else b""
        self._forward("POST", body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--upstream",
        default=os.environ.get(
            "OPENCODE_GO_BASE_URL",
            "https://opencode.ai/inference/go/openai/v1",
        ),
    )
    parser.add_argument(
        "--session",
        default=os.environ.get("OPENCODE_GO_SESSION", "repodoc-codewiki"),
    )
    parser.add_argument(
        "--user-agent",
        default=os.environ.get("OPENCODE_GO_USER_AGENT", "repodoc-ours/0.1"),
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)

    ProxyHandler.upstream = args.upstream
    ProxyHandler.session = args.session
    ProxyHandler.user_agent = args.user_agent
    ProxyHandler.timeout = args.timeout
    server = ThreadingHTTPServer((args.host, args.port), ProxyHandler)
    print(
        f"opencode proxy: http://{args.host}:{args.port}/v1 -> {args.upstream} "
        f"(session={args.session})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
