"""Minimal OpenAI-compatible client for OpenCode Go.

Confirmed request format (Go, DeepSeek V4.1 Flash):

    POST https://opencode.ai/inference/go/openai/v1/chat/completions
    Authorization: Bearer oc_sk_...
    Content-Type: application/json
    User-Agent: repodoc-ours/0.1
    x-opencode-session: <stable conversation/run id>

The client keeps the request/response as raw OpenAI chat-completions JSON so
later agent stages can use tools or inspect reasoning_content/usage.opencode.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterable, Iterator

import httpx

from .config import OpenCodeGoConfig
from .ledger import record_error, record_response


class LLMError(RuntimeError):
    def __init__(
        self, message: str, *, status_code: int | None = None, body: str | None = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class OpenCodeGoClient:
    def __init__(
        self,
        config: OpenCodeGoConfig | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config or OpenCodeGoConfig.from_env()
        if not self.config.api_key:
            raise LLMError(
                "OPENCODE_GO_API_KEY is not set. Put it in the environment or a local .env, "
                "then run `repodoc llm-check --no-call` to inspect the configuration."
            )
        self._client = httpx.Client(
            base_url=self.config.base_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": self.config.user_agent,
                "x-opencode-session": self.config.session_id,
            },
            timeout=self.config.timeout,
            transport=transport,
        )

    def chat(
        self,
        messages: Iterable[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        response_format: dict[str, Any] | None = None,
        session_id: str | None = None,
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": list(messages),
            "temperature": self.config.temperature
            if temperature is None
            else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if response_format is not None:
            payload["response_format"] = response_format
        if extra_body:
            payload.update(extra_body)
        headers = dict(extra_headers or {})
        if session_id:
            headers["x-opencode-session"] = session_id

        started = time.monotonic()
        try:
            response = self._client.post(
                "chat/completions", json=payload, headers=headers
            )
        except Exception as exc:
            record_error(
                seconds=time.monotonic() - started,
                model=str(payload["model"]),
                error=type(exc).__name__,
            )
            raise
        elapsed = time.monotonic() - started
        if response.status_code >= 400:
            record_error(
                seconds=elapsed,
                model=str(payload["model"]),
                error=f"HTTP {response.status_code}",
            )
            raise LLMError(
                f"OpenCode Go request failed with HTTP {response.status_code}",
                status_code=response.status_code,
                body=response.text[:4000],
            )
        try:
            data = response.json()
        except ValueError as exc:
            record_error(
                seconds=elapsed,
                model=str(payload["model"]),
                error="invalid JSON response",
            )
            raise LLMError(
                "OpenCode Go returned non-JSON response", body=response.text[:4000]
            ) from exc
        record_response(data, seconds=elapsed, model=str(payload["model"]))
        return data

    def stream_chat(
        self,
        messages: Iterable[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        response_format: dict[str, Any] | None = None,
        session_id: str | None = None,
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield OpenAI-compatible SSE chunks for a streaming completion."""

        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": list(messages),
            "temperature": self.config.temperature
            if temperature is None
            else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            "stream": True,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if response_format is not None:
            payload["response_format"] = response_format
        if extra_body:
            payload.update(extra_body)
        headers = dict(extra_headers or {})
        if session_id:
            headers["x-opencode-session"] = session_id

        started = time.monotonic()
        try:
            with self._client.stream(
                "POST", "chat/completions", json=payload, headers=headers
            ) as response:
                if response.status_code >= 400:
                    body = response.read().decode("utf-8", errors="replace")[:4000]
                    record_error(
                        seconds=time.monotonic() - started,
                        model=str(payload["model"]),
                        error=f"HTTP {response.status_code}",
                    )
                    raise LLMError(
                        f"OpenCode Go request failed with HTTP {response.status_code}",
                        status_code=response.status_code,
                        body=body,
                    )
                for line in response.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
                    if line.startswith("data:"):
                        data = line[len("data:") :].strip()
                    else:
                        data = line.strip()
                    if not data or data == "[DONE]":
                        if data == "[DONE]":
                            break
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as exc:
                        record_error(
                            seconds=time.monotonic() - started,
                            model=str(payload["model"]),
                            error="invalid stream chunk",
                        )
                        raise LLMError(
                            "OpenCode Go returned an invalid stream chunk",
                            body=data[:4000],
                        ) from exc
                    if isinstance(chunk, dict):
                        yield chunk
        except LLMError:
            raise
        except Exception as exc:
            record_error(
                seconds=time.monotonic() - started,
                model=str(payload["model"]),
                error=type(exc).__name__,
            )
            raise

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self.chat(messages, **kwargs)
        try:
            return response["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                "OpenCode Go response has no choices[0].message.content",
                body=str(response),
            ) from exc

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenCodeGoClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
