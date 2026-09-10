"""LLM calls (OpenAI-compatible API). Used by clustering and doc generation.

Single implementation lives in LLMClient (async). All calls — doc generation,
clustering, summarization — record token usage into a per-endpoint client
so metadata/ablation statistics never miss a call.
"""

import asyncio
import collections
import logging
import random
import threading
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from repodoc.config import Config

logger = logging.getLogger(__name__)


class TokenUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# Prompt safety cap: keep under 60K chars.
_MAX_PROMPT_CHARS = 60_000

# Backoff retry budget for transient API errors (429/5xx/timeouts).
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_MAX_SECONDS = 8.0
_BACKOFF_MAX_RETRIES = 3


class AdaptiveLimiter:
    """AIMD adaptive concurrency limiter for LLM API calls.

    Concurrency shrinks multiplicatively when failures spike (busy gateway)
    and grows additively when everything succeeds (idle gateway) — the TCP
    congestion-control pattern. One shared instance serves both the sync
    (clustering thread pool) and async (doc generation) paths; they never
    run concurrently (stage separation), so no double bookkeeping.
    """

    def __init__(
        self,
        initial: int = 16,
        min_permits: int = 4,
        max_permits: int = 24,
        window: int = 40,
    ) -> None:
        self._cond = threading.Condition()
        self._permits = initial
        self._min = min_permits
        self._max = max_permits
        self._window = window
        self._results: collections.deque[bool] = collections.deque(maxlen=window)
        self._inflight = 0
        self._result_count = 0

    def acquire(self) -> None:
        with self._cond:
            while self._inflight >= self._permits:
                self._cond.wait()
            self._inflight += 1

    async def acquire_async(self) -> None:
        await asyncio.to_thread(self.acquire)

    def release(self, success: bool) -> None:
        with self._cond:
            self._inflight -= 1
            self._results.append(success)
            self._result_count += 1
            # adjust at most once per half-window of new results
            if self._result_count % (self._window // 2) == 0:
                self._maybe_adjust()
            self._cond.notify()

    def _maybe_adjust(self) -> None:
        if len(self._results) < self._window // 2:
            return  # not enough samples yet
        fail_rate = 1.0 - sum(self._results) / len(self._results)
        old = self._permits
        if fail_rate > 0.3:
            self._permits = max(self._min, self._permits // 2)  # multiplicative decrease
        elif fail_rate < 0.1:
            self._permits = min(self._max, self._permits + 1)  # additive increase
        if self._permits != old:
            logger.info(
                "Adaptive limiter: permits %d -> %d (fail_rate %.0f%%)",
                old,
                self._permits,
                fail_rate * 100,
            )

    def force_decrease(self) -> int:
        """Explicitly halve permits (called by callers after failed batches)
        so retries run at lower concurrency until they succeed. Returns the
        new permit count."""
        with self._cond:
            old = self._permits
            self._permits = max(self._min, self._permits // 2)
            if self._permits != old:
                logger.info(
                    "Adaptive limiter: permits %d -> %d (forced decrease)",
                    old,
                    self._permits,
                )
            self._cond.notify_all()
            return self._permits

    @property
    def permits(self) -> int:
        with self._cond:
            return self._permits

    @property
    def min_permits(self) -> int:
        return self._min


# Shared across clustering and doc generation (they are stage-separated).
_LIMITER = AdaptiveLimiter()


class LLMClient:
    """Async LLM client with token usage history, shared per (base_url, api_key)."""

    def __init__(self, config: "Config"):
        self.config = config
        self.history: list[TokenUsage] = []

    def _build_headers(self) -> dict[str, str]:
        """Extra headers for every request.

        Base: config.llm_headers (LLM_EXTRA_HEADERS JSON via make_config).
        Built-in: opencode.ai gateways route requests by session and reject
        callers without `x-opencode-session` (400 MissingSessionID) — attach
        a stable id automatically; a user-configured value wins.
        """
        headers = dict(getattr(self.config, "llm_headers", None) or {})
        base_url = self.config.llm_base_url or ""
        if "opencode" in base_url and "x-opencode-session" not in headers:
            headers["x-opencode-session"] = "repodoc"
        return headers

    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        response_format: str | None = None,
    ) -> tuple[str, TokenUsage]:
        """Call the LLM and record token usage into self.history.

        response_format: OpenAI-compatible structured output mode, e.g.
        "json_object" (None = default free-form output). Endpoints that do
        not support it fail with APIError and fall back automatically.
        """
        from openai import AsyncOpenAI
        from openai import APIError

        if model is None:
            model = self.config.main_model
        openai_kwargs = dict(
            base_url=self.config.llm_base_url,
            api_key=self.config.llm_api_key,
            max_retries=1,  # SDK does one quick retry; backoff is ours
        )
        headers = self._build_headers()
        if headers:
            openai_kwargs["default_headers"] = headers
        client = AsyncOpenAI(**openai_kwargs)

        if len(prompt) > _MAX_PROMPT_CHARS:
            logger.warning(
                f"Prompt too long ({len(prompt)} chars), truncating to 60K"
            )
            prompt = prompt[:_MAX_PROMPT_CHARS]

        def _create(mt: int, rf: str | None):
            kwargs = dict(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=mt,
            )
            if rf:
                kwargs["response_format"] = {"type": rf}
            return client.chat.completions.create(**kwargs)

        try:
            response = await _create(max_tokens, response_format)
        except APIError as e:
            logger.error(f"API Error: {e}")
            # Config errors (4xx other than 408/429) never recover: no retry.
            # 408/429 ARE retryable (timeout / rate limit).
            if (
                e.status_code is not None
                and 400 <= e.status_code < 500
                and e.status_code not in (408, 429)
            ):
                raise
            # Transient errors (429/5xx/timeouts): exponential backoff.
            last_error = e
            success = False
            for attempt in range(_BACKOFF_MAX_RETRIES):
                delay = min(
                    _BACKOFF_BASE_SECONDS * (2**attempt)
                    + random.uniform(0, 0.3),
                    _BACKOFF_MAX_SECONDS,
                )
                logger.warning(
                    "Retrying in %.2fs (attempt %d/%d) after %s",
                    delay,
                    attempt + 1,
                    _BACKOFF_MAX_RETRIES,
                    e,
                )
                await asyncio.sleep(delay)
                try:
                    response = await _create(max_tokens, response_format)
                    success = True
                    break
                except APIError as e2:
                    last_error = e2
                    if (
                        e2.status_code is not None
                        and 400 <= e2.status_code < 500
                        and e2.status_code not in (408, 429)
                    ):
                        raise
            if not success:
                # Backoff exhausted: try the max_tokens fallback once
                # (existing behavior) before giving up.
                if max_tokens > 4096:
                    logger.info("Retrying with smaller max_tokens...")
                    try:
                        response = await _create(4096, None)
                        success = True
                    except APIError:
                        pass
                if not success:
                    raise last_error
        finally:
            await client.close()

        usage = response.usage
        token_usage: TokenUsage = {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }
        self.history.append(token_usage)
        return response.choices[0].message.content or "", token_usage

    def complete_sync(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        response_format: str | None = None,
    ) -> tuple[str, TokenUsage]:
        """Synchronous wrapper. Only call outside a running event loop."""
        return asyncio.run(
            self.complete(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        )


_clients: dict[tuple[str, str], LLMClient] = {}


def get_llm_client(config: "Config") -> LLMClient:
    """Return the shared LLM client for this endpoint (token history is global)."""
    key = (config.llm_base_url, config.llm_api_key)
    client = _clients.get(key)
    if client is None:
        client = LLMClient(config)
        _clients[key] = client
    return client


def call_llm(
    prompt: str,
    config: "Config",
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    response_format: str | None = None,
) -> tuple[str, TokenUsage]:
    """Call LLM with the given prompt (OpenAI-compatible chat API).

    Args:
        prompt: User message content.
        config: RepoDoc config with llm_base_url, llm_api_key, main_model, cluster_model.
        model: Model name override (default: config.main_model).
        temperature: Sampling temperature.
        max_tokens: Max response tokens.
        response_format: Optional structured output mode (e.g. "json_object").

    Returns:
        Tuple of (assistant message content, token usage dict).
    """
    _LIMITER.acquire()
    try:
        result = get_llm_client(config).complete_sync(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
    except Exception:
        _LIMITER.release(False)
        raise
    _LIMITER.release(True)
    return result


async def call_llm_async(
    prompt: str,
    config: "Config",
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    response_format: str | None = None,
) -> tuple[str, TokenUsage]:
    """Async version of call_llm with OpenAI-compatible async chat API."""
    await _LIMITER.acquire_async()
    try:
        result = await get_llm_client(config).complete(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
    except Exception:
        _LIMITER.release(False)
        raise
    _LIMITER.release(True)
    return result


def total_token_usage(config: "Config") -> dict:
    """Aggregate token usage across all calls made through the shared client."""
    client = get_llm_client(config)
    total = sum(u.get("total_tokens", 0) for u in client.history)
    prompt = sum(u.get("prompt_tokens", 0) for u in client.history)
    completion = sum(u.get("completion_tokens", 0) for u in client.history)
    return {
        "history": client.history,
        "total": total,
        "prompt": prompt,
        "completion": completion,
        "llm_calls": len(client.history),
    }
