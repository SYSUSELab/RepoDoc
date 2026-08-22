"""LLM calls (OpenAI-compatible API). Used by clustering and doc generation.

Single implementation lives in LLMClient (async). All calls — doc generation,
clustering, concept extraction — record token usage into a per-endpoint client
so metadata/ablation statistics never miss a call.
"""

import asyncio
import logging
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


class LLMClient:
    """Async LLM client with token usage history, shared per (base_url, api_key)."""

    def __init__(self, config: "Config"):
        self.config = config
        self.history: list[TokenUsage] = []

    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> tuple[str, TokenUsage]:
        """Call the LLM and record token usage into self.history."""
        from openai import AsyncOpenAI
        from openai import APIError

        if model is None:
            model = self.config.main_model
        client = AsyncOpenAI(
            base_url=self.config.llm_base_url,
            api_key=self.config.llm_api_key,
        )

        if len(prompt) > _MAX_PROMPT_CHARS:
            logger.warning(
                f"Prompt too long ({len(prompt)} chars), truncating to 60K"
            )
            prompt = prompt[:_MAX_PROMPT_CHARS]

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except APIError as e:
            logger.error(f"API Error: {e}")
            # Fallback to a smaller max_tokens if API error
            if max_tokens > 4096:
                logger.info("Retrying with smaller max_tokens...")
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=4096,
                )
            else:
                raise
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
    ) -> tuple[str, TokenUsage]:
        """Synchronous wrapper. Only call outside a running event loop."""
        return asyncio.run(
            self.complete(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
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
) -> tuple[str, TokenUsage]:
    """Call LLM with the given prompt (OpenAI-compatible chat API).

    Args:
        prompt: User message content.
        config: RepoDoc config with llm_base_url, llm_api_key, main_model, cluster_model.
        model: Model name override (default: config.main_model).
        temperature: Sampling temperature.
        max_tokens: Max response tokens.

    Returns:
        Tuple of (assistant message content, token usage dict).
    """
    return get_llm_client(config).complete_sync(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


async def call_llm_async(
    prompt: str,
    config: "Config",
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> tuple[str, TokenUsage]:
    """Async version of call_llm with OpenAI-compatible async chat API."""
    return await get_llm_client(config).complete(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


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
