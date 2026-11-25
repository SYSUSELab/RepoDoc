"""Model catalog and per-model token limits for OpenCode Go."""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from .config import OpenCodeGoConfig


class ModelCatalogError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelLimits:
    model: str
    context: int
    output: int
    source: str = "catalog"

    @classmethod
    def fallback(cls, model: str) -> "ModelLimits":
        """Conservative fallback only when the catalog cannot be fetched."""
        context = int(os.environ.get("OPENCODE_GO_CONTEXT_LIMIT", "200000"))
        output = int(os.environ.get("OPENCODE_GO_OUTPUT_LIMIT", "32768"))
        return cls(model=model, context=context, output=output, source="env-fallback")


def fetch_model_limits(
    config: OpenCodeGoConfig | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
    timeout: float = 30.0,
) -> ModelLimits:
    """Read the model's context/output limits from the Go catalog.

    The catalog endpoint is a plain GET and does not consume model credits.
    """
    config = config or OpenCodeGoConfig.from_env()
    headers = {
        "Accept": "application/json",
        "User-Agent": config.user_agent,
        "x-opencode-session": config.session_id,
    }
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    with httpx.Client(
        base_url=config.base_url,
        headers=headers,
        timeout=timeout,
        transport=transport,
    ) as client:
        response = client.get("models")
    if response.status_code >= 400:
        raise ModelCatalogError(
            f"model catalog request failed: HTTP {response.status_code}: {response.text[:1000]}"
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise ModelCatalogError("model catalog returned non-JSON") from exc
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list):
        raise ModelCatalogError("model catalog has no `data` list")
    for model in models:
        if not isinstance(model, dict) or model.get("id") != config.model:
            continue
        limit = model.get("limit") or {}
        context = limit.get("context")
        output = limit.get("output")
        if not isinstance(context, int) or not isinstance(output, int):
            raise ModelCatalogError(
                f"model {config.model!r} has no numeric limit.context/limit.output"
            )
        return ModelLimits(config.model, context, output)
    raise ModelCatalogError(f"model {config.model!r} not found in the Go catalog")


def model_limits_or_fallback(
    config: OpenCodeGoConfig | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
) -> ModelLimits:
    config = config or OpenCodeGoConfig.from_env()
    try:
        return fetch_model_limits(config, transport=transport)
    except (ModelCatalogError, httpx.HTTPError, OSError):
        return ModelLimits.fallback(config.model)
