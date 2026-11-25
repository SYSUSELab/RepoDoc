"""OpenCode Go / deepseek-v4.1-flash configuration.

The method is intentionally independent of CodeWiki and its config files. It
only reads OpenCode Go settings from environment variables (or a local .env).
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASE_URL = "https://opencode.ai/inference/go/openai/v1/"
DEFAULT_MODEL = "deepseek-v4.1-flash"
DEFAULT_USER_AGENT = "repodoc-ours/0.1"
DEFAULT_TIMEOUT = 300.0
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.2


def load_dotenv_simple(path: Path) -> None:
    """Load KEY=VALUE lines without overriding the real environment."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ[key] = value


def _load_env() -> None:
    env_file = os.environ.get("OPENCODE_GO_ENV_FILE")
    if env_file:
        load_dotenv_simple(Path(env_file).expanduser())
        return
    load_dotenv_simple(Path.cwd() / ".env")


def mask_key(key: str | None) -> str:
    if not key:
        return "<not set>"
    if len(key) <= 8:
        return "***"
    return f"{key[:5]}...{key[-4:]}"


@dataclass(frozen=True)
class OpenCodeGoConfig:
    api_key: str | None
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    session_id: str = ""
    user_agent: str = DEFAULT_USER_AGENT
    timeout: float = DEFAULT_TIMEOUT
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE

    @classmethod
    def from_env(cls) -> "OpenCodeGoConfig":
        _load_env()
        api_key = os.environ.get("OPENCODE_GO_API_KEY") or os.environ.get(
            "OPENCODE_API_KEY"
        )
        base_url = os.environ.get("OPENCODE_GO_BASE_URL", DEFAULT_BASE_URL)
        if not base_url.endswith("/"):
            base_url += "/"
        session_id = (
            os.environ.get("OPENCODE_GO_SESSION")
            or os.environ.get("REPODOC_SESSION_ID")
            or f"repodoc-{uuid.uuid4().hex[:16]}"
        )
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=os.environ.get("OPENCODE_GO_MODEL", DEFAULT_MODEL),
            session_id=session_id,
            user_agent=os.environ.get("OPENCODE_GO_USER_AGENT", DEFAULT_USER_AGENT),
            timeout=float(os.environ.get("OPENCODE_GO_TIMEOUT", DEFAULT_TIMEOUT)),
            max_tokens=int(
                os.environ.get("OPENCODE_GO_MAX_TOKENS", DEFAULT_MAX_TOKENS)
            ),
            temperature=float(
                os.environ.get("OPENCODE_GO_TEMPERATURE", DEFAULT_TEMPERATURE)
            ),
        )
