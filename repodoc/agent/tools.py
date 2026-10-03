"""Tools available to the DocAgent (file I/O and LLM calls)."""

import os
from typing import Any

from repodoc.config import Config
from repodoc.llm import TokenUsage, call_llm_async, get_llm_client


class ToolResult:
    """Result of tool execution"""

    def __init__(self, success: bool, output: Any = None, error: str | None = None):
        self.success = success
        self.output = output
        self.error = error

    def __repr__(self):
        if self.success:
            return f"ToolResult(success=True, output={self.output})"
        return f"ToolResult(success=False, error={self.error})"


class FileWriterTool:
    """Writes generated docs, constrained to a root directory.

    LLM output (e.g. module names) flows into output paths, so a hostile or
    buggy generation could otherwise escape the docs directory (`../../x`).
    All writes are rejected unless the resolved path stays under `root`."""

    name = "file_writer"
    description = "Write content to a file in the repository"

    def __init__(self, root: str | None = None):
        self.root = os.path.realpath(root) if root else None

    def _is_within_root(self, file_path: str) -> bool:
        if self.root is None:
            return True
        real = os.path.realpath(file_path)
        try:
            return os.path.commonpath([self.root, real]) == self.root
        except ValueError:
            return False

    async def execute(self, file_path: str, content: str, **kwargs) -> ToolResult:
        try:
            if not self._is_within_root(file_path):
                return ToolResult(
                    success=False,
                    error=f"Refusing to write outside {self.root}: {file_path}",
                )
            directory = os.path.dirname(file_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return ToolResult(success=True, output=f"Written to {file_path}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class LLMTool:
    name = "llm_caller"
    description = "Call LLM for code generation or analysis"

    def __init__(self, config: Config):
        self.config = config

    def get_token_usage_history(self) -> list[TokenUsage]:
        """Token history lives on the shared LLM client, so clustering
        calls are counted too."""
        return get_llm_client(self.config).history

    async def execute(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        system_prompt: str | None = None,
        **kwargs,
    ) -> ToolResult:
        try:
            result, _ = await call_llm_async(
                prompt=prompt,
                config=self.config,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )
            return ToolResult(success=True, output=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
