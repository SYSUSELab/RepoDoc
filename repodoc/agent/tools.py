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


class FileReaderTool:
    name = "file_reader"
    description = "Read file content from the repository"

    async def execute(self, file_path: str, **kwargs) -> ToolResult:
        try:
            if not os.path.exists(file_path):
                return ToolResult(success=False, error=f"File not found: {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            return ToolResult(success=True, output=content)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class FileWriterTool:
    name = "file_writer"
    description = "Write content to a file in the repository"

    async def execute(self, file_path: str, content: str, **kwargs) -> ToolResult:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
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
        **kwargs,
    ) -> ToolResult:
        try:
            result, _ = await call_llm_async(
                prompt=prompt,
                config=self.config,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return ToolResult(success=True, output=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
