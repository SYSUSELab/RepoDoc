"""DocAgent: LLM-driven documentation writer.

Merges the former Agent + DocWriterSkill + tools layers into one class.
The execute() protocol is unchanged for callers:
    task = {"type": "write_doc", "config": ..., "input": {"type": "...", ...}}
and the result keeps the SkillResult shape (success/output/error/metadata).
"""

from typing import Any

from pydantic import BaseModel

from repodoc.agent.prompts import (
    format_component_prompt,
    format_module_prompt,
    format_overview_prompt,
)
from repodoc.agent.tools import FileWriterTool, LLMTool
from repodoc.config import Config


class SkillResult(BaseModel):
    """Result of a doc generation task."""

    success: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = {}


# Unified prompt for the "direct" doc strategy (ablation baseline without
# per-doc-type templates).
DIRECT_DOC_PROMPT = """You are an expert technical documentation writer.

Generate a well-structured Markdown document for the following {doc_type} of this repository.

<CONTEXT>
{context}
</CONTEXT>

Requirements:
- Clear overview of purpose and functionality
- Accurate technical detail grounded ONLY in the provided context
- Markdown tables for any API listings, Mermaid diagrams where helpful
- Link to related modules/components using relative markdown links
- Do not invent APIs, files, or behavior that are not in the context

Return ONLY the Markdown document content, no extra explanation."""


class DocAgent:
    """Documentation agent: generates overview/module/component/architecture docs."""

    def __init__(self, config: Config):
        self.config = config
        self.tools = {
            "file_writer": FileWriterTool(),
            "llm_caller": LLMTool(config),
        }

    async def execute(self, task: dict[str, Any]) -> SkillResult:
        task_type = task.get("type")
        if task_type not in ("write_doc", None):
            return SkillResult(
                success=False, error=f"Unsupported task type: {task_type}"
            )
        config = task.get("config") or self.config
        input_data = task.get("input", {})
        doc_type = input_data.get("type", "component")

        llm_tool = self.tools["llm_caller"]
        file_tool = self.tools["file_writer"]

        try:
            if self.config.doc_strategy == "direct":
                return await self._generate_direct(doc_type, input_data, llm_tool, file_tool)
            if doc_type == "overview":
                return await self._generate_overview(input_data, llm_tool, file_tool)
            elif doc_type == "component":
                return await self._generate_component_doc(input_data, llm_tool, file_tool)
            elif doc_type == "module":
                return await self._generate_module_doc(input_data, llm_tool, file_tool)
            else:
                return SkillResult(
                    success=False, error=f"Unknown documentation type: {doc_type}"
                )
        except Exception as e:
            return SkillResult(success=False, error=str(e))

    # -- generation helpers ------------------------------------------------

    async def _call_llm(
        self, llm_tool, system_prompt: str, user_prompt: str, *, temperature=0.3, max_tokens=8192
    ) -> SkillResult:
        result = await llm_tool.execute(
            prompt=f"{system_prompt}\n\n{user_prompt}",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not result.success:
            return SkillResult(success=False, error=result.error)
        return SkillResult(success=True, output=result.output)

    async def _write_result(self, output_path: str, content: str, doc_type: str, file_tool) -> SkillResult:
        write_result = await file_tool.execute(file_path=output_path, content=content)
        if not write_result.success:
            return SkillResult(success=False, error=write_result.error)
        return SkillResult(
            success=True,
            output={"type": doc_type, "content": content, "path": output_path},
            metadata={"skill": "doc_writer", "doc_type": doc_type},
        )

    async def _generate_overview(self, input_data: dict, llm_tool, file_tool) -> SkillResult:
        repo_name = input_data.get("repo_name", "Repository")
        component_count = input_data.get("component_count", 0)
        module_count = input_data.get("module_count", 0)
        module_dependencies = input_data.get("module_dependencies", {})
        module_docs_summary = input_data.get("module_docs_summary", {})

        system_prompt, user_prompt = format_overview_prompt(
            repo_name=repo_name,
            component_count=component_count,
            module_count=module_count,
            module_dependencies=module_dependencies,
            module_docs_summary="\n\n".join(
                f"### {name}\n{content}" for name, content in module_docs_summary.items()
            ),
        )

        result = await self._call_llm(llm_tool, system_prompt, user_prompt)
        if not result.success:
            return result
        output_path = input_data.get("output_path", "docs/README.md")
        return await self._write_result(output_path, result.output, "overview", file_tool)

    async def _generate_component_doc(self, input_data: dict, llm_tool, file_tool) -> SkillResult:
        component = input_data.get("component", {})
        predecessors = input_data.get("predecessors", [])
        module_name = input_data.get("module_name", "")

        system_prompt, user_prompt = format_component_prompt(
            component_name=component.get("name", "Unknown"),
            component_type=component.get("component_type", "unknown"),
            file_path=component.get("file_path", ""),
            docstring=component.get("docstring", ""),
            source_code=component.get("source_code", ""),
            dependencies=list(component.get("depends_on", set())),
            predecessors=list(predecessors),
            module_name=module_name,
        )

        result = await self._call_llm(llm_tool, system_prompt, user_prompt)
        if not result.success:
            return result
        output_path = input_data.get(
            "output_path", f"docs/{component.get('relative_path', 'component')}.md"
        )
        return await self._write_result(output_path, result.output, "component", file_tool)

    async def _generate_module_doc(self, input_data: dict, llm_tool, file_tool) -> SkillResult:
        module_name = input_data.get("module_name", "Module")
        module_path = input_data.get("module_path", "")

        system_prompt, user_prompt = format_module_prompt(
            module_name=module_name,
            module_path=module_path,
            parent_module=input_data.get("parent_module", ""),
            sibling_modules=input_data.get("sibling_modules", []),
            module_dependencies=input_data.get("module_dependencies", []),
            dependent_modules=input_data.get("dependent_modules", []),
            sub_modules=input_data.get("sub_modules", []),
            # Already budgeted strings (budget_context) — never raw dicts.
            component_docs_summary=input_data.get("component_docs", ""),
            children_docs_summary=input_data.get("children_docs", ""),
        )

        result = await self._call_llm(llm_tool, system_prompt, user_prompt)
        if not result.success:
            return result
        output_path = input_data.get("output_path", f"docs/modules/{module_name}.md")
        return await self._write_result(output_path, result.output, "module", file_tool)

    async def _generate_direct(self, doc_type: str, input_data: dict, llm_tool, file_tool) -> SkillResult:
        """Unified single-prompt path (ablation baseline)."""
        import json as _json

        context = _json.dumps(input_data, ensure_ascii=False, default=str)[:24000]
        prompt = DIRECT_DOC_PROMPT.format(doc_type=doc_type, context=context)

        result = await self._call_llm(llm_tool, "", prompt)
        if not result.success:
            return result
        output_path = input_data.get("output_path", f"docs/{doc_type}.md")
        return await self._write_result(output_path, result.output, doc_type, file_tool)
