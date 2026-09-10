"""Prompts for architecture-narrative documentation.

Three layers, generated bottom-up:
1. Component docs — role/cooperation in the architecture, length adapts to
   the component's architectural importance (no API-manual tables).
2. Module docs — responsibilities/boundaries, position between upstream and
   downstream modules, member role list (from actual component docs).
3. Overview README — architecture overview with an embedded module-level
   mermaid diagram, theme map, and core flows.
"""

# ---------------------------------------------------------------------------
# Layer 1: Component docs (architecture role, adaptive length)
# ---------------------------------------------------------------------------

COMPONENT_SYSTEM_PROMPT = """You are an expert software architect writing documentation for a code repository.

Your documentation is READ by developers to understand the repository's architecture:
how components cooperate, what roles they play in flows, and why things are
structured the way they are. It is NOT an API manual.

Write in clear, precise technical prose in the repository's primary language
(English unless the context is clearly otherwise). Ground everything in the
provided context — never invent components, calls, or behaviors."""

COMPONENT_USER_PROMPT = """Describe the role of this component in the repository's architecture.

Component: {component_name}
Type: {component_type}
File: {file_path}
Belongs to module: {module_name}

{component_context}

## What to write

- **Architectural role**: what this component is for, and why it exists in the
  architecture (its responsibility at a high level — NOT an API listing)
- **Cooperation**: which components it calls and which components call it
  (use the dependency lists below); what flows it participates in
- **Position in flows**: entry points it serves, or outputs it produces

## Length: ADAPT to architectural importance

- **Architecturally important** (a core class, a flow hub, widely referenced):
  write a thorough section — role, cooperation, flows; a small mermaid
  diagram is welcome when it clarifies cooperation.
- **Secondary components** (config holders, small helpers, internal
  utilities): 1-3 short paragraphs. A short "Role" line plus key points is
  enough. Do NOT pad.

## Rules

- NO method/API tables. NO full API listings. Describe behavior and role.
- Never invent dependencies or flows not present in the context.
- Return ONLY the Markdown document content.

Component: {component_name}
Calls: {dependencies}
Called by: {predecessors}
Source:
```
{source_code}
```
Docstring: {docstring}"""


def format_component_prompt(
    component_name: str,
    component_type: str,
    file_path: str,
    docstring: str,
    source_code: str,
    dependencies: list[str],
    predecessors: list[str],
    module_name: str,
    component_context: str = "",
) -> tuple[str, str]:
    user_prompt = COMPONENT_USER_PROMPT.format(
        component_name=component_name,
        component_type=component_type,
        file_path=file_path,
        docstring=docstring or "No documentation provided.",
        source_code=source_code[:24000],
        dependencies=", ".join(dependencies) if dependencies else "None",
        predecessors=", ".join(predecessors) if predecessors else "None",
        module_name=module_name,
        component_context=component_context,
    )
    return COMPONENT_SYSTEM_PROMPT, user_prompt


# ---------------------------------------------------------------------------
# Layer 2: Module docs (architecture narrative per module-tree node)
# ---------------------------------------------------------------------------

MODULE_SYSTEM_PROMPT = """You are an expert software architect documenting a code repository.

Your documentation is READ by developers to understand the repository's
architecture. Describe modules at the ABSTRACT level: responsibilities,
boundaries, cooperation with other modules, and flows. Do NOT write API
manuals — no method listings, no API tables.

Ground everything in the provided context (module info, member component
docs, sub-module docs, module dependencies). Never invent modules, flows, or
dependencies."""

MODULE_USER_PROMPT = """Write the architecture-narrative documentation for the module `{module_name}`.

Module path: {module_path}
Parent module: {parent_module}
Sibling modules: {sibling_modules}

## Modules it depends on (calls)
{module_dependencies}

## Modules that depend on it (are called by)
{dependent_modules}

## Sub-modules
{sub_modules}

## Member components and their roles (from their component docs)
{component_docs_summary}

## Sub-module docs (actual content of child module docs)
{children_docs_summary}

## What to write

1. **Module purpose**: what this module is responsible for, and its boundary
   (what it deliberately does NOT do)
2. **Architectural position**: which modules feed it (upstream), which it
   serves (downstream), and what the dependency direction implies
3. **Member components**: the role list above, woven into a short narrative —
   which members are the core, which are supporting
4. **Sub-modules**: one line each, linking to their docs
5. **Key flows/cooperation**: 1-3 flows this module participates in, if the
   context supports them

Return ONLY the Markdown document content. Use relative markdown links to
member component docs (`{module_name}/ComponentName.md`) and sub-module docs
(`sub_module.md` for children)."""


def format_module_prompt(
    module_name: str,
    module_path: str,
    parent_module: str,
    sibling_modules: list[str],
    module_dependencies: list[str],
    dependent_modules: list[str],
    sub_modules: list[str],
    component_docs_summary: str,
    children_docs_summary: str,
) -> tuple[str, str]:
    user_prompt = MODULE_USER_PROMPT.format(
        module_name=module_name,
        module_path=module_path or ".",
        parent_module=parent_module or "(root)",
        sibling_modules=", ".join(sibling_modules) if sibling_modules else "None",
        module_dependencies=", ".join(module_dependencies) if module_dependencies else "None",
        dependent_modules=", ".join(dependent_modules) if dependent_modules else "None",
        sub_modules="\n".join(f"- {s}" for s in sub_modules) if sub_modules else "None",
        component_docs_summary=component_docs_summary or "No member component docs available",
        children_docs_summary=children_docs_summary or "No sub-module docs available",
    )
    return MODULE_SYSTEM_PROMPT, user_prompt


# ---------------------------------------------------------------------------
# Layer 3: Overview README (architecture overview with module-level diagram)
# ---------------------------------------------------------------------------

OVERVIEW_SYSTEM_PROMPT = """You are an expert software architect writing the top-level architecture
document (README) of a code repository.

The README is the ENTRY POINT for developers. It must explain the
repository's architecture at the abstract level: what the system does, how
its modules fit together, and which flows connect them. It is NOT an API
manual and NOT a changelog.

Ground everything in the provided context: module docs (actual generated
content), the module dependency map, and repository statistics. Never invent
modules, flows, or capabilities."""

OVERVIEW_USER_PROMPT = """Write the architecture overview (README.md) for the repository.

Repository: {repo_name}
Total components: {component_count}
Modules: {module_count}

## Module dependency map (module -> modules it depends on)
{module_dependencies}

## Module docs (actual content, one section per module)
{module_docs_summary}

## What to write

1. **Project positioning**: one clear paragraph — what this system is and
   what problem it solves
2. **Architecture overview**: describe the module layout (layers/domains),
   then include a **mermaid flowchart** showing modules and their dependency
   edges (use the dependency map above; every module must be a node)
3. **Theme map**: the top-level modules, one line of responsibility each,
   linking to each module doc (`module_name.md`)
4. **Core flows**: 2-4 key flows through the system (entry point -> modules
   involved -> outcome), based on the dependency map
5. **Module navigation**: a short list of all top-level module docs

Return ONLY the Markdown document content. Mermaid diagrams must be valid
(flowchart TD with `-->` edges)."""


def format_overview_prompt(
    repo_name: str,
    component_count: int,
    module_count: int,
    module_dependencies: dict[str, list[str]],
    module_docs_summary: str,
) -> tuple[str, str]:
    deps_str = "\n".join(
        f"- {m} -> {', '.join(ds) if ds else '(none)'}"
        for m, ds in sorted(module_dependencies.items())
    )
    user_prompt = OVERVIEW_USER_PROMPT.format(
        repo_name=repo_name,
        component_count=component_count,
        module_count=module_count,
        module_dependencies=deps_str or "None",
        module_docs_summary=module_docs_summary or "No module docs available",
    )
    return OVERVIEW_SYSTEM_PROMPT, user_prompt
