DOC_WRITER_SYSTEM_PROMPT = """You are an expert technical documentation writer. Your task is to generate comprehensive documentation for code components based on the provided analysis results.

## Role
You are an AI documentation assistant specializing in creating detailed, accurate technical documentation.

## Objectives
Create documentation that helps developers and maintainers understand:
1. The component's purpose and core functionality
2. Architecture and component relationships
3. How the component fits into the overall system
4. API reference with all public methods and attributes
5. Best practices and usage patterns
6. Common issues and troubleshooting

## Documentation Structure
Generate documentation with the following sections:
1. **Overview**: Brief introduction and purpose
2. **Key Responsibilities**: Core duties and features
3. **Constructor/Init Parameters**: All parameters with types, defaults, and descriptions
4. **Public Methods**: All public methods with descriptions - use a markdown TABLE format with columns: Method | Description | Parameters | Returns
5. **Attributes/Properties**: All attributes with types - use TABLE format
6. **Usage Examples**: Multiple code examples showing common use cases - each example must be runnable
7. **Best Practices**: Recommended patterns for using this component
8. **Common Issues**: Potential problems and solutions
9. **Integration Points**: How this component interacts with others
10. **Related Modules**: Links to related documentation

## Output Format Requirements
- Use markdown TABLE format for all API listings
- Include COMPLETE method signatures with all parameters
- Code examples must be complete and runnable
- Include import statements in code examples
- Add error handling examples showing common exceptions

## Guidelines
1. Write comprehensive documentation - do not be brief
2. Include ALL public API elements (methods, attributes, properties)
3. Provide multiple usage examples (at least 3 different use cases)
4. Use Markdown formatting for structure and readability
5. Focus on the "why" not just the "what"
6. Extract business concepts and design decisions from the code
7. Be detailed - aim for thorough coverage"""


DOC_WRITER_USER_PROMPT = """Generate comprehensive documentation for the following code component:

Component Name: {component_name}
Component Type: {component_type}
File Path: {file_path}
Docstring: {docstring}

Source Code:
{source_code}

Dependencies: {dependencies}

Module Context: {module_context}

Please generate detailed documentation following the structure in the system prompt. Include all public methods, attributes, parameters, and provide multiple usage examples."""


def format_doc_prompt(
    component_name: str,
    component_type: str,
    file_path: str,
    docstring: str,
    source_code: str,
    dependencies: list[str],
    module_context: str = "",
) -> tuple[str, str]:
    user_prompt = DOC_WRITER_USER_PROMPT.format(
        component_name=component_name,
        component_type=component_type,
        file_path=file_path,
        docstring=docstring or "No documentation provided.",
        source_code=source_code[:24000],
        dependencies=", ".join(dependencies) if dependencies else "None",
        module_context=module_context,
    )
    return DOC_WRITER_SYSTEM_PROMPT, user_prompt


REPO_OVERVIEW_PROMPT = """Generate a comprehensive repository overview document based on the following analysis:

Repository Name: {repo_name}
Total Components: {component_count}
Module Structure: {module_structure}

Code Analysis Statistics:
- Component Types: {component_types}
- Language Distribution: {language_counts}
- Total Files Analyzed: {total_files}
- Top Dependencies: {top_dependencies}

Top-level Modules:
{modules_summary}

## IMPORTANT: Available Modules for Cross-References
Only link to modules that exist in this list:
{available_modules}

When linking to modules, use the format:
- [ModuleName](module_name.md) - for top-level modules in ROOT directory

Please generate a well-structured README that includes:

1. **Repository Overview**:
   - Explain the repository purpose based on the code analysis
   - Describe the main functionality and what this project does

2. **Architecture Overview**:
   - Include a Mermaid diagram showing high-level architecture
   - Explain the main components and their relationships

3. **Module Documentation**:
   - List all main modules and their responsibilities
   - Link to each module's documentation: [ModuleName](module_name.md)

4. **Key Features**:
   - List the main features discovered from code analysis
   - Include technology stack and dependencies

5. **Getting Started**:
   - Installation/setup instructions if inferrable from code
   - Basic usage examples

6. **Project Structure**:
   - Directory structure overview
   - Key files and their purposes

7. **Related Documentation**:
   - Links to external resources (official docs, etc. if found in code)

Generate a comprehensive README that serves as the entry point for this repository's documentation."""


MODULE_OVERVIEW_PROMPT = """You are an AI documentation assistant. Your task is to generate a brief overview of `{module_name}` module.

## IMPORTANT: Available Links for Cross-References
Only link to modules and components that exist in this list:

Available ROOT modules: {available_root_modules}

Available components in THIS module ({module_name}): {available_components}

When linking:
- Use [ModuleName](module_name.md) for root-level modules (in ROOT directory)
- Use [ComponentName](module_name/ComponentName.md) for components in module subdirectory
  - Example: For Flask component in app module, use [Flask](app/Flask.md)
- DO NOT link to modules or components not in the above lists

Generate documentation following this structure:

1. **Module Overview**:
   - Brief introduction and purpose
   - Architecture overview with Mermaid diagrams
   - High-level functionality of each sub-module including references to its documentation file

2. **API Reference**:
   - Use TABLE format for all classes, methods, functions
   - Include parameter types, return types, descriptions
   - Group by class/namespace

3. **Usage Examples**:
   - At least 3 runnable code examples
   - Include import statements
   - Show common use cases

4. **Best Practices**:
   - Recommended patterns
   - Common pitfalls to avoid
   - Performance tips

5. **Common Issues & Troubleshooting**:
   - List common errors and their solutions
   - Debug tips

6. **Sub-module Documentation** (if applicable):
   - Detailed descriptions of each sub-module saved in the working directory under the name of `sub-module_name.md`
   - Core components and their responsibilities

7. **Child Module Documentation** (summaries from children's docs):
   {children_docs_summary}

8. **Visual Documentation**:
   - Mermaid diagrams for architecture, dependencies, and data flow
   - Component interaction diagrams
   - Process flow diagrams where relevant

9. **Cross-references**:
   - Link to other module documentation using [ModuleName](module_name.md) format instead of duplicating information
   - Link to component documentation in the same module directory using [ComponentName](ComponentName.md) format

10. **Related Modules**:
    - Link to related module documentation

Module Name: {module_name}
Module Path: {module_path}
Components: {components}
Component Documentation (use these as reference):
{component_docs}
Sub-modules: {sub_modules}
Children: {children}

Generate the module documentation now. Include Mermaid diagrams where appropriate."""





def format_overview_prompt(
    repo_name: str,
    component_count: int,
    module_structure: dict,
    code_statistics: dict = None,
    modules_summary: str = "",
    available_modules: list[str] = None,
) -> tuple[str, str]:
    comp_types = code_statistics.get("component_types", {}) if code_statistics else {}
    lang_counts = code_statistics.get("language_counts", {}) if code_statistics else {}
    total_files = code_statistics.get("total_files", 0) if code_statistics else 0
    top_deps = code_statistics.get("top_dependencies", []) if code_statistics else []

    avail_modules_str = (
        ", ".join(available_modules)
        if available_modules
        else "None (no modules available yet)"
    )

    user_prompt = REPO_OVERVIEW_PROMPT.format(
        repo_name=repo_name,
        component_count=component_count,
        component_types=", ".join([f"{k}: {v}" for k, v in comp_types.items()])
        or "N/A",
        language_counts=", ".join([f"{k}: {v}" for k, v in lang_counts.items()])
        or "N/A",
        total_files=total_files,
        top_dependencies=", ".join([d[0] for d in top_deps]) if top_deps else "N/A",
        module_structure=str(module_structure)[:1000],
        modules_summary=(
            modules_summary[:2000] if modules_summary else "No modules defined yet"
        ),
        available_modules=avail_modules_str,
    )
    return DOC_WRITER_SYSTEM_PROMPT, user_prompt


# Mermaid diagram prompts
DIAGRAM_SYSTEM_PROMPT = """You are an expert software architect. Your task is to generate Mermaid diagrams that visualize code structure and relationships.

## Guidelines
1. Generate syntactically correct Mermaid diagram code
2. Use appropriate diagram type for the information:
   - `flowchart TD` for process flows
   - `graph TD` for dependencies
   - `classDiagram` for class structures
   - `sequenceDiagram` for interactions
   - `stateDiagram-v2` for state machines
3. Include meaningful labels and descriptions
4. Keep diagrams focused and not overly complex"""

DIAGRAM_ARCHITECTURE_PROMPT = """Generate a Mermaid flowchart diagram showing the architecture of this module.

Module Name: {module_name}
Components: {components}
Dependencies: {dependencies}

Output a Mermaid flowchart (use `flowchart TD` or `graph TD`) that shows:
1. Main components and their relationships
2. Data flow direction
3. Key interfaces

Return ONLY the Mermaid code block, no additional text."""

def format_architecture_diagram_prompt(
    module_name: str,
    components: list[str],
    dependencies: dict[str, list[str]],
) -> tuple[str, str]:
    deps_str = "\n".join(
        [f"- {k} -> {v}" for k, vs in dependencies.items() for v in vs]
    )
    user_prompt = DIAGRAM_ARCHITECTURE_PROMPT.format(
        module_name=module_name,
        components="\n".join([f"- {c}" for c in components]),
        dependencies=deps_str or "None",
    )
    return DIAGRAM_SYSTEM_PROMPT, user_prompt


def format_module_prompt(
    module_name: str,
    module_path: str,
    components: list[str],
    sub_modules: list[str],
    children: dict,
    children_docs: dict[str, str] = None,
    component_docs: dict[str, str] = None,
    available_root_modules: list[str] = None,
    available_components: list[str] = None,
) -> tuple[str, str]:
    children_docs_summary = "No child documentation available"
    if children_docs:
        docs_parts = []
        for child_name, doc_content in children_docs.items():
            summary = (
                doc_content[:300] + "..." if len(doc_content) > 300 else doc_content
            )
            docs_parts.append(f"   - **{child_name}**: {summary}")
        children_docs_summary = "\n".join(docs_parts)

    component_docs_summary = "No component documentation available"
    if component_docs:
        docs_parts = []
        for comp_name, doc_content in component_docs.items():
            summary = (
                doc_content[:500] + "..." if len(doc_content) > 500 else doc_content
            )
            docs_parts.append(f"### {comp_name}\n{summary}")
        component_docs_summary = "\n\n".join(docs_parts)

    avail_root = ", ".join(available_root_modules) if available_root_modules else "None"
    avail_comps = ", ".join(available_components) if available_components else "None"

    user_prompt = MODULE_OVERVIEW_PROMPT.format(
        module_name=module_name,
        module_path=module_path,
        components="\n".join([f"- {c}" for c in components]) if components else "None",
        component_docs=component_docs_summary,
        sub_modules=(
            "\n".join([f"- {s}" for s in sub_modules]) if sub_modules else "None"
        ),
        children=str(children)[:1000] if children else "None",
        children_docs_summary=children_docs_summary,
        available_root_modules=avail_root,
        available_components=avail_comps,
    )
    return DOC_WRITER_SYSTEM_PROMPT, user_prompt
