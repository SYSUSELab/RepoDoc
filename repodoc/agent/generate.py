import asyncio
import os
import subprocess
import time
import glob
from datetime import datetime
from typing import Any

from repodoc.config import Config
from repodoc.graph import DependencyGraphBuilder
from repodoc.graph.builder import build_graph_from_analysis
from repodoc.agent.agent import DocAgent
from repodoc.utils import file_manager, validate_and_fix_links, log_operation


def flatten_module_tree_leaf_first(
    tree: dict[str, Any], parent_path: str = ""
) -> list[tuple[str, dict[str, Any], str]]:
    """Flatten module tree into leaf-first ordered list.

    Returns list of (module_name, module_info, full_path) tuples.
    Leaves are returned before their parents.
    """
    result = []

    def _collect(
        nodes: dict[str, Any], path: str
    ) -> list[tuple[str, dict[str, Any], str]]:
        collected = []
        for name, info in nodes.items():
            full_path = f"{path}/{name}" if path else name
            children = info.get("children", {})
            if children:
                collected.extend(_collect(children, full_path))
            collected.append((name, info, full_path))
        return collected

    return _collect(tree, parent_path)


def group_modules_by_depth(
    tree: dict[str, Any], parent_path: str = ""
) -> dict[int, list[tuple[str, dict[str, Any], str]]]:
    """Group modules by depth level for parallel processing.

    Returns dict of depth -> list of (module_name, module_info, full_path) tuples.
    Depth 0 = leaf modules (no children), higher depths = parent modules.
    """
    depth_groups = {}

    def _collect(nodes: dict[str, Any], path: str, depth: int):
        for name, info in nodes.items():
            full_path = f"{path}/{name}" if path else name
            children = info.get("children", {})
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append((name, info, full_path))
            if children:
                _collect(children, full_path, depth + 1)

    _collect(tree, parent_path, 0)
    return depth_groups


def get_module_depth(
    tree: dict[str, Any], module_name: str, current_depth: int = 0
) -> int:
    """Get the depth of a module in the tree (0 = top-level)."""
    for name, info in tree.items():
        if name == module_name:
            return current_depth
        children = info.get("children", {})
        if children:
            depth = get_module_depth(children, module_name, current_depth + 1)
            if depth >= 0:
                return depth
    return -1


class DocPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.agent = None
        self.components = {}
        self.leaf_nodes = []
        self.graph = None
        self.module_tree = {}
        self._start_time = None
        self._generated_files = []

    def get_data_dir(self) -> str:
        return os.path.join(self.config.output_dir, "data")

    def get_project_name(self) -> str:
        repo_name = os.path.basename(os.path.normpath(self.config.repo_path))
        return "".join(c if c.isalnum() else "_" for c in repo_name)

    def run_analysis(self):
        builder = DependencyGraphBuilder(self.config)
        self.components, self.leaf_nodes = builder.build_dependency_graph()
        self.graph = build_graph_from_analysis(self.components)
        return self.components, self.leaf_nodes

    def load_module_tree(self):
        project_name = self.get_project_name()
        data_dir = self.get_data_dir()
        new_path = os.path.join(data_dir, f"{project_name}_module_tree.json")
        old_docs_path = os.path.join(self.config.docs_dir, "module_tree.json")

        if os.path.exists(new_path):
            self.module_tree = file_manager.load_json(new_path) or {}
        elif os.path.exists(old_docs_path):
            self.module_tree = file_manager.load_json(old_docs_path) or {}
        else:
            self.module_tree = {}
        return self.module_tree

    def initialize_agent(self):
        self.agent = DocAgent(self.config)
        return self.agent

    def get_code_statistics(self) -> dict[str, Any]:
        """Extract code analysis statistics from components."""
        stats = {
            "total_components": len(self.components),
            "component_types": {},
            "language_counts": {},
            "top_dependencies": [],
            "total_files": set(),
            "modules_with_components": {},
        }

        for comp_id, node in self.components.items():
            ctype = node.component_type or "unknown"
            stats["component_types"][ctype] = stats["component_types"].get(ctype, 0) + 1

            if hasattr(node, "file_path") and node.file_path:
                ext = os.path.splitext(node.file_path)[1]
                if ext:
                    stats["language_counts"][ext] = (
                        stats["language_counts"].get(ext, 0) + 1
                    )
                stats["total_files"].add(node.file_path)

            if hasattr(node, "depends_on") and node.depends_on:
                deps_list = list(node.depends_on)[:3]
                for dep in deps_list:
                    if dep not in [d[0] for d in stats["top_dependencies"]]:
                        stats["top_dependencies"].append((dep, 1))
                    else:
                        for i, (d, c) in enumerate(stats["top_dependencies"]):
                            if d == dep:
                                stats["top_dependencies"][i] = (d, c + 1)
                                break

        stats["top_dependencies"] = sorted(
            stats["top_dependencies"], key=lambda x: x[1], reverse=True
        )[:10]
        stats["total_files"] = len(stats["total_files"])

        return stats

    async def generate_overview(self, output_path: str = None):
        if not self.agent:
            self.initialize_agent()

        module_structure = self.load_module_tree()
        code_stats = self.get_code_statistics()
        available_modules = list(module_structure.keys()) if module_structure else []

        result = await self.agent.execute(
            {
                "type": "write_doc",
                "config": self.config,
                "input": {
                    "type": "overview",
                    "repo_name": os.path.basename(self.config.repo_path),
                    "component_count": len(self.components),
                    "module_structure": module_structure,
                    "code_statistics": code_stats,
                    "available_modules": available_modules,
                    "output_path": output_path
                    or os.path.join(self.config.docs_dir, "README.md"),
                },
            }
        )
        return result

    async def generate_module_docs(
        self,
        module_tree: dict = None,
        parent_dir: str = None,
        children_docs: dict[str, str] = None,
    ):
        """Generate module documentation with parallel processing by depth level.

        Modules at the same depth level are processed in parallel.
        Parent modules wait for their children to complete before generation.
        """
        if not self.agent:
            self.initialize_agent()

        if module_tree is None:
            module_tree = self.load_module_tree()

        if parent_dir is None:
            parent_dir = self.config.docs_dir

        if children_docs is None:
            children_docs = {}

        depth_groups = group_modules_by_depth(module_tree)

        results = []

        for depth in sorted(depth_groups.keys()):
            modules_at_depth = depth_groups[depth]

            tasks = []
            for module_name, module_info, full_path in modules_at_depth:
                task = self._generate_single_module_doc(
                    module_name, module_info, full_path, module_tree
                )
                tasks.append(task)

            depth_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend([r for r in depth_results if not isinstance(r, Exception)])

        return results

    async def _generate_single_module_doc(
        self,
        module_name: str,
        module_info: dict,
        full_path: str,
        module_tree: dict,
    ):
        """Generate documentation for a single module."""
        components = module_info.get("components", [])
        children = module_info.get("children", {})
        path_parts = full_path.split("/")
        current_parent_dir = self.config.docs_dir

        for part in path_parts[:-1]:
            current_parent_dir = os.path.join(current_parent_dir, part)

        module_dir = os.path.join(current_parent_dir, module_name)
        if children:
            file_manager.ensure_directory(module_dir)

        child_docs_content = {}
        for child_name in children.keys():
            child_doc_path = os.path.join(module_dir, f"{child_name}.md")
            if os.path.exists(child_doc_path):
                try:
                    with open(child_doc_path, "r") as f:
                        child_docs_content[child_name] = f.read()[:2000]
                except:
                    pass

        component_docs_content = {}
        component_docs_dir = os.path.join(self.config.docs_dir, module_name)
        for comp_id in components:
            node = self.components.get(comp_id)
            if node and node.name:
                comp_doc_path = os.path.join(component_docs_dir, f"{node.name}.md")
                if os.path.exists(comp_doc_path):
                    try:
                        with open(comp_doc_path, "r") as f:
                            component_docs_content[node.name] = f.read()[:3000]
                    except Exception:
                        pass

        module_components = []
        for comp_id in components:
            node = self.components.get(comp_id)
            if node:
                module_components.append(
                    {
                        "name": node.name,
                        "component_type": node.component_type,
                        "docstring": node.docstring,
                        "file_path": node.file_path,
                    }
                )

        available_root_modules = list(module_tree.keys()) if module_tree else []
        available_components = [
            f"{module_name}/{node.name}"
            for node in [self.components.get(comp_id) for comp_id in components]
            if node and node.name
        ]

        result = await self.agent.execute(
            {
                "type": "write_doc",
                "config": self.config,
                "input": {
                    "type": "module",
                    "module_name": module_name,
                    "module_path": module_info.get("path", module_name),
                    "components": module_components,
                    "component_docs": component_docs_content,
                    "sub_modules": list(children.keys()),
                    "children": children,
                    "children_docs": child_docs_content,
                    "available_root_modules": available_root_modules,
                    "available_components": available_components,
                    "output_path": os.path.join(
                        current_parent_dir, f"{module_name}.md"
                    ),
                },
            }
        )
        return result

    async def _generate_single_component(
        self, node_id: str, output_dir: str, module_name: str
    ) -> dict:
        node = self.components.get(node_id)
        if not node:
            return None

        component = {
            "name": node.name,
            "component_type": node.component_type,
            "file_path": node.file_path,
            "relative_path": node.relative_path,
            "docstring": node.docstring,
            "source_code": node.source_code or "",
            "depends_on": node.depends_on,
        }

        result = await self.agent.execute(
            {
                "type": "write_doc",
                "config": self.config,
                "input": {
                    "type": "component",
                    "component": component,
                    "output_path": os.path.join(output_dir, f"{node.name}.md"),
                },
            }
        )
        return result

    async def generate_component_docs(
        self, docs_dir: str = None, module_tree: dict = None
    ):
        if not self.agent:
            self.initialize_agent()

        docs_dir = docs_dir or self.config.docs_dir

        if not module_tree:
            module_tree = self.load_module_tree()

        module_components_map = {}
        for module_name, module_info in module_tree.items():
            for comp_id in module_info.get("components", []):
                node = self.components.get(comp_id)
                if node and node.name:
                    if module_name not in module_components_map:
                        module_components_map[module_name] = []
                    module_components_map[module_name].append(comp_id)

        sem = asyncio.Semaphore(16)

        async def _run(comp_id: str, module_dir: str, module_name: str):
            async with sem:
                return await self._generate_single_component(
                    comp_id, module_dir, module_name
                )

        tasks = []
        for module_name, comp_ids in module_components_map.items():
            module_dir = os.path.join(docs_dir, module_name)
            file_manager.ensure_directory(module_dir)
            for comp_id in comp_ids:
                tasks.append(_run(comp_id, module_dir, module_name))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if r is not None and not isinstance(r, Exception)]

    async def generate_architecture_diagrams(self, output_dir: str = None):
        """Generate Mermaid diagrams for architecture and dependency visualization."""
        if not self.agent:
            self.initialize_agent()

        output_dir = output_dir or self.config.docs_dir
        file_manager.ensure_directory(output_dir)

        project_name = self.get_project_name()

        module_tree = self.load_module_tree()

        all_components = []
        dependencies = {}

        for comp_id, node in self.components.items():
            all_components.append(
                {
                    "name": node.name,
                    "component_type": node.component_type,
                }
            )
            if hasattr(node, "depends_on") and node.depends_on:
                dependencies[node.name] = list(node.depends_on)[:5]

        diagram_result = await self.agent.execute(
            {
                "type": "write_doc",
                "config": self.config,
                "input": {
                    "type": "architecture_diagram",
                    "module_name": os.path.basename(self.config.repo_path),
                    "components": all_components[:20],
                    "dependencies": dependencies,
                    "output_path": os.path.join(
                        output_dir, f"{project_name}_architecture.mmd"
                    ),
                },
            }
        )

        if diagram_result.success and hasattr(diagram_result, "output"):
            diagram_path = os.path.join(output_dir, f"{project_name}_architecture.mmd")
            diagram_content = diagram_result.output.get("diagram", "")

            # Validate Mermaid diagram
            from repodoc.utils import validate_mermaid_syntax

            is_valid, error_msg = validate_mermaid_syntax(diagram_content)
            if not is_valid:
                # Generate a simpler fallback diagram
                diagram_content = self._generate_fallback_diagram(
                    all_components, dependencies
                )

            with open(diagram_path, "w") as f:
                f.write(diagram_content)

        return diagram_result

    def _generate_fallback_diagram(self, components: list, dependencies: dict) -> str:
        """Generate a simple fallback Mermaid diagram if validation fails."""
        lines = ["```mermaid", "flowchart TD", "    %% Fallback architecture diagram"]

        # Add components as nodes
        for comp in components[:15]:
            node_id = comp["name"].replace(".", "_").replace("-", "_")
            lines.append(f"    {node_id}[{comp['name']}]")

        # Add dependencies as edges
        seen = set()
        for comp_name, deps in dependencies.items():
            source_id = comp_name.replace(".", "_").replace("-", "_")
            for dep in deps[:3]:
                dep_id = dep.replace(".", "_").replace("-", "_")
                edge_key = f"{source_id}--> {dep_id}"
                if edge_key not in seen:
                    seen.add(edge_key)
                    lines.append(f"    {source_id} --> {dep_id}")

        lines.append("```")
        return "\n".join(lines)

    def update_graph_with_docs(self, data_dir: str, docs_dir: str = None):
        if not self.graph:
            return

        if docs_dir is None:
            docs_dir = os.path.join(self.config.output_dir, "docs")

        from repodoc.graph.models import DocNode, REL_DESCRIBES

        overview_path = os.path.join(docs_dir, "README.md")
        if os.path.exists(overview_path):
            with open(overview_path) as f:
                content = f.read()
            doc_node = DocNode(id="overview", content=content, format="markdown")
            self.graph.add_doc_node(doc_node)

        for root, _, files in os.walk(docs_dir):
            for fname in files:
                if fname.endswith(".md") and not fname.startswith("_"):
                    fpath = os.path.join(root, fname)
                    rel_dir = os.path.relpath(os.path.dirname(fpath), docs_dir)
                    if rel_dir == ".":
                        continue
                    with open(fpath) as f:
                        content = f.read()
                    doc_id = f"doc/{rel_dir}/{fname[:-3]}"
                    doc_node = DocNode(
                        id=doc_id, content=content, format="markdown", file_path=fpath
                    )
                    self.graph.add_doc_node(doc_node)

                    for comp_id, node in self.components.items():
                        if node.name and node.name.lower() in fname.lower().replace(
                            ".md", ""
                        ).replace("_", " ").replace("-", " "):
                            self.graph.add_edge(doc_id, comp_id, REL_DESCRIBES, 1.0)
                            break

        project_name = self.get_project_name()
        kg_path = os.path.join(data_dir, f"{project_name}_knowledge_graph.json")
        self.graph.save(kg_path)

    def _collect_generated_files(self, docs_dir: str, data_dir: str) -> list[str]:
        files = []
        for ext in ["*.md", "*.json", "*.mmd"]:
            for f in glob.glob(os.path.join(docs_dir, "**", ext), recursive=True):
                rel = os.path.relpath(f, docs_dir)
                files.append(rel)
            for f in glob.glob(os.path.join(data_dir, "**", ext), recursive=True):
                rel = os.path.relpath(f, data_dir)
                if rel not in files:
                    files.append(rel)
        return sorted(files)

    async def _generate_module_docs_safe(self):
        try:
            return await self.generate_module_docs()
        except Exception as e:
            print(f"Warning: Module docs generation failed: {e}")
            return []

    async def _generate_overview_safe(self, output_path: str):
        try:
            return await self.generate_overview(output_path)
        except Exception as e:
            print(f"Warning: Overview generation failed: {e}")
            return None

    async def _generate_diagram_safe(self, output_dir: str):
        try:
            return await self.generate_architecture_diagrams(output_dir)
        except Exception as e:
            print(f"Warning: Architecture diagram generation failed: {e}")
            return None

    async def generate_all(self, docs_dir: str = None, data_dir: str = None):
        docs_dir = docs_dir or self.config.docs_dir
        data_dir = data_dir or self.get_data_dir()
        file_manager.ensure_directory(docs_dir)
        file_manager.ensure_directory(data_dir)

        self._start_time = time.time()
        self._generated_files = []

        module_tree = self.load_module_tree()

        component_results = await self.generate_component_docs(
            docs_dir=docs_dir, module_tree=module_tree
        )

        module_results, overview_result, diagram_results = await asyncio.gather(
            self._generate_module_docs_safe(),
            self._generate_overview_safe(os.path.join(docs_dir, "README.md")),
            self._generate_diagram_safe(data_dir),
        )

        self._update_graph_with_docs_safe(data_dir, docs_dir)
        self._extract_concepts_safe(data_dir)

        link_results = validate_and_fix_links(docs_dir)
        all_files = self._collect_generated_files(docs_dir, data_dir)
        total_duration = time.time() - self._start_time if self._start_time else 0

        token_usage = self._collect_token_usage()
        git_commit = self._get_git_commit()

        self._save_and_log_metadata(
            data_dir, all_files, total_duration, token_usage, git_commit, link_results
        )

        return {
            "overview": overview_result,
            "modules": module_results,
            "components": component_results,
            "diagrams": diagram_results,
            "total_components": len(self.components),
            "docs_directory": docs_dir,
        }

    def _update_graph_with_docs_safe(self, data_dir: str, docs_dir: str) -> None:
        """Update graph with docs, ignoring errors."""
        try:
            self.update_graph_with_docs(data_dir, docs_dir)
        except Exception as e:
            print(f"Warning: Graph update failed: {e}")

    def _extract_concepts_safe(self, data_dir: str) -> None:
        """Extract concepts into the graph (keyword or LLM-enhanced), ignoring errors."""
        try:
            from repodoc.graph.concepts import extract_concepts

            extract_concepts(self.graph, self.components, self.config, data_dir)
        except Exception as e:
            print(f"Warning: Concept extraction failed: {e}")

    def _collect_token_usage(self) -> dict[str, Any]:
        """Collect token usage from the shared LLM client (covers clustering,
        concept extraction, and doc generation)."""
        from repodoc.llm import total_token_usage

        usage = total_token_usage(self.config)
        return {
            "history": usage["history"],
            "total": usage["total"],
            "prompt": usage["prompt"],
            "completion": usage["completion"],
        }

    def _get_git_commit(self) -> str | None:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.config.repo_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _save_and_log_metadata(
        self,
        data_dir: str,
        all_files: list[str],
        total_duration: float,
        token_usage: dict[str, Any],
        git_commit: str | None,
        link_results: dict[str, Any],
    ) -> None:
        """Save metadata to JSON and log operation."""
        metadata = {
            "generation_info": {
                "timestamp": datetime.now().isoformat(),
                "total_duration_seconds": round(total_duration, 2),
                "main_model": self.config.main_model,
                "generator_version": "0.1.0",
                "repo_path": self.config.repo_path,
                "git_commit": git_commit,
            },
            "statistics": {
                "total_components": len(self.components),
                "leaf_nodes": len(self.leaf_nodes),
                "total_files": len(all_files),
            },
            "token_usage": {
                "total_tokens": token_usage["total"],
                "prompt_tokens": token_usage["prompt"],
                "completion_tokens": token_usage["completion"],
                "llm_calls": len(token_usage["history"]),
                "calls": token_usage["history"],
            },
            "files_generated": all_files,
            "link_validation": {
                "files_fixed": len(link_results.get("fixed", [])),
                "broken_links_removed": len(link_results.get("removed", [])),
            },
        }
        project_name = self.get_project_name()
        file_manager.save_json(
            metadata, os.path.join(data_dir, f"{project_name}_metadata.json")
        )

        log_operation(
            output_dir=self.config.output_dir,
            operation_type="full_generation",
            repo_path=self.config.repo_path,
            git_commit=git_commit,
            duration_seconds=total_duration,
            total_tokens=token_usage["total"],
            prompt_tokens=token_usage["prompt"],
            completion_tokens=token_usage["completion"],
            llm_calls=len(token_usage["history"]),
            components_processed=len(self.components),
            files_generated=len(all_files),
            status="success",
        )


def run_pipeline(config: Config, generate_docs: bool = True):
    pipeline = DocPipeline(config)
    components, leaf_nodes = pipeline.run_analysis()
    if generate_docs:
        results = asyncio.run(pipeline.generate_all())
        return results
    return {"components": components, "leaf_nodes": leaf_nodes}
