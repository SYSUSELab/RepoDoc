"""Impact computation: changed files -> affected components via the graph.

Split out of update.py. Graph edge convention: caller -> callee
(REL_CALLS), so a change to X affects X's predecessors (its callers).
"""

from pathlib import Path
from typing import Any, Optional

from repodoc.agent.diff import ChangeType, ComponentChange
from repodoc.graph.graph import HeterogeneousGraph
from repodoc.graph.models import REL_CALLS


# ---------------------------------------------------------------------------
# Impact computation
# ---------------------------------------------------------------------------


class IncrementalUpdater:
    """Finds affected components for changed files via the knowledge graph.

    Graph edge convention: caller -> callee (REL_CALLS). A change to X
    affects the components that CALL X, i.e. X's predecessors.
    """

    def __init__(self, graph: HeterogeneousGraph, config: Any):
        self.graph = graph
        self.config = config
        self._impact_cache: dict[str, set[str]] = {}

    def _find_components_in_file(self, file_path: str) -> set[str]:
        """Find all components defined in a file (by graph node file_path)."""
        components: set[str] = set()
        path = Path(file_path).as_posix()

        for node_id, node_attrs in self.graph.nodes(data=True):
            node_data = node_attrs.get("data", {})
            source_file = node_data.get("file_path", "")
            if source_file and self._paths_match(source_file, path):
                components.add(node_id)

        if not components:
            components = self._find_components_by_relative_path(file_path)
        return components

    def _find_components_by_relative_path(self, file_path: str) -> set[str]:
        """Match by relative path structure (filename + at least one dir)."""
        components: set[str] = set()
        try:
            current_path = Path(file_path)
            current_name = current_path.name
            current_parts = current_path.parts
            current_len = len(current_parts)

            for node_id, node_attrs in self.graph.nodes(data=True):
                node_data = node_attrs.get("data", {})
                source_file = node_data.get("file_path", "")
                if not source_file:
                    continue

                source_path = Path(source_file)
                source_parts = source_path.parts
                source_len = len(source_parts)

                if source_path.name != current_name:
                    continue

                # Need at least 2 matching parts (filename + 1 directory)
                min_match = 2
                match_count = 0
                src_idx, tgt_idx = source_len - 1, current_len - 1
                while (
                    src_idx >= 0 and tgt_idx >= 0 and match_count < min_match
                ):
                    if source_parts[src_idx] == current_parts[tgt_idx]:
                        match_count += 1
                    else:
                        break
                    src_idx -= 1
                    tgt_idx -= 1

                if match_count >= min_match:
                    components.add(node_id)
        except Exception as e:
            logger.debug(f"Error in relative path matching: {e}")
        return components

    def _paths_match(self, source: str, target: str) -> bool:
        try:
            return Path(source).resolve() == Path(target).resolve()
        except Exception:
            return source == target

    def _get_downstream_components(self, components: set[str]) -> set[str]:
        """Components affected by a change to the given ones = their CALLERS
        (predecessors in the caller -> callee graph)."""
        downstream: set[str] = set()
        to_process = list(components)
        visited: set[str] = set()

        while to_process:
            component = to_process.pop()
            if component in visited:
                continue
            visited.add(component)

            for pred in self.graph.predecessors(component):
                edge_data = self.graph.get_edge_data(pred, component)
                if edge_data and edge_data.get("relation_type") == REL_CALLS:
                    downstream.add(pred)
                    to_process.append(pred)
        return downstream

    def get_upstream_dependencies(self, component: str) -> set[str]:
        """Components this component depends on = its callees (successors)."""
        if component in self._impact_cache:
            return self._impact_cache[component]

        dependencies: set[str] = set()
        to_process = [component]
        visited: set[str] = set()

        while to_process:
            comp = to_process.pop()
            if comp in visited:
                continue
            visited.add(comp)

            for succ in self.graph.successors(comp):
                edge_data = self.graph.get_edge_data(comp, succ)
                if edge_data and edge_data.get("relation_type") == REL_CALLS:
                    dependencies.add(succ)
                    to_process.append(succ)

        self._impact_cache[component] = dependencies
        return dependencies

    def compute_impact_graph(
        self,
        changed_files: list[str],
        module_tree: Optional[dict[str, Any]] = None,
        change_map: Optional[dict[str, ComponentChange]] = None,
    ) -> dict[str, Any]:
        """Compute a complete impact analysis for changed files.

        Returns a structured result with:
        - changed_files / changed_components / affected_components
        - new_components: components detected as NEW via diff analysis that
          are not in the knowledge graph
        - not_in_module_tree / impact_order / total_affected
        """
        changed_components: set[str] = set()
        for f in changed_files:
            changed_components.update(self._find_components_in_file(f))

        all_affected = self.get_affected_components(changed_files)

        module_tree_components: set[str] = set()

        def _collect_tree_components(nodes: dict) -> None:
            """Collect member components from EVERY tree node (top-level and
            nested — a nested-module component filtered out here would never
            have its doc regenerated on update)."""
            for mod_info in nodes.values():
                module_tree_components.update(mod_info.get("components", []))
                children = mod_info.get("children")
                if isinstance(children, dict):
                    _collect_tree_components(children)

        if module_tree:
            _collect_tree_components(module_tree)

        graph_components: set[str] = set()
        for node_id, _attrs in self.graph.nodes(data=True):
            if isinstance(node_id, str):
                graph_components.add(node_id)

        if module_tree:
            filtered_affected = all_affected & module_tree_components
            not_in_module_tree = all_affected - module_tree_components
        else:
            filtered_affected = all_affected
            not_in_module_tree = set()

        # New components: derived from diff analysis (NEW_COMPONENT changes
        # whose ids are absent from the knowledge graph). The old
        # "changed - graph" computation was always empty.
        new_components: set[str] = set()
        if change_map:
            new_components = {
                cid
                for cid, change in change_map.items()
                if change.change_type == ChangeType.NEW_COMPONENT
                and cid not in graph_components
            }

        impact_order = self._compute_topological_order(filtered_affected)

        return {
            "changed_files": changed_files,
            "changed_components": list(changed_components),
            "affected_components": list(filtered_affected),
            "new_components": list(new_components),
            "not_in_module_tree": list(not_in_module_tree),
            "impact_order": impact_order,
            "total_affected": len(filtered_affected),
        }

    def get_affected_components(self, changed_files: list[str]) -> set[str]:
        """All components affected by the given file changes (changed + downstream)."""
        affected: set[str] = set()
        for file_path in changed_files:
            components = self._find_components_in_file(file_path)
            affected.update(components)
            affected.update(self._get_downstream_components(components))
        return affected

    def _compute_topological_order(self, components: set[str]) -> list[str]:
        """Processing order: dependencies first (deps = callees/successors)."""
        order: list[str] = []
        remaining = set(components)
        visited: set[str] = set()

        while remaining:
            ready = []
            for comp in remaining:
                deps = self.get_upstream_dependencies(comp)
                if deps.issubset(visited):
                    ready.append(comp)

            if not ready:
                # Cycle or dependencies outside the affected set — pick one
                ready = [remaining.pop()]
            else:
                for r in ready:
                    remaining.discard(r)

            order.extend(ready)
            visited.update(ready)

        return order


