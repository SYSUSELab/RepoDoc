"""Cluster leaf components into a module tree.

Two strategies (config.cluster_strategy):
- "llm": LLM-guided hierarchical clustering (batched, parallel).
- "heuristic": deterministic grouping by top-level directory segment.
"""

import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from repodoc.config import Config, MAX_TOKEN_PER_MODULE
from repodoc.graph.models import Node
from repodoc.llm import call_llm
from repodoc.utils import count_tokens

logger = logging.getLogger(__name__)

# Maximum components per batch: large batches produce huge JSON outputs that
# approach max_tokens and return empty responses (observed with 150-item
# batches on DeepSeek json_object mode; 30-70 items succeed reliably).
MAX_COMPONENTS_PER_BATCH = 70
# Modules with more components than this get clustered into sub-modules
# (recursion). Independent of the batch size: a 41-component module should
# still be split even though it fits in one batch. ≤12-component modules
# (SKIP_CLUSTER_MAX_LEAVES) are never split.
MIN_COMPONENTS_FOR_RECURSION = 20
# Max concurrent LLM calls for batch clustering (avoid rate limits)
MAX_BATCH_WORKERS = 16
# Only skip clustering when content is small AND we have few leaves (avoid one big module for medium projects)
SKIP_CLUSTER_MAX_LEAVES = 12


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CLUSTER_REPO_PROMPT = """
You are a code analysis assistant. Your task is to group software components into logical modules.

## Rules (STRICTLY FOLLOW):
1. ONLY output the JSON structure inside <GROUPED_COMPONENTS> tags - NO explanation, NO introduction, NO markdown formatting
2. The JSON must be a dictionary with module names as keys
3. Each module must have exactly: "path" (string) and "components" (list of strings)
4. Group components that are semantically related (e.g., all HTTP-related components together, all database-related together)
5. Include all provided components - do not exclude any unless clearly unrelated
6. Use descriptive module names that reflect the functionality
7. "components" values MUST be copied EXACTLY as given (full dotted identifiers, e.g. `django.http.HttpRequest`). NEVER shorten them to bare class names like `HttpRequest`
8. "path" MUST be a real directory that contains at least one of the listed components' file paths (e.g. `django/http`). NEVER invent paths; if unsure, use the directory of the first component
9. Every identifier from the provided component list MUST appear in exactly one module's "components" array. Omitting any component is an error — before finishing, verify the union of all "components" arrays equals the provided list

## Component list:
<POTENTIAL_CORE_COMPONENTS>
{potential_core_components}
</POTENTIAL_CORE_COMPONENTS>

## Output format (use this EXACT structure):
<GROUPED_COMPONENTS>
{{
    "http": {{
        "path": "django/http",
        "components": ["django.http.HttpRequest", "django.http.HttpResponse", "django.http.JsonResponse"]
    }},
    "database": {{
        "path": "django/db",
        "components": ["django.db.Model", "django.db.QuerySet", "django.db.Transaction"]
    }}
}}
</GROUPED_COMPONENTS>

## CRITICAL:
- Output ONLY the content between <GROUPED_COMPONENTS> and </GROUPED_COMPONENTS>
- Do NOT include any other text, explanations, or markdown
- The JSON must be valid and parseable
""".strip()

CLUSTER_MODULE_PROMPT = """
You are a code analysis assistant. Your task is to group software components into logical sub-modules.

## Existing module structure:
<MODULE_TREE>
{module_tree}
</MODULE_TREE>

## Components to group (in module {module_name}):
<POTENTIAL_CORE_COMPONENTS>
{potential_core_components}
</POTENTIAL_CORE_COMPONENTS>

## Rules (STRICTLY FOLLOW):
1. ONLY output the JSON structure inside <GROUPED_COMPONENTS> tags - NO explanation, NO introduction, NO markdown
2. Group components that are semantically related within this module
3. Use descriptive sub-module names that reflect the functionality
4. Include all provided components
5. "components" values MUST be copied EXACTLY as given (full dotted identifiers). NEVER shorten them to bare class names
6. "path" MUST be a real directory that contains at least one of the listed components' file paths. NEVER invent paths

## Output format:
<GROUPED_COMPONENTS>
{{
    "submodule_name": {{
        "path": "relative/path",
        "components": ["module.path.SubComponent1", "module.path.SubComponent2"]
    }}
}}
</GROUPED_COMPONENTS>

## CRITICAL: Output ONLY the JSON between the tags, nothing else.
""".strip()


def _format_cluster_prompt(
    potential_core_components: str,
    module_tree: dict[str, Any] | None = None,
    module_name: str | None = None,
) -> str:
    """Build cluster prompt: repo-level or module-level."""
    module_tree = module_tree or {}
    lines: list[str] = []

    def _format_tree(tree: dict[str, Any], indent: int = 0) -> None:
        for key, value in tree.items():
            if key == module_name:
                lines.append(f"{'  ' * indent}{key} (current module)")
            else:
                lines.append(f"{'  ' * indent}{key}")
            comps = value.get("components", [])
            lines.append(f"{'  ' * (indent + 1)} Core components: {', '.join(comps)}")
            children = value.get("children")
            if isinstance(children, dict) and children:
                lines.append(f"{'  ' * (indent + 1)} Children:")
                _format_tree(children, indent + 2)

    _format_tree(module_tree, 0)
    formatted_module_tree = "\n".join(lines)

    if not module_tree:
        return CLUSTER_REPO_PROMPT.format(
            potential_core_components=potential_core_components
        )
    return CLUSTER_MODULE_PROMPT.format(
        potential_core_components=potential_core_components,
        module_tree=formatted_module_tree,
        module_name=module_name or "",
    )


# ---------------------------------------------------------------------------
# LLM clustering (heuristic + batched)
# ---------------------------------------------------------------------------


def _format_potential_core_components(
    leaf_nodes: List[str],
    components: Dict[str, Node],
    include_code: bool = False,
) -> str:
    """Format leaf components for the cluster prompt (with or without code)."""
    valid_leaf_nodes = [n for n in leaf_nodes if n in components]
    for n in leaf_nodes:
        if n not in components:
            logger.warning(
                "Skipping invalid leaf node '%s' - not found in components", n
            )

    by_file: Dict[str, List[str]] = defaultdict(list)
    for n in valid_leaf_nodes:
        by_file[components[n].relative_path].append(n)

    lines = []
    for file, nodes in sorted(by_file.items()):
        lines.append(f"# {file}")
        for n in nodes:
            lines.append(f"  - {n}")
            if include_code:
                code = components[n].source_code or ""
                if code:
                    # Include only first 500 chars of code to save tokens
                    lines.append(f"    ```\n    {code[:500]}\n    ```")

    return "\n".join(lines)


def _parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse LLM response to extract grouped components.

    Tolerates all observed output shapes:
    - response_format=json_object: bare JSON object (possibly with trailing
      garbage like a dangling </GROUPED_COMPONENTS> tag)
    - free-form: JSON wrapped in <GROUPED_COMPONENTS> tags, with or without
      markdown code fences
    """
    content = response.strip()

    # 1. Exact parse of the whole response (clean json_object output)
    try:
        tree = json.loads(content)
        if isinstance(tree, dict):
            return tree
    except json.JSONDecodeError:
        pass

    # 2. First JSON object anywhere in the response (tolerates trailing junk)
    try:
        decoder = json.JSONDecoder()
        tree, _ = decoder.raw_decode(content.lstrip())
        if isinstance(tree, dict):
            return tree
    except json.JSONDecodeError:
        pass

    # 3. Content between <GROUPED_COMPONENTS> tags (free-form mode)
    if "<GROUPED_COMPONENTS>" in content and "</GROUPED_COMPONENTS>" in content:
        inner = content.split("<GROUPED_COMPONENTS>")[1].split(
            "</GROUPED_COMPONENTS>"
        )[0].strip()
        if inner.startswith("```"):
            inner = inner.split("```")[1]
            if inner.startswith("json"):
                inner = inner[4:]
            inner = inner.strip()
        try:
            tree = json.loads(inner)
            if isinstance(tree, dict):
                return tree
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not parse JSON from LLM response")


def _call_llm_with_retry(
    prompt: str,
    config: Config,
    max_retries: int = 2,
) -> Tuple[str, Dict[str, Any]]:
    """Call LLM with retry on parse failure, using lower temperature."""
    last_error = None
    last_response = ""

    for attempt in range(max_retries):
        try:
            # Use lower temperature on retry to get more structured output;
            # retries drop json_object mode, which is suspected of causing
            # empty responses on large prompts/outputs
            temperature = 0.0 if attempt == 0 else 0.1
            response_format = "json_object" if attempt == 0 else None
            last_response, _ = call_llm(
                prompt,
                config,
                model=config.cluster_model,
                temperature=temperature,
                response_format=response_format,
            )

            module_tree = _parse_llm_response(last_response)
            return last_response, module_tree

        except Exception as e:
            last_error = e
            logger.warning(
                f"Attempt {attempt + 1} failed: {e}. Response preview: {last_response[:200] if last_response else 'N/A'}"
            )
            continue

    raise ValueError(f"Failed after {max_retries} attempts: {last_error}")


# ---------------------------------------------------------------------------
# LLM output normalization (names -> full component ids) + path anchoring
# ---------------------------------------------------------------------------


def _common_dir(dirs: List[str]) -> str:
    """Longest common directory prefix; fall back to the first entry when
    dirs span different top-level roots (commonpath returns '' then, and
    only raises ValueError on cross-drive paths on Windows)."""
    if not dirs:
        return ""
    if len(dirs) == 1:
        return dirs[0]
    try:
        return os.path.commonpath(dirs) or dirs[0]
    except ValueError:
        return dirs[0]


def _build_id_lookup(components: Dict[str, Node]) -> Dict[str, List[str]]:
    """Inverted index: component name -> candidate full ids (name may collide)."""
    name_map: Dict[str, List[str]] = {}
    for cid, node in components.items():
        if node.name:
            name_map.setdefault(node.name, []).append(cid)
    return name_map


def _disambiguate(
    candidates: List[str],
    components: Dict[str, Node],
    known_dirs: set[str],
    path_hint: str | None,
) -> str:
    """Pick among same-named candidates: prefer one whose directory matches
    the module's known directories, then the module's path prefix, else first."""
    for c in candidates:
        rel = components[c].relative_path or ""
        if os.path.dirname(rel) in known_dirs:
            return c
    if path_hint:
        hint = path_hint.strip("/").replace("/", ".")
        for c in candidates:
            if c.startswith(hint + "."):
                return c
    return candidates[0]


def _normalize_module_tree(
    module_tree: Dict[str, Any], components: Dict[str, Node]
) -> None:
    """Rewrite bare component names in a module tree to full component ids
    (in place). LLMs often shorten ids to bare names, which downstream lookups
    (generate/update) can never match. Unresolvable refs are dropped with a
    warning; their real leaves are backfilled by _validate_and_backfill."""
    name_map = _build_id_lookup(components)

    def walk(nodes: Dict[str, Any]) -> None:
        for info in nodes.values():
            refs = info.get("components", [])
            if not isinstance(refs, list):
                refs = [refs]  # tolerate a single string
            resolved: List[str] = []
            ambiguous: List[str] = []
            for ref in refs:
                if ref in components:  # already a full id
                    resolved.append(ref)
                else:
                    cands = name_map.get(ref, [])
                    if not cands:
                        logger.warning("Dropping unresolved component ref '%s'", ref)
                    elif len(cands) == 1:
                        resolved.append(cands[0])
                    else:
                        ambiguous.append(ref)
            if ambiguous:
                dirs = {
                    os.path.dirname(components[c].relative_path)
                    for c in resolved
                    if c in components and components[c].relative_path
                }
                path_hint = info.get("path")
                for ref in ambiguous:
                    pick = _disambiguate(
                        name_map[ref], components, dirs, path_hint
                    )
                    resolved.append(pick)
                    logger.warning("Ambiguous component name '%s' -> %s", ref, pick)
            info["components"] = resolved
            children = info.get("children")
            if isinstance(children, dict):
                walk(children)

    walk(module_tree)


def _anchor_module_paths(
    module_tree: Dict[str, Any], components: Dict[str, Node]
) -> None:
    """Verify LLM-invented module paths: a path must be a real directory
    prefix of at least one member component; otherwise derive it from the
    component ids (commonpath). In place."""
    def walk(nodes: Dict[str, Any]) -> None:
        for info in nodes.values():
            comp_ids = [c for c in info.get("components", []) if c in components]
            rels = [
                components[c].relative_path
                for c in comp_ids
                if components[c].relative_path
            ]
            raw = str(info.get("path") or "")
            path = raw.strip().removeprefix("./").rstrip("/")
            # A path is valid if it matches a member's directory, a directory
            # prefix of it, or the member's file path without extension
            # (e.g. "src/atcher/config" for "src/atcher/config.py").
            member_dirs = {os.path.dirname(r) for r in rels}
            file_noext = {
                r[: r.rindex(".")] for r in rels if "." in os.path.basename(r)
            }
            ok = (
                bool(path)
                and not os.path.isabs(path)
                and (
                    any(d == path or d.startswith(path + "/") for d in member_dirs)
                    or path in file_noext
                )
            )
            if rels and not ok:
                dirs = [os.path.dirname(r) for r in rels]
                path = _common_dir(dirs)
                info["path"] = path
                logger.warning("Anchored module path '%s' -> '%s'", raw, path)
            elif rels and not info.get("path"):
                dirs = [os.path.dirname(r) for r in rels]
                info["path"] = _common_dir(dirs)
            children = info.get("children")
            if isinstance(children, dict):
                walk(children)

    walk(module_tree)


def _collect_tree_components(tree: Dict[str, Any]) -> set[str]:
    """Union of all component ids referenced across the module tree."""
    out: set[str] = set()

    def walk(nodes: Dict[str, Any]) -> None:
        for info in nodes.values():
            out.update(info.get("components", []))
            if isinstance(info.get("children"), dict):
                walk(info["children"])

    walk(tree)
    return out


def _validate_and_backfill(
    tree: Dict[str, Any],
    leaf_nodes: List[str],
    components: Dict[str, Node],
) -> Dict[str, Any]:
    """Ensure every leaf appears in the tree; missing ones are backfilled into
    directory-based modules (mirrors _cluster_heuristic). Only called at the
    outermost level, so leaves are never silently dropped."""
    have = _collect_tree_components(tree)
    missing = [ln for ln in leaf_nodes if ln not in have]
    if not missing:
        return tree
    logger.warning(
        "Backfilling %d missing leaf(s) into directory modules", len(missing)
    )
    for ln in missing:
        node = components.get(ln)
        rel = node.relative_path if node else ""
        parts = [p for p in rel.split("/") if p]
        module_name = parts[0] if len(parts) > 1 else "_unassigned"
        info = tree.setdefault(
            module_name,
            {"path": parts[0] if len(parts) > 1 else ".", "components": [], "children": {}},
        )
        if ln not in info["components"]:
            info["components"].append(ln)
    return tree


def _cluster_one_batch(
    idx: int,
    batch: List[str],
    components: Dict[str, Node],
    config: Config,
) -> Dict[str, Any]:
    """Process a single batch (used by parallel executor). Returns module_tree or fallback dict."""
    potential = _format_potential_core_components(
        batch, components, include_code=False
    )
    prompt = _format_cluster_prompt(potential)
    try:
        _, module_tree = _call_llm_with_retry(prompt, config)
        logger.info(f"Batch {idx + 1} completed: {len(module_tree)} modules")
        return module_tree
    except Exception as e:
        logger.error(f"Batch {idx + 1} failed: {e}")
        # Fallback: group by directory instead of one monolithic batch module
        return _cluster_heuristic(batch, components)


def _cluster_components_batch(
    components_to_cluster: List[str],
    components: Dict[str, Node],
    config: Config,
    batch_size: int = MAX_COMPONENTS_PER_BATCH,
) -> Dict[str, Any]:
    """Cluster a list of components using batched LLM calls (parallel when multiple batches)."""
    if not components_to_cluster:
        return {}

    total = len(components_to_cluster)
    logger.info(f"Batching {total} components into groups of {batch_size}")

    batches = []
    for i in range(0, total, batch_size):
        batches.append(components_to_cluster[i : i + batch_size])

    num_batches = len(batches)
    workers = min(MAX_BATCH_WORKERS, num_batches)
    batch_results: List[Dict[str, Any]] = [{}] * num_batches  # placeholder

    if num_batches == 1:
        batch_results[0] = _cluster_one_batch(0, batches[0], components, config)
    else:
        logger.info(f"Processing {num_batches} batches with {workers} workers")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(
                    _cluster_one_batch, idx, batch, components, config
                ): idx
                for idx, batch in enumerate(batches)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    batch_results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Batch {idx + 1} failed: {e}")
                    batch_results[idx] = _cluster_heuristic(batches[idx], components)

    # Merge all batch results. Same-named modules from different batches
    # (common for dir-based names like "routing") must have their components
    # combined, not overwritten.
    merged: Dict[str, Any] = {}
    for result in batch_results:
        if not isinstance(result, dict):
            continue
        for name, info in result.items():
            if name in merged:
                existing = merged[name]
                for comp in info.get("components", []):
                    if comp not in existing.get("components", []):
                        existing["components"].append(comp)
            else:
                merged[name] = info

    return merged


def _cluster_large_set(
    leaf_nodes: List[str],
    components: Dict[str, Node],
    config: Config,
    current_module_tree: dict[str, Any],
    current_module_path: List[str],
) -> Dict[str, Any]:
    """Handle clustering for large sets (> MAX_COMPONENTS_PER_BATCH)."""
    logger.info(
        f"Large module detected ({len(leaf_nodes)} components), using batch clustering"
    )

    valid_leaf_nodes = [n for n in leaf_nodes if n in components]
    module_tree = _cluster_components_batch(valid_leaf_nodes, components, config)

    if not module_tree:
        logger.warning("Batch clustering returned empty result")
        return {}

    # Normalize LLM output (bare names -> full ids) before merging/recurse
    _normalize_module_tree(module_tree, components)
    _anchor_module_paths(module_tree, components)

    if not current_module_tree:
        current_module_tree.update(module_tree)

    # Recursively cluster each module's components
    for module_name, module_info in list(module_tree.items()):
        sub_leaf = module_info.get("components", [])
        valid_sub = [n for n in sub_leaf if n in components]

        if len(valid_sub) > MIN_COMPONENTS_FOR_RECURSION:
            module_info["children"] = cluster_modules(
                valid_sub, components, config,
                current_module_tree, module_name, current_module_path + [module_name]
            )
        else:
            module_info["children"] = {}

    return module_tree


def _cluster_small_set(
    leaf_nodes: List[str],
    components: Dict[str, Node],
    config: Config,
    current_module_tree: dict[str, Any],
    current_module_name: str | None,
    current_module_path: List[str],
) -> Dict[str, Any]:
    """Handle clustering for smaller sets (<= MAX_COMPONENTS_PER_BATCH)."""
    potential = _format_potential_core_components(leaf_nodes, components, include_code=False)
    potential_with_code = _format_potential_core_components(leaf_nodes, components, include_code=True)
    token_count = count_tokens(potential_with_code)

    # Skip clustering for small content
    if token_count <= MAX_TOKEN_PER_MODULE and len(leaf_nodes) <= SKIP_CLUSTER_MAX_LEAVES:
        return _build_simple_module_tree(
            leaf_nodes, components, current_module_tree, current_module_name
        )

    # Call LLM for clustering
    prompt = _format_cluster_prompt(potential, current_module_tree, current_module_name)
    module_tree = _call_clustering_llm(prompt, config, current_module_name)
    if not module_tree:
        return {}

    return _process_clustering_results(
        module_tree, components, config,
        current_module_tree, current_module_path
    )


def _build_simple_module_tree(
    leaf_nodes: List[str],
    components: Dict[str, Node],
    current_module_tree: dict[str, Any],
    current_module_name: str | None,
) -> Dict[str, Any]:
    """Build a simple module tree when clustering is skipped."""
    logger.debug(
        "Skipping clustering for %s: token count and leaf count under threshold",
        current_module_name,
    )
    if not current_module_tree:
        module_path = "."
        dirs = []
        for n in leaf_nodes:
            node = components.get(n)
            if node and node.relative_path:
                dirs.append(os.path.dirname(node.relative_path))
        if dirs:
            module_path = _common_dir(dirs)
        repo_name = current_module_name or "root"
        return {
            repo_name: {
                "path": module_path,
                "components": leaf_nodes,
                "children": {},
            }
        }
    return {}


def _call_clustering_llm(
    prompt: str,
    config: Config,
    current_module_name: str | None,
) -> Dict[str, Any] | None:
    """Call LLM for clustering with retry logic."""
    response = ""
    try:
        response, module_tree = _call_llm_with_retry(prompt, config)
    except Exception as e:
        logger.error(
            "Failed to cluster modules after retries: %s. Response: %s...",
            e,
            response[:200] if response else "N/A",
        )
        return None

    if len(module_tree) <= 1:
        logger.debug(
            "Skipping clustering for %s: module tree too small (%s modules)",
            current_module_name,
            len(module_tree),
        )
        return None

    return module_tree


def _process_clustering_results(
    module_tree: Dict[str, Any],
    components: Dict[str, Node],
    config: Config,
    current_module_tree: dict[str, Any],
    current_module_path: List[str],
) -> Dict[str, Any]:
    """Process clustering results and recurse into sub-modules."""
    # Normalize LLM output (bare names -> full ids) before merge/recurse
    _normalize_module_tree(module_tree, components)
    _anchor_module_paths(module_tree, components)

    if not current_module_tree:
        current_module_tree.update(module_tree)
    else:
        # Navigate to the correct nested position. Modules merged so far may
        # not have a "children" key yet (LLM output only has path/components;
        # children are assigned as each module is recursed), so create it.
        value = current_module_tree
        for key in current_module_path:
            value = value[key].setdefault("children", {})
        # Merge new modules
        for name, info in module_tree.items():
            info.pop("path", None)
            value[name] = info

    # Recurse into each module's sub-components
    for module_name, module_info in module_tree.items():
        sub_leaf = module_info.get("components", [])
        valid_sub = [n for n in sub_leaf if n in components]

        for n in sub_leaf:
            if n not in components:
                logger.warning(
                    "Skipping invalid sub leaf node '%s' in module '%s'", n, module_name
                )

        module_info["children"] = cluster_modules(
            valid_sub, components, config,
            current_module_tree, module_name, current_module_path + [module_name]
        )

    return module_tree


# ---------------------------------------------------------------------------
# Heuristic clustering (deterministic baseline for ablation)
# ---------------------------------------------------------------------------


def _cluster_heuristic(
    leaf_nodes: List[str],
    components: Dict[str, Node],
) -> Dict[str, Any]:
    """Group leaf components by the first segment of their relative_path.

    Root-level components go into a "core" module. Produces the same
    module_tree shape as the LLM path: {name: {path, components, children}}.
    """
    modules: Dict[str, Any] = {}
    for ln in leaf_nodes:
        node = components.get(ln)
        if not node:
            continue
        parts = (node.relative_path or "").split("/")
        module_name = parts[0] if len(parts) > 1 and parts[0] else "core"
        info = modules.setdefault(
            module_name, {"path": module_name, "components": [], "children": {}}
        )
        info["components"].append(ln)
    return modules


def cluster_modules(
    leaf_nodes: List[str],
    components: Dict[str, Node],
    config: Config,
    current_module_tree: dict[str, Any] | None = None,
    current_module_name: str | None = None,
    current_module_path: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Cluster leaf components into a module tree.
    Returns a dict: module_name -> { "path", "components", "children" }.
    """
    if config.cluster_strategy == "heuristic":
        return _cluster_heuristic(leaf_nodes, components)

    # Private components (leading underscore) are not useful in docs and
    # never enter the tree; heuristic strategy and the dependency graph
    # are unaffected (filtered by name, ids are dotted paths).
    leaf_nodes = [
        n
        for n in leaf_nodes
        if n in components and not components[n].name.startswith("_")
    ]

    # Only the outermost call validates coverage; recursion passes the
    # current tree and is skipped.
    is_outermost = current_module_tree is None
    current_module_tree = current_module_tree or {}
    current_module_path = current_module_path or []

    # If too many leaf nodes, use batch clustering
    if len(leaf_nodes) > MAX_COMPONENTS_PER_BATCH:
        tree = _cluster_large_set(
            leaf_nodes, components, config, current_module_tree, current_module_path
        )
    else:
        tree = _cluster_small_set(
            leaf_nodes, components, config,
            current_module_tree, current_module_name, current_module_path
        )

    if is_outermost:
        tree = _validate_and_backfill(tree, leaf_nodes, components)
    return tree
