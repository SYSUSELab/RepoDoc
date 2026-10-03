"""Cluster leaf components into a module tree.

Two strategies (config.cluster_strategy):
- "llm": LLM-guided hierarchical clustering (batched, parallel).
- "heuristic": deterministic grouping by top-level directory segment.
"""

import hashlib
import json
import logging
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from repodoc.config import Config, MAX_TOKEN_PER_MODULE
from repodoc.graph.models import Node
from repodoc.llm import call_llm
from repodoc.utils import count_tokens, file_manager

logger = logging.getLogger(__name__)

# Maximum components per batch: large batches produce huge JSON outputs that
# approach max_tokens and return empty responses (150/70-item batches failed
# on DeepSeek, but short-id repos like libevent are fine at 100 on the
# opencode gateway; 50 was the conservative default that made big repos
# slow). Keep MAX_CONCURRENT_LLM_CALLS low when batches get large.
MAX_COMPONENTS_PER_BATCH = 100
# Modules with more components than this get clustered into sub-modules
# (recursion). Independent of the batch size: a 41-component module should
# still be split even though it fits in one batch. ≤SKIP_CLUSTER_MAX_LEAVES
# modules are never split. 40 skips LLM calls for small-ish modules, which
# cuts recursive LLM call count roughly in half on big repos.
MIN_COMPONENTS_FOR_RECURSION = 40
# Max concurrent LLM calls for batch clustering. The batch thread pools nest
# through recursion (a worker thread recursing into a >50-component module
# opens a NEW pool), so plain worker counts explode: libevent's 15 top-level
# batches + recursive 9/8/5/4-worker pools hit the API with dozens of
# concurrent requests and ~87% of batches returned empty responses. A single
# process-wide semaphore caps ALL LLM calls (top-level and recursive).
MAX_BATCH_WORKERS = 16
MAX_CONCURRENT_LLM_CALLS = 16
_LLM_CALL_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_LLM_CALLS)
# Only skip clustering when content is small AND we have few leaves (avoid one big module for medium projects).
# Keep low: at 25, atcher (18 leaves) skipped LLM clustering entirely and
# lost its config/core/services module split.
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
            if not isinstance(value, dict):
                continue  # defense: malformed module entry
            if key == module_name:
                lines.append(f"{'  ' * indent}{key} (current module)")
            else:
                lines.append(f"{'  ' * indent}{key}")
            comps = value.get("components", [])
            comp_str = ", ".join(comps)
            if len(comp_str) > 500:
                # keep recursive prompts small — the full id list would
                # otherwise blow past the 60K prompt cap on deep trees
                comp_str = f"{comp_str[:500]}... (+{len(comps)} total)"
            lines.append(f"{'  ' * (indent + 1)} Core components: {comp_str}")
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
            return _sanitize_module_tree(tree)
    except json.JSONDecodeError:
        pass

    # 2. First JSON object anywhere in the response (tolerates trailing junk)
    try:
        decoder = json.JSONDecoder()
        tree, _ = decoder.raw_decode(content.lstrip())
        if isinstance(tree, dict):
            return _sanitize_module_tree(tree)
    except json.JSONDecodeError:
        pass

    # 3. Content between <GROUPED_COMPONENTS> tags (free-form mode)
    if "<GROUPED_COMPONENTS>" in content and "</GROUPED_COMPONENTS>" in content:
        inner = content.split("<GROUPED_COMPONENTS>")[1].split(
            "</GROUPED_COMPONENTS>"
        )[0].strip()
        if inner.startswith("```"):
            fence_parts = inner.split("```")
            if len(fence_parts) >= 2:
                inner = fence_parts[1]
                if inner.startswith("json"):
                    inner = inner[4:]
                inner = inner.strip()
        try:
            tree = json.loads(inner)
            if isinstance(tree, dict):
                return _sanitize_module_tree(tree)
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not parse JSON from LLM response")


def _sanitize_module_tree(tree: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the LLM-produced module tree in place.

    LLMs occasionally emit malformed shapes ({"module_name": "string"},
    {"error": "..."} or components as a bare string) that crash downstream
    .get() calls. Only {name: {path: str, components: [str], children: {}}}
    entries survive; unknown component refs are left to _normalize_module_tree
    (which has the component map) and backfill."""
    for name in [k for k, v in tree.items() if not isinstance(v, dict)]:
        logger.warning("Dropping non-dict module value for '%s'", name)
        del tree[name]
    for name, info in tree.items():
        comps = info.get("components", [])
        if isinstance(comps, str):
            comps = [comps]
        if not isinstance(comps, list):
            logger.warning("Module '%s': components is not a list; reset", name)
            comps = []
        info["components"] = [
            c for c in comps if isinstance(c, str) and c.strip()
        ]
        if not isinstance(info.get("path"), str):
            info["path"] = ""
        children = info.get("children")
        if isinstance(children, dict):
            _sanitize_module_tree(children)
        elif "children" in info:
            info.pop("children", None)
    return tree


# ---------------------------------------------------------------------------
# Clustering response cache: same (prompt, model) -> same module tree.
# Reproducibility for reruns/ablations, and no token spend on identical work.
# ---------------------------------------------------------------------------

_CLUSTER_CACHE_LOCK = threading.RLock()
_cluster_cache_mem: Dict[str, str] | None = None


def _cluster_cache_path(config: Config) -> str:
    return os.path.join(config.dependency_graph_dir, "clustering_cache.json")


def _cache_key(prompt: str, config: Config) -> str:
    digest = hashlib.sha256()
    digest.update(prompt.encode("utf-8"))
    digest.update(b"\x00")
    digest.update((config.cluster_model or config.main_model or "").encode("utf-8"))
    return digest.hexdigest()


def _load_cluster_cache(config: Config) -> Dict[str, str]:
    global _cluster_cache_mem
    with _CLUSTER_CACHE_LOCK:
        if _cluster_cache_mem is None:
            path = _cluster_cache_path(config)
            data = file_manager.load_json(path) if os.path.exists(path) else None
            _cluster_cache_mem = data if isinstance(data, dict) else {}
        return _cluster_cache_mem


def _store_cluster_cache(config: Config, key: str, response: str) -> None:
    global _cluster_cache_mem
    with _CLUSTER_CACHE_LOCK:
        cache = _load_cluster_cache(config)
        cache[key] = response
        try:
            file_manager.save_json(cache, _cluster_cache_path(config))
        except OSError as e:  # cache is best-effort, never fail the run
            logger.warning("Could not persist clustering cache: %s", e)


def _call_llm_with_retry(
    prompt: str,
    config: Config,
    max_retries: int = 2,
) -> Tuple[str, Dict[str, Any]]:
    """Call LLM with retry on parse failure, using lower temperature.

    The process-wide semaphore serializes ALL clustering LLM calls (batch
    pools and recursive single calls alike) — nested thread pools would
    otherwise multiply concurrent requests (libevent: dozens concurrent ->
    ~87% empty responses).
    """
    with _LLM_CALL_SEMAPHORE:
        return _call_llm_with_retry_locked(prompt, config, max_retries)


def _call_llm_with_retry_locked(
    prompt: str,
    config: Config,
    max_retries: int = 2,
) -> Tuple[str, Dict[str, Any]]:
    """Internal: the retry loop, run under the global LLM semaphore."""
    cache_key = _cache_key(prompt, config)
    cached = _load_cluster_cache(config).get(cache_key)
    if cached is not None:
        try:
            cached_tree = _parse_llm_response(cached)
            logger.info(
                "Clustering cache hit (%d modules, key %s)",
                len(cached_tree),
                cache_key[:12],
            )
            return cached, cached_tree
        except Exception:
            logger.warning(
                "Discarding corrupt clustering cache entry %s", cache_key[:12]
            )

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
            _store_cluster_cache(config, cache_key, last_response)
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
    only raises ValueError on cross-drive paths on Windows).

    Empty directory parts (root-level files, e.g. index.d.ts) are ignored;
    if every member is root-level, "." is returned."""
    dirs = [d for d in dirs if d]
    if not dirs:
        return "."
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
            if not isinstance(info, dict):
                continue  # defense: malformed module entry from the LLM
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
                if not any(dirs):
                    # Root-layout repos (C projects: event.c sits at the repo
                    # root) have no directories — keep the LLM's semantic
                    # path ('libevent') instead of flattening to '.'
                    if not raw:
                        info["path"] = "."
                    continue
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


def _dedupe_tree_components(tree: Dict[str, Any]) -> None:
    """Remove duplicate component listings across the tree.

    Semantics: overlap with a module's OWN ancestors is by design (a parent
    keeps all members while children subdivide them — must not be removed).
    Overlap with any OTHER branch (e.g. top-level "utils" vs. a "utils"
    sub-module of "helpers") is a real duplicate and the later listing is
    dropped; first occurrence wins, so nothing is ever lost."""
    global_seen: set[str] = set()

    def walk(nodes: Dict[str, Any], ancestors: set[str]) -> None:
        # Sibling modules are mutually exclusive too
        level_seen: set[str] = set()
        for info in nodes.values():
            keep = []
            for x in info.get("components", []):
                if x in ancestors:
                    # parent/child subdivision is by design, but sibling
                    # modules are mutually exclusive even for those members
                    if x not in level_seen:
                        level_seen.add(x)
                        keep.append(x)
                elif x not in global_seen and x not in level_seen:
                    global_seen.add(x)
                    level_seen.add(x)
                    keep.append(x)
            info["components"] = keep
            own = set(info["components"])
            children = info.get("children")
            if isinstance(children, dict):
                walk(children, ancestors | own)

    walk(tree, set())


def _validate_and_backfill(
    tree: Dict[str, Any],
    leaf_nodes: List[str],
    components: Dict[str, Node],
) -> Dict[str, Any]:
    """Ensure every leaf appears in the tree; missing ones are backfilled into
    directory-based modules (mirrors _cluster_heuristic). Only called at the
    outermost level, so leaves are never silently dropped."""
    _dedupe_tree_components(tree)
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
        if len(parts) > 1:
            module_name = parts[0]
            module_path = parts[0]
        else:
            # Root-layout repos (C projects): group by file stem so
            # buffer.c components land in a "buffer" module that merges
            # with the LLM's same-named module
            module_name = os.path.splitext(parts[0])[0] if parts else "_unassigned"
            module_path = "."
        info = tree.setdefault(
            module_name,
            {"path": module_path, "components": [], "children": {}},
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
    else:
        # Recursive call: merge into the nested position (mirrors
        # _process_clustering_results, which handles the small-set path)
        value = current_module_tree
        for key in current_module_path:
            value = value[key].setdefault("children", {})
        for name, info in module_tree.items():
            info.pop("path", None)
            value[name] = info

    # Recursively cluster each module's components — in PARALLEL. Serial
    # recursion made libevent take 1h+ (10+ large modules each waiting for
    # its own LLM round-trips). The global LLM semaphore still caps total
    # API concurrency, so parallelizing here is safe.
    recursive = []
    for module_name, module_info in list(module_tree.items()):
        sub_leaf = module_info.get("components", [])
        valid_sub = [n for n in sub_leaf if n in components]
        if len(valid_sub) > MIN_COMPONENTS_FOR_RECURSION:
            recursive.append((module_name, module_info, valid_sub))
        else:
            module_info["children"] = {}

    if recursive:
        with ThreadPoolExecutor(
            max_workers=min(MAX_BATCH_WORKERS, len(recursive))
        ) as executor:
            futures = {}
            for module_name, module_info, valid_sub in recursive:
                future = executor.submit(
                    cluster_modules,
                    valid_sub, components, config,
                    current_module_tree, module_name, current_module_path + [module_name],
                )
                futures[future] = module_info
            for future in as_completed(futures):
                try:
                    futures[future]["children"] = future.result()
                except Exception as e:
                    logger.error("Recursive clustering failed: %s", e)
                    futures[future]["children"] = {}

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

    Root-level components group by file stem (buffer.c -> "buffer") so
    root-layout C repos get per-file modules instead of one "core" blob.
    Produces the same module_tree shape as the LLM path:
    {name: {path, components, children}}.
    """
    modules: Dict[str, Any] = {}
    for ln in leaf_nodes:
        node = components.get(ln)
        if not node:
            continue
        parts = (node.relative_path or "").split("/")
        if len(parts) > 1 and parts[0]:
            module_name, module_path = parts[0], parts[0]
        else:
            module_name = os.path.splitext(parts[0])[0] if parts and parts[0] else "core"
            module_path = "."
        info = modules.setdefault(
            module_name,
            {"path": module_path, "components": [], "children": {}},
        )
        info["components"].append(ln)
    return modules


def module_tree_stats(tree: Dict[str, Any]) -> Dict[str, Any]:
    """Structural quality stats of the clustering result (for metadata/logs):
    module count, depth, empty modules, largest module, leaf coverage."""
    module_count = 0
    max_depth = 0
    empty = 0
    largest = 0
    leaf_components: set[str] = set()

    def walk(nodes: Dict[str, Any], depth: int) -> None:
        nonlocal module_count, max_depth, empty, largest
        for info in nodes.values():
            if not isinstance(info, dict):
                continue
            module_count += 1
            max_depth = max(max_depth, depth)
            comps = info.get("components", [])
            largest = max(largest, len(comps))
            children = info.get("children")
            has_children = isinstance(children, dict) and bool(children)
            if not comps and not has_children:
                empty += 1
            if has_children:
                walk(children, depth + 1)
            else:
                leaf_components.update(comps)

    walk(tree, 1)
    return {
        "modules": module_count,
        "max_depth": max_depth,
        "empty_modules": empty,
        "largest_module_components": largest,
        "leaf_components": len(leaf_components),
    }


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
