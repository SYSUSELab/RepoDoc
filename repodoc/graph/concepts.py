"""Business concept extraction from components.

Two modes (config.llm_concept_extraction):
- False (default): fast keyword matching over component names/docstrings.
- True: keyword matching first, then LLM extraction for unmatched components.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List

from repodoc.config import Config
from repodoc.graph.graph import HeterogeneousGraph
from repodoc.graph.models import REL_SEMANTIC_IMPACT, ConceptNode, Node
from repodoc.llm import call_llm

logger = logging.getLogger(__name__)

CONCEPT_PATTERNS = {
    "authentication": [
        "auth", "login", "password", "token", "credential", "oauth",
    ],
    "authorization": ["permission", "role", "access", "acl", "authorize"],
    "database": ["db", "query", "model", "schema", "migration", "sql"],
    "cache": ["cache", "redis", "memcached", "session"],
    "api": ["endpoint", "rest", "graphql", "request", "response", "http"],
    "validation": ["validate", "schema", "parser", "sanitize"],
    "logging": ["log", "debug", "trace", "audit"],
    "error_handling": ["exception", "error", "fallback", "retry"],
    "configuration": ["config", "settings", "env", "option"],
    "serialization": ["json", "xml", "serialize", "deserialize", "marshal"],
}


def _build_name_index(components: Dict[str, Node]) -> Dict[str, str]:
    """Map component name -> component id (built once per extraction run,
    instead of the old per-edge O(n) scan)."""
    return {node.name: cid for cid, node in components.items() if node.name}


def _add_concept_edges(
    graph: HeterogeneousGraph,
    comp_name: str,
    name_to_id: Dict[str, str],
    concept_name: str,
    confidence: float,
    context: str,
) -> None:
    """Link a component to a concept node in the graph (idempotent)."""
    comp_id = name_to_id.get(comp_name)
    if not comp_id:
        return
    concept_id = f"concept/{concept_name.lower()}"
    try:
        concept_node = ConceptNode(
            id=concept_id,
            concept=concept_name,
            context=context,
            confidence=confidence,
        )
        graph.add_concept_node(concept_node)
        graph.add_edge(comp_id, concept_id, REL_SEMANTIC_IMPACT, 0.5)
    except Exception:
        pass


def _extract_keyword_concepts(
    graph: HeterogeneousGraph,
    components: Dict[str, Node],
    name_to_id: Dict[str, str],
) -> set[str]:
    """Keyword matching over component names and docstrings (fast path)."""
    matched: set[str] = set()
    for comp_id, node in components.items():
        if not node.name:
            continue

        name_lower = node.name.lower()
        docstring_lower = (node.docstring or "").lower()

        for concept_name, keywords in CONCEPT_PATTERNS.items():
            if any(kw in name_lower or kw in docstring_lower for kw in keywords):
                matched.add(comp_id)
                _add_concept_edges(
                    graph,
                    node.name,
                    name_to_id,
                    concept_name,
                    confidence=0.8,
                    context=f"{node.name}: {node.docstring[:200] if node.docstring else ''}",
                )
    return matched


def _extract_concept_with_llm(
    component_names: List[str], components: Dict[str, Node], config: Config
) -> Dict[str, List[str]]:
    """Use LLM to extract business concepts for components.

    Returns a dict mapping component name -> list of concept names.
    """
    component_info = []
    for comp_id in component_names[: config.llm_concept_max_components]:
        node = components.get(comp_id)
        if not node:
            continue
        info = f"- {node.name}: {node.docstring[:200] if node.docstring else 'No docstring'}"
        component_info.append(info)

    if not component_info:
        return {}

    prompt = f"""Analyze the following code components and identify business concepts they represent.

Components:
{chr(10).join(component_info)}

Business concept categories include: authentication, authorization, database, cache, api, validation, logging, error_handling, configuration, serialization, etc.

Return a JSON mapping each component name to its business concepts:
{{
    "ComponentName": ["concept1", "concept2"],
    ...
}}

Only include components that clearly represent a business concept. If a component is generic utility code with no business meaning, omit it from the output.
Return ONLY the JSON, no explanation."""

    try:
        response, _ = call_llm(prompt, config, temperature=0.1, max_tokens=2048)
        # Prefer a strict parse of the whole response, then fall back to the
        # first top-level JSON object (regex misses nested structures)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, dict):
                    return parsed
    except Exception as e:
        logger.warning(f"LLM concept extraction failed: {e}")

    return {}


def extract_concepts(
    graph: HeterogeneousGraph,
    components: Dict[str, Node],
    config: Config,
    data_dir: str,
) -> None:
    """Extract business concepts and attach them to the graph, then save.

    Keyword matching always runs; LLM enhancement runs only when
    config.llm_concept_extraction is enabled.
    """
    name_to_id = _build_name_index(components)
    matched = _extract_keyword_concepts(graph, components, name_to_id)

    if config.llm_concept_extraction:
        # Components that have no concept edges yet
        components_with_concepts = set()
        for node_id, attrs in graph.nodes(data=True):
            if isinstance(node_id, tuple):
                nid, _ = node_id
            else:
                nid = node_id
            for succ in graph.successors(nid):
                edge_data = graph.get_edge_data(nid, succ)
                if edge_data and edge_data.get("relation_type") == REL_SEMANTIC_IMPACT:
                    components_with_concepts.add(nid)

        components_without_concepts = [
            comp_id
            for comp_id in components.keys()
            if comp_id not in components_with_concepts
        ]

        if components_without_concepts:
            llm_concepts = _extract_concept_with_llm(
                components_without_concepts, components, config
            )
            for comp_name, concepts in llm_concepts.items():
                for concept_name in concepts:
                    _add_concept_edges(
                        graph,
                        comp_name,
                        name_to_id,
                        concept_name,
                        confidence=0.7,  # Slightly lower than keyword matching
                        context=f"{comp_name}: LLM extracted concept",
                    )

    # Persist the graph with concept nodes
    project_name = project_name_of(config.repo_path)
    kg_path = os.path.join(data_dir, f"{project_name}_knowledge_graph.json")
    graph.save(kg_path)
