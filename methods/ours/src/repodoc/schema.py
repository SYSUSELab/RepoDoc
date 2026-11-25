"""Shared schema constants, stable IDs and validation helpers."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"

NODE_TYPES = {
    "file",
    "symbol",
    "api",
    "artifact",
    "component",
    "flow",
    "doc_page",
    "doc_section",
    "change",
    "change_unit",
}
EDGE_TYPES = {
    "contains",
    "declares",
    "belongs_to",
    "imports",
    "exports",
    "calls",
    "called_by",
    "references",
    "reads",
    "writes",
    "inherits",
    "implements",
    "overrides",
    "instantiates",
    "exposes_api",
    "handles",
    "configured_by",
    "tests",
    "has_child_component",
    "part_of_component",
    "depends_on_component",
    "participates_in_flow",
    "documents",
    "cites",
    "derived_from",
    "child_of",
    "related_to",
    "changes",
    "impacts",
    "renamed_from",
    "replaces",
}
NODE_LAYERS = {"code", "semantic", "document", "change"}
COMPONENT_KINDS = {"repository", "subsystem", "component", "module", "file_group"}
DOC_TYPES = {
    "architecture",
    "component",
    "module",
    "api",
    "implementation",
    "operations",
    "security",
    "testing",
    "flow",
    "overview",
}
DOC_STATUSES = {"planned", "generated", "stale", "failed", "deleted"}
SECTION_LEVELS = {1, 2, 3, 4}

_LAYER_BY_TYPE = {
    "file": "code",
    "symbol": "code",
    "api": "code",
    "artifact": "code",
    "component": "semantic",
    "flow": "semantic",
    "doc_page": "document",
    "doc_section": "document",
    "change": "change",
    "change_unit": "change",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(data: Any, *, drop_keys: tuple[str, ...] = ("created_at",)) -> str:
    """Hash content while ignoring volatile metadata such as nested timestamps."""

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: scrub(item) for key, item in value.items() if key not in drop_keys
            }
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    payload = scrub(data)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slugify(value: str, *, fallback: str = "module") -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-").lower()
    return slug or fallback


def edge_id(edge_type: str, src: str, dst: str) -> str:
    digest = hashlib.sha1(f"{edge_type}|{src}|{dst}".encode("utf-8")).hexdigest()[:16]
    return f"edge:{digest}"


def file_node_id(path: str) -> str:
    return f"code:file:{path}"


def symbol_node_id(path: str, qualified_name: str, kind: str) -> str:
    return f"code:{kind}:{path}::{qualified_name}"


def api_node_id(path: str, qualified_name: str) -> str:
    return f"code:api:{path}::{qualified_name}"


def component_node_id(repo: str, parent_path: str, name: str) -> str:
    key = f"{repo}|{parent_path}|{name}"
    return "component:" + uuid.uuid5(uuid.NAMESPACE_URL, "repodoc:" + key).hex


def doc_page_id(path: str) -> str:
    return f"doc:{path}"


def doc_section_id(page_id: str, heading: str, order: int) -> str:
    return f"section:{page_id}:{slugify(heading, fallback='section')}:{order}"


def layer_for_node(node_type: str) -> str:
    return _LAYER_BY_TYPE[node_type]


def source_ref(path: str, start: dict[str, Any], end: dict[str, Any]) -> str:
    return f"{path}:{start['line']}-{end['line']}"


def schema_path(name: str) -> Path:
    return Path(__file__).resolve().parent / "schemas" / f"{name}.schema.json"


def load_schema(name: str) -> dict[str, Any]:
    with schema_path(name).open("r", encoding="utf-8") as f:
        return json.load(f)


def validate(name: str, data: Any) -> None:
    """Validate against one of the packaged JSON Schemas.

    `name` is e.g. "graph", "component_tree", "doc_tree", "manifest",
    "change_set". Raises jsonschema.ValidationError on failure.
    """
    from jsonschema import Draft202012Validator

    Draft202012Validator(load_schema(name)).validate(data)
