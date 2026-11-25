"""Experiment naming and manifest helpers.

Every comparable run lives under::

    runs/<experiment>/systems/<system>/<repo>/

The experiment label is part of the data contract, so reject names that would
make directories ambiguous or hard to scan later.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .config import Config
from .metrics import normalize_tokens
from .util import HarnessError, now_iso, read_json, safe_component, write_json

EXPERIMENT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
NAMING_HELP = (
    "Use lowercase letters, digits, dots, dashes and underscores, at most 80 "
    "characters: <language>-<repo>-<scope>-v<N>, e.g. python-rasa-full-v1"
)


def validate_experiment_name(name: str) -> str:
    if not EXPERIMENT_RE.fullmatch(str(name)):
        raise HarnessError(f"invalid experiment name '{name}'. {NAMING_HELP}")
    return str(name)


def _language(cfg: Config, repo: str) -> str | None:
    try:
        return cfg.repo(repo).get("language")
    except HarnessError:
        return None


def write_manifest(cfg: Config, experiment: str) -> Path:
    """Write/refresh ``runs/<experiment>/experiment.json``.

    This is a compact index for humans and downstream scripts; the authoritative
    per-run data remains ``metrics.json`` and ``meta.json``. Failed or stale runs
    are not hidden: a failed ``meta.json`` marks the corresponding row as failed.
    """
    validate_experiment_name(experiment)
    root = cfg.runs_dir / safe_component(experiment)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for meta_path in sorted(root.glob("systems/*/*/meta.json")):
        run_dir = meta_path.parent
        meta = read_json(meta_path)
        metrics = read_json(run_dir / "metrics.json")
        if not isinstance(meta, dict):
            continue
        generation: dict[str, Any] = {}
        docs: dict[str, Any] = {}
        score = None
        pages = None
        status = str(meta.get("status") or "unknown")
        if isinstance(metrics, dict):
            generation = metrics.get("generation") or {}
            docs = (metrics.get("artifacts") or {}).get("docs") or {}
            score = metrics.get("score_100")
            pages = docs.get("pages") or generation.get("docs_pages")
            status = str(metrics.get("status") or status)
        if meta.get("status") == "failed":
            status = "failed"
        if not generation:
            gen_stage = (meta.get("stages") or {}).get("generate") or {}
            generation = {
                "duration_sec": gen_stage.get("duration_sec"),
                "docs_pages": gen_stage.get("docs_pages"),
            }
            pages = gen_stage.get("docs_pages")
            adapter_stats = read_json(run_dir / "work" / "system_metadata.json")
            if isinstance(adapter_stats, dict):
                generation.update(adapter_stats)
        tokens = normalize_tokens(generation.get("tokens"))
        rows.append(
            {
                "system": meta.get("system"),
                "repo": meta.get("repo"),
                "language": _language(cfg, str(meta.get("repo"))),
                "status": status,
                "score_100": score,
                "docs_pages": pages,
                "generation_duration_sec": generation.get("duration_sec"),
                "generation_llm_duration_sec": generation.get("llm_duration_sec"),
                "tokens": tokens,
                "cost_usd": generation.get("cost_usd"),
                "cost_status": generation.get("cost_status"),
                "metrics": str(run_dir / "metrics.json")
                if (run_dir / "metrics.json").is_file()
                else None,
                "meta": str(meta_path),
            }
        )

    payload = {
        "experiment": experiment,
        "naming": "<language>-<repo>-<scope>-v<N>",
        "layout": "runs/<experiment>/systems/<system>/<repo>/",
        "updated_at": now_iso(),
        "systems": sorted({row["system"] for row in rows if row.get("system")}),
        "repos": sorted({row["repo"] for row in rows if row.get("repo")}),
        "languages": sorted({row["language"] for row in rows if row.get("language")}),
        "runs": rows,
    }
    path = root / "experiment.json"
    write_json(path, payload)
    return path
