"""FastAPI server for the RepoDoc web UI.

The Web UI is deliberately separate from local experiment runs:

    web-data/repos/   # repositories cloned from GitHub
    web-data/runs/    # states/logs produced by the Web UI

Local experiment outputs remain under ``runs/`` and are not mixed into the Web
project list. The Web UI only serves RepoDoc states; CodeWiki is not part of
this surface.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from repodoc.llm.config import load_dotenv_simple
from repodoc.qa.service import QAService

ROOT = Path(__file__).resolve().parents[5]
load_dotenv_simple(ROOT / ".env")
WEB_DATA = ROOT / "web-data"
DEFAULT_REPOS_DIR = WEB_DATA / "repos"
DEFAULT_RUNS_DIR = WEB_DATA / "runs"
WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"
_SKIP_PARTS = {".git", "node_modules", ".venv", "venv", "__pycache__"}
_SAFE = re.compile(r"[^a-z0-9._-]+")
_GITHUB_RE = re.compile(
    r"^(?:https?://github\.com/|git@github\.com:|github\.com/)([^/\s]+)/([^/\s#?]+)",
    re.I,
)
_OWNER_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_GITHUB_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_GITHUB_CACHE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# State discovery / projection
# ---------------------------------------------------------------------------
def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _artifact(state: Path, manifest: dict[str, Any], key: str, default: str) -> Path:
    value = (manifest.get("artifacts") or {}).get(key, default)
    path = Path(str(value))
    return path if path.is_absolute() else state / path


def _iso(value: Any, fallback: float | None = None) -> str | None:
    if isinstance(value, str) and value:
        return value
    if fallback is not None:
        return datetime.fromtimestamp(fallback, UTC).isoformat()
    return None


def _project_id(state: Path) -> str:
    return hashlib.sha1(str(state.resolve()).encode("utf-8")).hexdigest()[:12]


def _walk_tree(node: dict[str, Any], key: str = "children") -> list[dict[str, Any]]:
    result = [node]
    for child in node.get(key) or []:
        if isinstance(child, dict):
            result.extend(_walk_tree(child, key))
    return result


def _trim_doc_tree(node: dict[str, Any]) -> dict[str, Any]:
    children = [
        _trim_doc_tree(child)
        for child in node.get("children") or []
        if isinstance(child, dict)
    ]
    return {
        "id": node.get("id"),
        "path": node.get("path"),
        "title": node.get("title"),
        "level": node.get("level"),
        "doc_type": node.get("doc_type"),
        "component_id": node.get("component_id"),
        "status": node.get("status"),
        "section_count": len(node.get("sections") or []),
        "children": children,
    }


def _trim_component_tree(
    node: dict[str, Any], page_by_component: dict[str, str]
) -> dict[str, Any]:
    children = [
        _trim_component_tree(child, page_by_component)
        for child in node.get("children") or []
        if isinstance(child, dict)
    ]
    component_id = str(node.get("id") or "")
    return {
        "id": component_id,
        "name": node.get("name"),
        "path": node.get("path"),
        "kind": node.get("kind"),
        "depth": node.get("depth"),
        "summary": node.get("summary"),
        "member_count": len(node.get("members") or []),
        "page_path": page_by_component.get(component_id),
        "children": children,
    }


def _first_language(state: Path) -> str | None:
    nodes = state / "graph" / "nodes.jsonl"
    if not nodes.is_file():
        return None
    try:
        with nodes.open("r", encoding="utf-8") as stream:
            for _ in range(20000):
                line = stream.readline()
                if not line:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("type") == "file":
                    language = (row.get("attrs") or {}).get("language")
                    if language:
                        return str(language)
    except OSError:
        return None
    return None


def _usage_from_metadata(state: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    docs = _artifact(state, manifest, "markdown", "docs/markdown")
    metadata = _read_json(docs / "metadata.json", default={}) or {}
    usage = metadata.get("usage") if isinstance(metadata, dict) else None
    if not isinstance(usage, dict):
        usage = {}
    totals = usage.get("totals") if isinstance(usage.get("totals"), dict) else {}
    last_run = usage.get("last_run") if isinstance(usage.get("last_run"), dict) else {}
    if not totals:
        ledger_dir = state / "usage" / "runs"
        records: list[tuple[str, dict[str, Any]]] = []
        if ledger_dir.is_dir():
            for path in ledger_dir.glob("*.json"):
                row = _read_json(path, default={}) or {}
                if isinstance(row, dict) and row.get("totals"):
                    records.append((str(row.get("started_at") or ""), row))
        if records:
            records.sort(key=lambda item: item[0])
            usage = records[-1][1]
            totals = usage.get("totals") or {}
            last_run = usage
    return {
        "tokens": {
            "input": totals.get("input_tokens", totals.get("prompt_tokens")),
            "output": totals.get("output_tokens", totals.get("completion_tokens")),
            "reasoning": totals.get("reasoning_tokens"),
            "total": totals.get("total_tokens"),
        },
        "cost_usd": totals.get("cost_usd"),
        "cost_status": usage.get("cost_status") or "unavailable",
        "n_calls": usage.get("n_calls"),
        "llm_seconds": usage.get("llm_seconds") or (last_run or {}).get("elapsed_sec"),
        "by_phase": usage.get("by_phase") or {},
        "generation_info": metadata.get("generation_info")
        if isinstance(metadata, dict)
        else {},
    }


def _graph_summary(state: Path) -> dict[str, Any]:
    graph = _read_json(state / "graph" / "graph.json", default={}) or {}
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    node_types: dict[str, int] = {}
    edge_types: dict[str, int] = {}
    api_by_path: dict[str, int] = {}
    file_count = 0
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = str(node.get("type") or "unknown")
        node_types[node_type] = node_types.get(node_type, 0) + 1
        attrs = node.get("attrs") or {}
        if node_type == "file":
            file_count += 1
        if node_type == "api":
            path = str(attrs.get("path") or "")
            api_by_path[path] = api_by_path.get(path, 0) + 1
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        edge_type = str(edge.get("type") or "unknown")
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    top_files = sorted(api_by_path.items(), key=lambda item: (-item[1], item[0]))[:12]
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "files": file_count,
        "node_types": dict(sorted(node_types.items())),
        "edge_types": dict(sorted(edge_types.items())),
        "top_api_files": [
            {"path": path, "api_count": count} for path, count in top_files
        ],
    }


def _infer_identity(
    state: Path, runs_dir: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    repo = str(manifest.get("repo") or state.parent.name)
    experiment = "web"
    system = "ours"
    try:
        parts = state.resolve().relative_to(runs_dir.resolve()).parts
    except ValueError:
        parts = ()
    if parts:
        experiment = parts[0] if len(parts) > 1 else "web"
    run_meta = _read_json(state.parent / "run.json", default={}) or {}
    if isinstance(run_meta, dict) and run_meta.get("experiment"):
        experiment = str(run_meta["experiment"])
    return {"experiment": experiment, "system": system, "repo": repo}


def _describe_state(state: Path, runs_dir: Path) -> dict[str, Any] | None:
    manifest = _read_json(state / "manifest.json", default=None)
    if not isinstance(manifest, dict):
        return None
    docs_dir = _artifact(state, manifest, "markdown", "docs/markdown")
    doc_tree_path = _artifact(state, manifest, "doc_tree", "docs/doc_tree.json")
    if not docs_dir.is_dir() or not doc_tree_path.is_file():
        return None
    doc_tree = _read_json(doc_tree_path, default={}) or {}
    pages = [
        {
            "path": page.get("path"),
            "title": page.get("title"),
            "level": page.get("level"),
            "doc_type": page.get("doc_type"),
        }
        for page in _walk_tree(doc_tree.get("root") or {})
        if page.get("path")
    ]
    identity = _infer_identity(state, runs_dir, manifest)
    usage = _usage_from_metadata(state, manifest)
    updated_at = _iso(manifest.get("updated_at"), state.stat().st_mtime)
    last_run = (
        manifest.get("last_run") if isinstance(manifest.get("last_run"), dict) else {}
    )
    run_meta = _read_json(state.parent / "run.json", default={}) or {}
    return {
        "id": _project_id(state),
        "repo": identity["repo"],
        "experiment": identity["experiment"],
        "system": identity["system"],
        "state_dir": str(state),
        "relative_path": str(state.relative_to(ROOT))
        if state.is_relative_to(ROOT)
        else str(state),
        "commit": manifest.get("last_commit")
        or (doc_tree.get("root") or {}).get("commit"),
        "language": _first_language(state),
        "pages": len(pages),
        "updated_at": updated_at,
        "status": last_run.get("status") or "ready",
        "graph_version": manifest.get("graph_version"),
        "doc_version": manifest.get("doc_version"),
        "usage": usage,
        "source": "remote" if run_meta.get("remote_url") else "local",
        "remote_url": run_meta.get("remote_url"),
        "run_id": run_meta.get("run_id"),
    }


def _scan_states(runs_dir: Path) -> list[Path]:
    states: dict[Path, None] = {}
    if runs_dir.is_dir():
        for manifest_path in runs_dir.rglob("manifest.json"):
            state = manifest_path.parent
            if any(".prev-" in part for part in state.parts):
                continue
            if (
                state.name in {"state", "repodoc-state"}
                and (state / "docs" / "doc_tree.json").is_file()
            ):
                states[state.resolve()] = None
    return sorted(
        states, key=lambda path: (path.stat().st_mtime, str(path)), reverse=True
    )


def _project_list(runs_dir: Path) -> list[dict[str, Any]]:
    projects = []
    for state in _scan_states(runs_dir):
        described = _describe_state(state, runs_dir)
        if described:
            projects.append(described)
    return projects


def _project_detail(state: Path, runs_dir: Path) -> dict[str, Any]:
    described = _describe_state(state, runs_dir)
    if described is None:
        raise HTTPException(404, "project not found")
    manifest = _read_json(state / "manifest.json", default={}) or {}
    doc_tree = (
        _read_json(
            _artifact(state, manifest, "doc_tree", "docs/doc_tree.json"), default={}
        )
        or {}
    )
    component_tree = (
        _read_json(
            _artifact(
                state, manifest, "component_tree", "components/component_tree.json"
            ),
            default={},
        )
        or {}
    )
    pages = [
        page for page in _walk_tree(doc_tree.get("root") or {}) if page.get("path")
    ]
    page_by_component = {
        str(page.get("component_id")): str(page.get("path"))
        for page in pages
        if page.get("component_id") and page.get("path")
    }
    root = component_tree.get("root") or {}
    return {
        "project": described,
        "doc_tree": _trim_doc_tree(doc_tree.get("root") or {}),
        "pages": [
            {
                "id": page.get("id"),
                "path": page.get("path"),
                "title": page.get("title"),
                "level": page.get("level"),
                "doc_type": page.get("doc_type"),
                "component_id": page.get("component_id"),
                "section_count": len(page.get("sections") or []),
            }
            for page in pages
        ],
        "components": _trim_component_tree(root, page_by_component),
        "graph": _graph_summary(state),
        "usage": described["usage"],
        "manifest": {
            "repo": manifest.get("repo"),
            "repo_path": manifest.get("repo_path"),
            "last_commit": manifest.get("last_commit"),
            "graph_version": manifest.get("graph_version"),
            "doc_version": manifest.get("doc_version"),
            "updated_at": manifest.get("updated_at"),
            "last_run": manifest.get("last_run"),
        },
    }


def _find_project(project_id: str, runs_dir: Path) -> Path:
    for state in _scan_states(runs_dir):
        if _project_id(state) == project_id:
            return state
    raise HTTPException(404, "project not found")


def _render_page(state: Path, page_path: str) -> str:
    manifest = _read_json(state / "manifest.json", default={}) or {}
    docs_dir = _artifact(state, manifest, "markdown", "docs/markdown").resolve()
    candidate = (docs_dir / page_path).resolve()
    try:
        candidate.relative_to(docs_dir)
    except ValueError as exc:
        raise HTTPException(400, "invalid page path") from exc
    if not candidate.is_file() or candidate.suffix.lower() != ".md":
        raise HTTPException(404, "page not found")
    return candidate.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# GitHub repository discovery / cloning
# ---------------------------------------------------------------------------
def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _slug(value: str, fallback: str = "repo") -> str:
    cleaned = _SAFE.sub("-", value.lower()).strip("-._")
    return cleaned or fallback


def _github_slug(source: str) -> tuple[str, str] | None:
    text = source.strip()
    match = _GITHUB_RE.match(text)
    if match:
        owner, repo = match.group(1), match.group(2)
    elif _OWNER_REPO_RE.fullmatch(text) and not text.startswith(("/", ".", "~")):
        owner, repo = text.split("/", 1)
    else:
        return None
    repo = repo.removesuffix(".git")
    if not owner or not repo:
        return None
    return owner, repo


def _git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def _clone_info(path: Path) -> dict[str, Any]:
    info = _read_json(path / ".repodoc-source.json", default={}) or {}
    return info if isinstance(info, dict) else {}


def _clone_repo(source: str, repos_dir: Path) -> tuple[Path, dict[str, Any]]:
    slug = _github_slug(source)
    if slug is None:
        raise HTTPException(400, "unsupported repository URL")
    owner, repo = slug
    full_name = f"{owner}/{repo}"
    url = f"https://github.com/{full_name}.git"
    target = repos_dir / f"{owner}__{repo}"
    repos_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    if (target / ".git").exists():
        commands = [
            ["git", "-C", str(target), "remote", "set-url", "origin", url],
            ["git", "-C", str(target), "fetch", "--depth", "1", "origin"],
            ["git", "-C", str(target), "reset", "--hard", "FETCH_HEAD"],
        ]
    else:
        if target.exists():
            shutil.rmtree(target)
        commands = [
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--single-branch",
                url,
                str(target),
            ]
        ]
    for command in commands:
        try:
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=600,
                env=env,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(400, f"git clone failed: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[-1200:]
            raise HTTPException(400, f"git clone failed: {detail}")
    meta = {
        "source": "github",
        "full_name": full_name,
        "url": f"https://github.com/{full_name}",
        "cloned_at": datetime.now(UTC).isoformat(),
        "commit": _git_head(target),
    }
    (target / ".repodoc-source.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return target, meta


def _resolve_source(source: str, repos_dir: Path) -> tuple[Path, dict[str, Any] | None]:
    text = source.strip()
    if not text:
        raise HTTPException(400, "repository URL or local path is required")
    local = Path(text).expanduser()
    if local.is_dir():
        return local.resolve(), None
    if _github_slug(text) is not None:
        return _clone_repo(text, repos_dir)
    raise HTTPException(
        400,
        f"repository not found: {text}. Use a local path or a GitHub URL.",
    )


def _github_search(query: str, limit: int = 8) -> list[dict[str, Any]]:
    q = query.strip()
    if len(q) < 2:
        return []
    cache_key = q.lower()
    now = time.monotonic()
    with _GITHUB_CACHE_LOCK:
        cached = _GITHUB_CACHE.get(cache_key)
        if cached and now - cached[0] < 300:
            return cached[1][:limit]
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "repodoc-web/0.1",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = httpx.get(
            "https://api.github.com/search/repositories",
            params={
                "q": f"{q} in:name",
                "sort": "stars",
                "order": "desc",
                "per_page": max(1, min(limit, 20)),
            },
            headers=headers,
            timeout=10.0,
        )
        response.raise_for_status()
        items = response.json().get("items") or []
    except (httpx.HTTPError, ValueError):
        items = []
    results = []
    for item in items:
        if not isinstance(item, dict) or not item.get("full_name"):
            continue
        results.append(
            {
                "full_name": item["full_name"],
                "url": item.get("html_url")
                or f"https://github.com/{item['full_name']}",
                "description": item.get("description"),
                "stars": item.get("stargazers_count", 0),
                "language": item.get("language"),
                "source": "github",
            }
        )
    with _GITHUB_CACHE_LOCK:
        _GITHUB_CACHE[cache_key] = (now, results)
    return results[:limit]


def _local_repo_options(repos_dir: Path) -> list[dict[str, Any]]:
    from repodoc.graph.languages import detect_analyzer

    result = []
    if not repos_dir.is_dir():
        return result
    for path in sorted(repos_dir.iterdir()):
        if not path.is_dir() or not (path / ".git").exists():
            continue
        info = _clone_info(path)
        result.append(
            {
                "name": path.name,
                "full_name": info.get("full_name") or path.name,
                "url": info.get("url"),
                "path": str(path),
                "commit": info.get("commit") or _git_head(path),
                "language": detect_analyzer(path).name,
                "source": "local",
            }
        )
    return result


def _search_candidates(q: str, repos_dir: Path, limit: int = 8) -> list[dict[str, Any]]:
    needle = q.strip().lower()
    local = [
        row
        for row in _local_repo_options(repos_dir)
        if not needle
        or needle in row["name"].lower()
        or needle in row["full_name"].lower()
    ]
    remote = _github_search(q, limit=limit)
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in local + remote:
        key = str(row.get("full_name") or row.get("name") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(row)
    return merged[:limit]


# ---------------------------------------------------------------------------
# Web generation runner
# ---------------------------------------------------------------------------
class RunRequest(BaseModel):
    source: str = Field(min_length=1)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None
    mode: Literal["quick", "deep"] = "quick"


@dataclass
class RunRecord:
    run_id: str
    status: str
    source: str
    repo_path: str
    repo_name: str
    experiment: str
    state_dir: str
    log_path: str
    created_at: str
    updated_at: str
    remote_url: str | None = None
    repo_commit: str | None = None
    returncode: int | None = None
    finished_at: str | None = None
    project_id: str | None = None
    error: str | None = None
    progress: int = 0
    stage: str = "Queued"
    _process: subprocess.Popen[str] | None = field(default=None, repr=False)

    def public(self, logs: list[str] | None = None) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("_process", None)
        if logs is not None:
            payload["logs"] = logs
        return payload


def _parse_progress(lines: list[str]) -> tuple[int, str]:
    progress = 3
    stage = "Starting"
    for line in lines[-500:]:
        if "Documentation generation complete" in line:
            progress, stage = 100, "Complete"
        elif "STAGE Building" in line:
            progress, stage = 8, "Build"
        elif "STAGE Parsing" in line:
            progress, stage = 14, "Parse"
        elif "STAGE Preparing component" in line:
            progress, stage = 22, "Components"
        elif "STAGE LLM clustering" in line or "STAGE Clustering" in line:
            progress, stage = 30, "Clustering"
        elif "STAGE Planning document tree" in line:
            progress, stage = 36, "Planning"
        elif "STAGE Generating LLM documentation" in line:
            progress, stage = 40, "Documentation"
        elif "DONE Generated page" in line or "DONE Generated batch" in line:
            match = re.search(r"\[(\d+)/(\d+)\]", line)
            if match:
                current, total = int(match.group(1)), int(match.group(2))
                progress = 40 + int(58 * current / max(1, total))
                stage = f"Documentation {current}/{total}"
        elif "STAGE Saving" in line:
            progress, stage = 98, "Saving"
    return min(progress, 100), stage


class RunManager:
    def __init__(self, runs_dir: Path, repos_dir: Path) -> None:
        self.runs_root = runs_dir
        self.repos_dir = repos_dir
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.repos_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.records: dict[str, RunRecord] = {}
        self.processes: dict[str, subprocess.Popen[str]] = {}
        self._load()

    def _load(self) -> None:
        known = {item.name for item in fields(RunRecord)}
        for path in sorted(self.runs_root.glob("*/run.json")):
            payload = _read_json(path, default={}) or {}
            try:
                record = RunRecord(
                    **{key: value for key, value in payload.items() if key in known}
                )
            except TypeError:
                continue
            if record.status in {"running", "queued"}:
                try:
                    log_text = Path(record.log_path).read_text(
                        encoding="utf-8", errors="replace"
                    )
                except OSError:
                    log_text = ""
                if "Documentation generation complete" in log_text:
                    record.status = "completed"
                    record.progress = 100
                    record.stage = "Complete"
                    record.project_id = _project_id(Path(record.state_dir))
                    record.returncode = 0
                else:
                    record.status = "interrupted"
                    record.error = "server restarted before the run finished"
                self._save(record)
            self.records[record.run_id] = record

    def _save(self, record: RunRecord) -> None:
        path = Path(record.state_dir).parent / "run.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = record.public()
        payload.pop("logs", None)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def start(self, request: RunRequest) -> RunRecord:
        repo, remote = _resolve_source(request.source, self.repos_dir)
        repo_name = repo.name
        stamp = _stamp()
        experiment = _slug(f"web-{repo_name}")
        run_id = f"{stamp}-{_slug(repo_name)}"
        suffix = 1
        while (self.runs_root / run_id).exists():
            suffix += 1
            run_id = f"{stamp}-{_slug(repo_name)}-{suffix}"
        run_dir = self.runs_root / run_id
        state_dir = run_dir / "state"
        log_path = run_dir / "generate.log"
        run_dir.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(UTC).isoformat()
        record = RunRecord(
            run_id=run_id,
            status="queued",
            source=request.source,
            repo_path=str(repo),
            repo_name=repo_name,
            experiment=experiment,
            state_dir=str(state_dir),
            log_path=str(log_path),
            created_at=now,
            updated_at=now,
            remote_url=(remote or {}).get("url"),
            repo_commit=(remote or {}).get("commit") or _git_head(repo),
        )
        command = [
            sys.executable,
            "-m",
            "repodoc",
            "generate",
            "--repo",
            str(repo),
            "--state-dir",
            str(state_dir),
            "--repo-name",
            repo_name,
            "--mode",
            "full",
            "--cluster-mode",
            "auto",
            "--workers",
            "4",
            "--batch-size",
            "8",
        ]
        with self.lock:
            self.records[run_id] = record
            self._save(record)
            record.status = "running"
            record.updated_at = datetime.now(UTC).isoformat()
            self._save(record)
            log = log_path.open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            except Exception as exc:  # noqa: BLE001 - surface launch failure to the UI
                log.close()
                record.status = "failed"
                record.error = str(exc)
                record.finished_at = datetime.now(UTC).isoformat()
                self._save(record)
                raise HTTPException(500, f"failed to start repodoc: {exc}") from exc
            self.processes[run_id] = process
            threading.Thread(
                target=self._wait, args=(run_id, process, log), daemon=True
            ).start()
        return record

    def _wait(self, run_id: str, process: subprocess.Popen[str], log: Any) -> None:
        rc = process.wait()
        log.close()
        with self.lock:
            record = self.records.get(run_id)
            if record is None:
                return
            self._finalize(record, rc)

    def _logs(self, record: RunRecord, limit: int = 200) -> list[str]:
        try:
            lines = (
                Path(record.log_path)
                .read_text(encoding="utf-8", errors="replace")
                .splitlines()
            )
        except OSError:
            return []
        return lines[-limit:]

    def _finalize(self, record: RunRecord, returncode: int) -> None:
        record.returncode = returncode
        record.finished_at = datetime.now(UTC).isoformat()
        record.updated_at = record.finished_at
        if record.status == "cancelled":
            record.stage = "Cancelled"
        elif returncode == 0:
            record.status = "completed"
            record.project_id = _project_id(Path(record.state_dir))
            record.progress = 100
            record.stage = "Complete"
        else:
            record.status = "failed"
            record.error = f"repodoc exited with code {returncode}"
        self.processes.pop(record.run_id, None)
        self._save(record)

    def get(self, run_id: str) -> dict[str, Any]:
        process = self.processes.get(run_id)
        if process is not None and process.poll() is not None:
            with self.lock:
                record = self.records.get(run_id)
                if record is not None and record.status == "running":
                    self._finalize(record, process.returncode or 0)
        with self.lock:
            record = self.records.get(run_id)
            if record is None:
                raise HTTPException(404, "run not found")
            logs = self._logs(record)
            progress, stage = _parse_progress(logs)
            if record.status == "completed":
                progress, stage = 100, "Complete"
            elif progress >= 100 and record.status == "running":
                # The CLI has written its completion line; the process may be
                # finishing usage metadata before the wait thread observes exit.
                stage = "Finalizing"
            record.progress, record.stage = progress, stage
            return record.public(logs=logs)

    def cancel(self, run_id: str) -> dict[str, Any]:
        process = self.processes.get(run_id)
        with self.lock:
            record = self.records.get(run_id)
            if record is None:
                raise HTTPException(404, "run not found")
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except OSError:
                    pass
                record.status = "cancelled"
                record.updated_at = datetime.now(UTC).isoformat()
                self._save(record)
            return record.public(logs=self._logs(record))


def _read_source(
    state: Path, path: str, start: int = 1, end: int | None = None
) -> dict[str, Any]:
    manifest = _read_json(state / "manifest.json", default={}) or {}
    repo_path = Path(str(manifest.get("repo_path") or "")).resolve()
    if not repo_path.is_dir():
        raise HTTPException(404, "repository checkout not found")
    candidate = (repo_path / path).resolve()
    try:
        candidate.relative_to(repo_path)
    except ValueError as exc:
        raise HTTPException(400, "invalid source path") from exc
    if not candidate.is_file():
        raise HTTPException(404, "source file not found")
    lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
    first = max(1, int(start or 1))
    last = int(end or len(lines))
    last = min(len(lines), max(first, last))
    content = "\n".join(lines[first - 1 : last])
    return {
        "path": candidate.relative_to(repo_path).as_posix(),
        "start": first,
        "end": last,
        "content": content,
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
def create_app(
    runs_dir: str | Path | None = None,
    repos_dir: str | Path | None = None,
) -> FastAPI:
    runs_root = Path(runs_dir).expanduser().resolve() if runs_dir else DEFAULT_RUNS_DIR
    repos_root = (
        Path(repos_dir).expanduser().resolve() if repos_dir else DEFAULT_REPOS_DIR
    )
    manager = RunManager(runs_root, repos_root)
    qa_service = QAService()

    app = FastAPI(title="RepoDoc", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "runs_dir": str(runs_root),
            "repos_dir": str(repos_root),
        }

    @app.get("/api/repos")
    def repos() -> list[dict[str, Any]]:
        return _local_repo_options(repos_root)

    @app.get("/api/github/search")
    def github_search(
        q: str = Query(min_length=1), limit: int = Query(default=8, ge=1, le=20)
    ) -> list[dict[str, Any]]:
        return _search_candidates(q, repos_root, limit=limit)

    @app.get("/api/projects")
    def projects() -> list[dict[str, Any]]:
        return _project_list(runs_root)

    @app.get("/api/projects/{project_id}")
    def project(project_id: str) -> dict[str, Any]:
        return _project_detail(_find_project(project_id, runs_root), runs_root)

    @app.get("/api/projects/{project_id}/page", response_class=PlainTextResponse)
    def page(project_id: str, path: str = Query(min_length=1)) -> str:
        state = _find_project(project_id, runs_root)
        return _render_page(state, path)

    @app.get("/api/projects/{project_id}/usage")
    def usage(project_id: str) -> dict[str, Any]:
        state = _find_project(project_id, runs_root)
        described = _describe_state(state, runs_root)
        if described is None:
            raise HTTPException(404, "project not found")
        return described["usage"]

    @app.post("/api/projects/{project_id}/ask")
    def ask_project(project_id: str, request: AskRequest) -> dict[str, Any]:
        state = _find_project(project_id, runs_root)
        try:
            return qa_service.ask(
                state, request.question, request.session_id, request.mode
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(502, str(exc)) from exc

    @app.post("/api/projects/{project_id}/ask/stream")
    def ask_project_stream(project_id: str, request: AskRequest) -> StreamingResponse:
        state = _find_project(project_id, runs_root)

        def events() -> Any:
            try:
                for event in qa_service.ask_stream(
                    state, request.question, request.session_id, request.mode
                ):
                    yield (
                        "data: "
                        + json.dumps(event, ensure_ascii=False, default=str)
                        + "\n\n"
                    )
            except ValueError as exc:
                yield (
                    "data: "
                    + json.dumps(
                        {"type": "error", "message": str(exc)}, ensure_ascii=False
                    )
                    + "\n\n"
                )
            except Exception as exc:
                yield (
                    "data: "
                    + json.dumps(
                        {"type": "error", "message": str(exc)}, ensure_ascii=False
                    )
                    + "\n\n"
                )

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/projects/{project_id}/source")
    def source(
        project_id: str,
        path: str = Query(min_length=1),
        start: int = Query(default=1, ge=1),
        end: int | None = Query(default=None, ge=1),
    ) -> dict[str, Any]:
        state = _find_project(project_id, runs_root)
        return _read_source(state, path, start, end)

    @app.delete("/api/projects/{project_id}")
    def delete_project(project_id: str) -> dict[str, Any]:
        state = _find_project(project_id, runs_root)
        try:
            state.relative_to(runs_root)
        except ValueError as exc:
            raise HTTPException(403, "only Web projects can be deleted") from exc
        if state.name != "state" or state.parent.parent != runs_root:
            raise HTTPException(403, "only Web run states can be deleted")
        run_dir = state.parent
        process = manager.processes.get(run_dir.name)
        if process is not None and process.poll() is None:
            raise HTTPException(409, "stop the running generation before deleting it")
        shutil.rmtree(run_dir)
        manager.records.pop(run_dir.name, None)
        return {"deleted": True, "id": project_id}

    @app.post("/api/runs")
    def start_run(request: RunRequest) -> dict[str, Any]:
        return manager.start(request).public()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        return manager.get(run_id)

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict[str, Any]:
        return manager.cancel(run_id)

    if WEB_DIST.is_dir():
        assets = WEB_DIST / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(WEB_DIST / "index.html")

        @app.get("/{path:path}")
        def static_or_index(path: str) -> FileResponse:
            if path.startswith("api/"):
                raise HTTPException(404, "not found")
            candidate = (WEB_DIST / path).resolve()
            try:
                candidate.relative_to(WEB_DIST)
            except ValueError as exc:
                raise HTTPException(404, "not found") from exc
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(WEB_DIST / "index.html")

    @app.exception_handler(HTTPException)
    def http_error(_request: Any, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail, "status": exc.status_code},
        )

    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    data_dir: str | Path | None = None,
) -> None:
    import uvicorn

    root = Path(data_dir).expanduser().resolve() if data_dir else WEB_DATA
    app = create_app(runs_dir=root / "runs", repos_dir=root / "repos")
    print(
        f"RepoDoc web UI: http://{host}:{port} (data: {root})",
        flush=True,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
