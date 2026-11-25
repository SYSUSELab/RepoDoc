"""State manifest and change detection for the prototype CLI."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

from repodoc.schema import SCHEMA_VERSION, sha256_file, utc_now

_SKIP_PARTS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".repodoc"}


def git_head(repo_path: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def iter_source_files(
    repo_path: str | Path, exclude_dirs: Iterable[str | Path] = ()
) -> list[Path]:
    repo = Path(repo_path).resolve()
    excluded = [Path(item).resolve() for item in exclude_dirs]
    files: list[Path] = []
    for path in sorted(repo.rglob("*.py")):
        if not path.is_file():
            continue
        if any(part in _SKIP_PARTS for part in path.relative_to(repo).parts):
            continue
        resolved = path.resolve()
        if any(resolved.is_relative_to(item) for item in excluded):
            continue
        files.append(path)
    return files


def source_hash(repo_path: str | Path, exclude_dirs: Iterable[str | Path] = ()) -> str:
    import hashlib

    digest = hashlib.sha256()
    repo = Path(repo_path).resolve()
    for path in iter_source_files(repo, exclude_dirs):
        rel = path.relative_to(repo).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def read_manifest(state_dir: str | Path) -> dict[str, Any] | None:
    path = Path(state_dir) / "manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(state_dir: str | Path, manifest: dict[str, Any]) -> Path:
    path = Path(state_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)
    return path


def artifact_path(path: Path, state_dir: Path) -> str:
    try:
        return path.resolve().relative_to(state_dir.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def status_info(
    repo_path: str | Path,
    state_dir: str | Path,
    *,
    exclude_dirs: Iterable[str | Path] = (),
) -> dict[str, Any]:
    repo = Path(repo_path).resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"repository not found: {repo}")
    state = Path(state_dir).resolve()
    manifest = read_manifest(state)
    current_commit = git_head(repo)
    current_hash = source_hash(repo, exclude_dirs)

    if manifest is None:
        return {
            "state": "missing",
            "needs_update": True,
            "reason": "manifest.json not found",
            "current_commit": current_commit,
            "current_source_hash": current_hash,
            "manifest": None,
        }

    reasons: list[str] = []
    last_commit = manifest.get("last_commit")
    if current_commit and last_commit and current_commit != last_commit:
        reasons.append(f"commit changed: {last_commit[:12]} -> {current_commit[:12]}")
    if manifest.get("source_hash") != current_hash:
        reasons.append("source files changed")
    return {
        "state": "ok",
        "needs_update": bool(reasons),
        "reason": "; ".join(reasons) or "up to date",
        "current_commit": current_commit,
        "current_source_hash": current_hash,
        "manifest": manifest,
    }


def create_manifest(
    *,
    repo_path: Path,
    state_dir: Path,
    source_hash_value: str,
    code_graph_version: str,
    graph_version: str,
    component_version: str,
    doc_version: str,
    artifacts: dict[str, str],
    run_id: str,
    repo_name: str | None = None,
) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "repo": repo_name or repo_path.name,
        "repo_path": str(repo_path),
        "last_commit": git_head(repo_path),
        "source_hash": source_hash_value,
        "code_graph_version": code_graph_version,
        "graph_version": graph_version,
        "component_version": component_version,
        "doc_version": doc_version,
        "created_at": now,
        "updated_at": now,
        "artifacts": artifacts,
        "last_run": {
            "run_id": run_id,
            "status": "ok",
            "finished_at": now,
        },
    }
