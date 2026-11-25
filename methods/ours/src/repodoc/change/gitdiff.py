"""Small git diff wrapper used by the update skeleton."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

_STATUS = {
    "A": "added",
    "M": "modified",
    "D": "deleted",
    "R": "renamed",
    "C": "copied",
}


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def git_changed_files(
    repo_path: str | Path,
    base_commit: str,
    target_commit: str = "HEAD",
) -> list[dict[str, Any]]:
    repo = Path(repo_path).resolve()
    result = _git(
        repo, "diff", "--name-status", "-M", f"{base_commit}..{target_commit}"
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    files: list[dict[str, Any]] = []
    for raw in result.stdout.splitlines():
        if not raw.strip():
            continue
        parts = raw.split("\t")
        code = parts[0][0]
        status = _STATUS.get(code, "unknown")
        if status in {"renamed", "copied"} and len(parts) >= 3:
            files.append({"path": parts[2], "old_path": parts[1], "status": status})
        elif len(parts) >= 2:
            files.append({"path": parts[1], "old_path": None, "status": status})
    return files


def git_head(repo_path: str | Path) -> str | None:
    repo = Path(repo_path).resolve()
    result = _git(repo, "rev-parse", "HEAD")
    return result.stdout.strip() or None if result.returncode == 0 else None
