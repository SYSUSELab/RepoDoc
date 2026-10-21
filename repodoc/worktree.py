"""Working-tree snapshots for incremental updates.

A git-commit watermark cannot represent uncommitted work: re-running an
update against the same dirty tree re-processes the same changes every time,
and reverting a change leaves the knowledge graph describing code that no
longer exists. The snapshot records the hash of every managed source file
plus the content of files that were dirty (uncommitted) at snapshot time:

- unchanged tree + unchanged dirty state  -> no-op update;
- reverted change                         -> detected and synced back;
- old content of a dirty file             -> available without git.

Only dirty files store their text, so the metadata stays small.
"""

import hashlib
import os
import subprocess
from typing import Dict, Optional


def file_hash(path: str) -> str:
    """SHA-256 of a file's raw bytes (streamed; large files stay cheap)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_head_commit(repo_path: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def git_dirty_paths(repo_path: str) -> set[str]:
    """Repo-relative paths that differ from HEAD or are untracked."""
    paths: set[str] = set()
    commands = (
        ["git", "diff", "--name-only", "HEAD"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    )
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd, cwd=repo_path, capture_output=True, text=True, timeout=30
            )
        except Exception:
            continue
        if result.returncode != 0:
            continue
        paths.update(
            p.replace(os.sep, "/")
            for p in result.stdout.strip().split("\n")
            if p
        )
    return paths


def _managed_hashes(config) -> Dict[str, str]:
    """Hashes of every source file the analysis pipeline would read."""
    from repodoc.graph.builder import _collect_code_files

    files = _collect_code_files(
        config.repo_path,
        include_patterns=config.include_patterns or None,
        exclude_patterns=set(config.exclude_patterns) or None,
        max_file_size=config.max_file_size,
    )
    out: Dict[str, str] = {}
    for info in files:
        rel = info["path"].replace(os.sep, "/")
        try:
            out[rel] = file_hash(os.path.join(config.repo_path, info["path"]))
        except OSError:
            continue
    return out


def worktree_snapshot(config) -> dict:
    """Snapshot dict: commit, per-file hashes, and dirty-file contents."""
    files = _managed_hashes(config)
    dirty: Dict[str, str] = {}
    for rel in sorted(git_dirty_paths(config.repo_path) & set(files)):
        try:
            with open(
                os.path.join(config.repo_path, rel),
                "r",
                encoding="utf-8",
                errors="replace",
            ) as f:
                dirty[rel] = f.read()
        except OSError:
            continue
    return {
        "commit": get_head_commit(config.repo_path),
        "files": files,
        "dirty": dirty,
    }


def record_worktree_snapshot(metadata: dict, config) -> None:
    """Attach the current working-tree snapshot to generation/update metadata.

    Best effort: a snapshot failure must never fail a docs run."""
    try:
        metadata["working_tree"] = worktree_snapshot(config)
    except Exception:
        pass
