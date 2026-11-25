"""Prepare benchmark repositories, reference docs and frozen rubrics."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

from . import benchmark
from .config import Config
from .util import (
    HarnessError,
    git_head,
    now_iso,
    read_json,
    run_capture,
    run_logged,
    write_json,
)

DATASET_NAME = "anhnh2002/codewikibench"


# ----------------------------------------------------------------------
# HuggingFace dataset
# ----------------------------------------------------------------------
def _json_field(value: Any, field: str) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise HarnessError(
                f"dataset field '{field}' is not valid JSON: {exc}"
            ) from exc
    return value


def _row_repo_name(row: dict[str, Any]) -> str | None:
    for key in ("repo_name", "repository", "repo", "name"):
        if row.get(key):
            return str(row[key])
    return None


def _row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("metadata", "meta"):
        value = _json_field(row.get(key), key)
        if isinstance(value, dict):
            return value
    return {}


def _commit_from_metadata(metadata: dict[str, Any]) -> str | None:
    for key in ("commit_id", "commit", "sha", "revision", "head_commit"):
        if metadata.get(key):
            return str(metadata[key])
    return None


def _url_from_metadata(metadata: dict[str, Any]) -> str | None:
    for key in ("repo_url", "url", "repository", "github_url"):
        if metadata.get(key):
            return str(metadata[key])
    return None


def _row_commit(row: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    return _commit_from_metadata(metadata) or _commit_from_metadata(row)


def _row_url(row: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    return _url_from_metadata(metadata) or _url_from_metadata(row)


def _write_reference(
    cfg: Config, repo: str, row: dict[str, Any], metadata: dict[str, Any], force: bool
) -> bool:
    parsed_dir = cfg.reference_parsed_dir(repo)
    docs_tree = _json_field(row.get("docs_tree"), "docs_tree")
    structured = _json_field(row.get("structured_docs"), "structured_docs")
    if docs_tree is None and structured is None:
        print(
            f"  [{repo}] no docs_tree/structured_docs in dataset; skipping reference docs"
        )
        return False

    expected = []
    if docs_tree is not None:
        expected.append(parsed_dir / "docs_tree.json")
    if structured is not None:
        expected.append(parsed_dir / "structured_docs.json")
    if not force and all(path.is_file() for path in expected):
        print(
            f"  [{repo}] reference docs already exist: {parsed_dir} (use --force to overwrite)"
        )
        return False

    parsed_dir.mkdir(parents=True, exist_ok=True)
    if docs_tree is not None:
        write_json(parsed_dir / "docs_tree.json", docs_tree)
    if structured is not None:
        write_json(parsed_dir / "structured_docs.json", structured)
    write_json(
        cfg.reference_root / repo / "metadata.json",
        {
            "repo": repo,
            "dataset": DATASET_NAME,
            "metadata": metadata,
            "prepared_at": now_iso(),
        },
    )
    print(f"  [{repo}] reference docs -> {parsed_dir}")
    return True


def _normalize_rubrics(raw: Any, repo: str, metadata: dict[str, Any]) -> dict[str, Any]:
    raw = _json_field(raw, "rubrics")
    if isinstance(raw, dict) and isinstance(raw.get("rubrics"), list):
        result = dict(raw)
    elif isinstance(raw, list):
        result = {"rubrics": raw}
    elif isinstance(raw, dict) and "requirements" in raw:
        result = {"rubrics": [raw]}
    else:
        raise HarnessError(f"unrecognized rubrics structure for {repo}")

    if not isinstance(result.get("combination_metadata"), dict):
        result["combination_metadata"] = {}
    meta = result["combination_metadata"]
    meta.setdefault("combination_method", "frozen_dataset")
    meta.setdefault("source", f"huggingface:{DATASET_NAME}")
    meta.setdefault("repo", repo)
    meta.setdefault("commit_id", _commit_from_metadata(metadata))
    return result


def _write_rubrics(
    cfg: Config, repo: str, row: dict[str, Any], metadata: dict[str, Any], force: bool
) -> bool:
    raw = row.get("rubrics")
    if raw is None:
        print(f"  [{repo}] no rubrics in dataset; skipping")
        return False
    out = cfg.rubric_path(repo)
    if out.exists() and not force:
        data = read_json(out)
        if isinstance(data, dict) and isinstance(data.get("rubrics"), list):
            print(
                f"  [{repo}] frozen rubrics already exist: {out} (use --force to overwrite)"
            )
            return False
        print(f"  [{repo}] warning: invalid frozen rubrics at {out}; overwriting")
    rubrics = _normalize_rubrics(raw, repo, metadata)
    write_json(out, rubrics)
    print(f"  [{repo}] frozen rubrics -> {out}")
    return True


def _ingest_row(
    cfg: Config,
    repo: str,
    row: dict[str, Any],
    locks: dict[str, Any],
    *,
    force: bool,
) -> None:
    metadata = _row_metadata(row)
    commit = _row_commit(row, metadata)
    url = _row_url(row, metadata)
    lock = dict(locks.get(repo, {}))
    if commit:
        lock["commit"] = commit
    if url:
        lock["url"] = url
    lock.update({"source": DATASET_NAME, "prepared_at": now_iso()})
    locks[repo] = lock
    if commit:
        print(f"  [{repo}] commit {commit}")
    else:
        print(f"  [{repo}] warning: no commit id in metadata")
    _write_reference(cfg, repo, row, metadata, force)
    _write_rubrics(cfg, repo, row, metadata, force)


def fetch_hf(cfg: Config, repo_names: Iterable[str], *, force: bool = False) -> int:
    """Download reference docs + rubrics + commit ids from CodeWikiBench.

    Rows are processed one at a time so we do not hold every repository's
    documentation tree in memory at once.
    """
    repo_names = list(repo_names)
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise HarnessError(
            "the `datasets` package is required for --from-hf.\n"
            "Install it with: uv pip install --python .venv/bin/python datasets huggingface_hub"
        ) from exc

    wanted = {name.lower(): name for name in repo_names}
    revision = os.environ.get("REPODOC_DATASET_REVISION") or None
    print(
        f"Loading {DATASET_NAME}"
        f"{f' @ {revision}' if revision else ''} "
        "(this downloads/caches the dataset once)..."
    )
    dataset = load_dataset(DATASET_NAME, **({"revision": revision} if revision else {}))

    locks = dict(cfg.locks) if isinstance(cfg.locks, dict) else {}
    seen: set[str] = set()
    if hasattr(dataset, "items"):
        splits = dataset.items()
    else:
        splits = [("train", dataset)]
    for _split_name, split in splits:
        for row in split:
            name = _row_repo_name(row)
            if not name:
                continue
            canonical = wanted.get(name.lower())
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            _ingest_row(cfg, canonical, dict(row), locks, force=force)

    missing = sorted(set(repo_names) - seen)
    if missing:
        print(f"Warning: dataset has no rows for: {', '.join(missing)}")

    cfg.locks = locks
    write_json(cfg.data_dir / "locks" / "repos.json", locks)
    print(f"Wrote {cfg.data_dir / 'locks' / 'repos.json'}")
    return len(seen)


# ----------------------------------------------------------------------
# Git checkouts
# ----------------------------------------------------------------------
def _ensure_clone(cfg: Config, repo: str, log_path: Path) -> Path:
    clone = cfg.clone_dir(repo)
    if clone.is_dir() and (clone / ".git").exists():
        return clone
    if clone.exists():
        raise HarnessError(f"{clone} exists but is not a git clone")

    url = cfg.repo_url(repo)
    if not url:
        raise HarnessError(f"no repository URL configured for {repo}")
    clone.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {repo} from {url} (blobless)...")
    cmd = ["git", "clone", "--filter=blob:none", "--no-checkout", url, str(clone)]
    rc, _ = run_logged(cmd, log_path, cwd=cfg.root)
    if rc != 0:
        raise HarnessError(f"git clone failed for {repo}. See {log_path}")
    return clone


def _remote_head(url: str) -> str | None:
    try:
        proc = run_capture(["git", "ls-remote", url, "HEAD"])
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    line = proc.stdout.strip().splitlines()
    if not line:
        return None
    return line[0].split()[0]


def _commit_available(clone: Path, commit: str) -> bool:
    proc = run_capture(
        ["git", "-C", str(clone), "cat-file", "-e", f"{commit}^{{commit}}"]
    )
    return proc.returncode == 0


def _fetch_commit(clone: Path, commit: str, log_path: Path) -> None:
    if _commit_available(clone, commit):
        return
    run_logged(
        ["git", "-C", str(clone), "fetch", "origin", commit],
        log_path,
        cwd=clone,
        append=True,
    )
    if _commit_available(clone, commit):
        return
    run_logged(
        ["git", "-C", str(clone), "fetch", "--all", "--tags", "--prune"],
        log_path,
        cwd=clone,
        append=True,
    )


def _ensure_worktree(
    cfg: Config,
    repo: str,
    ref: str,
    commit: str,
    clone: Path,
    log_path: Path,
    *,
    force: bool = False,
) -> Path:
    dest = cfg.checkout_dir(repo, ref)
    if dest == clone:
        return clone
    if dest.exists():
        head = git_head(dest)
        if head == commit:
            print(f"  [{repo}] {ref} worktree ready: {dest}")
            return dest
        if not force:
            raise HarnessError(
                f"worktree {dest} is at {head}, expected {commit}; use --force to recreate"
            )
        run_capture(
            ["git", "-C", str(clone), "worktree", "remove", "--force", str(dest)]
        )
        if dest.exists():
            shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _fetch_commit(clone, commit, log_path)
    if not _commit_available(clone, commit):
        raise HarnessError(
            f"commit {commit} is not available in {clone}; if it is a later commit, "
            "make sure it is reachable from the remote refs"
        )
    run_logged(
        ["git", "-C", str(clone), "worktree", "add", "--detach", str(dest), commit],
        log_path,
        cwd=clone,
        append=True,
    )
    if not dest.is_dir():
        raise HarnessError(f"failed to create worktree: {dest}")
    print(f"  [{repo}] {ref} worktree {commit[:12]} -> {dest}")
    return dest


def prepare_checkouts(
    cfg: Config,
    repo_names: Iterable[str],
    *,
    allow_head: bool = False,
    force: bool = False,
) -> None:
    repo_names = list(repo_names)
    logs = cfg.data_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    locks = dict(cfg.locks) if isinstance(cfg.locks, dict) else {}
    errors: list[tuple[str, str]] = []

    for repo in repo_names:
        print(f"Preparing {repo}")
        base = cfg.repo_commit(repo, "base")
        if not base and not allow_head:
            message = "no base commit configured; run with --from-hf or --allow-head"
            print(f"  [{repo}] {message}")
            errors.append((repo, message))
            continue

        try:
            clone = _ensure_clone(cfg, repo, logs / f"{repo}-clone.log")
            if not base:
                base = _remote_head(cfg.repo_url(repo) or "") or None
                if not base:
                    raise HarnessError("could not resolve remote HEAD")
                lock = dict(locks.get(repo, {}))
                lock.update(
                    {"commit": base, "source": "remote-head", "prepared_at": now_iso()}
                )
                locks[repo] = lock
                cfg.locks = locks

            _ensure_worktree(
                cfg,
                repo,
                "base",
                base,
                clone,
                logs / f"{repo}-worktree.log",
                force=force,
            )

            update = cfg.repo_commit(repo, "update")
            if update:
                _ensure_worktree(
                    cfg,
                    repo,
                    "update",
                    update,
                    clone,
                    logs / f"{repo}-worktree.log",
                    force=force,
                )
            else:
                print(
                    f"  [{repo}] no update_commit configured (skipping update worktree)"
                )
        except Exception as exc:  # noqa: BLE001 - prepare all repos, then report
            print(f"  [{repo}] FAILED: {exc}", file=sys.stderr)
            errors.append((repo, str(exc)))

    cfg.locks = locks
    write_json(cfg.data_dir / "locks" / "repos.json", locks)
    if errors:
        details = "\n".join(f"  - {repo}: {message}" for repo, message in errors)
        raise HarnessError(
            f"{len(errors)} repository/repositories could not be prepared:\n{details}"
        )


# ----------------------------------------------------------------------
# Frozen rubrics
# ----------------------------------------------------------------------
def frozen_rubrics(
    cfg: Config,
    repo: str,
    *,
    models: Iterable[str] | None = None,
    force: bool = False,
) -> Path:
    parsed = cfg.reference_parsed_dir(repo)
    missing = [
        name
        for name in ("docs_tree.json", "structured_docs.json")
        if not (parsed / name).is_file()
    ]
    if missing:
        raise HarnessError(
            f"reference docs for {repo} are incomplete: missing {', '.join(missing)} in {parsed}\n"
            f"Run `python -m harness prepare --repo {repo} --from-hf` first."
        )
    out = cfg.rubric_path(repo)
    if out.exists() and not force:
        data = read_json(out)
        if isinstance(data, dict) and isinstance(data.get("rubrics"), list):
            print(f"Reusing frozen rubrics: {out}")
            return out
        print(f"warning: invalid frozen rubrics at {out}; regenerating")
    log = out.parent / "rubrics.log"
    print(f"Generating frozen rubrics for {repo} from {parsed}")
    benchmark.generate_rubrics(
        cfg, parsed, out, log, models=list(models) if models else None, force=force
    )
    if not out.is_file():
        raise HarnessError(f"rubric generation did not produce {out}")
    return out
