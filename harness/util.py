"""Small shared utilities: paths, JSON, subprocess logging, git metadata."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent


class HarnessError(RuntimeError):
    """A user-facing experiment failure."""


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_component(value: str) -> str:
    """Make a user-provided experiment/system/repo name safe as one path part."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "unnamed"


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_path(value: str | os.PathLike[str], root: Path = ROOT) -> Path:
    """Resolve a config path relative to the project root (not the CWD)."""
    path = Path(os.path.expanduser(str(value)))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def write_json(path: Path, data: Any) -> None:
    """Write JSON atomically so an interrupted run does not leave a broken file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError):
        return default


def load_dotenv_simple(path: Path) -> None:
    """Load KEY=VALUE lines without overriding the real environment.

    The harness intentionally avoids a python-dotenv dependency so it can run in
    a small venv. Values may be quoted; comments start with '#'.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ[key] = value


def format_command(cmd: Sequence[str | os.PathLike[str]]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_logged(
    cmd: Sequence[str | os.PathLike[str]],
    log_path: Path,
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    append: bool = False,
) -> tuple[int, float]:
    """Run a command, streaming combined stdout/stderr to the console and log.

    Returns (returncode, duration_seconds). The log is flushed line by line so
    long LLM runs can be tailed while they are still running.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8") as log:
        log.write(f"# started: {now_iso()}\n")
        log.write(f"# cwd: {cwd or Path.cwd()}\n")
        log.write(f"$ {format_command(cmd)}\n")
        log.flush()
        print(f"$ {format_command(cmd)}")
        print(f"  log: {log_path}")
        try:
            proc = subprocess.Popen(
                [str(part) for part in cmd],
                cwd=str(cwd) if cwd else None,
                env=dict(env) if env is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            log.write(f"# failed to start: {exc}\n")
            log.write(f"# finished: {now_iso()}  returncode=127\n")
            print(f"failed to start: {exc}", file=sys.stderr)
            return 127, time.monotonic() - started

        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
        finally:
            proc.stdout.close()
        returncode = proc.wait()
        duration = time.monotonic() - started
        log.write(
            f"# finished: {now_iso()}  returncode={returncode}  seconds={duration:.1f}\n"
        )
    return returncode, duration


def run_capture(
    cmd: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(part) for part in cmd],
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def git_head(path: Path) -> str | None:
    try:
        proc = run_capture(["git", "-C", str(path), "rev-parse", "HEAD"])
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def git_short_head(path: Path) -> str | None:
    head = git_head(path)
    return head[:12] if head else None


def git_remote(path: Path) -> str | None:
    try:
        proc = run_capture(["git", "-C", str(path), "remote", "get-url", "origin"])
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_markdown(docs_dir: Path) -> tuple[int, int]:
    """Return (number of .md files, total bytes)."""
    files = list(docs_dir.glob("*.md")) if docs_dir.is_dir() else []
    return len(files), sum(p.stat().st_size for p in files if p.is_file())


def fingerprint_markdown(docs_dir: Path) -> str | None:
    """Stable hash of top-level markdown names and contents."""
    import hashlib

    if not docs_dir.is_dir():
        return None
    files = sorted(p for p in docs_dir.glob("*.md") if p.is_file())
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def ensure_within(root: Path, path: Path) -> Path:
    root = root.resolve()
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HarnessError(f"path escapes {root}: {path}") from exc
    return path


def human_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def iter_json(paths: Iterable[Path]) -> Iterable[tuple[Path, Any]]:
    for path in paths:
        data = read_json(path)
        if data is not None:
            yield path, data
