"""Shared utilities for RepoDoc."""

import os
import json
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote

# Lazy-loaded for count_tokens to avoid import cost when not clustering
_tiktoken_enc = None


def count_tokens(text: str) -> int:
    """Approximate token count (GPT-4 style encoding). Used for clustering threshold."""
    global _tiktoken_enc
    if _tiktoken_enc is None:
        try:
            import tiktoken

            _tiktoken_enc = tiktoken.encoding_for_model("gpt-4")
        except ImportError:
            return len(text) // 4
    return len(_tiktoken_enc.encode(text))


# Diagram header keywords (first significant line of a Mermaid block).
MERMAID_DIAGRAM_TYPES = {
    "flowchart", "graph", "sequenceDiagram", "classDiagram",
    "stateDiagram", "stateDiagram-v2", "erDiagram", "gantt", "gitGraph",
    "journey", "pie", "mindmap", "timeline", "quadrantChart",
    "requirementDiagram", "C4Context", "C4Container", "C4Component",
    "sankey-beta", "xychart-beta", "block-beta", "packet-beta",
    "architecture-beta", "kanban", "zenuml",
}


def extract_mermaid_diagrams(content: str) -> list[str]:
    """Extract all Mermaid code blocks from markdown content (tolerant of
    trailing spaces after the fence tag and CRLF line endings)."""
    pattern = r"```mermaid[ \t]*\r?\n(.*?)```"
    matches = re.findall(pattern, content, re.DOTALL)
    return [m.strip() for m in matches]


def _first_significant_mermaid_line(code: str) -> str:
    """First non-empty line that is not a Mermaid directive/comment."""
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%%"):
            continue
        return stripped
    return ""


def validate_mermaid_syntax(diagram_code: str) -> tuple[bool, str]:
    """Validate a Mermaid diagram by its header.

    Deliberately conservative: without a real Mermaid parser, deeper checks
    produced false alarms and destructive "fixes" (sequence diagrams were
    rebuilt as flowcharts and deleted). Only an unknown/missing diagram type
    is reported here; anything else is left untouched.

    Returns:
        (is_valid, error_message) - tuple of validity and error message
    """
    if not diagram_code or not diagram_code.strip():
        return False, "Empty diagram code"

    first_line = _first_significant_mermaid_line(diagram_code)
    if not first_line:
        return False, "Empty diagram code"

    diagram_type = first_line.split()[0].rstrip(";,")
    base_type = diagram_type.split("-")[0]
    if (
        diagram_type not in MERMAID_DIAGRAM_TYPES
        and base_type not in MERMAID_DIAGRAM_TYPES
    ):
        return False, f"Unknown diagram type: {diagram_type}"

    return True, "Valid"


def fix_invalid_mermaid(content: str) -> str:
    """Never rewrite or delete Mermaid blocks.

    The previous implementation rebuilt "invalid" blocks with regexes and
    deleted anything it could not salvage — it corrupted valid diagrams
    (e.g. sequenceDiagram) because the validator itself was too strict.
    Invalid blocks are now reported and left for the next LLM/human pass."""
    return content


class FileManager:
    """Handles file I/O operations."""

    @staticmethod
    def ensure_directory(path: str) -> None:
        """Create directory if it doesn't exist."""
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def save_json(data: Any, filepath: str) -> None:
        """Save data as JSON to file (atomically: temp file + rename)."""
        FileManager.save_text(
            json.dumps(data, indent=2, ensure_ascii=False), filepath
        )

    @staticmethod
    def load_json(filepath: str) -> Optional[dict[str, Any]]:
        """Load JSON from file; return None if file doesn't exist."""
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_text(content: str, filepath: str) -> None:
        """Save text content to file atomically (temp file + rename).

        A crash mid-write never leaves a truncated file behind: readers see
        either the old or the new content, never a partial one."""
        directory = os.path.dirname(os.path.abspath(filepath))
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".repodoc-tmp-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            # mkstemp creates 0600 files; os.replace keeps that mode. Generated
            # artifacts are not secrets and should stay group/world readable.
            os.chmod(tmp_path, 0o644)
            os.replace(tmp_path, filepath)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def load_text(filepath: str) -> str:
        """Load text content from file."""
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()


file_manager = FileManager()


class FileLock:
    """Exclusive advisory lock for a single writer across processes.

    Used by `repodoc update` so two concurrent runs cannot interleave graph
    mutation and doc writes. POSIX flock semantics: the lock dies with the
    process (no stale lock files to clean up)."""

    def __init__(self, path: str, timeout: float = 0.0):
        self.path = path
        self.timeout = timeout
        self._fd: Optional[int] = None

    def acquire(self) -> "FileLock":
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX fallback
            fcntl = None

        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                if fcntl is not None:
                    fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    os.close(self._fd)
                    self._fd = None
                    raise TimeoutError(
                        f"Another RepoDoc update is already running "
                        f"(lock: {self.path})"
                    )
                time.sleep(0.2)
        os.ftruncate(self._fd, 0)
        os.write(self._fd, str(os.getpid()).encode("ascii"))
        return self

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            import fcntl

            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except Exception:
            pass
        os.close(self._fd)
        self._fd = None

    def __enter__(self) -> "FileLock":
        return self.acquire()

    def __exit__(self, *exc_info) -> None:
        self.release()


def get_all_markdown_files(directory: str) -> set[str]:
    """Get all markdown files in directory (relative, '/'-separated paths)."""
    md_files = set()
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".md"):
                rel_path = os.path.relpath(os.path.join(root, f), directory)
                md_files.add(rel_path.replace(os.sep, "/"))
    return md_files


def summarize_markdown(content: str, budget: int = 2000) -> str:
    """Structural summary of a generated doc for the parent layer's prompt.

    Keeps the title, the opening paragraph, the section outline, and the
    leading bullets of the document. Unlike a raw `content[:budget]` cut this
    preserves *what the doc covers* even when the body is long, and never
    ends mid-sentence (it truncates on a line boundary)."""
    if not content:
        return ""
    text = content.strip()
    if len(text) <= budget:
        return text

    lines = text.splitlines()
    title = ""
    first_para: list[str] = []
    headings: list[str] = []
    bullets: list[str] = []
    in_para = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            if not title and stripped.startswith("# "):
                title = stripped[:200]
            headings.append(stripped[:120])
            in_para = False
            continue
        if not stripped:
            in_para = False
            continue
        if stripped.startswith(("- ", "* ")) and len(bullets) < 12:
            bullets.append(stripped[:200])
            continue
        if not first_para and not in_para:
            first_para.append(stripped[:400])
            in_para = True
        elif in_para and len(first_para) < 4:
            first_para.append(stripped[:400])

    outline = [h for h in headings if h != title][:20]
    parts: list[str] = []
    if title:
        parts.append(title)
    if first_para:
        parts.append("\n".join(first_para))
    if outline:
        parts.append("Sections:\n" + "\n".join(f"- {h}" for h in outline))
    if bullets:
        parts.append("Key points:\n" + "\n".join(bullets))

    summary = "\n\n".join(parts)
    if len(summary) > budget:
        cut = summary.rfind("\n", 0, budget)
        summary = summary[: cut if cut > 0 else budget]
    return summary or text[:budget]


def extract_markdown_links(content: str) -> list[tuple[str, str]]:
    """Extract all markdown links from content.

    Returns:
        List of (link_text, link_target) tuples
    """
    pattern = r"\[([^\]]+)\]\(([^\)]+)\)"
    matches = re.findall(pattern, content)
    return matches


_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^\)]+)\)")
# Capturing group so re.split KEEPS the fenced blocks in the segment list.
_FENCED_BLOCK_PATTERN = re.compile(r"(```.*?```)", re.DOTALL)
_EXTERNAL_PREFIXES = (
    "http://", "https://", "mailto:", "tel:", "data:", "ftp://", "//", "#",
)


def _link_target_resolves(
    link_target: str, rel_path: str, available_files: set[str]
) -> bool:
    """True when a relative link resolves to an existing generated doc.

    Resolution is relative to the linking file's directory (a link like
    `Model.md` inside `core/A.md` means `core/Model.md`), with site-absolute
    links (`/core/Model.md`) resolved from the docs root."""
    clean = unquote(link_target.split("#")[0].split("?")[0]).strip()
    if not clean:
        return True  # pure anchor/query link
    if clean.startswith("/"):
        resolved = clean.lstrip("/")
    else:
        base_dir = os.path.dirname(rel_path)
        resolved = os.path.normpath(os.path.join(base_dir, clean))
    resolved = resolved.replace(os.sep, "/").replace("\\", "/")
    return resolved in available_files


def validate_and_fix_links(
    docs_dir: str, remove_broken: bool = True, fix_mermaid: bool = True
) -> dict[str, list[str]]:
    """Validate and optionally fix broken links in markdown files.

    Mermaid blocks are never rewritten (see fix_invalid_mermaid). Broken
    links are removed only outside fenced code blocks, and a link counts as
    valid only when it resolves relative to the linking file.

    Args:
        docs_dir: Root directory containing markdown files
        remove_broken: If True, remove broken links; if False, keep them
        fix_mermaid: Kept for API compatibility (no-op).

    Returns:
        Dict with 'fixed', 'skipped', 'removed', and 'mermaid_fixed' lists
    """
    available_files = get_all_markdown_files(docs_dir)

    results = {"fixed": [], "skipped": [], "removed": [], "mermaid_fixed": []}

    for rel_path in sorted(available_files):
        md_file = os.path.join(docs_dir, rel_path)
        try:
            with open(md_file, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as e:
            results["skipped"].append(f"{rel_path}: {e}")
            continue

        original_content = content

        # Split into text / fenced-code segments (capture group keeps the
        # fences); only the text segments are link-checked and rewritten.
        segments = _FENCED_BLOCK_PATTERN.split(content)
        for i in range(0, len(segments), 2):

            def _replace(match, _rel=rel_path):
                link_text, link_target = match.group(1), match.group(2)
                if link_target.startswith(_EXTERNAL_PREFIXES):
                    return match.group(0)
                if _link_target_resolves(link_target, _rel, available_files):
                    return match.group(0)
                broken = f"[{link_text}]({link_target})"
                results["removed"].append(f"{_rel}: {broken}")
                if not remove_broken:
                    return match.group(0)
                return link_text

            segments[i] = _LINK_PATTERN.sub(_replace, segments[i])

        content = "".join(segments)

        if fix_mermaid:
            mermaid_fixed = fix_invalid_mermaid(content)
            if mermaid_fixed != content:
                content = mermaid_fixed
                results["mermaid_fixed"].append(rel_path)

        if content != original_content:
            file_manager.save_text(content, md_file)
            results["fixed"].append(rel_path)

    return results


def log_operation(
    output_dir: str,
    operation_type: str,
    repo_path: str,
    git_commit: Optional[str] = None,
    base_commit: Optional[str] = None,
    duration_seconds: float = 0,
    total_tokens: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    llm_calls: int = 0,
    components_processed: int = 0,
    components_updated: int = 0,
    files_generated: int = 0,
    status: str = "success",
    error_message: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Log a LiveDoc operation to a JSONL file.

    Args:
        output_dir: Output directory for the documentation
        operation_type: Type of operation (full_generation, incremental_update, etc.)
        repo_path: Path to the repository
        git_commit: Current git commit hash
        base_commit: Base commit for comparison (for incremental updates)
        duration_seconds: Duration of the operation in seconds
        total_tokens: Total tokens used
        prompt_tokens: Prompt tokens used
        completion_tokens: Completion tokens used
        llm_calls: Number of LLM calls made
        components_processed: Number of components processed
        components_updated: Number of components updated (for incremental)
        files_generated: Number of files generated
        status: Operation status (success, failed, aborted)
        error_message: Error message if status is not success
        metadata: Additional metadata
    """
    log_file = os.path.join(output_dir, "data", "operations.log.jsonl")

    entry = {
        "timestamp": datetime.now().isoformat(),
        "operation_type": operation_type,
        "repo_path": repo_path,
        "git_commit": git_commit,
        "base_commit": base_commit,
        "duration_seconds": round(duration_seconds, 2),
        "token_usage": {
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "llm_calls": llm_calls,
        },
        "components_processed": components_processed,
        "components_updated": components_updated,
        "files_generated": files_generated,
        "status": status,
        "error_message": error_message,
    }

    if metadata:
        entry["metadata"] = metadata

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Safe file reads (moved from src/analysis/utils/security.py)
# ---------------------------------------------------------------------------


def _inside(base: Path, target: Path) -> bool:
    base_r = base.resolve()
    try:
        target_r = target.resolve()
        return target_r.is_relative_to(base_r)  # py>=3.9
    except AttributeError:
        return str(target.resolve()).startswith(str(base_r))


def assert_safe_path(base_dir: Path, target: Path) -> None:
    """Block symlinks and paths that escape the repo."""
    if target.is_symlink():
        raise PermissionError(f"Symlink blocked: {target}")
    if not _inside(base_dir, target):
        raise PermissionError(f"Path escapes repo: {target} -> {target.resolve()}")


def safe_open_text(base_dir: Path, target: Path, encoding: str = "utf-8") -> str:
    """Read file content safely (no symlink follow, no path escape)."""
    assert_safe_path(base_dir, target)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(str(target), flags)
    try:
        with os.fdopen(fd, "r", encoding=encoding, errors="replace") as f:
            return f.read()
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


__all__ = [
    "FileManager",
    "file_manager",
    "count_tokens",
    "extract_mermaid_diagrams",
    "validate_mermaid_syntax",
    "fix_invalid_mermaid",
    "validate_and_fix_links",
    "get_all_markdown_files",
    "extract_markdown_links",
    "log_operation",
    "safe_open_text",
    "assert_safe_path",
]
