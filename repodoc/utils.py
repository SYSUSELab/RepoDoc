"""Shared utilities for RepoDoc."""

import os
import json
import re
import glob
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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


def extract_mermaid_diagrams(content: str) -> list[str]:
    """Extract all Mermaid code blocks from markdown content."""
    pattern = r"```mermaid\n(.*?)```"
    matches = re.findall(pattern, content, re.DOTALL)
    return [m.strip() for m in matches]


def validate_mermaid_syntax(diagram_code: str) -> tuple[bool, str]:
    """Validate Mermaid diagram syntax.

    Returns:
        (is_valid, error_message) - tuple of validity and error message
    """
    if not diagram_code:
        return False, "Empty diagram code"

    diagram_type = diagram_code.split()[0] if diagram_code.split() else ""

    valid_types = [
        "flowchart",
        "graph",
        "pie",
        "sequenceDiagram",
        "classDiagram",
        "stateDiagram",
        "stateDiagram-v2",
        "erDiagram",
        "gantt",
        "gitGraph",
        "journey",
    ]

    if diagram_type not in valid_types:
        return False, f"Unknown diagram type: {diagram_type}"

    if diagram_type in ("flowchart", "graph"):
        return _validate_flowchart(diagram_code)
    elif diagram_type == "sequenceDiagram":
        return _validate_sequence(diagram_code)
    elif diagram_type == "classDiagram":
        return _validate_class(diagram_code)

    return True, "Valid"


def _validate_flowchart(code: str) -> tuple[bool, str]:
    """Validate flowchart/graph syntax."""
    if " --> " not in code and " --- " not in code:
        return False, "No edges found in flowchart"
    return True, "Valid"


def _validate_sequence(code: str) -> tuple[bool, str]:
    """Validate sequence diagram syntax."""
    lines = code.split("\n")
    participants = set()
    for line in lines:
        line = line.strip()
        if "->>" in line:
            parts = re.split(r"[->>]+", line)
            participants.update([p.strip() for p in parts if p.strip()])
    if not participants:
        return False, "No participants in sequence diagram"
    return True, "Valid"


def _validate_class(code: str) -> tuple[bool, str]:
    """Validate class diagram syntax."""
    if "class " not in code:
        return False, "No class definitions found"
    return True, "Valid"


def fix_invalid_mermaid(content: str) -> str:
    """Fix or remove invalid Mermaid diagrams in markdown content.

    Returns:
        Content with invalid Mermaid diagrams fixed or removed.
    """
    pattern = r"(```mermaid\n)(.*?)(```)"

    def replace_diagram(match):
        diagram_code = match.group(2).strip()
        is_valid, _ = validate_mermaid_syntax(diagram_code)

        if is_valid:
            return match.group(0)

        # Try to create a simple valid flowchart
        lines = diagram_code.split("\n")
        valid_lines = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            # Keep lines that look like valid mermaid syntax
            if any(x in line for x in ["-->", "---", "--", "==", "|", "+", "}", "{"]):
                valid_lines.append(line)

        if valid_lines:
            return (
                "```mermaid\nflowchart TD\n    "
                + "\n    ".join(valid_lines[:10])
                + "\n```"
            )

        # Remove the entire block if it can't be fixed
        return ""

    return re.sub(pattern, replace_diagram, content, flags=re.DOTALL)


class FileManager:
    """Handles file I/O operations."""

    @staticmethod
    def ensure_directory(path: str) -> None:
        """Create directory if it doesn't exist."""
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def save_json(data: Any, filepath: str) -> None:
        """Save data as JSON to file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_json(filepath: str) -> Optional[dict[str, Any]]:
        """Load JSON from file; return None if file doesn't exist."""
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_text(content: str, filepath: str) -> None:
        """Save text content to file."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def load_text(filepath: str) -> str:
        """Load text content from file."""
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()


file_manager = FileManager()


def get_all_markdown_files(directory: str) -> set[str]:
    """Get all markdown files in directory (relative paths)."""
    md_files = set()
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".md"):
                rel_path = os.path.relpath(os.path.join(root, f), directory)
                md_files.add(rel_path)
    return md_files


def extract_markdown_links(content: str) -> list[tuple[str, str]]:
    """Extract all markdown links from content.

    Returns:
        List of (link_text, link_target) tuples
    """
    pattern = r"\[([^\]]+)\]\(([^\)]+)\)"
    matches = re.findall(pattern, content)
    return matches


def validate_and_fix_links(
    docs_dir: str, remove_broken: bool = True, fix_mermaid: bool = True
) -> dict[str, list[str]]:
    """Validate and optionally fix broken links and Mermaid diagrams in markdown files.

    Args:
        docs_dir: Root directory containing markdown files
        remove_broken: If True, remove broken links; if False, keep them
        fix_mermaid: If True, fix invalid Mermaid diagrams; if False, remove them

    Returns:
        Dict with 'fixed', 'skipped', 'removed', and 'mermaid_fixed' lists
    """
    available_files = get_all_markdown_files(docs_dir)
    available_basenames = {os.path.splitext(f)[0] for f in available_files}

    results = {"fixed": [], "skipped": [], "removed": [], "mermaid_fixed": []}

    for md_file in glob.glob(os.path.join(docs_dir, "**/*.md"), recursive=True):
        rel_path = os.path.relpath(md_file, docs_dir)
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        links = extract_markdown_links(content)

        for link_text, link_target in links:
            if link_target.startswith(("http", "https", "mailto:", "#")):
                continue

            link_target_clean = link_target.split("#")[0].split("?")[0]
            if not link_target_clean:
                continue

            target_basename = os.path.splitext(os.path.basename(link_target_clean))[0]

            if link_target_clean in available_files:
                continue

            if target_basename in available_basenames:
                continue

            if not remove_broken:
                continue

            broken_link = f"[{link_text}]({link_target})"
            content = content.replace(broken_link, link_text)
            results["removed"].append(f"{rel_path}: {broken_link}")

        if fix_mermaid:
            mermaid_fixed = fix_invalid_mermaid(content)
            if mermaid_fixed != content:
                content = mermaid_fixed
                results["mermaid_fixed"].append(rel_path)

        if content != original_content:
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(content)
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
