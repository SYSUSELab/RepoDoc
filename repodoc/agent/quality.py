"""Lightweight quality checks on generated docs.

Hard check — mermaid blocks must be well-formed enough to render; a broken
diagram in the README is a visible defect. Kept conservative: only definite
problems are flagged (missing diagram-type header, dangling arrows, unclosed
brackets), so the check never cries wolf on valid diagrams.

Soft check — backticked dotted identifiers whose root matches a known
component/module are counted; results are recorded as statistics only.
Method-level references (``Server.serve``) cannot be validated without
method-level analysis, so no per-reference warnings are emitted.
"""

import re

from repodoc.utils import extract_mermaid_diagrams

# Valid first lines of a mermaid block (diagram type declarations).
_MERMAID_HEADER = re.compile(
    r"^\s*(flowchart|graph|sequenceDiagram|classDiagram|stateDiagram(-v2)?"
    r"|erDiagram|gantt|pie|journey|mindmap|timeline|gitGraph)\b"
)

# Lines that legitimately do not contain arrows/statements.
_MERMAID_STRUCTURAL = re.compile(
    r"^\s*(%%|subgraph\b|end\b|classDef\b|class\b|style\b|linkStyle\b"
    r"|click\b|direction\b|title\b|section\b)"
)

# A dangling arrow: '-->' or '-.->' with nothing after it on the line.
_DANGLING_ARROW = re.compile(r"(-{2,3}>|-\.->)\s*$")


def validate_mermaid(content: str) -> list[str]:
    """Return definite syntax problems in the mermaid blocks of a document
    (empty list = all blocks look renderable)."""
    problems: list[str] = []
    for i, block in enumerate(extract_mermaid_diagrams(content)):
        stripped = block.strip()
        if not stripped:
            problems.append(f"mermaid block {i + 1}: empty")
            continue
        lines = stripped.splitlines()
        first = lines[0].strip()
        if not _MERMAID_HEADER.match(first):
            problems.append(
                f"mermaid block {i + 1}: missing diagram type header "
                f"(starts with {first[:40]!r})"
            )
        for line in lines[1:]:
            if not line.strip() or _MERMAID_STRUCTURAL.match(line):
                continue
            if _DANGLING_ARROW.search(line):
                problems.append(
                    f"mermaid block {i + 1}: dangling arrow in {line.strip()[:50]!r}"
                )
            if line.count("[") != line.count("]") or line.count("(") != line.count(")"):
                problems.append(
                    f"mermaid block {i + 1}: unclosed bracket in {line.strip()[:50]!r}"
                )
    return problems


_DOTTED_REF = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)`")


def reference_stats(content: str, known_roots: set[str]) -> dict[str, int]:
    """Count backticked dotted references by whether their root matches a
    known project entity (component name or module name)."""
    known = 0
    unknown = 0
    for ref in _DOTTED_REF.findall(content):
        if ref.split(".")[0] in known_roots:
            known += 1
        else:
            unknown += 1
    return {"project_refs": known, "external_refs": unknown}
