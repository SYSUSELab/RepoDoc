"""P2 safety rails: LLM output validation, write containment, ignores,
mermaid/link non-destructiveness, TS method ids."""

import asyncio
from pathlib import Path

from repodoc.agent.agent import validate_llm_output
from repodoc.agent.tools import FileWriterTool
from repodoc.agent.update import ChangeType, DiffSpec, analyze_diff
from repodoc.config import Config
from repodoc.graph import DependencyGraphBuilder
from repodoc.graph.analyzers import DEFAULT_IGNORE_PATTERNS
from repodoc.graph.builder import _should_exclude_path
from repodoc.utils import (
    fix_invalid_mermaid,
    validate_and_fix_links,
    validate_mermaid_syntax,
)


# --- LLM output validation -------------------------------------------------


def test_validate_llm_output_rejects_empty_and_truncated():
    assert validate_llm_output("") is not None
    assert validate_llm_output("   \n") is not None
    assert validate_llm_output("# short") is not None
    assert validate_llm_output("# Title\n\nSome text ```code") is not None
    assert validate_llm_output("# Title\n\nA reasonable document body.") is None


def test_agent_rejects_empty_llm_output_before_writing(tmp_path):
    from repodoc.agent.agent import DocAgent

    config = Config(repo_path=".", output_dir=str(tmp_path / "out"))
    agent = DocAgent(config)
    out = Path(config.docs_dir) / "x.md"
    result = asyncio.run(
        agent._write_result(
            str(out), "   \n", "component", agent.tools["file_writer"]
        )
    )
    assert result.success is False
    assert "empty LLM output" in (result.error or "")
    assert not out.exists()


# --- Write containment ------------------------------------------------------


def test_file_writer_stays_inside_root(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    tool = FileWriterTool(root=str(docs))

    ok = asyncio.run(tool.execute(str(docs / "sub" / "a.md"), "# ok"))
    assert ok.success
    assert (docs / "sub" / "a.md").exists()

    # Bare filename (no directory part) must work, not crash on makedirs("").
    bare = asyncio.run(tool.execute(str(docs / "root.md"), "# root"))
    assert bare.success

    escape = asyncio.run(tool.execute(str(docs / ".." / "evil.md"), "x"))
    assert not escape.success
    assert not (tmp_path / "evil.md").exists()

    outside = asyncio.run(tool.execute(str(tmp_path / "evil2.md"), "x"))
    assert not outside.success


# --- Ignore patterns / oversized files --------------------------------------


def test_dependency_directories_are_ignored():
    patterns = set(DEFAULT_IGNORE_PATTERNS)
    assert _should_exclude_path(
        "node_modules/pkg/index.js", "index.js", patterns
    )
    assert _should_exclude_path("vendor/lib/x.c", "x.c", patterns)
    assert _should_exclude_path("target/debug/x.rs", "x.rs", patterns)
    assert not _should_exclude_path("src/index.js", "index.js", patterns)


def test_node_modules_not_analyzed(tmp_path):
    repo = tmp_path / "repo"
    (repo / "node_modules" / "pkg").mkdir(parents=True)
    (repo / "node_modules" / "pkg" / "index.js").write_text(
        "export function vendored() { return 1; }\n"
    )
    (repo / "src").mkdir()
    (repo / "src" / "app.js").write_text("export function app() { return 2; }\n")
    config = Config(repo_path=str(repo), output_dir=str(tmp_path / "out"))
    components, _ = DependencyGraphBuilder(config).build_dependency_graph()
    assert "src.app.app" in components
    assert not any("node_modules" in cid for cid in components)


# --- Mermaid / links --------------------------------------------------------


def test_valid_sequence_diagram_is_not_corrupted():
    code = "sequenceDiagram\n    Alice->Bob: Hello\n    Bob-->>Alice: Hi"
    valid, _ = validate_mermaid_syntax(code)
    assert valid
    content = f"# Doc\n\n```mermaid\n{code}\n```\n"
    assert fix_invalid_mermaid(content) == content


def test_spaced_and_unspaced_flowcharts_are_valid():
    assert validate_mermaid_syntax("flowchart LR\n    A-->B")[0]
    assert validate_mermaid_syntax("flowchart TD\n    A --> B")[0]


def test_unknown_mermaid_header_is_reported_but_kept():
    code = "notADiagram\n    x --> y"
    valid, error = validate_mermaid_syntax(code)
    assert not valid and "Unknown diagram type" in error
    content = f"```mermaid\n{code}\n```\n"
    assert fix_invalid_mermaid(content) == content


def test_link_validation_is_relative_and_skips_code_blocks(tmp_path):
    docs = tmp_path / "docs"
    (docs / "core").mkdir(parents=True)
    (docs / "core" / "A.md").write_text(
        "# A\n\n"
        "[Good](B.md)\n"
        "[Bad](missing.md)\n"
        "\n"
        "```python\n"
        "[Inside](missing.md)\n"
        "```\n"
    )
    (docs / "core" / "B.md").write_text("# B\n")

    results = validate_and_fix_links(str(docs))
    updated = (docs / "core" / "A.md").read_text()

    assert "[Good](B.md)" in updated
    assert "[Bad](missing.md)" not in updated
    assert "Bad" in updated  # broken link text is kept, link syntax removed
    assert "[Inside](missing.md)" in updated  # code blocks untouched
    assert any("A.md" in r for r in results["removed"])


def test_local_file_named_like_scheme_is_checked(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "http_notes.md").write_text("# notes\n")
    (docs / "A.md").write_text("# A\n\n[notes](http_notes.md)\n")
    validate_and_fix_links(str(docs))
    assert "[notes](http_notes.md)" in (docs / "A.md").read_text()


# --- TS method ids ----------------------------------------------------------


def test_ts_method_ids_are_qualified_by_class(tmp_path):
    old = (
        "export class A {\n"
        "  save(x: number) { return x; }\n"
        "}\n"
        "export class B {\n"
        "  save(x: number) { return x; }\n"
        "}\n"
    )
    analysis = analyze_diff(
        "m.ts",
        old,
        old.replace("export class B {\n  save(x: number) { return x; }\n}\n", ""),
        DiffSpec("typescript"),
    )
    removed = {(c.component_id, c.change_type) for c in analysis.changes}
    assert ("m.B.save", ChangeType.REMOVED_COMPONENT) in removed
