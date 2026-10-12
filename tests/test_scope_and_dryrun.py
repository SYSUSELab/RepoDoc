"""Analysis/doc scope options and update --dry-run."""

import asyncio
import json
from pathlib import Path

from repodoc.config import Config
from repodoc.graph import DependencyGraphBuilder


def _repo(tmp_path, files: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return repo


def test_include_patterns_limit_analysis(tmp_path):
    repo = _repo(
        tmp_path,
        {
            "src/app.py": "def alpha():\n    return 1\n",
            "src/app.js": "export function beta() { return 2; }\n",
        },
    )
    config = Config(
        repo_path=str(repo),
        output_dir=str(tmp_path / "out"),
        include_patterns=["*.py"],
    )
    components, _ = DependencyGraphBuilder(config).build_dependency_graph()
    assert any(".alpha" in cid for cid in components)
    assert not any("beta" in cid for cid in components)


def test_exclude_patterns_skip_paths(tmp_path):
    repo = _repo(
        tmp_path,
        {
            "src/keep/mod.py": "def keep():\n    return 1\n",
            "src/skip/mod.py": "def dropme():\n    return 1\n",
        },
    )
    config = Config(
        repo_path=str(repo),
        output_dir=str(tmp_path / "out"),
        exclude_patterns=["skip"],
    )
    components, _ = DependencyGraphBuilder(config).build_dependency_graph()
    assert any(".keep" in cid for cid in components)
    assert not any(".dropme" in cid for cid in components)


def test_doc_types_override_auto_selection(tmp_path):
    repo = _repo(
        tmp_path,
        {"m.py": "class Alpha:\n    pass\n\n\ndef helper():\n    return 1\n"},
    )
    auto = Config(repo_path=str(repo), output_dir=str(tmp_path / "o1"))
    _, auto_leaves = DependencyGraphBuilder(auto).build_dependency_graph()
    assert auto_leaves and all(cid.endswith(".Alpha") for cid in auto_leaves)

    functions = Config(
        repo_path=str(repo),
        output_dir=str(tmp_path / "o2"),
        doc_types=["function"],
    )
    _, fn_leaves = DependencyGraphBuilder(functions).build_dependency_graph()
    assert fn_leaves and all(cid.endswith(".helper") for cid in fn_leaves)


def test_max_docs_caps_and_ranks_by_dependents(tmp_path):
    repo = _repo(
        tmp_path,
        {
            "m.py": (
                "def f1():\n    return 1\n\n\n"
                "def f2():\n    return f1()\n\n\n"
                "def f3():\n    return 3\n"
            )
        },
    )
    config = Config(
        repo_path=str(repo), output_dir=str(tmp_path / "out"), max_docs=1
    )
    _, leaves = DependencyGraphBuilder(config).build_dependency_graph()
    assert len(leaves) == 1
    assert leaves[0].endswith(".f1")  # the most depended-upon function


def test_update_dry_run_changes_nothing(built_project):
    from repodoc.agent.update import run_update

    config, _graph, base_commit = built_project
    repo = Path(config.repo_path)
    source = repo / "pkg" / "mod.py"
    source.write_text(source.read_text().replace("return 1", "return 2"))

    data_dir = Path(config.output_dir) / "data"
    kg_path = data_dir / "repo_knowledge_graph.json"
    metadata_path = data_dir / "repo_metadata.json"
    kg_before = kg_path.read_bytes()
    metadata_before = metadata_path.read_bytes()
    docs_before = sorted(p.name for p in Path(config.docs_dir).rglob("*.md"))

    result = asyncio.run(run_update(config, dry_run=True))

    assert result["status"] == "dry_run"
    assert result["estimated_llm_calls"] >= 1
    assert result["files_changed"] >= 1
    assert kg_path.read_bytes() == kg_before
    assert metadata_path.read_bytes() == metadata_before
    assert not (data_dir / "repo_pending_updates.json").exists()
    assert sorted(p.name for p in Path(config.docs_dir).rglob("*.md")) == docs_before
    assert json.loads(metadata_path.read_text())["generation_info"][
        "git_commit"
    ] == base_commit
