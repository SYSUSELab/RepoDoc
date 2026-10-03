"""Shared fixtures: a tiny git repo + a prebuilt knowledge graph."""

import json
import os
import subprocess
from pathlib import Path

import pytest


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    """A committed git repo with one package and two components."""
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "mod.py").write_text(
        "class Alpha:\n"
        "    def run(self):\n"
        "        return 1\n"
        "\n"
        "\n"
        "def helper():\n"
        "    return Alpha().run()\n"
    )
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "init")
    return repo


@pytest.fixture
def built_project(mini_repo: Path, tmp_path: Path):
    """(config, graph, base_commit) with KG + metadata as `generate` leaves them."""
    from repodoc.config import Config
    from repodoc.graph import DependencyGraphBuilder
    from repodoc.graph.builder import (
        attach_module_tree_to_graph,
        build_graph_from_analysis,
    )

    config = Config(
        repo_path=str(mini_repo), output_dir=str(tmp_path / "output")
    )
    components, _leaves = DependencyGraphBuilder(config).build_dependency_graph()
    graph = build_graph_from_analysis(components)

    tree: dict = {}
    for cid, node in components.items():
        top = node.relative_path.split("/")[0]
        tree.setdefault(top, {"path": top, "components": [], "children": {}})
        tree[top]["components"].append(cid)
    attach_module_tree_to_graph(graph, tree, components)

    data_dir = Path(config.output_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    graph.save(str(data_dir / f"{mini_repo.name}_knowledge_graph.json"))

    base_commit = _git(mini_repo, "rev-parse", "HEAD")
    (data_dir / f"{mini_repo.name}_metadata.json").write_text(
        json.dumps({"generation_info": {"git_commit": base_commit}})
    )
    Path(config.docs_dir).mkdir(parents=True, exist_ok=True)
    return config, graph, base_commit


def make_fake_execute(fail_component_names: set[str] | None = None):
    """A DocAgent.execute replacement that writes files without any LLM.

    Components whose name is in `fail_component_names` return a failed
    SkillResult; everything else succeeds and writes the output file."""
    from pathlib import Path as _Path

    from repodoc.agent.agent import SkillResult

    async def fake_execute(self, task):
        data = task.get("input", {})
        doc_type = data.get("type", "")
        if doc_type in ("component", "update_component"):
            name = data.get("component", {}).get("name", "")
            if fail_component_names and name in fail_component_names:
                return SkillResult(success=False, error="injected failure")
        out = data.get("output_path", "")
        content = f"# {doc_type}: {data.get('module_name') or data.get('repo_name', 'doc')}\n"
        if out:
            _Path(out).parent.mkdir(parents=True, exist_ok=True)
            _Path(out).write_text(content)
        return SkillResult(
            success=True, output={"content": content, "path": out}
        )

    return fake_execute
