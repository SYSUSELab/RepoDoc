"""Knowledge-graph completeness: unresolved refs reconcile, cluster signals."""

from pathlib import Path

from repodoc.config import Config
from repodoc.graph import DependencyGraphBuilder
from repodoc.graph.builder import build_graph_from_analysis
from repodoc.graph.cluster import (
    MAX_CLUSTER_PROMPT_TOKENS,
    _format_cluster_input,
    cluster_modules,
)
from repodoc.graph.models import Node
from repodoc.graph.mutation import (
    apply_file_changes_to_graph,
    reconcile_unresolved_references,
)


def _build(tmp_path, files: dict[str, str]):
    repo = tmp_path / "repo"
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    config = Config(repo_path=str(repo), output_dir=str(tmp_path / "out"))
    components, _ = DependencyGraphBuilder(config).build_dependency_graph()
    return repo, components, build_graph_from_analysis(components)


def test_unresolved_reference_is_stored_on_the_node(tmp_path):
    _repo, components, graph = _build(
        tmp_path, {"src/a.py": "def caller():\n    return helper()\n"}
    )
    assert components["src.a.caller"].unresolved == {"calls": {"helper"}}
    assert graph.get_node("src.a.caller").unresolved == {"calls": ["helper"]}


def test_new_definition_reconciles_unresolved_reference(tmp_path):
    repo, _components, graph = _build(
        tmp_path, {"src/a.py": "def caller():\n    return helper()\n"}
    )
    (repo / "src" / "b.py").write_text("def helper():\n    return 1\n")
    result = apply_file_changes_to_graph(
        graph, "src/b.py", "def helper():\n    return 1\n", str(repo)
    )
    assert result.added == ["src.b.helper"]
    assert reconcile_unresolved_references(graph) == 1
    caller = graph.get_node("src.a.caller")
    assert caller.depends_on == ["src.b.helper"]
    assert caller.unresolved == {}
    assert graph.has_edge("src.a.caller", "src.b.helper")


def test_ambiguous_reference_stays_unresolved(tmp_path):
    repo, _components, graph = _build(
        tmp_path, {"src/a.py": "def caller():\n    return helper()\n"}
    )
    (repo / "src" / "b.py").write_text("def helper():\n    return 1\n")
    (repo / "src" / "c.py").write_text("def helper():\n    return 2\n")
    apply_file_changes_to_graph(
        graph, "src/b.py", "def helper():\n    return 1\n", str(repo)
    )
    apply_file_changes_to_graph(
        graph, "src/c.py", "def helper():\n    return 2\n", str(repo)
    )
    # One reconciliation pass for the whole run: two fresh definitions with
    # the same name are ambiguous and must not be guessed.
    assert reconcile_unresolved_references(graph) == 0
    caller = graph.get_node("src.a.caller")
    assert caller.depends_on == []
    assert caller.unresolved == {"calls": ["helper"]}
    assert not graph.has_edge("src.a.caller", "src.b.helper")
    assert not graph.has_edge("src.a.caller", "src.c.helper")


def test_cluster_prompt_includes_semantic_signals():
    components = {
        "m.fetch_user": Node(
            id="m.fetch_user",
            name="fetch_user",
            component_type="function",
            file_path="m.py",
            relative_path="m.py",
            docstring="Load a user record from the remote API.",
            parameters=["user_id"],
        )
    }
    text = _format_cluster_input(["m.fetch_user"], components)
    assert "[function]" in text
    assert "Load a user record from the remote API." in text
    assert "(user_id)" in text


def test_cluster_prompt_falls_back_to_names_when_too_large(monkeypatch):
    components = {
        f"m.f{i}": Node(
            id=f"m.f{i}",
            name=f"f{i}",
            component_type="function",
            file_path="m.py",
            relative_path="m.py",
            docstring="word " * 40,
        )
        for i in range(20)
    }
    monkeypatch.setattr(
        "repodoc.graph.cluster.MAX_CLUSTER_PROMPT_TOKENS", 1
    )
    text = _format_cluster_input(list(components), components)
    assert "—" not in text  # docstring signals dropped
    assert "m.f0" in text


def test_heuristic_clustering_excludes_private_components(tmp_path):
    _repo, components, _graph = _build(
        tmp_path,
        {
            "src/m.py": (
                "def public():\n    return 1\n\n\n"
                "def _private():\n    return 2\n"
            )
        },
    )
    config = Config(
        repo_path=str(tmp_path / "repo"),
        output_dir=str(tmp_path / "out2"),
        cluster_strategy="heuristic",
    )
    tree = cluster_modules(
        ["src.m.public", "src.m._private"], components, config
    )
    listed = {
        cid
        for info in tree.values()
        for cid in info.get("components", [])
    }
    assert "src.m.public" in listed
    assert "src.m._private" not in listed
