"""Knowledge-graph relation typing: extends/implements are not calls."""

import os

from repodoc.config import Config
from repodoc.graph import DependencyGraphBuilder
from repodoc.graph.builder import graph_to_components, build_graph_from_analysis
from repodoc.graph.mutation import apply_file_changes_to_graph


def _build(tmp_path, files: dict[str, str]):
    repo = tmp_path / "repo"
    repo.mkdir()
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    config = Config(repo_path=str(repo), output_dir=str(tmp_path / "out"))
    components, _ = DependencyGraphBuilder(config).build_dependency_graph()
    return repo, graph_to_components(build_graph_from_analysis(components)), (
        build_graph_from_analysis(components)
    )


def test_python_inheritance_becomes_implements_edge(tmp_path):
    repo, components, graph = _build(
        tmp_path,
        {
            "src/animal.py": (
                "class Animal:\n"
                "    def speak(self):\n"
                "        return '...'\n"
                "\n"
                "\n"
                "class Dog(Animal):\n"
                "    def speak(self):\n"
                "        return 'woof'\n"
            )
        },
    )
    dog = components["src.animal.Dog"]
    assert dog.implements == {"src.animal.Animal"}
    assert dog.depends_on == set()
    edges = {(s, t, rt) for s, t, rt, _w in graph.edges()}
    assert ("src.animal.Dog", "src.animal.Animal", "implements") in edges
    assert not any(rt == "calls" for _s, _t, rt in edges)


def test_typescript_implements_clause(tmp_path):
    _repo, components, graph = _build(
        tmp_path,
        {
            "src/pet.ts": (
                "export interface Pet { name: string }\n"
                "export class Cat implements Pet {\n"
                '  name = "cat";\n'
                "}\n"
            )
        },
    )
    assert components["src.pet.Cat"].implements == {"src.pet.Pet"}
    assert ("src.pet.Cat", "src.pet.Pet", "implements") in {
        (s, t, rt) for s, t, rt, _w in graph.edges()
    }


def test_impl_change_is_a_severe_api_change(tmp_path):
    from repodoc.agent.update import ChangeType, DiffSpec, analyze_diff

    old = "class A:\n    pass\n\n\nclass B(A):\n    pass\n"
    new = "class A:\n    pass\n\n\nclass C:\n    pass\n\n\nclass B(C):\n    pass\n"
    analysis = analyze_diff("m.py", old, new, DiffSpec("python"))
    change = next(c for c in analysis.changes if c.name == "B")
    assert change.change_type == ChangeType.API_SIGNATURE_CHANGED


def test_same_file_resolution_change_updates_subclass(tmp_path):
    """Adding the base class after the subclass must resolve the edge."""
    repo, _components, graph = _build(
        tmp_path,
        {
            "src/animal.py": (
                "class Animal:\n"
                "    pass\n"
                "\n"
                "\n"
                "class Dog(Animal):\n"
                "    pass\n"
            )
        },
    )
    changed = (
        "class Animal:\n"
        "    pass\n"
        "\n"
        "\n"
        "class Base:\n"
        "    pass\n"
        "\n"
        "\n"
        "class Dog(Base):\n"
        "    pass\n"
    )
    result = apply_file_changes_to_graph(
        graph, "src/animal.py", changed, str(repo)
    )
    assert "src.animal.Base" in result.added
    assert "src.animal.Dog" in result.updated
    assert graph.get_node("src.animal.Dog").implements == ["src.animal.Base"]
    assert graph.has_edge("src.animal.Dog", "src.animal.Base")


def test_empty_file_is_not_a_deletion(tmp_path):
    repo, _components, graph = _build(
        tmp_path,
        {"src/a.py": "def alpha():\n    return 1\n"},
    )
    result = apply_file_changes_to_graph(graph, "src/a.py", "", str(repo))
    assert result.removed == []
    assert graph.has_node("src.a.alpha")


def test_none_content_is_a_deletion(tmp_path):
    repo, _components, graph = _build(
        tmp_path,
        {"src/a.py": "def alpha():\n    return 1\n"},
    )
    result = apply_file_changes_to_graph(graph, "src/a.py", None, str(repo))
    assert result.removed == ["src.a.alpha"]
    assert not graph.has_node("src.a.alpha")


def test_impact_propagation_follows_implements(tmp_path):
    from repodoc.agent.update import IncrementalUpdater

    repo, _components, graph = _build(
        tmp_path,
        {
            "src/animal.py": (
                "class Animal:\n"
                "    pass\n"
                "\n"
                "\n"
                "class Dog(Animal):\n"
                "    pass\n"
            )
        },
    )
    updater = IncrementalUpdater(graph, Config(repo_path=str(repo)))
    impact = updater.compute_layered_impact(
        ["src.animal.Animal"], {"src.animal.Animal"}, max_hops=2
    )
    assert impact["caller_components"] == ["src.animal.Dog"]
