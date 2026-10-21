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


def test_python_method_signature_is_a_severe_class_change():
    from repodoc.agent.update import ChangeType, DiffSpec, analyze_diff

    old = "class A:\n    def run(self, x):\n        return x\n"
    new = "class A:\n    def run(self, x, y=1):\n        return x\n"
    analysis = analyze_diff("m.py", old, new, DiffSpec("python"))
    change = next(c for c in analysis.changes if c.name == "A")
    assert change.change_type == ChangeType.API_SIGNATURE_CHANGED


def test_java_method_node_points_at_owning_class():
    from repodoc.graph.analyzers import analyze_file

    nodes, _rels, _imports = analyze_file(
        "demo/User.java",
        "package demo;\npublic class User {\n  public void save() {}\n}\n",
        ".",
    )
    method = next(n for n in nodes if n.component_type == "method")
    assert method.id.endswith("User.save")
    assert method.owner is not None
    assert method.owner.endswith("User.User")


def test_in_file_component_rename_is_paired():
    from repodoc.agent.update import _detect_component_renames
    from repodoc.graph.graph import HeterogeneousGraph
    from repodoc.graph.models import CodeNode, Node
    from repodoc.graph.mutation import GraphMutationResult

    old_source = (
        "class OldName:\n"
        "    def run(self, value):\n"
        "        result = value + 1\n"
        "        return result\n"
    )
    new_source = old_source.replace("OldName", "NewName")
    pre = {
        "m.OldName": Node(
            id="m.OldName", name="OldName", component_type="class",
            file_path="m.py", relative_path="m.py", source_code=old_source,
        )
    }
    graph = HeterogeneousGraph()
    graph.add_code_node(
        CodeNode(
            id="m.NewName", type="class", component_type="class",
            file_path="m.py", relative_path="m.py", name="NewName",
            source_code=new_source,
        )
    )
    mutation = GraphMutationResult(
        added=["m.NewName"], removed=["m.OldName"]
    )
    assert _detect_component_renames(pre, graph, mutation) == [
        ("m.OldName", "m.NewName")
    ]


def test_method_changes_map_to_owner_docs():
    from repodoc.agent.update import _map_methods_to_owners
    from repodoc.graph.models import Node

    components = {
        "m.User": Node(
            id="m.User", name="User", component_type="class",
            file_path="m.py", relative_path="m.py",
        ),
        "m.User.save": Node(
            id="m.User.save", name="save", component_type="method",
            file_path="m.py", relative_path="m.py", owner="m.User",
        ),
    }
    assert _map_methods_to_owners(components, ["m.User.save", "m.User"]) == ["m.User"]


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
