"""Scope-aware reference resolution and typed reference relations."""

from repodoc.config import Config
from repodoc.graph import DependencyGraphBuilder


def _build(tmp_path, files: dict[str, str]):
    repo = tmp_path / "repo"
    repo.mkdir()
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    config = Config(repo_path=str(repo), output_dir=str(tmp_path / "out"))
    components, _ = DependencyGraphBuilder(config).build_dependency_graph()
    return components


def test_same_file_reference_wins_over_other_files(tmp_path):
    components = _build(
        tmp_path,
        {
            "a.java": (
                "package pkg.a;\n"
                "class ServiceA {}\n"
                "class Client {\n"
                "  ServiceA service;\n"
                "}\n"
            ),
            "b.java": (
                "package pkg.b;\n"
                "class ServiceA {}\n"
                "class Client {\n"
                "  ServiceA service;\n"
                "}\n"
            ),
        },
    )
    assert components["a.Client"].uses == {"a.ServiceA"}
    assert components["b.Client"].uses == {"b.ServiceA"}


def test_ambiguous_global_reference_is_dropped(tmp_path):
    components = _build(
        tmp_path,
        {
            "a.java": "package pkg.a;\nclass ServiceA {}\n",
            "b.java": "package pkg.b;\nclass ServiceA {}\n",
            "c.java": (
                "package pkg.c;\n"
                "class Client {\n"
                "  ServiceA service;\n"
                "}\n"
            ),
        },
    )
    client = components["c.Client"]
    # Bare "ServiceA" matches two files: guessing would create a wrong edge.
    assert client.uses == set()
    assert client.depends_on == set()


def test_unique_global_reference_resolves(tmp_path):
    components = _build(
        tmp_path,
        {
            "a.java": "package pkg.a;\nclass ServiceA {}\n",
            "c.java": (
                "package pkg.c;\n"
                "class Client {\n"
                "  ServiceA service;\n"
                "}\n"
            ),
        },
    )
    assert components["c.Client"].uses == {"a.ServiceA"}


def test_field_type_is_uses_and_new_is_instantiates(tmp_path):
    components = _build(
        tmp_path,
        {
            "A.java": (
                "package pkg;\n"
                "class ServiceA {}\n"
                "class Client {\n"
                "  ServiceA service;\n"
                "  void run() { ServiceA other = new ServiceA(); other.toString(); }\n"
                "}\n"
            ),
        },
    )
    client = components["A.Client"]
    assert client.uses == {"A.ServiceA"}
    assert client.instantiates == {"A.ServiceA"}
    assert client.depends_on == set()
