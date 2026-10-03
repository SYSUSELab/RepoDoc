"""Analyzer registry / file selection alignment with pyproject capabilities."""

from repodoc.config import Config
from repodoc.graph import DependencyGraphBuilder
from repodoc.graph.builder import _should_exclude_path


def _analyze(tmp_path, files: dict[str, str]):
    repo = tmp_path / "repo"
    repo.mkdir()
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    config = Config(repo_path=str(repo), output_dir=str(tmp_path / "out"))
    components, _ = DependencyGraphBuilder(config).build_dependency_graph()
    return components


def test_tsx_parses_with_tsx_grammar(tmp_path):
    components = _analyze(
        tmp_path,
        {
            "src/App.tsx": (
                "export function App() {\n"
                "  return <div>Hello</div>;\n"
                "}\n"
                "export const Button = () => <button>Click</button>;\n"
            )
        },
    )
    assert "src.App.App" in components
    assert "src.App.Button" in components


def test_previously_blocked_extensions_are_collected(tmp_path):
    components = _analyze(
        tmp_path,
        {
            "src/legacy.mjs": "export function legacy() { return 1; }\n",
            "src/thing.cc": "class Thing {\npublic:\n    void run();\n};\n",
            "src/old.cjs": "module.exports = function old() {};\n",
        },
    )
    assert "src.legacy.legacy" in components
    assert "src.thing.Thing" in components


def test_directory_exclude_matches_segment_not_prefix():
    patterns = {"bin/", "out"}
    assert _should_exclude_path("bin/x.py", "x.py", patterns)
    assert not _should_exclude_path("bindings/x.py", "x.py", patterns)
    assert not _should_exclude_path("binary/x.py", "x.py", patterns)
    assert _should_exclude_path("out/x.py", "x.py", patterns)


def test_hxx_strips_extension_from_module_id(tmp_path):
    components = _analyze(
        tmp_path,
        {"src/util.hxx": "class Util {\npublic:\n    void run();\n};\n"},
    )
    assert "src.util.Util" in components
