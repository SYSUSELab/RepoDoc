from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from repodoc.document.decompose import decompose_large_components  # noqa: E402
from repodoc.document.generator import generate_documents  # noqa: E402
from repodoc.document.planner import (  # noqa: E402
    build_component_tree,
    build_doc_plan,
    page_for_component,
)
from repodoc.pipeline import build_project  # noqa: E402
from repodoc.graph.builder import PythonGraphBuilder  # noqa: E402
from repodoc.schema import validate  # noqa: E402


class DecomposeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="repodoc-decompose-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.repo = self.tmp / "repo"
        (self.repo / "pkg").mkdir(parents=True)
        (self.repo / "app.py").write_text(
            '"""App."""\n\ndef run() -> int:\n    """Run."""\n    return 1\n',
            encoding="utf-8",
        )
        (self.repo / "pkg" / "a.py").write_text(
            '"""A."""\n\ndef alpha() -> int:\n    """Alpha."""\n    return 1\n',
            encoding="utf-8",
        )
        (self.repo / "pkg" / "b.py").write_text(
            '"""B."""\n\ndef beta() -> int:\n    """Beta."""\n    return 2\n',
            encoding="utf-8",
        )
        self.graph = PythonGraphBuilder(self.repo, repo_name="repo").build()
        self.tree = build_component_tree(self.graph, repo_name="repo")

    def test_oversized_component_splits_into_pages(self) -> None:
        changed = decompose_large_components(
            self.graph,
            self.tree,
            max_nodes=2,
            max_children=2,
            min_child_nodes=1,
        )
        self.assertEqual(changed, 1)
        pkg = next(
            child for child in self.tree["root"]["children"] if child["name"] == "pkg"
        )
        self.assertTrue(pkg["children"])
        self.assertEqual(pkg["members"], [])
        child_names = {child["name"] for child in pkg["children"]}
        self.assertTrue({"a", "b"}.issubset(child_names))
        validate("component_tree", self.tree)

        doc_tree = build_doc_plan(self.tree, self.graph, repo_name="repo")
        docs = self.tmp / "docs"
        doc_tree, _pages, _sections, _doc_version = generate_documents(
            self.graph, self.tree, doc_tree, docs_dir=docs
        )
        validate("doc_tree", doc_tree)
        self.assertTrue((docs / "pkg.md").is_file())
        self.assertTrue((docs / "a.md").is_file())
        self.assertTrue((docs / "b.md").is_file())
        pkg_page = next(
            page for page in doc_tree["root"]["children"] if page["path"] == "pkg.md"
        )
        self.assertEqual(
            {child["path"] for child in pkg_page["children"]}, {"a.md", "b.md"}
        )

    def test_page_names_use_context_instead_of_numeric_suffix(self) -> None:
        used = {"overview.md"}

        def component(name: str, path: str, component_id: str) -> dict[str, object]:
            return {
                "id": component_id,
                "name": name,
                "path": path,
                "depth": 1,
                "children": [],
            }

        parent = page_for_component(
            component("example-applications", "examples", "c1"),
            source_graph_version="g",
            used_paths=used,
        )
        child = page_for_component(
            component("example-applications", "examples", "c2"),
            source_graph_version="g",
            used_paths=used,
        )
        self.assertEqual(parent["path"], "example-applications.md")
        self.assertEqual(child["path"], "examples.md")
        self.assertNotIn("-2.md", child["path"])

        first = page_for_component(
            component("utils", "src/pkg1/utils", "u1"),
            source_graph_version="g",
            used_paths=used,
        )
        second = page_for_component(
            component("utils", "src/pkg2/utils", "u2"),
            source_graph_version="g",
            used_paths=used,
        )
        self.assertEqual(first["path"], "utils.md")
        self.assertNotIn("-2.md", second["path"])
        self.assertTrue(second["path"].startswith("pkg2-utils"))

    def test_build_project_decomposes_large_component(self) -> None:
        state = self.tmp / "state"
        result = build_project(
            self.repo,
            state,
            force=True,
            max_component_nodes=2,
            max_component_children=2,
            min_component_child_nodes=1,
        )
        self.assertGreaterEqual(result.split_components, 1)
        self.assertTrue((state / "docs" / "markdown" / "pkg.md").is_file())
        self.assertTrue((state / "docs" / "markdown" / "a.md").is_file())
        self.assertTrue((state / "docs" / "markdown" / "b.md").is_file())


if __name__ == "__main__":
    unittest.main()
