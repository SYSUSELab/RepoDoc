from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from repodoc.pipeline import build_project, plan_update, status_info  # noqa: E402
from repodoc.schema import validate  # noqa: E402


def make_toy_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "calculator.py").write_text(
        '"""Tiny calculator."""\n\n'
        "def add(a: int, b: int) -> int:\n"
        '    """Add two numbers."""\n'
        "    return a + b\n\n"
        "def multiply(a: int, b: int) -> int:\n"
        '    """Multiply two numbers."""\n'
        "    return a * b\n",
        encoding="utf-8",
    )
    (repo / "app.py").write_text(
        '"""Tiny app."""\n\n'
        "from calculator import add, multiply\n\n"
        "def run() -> int:\n"
        '    """Run one calculation."""\n'
        "    return multiply(add(1, 2), 3)\n\n"
        "class App:\n"
        "    def compute(self):\n"
        "        # use the module function\n"
        "        return run()\n",
        encoding="utf-8",
    )


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def init_git(repo: Path) -> str:
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "init",
    )
    return git(repo, "rev-parse", "HEAD")


class TempDirTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="repodoc-ours-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)


class BuildTests(TempDirTest):
    def test_build_graph_components_docs(self) -> None:
        repo = self.tmp / "repo"
        state = self.tmp / "state"
        make_toy_repo(repo)
        result = build_project(repo, state, force=True)

        self.assertEqual(result.pages, 3)
        self.assertTrue((state / "manifest.json").is_file())
        self.assertTrue((state / "graph" / "nodes.jsonl").is_file())
        self.assertTrue((state / "graph" / "edges.jsonl").is_file())
        self.assertTrue((state / "components" / "component_tree.json").is_file())
        self.assertTrue((state / "docs" / "doc_tree.json").is_file())
        self.assertTrue((state / "docs" / "markdown" / "overview.md").is_file())

        graph = json.loads((state / "graph" / "graph.json").read_text(encoding="utf-8"))
        component_tree = json.loads(
            (state / "components" / "component_tree.json").read_text(encoding="utf-8")
        )
        doc_tree = json.loads(
            (state / "docs" / "doc_tree.json").read_text(encoding="utf-8")
        )
        manifest = json.loads((state / "manifest.json").read_text(encoding="utf-8"))
        validate("graph", graph)
        validate("component_tree", component_tree)
        validate("doc_tree", doc_tree)
        validate("manifest", manifest)

        nodes = {node["id"]: node for node in graph["nodes"]}
        self.assertIn("code:file:app.py", nodes)
        self.assertIn("code:function:calculator.py::add", nodes)
        self.assertIn("doc:overview.md", nodes)
        self.assertIn("doc_section", {node["type"] for node in nodes.values()})
        section_nodes = [
            node for node in nodes.values() if node["type"] == "doc_section"
        ]
        self.assertTrue(section_nodes)
        self.assertTrue(all(node["attrs"]["content"].strip() for node in section_nodes))

        edges = {(edge["type"], edge["src"], edge["dst"]) for edge in graph["edges"]}
        self.assertIn(
            ("calls", "code:function:app.py::run", "code:function:calculator.py::add"),
            edges,
        )
        self.assertIn(
            (
                "calls",
                "code:function:app.py::run",
                "code:function:calculator.py::multiply",
            ),
            edges,
        )
        self.assertIn(("imports", "code:file:app.py", "code:file:calculator.py"), edges)
        self.assertIn(
            ("documents", "doc:overview.md", component_tree["root"]["id"]), edges
        )

        overview = (state / "docs" / "markdown" / "overview.md").read_text(
            encoding="utf-8"
        )
        app_page = (state / "docs" / "markdown" / "app.md").read_text(encoding="utf-8")
        self.assertIn("## Summary", overview)
        self.assertIn("[app](app.md)", overview)
        self.assertIn("App.compute", app_page)
        self.assertIn("def compute(self)", app_page)
        self.assertNotIn("# use the module function\n```", app_page)

        sections = [
            json.loads(line)
            for line in (state / "docs" / "sections.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertTrue(sections)
        self.assertTrue(all(section["content"].strip() for section in sections))
        self.assertTrue(
            all(section["id"].startswith("section:") for section in sections)
        )

        # Rebuilding identical inputs must produce the same content versions.
        result2 = build_project(repo, self.tmp / "state2", force=True)
        self.assertEqual(result.graph_version, result2.graph_version)
        self.assertEqual(result.component_version, result2.component_version)
        self.assertEqual(result.doc_version, result2.doc_version)

    def test_status_and_noop_update(self) -> None:
        repo = self.tmp / "repo"
        state = self.tmp / "state"
        make_toy_repo(repo)
        build_project(repo, state, force=True)

        info = status_info(repo, state, exclude_dirs=[state])
        self.assertFalse(info["needs_update"])
        self.assertEqual(plan_update(repo, state, write=False)["status"], "noop")

        (repo / "calculator.py").write_text(
            (repo / "calculator.py").read_text(encoding="utf-8") + "\n# changed\n",
            encoding="utf-8",
        )
        info = status_info(repo, state, exclude_dirs=[state])
        self.assertTrue(info["needs_update"])


class UpdateSkeletonTests(TempDirTest):
    def test_git_change_set_is_written(self) -> None:
        repo = self.tmp / "repo"
        state = self.tmp / "state"
        make_toy_repo(repo)
        init_git(repo)
        build_project(repo, state, force=True)

        (repo / "calculator.py").write_text(
            (repo / "calculator.py")
            .read_text(encoding="utf-8")
            .replace("return a + b", "return a + b  # changed"),
            encoding="utf-8",
        )
        git(repo, "add", ".")
        git(
            repo,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "change",
        )

        result = plan_update(repo, state, write=True)
        self.assertEqual(result["status"], "planned")
        change_set = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
        validate("change_set", change_set)
        self.assertEqual(change_set["files"][0]["path"], "calculator.py")
        self.assertEqual(change_set["files"][0]["status"], "modified")
        self.assertFalse(change_set["impact"]["requires_recluster"])


if __name__ == "__main__":
    unittest.main()
