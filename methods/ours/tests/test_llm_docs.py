from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from repodoc.document.llm_docs import (  # noqa: E402
    LLMDocError,
    build_page_context,
    generate_llm_pages,
)
from repodoc.graph.store import GraphStore  # noqa: E402
from repodoc.llm import ModelLimits  # noqa: E402
from repodoc.pipeline import build_project  # noqa: E402
from repodoc.schema import validate  # noqa: E402


def make_toy_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "README.md").write_text(
        "# Toy repository\n\nA tiny calculator app used by documentation tests.\n",
        encoding="utf-8",
    )
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


TEST_LIMITS = ModelLimits("fake-doc-model", 100_000, 20_000)


class FakeDocClient:
    def __init__(self):
        self.calls = []
        self.root_component_id = ""

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        user = messages[-1]["content"]
        if "PAGE: app (app.md)" in user:
            sections = [
                {
                    "heading": "Overview",
                    "level": 2,
                    "content": (
                        "LLM generated app overview.\n\n"
                        "```mermaid\n"
                        "flowchart LR\n"
                        '  A["App"] --> B["calculator"]\n'
                        "```"
                    ),
                    "covers": ["code:file:app.py", "code:run"],
                    "citations": ["app.py:1-12"],
                },
                {
                    "heading": "Public API",
                    "level": 2,
                    "content": "LLM generated app API.",
                    "covers": ["code:api:app.py::run"],
                    "citations": ["app.py:5-7"],
                },
            ]
        else:
            sections = [
                {
                    "heading": "Architecture",
                    "level": 2,
                    "content": (
                        "LLM generated repository overview.\n\n"
                        "```mermaid\n"
                        "flowchart TB\n"
                        '  A["app"] --> B["calculator"]\n'
                        "```\n\n"
                        "```mermaid\n"
                        "sequenceDiagram\n"
                        "  A->>B: run calculation\n"
                        "```"
                    ),
                    "covers": [self.root_component_id],
                    "citations": [],
                }
            ]
        return {
            "choices": [{"message": {"content": json.dumps({"sections": sections})}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 30},
        }


class LLMDocTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="repodoc-llm-docs-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.repo = self.tmp / "repo"
        self.state = self.tmp / "state"
        make_toy_repo(self.repo)
        build_project(self.repo, self.state, force=True)
        component_tree = json.loads(
            (self.state / "components" / "component_tree.json").read_text(
                encoding="utf-8"
            )
        )
        self.client = FakeDocClient()
        self.client.root_component_id = component_tree["root"]["id"]

    def test_generate_two_pages_and_write_back(self) -> None:
        before = json.loads((self.state / "manifest.json").read_text(encoding="utf-8"))
        result = generate_llm_pages(
            self.state,
            ["overview.md", "app.md"],
            client=self.client,
            max_pages=2,
            max_tokens=512,
            limits=TEST_LIMITS,
        )

        self.assertFalse(result["dry_run"])
        self.assertEqual(
            [call[1]["max_tokens"] for call in self.client.calls], [512, 512]
        )
        # level sorting must put app (L2) before overview (L3)
        self.assertIn("PAGE: app (app.md)", self.client.calls[0][0][-1]["content"])
        self.assertIn(
            "PAGE: Overview (overview.md)", self.client.calls[1][0][-1]["content"]
        )

        app_md = (self.state / "docs" / "markdown" / "app.md").read_text(
            encoding="utf-8"
        )
        overview_md = (self.state / "docs" / "markdown" / "overview.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("LLM generated app overview", app_md)
        self.assertIn("LLM generated repository overview", overview_md)

        doc_tree = json.loads(
            (self.state / "docs" / "doc_tree.json").read_text(encoding="utf-8")
        )
        validate("doc_tree", doc_tree)
        pages = {page["path"]: page for page in self._walk_pages(doc_tree["root"])}
        self.assertEqual(len(pages["app.md"]["sections"]), 2)
        self.assertEqual(len(pages["overview.md"]["sections"]), 1)
        self.assertNotIn("_content", pages["app.md"]["sections"][0])
        self.assertNotIn("content", pages["app.md"]["sections"][0])

        graph = json.loads(
            (self.state / "graph" / "graph.json").read_text(encoding="utf-8")
        )
        doc_sections = [
            node
            for node in graph["nodes"]
            if node["type"] == "doc_section"
            and node["attrs"]["page_id"] == "doc:app.md"
        ]
        self.assertEqual(len(doc_sections), 2)
        self.assertTrue(all(node["attrs"]["content"].strip() for node in doc_sections))
        self.assertTrue(all(node["attrs"]["covers"] for node in doc_sections))
        self.assertIn(
            "LLM generated app overview",
            " ".join(node["attrs"]["content"] for node in doc_sections),
        )

        after = json.loads((self.state / "manifest.json").read_text(encoding="utf-8"))
        self.assertNotEqual(before["graph_version"], after["graph_version"])
        self.assertNotEqual(before["doc_version"], after["doc_version"])
        self.assertEqual(after["last_run"]["status"], "llm-docs")
        self.assertTrue((self.state / "docs" / "llm_traces" / "app.json").is_file())
        self.assertTrue(
            (self.state / "docs" / "llm_traces" / "overview.json").is_file()
        )

    def test_dry_run_does_not_call_api(self) -> None:
        result = generate_llm_pages(
            self.state,
            ["app.md", "overview.md"],
            dry_run=True,
            max_pages=2,
            limits=TEST_LIMITS,
        )
        self.assertTrue(result["dry_run"])
        self.assertEqual(len(result["pages"]), 2)
        self.assertEqual(self.client.calls, [])

    def test_max_pages_guard(self) -> None:
        with self.assertRaises(LLMDocError):
            generate_llm_pages(
                self.state,
                ["app.md", "calculator.md", "overview.md"],
                client=self.client,
                max_pages=2,
            )
        self.assertEqual(self.client.calls, [])

    def test_unknown_page(self) -> None:
        with self.assertRaises(LLMDocError):
            generate_llm_pages(self.state, ["does-not-exist.md"], dry_run=True)

    def test_high_level_context_has_relationship_evidence(self) -> None:
        doc_tree = json.loads(
            (self.state / "docs" / "doc_tree.json").read_text(encoding="utf-8")
        )
        pages = {page["path"]: page for page in self._walk_pages(doc_tree["root"])}
        component_tree = json.loads(
            (self.state / "components" / "component_tree.json").read_text(
                encoding="utf-8"
            )
        )
        graph = GraphStore.load(self.state)
        context = build_page_context(
            graph,
            component_tree,
            doc_tree,
            pages["overview.md"],
            docs_dir=self.state / "docs" / "markdown",
            max_source_chars=30_000,
        )
        self.assertIn("TOP-LEVEL COMPONENT RELATIONSHIPS", context)
        self.assertIn("KEY ENTRY POINTS", context)
        self.assertIn("PROJECT README", context)

    @staticmethod
    def _walk_pages(page):
        yield page
        for child in page.get("children", []):
            yield from LLMDocTests._walk_pages(child)


if __name__ == "__main__":
    unittest.main()
