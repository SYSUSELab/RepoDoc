from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from repodoc.document.llm_cluster import LLMClusterAgent, LLMClusterError  # noqa: E402
from repodoc.graph.builder import PythonGraphBuilder  # noqa: E402
from repodoc.llm import ModelLimits  # noqa: E402
from repodoc.pipeline import build_project  # noqa: E402
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


def members_for(graph, path: str) -> list[str]:
    return sorted(
        node["id"]
        for node in graph.nodes.values()
        if node["layer"] == "code" and node["attrs"].get("path") == path
    )


class FakeClient:
    config = SimpleNamespace(model="fake-cluster-model", session_id="fake-session")

    def __init__(self, content: str):
        self.content = content
        self.calls: list[tuple[list[dict], dict]] = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {
            "choices": [{"message": {"content": self.content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }


class LLMClusterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="repodoc-llm-cluster-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.repo = self.tmp / "repo"
        make_toy_repo(self.repo)
        self.graph = PythonGraphBuilder(self.repo, repo_name="repo").build()

    def response_json(self) -> str:
        data = {
            "components": [
                {
                    "name": "calculator",
                    "path": "calculator.py",
                    "summary": "Arithmetic helpers.",
                    "members": members_for(self.graph, "calculator.py"),
                    "children": [],
                },
                {
                    "name": "app",
                    "path": "app.py",
                    "summary": "Application entry points.",
                    "members": members_for(self.graph, "app.py"),
                    "children": [],
                },
            ]
        }
        return json.dumps(data)

    def test_cluster_and_write_components(self) -> None:
        client = FakeClient(self.response_json())
        agent = LLMClusterAgent(client, max_nodes=50, max_tokens=512, temperature=0.2)
        tree, trace = agent.cluster(self.graph)

        validate("component_tree", tree)
        self.assertEqual(tree["cluster_meta"]["mode"], "llm")
        self.assertEqual(tree["cluster_meta"]["model"], "fake-cluster-model")
        self.assertEqual(tree["cluster_meta"]["usage"]["prompt_tokens"], 100)
        self.assertEqual(tree["root"]["coverage"]["unassigned"], [])
        self.assertEqual(len(tree["root"]["children"]), 2)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][1]["max_tokens"], 512)
        self.assertEqual(client.calls[0][1]["temperature"], 0.2)
        self.assertIn("FILE NODES", client.calls[0][0][1]["content"])
        self.assertEqual(trace["content"], self.response_json())

        component_ids = {
            node["id"]
            for node in self.graph.nodes.values()
            if node["type"] == "component"
        }
        self.assertEqual(len(component_ids), 3)  # repository + 2 components

    def test_file_only_members_are_expanded(self) -> None:
        response = json.dumps(
            {
                "components": [
                    {
                        "name": "calculator",
                        "path": "calculator.py",
                        "members": ["code:file:calculator.py"],
                    },
                    {"name": "app", "path": "app.py", "members": ["code:file:app.py"]},
                ]
            }
        )
        tree, _trace = LLMClusterAgent(FakeClient(response), max_nodes=50).cluster(
            self.graph
        )
        validate("component_tree", tree)
        self.assertEqual(tree["root"]["coverage"]["unassigned"], [])
        children = {child["name"]: child for child in tree["root"]["children"]}
        self.assertEqual(len(children["calculator"]["members"]), 5)
        self.assertEqual(len(children["app"]["members"]), 6)

    def test_hierarchical_cluster_bounds_prompts_and_merges_chunks(self) -> None:
        class DynamicClient:
            config = SimpleNamespace(
                model="fake-cluster-model", session_id="fake-session"
            )

            def __init__(self):
                self.calls = []

            def chat(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                user = messages[-1]["content"]
                files = re.findall(r"\[file\] (\S+)", user)
                components = [
                    {
                        "name": Path(file_id.split(":file:")[-1]).stem,
                        "path": file_id.split(":file:")[-1],
                        "members": [file_id],
                    }
                    for file_id in files
                ]
                return {
                    "choices": [
                        {"message": {"content": json.dumps({"components": components})}}
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                }

        client = DynamicClient()
        agent = LLMClusterAgent(client, max_nodes=50, max_tokens=512, max_chars=50_000)
        tree, trace = agent.cluster_hierarchical(
            self.graph, max_chunk_files=1, max_chunk_chars=50_000
        )
        validate("component_tree", tree)
        self.assertEqual(tree["cluster_meta"]["mode"], "llm-hierarchical")
        self.assertEqual(tree["cluster_meta"]["chunk_count"], 2)
        self.assertEqual(tree["cluster_meta"]["usage"]["prompt_tokens"], 200)
        self.assertEqual(tree["cluster_meta"]["usage"]["completion_tokens"], 100)
        self.assertEqual(tree["root"]["coverage"]["unassigned"], [])
        self.assertEqual(len(tree["root"]["children"]), 2)
        self.assertEqual(len(client.calls), 2)

    def test_hierarchical_llm_merge(self) -> None:
        class MergeClient:
            config = SimpleNamespace(
                model="fake-cluster-model", session_id="fake-session"
            )

            def __init__(self):
                self.calls = []

            def chat(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                system = messages[0]["content"]
                user = messages[-1]["content"]
                if "software architecture merge agent" in system:
                    components = [
                        {
                            "name": "merged",
                            "path": ".",
                            "summary": "Merged component.",
                            "children": [0, 1],
                        }
                    ]
                else:
                    files = re.findall(r"\[file\] (\S+)", user)
                    components = [
                        {
                            "name": Path(file_id.split(":file:")[-1]).stem,
                            "path": file_id.split(":file:")[-1],
                            "members": [file_id],
                        }
                        for file_id in files
                    ]
                return {
                    "choices": [
                        {"message": {"content": json.dumps({"components": components})}}
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                }

        client = MergeClient()
        agent = LLMClusterAgent(client, max_nodes=50, max_tokens=512, max_chars=50_000)
        tree, trace = agent.cluster_hierarchical(
            self.graph,
            max_chunk_files=1,
            max_chunk_chars=50_000,
            merge_mode="llm",
        )
        validate("component_tree", tree)
        self.assertEqual(tree["cluster_meta"]["merge_mode"], "llm")
        self.assertFalse(tree["cluster_meta"]["merge_fallback"])
        self.assertEqual(len(tree["root"]["children"]), 1)
        self.assertEqual(tree["root"]["coverage"]["unassigned"], [])
        self.assertEqual(len(client.calls), 3)
        self.assertIsNotNone(trace["merge"])

    def test_node_limit_prevents_call(self) -> None:
        client = FakeClient(self.response_json())
        agent = LLMClusterAgent(client, max_nodes=1)
        with self.assertRaises(LLMClusterError):
            agent.cluster(self.graph)
        self.assertEqual(client.calls, [])

    def test_invalid_json_raises(self) -> None:
        agent = LLMClusterAgent(FakeClient("not json"))
        with self.assertRaises(LLMClusterError):
            agent.cluster(self.graph)

    def test_build_project_auto_cluster_for_small_repo(self) -> None:
        client = FakeClient(self.response_json())
        state = self.tmp / "state-auto"
        result = build_project(
            self.repo,
            state,
            force=True,
            cluster_mode="auto",
            llm_client=client,
            max_cluster_nodes=50,
            llm_max_tokens=512,
            model_limits=ModelLimits("fake-cluster-model", 100_000, 20_000),
        )
        self.assertEqual(result.cluster_mode, "llm")
        self.assertEqual(result.pages, 3)

    def test_build_project_with_llm_cluster(self) -> None:
        client = FakeClient(self.response_json())
        state = self.tmp / "state"
        result = build_project(
            self.repo,
            state,
            force=True,
            cluster_mode="llm",
            llm_client=client,
            max_cluster_nodes=50,
            llm_max_tokens=512,
            model_limits=ModelLimits("fake-cluster-model", 100_000, 20_000),
        )
        self.assertEqual(result.cluster_mode, "llm")
        self.assertEqual(result.pages, 3)
        self.assertTrue((state / "components" / "llm_cluster_trace.json").is_file())
        tree = json.loads(
            (state / "components" / "component_tree.json").read_text(encoding="utf-8")
        )
        validate("component_tree", tree)
        self.assertEqual(tree["cluster_meta"]["mode"], "llm")
        self.assertIsNotNone(tree["cluster_meta"]["budget"])
        self.assertIsNotNone(result.llm_budget)
        self.assertTrue((state / "docs" / "markdown" / "calculator.md").is_file())


if __name__ == "__main__":
    unittest.main()
