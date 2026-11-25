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

from repodoc.llm import ModelLimits  # noqa: E402
from repodoc.pipeline import apply_update, build_project  # noqa: E402
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
        "    return multiply(add(1, 2), 3)\n",
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


def commit(repo: Path, message: str) -> str:
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        message,
    )
    return git(repo, "rev-parse", "HEAD")


class UpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="repodoc-update-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.repo = self.tmp / "repo"
        self.state = self.tmp / "state"
        make_toy_repo(self.repo)
        git(self.repo, "init", "-q", "-b", "main")
        self.base_commit = commit(self.repo, "base")
        build_project(self.repo, self.state, force=True)

    def test_apply_update_changes_graph_and_affected_docs(self) -> None:
        old_manifest = json.loads(
            (self.state / "manifest.json").read_text(encoding="utf-8")
        )
        old_calc = (self.state / "docs" / "markdown" / "calculator.md").read_text(
            encoding="utf-8"
        )

        source = self.repo / "calculator.py"
        source.write_text(
            source.read_text(encoding="utf-8").replace(
                "return a + b", "return a + b  # clamped later\n"
            ),
            encoding="utf-8",
        )
        target_commit = commit(self.repo, "change add")

        result = apply_update(self.repo, self.state)
        self.assertEqual(result["status"], "applied")
        self.assertIn("calculator.md", result["regenerated_pages"])
        self.assertIn("overview.md", result["regenerated_pages"])

        change_set = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
        validate("change_set", change_set)
        self.assertEqual(change_set["status"], "applied")
        self.assertEqual(change_set["target_commit"], target_commit)
        self.assertIn(
            "code:function:calculator.py::add",
            change_set["graph_delta"]["nodes_modified"],
        )
        self.assertIn("calculator.md", change_set["impact"]["affected_pages"])
        self.assertFalse(change_set["impact"]["requires_recluster"])

        manifest = json.loads(
            (self.state / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["last_commit"], target_commit)
        self.assertNotEqual(manifest["graph_version"], old_manifest["graph_version"])
        self.assertNotEqual(manifest["doc_version"], old_manifest["doc_version"])
        self.assertEqual(manifest["last_run"]["status"], "updated")

        new_calc = (self.state / "docs" / "markdown" / "calculator.md").read_text(
            encoding="utf-8"
        )
        self.assertNotEqual(old_calc, new_calc)
        overview = (self.state / "docs" / "markdown" / "overview.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(target_commit, overview)
        graph = json.loads(
            (self.state / "graph" / "graph.json").read_text(encoding="utf-8")
        )
        cites = [edge for edge in graph["edges"] if edge["type"] == "cites"]
        self.assertTrue(cites)
        add_node = next(
            node
            for node in graph["nodes"]
            if node["id"] == "code:function:calculator.py::add"
        )
        self.assertIn("clamped later", add_node["attrs"]["source"])

        component_tree = json.loads(
            (self.state / "components" / "component_tree.json").read_text(
                encoding="utf-8"
            )
        )
        validate("component_tree", component_tree)
        self.assertTrue(
            all(child["members"] for child in component_tree["root"]["children"])
        )
        doc_tree = json.loads(
            (self.state / "docs" / "doc_tree.json").read_text(encoding="utf-8")
        )
        validate("doc_tree", doc_tree)

    def test_direct_only_updates_only_cited_page(self) -> None:
        source = self.repo / "calculator.py"
        source.write_text(
            source.read_text(encoding="utf-8").replace(
                "return a + b", "return a + b  # direct-only\n"
            ),
            encoding="utf-8",
        )
        commit(self.repo, "change add directly")

        result = apply_update(self.repo, self.state, direct_only=True)
        self.assertEqual(result["status"], "applied")
        self.assertIn("calculator.md", result["regenerated_pages"])
        self.assertNotIn("overview.md", result["regenerated_pages"])
        change_set = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
        self.assertEqual(
            set(change_set["impact"]["affected_pages"]),
            {"app.md", "calculator.md"},
        )

    def test_new_file_creates_component_and_page(self) -> None:
        new_dir = self.repo / "plugins"
        new_dir.mkdir()
        (new_dir / "plugin.py").write_text(
            '"""New plugin."""\n\ndef activate() -> str:\n    """Activate the plugin."""\n    return "ok"\n',
            encoding="utf-8",
        )
        commit(self.repo, "add plugin")

        result = apply_update(self.repo, self.state, direct_only=True)
        self.assertEqual(result["status"], "applied")
        self.assertTrue(any("plugins" in name for name in result["new_components"]))
        self.assertTrue((self.state / "docs" / "markdown" / "plugins.md").is_file())
        change_set = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
        self.assertTrue(change_set["impact"]["new_pages"])
        self.assertIn("overview.md", result["regenerated_pages"])

    def test_new_file_in_existing_directory_affects_component_page(self) -> None:
        (self.repo / "extra_calculator.py").write_text(
            '"""Extra calculator helpers."""\n\ndef subtract(a, b):\n    return a - b\n',
            encoding="utf-8",
        )
        commit(self.repo, "add extra calculator")

        result = apply_update(self.repo, self.state, direct_only=True)
        self.assertEqual(result["status"], "applied")
        self.assertIn("overview.md", result["regenerated_pages"])
        self.assertTrue({"app.md", "calculator.md"} & set(result["regenerated_pages"]))

    def test_deleted_file_removes_component_and_page(self) -> None:
        (self.repo / "calculator.py").unlink()
        commit(self.repo, "delete calculator")

        result = apply_update(self.repo, self.state)
        self.assertEqual(result["status"], "applied")
        self.assertIn("calculator.md", result["deleted_pages"])
        self.assertFalse((self.state / "docs" / "markdown" / "calculator.md").exists())
        component_tree = json.loads(
            (self.state / "components" / "component_tree.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn(
            "calculator",
            [child["name"] for child in component_tree["root"]["children"]],
        )
        change_set = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
        self.assertIn("calculator.md", change_set["impact"]["deleted_pages"])

    def test_llm_triage_can_skip_a_page(self) -> None:
        source = self.repo / "calculator.py"
        source.write_text(
            source.read_text(encoding="utf-8").replace(
                "return a + b", "return a + b  # changed\n"
            ),
            encoding="utf-8",
        )
        commit(self.repo, "change add")

        class FakeTriageClient:
            def __init__(self):
                self.calls = []

            def chat(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                content = json.dumps(
                    {
                        "decisions": [
                            {
                                "page": "calculator.md",
                                "action": "noop",
                                "reason": "test skip",
                            },
                            {
                                "page": "overview.md",
                                "action": "regenerate",
                                "reason": "child summary changes",
                            },
                        ]
                    }
                )
                return {
                    "choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }

        client = FakeTriageClient()
        result = apply_update(
            self.repo,
            self.state,
            use_llm=True,
            llm_client=client,
            model_limits=ModelLimits("fake", 100_000, 20_000),
        )
        self.assertEqual(result["status"], "applied")
        self.assertEqual(len(client.calls), 1)
        self.assertNotIn("calculator.md", result["regenerated_pages"])
        self.assertIn("overview.md", result["regenerated_pages"])
        self.assertEqual(result["llm_decisions"]["calculator.md"], "noop")
        self.assertTrue(
            (
                self.state
                / "changes"
                / f"{git(self.repo, 'rev-parse', 'HEAD')}.llm_triage.json"
            ).is_file()
        )

    def test_llm_planner_structural_change_keeps_required_pages(self) -> None:
        new_dir = self.repo / "plugins"
        new_dir.mkdir()
        (new_dir / "plugin.py").write_text(
            '"""New plugin."""\n\ndef activate() -> str:\n    """Activate."""\n    return "ok"\n',
            encoding="utf-8",
        )
        commit(self.repo, "add plugin with planner")

        class FakePlannerClient:
            def chat(self, messages, **kwargs):
                content = json.dumps(
                    {
                        "change_scope": "architecture",
                        "decisions": [
                            {
                                "page": "overview.md",
                                "action": "regenerate",
                                "reason": "new component is architecturally visible",
                            }
                        ],
                    }
                )
                return {
                    "choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }

        result = apply_update(
            self.repo,
            self.state,
            use_planner=True,
            llm_client=FakePlannerClient(),
            model_limits=ModelLimits("fake", 100_000, 20_000),
        )
        self.assertEqual(result["status"], "applied")
        self.assertIn("overview.md", result["regenerated_pages"])
        self.assertTrue((self.state / "docs" / "markdown" / "plugins.md").is_file())
        change_set = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
        self.assertTrue(change_set["impact"]["structural"])
        self.assertTrue(change_set["impact"]["architecture_signal"])

    def test_llm_planner_guard_blocks_local_escalation(self) -> None:
        source = self.repo / "calculator.py"
        source.write_text(
            source.read_text(encoding="utf-8").replace(
                "return a + b", "return a + b  # planner escalation\n"
            ),
            encoding="utf-8",
        )
        commit(self.repo, "change add with escalation")

        class FakePlannerClient:
            def __init__(self):
                self.calls = []

            def chat(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                content = json.dumps(
                    {
                        "change_scope": "architecture",
                        "decisions": [
                            {
                                "page": "calculator.md",
                                "action": "regenerate",
                                "reason": "changed add behavior",
                            },
                            {
                                "page": "overview.md",
                                "action": "regenerate",
                                "reason": "architecture-level flow changed",
                            },
                        ],
                    }
                )
                return {
                    "choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }

        client = FakePlannerClient()
        result = apply_update(
            self.repo,
            self.state,
            use_planner=True,
            llm_client=client,
            model_limits=ModelLimits("fake", 100_000, 20_000),
        )
        self.assertEqual(result["status"], "applied")
        self.assertEqual(len(client.calls), 1)
        self.assertIn("calculator.md", result["regenerated_pages"])
        self.assertNotIn("overview.md", result["regenerated_pages"])
        self.assertIn("llm_plan", Path(result["path"]).read_text(encoding="utf-8"))
        trace = (
            self.state
            / "changes"
            / f"{git(self.repo, 'rev-parse', 'HEAD')}.llm_plan.json"
        )
        self.assertTrue(trace.is_file())
        self.assertEqual(result["llm_decisions"]["overview.md"], "noop")
        change_set = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
        self.assertFalse(change_set["impact"]["llm_plan"]["architecture_signal"])

    def test_llm_content_patch_updates_one_section(self) -> None:
        sections_path = self.state / "docs" / "sections.jsonl"
        section_rows = [
            json.loads(line)
            for line in sections_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        target = next(
            row
            for row in section_rows
            if row["page_path"] == "calculator.md"
            and row["heading"] == "Implementation"
        )

        source = self.repo / "calculator.py"
        source.write_text(
            source.read_text(encoding="utf-8").replace(
                "return a + b", "return a + b  # changed\n"
            ),
            encoding="utf-8",
        )
        commit(self.repo, "change add")

        class FakePatchClient:
            def __init__(self):
                self.calls = []

            def chat(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                system = messages[0]["content"]
                if "patch editor" in system:
                    content = json.dumps(
                        {
                            "patches": [
                                {
                                    "section_id": target["id"],
                                    "old_content_hash": target["content_hash"],
                                    "new_content": "Patched implementation section.",
                                    "reason": "changed add implementation",
                                }
                            ]
                        }
                    )
                else:
                    content = json.dumps(
                        {
                            "decisions": [
                                {
                                    "page": "calculator.md",
                                    "action": "patch",
                                    "reason": "small section change",
                                },
                                {
                                    "page": "overview.md",
                                    "action": "regenerate",
                                    "reason": "child summary changes",
                                },
                            ]
                        }
                    )
                return {
                    "choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }

        client = FakePatchClient()
        result = apply_update(
            self.repo,
            self.state,
            use_llm=True,
            use_patch=True,
            llm_client=client,
            model_limits=ModelLimits("fake", 100_000, 20_000),
        )
        self.assertEqual(result["status"], "applied")
        self.assertEqual(len(client.calls), 2)  # triage + one patch
        self.assertIn(
            "calculator.md", result["patched_pages"], result.get("patch_fallbacks")
        )
        self.assertIn("overview.md", result["regenerated_pages"])
        self.assertNotIn("calculator.md", result["regenerated_pages"])
        markdown = (self.state / "docs" / "markdown" / "calculator.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Patched implementation section.", markdown)
        graph = json.loads(
            (self.state / "graph" / "graph.json").read_text(encoding="utf-8")
        )
        section_node = next(
            node for node in graph["nodes"] if node["id"] == target["id"]
        )
        self.assertIn(
            "Patched implementation section.", section_node["attrs"]["content"]
        )

    def test_noop_when_no_changes(self) -> None:
        result = apply_update(self.repo, self.state)
        self.assertEqual(result["status"], "noop")


if __name__ == "__main__":
    unittest.main()
