from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from repodoc.llm import ledger  # noqa: E402
from repodoc.usage import collect_usage, write_usage_metadata  # noqa: E402


def _usage(prompt: int, completion: int, cost: float | None = None) -> dict:
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "completion_tokens_details": {"reasoning_tokens": 1},
    }
    if cost is not None:
        usage["opencode"] = {"list_cost": cost, "tokens": {"reasoning": 1}}
    return usage


class UsageTests(unittest.TestCase):
    def setUp(self) -> None:
        ledger.reset()
        self.tmp = Path(tempfile.mkdtemp(prefix="repodoc-usage-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.state = self.tmp / "state"
        (self.state / "components").mkdir(parents=True)
        (self.state / "docs" / "markdown").mkdir(parents=True)
        (self.state / "docs" / "llm_traces" / "failed").mkdir(parents=True)
        (self.state / "changes").mkdir()
        (self.state / "manifest.json").write_text(
            json.dumps(
                {
                    "repo_path": "/tmp/repo",
                    "last_commit": "abc123",
                }
            ),
            encoding="utf-8",
        )
        (self.state / "components" / "component_tree.json").write_text(
            json.dumps(
                {
                    "root": {
                        "name": "root",
                        "children": [
                            {"name": "a", "children": []},
                            {"name": "b", "children": []},
                        ],
                    },
                    "cluster_meta": {
                        "mode": "llm-hierarchical",
                        "model": "fake-model",
                        "chunk_count": 2,
                        "merge_mode": "llm",
                        "merge_fallback": False,
                        "llm_seconds": 1.5,
                        "usage": _usage(100, 50, 0.01),
                    },
                }
            ),
            encoding="utf-8",
        )
        (self.state / "docs" / "doc_tree.json").write_text(
            json.dumps({"root": {"path": "overview.md", "children": []}}),
            encoding="utf-8",
        )
        (self.state / "docs" / "markdown" / "overview.md").write_text(
            "# Overview\n", encoding="utf-8"
        )
        self._trace("overview.json", _usage(10, 5, 0.001))
        self._trace("failed/part.123.json", _usage(5, 0, None), failed=True)
        (self.state / "docs" / "llm_traces" / "failed" / "part.124.json").write_text(
            json.dumps({"page": "missing.json", "failed": True}), encoding="utf-8"
        )
        (self.state / "changes" / "abc.llm_plan.json").write_text(
            json.dumps(
                {
                    "prompt": "p",
                    "response": {
                        "model": "fake-model",
                        "usage": _usage(7, 3, 0.0005),
                    },
                }
            ),
            encoding="utf-8",
        )

    def _trace(self, name: str, usage: dict, *, failed: bool = False) -> None:
        (self.state / "docs" / "llm_traces" / name).write_text(
            json.dumps(
                {
                    "page": f"{Path(name).stem}.md",
                    "kind": "component",
                    "failed": failed,
                    "elapsed_sec": 0.5,
                    "response": {"model": "fake-model", "usage": usage},
                }
            ),
            encoding="utf-8",
        )

    def test_collect_usage_counts_failed_traces_and_phases(self) -> None:
        metadata = collect_usage(self.state)
        usage = metadata["usage"]
        self.assertEqual(usage["n_calls"], 7)
        self.assertEqual(usage["totals"]["prompt_tokens"], 122)
        self.assertEqual(usage["totals"]["completion_tokens"], 58)
        self.assertEqual(usage["totals"]["total_tokens"], 180)
        self.assertEqual(usage["totals"]["cost_usd"], 0.0115)
        self.assertEqual(usage["cost_status"], "partial")
        self.assertEqual(usage["unknown_usage_calls"], 1)
        self.assertEqual(set(usage["by_phase"]), {"clustering", "docs", "planning"})
        self.assertEqual(metadata["statistics"]["pages"], 1)
        self.assertEqual(metadata["generation_info"]["main_model"], "fake-model")

    def test_write_metadata(self) -> None:
        out = self.tmp / "metadata.json"
        write_usage_metadata(self.state, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        self.assertTrue(data["files_generated"])
        self.assertTrue(data["usage"]["calls"])
        self.assertEqual(data["generation_info"]["commit_id"], "abc123")

    def test_current_process_run_is_recorded_separately(self) -> None:
        ledger.reset()
        ledger.record_response(
            {"usage": _usage(30, 10, 0.003)}, seconds=1.25, model="fake-model"
        )
        out = self.tmp / "metadata.json"
        write_usage_metadata(self.state, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        last_run = data["usage"]["last_run"]
        self.assertEqual(last_run["totals"]["total_tokens"], 40)
        self.assertEqual(last_run["n_calls"], 1)
        self.assertEqual(last_run["cost_status"], "complete")
        self.assertGreaterEqual(last_run["elapsed_sec"], 0)
        ledger_file = self.state / "usage" / "runs" / f"{last_run['run_id']}.json"
        self.assertTrue(ledger_file.is_file())
        latest = collect_usage(self.state, scope="latest-run")
        self.assertEqual(latest["usage"]["run_id"], last_run["run_id"])
        self.assertEqual(latest["usage"]["totals"]["total_tokens"], 40)

    def test_failed_call_without_usage_is_unknown(self) -> None:
        ledger.reset()
        ledger.record_error(seconds=0.5, model="fake-model", error="TimeoutError")
        metadata = collect_usage(self.state)
        self.assertEqual(metadata["usage"]["last_run"]["unknown_usage_calls"], 1)
        self.assertEqual(metadata["usage"]["last_run"]["cost_status"], "unknown")


if __name__ == "__main__":
    unittest.main()
