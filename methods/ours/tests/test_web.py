from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - optional web extra
    TestClient = None

from repodoc.pipeline import build_project  # noqa: E402
from repodoc.qa.index import build_index  # noqa: E402
from repodoc.qa.service import QAService  # noqa: E402
from repodoc.web.server import (  # noqa: E402
    _github_slug,
    _local_repo_options,
    create_app,
)


@unittest.skipIf(TestClient is None, "install methods/ours[web] to run web tests")
class WebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="repodoc-web-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        (self.repo / "app.py").write_text(
            '"""Toy app."""\n\ndef run():\n    """Run it."""\n    return 1\n',
            encoding="utf-8",
        )
        self.state = self.tmp / "runs" / "run-1" / "state"
        build_project(self.repo, self.state, cluster_mode="toy")
        metadata = {
            "generation_info": {"main_model": "fake-model", "language": "python"},
            "usage": {
                "cost_status": "complete",
                "n_calls": 2,
                "totals": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "cost_usd": 0.01,
                },
            },
        }
        (self.state / "docs" / "markdown" / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        from repodoc.web.server import create_app

        self.client = TestClient(
            create_app(runs_dir=self.tmp / "runs", repos_dir=self.tmp / "repos")
        )

    def test_list_detail_and_page(self) -> None:
        projects = self.client.get("/api/projects")
        self.assertEqual(projects.status_code, 200)
        rows = projects.json()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["repo"], "repo")
        self.assertEqual(rows[0]["usage"]["tokens"]["total"], 15)

        project_id = rows[0]["id"]
        detail = self.client.get(f"/api/projects/{project_id}")
        self.assertEqual(detail.status_code, 200)
        body = detail.json()
        self.assertEqual(body["project"]["pages"], 2)
        self.assertTrue(body["doc_tree"]["children"])
        page_path = body["pages"][0]["path"]
        page = self.client.get(
            f"/api/projects/{project_id}/page", params={"path": page_path}
        )
        self.assertEqual(page.status_code, 200)
        self.assertIn("#", page.text)

        deleted = self.client.delete(f"/api/projects/{project_id}")
        self.assertEqual(deleted.status_code, 200)
        self.assertFalse(self.state.exists())
        self.assertEqual(self.client.get("/api/projects").json(), [])

    def test_page_path_traversal_is_rejected(self) -> None:
        project_id = self.client.get("/api/projects").json()[0]["id"]
        response = self.client.get(
            f"/api/projects/{project_id}/page", params={"path": "../manifest.json"}
        )
        self.assertEqual(response.status_code, 400)

    def test_index_and_source_endpoint(self) -> None:
        index = build_index(self.state)
        self.assertTrue(index.search("run"))
        project_id = self.client.get("/api/projects").json()[0]["id"]
        response = self.client.get(
            f"/api/projects/{project_id}/source",
            params={"path": "app.py", "start": 1, "end": 5},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Toy app", response.json()["content"])

    def test_ask_stream_endpoint_emits_sse_events(self) -> None:
        project_id = self.client.get("/api/projects").json()[0]["id"]

        def fake_stream(
            _self: QAService,
            _state: Path,
            _question: str,
            _session_id: str | None = None,
            _mode: str = "quick",
        ):
            yield {
                "type": "start",
                "session_id": "s1",
                "mode": "quick",
                "citations": [],
                "usage": None,
            }
            yield {"type": "delta", "content": "hello"}
            yield {
                "type": "done",
                "answer": "hello",
                "citations": [],
                "session_id": "s1",
                "mode": "quick",
                "usage": None,
            }

        with patch.object(QAService, "ask_stream", fake_stream):
            app = create_app(runs_dir=self.tmp / "runs", repos_dir=self.tmp / "repos")
            response = TestClient(app).post(
                f"/api/projects/{project_id}/ask/stream",
                json={"question": "hello", "mode": "quick"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["content-type"].split(";")[0], "text/event-stream"
        )
        self.assertIn('"type": "start"', response.text)
        self.assertIn('"content": "hello"', response.text)
        self.assertIn('"type": "done"', response.text)

    def test_github_url_parsing(self) -> None:
        self.assertEqual(
            _github_slug("https://github.com/microsoft/graphrag"),
            ("microsoft", "graphrag"),
        )
        self.assertEqual(
            _github_slug("https://github.com/microsoft/graphrag.git"),
            ("microsoft", "graphrag"),
        )
        self.assertEqual(_github_slug("microsoft/graphrag"), ("microsoft", "graphrag"))
        self.assertIsNone(_github_slug("/tmp/local/repo"))

    def test_local_repo_options_are_exposed(self) -> None:
        clone = self.tmp / "repos" / "owner__repo"
        clone.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(clone)], check=True)
        (clone / ".git").mkdir(exist_ok=True)
        (clone / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
        (clone / ".repodoc-source.json").write_text(
            json.dumps(
                {
                    "full_name": "owner/repo",
                    "url": "https://github.com/owner/repo",
                    "commit": "abc123",
                }
            ),
            encoding="utf-8",
        )
        rows = _local_repo_options(self.tmp / "repos")
        self.assertEqual(rows[0]["full_name"], "owner/repo")
        self.assertEqual(rows[0]["language"], "python")


if __name__ == "__main__":
    unittest.main()
