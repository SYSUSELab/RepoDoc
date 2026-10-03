"""Incremental update recovery: failed docs are journaled and retried,
and the git watermark only advances when nothing is left pending."""

import asyncio
import json
import subprocess
from pathlib import Path

from conftest import make_fake_execute


def _commit(repo: Path, message: str) -> str:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "."],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            message,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _metadata(config) -> dict:
    path = Path(config.output_dir) / "data" / "repo_metadata.json"
    return json.loads(path.read_text())


def test_failed_doc_is_journaled_and_watermark_held(
    built_project, monkeypatch
):
    from repodoc.agent.agent import DocAgent
    from repodoc.agent.update import run_update

    config, _graph, base_commit = built_project
    repo = Path(config.repo_path)

    # Change Alpha's body -> Alpha is in doc scope, module + README follow.
    source = repo / "pkg" / "mod.py"
    source.write_text(
        source.read_text().replace("return 1", "return 2")
    )
    current_commit = _commit(repo, "change alpha")

    monkeypatch.setattr(
        DocAgent, "execute", make_fake_execute({"Alpha"})
    )
    result = asyncio.run(run_update(config))

    assert result["status"] == "partial"
    assert result["pending"] is True
    pending_path = Path(config.output_dir) / "data" / "repo_pending_updates.json"
    pending = json.loads(pending_path.read_text())
    assert "pkg.mod.Alpha" in pending["components"]

    # Watermark must NOT advance while docs are pending.
    assert _metadata(config)["generation_info"]["git_commit"] == base_commit
    assert _metadata(config)["update_info"]["pending_docs"]["components"]


def test_next_run_resumes_and_advances_watermark(built_project, monkeypatch):
    from repodoc.agent.agent import DocAgent
    from repodoc.agent.update import run_update

    config, _graph, base_commit = built_project
    repo = Path(config.repo_path)
    source = repo / "pkg" / "mod.py"
    source.write_text(source.read_text().replace("return 1", "return 2"))
    current_commit = _commit(repo, "change alpha")

    monkeypatch.setattr(DocAgent, "execute", make_fake_execute({"Alpha"}))
    first = asyncio.run(run_update(config))
    assert first["status"] == "partial"

    # Retry with a healthy agent: no new code change, yet the pending doc
    # must be regenerated and the watermark must advance.
    monkeypatch.setattr(DocAgent, "execute", make_fake_execute())
    second = asyncio.run(run_update(config))

    assert second["status"] == "success"
    assert second["pending"] is False
    assert not (Path(config.output_dir) / "data" / "repo_pending_updates.json").exists()
    assert _metadata(config)["generation_info"]["git_commit"] == current_commit
    assert (Path(config.docs_dir) / "pkg" / "Alpha.md").exists()


def test_crash_after_graph_save_is_recovered(built_project, monkeypatch):
    """A doc that was never attempted is still recovered on the next run:
    the graph is already converged, so only the pending/fallback scope can
    bring it back."""
    from repodoc.agent.agent import DocAgent
    from repodoc.agent.update import run_update

    config, _graph, _base = built_project
    repo = Path(config.repo_path)
    source = repo / "pkg" / "mod.py"
    source.write_text(source.read_text().replace("return 1", "return 2"))
    _commit(repo, "change alpha")

    # Simulate a crash right after graph mutation: converge the graph by
    # running mutation-only through a failed run, then delete the journal
    # (as if the process died before it was written).
    monkeypatch.setattr(DocAgent, "execute", make_fake_execute({"Alpha"}))
    asyncio.run(run_update(config))
    pending_path = Path(config.output_dir) / "data" / "repo_pending_updates.json"
    pending_path.unlink()

    # Next run: mutation is a no-op, but the changed file's components are
    # re-scoped so docs still get generated.
    monkeypatch.setattr(DocAgent, "execute", make_fake_execute())
    result = asyncio.run(run_update(config))

    assert result["status"] == "success"
    assert (Path(config.docs_dir) / "pkg" / "Alpha.md").exists()
