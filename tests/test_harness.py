from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from harness import aggregate, architecture, benchmark, prepare, runner
from harness.config import Config
from harness.metrics import extract_results_metrics, normalize_tokens, usage_stats
from harness.util import ROOT, HarnessError, read_json, write_json


class TempDirTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="repodoc-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def make_config(self) -> Config:
        (self.tmp / "baselines" / "dummy").mkdir(parents=True)
        (self.tmp / "adapters").mkdir(parents=True)
        (self.tmp / "data" / "repos" / "Repo").mkdir(parents=True)
        (self.tmp / "data" / "repos" / "Repo" / "README.md").write_text(
            "# repo\n", encoding="utf-8"
        )
        adapter = self.tmp / "adapters" / "dummy.sh"
        adapter.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$REPODOC_OUT" "$REPODOC_WORKDIR"
printf '# Dummy\\n\\nhello\\n' > "$REPODOC_OUT/index.md"
printf '# Dummy overview\\n' > "$REPODOC_OUT/overview.md"
printf '{"model":"dummy-model","cost_usd":0.25,"tokens":{"input":10,"output":5}}\\n' \\
  > "$REPODOC_WORKDIR/system_metadata.json"
""",
            encoding="utf-8",
        )
        return Config(
            root=self.tmp,
            config_dir=self.tmp / "configs",
            systems={
                "dummy": {
                    "kind": "baseline",
                    "source": "baselines/dummy",
                    "adapter": "adapters/dummy.sh",
                    "extra_args": [],
                    "env": {},
                }
            },
            repos={
                "defaults": {},
                "repos": {"Repo": {"url": "https://example.invalid/repo.git"}},
            },
            evaluation={
                "benchmark": {"source": "benchmarks/none"},
                "rubrics": {"dir": "data/rubrics"},
                "judge": {
                    "models": ["judge-a"],
                    "use_tools": False,
                    "enable_retry": False,
                },
                "run": {
                    "experiments_dir": "runs",
                    "normalize_missing_module_tree": True,
                },
            },
            experiments={},
            locks={},
        )


class ConfigTests(unittest.TestCase):
    def test_project_config(self) -> None:
        cfg = Config.load(ROOT)
        self.assertEqual(len(cfg.repo_names()), 22)
        self.assertEqual(len(cfg.repo_names("core")), 7)
        self.assertIn("codewiki", cfg.system_names())
        self.assertEqual(cfg.system("codewiki")["kind"], "baseline")
        self.assertTrue(cfg.system("codewiki")["adapter_path"].is_file())
        self.assertIn("data/rubrics", str(cfg.rubric_root))


class MetricsTests(TempDirTest):
    def test_extract_results_metrics(self) -> None:
        results = {
            "rubrics": [
                {
                    "requirements": "Top",
                    "weight": 3,
                    "score": 2 / 3,
                    "sub_tasks": [
                        {
                            "requirements": "covered",
                            "weight": 2,
                            "score": 1,
                            "evaluation": {
                                "score": 1,
                                "tokens": {"input": 3, "output": 1},
                            },
                        },
                        {
                            "requirements": "missing",
                            "weight": 1,
                            "score": 0,
                            "evaluation": {
                                "score": 0,
                                "error": True,
                                "tokens": {"input": 2, "output": 1},
                            },
                        },
                    ],
                }
            ],
            "combination_metadata": {"overall_score": 0.1},
        }
        metrics = extract_results_metrics(results)
        self.assertAlmostEqual(metrics["score"], 2 / 3)
        self.assertAlmostEqual(metrics["score_100"], 66.67, places=2)
        self.assertEqual(metrics["num_leaves"], 2)
        self.assertEqual(metrics["num_errors"], 1)
        self.assertEqual(metrics["judge_tokens"], {"input": 5, "output": 2})
        self.assertEqual(metrics["topics"][0]["name"], "Top")
        self.assertIn("architecture", metrics)

    def test_architecture_metrics(self) -> None:
        results = {
            "rubrics": [
                {
                    "requirements": "Architecture",
                    "weight": 3,
                    "sub_tasks": [
                        {
                            "requirements": "Component interaction flow",
                            "weight": 3,
                            "score": 1.0,
                        },
                        {
                            "requirements": "Usage example",
                            "weight": 1,
                            "score": 0.0,
                        },
                    ],
                },
                {
                    "requirements": "Installation",
                    "weight": 1,
                    "sub_tasks": [
                        {"requirements": "Install package", "weight": 1, "score": 0.0}
                    ],
                },
            ]
        }
        metrics = architecture.architecture_metrics(
            results, keywords=["component", "flow"]
        )
        self.assertEqual(metrics["num_leaves"], 1)
        self.assertEqual(metrics["total_leaves"], 3)
        self.assertAlmostEqual(metrics["score"], 1.0)
        self.assertEqual(metrics["topics"][0]["name"], "Architecture")
        filtered = architecture.filter_architecture_rubrics(
            results, keywords=["component", "flow"]
        )
        self.assertEqual(len(list(architecture._walk_leaves(filtered["rubrics"]))), 1)
        topic_metrics = architecture.architecture_metrics(
            results, include_topics=["Architecture"]
        )
        self.assertEqual(topic_metrics["num_leaves"], 2)
        self.assertAlmostEqual(topic_metrics["score"], 0.75)

    def test_docs_architecture_stats(self) -> None:
        docs = self.tmp / "docs"
        docs.mkdir()
        (docs / "overview.md").write_text(
            "# Overview\n\n## Architecture\n\n"
            "```mermaid\nflowchart LR\n A --> B\n```\n\n"
            "[child](child.md)\n",
            encoding="utf-8",
        )
        (docs / "child.md").write_text(
            "# Child\n\n```mermaid\nsequenceDiagram\n A->>B: run\n```\n",
            encoding="utf-8",
        )
        stats = architecture.docs_architecture_stats(docs)
        self.assertEqual(stats["pages"], 2)
        self.assertEqual(stats["mermaid_blocks"], 2)
        self.assertEqual(stats["diagram_types"]["flowchart"], 1)
        self.assertEqual(stats["diagram_types"]["sequenceDiagram"], 1)
        self.assertEqual(stats["internal_markdown_links"], 1)
        self.assertGreaterEqual(stats["architecture_headings"], 1)


class BenchmarkTests(TempDirTest):
    def test_normalize_missing_module_tree(self) -> None:
        docs = self.tmp / "docs"
        docs.mkdir()
        (docs / "b.md").write_text("# B\n", encoding="utf-8")
        (docs / "a.md").write_text("# A\n", encoding="utf-8")
        (docs / "overview.md").write_text("# Overview\n", encoding="utf-8")

        self.assertTrue(benchmark.normalize_missing_module_tree(docs))
        tree = read_json(docs / "module_tree.json")
        self.assertEqual(list(tree), ["a", "b", "overview"])
        self.assertFalse(benchmark.normalize_missing_module_tree(docs))

    def test_deepwiki_numbering_is_not_synthesized(self) -> None:
        docs = self.tmp / "docs"
        docs.mkdir()
        (docs / "content_0.md").write_text(
            "1-Overview\n\n# Overview\n", encoding="utf-8"
        )
        (docs / "content_1.md").write_text("2-API\n\n# API\n", encoding="utf-8")
        self.assertFalse(benchmark.normalize_missing_module_tree(docs))
        self.assertFalse((docs / "module_tree.json").exists())

    def test_nested_markdown_detection(self) -> None:
        docs = self.tmp / "docs"
        (docs / "sub").mkdir(parents=True)
        (docs / "sub" / "page.md").write_text("# Page\n", encoding="utf-8")
        top, nested = benchmark.list_markdown(docs)
        self.assertEqual(top, [])
        self.assertEqual([p.name for p in nested], ["page.md"])


class BenchmarkCommandTests(TempDirTest):
    def test_parse_command_args(self) -> None:
        cfg = Config.load(ROOT)
        docs = self.tmp / "docs"
        docs.mkdir()
        (docs / "index.md").write_text("# Index\n", encoding="utf-8")
        parsed = self.tmp / "parsed"
        captured = {}

        def fake_run(cfg_, args, log_path, check=True):
            captured["args"] = args
            return {"returncode": 0}

        with mock.patch.object(benchmark, "_run", fake_run):
            benchmark.parse_docs(cfg, docs, parsed, "Repo", self.tmp / "parse.log")
        self.assertEqual(captured["args"][0], "parse")
        self.assertIn("--docs-dir", captured["args"])
        self.assertIn("--project-name", captured["args"])
        self.assertTrue((docs / "module_tree.json").is_file())

    def test_judge_command_args(self) -> None:
        cfg = Config.load(ROOT)
        parsed = self.tmp / "parsed"
        parsed.mkdir()
        write_json(parsed / "docs_tree.json", {})
        write_json(parsed / "structured_docs.json", {})
        rubrics = self.tmp / "rubrics.json"
        write_json(rubrics, {"rubrics": []})
        captured = {}

        def fake_run(cfg_, args, log_path, check=True):
            captured["args"] = args
            return {"returncode": 0}

        with mock.patch.object(benchmark, "_run", fake_run):
            benchmark.judge_docs(
                cfg,
                rubrics,
                parsed,
                self.tmp / "results.json",
                self.tmp / "judge.log",
                models=["judge-a"],
                batch_size=2,
                method="average",
            )
        args = captured["args"]
        self.assertEqual(args[0], "judge")
        self.assertIn("--rubrics", args)
        self.assertIn("--docs", args)
        self.assertIn("--batch-size", args)
        self.assertIn("judge-a", args)

    def test_rubrics_command_args(self) -> None:
        cfg = Config.load(ROOT)
        parsed = self.tmp / "parsed"
        parsed.mkdir()
        write_json(parsed / "docs_tree.json", {})
        write_json(parsed / "structured_docs.json", {})
        captured = {}

        def fake_run(cfg_, args, log_path, check=True):
            captured["args"] = args
            return {"returncode": 0}

        with mock.patch.object(benchmark, "_run", fake_run):
            benchmark.generate_rubrics(
                cfg,
                parsed,
                self.tmp / "rubrics.json",
                self.tmp / "rubrics.log",
                models=["rubric-model"],
            )
        args = captured["args"]
        self.assertEqual(args[0], "rubrics")
        self.assertIn("--docs", args)
        self.assertIn("--out", args)
        self.assertIn("rubric-model", args)
        self.assertIn("--use-tools", args)


class RunnerTests(TempDirTest):
    def test_generate_reuses_successful_run(self) -> None:
        cfg = self.make_config()
        paths, meta, skipped = runner.generate(
            cfg, "dummy", "Repo", "unit", ref="base", force=False
        )
        self.assertFalse(skipped)
        self.assertEqual(meta["status"], "generated")
        self.assertTrue((paths.docs / "index.md").is_file())
        self.assertTrue((paths.work / "system_metadata.json").is_file())

        paths2, meta2, skipped2 = runner.generate(
            cfg, "dummy", "Repo", "unit", ref="base", force=False
        )
        self.assertTrue(skipped2)
        self.assertEqual(paths2.docs, paths.docs)
        self.assertEqual(meta2["stages"]["generate"]["status"], "skipped")

    def test_update_copies_base_docs(self) -> None:
        cfg = self.make_config()
        base_paths, _meta, _skipped = runner.generate(
            cfg, "dummy", "Repo", "base", ref="base", force=False
        )
        paths, meta, skipped = runner.generate(
            cfg,
            "dummy",
            "Repo",
            "update",
            ref="base",
            mode="update",
            base_docs=str(base_paths.docs),
            force=False,
        )
        self.assertFalse(skipped)
        self.assertTrue((paths.docs / "index.md").is_file())
        self.assertEqual(meta["config"]["generate"]["mode"], "update")
        self.assertIn("base_docs_fingerprint", meta["versions"])

    def test_run_paths_reject_traversal(self) -> None:
        cfg = self.make_config()
        with self.assertRaises(HarnessError):
            runner.run_paths(cfg, "../evil", "dummy", "Repo")
        paths = runner.run_paths(cfg, "python-repo-full-v1", "dummy", "Repo")
        self.assertEqual(
            paths.root,
            cfg.runs_dir / "python-repo-full-v1" / "systems" / "dummy" / "Repo",
        )

    def test_evaluate_merges_adapter_metadata(self) -> None:
        cfg = self.make_config()
        paths, _meta, _skipped = runner.generate(
            cfg, "dummy", "Repo", "unit", ref="base", force=False
        )
        rubric = cfg.rubric_path("Repo")
        write_json(rubric, {"rubrics": []})

        sample_results = {
            "rubrics": [
                {
                    "requirements": "Top",
                    "weight": 1,
                    "score": 0.5,
                    "sub_tasks": [
                        {
                            "requirements": "A",
                            "weight": 1,
                            "score": 1,
                            "evaluation": {"score": 1},
                        },
                        {
                            "requirements": "B",
                            "weight": 1,
                            "score": 0,
                            "evaluation": {"score": 0},
                        },
                    ],
                }
            ]
        }

        def fake_parse(*args, **kwargs):
            return {"skipped": False, "duration_sec": 0.1}

        def fake_judge(cfg_, rubrics_path, parsed_dir, out_path, log_path, **kwargs):
            write_json(out_path, sample_results)
            return {"returncode": 0, "duration_sec": 0.2}

        def fake_reports(cfg_, results_path, results_dir, *, max_score=None):
            (results_dir / "report.md").write_text("# report\n", encoding="utf-8")
            return {"num_failed": 1}

        with (
            mock.patch.object(benchmark, "parse_docs", fake_parse),
            mock.patch.object(benchmark, "judge_docs", fake_judge),
            mock.patch.object(benchmark, "write_reports", fake_reports),
        ):
            paths2, metrics = runner.evaluate(cfg, "dummy", "Repo", "unit", force=False)

        self.assertEqual(paths2.metrics, paths.metrics)
        self.assertAlmostEqual(metrics["score"], 0.5)
        self.assertEqual(metrics["generation"]["model"], "dummy-model")
        self.assertEqual(metrics["generation"]["cost_usd"], 0.25)
        self.assertEqual(
            metrics["generation"]["tokens"],
            {"input": 10, "output": 5, "total": 15},
        )
        self.assertEqual(metrics["generation"]["cost_status"], "unknown")
        self.assertTrue(paths.metrics.is_file())
        manifest = read_json(cfg.runs_dir / "unit" / "experiment.json")
        self.assertEqual(manifest["runs"][0]["repo"], "Repo")
        self.assertEqual(manifest["runs"][0]["tokens"]["total"], 15)
        meta = read_json(paths.meta)
        self.assertEqual(meta["status"], "evaluated")
        self.assertIn("docs_fingerprint", meta["versions"])


class PrepareTests(TempDirTest):
    def test_local_clone_and_worktree(self) -> None:
        src = self.tmp / "source"
        src.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "main", str(src)], check=True)
        (src / "README.md").write_text("# source\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(src), "add", "README.md"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(src),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-q",
                "-m",
                "init",
            ],
            check=True,
        )
        commit = subprocess.run(
            ["git", "-C", str(src), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()

        cfg = Config(
            root=self.tmp,
            config_dir=self.tmp / "configs",
            systems={},
            repos={"defaults": {}, "repos": {"Repo": {"url": str(src)}}},
            evaluation={},
            experiments={},
            locks={"Repo": {"commit": commit, "url": str(src)}},
        )
        prepare.prepare_checkouts(cfg, ["Repo"])
        checkout = cfg.checkout_dir("Repo", "base")
        self.assertTrue((checkout / "README.md").is_file())
        self.assertEqual(
            subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip(),
            commit,
        )


class UsageTests(unittest.TestCase):
    def test_normalize_tokens_handles_provider_aliases(self) -> None:
        self.assertEqual(
            normalize_tokens({"prompt_tokens": 12, "completion_tokens": 3}),
            {"input": 12, "output": 3, "total": 15},
        )
        self.assertEqual(
            normalize_tokens(
                {
                    "input_tokens": 20,
                    "output_tokens": 5,
                    "requests": 2,
                }
            ),
            {"input": 20, "output": 5, "total": 25, "requests": 2},
        )

    def test_usage_stats_reads_codewiki_ledger(self) -> None:
        stats = usage_stats(
            {
                "usage": {
                    "schema_version": 2,
                    "currency": "USD",
                    "cost_status": "partial",
                    "incomplete": False,
                    "n_calls": 3,
                    "totals": {
                        "input_tokens": 100,
                        "output_tokens": 40,
                        "cost_usd": 0.12,
                    },
                    "calls": [
                        {"seconds": 1.5},
                        {"seconds": 2.25},
                    ],
                }
            }
        )
        self.assertEqual(stats["tokens"]["total"], 140)
        self.assertEqual(stats["cost_usd"], 0.12)
        self.assertEqual(stats["cost_status"], "partial")
        self.assertEqual(stats["llm_duration_sec"], 3.75)


class AggregateTests(TempDirTest):
    def test_summarize(self) -> None:
        cfg = self.make_config()
        run_root = cfg.runs_dir / "exp" / "systems" / "dummy" / "Repo"
        run_root.mkdir(parents=True)
        write_json(
            run_root / "metrics.json",
            {
                "system": "dummy",
                "repo": "Repo",
                "score_100": 75.5,
                "num_leaves": 10,
                "num_errors": 1,
                "generation": {"model": "dummy", "duration_sec": 12, "cost_usd": 0.2},
                "judge": {"models": ["judge-a"]},
                "architecture": {"score": 0.8, "score_100": 80.0},
                "artifacts": {
                    "docs": {
                        "pages": 3,
                        "architecture": {
                            "mermaid_blocks": 2,
                            "diagram_types": {
                                "flowchart": 1,
                                "sequenceDiagram": 1,
                            },
                            "internal_markdown_links": 4,
                        },
                    }
                },
                "topics": [{"name": "Architecture", "weight": 3, "score": 0.8}],
            },
        )
        write_json(run_root / "meta.json", {"status": "evaluated"})
        payload = aggregate.summarize(cfg, "exp")
        self.assertEqual(len(payload["records"]), 1)
        self.assertEqual(payload["averages"][0]["score_mean"], 75.5)
        self.assertTrue((cfg.runs_dir / "exp" / "summary.md").is_file())
        self.assertTrue((cfg.runs_dir / "exp" / "summary.csv").is_file())
        data = json.loads(
            (cfg.runs_dir / "exp" / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(data["topic_averages"]["dummy"]["Architecture"], 80.0)
        self.assertEqual(data["averages"][0]["architecture_mean"], 80.0)
        summary = (cfg.runs_dir / "exp" / "summary.md").read_text(encoding="utf-8")
        self.assertIn("Architecture", summary)
        self.assertIn("Mermaid", summary)
        self.assertIn("Total tokens", summary)


if __name__ == "__main__":
    unittest.main()
