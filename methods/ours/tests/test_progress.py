from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from repodoc.progress import Progress  # noqa: E402


class ProgressTests(unittest.TestCase):
    def test_progress_lines_are_flushed_and_include_stage_counts(self) -> None:
        stream = io.StringIO()
        progress = Progress(stream=stream)
        progress.stage("Parsing graph")
        progress.step("Generating page", 2, 5, detail="overview.md")
        progress.done("Generated page", 2, 5, detail="overview.md")

        output = stream.getvalue()
        self.assertIn("STAGE Parsing graph", output)
        self.assertIn("Generating page [2/5]", output)
        self.assertIn("overview.md", output)
        self.assertIn("DONE Generated page [2/5]", output)
        self.assertTrue(output.endswith("\n"))

    def test_quiet_progress_writes_nothing(self) -> None:
        stream = io.StringIO()
        Progress(stream=stream, quiet=True).stage("ignored")
        self.assertEqual(stream.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
