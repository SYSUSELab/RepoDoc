from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from repodoc.qa.dense import DenseIndex  # noqa: E402
from repodoc.qa.index import Chunk, RepoIndex  # noqa: E402


class DenseIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="repodoc-dense-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0, float(len(text)), 0.5] for text in texts]

    @staticmethod
    def query(text: str) -> list[float]:
        return [1.0, float(len(text)), 0.5]

    def test_cache_reuses_unchanged_chunks(self) -> None:
        chunks = [
            Chunk(id="a", kind="doc", title="Alpha", text="alpha text"),
            Chunk(id="b", kind="code", title="Beta", text="beta text"),
        ]
        index = DenseIndex(
            chunks,
            cache_dir=self.tmp / "index",
            embed_fn=self.embed,
            query_fn=self.query,
        )
        self.assertTrue(index.ensure())
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.calls[0]), 2)
        results = index.search("alpha", limit=2)
        self.assertEqual({chunk.id for chunk, _ in results}, {"a", "b"})
        self.assertTrue((self.tmp / "index" / "embeddings.json").is_file())

        self.calls.clear()
        changed = [
            Chunk(id="a", kind="doc", title="Alpha", text="alpha changed"),
            chunks[1],
        ]
        second = DenseIndex(
            changed,
            cache_dir=self.tmp / "index",
            embed_fn=self.embed,
            query_fn=self.query,
        )
        self.assertTrue(second.ensure())
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0], ["Alpha\nalpha changed"])

    def test_failed_backend_is_status_unavailable(self) -> None:
        chunks = [Chunk(id="a", kind="doc", title="Alpha", text="alpha")]

        def fail(_texts: list[str]) -> list[list[float]]:
            raise RuntimeError("no model")

        index = DenseIndex(
            chunks,
            cache_dir=self.tmp / "index",
            embed_fn=fail,
            query_fn=self.query,
        )
        self.assertFalse(index.ensure())
        self.assertEqual(index.status, "unavailable")
        self.assertEqual(index.search("alpha"), [])

    def test_rrf_fusion_includes_dense_only_results(self) -> None:
        fused = RepoIndex._rrf_fuse(
            lexical=[(0, 10.0), (1, 8.0)],
            dense=[(2, 0.9), (1, 0.8)],
        )
        self.assertEqual(fused, [1, 0, 2])


if __name__ == "__main__":
    unittest.main()
