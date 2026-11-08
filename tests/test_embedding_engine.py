#!/usr/bin/env python3
"""Tests for the Kuku Embedding Engine (embedding_engine.py).

Standard library + numpy only. Provider tests run against the offline test
double; the real EmbeddingGemma 2 provider is exercised when the local
download exists (integration section, skipped otherwise).
"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / 'src'))

import hashlib
import shutil
import unittest
from pathlib import Path

import numpy as np

import embedding_engine as ee
import tempfile

REPO_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = REPO_DIR / "models" / "embeddinggemma-2"


class InstructionFormatterTests(unittest.TestCase):
    def setUp(self):
        self.formatter = ee.TaskInstructionFormatter(ee.PROMPT_ASYMMETRIC_SEARCH)

    def test_documented_query_instruction(self):
        self.assertEqual(
            self.formatter.format_category_query("betrayal of trust"),
            "task: search result | query: betrayal of trust",
        )

    def test_titled_document_format(self):
        self.assertEqual(
            self.formatter.format_candidate_document("body", title="Episode 1"),
            "title: Episode 1 | text: body",
        )

    def test_untitled_document_format(self):
        self.assertEqual(
            self.formatter.format_candidate_document("body"),
            "title: none | text: body",
        )

    def test_symmetric_strategy_shares_instruction(self):
        formatter = ee.TaskInstructionFormatter(ee.PROMPT_SYMMETRIC_SIMILARITY)
        query = formatter.format_category_query("revenge moment")
        document = formatter.format_candidate_document("he plotted his revenge")
        self.assertEqual(query, "task: sentence similarity | query: revenge moment")
        self.assertEqual(document, "task: sentence similarity | query: he plotted his revenge")

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            ee.TaskInstructionFormatter("HYBRID")


class OfflineProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = ee.OfflineTestProvider(dimension=64)

    def test_controlled_token_overlap_similarity(self):
        shared = "the duke weds juliana in the cathedral"
        close = "juliana weds the duke in a cathedral today"
        far = "sowing turnips under a rainy sky"
        vectors = self.provider.embed_texts([shared, close, far])
        rows = ee._normalize_rows(vectors)
        sim = rows @ rows.T
        self.assertGreater(sim[0, 1], sim[0, 2])

    def test_identical_texts_cosine_one(self):
        vectors = ee._normalize_rows(self.provider.embed_texts(["same words repeated", "same words repeated"]))
        self.assertAlmostEqual(float(vectors[0] @ vectors[1]), 1.0, places=5)

    def test_unit_norm_and_determinism(self):
        vectors = ee._normalize_rows(self.provider.embed_texts(["alpha beta", "gamma delta"]))
        again = ee._normalize_rows(self.provider.embed_texts(["alpha beta", "gamma delta"]))
        np.testing.assert_allclose(vectors, again)
        for vector in vectors:
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)


class ConfigTests(unittest.TestCase):
    def test_defaults_validate(self):
        ee.EmbeddingConfig().validated()

    def test_float16_rejected(self):
        with self.assertRaises(ValueError):
            ee.EmbeddingConfig(dtype="float16").validated()

    def test_bad_truncate_dim_rejected(self):
        with self.assertRaises(ValueError):
            ee.EmbeddingConfig(truncate_dim=96).validated()

    def test_bad_batch_size_rejected(self):
        with self.assertRaises(ValueError):
            ee.EmbeddingConfig(batch_size=0).validated()


class MathHelperTests(unittest.TestCase):
    def test_zero_row_normalization_is_safe(self):
        rows = np.zeros((2, 3), dtype=np.float32)
        rows[0, 0] = 3.0
        normalized = ee._normalize_rows(rows)
        self.assertFalse(np.isnan(normalized).any())
        self.assertAlmostEqual(float(np.linalg.norm(normalized[0])), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(normalized[1])), 0.0, places=5)

    def test_truncate_and_renormalize(self):
        vector = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], dtype=np.float32)
        vector = ee._normalize_rows(vector)
        truncated = ee.truncate_and_normalize(vector, 4)
        self.assertEqual(truncated.shape, (1, 4))
        self.assertAlmostEqual(float(np.linalg.norm(truncated[0])), 1.0, places=5)
        np.testing.assert_allclose(truncated[0], vector[0][:4] / np.linalg.norm(vector[0][:4]), rtol=1e-5)

    def test_cosine_matrix_clipped_and_correct(self):
        row = ee._normalize_rows(np.array([[1.0, 0.0], [0.0, 2.0]]))
        col = ee._normalize_rows(np.array([[1.0, 1.0], [5.0, 0.0]]))
        matrix = ee.cosine_similarity_matrix(row, col)
        self.assertAlmostEqual(float(matrix[0, 0]), np.sqrt(0.5), places=5)
        self.assertAlmostEqual(float(matrix[0, 1]), 1.0, places=5)  # [1,0]·[1,0]
        self.assertAlmostEqual(float(matrix[1, 1]), 0.0, places=5)  # [0,1]·[1,0]
        self.assertTrue(((matrix <= 1.0) & (matrix >= -1.0)).all())


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        self.cache_root = Path(tempfile.gettempdir()) / "kuku_cache_tests"
        shutil.rmtree(self.cache_root, ignore_errors=True)
        self.engine_token = "test-token|ASYMMETRIC_SEARCH|float32|768"

    def tearDown(self):
        shutil.rmtree(self.cache_root, ignore_errors=True)

    def test_round_trip(self):
        cache = ee.EmbeddingCache(self.cache_root / "cache", self.engine_token, native_dim=4)
        key = cache.key_for("doc", "some text")
        vector = np.array([0.5, 0.5, 0.0, 0.7], dtype=np.float32)
        self.assertFalse(cache.get([key]))
        cache.put({key: vector / np.linalg.norm(vector)})
        reloaded = ee.EmbeddingCache(self.cache_root / "cache", self.engine_token, native_dim=4)
        hit = reloaded.get([key])
        self.assertIn(key, hit)
        np.testing.assert_allclose(hit[key], vector / np.linalg.norm(vector), rtol=1e-6)

    def test_role_separation(self):
        cache = ee.EmbeddingCache(self.cache_root / "cache2", self.engine_token, native_dim=4)
        self.assertNotEqual(cache.key_for("doc", "same"), cache.key_for("category", "same"))

    def test_dimension_mismatch_rejected_on_flush(self):
        cache = ee.EmbeddingCache(self.cache_root / "cache3", self.engine_token, native_dim=4)
        with self.assertRaises(ValueError):
            cache.put({"k": np.zeros(9, dtype=np.float32)})


class EmbeddingEngineTests(unittest.TestCase):
    def setUp(self):
        self.cache_root = Path(tempfile.gettempdir()) / "kuku_engine_tests"
        shutil.rmtree(self.cache_root, ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(self.cache_root, ignore_errors=True)

    def _engine(self, dimension: int = 64) -> ee.EmbeddingEngine:
        provider = ee.OfflineTestProvider(dimension=dimension)
        return ee.EmbeddingEngine(provider, ee.EmbeddingConfig(batch_size=4),
                                  cache_dir=self.cache_root / "cache")

    def test_deterministic_across_engines(self):
        first = self._engine().embed_candidate_documents(["betrayal scene filmed"], titles=["Episode 1"])[0]
        second = self._engine().embed_candidate_documents(["betrayal scene filmed"], titles=["Episode 1"])[0]
        np.testing.assert_allclose(first, second, rtol=1e-6)

    def test_cache_serves_second_run_without_provider_calls(self):
        engine = self._engine()
        texts = ["alpha", "beta", "gamma"]
        vectors, first_stats = engine.embed_candidate_documents(texts)
        self.assertEqual(first_stats["encoded_now"], 3)
        vectors_again, second_stats = engine.embed_candidate_documents(texts)
        self.assertEqual(second_stats["encoded_now"], 0)
        self.assertEqual(second_stats["cache_hits"], 3)
        np.testing.assert_allclose(vectors, vectors_again, rtol=1e-6)
        self.assertEqual(engine.provider.embed_calls, 1)  # only the first run hit the provider

    def test_cache_key_includes_formatting(self):
        engine = self._engine()
        engine.embed_candidate_documents(["same body"], titles=["Episode 1"])
        # A different title produces a different formatted document -> new key.
        latest_stats = engine.embed_candidate_documents(["same body"], titles=["Episode 2"])[1]
        self.assertEqual(latest_stats["encoded_now"], 1)

    def test_truncation_reduction(self):
        # The offline provider hashes tokens uniformly over 512 dims, so pick
        # tokens guaranteed to land inside the kept 128-dim window; then
        # truncation must preserve mass and renormalize to unit length.
        def find_low_token(tag: str) -> str:
            found = next(
                f"{tag}{i}"
                for i in range(10_000)
                if int(hashlib.sha1(f"{tag}{i}".encode("utf-8")).hexdigest(), 16) % 512 < 128
            )
            self.assertLess(len(found), 8)
            return found

        token_a = find_low_token("tk")
        token_b = find_low_token("sk")
        engine = ee.EmbeddingEngine(
            ee.OfflineTestProvider(dimension=512),
            ee.EmbeddingConfig(batch_size=4, truncate_dim=128),
            cache_dir=self.cache_root / "truncate",
        )
        vectors, _ = engine.embed_candidate_documents([f"alpha {token_a} beta", f"gamma {token_b} delta"])
        self.assertEqual(vectors.shape, (2, 128))
        for row in vectors:
            self.assertGreater(float(np.linalg.norm(row)), 0.0)
            self.assertAlmostEqual(float(np.linalg.norm(row)), 1.0, places=5)

    def test_head_tail_clipping(self):
        text = "A" * 150 + "MIDDLE-MARKER" + "B" * 150
        clipped = ee._clip_text(text, 100)
        # Output never exceeds limit: head half + separator + tail half.
        self.assertLessEqual(len(clipped), 100)
        self.assertTrue(clipped.startswith("A" * 40))
        self.assertTrue(clipped.endswith("B" * 40))
        # Texts within the limit pass through untouched.
        self.assertEqual(ee._clip_text("short text", 100), "short text")

    def test_empty_input(self):
        engine = self._engine()
        vectors, stats = engine.embed_candidate_documents([])
        self.assertEqual(vectors.shape, (0, 64))
        self.assertEqual(stats["total_texts"], 0)


@unittest.skipUnless(LOCAL_MODEL_DIR.exists(), "local EmbeddingGemma 2 download not present")
class RealProviderIntegrationTests(unittest.TestCase):
    """Smoke tests against the locally downloaded EmbeddingGemma 2 weights."""

    @classmethod
    def setUpClass(cls):
        cls.provider = ee.SentenceTransformerProvider(
            ee.EmbeddingConfig(
                local_model_dir=str(LOCAL_MODEL_DIR), text_only=True, batch_size=4
            )
        )
        cls.provider.load()

    def test_native_dimension_and_normalization(self):
        self.assertEqual(self.provider.dimension, 768)
        vectors = self.provider.embed_texts([
            "task: search result | query: a shocking betrayal",
            "title: none | text: she revealed the secret identity",
        ])
        self.assertEqual(vectors.shape, (2, 768))
        for row in vectors:
            self.assertAlmostEqual(float(np.linalg.norm(row)), 1.0, places=4)
        self.assertFalse(np.isnan(vectors).any())

    def test_search_semantics_sanity(self):
        query = self.provider.embed_texts(["task: search result | query: a shocking public betrayal of trust"])
        faithful = self.provider.embed_texts(["title: none | text: the matriarch betrayed the family before the court"])
        unrelated = self.provider.embed_texts(["title: none | text: tidy the sitting room of the duchy mansion"])
        sims = ee.cosine_similarity_matrix(query, np.vstack([faithful, unrelated]))[0]
        self.assertGreater(float(sims[0]), float(sims[1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
