#!/usr/bin/env python3
"""Tests for candidate_retriever.py (Stage 3 retrieval engine).

Standard library + numpy only, no model download required:
* retrieval math runs on hand-built similarity matrices,
* the CLI end-to-end test runs headless with ``--offline`` (OfflineTestProvider).

Covers: document builder composition, top-K/tie-break determinism, merge and
dedup accounting, recall sweep, strategy agreement, score distributions,
manual-label precision, renderers and the output-file contract.
"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / 'src'))

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import category_definitions as catdefs
import candidate_retriever as cr

REPO_DIR = Path(__file__).resolve().parent
CATEGORIES = catdefs.ordered_descriptions()  # 13 categories in documented order


def make_candidate(cid: str, batch: int = 0) -> dict:
    """Minimal candidate matching context_builder.py's v1.0 schema surface."""
    return {
        "candidate_id": cid,
        "episode_number": 1 + batch // 10,
        "scene_numbers": [1],
        "start_seconds": 10.0 + batch,
        "end_seconds": 40.0 + batch,
        "duration_seconds": 30.0,
        "span_seconds": 30.0,
        "start_timestamp": "00:10.00",
        "end_timestamp": "00:40.00",
        "segment_count": 1,
        "segments": [],
        "dialogue_transcript": [
            {"speaker": "SIR RODERICK", "speaker_canonical": "RODERICK",
             "delivery": "Furious", "content": "You betrayed this house!"},
        ],
        "actions_and_camera": [
            {"event_type": "camera", "content": "Push-in on GERALD's frozen face."},
        ],
        "speakers": ["RODERICK"],
        "untimestamped_events": [],
        "dialogue_event_count": 1,
        "crosses_scene_boundary": False,
        "scenes": [{"scene_number": 1, "heading": "INT. HALL – NIGHT"}],
        "episode": {"episode_number": 1, "episode_label": "Episode 1",
                     "logline": "A betrayal at the bridal altar."},
    }


def build_matrix(rows: dict[str, list[float]], candidate_ids: list[str]) -> np.ndarray:
    return np.array([rows[cid] for cid in candidate_ids], dtype=np.float32)


class DocumentBuilderTests(unittest.TestCase):
    def test_composition_includes_all_content_types(self):
        candidate = make_candidate("c_doc_1")
        title, body = cr.build_candidate_document(candidate)
        self.assertIn("Episode 1", title)
        self.assertIn("INT. HALL – NIGHT", title)
        self.assertIn("RODERICK", body)
        self.assertIn("(Furious)", body)  # delivery preserved
        self.assertIn("You betrayed this house!", body)
        self.assertIn("[camera]", body)  # staging preserved
        self.assertIn("Push-in on GERALD", body)
        self.assertIn("Runtime: 30.0", body)
        self.assertIn("Logline:", body)
        self.assertIn("A betrayal at the bridal altar.", body)

    def test_empty_candidate_falls_back(self):
        candidate = make_candidate("c_doc_2")
        candidate["dialogue_transcript"] = []
        candidate["actions_and_camera"] = []
        candidate["episode"]["logline"] = ""
        _title, body = cr.build_candidate_document(candidate)
        self.assertIn("No transcript content in this window.", body)

    def test_multi_segment_note(self):
        candidate = make_candidate("c_doc_3")
        candidate["segment_count"] = 2
        _title, body = cr.build_candidate_document(candidate)
        self.assertIn("multi-segment", body)

    def test_delivery_fallback_when_absent(self):
        candidate = make_candidate("c_doc_4")
        candidate["dialogue_transcript"][0] = {
            "speaker": "MAID", "speaker_canonical": None, "delivery": None, "content": "Yes, my lady.",
        }
        _title, body = cr.build_candidate_document(candidate)
        self.assertIn("MAID: Yes, my lady.", body)
        self.assertNotIn("(None)", body)


class RetrieveTopCandidatesTests(unittest.TestCase):
    def setUp(self):
        self.candidate_ids = [f"c{i:02d}" for i in range(8)]
        self.candidates = [make_candidate(cid, batch=i) for i, cid in enumerate(self.candidate_ids)]

    def test_top_k_selection_and_ranking(self):
        # Category A column: c07 best ... c00 worst; category B reversed.
        matrix = build_matrix(
            {cid: [7 - i, float(i)] for i, cid in enumerate(self.candidate_ids)},
            self.candidate_ids,
        )
        alpha, beta = CATEGORIES[0], CATEGORIES[1]
        results, stats = cr.retrieve_top_candidates(
            matrix, self.candidates, [alpha, beta], cr.RetrievalConfig(top_k_per_category=3)
        )
        self.assertEqual(stats["retrieval_slots"], 6)
        self.assertEqual(stats["duplicate_slots_merged"], 0)  # disjoint tops
        self.assertEqual(stats["distinct_candidates_retrieved"], 6)
        alpha_hits = [h for r in results for h in r["category_hits"] if h["category_id"] == alpha.category_id]
        ordered_ids = [h["score"] for h in sorted(alpha_hits, key=lambda h: h["category_rank"])]
        self.assertEqual(ordered_ids, [7.0, 6.0, 5.0])
        # Ranks are 1-based
        self.assertEqual({h["category_rank"] for h in alpha_hits}, {1, 2, 3})

    def test_ties_broken_by_candidate_index(self):
        matrix = np.zeros((4, 1), dtype=np.float32)  # all tied
        candidates = [make_candidate(f"c{i}", batch=i) for i in range(4)]
        results, _stats = cr.retrieve_top_candidates(
            matrix, candidates, [CATEGORIES[0]], cr.RetrievalConfig(top_k_per_category=2)
        )
        first = results[0]
        self.assertEqual(first["candidate_id"], "c0")  # lowest index wins under stable sort
        self.assertEqual(first["best_categories"][0]["score"], 0.0)

    def test_merge_keeps_all_hits_and_full_score_row(self):
        # Every category retrieves the same candidates -> one merged result
        # with all 13 category hits and a complete similarity row.
        column = [0.9, 0.8, 0.7, 0.6]
        matrix = np.tile(np.array(column, dtype=np.float32).reshape(4, 1), (1, len(CATEGORIES)))
        candidates = [make_candidate(f"c{i}", batch=i) for i in range(4)]
        results, stats = cr.retrieve_top_candidates(
            matrix, candidates, CATEGORIES, cr.RetrievalConfig(top_k_per_category=2)
        )
        # Every category retrieves {c0, c1}: two distinct entries, one per candidate.
        self.assertEqual(stats["retrieval_slots"], len(CATEGORIES) * 2)
        self.assertEqual(stats["duplicate_slots_merged"], len(CATEGORIES) * 2 - 2)
        self.assertEqual(len(results), 2)
        top_entry = results[0]
        self.assertEqual(top_entry["candidate_id"], "c0")
        self.assertEqual(top_entry["category_hits_count"], len(CATEGORIES))
        self.assertEqual(len(top_entry["category_hits"]), len(CATEGORIES))
        self.assertEqual(top_entry["best_category_rank"], 1)
        self.assertEqual(
            set(top_entry["similarity_scores_all_categories"]), {c.category_id for c in CATEGORIES}
        )
        self.assertEqual(len(top_entry["best_categories"]), 3)  # default best_categories_kept
        best_scores = [h["score"] for h in top_entry["best_categories"]]
        self.assertEqual(best_scores, sorted(best_scores, reverse=True))

    def test_traceability_fields_present(self):
        matrix = np.array([[0.5], [0.4]], dtype=np.float32)
        candidates = [make_candidate("c0", batch=0), make_candidate("c1", batch=1)]
        second = candidates[1]
        second["crosses_scene_boundary"] = True
        results, _stats = cr.retrieve_top_candidates(
            matrix, candidates, CATEGORIES[:1], cr.RetrievalConfig(top_k_per_category=2)
        )
        self.assertEqual({r["candidate_id"] for r in results}, {"c0", "c1"})
        by_id = {r["candidate_id"]: r for r in results}
        entry = by_id["c1"]
        self.assertEqual(entry["start_timestamp"], "00:10.00")
        self.assertEqual(entry["end_timestamp"], "00:40.00")
        self.assertEqual(entry["duration_seconds"], 30.0)
        self.assertEqual(entry["episode_number"], 1)
        self.assertTrue(entry["crosses_scene_boundary"])
        self.assertIn("betrayed", entry["document_snippet"])

    def test_top_k_larger_than_corpus(self):
        matrix = np.array([[0.5], [0.4]], dtype=np.float32)
        candidates = [make_candidate("c0"), make_candidate("c1")]
        results, stats = cr.retrieve_top_candidates(
            matrix, candidates, CATEGORIES[:1], cr.RetrievalConfig(top_k_per_category=99)
        )
        # Effectively clamps to corpus size; every candidate retrieved.
        self.assertEqual(len(results), 2)
        self.assertEqual(stats["duplicate_slots_merged"], 1 * 2 - 2)

    def test_original_candidates_untouched(self):
        matrix = np.array([[0.9], [0.8]], dtype=np.float32)
        candidates = [make_candidate("c0"), make_candidate("c1")]
        snapshot = json.dumps(candidates)
        cr.retrieve_top_candidates(matrix, candidates, CATEGORIES[:1], cr.RetrievalConfig(top_k_per_category=1))
        self.assertEqual(json.dumps(candidates), snapshot)


class RecallSweepTests(unittest.TestCase):
    def test_monotonic_growth_and_jaccard(self):
        candidate_ids = [f"c{i:02d}" for i in range(6)]
        candidates = [make_candidate(cid, batch=i) for i, cid in enumerate(candidate_ids)]
        # cat0 column: c00, c01 clearly on top; cat1 reversed. Rows are candidates.
        per_candidate = [
            [0.9, 0.8],  # c00
            [0.8, 0.9],  # c01
            [0.1, 0.1],  # c02
            [0.2, 0.2],  # c03
            [0.3, 0.3],  # c04
            [0.4, 0.4],  # c05
        ]
        matrix = build_matrix(
            {cid: values for cid, values in zip(candidate_ids, per_candidate)},
            candidate_ids,
        )
        sweep = cr.recall_sweep(matrix, candidates, CATEGORIES[:2], k_values=(1, 2, 3))
        distinct = [row["distinct_candidates"] for row in sweep]
        self.assertEqual(distinct, [2, 2, 3])  # grows, never shrinks
        self.assertIn("jaccard_vs_previous_k", sweep[1])
        self.assertEqual(sweep[1]["jaccard_vs_previous_k"], 1.0)  # same set at k=2
        self.assertLess(sweep[2]["jaccard_vs_previous_k"], 1.0)   # k=3 adds c02

    def test_multi_category_counting(self):
        candidate_ids = ["c0", "c1"]
        candidates = [make_candidate(cid, batch=i) for i, cid in enumerate(candidate_ids)]
        matrix = np.array([[0.9, 0.8], [0.85, 0.7]], dtype=np.float32)
        sweep = cr.recall_sweep(matrix, candidates, CATEGORIES[:2], k_values=(2,))
        self.assertEqual(sweep[0]["candidates_in_multiple_categories"], 2)


class StrategyAgreementTests(unittest.TestCase):
    def test_identical_matrices_full_agreement(self):
        matrix = np.array([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]], dtype=np.float32)
        ids = ["c0", "c1", "c2"]
        report = cr.strategy_agreement(matrix, matrix, CATEGORIES[:2], ids, top_k=2)
        self.assertEqual(len(report), 2)
        for row in report:
            self.assertEqual(row["jaccard_at_top2"], 1.0)

    def test_disjoint_topk_zero_agreement(self):
        matrix_a = np.array([[0.9], [0.8], [0.1], [0.05]], dtype=np.float32)
        matrix_b = np.array([[0.1], [0.05], [0.9], [0.8]], dtype=np.float32)
        ids = ["c0", "c1", "c2", "c3"]
        report = cr.strategy_agreement(matrix_a, matrix_b, CATEGORIES[:1], ids, top_k=2)
        self.assertEqual(report[0]["jaccard_at_top2"], 0.0)

    def test_partial_agreement_fraction(self):
        # Shared top: c0, c1 in both; then a has c2 while b has c3.
        # Intersection {c0,c1}=2; union {c0,c1,c2,c3}=4 -> jaccard 0.5.
        matrix_a = np.array([[0.9], [0.8], [0.7], [0.1]], dtype=np.float32)
        matrix_b = np.array([[0.9], [0.8], [0.1], [0.7]], dtype=np.float32)
        ids = ["c0", "c1", "c2", "c3"]
        report = cr.strategy_agreement(matrix_a, matrix_b, CATEGORIES[:1], ids, top_k=3)
        self.assertEqual(report[0]["jaccard_at_top3"], 0.5)


class ScoreDistributionTests(unittest.TestCase):
    def test_known_quantiles(self):
        # 20 values 0.0..0.95 in 0.05 steps: median is between 0.45 and 0.50.
        values = [i * 0.05 for i in range(20)]
        matrix = np.array(values, dtype=np.float32).reshape(20, 1)
        rows = cr.score_distributions(matrix, CATEGORIES[:1])
        row = rows[0]
        self.assertEqual(row["category_id"], CATEGORIES[0].category_id)
        self.assertAlmostEqual(row["min"], 0.0, places=5)
        self.assertAlmostEqual(row["max"], 0.95, places=5)
        self.assertAlmostEqual(row["median"], 0.475, places=5)
        self.assertAlmostEqual(row["mean"], sum(values) / 20, places=5)
        self.assertLessEqual(row["p10"], row["p25"])
        self.assertLessEqual(row["p25"], row["median"])
        self.assertLessEqual(row["median"], row["p75"])
        self.assertLessEqual(row["p75"], row["p90"])

    def test_monotonic_across_categories(self):
        matrix = np.array([[0.1, 0.9], [0.2, 0.95]], dtype=np.float32)
        rows = cr.score_distributions(matrix, CATEGORIES[:2])
        self.assertLess(rows[0]["max"], rows[1]["max"])


class ManualLabelTests(unittest.TestCase):
    def _payload(self):
        # Order for cat_a: c0(0.9) c1(0.8) c2(0.7) c3(0.6) c4(0.5) c5(0.4)...
        candidate_ids = [f"c{i}" for i in range(6)]
        matrix = np.array([[0.9 - 0.1 * i] for i in range(6)], dtype=np.float32)
        return {
            "similarity": {
                "candidate_ids": candidate_ids,
                "category_ids": ["cat_a"],
                "matrix": [[round(float(v), 6)] for v in matrix[:, 0]],
            }
        }

    def test_precision_counting(self):
        labels = {
            "cat_a": {"c0": "relevant", "c1": "not", "c2": "partial", "c3": "relevant", "c5": "not"}
        }
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as tmp:
            path = Path(tmp) / "labels.json"
            path.write_text(json.dumps(labels), encoding="utf-8")
            report = cr.apply_manual_labels(self._payload(), path, top_ks=(5, 10))
        category_report = report["categories"]["cat_a"]
        # top5: relevant, not, partial, relevant, unlabeled(c4 scored 0.5).
        # precision@5 = 3/5 = 0.6 (unlabeled counts as miss);
        # precision@10 includes c5 -> 3/6 = 0.5.
        self.assertEqual(category_report["precision"]["precision_at_5"], 0.6)
        self.assertEqual(category_report["precision"]["precision_at_10"], 0.5)
        self.assertEqual(category_report["ratings"]["c2"], "partial")

    def test_unreviewed_marks_rating_gaps(self):
        labels = {"cat_a": {"c0": "relevant"}}
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as tmp:
            path = Path(tmp) / "labels.json"
            path.write_text(json.dumps(labels), encoding="utf-8")
            report = cr.apply_manual_labels(self._payload(), path, top_ks=(5, 10))
        ratings = report["categories"]["cat_a"]["ratings"]
        self.assertEqual(ratings.get("c0"), "relevant")
        self.assertEqual(ratings.get("c4"), "unreviewed")  # rank 5 unlabeled

    def test_missing_file_returns_none(self):
        self.assertIsNone(cr.apply_manual_labels(self._payload(), Path("/tmp/opencode/definitely-missing.json")))

    def test_unknown_category_label_file_ignored(self):
        labels = {"not_a_category": {"c0": "relevant"}}
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as tmp:
            path = Path(tmp) / "labels.json"
            path.write_text(json.dumps(labels), encoding="utf-8")
            self.assertIsNone(cr.apply_manual_labels(self._payload(), path))


class RendererTests(unittest.TestCase):
    def _payload(self):
        candidate_ids = ["c1", "c0"]
        candidates = [make_candidate(cid, batch=1 if cid == "c1" else 0) for cid in candidate_ids]
        matrix = build_matrix({"c1": [0.8], "c0": [0.9]}, candidate_ids)
        results, stats = cr.retrieve_top_candidates(
            matrix, candidates, CATEGORIES[:1], cr.RetrievalConfig(top_k_per_category=2)
        )
        return {
            "config": {"categories": [CATEGORIES[0].category_id]},
            "results": results,
        }, stats

    def test_top10_markdown_shape(self):
        payload, _stats = self._payload()
        markdown = cr.render_top10_markdown(payload)
        self.assertIn("# Top 10 candidates per category", markdown)
        self.assertIn(f"## {CATEGORIES[0].category_id}", markdown)
        self.assertIn("c0", markdown)
        self.assertIn("c1", markdown)
        self.assertIn("score 0.9000", markdown)
        self.assertIn("not probabilities", markdown)

    def test_manual_review_markdown_shape(self):
        payload, _stats = self._payload()
        markdown = cr.render_manual_review_markdown(payload)
        self.assertIn("manual_eval_labels.json", markdown)
        self.assertIn("- [ ] c0", markdown)
        self.assertIn("- [ ] c1",markdown)


class TimerTests(unittest.TestCase):
    def test_stop_unknown_returns_zero(self):
        timer = cr._Timer()
        self.assertEqual(timer.stop("never-started"), 0.0)

    def test_report_rounds(self):
        import time
        timer = cr._Timer()
        timer.start("x")
        time.sleep(0.001)
        timer.stop("x")
        report = timer.report()
        self.assertIn("x", report)
        self.assertGreaterEqual(report["x"], 0.0)


class LoadCandidatesTests(unittest.TestCase):
    def test_wrapper_payload(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as tmp:
            path = Path(tmp) / "candidates.json"
            path.write_text(json.dumps({"candidates": [make_candidate("c0")]}), encoding="utf-8")
            loaded = cr.load_candidates(path)
        self.assertEqual(loaded[0]["candidate_id"], "c0")

    def test_bare_list(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as tmp:
            path = Path(tmp) / "candidates.json"
            path.write_text(json.dumps([make_candidate("c9")]), encoding="utf-8")
            loaded = cr.load_candidates(path)
        self.assertEqual(loaded[0]["candidate_id"], "c9")

    def test_invalid_payload_raises(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as tmp:
            path = Path(tmp) / "candidates.json"
            path.write_text(json.dumps({"other": 1}), encoding="utf-8")
            with self.assertRaises(ValueError):
                cr.load_candidates(path)


class CliEndToEndTests(unittest.TestCase):
    """Full CLI run headless (offline provider) asserting the output contract."""

    def setUp(self):
        self.tmp_root = Path(tempfile.mkdtemp(prefix="kuku_cli_", dir="/tmp/opencode"))
        self.candidates_path = self.tmp_root / "candidates.json"
        corpus = [make_candidate(f"c{i:02d}", batch=i) for i in range(12)]
        # Wrapper payload like context_output/candidates.json.
        self.candidates_path.write_text(
            json.dumps({"schema_version": "1.0", "candidates": corpus}), encoding="utf-8"
        )
        self.output_dir = self.tmp_root / "retrieval_output"

    def tearDown(self):
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def _run(self, extra_args: list[str]) -> int:
        return cr.main([
            "--candidates", str(self.candidates_path),
            "--output", str(self.output_dir),
            "--manual-labels", str(self.tmp_root / "no_labels_here.json"),
            "--offline",
            "--limit-candidates", "12",
            *extra_args,
        ])

    def test_offline_run_produces_full_contract(self):
        exit_code = self._run([])
        self.assertEqual(exit_code, 0)
        # Files contract
        for name in (
            "fast_discovery_results.json", "fast_discovery_results.jsonl",
            "evaluation_report.json", "candidate_embeddings.npz",
            "embedding_manifest.json", "top10_per_category.md",
            "manual_eval_review.md", "embedding_cache/embedding_cache.npz",
        ):
            self.assertTrue((self.output_dir / name).exists(), f"missing: {name}")

        payload = json.loads((self.output_dir / "fast_discovery_results.json").read_text(encoding="utf-8"))
        self.assertNotIn("_vectors", payload)  # vectors must never be inlined in JSON
        self.assertNotIn("_vectors", json.dumps(payload))
        self.assertEqual(len(payload["results"]), 12)
        self.assertEqual(payload["similarity"]["candidate_ids"], [f"c{i:02d}" for i in range(12)])
        self.assertEqual(len(payload["similarity"]["matrix"][0]), len(CATEGORIES))
        first = payload["results"][0]
        for field in ("candidate_id", "episode_number", "start_timestamp", "end_timestamp",
                      "duration_seconds", "best_categories", "similarity_scores_all_categories"):
            self.assertIn(field, first)
        self.assertTrue(payload["category_statistics"]["recall_sweep"])

        # JSONL rows match the JSON payload's results.
        jsonl_rows = [
            json.loads(line)
            for line in (self.output_dir / "fast_discovery_results.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([row["candidate_id"] for row in jsonl_rows],
                         [row["candidate_id"] for row in payload["results"]])

        # NPZ stores vectors separately, aligned with candidate_ids.
        with np.load(self.output_dir / "candidate_embeddings.npz") as archive:
            self.assertEqual(archive["candidate_embeddings"].shape, (12, 64))
            self.assertEqual(list(archive["candidate_ids"]), [f"c{i:02d}" for i in range(12)])
            norms = np.linalg.norm(archive["candidate_embeddings"], axis=1)
            np.testing.assert_allclose(norms, np.ones(12), rtol=1e-5)

        # Evaluation report carries the analysis sections.
        evaluation = json.loads((self.output_dir / "evaluation_report.json").read_text(encoding="utf-8"))
        self.assertIn("score_distributions", evaluation["category_statistics"])
        self.assertIn("runtime_seconds", evaluation)
        self.assertEqual(len(evaluation["category_statistics"]["score_distributions"]), len(CATEGORIES))
        # Strategy comparison runs by default (skip flag not passed).
        self.assertIsNotNone(evaluation.get("strategy_comparison"))

        # Markdown has a section per category
        top10 = (self.output_dir / "top10_per_category.md").read_text(encoding="utf-8")
        for category in CATEGORIES:
            self.assertIn(f"## {category.category_id}", top10)

    def test_strategy_comparison_detected_when_enabled(self):
        self.assertEqual(self._run([]), 0)
        evaluation = json.loads((self.output_dir / "evaluation_report.json").read_text(encoding="utf-8"))
        comparison = evaluation.get("strategy_comparison")
        self.assertIsNotNone(comparison)
        self.assertIn("engine_signature", comparison)
        agreement = comparison["agreement_top_k"]
        self.assertEqual(len(agreement), len(CATEGORIES))
        for row in agreement:
            self.assertIn("jaccard_at_top10", row)
            # Offline token-overlap vectors are deterministic -> both strategies
            # embed identical formatted text? No: document formatting differs
            # per strategy, so sets may differ; only assert well-formed values.
            self.assertGreaterEqual(row["jaccard_at_top10"], 0.0)
            self.assertLessEqual(row["jaccard_at_top10"], 1.0)

    def test_manual_labels_attached_when_present(self):
        labels = {"betrayal": {f"c{i:02d}": ("relevant" if i < 3 else "not") for i in range(5)}}
        labels_path = self.tmp_root / "labels.json"
        labels_path.write_text(json.dumps(labels), encoding="utf-8")
        exit_code = cr.main([
            "--candidates", str(self.candidates_path),
            "--output", str(self.output_dir),
            "--manual-labels", str(labels_path),
            "--offline", "--limit-candidates", "12",
        ])
        self.assertEqual(exit_code, 0)
        payload = json.loads((self.output_dir / "fast_discovery_results.json").read_text(encoding="utf-8"))
        manual_review = payload["category_statistics"].get("manual_review")
        self.assertIsNotNone(manual_review)
        self.assertIn("betrayal", manual_review["categories"])
        self.assertIn("precision_at_5", manual_review["categories"]["betrayal"]["precision"])

    def test_invalid_candidates_path_exits_two(self):
        exit_code = cr.main([
            "--candidates", str(self.tmp_root / "missing.json"),
            "--output", str(self.output_dir), "--offline",
        ])
        self.assertEqual(exit_code, 2)

    def test_limit_candidates_bounds_enforced(self):
        exit_code = cr.main([
            "--candidates", str(self.candidates_path),
            "--output", str(self.output_dir), "--offline", "--limit-candidates", "0",
        ])
        self.assertEqual(exit_code, 2)


class CacheReuseAcrossRunsTests(unittest.TestCase):
    """Second CLI run against the same output dir must not re-encode anything."""

    def test_second_run_hits_cache(self):
        tmp_root = Path(tempfile.mkdtemp(prefix="kuku_cache_", dir="/tmp/opencode"))
        try:
            candidates_path = tmp_root / "candidates.json"
            corpus = [make_candidate(f"c{i}", batch=i) for i in range(4)]
            candidates_path.write_text(json.dumps({"candidates": corpus}), encoding="utf-8")
            output_dir = tmp_root / "retrieval_output"
            base_args = ["--candidates", str(candidates_path), "--output", str(output_dir),
                         "--offline", "--skip-strategy-comparison", "--skip-recall-sweep",
                         "--manual-labels", str(tmp_root / "none.json")]
            self.assertEqual(cr.main(base_args), 0)
            first_cache = json.loads((output_dir / "fast_discovery_results.json").read_text(encoding="utf-8"))["cache"]
            self.assertEqual(cr.main(base_args), 0)
            second_cache = json.loads((output_dir / "fast_discovery_results.json").read_text(encoding="utf-8"))["cache"]
            self.assertEqual(first_cache["documents"]["encoded_now"], 4)
            self.assertEqual(second_cache["documents"]["encoded_now"], 0)
            self.assertEqual(second_cache["categories"]["encoded_now"], 0)
            self.assertEqual(second_cache["documents"]["cache_hits"], 4)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
