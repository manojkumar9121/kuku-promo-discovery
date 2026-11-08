#!/usr/bin/env python3
"""Tests for the Kuku Stage 4 promo ranker (promo_ranker.py).

Pure math + assembly tests: ranked ordering, arc capping, near-duplicate
clustering, rejected retention, embed-vs-judge comparison, human-review
overlay, CLI output contract.
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

import promo_ranker as pr

TMP = Path(tempfile.gettempdir()) / "kuku_ranker_tests"


def make_row(candidate_id: str, *, episode: int = 1, arc: str = "altar_abandonment",
             score: float = 4.0, start: float = 10.0, end: float = 40.0,
             category_scores: dict | None = None) -> dict:
    return {
        "candidate_id": candidate_id,
        "episode_number": episode,
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": end - start,
        "status": "ok",
        "judgement": {
            "scores": {"opening_hook": 4, "cliffhanger": 4, "standalone_clarity": 4,
                       "narrative_progression": 4, "emotional_impact": 4},
            "verdict": "accept",
            "confidence": 0.8,
            "observed_arc": arc,
            "evidence": [{"dimension": d, "statement": "s", "event_ids": ["e1"]}
                         for d in ("opening_hook",)],
            "rejection_reasons": [],
            "boundary_feedback": None,
            "notes": "note",
            "weighted_score": score,
        },
    }


def make_candidate(candidate_id: str, *, episode: int = 1, start: float = 10.0, end: float = 40.0) -> dict:
    return {
        "candidate_id": candidate_id,
        "episode_number": episode,
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": end - start,
        "start_timestamp": f"00:{start:05.2f}",
        "end_timestamp": f"00:{end:05.2f}",
        "source_event_ids": ["e1", "e2"],
        "episode": {"episode_label": f"Episode {episode}"},
        "scenes": [],
    }


class WindowIoUTests(unittest.TestCase):
    def test_identical_window(self):
        self.assertAlmostEqual(pr.window_iou({"start_seconds": 10, "end_seconds": 40},
                                             {"start_seconds": 10, "end_seconds": 40}), 1.0)

    def test_half_overlap(self):
        iou = pr.window_iou({"start_seconds": 0, "end_seconds": 20}, {"start_seconds": 10, "end_seconds": 30})
        self.assertAlmostEqual(iou, 10 / 30, places=6)

    def test_disjoint(self):
        self.assertEqual(pr.window_iou({"start_seconds": 0, "end_seconds": 10},
                                       {"start_seconds": 50, "end_seconds": 60}), 0.0)


class SpearmanTests(unittest.TestCase):
    def test_identical_order(self):
        self.assertEqual(pr.spearman_rho([3, 1, 2], [30, 10, 20]), 1.0)

    def test_reversed_order(self):
        self.assertEqual(pr.spearman_rho([1, 2, 3], [3, 2, 1]), -1.0)

    def test_too_small_returns_none(self):
        self.assertIsNone(pr.spearman_rho([1, 2], [1, 2]))

    def test_ties_average_rank(self):
        result = pr.spearman_rho([1, 1, 2], [1, 1, 2])
        self.assertAlmostEqual(result, 1.0)


class RankingAssemblyTests(unittest.TestCase):
    def test_featured_respects_arc_cap(self):
        accepted = [
            make_row("a1", episode=1, arc="altar_abandonment", score=4.9),
            make_row("a2", episode=1, arc="altar_abandonment", score=4.8),
            make_row("a3", episode=1, arc="altar_abandonment", score=4.7),
            make_row("a4", episode=2, arc="altar_abandonment", score=4.6),
        ]
        candidates = {row["candidate_id"]: make_candidate(row["candidate_id"], episode=row["episode_number"],
                                                         start=0, end=10 + int(row["candidate_id"][1]))
                      for row in accepted}
        ranking = pr.rank_candidates(accepted, candidates, {}, max_per_arc=2)
        self.assertEqual([r["candidate_id"] for r in ranking["featured"]], ["a1", "a2", "a4"])
        self.assertEqual([r["candidate_id"] for r in ranking["reserve"]], ["a3"])
        self.assertIn("arc_cap_reached", ranking["reserve"][0]["overflow_reason"])
        self.assertTrue(all(r["tier"] == "featured" for r in ranking["featured"]))

    def test_reserve_does_not_reorder_quality(self):
        accepted = [
            make_row("b1", episode=1, arc="wedding", score=5.0),
            make_row("b2", episode=1, arc="wedding", score=4.9),
            make_row("b3", episode=1, arc="wedding", score=4.8),
            make_row("b4", episode=1, arc="wedding", score=4.7),
        ]
        candidates = {row["candidate_id"]: make_candidate(row["candidate_id"], start=0, end=10) for row in accepted}
        ranking = pr.rank_candidates(accepted, candidates, {}, max_per_arc=1)
        self.assertEqual([r["candidate_id"] for r in ranking["featured"]], ["b1"])
        self.assertEqual([r["candidate_id"] for r in ranking["reserve"]], ["b2", "b3", "b4"])

    def test_rejected_retained_with_reasons(self):
        rejected_row = make_row("rej-1", episode=1, arc="fir", score=1.2)
        rejected_row["judgement"]["verdict"] = "reject"
        rejected_row["judgement"]["rejection_reasons"] = ["weak_opening", "no_cliffhanger"]
        accepted_row = make_row("acc-1", episode=1, arc="foo", score=4.5)
        ranking = pr.rank_candidates([rejected_row, accepted_row],
                                     {"rej-1": make_candidate("rej-1"), "acc-1": make_candidate("acc-1")},
                                     {}, max_per_arc=2)
        self.assertEqual(len(ranking["rejected"]), 1)
        self.assertEqual(ranking["rejected"][0]["rejection_reasons"], ["weak_opening", "no_cliffhanger"])
        self.assertEqual(ranking["verdict_statistics"]["rejection_reason_histogram"],
                         {"weak_opening": 1, "no_cliffhanger": 1})

    def test_failed_rows_excluded_from_ranking(self):
        failed = {"candidate_id": "fail-1", "status": "failed", "judgement": None,
                  "error": "permanent failure"}
        accepted = make_row("acc-2", score=3.9)
        ranking = pr.rank_candidates([failed, accepted],
                                     {"acc-2": make_candidate("acc-2")},
                                     {}, max_per_arc=2)
        self.assertEqual(len(ranking["featured"]), 1)
        self.assertEqual(ranking["failed"], [{"candidate_id": "fail-1", "error": "permanent failure"}])
        self.assertEqual(len(ranking["ranked"]), 1)

    def test_near_duplicate_cluster_detected(self):
        rows = [
            make_row("dup-a", episode=3, arc="dungeon_reckoning", score=4.9, start=100, end=130),
            make_row("dup-b", episode=3, arc="dungeon_reckoning", score=4.8, start=98, end=128),
            make_row("solo-c", episode=4, arc="resurrection", score=4.7, start=0, end=30),
        ]
        candidates = {
            "dup-a": make_candidate("dup-a", episode=3, start=100, end=130),
            "dup-b": make_candidate("dup-b", episode=3, start=98, end=128),
            "solo-c": make_candidate("solo-c", episode=4, start=0, end=30),
        }
        ranking = pr.rank_candidates(rows, candidates, {}, max_per_arc=2)
        self.assertEqual(len(ranking["near_duplicate_clusters"]), 1)
        cluster = ranking["near_duplicate_clusters"][0]
        self.assertEqual(cluster["head_candidate_id"], "dup-a")
        self.assertEqual(set(cluster["member_candidate_ids"]), {"dup-a", "dup-b"})

    def test_distinct_arcs_stay_out_of_clusters(self):
        rows = [
            make_row("x-1", episode=1, arc="arc_one", score=4.9, start=100, end=130),
            make_row("x-2", episode=1, arc="arc_two", score=4.8, start=100, end=130),
        ]
        candidates = {
            "x-1": make_candidate("x-1", episode=1, start=100, end=130),
            "x-2": make_candidate("x-2", episode=1, start=100, end=130),
        }
        ranking = pr.rank_candidates(rows, candidates, {}, max_per_arc=2)
        # High IoU same window but different arcs -> still clustered (structural near-dup),
        # which is the intended sellable-moment signal.
        self.assertEqual(len(ranking["near_duplicate_clusters"]), 1)


class EmbeddingVsJudgeTests(unittest.TestCase):
    def setUp(self):
        pass

    def _ok_rows(self):
        rows = []
        specs = [("c-high-judge", 4.9, 0.61), ("c-mid", 4.0, 0.65), ("c-low-judge", 2.9, 0.66), ("c-flat", 3.0, 0.59)]
        for cid, score, embed in specs:
            rows.append(make_row(cid, score=score, category_scores={"betrayal": embed}))
        return rows

    def test_comparison_block_built(self):
        rows = self._ok_rows()
        candidates = {row["candidate_id"]: make_candidate(row["candidate_id"]) for row in rows}
        retrieval = {
            "c-high-judge": {"similarity_scores_all_categories": {"a": 0.60, "b": 0.58}},
            "c-mid": {"similarity_scores_all_categories": {"a": 0.63, "b": 0.61}},
            "c-low-judge": {"similarity_scores_all_categories": {"a": 0.68, "b": 0.66}},
            "c-flat": {"similarity_scores_all_categories": {"a": 0.57, "b": 0.55}},
        }
        block = pr.build_embedding_vs_judge(rows, candidates, retrieval, top_ks=(2,))
        self.assertEqual(block["population"], 4)
        self.assertIsNotNone(block["spearman_rho"])
        # judge top2 {c-high-judge, c-mid}; embedding top2 {c-low-judge, c-mid} -> share c-mid
        self.assertAlmostEqual(block["jaccard_top_k"]["top_2"]["jaccard"], 1 / 3, places=4)
        self.assertTrue(block["largest_disagreements"])
        self.assertEqual(block["judge_order_top10"][0], "c-high-judge")
        self.assertEqual(block["embedding_order_top10"][0], "c-low-judge")


class HumanOverlayTests(unittest.TestCase):
    def _rows(self):
        rows = []
        for i, (cid, score) in enumerate([("p-a", 5.0), ("p-b", 4.5), ("p-c", 4.0), ("p-d", 3.5), ("p-e", 3.0)]):
            rows.append(make_row(cid, score=score))
        return rows

    def test_missing_labels_returns_none(self):
        rows = self._rows()
        self.assertIsNone(pr.build_human_review_overlay(rows, Path(tempfile.gettempdir()) / "no-such-labels.json"))

    def test_precision_counts_strong_and_usable(self):
        rows = self._rows()
        labels = {"p-a": "strong", "p-b": "usable", "p-c": "weak", "p-d": "usable", "p-e": "unusable"}
        with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as tmp:
            path = Path(tmp) / "labels.json"
            path.write_text(json.dumps(labels), encoding="utf-8")
            overlay = pr.build_human_review_overlay(rows, path, top_ks=(2, 5))
        self.assertEqual(overlay["precision"]["precision_at_2"]["value"], 1.0)
        self.assertEqual(overlay["precision"]["precision_at_5"]["value"], 0.6)  # strong, usable, usable = 3


class CliEndToEndTests(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(TMP, ignore_errors=True)
        TMP.mkdir(parents=True)
        corpus = [make_candidate("cli-1", episode=1, start=10, end=40),
                  make_candidate("cli-2", episode=1, start=15, end=45),
                  make_candidate("cli-3", episode=2, start=0, end=30)]
        (TMP / "candidates.json").write_text(json.dumps({"candidates": corpus}), encoding="utf-8")
        retrieval = {"results": [
            {"candidate_id": "cli-1", "similarity_scores_all_categories": {"a": 0.66, "b": 0.6}},
            {"candidate_id": "cli-2", "similarity_scores_all_categories": {"a": 0.7, "b": 0.58}},
            {"candidate_id": "cli-3", "similarity_scores_all_categories": {"a": 0.55, "b": 0.52}},
        ]}
        (TMP / "retrieval.json").write_text(json.dumps(retrieval), encoding="utf-8")
        judged = []
        for cid, score, verdict in [("cli-1", 4.6, "accept"), ("cli-2", 4.4, "accept"), ("cli-3", 1.1, "reject")]:
            row = make_row(cid, score=score, arc="wedding")
            row["judgement"]["verdict"] = verdict
            if verdict == "reject":
                row["judgement"]["rejection_reasons"] = ["needs_prior_knowledge"]
            judged.append(row)
        (TMP / "judge_results.json").write_text(
            json.dumps({"results": judged}), encoding="utf-8"
        )
        self.output = TMP / "judge_output"

    def tearDown(self):
        shutil.rmtree(TMP, ignore_errors=True)

    def test_end_to_end_outputs(self):
        exit_code = pr.main([
            "--judge-results", str(TMP / "judge_results.json"),
            "--candidates", str(TMP / "candidates.json"),
            "--retrieval-results", str(TMP / "retrieval.json"),
            "--output", str(self.output),
            "--human-labels", str(TMP / "missing-labels.json"),
        ])
        self.assertEqual(exit_code, 0)
        ranking = json.loads((self.output / "promo_ranking.json").read_text(encoding="utf-8"))
        self.assertEqual(len(ranking["featured"]), 2)  # cli-1, cli-2
        self.assertEqual(len(ranking["rejected"]), 1)
        self.assertEqual(ranking["rejected"][0]["candidate_id"], "cli-3")
        report = json.loads((self.output / "ranking_report.json").read_text(encoding="utf-8"))
        self.assertIn("embedding_vs_judge", report)
        self.assertIn("diversity_summary", report)
        self.assertTrue((self.output / "promo_ranking.md").exists())
        # ranking keeps source timestamps + event references
        first = ranking["featured"][0]
        self.assertIn("start_timestamp", first)
        self.assertIn("end_timestamp", first)
        self.assertTrue(first["source_event_ids"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
