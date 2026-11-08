#!/usr/bin/env python3
"""Tests for the Kuku Stage 4 LLM judge (judge_schema / judge_prompt / llm_judge).

No network access and no real LLM needed: the orchestrator tests use either
the OfflineScriptedJudge or tiny in-process scripted providers. The staging
llama.cpp server is NOT exercised here.
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

import judge_prompt
import judge_schema as js
import llm_judge as lj

TMP = Path(tempfile.gettempdir()) / "kuku_judge_tests"


def make_candidate(candidate_id: str, *, episode: int = 1, arc_seed: str = "x") -> dict:
    """Minimal candidate matching context_builder.py's v1.0 schema surface."""
    dialogue = [
        {"event_id": f"{candidate_id}_d1", "event_type": "dialogue", "scene_number": 1,
         "start_seconds": 10.0, "end_seconds": 12.0, "start_timestamp": "00:10.00",
         "content": "What do you mean?", "speaker": "LADY", "speaker_canonical": "LADY",
         "delivery": "Trembling"},
        {"event_id": f"{candidate_id}_d2", "event_type": "dialogue", "scene_number": 1,
         "start_seconds": 13.0, "end_seconds": 15.0, "start_timestamp": "00:13.00",
         "content": "Because he is marrying me today.", "speaker": "CLARISSA",
         "speaker_canonical": "CLARISSA", "delivery": "Gloating"},
        {"event_id": f"{candidate_id}_d3", "event_type": "dialogue", "scene_number": 1,
         "start_seconds": 16.0, "end_seconds": 18.0, "start_timestamp": "00:16.00",
         "content": "Why are you holding her hand?", "speaker": "LADY", "speaker_canonical": "LADY",
         "delivery": "Broken"},
    ]
    staging = [
        {"event_id": f"{candidate_id}_s1", "event_type": "camera", "scene_number": 1,
         "start_seconds": 9.0, "end_seconds": 11.0, "start_timestamp": "00:09.00",
         "content": "Close-up LADY."},
    ]
    preceding = [
        {"event_id": f"{candidate_id}_p1", "event_type": "dialogue", "scene_number": 1,
         "start_seconds": 3.0, "end_seconds": 5.0, "start_timestamp": "00:03.00",
         "content": "Are you ready?", "speaker": "MATRIARCH", "speaker_canonical": "MATRIARCH", "delivery": "Cold"},
    ]
    following = [
        {"event_id": f"{candidate_id}_f1", "event_type": "dialogue", "scene_number": 2,
         "start_seconds": 45.0, "end_seconds": 47.0, "start_timestamp": "00:45.00",
         "content": "Leave.", "speaker": "MATRIARCH", "speaker_canonical": "MATRIARCH", "delivery": "Firm"},
    ]
    source_ids = [item["event_id"] for item in dialogue] + [item["event_id"] for item in staging]
    context_ids = [item["event_id"] for item in preceding + following]
    return {
        "candidate_id": candidate_id,
        "episode_number": episode,
        "scene_numbers": [1],
        "start_seconds": 10.0,
        "end_seconds": 40.0,
        "duration_seconds": 30.0,
        "span_seconds": 30.0,
        "start_timestamp": "00:10.00",
        "end_timestamp": "00:40.00",
        "segment_count": 1,
        "dialogue_transcript": dialogue,
        "actions_and_camera": staging,
        "speakers": [
            {"speaker_canonical": "LADY", "deliveries": ["Trembling", "Broken"], "dialogue_event_count": 2},
            {"speaker_canonical": "CLARISSA", "deliveries": ["Gloating"], "dialogue_event_count": 1},
        ],
        "preceding_context": preceding,
        "following_context": following,
        "untimestamped_events": [],
        "dialogue_event_count": len(dialogue),
        "source_event_ids": source_ids,
        "context_event_ids": context_ids,
        "crosses_scene_boundary": False,
        "scenes": [{"scene_number": 1, "heading": "INT. CATHEDRAL – DAY", "start_seconds": 0.0, "end_seconds": 138.0}],
        "episode": {"episode_number": episode, "episode_label": f"Episode {episode}",
                    "logline": "A wedding interrupted by a prior claim on the groom.",
                    "continuity_notes": "First episode; establishes abandonment.",
                    "runtime_seconds": 138.0, "scene_count": 1},
    }


def valid_judgement(candidate: dict, verdict: str = "accept") -> dict:
    ids = [item["event_id"] for item in candidate["dialogue_transcript"]]
    scores = {
        "opening_hook": 4,
        "cliffhanger": 5,
        "standalone_clarity": 3,
        "narrative_progression": 4,
        "emotional_impact": 4,
    }
    evidence = [
        {"dimension": dimension, "statement": f"evidence for {dimension}", "event_ids": [ids[i % len(ids)]]}
        for i, dimension in enumerate(js.DIMENSIONS)
    ]
    return {
        "scores": scores,
        "verdict": verdict,
        "confidence": 0.85,
        "observed_arc": "altar_abandonment",
        "evidence": evidence,
        "rejection_reasons": [] if verdict == "accept" else ["weak_opening"],
        "boundary_feedback": None,
        "notes": "Compact but dramatic wedding-day detonation.",
    }


class SchemaValidationTests(unittest.TestCase):
    def setUp(self):
        self.candidate = make_candidate("c-0001")

    def test_valid_judgement_passes(self):
        judgement = js.validate_judgement(valid_judgement(self.candidate), self.candidate)
        self.assertEqual(judgement["verdict"], "accept")
        self.assertEqual(judgement["scores"]["opening_hook"], 4)
        self.assertEqual(set(judgement), {
            "scores", "verdict", "confidence", "observed_arc",
            "evidence", "rejection_reasons", "boundary_feedback", "notes",
        })

    def test_weighted_score_math(self):
        judgement = valid_judgement(self.candidate)
        weighted = js.weighted_score(judgement["scores"])
        expected = 4 * 0.25 + 5 * 0.25 + 3 * 0.20 + 4 * 0.15 + 4 * 0.15
        self.assertAlmostEqual(weighted, expected, places=6)
        self.assertEqual(js.weighted_score({d: 5 for d in js.DIMENSIONS}), 5.0)
        self.assertEqual(js.weighted_score({d: 0 for d in js.DIMENSIONS}), 0.0)

    def test_all_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(js.DIMENSION_WEIGHTS.values()), 1.0, places=9)
        self.assertEqual(set(js.DIMENSION_WEIGHTS), set(js.DIMENSIONS))

    def test_score_out_of_range_rejected(self):
        judgement = valid_judgement(self.candidate)
        judgement["scores"]["opening_hook"] = 6
        with self.assertRaises(js.JudgementValidationError) as ctx:
            js.validate_judgement(judgement, self.candidate)
        self.assertIn("out of range", "; ".join(ctx.exception.errors))

    def test_score_bool_rejected(self):
        judgement = valid_judgement(self.candidate)
        judgement["scores"]["opening_hook"] = True  # bool is subclass of int
        with self.assertRaises(js.JudgementValidationError):
            js.validate_judgement(judgement, self.candidate)

    def test_missing_dimension_rejected(self):
        judgement = valid_judgement(self.candidate)
        del judgement["scores"]["emotional_impact"]
        with self.assertRaises(js.JudgementValidationError):
            js.validate_judgement(judgement, self.candidate)

    def test_missing_evidence_for_dimension_rejected(self):
        judgement = valid_judgement(self.candidate)
        judgement["evidence"] = [e for e in judgement["evidence"] if e["dimension"] != "emotional_impact"]
        with self.assertRaises(js.JudgementValidationError) as ctx:
            js.validate_judgement(judgement, self.candidate)
        self.assertIn("emotional_impact", "; ".join(ctx.exception.errors))

    def test_invented_event_id_rejected(self):
        judgement = valid_judgement(self.candidate)
        judgement["evidence"][0]["event_ids"] = ["ep99_not_a_real_event"]
        with self.assertRaises(js.JudgementValidationError) as ctx:
            js.validate_judgement(judgement, self.candidate)
        self.assertIn("unknown id", "; ".join(ctx.exception.errors))

    def test_accept_with_rejection_reasons_rejected(self):
        judgement = valid_judgement(self.candidate, "accept")
        judgement["rejection_reasons"] = ["weak_opening"]
        with self.assertRaises(js.JudgementValidationError):
            js.validate_judgement(judgement, self.candidate)

    def test_reject_without_rejection_reasons_rejected(self):
        judgement = valid_judgement(self.candidate, "reject")
        judgement["rejection_reasons"] = []
        with self.assertRaises(js.JudgementValidationError):
            js.validate_judgement(judgement, self.candidate)

    def test_unknown_top_level_key_rejected(self):
        judgement = valid_judgement(self.candidate)
        judgement["reasoning"] = "the model thought about it"
        with self.assertRaises(js.JudgementValidationError) as ctx:
            js.validate_judgement(judgement, self.candidate)
        self.assertIn("unknown top-level keys", "; ".join(ctx.exception.errors))

    def test_confidence_range(self):
        judgement = valid_judgement(self.candidate)
        judgement["confidence"] = 1.5
        with self.assertRaises(js.JudgementValidationError):
            js.validate_judgement(judgement, self.candidate)

    def test_boundary_feedback_ordered_and_known(self):
        judgement = valid_judgement(self.candidate)
        judgement["boundary_feedback"] = {
            "start_event_id": self.candidate["dialogue_transcript"][1]["event_id"],
            "end_event_id": self.candidate["dialogue_transcript"][0]["event_id"],  # backwards
            "rationale": "Cut tighter around the turn.",
        }
        with self.assertRaises(js.JudgementValidationError) as ctx:
            js.validate_judgement(judgement, self.candidate)
        self.assertIn("not ordered", "; ".join(ctx.exception.errors))

    def test_boundary_feedback_unknown_id_rejected(self):
        judgement = valid_judgement(self.candidate)
        judgement["boundary_feedback"] = {
            "start_event_id": "invented_id_1",
            "end_event_id": self.candidate["dialogue_transcript"][0]["event_id"],
            "rationale": "Tighter cut.",
        }
        with self.assertRaises(js.JudgementValidationError):
            js.validate_judgement(judgement, self.candidate)

    def test_attach_weighted_score_idempotent_copy(self):
        judgement = js.validate_judgement(valid_judgement(self.candidate), self.candidate)
        out = js.attach_weighted_score(judgement)
        self.assertIn("weighted_score", out)
        self.assertNotIn("weighted_score", judgement)  # input not mutated


class ExtractJsonTests(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(lj.extract_json_object('{"a": 1}'), {"a": 1})

    def test_fenced(self):
        self.assertEqual(lj.extract_json_object('```json\n{"a": 1}\n```'), {"a": 1})

    def test_leading_and_trailing_chatter(self):
        self.assertEqual(lj.extract_json_object('thinking out loud {"a": 2} trailing'), {"a": 2})

    def test_no_json_raises(self):
        with self.assertRaises(ValueError):
            lj.extract_json_object("no braces here")

    def test_non_object_raises(self):
        with self.assertRaises(ValueError):
            lj.extract_json_object("[1, 2, 3]")


class PromptContractTests(unittest.TestCase):
    def setUp(self):
        self.candidate = make_candidate("c-0002")

    def test_blind_payload_carries_no_retrieval_markers(self):
        body = judge_prompt.build_user_message(self.candidate, include_retrieval_metadata=False)
        self.assertNotIn("retrieval_metadata", body)
        self.assertNotIn("similarity_scores_all_categories", body)
        self.assertNotIn("best_categories", body)

    def test_opt_in_payload_marks_metadata(self):
        candidate = dict(self.candidate)
        candidate["best_categories"] = [{"category_id": "betrayal"}]
        body = judge_prompt.build_user_message(candidate, include_retrieval_metadata=True)
        self.assertIn("retrieval_metadata", body)
        self.assertIn("best_categories", body)
        self.assertIn("betrayal", body)

    def test_payload_includes_all_contract_sections(self):
        payload = judge_prompt.build_payload(self.candidate)
        for key in ("candidate_id", "episode_number", "episode_logline", "window",
                    "cast_in_window", "dialogue", "staging",
                    "narrative_preceding", "narrative_following"):
            self.assertIn(key, payload)
        self.assertTrue(payload["dialogue"], "dialogue anchors missing")
        self.assertTrue(all("event_id" in item for item in payload["dialogue"]))
        self.assertTrue(all("event_id" in item for item in payload["staging"]))
        # blind by default
        self.assertIs(payload.get("retrieval_metadata"), None)

    def test_blind_payload_deterministic(self):
        first = judge_prompt.build_user_message(self.candidate)
        second = judge_prompt.build_user_message(self.candidate)
        self.assertEqual(first, second)

    def test_system_prompt_states_the_hard_rules(self):
        for phrase in ("Never invent", "Betrayal", "Standalone clarity", "cliffhanger",
                       "observed_arc", "ONLY this JSON object"):
            self.assertIn(phrase, judge_prompt.SYSTEM_PROMPT)


class OfflineJudgeTests(unittest.TestCase):
    def setUp(self):
        self.candidate = make_candidate("c-0003")

    def test_heuristic_is_deterministic_and_valid(self):
        provider = lj.OfflineScriptedJudge("heuristic")
        first = provider.complete_messages(
            [{"role": "user", "content": judge_prompt.build_user_message(self.candidate)}],
            max_output_tokens=800, temperature=0.0,
        )
        second = provider.complete_messages(
            [{"role": "user", "content": judge_prompt.build_user_message(self.candidate)}],
            max_output_tokens=800, temperature=0.0,
        )
        self.assertEqual(first.text, second.text)
        decoded = lj.extract_json_object(first.text)
        js.validate_judgement(decoded, self.candidate)

    def test_reject_all_mode(self):
        provider = lj.OfflineScriptedJudge("reject_all")
        completion = provider.complete_messages(
            [{"role": "user", "content": judge_prompt.build_user_message(self.candidate)}],
            max_output_tokens=800, temperature=0.0,
        )
        decoded = lj.extract_json_object(completion.text)
        self.assertEqual(decoded["verdict"], "reject")
        self.assertTrue(decoded["rejection_reasons"])


class ScriptedProvider(lj.OfflineScriptedJudge):
    """Plays back a queue of raw completion texts verbatim."""

    def __init__(self, script: list[str]):
        super().__init__("heuristic")
        self.script = list(script)
        self.calls = 0
        self.last_messages: list[dict] = []

    def complete_messages(self, messages, *, max_output_tokens, temperature):
        self.calls += 1
        self.last_messages = list(messages)
        text = self.script[min(self.calls - 1, len(self.script) - 1)]
        return lj.LlmCompletion(text=text, prompt_tokens=100, completion_tokens=50,
                                latency_seconds=0.01, raw={"provider": "scripted"})


class JudgeOneTests(unittest.TestCase):
    def setUp(self):
        self.candidate = make_candidate("c-0004")
        self.config = lj.JudgeConfig()

    def test_valid_first_attempt(self):
        provider = ScriptedProvider([json.dumps(valid_judgement(self.candidate))])
        row = lj.judge_one(self.candidate, provider, self.config)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["attempts"][0]["ok"], True)
        self.assertEqual(row["usage"]["prompt_tokens"], 100)

    def test_repair_loop_recovers(self):
        invalid = json.dumps({"scores": {"opening_hook": 4}})
        provider = ScriptedProvider([invalid, json.dumps(valid_judgement(self.candidate))])
        row = lj.judge_one(self.candidate, provider, self.config)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(len(row["attempts"]), 2)
        self.assertEqual(row["attempts"][0]["ok"], False)
        self.assertIn("Return corrected JSON only", str(provider.last_messages[-1]["content"]))

    def test_permanent_failure_marks_failed(self):
        provider = ScriptedProvider(["not json at all"] * 5)
        row = lj.judge_one(self.candidate, provider, self.config)
        self.assertEqual(row["status"], "failed")
        self.assertEqual(len(row["attempts"]), self.config.repair_attempts + 1)
        self.assertIsNone(row["judgement"])

    def test_usage_accumulates_across_attempts(self):
        provider = ScriptedProvider(["garbage {", json.dumps(valid_judgement(self.candidate))])
        row = lj.judge_one(self.candidate, provider, self.config)
        self.assertEqual(row["usage"]["prompt_tokens"], 200)
        self.assertEqual(row["usage"]["completion_tokens"], 100)


class CliEndToEndTests(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(TMP, ignore_errors=True)
        TMP.mkdir(parents=True)
        corpus = [make_candidate("cli-0001"), make_candidate("cli-0002", episode=2)]
        (TMP / "candidates.json").write_text(
            json.dumps({"candidates": corpus}), encoding="utf-8"
        )
        retrieval = {
            "schema_version": "1.1",
            "results": [
                {"candidate_id": "cli-0001", "similarity_scores_all_categories": {"betrayal": 0.66, "suspense": 0.61}},
                {"candidate_id": "cli-0002", "similarity_scores_all_categories": {"betrayal": 0.59, "suspense": 0.70}},
            ],
        }
        (TMP / "fast_discovery_results.json").write_text(json.dumps(retrieval), encoding="utf-8")
        self.output = TMP / "judge_output"

    def tearDown(self):
        shutil.rmtree(TMP, ignore_errors=True)

    def _run(self, extra=()):
        return lj.main([
            "--retrieval-results", str(TMP / "fast_discovery_results.json"),
            "--candidates", str(TMP / "candidates.json"),
            "--output", str(self.output),
            "--offline",
            *extra,
        ])

    def test_offline_run_produces_contract(self):
        self.assertEqual(self._run(), 0)
        for name in ("judge_results.json", "judge_results.jsonl", "judge_report.json", "judge_cache.jsonl"):
            self.assertTrue((self.output / name).exists(), msg=f"missing {name}")
        payload = json.loads((self.output / "judge_results.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["totals"]["candidates_in_run"], 2)
        self.assertEqual(len(payload["results"]), 2)
        self.assertEqual(payload["prompt_version"], judge_prompt.JUDGE_PROMPT_VERSION)
        for row in payload["results"]:
            self.assertEqual(row["status"], "ok")
            self.assertIsNotNone(row["judgement"]["weighted_score"])

    def test_cache_serves_second_run(self):
        self.assertEqual(self._run(), 0)
        self.assertEqual(self._run(), 0)
        payload = json.loads((self.output / "judge_results.json").read_text(encoding="utf-8"))
        self.assertTrue(all(row["cache_hit"] for row in payload["results"]))
        self.assertEqual(payload["totals"]["judged_now"], 0)

    def test_retrieval_union_missing_candidate_is_an_error(self):
        retrieval = {"results": [{"candidate_id": "nope", "similarity_scores_all_categories": {}}]}
        (TMP / "fast_discovery_results.json").write_text(json.dumps(retrieval), encoding="utf-8")
        self.assertEqual(self._run(), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
