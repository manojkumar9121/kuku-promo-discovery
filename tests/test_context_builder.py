#!/usr/bin/env python3
"""Tests for the Kuku Context Builder.

Standard library only. Unit tests run on synthetic parsed-episode fixtures;
the integration tests run the real 50-episode output when it exists and
re-validate it through an independent recomputation of every clip's contents.
"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / 'src'))

import copy
import json
import unittest
from pathlib import Path

import context_builder as cb

REPO_DIR = Path(__file__).resolve().parent
PARSED_EPISODES = REPO_DIR / "parsed_output" / "parsed_episodes.json"
CANDIDATES_JSON = REPO_DIR / "context_output" / "candidates.json"
CANDIDATES_JSONL = REPO_DIR / "context_output" / "candidates.jsonl"
PACK_ZIP = REPO_DIR / "I Married a Broken Soldier. He was a Duke-scripts (2).zip"


def _event(event_id, scene, start, end, event_type="camera", content="", **extra):
    event = {
        "event_id": event_id,
        "episode_number": 1,
        "scene_number": scene,
        "event_index_in_scene": 0,
        "event_type": event_type,
        "start_seconds": start,
        "end_seconds": end,
        "start_timestamp": f"{int((start or 0) // 60):02d}:{(start or 0) % 60:05.2f}" if start is not None else None,
        "end_timestamp": f"{int((end or 0) // 60):02d}:{(end or 0) % 60:05.2f}" if end is not None else None,
        "content": content,
        "speaker": None,
        "speaker_canonical": None,
        "speaker_alias_source": None,
        "delivery": None,
        "voice_direction": None,
        "raw_header": None,
        "raw_block": None,
        "source_file": "fixture.txt",
        "source_line_start": 0,
        "source_order_index": 0,
        "scene_occurrence_index": 1,
    }
    event.update(extra)
    return event


def _make_episode(events, scene_bounds=((0.0, 200.0),), episode_number=1, logline="L", notes="C"):
    """Build a parsed-episode-shaped dict from event specs and scene bounds."""
    scenes = []
    counter = 0
    for index, (start, end) in enumerate(scene_bounds, start=1):
        scene_events = []
        for event in [e for e in events if e["scene_number"] == index]:
            counter += 1
            event["event_index_in_scene"] = len(scene_events) + 1
            event["source_order_index"] = counter
            scene_events.append(event)
        scenes.append({
            "scene_number": index,
            "scene_occurrence_index": 1,
            "heading": f"SCENE {index}",
            "start_seconds": start,
            "end_seconds": end,
            "start_timestamp": f"{int(start // 60):02d}:{start % 60:05.2f}",
            "end_timestamp": f"{int(end // 60):02d}:{end % 60:05.2f}",
            "events": scene_events,
        })
    return {
        "episode_number": episode_number,
        "episode_label": f"Episode {episode_number}",
        "logline": logline,
        "continuity_notes": notes,
        "source_file": "fixture.txt",
        "scenes": scenes,
        "warnings": [],
    }


def _dialogue(event_id, scene, start, end, content="Line.", speaker="S"):
    return _event(event_id, scene, start, end, "dialogue", content,
                  speaker=speaker, speaker_canonical=speaker, delivery="Calm", voice_direction=[])


class ConfigTests(unittest.TestCase):
    def test_default_config_is_valid(self):
        cb.BuilderConfig().validated()

    def test_min_ge_max_rejected(self):
        with self.assertRaises(ValueError):
            cb.BuilderConfig(min_window_seconds=60, max_window_seconds=20).validated()

    def test_target_outside_bounds_rejected(self):
        with self.assertRaises(ValueError):
            cb.BuilderConfig(target_window_seconds=100).validated()

    def test_unknown_anchor_type_rejected(self):
        with self.assertRaises(ValueError):
            cb.BuilderConfig(anchor_event_types=("dialogue", "whoosh")).validated()

    def test_empty_anchor_types_rejected(self):
        with self.assertRaises(ValueError):
            cb.BuilderConfig(anchor_event_types=()).validated()

    def test_negative_context_rejected(self):
        with self.assertRaises(ValueError):
            cb.BuilderConfig(context_before_seconds=-1).validated()

    def test_bad_iou_rejected(self):
        with self.assertRaises(ValueError):
            cb.BuilderConfig(dedup_iou_threshold=1.5).validated()


class WindowGenerationTests(unittest.TestCase):
    def setUp(self):
        events = [
            _event("e1", 1, 0.0, 10.0, "camera", "Establishing."),
            _dialogue("e2", 1, 0.0, 20.0, "Opening line.", "HERO"),
            _event("e3", 1, 5.0, 25.0, "action", "Reacting."),
            _dialogue("e4", 1, 25.0, 30.0, "Second line.", "ALLY"),
            _dialogue("e5", 1, 40.0, 60.0, "Third line.", "VILLAIN"),
            _dialogue("e6", 1, 70.0, 90.0, "Fourth line.", "ALLY"),
        ]
        self.episode = _make_episode(events, scene_bounds=((0.0, 200.0),))
        self.candidates, self.stats = cb.build_candidates_for_episode(self.episode, cb.BuilderConfig())
        # Expected windows from these boundaries: [0,30] (e2), [25,60] (e4),
        # [40,60] dropped as near-dup of [25,60], [70,90] (e6).

    def test_all_candidates_within_configured_bounds(self):
        self.assertEqual(len(self.candidates), 3)
        for candidate in self.candidates:
            self.assertGreaterEqual(candidate["duration_seconds"], 20.0 - 1e-6)
            self.assertLessEqual(candidate["duration_seconds"], 60.0 + 1e-6)

    def test_start_snaps_to_anchor_start(self):
        self.assertEqual(self.candidates[0]["start_seconds"], 0.0)
        self.assertEqual(self.candidates[0]["anchor_event_id"], "e2")
        self.assertEqual([c["start_seconds"] for c in self.candidates], [0.0, 25.0, 70.0])

    def test_end_snaps_to_event_boundary(self):
        ends = {event["end_seconds"] for event in self.episode["scenes"][0]["events"]}
        for candidate in self.candidates:
            self.assertIn(round(candidate["end_seconds"], 3), ends)

    def test_timestamps_reused_verbatim(self):
        by_id = {event["event_id"]: event for event in self.episode["scenes"][0]["events"]}
        for candidate in self.candidates:
            self.assertEqual(candidate["start_timestamp"], by_id[candidate["anchor_event_id"]]["start_timestamp"])
            self.assertEqual(candidate["end_timestamp"], by_id[candidate["boundary_event_ids"]["end_event_id"]]["end_timestamp"])

    def test_overlapping_events_retained(self):
        first = self.candidates[0]
        ids = set(first["source_event_ids"])
        # e1 (0-10) and e3 (5-25) overlap the [0,30] window although neither
        # boundary defines it; e2 anchors it.
        self.assertEqual({"e1", "e2", "e3", "e4"}, ids)

    def test_speaker_information_preserved(self):
        first = self.candidates[0]
        canonical = [speaker["speaker_canonical"] for speaker in first["speakers"]]
        self.assertEqual(canonical, ["HERO", "ALLY"])
        hero = first["speakers"][0]
        self.assertEqual(hero["deliveries"], ["Calm"])
        self.assertEqual(hero["dialogue_event_count"], 1)
        ally = first["speakers"][1]
        self.assertEqual(ally["dialogue_event_count"], 1)  # e4 ends exactly at window end

    def test_candidate_shows_scene_and_episode_metadata(self):
        first = self.candidates[0]
        self.assertEqual(first["scene_numbers"], [1])
        self.assertEqual(first["scenes"][0]["heading"], "SCENE 1")
        self.assertEqual(first["episode"]["logline"], "L")
        self.assertEqual(first["episode"]["continuity_notes"], "C")
        self.assertTrue(first["segments"][0]["start_event_id"] == "e2")

    def test_transcripts_split_by_type(self):
        first = self.candidates[0]
        self.assertEqual([item["event_id"] for item in first["dialogue_transcript"]], ["e2", "e4"])
        self.assertEqual([item["event_id"] for item in first["actions_and_camera"]], ["e1", "e3"])
        for item in first["dialogue_transcript"]:
            self.assertIn("delivery", item)
            self.assertTrue(item["content"].endswith("line."))

    def test_context_contains_no_clip_event_ids(self):
        for candidate in self.candidates:
            self.assertFalse(set(candidate["source_event_ids"]) & set(candidate["context_event_ids"]))

    def test_context_lies_strictly_outside_window(self):
        for candidate in self.candidates:
            for item in candidate["preceding_context"]:
                self.assertLessEqual(item["end_seconds"], candidate["start_seconds"] + cb.EPS)
            for item in candidate["following_context"]:
                self.assertGreaterEqual(item["start_seconds"], candidate["end_seconds"] - cb.EPS)

    def test_configurable_context_span_changes_context(self):
        tight_candidates, _ = cb.build_candidates_for_episode(
            self.episode, cb.BuilderConfig(context_before_seconds=4.0, context_after_seconds=4.0)
        )
        loose_candidates, _ = cb.build_candidates_for_episode(
            self.episode, cb.BuilderConfig(context_before_seconds=60.0, context_after_seconds=60.0)
        )
        tight_ids = sum(len(c["context_event_ids"]) for c in tight_candidates)
        loose_ids = sum(len(c["context_event_ids"]) for c in loose_candidates)
        self.assertLess(tight_ids, loose_ids)


class PartialOverlapTests(unittest.TestCase):
    def setUp(self):
        events = [
            _event("p", 1, 8.0, 12.0, "camera", "Started before the window."),
            _dialogue("a", 1, 10.0, 12.0, "Anchor.", "S"),
            _dialogue("b", 1, 30.0, 40.0, "Boundary line.", "S"),
            _event("r", 1, 38.0, 42.0, "camera", "Ends after the window."),
        ]
        self.episode = _make_episode(events, scene_bounds=((0.0, 100.0),))
        self.candidates, _ = cb.build_candidates_for_episode(self.episode, cb.BuilderConfig())
        # Only window: anchor "a" start 10 -> end 40 (event boundary of "b").

    def test_single_candidate(self):
        self.assertEqual(len(self.candidates), 1)
        self.assertAlmostEqual(self.candidates[0]["duration_seconds"], 30.0)

    def test_events_poking_out_are_kept_and_flagged(self):
        candidate = self.candidates[0]
        snapshots = {item["event_id"]: item
                     for item in candidate["dialogue_transcript"] + candidate["actions_and_camera"]}
        self.assertEqual(snapshots["p"]["boundary_overlap"], "start")
        self.assertFalse(snapshots["p"]["fully_contained"])
        self.assertEqual(snapshots["r"]["boundary_overlap"], "end")
        self.assertFalse(snapshots["r"]["fully_contained"])
        self.assertTrue(snapshots["a"]["fully_contained"])
        self.assertEqual(candidate["fully_contained_event_count"], 2)
        self.assertEqual(candidate["partial_overlap_event_count"], 2)


class DedupTests(unittest.TestCase):
    def test_near_duplicates_suppressed(self):
        events = [
            _dialogue("d1", 1, 0.0, 30.0, "Anchor one."),
            _dialogue("d2", 1, 1.0, 31.0, "Anchor two."),
            _dialogue("d3", 1, 2.0, 32.0, "Anchor three."),
            _dialogue("d4", 1, 60.0, 90.0, "Anchor four."),
            _dialogue("tail", 1, 150.0, 200.0, "Tail."),
        ]
        episode = _make_episode(events, scene_bounds=((0.0, 210.0),))
        candidates, stats = cb.build_candidates_for_episode(episode, cb.BuilderConfig())
        close_starts = [c for c in candidates if c["start_seconds"] < 10]
        self.assertEqual(len(close_starts), 1)  # anchors 0/1/2 collapse to one window
        self.assertEqual(stats["dropped_near_duplicate"], 2)

    def test_exact_duplicates_suppressed(self):
        events = [
            _dialogue("a1", 1, 10.0, 40.0, "Dialogue at x.", "S"),
            _event("a2", 1, 10.0, 40.0, "action", "Shares the exact interval."),
        ]
        episode = _make_episode(events, scene_bounds=((0.0, 100.0),))
        config = cb.BuilderConfig(anchor_event_types=("dialogue", "action"))
        _candidates, stats = cb.build_candidates_for_episode(episode, config)
        self.assertEqual(stats["dropped_exact_duplicate"], 1)

    def test_same_anchor_variants_exempt_from_cross_dedup(self):
        events = [
            _dialogue("d1", 1, 0.0, 30.0, "Anchor with two boundary choices."),
            _event("c1", 1, 1.0, 29.0, "camera", "Provides an early boundary."),
            _event("c2", 1, 1.0, 55.0, "camera", "Provides the target boundary."),
            _dialogue("tail", 1, 150.0, 200.0, "Tail."),
        ]
        episode = _make_episode(events, scene_bounds=((0.0, 210.0),))
        config = cb.BuilderConfig(max_windows_per_anchor=2, target_window_seconds=45.0)
        candidates, stats = cb.build_candidates_for_episode(episode, config)
        windows_from_anchor = sorted(
            (c for c in candidates if c["anchor_event_id"] == "d1"),
            key=lambda c: c["end_seconds"],
        )
        self.assertEqual(len(windows_from_anchor), 2)
        self.assertEqual([c["end_seconds"] for c in windows_from_anchor], [29.0, 55.0])
        self.assertEqual({c["start_seconds"] for c in windows_from_anchor}, {0.0})
        # Same-anchor variants must not be deduped against each other...
        self.assertEqual(stats["dropped_near_duplicate"], 0)


class EdgeCaseTests(unittest.TestCase):
    def test_no_boundary_in_range_is_counted(self):
        events = [_dialogue("only", 1, 0.0, 5.0, "Short episode.")]
        episode = _make_episode(events, scene_bounds=((0.0, 10.0),))
        candidates, stats = cb.build_candidates_for_episode(episode, cb.BuilderConfig())
        self.assertEqual(candidates, [])
        self.assertEqual(stats["anchors_without_boundary"], 1)

    def test_minimum_exact_boundary_allowed(self):
        events = [
            _dialogue("a", 1, 0.0, 2.0, "Start line."),
            _dialogue("b", 1, 18.0, 20.0, "End line."),
        ]
        episode = _make_episode(events, scene_bounds=((0.0, 30.0),))
        candidates, _ = cb.build_candidates_for_episode(episode, cb.BuilderConfig(target_window_seconds=20.0))
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0]["duration_seconds"], 20.0)
        self.assertEqual(candidates[0]["end_seconds"], 20.0)

    def test_untimestamped_events_listed_but_never_timelined(self):
        events = [
            _dialogue("a", 1, 0.0, 2.0, "Start."),
            _event("bad", 1, None, None, "camera", "Lost its time range."),
            _dialogue("b", 1, 40.0, 50.0, "End."),
        ]
        episode = _make_episode(events, scene_bounds=((0.0, 100.0),))
        candidates, stats = cb.build_candidates_for_episode(episode, cb.BuilderConfig(target_window_seconds=45.0))
        self.assertEqual(stats["untimestamped_events_in_episode"], 1)
        timeline_ids = {event_id for candidate in candidates for event_id in candidate["source_event_ids"]}
        self.assertNotIn("bad", timeline_ids)
        listed = {item["event_id"] for candidate in candidates
                  for item in candidate["untimestamped_events"]}
        self.assertIn("bad", listed)

    def test_episode_without_anchor_types_yields_no_candidates(self):
        events = [_event("c1", 1, 0.0, 40.0, "camera", "All camera.")]
        episode = _make_episode(events, scene_bounds=((0.0, 100.0),))
        candidates, _ = cb.build_candidates_for_episode(episode, cb.BuilderConfig())
        self.assertEqual(candidates, [])

    def test_scene_crossing_window_flagged_with_both_scenes(self):
        events = [
            _dialogue("a", 1, 45.0, 60.0, "Scene one."),
            _dialogue("b", 2, 100.0, 105.0, "Scene two."),
        ]
        episode = _make_episode(events, scene_bounds=((0.0, 100.0), (100.0, 200.0)))
        candidates, _ = cb.build_candidates_for_episode(episode, cb.BuilderConfig())
        crossing = [c for c in candidates if c["crosses_scene_boundary"]]
        self.assertTrue(crossing)
        self.assertEqual(crossing[0]["scene_numbers"], [1, 2])
        self.assertEqual(crossing[0]["scenes"][1]["heading"], "SCENE 2")

    def test_logline_and_continuity_notes_carried(self):
        events = [
            _dialogue("a", 1, 0.0, 2.0, "Line."),
            _dialogue("b", 1, 40.0, 50.0, "Line2."),
        ]
        episode = _make_episode(events, scene_bounds=((0.0, 100.0),), logline="TEST LOGLINE", notes="NOTES-1")
        candidates, _ = cb.build_candidates_for_episode(episode, cb.BuilderConfig(target_window_seconds=45.0))
        self.assertEqual(candidates[0]["episode"]["logline"], "TEST LOGLINE")
        self.assertEqual(candidates[0]["episode"]["continuity_notes"], "NOTES-1")


class MultiSegmentTests(unittest.TestCase):
    def make_episode(self, gap_bounds):
        events = [
            _dialogue("a", 1, 0.0, 2.0, "Open."),
            _event("gap", 1, *gap_bounds, "camera", "Around the gap."),
            _dialogue("b", 1, 40.0, 50.0, "Close."),
        ]
        return _make_episode(events, scene_bounds=((0.0, 100.0),))

    def _specs(self):
        return [
            {"start_seconds": 0.0, "end_seconds": 10.0, "start_timestamp": "00:00.00",
             "end_timestamp": "00:10.00", "start_event_id": "a", "end_event_id": "a"},
            {"start_seconds": 40.0, "end_seconds": 50.0, "start_timestamp": "00:40.00",
             "end_timestamp": "00:50.00", "start_event_id": "b", "end_event_id": "b"},
        ]

    def test_multi_segment_sums_durations_not_spans(self):
        episode = self.make_episode((9.5, 12.0))
        timeline, _ = cb.episode_timeline(episode)
        candidate = cb._assemble_candidate(episode, timeline, [], self._specs(), cb.BuilderConfig(), "seed")
        self.assertEqual(candidate["duration_seconds"], 20.0)  # (10-0) + (50-40)
        self.assertEqual(candidate["span_seconds"], 50.0)
        self.assertEqual(candidate["segment_count"], 2)
        self.assertEqual(candidate["duration_seconds"], sum(s["duration_seconds"] for s in candidate["segments"]))
        ids = set(candidate["source_event_ids"])
        self.assertIn("a", ids)
        self.assertIn("b", ids)
        self.assertIn("gap", ids)  # overlaps segment 1's tail (9.5-10.0)

    def test_gap_only_event_excluded_from_multi_segment(self):
        episode = self.make_episode((11.0, 15.0))
        timeline, _ = cb.episode_timeline(episode)
        candidate = cb._assemble_candidate(episode, timeline, [], self._specs(), cb.BuilderConfig(), "seed")
        self.assertNotIn("gap", candidate["source_event_ids"])
        self.assertNotIn("gap", [item["event_id"] for item in candidate["actions_and_camera"]])


class ValidationTests(unittest.TestCase):
    def setUp(self):
        events = [
            _dialogue("a", 1, 0.0, 2.0, "Open."),
            _dialogue("b", 1, 41.0, 50.0, "Later."),
            _event("c", 1, 60.0, 70.0, "camera", "After the window."),
        ]
        episode = _make_episode(events, scene_bounds=((0.0, 100.0),))
        self.episode = episode
        self.payload = cb.build_all_candidates([episode], cb.BuilderConfig(target_window_seconds=45.0))
        self.assertFalse(cb.validate_candidates(self.payload, [episode]))

    def test_duration_out_of_range_detected(self):
        payload = copy.deepcopy(self.payload)
        payload["candidates"][0]["end_seconds"] += 40.0
        payload["candidates"][0]["duration_seconds"] += 40.0
        errors = cb.validate_candidates(payload, [self.episode])
        self.assertTrue(any("outside" in error for error in errors))

    def test_missing_clip_event_detected(self):
        payload = copy.deepcopy(self.payload)
        payload["candidates"][0]["source_event_ids"].pop()
        errors = cb.validate_candidates(payload, [self.episode])
        self.assertTrue(any("source_event_ids" in error for error in errors))

    def test_context_leak_detected(self):
        payload = copy.deepcopy(self.payload)
        leaking = payload["candidates"][0]["following_context"][0]
        leaking["start_seconds"] = payload["candidates"][0]["start_seconds"] + 1.0
        errors = cb.validate_candidates(payload, [self.episode])
        self.assertTrue(any("leaks into the clip" in error for error in errors))

    def test_timestamp_drift_detected(self):
        payload = copy.deepcopy(self.payload)
        payload["candidates"][0]["start_timestamp"] = "(00:99.99)"
        errors = cb.validate_candidates(payload, [self.episode])
        self.assertTrue(any("start_timestamp" in error for error in errors))

    def test_scene_numbers_tamper_detected(self):
        payload = copy.deepcopy(self.payload)
        payload["candidates"][0]["scene_numbers"] = [7]
        errors = cb.validate_candidates(payload, [self.episode])
        self.assertTrue(any("scene_numbers" in error for error in errors))

    def test_false_tamper_passes_and_structure_is_balanced(self):
        first = self.payload["candidates"][0]
        self.assertFalse(set(first["source_event_ids"]) & set(first["context_event_ids"]))
        self.assertEqual(first["dialogue_event_count"], len(first["dialogue_transcript"]))


class DeterminismTests(unittest.TestCase):
    def test_repeated_build_is_identical(self):
        events = [
            _dialogue("a", 1, 0.0, 2.0, "Open."),
            _dialogue("b", 1, 45.0, 55.0, "Close."),
        ]
        episode = _make_episode(events, scene_bounds=((0.0, 100.0),))
        first = json.dumps(
            cb.build_all_candidates([episode], cb.BuilderConfig(target_window_seconds=50.0)), sort_keys=True
        )
        second = json.dumps(
            cb.build_all_candidates([episode], cb.BuilderConfig(target_window_seconds=50.0)), sort_keys=True
        )
        self.assertEqual(first, second)


@unittest.skipUnless(PACK_ZIP.exists() and PARSED_EPISODES.exists() and CANDIDATES_JSON.exists(),
                     "pack or builder output not present")
class FullPackIntegrationTests(unittest.TestCase):
    """Regression integration tests for the real 50-episode builder output."""

    @classmethod
    def setUpClass(cls):
        cls.episodes = json.loads(PARSED_EPISODES.read_text(encoding="utf-8"))
        cls.payload = json.loads(CANDIDATES_JSON.read_text(encoding="utf-8"))
        cls.config = cb.BuilderConfig(**cls.payload["config"]).validated()

    def test_every_episode_yields_candidates(self):
        counts = {stats["episode_number"]: stats["candidates_kept"] for stats in self.payload["per_episode"]}
        for episode in self.episodes:
            count = counts.get(episode["episode_number"], 0)
            self.assertGreater(count, 0, f"episode {episode['episode_number']} has no candidates")

    def test_all_candidates_pass_validation(self):
        self.assertEqual(cb.validate_candidates(self.payload, self.episodes, self.config), [])

    def test_candidates_stay_inside_pipeline_bounds(self):
        for candidate in self.payload["candidates"]:
            self.assertGreaterEqual(candidate["duration_seconds"], 20.0 - 1e-6)
            self.assertLessEqual(candidate["duration_seconds"], 60.0 + 1e-6)

    def test_totals_match_candidates(self):
        self.assertEqual(len(self.payload["candidates"]), self.payload["totals"]["candidates"])
        self.assertEqual(self.payload["totals"]["episodes"], 50)
        self.assertEqual(self.payload["totals"]["candidates"], 399)

    def test_jsonl_matches_json_candidates(self):
        lines = CANDIDATES_JSONL.read_text(encoding="utf-8").splitlines()
        from_json = [json.dumps(candidate, ensure_ascii=False) for candidate in self.payload["candidates"]]
        self.assertEqual(lines, from_json)

    def test_rebuilding_with_same_config_reproduces_output(self):
        rebuilt = cb.build_all_candidates(
            self.episodes, self.config, source_path=self.payload["source"]["episodes_json"]
        )
        self.assertEqual(rebuilt, self.payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
