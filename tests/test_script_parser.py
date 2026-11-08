#!/usr/bin/env python3
"""Regression tests for the Kuku screenplay parser (Parser v2).

Standard library only. Parses synthetic fixtures in temporary directories and,
when the supplied 50-episode pack is present next to this file, re-checks the
pack baseline: 50 files, 100 scenes, 3,727 events, event type counts and the
single known EVENT_OUTSIDE_SCENE warning.
"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / 'src'))

import json
import tempfile
import unittest
from pathlib import Path

import script_parser as sp

REPO_DIR = Path(__file__).resolve().parent
PACK_ZIP = REPO_DIR / "I Married a Broken Soldier. He was a Duke-scripts (2).zip"
PARSED_EPISODES = REPO_DIR / "parsed_output" / "parsed_episodes.json"

SAMPLE_SCREENPLAY = """\
EPISODE NUMBER: Episode 1
EPISODE LOGLINE: A betrayal at the altar.
CONTINUITY NOTES: Sets up the rivalry.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. CATHEDRAL – DAY
Scene start: 00:00.00
Scene end: 01:00.00

**CAMERA:**
Start: (00:00.00) | End: (00:10.00)
Wide establishing shot of the cathedral doors.

**ACTION:**
Start: (00:05.00) | End: (00:12.00)
Juliana walks the aisle while the doors part. Overlaps the camera shot.

**LADY JULIANA (Hopeful, Soft):**
Start: (00:12.50) | End: (00:15.00)
"Today I finally marry him."

**MATRIARCH (V.O.)(Cold):**
Start: (00:15.00) | End: (00:17.00)
"Not for long."

**SOUND FX:**
Start: (00:17.00) | End: (00:20.00)
A distant thunderclap under the organ.
**BAD BLOCK:**
An event header with no time range at all.

**JULIANA (Whispering):**
Start: (00:20.50) | End: (00:22.00)
"I still choose him."

SCENE 02
Scene heading: EXT. GRAVEYARD – NIGHT
Scene start: 01:00.00
Scene end: 00:59.00

**CAMERA:**
Start: (01:02.00) | End: (01:00.90)
A slow push-in whose end precedes its start by design of the fixture.

SCENE 01
Scene heading: INT. HALL – DAY
Scene start: 01:10.00
Scene end: 01:20.00

**CAMERA:**
Start: (01:11.00) | End: (01:12.00)
--- SCRIPT END ---
"""


def _write_screenplay(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


class TimestampTests(unittest.TestCase):
    def test_mm_ss_with_fraction(self):
        self.assertEqual(sp.timestamp_to_seconds("00:39.10"), 39.1)

    def test_minutes_above_59(self):
        self.assertEqual(sp.timestamp_to_seconds("75:12"), 4512.0)

    def test_hh_mm_ss(self):
        self.assertEqual(sp.timestamp_to_seconds("01:02:03"), 3723.0)

    def test_parenthesised_value(self):
        self.assertEqual(sp.timestamp_to_seconds("(00:10.20)"), 10.2)

    def test_unbalanced_parens_rejected(self):
        with self.assertRaises(ValueError):
            sp.timestamp_to_seconds("(00:10.20")

    def test_seconds_ge_60_rejected(self):
        with self.assertRaises(ValueError):
            sp.timestamp_to_seconds("00:61.00")

    def test_minutes_ge_60_with_hours_rejected(self):
        with self.assertRaises(ValueError):
            sp.timestamp_to_seconds("01:60:00")

    def test_garbage_rejected(self):
        with self.assertRaises(ValueError):
            sp.timestamp_to_seconds("not a time")


class SpeakerHeaderTests(unittest.TestCase):
    def test_plain_speaker(self):
        self.assertEqual(sp._split_speaker_header("NARRATOR"), ("NARRATOR", None, []))

    def test_speaker_with_delivery(self):
        speaker, delivery, voice = sp._split_speaker_header("JULIANA (Trembling, Pleading)")
        self.assertEqual(speaker, "JULIANA")
        self.assertEqual(delivery, "Trembling, Pleading")
        self.assertEqual(voice, [])

    def test_speaker_with_voice_marker_only(self):
        speaker, delivery, voice = sp._split_speaker_header("MATRIARCH (V.O.)")
        self.assertEqual(speaker, "MATRIARCH")
        self.assertIsNone(delivery)
        self.assertEqual(voice, ["V.O."])

    def test_speaker_with_voice_marker_then_delivery(self):
        speaker, delivery, voice = sp._split_speaker_header("MATRIARCH (V.O.)(Cold)")
        self.assertEqual(speaker, "MATRIARCH")
        self.assertEqual(delivery, "Cold")
        self.assertEqual(voice, ["V.O."])


class AliasTests(unittest.TestCase):
    def test_identity_by_default(self):
        aliases, sources = sp._build_speaker_alias_details(["GERALD"])
        self.assertEqual(aliases["GERALD"], "GERALD")
        self.assertEqual(sources["GERALD"], "identity")

    def test_inferred_when_prefixless_exists(self):
        aliases, sources = sp._build_speaker_alias_details(["LADY JULIANA", "JULIANA"])
        self.assertEqual(aliases["LADY JULIANA"], "JULIANA")
        self.assertEqual(sources["LADY JULIANA"], "inferred")

    def test_manual_alias(self):
        aliases, sources = sp._build_speaker_alias_details(
            ["THE FORMER GROOM"], {"THE FORMER GROOM": "GERALD"}
        )
        self.assertEqual(aliases["THE FORMER GROOM"], "GERALD")
        self.assertEqual(sources["THE FORMER GROOM"], "manual")

    def test_manual_alias_case_insensitive_matching(self):
        aliases, _ = sp._build_speaker_alias_details(["the former groom"], {"The Former Groom": "GERALD"})
        self.assertEqual(aliases["the former groom"], "GERALD")

    def test_alias_cycle_rejected(self):
        with self.assertRaises(ValueError):
            sp._build_speaker_alias_details(["A", "B"], {"A": "B", "B": "A"})

    def test_alias_conflict_rejected(self):
        with self.assertRaises(ValueError):
            sp._build_speaker_alias_details(["A"], {"A": "B", "a": "C"})

    def test_alias_non_string_rejected(self):
        with self.assertRaises(ValueError):
            sp._build_speaker_alias_details(["A"], {"A": 3})


class ParserBehaviourTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        directory = Path(self._tmp.name)
        self.path = _write_screenplay(directory, "episode-001.txt", SAMPLE_SCREENPLAY)
        episodes, self.report = sp.parse_input(directory)
        self.episode = episodes[0]
        self.scene1 = self.episode.scenes[0]  # 6 events, incl. header-only BAD BLOCK
        self.scene2 = self.episode.scenes[1]

    def tearDown(self):
        self._tmp.cleanup()

    def test_metadata_extracted(self):
        self.assertEqual(self.episode.episode_number, 1)
        self.assertEqual(self.episode.logline, "A betrayal at the altar.")
        self.assertEqual(self.episode.continuity_notes, "Sets up the rivalry.")

    def test_event_classification(self):
        types = [event.event_type for scene in self.episode.scenes for event in scene.events]
        self.assertEqual(types.count("camera"), 3)
        self.assertEqual(types.count("action"), 1)
        self.assertEqual(types.count("sound_fx"), 1)
        self.assertEqual(types.count("dialogue"), 4)

    def test_speaker_and_delivery_parsed_and_canonicalized(self):
        juliana = self.scene1.events[2]
        self.assertEqual(juliana.speaker, "LADY JULIANA")
        self.assertEqual(juliana.speaker_canonical, "JULIANA")
        self.assertEqual(juliana.speaker_alias_source, "inferred")
        self.assertEqual(juliana.delivery, "Hopeful, Soft")

    def test_matriarch_voice_direction_recorded(self):
        matriarch = self.scene1.events[3]
        self.assertEqual(matriarch.voice_direction, ["V.O."])
        self.assertEqual(matriarch.delivery, "Cold")

    def test_overlapping_events_retained(self):
        camera = self.scene1.events[0]
        action = self.scene1.events[1]
        self.assertLess(camera.start_seconds, action.start_seconds)
        self.assertGreater(camera.end_seconds, action.start_seconds)  # overlap preserved

    def test_source_order_index_is_episode_wide(self):
        indexes = [event.source_order_index for scene in self.episode.scenes for event in scene.events]
        self.assertEqual(indexes, list(range(1, len(indexes) + 1)))

    def test_scene_events_remain_in_source_order(self):
        order = [event.event_index_in_scene for event in self.episode.scenes[0].events]
        self.assertEqual(order, list(range(1, len(order) + 1)))

    def test_chronological_events_sortable(self):
        starts = [event.start_seconds for event in self.episode.chronological_events if event.start_seconds is not None]
        self.assertEqual(starts, sorted(starts))
        timeline_ids = self.episode.chronological_events
        self.assertEqual(sorted(event.event_id for event in timeline_ids),
                         sorted(event.event_id for event in self.episode.events))

    def test_missing_time_range_warned_and_content_retained(self):
        warnings = [warning for warning in self.episode.warnings if warning.code == "EVENT_MISSING_TIME_RANGE"]
        self.assertTrue(warnings)
        bad = self.scene1.events[5]
        self.assertEqual(bad.content, "An event header with no time range at all.")
        self.assertIsNone(bad.start_seconds)

    def test_scene_end_before_start_warned(self):
        self.assertIn("SCENE_END_BEFORE_START", {w.code for w in self.episode.warnings})

    def test_event_end_before_start_warned(self):
        self.assertIn("EVENT_END_BEFORE_START", {w.code for w in self.episode.warnings})

    def test_empty_event_content_warned(self):
        self.assertIn("EMPTY_EVENT_CONTENT", {w.code for w in self.episode.warnings})
        scene3 = self.episode.scenes[2]
        self.assertEqual(scene3.events[0].content, "")

    def test_duplicate_scene_number_assigns_occurrence(self):
        duplicates = [scene for scene in self.episode.scenes if scene.scene_number == 1]
        self.assertEqual(len(duplicates), 2)
        self.assertEqual(duplicates[1].scene_occurrence_index, 2)
        self.assertIn("DUPLICATE_SCENE_NUMBER", {w.code for w in self.episode.warnings})

    def test_report_counts(self):
        self.assertEqual(self.report["episodes_parsed"], 1)
        self.assertEqual(self.report["scenes_parsed"], 3)
        self.assertEqual(self.report["events_parsed"], 9)
        self.assertEqual(self.report["timeline_event_ids_emitted"], True)


class ParserMalformedInputTests(unittest.TestCase):
    def test_missing_script_markers_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_screenplay(Path(tmp), "episode-001.txt",
                                     "EPISODE NUMBER: Episode 7\nEPISODE LOGLINE: x\nCONTINUITY NOTES: y\nSCENE 01\n"
                                     "Scene heading: INT. VOID – DAY\nScene start: 00:00.00\nScene end: 00:02.00\n")
            episodes, report = sp.parse_input(path.parent)
        self.assertIn("SCRIPT_START_MARKER_MISSING", {w.code for w in episodes[0].warnings})
        self.assertEqual(report["episodes_parsed"], 1)

    def test_invalid_event_timestamp_warned_and_value_retained(self):
        text = (
            "EPISODE NUMBER: Episode 2\nEPISODE LOGLINE: x\nCONTINUITY NOTES: y\n"
            "--- SCRIPT START ---\nSCENE 01\nScene heading: INT. ROOM – DAY\n"
            "Scene start: 00:00.00\nScene end: 00:30.00\n"
            "**CAMERA:**\nStart: (00:aa) | End: (00:05.00)\nBroken timestamp fixture.\n"
            "--- SCRIPT END ---\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            _write_screenplay(Path(tmp), "episode-002.txt", text)
            episodes, _ = sp.parse_input(Path(tmp))
        codes = {warning.code for warning in episodes[0].warnings}
        self.assertIn("INVALID_EVENT_TIMESTAMP", codes)
        event = episodes[0].scenes[0].events[0]
        self.assertEqual(event.content, "Broken timestamp fixture.")

    def test_event_outside_scene_bounds_warned_with_tolerance(self):
        text = (
            "EPISODE NUMBER: Episode 3\nEPISODE LOGLINE: x\nCONTINUITY NOTES: y\n"
            "--- SCRIPT START ---\nSCENE 01\nScene heading: INT. ROOM – DAY\n"
            "Scene start: 00:00.00\nScene end: 00:10.00\n"
            "**CAMERA:**\nStart: (00:09.99) | End: (00:11.40)\nJust outside by tolerance rules.\n"
            "--- SCRIPT END ---\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            _write_screenplay(Path(tmp), "episode-003.txt", text)
            episodes, _ = sp.parse_input(Path(tmp))
        codes = {warning.code for warning in episodes[0].warnings}
        self.assertIn("EVENT_OUTSIDE_SCENE", codes)


@unittest.skipUnless(PACK_ZIP.exists(), "supplied 50-episode ZIP not present")
class SuppliedPackRegressionTests(unittest.TestCase):
    """Baseline regression against the exact supplied input pack."""

    def setUp(self):
        self.episodes, self.report = sp.parse_input(PACK_ZIP)

    def test_pack_counts_match_baseline(self):
        self.assertEqual(self.report["files_discovered"], 50)
        self.assertEqual(self.report["episodes_parsed"], 50)
        self.assertEqual(self.report["scenes_parsed"], 100)
        self.assertEqual(self.report["events_parsed"], 3_727)

    def test_pack_event_type_counts_match_baseline(self):
        self.assertEqual(self.report["event_type_counts"], {
            "action": 372, "camera": 1754, "dialogue": 1487, "sound_fx": 114,
        })

    def test_pack_warning_baseline(self):
        self.assertEqual(self.report["warnings_count"], 1)
        self.assertEqual(self.report["warning_code_counts"], {"EVENT_OUTSIDE_SCENE": 1})

    def test_pack_episode_numbers_and_ordering(self):
        numbers = [episode.episode_number for episode in self.episodes]
        self.assertEqual(numbers, list(range(1, 51)))

    def test_pack_source_order_and_timeline_integrity(self):
        for episode in self.episodes:
            order = [event.source_order_index for scene in episode.scenes for event in scene.events]
            self.assertEqual(order, list(range(1, len(order) + 1)))
            starts = [event.start_seconds for event in episode.chronological_events
                      if event.start_seconds is not None]
            self.assertEqual(starts, sorted(starts))

    def test_pack_all_events_have_timestamps_and_content(self):
        for episode in self.episodes:
            for event in episode.events:
                self.assertIsNotNone(event.start_seconds, msg=event.event_id)
                self.assertIsNotNone(event.end_seconds, msg=event.event_id)
                self.assertTrue(event.content, msg=event.event_id)

    @unittest.skipUnless(PARSED_EPISODES.exists(), "parsed_output/parsed_episodes.json not present")
    def test_written_json_matches_in_memory_parse(self):
        with PARSED_EPISODES.open(encoding="utf-8") as handle:
            written = json.load(handle)
        self.assertEqual(len(written), len(self.episodes))
        for written_episode, episode in zip(written, self.episodes):
            self.assertEqual(written_episode["episode_number"], episode.episode_number)
            timeline_ids = [event.event_id for event in episode.chronological_events]
            self.assertEqual(written_episode["timeline_event_ids"], timeline_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
