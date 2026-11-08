#!/usr/bin/env python3
"""Kuku Context Builder — stage 2 of the promo-discovery pipeline.

Transforms parsed screenplay events into promo candidate windows that future
embedding and LLM components can consume.

Inputs
------
* parsed_episodes.json as produced by script_parser.py (Parser v2).
  The builder is a pure consumer of the parser's JSON contract: it never
  imports the parser and never re-parses screenplay text.

Outputs
-------
* candidates.json        every candidate window, wrapped with config + totals
* candidates.jsonl       one candidate per line, for streaming into embeddings
* context_report.json    detailed generation statistics (no candidates)

Determinism
-----------
* No randomness, no wall-clock timestamps, no LLM or model calls, no network.
* Every window boundary snaps to an original event boundary, so original
  timestamp strings are reused verbatim rather than recomputed.
* Repeated runs on the same input with the same config produce identical bytes.

Multi-segment readiness
-----------------------
Candidates store their content as a `segments` list. This first implementation
emits exactly one continuous segment per candidate, but `_assemble_candidate`
accepts any number of segment specs, so a later recommendation stage can emit
multi-segment promos (e.g. cold-open teaser + climax) without a schema change.
For multi-segment candidates `duration_seconds` is the SUM of segment durations
(the audio that would actually play), while `span_seconds` is the wall-clock
span from the first segment start to the last segment end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

SCHEMA_VERSION = "1.0"
BUILDER_NAME = "context_builder.py"
EPS = 1e-9  # float tolerance for interval comparisons
ROUND_DIGITS = 3

VALID_EVENT_TYPES = {"dialogue", "camera", "action", "sound_fx"}


@dataclass(frozen=True)
class BuilderConfig:
    """All tunable knobs. Defaults are frozen for reproducible regression runs."""

    min_window_seconds: float = 20.0
    max_window_seconds: float = 60.0
    target_window_seconds: float = 30.0
    max_windows_per_anchor: int = 1
    context_before_seconds: float = 12.0
    context_after_seconds: float = 12.0
    anchor_event_types: tuple[str, ...] = ("dialogue",)
    dedup_iou_threshold: float = 0.5

    def validated(self) -> "BuilderConfig":
        if self.min_window_seconds <= 0 or self.max_window_seconds <= self.min_window_seconds:
            raise ValueError(
                f"Window bounds invalid: min={self.min_window_seconds} max={self.max_window_seconds}"
            )
        if not (self.min_window_seconds <= self.target_window_seconds <= self.max_window_seconds):
            raise ValueError(
                f"Target window {self.target_window_seconds} must lie within "
                f"[{self.min_window_seconds}, {self.max_window_seconds}]."
            )
        if self.max_windows_per_anchor < 1:
            raise ValueError("max_windows_per_anchor must be >= 1.")
        if self.context_before_seconds < 0 or self.context_after_seconds < 0:
            raise ValueError("Context spans must be >= 0 seconds.")
        bad_types = set(self.anchor_event_types) - VALID_EVENT_TYPES
        if bad_types:
            raise ValueError(
                f"Unknown anchor event types {sorted(bad_types)}; valid: {sorted(VALID_EVENT_TYPES)}"
            )
        if not self.anchor_event_types:
            raise ValueError("anchor_event_types must not be empty.")
        if not (0.0 < self.dedup_iou_threshold <= 1.0):
            raise ValueError("dedup_iou_threshold must lie in (0, 1].")
        return self


# --------------------------------------------------------------------------
# Reading the parser contract
# --------------------------------------------------------------------------

def _timeline_key(event: dict):
    return (
        event["start_seconds"],
        event["end_seconds"],
        event.get("source_order_index") if event.get("source_order_index") is not None else float("inf"),
    )


def episode_timeline(episode: dict) -> tuple[list[dict], list[dict]]:
    """Return (timed events in chronological order, untimestamped events).

    This mirrors the parser's `timeline_event_ids` contract: chronological by
    start, then end, then source order. `source_order_index` stays on every
    event, so the original screenplay order is never lost.
    """
    timed: list[dict] = []
    untimed: list[dict] = []
    for scene in episode.get("scenes", []):
        for event in scene.get("events", []):
            if event.get("start_seconds") is not None and event.get("end_seconds") is not None:
                timed.append(event)
            else:
                untimed.append(event)
    timed.sort(key=_timeline_key)
    return timed, untimed


def episode_runtime_seconds(episode: dict) -> float:
    """Last known end of any timed event or scene edge, used as episode runtime."""
    ends: list[float] = []
    for scene in episode.get("scenes", []):
        for value in (scene.get("end_seconds"), scene.get("start_seconds")):
            if value is not None:
                ends.append(value)
        for event in scene.get("events", []):
            for value in (event.get("end_seconds"), event.get("start_seconds")):
                if value is not None:
                    ends.append(value)
    return max(ends) if ends else 0.0


def _intersected_scenes(episode: dict, start: float, end: float) -> list[dict]:
    """Scenes that the window [start, end] actually overlaps (touching excluded)."""
    hits = []
    for scene in episode.get("scenes", []):
        scene_start = scene.get("start_seconds")
        scene_end = scene.get("end_seconds")
        if scene_start is None or scene_end is None:
            continue
        if scene_end > start + EPS and scene_start < end - EPS:
            hits.append(scene)
    return hits


def _clip_events(timeline: list[dict], start: float, end: float) -> list[dict]:
    """Every timed event whose interval overlaps [start, end], chronological.

    Overlapping camera/action/dialogue/sound events are all retained. Events
    that poke slightly outside the window edges are kept too (flagged as
    partial overlaps in `_event_snapshot`), because their audible/visible
    content really is inside the clip; only fully outside events are excluded.
    """
    return [event for event in timeline if event["end_seconds"] > start + EPS and event["start_seconds"] < end - EPS]


def _context_events(
    timeline: list[dict], start: float, end: float, before: float, after: float
) -> tuple[list[dict], list[dict]]:
    """Events strictly outside the window but within the context span.

    Strictly outside: an event ending exactly at `start` (gap 0.0) is context;
    context is never part of the promo duration.
    """
    precedence = []
    following = []
    if before > 0:
        lower = start - before
        precedence = [e for e in timeline if e["end_seconds"] <= start + EPS and e["end_seconds"] > lower - EPS]
    if after > 0:
        upper = end + after
        following = [e for e in timeline if e["start_seconds"] >= end - EPS and e["start_seconds"] < upper - EPS]
    return precedence, following


def _boundary_overlap(event: dict, start: float, end: float) -> Optional[str]:
    """Classify how a partially-overlapping event straddles a window edge."""
    fully = event["start_seconds"] >= start - EPS and event["end_seconds"] <= end + EPS
    if fully:
        return None
    crosses_start = event["start_seconds"] < start - EPS
    crosses_end = event["end_seconds"] > end + EPS
    if crosses_start and crosses_end:
        return "wraps"
    if crosses_start:
        return "start"
    if crosses_end:
        return "end"
    return None


def _event_snapshot(event: dict, start: float, end: float) -> dict:
    """Candidate-local view of a clip event with original references intact."""
    overlap = min(event["end_seconds"], end) - max(event["start_seconds"], start)
    is_dialogue = event.get("event_type") == "dialogue"
    return {
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "scene_number": event.get("scene_number"),
        "source_order_index": event.get("source_order_index"),
        "start_seconds": round(event["start_seconds"], ROUND_DIGITS),
        "end_seconds": round(event["end_seconds"], ROUND_DIGITS),
        "start_timestamp": event.get("start_timestamp"),
        "end_timestamp": event.get("end_timestamp"),
        "content": event.get("content"),
        "speaker": event.get("speaker") if is_dialogue else None,
        "speaker_canonical": event.get("speaker_canonical") if is_dialogue else None,
        "delivery": event.get("delivery") if is_dialogue else None,
        "voice_direction": event.get("voice_direction") if is_dialogue else None,
        "fully_contained": event["start_seconds"] >= start - EPS and event["end_seconds"] <= end + EPS,
        "overlap_seconds": round(max(overlap, 0.0), ROUND_DIGITS),
        "boundary_overlap": _boundary_overlap(event, start, end),
    }


def _context_snapshot(event: dict, gap: float) -> dict:
    is_dialogue = event.get("event_type") == "dialogue"
    return {
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "scene_number": event.get("scene_number"),
        "source_order_index": event.get("source_order_index"),
        "start_seconds": round(event["start_seconds"], ROUND_DIGITS),
        "end_seconds": round(event["end_seconds"], ROUND_DIGITS),
        "start_timestamp": event.get("start_timestamp"),
        "end_timestamp": event.get("end_timestamp"),
        "content": event.get("content"),
        "speaker": event.get("speaker") if is_dialogue else None,
        "speaker_canonical": event.get("speaker_canonical") if is_dialogue else None,
        "delivery": event.get("delivery") if is_dialogue else None,
        "gap_to_window_seconds": round(gap, ROUND_DIGITS),
    }


def _untimed_snapshot(event: dict) -> dict:
    return {
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type"),
        "scene_number": event.get("scene_number"),
        "source_order_index": event.get("source_order_index"),
        "content": event.get("content"),
        "speaker": event.get("speaker"),
        "speaker_canonical": event.get("speaker_canonical"),
    }


def _speaker_summary(dialogue_events: Iterable[dict]) -> list[dict]:
    """Canonical speaker aggregation preserving first-appearance order."""
    ordered: list[str] = []
    grouped: dict[str, dict] = {}
    for event in dialogue_events:
        canonical = event.get("speaker_canonical") or event.get("speaker") or "UNKNOWN"
        if canonical not in grouped:
            ordered.append(canonical)
            grouped[canonical] = {
                "speaker_canonical": canonical,
                "raw_spelling": event.get("speaker"),
                "dialogue_event_ids": [],
                "deliveries": [],
                "dialogue_event_count": 0,
            }
        entry = grouped[canonical]
        if entry["raw_spelling"] is None:
            entry["raw_spelling"] = event.get("speaker")
        entry["dialogue_event_ids"].append(event.get("event_id"))
        entry["dialogue_event_count"] += 1
        delivery = event.get("delivery")
        if delivery and delivery not in entry["deliveries"]:
            entry["deliveries"].append(delivery)
    return [grouped[name] for name in ordered]


def _iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    intersection = min(end_a, end_b) - max(start_a, start_b)
    union = max(end_a, end_b) - min(start_a, start_b)
    if union <= EPS:
        return 1.0
    return max(intersection, 0.0) / union


# --------------------------------------------------------------------------
# Candidate assembly
# --------------------------------------------------------------------------

def _assemble_candidate(
    episode: dict,
    timeline: list[dict],
    untimed: list[dict],
    segment_specs: list[dict],
    config: BuilderConfig,
    candidate_seed: str,
) -> dict:
    """Build one candidate from ordered segment specs.

    Segment spec fields: start_seconds, start_timestamp, end_seconds,
    end_timestamp, start_event_id, end_event_id. Boundary times MUST be
    original event boundaries so timestamps stay verbatim.
    """
    segments = []
    covered_intervals: list[tuple[float, float]] = []
    for index, spec in enumerate(segment_specs, start=1):
        start = round(spec["start_seconds"], ROUND_DIGITS)
        end = round(spec["end_seconds"], ROUND_DIGITS)
        if end <= start:
            raise ValueError(f"Segment {index} has non-positive duration: {spec}")
        covered_intervals.append((start, end))
        segment_events = _clip_events(timeline, start, end)
        segments.append({
            "segment_index": index,
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": round(end - start, ROUND_DIGITS),
            "start_timestamp": spec.get("start_timestamp"),
            "end_timestamp": spec.get("end_timestamp"),
            "start_event_id": spec.get("start_event_id"),
            "end_event_id": spec.get("end_event_id"),
            "event_ids": [event.get("event_id") for event in segment_events],
        })

    start = min(segment["start_seconds"] for segment in segments)
    end = max(segment["end_seconds"] for segment in segments)
    duration = round(sum(segment["duration_seconds"] for segment in segments), ROUND_DIGITS)

    clip_events: list[dict] = []
    for event in timeline:
        if any(event["end_seconds"] > s + EPS and event["start_seconds"] < e - EPS for s, e in covered_intervals):
            clip_events.append(event)
    clip_events.sort(key=_timeline_key)

    dialogue_events = [event for event in clip_events if event.get("event_type") == "dialogue"]
    action_camera_events = [
        event for event in clip_events if event.get("event_type") in {"camera", "action", "sound_fx"}
    ]
    scenes = _intersected_scenes(episode, start, end)
    scene_numbers = [scene.get("scene_number") for scene in scenes]
    scene_numbers = sorted(set(scene_numbers), key=scene_numbers.index)
    preceding, following = _context_events(
        timeline, start, end, config.context_before_seconds, config.context_after_seconds
    )
    preceding_context = [_context_snapshot(event, start - event["end_seconds"]) for event in preceding]
    following_context = [_context_snapshot(event, event["start_seconds"] - end) for event in following]
    scene_lookup = {scene.get("scene_number") for scene in scenes}
    untimestamped_events = [_untimed_snapshot(event) for event in untimed if event.get("scene_number") in scene_lookup]
    partly_contained = [event for event in clip_events if _boundary_overlap(event, start, end) is not None]

    episode_number = episode.get("episode_number")
    digest = hashlib.sha1(candidate_seed.encode("utf-8")).hexdigest()[:6]
    candidate_id = f"cand_ep{episode_number:02d}_t{start:.1f}-{end:.1f}_{digest}"

    return {
        "candidate_id": candidate_id,
        "episode_number": episode_number,
        "scene_numbers": scene_numbers,
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": duration,
        "span_seconds": round(end - start, ROUND_DIGITS),
        "start_timestamp": segments[0]["start_timestamp"],
        "end_timestamp": segments[-1]["end_timestamp"],
        "segment_count": len(segments),
        "segments": segments,
        "dialogue_transcript": [_event_snapshot(event, start, end) for event in dialogue_events],
        "actions_and_camera": [_event_snapshot(event, start, end) for event in action_camera_events],
        "speakers": _speaker_summary(dialogue_events),
        "preceding_context": preceding_context,
        "following_context": following_context,
        "untimestamped_events": untimestamped_events,
        "anchor_event_id": segment_specs[0].get("start_event_id"),
        "boundary_event_ids": {
            "start_event_id": segments[0].get("start_event_id"),
            "end_event_id": segments[-1].get("end_event_id"),
        },
        "source_event_ids": [event.get("event_id") for event in clip_events],
        "context_event_ids": [item["event_id"] for item in preceding_context] + [
            item["event_id"] for item in following_context
        ],
        "event_count": len(clip_events),
        "dialogue_event_count": len(dialogue_events),
        "fully_contained_event_count": len(clip_events) - len(partly_contained),
        "partial_overlap_event_count": len(partly_contained),
        "crosses_scene_boundary": len(scenes) > 1,
        "scenes": [
            {
                "scene_number": scene.get("scene_number"),
                "scene_occurrence_index": scene.get("scene_occurrence_index"),
                "heading": scene.get("heading"),
                "start_seconds": scene.get("start_seconds"),
                "end_seconds": scene.get("end_seconds"),
                "start_timestamp": scene.get("start_timestamp"),
                "end_timestamp": scene.get("end_timestamp"),
            }
            for scene in scenes
        ],
        "episode": {
            "episode_number": episode.get("episode_number"),
            "episode_label": episode.get("episode_label"),
            "logline": episode.get("logline"),
            "continuity_notes": episode.get("continuity_notes"),
            "source_file": episode.get("source_file"),
            "runtime_seconds": round(episode_runtime_seconds(episode), ROUND_DIGITS),
            "scene_count": len(episode.get("scenes", [])),
        },
    }


# --------------------------------------------------------------------------
# Window generation
# --------------------------------------------------------------------------

def build_candidates_for_episode(episode: dict, config: BuilderConfig) -> tuple[list[dict], dict]:
    """Generate deduplicated candidate windows for one parsed episode.

    Strategy
    --------
    1. Anchor the window start exactly on the start boundary of each anchor
       event (default: every dialogue event).
    2. Snap the window end to an actual event end boundary inside
       [start + min_window, start + max_window]. Up to `max_windows_per_anchor`
       boundaries are picked: the one closest to `target_window_seconds`
       (primary) and the earliest qualifying boundary (secondary variant).
    3. Windows are generated chronologically; a new window is dropped if it
       is an exact duplicate of, or its temporal IoU with any *already
       accepted window from another anchor* reaches `dedup_iou_threshold`.
       Same-anchor size variants (short + target) are intentionally exempt.
    """
    if config.min_window_seconds <= 0 or config.max_window_seconds <= config.min_window_seconds:
        raise ValueError(
            f"Window bounds invalid: min={config.min_window_seconds} max={config.max_window_seconds}"
        )
    timeline, untimed = episode_timeline(episode)
    anchors = [event for event in timeline if event.get("event_type") in set(config.anchor_event_types)]
    boundaries = sorted({round(event["end_seconds"], ROUND_DIGITS) for event in timeline})

    accepted: list[dict] = []
    stats = {
        "episode_number": episode.get("episode_number"),
        "anchors_considered": len(anchors),
        "anchors_without_boundary": 0,
        "windows_generated": 0,
        "dropped_exact_duplicate": 0,
        "dropped_near_duplicate": 0,
        "candidates_kept": 0,
    }

    for anchor in anchors:
        anchor_start = round(anchor["start_seconds"], ROUND_DIGITS)
        low = anchor_start + config.min_window_seconds
        high = anchor_start + config.max_window_seconds
        in_range = [b for b in boundaries if low - EPS <= b <= high + EPS]
        if not in_range:
            stats["anchors_without_boundary"] += 1
            continue

        target = anchor_start + config.target_window_seconds
        chosen = {min(in_range, key=lambda b: (abs(b - target), b))}
        if config.max_windows_per_anchor >= 2:
            chosen.add(in_range[0])

        window_ends = sorted(chosen)[: config.max_windows_per_anchor]
        for window_end in window_ends:
            end_snapshot = next(
                (event for event in timeline if round(event["end_seconds"], ROUND_DIGITS) == window_end),
                None,
            )
            spec = {
                "start_seconds": anchor_start,
                "end_seconds": window_end,
                "start_timestamp": anchor.get("start_timestamp"),
                "end_timestamp": end_snapshot.get("end_timestamp") if end_snapshot else None,
                "start_event_id": anchor.get("event_id"),
                "end_event_id": end_snapshot.get("event_id") if end_snapshot else None,
            }
            seed = "|".join([
                str(episode.get("episode_number")),
                str(episode.get("source_file")),
                f"{anchor_start:.3f}",
                f"{window_end:.3f}",
                str(anchor.get("event_id")),
            ])
            candidate = _assemble_candidate(episode, timeline, untimed, [spec], config, seed)
            stats["windows_generated"] += 1

            duplicate = any(
                other["start_seconds"] == candidate["start_seconds"]
                and other["end_seconds"] == candidate["end_seconds"]
                for other in accepted
            )
            if duplicate:
                stats["dropped_exact_duplicate"] += 1
                continue
            near_duplicate = any(
                anchor.get("event_id") != other.get("anchor_event_id")
                and _iou(
                    candidate["start_seconds"], candidate["end_seconds"],
                    other["start_seconds"], other["end_seconds"],
                ) >= config.dedup_iou_threshold - EPS
                for other in accepted
            )
            if near_duplicate:
                stats["dropped_near_duplicate"] += 1
                continue
            accepted.append(candidate)

    accepted.sort(key=lambda c: (c["start_seconds"], c["end_seconds"]))
    stats["candidates_kept"] = len(accepted)
    if accepted:
        durations = [candidate["duration_seconds"] for candidate in accepted]
        stats["duration_min"] = round(min(durations), ROUND_DIGITS)
        stats["duration_median"] = round(statistics.median(durations), ROUND_DIGITS)
        stats["duration_max"] = round(max(durations), ROUND_DIGITS)
    stats["untimestamped_events_in_episode"] = len(untimed)
    return accepted, stats


def build_all_candidates(
    episodes: list[dict], config: BuilderConfig, source_path: str = ""
) -> dict:
    """Build the full candidates payload (wrapper + per-episode stats)."""
    per_episode = []
    all_candidates: list[dict] = []
    for episode in sorted(
        episodes, key=lambda ep: (ep.get("episode_number") is None, ep.get("episode_number") or 0)
    ):
        candidates, stats = build_candidates_for_episode(episode, config)
        per_episode.append(stats)
        all_candidates.extend(candidates)

    totals = {
        "episodes": len(episodes),
        "candidates": len(all_candidates),
        "anchors_considered": sum(stats["anchors_considered"] for stats in per_episode),
        "anchors_without_boundary": sum(stats["anchors_without_boundary"] for stats in per_episode),
        "windows_generated": sum(stats["windows_generated"] for stats in per_episode),
        "dropped_exact_duplicate": sum(stats["dropped_exact_duplicate"] for stats in per_episode),
        "dropped_near_duplicate": sum(stats["dropped_near_duplicate"] for stats in per_episode),
    }
    durations = [candidate["duration_seconds"] for candidate in all_candidates]
    if durations:
        totals["duration_min"] = round(min(durations), ROUND_DIGITS)
        totals["duration_median"] = round(statistics.median(durations), ROUND_DIGITS)
        totals["duration_max"] = round(max(durations), ROUND_DIGITS)
    totals["candidates_crossing_scene_boundary"] = sum(
        1 for candidate in all_candidates if candidate["crosses_scene_boundary"]
    )
    totals["candidates_with_partial_overlaps"] = sum(
        1 for candidate in all_candidates if candidate["partial_overlap_event_count"] > 0
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "builder": BUILDER_NAME,
        "source": {"episodes_json": source_path},
        "config": asdict(config),
        "totals": totals,
        "per_episode": per_episode,
        "candidates": all_candidates,
    }


# --------------------------------------------------------------------------
# Validation (independent recomputation from the parser output)
# --------------------------------------------------------------------------

def validate_candidates(
    payload: dict,
    episodes: list[dict],
    config: Optional[BuilderConfig] = None,
) -> list[str]:
    """Validate a builder payload against the parser JSON it claims to come from.

    Recomputes clip membership, scene intersections and speaker values from
    the parser output instead of trusting the candidate, so tampering or
    generation bugs surface as explicit error strings. Returns a list of
    human-readable errors; empty list means the payload is sound.
    """
    errors: list[str] = []
    if isinstance(payload, list):
        payload = {"candidates": payload}
    config = config or BuilderConfig().validated()

    episode_by_number = {episode.get("episode_number"): episode for episode in episodes}
    episode_lens: dict[int, tuple[list[dict], list[dict]]] = {}

    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return ["payload has no candidates list"]

    seen_ids: set[str] = set()
    per_episode_windows: dict[int, list[dict]] = {}

    for candidate in candidates:
        cid = candidate.get("candidate_id")
        where = f"candidate {cid!r}"
        episode_number = candidate.get("episode_number")
        episode = episode_by_number.get(episode_number)
        if episode is None:
            errors.append(f"{where}: unknown episode_number {episode_number!r}")
            continue
        if episode_number not in episode_lens:
            episode_lens[episode_number] = episode_timeline(episode)
        timeline, _untimed = episode_lens[episode_number]

        if cid in seen_ids:
            errors.append(f"{where}: duplicate candidate_id")
        seen_ids.add(cid)

        required = [
            "candidate_id", "episode_number", "scene_numbers", "start_seconds", "end_seconds",
            "duration_seconds", "segments", "dialogue_transcript", "actions_and_camera",
            "speakers", "preceding_context", "following_context", "source_event_ids",
            "anchor_event_id", "boundary_event_ids", "scenes",
        ]
        missing = [key for key in required if key not in candidate]
        if missing:
            errors.append(f"{where}: missing schema keys {missing}")
            continue

        start = candidate["start_seconds"]
        end = candidate["end_seconds"]
        if start is None or end is None or end <= start:
            errors.append(f"{where}: start/end invalid ({start} -> {end})")
            continue
        duration = sum(segment["duration_seconds"] for segment in candidate["segments"])
        if abs(duration - candidate["duration_seconds"]) > 1e-3:
            errors.append(f"{where}: duration_seconds {candidate['duration_seconds']} != segment sum {duration}")
        if not (config.min_window_seconds - 1e-6 <= candidate["duration_seconds"] <= config.max_window_seconds + 1e-6):
            errors.append(
                f"{where}: duration {candidate['duration_seconds']} outside "
                f"[{config.min_window_seconds}, {config.max_window_seconds}]"
            )
        if abs((end - start) - candidate["span_seconds"]) > 1e-3:
            errors.append(f"{where}: span_seconds mismatch")

        # Boundaries must remain original event boundaries, not invented times.
        boundary_ids = candidate.get("boundary_event_ids") or {}
        start_event = next((e for e in timeline if e.get("event_id") == boundary_ids.get("start_event_id")), None)
        end_event = next((e for e in timeline if e.get("event_id") == boundary_ids.get("end_event_id")), None)
        if start_event is None or end_event is None:
            errors.append(f"{where}: boundary event ids not resolvable in parser output")
        else:
            if abs(start_event["start_seconds"] - start) > 1e-3:
                errors.append(f"{where}: start snaps away from its anchor boundary")
            if abs(end_event["end_seconds"] - end) > 1e-3:
                errors.append(f"{where}: end snaps away from its boundary event end")
            if candidate.get("start_timestamp") != start_event.get("start_timestamp"):
                errors.append(f"{where}: start_timestamp drifted from source string")
            if candidate.get("end_timestamp") != end_event.get("end_timestamp"):
                errors.append(f"{where}: end_timestamp drifted from source string")

        # Clip membership recomputed independently; overlapping events must stay.
        expected_events = _clip_events(timeline, start, end)
        expected_ids = [event.get("event_id") for event in expected_events]
        if candidate.get("source_event_ids") != expected_ids:
            errors.append(f"{where}: source_event_ids do not match recomputed clip events")

        expected_scenes = [scene.get("scene_number") for scene in _intersected_scenes(episode, start, end)]
        expected_scenes = sorted(set(expected_scenes), key=expected_scenes.index)
        if candidate.get("scene_numbers") != expected_scenes:
            errors.append(f"{where}: scene_numbers {candidate.get('scene_numbers')} != recomputed {expected_scenes}")

        timeline_by_id = {event.get("event_id"): event for event in timeline}
        for event_id in candidate.get("source_event_ids", []):
            if event_id not in timeline_by_id:
                errors.append(f"{where}: clip event {event_id!r} not traceable in parser output")
        for item in candidate.get("dialogue_transcript", []):
            source = timeline_by_id.get(item.get("event_id"))
            if source is None:
                errors.append(f"{where}: dialogue item {item.get('event_id')!r} not traceable")
                continue
            if item.get("event_type") != "dialogue":
                errors.append(f"{where}: non-dialogue event in dialogue_transcript")
            if item.get("start_timestamp") != source.get("start_timestamp") or (
                item.get("end_timestamp") != source.get("end_timestamp")
            ):
                errors.append(f"{where}: dialogue {item.get('event_id')!r} timestamps drifted")
            if item.get("speaker_canonical") != source.get("speaker_canonical"):
                errors.append(f"{where}: dialogue {item.get('event_id')!r} speaker_canonical drifted")
        for snapshot_list, expected_type in (
            (candidate.get("actions_and_camera"), {"camera", "action", "sound_fx"}),
        ):
            for item in snapshot_list:
                if item.get("event_type") not in expected_type:
                    errors.append(f"{where}: unexpected event_type in actions_and_camera")

        # Context must sit strictly outside the clip and never inside it.
        for item in candidate.get("preceding_context", []):
            if item.get("end_seconds") > start + EPS:
                errors.append(f"{where}: preceding context {item.get('event_id')!r} leaks into the clip")
        for item in candidate.get("following_context", []):
            if item.get("start_seconds") < end - EPS:
                errors.append(f"{where}: following context {item.get('event_id')!r} leaks into the clip")
        clip_id_set = set(candidate.get("source_event_ids", []))
        leaked = clip_id_set.intersection(candidate.get("context_event_ids", []))
        if leaked:
            errors.append(f"{where}: context ids also present in the clip: {sorted(leaked)}")

        if not candidate.get("source_event_ids"):
            errors.append(f"{where}: candidate contains no clip events")

        per_episode_windows.setdefault(episode_number, []).append(candidate)

    # Near-duplicate sweep across the final candidate set, per episode.
    for episode_number, windows in per_episode_windows.items():
        ordered = sorted(windows, key=lambda c: (c["start_seconds"], c["end_seconds"]))
        for i, current in enumerate(ordered):
            for previous in ordered[:i]:
                score = _iou(current["start_seconds"], current["end_seconds"], previous["start_seconds"], previous["end_seconds"])
                if score >= config.dedup_iou_threshold - EPS and current.get("anchor_event_id") != previous.get("anchor_event_id"):
                    errors.append(
                        f"candidate {current.get('candidate_id')!r}: IoU {score:.3f} with "
                        f"{previous.get('candidate_id')!r} exceeds dedup threshold"
                    )
    return errors


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _write_jsonl_atomic(path: Path, rows: Iterable[object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _config_from_args(args: argparse.Namespace) -> BuilderConfig:
    return BuilderConfig(
        min_window_seconds=args.min_window,
        max_window_seconds=args.max_window,
        target_window_seconds=args.target_window,
        max_windows_per_anchor=args.max_windows_per_anchor,
        context_before_seconds=args.context_before,
        context_after_seconds=args.context_after,
        anchor_event_types=tuple(t.strip().lower() for t in args.anchor_types.split(",") if t.strip()),
        dedup_iou_threshold=args.iou,
    ).validated()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build promo candidate windows from parsed screenplay episodes."
    )
    parser.add_argument("--input", help="Path to parsed_episodes.json from script_parser.py")
    parser.add_argument("--output", default="context_output", help="Output directory (default: context_output)")
    parser.add_argument("--min-window", type=float, default=20.0)
    parser.add_argument("--max-window", type=float, default=60.0)
    parser.add_argument("--target-window", type=float, default=30.0)
    parser.add_argument("--context-before", type=float, default=12.0)
    parser.add_argument("--context-after", type=float, default=12.0)
    parser.add_argument("--max-windows-per-anchor", type=int, default=1, choices=(1, 2))
    parser.add_argument("--iou", type=float, default=0.5, help="Temporal IoU dedup threshold")
    parser.add_argument("--anchor-types", default="dialogue",
                        help="Comma-separated anchor event types (dialogue,camera,action,sound_fx)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate an existing candidates.json instead of building")
    parser.add_argument("--candidates", help="Path to candidates.json for --validate-only mode")
    args = parser.parse_args(argv)

    try:
        config = _config_from_args(args)
        if args.validate_only:
            if not args.input or not args.candidates:
                parser.error("--validate-only requires --input (parsed_episodes.json) and --candidates")
            episodes = load_payload(Path(args.input))
            payload = load_payload(Path(args.candidates))
            validation_config = BuilderConfig(**payload.get("config", {})).validated() \
                if isinstance(payload, dict) and payload.get("config") else config
            errors = validate_candidates(payload, episodes, validation_config)
            if errors:
                for error in errors:
                    print(f"VALIDATION ERROR: {error}", file=sys.stderr)
                return 2
            print(f"OK: {len(payload.get('candidates', []))} candidates validated with no errors.")
            return 0

        if not args.input:
            parser.error("--input is required to build candidates")
        episodes = load_payload(Path(args.input))
        payload = build_all_candidates(episodes, config, source_path=str(args.input))
        output_dir = Path(args.output).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        candidates_path = output_dir / "candidates.json"
        report = {key: value for key, value in payload.items() if key != "candidates"}
        _write_json_atomic(candidates_path, payload)
        _write_jsonl_atomic(output_dir / "candidates.jsonl", payload["candidates"])
        _write_json_atomic(output_dir / "context_report.json", report)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    totals = payload["totals"]
    print(f"Built {totals['candidates']} candidates from {totals['episodes']} episodes "
          f"({totals['anchors_considered']} anchors, {totals['dropped_near_duplicate']} near-dups dropped)")
    if "duration_min" in totals:
        print(f"Durations: min {totals['duration_min']}s | median {totals['duration_median']}s | max {totals['duration_max']}s")
    print(f"Candidates JSON:        {output_dir / 'candidates.json'}")
    print(f"Candidates JSONL:       {output_dir / 'candidates.jsonl'}")
    print(f"Context report:         {output_dir / 'context_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
