#!/usr/bin/env python3
"""Structured evaluation schema for the Kuku Stage 4 LLM judge.

Single source of truth for judge outputs: rubric dimensions, weights,
rejection vocabulary, per-dimension evidence requirements, and mechanical
validation of every judgement against the *candidate's own event universe*
(no invented dialogue, IDs, actions or timestamps can survive validation).

Design notes:
* Dimension scores are INTEGERS 0..5. The weighted score is computed here
  (`weighted_score`), never taken from the model.
* Every dimension must be covered by at least one evidence entry, each of
  which names the source event_ids from the payload — including candidate
  events AND its surrounding-context events (never invented).
* `boundary_feedback` is optional and strictly advisory: proposed cut points
  may only reference known event IDs, are validated for ordering, and are
  NEVER applied by the judge or the ranker (timestamp refinement is a
  separate future operation).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

SCHEMA_VERSION = "kuku-judge-schema-1.0.0"
JUDGE_NAME = "llm_judge.py"

# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------

DIMENSIONS: tuple[str, ...] = (
    "opening_hook",
    "cliffhanger",
    "standalone_clarity",
    "narrative_progression",
    "emotional_impact",
)

DIMENSION_WEIGHTS: dict[str, float] = {
    "opening_hook": 0.25,
    "cliffhanger": 0.25,
    "standalone_clarity": 0.20,
    "narrative_progression": 0.15,
    "emotional_impact": 0.15,
}

DIMENSION_LABELS: dict[str, str] = {
    "opening_hook": "Opening hook (25%) — grabs attention within the first 3 seconds",
    "cliffhanger": "Cliffhanger (25%) — the ending leaves a question the viewer must see answered",
    "standalone_clarity": "Standalone clarity (20%) — understandable with zero series knowledge",
    "narrative_progression": "Narrative progression (15%) — a coherent mini-story inside the clip",
    "emotional_impact": "Emotional impact (15%) — a strong reaction: anger, sorrow, shock or curiosity",
}

VERDICT_ACCEPT = "accept"
VERDICT_REJECT = "reject"
VALID_VERDICTS: tuple[str, ...] = (VERDICT_ACCEPT, VERDICT_REJECT)

# Canonical rejection vocabulary. `judge_failure` is produced only when the
# model cannot produce a valid judgement after repairs (never by the model).
REJECTION_REASONS: tuple[str, ...] = (
    "weak_opening",          # first 3 seconds do not grab attention
    "no_cliffhanger",        # nothing unresolved at the cut (question marks don't count)
    "needs_prior_knowledge", # meaning does not survive zero-context viewing
    "incoherent_narrative",  # no setup→escalation→peak inside the clip
    "weak_emotional_impact", # flat; no strong reaction produced
    "not_promotional_moment",# content is not advertisement material for the drama
    "other",
)
INFRA_REJECTION_REASONS: tuple[str, ...] = ("judge_failure",)
ALL_REJECTION_REASONS: tuple[str, ...] = REJECTION_REASONS + INFRA_REJECTION_REASONS

MAX_SCORE = 5
MAX_NOTES_CHARS = 600
MAX_STATEMENT_CHARS = 420
MAX_RATIONALE_CHARS = 420
OBSERVED_ARC_PATTERN = re.compile(r"[a-z0-9_\-]{2,64}")

TOP_LEVEL_KEYS: frozenset[str] = frozenset({
    "scores",
    "verdict",
    "confidence",
    "observed_arc",
    "evidence",
    "rejection_reasons",
    "boundary_feedback",
    "notes",
})
EVIDENCE_KEYS: frozenset[str] = frozenset({"dimension", "statement", "event_ids"})
BOUNDARY_KEYS: frozenset[str] = frozenset({"start_event_id", "end_event_id", "rationale"})


class JudgementValidationError(ValueError):
    """Raised when a judgement fails the contract; carries machine-readable errors."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("Judgement failed validation: " + "; ".join(self.errors))


def weighted_score(scores: dict[str, int]) -> float:
    """Weighted promo-quality score in [0, 5]. Computed by code, never taken from the model."""
    total = sum(float(scores[dimension]) * DIMENSION_WEIGHTS[dimension] for dimension in DIMENSIONS)
    return round(total, 3)


def compute_verdict_from_scores(scores: dict[str, int], accept_threshold: float = 2.8) -> str:
    """Advisory fallback only — production verdicts always come from the model + validation."""
    return VERDICT_ACCEPT if weighted_score(scores) >= accept_threshold else VERDICT_REJECT


def build_event_universe(candidate: dict) -> set[str]:
    """Every event id the judge is allowed to cite: clip contents + both context sides."""
    universe: set[str] = set()
    for key in ("source_event_ids", "context_event_ids"):
        universe.update(candidate.get(key) or [])
    # boundary ids + segment ids are subsets of source; they add nothing new.
    return universe


def _event_times(candidate: dict) -> dict[str, float]:
    """Map known event ids → start_seconds (for boundary ordering checks)."""
    times: dict[str, float] = {}
    for section in ("dialogue_transcript", "actions_and_camera", "preceding_context", "following_context"):
        for item in candidate.get(section) or []:
            event_id = item.get("event_id")
            start = item.get("start_seconds")
            if event_id is not None and start is not None:
                times[event_id] = float(start)
    return times


def _validate_evidence(
    evidence: object,
    universe: set[str],
    errors: list[str],
) -> dict[str, list[dict]]:
    covered: dict[str, list[dict]] = {dimension: [] for dimension in DIMENSIONS}
    if not isinstance(evidence, list) or not evidence:
        errors.append("evidence must be a non-empty list.")
        return covered
    for index, entry in enumerate(evidence):
        where = f"evidence[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{where} must be an object.")
            continue
        unknown = set(entry) - EVIDENCE_KEYS
        if unknown:
            errors.append(f"{where} has unknown keys {sorted(unknown)} (allowed: {sorted(EVIDENCE_KEYS)}).")
        dimension = entry.get("dimension")
        statement = entry.get("statement")
        event_ids = entry.get("event_ids")
        if dimension not in DIMENSIONS:
            errors.append(f"{where}.dimension must be one of {DIMENSIONS}, got {dimension!r}.")
            continue
        if not isinstance(statement, str) or not statement.strip():
            errors.append(f"{where}.statement must be a non-empty string.")
        elif len(statement) > MAX_STATEMENT_CHARS:
            errors.append(f"{where}.statement exceeds {MAX_STATEMENT_CHARS} chars.")
        if not isinstance(event_ids, list) or not event_ids:
            errors.append(f"{where}.event_ids must be a non-empty list.")
            continue
        for event_id in event_ids:
            if not isinstance(event_id, str) or not event_id:
                errors.append(f"{where}.event_ids entries must be non-empty strings.")
            elif event_id not in universe:
                errors.append(f"{where}.event_ids contains unknown id {event_id!r} (would be an invented source).")
        covered[dimension].append({"statement": statement, "event_ids": list(event_ids)})
    for dimension in DIMENSIONS:
        if not covered[dimension]:
            errors.append(f"dimension {dimension!r} is not covered by any evidence entry.")
    return covered


def _validate_boundary_feedback(
    boundary: object,
    universe: set[str],
    candidate: dict,
    errors: list[str],
) -> Optional[dict]:
    if boundary is None:
        return None
    if not isinstance(boundary, dict):
        errors.append("boundary_feedback must be an object or null.")
        return None
    unknown = set(boundary) - BOUNDARY_KEYS
    if unknown:
        errors.append(f"boundary_feedback has unknown keys {sorted(unknown)}.")
        return None
    start_id = boundary.get("start_event_id")
    end_id = boundary.get("end_event_id")
    rationale = boundary.get("rationale")
    valid = True
    for label, event_id in (("start_event_id", start_id), ("end_event_id", end_id)):
        if not isinstance(event_id, str) or event_id not in universe:
            errors.append(f"boundary_feedback.{label} must reference a known event id, got {event_id!r}.")
            valid = False
    if not isinstance(rationale, str) or not rationale.strip():
        errors.append("boundary_feedback.rationale must be a non-empty string.")
        valid = False
    elif len(rationale) > MAX_RATIONALE_CHARS:
        errors.append(f"boundary_feedback.rationale exceeds {MAX_RATIONALE_CHARS} chars.")
        valid = False
    if valid:
        times = _event_times(candidate)
        start_t, end_t = times.get(start_id), times.get(end_id)
        if start_t is not None and end_t is not None and start_t > end_t:
            errors.append(
                f"boundary_feedback is not ordered: start {start_id!r} begins after end {end_id!r}."
            )
            valid = False
    return boundary if valid else None


def validate_judgement(raw: object, candidate: dict) -> dict:
    """Validate + normalize a raw model JSON object against the candidate's world.

    Returns a normalised judgement dict (without weighted_score — add it with
    `attach_weighted_score` after collecting scores) or raises
    JudgementValidationError. Validation failures are the caller's signal to
    re-ask the model (the repair loop) or to record a `judge_failure`.
    """
    errors: list[str] = []
    universe = build_event_universe(candidate)

    if not isinstance(raw, dict):
        raise JudgementValidationError(["top level must be a JSON object."])

    unknown = set(raw) - TOP_LEVEL_KEYS
    if unknown:
        errors.append(f"unknown top-level keys {sorted(unknown)} (allowed: {sorted(TOP_LEVEL_KEYS)}).")

    missing = [key for key in ("scores", "verdict", "confidence", "evidence", "rejection_reasons", "notes", "observed_arc") if key not in raw]
    if missing:
        errors.append(f"missing required keys {missing}.")

    scores = raw.get("scores")
    clean_scores: dict[str, int] = {}
    if not isinstance(scores, dict):
        errors.append("scores must be an object with exactly the five dimensions.")
    else:
        if set(scores) != set(DIMENSIONS):
            errors.append(
                f"scores must have exactly {sorted(DIMENSIONS)}, got {sorted(scores)}."
            )
        for dimension in DIMENSIONS:
            value = scores.get(dimension)
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append(f"scores.{dimension} must be an integer 0..{MAX_SCORE}, got {value!r}.")
                continue
            if not 0 <= value <= MAX_SCORE:
                errors.append(f"scores.{dimension} out of range: {value}.")
                continue
            clean_scores[dimension] = value

    verdict = raw.get("verdict")
    if verdict not in VALID_VERDICTS:
        errors.append(f"verdict must be one of {VALID_VERDICTS}, got {verdict!r}.")

    confidence = raw.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        errors.append(f"confidence must be a number in [0, 1], got {confidence!r}.")
        confidence = None
    elif not 0.0 <= float(confidence) <= 1.0:
        errors.append(f"confidence out of range: {confidence}.")

    observed_arc = raw.get("observed_arc")
    if not isinstance(observed_arc, str) or not OBSERVED_ARC_PATTERN.fullmatch(observed_arc.strip().lower()):
        errors.append("observed_arc must match [a-z0-9_-]{2,64} (short snake_case beat tag).")

    evidence_coverage = _validate_evidence(raw.get("evidence"), universe, errors)

    reasons = raw.get("rejection_reasons")
    if not isinstance(reasons, list):
        errors.append("rejection_reasons must be a list (empty list for accepts).")
        reasons = None
    else:
        for reason in reasons:
            if reason not in REJECTION_REASONS:
                errors.append(f"rejection_reasons contains non-canonical reason {reason!r}.")
        if verdict == VERDICT_REJECT and len(reasons) == 0:
            errors.append("verdict is reject but rejection_reasons is empty.")
        if verdict == VERDICT_ACCEPT and len(reasons) != 0:
            errors.append("verdict is accept but rejection_reasons is non-empty.")

    notes = raw.get("notes")
    if not isinstance(notes, str) or not notes.strip():
        errors.append("notes must be a non-empty string.")
    elif len(notes) > MAX_NOTES_CHARS:
        errors.append(f"notes exceeds {MAX_NOTES_CHARS} chars.")

    boundary = _validate_boundary_feedback(raw.get("boundary_feedback"), universe, candidate, errors)

    if errors:
        raise JudgementValidationError(errors)

    return {
        "scores": clean_scores,
        "verdict": verdict,
        "confidence": round(float(confidence), 4),
        "observed_arc": observed_arc.strip().lower(),
        "evidence": [
            {"dimension": dimension, "statement": entry["statement"].strip(), "event_ids": entry["event_ids"]}
            # preserve model order; coverage keyed by dimension
            for dimension in DIMENSIONS for entry in evidence_coverage[dimension]
        ],
        "rejection_reasons": list(reasons),
        "boundary_feedback": boundary,
        "notes": notes.strip(),
    }


def attach_weighted_score(judgement: dict) -> dict:
    """Return a copy of a validated judgement with the code-computed weighted_score."""
    out = dict(judgement)
    out["weighted_score"] = weighted_score(judgement["scores"])
    return out


def fingerprint_for_cache(candidate: dict) -> str:
    """Stable digest of the content the judge sees (used for cache reuse)."""
    import hashlib

    episode = candidate.get("episode")
    episode = episode if isinstance(episode, dict) else {}
    digest = hashlib.sha1()
    parts = [
        str(candidate.get("candidate_id", "")),
        str(candidate.get("start_timestamp", "")),
        str(candidate.get("end_timestamp", "")),
        json.dumps(candidate.get("dialogue_transcript") or [], sort_keys=True, ensure_ascii=False),
        json.dumps(candidate.get("actions_and_camera") or [], sort_keys=True, ensure_ascii=False),
        json.dumps(candidate.get("preceding_context") or [], sort_keys=True, ensure_ascii=False),
        json.dumps(candidate.get("following_context") or [], sort_keys=True, ensure_ascii=False),
        json.dumps(episode, sort_keys=True, ensure_ascii=False),
    ]
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Orchestration helpers (pure functions so they are shared by judge and ranker)
# ---------------------------------------------------------------------------

def serialize_judgement_jsonl(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=False)
