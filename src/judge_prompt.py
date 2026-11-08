#!/usr/bin/env python3
"""Prompt assembly for the Kuku Stage 4 LLM judge.

Versioned so judgements are traceable to an exact prompt contract:
JUDGE_PROMPT_VERSION is embedded in every result and the cache key.

Blind evaluation: the user payload deliberately OMITS retrieval categories,
scores and rankings (the default, and the only mode used for the production
run). `include_retrieval_metadata=True` exists purely for experiments and is
recorded in the outputs when used.

Evidence contract: the model must support every decisive judgement with the
payload's real event_ids — never with invented ones. Mechanical validation in
`judge_schema.py` cross-checks every cited id against the candidate's own
event universe, so unsupported claims cannot survive parsing.
"""

from __future__ import annotations

import json
from typing import Optional

import judge_schema as js

JUDGE_PROMPT_VERSION = "kuku-judge-v1.0.0"

# Content clipped for context sections: enough to read the beat, not the whole episode.
CONTEXT_LINE_CHARS = 200
MAX_CONTEXT_LINES = 5
MAX_CAST_ENTRIES = 10
MAX_CONTEXT_DELIVERY_CHARS = 40

SYSTEM_PROMPT = """\
You are Kuku's senior promo editor. You judge 20-60 second vertical promo\
candidates cut from serialized microdrama screenplays. For each candidate \
you return a single JSON object that tells the ad-buy whether this clip is \
worth spending a media budget on, and why.

THE CANDIDATE IS JUDGED STANDALONE. A stranger who has never seen the series\
must be hooked, understand the stakes, and want the next episode. Internal \
series facts (who Serena's father is, what happened in episode 9) are NOT \
ours to lean on.

RULES OF JUDGEMENT (non-negotiable):
1. Use ONLY the payload. Never invent dialogue, actions, timestamps, order of \
   events or character relationships. If a relationship is not signalled by the \
   clip or the logline/continuity notes, do not assume it.
2. Standalone clarity is scored on the CLIP ALONE. The surrounding context is\
   there so you understand what happens — do NOT use it to paper over \
   confusion. If a cold viewer could not follow, standalone_clarity must fall.
3. Betrayal ≠ coercion ≠ threat ≠ accusation. Betrayal is loyalty or love \
   broken by someone close: the chosen sister, the vow broken at the altar. A \
   strong man squeezing a servant into false testimony is coercion; an armed \
   guard and a raised voice are threat; "you have been deceived" is an \
   accusation until proven. Judge what actually happens in the clip.
4. A shocking line is not a promo. Shock must hold inside a clear mini-story \
   with a reason to watch on; otherwise it is noise.
5. A rhetorical question mark is NOT a cliffhanger. Cliffhanger means a real \
   unanswered question the cut weaponises: an accusation unanswered, a demand \
   with no reply, a door opening on something hidden, a verdict mid-swing.
6. A clip that ends mid-beat with a question can be a 4 or 5 EVEN IF the \
   preceding 25 seconds are inert; but a weak OPENING alone disqualifies the \
   clip regardless of a strong ending — mark first-seconds quality down and \
   reject if there is no grip in the first 3 seconds at all.
7. Prior knowledge MUST NOT be required. Clips that only land if the viewer \
   knows episodes 1-3, an off-screen letter, or who a "scarred nobody" is in \
   the arc, must be scored down and rejected.
8. Rejection is a valid and expected verdict. Roughly half the pool may be \
   unusable. Do not be polite about it.
9. Every decisive judgement must carry the payload's real event_ids. Every \
   one of the five dimensions needs at least one evidence entry. Cite ids \
   exactly as given.
10. Do NOT propose timestamps. If you think a different cut would be dramatic-\
   ally stronger, put it in boundary_feedback using the payload's own event \
   ids — as advisory feedback only; it is never applied during this pass.
11. observed_arc is a short snake_case tag for the dramatic beat (e.g. \
   altar_abandonment, coerced_testimony, resurrection_reveal, dungeon_reckoning). \
   Use one of your own devising consistently across the run.

RUBRIC (integers 0-5 each; the system computes the weighted score, you never do):

opening_hook — does the first 3 seconds grab an uninterested swiper?
  0: dead air (silent corridor, report reading, walking).
  1: polite exchange, neutral.
  2: mild unrest; some tension but no snap.
  3: tension is clear within seconds.
  4: immediate confrontation / accusation / command in-frame.
  5: instant shock — a slap, a revelation, an abandonment at the altar.

cliffhanger — does the cut leave a question the viewer must see answered?
  0: fully resolved; closure inside the clip.
  1: information delivered; nothing at stake forward.
  2: a mild dangling thread, not urgent.
  3: a clear open question, mildly urgent.
  4: sharp unresolved tension at the cut; accusation unanswered.
  5: hard mid-moment cut that stops the pulse — viewer physically cannot not \
     watch the next second.

standalone_clarity — can a cold viewer follow WHO/WHAT/WHY within the clip?
  0: meaningless without series context.
  1: scattered; viewer has no idea who wronged whom.
  2: one fact lands, the rest is guesswork.
  3: stakes visible but names/relations partly assumed.
  4: clear wronged vs wrong, stakes obvious mid-clip.
  5: anyone can parse it in five seconds cold.

narrative_progression — mini-story with setup -> escalation -> peak?
  0: static; nothing develops.
  1: drifting conversation; no escalation.
  2: mild build with a single beat.
  3: clear two-step arc (setup then turn).
  4: setup -> escalation -> peak structure present.
  5: textbook mini-arc with a gut-punch at the crest.

emotional_impact — does the clip create a strong reaction?
  0: no feeling produced.
  1: faint acknowledgment.
  2: mild sympathy or curiosity.
  3: a clear reaction (anger, pity, dread).
  4: strong anger/shock/ache at the peak.
  5: ferocious reaction — the kind that makes a thumb hover over play.

OUTPUT CONTRACT — ONLY this JSON object, nothing before or after, no code fences:
{
  "scores": {"opening_hook": <0-5 int>, "cliffhanger": <0-5 int>,
             "standalone_clarity": <0-5 int>, "narrative_progression": <0-5 int>,
             "emotional_impact": <0-5 int>},
  "verdict": "accept" | "reject",
  "confidence": <0.0-1.0>,
  "observed_arc": "<snake_case beat tag>",
  "evidence": [
    {"dimension": "<dimension name>", "statement": "<<=400 chars>", "event_ids": ["<id>", ...]},
    ...
  ],
  "rejection_reasons": ["<one or more of: weak_opening, no_cliffhanger, needs_prior_knowledge, incoherent_narrative, weak_emotional_impact, not_promotional_moment>"],
  "boundary_feedback": {"start_event_id": "<id>", "end_event_id": "<id>", "rationale": "<short>"} or null,
  "notes": "<=600 chars on what a buyer should know"
}
rejection_reasons MUST be an empty list when verdict = "accept" and a non-empty list when "reject".\
""".strip()


def _clip(text: object, limit: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _window_fields(candidate: dict) -> dict:
    return {
        "start_timestamp": candidate.get("start_timestamp"),
        "end_timestamp": candidate.get("end_timestamp"),
        "duration_seconds": candidate.get("duration_seconds"),
        "start_seconds": candidate.get("start_seconds"),
        "end_seconds": candidate.get("end_seconds"),
    }


def _event_rows(items: list[dict], kinds: tuple[str, ...]) -> list[dict]:
    """Project event-like dicts down to the payload's minimal rows."""
    rows = []
    for item in items or []:
        event_type = item.get("event_type") or ""
        if event_type not in kinds:
            continue
        speaker = item.get("speaker_canonical") or item.get("speaker") or None
        rows.append({
            "event_id": item.get("event_id"),
            "timestamp": item.get("start_timestamp"),
            "type": event_type,
            **({"speaker": speaker} if speaker else {}),
            **({"delivery": _clip(item.get("delivery"), MAX_CONTEXT_DELIVERY_CHARS)} if item.get("delivery") else {}),
            "content": _clip(item.get("content"), 400),
        })
    return rows


def _context_rows(items: list[dict]) -> list[dict]:
    """Compressed preceding/following context: the tail of the past / head of the future."""
    tail = (items or [])[-MAX_CONTEXT_LINES:] if len(items or []) > MAX_CONTEXT_LINES else (items or [])
    rows = []
    for item in tail:
        speaker = item.get("speaker_canonical") or item.get("speaker")
        rows.append({
            "event_id": item.get("event_id"),
            "timestamp": item.get("start_timestamp"),
            "type": item.get("event_type"),
            **({"speaker": speaker} if speaker else {}),
            "content": _clip(item.get("content"), CONTEXT_LINE_CHARS),
        })
    return rows


def _context_following_rows(items: list[dict]) -> list[dict]:
    head = (items or [])[:MAX_CONTEXT_LINES]
    rows = []
    for item in head:
        speaker = item.get("speaker_canonical") or item.get("speaker")
        rows.append({
            "event_id": item.get("event_id"),
            "timestamp": item.get("start_timestamp"),
            "type": item.get("event_type"),
            **({"speaker": speaker} if speaker else {}),
            "content": _clip(item.get("content"), CONTEXT_LINE_CHARS),
        })
    return rows


def _cast_rows(candidate: dict) -> list[dict]:
    rows = []
    for entry in (candidate.get("speakers") or [])[:MAX_CAST_ENTRIES]:
        canonical = entry.get("speaker_canonical") or entry.get("speaker")
        deliveries = entry.get("deliveries") or []
        rows.append({
            "speaker": canonical,
            "scene_lines": entry.get("dialogue_event_count"),
            **({"voice_flavours": deliveries[:4]} if deliveries else {}),
        })
    return rows


def build_payload(candidate: dict, include_retrieval_metadata: bool = False) -> dict:
    """Assemble the judge payload for one candidate (canonical dict, json.dumps-able)."""
    episode = candidate.get("episode") if isinstance(candidate.get("episode"), dict) else {}
    payload = {
        "candidate_id": candidate.get("candidate_id"),
        "episode_number": candidate.get("episode_number"),
        "episode_label": episode.get("episode_label"),
        "episode_logline": _clip(episode.get("logline"), 600),
        "episode_continuity_notes": _clip(episode.get("continuity_notes"), 500),
        "scene_headings": [
            _clip(scene.get("heading"), 120)
            for scene in (candidate.get("scenes") or [])
        ],
        "window": _window_fields(candidate),
        "cast_in_window": _cast_rows(candidate),
        "dialogue": _event_rows(candidate.get("dialogue_transcript") or [], ("dialogue",)),
        "staging": _event_rows(candidate.get("actions_and_camera") or [],
                               ("camera", "action", "sound_fx")),
        "narrative_preceding": _context_rows(candidate.get("preceding_context") or []),
        "narrative_following": _context_following_rows(candidate.get("following_context") or []),
    }
    if include_retrieval_metadata:
        payload["retrieval_metadata"] = {
            "note": "Anchoring metadata deliberately excluded from the production (blind) pass.",
            "best_categories": candidate.get("best_categories"),
            "similarity_scores_all_categories": candidate.get("similarity_scores_all_categories"),
        }
    return payload


def build_user_message(candidate: dict, include_retrieval_metadata: bool = False) -> str:
    payload = build_payload(candidate, include_retrieval_metadata=include_retrieval_metadata)
    return (
        "Judge this candidate for Kuku promo use. Return ONLY the contracted JSON object.\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def available_event_ids(candidate: dict) -> list[str]:
    ids: list[str] = []
    for key in ("dialogue_transcript", "actions_and_camera", "preceding_context", "following_context"):
        for item in candidate.get(key) or []:
            if item.get("event_id"):
                ids.append(item["event_id"])
    return ids
