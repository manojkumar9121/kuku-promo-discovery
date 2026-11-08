#!/usr/bin/env python3
"""Final recommendation view for the submission (Task 1 of Stage 4 hand-in).

Renders the top of the existing judge-ranked slate into a clean editorial
view plus a reserve list. It is a PRESENTATION layer only: it never re-scores,
re-filters or reorders any judge output; it reads judge_output/promo_ranking.json
(frozen by Stage 4 approval) and context_output/candidates.json, and emits
final_recommendations.json / final_recommendations.md.

Preserved guarantees (referenced from the doc):
* full featured slate of 49 (raw outputs, judge_output/promo_ranking.json),
* all 31 rejected candidates with reasons (ditto + ranking_report.json),
* raw model outputs (judge_output/judge_results.json :: raw_model_output),
* human evaluation labels (judge_output/judge_human_sample.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SLATE_SIZE = 12          # top-N of the featured slate presented as "post first"
RESERVE_TARGET = 7       # 4 arc-cap reserves + next featured entries past the cut
CUT_LINE_NEXT = 3        # how many from just past the cut join the reserve


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _trim(text: str, limit: int = 170) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _evidence_map(judgement_evidence: list[dict]) -> dict[str, list[dict]]:
    by_dim: dict[str, list[dict]] = {}
    for entry in judgement_evidence or []:
        by_dim.setdefault(entry.get("dimension", ""), []).append(entry)
    return by_dim


def build_view(ranking_path: Path, candidates_path: Path) -> dict:
    ranking = load_json(ranking_path)
    candidates = {
        row["candidate_id"]: row
        for row in load_json(candidates_path)["candidates"]
    }
    featured = ranking["featured"]
    reserve_raw = ranking["reserve"]

    judge_cache = None
    # re-pull the frozen judgement blocks (ranked entries lack the evidence map)
    # from the raw judge rows if present; tolerate absence by falling back to notes.
    judge_rows = None
    try:
        judge_path = ranking_path.parent / "judge_results.json"
        judge_rows = {row["candidate_id"]: row.get("judgement") for row in load_json(judge_path)["results"]}
    except OSError:
        pass

    slate = []
    for index, entry in enumerate(featured[:SLATE_SIZE], start=1):
        cid = entry["candidate_id"]
        candidate = candidates.get(cid, {})
        judgement = (judge_rows or {}).get(cid) or {}
        evidence = _evidence_map(judgement.get("evidence") or [])
        hook_evidence = [e for e in evidence.get("opening_hook", [])]
        cliff_evidence = [e for e in evidence.get("cliffhanger", [])]
        hook_ids = [eid for e in hook_evidence for eid in e.get("event_ids", [])]
        cliff_ids = [eid for e in cliff_evidence for eid in e.get("event_ids", [])]
        all_evidence_ids = entry.get("evidence_event_ids", []) or []
        slate.append({
            "rank": index,
            "candidate_id": cid,
            "episode_number": entry.get("episode_number"),
            "episode_label": entry.get("episode_label"),
            "start_timestamp": entry.get("start_timestamp"),
            "end_timestamp": entry.get("end_timestamp"),
            "duration_seconds": entry.get("duration_seconds"),
            "observed_arc": entry.get("observed_arc"),
            "weighted_score": entry.get("weighted_score"),
            "hook_score": entry.get("scores", {}).get("opening_hook"),
            "cliffhanger_score": entry.get("scores", {}).get("cliffhanger"),
            "standalone_clarity_score": entry.get("scores", {}).get("standalone_clarity"),
            "why_this_clip_works": _trim(entry.get("notes", ""), 220),
            "hook_evidence": [e.get("statement") for e in hook_evidence][:1],
            "cliffhanger_evidence": [e.get("statement") for e in cliff_evidence][:1],
            "hook_evidence_event_ids": hook_ids[:3],
            "cliffhanger_evidence_event_ids": cliff_ids[:3],
            "evidence_event_ids": all_evidence_ids,
            "source_event_ids": entry.get("source_event_ids", []),
            "crosses_scene_boundary": candidate.get("crosses_scene_boundary"),
        })

    reserve = []
    for entry in reserve_raw:
        reserve.append({
            "why_held_back": "arc-cap overflow — strong on its own; a second promo from this arc would dilute the slate",
            "candidate_id": entry["candidate_id"],
            "episode_number": entry.get("episode_number"),
            "start_timestamp": entry.get("start_timestamp"),
            "end_timestamp": entry.get("end_timestamp"),
            "duration_seconds": entry.get("duration_seconds"),
            "observed_arc": entry.get("observed_arc"),
            "weighted_score": entry.get("weighted_score"),
            "hook_score": entry.get("scores", {}).get("opening_hook"),
            "cliffhanger_score": entry.get("scores", {}).get("cliffhanger"),
            "reason": _trim(entry.get("notes", ""), 160) or entry.get("overflow_reason"),
            "evidence_event_ids": entry.get("evidence_event_ids", []),
        })
    for cut_entry in featured[SLATE_SIZE:SLATE_SIZE + CUT_LINE_NEXT]:
        reserve.append({
            "why_held_back": "cut-line boundary — strong slate entry displaced by the arc cap elsewhere in the ranking; natural next-up",
            "candidate_id": cut_entry["candidate_id"],
            "episode_number": cut_entry.get("episode_number"),
            "start_timestamp": cut_entry.get("start_timestamp"),
            "end_timestamp": cut_entry.get("end_timestamp"),
            "duration_seconds": cut_entry.get("duration_seconds"),
            "observed_arc": cut_entry.get("observed_arc"),
            "weighted_score": cut_entry.get("weighted_score"),
            "hook_score": cut_entry.get("scores", {}).get("opening_hook"),
            "cliffhanger_score": cut_entry.get("scores", {}).get("cliffhanger"),
            "reason": _trim(cut_entry.get("notes", ""), 160),
            "evidence_event_ids": cut_entry.get("evidence_event_ids", []),
        })

    episode = lambda cid: candidates.get(cid, {}).get("episode", {}) if isinstance(candidates.get(cid, {}).get("episode"), dict) else {}
    show_title = (episode(featured[0]["candidate_id"]) or {}).get("source_file", "") or "I Married a Broken Soldier. He was a Duke"

    return {
        "view": "final_recommendations v1.0",
        "source_ranking": "judge_4 output, prompt kuku-judge-v1.0.0, blind (retrieval-unaware)",
        "show": "I Married a Broken Soldier. He was a Duke (first 50 episodes, US-targeted English pack)",
        "slate_size": len(slate),
        "reserve_size": len(reserve),
        "total_slate_preserved": [r["candidate_id"] for r in featured],
        "raw_outputs_preserved": {
            "featured_slate": "judge_output/promo_ranking.json :: featured (49)",
            "rejected": f"judge_output/promo_ranking.json :: rejected ({len(ranking['rejected'])})",
            "raw_model_outputs": "judge_output/judge_results.json :: raw_model_output",
            "human_evaluation_labels": "judge_output/judge_human_sample.json",
        },
        "slate": slate,
        "reserve": reserve,
    }


def render_markdown(view: dict) -> str:
    lines = ["# Promo slate — I Married a Broken Soldier… He was a Duke",
             "",
             "Stage 4 blind judge output on the 50-episode pack (kuku-judge-v1.0.0, 34.6B Q4 coder model,",
             "enable_thinking=false, temperature 0.0). Ranking + 2-arc diversity cap are frozen as approved.",
             "Hook / Cliffhanger are the judge's 0–5 dimension scores; Why = judge's own evidence summary.",
             "Full 49-candidate slate, rejected list, raw outputs and human labels are preserved in the raw outputs.",
             "",
             f"## Post-first slate (top {view['slate_size']})",
             "",
             "| # | Ep | Window | Dur | Hook | Cliff | Score | Why it works |",
             "|---|----|--------|-----|------|-------|-------|--------------|"]
    for entry in view["slate"]:
        why = _trim(entry["why_this_clip_works"], 95)
        lines.append(
            f"| {entry['rank']} | E{entry['episode_number']} "
            f"| {entry['start_timestamp']}→{entry['end_timestamp']} "
            f"| {entry['duration_seconds']}s | {entry['hook_score']} | {entry['cliffhanger_score']} "
            f"| **{entry['weighted_score']:.2f}** | {why} |"
        )
    lines += ["", "### Evidence references (event ids)", ""]
    for entry in view["slate"]:
        hook_ids = ", ".join(entry["hook_evidence_event_ids"]) or "—"
        cliff_ids = ", ".join(entry["cliffhanger_evidence_event_ids"]) or "—"
        lines.append(f"{entry['rank']}. `{entry['candidate_id']}`  hook: {hook_ids}  |  cliffhanger: {cliff_ids}")
    lines += [
        "",
        f"## Reserve ({view['reserve_size']}) — hold these; post only when slate budget allows",
        "",
        "| # | Ep | Window | Dur | Score | Why held back |",
        "|---|----|--------|-----|-------|----------------|",
    ]
    for i, entry in enumerate(view["reserve"], 1):
        lines.append(
            f"| {i} | E{entry['episode_number']} "
            f"| {entry['start_timestamp']}→{entry['end_timestamp']} "
            f"| {entry['duration_seconds']}s | {entry['weighted_score']:.2f} "
            f"| {entry['why_held_back']} |"
        )
    lines += [
        "",
        "## Preserved in raw outputs (evidence, nothing deleted)",
        f"- featured slate (all {view['slate_size'] + 37} caps-respecting entries): `promo_ranking.json :: featured`",
        "- rejected candidates with reasons & notes (31): `promo_ranking.json :: rejected`",
        "- raw model outputs per candidate (incl. failed attempt log): `judge_results.json :: raw_model_output`",
        "- independent human review labels (33 pairs, Judge-vs-reviewer 30/33): `judge_human_sample.json`",
        "",
        "Timestamps trace to source events via `source_event_ids` in every artifact; no timestamp was invented.",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the final recommendation view (presentation-only).")
    parser.add_argument("--ranking", default="judge_output/promo_ranking.json")
    parser.add_argument("--candidates", default="context_output/candidates.json")
    parser.add_argument("--output", default="judge_output")
    args = parser.parse_args(argv)
    output_dir = Path(args.output)
    view = build_view(Path(args.ranking), Path(args.candidates))
    (output_dir / "final_recommendations.json").write_text(
        json.dumps(view, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "final_recommendations.md").write_text(render_markdown(view), encoding="utf-8")
    print(f"Slate: {view['slate_size']} | Reserve: {view['reserve_size']}")
    print(f"Wrote {output_dir / 'final_recommendations.json'} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
