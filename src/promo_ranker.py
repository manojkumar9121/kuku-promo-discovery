#!/usr/bin/env python3
"""Kuku Stage 4 — Promo ranking engine.

Two separate, non-negotiable ranking considerations:

* **Quality** — the LLM judge's weighted rubric score (plus confidence as
  tie-break) order accepted candidates. The rejected ones are RETAINED with
  their rejection explanations, not dropped.
* **Diversity** — the recommended shortlist must not collapse onto a single
  story arc / episode. A configurable arc cap (default 2 per
  (episode, observed_arc)) governs which candidates are marked ``featured``
  (the recommended promo slate) vs ``reserve`` (overflow, kept in full with
  reasons). Quality ordering itself is never altered by the diversity pass —
  diversity selects WHICH of the best to recommend first, it does not fake a
  quality ranking.

Near-duplicate detection (same episode + overlapping windows or identical arc
tag with meaningful overlap) is reported as clusters so buyers can see when
five recommendations all trace to the same wedding-morning beat.

Also produces the embedding-vs-judge comparison over the shared candidate set.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Optional

REPORT_SCHEMA_VERSION = "kuku-rank-report-1.0.0"
RANKING_SCHEMA_VERSION = "kuku-promo-ranking-1.0.0"

DEFAULT_MAX_PER_ARC = 2
DEFAULT_DUPLICATE_IOU = 0.5
DEFAULT_ARC_WITH_OVERLAP_IOU = 0.3


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_candidates(path: Path) -> dict[str, dict]:
    payload = _load_json(path)
    rows = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"No candidates list in {path}")
    return {row["candidate_id"]: row for row in rows}


def load_judge_results(path: Path) -> list[dict]:
    payload = _load_json(path)
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"No results list in {path}")
    return rows


def load_retrieval_results(path: Path) -> dict[str, dict]:
    payload = _load_json(path)
    rows = payload.get("results") or []
    return {row["candidate_id"]: row for row in rows if row.get("candidate_id")}


# ---------------------------------------------------------------------------
# Ranking math
# ---------------------------------------------------------------------------

def split_acceptance(judged_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    accepted, rejected = [], []
    for row in judged_rows:
        if row.get("status") != "ok":
            continue
        judgement = row["judgement"]
        if judgement["verdict"] == "accept":
            accepted.append(row)
        else:
            rejected.append(row)
    return accepted, rejected


def quality_sort(accepted: list[dict]) -> list[dict]:
    return sorted(
        accepted,
        key=lambda row: (
            -row["judgement"]["weighted_score"],
            -row["judgement"]["confidence"],
            str(row.get("candidate_id")),
        ),
    )


def window_iou(a: dict, b: dict) -> float:
    a_start, a_end = float(a.get("start_seconds", 0.0)), float(a.get("end_seconds", 0.0))
    b_start, b_end = float(b.get("start_seconds", 0.0)), float(b.get("end_seconds", 0.0))
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0 else 0.0


def _union_find_connect(nodes: list, edges: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(len(nodes)))

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != root:
            parent[index], index = root, parent[index]
        return root

    for first, second in edges:
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[root_a] = root_b
    groups: dict[int, list[int]] = {}
    for index in range(len(nodes)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def detect_near_duplicates(
    accepted: list[dict],
    candidates_map: dict[str, dict],
    iou_threshold: float,
) -> list[dict]:
    """Cluster accepted candidates that are effectively the same moment."""
    n = len(accepted)
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            row_a, row_b = accepted[i], accepted[j]
            cand_a = candidates_map.get(row_a["candidate_id"])
            cand_b = candidates_map.get(row_b["candidate_id"])
            if cand_a is None or cand_b is None:
                continue
            if cand_a.get("episode_number") != cand_b.get("episode_number"):
                continue
            iou = window_iou(cand_a, cand_b)
            same_arc = (
                row_a["judgement"]["observed_arc"] == row_b["judgement"]["observed_arc"]
            )
            if iou >= iou_threshold or (same_arc and iou >= DEFAULT_ARC_WITH_OVERLAP_IOU):
                edges.append((i, j))

    clusters = []
    cluster_id = 0
    for group in _union_find_connect(accepted, edges):
        members = [accepted[index] for index in group]
        if len(members) < 2:
            continue
        cluster_id += 1
        ordered = quality_sort(members)
        head = ordered[0]
        window = lambda item: _window_label(candidates_map[item["candidate_id"]])
        clusters.append({
            "cluster_id": f"cluster-{cluster_id:02d}",
            "episode_number": candidates_map[head["candidate_id"]].get("episode_number"),
            "observed_arc": head["judgement"]["observed_arc"],
            "head_candidate_id": head["candidate_id"],
            "member_candidate_ids": [item["candidate_id"] for item in ordered],
            "member_windows": [window(item) for item in ordered],
            "note": "Near-identical cut windows or identical arc tag on an overlapping moment — recommend at most the head.",
        })
    return clusters


def _window_label(candidate: dict) -> str:
    return f"{candidate.get('start_timestamp')}→{candidate.get('end_timestamp')}"


def assign_featured_and_reserve(
    accepted_sorted: list[dict],
    candidates_map: dict[str, dict],
    max_per_arc: int,
) -> tuple[list[dict], list[dict]]:
    """Walk quality-sorted accepted list; cap per (episode, observed_arc).

    Featured = everyone within the cap at the time of walking (recommended
    slate, in quality order); everyone past the cap is marked overflow with a
    reason. Both lists preserve quality order — diversity is applied in the
    walk, it does not reorder quality.
    """
    featured: list[dict] = []
    reserve: list[dict] = []
    arc_counts: dict[tuple[int, str], int] = {}
    for row in accepted_sorted:
        candidate = candidates_map.get(row["candidate_id"], {})
        key = (int(candidate.get("episode_number") or candidate.get("episode_number") or 0),
               row["judgement"]["observed_arc"])
        count = arc_counts.get(key, 0)
        entry = dict(row)
        entry["recommendation_tier"] = "featured" if count < max_per_arc else "reserve"
        entry["overflow_reason"] = None if count < max_per_arc else (
            f"arc_cap_reached: >= {max_per_arc} recommended candidates already assigned to arc "
            f"'{key[1]}' on episode {key[0]}"
        )
        entry["arc_key"] = f"{key[0]}:{key[1]}"
        if count < max_per_arc:
            featured.append(entry)
            arc_counts[key] = count + 1
        else:
            reserve.append(entry)
    return featured, reserve


# ---------------------------------------------------------------------------
# Statistics + comparisons
# ---------------------------------------------------------------------------

def mean_or_none(values: list[float]) -> Optional[float]:
    return round(sum(values) / len(values), 3) if values else None


def percentile_or_none(values: list[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    interpolated = ordered[lower] + (ordered[upper] - ordered[lower]) * weight
    return round(float(interpolated), 3)


def spearman_rho(scores_a: list[float], scores_b: list[float]) -> Optional[float]:
    """Average-rank Spearman over aligned arrays (ties share the average rank)."""
    if len(scores_a) != len(scores_b) or len(scores_a) < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        indexed = sorted(enumerate(values), key=lambda pair: pair[1])
        rank_values = [0.0] * len(values)
        index = 0
        while index < len(indexed):
            group = [indexed[index]]
            while index + 1 < len(indexed) and indexed[index + 1][1] == indexed[index][1]:
                index += 1
                group.append(indexed[index])
            average = sum(position + 1 for position in range(index - len(group) + 1, index + 1)) / len(group)
            for original_index, _ in group:
                rank_values[original_index] = average
            index += 1
        return rank_values

    r_a, r_b = ranks(scores_a), ranks(scores_b)
    n = len(r_a)
    mean_a, mean_b = sum(r_a) / n, sum(r_b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(r_a, r_b))
    var_a = sum((x - mean_a) ** 2 for x in r_a)
    var_b = sum((y - mean_b) ** 2 for y in r_b)
    if var_a == 0 or var_b == 0:
        return None
    return round(cov / (var_a * var_b) ** 0.5, 4)


def jaccard(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 1.0


def embedding_rank_scores(
    retrieval_rows: dict[str, dict],
    candidate_ids: list[str],
    mode: str = "max_category",
) -> dict[str, float]:
    scores = {}
    for cid in candidate_ids:
        row = retrieval_rows.get(cid)
        if row is None:
            continue
        similarity = row.get("similarity_scores_all_categories") or {}
        if not similarity:
            scores[cid] = 0.0
            continue
        if mode == "max_category":
            scores[cid] = float(max(similarity.values()))
        elif mode == "mean_category":
            scores[cid] = float(sum(similarity.values()) / len(similarity))
        else:
            raise ValueError(f"Unknown embedding rank mode {mode!r}")
    return scores


def build_embedding_vs_judge(
    ok_rows: list[dict],
    candidates_map: dict[str, dict],
    retrieval_rows: dict[str, dict],
    top_ks: tuple[int, ...] = (10, 20),
) -> dict:
    ids = [row["candidate_id"] for row in ok_rows]
    judge_scores = {row["candidate_id"]: row["judgement"]["weighted_score"] for row in ok_rows}
    embed_scores = embedding_rank_scores(retrieval_rows, ids, mode="max_category")
    aligned = [(cid, judge_scores[cid], embed_scores.get(cid)) for cid in ids]
    aligned = [(cid, j, e) for cid, j, e in aligned if e is not None]
    judge_order = [cid for cid, _, _ in sorted(aligned, key=lambda r: (-r[1], r[0]))]
    embed_order = [cid for cid, _, _ in sorted(aligned, key=lambda r: (-r[2], r[0]))]

    spearman = spearman_rho([row[1] for row in aligned], [row[2] for row in aligned])
    top_k = {}
    for k in top_ks:
        set_j = set(judge_order[:k])
        set_e = set(embed_order[:k])
        top_k[f"top_{k}"] = {
            "jaccard": round(jaccard(set_j, set_e), 4),
            "judge_only": sorted(set_j - set_e),
            "embed_only": sorted(set_e - set_j),
        }
    judge_rank = {cid: index + 1 for index, cid in enumerate(judge_order)}
    embed_rank = {cid: index + 1 for index, cid in enumerate(embed_order)}
    disagreements = sorted(
        (
            {
                "candidate_id": cid,
                "judge_rank": judge_rank[cid],
                "embedding_rank": embed_rank[cid],
                "rank_delta_judge_minus_embedding": judge_rank[cid] - embed_rank[cid],
                "judge_weighted_score": judge_scores[cid],
                "embedding_best_category_score": embed_scores[cid],
                "judge_verdict": next(r["judgement"]["verdict"] for r in ok_rows if r["candidate_id"] == cid),
            }
            for cid in ids
            if cid in embed_scores
        ),
        key=lambda item: -abs(item["rank_delta_judge_minus_embedding"]),
    )
    return {
        "population": len(aligned),
        "spearman_rho": spearman,
        "jaccard_top_k": top_k,
        "embedding_order_top10": embed_order[:10],
        "judge_order_top10": judge_order[:10],
        "largest_disagreements": disagreements[:20],
    }


def build_human_review_overlay(
    ranked_rows: list[dict],
    labels_path: Optional[Path],
    top_ks: tuple[int, ...] = (5, 10, 20),
) -> Optional[dict]:
    """Optional human-promo-quality overlay for the independent sample.

    `ranked_rows` is the final `ranked` list from ranking (featured slate then
    reserve), already quality+diversity ordered. Labels:
    {candidate_id: "strong"|"usable"|"weak"|"unusable"}. Precision counts
    strong+usable as hits. This overlay is built AFTER the ranking and is
    NEVER used to tune the prompt or selection weights.
    """
    if labels_path is None or not labels_path.exists():
        return None
    labels = _load_json(labels_path)
    if not isinstance(labels, dict) or not labels:
        return None
    accepted_order = [row["candidate_id"] for row in ranked_rows]
    hit_labels = {"strong", "usable"}
    report: dict[str, dict] = {}
    for k in top_ks:
        hits = sum(1 for cid in accepted_order[:k] if labels.get(cid) in hit_labels)
        report[f"precision_at_{k}"] = {"hits": hits, "of_top": min(k, len(accepted_order)), "value": round(hits / min(k, len(accepted_order)), 4) if accepted_order else None}
    judge_ok_top20 = accepted_order[:20]
    verdicts = {cid: labels.get(cid, "unlabelled") for cid in judge_ok_top20}
    overall = {
        f"{label}": sum(1 for v in verdicts.values() if v == label)
        for label in sorted(set(verdicts.values()))
    }
    return {
        "labels_file": str(labels_path),
        "precision": report,
        "top20_verdict_histogram": overall,
        "note": "Independent review; no prompt tuning was performed against these labels.",
    }


# ---------------------------------------------------------------------------
# Ranking + report assembly
# ---------------------------------------------------------------------------

def rank_candidates(
    judged_rows: list[dict],
    candidates_map: dict[str, dict],
    retrieval_rows: dict[str, dict],
    max_per_arc: int = DEFAULT_MAX_PER_ARC,
) -> dict:
    accepted_raw, rejected_raw = split_acceptance(judged_rows)
    failed = [row for row in judged_rows if row.get("status") != "ok"]
    accepted_sorted = quality_sort(accepted_raw)

    clusters = detect_near_duplicates(
        accepted_sorted, candidates_map, iou_threshold=DEFAULT_DUPLICATE_IOU
    )
    featured, reserve = assign_featured_and_reserve(accepted_sorted, candidates_map, max_per_arc)

    def joined_entry(row: dict, tier: str) -> dict:
        candidate = candidates_map.get(row["candidate_id"], {})
        episode = candidate.get("episode") if isinstance(candidate.get("episode"), dict) else {}
        judgement = row["judgement"]
        return {
            "candidate_id": row["candidate_id"],
            "episode_number": candidate.get("episode_number"),
            "episode_label": episode.get("episode_label"),
            "start_timestamp": candidate.get("start_timestamp"),
            "end_timestamp": candidate.get("end_timestamp"),
            "duration_seconds": candidate.get("duration_seconds"),
            "observed_arc": judgement["observed_arc"],
            "arc_key": row.get("arc_key"),
            "weighted_score": judgement["weighted_score"],
            "confidence": judgement["confidence"],
            "scores": judgement["scores"],
            "verdict": judgement["verdict"],
            "notes": judgement["notes"],
            "evidence_dimensions": sorted({e["dimension"] for e in judgement.get("evidence", [])}),
            "evidence_event_ids": [eid for e in judgement.get("evidence", []) for eid in e["event_ids"]],
            "source_event_ids": candidate.get("source_event_ids", []),
            "boundary_feedback": judgement.get("boundary_feedback"),
            "tier": tier,
            "overflow_reason": row.get("overflow_reason"),
        }

    featured_rows = [joined_entry(row, "featured") for row in featured]
    reserve_rows = [joined_entry(row, "reserve") for row in reserve]
    ranked_rows = featured_rows + reserve_rows

    rejected_rows = []
    rejection_histogram: dict[str, int] = {}
    for row in rejected_raw:
        judgement = row["judgement"]
        episode = (candidates_map.get(row["candidate_id"], {}) or {}).get("episode") or {}
        for reason in judgement.get("rejection_reasons", []):
            rejection_histogram[reason] = rejection_histogram.get(reason, 0) + 1
        rejected_rows.append({
            "candidate_id": row["candidate_id"],
            "episode_number": (candidates_map.get(row["candidate_id"], {}) or {}).get("episode_number"),
            "episode_label": episode.get("episode_label"),
            "start_timestamp": (candidates_map.get(row["candidate_id"], {}) or {}).get("start_timestamp"),
            "end_timestamp": (candidates_map.get(row["candidate_id"], {}) or {}).get("end_timestamp"),
            "weighted_score": judgement["weighted_score"],
            "verdict": judgement["verdict"],
            "rejection_reasons": judgement["rejection_reasons"],
            "notes": judgement["notes"],
            "observed_arc": judgement["observed_arc"],
        })
    rejected_rows.sort(key=lambda r: (-r["weighted_score"], r["candidate_id"]))

    failed_rows = [
        {"candidate_id": row.get("candidate_id"), "error": row.get("error") or "judge failure"}
        for row in failed
    ]

    # ---- stats ------------------------------------------------------------
    ok_rows = accepted_raw + rejected_raw
    scores = [row["judgement"]["weighted_score"] for row in ok_rows]
    dimension_means = {}
    for dimension in ("opening_hook", "cliffhanger", "standalone_clarity", "narrative_progression", "emotional_impact"):
        dimension_means[dimension] = mean_or_none([row["judgement"]["scores"][dimension] for row in ok_rows])

    arc_counts_rows: dict[str, int] = {}
    for row in featured_rows:
        arc_counts_rows[row["arc_key"]] = arc_counts_rows.get(row["arc_key"], 0) + 1
    episode_counts: dict[int, int] = {}
    for row in featured_rows:
        episode_counts[int(row["episode_number"] or 0)] = episode_counts.get(int(row["episode_number"] or 0), 0) + 1

    embedding_vs_judge = build_embedding_vs_judge(ok_rows, candidates_map, retrieval_rows)

    return {
        "schema_version": RANKING_SCHEMA_VERSION,
        "config": {"max_per_arc": max_per_arc,
                   "acceptance_rule": "verdict == accept (judge-computed, validated)",
                   "quality_sort": ["weighted_score desc", "confidence desc", "candidate_id asc"],
                   "diversity": "featured requires arc_count < max_per_arc for (episode_number, observed_arc)"},
        "featured": featured_rows,
        "reserve": reserve_rows,
        "rejected": rejected_rows,
        "failed": failed_rows,
        "ranked": ranked_rows,
        "near_duplicate_clusters": clusters,
        "arc_statistics": [
            {"arc_key": key, "featured_count": value} for key, value in sorted(arc_counts_rows.items())
        ],
        "episode_statistics": [
            {"episode_number": key, "featured_count": value} for key, value in sorted(episode_counts.items())
        ],
        "diversity_summary": {
            "featured_total": len(featured_rows),
            "featured_distinct_arcs": len(arc_counts_rows),
            "featured_distinct_episodes": len(episode_counts),
            "reserve_total": len(reserve_rows),
        },
        "score_statistics": {
            "n_judged_ok": len(ok_rows),
            "mean": mean_or_none(scores),
            "median": round(median(scores), 3) if scores else None,
            "p10": percentile_or_none(scores, 0.1),
            "p90": percentile_or_none(scores, 0.9),
            "min": round(min(scores), 3) if scores else None,
            "max": round(max(scores), 3) if scores else None,
            "dimension_means": dimension_means,
        },
        "verdict_statistics": {
            "accepted": len(accepted_raw),
            "rejected": len(rejected_raw),
            "failed": len(failed_rows),
            "rejection_reason_histogram": dict(sorted(rejection_histogram.items(), key=lambda kv: (-kv[1], kv[0]))),
        },
        "embedding_vs_judge": embedding_vs_judge,
    }


def render_ranking_markdown(ranking: dict) -> str:
    lines = ["# Promo ranking — Kuku Stage 4 (LLM-judged)", ""]
    lines.append(f"Featured (max {ranking['config']['max_per_arc']} per episode+arc): "
                 f"{len(ranking['featured'])} | Reserve: {len(ranking['reserve'])} | "
                 f"Rejected: {len(ranking['rejected'])} | Failed: {len(ranking['failed'])}")
    lines.append("")
    lines.append("## Featured slate (quality order)")
    for index, row in enumerate(ranking["featured"], start=1):
        lines.append(
            f"{index}. **{row['candidate_id']}** ep{row['episode_number']} "
            f"[{row['start_timestamp']} → {row['end_timestamp']}] "
            f"arc `{row['observed_arc']}` — score {row['weighted_score']:.3f} "
            f"(conf {row['confidence']:.2f}) — {row['notes'][:150]}"
        )
    lines.append("")
    lines.append("## Reserve (diversity overflow — kept in quality order)")
    for index, row in enumerate(ranking["reserve"], start=1):
        lines.append(f"- **{row['candidate_id']}** ep{row['episode_number']} — score {row['weighted_score']:.3f} — {row['overflow_reason']}")
    lines.append("")
    lines.append("## Near-duplicate clusters")
    for cluster in ranking["near_duplicate_clusters"]:
        lines.append(
            f"- {cluster['cluster_id']} ep{cluster['episode_number']} arc "
            f"`{cluster['observed_arc']}` — recommended head **{cluster['head_candidate_id']}**; "
            f"members: {', '.join(cluster['member_candidate_ids'])}"
        )
    if not ranking["near_duplicate_clusters"]:
        lines.append("- none")
    lines.append("")
    lines.append("## Rejection reasons")
    for reason, count in ranking["verdict_statistics"]["rejection_reason_histogram"].items():
        lines.append(f"- {reason}: {count}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rank judged promo candidates (quality + diversity).")
    parser.add_argument("--judge-results", default="judge_output/judge_results.json")
    parser.add_argument("--candidates", default="context_output/candidates.json")
    parser.add_argument("--retrieval-results", default="retrieval_output/fast_discovery_results.json")
    parser.add_argument("--output", default="judge_output")
    parser.add_argument("--max-per-arc", type=int, default=DEFAULT_MAX_PER_ARC)
    parser.add_argument("--human-labels", default="judge_output/judge_human_sample.json",
                        help="Optional independent human-promo-quality labels for precision reporting (never used at selection time)")
    args = parser.parse_args(argv)

    try:
        judge_path = Path(args.judge_results)
        candidates = load_candidates(Path(args.candidates))
        retrieval_rows = load_retrieval_results(Path(args.retrieval_results))
        judged_rows = load_judge_results(judge_path)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    ranking = rank_candidates(judged_rows, candidates, retrieval_rows, max_per_arc=args.max_per_arc)

    overlay = build_human_review_overlay(ranking["ranked"], Path(args.human_labels) if args.human_labels else None)
    if overlay is not None:
        ranking["human_review"] = overlay

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "promo_ranking.json", ranking)
    _write_json_atomic(
        output_dir / "ranking_report.json",
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "verdict_statistics": ranking["verdict_statistics"],
            "score_statistics": ranking["score_statistics"],
            "diversity_summary": ranking["diversity_summary"],
            "arc_statistics": ranking["arc_statistics"],
            "episode_statistics": ranking["episode_statistics"],
            "near_duplicate_clusters": ranking["near_duplicate_clusters"],
            "embedding_vs_judge": ranking["embedding_vs_judge"],
            "human_review": ranking.get("human_review"),
        },
    )
    (output_dir / "promo_ranking.md").write_text(render_ranking_markdown(ranking), encoding="utf-8")

    stats = ranking["verdict_statistics"]
    div = ranking["diversity_summary"]
    print(f"Accepted: {stats['accepted']} | Rejected: {stats['rejected']} | Failed: {stats['failed']}")
    print(f"Featured slate: {div['featured_total']} across {div['featured_distinct_episodes']} episodes / {div['featured_distinct_arcs']} arcs")
    print(f"Reserve (arc-cap overflow): {div['reserve_total']}")
    if ranking["near_duplicate_clusters"]:
        print(f"Near-duplicate clusters: {len(ranking['near_duplicate_clusters'])}")
    evj = ranking["embedding_vs_judge"]
    print(f"Embed-vs-judge Spearman ρ: {evj['spearman_rho']} | Jaccard@10 {evj['jaccard_top_k']['top_10']['jaccard']}")
    print(f"Ranking: {output_dir / 'promo_ranking.json'}")
    return 0


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
