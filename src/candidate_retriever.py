#!/usr/bin/env python3
"""Candidate retrieval engine — semantic category matching over Context-Builder candidates.

Consumes the candidate JSON produced by context_builder.py (v1.0 schema:
candidate_id / episode metadata / dialogue transcript / actions & camera /
timestamps), embeds

  * candidate content (dialogue + speakers + deliveries + camera/action/sound), and
  * semantic category descriptions (category_definitions.py),

with EmbeddingGemma 2 through `embedding_engine.EmbeddingEngine`, then
retrieves top candidates per category by cosine similarity and merges them
into a deduplicated, fully traceable result set.

Retrieval philosophy: **high recall — no arbitrary similarity thresholds.**
Scores are cosine similarities in [-1, 1]; they are orientation indicators,
NOT probabilities and NOT promo-quality judgements. Ranking for promotion is
the later LLM judge's job; this module stays LLM-free.

Modality note: the text interface below is the contract future audio/video
adapters will reuse — a provider embeds a list of (id, text) documents; only
`build_candidate_document` is screenplay-specific.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Optional

import numpy as np

import category_definitions as catdefs
from embedding_engine import (
    PROMPT_ASYMMETRIC_SEARCH,
    PROMPT_SYMMETRIC_SIMILARITY,
    VALID_PROMPT_STRATEGIES,
    EmbeddingConfig,
    EmbeddingEngine,
    EmbeddingProvider,
    OfflineTestProvider,
    SentenceTransformerProvider,
    cosine_similarity_matrix,
    memory_usage_bytes,
)

SCHEMA_VERSION = "1.1"
RETRIEVER_NAME = "candidate_retriever.py"
BEST_CATEGORIES_KEPT = 3
DEFAULT_TOP_K = 10
DIALOGUE_LINES_IN_SNIPPET = 2
DEFAULT_K_SWEEP = (3, 5, 10, 15, 20, 30, 50)


# --------------------------------------------------------------------------
# Candidate -> embeddable document text
# --------------------------------------------------------------------------

def build_candidate_document(candidate: dict) -> tuple[str, str]:
    """Return (title, body) for a candidate, retaining every source content type.

    The title uses EmbeddingGemma 2's documented titled-document format
    (`title: {title} | text: {content}`); the body keeps speakers, deliveries
    and chronological camera/action/sound lines so the embedding carries the
    full narrative surface — spoken drama *and* stage directions.
    """
    episode_number = candidate.get("episode_number")
    scene_numbers = candidate.get("scene_numbers") or []
    headings = [
        f"{scene.get('scene_number')}: {scene.get('heading')}"
        for scene in (candidate.get("scenes") or [])
        if scene.get("heading")
    ]
    title = f"Episode {episode_number}; scenes {scene_numbers}"
    if headings:
        title += f" | {headings[0]}"

    dialogue_lines = []
    for item in candidate.get("dialogue_transcript") or []:
        speaker = item.get("speaker") or item.get("speaker_canonical") or "UNKNOWN"
        delivery = item.get("delivery")
        prefix = f"{speaker} ({delivery})" if delivery else speaker
        dialogue_lines.append(f"- {prefix}: {item.get('content', '')}")

    stage_lines = []
    for item in candidate.get("actions_and_camera") or []:
        stage_lines.append(f"- [{item.get('event_type')}] {item.get('content', '')}")

    multi_segment_note = "  This clip is one segment of a multi-segment promo." \
        if (candidate.get("segment_count", 1) or 1) > 1 else ""

    body_parts = []
    if dialogue_lines:
        body_parts.append("Drama dialogue:\n" + "\n".join(dialogue_lines))
    if stage_lines:
        body_parts.append("Staging (camera, action and sound):\n" + "\n".join(stage_lines))
    if not body_parts:
        body_parts.append("No transcript content in this window.")
    body = multi_segment_note + "\n\n".join(body_parts)
    body += f"\nRuntime: {candidate.get('duration_seconds')} seconds. Logline: {candidate.get('episode', {}).get('logline', '')}"
    return title, body


def load_candidates(path: Path) -> list[dict]:
    """Load candidates.json (wrapper payload or bare list)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        candidates = payload.get("candidates")
    else:
        candidates = payload
    if not isinstance(candidates, list):
        raise ValueError(f"No candidates list found in {path}")
    return candidates


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalConfig:
    top_k_per_category: int = DEFAULT_TOP_K
    best_categories_kept: int = BEST_CATEGORIES_KEPT

    def validated(self) -> "RetrievalConfig":
        if self.top_k_per_category < 1:
            raise ValueError("top_k_per_category must be >= 1.")
        if self.best_categories_kept < 1:
            raise ValueError("best_categories_kept must be >= 1.")
        return self


def _rank_indices_desc(values: np.ndarray, top_k: int) -> list[int]:
    """Top row indices by value, ties broken by ascending index (deterministic)."""
    order = np.argsort(-np.asarray(values), kind="stable")[:top_k]
    return [int(index) for index in order]


def retrieve_top_candidates(
    similarity: np.ndarray,        # (n_candidates, n_categories)
    candidates: list[dict],
    categories: list[catdefs.CategoryDefinition],
    config: RetrievalConfig,
) -> tuple[list[dict], dict]:
    """Per-category top-K retrieval merged candidate-wise.

    Merging never drops source information: a candidate retrieved by several
    categories keeps every hit (score + per-category rank), and its row from
    the similarity matrix is carried in full (`similarity_scores_all_categories`).
    Stable ordering: best hit score desc, then candidate_id.
    """
    config = config.validated()
    n_candidates = similarity.shape[0]
    per_category: dict[str, list[dict]] = {}
    merged_by_id: dict[str, dict] = {}
    effective_slots = 0

    for category_index, category in enumerate(categories):
        column = similarity[:, category_index]
        order = _rank_indices_desc(column, min(config.top_k_per_category, n_candidates))
        effective_slots += len(order)
        ranked = []
        for category_rank, candidate_index in enumerate(order, start=1):
            candidate = candidates[candidate_index]
            hit = {
                "category_id": category.category_id,
                "category_name": category.name,
                "category_group": category.group,
                "score": round(float(column[candidate_index]), 6),
                "category_rank": category_rank,
            }
            ranked.append(hit)
            candidate_id = candidate.get("candidate_id")
            merged = merged_by_id.get(candidate_id)
            if merged is None:
                # Keep the ORIGINAL candidate payload verbatim for traceability.
                merged = deepcopy_candidate(candidate)
                merged["category_hits"] = []
                merged["_row_index"] = candidate_index
                merged_by_id[candidate_id] = merged
            merged["category_hits"].append(hit)
        per_category[category.category_id] = ranked

    results: list[dict] = []
    for merged in merged_by_id.values():
        hits = sorted(merged["category_hits"], key=lambda hit: (-hit["score"], hit["category_id"]))
        row = similarity[merged["_row_index"]]
        del merged["_row_index"]
        entry = {
            "candidate_id": merged.get("candidate_id"),
            "episode_number": merged.get("episode_number"),
            "episode_label": merged.get("episode", {}).get("episode_label"),
            "logline": merged.get("episode", {}).get("logline"),
            "scene_numbers": merged.get("scene_numbers"),
            "start_seconds": merged.get("start_seconds"),
            "end_seconds": merged.get("end_seconds"),
            "duration_seconds": merged.get("duration_seconds"),
            "start_timestamp": merged.get("start_timestamp"),
            "end_timestamp": merged.get("end_timestamp"),
            "dialogue_event_count": merged.get("dialogue_event_count"),
            "crosses_scene_boundary": merged.get("crosses_scene_boundary"),
            "best_categories": hits[: config.best_categories_kept],
            "category_hits_count": len(hits),
            "category_hits": hits,
            "best_category_rank": min(hit["category_rank"] for hit in hits),
            "similarity_scores_all_categories": {
                category.category_id: round(float(row[index]), 6)
                for index, category in enumerate(categories)
            },
            "document_snippet": _document_snippet(merged),
        }
        results.append(entry)

    results.sort(key=lambda entry: (-entry["best_categories"][0]["score"], entry["candidate_id"]))

    hits_histogram: dict[int, int] = {}
    for entry in results:
        hits_histogram[entry["category_hits_count"]] = hits_histogram.get(entry["category_hits_count"], 0) + 1

    stats = {
        "top_k_per_category": config.top_k_per_category,
        "category_count": len(categories),
        "retrieval_slots": effective_slots,  # clamped per category to corpus size
        "distinct_candidates_retrieved": len(results),
        "duplicate_slots_merged": max(effective_slots - len(results), 0),
        "corpus_size": n_candidates,
        "corpus_coverage_fraction": round(len(results) / n_candidates, 4) if n_candidates else 0.0,
        "hits_histogram": {str(key): value for key, value in sorted(hits_histogram.items())},
    }
    return results, stats


def deepcopy_candidate(candidate: dict) -> dict:
    return json.loads(json.dumps(candidate))


def recall_sweep(
    similarity: np.ndarray,
    candidates: list[dict],
    categories: list[catdefs.CategoryDefinition],
    k_values: tuple[int, ...],
) -> list[dict]:
    """How the combined distinct set grows with `top_k_per_category`.

    Setting-effect analysis over coverage (no accuracy claim — no ground-truth
    labels exist at the embedding-only stage).
    """
    rows = []
    previous_ids: Optional[set[str]] = None
    for k in k_values:
        results, _stats = retrieve_top_candidates(
            similarity, candidates, categories, RetrievalConfig(top_k_per_category=k)
        )
        ids = {entry["candidate_id"] for entry in results}
        multi = sum(1 for entry in results if entry["category_hits_count"] > 1)
        row = {
            "top_k_per_category": k,
            "distinct_candidates": len(ids),
            "corpus_coverage_fraction": round(len(ids) / max(len(candidates), 1), 4),
            "candidates_in_multiple_categories": multi,
        }
        if previous_ids is not None:
            union = ids | previous_ids
            row["jaccard_vs_previous_k"] = round(len(ids & previous_ids) / len(union), 4) if union else 1.0
        previous_ids = ids
        rows.append(row)
    return rows


def strategy_agreement(
    similarity_a: np.ndarray,
    similarity_b: np.ndarray,
    categories: list[catdefs.CategoryDefinition],
    candidate_ids: list[str],
    top_k: int,
) -> list[dict]:
    """Jaccard agreement of top-K sets between two embedding strategies."""
    report = []
    for index, category in enumerate(categories):
        set_a = {candidate_ids[i] for i in _rank_indices_desc(similarity_a[:, index], top_k)}
        set_b = {candidate_ids[i] for i in _rank_indices_desc(similarity_b[:, index], top_k)}
        union = set_a | set_b
        report.append({
            "category_id": category.category_id,
            f"jaccard_at_top{top_k}": round(len(set_a & set_b) / len(union), 4) if union else 1.0,
        })
    return report


def score_distributions(similarity: np.ndarray, categories: list[catdefs.CategoryDefinition]) -> list[dict]:
    rows = []
    for index, category in enumerate(categories):
        column = similarity[:, index]
        quantiles = [float(q) for q in np.quantile(column, [0.1, 0.25, 0.5, 0.75, 0.9])]
        rows.append({
            "category_id": category.category_id,
            "min": round(float(column.min()), 6),
            "p10": round(quantiles[0], 6),
            "p25": round(quantiles[1], 6),
            "median": round(quantiles[2], 6),
            "p75": round(quantiles[3], 6),
            "p90": round(quantiles[4], 6),
            "max": round(float(column.max()), 6),
            "mean": round(float(column.mean()), 6),
            "std": round(float(column.std()), 6),
        })
    return rows


def _document_snippet(candidate: dict) -> str:
    lines = [
        f"{item.get('speaker_canonical') or item.get('speaker')}: {item.get('content', '')}"
        for item in (candidate.get("dialogue_transcript") or [])[:DIALOGUE_LINES_IN_SNIPPET]
    ]
    return " | ".join(lines)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

class _Timer:
    """Simple (start, stop) timing helper producing rounded seconds."""

    def __init__(self):
        self._marks: dict[str, float] = {}
        self._durations: dict[str, float] = {}

    @staticmethod
    def _now() -> float:
        return time.perf_counter()

    def start(self, name: str) -> None:
        self._marks[name] = self._now()

    def stop(self, name: str) -> float:
        started = self._marks.pop(name, None)
        if started is None:
            return 0.0
        duration = self._now() - started
        self._durations[name] = duration
        return duration

    def report(self) -> dict[str, float]:
        return {name: round(value, 3) for name, value in sorted(self._durations.items())}


def run_retrieval(
    candidates: list[dict],
    engine: EmbeddingEngine,
    categories: list[catdefs.CategoryDefinition],
    config: RetrievalConfig,
    compare_symmetric_strategy: bool = False,
) -> dict:
    """Embed, score and retrieve; returns the full retrieval payload in memory."""
    timer = _Timer()
    memory_start = memory_usage_bytes()

    timer.start("document_text_build")
    documents = [build_candidate_document(candidate) for candidate in candidates]
    titles = [title for title, _body in documents]
    bodies = [body for _title, body in documents]
    timer.stop("document_text_build")

    timer.start("embed_candidates")
    candidate_vectors, doc_cache_stats = engine.embed_candidate_documents(bodies, titles)
    timer.stop("embed_candidates")

    timer.start("embed_categories")
    category_vectors, category_cache_stats = engine.embed_category_queries(
        [category.description for category in categories]
    )
    timer.stop("embed_categories")

    timer.start("similarity_matrix")
    similarity = cosine_similarity_matrix(candidate_vectors, category_vectors)
    timer.stop("similarity_matrix")

    timer.start("retrieve")
    results, retrieval_stats = retrieve_top_candidates(similarity, candidates, categories, config)
    timer.stop("retrieve")

    symmetric_comparison: Optional[dict] = None
    if compare_symmetric_strategy:
        timer.start("strategy_comparison")
        symmetric_engine = EmbeddingEngine(
            engine.provider,
            config=engine.config,
            strategy=PROMPT_SYMMETRIC_SIMILARITY,
            cache_dir=engine.cache_dir,
        )
        sym_candidate_vectors, _sym_doc_stats = symmetric_engine.embed_candidate_documents(bodies, titles)
        sym_category_vectors, _sym_cat_stats = symmetric_engine.embed_category_queries(
            [category.description for category in categories]
        )
        symmetric_similarity = cosine_similarity_matrix(sym_candidate_vectors, sym_category_vectors)
        candidate_ids = [candidate.get("candidate_id") for candidate in candidates]
        agreement = strategy_agreement(
            similarity, symmetric_similarity, categories, candidate_ids,
            top_k=config.top_k_per_category,
        )
        symmetric_comparison = {
            "engine_signature": symmetric_engine.signature(),
            "agreement_top_k": agreement,
        }
        timer.stop("strategy_comparison")

    memory_end = memory_usage_bytes()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "retriever": RETRIEVER_NAME,
        "engine_signature": engine.signature(),
        "config": {
            "top_k_per_category": config.top_k_per_category,
            "best_categories_kept": config.best_categories_kept,
            "embedding": {
                "model_id": engine.provider.model_id,
                "dimension": engine.dimension,
                "native_dimension": engine.native_dimension,
                "truncate_dim": engine.config.truncate_dim,
                "dtype": engine.config.dtype,
                "device": engine.config.device,
                "batch_size": engine.config.batch_size,
                "text_only": engine.config.text_only,
                "max_chars_per_text": engine.config.max_chars_per_text,
            },
            "categories": [category.category_id for category in categories],
        },
        "cache": {
            "documents": doc_cache_stats,
            "categories": category_cache_stats,
        },
        "runtime_seconds": timer.report(),
        "memory": {
            "rss_peak_bytes_before": memory_start.get("rss_peak_bytes"),
            "rss_peak_bytes_after": memory_end.get("rss_peak_bytes"),
        },
        "category_statistics": {
            "score_distributions": score_distributions(similarity, categories),
            "retrieval_stats": retrieval_stats,
        },
        "results": results,
        "similarity": {
            "candidate_ids": [candidate.get("candidate_id") for candidate in candidates],
            "category_ids": [category.category_id for category in categories],
            "matrix": [[round(float(value), 6) for value in row] for row in similarity],
        },
        # Vectors travel with the in-memory payload but are excluded from JSON
        # output and stored in candidate_embeddings.npz (never inline in JSON).
        "_vectors": {
            "candidates": candidate_vectors,
            "categories": category_vectors,
        },
    }
    if symmetric_comparison is not None:
        payload["strategy_comparison"] = symmetric_comparison
    return payload


# --------------------------------------------------------------------------
# Output files
# --------------------------------------------------------------------------

def write_npz_embeddings(
    output_dir: Path,
    candidate_ids: list[str],
    category_ids: list[str],
    candidate_vectors: np.ndarray,
    category_vectors: np.ndarray,
    engine: EmbeddingEngine,
) -> dict:
    """Store raw vectors separately from the candidate JSON (requirement 9)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "candidate_embeddings.npz"
    np.savez_compressed(
        path,
        candidate_ids=np.asarray(candidate_ids, dtype="<U256"),
        category_ids=np.asarray(category_ids, dtype="<U64"),
        candidate_embeddings=candidate_vectors.astype(np.float32),
        category_embeddings=category_vectors.astype(np.float32),
    )
    manifest = {
        "path": str(path),
        "engine_signature": engine.signature(),
        "dimensions": {
            "candidates": list(candidate_vectors.shape),
            "categories": list(category_vectors.shape),
        },
        "note": "Vectors are float32 and L2-normalized; candidate order matches candidate_ids.",
    }
    (output_dir / "embedding_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def render_top10_markdown(results_payload: dict) -> str:
    """Human-readable top-10 per category from the full per-category rankings."""
    lines = ["# Top 10 candidates per category (Fast Discovery, embedding-only)", ""]
    lines.append("Ranking is cosine similarity under the configured prompt strategy.")
    lines.append("Scores are orientation indicators, not probabilities or quality verdicts.")
    lines.append("")
    for category_id in results_payload["config"]["categories"]:
        lines.append(f"## {category_id}")
        ranked = [
            (entry, hit)
            for entry in results_payload["results"]
            for hit in entry["category_hits"]
            if hit["category_id"] == category_id
        ]
        ranked.sort(key=lambda pair: pair[1]["category_rank"])
        for entry, hit in ranked[:10]:
            lines.append(
                f"{hit['category_rank']}. **{entry['candidate_id']}** ep{entry['episode_number']} "
                f"[{entry['start_timestamp']} → {entry['end_timestamp']}] "
                f"({entry['duration_seconds']}s) — score {hit['score']:.4f} "
                f"({entry['category_hits_count']} category hits) — {entry['document_snippet'][:110]} …"
            )
        lines.append("")
    return "\n".join(lines)


def render_manual_review_markdown(results_payload: dict) -> str:
    """Review worksheet scaffolding; values below are curated by hand after reading."""
    lines = [
        "# Manual review worksheet",
        "",
        "Read the referenced candidates in `context_output/candidates.json`, then tick",
        "judgements. Persist them in `manual_eval_labels.json`:",
        "`{\"category_id\": {\"candidate_id\": \"relevant\"|\"partial\"|\"not\"}}`",
        "",
    ]
    for category_id in results_payload["config"]["categories"]:
        lines.append(f"## {category_id}")
        ranked = [
            (entry, hit)
            for entry in results_payload["results"]
            for hit in entry["category_hits"]
            if hit["category_id"] == category_id
        ]
        ranked.sort(key=lambda pair: pair[1]["category_rank"])
        for entry, hit in ranked[:10]:
            lines.append(
                f"- [ ] {entry['candidate_id']} (score {hit['score']:.4f}, rank {hit['category_rank']}) — "
                f"{entry['document_snippet'][:110]} …"
            )
        lines.append("")
    return "\n".join(lines)


def apply_manual_labels(payload: dict, labels_path: Path, top_ks: tuple[int, ...] = (5, 10)) -> Optional[dict]:
    """Attach precision@K measured against a human-reviewed label file.

    Labels: {category_id: {candidate_id: relevant|partial|not}}. Precision
    counts `relevant` and `partial` as hits by default; the review file
    explains the rubric. Missing labels file returns None (evaluation skips).
    """
    if not labels_path.exists():
        return None
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    matrix = np.asarray(payload["similarity"]["matrix"], dtype=np.float32)
    candidate_ids = payload["similarity"]["candidate_ids"]
    category_ids = payload["similarity"]["category_ids"]
    report: dict[str, dict] = {}
    for category_id in category_ids:
        category_labels = labels.get(category_id)
        if not category_labels:
            continue
        index = category_ids.index(category_id)
        order = _rank_indices_desc(matrix[:, index], min(top_ks[-1], len(candidate_ids)))
        per_review: dict[str, str] = {}
        for rating, candidate_rank, candidate_index in (
            (category_labels.get(candidate_ids[i]), rank, i) for rank, i in enumerate(order, start=1)
        ):
            if rating and candidate_rank <= top_ks[-1]:
                per_review[candidate_ids[candidate_index]] = rating
            if rating is None and candidate_rank in top_ks:
                per_review.setdefault(candidate_ids[candidate_index], "unreviewed")
        scored = {
            f"precision_at_{k}": round(
                sum(
                    1
                    for i in order[:k]
                    if category_labels.get(candidate_ids[i]) in {"relevant", "partial"}
                ) / min(k, len(order)), 4)
            for k in top_ks
        }
        report[category_id] = {"precision": scored, "ratings": per_review}
    if not report:
        return None
    return {"labels_file": str(labels_path), "categories": report}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Embedding-based fast promo discovery over Context-Builder candidates."
    )
    parser.add_argument("--candidates", default="context_output/candidates.json",
                        help="path to candidates.json from context_builder.py")
    parser.add_argument("--output", default="retrieval_output")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--best-categories", type=int, default=BEST_CATEGORIES_KEPT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-path", default=None,
                        help="Local path to EmbeddingGemma 2 (default: models/embeddinggemma-2 if present, else the HF id)")
    parser.add_argument("--model-id", default="google/embeddinggemma-2")
    parser.add_argument("--prompt-strategy", default=PROMPT_ASYMMETRIC_SEARCH, choices=list(VALID_PROMPT_STRATEGIES))
    parser.add_argument("--truncate-dim", type=int, default=None, choices=(128, 256, 512))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-candidates", type=int, default=None, help="Debug: embed only the first N candidates")
    parser.add_argument("--skip-strategy-comparison", action="store_true")
    parser.add_argument("--skip-recall-sweep", action="store_true")
    parser.add_argument("--offline", action="store_true",
                        help="Use the OfflineTestProvider (token-overlap vectors; tests/debug only)")
    parser.add_argument("--manual-labels", default="retrieval_output/manual_eval_labels.json",
                        help="Human-reviewed category labels to compute precision@K against")
    args = parser.parse_args(argv)

    try:
        candidates_path = Path(args.candidates)
        candidates = load_candidates(candidates_path)
        if args.limit_candidates is not None:
            if args.limit_candidates < 1 or args.limit_candidates > len(candidates):
                raise ValueError("--limit-candidates must be between 1 and the corpus size.")
            candidates = candidates[: args.limit_candidates]
        model_path = args.model_path or (
            "models/embeddinggemma-2" if Path("models/embeddinggemma-2").exists() else None
        )
        embedding_config = EmbeddingConfig(
            model_id=args.model_id,
            local_model_dir=model_path,
            device=args.device,
            batch_size=args.batch_size,
            truncate_dim=args.truncate_dim,
        ).validated()
        categories = catdefs.ordered_descriptions()
        retrieval_config = RetrievalConfig(
            top_k_per_category=args.top_k, best_categories_kept=args.best_categories
        ).validated()
        output_dir = Path(args.output)
        if args.offline:
            provider: EmbeddingProvider = OfflineTestProvider()
        else:
            provider = SentenceTransformerProvider(embedding_config)
        engine = EmbeddingEngine(
            provider,
            config=embedding_config,
            strategy=args.prompt_strategy,
            cache_dir=output_dir / "embedding_cache",
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    payload = run_retrieval(
        candidates, engine, categories, retrieval_config,
        compare_symmetric_strategy=not args.skip_strategy_comparison,
    )
    if not args.skip_recall_sweep:
        payload["category_statistics"]["recall_sweep"] = recall_sweep(
            np.asarray(payload["similarity"]["matrix"], dtype=np.float32),
            candidates,
            categories,
            DEFAULT_K_SWEEP,
        )

    # ---- vectors artifacts (stored separately from candidate JSON) ----
    manifest = write_npz_embeddings(
        output_dir,
        payload["similarity"]["candidate_ids"],
        payload["similarity"]["category_ids"],
        payload["_vectors"]["candidates"],
        payload["_vectors"]["categories"],
        engine,
    )
    del payload["_vectors"]

    # ---- optional human review overlay ----
    manual_eval = apply_manual_labels(payload, Path(args.manual_labels))
    if manual_eval is not None:
        payload["category_statistics"]["manual_review"] = manual_eval

    # ---- results artifacts ----
    _write_json_atomic(output_dir / "fast_discovery_results.json", payload)
    _write_jsonl(output_dir / "fast_discovery_results.jsonl", payload["results"])
    _write_json_atomic(
        output_dir / "evaluation_report.json",
        {
            "schema_version": payload["schema_version"],
            "engine_signature": payload["engine_signature"],
            "config": payload["config"],
            "cache": payload["cache"],
            "runtime_seconds": payload["runtime_seconds"],
            "memory": payload["memory"],
            "category_statistics": payload["category_statistics"],
            "strategy_comparison": payload.get("strategy_comparison"),
        },
    )
    (output_dir / "top10_per_category.md").write_text(
        render_top10_markdown(payload), encoding="utf-8"
    )
    (output_dir / "manual_eval_review.md").write_text(
        render_manual_review_markdown(payload), encoding="utf-8"
    )

    retrieval_stats = payload["category_statistics"]["retrieval_stats"]
    print(f"Embedded {len(candidates)} candidates | engine {payload['engine_signature']}")
    print(f"Retrieved {retrieval_stats['distinct_candidates_retrieved']} distinct candidates from "
          f"{retrieval_stats['retrieval_slots']} slots (corpus coverage "
          f"{retrieval_stats['corpus_coverage_fraction']:.1%})")
    print(f"Cache after run: {payload['cache']['documents']['encoded_now']} document encodes, "
          f"{payload['cache']['categories']['encoded_now']} category encodes")
    print(f"Runtime: {payload['runtime_seconds']}s | Peak RSS: "
          f"{(payload['memory']['rss_peak_bytes_after'] or 0) / 1_048_576:.0f} MB")
    print(f"Embeddings: {manifest['path']}")
    print(f"Results:    {output_dir / 'fast_discovery_results.json'}")
    print(f"Evaluation: {output_dir / 'evaluation_report.json'}")
    print(f"Top-10 review: {output_dir / 'top10_per_category.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
