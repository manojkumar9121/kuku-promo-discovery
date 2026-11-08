# Kuku — Submission Manifest (Problem 1: Find the Promos)

Every reviewer link opens without login. This manifest lists the three link
artifacts to submit in the form, and what each contains.

## Link 1 — GitHub repository (code + docs, no pack-derived text)

Subtree: `kuku-promo-discovery/` (git repo, branch `main`; tip updated for these final corrections).

Contains: full source, 198-test offline suite, the 2-page solution PDF
(`docs/Kuku_Solution.pdf`), architecture + operations docs, metrics-only JSON
artifacts. **Deliberately excludes** the screenplay pack, parsed dialogue,
raw model outputs, and any file quoting the show's scripts verbatim.
Run instructions: `README.md` (setup ≈ 2 commands + one 1.55 GB model pull;
5 pipeline commands; ~20 min cold, ~4 s warm).

## Link 2 — Drive folder (raw outputs — Kuku confidential, do not redistribute)

Complete, unedited outputs from the final run on the supplied 50-episode pack:

```
/parsed_output/          parser v2 output (parsed_episodes.json + parse_report.json)
/context_output/         candidates.json / .jsonl + context_report.json
/retrieval_output/       fast_discovery_results.json/.jsonl, evaluation_report.json,
                         candidate_embeddings.npz (+manifest), embedding_cache/,
                         manual_eval_labels.json (Stage-3 50-pair human labels),
                         top10_per_category.md
/judge_output/           judge_results.json/.jsonl (raw_model_output for all 84),
                         promo_ranking.json, promo_ranking.md,
                         ranking_report.json, judge_report.json,
                         judge_human_sample.json (Stage-4 33-pair human labels),
                         judge_cache.jsonl
/final/                  final_recommendations.json + final_recommendations.md (the slate; printable)
+ README.md (this manifest) + Kuku_Solution.pdf (same 2 pages)
```

Judgement evidence: all 916 cited event_ids resolve against the corpus
(verified by cross-check; 0 invented). Timestamps trace to source events.

## Link 3 — Walkthrough video (optional, ~3:00)

Script: `docs/WALKTHROUGH.md` (in Link 1). Record in your own voice; shows
parser → context builder → Fast Discovery → judge → ranking, human-validation
numbers, and honest failures.

## What is implemented / planned / incomplete (submission summary)

**Implemented and verified:** parser v2, context builder, embedding retrieval
(13 categories, no thresholds, cached), blind LLM judge with mechanically
enforced event evidence, diversity-aware ranker, reproducibility audit
(fresh re-run reproduces 50/100/3,727 → 399 → 84 → 53/31), 198 tests.

**Planned, not built:** audio adapter (`Kuku_Multimodal_Architecture.md` §2),
video adapter (§3), cross-modal alignment projection (§4), cross-episode
multi-segment promos (schema-supported, unbenchmarked), benchmarking on a
second show.

**Known incomplete / limitation items:** single-judge-model evaluation
(results stamp `kuku-judge-v1.0.0` + the staged 34.6B coder model); the
diversity cap is a policy lever not a quality proof; precision@K measured on
head-stratified sample (not corpus-unbiased); embedding cosine is orientation
only, never a probability.
