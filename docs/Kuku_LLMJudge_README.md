# Kuku Stage 4 — Contextual LLM Judge & Promo Ranker

Stage 4 of the Kuku FM AI Promo Discovery Engine. It consumes the 84-candidate
union produced by Stage 3 Fast Discovery (`retrieval_output/fast_discovery_results.json`,
all 84, no embedding-based pre-filtering beyond that union) and judges,
**blindly**, which candidates are genuinely usable as vertical microdrama
promos. The judge is LLM-free of any ranking influence: no retrieved
categories, scores or ranks appear in the prompt, so retrieval can never
anchor the verdict.

Parser v2, Context Builder v1.0 and the embedding/retrieval implementation
remain untouched. Audio/video ingestion is intentionally out of scope.

> **Model colocation note (2026-11-08).** The staging provider is a
> 34.6B Q4_K_M coder model on a llama.cpp endpoint
> (`llama-server` @ `http://100.90.211.24:9931`), driven with
> `chat_template_kwargs.enable_thinking=false` so the content channel carries
> plain JSON. Any stronger/different model can be wired behind the
> `LlmJudgeProvider` protocol — register it in `llm_judge.build_provider`.

---

## 1. Pipeline position

```
Parser v2 ─► Context Builder ─► Embedding Retrieval (union-84) ─► LLM Judge ─► Promo Ranker
EVAL:           Parser raw corpus    K04 sanity audit    Stage-3 eval      this stage
```

The jury never relaxes the earlier stages' contracts: it reads the full
Context-Builder candidate payload (dialogue, camera/action/sound, speakers +
deliveries, logline, continuity notes, preceding/following narrative ±12s).

## 2. Modules

| File | Role |
| --- | --- |
| `judge_schema.py` | Rubric dimensions/weights, judgement validation against the candidate's event universe (no invented ids survive), weighted score (computed by code), boundary-feedback validation. |
| `judge_prompt.py` | Versioned prompt (`kuku-judge-v1.0.0`): role, critical rules, rubric anchors, JSON output contract, payload builder (full candidate view). |
| `llm_judge.py` | Provider protocol (`LlamaServerProvider` llama.cpp endpoint, `OfflineScriptedJudge` test double), repair loop (validation-guided JSON correction, ≤2 repairs), crash-safe JSONL judge cache, token/latency accounting, CLI. |
| `promo_ranker.py` | Quality ranking (weighted score desc, confidence tie-break), near-duplicate clustering (same episode + IoU ≥ 0.5, or same `observed_arc` with IoU ≥ 0.3), diversity-aware slate selection, rejection retention, embedding-vs-judge comparison, report renderer, CLI. |

## 3. Rubric (machine-weighted)

| Dimension | Weight | Meaning |
| --- | ---: | --- |
| Opening hook | 0.25 | Attention inside the first 3 seconds |
| Cliffhanger | 0.25 | Cut leaves a question the viewer must see answered |
| Standalone clarity | 0.20 | Understandable cold; scored only from the clip |
| Narrative progression | 0.15 | Setup → escalation → peak inside ~30s |
| Emotional impact | 0.15 | Anger/shock/ache/curiosity at the peak |

Weighted score ∈ [0, 5] computed in code from the ints. Confidence and
per-dimension evidence (with real payload event ids, at least one entry per
dimension) come from the model. Rejection vocabulary is fixed
(`weak_opening`, `no_cliffhanger`, `needs_prior_knowledge`,
`incoherent_narrative`, `weak_emotional_impact`, `not_promotional_moment`).

## 4. Judging protocol

* **Blind by default** — `include_retrieval_metadata=False` for production;
  flag `include_retrieval_metadata` exists for experiments only and is
  recorded in outputs.
* **Separate judging from boundary refinement** — `boundary_feedback` is
  advisory: it may only cite payload event ids, is order-validated, and is
  **never applied** or fed to the ranker. Cut refinement is a future,
  validator-gated stage.
* Every major judgement carries event ids; mechanical validation guarantees
  none of them are invented (every id must exist in the candidate's own
  source/context event universe).
* Coarse signal classes (betrayal / coercion / threat / accusation) are
  discriminated in the prompt by explicit examples; judges must name what
  actually happens, not the genre label a scene *looks* like.

## 5. Ranking & diversity

Quality ordering: weighted score desc, then confidence desc, then
candidate_id. The diversity pass is separate from quality:

* Arc key = `(episode_number, observed_arc)`; `--max-per-arc` (default 2).
* `featured` = candidates within the arc cap in quality order (the
  recommended slate). `reserve` = everyone past the cap, in quality order,
  carrying `overflow_reason`. Both lists are emitted with timestamps, event
  ids and feedback — nothing is dropped.
* Rejected candidates are retained with rejection reasons and notes.
* Near-duplicate clusters are reported with a recommended head.
* The spread is deliberately wide: five shrine-removal clips from one arc
  cannot all occupy the slate when other episodes qualify.

## 6. Runtime & offline testing

The whole pipeline is end-to-end testable offline with
`python3 llm_judge.py --offline` (deterministic `OfflineScriptedJudge`
double). No network calls, no LLM in unit tests.

## 7. Running against the staged endpoint

```bash
# Full blind pass over the retrieval union (resume supported)
python3 llm_judge.py \
  --retrieval-results retrieval_output/fast_discovery_results.json \
  --candidates context_output/candidates.json \
  --output judge_output

# Ranking (add --human-labels to attach the independent review overlay)
python3 promo_ranker.py \
  --judge-results judge_output/judge_results.json \
  --candidates context_output/candidates.json \
  --retrieval-results retrieval_output/fast_discovery_results.json \
  --output judge_output
```

Cache: `judge_output/judge_cache.jsonl` — one JSONL record per candidate,
**streamed to disk as it lands** (a killed run loses nothing already judged).
Key: provider model id | prompt version | blind flag | sha1(candidate content).

## 8. Evaluation protocol (what the numbers mean)

* Embedding-vs-judge comparison computed on the 84 shared candidates:
  Spearman ρ (rank correlation), Jaccard overlap at top-10/top-20, and every
  |rank delta| ≥ 10 case with verdicts.
* Precision@K against an **independent, post-hoc human sample**
  (`judge_output/judge_human_sample.json`), labelled `strong|usable|weak|unusable`.
  Neither the prompt nor the ranker nor any threshold was tuned against
  these labels. The Stage-3 category-relevance labels were NOT reused here —
  different question (promo quality vs category membership).
* Score/rejection distributions over the judged population.
* Runtime + token accounting per candidate from provider-reported usage
  (or a documented heuristic when the provider reports none).

## 9. Output artifacts (`judge_output/`)

| File | Content |
| --- | --- |
| `judge_results.json` | Full machine-readable payload: config, totals, every row incl. attempts |
| `judge_results.jsonl` | One row/candidate (crash-safe stream) |
| `raw outputs` (field `raw_model_output`) | Verbatim model text per candidate |
| `judge_report.json` | Phase-1 report: status/verdict counts, mean score |
| `promo_ranking.json` | Featured slate, reserve, rejected, clusters, stats, comparison, human review |
| `promo_ranking.md` | Human-readable slate |
| `ranking_report.json` | Eval-report variant (same stats, separable for docs) |
| `judge_cache.jsonl` | Incremental cache |
| `judge_human_sample.json` | The independent sample labels (filled post-run) |

## 10. Run results (this machine, `judge_output/`, 2026-11-08)

Production blind pass over the 84 retrieval-union candidates
(`models/…KAT-Coder-V2.5-Dev-Q4_K_M`, `kuku-judge-v1.0.0`, temperature 0.0,
`enable_thinking=false`). All 84 candidates were judged; the one hard
truncation (max output tokens) was resolved by a documented budget increase
to 2048 — no prompt content, weights, or candidate selection changed.

### Cost, runtime and token usage

| Measurement | Value |
| --- | --- |
| Candidates attempted / ok | 84 / 84 |
| Prompt tokens — total model work (pass 1 + retry pass) | 454,662 / ≈5,412 per candidate |
| Completion tokens — total model work | 91,399 / ≈1,088 per candidate |
| Per-row attribution (written in `judge_results.json`) | 436,320 prompt · 87,799 completion · 3,750.5 s latency |
| Pass 1 (83 ok + 1 truncated failure) | 449,814 prompt / 90,035 completion / 3,841.4 s |
| Pass 2 (budget-only retry, 2048 tokens) | 4,848 prompt / 1,364 completion / 53.5 s (83 rows served from cache) |
| Accumulated latency (both passes) | ≈ 3,895 s (~65 min; ~46 s wall per candidate) |
| LLM calls | 88 (84 first-pass + 4 repair attempts on the truncated candidate) |

### Verdicts and score distribution

| Stat | Value | Stat | Value |
| --- | --- | --- | ---: |
| accepted | **53** (63%) | mean weighted score | 3.429 / 5 |
| rejected | **31** (37%) | median | 3.550 |
| failed | 0 | p10 / p90 | 2.595 / 4.200 |
| | | min / max | 1.950 / 4.400 |

Dimension means (interesting: clarity is the weakest axis — the exact one
that gates paid ad effectiveness): opening_hook 3.417 · cliffhanger 3.274 ·
**standalone_clarity 3.060** · narrative_progression 3.690 · emotional
impact 3.940.

Rejection reasons (multi-valued; 31 rejections): `no_cliffhanger` **20** ·
`weak_opening` **16** · `needs_prior_knowledge` **10** ·
`not_promotional_moment` **9** · `weak_emotional_impact` **6** ·
`incoherent_narrative` **5**.

### Ranking & diversity

* Featured slate: **49 candidates** across **19 episodes** and **41 distinct
  arcs**, capped at 2 per (episode, arc).
* Reserve (arc-cap overflow, quality-ordered, demoted but retained):
  4 candidates — `cand_ep32_t90.1-119.5_3df514`, `cand_ep33_t140.5-174.0_28a108`,
  `cand_ep03_t58.3-88.1_f06bc0`, `cand_ep33_t69.5-100.0_aedd66`.
* Near-duplicate clusters: **6** (ep32×2 dungeon beats, ep2 altar speech,
  ep3 rescue intervention, ep43 treason-reveal briefing) — buyers should
  take only the cluster head; ~9 recommendations collapse away.

### Embedding-only ranking vs LLM-judged ranking (shared 84 candidates)

| Metric | Value |
| --- | ---: |
| Spearman ρ | **0.044** (≈ 0 — the two stages rank independently) |
| Jaccard @10 | 0.053 (1 shared: ep12 ultimatum `cand_ep12_t57.3-87.2_fb29da`) |
| Jaccard @20 | 0.111 (2 shared: ep12 + `cand_ep16_t48.8-78.8_96aa87`) |

Disagreements are the interesting signal, seen by |rank delta| ≥ 40:

| Candidate | embed # | judge # | judge verdict | note |
| --- | ---: | ---: | --- | --- |
| `cand_ep35_t83.0-113.0_807f9f` | 1 | 73 | reject | static backstory monologue; clear cut, no grip |
| `cand_ep31_t70.8-103.3_cc46ef` | 8 | 80 | reject | "I call it love" — tenderness, not a hook |
| `cand_ep14_t35.8-66.0_84fb9f` | 14 | 81 | reject | soft intimacy, no promo urgency |
| `cand_ep01_t70.6-101.5_39a710` | 70 | **1** | **accept** | Gerald's abandonment + family disowning; judge 4.4 |
| `cand_ep16_t48.8-78.8_96aa87` | 83 | 19 | accept | Harrington-framed confession; 4.0 |
| `cand_ep24_t66.2-92.0_87686c` | 84 | 21 | accept | captive healer's subversion line; 4.05 |
| `cand_ep33_t104.5-134.5_ffa17e` | 69 | 10 | accept | coerced testimony; menace-as-promo 4.2 |

Special-attention clusters:

* **Ep33 betrayal/coercion**: embedding's best single slot (rank 1) is rejected
  by the judge as dead air; the coercion windows that survive (e.g.
  `ffa17e`, `682d10`, `69.5-100.0`, `140.5-174.0`) are accepted on menace +
  outrage terms with standalone clarity as the swing factor — 4 accepts, 3
  rejects, with `682d10` accepted after the documented budget bump.
* **Ep1 wedding betrayal**: embedding spread it to #3/#5; judge consolidates it
  to the very top — `39a710` #1, `91e509` #4, `f0e9f8` #16-area accepts;
  `f27ba8` (same scenes, flatter cut) rejected for no_cliffhanger.
* **Ep42–43 shocking reveal**: uniform accepts 3.55–4.05 — the only cluster
  both embedding and judge agree on; usable fallback for a different ad flight.
* **Ep32 dungeon suite**: 9/9 accepts 3.05–4.2; capped at 2 in the slate per arc
  (`dungeon_irony_peak` hold = `3df514` demoted to reserve).
* **Ep48 suspense**: both candidates rejected (prior knowledge + no resolved
  opener) — embedding ranked them high; judge disagreed.

### Independent human review (33 labels, GLM reviewer, top-20 + reserve + rejected high-embeds)

| Metric | Value |
| --- | ---: |
| Precision@5 | **1.00** (5/5 strong featured) |
| Precision@10 | **1.00** (10/10 labelled/hit = strong-or-usable) |
| Precision@20 | **1.00** (16 strong + 4 usable, all judges' acceptances) |
| Judge vs reviewer agreement | **30/33** (91%) — 3 disagreements, all judge-stricter cases (§11) |

In the reviewed subset, the judge **never** put a weak promo into the top
twenty; the residual disagreements run the other way (judge rejected clips
I rated usable: `ep47_0492b1` revenge-labor scene — judge: no_cliffhanger;
`ep48_6d7608` Catherine's ten-days threat; `ep33_4e679b` coercion beat —
all three need stage knowledge to land cold).

This overlay is measured against the **independent** sample; the Stage-3
category labels were never consulted, and neither prompt, weights nor
selection policy were tuned on these labels.

### Caveats

* Precision@K is exact for this head-stratified sample, *not* an unbiased
  corpus-wide estimate; the bottom of the ranking is unreviewed.
* Absolute scores are not calibrated probabilities; they are practice-oriented
  rubrics to rank candidates, and they were produced by ONE model, ONE prompt
  version. The field `prompt_version` is in every output row.
* The staging judge is a coder-leaning 34B; connection to an instruct chat
  model (e.g. at the Kuku interview) may reshuffle mid-tier verdicts.

## 11. Known limitations

* Episode-specific hallucination is impossible only mechanically across the
  event universe, not semantically — the judge can still misjudge a moment's
  dramatic value; hence human sample + retained rejections with reasons.
* The staging provider is a coder-leaning 34B; a genuinely stronger chat
  model should be connected at the Kuku interview via `build_provider`.
* Arc capping is a policy lever, not a quality claim: overflow (`reserve`)
  may contain excellent promos that simply lost the arc cap.
