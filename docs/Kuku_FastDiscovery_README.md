# Kuku Stage 3 — Embedding-Based Promo Discovery (Fast Discovery Mode)

Stage 3 of the Kuku FM AI Promo Discovery Engine. It consumes the 399 candidates
produced by the approved Context Builder (`Kuku_ContextBuilder_README.md`,
candidate schema v1.0) and ranks them against semantic promo categories using
**Google's EmbeddingGemma 2** (`google/embeddinggemma-2`) — text-only retrieval,
no LLM calls anywhere in this stage.

Three modules, one job each:

| Module | Role |
| --- | --- |
| `category_definitions.py` | The 13 promo categories as rich natural-language descriptions (pure data). |
| `embedding_engine.py` | EmbeddingGemma 2 behind a swappable provider interface: task instructions, batching, truncation, cosine math, disk cache. |
| `candidate_retriever.py` | Candidate → document text, per-category top-K retrieval, merge/dedup, evaluation artifacts, Fast Discovery CLI. |

---

## 1. Model contract (EmbeddingGemma 2)

Implemented exactly as prescribed by the official model card / checkpoint:

* **Implementation**: `sentence-transformers.SentenceTransformer` (>= 6.1.0, as
  pinned by the checkpoint's `config_sentence_transformers.json`).
* **Task instructions** — applied manually as text, byte-identical to the
  checkpoint's prompt dict (verified against `config_sentence_transformers.json`):
  * Category queries (retrieval): `task: search result | query: {description}`
    (the `SearchQuery` prompt string).
  * Candidate documents: `title: {title} | text: {content}`; candidates have
    real titles (episode/scenes/heading), so no `title: none` instance appears
    in the default run.
  * Symmetric comparison baseline: both sides use
    `task: sentence similarity | query: {text}` (`SentenceSimilarity`).
* **Text-only loading**: `config_kwargs={"vision_config": None, "audio_config": None}`
  → 270M effective parameters; no vision/audio tower is loaded (the official
  selective-encoder path). This is also the seam where future audio/video
  adapters plug in — the towers can be enabled without touching the text path.
* **Precision**: float32 on CPU (float16 is rejected by `EmbeddingConfig`).
* **Dimension**: native 768 output (`provider.dimension`); Matryoshka truncation
  to 128/256/512 is supported post-encode **with L2 renormalization** as the
  card requires.
* **Similarity**: cosine. Scores are in [-1, 1]. **They are not probabilities
  and not promo-quality judgements** — they only orient the recall stage;
  promotion ranking is the LLM judge's job (Stage 4, explicitly not built here).

## 2. Architecture

### Swappable providers

`embedding_engine.EmbeddingProvider` is a structural `Protocol`:

```python
class EmbeddingProvider(Protocol):
    model_id: str
    dimension: int          # native output dim
    metadata: dict          # provenance (device, dtype, prompts available)
    def embed_texts(self, texts: list[str], prompt_name: Optional[str] = None) -> np.ndarray: ...
```

Implementations shipped:

* `SentenceTransformerProvider` — the official EmbeddingGemma 2 implementation.
* `OfflineTestProvider` — deterministic token-overlap double used by the test
  suite and the CLI's `--offline` flag so retrieval math and output contracts
  can be exercised headlessly (never used in production runs).

The retrieval code only ever talks to `EmbeddingProvider`, so a future
audio/video adapter (e.g. a multimodal provider embedding candidate
transcripts + audio snippets) can be dropped in without touching retrieval.

### Document format (`build_candidate_document`)

Each candidate becomes one titled document:

* **title**: `Episode {n}; scenes [s] | {heading}` — the real-title path of the
  documented `title | text` instruction.
* **body**: chronological *Drama dialogue* lines with speaker + delivery, then
  *Staging* lines ({camera/action/sound} event types), then runtime + episode
  logline. Nothing is dropped: speech, stage directions and delivery cues all
  reach the embedding.
* Head+tail clipping at `max_chars_per_text` (4000) for safety (observed
  documents: 1.3–3.3 KB, so it never binds on this corpus).

### Caching

`EmbeddingCache` stores full-resolution float32 vectors in
`retrieval_output/embedding_cache/embedding_cache.npz`, atomically written.
Keys are sha1 digests over `engine_token | role | formatted_text`, where
`engine_token = {provider model path} | {strategy} | {dtype} | {dim}`.
Consequence: re-runs of the same configuration are free; edits to a
description or to a candidate re-embed exactly what changed. Different
prompt-strategies never share cache collections.

### Retrieval strategy — high recall, no thresholds

For each of the 13 categories: rank all 399 candidates by cosine similarity,
take top-K (default 10). **No similarity cutoff whatsoever** — the pipeline's
philosophy is that rejecting by an arbitrary threshold at this stage throws
away boundary cases the more accurate LLM judge would catch.

Merge: the per-category top-K lists (13 × K slots) are merged by
`candidate_id` without dropping any information:

* multi-category candidates keep **every** hit (score + per-category rank),
* each retained entry carries its **complete** similarity row
  (`similarity_scores_all_categories`), so dedup loses no source signal,
* original candidate IDs and timestamps are preserved verbatim,
* stable final ordering: best hit score desc, then `candidate_id`.

Top-K is configurable (`--top-k`, `RecallConfig`); a `recall_sweep`
(K ∈ {3, 5, 10, 15, 20, 30, 50}) reports how coverage grows with K —
a settings-effect analysis, **not** an accuracy claim (there are no
ground-truth labels at the embedding stage).

## 3. Categories

Ten drama categories plus three structural categories
(`category_definitions.py`; every description is embedded verbatim as a query):

drama: `betrayal`, `shocking_reveal`, `revenge`, `humiliation`, `threat`,
`confession`, `romantic_tension`, `suspense`, `unexpected_entrance`,
`emotional_confrontation`

structural: `strong_opening_hook`, `unresolved_cliffhanger`,
`standalone_dramatic_moment`

## 4. Usage

```bash
# Fast Discovery over the full candidate corpus (defaults documented below)
python3 candidate_retriever.py \
    --candidates context_output/candidates.json \
    --output retrieval_output

# Useful flags
--top-k 10                    # per-category K (default 10)
--best-categories 3            # kept per results entry (default 3)
--prompt-strategy ASYMMETRIC_SEARCH | SYMMETRIC_SIMILARITY
--truncate-dim 128|256|512     # MRL truncation (default: native 768)
--device cpu                   # embedding device
--batch-size 8                 # encoder batching
--manual-labels PATH           # attach precision@5/@10 against reviewed labels
--offline                      # headless debug (OfflineTestProvider)
--skip-strategy-comparison     # skip the symmetric-baseline agreement pass
--skip-recall-sweep            # skip the K sweep
```

Default model resolution: `models/embeddinggemma-2` if present (recommended —
the local snapshot of `google/embeddinggemma-2`), else the HF hub id.

## 5. Outputs (`retrieval_output/`)

| File | Contents |
| --- | --- |
| `fast_discovery_results.json` | Full payload: config, cache stats, runtime/memory, score distributions, recall sweep, results. Every result entry: candidate_id, episode, scene numbers, start/end timestamps, duration, best matching categories, all category hits + ranks, full similarity row. |
| `fast_discovery_results.jsonl` | The same results, one JSON object per line (streaming-friendly). |
| `candidate_embeddings.npz` | Raw float32 vectors, stored **separately** from candidate JSON (requirement): `candidate_embeddings` (n × dim, L2-normalized, order == `candidate_ids`), `category_embeddings` (13 × dim), plus the id arrays. |
| `embedding_manifest.json` | npz provenance: engine signature, shapes, normalization note. |
| `embedding_cache/` | Content-addressed vector cache (see §2). |
| `evaluation_report.json` | Retrieval-only evaluation: config, cache, runtime, memory, score distributions, recall sweep, strategy comparison, manual review overlay when labels exist. |
| `top10_per_category.md` | Human-readable top 10 per category with timestamps and scores. |
| `manual_eval_review.md` | Worksheet for the hand-reviewed evaluation subset. |
| `manual_eval_labels.json` | Human judgements: `{category_id: {candidate_id: relevant\|partial\|not}}`. |

Fast Discovery Mode results entries contain exactly: **candidate ID, episode,
start/end timestamp, duration, best matching categories, similarity scores
(every category), and retrieval ranking** (per-category `category_rank`, plus
`best_category_rank`).

## 6. Run results (all 399 candidates)

All numbers below come from the actual production runs of
`candidate_retriever.py` against `context_output/candidates.json`
(engine `models/embeddinggemma-2|ASYMMETRIC_SEARCH|768|768`, float32, CPU).

### Cost, runtime and memory

| Measurement | Value |
| --- | --- |
| Cold embedding pass (412 texts: 399 documents + 13 categories) | measured 580.5 s on the symmetric pass; asymmetric pass costs the same (identical text volume) |
| Warm re-run (everything cache-hits, model never loaded) | 4.1 s end-to-end incl. retrieval + recall sweep |
| Peak RSS, full run with model resident | 3,456 MB (3.38 GiB `ru_maxrss`) |
| Peak RSS, fully-cached re-run | 78 MB |
| Cache payload | 412 × 768-dim float32 vectors + 412 × 768 symmetric vectors, `embedding_cache.npz` ≈ 2.6 MB compressed |

### Retrieval accounting (top-K = 10, 13 categories)

| Metric | Value |
| --- | --- |
| Retrieval slots | 130 |
| Distinct candidates retrieved | 84 (21.1 % of the 399-candidate corpus) |
| Slots merged as duplicates | 46 |
| Category hits per retrieved candidate | 57 × 1 hit, 18 × 2, 4 × 3, 2 × 4, 1 × 5, 2 × 6 |

No similarity threshold is applied anywhere: 130 category slots produce 84
union candidates purely from top-K rank + dedup by `candidate_id`.

### Effect of top-K on recall (no accuracy claim — embedding stage has no labels)

| K per category | Distinct candidates | Corpus coverage | In ≥2 categories | Jaccard vs previous K |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 27 | 6.8 % | 8 | — |
| 5 | 42 | 10.5 % | 16 | 0.643 |
| 10 | 84 | 21.1 % | 27 | 0.500 |
| 15 | 107 | 26.8 % | 50 | 0.785 |
| 20 | 124 | 31.1 % | 67 | 0.863 |
| 30 | 168 | 42.1 % | 91 | 0.738 |
| 50 | 228 | 57.1 % | 147 | 0.737 |

Reading: growth is monotone but not exhaustive — even K=50 reaches only
~57 % of the corpus, i.e. category-specific ranking really concentrates; the
bright head (K≤10) already contains every candidate the LLM judge will need
for the strongest picks. Multi-category candidates grow superlinearly with K
(27 → 147 as K 3 → 50): broad categories (#structural family and
`emotional_confrontation`) repeatedly claim the same dramatic beats, which
is expected behaviour, not leakage.

### Similarity score distributions (cosine, all 399 candidates per category)

| Category | min | p10 | median | p90 | max | mean | std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| betrayal | 0.574 | 0.596 | 0.618 | 0.644 | 0.666 | 0.620 | 0.018 |
| shocking_reveal | 0.608 | 0.632 | 0.653 | 0.680 | **0.732** | 0.655 | 0.019 |
| revenge | 0.562 | 0.603 | 0.626 | 0.651 | 0.668 | 0.626 | 0.018 |
| humiliation | 0.571 | 0.607 | 0.631 | 0.661 | 0.688 | 0.632 | 0.022 |
| threat | 0.570 | 0.597 | 0.619 | 0.646 | 0.674 | 0.620 | 0.018 |
| confession | 0.584 | 0.616 | 0.632 | 0.650 | 0.670 | 0.633 | 0.014 |
| romantic_tension | 0.579 | 0.613 | 0.635 | 0.659 | 0.683 | 0.636 | 0.018 |
| suspense | 0.590 | 0.614 | 0.642 | 0.666 | 0.692 | 0.641 | 0.020 |
| unexpected_entrance | 0.605 | 0.635 | 0.656 | 0.675 | 0.709 | 0.655 | 0.016 |
| emotional_confrontation | 0.603 | 0.622 | 0.645 | 0.666 | 0.688 | 0.644 | 0.017 |
| strong_opening_hook | 0.601 | 0.633 | 0.653 | 0.675 | 0.704 | 0.654 | 0.017 |
| unresolved_cliffhanger | 0.603 | 0.622 | 0.646 | 0.668 | 0.697 | 0.646 | 0.017 |
| standalone_dramatic_moment | 0.634 | 0.651 | 0.677 | 0.700 | 0.726 | 0.676 | 0.019 |

Distributions are tight (std ≈ 0.018) — the ABSOLUTE cosine level of a
document score vs a query prompt, not the gap to another candidate, carries
little meaning. Exactly why thresholds are banned and rank-ordering is used.

### Prompt-strategy comparison (asymmetric search vs symmetric similarity)

Jaccard agreement of each category's top-10 sets:

| Category | Jaccard @10 | Category | Jaccard @10 |
| --- | ---: | --- | ---: |
| betrayal | 0.111 | emotional_confrontation | 0.176 |
| shocking_reveal | 0.429 | strong_opening_hook | 0.111 |
| revenge | 0.111 | unresolved_cliffhanger | 0.000 |
| humiliation | 0.111 | suspense | 0.000 |
| threat | 0.538 | standalone_dramatic_moment | 0.333 |
| confession | 0.111 | unexpected_entrance | 0.250 |
| romantic_tension | 0.111 | | |

The two strategies disagree strongly at rank depth 10 — asymmetric
retrieval instructions matter and the documented `ASYMMETRIC_SEARCH`
question/document pairing is retained as the default. Agreement is
highest where clips resemble abstract scene summaries (`threat`,
`shocking_reveal`).

### Manual review of the top-10 subset (50 (category, candidate) pairs, 44 distinct candidates)

Reviewer read the complete dialogue + staging of every listed candidate.
Rubric: **relevant** = canonical instance a viewer would instantly name as
the category; **partial** = category present in weakened/adjacent form or
content-overlapping with a stronger ranked entry; **not** = wrong category.
Precision@K counts `relevant` + `partial` as hits.

| Category | Precision@5 | Precision@10 | Dominant error |
| --- | ---: | ---: | --- |
| shocking_reveal | 1.00 | 1.00 | — (ep35/ep42/ep43 "Alexander is alive" arc all correctly ranked) |
| standalone_dramatic_moment | 1.00 | 1.00 | one partial: ep32/ep02 windows overlap their rank neighbours |
| humiliation | 0.80 | 0.80 | the ep16 thug attack (action scene, no shame beat) and the mid-resolution ep3 rescue |
| suspense | 0.80 | 0.80 | the gentle ep6 fireside chat and one ep3 mid-resolution struggle |
| betrayal | 0.40 | 0.50 | the ep33 interrogation quintet — **coercion, not violated trust** — floods the category |

Full judgements: `retrieval_output/manual_eval_labels.json`.

**Known systematic weakness.** Five ep33 "Lord coerces the maid" windows
occupy the betrayal top-10 yet are textbook *threat/coercion* scenes. The
embedding keys on "a close person turns against", which surfaces the coerced
testimony as a weak "switching sides" signal. Options for the future (any of):
tune the betrayal description toward "affection/trust flip", add a
"coerced confession" distractor description to absorb these, or let the
LLM judge re-rank them out — the recall-first design deliberately leaves
them in. No threshold gating was applied at this stage.

### Qualitative observations worth carrying into Stage 4

* `shocking_reveal` and `standalone_dramatic_moment` retrieval is already
  judge-ready material — the ep43 "intelligence report" and ep32 dungeon
  suites are exactly the moments a promo editor wants.
* Multi-category hits (27 candidates) are a feature for the judge: the same
  clip legitimately reads as humiliation + emotional_confrontation +
  standalone_dramatic_moment. Full similarity rows travel with each result
  so the judge sees every angle.
* Structural categories rank ep11/ep3 hospital-raid/chapel-expulsion
  clusters high — plausible for "opening hook", but the LLM judge should
  verify the "works without prior knowledge" constraint explicitly.


## 7. Evaluation caveats

* Similarity scores are **not probabilities** and do not measure promo
  quality — a high score means "this candidate text sits semantically close
  to the category description", nothing more.
* There is no ground truth for promo-worthiness at this stage; the recall
  sweep and precision@K are measured against a **small, manually reviewed
  subset** labelled by reading the candidates (see `manual_eval_review.md`),
  and those precision numbers are only indicative for the reviewed categories.
* The LLM judge (Stage 4) is deliberately not integrated yet: it is the
  component that will rank and select among these recall-first results.

## 8. Tests

`python3 -m unittest test_script_parser test_context_builder test_embedding_engine
test_candidate_retriever` — 147 tests, all passing:

* Parser v2 + Context Builder regression (87) — approved baselines untouched.
* Embedding engine (26): instruction formatting byte-equality, offline provider
  determinism, config validation, normalization/truncation math, cache
  round-trips, engine cache-hit behavior, plus real-model integration smoke
  tests (768-dim, unit norms, betrayal-vs-unrelated sanity) when the local
  weights exist.
* Retriever (34): top-K determinism/tie-breaking, merge/dedup accounting,
  recall sweep, strategy agreement, score distributions, manual-label
  precision, renderers, CLI end-to-end contract, and cross-run cache reuse.
