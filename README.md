# Kuku Promo Discovery — AI Generalist Assignment (Problem 1: Find the Promos)

Finds the promo moments in a timestamped microdrama screenplay, measures
stake vs. a 100% AI pipeline, and ranks candidates for paid placement.
Two modes: **Fast Discovery** (embedding retrieval, high recall) and a
**blind LLM Judge** (contextual selection), with a diversity-aware ranker.

> **Confidentiality (assignment fine print).** The sample show's screenplay
> pack is Kuku's content and is **not** included in this repository. Place it
> at `input/` yourself (see *Input* below). Raw outputs full of the pack's own
> dialogue are likewise excluded; the numbers-only evaluation files are under
> `metrics/`. The reviewer-facing Drive folder carries those raw outputs.
> Nothing verbatim from the pack is committed here.

## What's implemented vs planned

| Area | State |
|---|---|
| Parser → context builder → embedding recall → blind LLM judge → diversity ranking | **implemented + evaluated** (see `docs/Kuku_Solution.pdf`, 2 pp) |
| Audio adapter (`AudioAdapter`) | **future work** (interface contract in `docs/Kuku_Multimodal_Architecture.md`) |
| Video adapter (`VideoAdapter`) | **future work** (ditto) |
| Cross-modal embedding projection | **future work** |
| Multi-segment (cross-cluster) promos | schema-supported; **not benchmarked** in this submission |

## Layout

```
src/        pipeline code (python 3.10+, stdlib + numpy)
tests/      198-test suite (unittest) — runs offline, no model download needed
docs/       architecture + operations + the 2-page submission PDF
metrics/    evaluation artifacts (IDs + numbers only — no pack-derived text)
input/      place the supplied pack folder here (untracked by git)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy
# 1.55 GB model, once (text-only path); or point --model-path at any local snapshot
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/embeddinggemma-2', local_dir='models/embeddinggemma-2')"
```

The judge stage targets a llama.cpp OpenAI-compatible endpoint
(`--base-url`, `--model`); the test suite and the *embedding* stages need no
server. Swap judge models in one line (`llm_judge.build_provider`).

## Run

```bash
# 1) parse (50 episodes, one warning kept verbatim)
python src/script_parser.py --input "input/I Married a Broken Soldier. He was a Duke" --output parsed_output/
# 2) build candidates (30 s targets, 20–60 s clamps, ±12 s context retained)
python src/context_builder.py --input parsed_output/parsed_episodes.json --output context_output/
# 3) Fast Discovery — writes top-K per category, deduped, high recall
python src/candidate_retriever.py --candidates context_output/candidates.json --output retrieval_output/
# 4) blind LLM judge — no retrieval data in the prompt, event-id evidence enforced
python src/llm_judge.py --retrieval-results retrieval_output/fast_discovery_results.json \
    --candidates context_output/candidates.json --output judge_output/
# 5) ranking + final recommendation view (2-per-arc diversity cap)
python src/promo_ranker.py --judge-results judge_output/judge_results.json \
    --candidates context_output/candidates.json --retrieval-results retrieval_output/fast_discovery_results.json \
    --output judge_output/
python src/final_recommendations.py --output judge_output/
```

## Verify

```bash
cd tests && python3 -m unittest test_script_parser test_context_builder \
  test_embedding_engine test_candidate_retriever test_llm_judge test_promo_ranker
```

## Headline results (pack in the Drive folder)

| Signal | Value |
|---|---|
| Parser → candidates | 50 eps / 100 scenes / 3,727 events → 399 windows |
| Fast Discovery | 84 distinct from 130 slots, 46 merged dups, 21.1% corpus coverage |
| Judge pass | 84 ok · 53 accept · 31 reject · mean 3.43 / median 3.55 |
| Independent human review (33 pairs) | P@5 = P@10 = P@20 = **1.00**; 30/33 judge↔reviewer agreement |
| Embed-vs-judge | Spearman ρ ≈ 0.04; Jaccard@10 = 5.3% (orthogonal lenses — kept on purpose) |
| Diversity of featured slate | 49 candidates · 19 episodes · 41 arcs · cap 2 per (episode, arc) |

Full discussion, honest failure cases and scale-out plan: `docs/Kuku_Solution.pdf`.

## Notes for running on a different show

Ep/scene scaffold and timestamp regex are already packed-agnostic; alias maps
are needed only for recurring epithets (`LADY CATHERINE → CATHERINE` etc.).
Pass `--aliases aliases.json` to `script_parser.py`. Retrieval re-embeds only
changed candidates (content-addressed cache); the judge judging only changed
candidates (prompt-version + fingerprint cache).
