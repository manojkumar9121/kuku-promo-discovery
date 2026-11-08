# 3-Minute Walkthrough Script — Kuku Promo Discovery

Format: screen recording in your own voice. Keep under 3:00. Targets the five lenses: problem framing, approach, quality checking, the working prototype, how you ran it.

---

**0:00–0:20 — Hook (the problem).**
"Kuku releases a show every week. An editor watches fifty episodes and hand-picks promo moments for Instagram and TikTok. I built the AI first pass: it finds the best 20-to-60-second clips per episode and ranks them for placement. Here is what it produced on the supplied pack, and where it fails."

**0:20–0:45 — The input (trust the parser).**
Show terminal: `python src/script_parser.py --input input/… --output parsed_output/`.
Narrate: "The pack has one timestamped line per scene, dialogue and action. The parser builds an ordered event ledger — 50 episodes, 100 scenes, 3,727 events — keeping every anomaly verbatim instead of cleaning it. One warning lives in the report: episode 22, a sound FX a half-second after the scene closes. It stays observable; no silent correction."

**0:45–1:10 — The funnel (fast, then careful).**
Show terminal: `python src/context_builder.py` then `python src/candidate_retriever.py`.
Narrate: "Windows built on dialogue boundaries, ~30 seconds: 399 candidates, no camera movement invented. Two lenses then rank them. Lens one: embedding retrieval over thirteen drama categories — high recall, deliberately no similarity thresholds. Lens two: a blind LLM judge that never sees the retrieval scores — it only reads the clip and ±12 seconds of context, scoring hook, cliffhanger, standalone clarity, progression and impact, with event-evidence enforced."

**1:10–1:40 — The result (show, don't tell).**
Show `judge_output/final_recommendations.md` — scroll the top 5.
Narrate: "Top clip: episode 1, the altar abandonment — 4.4, hook 5, cliffhanger 4. Episode 2's 'I cannot marry you' cold open — 4.4. The judge rejected 31 of 84, including the embedding's favorite — a backstory monologue with dead air at the top, useless as an ad. Embedding and judge disagree structurally — Spearman point-oh-four — and that disagreement is the product working: the retrieval widens cheaply, the judge filters expensively."

**1:40–2:20 — Human validation (and the honesty).**
Show `metrics/ranking_report.json` + `metrics/judge_human_sample.json`.
Narrate: "Independent check, none of it tuned: I hand-read the top-twenty and some rejected ones — 33 labels. Precision at five, ten and twenty is one-point-oh. Judge and reviewer agree thirty of thirty-three times. The three misses are all cases where the judge was stricter than me on promo-value — recorded, not hidden."

**2:20–2:45 — Where it fails + honesty contract.**
"It wraps a text-only world: no audio scream, no visual slap. Audio/video adapters are designed, not built — interface contracts are in `docs/Kuku_Multimodal_Architecture.md`. Multi-segment promos are schema-supported, not benchmarked — I did not claim them. One prompt, one model, one English-speaking pack."

**2:45–3:00 — Prototype taste.**
"Run it fresh: five commands in the README, no API keys, about twenty minutes; or warm-cache re-screen in seconds. Raw outputs, rejection log and the reviewer labels are all in the submission folder — nothing was deleted. Thanks."
