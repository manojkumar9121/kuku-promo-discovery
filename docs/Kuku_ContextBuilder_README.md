# Kuku Context Builder (Stage 2 of the Promo Discovery Engine)

Transforms parsed screenplay events from `parsed_episodes.json` into **promo
candidate windows** that future embedding and LLM components can consume.
Stage 1 is the hardened screenplay parser (`script_parser.py`, see
`Kuku_Parser_README.md`); this stage is its downstream consumer.

It is deterministic: no LLM calls, no embeddings, no randomness, no network,
stdlib only. All arithmetic (timestamps, durations, snapping) is classical,
bit-reproducible Python — re-running the pipeline yields byte-identical output.

## Repository layout

```
Kuku FM/
├── script_parser.py                 # Stage 1: screenplay -> parsed_episodes.json (Parser v2)
├── context_builder.py               # Stage 2: parsed events -> promo candidate windows
├── test_script_parser.py            # Parser regression suite (43 tests)
├── test_context_builder.py          # Builder unit + integration suite (44 tests)
├── parsed_output/                   # Stage 1 output
│   ├── parsed_episodes.json
│   └── parse_report.json
├── context_output/                  # Stage 2 output
│   ├── candidates.json              # all candidates + config + totals
│   ├── candidates.jsonl             # one candidate per line (streaming for embeddings)
│   └── context_report.json          # generation statistics
├── Kuku_Parser_Agent_Handoff.md     # original architecture reference
└── Kuku_ContextBuilder_README.md    # this file
```

## Run

```bash
# Stage 1 (already integrated and verified — Parser v2)
python script_parser.py \
  --input "I Married a Broken Soldier. He was a Duke-scripts (2).zip" \
  --output parsed_output

# Stage 2
python context_builder.py \
  --input parsed_output/parsed_episodes.json \
  --output context_output

# Re-validate a previously written output without re-building
python context_builder.py --validate-only \
  --candidates context_output/candidates.json \
  --input parsed_output/parsed_episodes.json
```

### Full test suite

```bash
python -m unittest test_script_parser test_context_builder -v
# 87 tests, stdlib only
```

## Context Builder CLI options

| Flag | Default | Meaning |
|---|---|---|
| `--input` | required | `parsed_episodes.json` from the parser |
| `--output` | `context_output` | output directory |
| `--min-window` | `20.0` | shortest allowed promo duration (seconds) |
| `--max-window` | `60.0` | longest allowed promo duration (seconds) |
| `--target-window` | `30.0` | preferred duration; end boundary snaps near it |
| `--context-before` | `12.0` | preceding narrative context span (seconds); never part of the clip |
| `--context-after` | `12.0` | following narrative context span (seconds); never part of the clip |
| `--max-windows-per-anchor` | `1` | `1` = one window per anchor; `2` adds an earlier-boundary variant |
| `--iou` | `0.5` | temporal-IoU threshold at which two windows count as near-duplicates |
| `--anchor-types` | `dialogue` | event types used as window anchors (comma-separated) |

## How candidates are generated

1. **Anchors.** Each dialogue event (configurable via `--anchor-types`) is an
   anchor. A candidate's start is *exactly* the anchor event's start boundary.
2. **End snapping.** The window end must be an *actual event end boundary*
   lying inside `[start + min_window, start + max_window]`. The boundary closest
   to the target duration is chosen (plus an earlier-boundary variant when
   `--max-windows-per-anchor 2`). This guarantees every candidate is 20–60 s,
   every boundary carries its original timestamp string, and no invented times
   enter the pipeline.
3. **Dedup.** Windows are generated in chronological anchor order. A window is
   dropped if it duplicates an already-accepted window (exact start+end) or if
   its temporal IoU with an accepted window from *another* anchor is ≥ 0.5.
   Variants from the *same* anchor are deliberately exempt, so multi-length
   probes survive. All decisions are counted in the report.
4. **Content.** A candidate includes *every* timed event overlapping its
   interval — dialogue, camera, action and sound FX, including events that
   straddle a window edge (flagged `partial_overlap_event_count` /
   `boundary_overlap: "start"|"end"|"wraps"`, with `overlap_seconds`). Untimed
   events (rare parser edge cases) are listed separately and never count
   toward clip math.
5. **Context.** Up to N seconds of events strictly *before* and *after* the
   window (`preceding_context` / `following_context`, each item carrying the
   width of its gap to the window). Context is reasoning material only — it is
   structurally separate from the clip and never included in
   `duration_seconds`.
6. **Traceability.** Every candidate keeps original `event_id`s,
   `source_order_index`, original `MM:SS.ss` timestamp strings, scene numbers
   and episode metadata. Candidates may cross scene boundaries (flagged
   `crosses_scene_boundary`).

## Candidate JSON schema (v1.0)

Big-truth field reference:

| Field | Type | Meaning |
|---|---|---|
| `candidate_id` | string | stable id `cand_epNN_tSTART-END_hash6` |
| `episode_number` | int | episode the candidate belongs to |
| `scene_numbers` | int[] | scenes the window overlaps, in play order |
| `start_seconds` / `end_seconds` | float | clip span boundaries (event boundaries) |
| `duration_seconds` | float | **sum of segment durations** — the audio that would play; for single-segment candidates equals `span_seconds` |
| `span_seconds` | float | wall-clock span from first start to last end |
| `start_timestamp` / `end_timestamp` | string | original `MM:SS.ss` strings of the boundary events |
| `segment_count` | int | always 1 today (schema-ready for multi-segment) |
| `segments[]` | object[] | per-segment `{segment_index, start_seconds, end_seconds, duration_seconds, start_timestamp, end_timestamp, start_event_id, end_event_id, event_ids[]}` |
| `dialogue_transcript[]` | object[] | chronological dialogue items: `{event_id, event_type, scene_number, source_order_index, start_seconds, end_seconds, start_timestamp, end_timestamp, content, speaker, speaker_canonical, delivery, voice_direction, fully_contained, overlap_seconds, boundary_overlap}` |
| `actions_and_camera[]` | object[] | same shape, for `camera` / `action` / `sound_fx` events |
| `speakers[]` | object[] | per canonical speaker: `{speaker_canonical, raw_spelling, dialogue_event_ids, deliveries, dialogue_event_count}` |
| `preceding_context[]` / `following_context[]` | object[] | narrative context items (same transcript shape plus `gap_to_window_seconds`); guaranteed strictly outside the clip |
| `untimestamped_events[]` | object[] | parser events without valid time ranges that fall in the involved scenes; never in clip math |
| `anchor_event_id` | string | event whose start defines the window |
| `boundary_event_ids` | object | `{start_event_id, end_event_id}` the exact events the boundaries snap to |
| `source_event_ids` | string[] | all clip event ids, chronological |
| `context_event_ids` | string[] | ids of context items (disjoint from `source_event_ids`) |
| `event_count`, `dialogue_event_count`, `fully_contained_event_count`, `partial_overlap_event_count` | int | content counters |
| `crosses_scene_boundary` | bool | window overlaps > 1 scene |
| `scenes[]` | object[] | verbatim scene info: `{scene_number, scene_occurrence_index, heading, start_seconds, end_seconds, start_timestamp, end_timestamp}` |
| `episode` | object | `{episode_number, episode_label, logline, continuity_notes, source_file, runtime_seconds, scene_count}` |

Representative single-segment candidate (from the real pack, trimmed of
repeated content lines):

```json
{
  "candidate_id": "cand_ep01_t11.3-41.2_5a0614",
  "episode_number": 1,
  "scene_numbers": [1],
  "start_seconds": 11.3,
  "end_seconds": 41.2,
  "duration_seconds": 29.9,
  "span_seconds": 29.9,
  "start_timestamp": "00:11.30",
  "end_timestamp": "00:41.20",
  "segment_count": 1,
  "segments": [{
    "segment_index": 1,
    "start_seconds": 11.3, "end_seconds": 41.2, "duration_seconds": 29.9,
    "start_timestamp": "00:11.30", "end_timestamp": "00:41.20",
    "start_event_id": "ep1_..._ev008", "end_event_id": "ep1_..._ev027",
    "event_ids": ["ep1_..._ev007", "ep1_..._ev008", "ep1_..._ev009", "..."]
  }],
  "dialogue_transcript": [{
    "event_id": "ep1_..._ev008", "event_type": "dialogue", "scene_number": 1,
    "source_order_index": 8,
    "start_seconds": 11.3, "end_seconds": 13.1,
    "start_timestamp": "00:11.30", "end_timestamp": "00:13.10",
    "content": "\"You should turn around, Juliana.\"",
    "speaker": "MATRIARCH", "speaker_canonical": "MATRIARCH",
    "delivery": "Cold, Commanding", "voice_direction": [],
    "fully_contained": true, "overlap_seconds": 1.8, "boundary_overlap": null
  }],
  "actions_and_camera": [{
    "event_id": "ep1_..._ev007", "event_type": "camera", "source_order_index": 7,
    "start_seconds": 11.0, "end_seconds": 13.2,
    "start_timestamp": "00:11.00", "end_timestamp": "00:13.20",
    "content": "Medium shot as Juliana stops midway down the aisle...",
    "fully_contained": false, "overlap_seconds": 1.9, "boundary_overlap": "start"
  }],
  "speakers": [
    {"speaker_canonical": "MATRIARCH", "raw_spelling": "MATRIARCH",
     "dialogue_event_ids": ["ep1_..._ev008", "ep1_..._ev013", "ep1_..._ev017"],
     "deliveries": ["Cold, Commanding", "Blunt, Unforgiving", "Dismissive"],
     "dialogue_event_count": 3},
    {"speaker_canonical": "JULIANA", "raw_spelling": "LADY JULIANA",
     "deliveries": ["Confused, Breathless", "Desperate, Unbelieving",
                    "Trembling, Pleading", "Shaken, Whispering", "Heartbroken"],
     "dialogue_event_count": 5},
    {"speaker_canonical": "LADY IN GOLD GOWN", "raw_spelling": "LADY IN GOLD GOWN",
     "deliveries": ["Gloating, Smug"], "dialogue_event_count": 1}
  ],
  "preceding_context": [{
    "event_id": "ep1_..._ev001", "event_type": "camera",
    "start_timestamp": "00:00.00", "end_timestamp": "00:04.80",
    "content": "Symmetrical framing on the heavy church doors...",
    "gap_to_window_seconds": 6.5
  }],
  "following_context": [{
    "event_id": "ep1_..._ev028", "event_type": "camera",
    "gap_to_window_seconds": 0.3,
    "content": "Close-up on Juliana as tears spill over..."
  }],
  "untimestamped_events": [],
  "anchor_event_id": "ep1_..._ev008",
  "boundary_event_ids": {"start_event_id": "ep1_..._ev008", "end_event_id": "ep1_..._ev027"},
  "source_event_ids": ["ep1_..._ev007", "ep1_..._ev008", "ep1_..._ev009", "..."],
  "context_event_ids": ["ep1_..._ev001", "...", "ep1_..._ev031"],
  "event_count": 21, "dialogue_event_count": 9,
  "fully_contained_event_count": 19, "partial_overlap_event_count": 2,
  "crosses_scene_boundary": false,
  "scenes": [{"scene_number": 1, "scene_occurrence_index": 1,
              "heading": "INT. CATHEDRAL – DAY",
              "start_seconds": 0.0, "end_seconds": 138.0,
              "start_timestamp": "00:00.00", "end_timestamp": "02:18.00"}],
  "episode": {"episode_number": 1, "episode_label": "Episode 1",
              "logline": "Lady Juliana arrives at the cathedral...",
              "continuity_notes": "First episode of the series...",
              "source_file": ".../episode-001-Episode 1671703.txt",
              "runtime_seconds": 138.0, "scene_count": 2}
}
```

### Multi-segment support

`segments` is the single point of extension: `_assemble_candidate()` accepts
any number of segment specs, and `candidates.json` already records
`duration_seconds` as the **sum** of segment durations while `span_seconds`
spans the first start through the last end. A later recommendation stage can
produce a two-piece promo (e.g. cold-open teaser + climax `{0-10s} + {40-50s}`
→ `duration=20`, `span=50`) with no schema change; the first implementation
emits exactly one continuous segment per candidate by design.

## Validation

The builder validates its own output by **independent recomputation** from the
parser JSON (never by re-reading its own intermediate state): every clip's
event membership, scene intersections, boundary provenance, timestamp
verbatim-ness, context leakage, duration bounds and cross-candidate near-dup
scores are recomputed and compared. `context_builder.py --validate-only`
exits non-zero with explicit per-violation messages if anything drifts, and
the same function powers the test suite.

## Results on the supplied 50-episode pack

Stage 1 (Parser v2) baseline — unchanged and re-verified:

| Metric | Result |
|---|---:|
| Screenplay files | 50 |
| Scenes | 100 |
| Events | 3,727 |
| Camera / Dialogue / Action / Sound FX | 1,754 / 1,487 / 372 / 114 |
| Parser warnings | 1 (`EVENT_OUTSIDE_SCENE`, ep 22 sound FX 0.30 s past its scene end — retained verbatim) |

Stage 2 (this builder, defaults):

| Metric | Result |
|---|---:|
| Candidate windows | 399 (min 4 / median 8 / max 15 per episode) |
| Dialogue anchors considered | 1,487 |
| Near-duplicates dropped (IoU ≥ 0.5) | 922 |
| Exact duplicates dropped | 0 |
| Anchors without qualifying end boundary | 166 (all within 20 s of episode tail) |
| Duration min / median / mean / max | 20.2 s / 30.0 s / 29.8 s / 33.5 s |
| Events referenced across all candidates | 7,542 (~2× pack coverage: overlaps included) |
| Candidates including partial edge overlaps | 368 |
| Candidates crossing a scene boundary | 86 |
| Distinct canonical speakers in candidates | 60 |
| Mean preceding / following context items | 7.3 / 6.8 |
| Untimestamped events in pack | 0 |

## Architectural boundaries (deliberate)

* **No EmbeddingGemma, no vector work, no LLM calls.** The output is plain
  JSON; the embedding and ranking stages are future consumers.
* **No LLM for deterministic timestamp math.** All snapping/durations are
  computed in Python and re-validated; timestamps are reused verbatim from the
  parser rather than recomputed.
* **The builder is independent of the parser**: it imports nothing from
  `script_parser.py`, it only reads its JSON contract. Future audio/video
  adapters can emit the same intermediate representation and the builder will
  consume them unchanged.
* **No over-processing:** candidates are ranked nowhere; dedup is purely
  geometric. Ordering/selection is the job of the next stage.

## Current limitations

* Windows never extend past the last event boundary; dialogue anchors in the
  final ~20 s of an episode cannot form valid windows (counted per episode in
  `context_report.json`).
* Near-duplicate suppression is purely temporal (IoU): two windows with
  very different narratives that happen to overlap heavily will still be
  deduplicated; semantic ranking happens in a later stage.
* Multi-segment generation is supported by the schema and the assembly code,
  but no candidate generator emits multiple segments yet — by design.
