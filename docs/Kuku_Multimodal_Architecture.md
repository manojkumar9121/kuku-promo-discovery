# Kuku — Multimodal Architecture & Future Audio/Video Extension

Status at submission: **only the screenplay (text) path is implemented and
evaluated.** The audio and video adapters below are *designed*, not built;
every section calling them "planned" or "future work" is a promise with an
interface contract, not shipped code. Nothing in this document should be read
as completed multimodal functionality.

---

## 1. What the screenplay ingestion does today (IMPLEMENTED)

The project exploits a fact about AI-generated microdramas: screenplays are
the *source of truth* and they already describe most of what an audio/video
adapter would otherwise have to detect. The ingestion pipeline therefore
reads one unified, timestamped token stream per episode:

```
TXT pack (50 episodes)
  └─► script_parser.py      → ordered TimestampedEvents (camera/dialogue/action/sound_fx),
        scene + episode scaffolds, warnings (missing timestamps, scene overshoot) kept verbatim
  └─► context_builder.py    → candidate windows anchored on dialogue event boundaries:
        (target 30s, 20–60s clamp, ±12s context retained, scene-boundary cross allowed & flagged)
  └─► candidate_retriever.py→ text embeddings (EmbeddingGemma-2, 768-dim, float32 CPU)
        → union-84 → blind LLM judge (kuku-judge-v1.0.0) → 2-per-arc ranking
  └─► promo_ranker.py       → featured/reserve + full evaluation comparison
```

The `TimestampedEvent` contract is the whole trick. Every event from every
pass through the system carries the same shape:

```json
{"event_id": "ep1_..._ev008", "event_type": "dialogue|camera|action|sound_fx",
 "start_seconds": 11.3, "end_seconds": 13.1, "content": "...",
 "speaker_canonical": "JULIANA", "delivery": "Trembling, Pleading",
 "fully_contained": true, "context_out_of_candidate": false}
```

Three remarkable properties already in place:

1. **Heterogeneous event types are already interleaved** in source order via
   `source_order_index`: the interleaved `actions_and_camera` vs
   `dialogue_transcript` distinction is purely *descriptive*, not structural.
   A future `camera_push_in` from *video* or `gasp_reaction` from *audio*
   parses as more rows in the same ledger.
2. **Context windows are modality-agnostic**: ±12 s of whatever precedes/follows
   is kept as context; new event types just inherit it.
3. **The judge's evidence contract generalizes**: the trial validates ids
   against "the candidate's event universe". Add event types ⇒ the universe
   grows ⇒ the same validation holds with zero conceptual change.

## 2. Planned audio adapter (`AudioAdapter`) — FUTURE WORK

Interface contract (not implemented):

```python
class AudioAdapter(Protocol):
    """Produces timestamped events from an episode's audio track."""
    def ingest(self, audio_path: Path, episode_meta: dict) -> list[TimestampedEvent]: ...
    # Required behaviour for any implementation:
    # * emit event_type ∈ {voice_over, scream, gasp, door_slam, silence_drop,
    #                      music_sting, overlapping_argument, whisper, cry}
    # * timestamps in episode-common seconds, ±10 ms tolerance, monotonic
    # * event_id in the scheme ep{N}_aud{order}_{obs}_evNNN  (id disjointness
    #   enforced against text-path ids by a uniqueness check)
    # * populate `confidence` [0,1] per event; below-threshold events enter a
    #   `weak_signals` list, never silently dropped
    # * stop at the episode end (no cross-episode bleed)
```

Planned event generation (options, to be benchmarked — none implemented):

| Source | Signal | Adapter behaviour |
| --- | --- | --- |
| faster-whisper/whisperX | words + times | `voice_over`, dialogue-confirmation pass (agreement check vs text) |
| VAD + energy segmentation | screams, gasps, door slams | `scream/gasp/door_slam/silence_drop` (threshold-based segments) |
| pyannote or light VAD | overlapping argument | `overlapping_argument` (2+ simultaneous voices above floor) |
| loudness dynamics | music_sting | `music_sting`: sudden 2 s energy jump post-silence |
| spectral classifier | whisper vs speech | `whisper` (underestimated in text; high-value hook) |

**Fusion seam:** `ingest()` output plugs directly into the existing
`TimestampedEvent` list before Context Builder — no downstream change.
The candidate document picks up an `Audio cues:` block automatically.

## 3. Planned video adapter (`VideoAdapter`) — FUTURE WORK

Interface contract (not implemented):

```python
class VideoAdapter(Protocol):
    """Produces timestamped events from an episode's visual track."""
    def ingest(self, video_path: Path, episode_meta: dict) -> list[TimestampedEvent]: ...
    # event_type ∈ {hard_cut, push_in, face_close_up, slap_contact,
    #               door_reveal, tears_on_face, crowd_reaction, text_overlay,
    #               slow_zoom_reveal, blood_or_bruising}
    # Same timestamp contract, same id scheme with `_vid_` infix.
```

Planned event generation: shot-boundary detection (histogram delta),
face-size heuristic for close-ups, simple temporal-CLIP / VideoMAE scoring
against a handful of promo-relevant action prompts (kept at embedding level,
no per-frame LLM calls), text-on-screen OCR for on-screen title beats.
Dense frames are never fed to the judge; only derived event rows are.

**Embedding path for adapters (sketch):** modal-specific embeds are stored in
an `.npz` parallel store (float32, 768-dim, aligned to the shared query
space by a learned linear projection fitted on text–audio pairs from the
same timeline). Retrieval treats projected vectors exactly as the current
pipeline treats candidate text vectors; the merge/dedup/judge/ranker path is
untouched.

## 4. What fuses with the judge today if adapters land

```text
video frames ─► VideoAdapter ─► TimestampedEvents ─┐
audio waveform ─► AudioAdapter ─► TimestampedEvents ├─► ONE merged event list
screenplay txt ─► script_parser ─► TimestampedEvents┘         │
                                                              ▼
                                              context_builder (±12s retained)
                                                              ▼
                                              candidate windows (modality-free)
                                                              ▼
                    EmbeddingGemma-2 text catalog (+ projected modal vectors)
                                                              ▼
                    blind LLM judge — payload gains `audio_cues`, `video_cues`
                    sections; validation universe absorbs new ids automatically
                                                              ▼
                    ranking, diversity cap, human review (unchanged)
```

## 5. Honest inventory (at submission time)

| Capability | State |
| --- | --- |
| Screenplay → timestamped events (parser v2) | **implemented, evaluated** |
| Candidate windows with ±12s context (builder) | **implemented, evaluated** |
| 13-category embed catalog + Fast Discovery | **implemented, evaluated** |
| Blind LLM judge + ranking + human sample | **implemented, evaluated** |
| Audio adapter & its events | ready-to-write; contract above; **not implemented** |
| Video adapter & its events | ready-to-write; contract above; **not implemented** |
| Cross-modal embedding alignment | research-grade; **not implemented** |
| Multi-segment (cross-cluster) promos | supported in schema; runs produced single-segment only; **not benchmarked** |
| Alignment solver (audio↔screenplay vs video↔screenplay sync) | **not implemented** |

## 6. Suggested build order & risks

1. Audio adapter first — cheapest signals, highest red-flag value
   (scream/gasp detection is exactly the hook the screenplay denotes).
   Risk: whisper-word drift on music-heavy segments → require ffprobe
   scene timestamps as weak priors; drop low-agreement segments.
2. Video adapter (shot + face-size) as second. Risk: compute cost —
   run at 4 fps; store embeddings sparsely, never decode in batch jobs.
3. Projection layer (~200 samples of same-timeline pairs) — validate
   retrieval lift before wiring the judge field.
4. Re-run blind judging; extend `guard:` (evidence ids from union of
   allowed types) — free but must be explicitly extended in `judge_schema.py`.
5. Cost ceiling: single GPU or 8-core CPU with batch embeddings; the
   latency dominates only at fusion calibration time.
