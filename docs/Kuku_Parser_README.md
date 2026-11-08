# Kuku Screenplay Parser

A deterministic, dependency-free parser for Kuku's timestamped screenplay `.txt` files.

## Run

```bash
python script_parser.py \
  --input "/path/to/screenplay-pack.zip" \
  --output parsed_output
```

The input may be a `.zip`, a single `.txt`, or a directory containing `.txt` files. File-extension matching is case-insensitive.

Optional manual speaker aliases:

```json
{
  "THE FORMER GROOM": "GERALD",
  "LADY JULIANA": "JULIANA"
}
```

```bash
python script_parser.py --input scripts.zip --output parsed_output --aliases aliases.json
```

## Test

```bash
python -m unittest -v test_script_parser.py
```

The test suite uses only Python's standard library.

## Output files

- `parsed_episodes.json`: episode metadata, scenes, events and warnings.
- `parse_report.json`: aggregate counts, applied speaker aliases, alias provenance and warnings.

## Data-ordering contract

- `scenes[].events` remain in original screenplay/source order.
- Each event has a 1-based `source_order_index` across its episode.
- `timeline_event_ids` provides a stable chronological view sorted by start time, then end time, then source order. Events without valid timestamps are placed after timed events.
- Overlapping camera, action, dialogue and sound events are retained. They are not treated as parse errors merely because their times overlap.
- Original speaker spelling, timestamp strings, raw headers, raw blocks, source filename and source line are retained.
- `speaker_canonical` is normalized for matching. `speaker_alias_source` is `identity`, `inferred` or `manual`.

## Validation behaviour

The parser reports malformed/missing timestamps, missing scene fields, duplicate metadata/scene fields, duplicate scene or episode numbers, missing script boundary markers, empty content, and events outside scene bounds. Warnings do not silently rewrite the source data.

Manual aliases are validated for non-empty string keys/values and cycles. Matching is case-insensitive and whitespace-normalized. Automatically inferred `LADY`/`THE` aliases are applied only when the unprefixed speaker exists in the same corpus; they are marked as inferred.

Source files are size-limited. UTF-8 and UTF-16 BOM files are supported; invalid UTF-8 is decoded using Windows-1252 with a warning rather than silently replacing characters.

## Validation against the supplied Kuku pack

The parser was run on the supplied 50-episode ZIP:

| Metric | Result |
|---|---:|
| Screenplay files | 50 |
| Scenes | 100 |
| Events | 3,727 |
| Camera | 1,754 |
| Dialogue | 1,487 |
| Action | 372 |
| Sound FX | 114 |
| Parser warnings | 1 |

The single warning is a sound-effect event in Episode 22 ending at `00:42.80`, while its scene ends at `00:42.50`. The parser retains the original event timestamp and reports the discrepancy; it does not clip the event.

These counts validate extraction against this pack only. New screenplay formats should be added as fixtures and tested before claiming general format support.
