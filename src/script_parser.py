#!/usr/bin/env python3
"""Robust deterministic parser for Kuku's timestamped screenplay TXT files.

Supported inputs
----------------
* A directory containing screenplay .txt files (extension matching is case-insensitive)
* A single .txt screenplay
* A .zip archive containing screenplay .txt files

Outputs
-------
* parsed_episodes.json: normalized episode/scene/event data
* parse_report.json: extraction counts, alias decisions and parser warnings

The parser is intentionally deterministic and has no third-party dependencies.
It preserves source order, records source locations, and also emits a chronological
list of event IDs for downstream timeline/context-building code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional


# Timestamp forms: MM:SS, MM:SS.ss, HH:MM:SS, HH:MM:SS.ss.
# In MM:SS form, MM may exceed 59 (e.g. 75:12 for a 75-minute runtime).
_TIMESTAMP_RE = re.compile(
    r"^(?:(?P<hours>\d{1,2}):)?(?P<minutes>\d{1,2}):(?P<seconds>\d{1,2}(?:\.\d+)?)$"
)
_EVENT_TIME_RE = re.compile(
    r"^\s*Start:\s*\(?\s*([^)|]+?)\s*\)?\s*\|\s*End:\s*\(?\s*([^)|]+?)\s*\)?\s*$",
    re.IGNORECASE,
)
_START_PREFIX_RE = re.compile(r"^\s*Start\s*:", re.IGNORECASE)
_SCENE_RE = re.compile(r"^\s*SCENE\s+(?P<number>\d+)\s*$", re.IGNORECASE)
_BOLD_HEADER_RE = re.compile(r"^\s*\*\*(?P<header>.+?)\*\*\s*$")
_METADATA_RE = re.compile(
    r"^\s*(?P<key>EPISODE NUMBER|EPISODE LOGLINE|CONTINUITY NOTES):\s*(?P<value>.*)$",
    re.IGNORECASE,
)
_SCENE_FIELD_RE = re.compile(
    r"^\s*(?P<key>Scene heading|Scene start|Scene end):\s*(?P<value>.*)$",
    re.IGNORECASE,
)

KNOWN_EVENT_HEADERS = {
    "CAMERA": "camera",
    "ACTION": "action",
    "SOUND FX": "sound_fx",
    "SOUND EFFECT": "sound_fx",
    "SOUND EFFECTS": "sound_fx",
}

SCENE_BOUNDS_TOLERANCE_SECONDS = 0.05
_ALIAS_PREFIXES = ("LADY", "THE")
_VOICE_MARKERS = {
    "V.O.", "VO", "O.S.", "OS", "O.C.", "OC", "CONT'D",
    "CONTINUING", "OFF-SCREEN", "OFFSCREEN",
}
MAX_SOURCE_FILES = 2_000
MAX_SOURCE_FILE_BYTES = 50 * 1024 * 1024
MAX_TOTAL_SOURCE_BYTES = 500 * 1024 * 1024


@dataclass
class ParseWarning:
    code: str
    message: str
    source_file: str
    scene_number: Optional[int] = None
    line_number: Optional[int] = None
    raw_text: Optional[str] = None


@dataclass
class Event:
    event_id: str
    episode_number: Optional[int]
    scene_number: Optional[int]
    event_index_in_scene: int
    event_type: str
    start_seconds: Optional[float]
    end_seconds: Optional[float]
    start_timestamp: Optional[str]
    end_timestamp: Optional[str]
    content: str
    speaker: Optional[str] = None
    speaker_canonical: Optional[str] = None
    speaker_alias_source: Optional[str] = None  # identity | inferred | manual
    delivery: Optional[str] = None
    voice_direction: list[str] = field(default_factory=list)
    raw_header: Optional[str] = None
    raw_block: Optional[str] = None
    source_file: Optional[str] = None
    source_line_start: Optional[int] = None
    source_order_index: Optional[int] = None
    scene_occurrence_index: Optional[int] = None


@dataclass
class Scene:
    scene_number: int
    heading: Optional[str]
    start_seconds: Optional[float]
    end_seconds: Optional[float]
    start_timestamp: Optional[str]
    end_timestamp: Optional[str]
    events: list[Event] = field(default_factory=list)
    scene_occurrence_index: int = 1


@dataclass
class Episode:
    episode_number: Optional[int]
    episode_label: Optional[str]
    logline: Optional[str]
    continuity_notes: Optional[str]
    source_file: str
    scenes: list[Scene] = field(default_factory=list)
    warnings: list[ParseWarning] = field(default_factory=list)

    @property
    def events(self) -> list[Event]:
        """Events in original screenplay order (never silently reordered)."""
        return [event for scene in self.scenes for event in scene.events]

    @property
    def chronological_events(self) -> list[Event]:
        """Stable timeline view; source order breaks ties and is retained on each event."""
        return sorted(
            self.events,
            key=lambda event: (
                event.start_seconds is None,
                event.start_seconds if event.start_seconds is not None else float("inf"),
                event.end_seconds is None,
                event.end_seconds if event.end_seconds is not None else float("inf"),
                event.source_order_index if event.source_order_index is not None else float("inf"),
            ),
        )


@dataclass
class SourceDocument:
    name: str
    text: str
    decode_warning: Optional[str] = None


def _normalize_name(value: str) -> str:
    """Normalize names for matching without changing the preserved source text."""
    return re.sub(r"\s+", " ", value.strip()).upper()


def timestamp_to_seconds(value: str) -> float:
    """Convert MM:SS[.sss] or HH:MM:SS[.sss] into seconds with range checks."""
    value = value.strip()
    has_open = value.startswith("(")
    has_close = value.endswith(")")
    if has_open != has_close:
        raise ValueError(f"Unbalanced timestamp parentheses: {value!r}")
    if has_open:
        value = value[1:-1].strip()

    match = _TIMESTAMP_RE.fullmatch(value)
    if not match:
        raise ValueError(f"Unsupported timestamp format: {value!r}")

    hours_text = match.group("hours")
    hours = int(hours_text or 0)
    minutes = int(match.group("minutes"))
    seconds = float(match.group("seconds"))
    if seconds >= 60:
        raise ValueError(f"Seconds component must be below 60: {value!r}")
    if hours_text is not None and minutes >= 60:
        raise ValueError(f"Minutes component must be below 60 in HH:MM:SS: {value!r}")
    return hours * 3600 + minutes * 60 + seconds


def _normalize_header(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().rstrip(":")).upper()


def _split_speaker_header(header: str) -> tuple[str, Optional[str], list[str]]:
    """Return speaker, delivery/emotion and production/voice directions."""
    value = header.strip().rstrip(":").strip()
    groups = list(re.finditer(r"\(([^()]*)\)", value))
    if not groups:
        return value, None, []

    speaker = value[:groups[0].start()].strip()
    parenthetical_values = [match.group(1).strip() for match in groups]
    delivery = parenthetical_values[-1] if parenthetical_values else None
    voice_direction = parenthetical_values[:-1]

    if len(parenthetical_values) == 1 and _normalize_name(parenthetical_values[0]) in _VOICE_MARKERS:
        delivery = None
        voice_direction = parenthetical_values

    return speaker or value, delivery, voice_direction


def _validate_manual_aliases(manual: Optional[dict[str, str]]) -> dict[str, str]:
    """Validate and normalize a manual alias map; reject cycles deterministically."""
    if manual is None:
        return {}
    if not isinstance(manual, dict):
        raise ValueError("Manual aliases must be a JSON object mapping speaker names to canonical names.")

    normalized: dict[str, str] = {}
    for raw_key, raw_value in manual.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ValueError("Every manual alias key must be a non-empty string.")
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(f"Manual alias target for {raw_key!r} must be a non-empty string.")
        key = _normalize_name(raw_key)
        value = _normalize_name(raw_value)
        if key in normalized and normalized[key] != value:
            raise ValueError(f"Conflicting manual alias definitions for {raw_key!r}.")
        normalized[key] = value

    # Resolve every chain once here so cycles fail before parsing output is written.
    for origin in normalized:
        current = origin
        visited: list[str] = []
        visited_set: set[str] = set()
        while current in normalized:
            if current in visited_set:
                cycle_start = visited.index(current)
                chain = " -> ".join([*visited[cycle_start:], current])
                raise ValueError(f"Cycle detected in manual speaker aliases: {chain}")
            visited.append(current)
            visited_set.add(current)
            current = normalized[current]
    return normalized


def _resolve_alias_target(name: str, manual: dict[str, str]) -> str:
    current = name
    visited: set[str] = set()
    while current in manual:
        if current in visited:  # Defensive; cycles are also checked at load time.
            raise ValueError(f"Cycle detected while resolving speaker alias {name!r}.")
        visited.add(current)
        current = manual[current]
    return current


def _build_speaker_alias_details(
    speakers: Iterable[str], manual: Optional[dict[str, str]] = None
) -> tuple[dict[str, str], dict[str, str]]:
    """Return raw-name -> canonical-name and raw-name -> decision source."""
    manual_normalized = _validate_manual_aliases(manual)
    raw_speakers = {speaker.strip() for speaker in speakers if isinstance(speaker, str) and speaker.strip()}
    normalized_speakers = {_normalize_name(speaker) for speaker in raw_speakers}

    aliases: dict[str, str] = {}
    sources: dict[str, str] = {}
    for raw_name in sorted(raw_speakers, key=lambda value: (_normalize_name(value), value)):
        normalized_name = _normalize_name(raw_name)
        if normalized_name in manual_normalized:
            canonical = _resolve_alias_target(normalized_name, manual_normalized)
            source = "manual"
        else:
            canonical = normalized_name
            source = "identity"
            for prefix in _ALIAS_PREFIXES:
                marker = prefix + " "
                if normalized_name.startswith(marker):
                    remainder = normalized_name[len(marker):].strip()
                    if remainder in normalized_speakers:
                        canonical = remainder
                        source = "inferred"
                        break
        aliases[raw_name] = canonical
        sources[raw_name] = source
    return aliases, sources


def build_speaker_aliases(
    speakers: Iterable[str], manual: Optional[dict[str, str]] = None
) -> dict[str, str]:
    """Public helper returning raw speaker spellings mapped to canonical names."""
    aliases, _ = _build_speaker_alias_details(speakers, manual)
    return aliases


def _apply_speaker_aliases(
    episodes: list[Episode], manual: Optional[dict[str, str]] = None
) -> tuple[dict[str, str], dict[str, str]]:
    speakers = [event.speaker for episode in episodes for event in episode.events if event.speaker]
    aliases, sources = _build_speaker_alias_details(speakers, manual)
    normalized_lookup: dict[str, tuple[str, str]] = {}
    for raw_name, canonical in aliases.items():
        normalized_lookup[_normalize_name(raw_name)] = (canonical, sources[raw_name])

    applied: dict[str, str] = {}
    provenance: dict[str, str] = {}
    for episode in episodes:
        for event in episode.events:
            if not event.speaker:
                continue
            canonical, source = normalized_lookup.get(
                _normalize_name(event.speaker), (_normalize_name(event.speaker), "identity")
            )
            event.speaker_canonical = canonical
            event.speaker_alias_source = source
            if canonical != _normalize_name(event.speaker):
                applied[event.speaker] = canonical
                provenance[event.speaker] = source
    return dict(sorted(applied.items())), dict(sorted(provenance.items()))


def _decode_bytes(raw: bytes) -> tuple[str, Optional[str]]:
    """Decode common text encodings without silently replacing invalid bytes."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return raw.decode("utf-16"), None
        except UnicodeDecodeError as exc:
            raise ValueError(f"Could not decode UTF-16 source: {exc}") from exc
    try:
        return raw.decode("utf-8-sig"), None
    except UnicodeDecodeError:
        # Windows screenplay exports are sometimes cp1252. Preserve text and flag it.
        return raw.decode("cp1252"), "UTF-8 decoding failed; decoded source using Windows-1252 fallback."


def _make_source_document(name: str, raw: bytes) -> SourceDocument:
    if len(raw) > MAX_SOURCE_FILE_BYTES:
        raise ValueError(
            f"Source file {name!r} is {len(raw)} bytes; limit is {MAX_SOURCE_FILE_BYTES} bytes."
        )
    text, decode_warning = _decode_bytes(raw)
    return SourceDocument(name=name, text=text, decode_warning=decode_warning)


def _read_sources(input_path: Path) -> list[SourceDocument]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    docs: list[SourceDocument] = []
    total_bytes = 0

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as archive:
            infos = [
                info for info in archive.infolist()
                if not info.is_dir() and PurePosixPath(info.filename).suffix.lower() == ".txt"
            ]
            if len(infos) > MAX_SOURCE_FILES:
                raise ValueError(f"Archive contains {len(infos)} TXT files; limit is {MAX_SOURCE_FILES}.")
            for info in sorted(infos, key=lambda item: item.filename.casefold()):
                if info.flag_bits & 0x1:
                    raise ValueError(f"Encrypted ZIP member is not supported: {info.filename}")
                if info.file_size > MAX_SOURCE_FILE_BYTES:
                    raise ValueError(f"ZIP member exceeds size limit: {info.filename}")
                total_bytes += info.file_size
                if total_bytes > MAX_TOTAL_SOURCE_BYTES:
                    raise ValueError(f"Total screenplay input exceeds {MAX_TOTAL_SOURCE_BYTES} bytes.")
                docs.append(_make_source_document(info.filename, archive.read(info)))

    elif input_path.is_file() and input_path.suffix.lower() == ".txt":
        total_bytes = input_path.stat().st_size
        if total_bytes > MAX_SOURCE_FILE_BYTES:
            raise ValueError(f"Source file exceeds size limit: {input_path}")
        raw = input_path.read_bytes()
        docs.append(_make_source_document(input_path.name, raw))

    elif input_path.is_dir():
        files = sorted(
            (path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() == ".txt"),
            key=lambda path: str(path.relative_to(input_path)).casefold(),
        )
        if len(files) > MAX_SOURCE_FILES:
            raise ValueError(f"Directory contains {len(files)} TXT files; limit is {MAX_SOURCE_FILES}.")
        for file_path in files:
            size = file_path.stat().st_size
            if size > MAX_SOURCE_FILE_BYTES:
                raise ValueError(f"Source file exceeds size limit: {file_path}")
            total_bytes += size
            if total_bytes > MAX_TOTAL_SOURCE_BYTES:
                raise ValueError(f"Total screenplay input exceeds {MAX_TOTAL_SOURCE_BYTES} bytes.")
            docs.append(_make_source_document(str(file_path.relative_to(input_path)), file_path.read_bytes()))
    else:
        raise ValueError("Input must be a .txt file, a directory of .txt files, or a .zip archive.")

    if not docs:
        raise ValueError(f"No .txt screenplay files found in: {input_path}")
    return docs


def _warning(
    code: str,
    message: str,
    source_file: str,
    scene_number: Optional[int] = None,
    line_number: Optional[int] = None,
    raw_text: Optional[str] = None,
) -> ParseWarning:
    return ParseWarning(code, message, source_file, scene_number, line_number, raw_text)


def _event_body(lines: list[str], body_start: int) -> tuple[str, int]:
    """Collect an event body until the next structural header; return content and stop index."""
    body_lines: list[str] = []
    index = body_start
    while index < len(lines):
        candidate = lines[index]
        stripped = candidate.strip()
        if (
            _BOLD_HEADER_RE.match(candidate)
            or _SCENE_RE.match(candidate)
            or stripped == "--- SCRIPT END ---"
        ):
            break
        if stripped not in {"--- SCRIPT START ---", "--- SCRIPT END ---"}:
            body_lines.append(candidate.rstrip())
        index += 1
    return "\n".join(body_lines).strip(), index


def _source_id(source_file: str) -> str:
    stem = Path(source_file).stem
    slug = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()[:28] or "source"
    digest = hashlib.sha1(source_file.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}"


def parse_episode(document: SourceDocument) -> Episode:
    lines = document.text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    metadata: dict[str, str] = {}
    warnings: list[ParseWarning] = []
    scenes: list[Scene] = []
    current_scene: Optional[Scene] = None
    current_scene_fields: set[str] = set()
    scene_occurrences: dict[int, int] = {}

    if document.decode_warning:
        warnings.append(_warning("SOURCE_ENCODING_FALLBACK", document.decode_warning, document.name))

    start_markers = [idx for idx, line in enumerate(lines) if line.strip() == "--- SCRIPT START ---"]
    end_markers = [idx for idx, line in enumerate(lines) if line.strip() == "--- SCRIPT END ---"]
    if not start_markers:
        in_script_initially = True
        warnings.append(_warning(
            "SCRIPT_START_MARKER_MISSING",
            "No SCRIPT START marker found; parsed the entire file as screenplay content.",
            document.name,
        ))
    else:
        in_script_initially = False
        if len(start_markers) > 1:
            warnings.append(_warning("MULTIPLE_SCRIPT_START_MARKERS", "Multiple SCRIPT START markers found.", document.name))
        valid_end_markers = [index for index in end_markers if index > start_markers[0]]
        if not valid_end_markers:
            warnings.append(_warning(
                "SCRIPT_END_MARKER_MISSING",
                "No SCRIPT END marker appears after the first SCRIPT START marker.",
                document.name,
            ))
        if end_markers and end_markers[0] < start_markers[0]:
            warnings.append(_warning(
                "SCRIPT_MARKERS_OUT_OF_ORDER",
                "A SCRIPT END marker appears before SCRIPT START; it was ignored.",
                document.name,
            ))
        if len(valid_end_markers) > 1:
            warnings.append(_warning("MULTIPLE_SCRIPT_END_MARKERS", "Multiple SCRIPT END markers follow SCRIPT START.", document.name))

    for line_number, line in enumerate(lines, start=1):
        match = _METADATA_RE.match(line)
        if not match:
            continue
        key = match.group("key").upper()
        value = match.group("value").strip()
        if key in metadata:
            warnings.append(_warning(
                "DUPLICATE_METADATA_FIELD",
                f"Metadata field {key!r} appears more than once; first value retained.",
                document.name,
                line_number=line_number,
                raw_text=line,
            ))
            continue
        metadata[key] = value
        if not value:
            warnings.append(_warning(
                "EMPTY_METADATA_FIELD", f"Metadata field {key!r} is empty.",
                document.name, line_number=line_number, raw_text=line,
            ))

    episode_label = metadata.get("EPISODE NUMBER") or None
    episode_number: Optional[int] = None
    if episode_label:
        number_match = re.search(r"\d+", episode_label)
        if number_match:
            episode_number = int(number_match.group(0))
        else:
            warnings.append(_warning(
                "INVALID_EPISODE_NUMBER", f"Could not extract a number from {episode_label!r}.", document.name,
            ))
    else:
        warnings.append(_warning("EPISODE_NUMBER_MISSING", "Episode number metadata was not found.", document.name))

    in_script = in_script_initially
    source_order_index = 0
    source_key = _source_id(document.name)
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "--- SCRIPT START ---":
            in_script = True
            i += 1
            continue
        if stripped == "--- SCRIPT END ---":
            if in_script:
                break
            i += 1
            continue
        if not in_script:
            i += 1
            continue

        scene_match = _SCENE_RE.match(line)
        if scene_match:
            scene_number = int(scene_match.group("number"))
            scene_occurrences[scene_number] = scene_occurrences.get(scene_number, 0) + 1
            occurrence = scene_occurrences[scene_number]
            if occurrence > 1:
                warnings.append(_warning(
                    "DUPLICATE_SCENE_NUMBER",
                    f"Scene number {scene_number} appears more than once; a distinct occurrence ID was assigned.",
                    document.name, scene_number=scene_number, line_number=i + 1, raw_text=line,
                ))
            current_scene_fields = set()
            current_scene = Scene(
                scene_number=scene_number,
                heading=None,
                start_seconds=None,
                end_seconds=None,
                start_timestamp=None,
                end_timestamp=None,
                events=[],
                scene_occurrence_index=occurrence,
            )
            scenes.append(current_scene)
            i += 1
            continue

        field_match = _SCENE_FIELD_RE.match(line)
        if field_match and current_scene is not None:
            key = field_match.group("key").lower()
            value = field_match.group("value").strip()
            if key in current_scene_fields:
                warnings.append(_warning(
                    "DUPLICATE_SCENE_FIELD",
                    f"{field_match.group('key')} appears more than once; first value retained.",
                    document.name, current_scene.scene_number, i + 1, line,
                ))
                i += 1
                continue
            current_scene_fields.add(key)
            if key == "scene heading":
                current_scene.heading = value or None
                if not value:
                    warnings.append(_warning("EMPTY_SCENE_HEADING", "Scene heading is empty.", document.name,
                                             current_scene.scene_number, i + 1, line))
            elif key in {"scene start", "scene end"}:
                if key == "scene start":
                    current_scene.start_timestamp = value or None
                else:
                    current_scene.end_timestamp = value or None
                if not value:
                    warnings.append(_warning("EMPTY_SCENE_TIMESTAMP", f"{key} is empty.", document.name,
                                             current_scene.scene_number, i + 1, line))
                else:
                    try:
                        seconds = timestamp_to_seconds(value)
                        if key == "scene start":
                            current_scene.start_seconds = seconds
                        else:
                            current_scene.end_seconds = seconds
                    except ValueError as exc:
                        warnings.append(_warning(
                            "INVALID_SCENE_TIMESTAMP", str(exc), document.name,
                            current_scene.scene_number, i + 1, line,
                        ))
            i += 1
            continue

        header_match = _BOLD_HEADER_RE.match(line)
        if header_match and current_scene is not None:
            raw_header = header_match.group("header").strip()
            normalized_header = _normalize_header(raw_header)

            time_line_index = i + 1
            while time_line_index < len(lines) and not lines[time_line_index].strip():
                time_line_index += 1
            time_line = lines[time_line_index] if time_line_index < len(lines) else ""
            time_match = _EVENT_TIME_RE.match(time_line)

            start_raw: Optional[str] = None
            end_raw: Optional[str] = None
            start_seconds: Optional[float] = None
            end_seconds: Optional[float] = None
            if time_match:
                start_raw, end_raw = time_match.group(1).strip(), time_match.group(2).strip()
                try:
                    start_seconds = timestamp_to_seconds(start_raw)
                    end_seconds = timestamp_to_seconds(end_raw)
                except ValueError as exc:
                    warnings.append(_warning(
                        "INVALID_EVENT_TIMESTAMP", str(exc), document.name,
                        current_scene.scene_number, time_line_index + 1, time_line,
                    ))
                body_start = time_line_index + 1
                source_block_start = i
            else:
                warnings.append(_warning(
                    "EVENT_MISSING_TIME_RANGE",
                    "Bold event header was found, but no valid 'Start | End' range followed it; content retained without timestamps.",
                    document.name, current_scene.scene_number, i + 1, line,
                ))
                if _START_PREFIX_RE.match(time_line):
                    warnings.append(_warning(
                        "MALFORMED_EVENT_TIME_LINE",
                        "A Start line was present but did not match the expected Start | End format.",
                        document.name, current_scene.scene_number, time_line_index + 1, time_line,
                    ))
                    body_start = time_line_index + 1
                else:
                    body_start = i + 1
                source_block_start = i

            content, body_end = _event_body(lines, body_start)
            # If the timestamp was validly located, include the timestamp line in raw_block.
            raw_end = body_end
            raw_block = "\n".join(lines[source_block_start:raw_end]).strip()

            source_order_index += 1
            scene_event_index = len(current_scene.events) + 1
            base_header = normalized_header.rstrip(":").strip()
            event_type = KNOWN_EVENT_HEADERS.get(base_header)
            speaker = None
            delivery = None
            voice_direction: list[str] = []
            if event_type is None:
                event_type = "dialogue"
                speaker, delivery, voice_direction = _split_speaker_header(raw_header)

            event_id = (
                f"ep{episode_number if episode_number is not None else 'unknown'}_{source_key}"
                f"_sc{current_scene.scene_number:02d}_occ{current_scene.scene_occurrence_index:02d}"
                f"_ev{scene_event_index:03d}"
            )
            current_scene.events.append(Event(
                event_id=event_id,
                episode_number=episode_number,
                scene_number=current_scene.scene_number,
                event_index_in_scene=scene_event_index,
                event_type=event_type,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                start_timestamp=start_raw,
                end_timestamp=end_raw,
                content=content,
                speaker=speaker,
                delivery=delivery,
                voice_direction=voice_direction,
                raw_header=raw_header,
                raw_block=raw_block,
                source_file=document.name,
                source_line_start=i + 1,
                source_order_index=source_order_index,
                scene_occurrence_index=current_scene.scene_occurrence_index,
            ))

            if start_seconds is not None and end_seconds is not None and end_seconds < start_seconds:
                warnings.append(_warning(
                    "EVENT_END_BEFORE_START",
                    f"Event end ({end_raw}) is earlier than start ({start_raw}).",
                    document.name, current_scene.scene_number, i + 1, raw_header,
                ))
            if not content:
                warnings.append(_warning(
                    "EMPTY_EVENT_CONTENT", "Event has a header/time range but no body content.",
                    document.name, current_scene.scene_number, i + 1, raw_header,
                ))

            i = body_end
            continue

        i += 1

    # Scene and event integrity checks. Overlapping events are expected and are not errors.
    for scene in scenes:
        if scene.heading is None:
            warnings.append(_warning("SCENE_MISSING_HEADING", "Scene heading was not found.",
                                     document.name, scene.scene_number))
        if scene.start_timestamp is None:
            warnings.append(_warning("SCENE_START_MISSING", "Scene start timestamp was not found.",
                                     document.name, scene.scene_number))
        if scene.end_timestamp is None:
            warnings.append(_warning("SCENE_END_MISSING", "Scene end timestamp was not found.",
                                     document.name, scene.scene_number))
        if not scene.events:
            warnings.append(_warning("SCENE_NO_EVENTS", "Scene contains no parsed events.",
                                     document.name, scene.scene_number))
        valid_scene_bounds = (
            scene.start_seconds is not None
            and scene.end_seconds is not None
            and scene.end_seconds >= scene.start_seconds
        )
        if (scene.start_seconds is not None and scene.end_seconds is not None
                and scene.end_seconds < scene.start_seconds):
            warnings.append(_warning("SCENE_END_BEFORE_START", "Scene end is earlier than scene start.",
                                     document.name, scene.scene_number))
        if valid_scene_bounds:
            for event in scene.events:
                if event.start_seconds is None or event.end_seconds is None:
                    continue
                if (event.end_seconds > scene.end_seconds + SCENE_BOUNDS_TOLERANCE_SECONDS
                        or event.start_seconds < scene.start_seconds - SCENE_BOUNDS_TOLERANCE_SECONDS):
                    warnings.append(_warning(
                        "EVENT_OUTSIDE_SCENE",
                        (f"Event {event.start_timestamp}-{event.end_timestamp} falls outside "
                         f"scene {scene.start_timestamp}-{scene.end_timestamp}."),
                        document.name, scene.scene_number, event.source_line_start, event.raw_header,
                    ))

    if not scenes:
        warnings.append(_warning("NO_SCENES_FOUND", "No SCENE headings were detected in this file.", document.name))

    return Episode(
        episode_number=episode_number,
        episode_label=episode_label,
        logline=metadata.get("EPISODE LOGLINE"),
        continuity_notes=metadata.get("CONTINUITY NOTES"),
        source_file=document.name,
        scenes=scenes,
        warnings=warnings,
    )


def _episode_to_dict(episode: Episode) -> dict:
    return {
        "episode_number": episode.episode_number,
        "episode_label": episode.episode_label,
        "logline": episode.logline,
        "continuity_notes": episode.continuity_notes,
        "source_file": episode.source_file,
        "event_ordering": {
            "scenes_and_events": "source_order",
            "timeline_event_ids": "chronological_by_start_then_end_then_source_order",
        },
        "timeline_event_ids": [event.event_id for event in episode.chronological_events],
        "scenes": [
            {
                "scene_number": scene.scene_number,
                "scene_occurrence_index": scene.scene_occurrence_index,
                "heading": scene.heading,
                "start_seconds": scene.start_seconds,
                "end_seconds": scene.end_seconds,
                "start_timestamp": scene.start_timestamp,
                "end_timestamp": scene.end_timestamp,
                "events": [asdict(event) for event in scene.events],
            }
            for scene in episode.scenes
        ],
        "warnings": [asdict(warning) for warning in episode.warnings],
    }


def parse_input(
    input_path: Path, manual_aliases: Optional[dict[str, str]] = None
) -> tuple[list[Episode], dict]:
    # Validate aliases before spending time parsing files.
    manual_aliases = _validate_manual_aliases(manual_aliases)
    documents = _read_sources(input_path)
    episodes = [parse_episode(document) for document in documents]
    episodes.sort(key=lambda item: (
        item.episode_number is None,
        item.episode_number if item.episode_number is not None else 0,
        item.source_file.casefold(),
    ))

    number_to_episodes: dict[int, list[Episode]] = {}
    for episode in episodes:
        if episode.episode_number is not None:
            number_to_episodes.setdefault(episode.episode_number, []).append(episode)
    for number, matches in number_to_episodes.items():
        if len(matches) > 1:
            for episode in matches:
                episode.warnings.append(_warning(
                    "DUPLICATE_EPISODE_NUMBER",
                    f"Episode number {number} is used by {len(matches)} source files.",
                    episode.source_file,
                ))

    speaker_merges, alias_provenance = _apply_speaker_aliases(episodes, manual_aliases)
    total_scenes = sum(len(episode.scenes) for episode in episodes)
    total_events = sum(len(episode.events) for episode in episodes)
    event_type_counts: dict[str, int] = {}
    for episode in episodes:
        for event in episode.events:
            event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1

    warnings = [warning for episode in episodes for warning in episode.warnings]
    warning_code_counts = {
        code: sum(1 for warning in warnings if warning.code == code)
        for code in sorted({warning.code for warning in warnings})
    }
    report = {
        "input": str(input_path),
        "files_discovered": len(documents),
        "episodes_parsed": len(episodes),
        "episodes_with_number": sum(episode.episode_number is not None for episode in episodes),
        "scenes_parsed": total_scenes,
        "events_parsed": total_events,
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "source_order_preserved": True,
        "timeline_event_ids_emitted": True,
        "speaker_aliases_applied": speaker_merges,
        "speaker_alias_provenance": alias_provenance,
        "warnings_count": len(warnings),
        "warning_code_counts": warning_code_counts,
        "warnings": [asdict(warning) for warning in warnings],
    }
    return episodes, report


def load_manual_aliases(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in aliases file {path}: {exc.msg} (line {exc.lineno}, column {exc.colno}).") from exc
    except OSError as exc:
        raise ValueError(f"Could not read aliases file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Aliases file must contain a JSON object mapping speaker names to canonical names.")
    return _validate_manual_aliases(payload)


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Parse Kuku timestamped screenplay TXT files.")
    parser.add_argument("--input", required=True, help="Path to .zip archive, .txt file, or directory of .txt files")
    parser.add_argument("--output", default="parsed_output", help="Output directory (default: parsed_output)")
    parser.add_argument("--aliases", help='Optional JSON alias map, e.g. {"THE FORMER GROOM": "GERALD"}')
    args = parser.parse_args(argv)

    try:
        manual_aliases = load_manual_aliases(Path(args.aliases).expanduser()) if args.aliases else None
        input_path = Path(args.input).expanduser().resolve()
        output_dir = Path(args.output).expanduser().resolve()
        episodes, report = parse_input(input_path, manual_aliases)
        output_dir.mkdir(parents=True, exist_ok=True)
        episodes_path = output_dir / "parsed_episodes.json"
        report_path = output_dir / "parse_report.json"
        _write_json_atomic(episodes_path, [_episode_to_dict(episode) for episode in episodes])
        _write_json_atomic(report_path, report)
    except (OSError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Parsed {report['episodes_parsed']} screenplay files")
    print(f"Scenes: {report['scenes_parsed']} | Events: {report['events_parsed']}")
    print(f"Event types: {json.dumps(report['event_type_counts'], sort_keys=True)}")
    print(f"Speaker aliases merged: {json.dumps(report['speaker_aliases_applied'], sort_keys=True)}")
    print(f"Warnings: {report['warnings_count']} {json.dumps(report['warning_code_counts'], sort_keys=True)}")
    print(f"Episodes JSON: {episodes_path}")
    print(f"Parse report:  {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
