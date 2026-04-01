#!/usr/bin/env python3
"""
MIDI Analyzer MCP Server
========================
Exposes three tools to the agent pipeline via the MCP stdio transport:

  • analyze_midi(file_path)
        Returns a JSON object with melody structure metadata:
        bpm, key_signature, time_signature, note_count,
        syllable_count, and strong_beat_positions.

  • get_syllable_durations(file_path)
        Returns a JSON array of per-note durations in seconds.

  • suggest_rhyme_positions(file_path)
        Returns a JSON array of 0-based syllable indices that fall on
        phrase endings / strong beats – good candidates for rhymes.

Dependencies (install via uv or pip):
    mcp>=1.0.0
    mido>=1.3.0

Run for manual testing (MCP Inspector):
    python server.py
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import mido
from mcp.server.fastmcp import FastMCP
from xfmido import XFMidiFile, extract_xf_karaoke_info

# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="midi-analyzer",
    instructions=(
        "Analyses MIDI files and extracts melody structure for "
        "Cantonese lyrics generation: syllable count, tempo, key, "
        "note durations, and rhyme-position suggestions."
    ),
)

# ---------------------------------------------------------------------------
# Internal MIDI utilities
# ---------------------------------------------------------------------------

# MIDI note number → pitch class name (sharps notation)
_PITCH_CLASS: dict[int, str] = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}

# Simple key-detection weight table (Krumhansl-Schmuckler tonal profiles)
_MAJOR_PROFILE = [
    6.35,
    2.23,
    3.48,
    2.33,
    4.38,
    4.09,
    2.52,
    5.19,
    2.39,
    3.66,
    2.29,
    2.88,
]
_MINOR_PROFILE = [
    6.33,
    2.68,
    3.52,
    5.38,
    2.60,
    3.53,
    2.54,
    4.75,
    3.98,
    2.69,
    3.34,
    3.17,
]


def _load_midi(file_path: str) -> mido.MidiFile:
    """Load and return a MidiFile, raising ValueError for bad paths."""
    p = Path(file_path)
    if not p.exists():
        raise ValueError(f"MIDI file not found: {p}")
    if p.suffix.lower() not in {".mid", ".midi"}:
        raise ValueError(f"Not a MIDI file: {p}")
    return mido.MidiFile(str(p))


def _ticks_to_seconds(ticks: int, tempo_us: int, ticks_per_beat: int) -> float:
    """Convert MIDI ticks to wall-clock seconds."""
    if ticks_per_beat <= 0 or tempo_us <= 0:
        return 0.0
    beats = ticks / ticks_per_beat
    return beats * (tempo_us / 1_000_000)


def _extract_note_events(mid: mido.MidiFile) -> list[dict[str, Any]]:
    """
    Merge all tracks into a flat, time-sorted list of note events.

    Each event dict:
        {
            "note":        int,       # MIDI note number 0-127
            "velocity":    int,       # 0-127
            "start_tick":  int,
            "end_tick":    int,
            "start_sec":   float,
            "duration_sec":float,
            "channel":     int,
        }
    """
    ticks_per_beat: int = mid.ticks_per_beat or 480
    tempo_us: int = 500_000  # default 120 BPM

    # Collect all messages with absolute tick times across all tracks
    raw_events: list[tuple[int, mido.Message]] = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            raw_events.append((abs_tick, msg))

    raw_events.sort(key=lambda x: x[0])

    # Build absolute-time-in-seconds mapping + extract note on/off pairs
    pending: dict[
        tuple[int, int], tuple[int, float]
    ] = {}  # (channel, note) → (tick, sec)
    completed: list[dict[str, Any]] = []
    current_sec = 0.0
    last_tick = 0

    for abs_tick, msg in raw_events:
        # Advance time
        delta_ticks = abs_tick - last_tick
        current_sec += _ticks_to_seconds(delta_ticks, tempo_us, ticks_per_beat)
        last_tick = abs_tick

        # Access mido.Message attributes (type stubs incomplete)
        if msg.type == "set_tempo":  # type: ignore[attr-defined]
            tempo_us = msg.tempo  # type: ignore[attr-defined]

        elif msg.type == "note_on" and msg.velocity > 0:  # type: ignore[attr-defined]
            key = (msg.channel, msg.note)  # type: ignore[attr-defined]
            pending[key] = (abs_tick, current_sec)

        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):  # type: ignore[attr-defined]
            key = (msg.channel, msg.note)  # type: ignore[attr-defined]
            if key in pending:
                start_tick, start_sec = pending.pop(key)
                duration_sec = current_sec - start_sec
                if duration_sec > 0:
                    completed.append(
                        {
                            "note": msg.note,  # type: ignore[attr-defined]
                            "velocity": msg.velocity,  # type: ignore[attr-defined]
                            "start_tick": start_tick,
                            "end_tick": abs_tick,
                            "start_sec": round(start_sec, 6),
                            "duration_sec": round(duration_sec, 6),
                            "channel": msg.channel,  # type: ignore[attr-defined]
                        }
                    )

    # Close any still-open notes (file ended without note_off)
    for (channel, note), (start_tick, start_sec) in pending.items():
        duration_sec = current_sec - start_sec
        if duration_sec > 0:
            completed.append(
                {
                    "note": note,
                    "velocity": 64,
                    "start_tick": start_tick,
                    "end_tick": last_tick,
                    "start_sec": round(start_sec, 6),
                    "duration_sec": round(duration_sec, 6),
                    "channel": channel,
                }
            )

    completed.sort(key=lambda e: e["start_sec"])
    return completed


def _group_events_by_channel(
    note_events: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ev in note_events:
        grouped[int(ev["channel"])].append(ev)
    for events in grouped.values():
        events.sort(key=lambda e: (e["start_sec"], -e["note"], -e["duration_sec"]))
    return dict(grouped)


def _max_polyphony(events: list[dict[str, Any]]) -> int:
    points: list[tuple[float, int]] = []
    for ev in events:
        start = float(ev["start_sec"])
        end = start + float(ev["duration_sec"])
        points.append((start, 1))
        points.append((end, -1))
    points.sort(key=lambda item: (item[0], item[1]))

    active = 0
    max_active = 0
    for _, delta in points:
        active += delta
        max_active = max(max_active, active)
    return max_active


def _extract_top_note_line(
    events: list[dict[str, Any]],
    onset_tolerance: float = 0.05,
) -> list[dict[str, Any]]:
    if not events:
        return []

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = [events[0]]
    current_start = float(events[0]["start_sec"])

    for ev in events[1:]:
        start = float(ev["start_sec"])
        if start - current_start <= onset_tolerance:
            current.append(ev)
        else:
            groups.append(current)
            current = [ev]
            current_start = start
    groups.append(current)

    topline: list[dict[str, Any]] = []
    for group in groups:
        chosen = max(
            group,
            key=lambda ev: (
                int(ev["note"]),
                float(ev["duration_sec"]),
                int(ev["velocity"]),
            ),
        )
        topline.append(dict(chosen))
    topline.sort(key=lambda e: e["start_sec"])
    return topline


def _merge_melody_ornaments(
    melody_notes: list[dict[str, Any]],
    gap_threshold: float = 0.03,
) -> list[dict[str, Any]]:
    if not melody_notes:
        return []

    merged: list[dict[str, Any]] = []
    prev = dict(melody_notes[0])

    for curr in melody_notes[1:]:
        onset_gap = float(curr["start_sec"]) - float(prev["start_sec"])
        if onset_gap <= gap_threshold:
            curr_end = float(curr["start_sec"]) + float(curr["duration_sec"])
            prev_end = float(prev["start_sec"]) + float(prev["duration_sec"])
            prev["end_tick"] = curr["end_tick"]
            prev["duration_sec"] = round(
                max(prev_end, curr_end) - float(prev["start_sec"]), 6
            )
            if int(curr["note"]) > int(prev["note"]):
                prev["note"] = curr["note"]
                prev["velocity"] = curr["velocity"]
        else:
            merged.append(prev)
            prev = dict(curr)

    merged.append(prev)
    return merged


def _select_melody_channel(
    note_events: list[dict[str, Any]],
    *,
    is_xf: bool = False,
    xf_melody_channel: int | None = None,
) -> tuple[int, list[dict[str, Any]], str]:
    grouped = _group_events_by_channel(note_events)
    if not grouped:
        return 0, [], "no_notes"

    if is_xf and xf_melody_channel is not None and xf_melody_channel in grouped:
        topline = _extract_top_note_line(grouped[xf_melody_channel])
        return xf_melody_channel, topline, "yamaha_xf_karaoke_info"
    if is_xf and 0 in grouped:
        topline = _extract_top_note_line(grouped[0])
        return 0, topline, "yamaha_xf_default_channel_0"

    best_channel = 0
    best_score = float("-inf")
    best_topline: list[dict[str, Any]] = []

    for channel, events in grouped.items():
        topline = _extract_top_note_line(events)
        if not topline:
            continue

        avg_pitch = statistics.mean(float(ev["note"]) for ev in topline)
        polyphony = _max_polyphony(events)
        monophony_ratio = len(topline) / max(len(events), 1)
        score = (
            avg_pitch * 0.7
            + monophony_ratio * 45.0
            - max(polyphony - 1, 0) * 16.0
            + min(len(topline), 1200) * 0.01
        )

        if score > best_score:
            best_score = score
            best_channel = channel
            best_topline = topline

    return best_channel, best_topline, "heuristic_channel_score"


def _extract_melody_notes(
    note_events: list[dict[str, Any]],
    *,
    is_xf: bool = False,
    xf_melody_channel: int | None = None,
) -> tuple[list[dict[str, Any]], int, str]:
    channel, topline, reason = _select_melody_channel(
        note_events,
        is_xf=is_xf,
        xf_melody_channel=xf_melody_channel,
    )
    return _merge_melody_ornaments(topline), channel, reason


def _extract_tempo(mid: mido.MidiFile) -> int:
    """Return the first set_tempo value found, or the default 120 BPM."""
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                return msg.tempo
    return 500_000  # 120 BPM


def _is_yamaha_xf(mid: mido.MidiFile) -> bool:
    for track in mid.tracks:
        for msg in track:
            if msg.type == "sequencer_specific" and hasattr(msg, "data"):
                data = bytes(msg.data)
                if b"XF" in data or b"XF02" in data:
                    return True
    return False


def _get_xf_melody_channel(file_path: str) -> int | None:
    try:
        info = extract_xf_karaoke_info(filename=file_path)
    except Exception:
        return None
    melody_channel = int(info.get("melody_channel", 0))
    return melody_channel - 1 if melody_channel > 0 else None


def _decode_lyric_bytes(data: bytes, *, prefer_xf: bool = False) -> str:
    encodings = (
        ["cp932", "shift_jis", "euc_jp", "gbk", "utf-8"]
        if prefer_xf
        else ["euc_jp", "cp932", "shift_jis", "gbk", "utf-8"]
    )
    for enc in encodings:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin1", errors="replace")


def _normalize_embedded_lyric_unit(text: str) -> str | None:
    unit = text.strip()
    if not unit:
        return None

    # Remove only control markers that are not part of the visible lyric stream.
    unit = unit.lstrip("<")
    unit = unit.rstrip("/^")
    unit = unit.strip()
    unit = unit.rstrip("・･")
    unit = unit.strip()

    if not unit:
        return None

    return unit


def _extract_xf_lyrics(file_path: str) -> list[str]:
    try:
        xf = XFMidiFile(filename=file_path)
    except Exception:
        return []

    if not getattr(xf, "xfkm", None):
        return []

    units: list[str] = []
    for msg in xf.xfkm:
        if getattr(msg, "type", None) != "lyrics":
            continue
        raw = getattr(msg, "text", "")
        data = str(raw).encode("latin1", errors="ignore")
        text = _decode_lyric_bytes(data, prefer_xf=True)
        unit = _normalize_embedded_lyric_unit(text)
        if unit is None:
            continue
        units.append(unit)
    return units


def _extract_standard_lyrics(mid: mido.MidiFile) -> list[str]:
    units: list[str] = []
    for track in mid.tracks:
        for msg in track:
            if getattr(msg, "type", None) != "lyrics":
                continue
            raw = getattr(msg, "text", "")
            data = str(raw).encode("latin1", errors="ignore")
            text = _decode_lyric_bytes(data)
            unit = _normalize_embedded_lyric_unit(text)
            if unit is None:
                continue
            units.append(unit)
    return units


def _extract_embedded_lyrics(
    mid: mido.MidiFile, file_path: str
) -> tuple[list[str], str | None]:
    if _is_yamaha_xf(mid):
        xf_units = _extract_xf_lyrics(file_path)
        if xf_units:
            return xf_units, "xfkm"

    std_units = _extract_standard_lyrics(mid)
    if std_units:
        return std_units, "lyrics_meta"

    return [], None


def _extract_time_signature(mid: mido.MidiFile) -> str:
    """Return the first time_signature as 'N/D', defaulting to '4/4'."""
    for track in mid.tracks:
        for msg in track:
            if msg.type == "time_signature":
                return f"{msg.numerator}/{msg.denominator}"
    return "4/4"


def _detect_key(mid: mido.MidiFile) -> str:
    """
    Detect the key signature using a simplified Krumhansl-Schmuckler algorithm.

    Returns a string like 'C major' or 'A minor'.
    If a key_signature meta message exists, use it directly.
    """
    # Priority 1: key_signature meta message
    for track in mid.tracks:
        for msg in track:
            if msg.type == "key_signature":
                return msg.key  # e.g. "C" or "Am"

    # Priority 2: pitch-class histogram correlation
    pitch_class_counts = [0] * 12
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                pitch_class_counts[msg.note % 12] += 1

    total = sum(pitch_class_counts)
    if total == 0:
        return "C major"

    profile = [c / total for c in pitch_class_counts]

    best_key = "C major"
    best_score = -1.0

    for root in range(12):
        # Major correlation
        major_corr = sum(
            profile[(root + i) % 12] * _MAJOR_PROFILE[i] for i in range(12)
        )
        # Minor correlation
        minor_corr = sum(
            profile[(root + i) % 12] * _MINOR_PROFILE[i] for i in range(12)
        )

        if major_corr > best_score:
            best_score = major_corr
            best_key = f"{_PITCH_CLASS[root]} major"

        if minor_corr > best_score:
            best_score = minor_corr
            best_key = f"{_PITCH_CLASS[root]} minor"

    return best_key


def _estimate_syllable_count(note_events: list[dict[str, Any]]) -> int:
    """
    Estimate the number of lyric syllables from note events.

    Strategy:
    - Each distinct note-on event on the melody track corresponds to one syllable.
    - Filter to the channel with the most notes (assumed to be the melody).
    - Merge notes that are very close together (< 30 ms gap) as ornaments/trills.
    """
    if not note_events:
        return 0

    melody_notes, _, _ = _extract_melody_notes(note_events)
    return len(melody_notes) if melody_notes else len(note_events)


def _find_strong_beat_positions(
    note_events: list[dict[str, Any]],
    ticks_per_beat: int,
    numerator: int,
) -> list[int]:
    """
    Return 0-based syllable indices that fall on strong beats.

    A "strong beat" is any note whose start_tick is aligned (within 10%)
    to a multiple of ticks_per_beat (beat 1 of each measure) or a half-bar.
    """
    if not note_events or ticks_per_beat <= 0:
        return []

    tolerance = ticks_per_beat * 0.10

    strong: list[int] = []
    for idx, ev in enumerate(note_events):
        tick = ev["start_tick"]
        # Is it on the downbeat (beat 1 of measure)?
        beats_per_bar = numerator
        ticks_per_bar = ticks_per_beat * beats_per_bar
        pos_in_bar = tick % ticks_per_bar

        # Strong positions: beat 1 and beat 3 (in 4/4)
        beat1 = 0
        beat3 = ticks_per_beat * 2 if beats_per_bar >= 3 else ticks_per_beat

        on_strong = (
            abs(pos_in_bar - beat1) <= tolerance
            or abs(pos_in_bar - beat3) <= tolerance
            or abs(pos_in_bar - ticks_per_bar) <= tolerance  # wrap-around
        )

        if on_strong:
            strong.append(idx)

    return strong


def _find_phrase_endings(
    note_events: list[dict[str, Any]],
    ticks_per_beat: int,
) -> list[int]:
    """
    Identify phrase-ending syllable indices by finding large gaps between notes.

    Uses a higher threshold (2.0× median IOI) and requires a minimum absolute
    gap (>= 0.5 seconds) to avoid false positives in fast-paced melodies.
    """
    if len(note_events) < 2:
        return []

    ioi_list = [
        note_events[i + 1]["start_sec"] - note_events[i]["start_sec"]
        for i in range(len(note_events) - 1)
    ]

    if not ioi_list:
        return []

    median_ioi = statistics.median(ioi_list)
    # Use the larger of 2.0× median or 0.5s absolute minimum
    threshold = max(median_ioi * 2.0, 0.5)

    endings: list[int] = []
    for i, ioi in enumerate(ioi_list):
        if ioi >= threshold:
            endings.append(i)  # note i is the end of a phrase

    # Always include the final note as a phrase ending
    if note_events and len(note_events) - 1 not in endings:
        endings.append(len(note_events) - 1)

    return sorted(set(endings))


def _build_phrase_segments(
    melody_notes: list[dict[str, Any]],
    phrase_end_indices: list[int],
) -> list[dict[str, Any]]:
    """
    Build phrase segments from melody notes and phrase end indices.

    Each segment represents one lyric line, with:
    - start_idx, end_idx: syllable range (inclusive)
    - note_count: number of syllables in this phrase
    - duration_sec: total duration of the phrase
    - gap_before_sec: gap before this phrase (0 for first phrase)

    Returns a list of segment dicts.
    """
    if not melody_notes:
        return []

    segments: list[dict[str, Any]] = []
    prev_end = -1

    for phrase_end_idx in phrase_end_indices:
        start_idx = prev_end + 1
        end_idx = phrase_end_idx

        if start_idx > end_idx or end_idx >= len(melody_notes):
            continue

        # Calculate phrase duration
        start_sec = float(melody_notes[start_idx]["start_sec"])
        end_sec = float(melody_notes[end_idx]["start_sec"]) + float(
            melody_notes[end_idx]["duration_sec"]
        )
        duration = end_sec - start_sec

        # Calculate gap before this phrase
        gap_before = 0.0
        if prev_end >= 0 and prev_end + 1 < len(melody_notes):
            gap_before = float(melody_notes[start_idx]["start_sec"]) - (
                float(melody_notes[prev_end]["start_sec"])
                + float(melody_notes[prev_end]["duration_sec"])
            )
            gap_before = max(0.0, gap_before)

        segments.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "note_count": end_idx - start_idx + 1,
                "duration_sec": round(duration, 3),
                "gap_before_sec": round(gap_before, 3),
            }
        )

        prev_end = end_idx

    return segments


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def analyze_midi(file_path: str) -> str:
    """
    Analyse a MIDI file and extract melody structure metadata.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to a .mid / .midi file.

    Returns
    -------
    JSON string with:
        {
            "syllable_count":        int,
            "bpm":                   float,
            "key":                   str,       # e.g. "C major"
            "time_signature":        str,       # e.g. "4/4"
            "note_count":            int,
            "strong_beat_positions": list[int], # 0-based syllable indices
            "duration_seconds":      float,     # total playback duration
            "ticks_per_beat":        int,
            "track_count":           int,
        }
    """
    try:
        mid = _load_midi(file_path)

        tempo_us = _extract_tempo(mid)
        bpm = round(60_000_000 / tempo_us, 2)

        time_sig_str = _extract_time_signature(mid)
        numerator = int(time_sig_str.split("/")[0])

        key = _detect_key(mid)
        is_xf = _is_yamaha_xf(mid)
        xf_melody_channel = _get_xf_melody_channel(file_path) if is_xf else None
        embedded_lyrics, embedded_source = _extract_embedded_lyrics(mid, file_path)

        note_events = _extract_note_events(mid)

        melody_notes, melody_channel, selection_reason = _extract_melody_notes(
            note_events,
            is_xf=is_xf,
            xf_melody_channel=xf_melody_channel,
        )
        note_syllable_count = len(melody_notes)
        effective_syllable_count = (
            len(embedded_lyrics) if embedded_lyrics else note_syllable_count
        )
        effective_syllable_count_source = (
            "embedded_lyrics" if embedded_lyrics else "melody_notes"
        )

        ticks_per_beat: int = mid.ticks_per_beat or 480
        strong_beats = _find_strong_beat_positions(
            melody_notes, ticks_per_beat, numerator
        )

        # Phrase boundaries for lyric line breaks
        phrase_ends = _find_phrase_endings(melody_notes, ticks_per_beat)
        phrase_boundaries = _build_phrase_segments(melody_notes, phrase_ends)

        # Total duration
        duration_sec = (
            melody_notes[-1]["start_sec"] + melody_notes[-1]["duration_sec"]
            if melody_notes
            else 0.0
        )

        result = {
            "syllable_count": effective_syllable_count,
            "note_syllable_count": note_syllable_count,
            "effective_syllable_count": effective_syllable_count,
            "effective_syllable_count_source": effective_syllable_count_source,
            "bpm": bpm,
            "key": key,
            "time_signature": time_sig_str,
            "note_count": len(note_events),
            "strong_beat_positions": strong_beats,
            "phrase_boundaries": phrase_boundaries,
            "phrase_end_indices": phrase_ends,
            "duration_seconds": round(duration_sec, 3),
            "ticks_per_beat": ticks_per_beat,
            "track_count": len(mid.tracks),
            "is_xf": is_xf,
            "xf_melody_channel": xf_melody_channel,
            "melody_channel": melody_channel,
            "melody_selection_reason": selection_reason,
            "embedded_lyrics_source": embedded_source,
            "embedded_lyric_unit_count": len(embedded_lyrics),
            "embedded_lyrics_preview": embedded_lyrics[:16],
        }
        return json.dumps(result, ensure_ascii=False)

    except Exception as exc:
        error = {"error": str(exc), "file_path": file_path}
        return json.dumps(error)


@mcp.tool()
def get_syllable_durations(file_path: str) -> str:
    """
    Return the duration (in seconds) of every note in the melody track.

    Each duration corresponds to one lyric syllable that must be sung.
    Short ornamental notes (< 30 ms) are merged into the preceding note.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to a .mid / .midi file.

    Returns
    -------
    JSON array of floats, e.g. [0.5, 0.25, 0.75, ...]
    One float per syllable in melody order.
    """
    try:
        mid = _load_midi(file_path)
        is_xf = _is_yamaha_xf(mid)
        xf_melody_channel = _get_xf_melody_channel(file_path) if is_xf else None
        note_events = _extract_note_events(mid)

        if not note_events:
            return json.dumps([])

        melody_notes, _, _ = _extract_melody_notes(
            note_events,
            is_xf=is_xf,
            xf_melody_channel=xf_melody_channel,
        )

        # Merge ornamental notes
        merged_durations: list[float] = []
        prev: dict[str, Any] | None = melody_notes[0] if melody_notes else None
        accumulated_dur = prev["duration_sec"] if prev else 0.0

        for curr in melody_notes[1:]:
            gap = curr["start_sec"] - (prev["start_sec"] + prev["duration_sec"])  # type: ignore[union-attr]
            if gap < 0.030:
                # Same syllable – accumulate duration
                accumulated_dur += curr["duration_sec"] + max(gap, 0.0)
                prev = curr
            else:
                merged_durations.append(round(accumulated_dur, 4))
                prev = curr
                accumulated_dur = curr["duration_sec"]

        if prev is not None:
            merged_durations.append(round(accumulated_dur, 4))

        return json.dumps(merged_durations)

    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def suggest_rhyme_positions(file_path: str) -> str:
    """
    Suggest which syllable indices are good candidates for rhyme placement.

    Uses two complementary heuristics:
      1. Phrase endings (large inter-onset-interval gaps → natural breath marks).
      2. Strong beats that occur near the median phrase length.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to a .mid / .midi file.

    Returns
    -------
    JSON array of 0-based integers – syllable indices where the lyricist
    should place rhyming syllables, e.g. [7, 15, 23, 31].
    """
    try:
        mid = _load_midi(file_path)
        is_xf = _is_yamaha_xf(mid)
        xf_melody_channel = _get_xf_melody_channel(file_path) if is_xf else None
        note_events = _extract_note_events(mid)

        if not note_events:
            return json.dumps([])

        ticks_per_beat: int = mid.ticks_per_beat or 480

        melody_notes, _, _ = _extract_melody_notes(
            note_events,
            is_xf=is_xf,
            xf_melody_channel=xf_melody_channel,
        )

        if len(melody_notes) < 2:
            return json.dumps([len(melody_notes) - 1] if melody_notes else [])

        phrase_ends = _find_phrase_endings(melody_notes, ticks_per_beat)

        # Add regular phrase boundaries every 8 syllables as a fallback
        # Only add positions NOT already covered by gap-based detection
        syllable_count = len(melody_notes)
        regular_positions: list[int] = []
        if syllable_count >= 8:
            # Use 8-syllable phrases (typical for Cantonese pop songs)
            step = 8
            regular_positions = list(range(step - 1, syllable_count, step))

        # Merge: prefer phrase_ends (gap-based), fill gaps with regular positions
        combined = sorted(set(phrase_ends + regular_positions))

        # Limit to a reasonable number (max ~15% of syllables)
        max_rhymes = max(4, syllable_count // 6)
        if len(combined) > max_rhymes:
            # Keep the most significant ones: prioritize phrase_ends, then regular
            combined = (phrase_ends + regular_positions)[:max_rhymes]
            combined = sorted(set(combined))

        # Ensure the last syllable is always included
        last = syllable_count - 1
        if last >= 0 and last not in combined:
            combined.append(last)
            combined.sort()

        return json.dumps(combined)

    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
def extract_embedded_lyrics(file_path: str) -> str:
    """
    Extract embedded lyric units from a MIDI file.

    Priority:
    1. Yamaha XF karaoke payload (`XFKM`)
    2. Standard MIDI `lyrics` meta messages

    Returns
    -------
    JSON object:
        {
            "source": "xfkm" | "lyrics_meta" | null,
            "unit_count": int,
            "units": ["こ", "ん", "な", ...]
        }
    """
    try:
        mid = _load_midi(file_path)
        units, source = _extract_embedded_lyrics(mid, file_path)
        return json.dumps(
            {
                "source": source,
                "unit_count": len(units),
                "units": units,
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Runs the MCP server over stdio (default transport expected by
    # langchain-mcp-adapters and the MCP Inspector).
    mcp.run(transport="stdio")
