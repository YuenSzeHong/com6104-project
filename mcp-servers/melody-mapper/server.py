#!/usr/bin/env python3
"""
Melody Mapper MCP Server
========================
Maps MIDI melody contours to the 0243.hk lean-mode tone system.

The 0243 Tone System (Lean Mode)
--------------------------------
0243.hk uses a simplified 4-tone system for Cantonese lyrics composition:

| Code | Tone Name  | Contour | Pitch Range | Description      | Example |
|------|------------|---------|-------------|------------------|---------|
|  0   | 阴平 (Yin Ping)  | 55/53   | High level/falling  | 高平/微降  | 诗、天、高 |
|  2   | 阴上 (Yin Shang) | 35      | Mid rising    | 高升          | 史、水、走 |
|  4   | 阴去 (Yin Qu)  | 33      | Mid level     | 中平          | 试、去、笑 |
|  3   | 阳平 (Yang Ping) | 21/11   | Low falling/level | 低降/低平  | 时、来、求 |

Note: This system excludes the entering tones (入声 7, 8, 9) which end in
-p, -t, -k, as they are less common in modern song lyrics.

Melody-to-Tone Mapping Rules
----------------------------
The algorithm analyzes the MIDI melody contour and maps each note to a
0243 tone code based on:

1. **Relative pitch within phrase**: Notes are normalized to the local
   key/range, then classified by their scale degree position.

2. **Contour direction**: Rising/falling/level melody segments map to
   corresponding tone contours.

3. **Strong beat preference**: Strong beats prefer stable tones (0, 4),
   weak beats can use flowing tones (2, 3).

4. **Phrase position**: Phrase endings prefer low tones (3, 4) for
   natural cadence.

Tools
-----
analyze_melody_contour(file_path)
    Extract the melody contour from MIDI and return 0243 tone codes
    for each syllable position.

get_tone_requirements(file_path, position)
    Get the required 0243 tone code for a specific syllable position
    based on melody analysis.

find_words_by_melody(file_path, position, count=10)
    Find Chinese words that match the melody's tone requirement at
    a specific position. Calls 0243.hk API with the derived tone code.

suggest_tone_sequence(file_path)
    Return the complete 0243 tone sequence for the entire melody.

Transport
---------
stdio (compatible with langchain-mcp-adapters MultiServerMCPClient)

Testing with MCP Inspector
--------------------------
    npx @modelcontextprotocol/inspector python mcp-servers/melody-mapper/server.py
"""

from __future__ import annotations

import json
import logging
import sys
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
import mido
from mcp.server.fastmcp import FastMCP
from xfmido import extract_xf_karaoke_info

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[melody-mapper] %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("melody-mapper")

# ---------------------------------------------------------------------------
# 0243.hk API client (for word lookup)
# ---------------------------------------------------------------------------

_API_URL = "https://www.0243.hk/api/cls/"
_TIMEOUT = 15.0
_MAX_RETRIES = 3


async def _call_0243_api(nums: str) -> list[str]:
    """Call 0243.hk API with a tone code, return Chinese word candidates."""
    payload = {"nums": nums}

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = await client.post(
                    _API_URL,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "User-Agent": "MelodyMapper/1.0 (MCP; 0243.hk)",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                if isinstance(data, list):
                    return [str(item) for item in data if item is not None]
                return []

            except Exception as exc:
                logger.warning(
                    "0243.hk API error (attempt %d/%d): %s",
                    attempt, _MAX_RETRIES, exc
                )
                if attempt < _MAX_RETRIES:
                    import asyncio
                    await asyncio.sleep(1.0 * attempt)

    return []

# ---------------------------------------------------------------------------
# 0243 Tone System Constants
# ---------------------------------------------------------------------------

# 0243 tone codes and their characteristics
_TONE_0 = 0  # 阴平：高平/微降 (55/53) - stable, high
_TONE_2 = 2  # 阴上：高升 (35) - rising
_TONE_4 = 4  # 阴去：中平 (33) - neutral, mid
_TONE_3 = 3  # 阳平：低降/低平 (21/11) - low, falling

# Stable tones for strong beats
_STABLE_TONES = {_TONE_0, _TONE_4}

# Flowing tones for weak beats/passing notes
_FLOWING_TONES = {_TONE_2, _TONE_3}

# Phrase-ending tones (natural cadence)
_CADENCE_TONES = {_TONE_3, _TONE_4}

# ---------------------------------------------------------------------------
# MIDI Analysis Utilities
# ---------------------------------------------------------------------------


def _load_midi(file_path: str) -> mido.MidiFile:
    """Load and validate a MIDI file."""
    p = Path(file_path)
    if not p.exists():
        raise ValueError(f"MIDI file not found: {p}")
    if p.suffix.lower() not in {".mid", ".midi"}:
        raise ValueError(f"Not a MIDI file: {p}")
    return mido.MidiFile(str(p))


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


def _extract_melody_notes(mid: mido.MidiFile) -> list[dict[str, Any]]:
    """
    Extract melody notes from MIDI, merged by channel.

    Returns list of note events with:
    - note: MIDI note number
    - start_tick, end_tick
    - start_sec, duration_sec
    - channel
    """
    ticks_per_beat = mid.ticks_per_beat or 480
    tempo_us = 500_000  # default 120 BPM

    # Collect all messages with absolute tick times
    raw_events = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            raw_events.append((abs_tick, msg))

    raw_events.sort(key=lambda x: x[0])

    # Build note events
    pending = {}  # (channel, note) -> (tick, sec)
    completed = []
    current_sec = 0.0
    last_tick = 0

    for abs_tick, msg in raw_events:
        delta_ticks = abs_tick - last_tick
        current_sec += (delta_ticks / ticks_per_beat) * (tempo_us / 1_000_000)
        last_tick = abs_tick

        if msg.type == "set_tempo":
            tempo_us = msg.tempo

        elif msg.type == "note_on" and msg.velocity > 0:
            pending[(msg.channel, msg.note)] = (abs_tick, current_sec)

        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in pending:
                start_tick, start_sec = pending.pop(key)
                duration_sec = current_sec - start_sec
                if duration_sec > 0.03:  # filter very short notes
                    completed.append({
                        "note": msg.note,
                        "velocity": msg.velocity,
                        "start_tick": start_tick,
                        "end_tick": abs_tick,
                        "start_sec": start_sec,
                        "duration_sec": duration_sec,
                        "channel": msg.channel,
                    })

    completed.sort(key=lambda e: e["start_sec"])
    return completed


def _group_events_by_channel(note_events: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
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
            key=lambda ev: (int(ev["note"]), float(ev["duration_sec"]), int(ev["velocity"])),
        )
        topline.append(dict(chosen))
    topline.sort(key=lambda e: e["start_sec"])
    return topline


def _get_melody_channel(
    note_events: list[dict],
    *,
    is_xf: bool = False,
    xf_melody_channel: int | None = None,
) -> tuple[int, str]:
    """Find the channel that best resembles a monophonic lead melody."""
    if not note_events:
        return 0, "no_notes"

    grouped = _group_events_by_channel(note_events)
    if is_xf and xf_melody_channel is not None and xf_melody_channel in grouped:
        return xf_melody_channel, "yamaha_xf_karaoke_info"
    if is_xf and 0 in grouped:
        return 0, "yamaha_xf_default_channel_0"
    best_channel = 0
    best_score = float("-inf")

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

    return best_channel, "heuristic_channel_score"


def _merge_ornaments(note_events: list[dict], melody_channel: int,
                     gap_threshold: float = 0.03) -> list[dict]:
    """
    Merge ornamental notes (very close together) into single syllables.
    Returns one event per syllable with averaged pitch.
    """
    channel_events = [e for e in note_events if e["channel"] == melody_channel]
    melody_notes = _extract_top_note_line(channel_events)
    if not melody_notes:
        return []

    merged = []
    cluster_start_sec = melody_notes[0]["start_sec"]
    cluster_start_tick = melody_notes[0]["start_tick"]
    prev = melody_notes[0]
    accumulated_dur = prev["duration_sec"]
    pitch_sum = prev["note"] * prev["duration_sec"]

    for curr in melody_notes[1:]:
        onset_gap = curr["start_sec"] - prev["start_sec"]

        if onset_gap <= gap_threshold:
            # Same syllable - accumulate
            prev_end = prev["start_sec"] + prev["duration_sec"]
            curr_end = curr["start_sec"] + curr["duration_sec"]
            accumulated_dur = max(prev_end, curr_end) - cluster_start_sec
            pitch_sum += curr["note"] * curr["duration_sec"]
            prev = curr
        else:
            # New syllable - emit previous
            avg_pitch = pitch_sum / accumulated_dur if accumulated_dur > 0 else prev["note"]
            merged.append({
                "note": round(avg_pitch),
                "start_sec": cluster_start_sec,
                "duration_sec": accumulated_dur,
                "start_tick": cluster_start_tick,
            })
            cluster_start_sec = curr["start_sec"]
            cluster_start_tick = curr["start_tick"]
            prev = curr
            accumulated_dur = curr["duration_sec"]
            pitch_sum = curr["note"] * curr["duration_sec"]

    # Emit final syllable
    if prev:
        avg_pitch = pitch_sum / accumulated_dur if accumulated_dur > 0 else prev["note"]
        merged.append({
            "note": round(avg_pitch),
            "start_sec": cluster_start_sec,
            "duration_sec": accumulated_dur,
            "start_tick": cluster_start_tick,
        })

    return merged


def _detect_key_signature(note_events: list[dict]) -> tuple[int, bool]:
    """
    Detect key signature from note histogram.
    Returns (root_note, is_minor).
    """
    if not note_events:
        return (60, False)  # default C4

    pitch_class_counts = [0] * 12
    for ev in note_events:
        pitch_class_counts[ev["note"] % 12] += 1

    # Simple heuristic: most common note is likely the tonic
    root = max(range(12), key=lambda i: pitch_class_counts[i])

    # Check if minor by looking for minor third
    minor_third = (root + 3) % 12
    major_third = (root + 4) % 12

    is_minor = pitch_class_counts[minor_third] > pitch_class_counts[major_third]

    return (root, is_minor)


def _normalize_pitch_to_scale(note: int, root: int, is_minor: bool) -> int:
    """
    Convert MIDI note to scale degree (0-7, where 0=tonic).
    """
    pitch_class = note % 12
    root_pc = root % 12

    if is_minor:
        # Natural minor scale intervals
        scale_intervals = [0, 2, 3, 5, 7, 8, 10]
    else:
        # Major scale intervals
        scale_intervals = [0, 2, 4, 5, 7, 9, 11]

    # Find closest scale degree
    diff = (pitch_class - root_pc) % 12
    closest = min(scale_intervals, key=lambda x: min(abs(x - diff), 12 - abs(x - diff)))
    return scale_intervals.index(closest)


# ---------------------------------------------------------------------------
# 0243 Tone Mapping Algorithm
# ---------------------------------------------------------------------------


def _map_pitch_to_0243(
    scale_degree: int,
    contour: str,
    is_strong_beat: bool,
    is_phrase_end: bool,
    duration: float,
    median_duration: float,
) -> int:
    """
    Map a melody note to a 0243 tone code based on multiple factors.

    Parameters
    ----------
    scale_degree : int
        Scale degree (0=tonic, 1=supertonic, ..., 7=octave)
    contour : str
        Melody contour: 'rising', 'falling', 'level_high', 'level_mid', 'level_low'
    is_strong_beat : bool
        True if this note falls on a strong beat
    is_phrase_end : bool
        True if this is at a phrase ending
    duration : float
        Note duration in seconds
    median_duration : float
        Median note duration for this melody

    Returns
    -------
    int : 0243 tone code (0, 2, 3, or 4)
    """
    # High scale degrees (5, 6, 7) tend toward tone 0 (high level)
    # Mid scale degrees (2, 3, 4) tend toward tone 4 (mid level)
    # Low scale degrees (0, 1) tend toward tone 3 (low)

    base_tone: int

    if scale_degree >= 5:
        # High register
        if contour == 'level_high' or contour == 'level_mid':
            base_tone = _TONE_0  # 高平
        elif contour == 'rising':
            base_tone = _TONE_2  # 高升
        else:
            base_tone = _TONE_0  # default high level

    elif scale_degree >= 2:
        # Mid register
        if contour == 'level_mid':
            base_tone = _TONE_4  # 中平
        elif contour == 'rising':
            base_tone = _TONE_2  # 高升
        elif contour == 'falling':
            base_tone = _TONE_3  # 低降
        else:
            base_tone = _TONE_4  # default mid

    else:
        # Low register
        if contour == 'level_low' or contour == 'falling':
            base_tone = _TONE_3  # 低降/低平
        elif contour == 'rising':
            base_tone = _TONE_2  # 高升 from low
        else:
            base_tone = _TONE_3  # default low

    # Adjust for strong beats (prefer stable tones)
    if is_strong_beat and base_tone in _FLOWING_TONES:
        if scale_degree >= 4:
            base_tone = _TONE_0  # stable high
        else:
            base_tone = _TONE_4  # stable mid

    # Adjust for phrase endings (prefer cadence tones)
    if is_phrase_end and base_tone not in _CADENCE_TONES:
        if scale_degree >= 3:
            base_tone = _TONE_4  # mid cadence
        else:
            base_tone = _TONE_3  # low cadence

    # Long notes prefer stable tones
    if duration > median_duration * 1.5:
        if base_tone in _FLOWING_TONES:
            base_tone = _TONE_0 if scale_degree >= 4 else _TONE_4

    return base_tone


def _analyze_contour(
    syllables: list[dict],
    window: int = 3
) -> list[str]:
    """
    Analyze melody contour for each syllable.

    Returns list of contour labels:
    'rising', 'falling', 'level_high', 'level_mid', 'level_low'
    """
    if len(syllables) < 2:
        return ['level_mid'] * len(syllables)

    contours = []
    notes = [s["note"] for s in syllables]

    for i in range(len(syllables)):
        # Look at context window
        start = max(0, i - window // 2)
        end = min(len(notes), i + window // 2 + 1)
        context = notes[start:end]

        if len(context) < 2:
            contours.append('level_mid')
            continue

        # Calculate local trend
        first_half = context[:len(context)//2+1]
        second_half = context[len(context)//2:]

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        diff = second_avg - first_avg

        # Classify contour
        if diff > 1.5:
            contours.append('rising')
        elif diff < -1.5:
            contours.append('falling')
        else:
            # Level - determine register
            avg_note = sum(context) / len(context)
            local_min = min(notes[max(0, i-3):min(len(notes), i+4)])
            local_max = max(notes[max(0, i-3):min(len(notes), i+4)])
            range_mid = (local_min + local_max) / 2

            if avg_note > range_mid + 2:
                contours.append('level_high')
            elif avg_note < range_mid - 2:
                contours.append('level_low')
            else:
                contours.append('level_mid')

    return contours


def _find_phrase_ends(syllables: list[dict], threshold_factor: float = 1.5) -> set[int]:
    """Find syllable indices that are phrase endings (followed by long gaps)."""
    if len(syllables) < 2:
        return {len(syllables) - 1} if syllables else set()

    # Calculate inter-onset intervals
    iois = [
        syllables[i + 1]["start_sec"] - syllables[i]["start_sec"]
        for i in range(len(syllables) - 1)
    ]

    if not iois:
        return {0}

    median_ioi = sorted(iois)[len(iois) // 2]
    threshold = median_ioi * threshold_factor

    phrase_ends = set()
    for i, ioi in enumerate(iois):
        if ioi >= threshold:
            phrase_ends.add(i)

    # Always include final syllable
    phrase_ends.add(len(syllables) - 1)

    return phrase_ends


def _find_strong_beats(syllables: list[dict], ticks_per_beat: int,
                       numerator: int = 4) -> set[int]:
    """Find syllable indices that fall on strong beats."""
    if not syllables or ticks_per_beat <= 0:
        return set()

    tolerance = ticks_per_beat * 0.15
    strong_beats = set()

    ticks_per_bar = ticks_per_beat * numerator

    for i, syl in enumerate(syllables):
        tick = syl.get("start_tick", 0)
        pos_in_bar = tick % ticks_per_bar

        # Beat 1 (downbeat) and beat 3 (in 4/4) are strong
        if abs(pos_in_bar) <= tolerance:
            strong_beats.add(i)
        elif abs(pos_in_bar - ticks_per_beat * 2) <= tolerance and numerator >= 3:
            strong_beats.add(i)
        elif abs(pos_in_bar - ticks_per_bar) <= tolerance:
            strong_beats.add(i)

    return strong_beats


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="melody-mapper",
    instructions=(
        "Maps MIDI melody contours to 0243.hk lean-mode tone system. "
        "Analyzes melody pitch, contour, and rhythm to derive tone requirements "
        "for Cantonese lyrics composition."
    ),
)


@mcp.tool()
async def analyze_melody_contour(file_path: str) -> str:
    """
    Analyze MIDI melody and return 0243 tone codes for each syllable.

    This is the main entry point for melody-to-tone mapping.

    Parameters
    ----------
    file_path : str
        Path to MIDI file.

    Returns
    -------
    str
        JSON object with:
        {
            "tone_sequence": [0, 2, 4, 3, ...],
            "syllable_count": int,
            "contours": ["rising", "level_high", ...],
            "strong_beats": [0, 4, 8, ...],
            "phrase_ends": [7, 15, ...],
            "key": {"root": int, "is_minor": bool},
        }
    """
    try:
        mid = _load_midi(file_path)
        is_xf = _is_yamaha_xf(mid)
        xf_melody_channel = _get_xf_melody_channel(file_path) if is_xf else None
        note_events = _extract_melody_notes(mid)

        if not note_events:
            return json.dumps({"error": "No melody notes found"})

        # Find melody channel and merge ornaments
        melody_ch, selection_reason = _get_melody_channel(
            note_events,
            is_xf=is_xf,
            xf_melody_channel=xf_melody_channel,
        )
        syllables = _merge_ornaments(note_events, melody_ch)

        if not syllables:
            return json.dumps({"error": "No melody syllables after merging"})

        # Detect key
        root, is_minor = _detect_key_signature(syllables)

        # Analyze contours
        contours = _analyze_contour(syllables)

        # Find structural positions
        ticks_per_beat = mid.ticks_per_beat or 480
        strong_beats = _find_strong_beats(syllables, ticks_per_beat)
        phrase_ends = _find_phrase_ends(syllables)

        # Calculate median duration
        durations = [s["duration_sec"] for s in syllables]
        median_dur = sorted(durations)[len(durations) // 2] if durations else 0.5

        # Map each syllable to 0243 tone
        tone_sequence = []
        for i, syl in enumerate(syllables):
            scale_deg = _normalize_pitch_to_scale(syl["note"], root, is_minor)
            tone = _map_pitch_to_0243(
                scale_degree=scale_deg,
                contour=contours[i],
                is_strong_beat=i in strong_beats,
                is_phrase_end=i in phrase_ends,
                duration=syl["duration_sec"],
                median_duration=median_dur,
            )
            tone_sequence.append(tone)

        result = {
            "tone_sequence": tone_sequence,
            "syllable_count": len(syllables),
            "contours": contours,
            "strong_beats": sorted(strong_beats),
            "phrase_ends": sorted(phrase_ends),
            "key": {"root": root, "is_minor": is_minor},
            "is_xf": is_xf,
            "xf_melody_channel": xf_melody_channel,
            "melody_channel": melody_ch,
            "melody_selection_reason": selection_reason,
            "syllables": [
                {"note": s["note"], "duration": round(s["duration_sec"], 3)}
                for s in syllables
            ],
        }

        return json.dumps(result, ensure_ascii=False)

    except Exception as exc:
        logger.exception("analyze_melody_contour error: %s", exc)
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def get_tone_requirements(file_path: str, position: int) -> str:
    """
    Get the required 0243 tone code for a specific syllable position.

    Parameters
    ----------
    file_path : str
        Path to MIDI file.
    position : int
        0-based syllable position.

    Returns
    -------
    str
        JSON object with:
        {
            "position": int,
            "tone_code": int,  # 0, 2, 3, or 4
            "is_strong_beat": bool,
            "is_phrase_end": bool,
            "contour": str,
            "scale_degree": int,
            "alternatives": [int, ...],  # other acceptable tones
        }
    """
    try:
        mid = _load_midi(file_path)
        is_xf = _is_yamaha_xf(mid)
        xf_melody_channel = _get_xf_melody_channel(file_path) if is_xf else None
        note_events = _extract_melody_notes(mid)
        melody_ch, _ = _get_melody_channel(
            note_events,
            is_xf=is_xf,
            xf_melody_channel=xf_melody_channel,
        )
        syllables = _merge_ornaments(note_events, melody_ch)

        if position < 0 or position >= len(syllables):
            return json.dumps({"error": f"Position {position} out of range"})

        root, is_minor = _detect_key_signature(syllables)
        contours = _analyze_contour(syllables)
        strong_beats = _find_strong_beats(syllables, mid.ticks_per_beat or 480)
        phrase_ends = _find_phrase_ends(syllables)

        syl = syllables[position]
        scale_deg = _normalize_pitch_to_scale(syl["note"], root, is_minor)
        primary_tone = _map_pitch_to_0243(
            scale_degree=scale_deg,
            contour=contours[position],
            is_strong_beat=position in strong_beats,
            is_phrase_end=position in phrase_ends,
            duration=syl["duration_sec"],
            median_duration=sorted([s["duration_sec"] for s in syllables])[len(syllables)//2],
        )

        # Determine acceptable alternatives
        alternatives = []
        if primary_tone in _STABLE_TONES:
            alternatives = [t for t in _STABLE_TONES if t != primary_tone]
        else:
            alternatives = [t for t in _FLOWING_TONES if t != primary_tone]

        result = {
            "position": position,
            "tone_code": primary_tone,
            "is_strong_beat": position in strong_beats,
            "is_phrase_end": position in phrase_ends,
            "contour": contours[position],
            "scale_degree": scale_deg,
            "alternatives": alternatives,
        }

        return json.dumps(result, ensure_ascii=False)

    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def suggest_tone_sequence(file_path: str) -> str:
    """
    Return the complete 0243 tone sequence for the melody.

    Simplified version of analyze_melody_contour - returns only the
    essential tone sequence string.

    Parameters
    ----------
    file_path : str
        Path to MIDI file.

    Returns
    -------
    str
        Space-separated 0243 tone codes, e.g. "0 2 4 3 0 4 2 3"
    """
    try:
        result = json.loads(await analyze_melody_contour(file_path))

        if "error" in result:
            return json.dumps(result)

        tone_seq = " ".join(str(t) for t in result["tone_sequence"])
        return json.dumps({
            "tone_sequence_str": tone_seq,
            "syllable_count": result["syllable_count"],
        })

    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def find_words_by_melody(
    file_path: str,
    position: int,
    count: int = 10,
) -> str:
    """
    Find Chinese words that match the melody's tone requirement at a position.

    This tool combines melody analysis with 0243.hk API lookup.

    Parameters
    ----------
    file_path : str
        Path to MIDI file.
    position : int
        0-based syllable position in the melody.
    count : int
        Maximum number of word candidates to return.

    Returns
    -------
    str
        JSON array of Chinese words that match the tone requirement.
    """
    try:
        # Get tone requirement
        req_result = json.loads(await get_tone_requirements(file_path, position))

        if "error" in req_result:
            return json.dumps(req_result)

        tone_code = str(req_result["tone_code"])

        # Call 0243.hk API
        candidates = await _call_0243_api(tone_code)

        # Filter to Chinese words only
        chinese_words = [
            c for c in candidates
            if any('\u4e00' <= ch <= '\u9fff' for ch in c)
        ][:count]

        return json.dumps({
            "position": position,
            "tone_code": tone_code,
            "words": chinese_words,
            "alternatives": req_result.get("alternatives", []),
        })

    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def find_phrase_words(
    file_path: str,
    phrase_start: int,
    phrase_length: int,
) -> str:
    """
    Find Chinese words/phrases that match a multi-syllable phrase in the melody.

    Parameters
    ----------
    file_path : str
        Path to MIDI file.
    phrase_start : int
        Starting syllable position (0-based).
    phrase_length : int
        Number of syllables in the phrase.

    Returns
    -------
    str
        JSON object with tone sequence and word candidates for the phrase.
    """
    try:
        mid = _load_midi(file_path)
        is_xf = _is_yamaha_xf(mid)
        xf_melody_channel = _get_xf_melody_channel(file_path) if is_xf else None
        note_events = _extract_melody_notes(mid)
        melody_ch, _ = _get_melody_channel(
            note_events,
            is_xf=is_xf,
            xf_melody_channel=xf_melody_channel,
        )
        syllables = _merge_ornaments(note_events, melody_ch)

        if phrase_start + phrase_length > len(syllables):
            return json.dumps({"error": "Phrase extends beyond melody"})

        # Get tone sequence for the phrase
        tone_codes = []
        for i in range(phrase_start, phrase_start + phrase_length):
            req = json.loads(await get_tone_requirements(file_path, i))
            if "error" not in req:
                tone_codes.append(str(req["tone_code"]))

        tone_seq = "".join(tone_codes)

        # Query 0243.hk for multi-syllable words
        candidates = await _call_0243_api(tone_seq)

        chinese_words = [
            c for c in candidates
            if any('\u4e00' <= ch <= '\u9fff' for ch in c)
        ][:20]

        return json.dumps({
            "phrase_start": phrase_start,
            "phrase_length": phrase_length,
            "tone_sequence": tone_seq,
            "words": chinese_words,
        })

    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Melody Mapper MCP server starting (stdio transport)")
    mcp.run(transport="stdio")
