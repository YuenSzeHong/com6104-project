from __future__ import annotations

import json
from pathlib import Path

import pytest


def _structural_score(
    analyzer_payload: dict,
    melody_payload: dict,
    rhyme_positions: list[int],
) -> float:
    expected = int(
        analyzer_payload.get("effective_syllable_count", 0)
        or analyzer_payload.get("syllable_count", 0)
    )
    actual = int(melody_payload.get("syllable_count", 0))

    max_len = max(expected, actual, 1)
    syllable_match = max(0.0, 1.0 - abs(expected - actual) / max_len)

    tones = melody_payload.get("tone_sequence", [])
    valid_tones = {0, 2, 3, 4}
    if tones:
        tone_validity = sum(1 for tone in tones if tone in valid_tones) / len(tones)
    else:
        tone_validity = 0.0

    if actual > 0:
        rhyme_density = min(1.0, len(rhyme_positions) / max(1.0, actual / 8.0))
    else:
        rhyme_density = 0.0

    return round(0.5 * syllable_match + 0.3 * tone_validity + 0.2 * rhyme_density, 4)


@pytest.mark.asyncio
async def test_structural_regression_baseline(
    mcp_session_midi_analyzer,
    mcp_session_melody_mapper,
    repo_root: Path,
):
    midi_files = [
        repo_root / "test" / "midi" / "R00317G2.mid",
        repo_root / "test" / "midi" / "X54896G2.mid",
        repo_root / "test" / "midi" / "ドラえもんのうた.mid",
    ]

    scores: list[float] = []

    async with (
        mcp_session_midi_analyzer() as midi_session,
        mcp_session_melody_mapper() as melody_session,
    ):
        for midi_path in midi_files:
            analyze_res = await midi_session.call_tool(
                "analyze_midi",
                {"file_path": str(midi_path)},
            )
            melody_res = await melody_session.call_tool(
                "analyze_melody_contour",
                {"file_path": str(midi_path)},
            )
            rhyme_res = await midi_session.call_tool(
                "suggest_rhyme_positions",
                {"file_path": str(midi_path)},
            )

            analyze_payload = json.loads(analyze_res.content[0].text)
            melody_payload = json.loads(melody_res.content[0].text)
            rhyme_positions = json.loads(rhyme_res.content[0].text)

            assert "error" not in analyze_payload
            assert "error" not in melody_payload
            assert isinstance(rhyme_positions, list)

            score = _structural_score(analyze_payload, melody_payload, rhyme_positions)
            scores.append(score)

    mean_score = sum(scores) / len(scores)
    variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)

    assert min(scores) >= 0.75
    assert mean_score >= 0.82
    assert variance <= 0.03
