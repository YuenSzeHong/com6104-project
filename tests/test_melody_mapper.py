from __future__ import annotations

import asyncio
import json


def test_analyze_melody_contour_returns_lean_0243_sequence(
    melody_mapper_module,
    doraemon_midi,
):
    payload = json.loads(
        asyncio.run(melody_mapper_module.analyze_melody_contour(str(doraemon_midi)))
    )

    assert "error" not in payload
    assert payload["melody_channel"] == 0
    assert payload["syllable_count"] > 0
    assert len(payload["tone_sequence"]) == payload["syllable_count"]
    assert set(payload["tone_sequence"]).issubset({0, 2, 3, 4})
    assert payload["strong_beats"]
    assert payload["phrase_ends"]


def test_suggest_tone_sequence_matches_contour_result(
    melody_mapper_module,
    doraemon_midi,
):
    contour = json.loads(
        asyncio.run(melody_mapper_module.analyze_melody_contour(str(doraemon_midi)))
    )
    summary = json.loads(
        asyncio.run(melody_mapper_module.suggest_tone_sequence(str(doraemon_midi)))
    )

    assert "error" not in summary
    assert summary["syllable_count"] == contour["syllable_count"]
    assert summary["tone_sequence_str"].split()[:8] == [
        str(x) for x in contour["tone_sequence"][:8]
    ]
