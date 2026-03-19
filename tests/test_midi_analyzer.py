from __future__ import annotations

import json


def test_analyze_midi_returns_expected_metadata(midi_analyzer_module, doraemon_midi):
    payload = json.loads(midi_analyzer_module.analyze_midi(str(doraemon_midi)))

    assert "error" not in payload
    assert payload["syllable_count"] == 98
    assert payload["effective_syllable_count"] == 98
    assert payload["effective_syllable_count_source"] == "embedded_lyrics"
    assert payload["embedded_lyrics_source"] == "lyrics_meta"
    assert payload["embedded_lyric_unit_count"] == 98
    assert payload["melody_channel"] == 0
    assert payload["track_count"] == 4
    assert payload["bpm"] > 0
    assert payload["strong_beat_positions"]


def test_extract_embedded_lyrics_returns_units(midi_analyzer_module, doraemon_midi):
    payload = json.loads(midi_analyzer_module.extract_embedded_lyrics(str(doraemon_midi)))

    assert "error" not in payload
    assert payload["source"] == "lyrics_meta"
    assert payload["unit_count"] == 98
    assert payload["units"][:4] == ["こん", "な", "こ", "と"]
