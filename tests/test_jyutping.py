from __future__ import annotations

import asyncio
import json
import os

import pytest


def test_find_words_by_tone_code_accepts_int_and_str(jyutping_module, monkeypatch):
    calls: list[str] = []

    async def fake_call_api(query: str):
        calls.append(query)
        return ["世界", "唱歌"]

    monkeypatch.setattr(jyutping_module, "_call_api", fake_call_api)

    result_from_int = json.loads(asyncio.run(jyutping_module.find_words_by_tone_code(0)))
    result_from_str = json.loads(asyncio.run(jyutping_module.find_words_by_tone_code("43")))

    assert result_from_int == ["世界", "唱歌"]
    assert result_from_str == ["世界", "唱歌"]
    assert calls == ["0", "43"]


def test_find_tone_continuation_accepts_int_digits(jyutping_module, monkeypatch):
    calls: list[str] = []

    async def fake_call_api(query: str):
        calls.append(query)
        return ["更多", "唱歌"]

    monkeypatch.setattr(jyutping_module, "_call_api", fake_call_api)

    result = asyncio.run(jyutping_module.find_tone_continuation("我要", 43))
    payload = json.loads(result)

    assert payload == ["更多", "唱歌"]
    assert calls == ["我要43"]


@pytest.mark.skipif(
    os.getenv("SKIP_NETWORK_TESTS", "").lower() == "true",
    reason="Set SKIP_NETWORK_TESTS=true to skip tests that require internet access.",
)
def test_query_raw_live_smoke(jyutping_module):
    payload = json.loads(asyncio.run(jyutping_module.query_raw("43")))

    assert isinstance(payload, list)
    assert payload
