from __future__ import annotations

import pytest

from agent.agents.lyrics_composer import LyricsComposerAgent
from agent.config import AGENT_MAP
from agent.memory import ShortTermMemory


class _DummyLlm:
    pass


@pytest.mark.asyncio
async def test_compose_prefers_structured_output(monkeypatch):
    agent = LyricsComposerAgent(
        config=AGENT_MAP["lyrics-composer"],
        llm=_DummyLlm(),
        memory=ShortTermMemory(),
        tools=[],
    )

    async def fake_structured(*args, **kwargs):
        return {
            "lyrics": "晨光照海面",
            "jyutping": "san4 gwong1 ziu3 hoi2 min6",
            "lines": [
                {
                    "text": "晨光照海面",
                    "jyutping": "san4 gwong1 ziu3 hoi2 min6",
                    "syllable_count": 5,
                }
            ],
            "rhyme_scheme": "A",
            "changes_made": "first draft",
            "notes": "ok",
        }

    async def should_not_call_legacy(*args, **kwargs):
        raise AssertionError("legacy _invoke_llm should not be called when schema output is valid")

    monkeypatch.setattr(agent, "_invoke_llm_structured", fake_structured)
    monkeypatch.setattr(agent, "_invoke_llm", should_not_call_legacy)

    result = await agent._compose_with_schema(
        prompt="compose",
        syllable_count=5,
        tone_sequence=[0, 2, 4, 3, 0],
    )

    assert result["lyrics"] == "晨光照海面"
    assert result["target_syllable_count"] == 5
    assert len(result["lines"]) == 1


@pytest.mark.asyncio
async def test_compose_falls_back_when_structured_missing_lyrics(monkeypatch):
    agent = LyricsComposerAgent(
        config=AGENT_MAP["lyrics-composer"],
        llm=_DummyLlm(),
        memory=ShortTermMemory(),
        tools=[],
    )

    async def fake_structured(*args, **kwargs):
        return {"lyrics": "", "lines": []}

    async def fake_legacy(*args, **kwargs):
        return '{"lyrics": "晚風裡想你", "jyutping": "maan5 fung1 leoi5 soeng2 nei5"}'

    monkeypatch.setattr(agent, "_invoke_llm_structured", fake_structured)
    monkeypatch.setattr(agent, "_invoke_llm", fake_legacy)

    result = await agent._compose_with_schema(
        prompt="compose",
        syllable_count=4,
        tone_sequence=[0, 2, 4, 3],
    )

    assert result["lyrics"] == "晚風裡想你"
    assert result["target_syllable_count"] == 4
