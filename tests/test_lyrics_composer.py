from __future__ import annotations

import asyncio
from typing import Any, cast

from langchain_core.language_models import BaseChatModel

import pytest

from agent.agents import lyrics_composer as lyrics_composer_module
from agent.agents.lyrics_composer import LyricsComposerAgent
from agent.config import AGENT_MAP
from agent.memory import ShortTermMemory
from agent.base_agent import AgentResult


class _DummyLlm:
    pass


@pytest.mark.asyncio
async def test_compose_prefers_structured_output(monkeypatch):
    dummy_llm = cast(BaseChatModel, _DummyLlm())
    agent = LyricsComposerAgent(
        config=AGENT_MAP["lyrics-composer"],
        llm=dummy_llm,
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
        raise AssertionError(
            "legacy _invoke_llm should not be called when schema output is valid"
        )

    monkeypatch.setattr(agent, "_invoke_llm_structured", fake_structured)
    monkeypatch.setattr(agent, "_invoke_llm", should_not_call_legacy)

    agent_any: Any = agent
    result = await agent_any._compose_with_schema(
        prompt="compose",
        syllable_count=5,
        tone_sequence=[0, 2, 4, 3, 0],
    )

    assert result["lyrics"] == "晨光照海面"
    assert result["target_syllable_count"] == 5
    assert len(result["lines"]) == 1


@pytest.mark.asyncio
async def test_compose_falls_back_when_structured_missing_lyrics(monkeypatch):
    dummy_llm = cast(BaseChatModel, _DummyLlm())
    agent = LyricsComposerAgent(
        config=AGENT_MAP["lyrics-composer"],
        llm=dummy_llm,
        memory=ShortTermMemory(),
        tools=[],
    )

    async def fake_structured(*args, **kwargs):
        return {"lyrics": "", "lines": []}

    async def fake_legacy(*args, **kwargs):
        return '{"lyrics": "晚風裡想你", "jyutping": "maan5 fung1 leoi5 soeng2 nei5"}'

    monkeypatch.setattr(agent, "_invoke_llm_structured", fake_structured)
    monkeypatch.setattr(agent, "_invoke_llm", fake_legacy)

    agent_any: Any = agent
    result = await agent_any._compose_with_schema(
        prompt="compose",
        syllable_count=4,
        tone_sequence=[0, 2, 4, 3],
    )

    assert result["lyrics"] == "晚風裡想你"
    assert result["target_syllable_count"] == 4


@pytest.mark.asyncio
async def test_refine_word_selection_runs_positions_concurrently(monkeypatch):
    dummy_llm = cast(BaseChatModel, _DummyLlm())
    agent = LyricsComposerAgent(
        config=AGENT_MAP["lyrics-composer"],
        llm=dummy_llm,
        memory=ShortTermMemory(),
        tools=[],
    )

    started = 0
    release = asyncio.Event()

    class FakeWordSelector:
        def __init__(self, **kwargs):
            self.memory = kwargs.get("memory")

        async def run(self, task: str, **kwargs):
            nonlocal started
            started += 1
            if started >= 2:
                release.set()
            await asyncio.wait_for(release.wait(), timeout=0.5)
            word = "天" if "位置 0" in task else "海"
            return AgentResult(
                agent_name="word-selector",
                success=True,
                output=word,
                data={
                    "selected_words": [
                        {"word": word, "reason": "ok", "alternatives": []}
                    ]
                },
                metadata={"selection_reason": "ok"},
            )

    monkeypatch.setattr(
        lyrics_composer_module,
        "_get_word_selector_class",
        lambda: FakeWordSelector,
    )

    structured = {"lyrics": "AB", "jyutping": "", "lines": []}
    agent_any: Any = agent
    result = await asyncio.wait_for(
        agent_any._refine_word_selection(
            structured=structured,
            candidate_map={
                0: [f"甲{i}" for i in range(11)],
                1: [f"乙{i}" for i in range(11)],
            },
            melody_tone_sequence=[0, 2],
            rhyme_positions=[],
            reference_text="海天",
        ),
        timeout=1.0,
    )

    assert result["lyrics"] == "天海"
