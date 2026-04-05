from __future__ import annotations

import asyncio
from typing import cast

from langchain_core.language_models import BaseChatModel

import pytest

from agent.errors import ConstraintViolation, ToolInvokeError
from agent.memory import ShortTermMemory
from agent.orchestrator import AgentOrchestrator
from agent.registry import MCP_REGISTRY


class _DummyLlm:
    pass


@pytest.mark.asyncio
async def test_call_tool_direct_missing_tool(monkeypatch):
    dummy_llm = cast(BaseChatModel | None, _DummyLlm())
    orchestrator = AgentOrchestrator(llm=dummy_llm, memory=ShortTermMemory())

    monkeypatch.setattr(MCP_REGISTRY, "get_all_tools", lambda: [])

    with pytest.raises(ToolInvokeError):
        await orchestrator._call_tool_direct(
            server_name="jyutping",
            tool_name="find_words_by_tone_code",
            args={"code": "43"},
            parse_json=True,
        )


@pytest.mark.asyncio
async def test_orchestrator_explicit_word_selector_trigger(monkeypatch):
    dummy_llm = cast(BaseChatModel | None, _DummyLlm())
    orchestrator = AgentOrchestrator(llm=dummy_llm, memory=ShortTermMemory())

    orchestrator._agents["word-selector"] = object()

    async def fake_isolated(task, candidates, context, timeout_s):
        return {"word": "海"}

    monkeypatch.setattr(
        orchestrator,
        "_run_word_selector_isolated",
        fake_isolated,
    )

    draft = {"lyrics": "天地人和"}
    result = await orchestrator._apply_orchestrator_word_selection(
        draft_output=draft,
        candidate_map={
            "1": ["天", "海", "云", "风", "山", "雨", "夜", "光", "星", "梦", "火"],
            "2": [f"短{i}" for i in range(4)],
        },
        strong_beats=[1],
        rhyme_positions=[],
        melody_tone_sequence=[0, 2, 4, 3],
        reference_text="海阔天空",
        event_callback=None,
    )

    assert result["lyrics"] == "天海人和"


@pytest.mark.asyncio
async def test_orchestrator_word_selection_concurrent(monkeypatch):
    dummy_llm = cast(BaseChatModel | None, _DummyLlm())
    orchestrator = AgentOrchestrator(llm=dummy_llm, memory=ShortTermMemory())

    orchestrator._agents["word-selector"] = object()
    orchestrator._word_selector_max_llm_calls = 3
    orchestrator._word_selector_max_targets = 4
    orchestrator._word_selector_fast_mode = "never"

    started = 0
    release = asyncio.Event()

    async def fake_isolated(task, candidates, context, timeout_s):
        nonlocal started
        started += 1
        if started >= 2:
            release.set()
        await asyncio.wait_for(release.wait(), timeout=0.5)
        return {"word": "天" if "位置 0" in task else "海"}

    monkeypatch.setattr(
        orchestrator,
        "_run_word_selector_isolated",
        fake_isolated,
    )

    draft = {"lyrics": "AB"}
    result = await asyncio.wait_for(
        orchestrator._apply_orchestrator_word_selection(
            draft_output=draft,
            candidate_map={
                "0": ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸", "子"],
                "1": ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "百"],
            },
            strong_beats=[0, 1],
            rhyme_positions=[],
            melody_tone_sequence=[0, 2],
            reference_text="海天",
            event_callback=None,
        ),
        timeout=1.0,
    )

    assert result["lyrics"] == "天海"


def test_pipeline_state_transition_guard_raises_on_invalid_transition():
    dummy_llm = cast(BaseChatModel | None, _DummyLlm())
    orchestrator = AgentOrchestrator(llm=dummy_llm, memory=ShortTermMemory())

    orchestrator._pipeline_state = "validation"
    with pytest.raises(ConstraintViolation):
        orchestrator._transition_state("midi_analysis")
