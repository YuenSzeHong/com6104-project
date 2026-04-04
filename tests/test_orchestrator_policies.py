from __future__ import annotations

import pytest

from agent.errors import ToolInvokeError
from agent.memory import ShortTermMemory
from agent.orchestrator import AgentOrchestrator
from agent.registry import MCP_REGISTRY


class _DummyLlm:
    pass


@pytest.mark.asyncio
async def test_call_tool_direct_raises_typed_error_when_tool_missing(monkeypatch):
    orchestrator = AgentOrchestrator(llm=_DummyLlm(), memory=ShortTermMemory())

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
    orchestrator = AgentOrchestrator(llm=_DummyLlm(), memory=ShortTermMemory())

    orchestrator._agents["word-selector"] = object()

    async def fake_run_agent(agent_name: str, task: str, context_key: str, **kwargs):
        assert agent_name == "word-selector"
        assert context_key == "selected_words"
        return {"word": "海"}

    monkeypatch.setattr(orchestrator, "_run_agent", fake_run_agent)

    draft = {"lyrics": "天地人和"}
    result = await orchestrator._apply_orchestrator_word_selection(
        draft_output=draft,
        candidate_map={
            "1": [f"候选{i}" for i in range(11)],
            "2": [f"短{i}" for i in range(4)],
        },
        strong_beats=[1],
        rhyme_positions=[],
        melody_tone_sequence=[0, 2, 4, 3],
        reference_text="海阔天空",
        event_callback=None,
    )

    assert result["lyrics"] == "天海人和"
