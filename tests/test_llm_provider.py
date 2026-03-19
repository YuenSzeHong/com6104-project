from __future__ import annotations

import importlib
import os

import pytest


def test_build_llm_uses_lmstudio_by_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "lmstudio")
    monkeypatch.setenv("LMSTUDIO_MODEL", "qwen3.5-4b@q4_k_m")
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")

    import agent.config as config_module
    import agent.orchestrator as orchestrator_module

    importlib.reload(config_module)
    importlib.reload(orchestrator_module)

    llm = orchestrator_module.AgentOrchestrator._build_llm()

    assert llm.__class__.__name__ == "ChatOpenAI"
    assert getattr(llm, "model_name", None) == "qwen3.5-4b@q4_k_m"


@pytest.mark.asyncio
async def test_lmstudio_live_smoke(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "lmstudio")
    monkeypatch.setenv("LMSTUDIO_MODEL", "qwen3.5-4b@q4_k_m")
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")

    import agent.config as config_module
    import agent.orchestrator as orchestrator_module

    importlib.reload(config_module)
    importlib.reload(orchestrator_module)

    llm = orchestrator_module.AgentOrchestrator._build_llm()
    response = await llm.ainvoke("请只回复OK")
    content = getattr(response, "content", "")

    assert "OK" in str(content).upper()
