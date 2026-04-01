from __future__ import annotations

import importlib

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


def test_build_llm_uses_ollama(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:4b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    import agent.config as config_module
    import agent.orchestrator as orchestrator_module

    importlib.reload(config_module)
    importlib.reload(orchestrator_module)

    llm = orchestrator_module.AgentOrchestrator._build_llm()

    assert llm.__class__.__name__ == "ChatOllama"
    assert getattr(llm, "model", None) == "qwen3.5:4b"


def test_build_llm_uses_ollama_cloud(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama-cloud")
    monkeypatch.setenv("OLLAMA_CLOUD_MODEL", "qwen3.5:cloud")
    monkeypatch.setenv("OLLAMA_CLOUD_BASE_URL", "https://ollama.com/v1")
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")

    import agent.config as config_module
    import agent.orchestrator as orchestrator_module

    importlib.reload(config_module)
    importlib.reload(orchestrator_module)

    llm = orchestrator_module.AgentOrchestrator._build_llm()

    assert llm.__class__.__name__ == "ChatOpenAI"
    assert getattr(llm, "model_name", None) == "qwen3.5:cloud"


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


@pytest.mark.asyncio
@pytest.mark.skip(reason="Requires local Ollama service running on localhost:11434")
async def test_ollama_live_smoke(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:4b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

    import agent.config as config_module
    import agent.orchestrator as orchestrator_module

    importlib.reload(config_module)
    importlib.reload(orchestrator_module)

    llm = orchestrator_module.AgentOrchestrator._build_llm()
    response = await llm.ainvoke("请只回复OK")
    content = getattr(response, "content", "")

    assert "OK" in str(content).upper()
