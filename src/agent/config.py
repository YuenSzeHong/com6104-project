"""
Agent configuration: LLM provider settings, MCP server definitions, and agent pipeline config.

Edit this file to:
  - Switch between LLM providers (Ollama / LM Studio)
  - Add new MCP servers
  - Add new pipeline agents
  - Adjust workflow parameters

Provider quick-switch
---------------------
Set the environment variable LLM_PROVIDER to either "ollama" or "lmstudio":

    # PowerShell
    $env:LLM_PROVIDER = "lmstudio"

    # bash / zsh
    export LLM_PROVIDER=lmstudio

Or just change the DEFAULT_PROVIDER constant below.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

ROOT_DIR        = Path(__file__).resolve().parents[2]   # repo root
MCP_SERVERS_DIR = ROOT_DIR / "mcp-servers"
PROMPTS_DIR     = ROOT_DIR / "prompts"
ENV_FILE        = ROOT_DIR / ".env"
PYTHON_BIN      = sys.executable

# Load local development overrides from .env without clobbering shell-provided vars.
load_dotenv(ENV_FILE, override=False)

# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

ProviderName = Literal["ollama", "lmstudio"]

# Change this constant OR set env-var LLM_PROVIDER at runtime.
DEFAULT_PROVIDER: ProviderName = "lmstudio"

PROVIDER: ProviderName = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).lower()  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Ollama configuration  (local, no API key required)
#
# Default model: qwen3.5:4b
# Pull it first:  ollama pull qwen3.5:4b
# ---------------------------------------------------------------------------

OLLAMA_CONFIG: dict[str, Any] = {
    "model":       os.getenv("OLLAMA_MODEL",    "qwen3.5:4b"),
    "base_url":    os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
    "num_ctx":     int(os.getenv("LLM_CTX",    "8192")),
}

# ---------------------------------------------------------------------------
# LM Studio configuration  (OpenAI-compatible API, no API key required)
#
# LM Studio exposes an OpenAI-compatible server on port 1234 by default.
# Enable it in LM Studio → Local Server → Start Server.
# Make sure you have the qwen3.5-4b@q4_k_m model loaded there.
#
# We use langchain-openai's ChatOpenAI with a custom base_url.
# ---------------------------------------------------------------------------

LMSTUDIO_CONFIG: dict[str, Any] = {
    "model":       os.getenv("LMSTUDIO_MODEL",    "qwen3.5-4b@q4_k_m"),
    "base_url":    os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
    # LM Studio does not require a real API key; the string below is a placeholder.
    "api_key":     os.getenv("LMSTUDIO_API_KEY",  "lm-studio"),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
    # max_tokens maps to num_ctx equivalent for OpenAI-compat providers
    "max_tokens":  int(os.getenv("LLM_CTX", "8192")),
}

# Resolved active LLM config (used by the orchestrator)
LLM_CONFIG: dict[str, Any] = OLLAMA_CONFIG if PROVIDER == "ollama" else LMSTUDIO_CONFIG

# ---------------------------------------------------------------------------
# Short-term memory configuration
# ---------------------------------------------------------------------------

MEMORY_CONFIG: dict[str, Any] = {
    # Maximum conversation turns kept in the sliding window
    "max_turns":         int(os.getenv("MEMORY_MAX_TURNS", "20")),
    # Shared system prompt file (Chinese)
    "system_prompt_file": PROMPTS_DIR / "system.md",
}

# ---------------------------------------------------------------------------
# MCP Server definitions
#
# Fields
# ------
# name        – unique key used throughout the agent code
# command     – executable (e.g. "python", "node")
# args        – list of CLI arguments
# tool_names  – EXACT names of tools this server exposes.
#               Used for per-agent tool filtering without hitting a server API.
#               Must match the names declared in the server's @tool / Tool() calls.
# env         – optional extra env-vars (merged with os.environ at startup)
# description – shown in logs / UI
# enabled     – set False to skip without deleting the entry
# ---------------------------------------------------------------------------

@dataclass
class MCPServerConfig:
    name:        str
    command:     str
    args:        list[str]
    tool_names:  list[str]             = field(default_factory=list)
    env:         dict[str, str]        = field(default_factory=dict)
    description: str                   = ""
    enabled:     bool                  = True

    def to_langchain_params(self) -> dict[str, Any]:
        """
        Return the connection-params dict expected by MultiServerMCPClient.

        The 'tool_names' field is framework-internal and intentionally
        excluded from the MCP client params.
        """
        params: dict[str, Any] = {
            "command":   self.command,
            "args":      self.args,
            "transport": "stdio",
            "encoding":  "utf-8",
            "encoding_error_handler": "replace",
        }
        if self.env:
            params["env"] = {**os.environ, **self.env}
        return params


MCP_SERVERS: list[MCPServerConfig] = [

    # ------------------------------------------------------------------
    # 1. Jyutping / 0243.hk IME engine
    #
    # Three query modes (all via the same endpoint):
    #   Mode 1 – numeric tone code → Chinese words
    #   Mode 2 – Chinese text → Jyutping candidates / tone codes
    #   Mode 3 – Chinese prefix + tone digits → context-aware continuation
    # ------------------------------------------------------------------
    MCPServerConfig(
        name        = "jyutping",
        command     = PYTHON_BIN,
        args        = [str(MCP_SERVERS_DIR / "jyutping" / "server.py")],
        tool_names  = [
            "query_raw",                # 直接透传 0243.hk API（全格式）
            "chinese_to_jyutping",      # Mode 2：中文 → 粤拼候选列表
            "get_tone_code",            # Mode 2：中文 → 数字声调码
            "get_tone_pattern",         # Mode 2：中文 → 空格分隔声调序列
            "find_words_by_tone_code",  # Mode 1：数字声调码 → 中文词语
            "find_tone_continuation",   # Mode 3：中文前缀 + 声调数字 → 续词候选
        ],
        description = "粤语 IME 引擎 – 通过 0243.hk API 提供粤拼转换、声调码查词及上下文续词",
    ),

    # ------------------------------------------------------------------
    # 2. MIDI analyser
    #
    # Pure computational skills: no LLM needed, fully testable via
    # MCP Inspector. Previously these lived inside MidiAnalyserAgent;
    # moved here so they satisfy the spec's MCP testability requirement.
    # ------------------------------------------------------------------
    MCPServerConfig(
        name        = "midi-analyzer",
        command     = PYTHON_BIN,
        args        = [str(MCP_SERVERS_DIR / "midi-analyzer" / "server.py")],
        tool_names  = [
            "analyze_midi",            # 完整 MIDI 元数据（音节数、速度、调性、强拍位置）
            "get_syllable_durations",  # 每音符时值列表（秒）
            "suggest_rhyme_positions", # 建议押韵位置索引
        ],
        description = "MIDI 旋律分析 – 音节数、速度、调性、强拍位置（纯计算，无需 LLM）",
    ),

    # ------------------------------------------------------------------
    # 3. Lyrics validator
    #
    # All computational validation skills moved OUT of ValidatorAgent
    # into this MCP server so they are independently testable.
    # The MCP server now treats lean 0243 melody fit as the primary
    # computational constraint, while ValidatorAgent adds the final
    # artistic-quality judgment on top.
    # ------------------------------------------------------------------
    MCPServerConfig(
        name        = "lyrics-validator",
        command     = PYTHON_BIN,
        args        = [str(MCP_SERVERS_DIR / "lyrics-validator" / "server.py")],
        tool_names  = [
            "count_syllables",       # 统计汉字音节数（逐行分解）
            "check_tone_accuracy",   # 0243 旋律贴合度：1-6 声调映射到 lean 0243 + 强拍检查
            "check_rhyme_scheme",    # 押韵一致性：提取韵尾、多数原则评分
            "score_lyrics",          # 综合评分：运行全部三项检查 + 生成修改建议
            "suggest_corrections",   # 将评分报告转换为优先级排序的中文修改指令
        ],
        description = "歌词计算验证 – 音节计数、lean 0243 旋律贴合度、押韵分析、综合评分（纯计算，无需 LLM）",
    ),

    # ------------------------------------------------------------------
    # 4. Melody mapper (0243.hk lean mode)
    #
    # Maps MIDI melody contours to the 0243.hk lean-mode tone system.
    # This is the KEY tool for accurate melody-to-tone mapping.
    #
    # 0243 Tone System (Lean Mode):
    #   0 = 阴平 (高平/微降 55/53) – stable high
    #   2 = 阴上 (高升 35)         – rising
    #   4 = 阴去 (中平 33)         – neutral mid
    #   3 = 阳平 (低降/低平 21/11) – low falling
    #
    # Tools:
    #   - analyze_melody_contour   : Full melody → 0243 tone sequence
    #   - get_tone_requirements    : Tone code for specific position
    #   - suggest_tone_sequence    : Simplified tone sequence string
    #   - find_words_by_melody     : Find words matching melody tone
    #   - find_phrase_words        : Find multi-syllable phrase matches
    # ------------------------------------------------------------------
    MCPServerConfig(
        name        = "melody-mapper",
        command     = PYTHON_BIN,
        args        = [str(MCP_SERVERS_DIR / "melody-mapper" / "server.py")],
        tool_names  = [
            "analyze_melody_contour",   # 完整旋律轮廓分析 → 0243 声调序列
            "get_tone_requirements",    # 获取特定位置的声调需求
            "suggest_tone_sequence",    # 简化版声调序列字符串
            "find_words_by_melody",     # 根据旋律声调找词
            "find_phrase_words",        # 多音节短语匹配
        ],
        description = "旋律→0243 声调映射器 – 将 MIDI 旋律线映射到 0243.hk 精简声调系统",
    ),

    # ------------------------------------------------------------------
    # Template: add more MCP servers below
    # ------------------------------------------------------------------
    # MCPServerConfig(
    #     name        = "my-new-server",
    #     command     = "python",
    #     args        = [str(MCP_SERVERS_DIR / "my-new-server" / "server.py")],
    #     tool_names  = ["tool_a", "tool_b"],
    #     description = "新服务的描述",
    #     enabled     = False,   # 准备好后改为 True
    # ),

]

# Convenience lookup: name → config
MCP_SERVER_MAP: dict[str, MCPServerConfig] = {s.name: s for s in MCP_SERVERS}

# ---------------------------------------------------------------------------
# Agent pipeline definitions
#
# Lists the agents that make up the multi-agent workflow in execution order.
#
# Fields
# ------
# name                  – unique identifier; must match a registered class
# description           – shown in logs
# allowed_mcp_servers   – names of MCP servers this agent may call.
#                         Empty list = all enabled servers.
# prompt_file           – path to the agent's Chinese system-prompt file.
#                         If None, auto-resolved to prompts/<name>.md
# enabled               – set False to skip without deleting the entry
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    name:                  str
    description:           str
    allowed_mcp_servers:   list[str]    = field(default_factory=list)
    prompt_file:           Path | None  = None
    enabled:               bool         = True

    def __post_init__(self) -> None:
        if self.prompt_file is None:
            candidate = PROMPTS_DIR / f"{self.name}.md"
            if candidate.exists():
                self.prompt_file = candidate


# ---------------------------------------------------------------------------
# Agent pipeline definitions
#
# Architecture note
# -----------------
# The previous 4-agent design (midi-analyser, jyutping-mapper, lyrics-composer,
# validator) has been simplified to 2 LLM agents.
#
# Why?
# ----
# The spec requires tool codes to be MCP-testable.  The computational skills
# that previously lived inside MidiAnalyserAgent and JyutpingMapperAgent
# (MIDI parsing, Jyutping lookup, tone/rhyme scoring) have been moved into
# MCP servers where they are independently testable by MCP Inspector.
#
# The two remaining agents are the parts that genuinely require LLM reasoning:
#   1. LyricsComposerAgent  – creative generation using all MCP tools
#   2. ValidatorAgent       – 在计算评分之上补充艺术质量判断与修改建议
#
# The orchestrator calls the midi-analyzer and jyutping MCP tools directly
# (as pipeline setup steps) before handing control to the LLM agents.
# ---------------------------------------------------------------------------

AGENTS: list[AgentConfig] = [

    # ------------------------------------------------------------------
    # 1. 歌词创作代理  (LLM agent)
    #
    # Has access to ALL three MCP servers:
    #   - midi-analyzer   → read MIDI analysis results
    #   - jyutping        → candidate lookup, Jyutping conversion, continuation
    #   - lyrics-validator → self-check before submitting draft
    # ------------------------------------------------------------------
    AgentConfig(
        name        = "lyrics-composer",
        description = (
            "粤语歌词再创作代理（LLM 驱动）。"
            "主用途是将外语歌或现有歌词改编成可唱的粤语版本；"
            "若没有原歌词，也可根据主题或情景原创填词。"
        ),
        allowed_mcp_servers = ["midi-analyzer", "jyutping", "lyrics-validator"],
        prompt_file = PROMPTS_DIR / "lyrics-composer.md",
    ),

    # ------------------------------------------------------------------
    # 2. 验收代理  (LLM agent)
    #
    # Uses lyrics-validator for all computational checks, then adds
    # an LLM-based artistic-quality judgment on top.
    # ------------------------------------------------------------------
    AgentConfig(
        name        = "validator",
        description = (
            "歌词验收代理（LLM 驱动）。"
            "调用 lyrics-validator MCP 工具完成可唱性检查（音节数、lean 0243 旋律贴合度、押韵），"
            "再由 LLM 评判艺术质量，给出是否接受当前改编结果及修改建议。"
        ),
        allowed_mcp_servers = ["jyutping", "lyrics-validator"],
        prompt_file = PROMPTS_DIR / "validator.md",
    ),

    # ------------------------------------------------------------------
    # 3. 选字代理  (LLM agent)
    #
    # 从 0243.hk API 返回的大量候选词中选择最合适的词语。
    # 根据上下文、语义、韵律等因素进行智能选择。
    # ------------------------------------------------------------------
    AgentConfig(
        name        = "word-selector",
        description = (
            "选字代理（LLM 驱动）。"
            "从 0243.hk API 返回的候选词列表中选择最合适的词语，"
            "考虑语义连贯性、声调匹配、韵律协调、主题一致性等因素。"
        ),
        allowed_mcp_servers = ["jyutping"],
        prompt_file = PROMPTS_DIR / "word-selector-task.md",
    ),

    # ------------------------------------------------------------------
    # Template: add more agents below
    # ------------------------------------------------------------------
    # AgentConfig(
    #     name        = "my-new-agent",
    #     description = "新代理的描述",
    #     allowed_mcp_servers = ["jyutping", "lyrics-validator"],
    #     enabled     = False,
    # ),

]

# Convenience lookup: name → config
AGENT_MAP: dict[str, AgentConfig] = {a.name: a for a in AGENTS}

# ---------------------------------------------------------------------------
# Workflow parameters
# ---------------------------------------------------------------------------

WORKFLOW_CONFIG: dict[str, Any] = {
    # 创作→校验 最大重试轮数
    "max_revision_loops": int(os.getenv("MAX_REVISION_LOOPS", "3")),
    # 最低验收质量分（0–1）
    "min_quality_score":  float(os.getenv("MIN_QUALITY_SCORE", "0.75")),
    # 是否将 LLM token 流式输出到 stdout
    "stream_output":      os.getenv("STREAM_OUTPUT", "false").lower() == "true",
}
