"""
AgentOrchestrator – the central coordinator of the multi-agent workflow.

Responsibilities
----------------
1. Start all enabled MCP servers via langchain-mcp-adapters MultiServerMCPClient.
2. Load ALL tools with get_tools() (the only supported call), then filter each
   agent's tool-set by the tool_names declared in its MCPServerConfig.
3. Build the LLM from whichever provider is selected (Ollama or LM Studio).
4. Instantiate every agent defined in config.AGENTS with the correct LLM,
   memory, and filtered tool-set.
5. Run the pipeline: MIDI Analyser → Jyutping Mapper → Lyrics Composer →
   Validator (with a configurable revision loop).

Provider quick-switch
---------------------
    # use Ollama  (default)
    $env:LLM_PROVIDER = "ollama"

    # use LM Studio
    $env:LLM_PROVIDER = "lmstudio"

Usage
-----
    import asyncio
    from agent.orchestrator import AgentOrchestrator

    async def main():
        async with AgentOrchestrator() as orch:
            result = await orch.run(
                midi_path="song.mid",
                reference_text="青山依舊在，幾度夕陽紅",
            )
            print(result.lyrics)

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import (
    AGENTS,
    LLM_CONFIG,
    MEMORY_CONFIG,
    PROMPTS_DIR,
    PROVIDER,
    WORKFLOW_CONFIG,
)
from .mcp_utils import normalize_mcp_result
from .memory import ShortTermMemory
from .registry import AGENT_REGISTRY, MCP_REGISTRY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Holds every artefact produced by a single orchestrator run."""

    lyrics: str = ""

    midi_analysis:     dict[str, Any] = field(default_factory=dict)
    jyutping_map:      dict[str, Any] = field(default_factory=dict)
    draft_history:     list[str]      = field(default_factory=list)
    validator_scores:  list[float]    = field(default_factory=list)
    validator_feedback: list[str]     = field(default_factory=list)

    revision_count:   int   = 0
    accepted:         bool  = False
    elapsed_seconds:  float = 0.0
    session_id:       str   = ""
    error:            str | None = None

    def __str__(self) -> str:  # pragma: no cover
        status = "✓ 已接受" if self.accepted else "✗ 未通过验收"
        score  = f"{self.validator_scores[-1]:.2f}" if self.validator_scores else "N/A"
        return (
            f"PipelineResult [{status}] "
            f"revisions={self.revision_count} score={score} "
            f"elapsed={self.elapsed_seconds:.1f}s\n"
            f"--- 歌词 ---\n{self.lyrics}"
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class AgentOrchestrator:
    """
    Wires MCP servers, agents, and shared memory into a runnable pipeline.

    Lifecycle
    ---------
    Use as an async context manager so MCP server subprocesses are started
    and cleanly terminated:

        async with AgentOrchestrator() as orch:
            result = await orch.run(midi_path, reference_text)

    Alternatively call ``await orch.start()`` / ``await orch.stop()``
    explicitly for long-running server processes.
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        memory: ShortTermMemory | None = None,
        *,
        session_id: str | None = None,
    ) -> None:
        self._llm:    BaseChatModel   = llm    or self._build_llm()
        self._memory: ShortTermMemory = memory or self._build_memory(session_id)

        self._mcp_client: MultiServerMCPClient | None = None
        self._agents:     dict[str, Any]              = {}
        self._started:    bool                        = False

        self._max_revision_loops: int   = WORKFLOW_CONFIG["max_revision_loops"]
        self._min_quality_score:  float = WORKFLOW_CONFIG["min_quality_score"]

        logger.info(
            "AgentOrchestrator 已初始化  provider=%s  session=%s",
            PROVIDER,
            self._memory.session_id,
        )

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "AgentOrchestrator":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Startup / teardown
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect MCP servers and instantiate all pipeline agents."""
        if self._started:
            logger.warning("AgentOrchestrator.start() 重复调用，已忽略")
            return

        logger.info("=== AgentOrchestrator: 启动中 ===")

        MCP_REGISTRY.build_from_config()
        await self._connect_mcp_servers()
        self._instantiate_agents()

        self._started = True
        logger.info("=== AgentOrchestrator: 就绪 ===")

    async def stop(self) -> None:
        """Gracefully stop all MCP server subprocesses."""
        logger.info("=== AgentOrchestrator: 关闭中 ===")

        self._mcp_client = None

        MCP_REGISTRY.mark_disconnected()
        AGENT_REGISTRY.clear_instances()
        self._agents.clear()
        self._started = False
        logger.info("=== AgentOrchestrator: 已停止 ===")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        midi_path: str | Path,
        reference_text: str,
    ) -> PipelineResult:
        """
        执行完整的歌词生成流水线。

        架构说明
        --------
        原来的四代理流水线已简化为两步：

        Step 1 — 直接调用 MCP 工具完成计算性准备工作（无需 LLM）：
            a. analyze_midi          → 提取音节数、速度、调性、强拍位置
            b. get_syllable_durations → 获取每音符时值
            c. suggest_rhyme_positions → 获取押韵位置建议
            d. analyze_melody_contour → 从旋律线推导 0243 声调序列（核心！）
            e. chinese_to_jyutping   → 获取参考文本粤拼候选
            f. get_tone_pattern      → 获取参考文本声调序列
            g. get_tone_code         → 获取参考文本数字声调码
            h. find_words_by_tone_code（强拍位置预查询）

        Step 2 — 两个 LLM 代理循环（创作 → 校验）：
            LyricsComposerAgent  调用工具创作歌词
            ValidatorAgent       调用 lyrics-validator 工具打分 + LLM 评判艺术质量

        关键改进
        --------
        新增的 melody-mapper 服务使用 0243.hk lean mode 声调系统：
          - 0 = 阴平 (高平/微降 55/53)
          - 2 = 阴上 (高升 35)
          - 4 = 阴去 (中平 33)
          - 3 = 阳平 (低降/低平 21/11)

        通过分析 MIDI 旋律的音高、轮廓、节奏，准确推导每个音节位置
        所需的声调码，再调用 0243.hk API 查找匹配的中文词语。

        Parameters
        ----------
        midi_path      : MIDI 文件路径
        reference_text : 参考文本 / 创作意境（中文）

        Returns
        -------
        PipelineResult  包含最终歌词及所有中间产物。
        """
        if not self._started:
            raise RuntimeError(
                "请先调用 start() 或使用 'async with AgentOrchestrator() as orch'"
            )

        t0     = time.perf_counter()
        result = PipelineResult(session_id=self._memory.session_id)

        logger.info("流水线开始 – midi=%s | text='%s'", midi_path, reference_text[:60])

        self._memory.add_user_message(
            f"请根据 MIDI 文件「{midi_path}」和以下参考文本，创作粤语歌词：\n\n{reference_text}"
        )
        self._memory.set_artifact("source_text", reference_text)
        self._memory.set_run_status(
            stage="starting",
            accepted=False,
            revision_count=0,
            error=None,
        )

        try:
            # ----------------------------------------------------------
            # Step 1a: 直接调用 MCP 工具完成 MIDI 分析（无 LLM）
            # ----------------------------------------------------------
            logger.info("Step 1a: 并行调用 MCP 工具分析 MIDI 文件…")

            # 并行调用 MIDI analyzer 的三个工具
            midi_task = self._call_tool_direct(
                server_name="midi-analyzer",
                tool_name="analyze_midi",
                args={"file_path": str(midi_path)},
                parse_json=True,
            )
            durations_task = self._call_tool_direct(
                server_name="midi-analyzer",
                tool_name="get_syllable_durations",
                args={"file_path": str(midi_path)},
                parse_json=True,
            )
            rhyme_task = self._call_tool_direct(
                server_name="midi-analyzer",
                tool_name="suggest_rhyme_positions",
                args={"file_path": str(midi_path)},
                parse_json=True,
            )

            # 等待所有任务完成
            midi_analysis, durations, rhyme_positions_raw = await asyncio.gather(
                midi_task, durations_task, rhyme_task,
                return_exceptions=True,
            )

            # 处理结果
            result.midi_analysis = midi_analysis if isinstance(midi_analysis, dict) else {}
            result.midi_analysis["syllable_durations"] = (
                durations if isinstance(durations, list) else []
            )
            rhyme_positions: list[int] = (
                rhyme_positions_raw if isinstance(rhyme_positions_raw, list) else []
            )
            result.midi_analysis["rhyme_positions"] = rhyme_positions

            syllable_count: int = int(
                result.midi_analysis.get("effective_syllable_count", 0)
                or result.midi_analysis.get("syllable_count", 0)
            )
            strong_beats: list[int] = result.midi_analysis.get("strong_beat_positions", [])

            self._memory.set_pipeline_value("midi_analysis", result.midi_analysis)
            logger.info(
                "Step 1a 完成: syllable_count=%d bpm=%s key=%s",
                syllable_count,
                result.midi_analysis.get("bpm", "?"),
                result.midi_analysis.get("key", "?"),
            )

            # ----------------------------------------------------------
            # Step 1b: 并行调用 MCP 工具完成 0243 旋律映射（无 LLM）
            # ----------------------------------------------------------
            logger.info("Step 1b: 并行调用 MCP 工具映射 0243 旋律声调…")

            # 并行调用 melody-mapper 和 jyutping 工具
            melody_task = self._call_tool_direct(
                server_name="melody-mapper",
                tool_name="analyze_melody_contour",
                args={"file_path": str(midi_path)},
                parse_json=True,
            )
            jp_task = self._call_tool_direct(
                server_name="jyutping",
                tool_name="chinese_to_jyutping",
                args={"text": reference_text},
                parse_json=True,
            )
            tone_pattern_task = self._call_tool_direct(
                server_name="jyutping",
                tool_name="get_tone_pattern",
                args={"text": reference_text},
                parse_json=False,
            )
            tone_codes_task = self._call_tool_direct(
                server_name="jyutping",
                tool_name="get_tone_code",
                args={"text": reference_text},
                parse_json=True,
            )

            # 等待所有任务完成
            melody_analysis_raw, jp_candidates_raw, tone_pattern_raw, tone_codes_raw = await asyncio.gather(
                melody_task, jp_task, tone_pattern_task, tone_codes_task,
                return_exceptions=True,
            )

            # 处理结果
            melody_analysis: dict[str, Any] = (
                melody_analysis_raw if isinstance(melody_analysis_raw, dict) else {}
            )
            melody_tone_sequence: list[int] = [
                int(t) for t in melody_analysis.get("tone_sequence", [])
                if isinstance(t, int | float | str) and str(t).isdigit()
            ]
            jp_candidates: list[str] = (
                jp_candidates_raw if isinstance(jp_candidates_raw, list) else []
            )
            tone_pattern_raw = tone_pattern_raw or ""
            tone_codes: list[str] = (
                tone_codes_raw if isinstance(tone_codes_raw, list) else []
            )

            result.midi_analysis["melody_0243"] = melody_analysis
            self._memory.set_pipeline_value("melody_analysis", melody_analysis)

            # Parse 1-6 Jyutping tone sequence from the space-separated pattern string
            reference_tone_sequence: list[int] = []
            if isinstance(tone_pattern_raw, list):
                tone_pattern_tokens = [str(tok) for tok in tone_pattern_raw]
            else:
                tone_pattern_tokens = str(tone_pattern_raw).split()
            tone_pattern = " ".join(tone_pattern_tokens)

            if tone_pattern_tokens:
                for tok in tone_pattern_tokens:
                    try:
                        reference_tone_sequence.append(int(tok))
                    except ValueError:
                        pass

            # Pre-query tone-constrained word candidates using BATCH API call
            # Collect all unique tone codes needed for strong beats and rhyme positions
            logger.info("Step 1c: 批量查询声调码候选词...")

            # Gather all positions that need candidate words
            positions_needing_candidates = set()
            positions_needing_candidates.update(str(p) for p in strong_beats[:16])  # First 16 strong beats
            positions_needing_candidates.update(str(p) for p in rhyme_positions[:8])  # First 8 rhyme positions

            # Map position -> tone code
            position_tone_map: dict[str, str] = {}
            for pos_str in positions_needing_candidates:
                pos = int(pos_str)
                beat_tone = (
                    str(melody_tone_sequence[pos])
                    if pos < len(melody_tone_sequence) else "4"
                )
                position_tone_map[pos_str] = beat_tone

            # Get unique tone codes and batch query them
            unique_tone_codes = list(set(position_tone_map.values()))
            if unique_tone_codes:
                # BATCH API call - query all tone codes at once
                batch_candidates_raw = await self._call_tool_direct(
                    server_name="jyutping",
                    tool_name="find_words_by_tone_code",
                    args={"code": unique_tone_codes},  # Batch call!
                    parse_json=True,
                )

                # Parse batch results: returns list of lists [[words_for_code1], [words_for_code2], ...]
                batch_candidates: list[list[str]] = (
                    batch_candidates_raw if isinstance(batch_candidates_raw, list) and
                    all(isinstance(item, list) for item in batch_candidates_raw)
                    else []
                )

                # Build tone_code -> candidates map
                tone_to_candidates: dict[str, list[str]] = {}
                for i, tone_code in enumerate(unique_tone_codes):
                    if i < len(batch_candidates):
                        tone_to_candidates[tone_code] = batch_candidates[i][:15]  # Keep top 15 per tone

                # Build position -> candidates map
                strong_beat_candidates: dict[str, list[str]] = {}
                for pos_str, tone_code in position_tone_map.items():
                    strong_beat_candidates[pos_str] = tone_to_candidates.get(tone_code, [])
            else:
                strong_beat_candidates = {}

            # ----------------------------------------------------------
            # Step 1d: Pre-query theme-related common words
            # ----------------------------------------------------------
            logger.info("Step 1d: 查询主题相关常用词...")

            # Extract theme keywords from reference text
            theme_tone_codes = await self._extract_theme_tone_codes(reference_text)

            theme_candidates: dict[str, list[str]] = {}
            if theme_tone_codes:
                # Batch query theme-related tone codes
                theme_candidates_raw = await self._call_tool_direct(
                    server_name="jyutping",
                    tool_name="find_words_by_tone_code",
                    args={"code": theme_tone_codes},
                    parse_json=True,
                )

                theme_candidates_list: list[list[str]] = (
                    theme_candidates_raw if isinstance(theme_candidates_raw, list) and
                    all(isinstance(item, list) for item in theme_candidates_raw)
                    else []
                )

                # Map tone code -> candidates
                for i, tone_code in enumerate(theme_tone_codes):
                    if i < len(theme_candidates_list):
                        theme_candidates[tone_code] = theme_candidates_list[i][:20]

                logger.info("主题相关声调码：%s → 候选词 %d 个", theme_tone_codes, sum(len(c) for c in theme_candidates.values()))

            jyutping_map: dict = {
                "reference_text":             reference_text,
                "selected_jyutping":          jp_candidates[0] if jp_candidates else "",
                "all_candidates":             jp_candidates,
                "reference_tone_pattern":     tone_pattern,
                "reference_tone_sequence":    reference_tone_sequence,
                "tone_codes":                 tone_codes,
                "melody_tone_sequence_0243":  melody_tone_sequence,
                "rhyme_positions":            rhyme_positions,
                "strong_beat_positions":      strong_beats,
                "strong_beat_candidates":     strong_beat_candidates,
                "theme_candidates":           theme_candidates,  # Theme-related words
                "target_syllable_count":      syllable_count,
            }
            result.jyutping_map = jyutping_map
            self._memory.set_pipeline_value("jyutping_map", jyutping_map)

            logger.info(
                "Step 1c 完成: jp_candidates=%d melody_0243=%s reference_1_6=%s",
                len(jp_candidates),
                " ".join(str(t) for t in melody_tone_sequence[:16]) or "(none)",
                tone_pattern[:40] if tone_pattern else "(none)",
            )

            # ----------------------------------------------------------
            # Step 2: LLM 代理循环（创作 → 校验）
            # ----------------------------------------------------------
            revision_instructions: str = ""

            for attempt in range(self._max_revision_loops + 1):
                logger.info(
                    "创作尝试 %d/%d", attempt + 1, self._max_revision_loops + 1
                )

                # --- 歌词创作（LLM 代理）---
                compose_task = self._build_compose_task(
                    reference_text=reference_text,
                    syllable_count=syllable_count,
                    revision_instructions=revision_instructions,
                    attempt=attempt,
                )
                draft_output = await self._run_agent(
                    agent_name="lyrics-composer",
                    task=compose_task,
                    context_key="draft_lyrics",
                )
                draft_lyrics: str = (
                    draft_output.get("lyrics", "")
                    if isinstance(draft_output, dict)
                    else str(draft_output)
                )
                draft_jyutping: str = (
                    draft_output.get("jyutping", "")
                    if isinstance(draft_output, dict) else ""
                )
                result.draft_history.append(draft_lyrics)

                # --- 歌词校验（LLM 代理）---
                validate_task = self._build_validate_task(
                    draft_lyrics=draft_lyrics,
                    draft_jyutping=draft_jyutping,
                    syllable_count=syllable_count,
                    melody_tone_sequence=melody_tone_sequence,
                    strong_beats=strong_beats,
                    rhyme_positions=rhyme_positions,
                    reference_text=reference_text,
                )
                validation_output = await self._run_agent(
                    agent_name="validator",
                    task=validate_task,
                    context_key="validation_result",
                )

                score: float = float(
                    validation_output.get("score", 0.0)
                    if isinstance(validation_output, dict) else 0.0
                )
                feedback: str = (
                    validation_output.get("feedback", "")
                    if isinstance(validation_output, dict)
                    else str(validation_output)
                )
                result.validator_scores.append(score)
                result.validator_feedback.append(feedback)
                result.revision_count = attempt
                self._memory.update_attempt_state(
                    {
                        "attempt_index": attempt,
                        "last_score": score,
                        "last_feedback": feedback,
                    }
                )

                logger.info(
                    "校验第 %d 次：score=%.2f（阈值=%.2f）",
                    attempt + 1, score, self._min_quality_score,
                )

                if score >= self._min_quality_score:
                    result.lyrics   = draft_lyrics
                    result.accepted = True
                    self._memory.set_best_result(
                        draft=draft_output if isinstance(draft_output, dict) else {},
                        validation=validation_output if isinstance(validation_output, dict) else {},
                        score=score,
                    )
                    logger.info("✓ 歌词已在第 %d 次尝试时通过验收", attempt + 1)
                    break

                corrections: list[str] = (
                    validation_output.get("corrections", [])
                    if isinstance(validation_output, dict) else []
                )
                revision_instructions = (
                    f"上一草稿未通过校验（score={score:.2f}）。\n"
                    f"校验反馈：{feedback}\n"
                    "需修改的问题：\n"
                    + "\n".join(f"  - {c}" for c in corrections)
                )
                self._memory.set_attempt_value(
                    "revision_instructions", revision_instructions
                )
                self._memory.add_ai_message(
                    f"[第 {attempt + 1} 轮修改] 得分偏低（{score:.2f}），"
                    f"根据以下反馈修改：{feedback}"
                )

                best_result = self._memory.get_best_result()
                best_score = float(best_result.get("score", float("-inf")) or float("-inf"))
                if score > best_score:
                    self._memory.set_best_result(
                        draft=draft_output if isinstance(draft_output, dict) else {},
                        validation=validation_output if isinstance(validation_output, dict) else {},
                        score=score,
                    )

            # 所有轮次用尽时，选最高分草稿
            if not result.accepted and result.draft_history:
                best_idx = (
                    result.validator_scores.index(max(result.validator_scores))
                    if result.validator_scores else 0
                )
                result.lyrics = result.draft_history[best_idx]
                logger.warning(
                    "已用尽全部 %d 次尝试，选用第 %d 次草稿（score=%.2f）。",
                    self._max_revision_loops + 1,
                    best_idx + 1,
                    result.validator_scores[best_idx] if result.validator_scores else 0,
                )

        except Exception as exc:  # noqa: BLE001
            logger.exception("流水线错误: %s", exc)
            result.error = str(exc)
            self._memory.set_run_status(error=str(exc))
            self._memory.add_ai_message(f"[错误] 流水线失败：{exc}")

        result.elapsed_seconds = time.perf_counter() - t0
        logger.info(
            "流水线完成  elapsed=%.1fs  accepted=%s  revisions=%d",
            result.elapsed_seconds, result.accepted, result.revision_count,
        )

        self._memory.set_final_result(
            {
                "lyrics": result.lyrics,
                "accepted": result.accepted,
                "revision_count": result.revision_count,
                "elapsed_seconds": result.elapsed_seconds,
                "error": result.error,
            }
        )
        self._memory.set_run_status(
            stage="completed",
            accepted=result.accepted,
            revision_count=result.revision_count,
            error=result.error,
        )
        return result

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def memory(self) -> ShortTermMemory:
        return self._memory

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    def get_agent(self, name: str) -> Any:
        if name not in self._agents:
            raise KeyError(
                f"找不到代理 '{name}'，可用代理：{list(self._agents)}"
            )
        return self._agents[name]

    def list_agents(self) -> list[str]:
        return list(self._agents)

    def list_mcp_servers(self) -> list[str]:
        return [s.name for s in MCP_REGISTRY.enabled_servers]

    # ------------------------------------------------------------------
    # Internal: MCP connection
    #
    # NOTE: MultiServerMCPClient.get_tools() returns ALL tools from ALL
    # connected servers in a flat list.  There is no server_name filter.
    # We identify which server a tool belongs to by matching its name
    # against the tool_names declared in each MCPServerConfig.
    # ------------------------------------------------------------------

    async def _connect_mcp_servers(self) -> None:
        """Start all enabled MCP server subprocesses and load their tools."""
        server_params = MCP_REGISTRY.langchain_server_params()

        if not server_params:
            logger.warning("没有启用的 MCP 服务器，代理将无法使用外部工具。")
            return

        logger.info(
            "正在连接 %d 个 MCP 服务器：%s", len(server_params), list(server_params)
        )

        self._mcp_client = MultiServerMCPClient(server_params)

        # Load ALL tools (flat list – no per-server filter available)
        all_tools = await self._mcp_client.get_tools()

        logger.info(
            "MCP 共加载 %d 个工具：%s",
            len(all_tools),
            [t.name for t in all_tools],
        )

        # Register every tool in the global tool map
        for tool in all_tools:
            MCP_REGISTRY.register_tool(tool.name, tool)

        # Associate tools with servers using the declared tool_names list.
        # This avoids any undocumented API call and works reliably.
        for srv_config in MCP_REGISTRY.enabled_servers:
            declared   = set(srv_config.tool_names)
            found_names = [t.name for t in all_tools if t.name in declared]

            if not found_names and declared:
                # Warn if the server declared tools that weren't found
                loaded_names = {t.name for t in all_tools}
                missing = declared - loaded_names
                logger.warning(
                    "服务器 '%s' 声明了以下工具但未从 MCP 加载到：%s",
                    srv_config.name, missing,
                )

            MCP_REGISTRY.set_server_tools(srv_config.name, found_names)
            logger.debug(
                "服务器 '%s' 关联工具：%s", srv_config.name, found_names
            )

        MCP_REGISTRY.mark_connected()

    # ------------------------------------------------------------------
    # Internal: agent instantiation
    # ------------------------------------------------------------------

    def _instantiate_agents(self) -> None:
        """
        Instantiate the two LLM agents defined in config.AGENTS.

        MidiAnalyserAgent and JyutpingMapperAgent have been removed as
        standalone agent classes — their skills now live in MCP servers
        (midi-analyzer and jyutping) and are called directly by the
        orchestrator in Step 1 of run().

        For each remaining agent:
          1. Look up the registered class in AGENT_REGISTRY.
          2. Build the filtered tool list from allowed_mcp_servers.
          3. Construct with shared LLM and memory.
          4. Store in AGENT_REGISTRY and self._agents.
        """
        from .agents import (
            LyricsComposerAgent,
            ValidatorAgent,
        )

        _builtin: dict[str, type] = {
            "lyrics-composer": LyricsComposerAgent,
            "validator":       ValidatorAgent,
        }
        for agent_name, cls in _builtin.items():
            if not AGENT_REGISTRY.has_class(agent_name):
                AGENT_REGISTRY.register_class(agent_name, cls)

        for agent_cfg in AGENTS:
            if not agent_cfg.enabled:
                logger.debug("跳过已禁用的代理 '%s'", agent_cfg.name)
                continue

            if not AGENT_REGISTRY.has_class(agent_cfg.name):
                logger.warning(
                    "代理 '%s' 没有注册对应的类。请使用 "
                    "@AGENT_REGISTRY.register('%s') 或在 _builtin 字典中添加。",
                    agent_cfg.name, agent_cfg.name,
                )
                continue

            cls   = AGENT_REGISTRY.get_class(agent_cfg.name)
            tools = MCP_REGISTRY.get_tools_for_servers(agent_cfg.allowed_mcp_servers)

            instance = cls(
                config=agent_cfg,
                llm=self._llm,
                memory=self._memory,
                tools=tools,
            )

            AGENT_REGISTRY.add_instance(instance)
            self._agents[agent_cfg.name] = instance

            logger.info(
                "代理 '%s' 已初始化，绑定 %d 个工具：%s",
                agent_cfg.name, len(tools), [t.name for t in tools],
            )

    # ------------------------------------------------------------------
    # Internal: single-stage runner
    # ------------------------------------------------------------------

    async def _run_agent(
        self,
        agent_name: str,
        task: str,
        context_key: str,
    ) -> dict[str, Any]:
        """
        调用一个 LLM 代理执行任务，返回其结构化输出 dict。

        与旧版 _run_stage 的区别：
        - 仅用于真正的 LLM 代理（lyrics-composer、validator）
        - 计算性步骤已由 _call_tool_direct 直接处理
        """
        if agent_name not in self._agents:
            logger.error(
                "代理 '%s' 不存在。可用：%s", agent_name, list(self._agents)
            )
            return {}

        agent = self._agents[agent_name]
        logger.info(">>> LLM 代理：%s", agent_name)

        try:
            agent_result = await agent.run(task)
            data: dict[str, Any] = (
                agent_result.data if hasattr(agent_result, "data") else {}
            )
            payload = data.get(context_key, data)
            if context_key == "draft_lyrics" and isinstance(payload, dict):
                self._memory.set_current_draft(payload)
            elif context_key == "validation_result" and isinstance(payload, dict):
                self._memory.set_validation_result(payload)
            else:
                self._memory.set_pipeline_value(context_key, payload)
            logger.info("<<< LLM 代理 '%s' 完成", agent_name)
            return payload
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM 代理 '%s' 失败: %s", agent_name, exc)
            self._memory.add_ai_message(f"[{agent_name}] 错误：{exc}")
            return {"error": str(exc)}

    async def _call_tool_direct(
        self,
        server_name: str,
        tool_name: str,
        args: dict[str, Any],
        parse_json: bool = True,
    ) -> Any:
        """
        直接调用 MCP 工具，绕过 LLM 代理层。

        用于流水线 Step 1 中纯计算性的工具调用：
        MIDI 分析、粤拼查询、声调码预查询等。

        Parameters
        ----------
        server_name : str   MCP 服务器名称（用于日志）
        tool_name   : str   工具名称
        args        : dict  传给工具的参数
        parse_json  : bool  若为 True，尝试将字符串结果解析为 JSON

        Returns
        -------
        解析后的 Python 对象（dict / list / str），出错时返回 None。
        """
        tool = MCP_REGISTRY.get_all_tools()
        tool_obj = next((t for t in tool if t.name == tool_name), None)

        if tool_obj is None:
            logger.warning(
                "_call_tool_direct: 工具 '%s' 未在 MCP 注册表中找到（服务器：%s）",
                tool_name, server_name,
            )
            return None

        logger.debug("_call_tool_direct: %s.%s(%s)", server_name, tool_name, args)

        try:
            raw = await tool_obj.ainvoke(args)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "_call_tool_direct: %s.%s 调用失败: %s", server_name, tool_name, exc
            )
            return None

        return normalize_mcp_result(raw, parse_json=parse_json)

    # ------------------------------------------------------------------
    # Internal: compose-task builder
    # ------------------------------------------------------------------

    def _build_compose_task(
        self,
        reference_text: str,
        syllable_count: int,
        revision_instructions: str,
        attempt: int,
    ) -> str:
        midi_analysis = self._memory.get_pipeline_value("midi_analysis", {})
        jyutping_map  = self._memory.get_pipeline_value("jyutping_map",  {})

        # Summarise key constraints inline so the LLM doesn't need to re-fetch them
        strong_beats   = midi_analysis.get("strong_beat_positions", [])
        rhyme_positions = midi_analysis.get("rhyme_positions", [])
        melody_tone_sequence = jyutping_map.get("melody_tone_sequence_0243", [])
        reference_tone_sequence = jyutping_map.get("reference_tone_sequence", [])
        selected_jp    = jyutping_map.get("selected_jyutping", "")
        sbc            = jyutping_map.get("strong_beat_candidates", {})
        embedded_lyrics_source = midi_analysis.get("embedded_lyrics_source")
        embedded_lyrics_preview = midi_analysis.get("embedded_lyrics_preview", [])
        embedded_lyric_unit_count = midi_analysis.get("embedded_lyric_unit_count", 0)
        effective_syllable_count_source = midi_analysis.get(
            "effective_syllable_count_source", "melody_notes"
        )

        sbc_summary = "\n".join(
            f"  位置 {pos}: {', '.join(words[:5])}"
            for pos, words in list(sbc.items())[:6]
        ) or "  （无预查询候选）"

        # Build theme candidate summary
        theme_candidates = jyutping_map.get("theme_candidates", {})
        theme_summary = "\n".join(
            f"  声调{tone}: {', '.join(words[:8])}"
            for tone, words in list(theme_candidates.items())[:6]
        ) or "  （无主题候选词）"

        melody_tone_seq_str = (
            " ".join(str(t) for t in melody_tone_sequence[:syllable_count]) or "（未知）"
        )
        reference_tone_seq_str = (
            " ".join(str(t) for t in reference_tone_sequence[:syllable_count]) or "（未知）"
        )
        strong_str   = ", ".join(str(b) for b in strong_beats[:12]) or "（未知）"
        rhyme_str    = ", ".join(str(r) for r in rhyme_positions[:8]) or "（未知）"
        embedded_lyrics_str = (
            " ".join(str(unit) for unit in embedded_lyrics_preview[:32])
            if embedded_lyrics_preview else "（无）"
        )

        base = self._render_prompt_template(
            "compose-task.md",
            syllable_count=syllable_count,
            reference_text=reference_text,
            embedded_lyrics_source=embedded_lyrics_source or "（无）",
            embedded_lyric_unit_count=embedded_lyric_unit_count,
            effective_syllable_count_source=effective_syllable_count_source,
            embedded_lyrics_str=embedded_lyrics_str,
            bpm=midi_analysis.get("bpm", "?"),
            key=midi_analysis.get("key", "?"),
            strong_str=strong_str,
            rhyme_str=rhyme_str,
            melody_tone_seq_str=melody_tone_seq_str,
            selected_jp=selected_jp or "（未查询到）",
            reference_tone_seq_str=reference_tone_seq_str,
            sbc_summary=sbc_summary,
            theme_summary=theme_summary,
        )

        if attempt > 0 and revision_instructions:
            base = f"[第 {attempt + 1} 次修改]\n{revision_instructions}\n\n{base}"

        return base

    def _build_validate_task(
        self,
        draft_lyrics: str,
        draft_jyutping: str,
        syllable_count: int,
        melody_tone_sequence: list[int],
        strong_beats: list[int],
        rhyme_positions: list[int],
        reference_text: str,
    ) -> str:
        """构建校验代理的任务提示，包含所有必要参数供其调用 MCP 工具。"""
        midi_analysis = self._memory.get_pipeline_value("midi_analysis", {})
        tone_seq_str   = " ".join(str(t) for t in melody_tone_sequence) or "（无）"
        strong_str     = ", ".join(str(b) for b in strong_beats) or "（无）"
        rhyme_str      = ", ".join(str(r) for r in rhyme_positions) or "（无）"
        strong_json    = str(strong_beats)
        rhyme_json     = str(rhyme_positions)
        tone_json      = str(melody_tone_sequence)
        embedded_lyrics_source = midi_analysis.get("embedded_lyrics_source")
        embedded_lyrics_preview = midi_analysis.get("embedded_lyrics_preview", [])
        embedded_lyric_unit_count = midi_analysis.get("embedded_lyric_unit_count", 0)
        effective_syllable_count_source = midi_analysis.get(
            "effective_syllable_count_source", "melody_notes"
        )
        embedded_lyrics_str = (
            " ".join(str(unit) for unit in embedded_lyrics_preview[:24])
            if embedded_lyrics_preview else "（无）"
        )

        return self._render_prompt_template(
            "validate-task.md",
            draft_lyrics=draft_lyrics,
            draft_jyutping=draft_jyutping or "（创作代理未提供，请调用 chinese_to_jyutping 获取）",
            syllable_count=syllable_count,
            tone_seq_str=tone_seq_str,
            strong_str=strong_str,
            rhyme_str=rhyme_str,
            reference_text=reference_text,
            embedded_lyrics_source=embedded_lyrics_source or "（无）",
            embedded_lyric_unit_count=embedded_lyric_unit_count,
            effective_syllable_count_source=effective_syllable_count_source,
            embedded_lyrics_str=embedded_lyrics_str,
            tone_json=tone_json,
            strong_json=strong_json,
            rhyme_json=rhyme_json,
        )

    @staticmethod
    def _render_prompt_template(template_name: str, **kwargs: Any) -> str:
        template_path = PROMPTS_DIR / template_name
        return template_path.read_text(encoding="utf-8").format(**kwargs).strip()

    # ------------------------------------------------------------------
    # Internal: LLM builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_llm() -> BaseChatModel:
        """
        根据 PROVIDER 配置构建对应的 LLM 实例。

        Supported providers
        -------------------
        ollama   – ChatOllama (langchain-ollama)
        lmstudio – ChatOpenAI with custom base_url (langchain-openai)
                   LM Studio exposes an OpenAI-compatible server on port 1234.
        """
        if PROVIDER == "lmstudio":
            # LM Studio: OpenAI-compatible API, no real key needed
            from langchain_openai import ChatOpenAI

            cfg = LLM_CONFIG
            logger.info(
                "使用 LM Studio 提供商  model=%s  base_url=%s",
                cfg["model"], cfg["base_url"],
            )
            return ChatOpenAI(
                model=cfg["model"],
                base_url=cfg["base_url"],
                api_key=cfg["api_key"],
                temperature=cfg["temperature"],
                max_tokens=cfg.get("max_tokens", 8192),
                # Disable reasoning/thinking to reduce latency
                extra_body={"thinking": False},
            )
        else:
            # Default: Ollama
            from langchain_ollama import ChatOllama

            cfg = LLM_CONFIG
            logger.info(
                "使用 Ollama 提供商  model=%s  base_url=%s",
                cfg["model"], cfg["base_url"],
            )
            return ChatOllama(
                model=cfg["model"],
                base_url=cfg["base_url"],
                temperature=cfg["temperature"],
                num_ctx=cfg.get("num_ctx", 8192),
                # Disable reasoning/thinking to reduce latency
                num_predict=cfg.get("max_tokens", 8192),
            )

    # ------------------------------------------------------------------
    # Internal: theme keyword extractor
    # ------------------------------------------------------------------

    async def _extract_theme_tone_codes(self, reference_text: str) -> list[str]:
        """
        从参考文本中提取主题相关的声调码。

        策略：
        1. 提取参考文本中的 2-4 字词语作为关键词
        2. 查询每个关键词的声调码
        3. 返回去重后的声调码列表

        Parameters
        ----------
        reference_text : str
            参考文本/主题描述

        Returns
        -------
        list[str]
            主题相关的声调码列表（去重）
        """
        import re

        # 简单策略：提取 2-4 字词语作为关键词
        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,4}', reference_text)

        # 去重并限制数量
        unique_words = list(set(chinese_words))[:10]

        if not unique_words:
            return []

        logger.debug("提取主题关键词：%s", unique_words)

        # 并行查询每个关键词的声调码
        tone_code_tasks = [
            self._call_tool_direct(
                server_name="jyutping",
                tool_name="get_tone_code",
                args={"text": word},
                parse_json=True,
            )
            for word in unique_words
        ]

        tone_code_results = await asyncio.gather(*tone_code_tasks, return_exceptions=True)

        # 收集所有声调码
        all_tone_codes: set[str] = set()
        for result in tone_code_results:
            if isinstance(result, list):
                all_tone_codes.update(str(code) for code in result if code)

        # 限制声调码数量（避免过多 API 调用）
        unique_tone_codes = list(all_tone_codes)[:8]

        if unique_tone_codes:
            logger.info("主题关键词声调码：%s", unique_tone_codes)

        return unique_tone_codes

    @staticmethod
    def _build_memory(session_id: str | None = None) -> ShortTermMemory:
        """从 MEMORY_CONFIG 构建共享短期记忆实例。"""
        prompt_file: Path = MEMORY_CONFIG["system_prompt_file"]
        if prompt_file.exists():
            system_prompt = prompt_file.read_text(encoding="utf-8").strip()
        else:
            # 默认中文系统提示（当 prompts/system.md 不存在时使用）
            system_prompt = (
                "你是一位专业的粤语作词人和语言学家。"
                "你的任务是根据 MIDI 旋律生成符合粤语声调规律、押韵优美的粤语歌词。"
                "请始终以中文思考和回答，输出结构化 JSON。"
            )
            logger.warning(
                "系统提示文件不存在：%s，使用默认提示。", prompt_file
            )

        return ShortTermMemory(
            max_turns=MEMORY_CONFIG["max_turns"],
            system_prompt=system_prompt,
            session_id=session_id,
        )
