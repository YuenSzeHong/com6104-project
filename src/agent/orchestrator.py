"""
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
                reference_text_kind="theme",
            )
            print(result.lyrics)

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from langchain_core.language_models import BaseChatModel
from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import (
    AGENTS,
    LLM_CONFIG,
    MEMORY_CONFIG,
    PROMPTS_DIR,
    PROMPTS_AGENTS_DIR,
    PROVIDER,
    WORKFLOW_CONFIG,
)
from .errors import ConstraintViolation, ParseError, ToolInvokeError
from .workflow_graph import build_workflow_graph, decide_after_validation
from .utils.constraint_filter import CandidateConstraintEngine
from .utils.mcp import normalize_mcp_result
from .memory import ShortTermMemory
from .registry import AGENT_REGISTRY, MCP_REGISTRY

logger = logging.getLogger(__name__)

PipelineEventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]

_STATE_STARTING = "starting"
_STATE_MIDI_ANALYSIS = "midi_analysis"
_STATE_MELODY_MAPPING = "melody_mapping"
_STATE_COMPOSITION = "composition"
_STATE_VALIDATION = "validation"
_STATE_COMPLETED = "completed"
_STATE_ERROR = "error"

_ALLOWED_STATE_TRANSITIONS: dict[str, set[str]] = {
    _STATE_STARTING: {_STATE_MIDI_ANALYSIS, _STATE_ERROR},
    _STATE_MIDI_ANALYSIS: {_STATE_MELODY_MAPPING, _STATE_ERROR},
    _STATE_MELODY_MAPPING: {_STATE_COMPOSITION, _STATE_ERROR},
    _STATE_COMPOSITION: {_STATE_VALIDATION, _STATE_ERROR},
    _STATE_VALIDATION: {_STATE_COMPOSITION, _STATE_COMPLETED, _STATE_ERROR},
    _STATE_COMPLETED: set(),
    _STATE_ERROR: set(),
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Holds every artefact produced by a single orchestrator run."""

    lyrics: str = ""

    midi_analysis: dict[str, Any] = field(default_factory=dict)
    jyutping_map: dict[str, Any] = field(default_factory=dict)
    draft_history: list[str] = field(default_factory=list)
    validator_scores: list[float] = field(default_factory=list)
    validator_feedback: list[str] = field(default_factory=list)

    revision_count: int = 0
    accepted: bool = False
    elapsed_seconds: float = 0.0
    session_id: str = ""
    error: str | None = None

    def __str__(self) -> str:  # pragma: no cover
        status = "✓ 已接受" if self.accepted else "✗ 未通过验收"
        score = f"{self.validator_scores[-1]:.2f}" if self.validator_scores else "N/A"
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
        self._llm: BaseChatModel = llm or self._build_llm()
        self._memory: ShortTermMemory = memory or self._build_memory(session_id)

        self._mcp_client: MultiServerMCPClient | None = None
        self._agents: dict[str, Any] = {}
        self._started: bool = False

        # Concurrency control: limits parallel LLM requests to avoid GPU thrashing
        # Based on stress test: concurrency > 1 causes significant degradation (34%)
        self._llm_semaphore: asyncio.Semaphore = asyncio.Semaphore(
            int(os.getenv("LLM_MAX_CONCURRENT_REQUESTS", "1"))
        )

        self._max_revision_loops: int = WORKFLOW_CONFIG["max_revision_loops"]
        self._min_quality_score: float = WORKFLOW_CONFIG["min_quality_score"]
        self._word_selector_threshold: int = int(
            os.getenv("WORD_SELECTOR_THRESHOLD", "10")
        )
        self._phrase_selector_max_len: int = int(
            os.getenv("WORD_SELECTOR_PHRASE_MAX_LEN", "3")
        )
        self._word_selector_fast_mode: str = (
            os.getenv("WORD_SELECTOR_FAST_MODE", "auto").strip().lower()
        )
        self._word_selector_max_llm_calls: int = max(
            1,
            int(os.getenv("WORD_SELECTOR_MAX_LLM_CALLS", "3")),
        )
        self._word_selector_max_targets: int = max(
            1,
            int(os.getenv("WORD_SELECTOR_MAX_TARGETS", "8")),
        )
        self._word_selector_call_timeout_s: float = max(
            1.0,
            float(os.getenv("WORD_SELECTOR_CALL_TIMEOUT_S", "12")),
        )
        self._pipeline_state: str = _STATE_STARTING
        self._workflow_graph = build_workflow_graph()
        self._candidate_constraint_engine = CandidateConstraintEngine()

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
        reference_text_kind: str = "theme",
        event_callback: PipelineEventCallback | None = None,
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
        reference_text_kind : 参考文本类型（theme / original_lyrics）

        Returns
        -------
        PipelineResult  包含最终歌词及所有中间产物。
        """
        if not self._started:
            raise RuntimeError(
                "请先调用 start() 或使用 'async with AgentOrchestrator() as orch'"
            )

        t0 = time.perf_counter()
        result = PipelineResult(session_id=self._memory.session_id)
        self._pipeline_state = _STATE_STARTING

        logger.info("流水线开始 – midi=%s | text='%s'", midi_path, reference_text[:60])
        logger.info("参考文本类型 – %s", reference_text_kind)
        await self._emit_event(
            event_callback,
            {
                "type": "run_started",
                "session_id": self._memory.session_id,
                "midi_path": str(midi_path),
                "reference_text_kind": reference_text_kind,
            },
        )

        self._memory.set_artifact("source_text", reference_text)
        self._memory.set_artifact("source_text_kind", reference_text_kind)
        self._memory.set_pipeline_value("current_midi_path", str(midi_path))
        self._memory.set_pipeline_value("reference_text_kind", reference_text_kind)
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
            await self._emit_event(
                event_callback,
                {
                    "type": "step_started",
                    "step": "midi_analysis",
                    "message": "Analyzing MIDI file structure...",
                },
            )
            self._transition_state(_STATE_MIDI_ANALYSIS)
            logger.info("Step 1a: 并行调用 MCP 工具分析 MIDI 文件…")

            # 并行调用 MIDI analyzer 工具（结构 + 押韵 + 时值）
            midi_task = self._call_tool_direct_safe(
                server_name="midi-analyzer",
                tool_name="analyze_midi",
                args={"file_path": str(midi_path)},
                parse_json=True,
                default={},
                event_callback=event_callback,
            )
            rhyme_task = self._call_tool_direct_safe(
                server_name="midi-analyzer",
                tool_name="suggest_rhyme_positions",
                args={"file_path": str(midi_path)},
                parse_json=True,
                default=[],
                event_callback=event_callback,
            )
            durations_task = self._call_tool_direct_safe(
                server_name="midi-analyzer",
                tool_name="get_syllable_durations",
                args={"file_path": str(midi_path)},
                parse_json=True,
                default=[],
                event_callback=event_callback,
            )

            # 等待所有任务完成
            (
                midi_analysis,
                rhyme_positions_raw,
                syllable_durations_raw,
            ) = await asyncio.gather(
                midi_task,
                rhyme_task,
                durations_task,
                return_exceptions=True,
            )

            # 处理结果
            result.midi_analysis = (
                midi_analysis if isinstance(midi_analysis, dict) else {}
            )

            syllable_durations: list[float] = (
                [
                    float(value)
                    for value in syllable_durations_raw
                    if isinstance(value, int | float)
                ]
                if isinstance(syllable_durations_raw, list)
                else []
            )
            if syllable_durations:
                result.midi_analysis["syllable_durations"] = syllable_durations
                result.midi_analysis["note_durations"] = syllable_durations
            else:
                result.midi_analysis.setdefault("syllable_durations", [])

            rhyme_positions: list[int] = (
                rhyme_positions_raw if isinstance(rhyme_positions_raw, list) else []
            )
            result.midi_analysis["rhyme_positions"] = rhyme_positions

            syllable_count: int = int(
                result.midi_analysis.get("effective_syllable_count", 0)
                or result.midi_analysis.get("syllable_count", 0)
            )
            strong_beats: list[int] = result.midi_analysis.get(
                "strong_beat_positions", []
            )

            self._memory.set_pipeline_value("midi_analysis", result.midi_analysis)
            logger.info(
                "Step 1a 完成: syllable_count=%d bpm=%s key=%s",
                syllable_count,
                result.midi_analysis.get("bpm", "?"),
                result.midi_analysis.get("key", "?"),
            )
            await self._emit_event(
                event_callback,
                {
                    "type": "step_completed",
                    "step": "midi_analysis",
                    "message": f"{syllable_count} syllables, {result.midi_analysis.get('bpm', 0):.0f} BPM",
                    "metrics": {
                        "syllable_count": syllable_count,
                        "bpm": result.midi_analysis.get("bpm", 0),
                    },
                },
            )

            # ----------------------------------------------------------
            # Step 1b: 并行调用 MCP 工具完成 0243 旋律映射（无 LLM）
            # ----------------------------------------------------------
            await self._emit_event(
                event_callback,
                {
                    "type": "step_started",
                    "step": "melody_mapping",
                    "message": "Mapping melody to 0243 tone sequence...",
                },
            )
            self._transition_state(_STATE_MELODY_MAPPING)
            logger.info("Step 1b: 并行调用 MCP 工具映射 0243 旋律声调…")

            # 并行调用 melody-mapper 和 jyutping 工具
            melody_task = self._call_tool_direct_safe(
                server_name="melody-mapper",
                tool_name="analyze_melody_contour",
                args={"file_path": str(midi_path)},
                parse_json=True,
                default={},
                event_callback=event_callback,
            )
            jp_task = self._call_tool_direct_safe(
                server_name="jyutping",
                tool_name="chinese_to_jyutping",
                args={"text": reference_text},
                parse_json=True,
                default=[],
                event_callback=event_callback,
            )
            tone_pattern_task = self._call_tool_direct_safe(
                server_name="jyutping",
                tool_name="get_tone_pattern",
                args={"text": reference_text},
                parse_json=False,
                default="",
                event_callback=event_callback,
            )
            tone_codes_task = self._call_tool_direct_safe(
                server_name="jyutping",
                tool_name="get_tone_code",
                args={"text": reference_text},
                parse_json=True,
                default=[],
                event_callback=event_callback,
            )

            # 等待所有任务完成
            (
                melody_analysis_raw,
                jp_candidates_raw,
                tone_pattern_raw,
                tone_codes_raw,
            ) = await asyncio.gather(
                melody_task,
                jp_task,
                tone_pattern_task,
                tone_codes_task,
                return_exceptions=True,
            )

            # 处理结果
            melody_analysis: dict[str, Any] = (
                melody_analysis_raw if isinstance(melody_analysis_raw, dict) else {}
            )
            melody_tone_sequence: list[int] = [
                int(t)
                for t in melody_analysis.get("tone_sequence", [])
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
            positions_needing_candidates.update(
                str(p) for p in strong_beats[:16]
            )  # First 16 strong beats
            positions_needing_candidates.update(
                str(p) for p in rhyme_positions[:8]
            )  # First 8 rhyme positions

            # Map position -> tone code
            position_tone_map: dict[str, str] = {}
            for pos_str in positions_needing_candidates:
                pos = int(pos_str)
                beat_tone = (
                    str(melody_tone_sequence[pos])
                    if pos < len(melody_tone_sequence)
                    else "4"
                )
                position_tone_map[pos_str] = beat_tone

            # Get unique tone codes and batch query them
            unique_tone_codes = list(set(position_tone_map.values()))
            if unique_tone_codes:
                # BATCH API call - query all tone codes at once
                batch_candidates_raw = await self._call_tool_direct_safe(
                    server_name="jyutping",
                    tool_name="find_words_by_tone_code",
                    args={"code": unique_tone_codes},  # Batch call!
                    parse_json=True,
                    default=[],
                    event_callback=event_callback,
                )

                # Parse batch results: returns list of lists [[words_for_code1], [words_for_code2], ...]
                batch_candidates: list[list[str]] = (
                    batch_candidates_raw
                    if isinstance(batch_candidates_raw, list)
                    and all(isinstance(item, list) for item in batch_candidates_raw)
                    else []
                )

                # Build tone_code -> candidates map
                tone_to_candidates: dict[str, list[str]] = {}
                for i, tone_code in enumerate(unique_tone_codes):
                    if i < len(batch_candidates):
                        tone_to_candidates[tone_code] = batch_candidates[i][
                            :15
                        ]  # Keep top 15 per tone

                # Build position -> candidates map
                strong_beat_candidates: dict[str, list[str]] = {}
                for pos_str, tone_code in position_tone_map.items():
                    strong_beat_candidates[pos_str] = tone_to_candidates.get(
                        tone_code, []
                    )
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
                theme_candidates_raw = await self._call_tool_direct_safe(
                    server_name="jyutping",
                    tool_name="find_words_by_tone_code",
                    args={"code": theme_tone_codes},
                    parse_json=True,
                    default=[],
                    event_callback=event_callback,
                )

                theme_candidates_list: list[list[str]] = (
                    theme_candidates_raw
                    if isinstance(theme_candidates_raw, list)
                    and all(isinstance(item, list) for item in theme_candidates_raw)
                    else []
                )

                # Map tone code -> candidates
                for i, tone_code in enumerate(theme_tone_codes):
                    if i < len(theme_candidates_list):
                        theme_candidates[tone_code] = theme_candidates_list[i][:20]

                logger.info(
                    "主题相关声调码：%s → 候选词 %d 个",
                    theme_tone_codes,
                    sum(len(c) for c in theme_candidates.values()),
                )

            jyutping_map: dict = {
                "reference_text": reference_text,
                "selected_jyutping": jp_candidates[0] if jp_candidates else "",
                "all_candidates": jp_candidates,
                "reference_tone_pattern": tone_pattern,
                "reference_tone_sequence": reference_tone_sequence,
                "tone_codes": tone_codes,
                "melody_tone_sequence_0243": melody_tone_sequence,
                "rhyme_positions": rhyme_positions,
                "strong_beat_positions": strong_beats,
                "strong_beat_candidates": strong_beat_candidates,
                "theme_candidates": theme_candidates,  # Theme-related words
                "target_syllable_count": syllable_count,
            }
            result.jyutping_map = jyutping_map
            self._memory.set_pipeline_value("jyutping_map", jyutping_map)

            logger.info(
                "Step 1c 完成: jp_candidates=%d melody_0243=%s reference_1_6=%s",
                len(jp_candidates),
                " ".join(str(t) for t in melody_tone_sequence[:16]) or "(none)",
                tone_pattern[:40] if tone_pattern else "(none)",
            )
            await self._emit_event(
                event_callback,
                {
                    "type": "step_completed",
                    "step": "melody_mapping",
                    "message": f"{len(melody_tone_sequence)} tone positions mapped",
                },
            )
            await self._emit_event(
                event_callback,
                {
                    "type": "step_completed",
                    "step": "jyutping_conversion",
                    "message": f"{len(jp_candidates)} Jyutping candidates",
                },
            )

            await self._emit_event(
                event_callback,
                {
                    "type": "step_completed",
                    "step": "candidate_query",
                    "message": (
                        f"{len(unique_tone_codes)} tone codes, "
                        f"theme groups={len(theme_candidates)}"
                    ),
                },
            )

            # ----------------------------------------------------------
            # Step 2: LLM 代理循环（创作 → 校验）
            # ----------------------------------------------------------
            revision_instructions: str = ""

            await self._emit_event(
                event_callback,
                {
                    "type": "step_started",
                    "step": "lyrics_composition",
                    "message": "LLM composing lyrics...",
                },
            )
            self._transition_state(_STATE_COMPOSITION)

            for attempt in range(self._max_revision_loops + 1):
                logger.info("创作尝试 %d/%d", attempt + 1, self._max_revision_loops + 1)
                await self._emit_event(
                    event_callback,
                    {
                        "type": "attempt_started",
                        "attempt": attempt + 1,
                        "max_attempts": self._max_revision_loops + 1,
                    },
                )

                # --- 歌词创作（LLM 代理）---
                compose_task = self._build_compose_task(
                    reference_text=reference_text,
                    reference_text_kind=reference_text_kind,
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
                    if isinstance(draft_output, dict)
                    else ""
                )

                # --- Self-check before submitting to validator ---
                self_check_result = None
                if isinstance(draft_output, dict):
                    self_check_result = draft_output.get("self_check", {})
                    if self_check_result and not self_check_result.get("passed", False):
                        issues = self_check_result.get("issues", [])
                        logger.warning(
                            "composer self-check FAILED (attempt %d): %s",
                            attempt + 1,
                            "; ".join(issues),
                        )
                        await self._emit_event(
                            event_callback,
                            {
                                "type": "self_check_failed",
                                "attempt": attempt + 1,
                                "issues": issues,
                                "message": f"Self-check failed: {'; '.join(issues)}",
                            },
                        )
                        # Skip to next attempt without calling validator
                        continue

                if isinstance(draft_output, dict):
                    draft_output = await self._apply_orchestrator_word_selection(
                        draft_output=draft_output,
                        candidate_map=strong_beat_candidates,
                        strong_beats=strong_beats,
                        rhyme_positions=rhyme_positions,
                        melody_tone_sequence=melody_tone_sequence,
                        reference_text=reference_text,
                        event_callback=event_callback,
                    )
                    draft_lyrics = str(draft_output.get("lyrics", draft_lyrics))

                result.draft_history.append(draft_lyrics)
                await self._emit_event(
                    event_callback,
                    {
                        "type": "step_completed",
                        "step": "lyrics_composition",
                        "message": f"Lyrics composed ({len(draft_lyrics)} chars)",
                        "lyrics": draft_lyrics,
                        "attempt": attempt + 1,
                    },
                )

                await self._emit_event(
                    event_callback,
                    {
                        "type": "step_started",
                        "step": "lyrics_validation",
                        "message": "Validating lyrics quality...",
                        "attempt": attempt + 1,
                    },
                )
                self._transition_state(_STATE_VALIDATION)

                # --- 歌词校验（LLM 代理）---
                validate_task = self._build_validate_task(
                    draft_lyrics=draft_lyrics,
                    draft_jyutping=draft_jyutping,
                    syllable_count=syllable_count,
                    melody_tone_sequence=melody_tone_sequence,
                    strong_beats=strong_beats,
                    rhyme_positions=rhyme_positions,
                    reference_text=reference_text,
                    reference_text_kind=reference_text_kind,
                )
                validation_output = await self._run_agent(
                    agent_name="validator",
                    task=validate_task,
                    context_key="validation_result",
                )

                score: float = float(
                    validation_output.get("score", 0.0)
                    if isinstance(validation_output, dict)
                    else 0.0
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
                    attempt + 1,
                    score,
                    self._min_quality_score,
                )
                await self._emit_event(
                    event_callback,
                    {
                        "type": "step_completed",
                        "step": "lyrics_validation",
                        "message": f"Score: {score:.2f}/1.00",
                        "score": score,
                        "attempt": attempt + 1,
                    },
                )

                if score >= self._min_quality_score:
                    result.lyrics = draft_lyrics
                    result.accepted = True
                    self._memory.set_best_result(
                        draft=draft_output if isinstance(draft_output, dict) else {},
                        validation=validation_output
                        if isinstance(validation_output, dict)
                        else {},
                        score=score,
                    )
                    logger.info("✓ 歌词已在第 %d 次尝试时通过验收", attempt + 1)
                    await self._emit_event(
                        event_callback,
                        {
                            "type": "accepted",
                            "attempt": attempt + 1,
                            "score": score,
                            "lyrics": draft_lyrics,
                        },
                    )
                    self._transition_state(_STATE_COMPLETED)
                    break

                corrections: list[str] = (
                    validation_output.get("corrections", [])
                    if isinstance(validation_output, dict)
                    else []
                )
                revision_instructions = (
                    f"上一草稿未通过校验（score={score:.2f}）。\n"
                    f"校验反馈：{feedback}\n"
                    "需修改的问题：\n" + "\n".join(f"  - {c}" for c in corrections)
                )
                self._memory.set_attempt_value(
                    "revision_instructions", revision_instructions
                )
                self._memory.add_ai_message(
                    f"[第 {attempt + 1} 轮修改] 得分偏低（{score:.2f}），"
                    f"根据以下反馈修改：{feedback}"
                )
                await self._emit_event(
                    event_callback,
                    {
                        "type": "revision_requested",
                        "attempt": attempt + 1,
                        "score": score,
                        "message": feedback,
                    },
                )
                decision = decide_after_validation(
                    {
                        "score": score,
                        "min_quality_score": self._min_quality_score,
                        "attempt": attempt,
                        "max_attempts": self._max_revision_loops + 1,
                    }
                )
                if decision.next_stage == "compose":
                    self._transition_state(_STATE_COMPOSITION)
                else:
                    self._transition_state(_STATE_COMPLETED)
                    break

                best_result = self._memory.get_best_result()
                best_score = float(
                    best_result.get("score", float("-inf")) or float("-inf")
                )
                if score > best_score:
                    self._memory.set_best_result(
                        draft=draft_output if isinstance(draft_output, dict) else {},
                        validation=validation_output
                        if isinstance(validation_output, dict)
                        else {},
                        score=score,
                    )

            # 所有轮次用尽时，选最高分草稿
            if not result.accepted and result.draft_history:
                best_idx = (
                    result.validator_scores.index(max(result.validator_scores))
                    if result.validator_scores
                    else 0
                )
                result.lyrics = result.draft_history[best_idx]
                logger.warning(
                    "已用尽全部 %d 次尝试，选用第 %d 次草稿（score=%.2f）。",
                    self._max_revision_loops + 1,
                    best_idx + 1,
                    result.validator_scores[best_idx] if result.validator_scores else 0,
                )
                await self._emit_event(
                    event_callback,
                    {
                        "type": "fallback_best_draft",
                        "attempt": best_idx + 1,
                        "score": result.validator_scores[best_idx]
                        if result.validator_scores
                        else 0,
                        "lyrics": result.lyrics,
                    },
                )

        except Exception as exc:  # noqa: BLE001
            logger.exception("流水线错误: %s", exc)
            result.error = str(exc)
            self._pipeline_state = _STATE_ERROR
            self._memory.set_run_status(error=str(exc))
            self._memory.add_ai_message(f"[错误] 流水线失败：{exc}")
            await self._emit_event(
                event_callback,
                {
                    "type": "error",
                    "message": str(exc),
                },
            )

        result.elapsed_seconds = time.perf_counter() - t0
        logger.info(
            "流水线完成  elapsed=%.1fs  accepted=%s  revisions=%d",
            result.elapsed_seconds,
            result.accepted,
            result.revision_count,
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
        await self._emit_event(
            event_callback,
            {
                "type": "run_completed",
                "accepted": result.accepted,
                "revision_count": result.revision_count,
                "elapsed_seconds": result.elapsed_seconds,
                "lyrics": result.lyrics,
                "error": result.error,
            },
        )
        return result

    def _transition_state(self, next_state: str) -> None:
        """Validate and apply a pipeline state transition."""
        allowed = _ALLOWED_STATE_TRANSITIONS.get(self._pipeline_state, set())
        if next_state not in allowed and next_state != self._pipeline_state:
            raise ConstraintViolation(
                "Invalid pipeline state transition",
                context={"from": self._pipeline_state, "to": next_state},
            )
        self._pipeline_state = next_state

    async def _emit_event(
        self,
        callback: PipelineEventCallback | None,
        event: dict[str, Any],
    ) -> None:
        """Emit a pipeline event to observers without interrupting the run on callback errors."""
        if callback is None:
            return
        try:
            maybe_awaitable = callback(event)
            if asyncio.iscoroutine(maybe_awaitable):
                await maybe_awaitable
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pipeline event callback failed: %s", exc)

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
            raise KeyError(f"找不到代理 '{name}'，可用代理：{list(self._agents)}")
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
            declared = set(srv_config.tool_names)
            found_names = [t.name for t in all_tools if t.name in declared]

            if not found_names and declared:
                # Warn if the server declared tools that weren't found
                loaded_names = {t.name for t in all_tools}
                missing = declared - loaded_names
                logger.warning(
                    "服务器 '%s' 声明了以下工具但未从 MCP 加载到：%s",
                    srv_config.name,
                    missing,
                )

            MCP_REGISTRY.set_server_tools(srv_config.name, found_names)
            logger.debug("服务器 '%s' 关联工具：%s", srv_config.name, found_names)

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
            WordSelectorAgent,
        )

        _builtin: dict[str, type] = {
            "lyrics-composer": LyricsComposerAgent,
            "validator": ValidatorAgent,
            "word-selector": WordSelectorAgent,
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
                    agent_cfg.name,
                    agent_cfg.name,
                )
                continue

            cls = AGENT_REGISTRY.get_class(agent_cfg.name)
            if agent_cfg.name in {"lyrics-composer", "validator"}:
                tools = []
            else:
                tools = MCP_REGISTRY.get_tools_for_servers(
                    agent_cfg.allowed_mcp_servers
                )

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
                agent_cfg.name,
                len(tools),
                [t.name for t in tools],
            )

    # ------------------------------------------------------------------
    # Internal: single-stage runner
    # ------------------------------------------------------------------

    async def _run_agent(
        self,
        agent_name: str,
        task: str,
        context_key: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        调用一个 LLM 代理执行任务，返回其结构化输出 dict。

        与旧版 _run_stage 的区别：
        - 仅用于真正的 LLM 代理（lyrics-composer、validator）
        - 计算性步骤已由 _call_tool_direct 直接处理
        - 使用信号量限制并发 LLM 请求，防止 GPU 过载
        """
        if agent_name not in self._agents:
            logger.error("代理 '%s' 不存在。可用：%s", agent_name, list(self._agents))
            return {}

        agent = self._agents[agent_name]
        logger.info(">>> LLM 代理：%s（等待许可）", agent_name)

        async with self._llm_semaphore:
            logger.info(">>> LLM 代理：%s 开始执行", agent_name)
            try:
                agent_result = await agent.run(task, **kwargs)
                if not getattr(agent_result, "success", False):
                    logger.warning(
                        "LLM 代理 '%s' 执行失败：%s",
                        agent_name,
                        getattr(agent_result, "error", "unknown error"),
                    )
                    self._memory.add_ai_message(
                        f"[{agent_name}] 执行失败：{getattr(agent_result, 'error', 'unknown error')}"
                    )
                    return {"error": getattr(agent_result, "error", "agent failed")}

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

    async def _run_word_selector_isolated(
        self,
        task: str,
        candidates: list[str],
        context: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        """Run word selection with a cloned memory snapshot."""
        if "word-selector" not in self._agents:
            return {}

        selector_template = self._agents["word-selector"]
        selector_cls = type(selector_template)
        selector = selector_cls(
            config=selector_template.config,
            llm=selector_template.llm,
            memory=ShortTermMemory.from_dict(self._memory.to_dict()),
            tools=[],
        )

        try:
            result = await asyncio.wait_for(
                selector.run(
                    task,
                    candidates=candidates,
                    context=context,
                    count=1,
                ),
                timeout=timeout_s,
            )
        except TimeoutError:
            return {}

        if not getattr(result, "success", False):
            return {}

        if isinstance(result.data, dict):
            selected_words = result.data.get("selected_words")
            if (
                isinstance(selected_words, list)
                and selected_words
                and isinstance(selected_words[0], dict)
            ):
                return selected_words[0]

        if isinstance(result.output, str):
            payload: dict[str, Any] = {"word": result.output}
            if isinstance(result.metadata, dict):
                payload["reason"] = str(result.metadata.get("selection_reason", ""))
            return payload

        return {}

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
        解析后的 Python 对象（dict / list / str）。

        Raises
        ------
        ToolInvokeError
            When the tool is not registered or invocation fails.
        ParseError
            When strict JSON parsing is requested and payload is malformed.
        """
        tool = MCP_REGISTRY.get_all_tools()
        tool_obj = next((t for t in tool if t.name == tool_name), None)

        if tool_obj is None:
            raise ToolInvokeError(
                "Tool not found in MCP registry",
                context={"server": server_name, "tool": tool_name, "args": args},
            )

        # Log direct tool call at INFO level
        logger.info("[TOOL] Direct call: %s.%s(%s)", server_name, tool_name, args)

        # Emit event for UI/progress display if possible
        await self._emit_event(
            getattr(self, "_event_callback", None),
            {
                "type": "tool_call",
                "server": server_name,
                "tool": tool_name,
                "args": args,
            },
        )

        try:
            raw = await tool_obj.ainvoke(args)
        except Exception as exc:  # noqa: BLE001
            raise ToolInvokeError(
                "MCP tool invocation failed",
                context={
                    "server": server_name,
                    "tool": tool_name,
                    "args": args,
                    "cause": str(exc),
                },
            ) from exc

        return normalize_mcp_result(
            raw,
            parse_json=parse_json,
            strict_json=parse_json,
        )

    async def _call_tool_direct_safe(
        self,
        server_name: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        parse_json: bool,
        default: Any,
        event_callback: PipelineEventCallback | None = None,
    ) -> Any:
        """Best-effort wrapper that degrades gracefully at pipeline boundary."""
        try:
            return await self._call_tool_direct(
                server_name=server_name,
                tool_name=tool_name,
                args=args,
                parse_json=parse_json,
            )
        except (ToolInvokeError, ParseError) as exc:
            logger.warning(
                "Tool boundary degraded: %s.%s | %s", server_name, tool_name, exc
            )
            await self._emit_event(
                event_callback,
                {
                    "type": "tool_error",
                    "server": server_name,
                    "tool": tool_name,
                    "message": str(exc),
                },
            )
            return default

    # ------------------------------------------------------------------
    # Internal: compose-task builder
    # ------------------------------------------------------------------

    async def _apply_orchestrator_word_selection(
        self,
        draft_output: dict[str, Any],
        candidate_map: dict[str, list[str]],
        strong_beats: list[int],
        rhyme_positions: list[int],
        melody_tone_sequence: list[int],
        reference_text: str,
        event_callback: PipelineEventCallback | None,
    ) -> dict[str, Any]:
        """Apply phrase-first word selection, then fallback to single characters."""
        if "word-selector" not in self._agents:
            return draft_output

        raw_lyrics = str(draft_output.get("lyrics", ""))
        lyric_chars = list(raw_lyrics.replace("\n", ""))
        if not lyric_chars:
            return draft_output

        strong_set = set(str(pos) for pos in strong_beats)
        rhyme_set = set(str(pos) for pos in rhyme_positions)
        selection_targets: list[tuple[int, list[str]]] = []
        for pos_str, candidates in candidate_map.items():
            if (
                not isinstance(candidates, list)
                or len(candidates) <= self._word_selector_threshold
            ):
                continue
            if pos_str not in strong_set and pos_str not in rhyme_set:
                continue

            constrained_candidates = self._candidate_constraint_engine.apply(candidates)
            if not constrained_candidates:
                continue

            try:
                position = int(pos_str)
            except ValueError:
                continue
            if position < 0 or position >= len(lyric_chars):
                continue
            selection_targets.append((position, constrained_candidates))

        if not selection_targets:
            return draft_output

        selected_by_position: dict[int, list[str]] = {
            pos: cands for pos, cands in sorted(selection_targets)
        }
        pending_positions = sorted(selected_by_position)
        limited_positions = pending_positions[: self._word_selector_max_targets]
        consumed_positions: set[int] = set()
        applied = 0
        phrase_applied = 0
        llm_calls = 0

        use_fast_selector = self._word_selector_fast_mode == "always" or (
            self._word_selector_fast_mode == "auto"
            and len(limited_positions) > self._word_selector_max_llm_calls
        )

        phrase_spans = self._build_phrase_spans(
            limited_positions,
            max_len=max(2, self._phrase_selector_max_len),
        )

        # Tool calls are parallelized here by the orchestrator so we don't rely
        # on the model to discover/use parallel execution patterns.
        phrase_payload_map: dict[tuple[int, int], dict[str, Any]] = {}
        if phrase_spans and melody_tone_sequence:
            phrase_codes: list[str] = []
            valid_spans: list[tuple[int, int]] = []
            for start, length in phrase_spans:
                tone_slice = melody_tone_sequence[
                    start:min(start + length, len(melody_tone_sequence))
                ]
                if len(tone_slice) != length:
                    continue
                phrase_codes.append("".join(str(t) for t in tone_slice))
                valid_spans.append((start, length))

            if phrase_codes:
                phrase_results = await self._call_tool_direct_safe(
                    server_name="jyutping",
                    tool_name="find_words_by_tone_code",
                    args={"code": phrase_codes},
                    parse_json=True,
                    default=[],
                    event_callback=event_callback,
                )

                if isinstance(phrase_results, list):
                    for i, span in enumerate(valid_spans):
                        words = phrase_results[i] if i < len(phrase_results) else []
                        if isinstance(words, list):
                            phrase_payload_map[span] = {"words": words}

        # Phrase-first: one LLM decision can replace multiple character-level calls.
        phrase_specs: list[tuple[int, int, list[str]]] = []
        phrase_tasks: list[Any] = []
        for start, length in phrase_spans:
            phrase_payload = phrase_payload_map.get((start, length), {})
            if not isinstance(phrase_payload, dict):
                continue

            phrase_candidates = phrase_payload.get("words", [])
            if not isinstance(phrase_candidates, list) or not phrase_candidates:
                continue

            phrase_candidates = [
                str(w).strip() for w in phrase_candidates if str(w).strip()
            ]
            phrase_candidates = [w for w in phrase_candidates if len(w) == length]
            phrase_candidates = self._candidate_constraint_engine.apply(
                phrase_candidates
            )
            if not phrase_candidates:
                continue

            if use_fast_selector:
                selected_phrase = phrase_candidates[0]
                if len(selected_phrase) == length:
                    for offset, ch in enumerate(selected_phrase):
                        pos = start + offset
                        if pos >= len(lyric_chars):
                            break
                        lyric_chars[pos] = ch
                        consumed_positions.add(pos)
                        applied += 1
                    phrase_applied += 1
                continue

            if llm_calls >= self._word_selector_max_llm_calls:
                continue

            phrase_context = {
                "position": f"第 {start + 1} 到第 {start + length} 字",
                "surrounding_before": raw_lyrics[max(0, start - 5):start],
                "surrounding_after": raw_lyrics[
                    start + length:start + length + 5
                ],
                "melody_tone": " ".join(
                    str(melody_tone_sequence[i])
                    for i in range(
                        start, min(start + length, len(melody_tone_sequence))
                    )
                ),
                "semantic_field": reference_text[:50],
                "theme": "歌词创作",
                "rhyme_requirement": (
                    "需与押韵位置协调"
                    if any(str(i) in rhyme_set for i in range(start, start + length))
                    else ""
                ),
            }
            phrase_specs.append((start, length, phrase_candidates))
            phrase_tasks.append(
                asyncio.wait_for(
                    self._run_agent(
                        agent_name="word-selector",
                        task=f"为歌词片段位置 {start}-{start + length - 1} 选择最合适的短语",
                        context_key="selected_words",
                        candidates=phrase_candidates[:20],
                        context=phrase_context,
                        count=1,
                    ),
                    timeout=self._word_selector_call_timeout_s,
                )
            )
            llm_calls += 1

        if phrase_tasks:
            phrase_results = await asyncio.gather(*phrase_tasks, return_exceptions=True)
            for (start, length, phrase_candidates), phrase_selection in zip(
                phrase_specs,
                phrase_results,
            ):
                selected_phrase = ""
                if isinstance(phrase_selection, Exception):
                    logger.warning("Word selector phrase failed: %s", phrase_selection)
                elif isinstance(phrase_selection, dict):
                    if isinstance(phrase_selection.get("word"), str):
                        selected_phrase = str(phrase_selection.get("word", ""))
                    elif (
                        isinstance(phrase_selection.get("selected_words"), list)
                        and phrase_selection["selected_words"]
                    ):
                        first = phrase_selection["selected_words"][0]
                        if isinstance(first, dict):
                            selected_phrase = str(first.get("word", ""))

                if len(selected_phrase) != length:
                    selected_phrase = phrase_candidates[0]

                if len(selected_phrase) != length:
                    continue

                for offset, ch in enumerate(selected_phrase):
                    pos = start + offset
                    if pos >= len(lyric_chars):
                        break
                    lyric_chars[pos] = ch
                    consumed_positions.add(pos)
                    applied += 1
                phrase_applied += 1

        # Fallback: remaining positions still use single-character selection.
        fallback_tasks: list[asyncio.Task[dict[str, Any]]] = []
        fallback_specs: list[tuple[int, list[str], dict[str, Any]]] = []
        for position in limited_positions:
            if position in consumed_positions:
                continue
            candidates = selected_by_position[position]
            context = {
                "position": f"第 {position + 1} 字",
                "surrounding_before": raw_lyrics[max(0, position - 5) : position],
                "surrounding_after": raw_lyrics[position + 1 : position + 6],
                "melody_tone": (
                    str(melody_tone_sequence[position])
                    if position < len(melody_tone_sequence)
                    else None
                ),
                "semantic_field": reference_text[:50],
                "theme": "歌词创作",
                "rhyme_requirement": "需与押韵位置协调"
                if str(position) in rhyme_set
                else "",
            }

            selected_word = ""
            if use_fast_selector or llm_calls >= self._word_selector_max_llm_calls:
                selected_word = str(candidates[0]) if candidates else ""
            else:
                llm_calls += 1

                fallback_specs.append((position, candidates, context))
                fallback_tasks.append(
                    asyncio.create_task(
                        self._run_word_selector_isolated(
                            task=f"为歌词位置 {position} 选择最合适的词语",
                            candidates=candidates,
                            context=context,
                            timeout_s=self._word_selector_call_timeout_s,
                        )
                    )
                )

            if selected_word:
                if len(selected_word) != 1:
                    selected_word = str(candidates[0]) if candidates else ""

                if len(selected_word) == 1:
                    lyric_chars[position] = selected_word
                    applied += 1

        if fallback_tasks:
            selection_results = await asyncio.gather(
                *fallback_tasks,
                return_exceptions=True,
            )

            for (position, candidates, _), selection in zip(
                fallback_specs,
                selection_results,
            ):
                selected_word = ""
                if isinstance(selection, Exception):
                    logger.warning("Word selector fallback failed: %s", selection)
                elif isinstance(selection, dict):
                    if isinstance(selection.get("word"), str):
                        selected_word = str(selection.get("word", ""))
                    elif (
                        isinstance(selection.get("selected_words"), list)
                        and selection["selected_words"]
                    ):
                        first = selection["selected_words"][0]
                        if isinstance(first, dict):
                            selected_word = str(first.get("word", ""))

                if len(selected_word) != 1:
                    selected_word = str(candidates[0]) if candidates else ""

                if len(selected_word) == 1:
                    lyric_chars[position] = selected_word
                    applied += 1

        if applied == 0:
            return draft_output

        rebuilt = self._rebuild_lyrics_with_original_breaks(raw_lyrics, lyric_chars)
        draft_output["lyrics"] = rebuilt
        self._memory.set_pipeline_value("orchestrator_word_selection_applied", True)

        await self._emit_event(
            event_callback,
            {
                "type": "word_selector_applied",
                "applied_count": applied,
                "target_count": len(selection_targets),
                "threshold": self._word_selector_threshold,
                "phrase_applied_count": phrase_applied,
                "llm_calls": llm_calls,
                "fast_mode": use_fast_selector,
                "limited_targets": len(limited_positions),
            },
        )
        return draft_output

    @staticmethod
    def _build_phrase_spans(
        positions: list[int],
        max_len: int,
    ) -> list[tuple[int, int]]:
        """Build contiguous phrase spans from target positions."""
        if not positions:
            return []

        spans: list[tuple[int, int]] = []
        run_start = positions[0]
        run_prev = positions[0]

        def flush_run(start: int, end: int) -> None:
            run_len = end - start + 1
            if run_len < 2:
                return
            cursor = start
            while cursor <= end:
                remaining = end - cursor + 1
                if remaining < 2:
                    break
                length = min(max_len, remaining)
                spans.append((cursor, length))
                cursor += length

        for pos in positions[1:]:
            if pos == run_prev + 1:
                run_prev = pos
                continue
            flush_run(run_start, run_prev)
            run_start = pos
            run_prev = pos

        flush_run(run_start, run_prev)
        return spans

    @staticmethod
    def _rebuild_lyrics_with_original_breaks(
        original_lyrics: str,
        flat_chars: list[str],
    ) -> str:
        """Rebuild line breaks after character-level replacements."""
        lines = original_lyrics.splitlines()
        if not lines:
            return "".join(flat_chars)

        rebuilt_lines: list[str] = []
        cursor = 0
        for line in lines:
            line_len = len(line)
            rebuilt_lines.append("".join(flat_chars[cursor: cursor + line_len]))
            cursor += line_len
        return "\n".join(rebuilt_lines)

    @staticmethod
    def _nearest_note_value_label(beats: float) -> str:
        """Map quarter-note beats to a nearest human-readable note value."""
        if beats <= 0:
            return "unknown"

        candidates = [
            (4.0, "whole"),
            (3.0, "dotted-half"),
            (2.0, "half"),
            (1.5, "dotted-quarter"),
            (1.0, "quarter"),
            (0.75, "dotted-eighth"),
            (2.0 / 3.0, "quarter-triplet"),
            (0.5, "eighth"),
            (1.0 / 3.0, "eighth-triplet"),
            (0.25, "sixteenth"),
        ]
        nearest = min(candidates, key=lambda item: abs(item[0] - beats))
        return nearest[1]

    def _format_note_values(
        self,
        durations: list[Any],
        bpm: Any,
        syllable_count: int,
    ) -> str:
        """Format durations as quarter-note beats plus nearest note labels."""
        try:
            bpm_value = float(bpm)
        except (TypeError, ValueError):
            bpm_value = 0.0

        if bpm_value <= 0:
            return "（无）"

        quarter_sec = 60.0 / bpm_value
        if quarter_sec <= 0:
            return "（无）"

        formatted: list[str] = []
        for value in durations[:syllable_count]:
            try:
                duration_sec = float(value)
            except (TypeError, ValueError):
                continue
            beats = duration_sec / quarter_sec
            label = self._nearest_note_value_label(beats)
            formatted.append(f"{beats:.2f}b({label})")

        return " ".join(formatted) if formatted else "（无）"

    def _build_compose_task(
        self,
        reference_text: str,
        reference_text_kind: str,
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
        syllable_durations = midi_analysis.get("note_durations") or midi_analysis.get(
            "syllable_durations", []
        )
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
        note_values_str = self._format_note_values(
            syllable_durations,
            midi_analysis.get("bpm", 0),
            syllable_count,
        )

        base = self._render_prompt_template(
            "compose-task.md",
            syllable_count=syllable_count,
            reference_text=reference_text,
            reference_text_kind=reference_text_kind,
            reference_text_kind_label=self._reference_text_kind_label(
                reference_text_kind
            ),
            embedded_lyrics_source=embedded_lyrics_source or "（无）",
            embedded_lyric_unit_count=embedded_lyric_unit_count,
            effective_syllable_count_source=effective_syllable_count_source,
            embedded_lyrics_str=embedded_lyrics_str,
            bpm=midi_analysis.get("bpm", "?"),
            key=midi_analysis.get("key", "?"),
            strong_str=strong_str,
            rhyme_str=rhyme_str,
            note_values_str=note_values_str,
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
        reference_text_kind: str,
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
        syllable_durations = midi_analysis.get("note_durations") or midi_analysis.get(
            "syllable_durations", []
        )
        effective_syllable_count_source = midi_analysis.get(
            "effective_syllable_count_source", "melody_notes"
        )
        embedded_lyrics_str = (
            " ".join(str(unit) for unit in embedded_lyrics_preview[:24])
            if embedded_lyrics_preview else "（无）"
        )
        note_values_str = self._format_note_values(
            syllable_durations,
            midi_analysis.get("bpm", 0),
            syllable_count,
        )

        return self._render_prompt_template(
            "validate-task.md",
            draft_lyrics=draft_lyrics,
            draft_jyutping=draft_jyutping
            or "（创作代理未提供，请调用 chinese_to_jyutping 获取）",
            syllable_count=syllable_count,
            tone_seq_str=tone_seq_str,
            strong_str=strong_str,
            rhyme_str=rhyme_str,
            reference_text=reference_text,
            reference_text_kind=reference_text_kind,
            reference_text_kind_label=self._reference_text_kind_label(
                reference_text_kind
            ),
            embedded_lyrics_source=embedded_lyrics_source or "（无）",
            embedded_lyric_unit_count=embedded_lyric_unit_count,
            effective_syllable_count_source=effective_syllable_count_source,
            embedded_lyrics_str=embedded_lyrics_str,
            note_values_str=note_values_str,
            tone_json=tone_json,
            strong_json=strong_json,
            rhyme_json=rhyme_json,
        )

    @staticmethod
    def _render_prompt_template(template_name: str, **kwargs: Any) -> str:
        # Try agents/ subdirectory first, then fall back to prompts/ root
        template_path = PROMPTS_AGENTS_DIR / template_name
        if not template_path.exists():
            template_path = PROMPTS_DIR / template_name
        return template_path.read_text(encoding="utf-8").format(**kwargs).strip()

    @staticmethod
    def _reference_text_kind_label(reference_text_kind: str) -> str:
        if reference_text_kind == "original_lyrics":
            return "原歌词"
        return "主题灵感"

    # ------------------------------------------------------------------
    # Internal: LLM builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_llm() -> BaseChatModel:
        """
        根据 PROVIDER 配置构建对应的 LLM 实例。

        Supported providers
        -------------------
        ollama        – ChatOllama (langchain-ollama)
        ollama-cloud  – ChatOpenAI with ollama.com API (langchain-openai)
        lmstudio      – ChatOpenAI with custom base_url (langchain-openai)
        """
        if PROVIDER == "ollama-cloud":
            # Ollama Cloud: OpenAI-compatible API at ollama.com/v1
            from langchain_openai import ChatOpenAI

            cfg = LLM_CONFIG
            if not cfg.get("api_key"):
                raise ValueError(
                    "OLLAMA_API_KEY is required for ollama-cloud provider. "
                    "Set it in .env or as environment variable."
                )
            logger.info(
                "使用 Ollama Cloud 提供商  model=%s  base_url=%s",
                cfg["model"],
                cfg["base_url"],
            )
            kwargs: dict[str, Any] = {
                "model": cfg["model"],
                "base_url": cfg["base_url"],
                "api_key": cfg["api_key"],
                "temperature": cfg["temperature"],
                "max_tokens": cfg.get("max_tokens", 8192),
                "extra_body": {"thinking": False},
            }
            return ChatOpenAI(**kwargs)
        elif PROVIDER == "lmstudio":
            # LM Studio: OpenAI-compatible API, no real key needed
            from langchain_openai import ChatOpenAI

            cfg = LLM_CONFIG
            logger.info(
                "使用 LM Studio 提供商  model=%s  base_url=%s",
                cfg["model"], cfg["base_url"],
            )
            kwargs: dict[str, Any] = {
                "model": cfg["model"],
                "base_url": cfg["base_url"],
                "api_key": cfg["api_key"],
                "temperature": cfg["temperature"],
                "max_tokens": cfg.get("max_tokens", 8192),
                "extra_body": {"thinking": False},
            }
            return ChatOpenAI(**kwargs)
        else:
            # Default: Ollama (local)
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
