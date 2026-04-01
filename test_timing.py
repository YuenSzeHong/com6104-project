#!/usr/bin/env python3
"""
带计时的全流程测试脚本
运行完整的 Lyrics Agent 流程，并记录每个阶段的耗时
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# 添加项目路径
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("timing-test")

# 测试配置
_ROOT_DIR = Path(__file__).resolve().parent  # 项目根目录
MIDI_PATH = _ROOT_DIR / "test" / "midi" / "ドラえもんのうた.mid"
LYRICS_PATH = _ROOT_DIR / "test" / "lyrics" / "ドラえもんのうた.kanji-yomi.txt"
SESSION_ID = "timing-test-001"
TIMEOUT_MINUTES = 15


def _read_text_file(path: str, encoding: str | None = None) -> str:
    """Read a source lyric / theme file with optional explicit encoding."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    if encoding:
        return file_path.read_text(encoding=encoding)

    candidates = ("utf-8", "utf-8-sig", "cp932", "shift_jis", "euc_jp", "gbk")
    errors: list[str] = []
    for candidate in candidates:
        try:
            return file_path.read_text(encoding=candidate)
        except UnicodeDecodeError as exc:
            errors.append(f"{candidate}: {exc}")

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        0,
        "Could not decode text file. Tried: " + "; ".join(errors),
    )


async def run_timed_pipeline():
    """运行带计时的全流程测试"""

    from agent.orchestrator import AgentOrchestrator

    # 读取歌词文本
    reference_text = _read_text_file(str(LYRICS_PATH))
    global REFERENCE_TEXT
    REFERENCE_TEXT = reference_text

    # 总计时器
    total_start = time.perf_counter()

    # 阶段计时器
    phase_elapsed = {}

    async with AgentOrchestrator(session_id=SESSION_ID) as orch:
        # Step 1: MIDI 分析
        logger.info("=" * 60)
        logger.info("Step 1: MIDI 分析")
        phase_start = time.perf_counter()

        # 调用 MIDI analyzer 工具
        midi_result = await orch._call_tool_direct(
            server_name="midi-analyzer",
            tool_name="analyze_midi",
            args={"file_path": str(MIDI_PATH)},
            parse_json=True,
        )

        phase_elapsed["midi_analysis"] = time.perf_counter() - phase_start
        logger.info(f"MIDI 分析耗时：{phase_elapsed['midi_analysis']:.2f}s")
        logger.info(f"音节数：{midi_result.get('syllable_count', 'N/A') if isinstance(midi_result, dict) else 'N/A'}")

        # Step 2: 声调映射 (melody-mapper)
        logger.info("=" * 60)
        logger.info("Step 2: 声调映射 (Melody Mapper)")
        phase_start = time.perf_counter()

        melody_result = await orch._call_tool_direct(
            server_name="melody-mapper",
            tool_name="analyze_melody_contour",
            args={"file_path": str(MIDI_PATH)},
            parse_json=True,
        )

        phase_elapsed["melody_mapping"] = time.perf_counter() - phase_start
        logger.info(f"旋律映射耗时：{phase_elapsed['melody_mapping']:.2f}s")
        tone_seq = melody_result.get("tone_sequence", []) if isinstance(melody_result, dict) else []
        logger.info(f"声调序列长度：{len(tone_seq)}")

        # Step 3: 粤拼转换
        logger.info("=" * 60)
        logger.info("Step 3: 粤拼转换 (Jyutping)")
        phase_start = time.perf_counter()

        jyutping_result = await orch._call_tool_direct(
            server_name="jyutping",
            tool_name="chinese_to_jyutping",
            args={"text": REFERENCE_TEXT},
            parse_json=True,
        )

        phase_elapsed["jyutping_conversion"] = time.perf_counter() - phase_start
        logger.info(f"粤拼转换耗时：{phase_elapsed['jyutping_conversion']:.2f}s")
        jp_list = jyutping_result if isinstance(jyutping_result, list) else []
        logger.info(f"粤拼候选数：{len(jp_list)}")

        # Step 3b: 批量查询声调码候选词（模拟 orchestrator run 方法中的逻辑）
        logger.info("=" * 60)
        logger.info("Step 3b: 批量查询声调码候选词")
        phase_start = time.perf_counter()

        strong_beats = midi_result.get("strong_beat_positions", []) if isinstance(midi_result, dict) else []
        rhyme_positions = jyutping_result.get("rhyme_positions", []) if isinstance(jyutping_result, dict) else []
        melody_tone_sequence = melody_result.get("tone_sequence", []) if isinstance(melody_result, dict) else []

        # Gather positions needing candidates
        positions_needing_candidates = set()
        positions_needing_candidates.update(str(p) for p in strong_beats[:16])
        positions_needing_candidates.update(str(p) for p in rhyme_positions[:8])

        # Map position -> tone code
        position_tone_map = {}
        for pos_str in positions_needing_candidates:
            pos = int(pos_str)
            beat_tone = str(melody_tone_sequence[pos]) if pos < len(melody_tone_sequence) else "4"
            position_tone_map[pos_str] = beat_tone

        # Batch query unique tone codes
        unique_tone_codes = list(set(position_tone_map.values()))
        if unique_tone_codes:
            batch_candidates_raw = await orch._call_tool_direct(
                server_name="jyutping",
                tool_name="find_words_by_tone_code",
                args={"code": unique_tone_codes},  # BATCH CALL!
                parse_json=True,
            )
            phase_elapsed["batch_word_lookup"] = time.perf_counter() - phase_start
            batch_candidates = batch_candidates_raw if isinstance(batch_candidates_raw, list) else []
            logger.info(f"批量查询耗时：{phase_elapsed['batch_word_lookup']:.2f}s")
            logger.info(f"查询声调码：{len(unique_tone_codes)} 个，返回候选词：{sum(len(c) for c in batch_candidates if isinstance(c, list))} 个")
        else:
            phase_elapsed["batch_word_lookup"] = 0
            logger.info("无需查询候选词")

        # Step 4: 歌词创作
        logger.info("=" * 60)
        logger.info("Step 4: 歌词创作 (Lyrics Composer)")
        phase_start = time.perf_counter()

        # 准备创作任务
        syllable_count = midi_result.get("syllable_count", 0) if isinstance(midi_result, dict) else 0
        compose_task = orch._build_compose_task(
            reference_text=REFERENCE_TEXT,
            syllable_count=syllable_count,
            revision_instructions="",
            attempt=0,
        )

        draft_output = await orch._run_agent(
            agent_name="lyrics-composer",
            task=compose_task,
            context_key="draft_lyrics",
        )

        phase_elapsed["lyrics_composition"] = time.perf_counter() - phase_start
        logger.info(f"歌词创作耗时：{phase_elapsed['lyrics_composition']:.2f}s")

        # Step 5: 验证
        logger.info("=" * 60)
        logger.info("Step 5: 歌词验证 (Validator)")
        phase_start = time.perf_counter()

        draft_lyrics = draft_output.get("lyrics", "") if isinstance(draft_output, dict) else ""
        draft_jyutping = draft_output.get("jyutping", "") if isinstance(draft_output, dict) else ""

        validate_task = orch._build_validate_task(
            draft_lyrics=draft_lyrics,
            draft_jyutping=draft_jyutping,
            syllable_count=syllable_count,
            melody_tone_sequence=tone_seq,
            strong_beats=midi_result.get("strong_beat_positions", []) if isinstance(midi_result, dict) else [],
            rhyme_positions=jyutping_result.get("rhyme_positions", []) if isinstance(jyutping_result, dict) else [],
            reference_text=REFERENCE_TEXT,
        )

        validation_output = await orch._run_agent(
            agent_name="validator",
            task=validate_task,
            context_key="validation_result",
        )

        phase_elapsed["validation"] = time.perf_counter() - phase_start
        logger.info(f"验证耗时：{phase_elapsed['validation']:.2f}s")
        score = validation_output.get("score", 0) if isinstance(validation_output, dict) else 0
        logger.info(f"验证评分：{score:.2f}")

        # 输出最终歌词
        logger.info("=" * 60)
        logger.info("最终生成的歌词")
        logger.info("=" * 60)
        if draft_lyrics:
            # 使用 stderr 直接输出歌词，确保能看到
            print("\n" + "=" * 60, file=sys.stderr)
            print(draft_lyrics, file=sys.stderr)
            print("=" * 60 + "\n", file=sys.stderr)

            # 写入歌词文件
            from pathlib import Path
            lyrics_file = Path(MIDI_PATH).with_suffix('.lyrics.txt')
            lyrics_file.write_text(draft_lyrics, encoding='utf-8')
            logger.info("歌词已写入：%s", lyrics_file)
        else:
            logger.warning("未生成歌词")

        # 清理在 __aexit__ 中自动处理
        phase_start = time.perf_counter()

    phase_elapsed["cleanup"] = time.perf_counter() - phase_start

    # 总耗时
    total_elapsed = time.perf_counter() - total_start
    phase_elapsed["total"] = total_elapsed

    # 打印汇总
    logger.info("=" * 60)
    logger.info("耗时汇总")
    logger.info("=" * 60)
    for stage, elapsed in sorted(phase_elapsed.items(), key=lambda x: x[1], reverse=True):
        pct = (elapsed / total_elapsed * 100) if total_elapsed > 0 else 0
        logger.info(f"{stage:25s}: {elapsed:8.2f}s ({pct:5.1f}%)")
    logger.info("=" * 60)

    return phase_elapsed


async def main():
    """主函数"""
    # 读取歌词文件
    try:
        reference_text = _read_text_file(str(LYRICS_PATH))
    except Exception as e:
        logger.error(f"无法读取歌词文件：{e}")
        return 1

    logger.info("=" * 60)
    logger.info("全流程计时测试")
    logger.info(f"MIDI 文件：{MIDI_PATH}")
    logger.info(f"歌词文件：{LYRICS_PATH}")
    logger.info(f"歌词预览：{reference_text[:50]}...")
    logger.info(f"超时限制：{TIMEOUT_MINUTES} 分钟")
    logger.info("=" * 60)

    try:
        # 设置超时
        stage_times = await asyncio.wait_for(
            run_timed_pipeline(),
            timeout=TIMEOUT_MINUTES * 60,
        )

        # 输出 JSON 结果
        import json
        print("\n\n")
        print("=" * 60)
        print("计时结果 (JSON)")
        print("=" * 60)
        print(json.dumps(stage_times, indent=2))

        # 找出最慢的阶段
        slowest_stage = max(stage_times.items(), key=lambda x: x[1])
        print(f"\n最耗时的阶段：{slowest_stage[0]} ({slowest_stage[1]:.2f}s)")

        return 0

    except asyncio.TimeoutError:
        logger.error(f"测试超时 ({TIMEOUT_MINUTES} 分钟)")
        return 1
    except Exception as e:
        logger.exception(f"测试失败：{e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
