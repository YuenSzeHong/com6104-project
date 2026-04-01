"""
Pipeline runner with progress tracking.

Executes the Cantonese lyrics generation pipeline step-by-step,
yielding progress updates and agent conversation logs for real-time UI feedback.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
import asyncio
from typing import AsyncGenerator

from .progress import PipelineProgress

logger = logging.getLogger("gui.pipeline")


def read_text_file(path: str, encoding: str | None = None) -> str:
    """Read a text file with encoding auto-detection."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

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
        "Could not decode file. Tried: " + "; ".join(errors),
    )


def _format_conversation_log(memory) -> str:
    """Format the agent conversation history from memory as markdown."""
    lines = []

    for turn in memory._turns:
        messages = turn.messages if hasattr(turn, "messages") else turn
        for msg in messages:
            msg_type = msg.__class__.__name__
            content = msg.content if hasattr(msg, "content") else str(msg)

            if msg_type == "HumanMessage":
                lines.append("### 👤 **User:**")
                lines.append("```text")
                # Show up to 2000 chars of the prompt to avoid excessive truncation
                lines.append(f"{content[:2000]}{'...' if len(content) > 2000 else ''}")
                lines.append("```")
                lines.append("")
            elif msg_type == "AIMessage":
                # Check if this is a tool call or regular response
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        lines.append(
                            f"### 🤖 **Agent → Tool:** `{tc.get('name', '?')}`"
                        )
                        args = tc.get("args", {})
                        try:
                            args_str = json.dumps(args, ensure_ascii=False, indent=2)
                        except Exception:
                            args_str = str(args)
                        lines.append(
                            f"```json\n{args_str[:500]}{'...' if len(args_str) > 500 else ''}\n```"
                        )
                        lines.append("")
                if content:
                    lines.append("### 🤖 **Agent:**")
                    lines.append("```text")
                    lines.append(
                        f"{content[:2000]}{'...' if len(content) > 2000 else ''}"
                    )
                    lines.append("```")
                    lines.append("")
            elif msg_type == "ToolMessage":
                tool_name = getattr(msg, "name", "unknown")
                lines.append(f"### 🔧 **Tool Result** (`{tool_name}`):")
                result_preview = content[:800]
                lines.append(
                    f"```text\n{result_preview}{'...' if len(content) > 800 else ''}\n```"
                )
                lines.append("")

    return "\n".join(lines)


async def run_pipeline_with_progress(
    midi_path: str,
    reference_text: str,
    text_file: str | None = None,
    session_id: str | None = None,
) -> AsyncGenerator[tuple[str, str, str], None]:
    """
    Run the pipeline and yield progress updates.

    Parameters
    ----------
    midi_path : str
        Path to the MIDI file.
    reference_text : str
        Direct text input (used if text_file is None).
    text_file : str | None
        Path to a text file (overrides reference_text if provided).
    session_id : str | None
        Optional session identifier.

    Yields
    ------
    (progress_markdown, lyrics_text, conversation_log) tuples
    """
    # Resolve text input
    if text_file:
        try:
            reference_text = read_text_file(text_file)
        except Exception as e:
            yield (
                f"## ❌ Error\n\nFailed to read text file: {e}",
                "",
                "",
            )
            return

    if not reference_text:
        yield (
            "## ⚠️ Warning\n\nNo reference text provided.",
            "",
            "",
        )
        return

    from agent.orchestrator import AgentOrchestrator

    progress = PipelineProgress()
    progress.overall_status = "running"

    # Define pipeline steps
    progress.add_step("MIDI Analysis")
    progress.add_step("Melody Mapping (0243)")
    progress.add_step("Jyutping Conversion")
    progress.add_step("Batch Word Candidate Query")
    progress.add_step("Lyrics Composition (LLM)")
    progress.add_step("Lyrics Validation")

    # Yield initial state
    yield progress.format_progress(), "", "*Waiting for pipeline to start...*"

    try:
        async with AgentOrchestrator(session_id=session_id) as orch:
            memory = orch.memory

            # Step 1: MIDI Analysis
            progress.start_step(0, "Analyzing MIDI file structure...")
            yield progress.format_progress(), "", _format_conversation_log(memory)

            midi_task = orch._call_tool_direct(
                server_name="midi-analyzer",
                tool_name="analyze_midi",
                args={"file_path": midi_path},
                parse_json=True,
            )
            durations_task = orch._call_tool_direct(
                server_name="midi-analyzer",
                tool_name="get_syllable_durations",
                args={"file_path": midi_path},
                parse_json=True,
            )
            rhyme_task = orch._call_tool_direct(
                server_name="midi-analyzer",
                tool_name="suggest_rhyme_positions",
                args={"file_path": midi_path},
                parse_json=True,
            )
            midi_analysis, durations, rhyme_positions_raw = await asyncio.gather(
                midi_task, durations_task, rhyme_task, return_exceptions=True
            )

            midi_result = midi_analysis if isinstance(midi_analysis, dict) else {}
            midi_result["syllable_durations"] = (
                durations if isinstance(durations, list) else []
            )
            midi_result["rhyme_positions"] = (
                rhyme_positions_raw if isinstance(rhyme_positions_raw, list) else []
            )

            syllable_count = int(
                midi_result.get("effective_syllable_count", 0)
                or midi_result.get("syllable_count", 0)
            )
            bpm = midi_result.get("bpm", 0)
            progress.complete_step(0, f"✅ {syllable_count} syllables, {bpm:.0f} BPM")
            orch.memory.set_pipeline_value("midi_analysis", midi_result)
            yield progress.format_progress(), "", _format_conversation_log(memory)

            # Step 2: Melody Mapping
            progress.start_step(1, "Mapping melody to 0243 tone sequence...")
            yield progress.format_progress(), "", _format_conversation_log(memory)

            melody_result = await orch._call_tool_direct(
                server_name="melody-mapper",
                tool_name="analyze_melody_contour",
                args={"file_path": midi_path},
                parse_json=True,
            )
            melody_analysis = melody_result if isinstance(melody_result, dict) else {}
            tone_seq = [
                int(t)
                for t in melody_analysis.get("tone_sequence", [])
                if isinstance(t, int | float | str) and str(t).isdigit()
            ]
            orch.memory.set_pipeline_value("melody_analysis", melody_analysis)

            progress.complete_step(1, f"✅ {len(tone_seq)} tone positions mapped")
            yield progress.format_progress(), "", _format_conversation_log(memory)

            # Step 3: Jyutping Conversion
            progress.start_step(2, "Converting reference text to Jyutping...")
            yield progress.format_progress(), "", _format_conversation_log(memory)

            jp_task = orch._call_tool_direct(
                server_name="jyutping",
                tool_name="chinese_to_jyutping",
                args={"text": reference_text},
                parse_json=True,
            )
            tone_pattern_task = orch._call_tool_direct(
                server_name="jyutping",
                tool_name="get_tone_pattern",
                args={"text": reference_text},
                parse_json=False,
            )
            tone_codes_task = orch._call_tool_direct(
                server_name="jyutping",
                tool_name="get_tone_code",
                args={"text": reference_text},
                parse_json=True,
            )
            jp_raw, tone_pattern_raw, tone_codes_raw = await asyncio.gather(
                jp_task, tone_pattern_task, tone_codes_task, return_exceptions=True
            )

            jp_result = jp_raw if isinstance(jp_raw, list) else []
            tone_codes = tone_codes_raw if isinstance(tone_codes_raw, list) else []
            tone_pattern = str(tone_pattern_raw).strip() if tone_pattern_raw else ""

            # Parse reference tone sequence
            reference_tone_sequence = []
            if isinstance(tone_pattern_raw, list):
                tone_pattern_tokens = [str(tok) for tok in tone_pattern_raw]
            else:
                tone_pattern_tokens = str(tone_pattern_raw).split()

            for tok in tone_pattern_tokens:
                try:
                    reference_tone_sequence.append(int(tok))
                except ValueError:
                    pass

            jp_count = len(jp_result)
            progress.complete_step(2, f"✅ {jp_count} Jyutping candidates")
            yield progress.format_progress(), "", _format_conversation_log(memory)

            # Step 4: Batch Word Query
            progress.start_step(3, "Querying word candidates by tone code...")
            yield progress.format_progress(), "", _format_conversation_log(memory)

            strong_beats = midi_result.get("strong_beat_positions", [])
            melody_tone_sequence = tone_seq

            positions_needing_candidates = set()
            positions_needing_candidates.update(str(p) for p in strong_beats[:16])

            rhyme_positions = midi_result.get("rhyme_positions", [])
            positions_needing_candidates.update(str(p) for p in rhyme_positions[:8])

            position_tone_map = {}
            for pos_str in positions_needing_candidates:
                pos = int(pos_str)
                beat_tone = (
                    str(melody_tone_sequence[pos])
                    if pos < len(melody_tone_sequence)
                    else "4"
                )
                position_tone_map[pos_str] = beat_tone

            unique_tone_codes = list(set(position_tone_map.values()))
            strong_beat_candidates = {}
            if unique_tone_codes:
                batch_result = await orch._call_tool_direct(
                    server_name="jyutping",
                    tool_name="find_words_by_tone_code",
                    args={"code": unique_tone_codes},
                    parse_json=True,
                )
                batch_candidates = (
                    batch_result if isinstance(batch_result, list) else []
                )
                tone_to_candidates = {}
                for i, tone_code in enumerate(unique_tone_codes):
                    if i < len(batch_candidates) and isinstance(
                        batch_candidates[i], list
                    ):
                        tone_to_candidates[tone_code] = batch_candidates[i][:15]

                for pos_str, tone_code in position_tone_map.items():
                    strong_beat_candidates[pos_str] = tone_to_candidates.get(
                        tone_code, []
                    )

                total_candidates = sum(len(c) for c in tone_to_candidates.values())
                progress.complete_step(
                    3,
                    f"✅ {len(unique_tone_codes)} tone codes → {total_candidates} candidates",
                )
            else:
                progress.complete_step(3, "✅ No candidates needed")

            theme_tone_codes = await orch._extract_theme_tone_codes(reference_text)
            theme_candidates = {}
            if theme_tone_codes:
                theme_candidates_raw = await orch._call_tool_direct(
                    server_name="jyutping",
                    tool_name="find_words_by_tone_code",
                    args={"code": theme_tone_codes},
                    parse_json=True,
                )
                theme_candidates_list = (
                    theme_candidates_raw
                    if isinstance(theme_candidates_raw, list)
                    else []
                )
                for i, tone_code in enumerate(theme_tone_codes):
                    if i < len(theme_candidates_list) and isinstance(
                        theme_candidates_list[i], list
                    ):
                        theme_candidates[tone_code] = theme_candidates_list[i][:20]

            jyutping_map = {
                "reference_text": reference_text,
                "selected_jyutping": jp_result[0]
                if jp_result and isinstance(jp_result, list)
                else "",
                "all_candidates": jp_result,
                "reference_tone_pattern": tone_pattern,
                "reference_tone_sequence": reference_tone_sequence,
                "tone_codes": tone_codes,
                "melody_tone_sequence_0243": melody_tone_sequence,
                "strong_beat_positions": strong_beats,
                "strong_beat_candidates": strong_beat_candidates,
                "rhyme_positions": rhyme_positions,
                "theme_candidates": theme_candidates,
                "target_syllable_count": syllable_count,
            }
            orch.memory.set_pipeline_value("jyutping_map", jyutping_map)

            yield progress.format_progress(), "", _format_conversation_log(memory)

            # Step 5: Lyrics Composition
            progress.start_step(4, "LLM composing lyrics... (this may take a while)")
            yield progress.format_progress(), "", _format_conversation_log(memory)

            compose_task = orch._build_compose_task(
                reference_text=reference_text,
                syllable_count=syllable_count,
                revision_instructions="",
                attempt=0,
            )

            # Run agent as a background task so we can stream memory updates to GUI
            agent_task = asyncio.create_task(
                orch._run_agent(
                    agent_name="lyrics-composer",
                    task=compose_task,
                    context_key="draft_lyrics",
                )
            )

            # Yield updates every 0.1s while LLM runs
            while not agent_task.done():
                yield progress.format_progress(), "", _format_conversation_log(memory)
                await asyncio.sleep(0.1)

            try:
                draft_output = agent_task.result()
            except Exception as exc:
                logger.error("Composer LLM failed: %s", exc)
                draft_output = {}

            draft_lyrics = (
                draft_output.get("lyrics", "") if isinstance(draft_output, dict) else ""
            )
            draft_jyutping = (
                draft_output.get("jyutping", "")
                if isinstance(draft_output, dict)
                else ""
            )
            progress.complete_step(4, f"✅ Lyrics composed ({len(draft_lyrics)} chars)")
            yield (
                progress.format_progress(),
                draft_lyrics,
                _format_conversation_log(memory),
            )

            # Step 6: Validation
            progress.start_step(5, "Validating lyrics quality...")
            yield (
                progress.format_progress(),
                draft_lyrics,
                _format_conversation_log(memory),
            )

            validate_task = orch._build_validate_task(
                draft_lyrics=draft_lyrics,
                draft_jyutping=draft_jyutping,
                syllable_count=syllable_count,
                melody_tone_sequence=tone_seq,
                strong_beats=strong_beats,
                rhyme_positions=rhyme_positions,
                reference_text=reference_text,
            )

            # Yield updates every 0.1s while validator runs
            val_agent_task = asyncio.create_task(
                orch._run_agent(
                    agent_name="validator",
                    task=validate_task,
                    context_key="validation_result",
                )
            )

            while not val_agent_task.done():
                yield (
                    progress.format_progress(),
                    draft_lyrics,
                    _format_conversation_log(memory),
                )
                await asyncio.sleep(0.1)

            try:
                validation_output = val_agent_task.result()
            except Exception as exc:
                logger.error("Validator LLM failed: %s", exc)
                validation_output = {}

            score = (
                validation_output.get("score", 0)
                if isinstance(validation_output, dict)
                else 0
            )
            progress.complete_step(5, f"✅ Score: {score:.2f}/1.00")
            yield (
                progress.format_progress(),
                draft_lyrics,
                _format_conversation_log(memory),
            )

            # Done
            progress.overall_status = "done"
            yield (
                progress.format_progress(),
                draft_lyrics,
                _format_conversation_log(memory),
            )

    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        progress.overall_status = "error"
        progress.error_message = str(exc)
        yield progress.format_progress(), f"Error: {exc}", f"**Error:** {exc}"
