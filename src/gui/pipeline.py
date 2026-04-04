"""
Pipeline runner with progress tracking.

Subscribes to AgentOrchestrator events and streams progress updates,
without duplicating orchestration business logic.
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

    # Define UI steps (driven by orchestrator events)
    step_keys = [
        "midi_analysis",
        "melody_mapping",
        "jyutping_conversion",
        "candidate_query",
        "lyrics_composition",
        "lyrics_validation",
    ]
    step_labels = {
        "midi_analysis": "MIDI Analysis",
        "melody_mapping": "Melody Mapping (0243)",
        "jyutping_conversion": "Jyutping Conversion",
        "candidate_query": "Batch Word Candidate Query",
        "lyrics_composition": "Lyrics Composition (LLM)",
        "lyrics_validation": "Lyrics Validation",
    }
    step_index = {key: idx for idx, key in enumerate(step_keys)}
    for key in step_keys:
        progress.add_step(step_labels[key])

    latest_lyrics = ""

    # Yield initial state
    yield progress.format_progress(), "", "*Waiting for pipeline to start...*"

    try:
        async with AgentOrchestrator(session_id=session_id) as orch:
            memory = orch.memory
            event_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()

            async def _on_event(event: dict[str, object]) -> None:
                await event_queue.put(event)

            run_task = asyncio.create_task(
                orch.run(
                    midi_path=midi_path,
                    reference_text=reference_text,
                    event_callback=_on_event,
                )
            )

            while not run_task.done() or not event_queue.empty():
                drained = False
                while not event_queue.empty():
                    drained = True
                    event = await event_queue.get()
                    ev_type = str(event.get("type", ""))
                    ev_step = str(event.get("step", ""))
                    ev_message = str(event.get("message", ""))

                    if ev_type == "run_started":
                        progress.overall_status = "running"
                    elif ev_type == "step_started" and ev_step in step_index:
                        progress.start_step(step_index[ev_step], ev_message)
                    elif ev_type == "step_completed" and ev_step in step_index:
                        # Complete any previous pending steps up to this one to keep UI coherent.
                        idx = step_index[ev_step]
                        for i in range(idx):
                            if progress.steps[i].status == "pending":
                                progress.complete_step(i, "✅ Completed")
                        progress.complete_step(
                            idx, f"✅ {ev_message}" if ev_message else "✅ Completed"
                        )
                    elif ev_type in {"accepted", "fallback_best_draft"}:
                        lyrics_value = event.get("lyrics")
                        if isinstance(lyrics_value, str):
                            latest_lyrics = lyrics_value
                    elif ev_type == "error":
                        progress.overall_status = "error"
                        progress.error_message = ev_message
                    elif ev_type == "run_completed":
                        if event.get("error"):
                            progress.overall_status = "error"
                            progress.error_message = str(event.get("error"))
                        else:
                            progress.overall_status = "done"
                        lyrics_value = event.get("lyrics")
                        if isinstance(lyrics_value, str):
                            latest_lyrics = lyrics_value

                if drained:
                    yield (
                        progress.format_progress(),
                        latest_lyrics,
                        _format_conversation_log(memory),
                    )
                else:
                    await asyncio.sleep(0.1)

            result = await run_task
            latest_lyrics = result.lyrics or latest_lyrics
            progress.overall_status = "error" if result.error else "done"
            if result.error:
                progress.error_message = result.error

            yield (
                progress.format_progress(),
                latest_lyrics,
                _format_conversation_log(memory),
            )

    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        progress.overall_status = "error"
        progress.error_message = str(exc)
        yield progress.format_progress(), f"Error: {exc}", f"**Error:** {exc}"
