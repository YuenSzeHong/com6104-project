"""
Pipeline runner with progress tracking.

Subscribes to AgentOrchestrator events and streams progress updates,
without duplicating orchestration business logic.
"""

# ruff: noqa: E501

from __future__ import annotations

import logging
from pathlib import Path
import asyncio
import time
from html import escape
from typing import AsyncGenerator

from .progress import PipelineProgress

logger = logging.getLogger("gui.pipeline")


_ACTOR_ORDER = [
    "orchestrator",
    "midi-analyzer",
    "melody-mapper",
    "jyutping",
    "lyrics-composer",
    "validator",
    "word-selector",
]


def _friendly_step_name(step: str) -> str:
    """Map internal step keys to user-facing labels."""
    return {
        "midi_analysis": "midi-analyzer",
        "melody_mapping": "melody-mapper",
        "jyutping_conversion": "jyutping",
        "candidate_query": "jyutping + orchestrator",
        "lyrics_composition": "lyrics-composer",
        "lyrics_validation": "validator",
    }.get(step, step or "-")


def _format_agent_status_panel(state: dict[str, str]) -> str:
    """Render a concise status card showing current pipeline actor and state."""
    run_state = escape(state.get("run_state", "idle"))
    current_step = escape(state.get("current_step", "-"))
    current_actor = escape(state.get("current_actor", "-"))
    attempt = escape(state.get("attempt", "-"))
    last_score = escape(state.get("last_score", "-"))

    status_line = f"**{run_state}** | Step: {current_step} | Agent: {current_actor} | Attempt: {attempt}"
    if last_score and last_score != "-":
        status_line += f" | Score: {last_score}"
    
    return status_line


def _format_activity_panel(events: list[dict[str, str]], status: str) -> str:
    """Render a clean timeline with collapsible agent details."""
    if not events:
        return "Waiting for pipeline to start..."

    recent = events[-40:]
    
    # Build main timeline (summary)
    timeline_lines = []
    for item in recent:
        ts = item.get("time", "")[-8:]  # Just HH:MM:SS
        typ = item.get("type", "")
        detail = item.get("detail", "")[:100]
        
        if typ == "step_start":
            timeline_lines.append(f"▶ {ts} | {detail}")
        elif typ == "step_complete":
            timeline_lines.append(f"✓ {ts} | {detail}")
        elif typ == "score":
            timeline_lines.append(f"📊 {ts} | {detail}")
        elif typ == "attempt":
            timeline_lines.append(f"━ {ts} | {detail}")
        elif typ == "error":
            timeline_lines.append(f"⚠ {ts} | {detail}")
        else:
            timeline_lines.append(f"• {ts} | {detail}")
    
    timeline = "\n".join(timeline_lines[-16:])  # Last 16 events
    
    # Group by actor for expandable details
    grouped: dict[str, list[str]] = {}
    for item in recent:
        actor = str(item.get("actor", "orchestrator") or "orchestrator")
        ts = item.get("time", "")[-8:]
        detail = item.get("detail", "")
        grouped.setdefault(actor, []).append(f"[{ts}] {detail}")
    
    details_html = ""
    for actor in _ACTOR_ORDER:
        if actor in grouped:
            items = grouped[actor]
            details_html += (
                f"<details style='margin:6px 0;'>"
                f"<summary style='cursor:pointer;font-weight:600;'>{escape(actor)} ({len(items)} events)</summary>"
                f"<div style='margin-top:8px;padding-left:12px;border-left:2px solid #3a3a3a;'>"
                f"{'<br/>'.join(escape(item) for item in items[-8:])}"
                f"</div>"
                f"</details>"
            )
    
    return f"{timeline}\n\n**Details by Agent:**\n{details_html}"


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
    """Format conversation history grouped by agent—collapsed by default, expandable."""
    if not memory or not hasattr(memory, "_turns") or not memory._turns:
        return "No conversation messages yet."
    
    grouped_blocks: dict[str, list[dict]] = {}
    
    for turn in memory._turns:
        messages = turn.messages if hasattr(turn, "messages") else turn
        turn_agent = ""
        if hasattr(turn, "metadata") and isinstance(turn.metadata, dict):
            turn_agent = str(turn.metadata.get("agent", ""))
        bucket = turn_agent or "orchestrator"
        grouped_blocks.setdefault(bucket, [])
        
        for msg in messages:
            msg_type = msg.__class__.__name__
            content = msg.content if hasattr(msg, "content") else str(msg)
            grouped_blocks[bucket].append({
                "type": msg_type,
                "content": str(content)[:500] if msg_type == "AIMessage" else str(content)[:300],
                "full_content": str(content)
            })
    
    # Build collapsible sections by agent
    sections: list[str] = []
    section_keys = [k for k in _ACTOR_ORDER if k in grouped_blocks]
    section_keys.extend(k for k in grouped_blocks if k not in section_keys)
    
    for actor in section_keys:
        msgs = grouped_blocks[actor]
        user_count = sum(1 for m in msgs if m["type"] == "HumanMessage")
        ai_count = sum(1 for m in msgs if m["type"] == "AIMessage")
        
        msg_preview = ""
        for msg in msgs[-3:]:  # Show last 3 messages
            if msg["type"] == "HumanMessage":
                msg_preview += f"<div style='margin:4px 0;padding:4px;background:#0f172a;border-radius:4px;'><strong>User:</strong> {escape(msg['content'][:80])}</div>"
            elif msg["type"] == "AIMessage":
                msg_preview += f"<div style='margin:4px 0;padding:4px;background:#1a1f3a;border-radius:4px;'><strong>Agent:</strong> {escape(msg['content'][:80])}</div>"
        
        sections.append(
            f"<details style='margin:8px 0;border:1px solid #2f2f2f;border-radius:6px;padding:8px;'>"
            f"<summary style='cursor:pointer;font-weight:600;'>{escape(actor)} ({user_count} inputs, {ai_count} responses)</summary>"
            f"<div style='margin-top:8px;padding:8px;border-top:1px solid #2f2f2f;'>"
            f"{msg_preview}"
            f"</div>"
            f"</details>"
        )
    
    return "".join(sections) if sections else "No conversation messages yet."


async def run_pipeline_with_progress(
    midi_path: str,
    reference_text: str,
    text_file: str | None = None,
    session_id: str | None = None,
) -> AsyncGenerator[tuple[str, str, str, str, str], None]:
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
    (
        progress_markdown,
        agent_status_markdown,
        lyrics_text,
        activity_panel_html,
        conversation_log_html,
    ) tuples
    """
    # Resolve text input
    reference_text_kind = "theme"
    if text_file:
        try:
            reference_text = read_text_file(text_file)
            reference_text_kind = "original_lyrics"
        except Exception as e:
            yield (
                f"## ❌ Error\n\nFailed to read text file: {e}",
                "**Run State:** error",
                "",
                _format_activity_panel([], "error"),
                "<div style='padding:10px;border:1px solid #3a3a3a;border-radius:8px;'>Failed to read text file</div>",
            )
            return

    if not reference_text:
        yield (
            "## ⚠️ Warning\n\nNo reference text provided.",
            "**Run State:** warning",
            "",
            _format_activity_panel([], "warning"),
            "<div style='padding:10px;border:1px solid #3a3a3a;border-radius:8px;'>Missing reference text</div>",
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
    recent_events: list[dict[str, str]] = []
    status_state: dict[str, str] = {
        "run_state": "running",
        "current_step": "-",
        "current_actor": "-",
        "active_tool": "-",
        "attempt": "-",
        "last_score": "-",
    }

    # Yield initial state
    yield (
        progress.format_progress(),
        _format_agent_status_panel(status_state),
        "",
        _format_activity_panel(recent_events, "running"),
        "<div style='padding:10px;border:1px solid #3a3a3a;border-radius:8px;'>Waiting for pipeline to start...</div>",
    )

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
                    reference_text_kind=reference_text_kind,
                    event_callback=_on_event,
                )
            )

            last_emit_at = 0.0
            last_progress_md = ""
            last_status_md = ""
            last_activity_md = ""
            last_conversation_md = ""
            last_lyrics_emitted = ""

            while not run_task.done() or not event_queue.empty():
                drained = False
                while not event_queue.empty():
                    drained = True
                    event = await event_queue.get()
                    ev_type = str(event.get("type", ""))
                    ev_step = str(event.get("step", ""))
                    ev_message = str(event.get("message", ""))
                    actor = "orchestrator"

                    if ev_step:
                        actor = _friendly_step_name(ev_step)

                    if ev_type == "tool_error":
                        server = str(event.get("server", ""))
                        tool = str(event.get("tool", ""))
                        actor = server or actor
                        status_state["active_tool"] = (
                            tool or status_state["active_tool"]
                        )

                    if ev_type == "word_selector_applied":
                        actor = "word-selector"

                    if ev_type == "attempt_started":
                        status_state["attempt"] = str(event.get("attempt", "-"))

                    if ev_type == "step_started" and ev_step:
                        status_state["current_step"] = ev_step
                        status_state["current_actor"] = _friendly_step_name(ev_step)
                        status_state["active_tool"] = "-"

                    if ev_type == "step_completed" and ev_step == "lyrics_validation":
                        score = event.get("score")
                        if isinstance(score, (int, float)):
                            status_state["last_score"] = f"{float(score):.2f}"

                    recent_events.append(
                        {
                            "time": time.strftime("%H:%M:%S"),
                            "type": ev_type or "event",
                            "detail": ev_message or ev_step or "-",
                            "actor": actor,
                        }
                    )
                    if len(recent_events) > 40:
                        recent_events = recent_events[-40:]

                    if ev_type == "run_started":
                        progress.overall_status = "running"
                        status_state["run_state"] = "running"
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
                        status_state["run_state"] = "error"
                    elif ev_type == "run_completed":
                        if event.get("error"):
                            progress.overall_status = "error"
                            progress.error_message = str(event.get("error"))
                            status_state["run_state"] = "error"
                        else:
                            progress.overall_status = "done"
                            status_state["run_state"] = "done"
                        lyrics_value = event.get("lyrics")
                        if isinstance(lyrics_value, str):
                            latest_lyrics = lyrics_value

                now = time.monotonic()
                heartbeat_due = (now - last_emit_at) >= 0.8
                should_emit = drained or heartbeat_due

                if should_emit:
                    progress_md = progress.format_progress()
                    status_md = _format_agent_status_panel(status_state)
                    activity_md = _format_activity_panel(
                        recent_events, progress.overall_status
                    )
                    conversation_md = _format_conversation_log(memory)
                    has_delta = (
                        progress_md != last_progress_md
                        or status_md != last_status_md
                        or activity_md != last_activity_md
                        or conversation_md != last_conversation_md
                        or latest_lyrics != last_lyrics_emitted
                    )

                    if has_delta or drained:
                        last_emit_at = now
                        last_progress_md = progress_md
                        last_status_md = status_md
                        last_activity_md = activity_md
                        last_conversation_md = conversation_md
                        last_lyrics_emitted = latest_lyrics

                        yield (
                            progress_md,
                            status_md,
                            latest_lyrics,
                            activity_md,
                            conversation_md,
                        )
                    else:
                        await asyncio.sleep(0.1)
                else:
                    await asyncio.sleep(0.1)

            result = await run_task
            latest_lyrics = result.lyrics or latest_lyrics
            progress.overall_status = "error" if result.error else "done"
            if result.error:
                progress.error_message = result.error

            yield (
                progress.format_progress(),
                _format_agent_status_panel(status_state),
                latest_lyrics,
                _format_activity_panel(recent_events, progress.overall_status),
                _format_conversation_log(memory),
            )

    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        progress.overall_status = "error"
        progress.error_message = str(exc)
        yield (
            progress.format_progress(),
            _format_agent_status_panel(
                {
                    "run_state": "error",
                    "current_step": "-",
                    "current_actor": "-",
                    "active_tool": "-",
                    "attempt": "-",
                    "last_score": "-",
                }
            ),
            f"Error: {exc}",
            _format_activity_panel(recent_events, "error"),
            (
                "<div style='padding:10px;border:1px solid #3a3a3a;border-radius:8px;'>"
                f"Error: {escape(str(exc))}"
                "</div>"
            ),
        )
