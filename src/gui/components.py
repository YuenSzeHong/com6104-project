"""
Gradio UI component builders.

Each function returns a configured Gradio component.
"""

from __future__ import annotations

import gradio as gr


def build_input_panel() -> tuple[
    gr.File,
    gr.Textbox,
    gr.File,
    gr.Textbox,
    gr.Button,
]:
    """Build the left-column input panel.

    Returns
    -------
    (midi_input, text_input, text_file_input, session_id_input, run_btn)
    """
    gr.Markdown("### 📥 Input")
    gr.Markdown(
        "Use the file upload for original lyrics or source text. "
        "Use the textbox for a theme, idea, or short backup note."
    )

    midi_input = gr.File(
        label="MIDI File",
        file_types=[".mid", ".midi"],
        type="filepath",
        elem_id="midi-input",
    )

    text_file_input = gr.File(
        label="Original Lyrics / Source Text File",
        file_types=[".txt", ".md"],
        type="filepath",
        elem_id="text-file-input",
    )

    text_input = gr.Textbox(
        label="Idea / Theme / Backup Text",
        placeholder=(
            "Paste a theme, idea, or short reference if you do not have a source file."
        ),
        lines=3,
        elem_id="text-input",
    )

    session_id_input = gr.Textbox(
        label="Session ID (optional)",
        placeholder="auto-generated if empty",
        value="",
        elem_id="session-id-input",
    )

    run_btn = gr.Button(
        "🎵 Generate Lyrics",
        variant="primary",
        size="lg",
        elem_id="run-button",
    )

    return midi_input, text_input, text_file_input, session_id_input, run_btn


def build_output_panel() -> tuple[
    gr.Markdown,
    gr.HTML,
    gr.Textbox,
    gr.HTML,
    gr.HTML,
    gr.Button,
    gr.Textbox,
]:
    """Build the right-column output panel.

    Returns
    -------
    (
        progress_output,
        agent_status_output,
        lyrics_output,
        activity_log,
        conversation_log,
        save_btn,
        save_path,
    )
    """
    gr.Markdown("### 📊 Progress")

    progress_output = gr.Markdown(
        value="*Upload a MIDI file and click 'Generate Lyrics' to start.*",
        elem_id="progress-output",
    )

    gr.Markdown("### 🧭 Agent Status")

    agent_status_output = gr.HTML(
        value=(
            "<div class='status-empty'>"
            "No run yet. Start the pipeline to see active agents and tools."
            "</div>"
        ),
        elem_id="agent-status-output",
    )

    gr.Markdown("### 📝 Generated Lyrics")

    lyrics_output = gr.Textbox(
        label="Lyrics",
        lines=12,
        interactive=False,
        elem_id="lyrics-output",
    )

    gr.Markdown("### 📡 Live Activity")

    activity_log = gr.HTML(
        value=(
            "<div class='panel-empty'>"
            "Recent agent activity will stream here in real time."
            "</div>"
        ),
        max_height=320,
        autoscroll=True,
        elem_id="activity-log",
    )

    gr.Markdown("### 💬 Agent Conversation Log")

    conversation_log = gr.HTML(
        value=(
            "<div class='panel-empty'>"
            "Agent conversations will appear here during pipeline execution."
            "</div>"
        ),
        max_height=360,
        autoscroll=True,
        elem_id="conversation-log",
    )

    save_btn = gr.Button(
        "💾 Save Lyrics",
        variant="secondary",
        elem_id="save-button",
    )

    save_path = gr.Textbox(
        label="Save Path",
        placeholder="Leave empty to save next to MIDI file",
        elem_id="save-path",
    )

    return (
        progress_output,
        agent_status_output,
        lyrics_output,
        activity_log,
        conversation_log,
        save_btn,
        save_path,
    )
