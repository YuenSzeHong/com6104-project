"""
Gradio UI component builders.

Each function returns a configured Gradio component.
"""

from __future__ import annotations

import gradio as gr


def build_input_panel() -> tuple[gr.File, gr.Textbox, gr.File, gr.Textbox, gr.Button]:
    """Build the left-column input panel.

    Returns
    -------
    (midi_input, text_input, text_file_input, session_id_input, run_btn)
    """
    gr.Markdown("### 📥 Input")

    midi_input = gr.File(
        label="MIDI File",
        file_types=[".mid", ".midi"],
        type="filepath",
    )

    text_input = gr.Textbox(
        label="Reference Text / Theme",
        placeholder="Enter source lyric or theme text...",
        lines=3,
    )

    text_file_input = gr.File(
        label="Or upload text file",
        file_types=[".txt", ".md"],
        type="filepath",
    )

    session_id_input = gr.Textbox(
        label="Session ID (optional)",
        placeholder="auto-generated if empty",
        value="",
    )

    run_btn = gr.Button("🎵 Generate Lyrics", variant="primary", size="lg")

    return midi_input, text_input, text_file_input, session_id_input, run_btn


def build_output_panel() -> tuple[
    gr.Markdown, gr.Textbox, gr.Markdown, gr.Button, gr.Textbox
]:
    """Build the right-column output panel.

    Returns
    -------
    (progress_output, lyrics_output, conversation_log, save_btn, save_path)
    """
    gr.Markdown("### 📊 Progress")

    progress_output = gr.Markdown(
        value="*Upload a MIDI file and click 'Generate Lyrics' to start.*"
    )

    gr.Markdown("### 📝 Generated Lyrics")

    lyrics_output = gr.Textbox(
        label="Lyrics",
        lines=12,
        interactive=False,
    )

    gr.Markdown("### 💬 Agent Conversation Log")

    conversation_log = gr.Markdown(
        value="*Agent conversations will appear here during pipeline execution.*",
    )

    save_btn = gr.Button("💾 Save Lyrics", variant="secondary")

    save_path = gr.Textbox(
        label="Save Path",
        placeholder="Leave empty to save next to MIDI file",
    )

    return progress_output, lyrics_output, conversation_log, save_btn, save_path
