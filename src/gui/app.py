"""
Main Gradio Blocks assembly.

Wires together components, handlers, and pipeline runner.
"""

from __future__ import annotations

import gradio as gr

from .components import build_input_panel, build_output_panel
from .handlers import handle_save_lyrics
from .pipeline import run_pipeline_with_progress


def create_ui() -> gr.Blocks:
    """Create and configure the Gradio interface."""

    with gr.Blocks(title="Cantonese Lyrics Agent") as app:
        gr.Markdown(
            """
# 🎵 Cantonese Lyrics Agent

AI-powered Cantonese lyrics generation from MIDI melodies.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                (
                    midi_input,
                    text_input,
                    text_file_input,
                    session_id_input,
                    run_btn,
                ) = build_input_panel()

            with gr.Column(scale=2):
                (
                    progress_output,
                    lyrics_output,
                    conversation_log,
                    save_btn,
                    save_path,
                ) = build_output_panel()

        # --- Event bindings ---
        # Gradio 6.x: pass the async generator function directly.
        # Gradio will handle awaiting and streaming the yields.

        run_btn.click(
            fn=run_pipeline_with_progress,
            inputs=[midi_input, text_input, text_file_input, session_id_input],
            outputs=[progress_output, lyrics_output, conversation_log],
        )

        save_btn.click(
            fn=handle_save_lyrics,
            inputs=[lyrics_output, midi_input, save_path],
            outputs=[gr.Textbox(label="Save Status")],
        )

    return app
