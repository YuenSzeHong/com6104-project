# ruff: noqa: E501
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
        gr.Markdown("## Cantonese Lyrics Agent")
        gr.Markdown(
            "Generate Cantonese lyrics from MIDI, then review the pipeline output below."
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                with gr.Group():
                    gr.Markdown("### Input")
                    (
                        midi_input,
                        text_input,
                        text_file_input,
                        session_id_input,
                        run_btn,
                    ) = build_input_panel()

            with gr.Column(scale=2, min_width=520):
                with gr.Group():
                    gr.Markdown("### Output")
                    (
                        progress_output,
                        agent_status_output,
                        lyrics_output,
                        activity_log,
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
            outputs=[
                progress_output,
                agent_status_output,
                lyrics_output,
                activity_log,
                conversation_log,
            ],
        )

        save_btn.click(
            fn=handle_save_lyrics,
            inputs=[lyrics_output, midi_input, save_path],
            outputs=[gr.Textbox(label="Save Status")],
        )

    return app
