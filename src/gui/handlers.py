"""
Event handler functions for the Gradio UI.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from .pipeline import read_text_file


def handle_text_input(
    text: str | None,
    text_file: str | None,
) -> str:
    """Read text from file if provided, otherwise use direct text."""
    if text_file:
        try:
            return read_text_file(text_file)
        except Exception as e:
            raise gr.Error(f"Failed to read text file: {e}")
    return text or ""


def handle_save_lyrics(
    lyrics: str,
    midi_file: str | None,
    save_path: str | None,
) -> str:
    """Save lyrics to file and return status message."""
    if not lyrics or lyrics.startswith("Error:"):
        return "⚠️ No lyrics to save."

    if save_path:
        path = Path(save_path)
    elif midi_file:
        path = Path(midi_file).with_suffix(".lyrics.txt")
    else:
        path = Path("lyrics.txt")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(lyrics, encoding="utf-8")
    return f"✅ Saved to: {path}"
