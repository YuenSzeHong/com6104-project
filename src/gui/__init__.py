"""
GUI package – Gradio web interface for the Cantonese Lyrics Agent.

Structure
---------
gui/
  __init__.py      – package entry point
  progress.py      – progress tracking dataclasses
  pipeline.py      – pipeline runner with progress yields
  components.py    – Gradio UI component builders
  handlers.py      – event handler functions
  app.py           – main Gradio Blocks assembly
  main.py          – CLI entry point (python -m gui.main)
"""

from .app import create_ui

__all__ = ["create_ui"]
