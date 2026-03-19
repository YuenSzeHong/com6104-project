"""
Concrete agent implementations for the Cantonese Lyrics pipeline.

Architecture note
-----------------
MidiAnalyserAgent and JyutpingMapperAgent have been removed as standalone
agent classes. Their computational skills now live in MCP servers:

  - midi-analyzer MCP server  →  analyze_midi, get_syllable_durations,
                                  suggest_rhyme_positions
  - jyutping MCP server       →  chinese_to_jyutping, get_tone_pattern,
                                  get_tone_code, find_words_by_tone_code,
                                  find_tone_continuation

The orchestrator calls those tools directly (no LLM) in Step 1 of run().
Only the two agents that genuinely require LLM reasoning remain here.

Agents
------
LyricsComposerAgent  – generates Cantonese lyrics using all MCP tools + LLM
ValidatorAgent       – calls lyrics-validator MCP tools for computational
                       scoring, then adds LLM artistic quality judgment
"""

from .lyrics_composer import LyricsComposerAgent
from .validator import ValidatorAgent

__all__ = [
    "LyricsComposerAgent",
    "ValidatorAgent",
]
