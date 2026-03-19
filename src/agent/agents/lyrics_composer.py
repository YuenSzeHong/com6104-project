"""
LyricsComposerAgent – Stage 3 of the Cantonese Lyrics pipeline.

Responsibilities
----------------
- Read the MIDI analysis and Jyutping mapping from shared memory context.
- Compose Cantonese lyrics that:
    1. Match the MIDI syllable count exactly.
    2. Follow Cantonese tonal rules at strong beats.
    3. Maintain rhyme scheme at line/phrase endings.
    4. Preserve the artistic mood of the reference text.
- Support revision mode: when given validator feedback, incorporate the
  corrections and re-generate improved lyrics.
- Return structured output:
    {
        "lyrics":    "<Chinese lyrics>",
        "jyutping":  "<space-separated Jyutping romanisation>",
        "lines":     ["line1", "line2", ...],
        "attempt":   <int>,
    }

Context keys read
~~~~~~~~~~~~~~~~~
- ``midi_analysis``  (dict) – syllable_count, bpm, strong_beat_positions
- ``jyutping_map``   (dict) – melody_tone_sequence_0243, rhyme_positions, selected_jyutping

Context keys written
~~~~~~~~~~~~~~~~~~~~
- ``draft_lyrics``   (dict) – the raw structured result from this agent
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from agent.base_agent import BaseAgent, AgentResult
from agent.config import PROMPTS_DIR
from agent.registry import AGENT_REGISTRY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tone-fitness helpers
# ---------------------------------------------------------------------------

# Tones considered "stable" – good for landing on strong melody beats
_STABLE_TONES = {1, 3, 5}

# Tones that are "light/rising" – flow well on weak passing beats
_FLOWING_TONES = {2, 4, 6}

# Cantonese finals that are common rhyme endings
_COMMON_RHYME_FINALS = {
    "aa", "aai", "aau", "aam", "aan", "aang", "aap", "aat", "aak",
    "ai", "au", "am", "an", "ang", "ap", "at", "ak",
    "e", "ei", "eu", "em", "eng", "ep", "ek",
    "i", "iu", "im", "in", "ing", "ip", "it", "ik",
    "o", "oi", "ou", "on", "ong", "ot", "ok",
    "u", "ui", "un", "ung", "ut", "uk",
    "oe", "oei", "oeng", "oet", "oek",
    "yu", "yun", "yut",
    "m", "ng",
}


def _get_rhyme_final(jyutping_syllable: str) -> str:
    """
    Extract the rhyme final (vowel nucleus + coda) from a Jyutping syllable.

    Examples: "sing1" → "ing",  "maau5" → "aau",  "zau2" → "au"
    """
    # Strip tone number
    syllable = re.sub(r"[1-6]$", "", jyutping_syllable.lower())

    # Common initials to strip
    _INITIALS: tuple[str, ...] = (
        "ng", "gw", "kw",
        "b", "p", "m", "f",
        "d", "t", "n", "l",
        "g", "k", "h",
        "z", "c", "s",
        "j", "w",
    )
    for initial in sorted(_INITIALS, key=len, reverse=True):
        if syllable.startswith(str(initial)):
            return syllable[len(initial):]

    return syllable


def _rhyme_matches(syl_a: str, syl_b: str) -> bool:
    """Return True if two Jyutping syllables share the same rhyme final."""
    return _get_rhyme_final(syl_a) == _get_rhyme_final(syl_b) and bool(syl_a)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@AGENT_REGISTRY.register("lyrics-composer")
class LyricsComposerAgent(BaseAgent):
    """
    Generates Cantonese lyrics constrained by melody structure and tonal rules.

    The agent operates in two modes:

    First-draft mode
    ~~~~~~~~~~~~~~~~
    Called with a plain composition task string.  Reads midi_analysis and
    jyutping_map from shared memory and composes an initial draft.

    Revision mode
    ~~~~~~~~~~~~~
    Called with a task string that begins with ``[第 N 次修改]`` or
    ``[REVISION ATTEMPT N]``.
    Reads the previous draft and validator feedback from context and produces
    an improved version that addresses the specific corrections requested.
    """

    # ------------------------------------------------------------------
    # Entry-point
    # ------------------------------------------------------------------

    async def _execute(self, task: str, **kwargs: Any) -> AgentResult:
        attempt: int = self._parse_attempt_number(task)
        is_revision: bool = attempt > 0

        # ----------------------------------------------------------------
        # Pull shared pipeline context
        # ----------------------------------------------------------------
        midi: dict[str, Any] = self._memory.get_pipeline_value("midi_analysis", {}) or {}
        jmap: dict[str, Any] = self._memory.get_pipeline_value("jyutping_map", {}) or {}
        validation: dict[str, Any] = self._memory.get_validation_result()

        syllable_count: int = int(midi.get("syllable_count", 0))
        strong_beats: list[int] = midi.get("strong_beat_positions", [])
        melody_tone_sequence: list[int] = jmap.get("melody_tone_sequence_0243", [])
        rhyme_positions: list[int] = jmap.get("rhyme_positions", [])
        reference_text: str = jmap.get("reference_text", "")

        self._log.info(
            "Composing lyrics | attempt=%d | syllables=%d | revision=%s",
            attempt + 1, syllable_count, is_revision,
        )

        # ----------------------------------------------------------------
        # Build the composition prompt
        # ----------------------------------------------------------------
        if is_revision:
            prompt = self._build_revision_prompt(
                task=task,
                syllable_count=syllable_count,
                strong_beats=strong_beats,
                melody_tone_sequence=melody_tone_sequence,
                rhyme_positions=rhyme_positions,
                reference_text=reference_text,
                validation=validation,
            )
        else:
            prompt = self._build_composition_prompt(
                task=task,
                syllable_count=syllable_count,
                strong_beats=strong_beats,
                melody_tone_sequence=melody_tone_sequence,
                rhyme_positions=rhyme_positions,
                reference_text=reference_text,
                jyutping_map=jmap,
                midi=midi,
            )

        # ----------------------------------------------------------------
        # Call the LLM (with tools if available, plain otherwise)
        # ----------------------------------------------------------------
        self._memory.add_user_message(prompt)

        if self._tools:
            # Use tool-calling executor so the LLM can validate Jyutping
            # candidates via the jyutping MCP server mid-composition
            raw_response = await self._invoke_with_tools(prompt)
        else:
            raw_response = await self._invoke_llm()

        # ----------------------------------------------------------------
        # Parse the LLM response
        # ----------------------------------------------------------------
        structured = self._parse_llm_response(
            raw_response,
            syllable_count=syllable_count,
            tone_sequence=melody_tone_sequence,
        )
        structured["attempt"] = attempt + 1

        # ----------------------------------------------------------------
        # Quality self-check before handing off to validator
        # ----------------------------------------------------------------
        self_check = self._self_check(structured, syllable_count, rhyme_positions)
        structured["self_check"] = self_check

        if not self_check["syllable_count_ok"]:
            self._log.warning(
                "Self-check FAILED: expected %d syllables, got %d",
                syllable_count,
                self_check["actual_syllable_count"],
            )

        self._log.info(
            "Composition done | syllables_ok=%s | rhyme_ok=%s",
            self_check["syllable_count_ok"],
            self_check["rhyme_ok"],
        )

        return AgentResult(
            agent_name=self.name,
            success=True,
            output=structured.get("lyrics", ""),
            data={"draft_lyrics": structured},
            metadata={
                "attempt": attempt + 1,
                "syllable_count": syllable_count,
                "self_check": self_check,
            },
        )

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_composition_prompt(
        self,
        task: str,
        syllable_count: int,
        strong_beats: list[int],
        melody_tone_sequence: list[int],
        rhyme_positions: list[int],
        reference_text: str,
        jyutping_map: dict[str, Any],
        midi: dict[str, Any],
    ) -> str:
        """Build the first-draft composition prompt."""
        tone_seq_str = (
            " ".join(str(t) for t in melody_tone_sequence[:syllable_count])
            if melody_tone_sequence
            else "(derive from lean 0243 melody mapping)"
        )
        strong_str = (
            ", ".join(str(b) for b in strong_beats)
            if strong_beats
            else "unknown"
        )
        rhyme_str = (
            ", ".join(str(r) for r in rhyme_positions)
            if rhyme_positions
            else "phrase endings"
        )

        selected_jp: str = jyutping_map.get("selected_jyutping", "")
        breakdown: list[dict] = jyutping_map.get("syllable_breakdown", [])
        breakdown_str = (
            "\n".join(
                f"  {item.get('char', '?')} → {item.get('jyutping', '?')}"
                for item in breakdown[:20]
            )
            if breakdown
            else "(not available)"
        )

        tempo: float = float(midi.get("bpm", 0))
        key: str = midi.get("key", "unknown")
        embedded_source: str = str(midi.get("embedded_lyrics_source", "")) or "(not available)"
        embedded_units: list[str] = midi.get("embedded_lyrics_preview", []) or []
        embedded_count: int = int(midi.get("embedded_lyric_unit_count", 0))
        effective_count_source: str = str(
            midi.get("effective_syllable_count_source", "melody_notes")
        )
        embedded_str = (
            " ".join(str(unit) for unit in embedded_units[:32])
            if embedded_units else "(not available)"
        )

        return self._render_prompt_template(
            "lyrics-composer-task.md",
            syllable_count=syllable_count,
            tempo=f"{tempo:.0f}",
            key=key,
            strong_str=strong_str,
            rhyme_str=rhyme_str,
            reference_text=reference_text or "(not provided)",
            embedded_source=embedded_source,
            embedded_count=embedded_count,
            effective_count_source=effective_count_source,
            embedded_str=embedded_str,
            selected_jp=selected_jp or "(not available)",
            breakdown_str=breakdown_str,
            tone_seq_str=tone_seq_str,
        )

    def _build_revision_prompt(
        self,
        task: str,
        syllable_count: int,
        strong_beats: list[int],
        melody_tone_sequence: list[int],
        rhyme_positions: list[int],
        reference_text: str,
        validation: dict[str, Any],
    ) -> str:
        """Build a revision prompt that incorporates validator feedback."""
        previous_draft: dict[str, Any] = self._memory.get_current_draft()
        prev_lyrics: str = previous_draft.get("lyrics", "(not available)")
        prev_jyutping: str = previous_draft.get("jyutping", "")

        score: float = float(validation.get("score", 0.0))
        feedback: str = validation.get("feedback", "")
        corrections: list[str] = validation.get("corrections", [])
        corrections_str = (
            "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(corrections))
            if corrections
            else "  (see feedback above)"
        )

        tone_seq_str = (
            " ".join(str(t) for t in melody_tone_sequence[:syllable_count])
            if melody_tone_sequence
            else "(derive from melody)"
        )

        # Extract attempt number from task string for logging clarity
        attempt_match = (
            re.search(r"REVISION ATTEMPT (\d+)", task, re.IGNORECASE)
            or re.search(r"\[第\s*(\d+)\s*次修改\]", task)
        )
        attempt_label = (
            f"Revision Attempt {attempt_match.group(1)}" if attempt_match
            else "Revision"
        )

        return self._render_prompt_template(
            "lyrics-composer-revision-task.md",
            attempt_label=attempt_label,
            score=score,
            prev_lyrics=prev_lyrics,
            prev_jyutping=prev_jyutping or "(not available)",
            feedback=feedback or "(no detailed feedback provided)",
            corrections_str=corrections_str,
            syllable_count=syllable_count,
            strong_str=", ".join(str(b) for b in strong_beats) or "unknown",
            rhyme_str=", ".join(str(r) for r in rhyme_positions) or "phrase endings",
            reference_text=reference_text or "(not provided)",
            tone_seq_str=tone_seq_str,
        )

    @staticmethod
    def _render_prompt_template(template_name: str, **kwargs: Any) -> str:
        template_path: Path = PROMPTS_DIR / template_name
        return template_path.read_text(encoding="utf-8").format(**kwargs).strip()

    # ------------------------------------------------------------------
    # Response parser
    # ------------------------------------------------------------------

    def _parse_llm_response(
        self,
        raw: str,
        syllable_count: int,
        tone_sequence: list[int],
    ) -> dict[str, Any]:
        """
        Parse the LLM's JSON response into a typed dict.

        Gracefully handles:
        - Properly formatted JSON
        - JSON wrapped in markdown fences
        - Partial / malformed JSON (best-effort extraction)
        - Plain text (treated as the lyrics with no Jyutping)
        """
        # 1. Try to extract a JSON block
        json_str = self._extract_json_block(raw)

        parsed: dict[str, Any] = {}
        if json_str:
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError as exc:
                self._log.warning(
                    "Could not parse LLM JSON: %s. Attempting field extraction.", exc
                )
                parsed = self._extract_fields_heuristically(raw)
        else:
            # No JSON found – treat the whole response as the lyrics
            self._log.warning(
                "LLM did not return JSON. Treating raw response as lyrics."
            )
            parsed = {"lyrics": raw.strip()}

        # 2. Normalise / fill missing fields
        lyrics: str = parsed.get("lyrics", "").strip()
        jyutping: str = parsed.get("jyutping", "").strip()
        lines: list[dict] = parsed.get("lines", [])

        # If lines list is empty but lyrics is populated, split by newline
        if lyrics and not lines:
            lines = self._split_lyrics_to_lines(lyrics, jyutping)

        # Fill syllable_count per line if missing
        for line in lines:
            if not line.get("syllable_count"):
                jp = line.get("jyutping", "")
                line["syllable_count"] = len(re.findall(r"[a-z]+[1-6]", jp, re.I))

        return {
            "lyrics":        lyrics,
            "jyutping":      jyutping,
            "lines":         lines,
            "rhyme_scheme":  parsed.get("rhyme_scheme", ""),
            "changes_made":  parsed.get("changes_made", ""),
            "notes":         parsed.get("notes", ""),
            "target_syllable_count": syllable_count,
        }

    # ------------------------------------------------------------------
    # Self-check
    # ------------------------------------------------------------------

    def _self_check(
        self,
        structured: dict[str, Any],
        syllable_count: int,
        rhyme_positions: list[int],
    ) -> dict[str, Any]:
        """
        Perform a lightweight local quality check on the composed lyrics.

        Returns a dict with:
            syllable_count_ok     – bool
            actual_syllable_count – int
            rhyme_ok              – bool (True when no rhyme positions specified)
            issues                – list[str] of human-readable problems found
        """
        issues: list[str] = []

        jyutping: str = structured.get("jyutping", "")
        syllables = re.findall(r"[a-z]+[1-6]", jyutping, re.IGNORECASE)
        actual_count = len(syllables)

        # --- Syllable count ---
        syllable_ok = (syllable_count == 0) or (actual_count == syllable_count)
        if not syllable_ok:
            issues.append(
                f"Syllable count mismatch: expected {syllable_count}, "
                f"got {actual_count}."
            )

        # --- Rhyme consistency ---
        rhyme_ok = True
        if rhyme_positions and len(syllables) > 0:
            rhyme_syllables = [
                syllables[i] for i in rhyme_positions if i < len(syllables)
            ]
            if len(rhyme_syllables) >= 2:
                finals = [_get_rhyme_final(s) for s in rhyme_syllables]
                # Check that at least 50% of rhyme positions share the same final
                most_common_final = max(set(finals), key=finals.count)
                matching = finals.count(most_common_final)
                if matching / len(finals) < 0.5:
                    rhyme_ok = False
                    issues.append(
                        "Rhyme inconsistency: fewer than 50% of rhyme positions "
                        f"share a final vowel. Finals found: {finals}"
                    )

        # --- Empty lyrics ---
        lyrics: str = structured.get("lyrics", "")
        if not lyrics.strip():
            issues.append("Lyrics field is empty.")

        return {
            "syllable_count_ok":     syllable_ok,
            "actual_syllable_count": actual_count,
            "expected_syllable_count": syllable_count,
            "rhyme_ok":              rhyme_ok,
            "issues":                issues,
            "passed":                syllable_ok and rhyme_ok and bool(lyrics.strip()),
        }

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_attempt_number(task: str) -> int:
        """
        Extract the 0-based attempt index from a revision task string.

        ``[第 2 次修改]`` / ``[REVISION ATTEMPT 2]`` → 1  (0-based)
        A plain task string (no revision marker) → 0
        """
        match = (
            re.search(r"REVISION ATTEMPT\s+(\d+)", task, re.IGNORECASE)
            or re.search(r"\[第\s*(\d+)\s*次修改\]", task)
        )
        if match:
            return int(match.group(1)) - 1
        return 0

    @staticmethod
    def _extract_json_block(text: str) -> str:
        """
        Extract the first JSON object from *text*.

        Handles markdown code fences (```json ... ```) and bare JSON objects.
        """
        # Strip markdown fence
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fenced:
            return fenced.group(1).strip()

        # Find balanced { ... }
        start = text.find("{")
        if start == -1:
            return ""

        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        return ""

    @staticmethod
    def _extract_fields_heuristically(text: str) -> dict[str, Any]:
        """
        Best-effort extraction of key fields when JSON parsing fails.

        Looks for patterns like:  "lyrics": "...",
        """
        result: dict[str, Any] = {}

        for key in ("lyrics", "jyutping", "rhyme_scheme", "notes", "changes_made"):
            pattern = rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                result[key] = match.group(1).replace("\\n", "\n").replace('\\"', '"')

        return result

    @staticmethod
    def _split_lyrics_to_lines(
        lyrics: str, jyutping: str
    ) -> list[dict[str, Any]]:
        """
        Split a flat lyrics string into line dicts.

        Tries to pair Chinese lines with Jyutping lines; falls back to
        splitting only on the Chinese text if Jyutping is unavailable.
        """
        chin_lines = [ln.strip() for ln in lyrics.splitlines() if ln.strip()]
        jp_lines = [ln.strip() for ln in jyutping.splitlines() if ln.strip()]

        result: list[dict[str, Any]] = []
        for i, chin in enumerate(chin_lines):
            jp = jp_lines[i] if i < len(jp_lines) else ""
            syl_count = len(re.findall(r"[a-z]+[1-6]", jp, re.IGNORECASE))
            result.append({
                "text": chin,
                "jyutping": jp,
                "syllable_count": syl_count,
            })
        return result
