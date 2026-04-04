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

from pydantic import BaseModel, Field

from agent.base_agent import BaseAgent, AgentResult
from agent.config import PROMPTS_AGENTS_DIR, PROVIDER
from agent.registry import AGENT_REGISTRY

# Lazy import WordSelectorAgent to avoid circular dependency
_word_selector_cls = None


def _get_word_selector_class():
    """Lazy import of WordSelectorAgent to avoid circular dependency."""
    global _word_selector_cls
    if _word_selector_cls is None:
        from agent.agents.word_selector import WordSelectorAgent

        _word_selector_cls = WordSelectorAgent
    return _word_selector_cls


logger = logging.getLogger(__name__)


class LyricsLineSchema(BaseModel):
    """Structured line-level output schema for lyric drafts."""

    text: str = Field(default="", description="中文歌词行")
    jyutping: str = Field(default="", description="对应粤拼行")
    syllable_count: int = Field(default=0, ge=0, description="该行音节数")


class LyricsDraftSchema(BaseModel):
    """Structured output schema for the composer agent."""

    lyrics: str = Field(default="", description="完整歌词（可含换行）")
    jyutping: str = Field(default="", description="完整歌词对应粤拼")
    lines: list[LyricsLineSchema] = Field(default_factory=list, description="逐行歌词与粤拼")
    rhyme_scheme: str = Field(default="", description="押韵方案")
    changes_made: str = Field(default="", description="本轮修改说明")
    notes: str = Field(default="", description="补充备注")


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

    Word Selection Integration
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    When 0243.hk API returns too many candidates (>10), the agent automatically
    invokes WordSelectorAgent to select the best word based on:
    - Context (surrounding lyrics)
    - Semantic field / theme
    - Tone matching with melody
    - Rhyme consistency at phrase endings
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
            # If the task string already looks like a formatted prompt (e.g. from orchestrator), use it directly
            if "请创作恰好" in task or "0243 旋律目标" in task:
                prompt = task
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
        self._memory.add_ai_message(
            "[填词代理] 已收到约束，正在生成歌词草稿...",
            metadata={"agent": self.name, "event": "compose_started"},
        )

        structured = await self._compose_with_schema(
            prompt=prompt,
            syllable_count=syllable_count,
            tone_sequence=melody_tone_sequence,
        )
        structured["attempt"] = attempt + 1

        # ----------------------------------------------------------------
        # Word selection refinement: invoke WordSelectorAgent for positions
        # where 0243.hk returned too many candidates
        # ----------------------------------------------------------------
        candidate_map: dict[int, list[str]] = jmap.get("candidate_words_map", {})
        if candidate_map:
            structured = await self._refine_word_selection(
                structured=structured,
                candidate_map=candidate_map,
                melody_tone_sequence=melody_tone_sequence,
                rhyme_positions=rhyme_positions,
                reference_text=reference_text,
            )

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

    async def _compose_with_schema(
        self,
        prompt: str,
        syllable_count: int,
        tone_sequence: list[int],
    ) -> dict[str, Any]:
        """Try schema-constrained output first, then fall back to legacy parsing."""
        structured_payload = None
        attempted_structured = PROVIDER != "ollama-cloud"
        if attempted_structured:
            structured_payload = await self._invoke_llm_structured(
                schema=LyricsDraftSchema,
            )
        else:
            self._memory.add_ai_message(
                "[填词代理] ollama-cloud 结构化输出不稳定，直接使用文本生成。",
                metadata={"agent": self.name, "event": "skip_structured"},
            )
        if structured_payload is not None:
            payload = (
                structured_payload.model_dump()
                if hasattr(structured_payload, "model_dump")
                else dict(structured_payload)
                if isinstance(structured_payload, dict)
                else {}
            )
            normalized = self._normalize_structured_payload(payload)
            if normalized.get("lyrics"):
                return {
                    **normalized,
                    "target_syllable_count": syllable_count,
                }

            self._log.warning("Structured output missing lyrics, fallback to legacy parsing.")

        # Prefer a plain text generation fallback first because some cloud
        # providers return fenced JSON that breaks strict structured parsing.
        # This path is faster and keeps GUI feedback responsive.
        if attempted_structured:
            self._memory.add_ai_message(
                "[填词代理] 结构化输出失败，回退为纯文本生成并做本地解析。",
                metadata={"agent": self.name, "event": "text_fallback"},
            )
        raw_response = await self._invoke_llm()

        # If plain generation is empty, try tool-calling as a last resort.
        if not str(raw_response).strip() and self._tools:
            self._memory.add_ai_message(
                "[填词代理] 纯文本响应为空，回退到工具代理重试。",
                metadata={"agent": self.name, "event": "tool_fallback"},
            )
            raw_response = await self._invoke_with_tools(prompt)

        return self._parse_llm_response(
            raw_response,
            syllable_count=syllable_count,
            tone_sequence=tone_sequence,
        )

    def _normalize_structured_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Normalize schema payload into the same shape expected by downstream logic."""
        lyrics = str(payload.get("lyrics", "")).strip()
        jyutping = str(payload.get("jyutping", "")).strip()

        lines_raw = payload.get("lines", [])
        lines: list[dict[str, Any]] = []
        if isinstance(lines_raw, list):
            for item in lines_raw:
                if not isinstance(item, dict):
                    continue
                line_text = str(item.get("text", "")).strip()
                line_jp = str(item.get("jyutping", "")).strip()
                line_syllable_count = item.get("syllable_count", 0)
                if not isinstance(line_syllable_count, int):
                    line_syllable_count = 0
                lines.append(
                    {
                        "text": line_text,
                        "jyutping": line_jp,
                        "syllable_count": max(0, line_syllable_count),
                    }
                )

        if lyrics and not lines:
            lines = self._split_lyrics_to_lines(lyrics, jyutping)

        for line in lines:
            if not line.get("syllable_count"):
                line["syllable_count"] = len(
                    re.findall(r"[a-z]+[1-6]", str(line.get("jyutping", "")), re.I)
                )

        return {
            "lyrics": lyrics,
            "jyutping": jyutping,
            "lines": lines,
            "rhyme_scheme": str(payload.get("rhyme_scheme", "")),
            "changes_made": str(payload.get("changes_made", "")),
            "notes": str(payload.get("notes", "")),
        }

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
        template_path: Path = PROMPTS_AGENTS_DIR / template_name
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
                    return text[start: i + 1]

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

    # ------------------------------------------------------------------
    # Word selection refinement
    # ------------------------------------------------------------------

    async def _refine_word_selection(
        self,
        structured: dict[str, Any],
        candidate_map: dict[int, list[str]],
        melody_tone_sequence: list[int],
        rhyme_positions: list[int],
        reference_text: str,
    ) -> dict[str, Any]:
        """
        Refine word selection using WordSelectorAgent for positions with many candidates.

        When 0243.hk API returns too many candidates (>10) for a position,
        invoke WordSelectorAgent to select the best word based on context.

        Parameters
        ----------
        structured : dict
            The parsed lyrics structure from LLM
        candidate_map : dict[int, list[str]]
            Map of position -> candidate words from 0243.hk API
        melody_tone_sequence : list[int]
            The melody's tone sequence (0243 system)
        rhyme_positions : list[int]
            Positions that should rhyme (phrase endings)
        reference_text : str
            Reference text / theme for semantic context

        Returns
        -------
        dict
            Updated structured lyrics with refined word selections
        """
        WordSelectorAgent = _get_word_selector_class()

        lyrics_text = structured.get("lyrics", "")

        # Extract current lyrics as a list of characters/words for easy modification
        lyrics_chars = list(lyrics_text.replace("\n", ""))

        # Threshold for invoking word selector
        CANDIDATE_THRESHOLD = 10

        self._log.info(
            "Refining word selection | positions_with_many_candidates=%d",
            sum(
                1
                for cands in candidate_map.values()
                if len(cands) > CANDIDATE_THRESHOLD
            ),
        )

        # Process positions with many candidates
        for position, candidates in sorted(candidate_map.items()):
            if len(candidates) <= CANDIDATE_THRESHOLD:
                continue  # Skip positions with few candidates

            if position >= len(lyrics_chars):
                continue  # Position out of bounds

            # Build selection context
            context = self._build_word_selection_context(
                position=position,
                lyrics_text=lyrics_text,
                melody_tone=melody_tone_sequence[position]
                if position < len(melody_tone_sequence)
                else None,
                is_rhyme_position=position in rhyme_positions,
                reference_text=reference_text,
            )

            # Invoke WordSelectorAgent
            selector = WordSelectorAgent(
                config=self.config,
                llm=self.llm,
                memory=self.memory,
                tools=[],  # Word selector doesn't need tools
            )

            result = await selector.run(
                task=f"为歌词位置 {position} 选择最合适的词语",
                candidates=candidates,
                context=context,
                count=1,
            )

            selected_word = result.output
            if selected_word and len(selected_word) == 1:
                # Replace the character at this position
                lyrics_chars[position] = selected_word
                self._log.info(
                    "Position %d: selected '%s' from %d candidates | reason=%s",
                    position,
                    selected_word,
                    len(candidates),
                    result.metadata.get("selection_reason", "?")[:60],
                )

        # Reconstruct lyrics with selected words
        "".join(lyrics_chars)

        # Re-split into lines preserving original line breaks
        original_lines = lyrics_text.split("\n")
        char_idx = 0
        new_lines_text = []
        for orig_line in original_lines:
            line_len = len(orig_line.strip())
            if line_len > 0:
                new_lines_text.append(
                    "".join(lyrics_chars[char_idx : char_idx + line_len])
                )
                char_idx += line_len
            else:
                new_lines_text.append("")

        structured["lyrics"] = "\n".join(new_lines_text)
        structured["lines"] = self._split_lyrics_to_lines(
            structured["lyrics"],
            structured.get("jyutping", ""),
        )

        return structured

    def _build_word_selection_context(
        self,
        position: int,
        lyrics_text: str,
        melody_tone: int | None,
        is_rhyme_position: bool,
        reference_text: str,
    ) -> dict[str, Any]:
        """
        Build context for word selection at a specific position.

        Returns a dict with:
        - surrounding_before: preceding characters/words
        - surrounding_after: following characters/words
        - semantic_field: inferred from reference text
        - theme: overall theme
        - rhyme_final: if rhyme position, the expected rhyme final
        """
        # Extract surrounding context (±5 characters)
        window = 5
        before_start = max(0, position - window)
        after_end = min(len(lyrics_text), position + window + 1)

        surrounding_before = lyrics_text[before_start:position] if position > 0 else ""
        surrounding_after = (
            lyrics_text[position + 1 : after_end]
            if position < len(lyrics_text) - 1
            else ""
        )

        context: dict[str, Any] = {
            "position": f"第 {position + 1} 字",
            "surrounding_before": surrounding_before,
            "surrounding_after": surrounding_after,
            "melody_tone": str(melody_tone) if melody_tone is not None else None,
            "semantic_field": reference_text[:50] if reference_text else "",
            "theme": "歌词创作",
        }

        if is_rhyme_position:
            context["rhyme_requirement"] = "需与押韵位置协调"

        return context
