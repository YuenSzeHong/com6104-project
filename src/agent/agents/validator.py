"""
ValidatorAgent – Stage 4 of the Cantonese Lyrics pipeline.

Responsibilities
----------------
- Receive the draft lyrics and the full pipeline context (MIDI analysis,
  Jyutping map, reference text).
- Call the ``chinese_to_jyutping`` and ``get_tone_pattern`` MCP tools to
  independently romanise the draft lyrics and compare against the expected
  tone sequence.
- Compute a quality score (0.0 – 1.0) across four dimensions:
    1. Syllable count match     (weight 0.30)
    2. Tonal accuracy           (weight 0.35)
    3. Rhyme consistency        (weight 0.20)
    4. Artistic quality (LLM)   (weight 0.15)
- Return an AgentResult whose .data dict contains:
    {
        "validation_result": {
            "score":       float,   # weighted composite 0-1
            "passed":      bool,    # score >= threshold
            "feedback":    str,     # human-readable summary
            "corrections": list[str],   # actionable fixes for the composer
            "dimension_scores": {
                "syllable_count": float,
                "tonal_accuracy": float,
                "rhyme_consistency": float,
                "artistic_quality": float,
            },
            "draft_lyrics":       str,
            "draft_jyutping":     str,
            "expected_jyutping":  str,
            "syllable_count_expected": int,
            "syllable_count_actual":   int,
        }
    }

Usage (via orchestrator)
------------------------
The orchestrator calls this agent automatically as Stage 4 and passes
the result back into the composer's revision loop if the score is too low.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.base_agent import BaseAgent, AgentResult
from agent.config import AgentConfig
from agent.registry import AGENT_REGISTRY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dimension weights (must sum to 1.0)
_WEIGHT_SYLLABLE  = 0.30
_WEIGHT_TONAL     = 0.35
_WEIGHT_RHYME     = 0.20
_WEIGHT_ARTISTIC  = 0.15

# Cantonese tones
_CHECKED_TONES = {3, 6}
_VALID_TONES   = {1, 2, 3, 4, 5, 6}

# Jyutping syllable pattern: one or more lowercase letters + tone digit 1-6
_JP_SYLLABLE_RE = re.compile(r"[a-z]+[1-6]", re.IGNORECASE)

# Common Cantonese rhyme finals (simplified; extend as needed)
_RHYME_FINALS: dict[str, str] = {
    "aa":  "aa",  "aai": "aai", "aau": "aau", "aam": "aam",
    "aan": "aan", "aang": "aang", "aap": "aap", "aat": "aat",
    "aak": "aak", "ai":  "ai",  "au":  "au",  "am":  "am",
    "an":  "an",  "ang": "ang", "ap":  "ap",  "at":  "at",
    "ak":  "ak",  "e":   "e",   "ei":  "ei",  "eu":  "eu",
    "em":  "em",  "eng": "eng", "ep":  "ep",  "ek":  "ek",
    "i":   "i",   "iu":  "iu",  "im":  "im",  "in":  "in",
    "ing": "ing", "ip":  "ip",  "it":  "it",  "ik":  "ik",
    "o":   "o",   "oi":  "oi",  "ou":  "ou",  "on":  "on",
    "ong": "ong", "ot":  "ot",  "ok":  "ok",  "oe":  "oe",
    "oeng":"oeng","oek": "oek", "eoi": "eoi", "eon": "eon",
    "eot": "eot", "u":   "u",   "ui":  "ui",  "un":  "un",
    "ung": "ung", "ut":  "ut",  "uk":  "uk",  "yu":  "yu",
    "yun": "yun", "yut": "yut", "m":   "m",   "ng":  "ng",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_syllables(jyutping_string: str) -> list[str]:
    """Return a list of Jyutping syllables (with tone digits) from a string."""
    return _JP_SYLLABLE_RE.findall(jyutping_string.lower())


def _extract_tones(jyutping_string: str) -> list[int]:
    """Return only the tone numbers from a Jyutping string."""
    tones: list[int] = []
    for syl in _extract_syllables(jyutping_string):
        try:
            tones.append(int(syl[-1]))
        except ValueError:
            pass
    return tones


def _get_rhyme_final(syllable: str) -> str:
    """
    Extract the rhyme final from a single Jyutping syllable.

    E.g.  "sing1" → "ing",  "jau2" → "au",  "dak1" → "ak"
    """
    # Strip tone digit
    base = syllable.rstrip("0123456789")
    # Try longest-match against known finals
    for length in (4, 3, 2, 1):
        candidate = base[-length:] if len(base) >= length else ""
        if candidate and candidate in _RHYME_FINALS:
            return candidate
    return base  # fallback: return the whole base


def _count_cjk(text: str) -> int:
    """Count CJK characters in *text*."""
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@AGENT_REGISTRY.register("validator")
class ValidatorAgent(BaseAgent):
    """
    Multi-dimensional quality validator for draft Cantonese lyrics.

    Scoring dimensions
    ------------------
    1. **Syllable count**   – Does the draft have the exact syllable count
                              required by the MIDI melody?
    2. **Tonal accuracy**   – Do the tones at strong-beat positions match the
                              expected Jyutping tone sequence?
    3. **Rhyme consistency**– Do the line-ending syllables share a consistent
                              rhyme final pattern?
    4. **Artistic quality** – LLM-assessed poetic quality, imagery coherence,
                              and fidelity to the reference text concept.
    """

    # Tool names exposed by the jyutping MCP server
    _TOOL_CHINESE_TO_JYUTPING = "chinese_to_jyutping"
    _TOOL_GET_TONE_PATTERN     = "get_tone_pattern"

    # ------------------------------------------------------------------
    # Entry-point
    # ------------------------------------------------------------------

    async def _execute(self, task: str, **kwargs: Any) -> AgentResult:
        # ----------------------------------------------------------------
        # 0. Extract inputs from task / context
        # ----------------------------------------------------------------
        draft_lyrics = self._extract_draft_from_task(task) or kwargs.get("draft_lyrics", "")
        if not draft_lyrics:
            stored_draft = self._memory.get_current_draft()
            if isinstance(stored_draft, dict):
                draft_lyrics = str(stored_draft.get("lyrics", ""))
            else:
                draft_lyrics = str(stored_draft)

        if not draft_lyrics:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error="No draft lyrics provided to validate.",
            )

        midi_analysis: dict[str, Any] = self._memory.get_pipeline_value("midi_analysis", {}) or {}
        jyutping_map: dict[str, Any] = self._memory.get_pipeline_value("jyutping_map", {}) or {}
        reference_text: str = (
            jyutping_map.get("reference_text", "")
            or self._extract_reference_text_from_task(task)
        )

        expected_syllable_count: int = int(
            midi_analysis.get("effective_syllable_count", 0)
            or midi_analysis.get("syllable_count", 0)
            or jyutping_map.get("target_syllable_count", 0)
        )
        expected_tone_sequence: list[int] = (
            jyutping_map.get("melody_tone_sequence_0243", [])
        )
        strong_beat_positions: list[int] = (
            midi_analysis.get("strong_beat_positions", [])
        )

        self._log.info(
            "Validating draft: syllable_count_expected=%d, melody_0243_len=%d",
            expected_syllable_count,
            len(expected_tone_sequence),
        )

        # ----------------------------------------------------------------
        # 1. Obtain actual Jyutping for the draft lyrics via MCP tools
        # ----------------------------------------------------------------
        draft_jyutping: str = ""
        actual_tone_sequence: list[int] = []

        if self._has_tool(self._TOOL_CHINESE_TO_JYUTPING):
            candidates = await self._call_chinese_to_jyutping(draft_lyrics)
            if candidates:
                # Pick the first candidate (most likely reading)
                draft_jyutping = candidates[0]

        if self._has_tool(self._TOOL_GET_TONE_PATTERN):
            tone_str = await self._call_get_tone_pattern(draft_lyrics)
            if tone_str:
                actual_tone_sequence = self._parse_tone_string(tone_str)
                if not draft_jyutping:
                    draft_jyutping = tone_str

        # Fallback: derive tones from the Jyutping string itself
        if not actual_tone_sequence and draft_jyutping:
            actual_tone_sequence = _extract_tones(draft_jyutping)

        actual_syllable_count = (
            len(actual_tone_sequence)
            or len(_extract_syllables(draft_jyutping))
            or _count_cjk(draft_lyrics)
        )

        # ----------------------------------------------------------------
        # 2. Compute dimension scores
        # ----------------------------------------------------------------
        dim_syllable = self._score_syllable_count(
            actual=actual_syllable_count,
            expected=expected_syllable_count,
        )

        dim_tonal = self._score_tonal_accuracy(
            actual_tones=actual_tone_sequence,
            expected_tones=expected_tone_sequence,
            strong_beat_positions=strong_beat_positions,
        )

        dim_rhyme = self._score_rhyme_consistency(
            draft_jyutping=draft_jyutping,
            draft_lyrics=draft_lyrics,
        )

        dim_artistic = await self._score_artistic_quality(
            draft_lyrics=draft_lyrics,
            reference_text=reference_text,
            expected_syllable_count=expected_syllable_count,
        )

        # ----------------------------------------------------------------
        # 3. Weighted composite score
        # ----------------------------------------------------------------
        composite_score: float = (
            dim_syllable  * _WEIGHT_SYLLABLE
            + dim_tonal   * _WEIGHT_TONAL
            + dim_rhyme   * _WEIGHT_RHYME
            + dim_artistic * _WEIGHT_ARTISTIC
        )
        composite_score = round(min(max(composite_score, 0.0), 1.0), 4)

        self._log.info(
            "Scores – syllable=%.2f tonal=%.2f rhyme=%.2f artistic=%.2f → composite=%.3f",
            dim_syllable, dim_tonal, dim_rhyme, dim_artistic, composite_score,
        )

        # ----------------------------------------------------------------
        # 4. Build corrections list
        # ----------------------------------------------------------------
        corrections = self._build_corrections(
            dim_syllable=dim_syllable,
            dim_tonal=dim_tonal,
            dim_rhyme=dim_rhyme,
            dim_artistic=dim_artistic,
            actual_syllable_count=actual_syllable_count,
            expected_syllable_count=expected_syllable_count,
            actual_tones=actual_tone_sequence,
            expected_tones=expected_tone_sequence,
            strong_beat_positions=strong_beat_positions,
            draft_lyrics=draft_lyrics,
        )

        # ----------------------------------------------------------------
        # 5. Build human-readable feedback
        # ----------------------------------------------------------------
        feedback = self._format_feedback(
            composite_score=composite_score,
            dim_syllable=dim_syllable,
            dim_tonal=dim_tonal,
            dim_rhyme=dim_rhyme,
            dim_artistic=dim_artistic,
            actual_syllable_count=actual_syllable_count,
            expected_syllable_count=expected_syllable_count,
            corrections=corrections,
        )

        # ----------------------------------------------------------------
        # 6. Assemble result
        # ----------------------------------------------------------------
        validation_result: dict[str, Any] = {
            "score":                   composite_score,
            "passed":                  composite_score >= self._quality_threshold(),
            "feedback":                feedback,
            "corrections":             corrections,
            "melody_0243_score":       round(dim_tonal, 4),
            "dimension_scores": {
                "syllable_count":  round(dim_syllable, 4),
                "tonal_accuracy":  round(dim_tonal, 4),
                "rhyme_consistency": round(dim_rhyme, 4),
                "artistic_quality": round(dim_artistic, 4),
            },
            "draft_lyrics":            draft_lyrics,
            "draft_jyutping":          draft_jyutping,
            "expected_jyutping":       jyutping_map.get("selected_jyutping", ""),
            "syllable_count_expected": expected_syllable_count,
            "syllable_count_actual":   actual_syllable_count,
        }

        self._memory.set_validation_result(validation_result)

        return AgentResult(
            agent_name=self.name,
            success=True,
            output=feedback,
            data={"validation_result": validation_result},
            metadata={
                "composite_score": composite_score,
                "passed": validation_result["passed"],
            },
        )

    # ------------------------------------------------------------------
    # Dimension scorers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_syllable_count(actual: int, expected: int) -> float:
        """
        Score based on how close the actual syllable count is to the expected.

        - Exact match → 1.0
        - Off by 1    → 0.7
        - Off by 2    → 0.4
        - Off by 3+   → linear decay toward 0.0 (min 0.0)
        """
        if expected == 0:
            # No expected count available – cannot evaluate; give neutral score
            return 0.5

        diff = abs(actual - expected)
        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.7
        elif diff == 2:
            return 0.4
        else:
            # Each additional syllable off reduces score by 0.1 beyond diff=2
            score = max(0.0, 0.4 - (diff - 2) * 0.1)
            return round(score, 4)

    @staticmethod
    def _score_tonal_accuracy(
        actual_tones: list[int],
        expected_tones: list[int],
        strong_beat_positions: list[int],
    ) -> float:
        """
        Compare the actual 1-6 tone sequence against the expected lean 0243 melody tones.

        Two sub-scores are averaged:
        a) Overall positional fit to the lean 0243 target
        b) Strong-beat singability
        """
        if not actual_tones:
            return 0.5   # can't evaluate

        if not expected_tones:
            # No reference tones – check that at least all tones are valid values
            invalid = sum(1 for t in actual_tones if t not in _VALID_TONES)
            return round(1.0 - (invalid / len(actual_tones)), 4)

        def tone_matches_0243(actual_tone: int, expected_0243: int) -> bool:
            mapping = {
                0: {1},
                2: {2, 5},
                4: {3, 4, 6},
                3: {4, 6},
            }
            return actual_tone in mapping.get(expected_0243, set())

        # Sub-score a: positional fit from 1-6 into lean 0243 buckets
        min_len = min(len(actual_tones), len(expected_tones))
        matches = sum(
            1 for a, e in zip(actual_tones[:min_len], expected_tones[:min_len])
            if tone_matches_0243(a, e)
        )
        coverage_penalty = abs(len(actual_tones) - len(expected_tones)) / max(
            len(actual_tones), len(expected_tones)
        )
        sub_a = (matches / min_len) * (1.0 - 0.5 * coverage_penalty)

        # Sub-score b: strong-beat positions
        if strong_beat_positions:
            valid_positions = [
                p for p in strong_beat_positions if p < len(actual_tones)
            ]
            if valid_positions:
                beat_matches = 0
                for pos in valid_positions:
                    actual_t = actual_tones[pos]
                    expected_t = (
                        expected_tones[pos]
                        if pos < len(expected_tones)
                        else None
                    )
                    if expected_t is not None:
                        if tone_matches_0243(actual_t, expected_t):
                            beat_matches += 2
                        elif actual_t not in _CHECKED_TONES:
                            beat_matches += 1
                    else:
                        if actual_t not in _CHECKED_TONES:
                            beat_matches += 1
                sub_b = beat_matches / (2 * len(valid_positions))
            else:
                sub_b = sub_a
        else:
            sub_b = sub_a

        return round((sub_a + sub_b) / 2, 4)

    @staticmethod
    def _score_rhyme_consistency(
        draft_jyutping: str,
        draft_lyrics: str,
    ) -> float:
        """
        Score how consistent the rhyme scheme is across line endings.

        Strategy:
        - Split lyrics by newline (or Chinese punctuation) to get lines.
        - For each line-ending syllable extract its rhyme final.
        - Compute the proportion of line-end rhyme finals that match the
          most common one (plurality-wins rhyme scheme).
        """
        # Split by line or Chinese sentence-ending punctuation
        lines = re.split(r"[\n。！？，、；｜]", draft_lyrics)
        lines = [ln.strip() for ln in lines if ln.strip()]

        if len(lines) < 2:
            return 0.5   # single line – can't evaluate rhyme scheme

        # Map each line to its ending CJK character's index in the Jyutping string
        syllables = _extract_syllables(draft_jyutping)

        if len(syllables) < len(lines):
            # Not enough syllables to map to lines – partial evaluation
            return 0.5

        # Approximate: line-ending syllable ≈ the syllable at proportional position
        cjk_counts = [_count_cjk(ln) for ln in lines]
        total_cjk = sum(cjk_counts)

        if total_cjk == 0 or not syllables:
            return 0.5

        # Walk through syllables and collect the last syllable of each "line"
        line_end_rhymes: list[str] = []
        cursor = 0
        for count in cjk_counts:
            end_idx = min(cursor + count - 1, len(syllables) - 1)
            if end_idx >= 0:
                line_end_rhymes.append(_get_rhyme_final(syllables[end_idx]))
            cursor += count

        if not line_end_rhymes:
            return 0.5

        # Plurality-wins rhyme
        from collections import Counter
        rhyme_counts = Counter(line_end_rhymes)
        most_common_count = rhyme_counts.most_common(1)[0][1]
        score = most_common_count / len(line_end_rhymes)

        # Penalise slightly if all line-ending syllables are checked tones
        # (checked tones are considered weak for rhyming in Cantonese)
        last_syllables_tones = [
            _extract_tones(syl) for syl in
            [syllables[min(sum(cjk_counts[:i+1]) - 1, len(syllables)-1)]
             for i in range(len(cjk_counts))]
        ]
        checked_endings = sum(
            1 for t_list in last_syllables_tones
            if t_list and t_list[-1] in _CHECKED_TONES
        )
        if checked_endings > len(last_syllables_tones) / 2:
            score *= 0.85  # gentle penalty

        return round(min(score, 1.0), 4)

    async def _score_artistic_quality(
        self,
        draft_lyrics: str,
        reference_text: str,
        expected_syllable_count: int,
    ) -> float:
        """
        Ask the LLM to evaluate the artistic quality of the draft lyrics
        on a scale of 0–10 and normalise to 0.0–1.0.

        Criteria assessed by the LLM:
        - Poetic imagery and metaphor
        - Coherence with the reference text's artistic intent
        - Natural Cantonese phrasing and colloquialism balance
        - Emotional resonance
        """
        prompt = f"""
You are an expert Cantonese lyricist and literary critic.
Rate the following draft Cantonese lyrics on artistic quality from 0 to 10.

### Draft Lyrics
{draft_lyrics}

### Reference Text / Artistic Intent
{reference_text or "(not provided)"}

### Evaluation Criteria
1. **Poetic imagery**: vivid, evocative language and metaphor.
2. **Coherence**: does the lyric honour the mood and theme of the reference text?
3. **Natural Cantonese phrasing**: sounds like authentic Cantonese song language.
4. **Emotional resonance**: does it feel expressive and moving?

### Output Format
Return ONLY a JSON object like:
{{"score": 7.5, "reasoning": "brief explanation"}}

Do not include any other text.
""".strip()

        try:
            raw = await self._invoke_llm(extra_user_message=prompt)
            json_str = self._extract_json_from_text(raw)
            if json_str:
                obj = json.loads(json_str)
                raw_score = float(obj.get("score", 5.0))
                return round(min(max(raw_score / 10.0, 0.0), 1.0), 4)
        except Exception as exc:  # noqa: BLE001
            self._log.warning("Artistic quality scoring failed: %s", exc)

        return 0.5  # neutral fallback

    # ------------------------------------------------------------------
    # Corrections builder
    # ------------------------------------------------------------------

    def _build_corrections(
        self,
        dim_syllable: float,
        dim_tonal: float,
        dim_rhyme: float,
        dim_artistic: float,
        actual_syllable_count: int,
        expected_syllable_count: int,
        actual_tones: list[int],
        expected_tones: list[int],
        strong_beat_positions: list[int],
        draft_lyrics: str,
    ) -> list[str]:
        """Build an actionable list of correction instructions for the composer."""
        corrections: list[str] = []

        # Syllable count corrections
        if dim_syllable < 0.95 and expected_syllable_count > 0:
            diff = actual_syllable_count - expected_syllable_count
            if diff > 0:
                corrections.append(
                    f"Remove {diff} syllable(s). "
                    f"The draft has {actual_syllable_count} syllables but the melody "
                    f"requires exactly {expected_syllable_count}."
                )
            elif diff < 0:
                corrections.append(
                    f"Add {abs(diff)} syllable(s). "
                    f"The draft has only {actual_syllable_count} syllables but the melody "
                    f"requires exactly {expected_syllable_count}."
                )

        # Tonal accuracy corrections
        if dim_tonal < 0.70 and actual_tones and expected_tones:
            mismatches: list[str] = []
            for i, (a, e) in enumerate(
                zip(actual_tones[:len(expected_tones)], expected_tones)
            ):
                if a != e:
                    mismatches.append(f"syllable {i+1}: got tone {a}, expected tone {e}")
            if mismatches:
                corrections.append(
                    "Fix tonal mismatches at: " + "; ".join(mismatches[:5])
                    + ("…" if len(mismatches) > 5 else "")
                )

        if dim_tonal < 0.70 and strong_beat_positions and actual_tones:
            bad_beats: list[int] = []
            for pos in strong_beat_positions:
                if pos < len(actual_tones) and actual_tones[pos] in _CHECKED_TONES:
                    bad_beats.append(pos + 1)  # 1-indexed for human readability
            if bad_beats:
                corrections.append(
                    f"Strong beat positions {bad_beats[:5]} have checked tones (3 or 6). "
                    "Replace with sustained tones (1, 2, 4, or 5) at these positions."
                )

        # Rhyme consistency corrections
        if dim_rhyme < 0.65:
            corrections.append(
                "Improve rhyme consistency: line-ending syllables do not share a "
                "common rhyme final. Choose a consistent rhyme vowel (e.g. all -ing, "
                "all -an, all -au) and apply it to every line ending."
            )

        # Artistic quality corrections
        if dim_artistic < 0.55:
            corrections.append(
                "Improve artistic quality: the draft feels prosaic or disconnected "
                "from the reference text's imagery. Use richer metaphors, more vivid "
                "Cantonese vocabulary, and ensure emotional coherence with the theme."
            )

        # If no specific corrections found but score is still low
        if not corrections and (
            dim_syllable * _WEIGHT_SYLLABLE
            + dim_tonal   * _WEIGHT_TONAL
            + dim_rhyme   * _WEIGHT_RHYME
            + dim_artistic * _WEIGHT_ARTISTIC
        ) < self._quality_threshold():
            corrections.append(
                "Overall quality is below the acceptance threshold. "
                "Consider a full rewrite while keeping the syllable count exact and "
                "tones aligned with the melody."
            )

        return corrections

    # ------------------------------------------------------------------
    # Feedback formatter
    # ------------------------------------------------------------------

    @staticmethod
    def _format_feedback(
        composite_score: float,
        dim_syllable: float,
        dim_tonal: float,
        dim_rhyme: float,
        dim_artistic: float,
        actual_syllable_count: int,
        expected_syllable_count: int,
        corrections: list[str],
    ) -> str:
        """Format a human-readable validation summary."""
        passed_marker = "✓ PASSED" if composite_score >= 0.75 else "✗ NEEDS REVISION"
        bar_len = 20
        filled = round(composite_score * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        lines = [
            f"## Validation Report  [{passed_marker}]",
            f"**Composite Score:** {composite_score:.3f}  [{bar}]",
            "",
            "### Dimension Scores",
            f"| Dimension        | Score | Weight |",
            f"|------------------|-------|--------|",
            f"| Syllable Count   | {dim_syllable:.3f} | 30%    |",
            f"| Melody 0243 Fit  | {dim_tonal:.3f} | 35%    |",
            f"| Rhyme Consistency| {dim_rhyme:.3f} | 20%    |",
            f"| Artistic Quality | {dim_artistic:.3f} | 15%    |",
            "",
            f"**Syllable Count:** {actual_syllable_count} actual "
            f"vs {expected_syllable_count} expected",
        ]

        if corrections:
            lines.append("")
            lines.append("### Required Corrections")
            for i, c in enumerate(corrections, 1):
                lines.append(f"{i}. {c}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # MCP tool callers
    # ------------------------------------------------------------------

    async def _call_chinese_to_jyutping(self, text: str) -> list[str]:
        """Invoke ``chinese_to_jyutping`` MCP tool and return candidates."""
        try:
            tool = self._get_tool(self._TOOL_CHINESE_TO_JYUTPING)
            if tool is None:
                return []
            raw = await tool.ainvoke({"text": text})
            if isinstance(raw, list):
                return [str(c) for c in raw]
            parsed = json.loads(str(raw))
            return [str(c) for c in parsed] if isinstance(parsed, list) else [str(raw)]
        except Exception as exc:  # noqa: BLE001
            self._log.warning("chinese_to_jyutping failed: %s", exc)
            return []

    async def _call_get_tone_pattern(self, text: str) -> str:
        """Invoke ``get_tone_pattern`` MCP tool and return raw tone string."""
        try:
            tool = self._get_tool(self._TOOL_GET_TONE_PATTERN)
            if tool is None:
                return ""
            raw = await tool.ainvoke({"text": text})
            return str(raw).strip()
        except Exception as exc:  # noqa: BLE001
            self._log.warning("get_tone_pattern failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _extract_draft_from_task(self, task: str) -> str | None:
        """Extract draft lyrics from task string if present."""
        # Try to find JSON block with lyrics field
        import re
        json_match = re.search(r'\{[^{}]*"lyrics"[^{}]*\}', task, re.DOTALL)
        if json_match:
            try:
                import json
                data = json.loads(json_match.group())
                return data.get("lyrics")
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    def _extract_reference_text_from_task(self, task: str) -> str:
        """Extract reference text from task string if present."""
        import re
        match = re.search(r"参考文本 [:：]\s*(.+?)(?:\n|$)", task)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_json_from_text(self, text: str) -> str | None:
        """Extract JSON block from text response."""
        import re
        # Try markdown code fence first
        fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if fence_match:
            return fence_match.group(1).strip()
        # Try to find balanced JSON object
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def _quality_threshold(self) -> float:
        """Return the configured minimum quality score (from WORKFLOW_CONFIG)."""
        try:
            from agent.config import WORKFLOW_CONFIG
            return float(WORKFLOW_CONFIG.get("min_quality_score", 0.75))
        except Exception:  # noqa: BLE001
            return 0.75

    def _has_tool(self, name: str) -> bool:
        return any(t.name == name for t in self._tools)

    def _get_tool(self, name: str):
        for t in self._tools:
            if t.name == name:
                return t
        return None

    @staticmethod
    def _parse_tone_string(tone_str: str) -> list[int]:
        """Parse a space-separated tone string like '1 4 3 2 1 6' into ints."""
        result: list[int] = []
        for token in tone_str.split():
            try:
                val = int(token.strip().rstrip(".,:;"))
                result.append(val)
            except ValueError:
                pass
        return result
