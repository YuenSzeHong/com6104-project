#!/usr/bin/env python3
"""
Lyrics Validator MCP Server
============================
Moves all computational validation skills out of agent Python code and into
a proper MCP-testable server. Every tool here is independently inspectable
via the MCP Inspector.

Tools
-----
count_syllables(lyrics)
    Count Cantonese syllables (= CJK characters) in a lyrics string.

check_tone_accuracy(draft_jyutping, expected_tone_sequence, strong_beat_positions)
    Compare the tone sequence of a draft against the expected sequence.
    Returns per-position match data and an overall accuracy score.

check_rhyme_scheme(draft_jyutping, rhyme_positions)
    Analyse rhyme consistency at the specified syllable positions.
    Returns the dominant rhyme final, per-position finals, and a score.

score_lyrics(lyrics, draft_jyutping, expected_tone_sequence,
             strong_beat_positions, rhyme_positions, expected_syllable_count)
    Composite scorer: runs all three checks and returns a weighted overall
    score (0.0–1.0) plus per-dimension breakdown and actionable corrections.

suggest_corrections(score_report)
    Given the JSON score report produced by score_lyrics, return a
    human-readable list of prioritised correction instructions in Chinese.

Transport
---------
stdio  (compatible with langchain-mcp-adapters MultiServerMCPClient)

Testing with MCP Inspector
--------------------------
    npx @modelcontextprotocol/inspector python mcp-servers/lyrics-validator/server.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter
from typing import Any

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[lyrics-validator] %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("lyrics-validator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dimension weights – must sum to 1.0
_W_SYLLABLE  = 0.30
_W_TONAL     = 0.35
_W_RHYME     = 0.20
_W_STRUCTURE = 0.15   # replaces "artistic" for the computational part

# Cantonese tone classification
_CHECKED_TONES  = {3, 6}          # short, abrupt – bad on long/strong notes
_SMOOTH_TONES   = {1, 2, 4, 5}    # sustained – good on strong beats
_VALID_TONES    = {1, 2, 3, 4, 5, 6}

# 0243.hk lean-mode tone buckets
_LEAN_0243_BUCKETS: dict[int, set[int]] = {
    0: {1},       # high level
    2: {2, 5},    # rising
    4: {3, 4, 6}, # mid / neutral
    3: {4, 6},    # low / falling
}

# Jyutping syllable pattern
_JP_SYLLABLE_RE = re.compile(r"[a-z]+[1-6]", re.IGNORECASE)

# CJK character range
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# Common Cantonese rhyme finals (vowel nucleus + coda, without tone digit)
_RHYME_FINALS: frozenset[str] = frozenset({
    "aa",  "aai", "aau", "aam", "aan", "aang", "aap", "aat", "aak",
    "ai",  "au",  "am",  "an",  "ang", "ap",   "at",  "ak",
    "e",   "ei",  "eu",  "em",  "eng", "ep",   "ek",
    "i",   "iu",  "im",  "in",  "ing", "ip",   "it",  "ik",
    "o",   "oi",  "ou",  "on",  "ong", "ot",   "ok",
    "oe",  "oei", "oeng","oek", "eoi", "eon",  "eot",
    "u",   "ui",  "un",  "ung", "ut",  "uk",
    "yu",  "yun", "yut",
    "m",   "ng",
})

# Cantonese initials (longest-first for greedy strip)
_INITIALS: tuple[str, ...] = (
    "ng", "gw", "kw",
    "b",  "p",  "m",  "f",
    "d",  "t",  "n",  "l",
    "g",  "k",  "h",
    "z",  "c",  "s",
    "j",  "w",
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cjk_count(text: str) -> int:
    """Count CJK characters (each = one Cantonese syllable)."""
    return len(_CJK_RE.findall(text))


def _extract_syllables(jyutping: str) -> list[str]:
    """Return a list of Jyutping syllables (with tone digit) from a string."""
    return _JP_SYLLABLE_RE.findall(jyutping.lower())


def _extract_tones(jyutping: str) -> list[int]:
    """Return tone digits from a Jyutping string."""
    tones: list[int] = []
    for syl in _extract_syllables(jyutping):
        try:
            tones.append(int(syl[-1]))
        except ValueError:
            pass
    return tones


def _get_rhyme_final(syllable: str) -> str:
    """
    Extract the rhyme final (vowel nucleus + coda) from a Jyutping syllable.

    Strategy: strip tone digit, then strip the longest matching initial,
    then match the remainder against known finals.

    Examples:
        "sing1" -> "ing"
        "jau2"  -> "au"
        "dak1"  -> "ak"
        "ng4"   -> "ng"
    """
    base = syllable.rstrip("0123456789").lower()

    # Strip longest matching initial
    for initial in _INITIALS:
        if base.startswith(initial) and base != initial:
            candidate = base[len(initial):]
            # Verify the remainder is a known final or non-empty
            if candidate:
                base = candidate
                break

    # Try matching known finals (longest first)
    for length in (4, 3, 2, 1):
        if len(base) >= length:
            suffix = base[-length:]
            if suffix in _RHYME_FINALS:
                return suffix

    return base  # fallback: return whole remainder


def _parse_int_list(raw: Any, field_name: str) -> list[int]:
    """Parse a JSON string or list into a list[int]."""
    if isinstance(raw, list):
        try:
            return [int(x) for x in raw]
        except (TypeError, ValueError):
            pass
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [int(x) for x in parsed]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        # Space-separated fallback
        tokens = raw.strip().split()
        result = []
        for t in tokens:
            try:
                result.append(int(t))
            except ValueError:
                pass
        if result:
            return result
    logger.warning("_parse_int_list: could not parse field '%s': %r", field_name, raw)
    return []


def _tone_matches_lean_0243(actual_tone: int, expected_code: int) -> bool:
    """Return True when a 1-6 Jyutping tone fits the target lean 0243 code."""
    return actual_tone in _LEAN_0243_BUCKETS.get(expected_code, set())


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="lyrics-validator",
    instructions=(
        "粤语歌词计算验证工具服务器。提供音节计数、声调准确性检查、"
        "押韵一致性分析及综合评分，所有工具均可通过 MCP Inspector 独立测试。"
    ),
)


# ---------------------------------------------------------------------------
# Tool 1: count_syllables
# ---------------------------------------------------------------------------

@mcp.tool()
def count_syllables(lyrics: str) -> str:
    """
    计算粤语歌词中的音节总数。

    粤语中每个汉字（CJK 字符）对应一个音节。标点符号、空格和换行符不计入音节数。
    同时也返回按行的分解数据，方便定位哪一行音节数有问题。

    参数
    ----
    lyrics : str
        粤语歌词文本，可含换行符，例如：
        "青山依舊在\\n幾度夕陽紅"

    返回
    ----
    str
        JSON 对象字符串，包含：
        {
          "total": int,              // 全部音节数
          "lines": [                 // 按行分解
            {"text": str, "count": int},
            ...
          ],
          "line_count": int          // 歌词行数
        }
    """
    if not lyrics or not lyrics.strip():
        return json.dumps({"total": 0, "lines": [], "line_count": 0})

    line_data: list[dict[str, Any]] = []
    for raw_line in lyrics.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        count = _cjk_count(stripped)
        line_data.append({"text": stripped, "count": count})

    total = sum(row["count"] for row in line_data)

    result = {
        "total":      total,
        "lines":      line_data,
        "line_count": len(line_data),
    }
    logger.info("count_syllables: total=%d lines=%d", total, len(line_data))
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 2: check_tone_accuracy
# ---------------------------------------------------------------------------

@mcp.tool()
def check_tone_accuracy(
    draft_jyutping: str,
    expected_tone_sequence: str,
    strong_beat_positions: str,
) -> str:
    """
    检查草稿歌词对 lean 0243 旋律目标的贴合度。

    将草稿粤拼的实际 1-6 声调序列映射到 0243.hk 默认 lean 0243
    类别，与预期的 0243 序列逐位对比，并特别检查强拍位置是否使用了
    不适合长音/重拍的入声字（声调3、6）。

    参数
    ----
    draft_jyutping : str
        草稿歌词的粤拼字符串，音节之间用空格分隔。
        例如："cing1 saan1 ji1 gau6 zoi6"

    expected_tone_sequence : str
        预期的 lean 0243 数字序列，可以是：
        - JSON 数组字符串：  "[0, 2, 4, 3, 4]"
        - 空格分隔的字符串：  "0 2 4 3 4"

    strong_beat_positions : str
        强拍位置的音节索引列表（0 起始），可以是：
        - JSON 数组字符串：  "[0, 4, 8, 12]"
        - 空格分隔的字符串：  "0 4 8 12"

    返回
    ----
    str
        JSON 对象字符串，包含：
        {
          "score": float,                     // 0.0–1.0，越高越好
          "actual_tones": [int, ...],         // 从草稿粤拼提取的实际声调
          "expected_tones": [int, ...],       // 预期 lean 0243 序列
          "positional_match": [               // 逐位对比
            {"position": int, "actual": int, "expected": int, "match": bool},
            ...
          ],
          "match_rate": float,                // 位置匹配率（0.0–1.0）
          "strong_beat_issues": [             // 强拍上的入声字问题
            {"position": int, "tone": int, "problem": str},
            ...
          ],
          "strong_beat_score": float,         // 强拍声调得分（0.0–1.0）
          "length_match": bool,               // 实际音节数 == 预期音节数
          "actual_syllable_count": int,
          "expected_syllable_count": int
        }
    """
    actual_tones    = _extract_tones(draft_jyutping)
    expected_tones  = _parse_int_list(expected_tone_sequence, "expected_tone_sequence")
    strong_beats    = _parse_int_list(strong_beat_positions,  "strong_beat_positions")

    actual_count   = len(actual_tones)
    expected_count = len(expected_tones)

    # ------------------------------------------------------------------
    # Sub-score A: positional fit to lean 0243 target
    # ------------------------------------------------------------------
    min_len = min(actual_count, expected_count)
    positional: list[dict[str, Any]] = []
    matches = 0

    for i in range(min_len):
        match = _tone_matches_lean_0243(actual_tones[i], expected_tones[i])
        if match:
            matches += 1
        positional.append({
            "position": i,
            "actual":   actual_tones[i],
            "expected": expected_tones[i],
            "match":    match,
        })

    # Penalise length mismatch
    max_len = max(actual_count, expected_count, 1)
    coverage = min_len / max_len
    match_rate = (matches / min_len * coverage) if min_len > 0 else 0.0
    sub_a = match_rate

    # ------------------------------------------------------------------
    # Sub-score B: strong beat quality
    # ------------------------------------------------------------------
    strong_beat_issues: list[dict[str, Any]] = []
    beat_score_sum = 0.0
    valid_beats = 0

    for pos in strong_beats:
        if pos >= actual_count:
            continue
        valid_beats += 1
        tone = actual_tones[pos]
        expected_at_pos = expected_tones[pos] if pos < expected_count else None

        if tone in _CHECKED_TONES:
            strong_beat_issues.append({
                "position": pos,
                "tone":     tone,
                "problem":  f"入声字（声调{tone}）落在强拍位置 {pos}，会被旋律音高扭曲",
            })
            beat_score_sum += 0.0
        elif expected_at_pos is not None and _tone_matches_lean_0243(tone, expected_at_pos):
            beat_score_sum += 1.0
        elif tone in _SMOOTH_TONES:
            beat_score_sum += 0.6    # right category (smooth), wrong specific tone
        else:
            beat_score_sum += 0.3

    strong_beat_score = (beat_score_sum / valid_beats) if valid_beats > 0 else sub_a

    # Composite tonal score
    score = round((sub_a + strong_beat_score) / 2, 4)

    result: dict[str, Any] = {
        "score":                  score,
        "actual_tones":           actual_tones,
        "expected_tones":         expected_tones,
        "positional_match":       positional,
        "match_rate":             round(match_rate, 4),
        "strong_beat_issues":     strong_beat_issues,
        "strong_beat_score":      round(strong_beat_score, 4),
        "length_match":           actual_count == expected_count,
        "actual_syllable_count":  actual_count,
        "expected_syllable_count": expected_count,
    }

    logger.info(
            "check_tone_accuracy: score=%.3f fit_rate=%.3f beat_issues=%d",
            score, match_rate, len(strong_beat_issues),
    )
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 3: check_rhyme_scheme
# ---------------------------------------------------------------------------

@mcp.tool()
def check_rhyme_scheme(
    draft_jyutping: str,
    rhyme_positions: str,
) -> str:
    """
    分析草稿歌词的押韵一致性。

    提取指定押韵位置的音节韵尾，检查是否共享同一韵脚。
    采用多数原则（plurality-wins）确定主押韵，并给出一致性评分。

    参数
    ----
    draft_jyutping : str
        草稿歌词的粤拼字符串，音节之间用空格分隔。

    rhyme_positions : str
        押韵位置的音节索引列表（0 起始），可以是 JSON 数组或空格分隔字符串。
        例如："[7, 15, 23, 31]" 或 "7 15 23 31"

    返回
    ----
    str
        JSON 对象字符串，包含：
        {
          "score": float,                       // 0.0–1.0，押韵一致性得分
          "dominant_final": str,                // 最常见的韵尾（主押韵）
          "rhyme_positions_analysis": [         // 各押韵位置详情
            {
              "position": int,
              "syllable": str,
              "final":    str,
              "matches_dominant": bool,
              "is_checked_tone": bool
            },
            ...
          ],
          "consistency_rate": float,            // 使用主押韵的位置比例
          "total_rhyme_positions": int,
          "valid_rhyme_positions": int,         // 索引在范围内的位置数
          "unique_finals": [str, ...],          // 出现的所有韵尾（去重）
          "checked_tone_penalty_applied": bool  // 是否因入声押韵扣分
        }
    """
    syllables       = _extract_syllables(draft_jyutping)
    rhyme_pos_list  = _parse_int_list(rhyme_positions, "rhyme_positions")

    if not syllables or not rhyme_pos_list:
        result = {
            "score":                      0.5,
            "dominant_final":             "",
            "rhyme_positions_analysis":   [],
            "consistency_rate":           0.5,
            "total_rhyme_positions":      len(rhyme_pos_list),
            "valid_rhyme_positions":      0,
            "unique_finals":              [],
            "checked_tone_penalty_applied": False,
        }
        return json.dumps(result, ensure_ascii=False)

    # Analyse each rhyme position
    analysis: list[dict[str, Any]] = []
    finals: list[str] = []

    for pos in rhyme_pos_list:
        if pos >= len(syllables):
            analysis.append({
                "position":         pos,
                "syllable":         "",
                "final":            "",
                "matches_dominant": False,
                "is_checked_tone":  False,
                "error":            f"索引 {pos} 超出音节范围（共 {len(syllables)} 个）",
            })
            continue

        syl   = syllables[pos]
        final = _get_rhyme_final(syl)
        tone  = int(syl[-1]) if syl[-1].isdigit() else 0
        finals.append(final)
        analysis.append({
            "position":         pos,
            "syllable":         syl,
            "final":            final,
            "matches_dominant": False,   # filled in below
            "is_checked_tone":  tone in _CHECKED_TONES,
        })

    # Determine dominant (plurality-wins) final
    dominant_final = ""
    consistency_rate = 0.5

    if finals:
        counter        = Counter(finals)
        dominant_final = counter.most_common(1)[0][0]
        dominant_count = counter[dominant_final]
        consistency_rate = dominant_count / len(finals)

        # Back-fill matches_dominant
        for entry in analysis:
            if "error" not in entry:
                entry["matches_dominant"] = entry["final"] == dominant_final

    # Penalty: if more than half of rhyme positions are checked tones, reduce score
    checked_endings = sum(
        1 for e in analysis
        if e.get("is_checked_tone", False) and "error" not in e
    )
    valid_count = len(finals)
    checked_penalty = valid_count > 0 and checked_endings > valid_count / 2

    score = consistency_rate
    if checked_penalty:
        score *= 0.85   # gentle deduction; checked tone rhyme is weaker in Cantonese

    score = round(min(max(score, 0.0), 1.0), 4)

    result = {
        "score":                       score,
        "dominant_final":              dominant_final,
        "rhyme_positions_analysis":    analysis,
        "consistency_rate":            round(consistency_rate, 4),
        "total_rhyme_positions":       len(rhyme_pos_list),
        "valid_rhyme_positions":       valid_count,
        "unique_finals":               sorted(set(finals)),
        "checked_tone_penalty_applied": checked_penalty,
    }

    logger.info(
        "check_rhyme_scheme: score=%.3f dominant=%r consistency=%.3f",
        score, dominant_final, consistency_rate,
    )
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 4: score_lyrics  (composite scorer)
# ---------------------------------------------------------------------------

@mcp.tool()
def score_lyrics(
    lyrics: str,
    draft_jyutping: str,
    expected_tone_sequence: str,
    strong_beat_positions: str,
    rhyme_positions: str,
    expected_syllable_count: int,
) -> str:
    """
    综合评分：对粤语歌词草稿进行全面的计算验证。

    依次运行音节计数、声调准确性检查、押韵一致性分析，
    并按固定权重合并为一个 0.0–1.0 的综合评分。
    同时生成可操作的修改建议列表。

    权重说明
    --------
    - 音节数匹配   30%  (syllable)
    - 声调准确性   35%  (tonal)
    - 押韵一致性   20%  (rhyme)
    - 结构完整性   15%  (structure：行数、空行、标点合理性)

    参数
    ----
    lyrics : str
        草稿歌词文本（中文字符，可含换行符）。

    draft_jyutping : str
        草稿歌词的粤拼字符串（音节间空格分隔）。

    expected_tone_sequence : str
        预期的 lean 0243 序列，JSON 数组或空格分隔字符串，如 "[0,2,4,3,4]"。

    strong_beat_positions : str
        强拍音节索引列表，JSON 数组或空格分隔字符串，如 "[0,4,8,12]"。

    rhyme_positions : str
        押韵音节索引列表，JSON 数组或空格分隔字符串，如 "[7,15,23,31]"。

    expected_syllable_count : int
        MIDI 旋律要求的音节总数。

    返回
    ----
    str
        JSON 对象字符串，包含：
        {
          "score":   float,                    // 综合评分 0.0–1.0
          "passed":  bool,                     // score >= 0.75
          "dimension_scores": {
            "syllable_count":    float,
            "tonal_accuracy":    float,
            "rhyme_consistency": float,
            "structure":         float,
          },
          "syllable_report":  object,          // count_syllables 的完整输出
          "tone_report":      object,          // check_tone_accuracy 的完整输出
          "rhyme_report":     object,          // check_rhyme_scheme 的完整输出
          "corrections":      [str, ...],      // 中文修改建议列表
          "actual_syllable_count":   int,
          "expected_syllable_count": int,
        }
    """
    # ------------------------------------------------------------------
    # 1. Syllable count
    # ------------------------------------------------------------------
    syl_raw    = count_syllables(lyrics)
    syl_report = json.loads(syl_raw)
    actual_syl = syl_report["total"]

    if expected_syllable_count <= 0:
        dim_syllable = 0.5
    else:
        diff = abs(actual_syl - expected_syllable_count)
        if diff == 0:
            dim_syllable = 1.0
        elif diff == 1:
            dim_syllable = 0.70
        elif diff == 2:
            dim_syllable = 0.40
        elif diff == 3:
            dim_syllable = 0.20
        else:
            dim_syllable = max(0.0, 0.20 - (diff - 3) * 0.05)

    # ------------------------------------------------------------------
    # 2. Melody / tonal fit
    # ------------------------------------------------------------------
    tone_raw    = check_tone_accuracy(draft_jyutping, expected_tone_sequence, strong_beat_positions)
    tone_report = json.loads(tone_raw)
    dim_tonal   = float(tone_report.get("score", 0.5))

    # ------------------------------------------------------------------
    # 3. Rhyme consistency
    # ------------------------------------------------------------------
    rhyme_raw    = check_rhyme_scheme(draft_jyutping, rhyme_positions)
    rhyme_report = json.loads(rhyme_raw)
    dim_rhyme    = float(rhyme_report.get("score", 0.5))

    # ------------------------------------------------------------------
    # 4. Structure score
    # ------------------------------------------------------------------
    # Pure computational heuristics (no LLM required):
    # - At least 2 non-empty lines
    # - Each line has at least 3 CJK characters
    # - No line is more than 3× the average line length
    # - Jyutping syllable count roughly matches CJK character count
    structure_issues: list[str] = []
    lines = syl_report.get("lines", [])

    if len(lines) < 2:
        structure_issues.append("歌词少于两行，结构不完整。")

    if lines:
        avg_line = actual_syl / len(lines) if len(lines) > 0 else 0
        for row in lines:
            if row["count"] < 3:
                structure_issues.append(f"行「{row['text'][:10]}」过短（{row['count']} 字）。")
            if avg_line > 0 and row["count"] > avg_line * 3:
                structure_issues.append(f"行「{row['text'][:10]}」过长（{row['count']} 字，均值的 {row['count']/avg_line:.1f} 倍）。")

    # Jyutping coverage check
    jp_syllables = _extract_syllables(draft_jyutping)
    if actual_syl > 0 and jp_syllables:
        coverage_ratio = len(jp_syllables) / actual_syl
        if coverage_ratio < 0.8:
            structure_issues.append(
                f"粤拼覆盖率偏低（{len(jp_syllables)} 个粤拼 / {actual_syl} 个汉字 = {coverage_ratio:.0%}），"
                "可能存在漏标。"
            )
        elif coverage_ratio > 1.3:
            structure_issues.append(
                f"粤拼音节数（{len(jp_syllables)}）远多于汉字数（{actual_syl}），"
                "可能存在多余粤拼。"
            )

    dim_structure = max(0.0, 1.0 - len(structure_issues) * 0.20)

    # ------------------------------------------------------------------
    # 5. Composite score
    # ------------------------------------------------------------------
    composite = (
        dim_syllable  * _W_SYLLABLE
        + dim_tonal   * _W_TONAL
        + dim_rhyme   * _W_RHYME
        + dim_structure * _W_STRUCTURE
    )
    composite = round(min(max(composite, 0.0), 1.0), 4)

    # ------------------------------------------------------------------
    # 6. Build corrections list
    # ------------------------------------------------------------------
    corrections: list[str] = []

    # Syllable corrections
    if dim_syllable < 0.95 and expected_syllable_count > 0:
        diff = actual_syl - expected_syllable_count
        if diff > 0:
            corrections.append(
                f"【音节数】需删减 {diff} 个音节。当前歌词共 {actual_syl} 字，"
                f"MIDI 旋律要求恰好 {expected_syllable_count} 字。"
            )
        elif diff < 0:
            corrections.append(
                f"【音节数】需增加 {abs(diff)} 个音节。当前歌词只有 {actual_syl} 字，"
                f"MIDI 旋律要求恰好 {expected_syllable_count} 字。"
            )

    # Melody / tonal corrections
    if dim_tonal < 0.70:
        beat_issues = tone_report.get("strong_beat_issues", [])
        if beat_issues:
            positions_str = "、".join(
                f"第{iss['position']+1}个音节（声调{iss['tone']}）"
                for iss in beat_issues[:5]
            )
            corrections.append(
                f"【旋律贴合】以下强拍位置使用了不适合长音/重拍的入声字，建议替换为更贴合 lean 0243 落点的字：{positions_str}。"
            )
        positional = tone_report.get("positional_match", [])
        mismatches = [p for p in positional if not p["match"]]
        if mismatches:
            mismatch_str = "、".join(
                f"第{p['position']+1}位（实际{p['actual']}，预期{p['expected']}）"
                for p in mismatches[:4]
            )
            corrections.append(f"【旋律贴合】以下位置与目标 lean 0243 序列不贴合：{mismatch_str}。")

    # Rhyme corrections
    if dim_rhyme < 0.65:
        dominant = rhyme_report.get("dominant_final", "")
        unique    = rhyme_report.get("unique_finals", [])
        analysis  = rhyme_report.get("rhyme_positions_analysis", [])
        bad_rhymes = [
            e for e in analysis
            if not e.get("matches_dominant") and "error" not in e
        ]
        if bad_rhymes:
            bad_str = "、".join(
                f"第{e['position']+1}个音节「{e['syllable']}」韵尾-{e['final']}"
                for e in bad_rhymes[:3]
            )
            corrections.append(
                f"【押韵】押韵不一致：主韵为 -{dominant}，但以下位置韵尾不符：{bad_str}。"
                f"建议将这些位置替换为韵尾 -{dominant} 的字。"
            )
        elif len(unique) > 2:
            corrections.append(
                f"【押韵】韵脚散乱，出现了 {len(unique)} 种不同韵尾：{unique}。"
                "建议选定一个主押韵并统一全段。"
            )

    # Structure corrections
    for issue in structure_issues:
        corrections.append(f"【结构】{issue}")

    # ------------------------------------------------------------------
    # 7. Assemble final result
    # ------------------------------------------------------------------
    result: dict[str, Any] = {
        "score":   composite,
        "passed":  composite >= 0.75,
        "melody_0243_score": round(dim_tonal, 4),
        "dimension_scores": {
            "syllable_count":    round(dim_syllable,   4),
            "tonal_accuracy":    round(dim_tonal,      4),
            "rhyme_consistency": round(dim_rhyme,      4),
            "structure":         round(dim_structure,  4),
        },
        "syllable_report":         syl_report,
        "tone_report":             tone_report,
        "rhyme_report":            rhyme_report,
        "corrections":             corrections,
        "actual_syllable_count":   actual_syl,
        "expected_syllable_count": expected_syllable_count,
    }

    logger.info(
        "score_lyrics: composite=%.3f passed=%s syllable=%.2f melody0243=%.2f "
        "rhyme=%.2f structure=%.2f corrections=%d",
        composite, result["passed"],
        dim_syllable, dim_tonal, dim_rhyme, dim_structure,
        len(corrections),
    )
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 5: suggest_corrections
# ---------------------------------------------------------------------------

@mcp.tool()
def suggest_corrections(score_report: str) -> str:
    """
    根据 score_lyrics 生成的评分报告，输出优先级排序的中文修改建议列表。

    此工具将 score_lyrics 的 JSON 报告转换为易于阅读的修改指令，
    并按严重程度排序——音节数问题最优先（影响整体结构），
    其次是声调问题（影响演唱自然度），再次是押韵，最后是结构。

    参数
    ----
    score_report : str
        由 score_lyrics 工具返回的 JSON 字符串。

    返回
    ----
    str
        JSON 对象字符串，包含：
        {
          "total_issues": int,
          "passed": bool,
          "score": float,
          "prioritized_corrections": [
            {
              "priority": int,        // 1 = 最高优先级
              "category": str,        // "音节数" / "声调" / "押韵" / "结构"
              "severity": str,        // "严重" / "中等" / "轻微"
              "instruction": str,     // 具体修改指令（中文）
              "dimension_score": float
            },
            ...
          ],
          "summary": str              // 一句话总结
        }
    """
    try:
        report = json.loads(score_report)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"无法解析评分报告 JSON: {exc}"}, ensure_ascii=False)

    composite  = float(report.get("score", 0.0))
    passed     = bool(report.get("passed", False))
    dim_scores = report.get("dimension_scores", {})
    raw_corrections: list[str] = report.get("corrections", [])

    # Map each correction to a category, severity and priority
    _CATEGORY_MAP = {
        "【音节数】": ("音节数", 1),
        "【旋律贴合】": ("旋律贴合", 2),
        "【声调】":   ("声调",   2),
        "【押韵】":   ("押韵",   3),
        "【结构】":   ("结构",   4),
    }
    _SCORE_KEY_MAP = {
        "音节数": "syllable_count",
        "旋律贴合": "tonal_accuracy",
        "声调":   "tonal_accuracy",
        "押韵":   "rhyme_consistency",
        "结构":   "structure",
    }

    def _severity(score: float) -> str:
        if score < 0.40:
            return "严重"
        elif score < 0.70:
            return "中等"
        else:
            return "轻微"

    prioritized: list[dict[str, Any]] = []
    for instruction in raw_corrections:
        category = "其他"
        priority = 5
        for prefix, (cat, pri) in _CATEGORY_MAP.items():
            if instruction.startswith(prefix):
                category = cat
                priority = pri
                break

        score_key = _SCORE_KEY_MAP.get(category, "")
        dim_score = float(dim_scores.get(score_key, 0.5)) if score_key else 0.5

        prioritized.append({
            "priority":        priority,
            "category":        category,
            "severity":        _severity(dim_score),
            "instruction":     instruction,
            "dimension_score": round(dim_score, 4),
        })

    # Sort by priority (ascending), then by dimension score (ascending = worse first)
    prioritized.sort(key=lambda x: (x["priority"], x["dimension_score"]))

    # Build one-line summary
    if passed:
        summary = f"歌词质量通过验收（综合评分 {composite:.3f}），可进入最终输出阶段。"
    elif not prioritized:
        summary = f"综合评分 {composite:.3f}，略低于阈值，建议整体润色。"
    else:
        top = prioritized[0]
        summary = (
            f"综合评分 {composite:.3f}，未通过验收。最优先修改：{top['category']}（{top['severity']}）— "
            + top["instruction"][:60] + ("…" if len(top["instruction"]) > 60 else "")
        )

    result: dict[str, Any] = {
        "total_issues":           len(prioritized),
        "passed":                 passed,
        "score":                  composite,
        "prioritized_corrections": prioritized,
        "summary":                summary,
    }

    logger.info(
        "suggest_corrections: score=%.3f passed=%s issues=%d",
        composite, passed, len(prioritized),
    )
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("歌词校验 MCP 服务器启动（stdio 传输）")
    mcp.run(transport="stdio")
