"""
Jyutping MCP Server
===================
Exposes Cantonese language tools backed by the 0243.hk search API.

The 0243.hk API is a Cantonese IME (Input Method Engine) search endpoint that
supports THREE distinct query modes — all via the same POST /api/cls/ endpoint:

Mode 1 — Numeric tone code → Chinese words
    Input a numeric string (e.g. "0243", "43")
    Returns: Chinese words/phrases whose tones match that code pattern.
    Use case: "I need a 2-syllable word with tones 4 then 3 — what fits?"

Mode 2 — Chinese text → Jyutping candidates + codes + related phrases
    Input Chinese characters (e.g. "透明嘅彗星")
    Returns: Jyutping romanisation candidates, the numeric code for that text,
             and phonetically/semantically related Chinese phrases.
    Use case: "How is this text pronounced in Cantonese?"

Mode 3 — Mixed prefix + tone digits (continuation/completion)
    Input existing Chinese text followed by tone digit(s) (e.g. "我要43")
    Returns: Chinese words/phrases that could follow the prefix text,
             constrained to having the specified tone pattern.
    Use case: "After '我要', give me words with tones 4 then 3."

Tools
-----
query_raw(nums)
    Direct pass-through to the 0243.hk API. Returns the raw response array.
    Useful when you want full control over the query format.

chinese_to_jyutping(text)
    Mode 2: Convert Chinese text to Jyutping romanisation candidates.
    Returns only the Jyutping strings from the response (sorted to front).

get_tone_code(text)
    Mode 2: Get the numeric tone code(s) for a Chinese phrase.
    Returns only the numeric strings from the response.

get_tone_pattern(text)
    Mode 2: Extract the space-separated tone digit sequence (e.g. "3 4 3 6 1")
    from the best Jyutping candidate for the input text.

find_words_by_tone_code(code)
    Mode 1: Given a numeric tone code, return Chinese words that match it.
    E.g. "43" → words with tone-4 then tone-3 syllables.
    Core tool for the lyrics composer to find tonally-constrained word candidates.

find_tone_continuation(chinese_prefix, tone_digits)
    Mode 3: Given existing Chinese lyrics text + required tone digits for the
    NEXT syllable(s), return Chinese words that fit.
    E.g. prefix="青山", tone_digits="13" → words with tone-1 then tone-3
    that could follow "青山" in lyrics.
    Most powerful tool for tone-aware lyric completion.

Transport
---------
stdio  (compatible with langchain-mcp-adapters MultiServerMCPClient)

Testing with MCP Inspector
--------------------------
    npx @modelcontextprotocol/inspector python mcp-servers/jyutping/server.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[jyutping-server] %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("jyutping-server")

# ---------------------------------------------------------------------------
# 0243.hk API client
# ---------------------------------------------------------------------------

_API_URL        = "https://www.0243.hk/api/cls/"
_TIMEOUT        = 15.0   # seconds
_MAX_RETRIES    = 3
_RETRY_DELAY    = 1.0    # seconds between retries

# Matches a complete Jyutping syllable: letters + tone digit 1-6
# e.g. "sing1", "tau3", "ming4"
_JP_SYLLABLE_RE = re.compile(r"[a-z]+[1-6]", re.IGNORECASE)

# Matches a purely numeric string (the tone codes returned by the API)
_NUMERIC_RE = re.compile(r"^\d+$")


async def _call_api(nums: str) -> list[str]:
    """
    POST {"nums": nums} to https://www.0243.hk/api/cls/.

    Returns the JSON array of candidate strings, or [] on error.
    Retries up to _MAX_RETRIES times on transient failures.
    """
    payload = {"nums": nums}

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = await client.post(
                    _API_URL,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept":       "application/json",
                        "User-Agent":   "CantoneseAgent/1.0 (MCP; 0243.hk)",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

                if isinstance(data, list):
                    return [str(item) for item in data if item is not None]

                logger.warning("0243.hk 返回非数组响应: %s", type(data).__name__)
                return []

            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "0243.hk HTTP 错误（第 %d/%d 次）: %s",
                    attempt, _MAX_RETRIES, exc.response.status_code,
                )
            except httpx.RequestError as exc:
                logger.warning(
                    "0243.hk 请求错误（第 %d/%d 次）: %s",
                    attempt, _MAX_RETRIES, exc,
                )
            except json.JSONDecodeError as exc:
                logger.warning(
                    "0243.hk JSON 解析失败（第 %d/%d 次）: %s",
                    attempt, _MAX_RETRIES, exc,
                )
                return []

            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_DELAY * attempt)

    return []


# ---------------------------------------------------------------------------
# Response classifiers
# ---------------------------------------------------------------------------

def _is_jyutping(s: str) -> bool:
    """Return True if s looks like a Jyutping romanisation string.

    A Jyutping string contains at least one syllable+tone pair and no CJK chars.
    E.g. "tau3 ming4 ge3 seoi6 sing1" → True
         "羅曼蒂克" → False
    """
    if re.search(r"[\u4e00-\u9fff]", s):
        return False
    return bool(_JP_SYLLABLE_RE.search(s))


def _is_numeric_code(s: str) -> bool:
    """Return True if s is a purely numeric string (a tone code)."""
    return bool(_NUMERIC_RE.match(s.strip()))


def _is_chinese(s: str) -> bool:
    """Return True if s contains any CJK characters."""
    return bool(re.search(r"[\u4e00-\u9fff]", s))


def _extract_tones_from_jyutping(jp_string: str) -> list[int]:
    """Extract tone numbers from a Jyutping string.

    "tau3 ming4 ge3 seoi6 sing1" → [3, 4, 3, 6, 1]
    """
    tones: list[int] = []
    for syllable in _JP_SYLLABLE_RE.findall(jp_string.lower()):
        try:
            tones.append(int(syllable[-1]))
        except ValueError:
            pass
    return tones


def _best_jyutping(candidates: list[str]) -> str:
    """
    From a mixed API response, pick the best Jyutping reading.

    Preference:
    1. First candidate where every space-separated token is a valid Jyutping syllable
       (full Jyutping reading, no Chinese mixed in).
    2. First candidate that contains any Jyutping syllable.
    3. Empty string if none found.
    """
    # Full Jyutping reading: all tokens are syllable+tone
    for c in candidates:
        tokens = c.strip().split()
        if tokens and all(_JP_SYLLABLE_RE.fullmatch(t.lower()) for t in tokens):
            return c

    # Partial Jyutping reading
    for c in candidates:
        if _is_jyutping(c):
            return c

    return ""


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="jyutping-mcp-server",
    instructions=(
        "粤语语言工具服务器：通过 0243.hk IME 引擎提供粤拼转换、"
        "声调码查词及声调约束续词功能。"
    ),
)


# ---------------------------------------------------------------------------
# Tool 1: Raw pass-through
# ---------------------------------------------------------------------------

@mcp.tool()
async def query_raw(nums: str) -> str:
    """
    直接调用 0243.hk API，返回原始响应数组（JSON 字符串）。

    0243.hk 接受三种输入格式：
    - 纯数字码（如 "0243", "43"）→ 返回具有该声调码的中文词语
    - 中文文本（如 "透明嘅彗星"）→ 返回粤拼候选、数字码及近音词
    - 中文+数字混合（如 "我要43"）→ 返回在该前缀文本之后、具有指定声调模式的词语

    参数
    ----
    nums : str
        传入 0243.hk API 的查询字符串。

    返回
    ----
    str
        JSON 数组字符串，包含所有候选结果（粤拼字符串、数字码、中文词语的混合）。
    """
    logger.info("query_raw: nums=%r", nums)
    results = await _call_api(nums)
    return json.dumps(results, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 2: Chinese text → Jyutping candidates
# ---------------------------------------------------------------------------

@mcp.tool()
async def chinese_to_jyutping(text: str) -> str:
    """
    将中文文本转换为粤拼罗马化候选列表。

    调用 0243.hk API（Mode 2），从响应中筛选出粤拼字符串，
    排在最前面的是完整的粤拼读音（所有音节均含声调数字）。

    参数
    ----
    text : str
        输入的中文文本，例如 "透明嘅彗星" 或 "青山依舊在"。

    返回
    ----
    str
        JSON 数组字符串，仅包含粤拼候选（已过滤掉数字码和中文词语）。
        示例：["tau3 ming4 ge3 seoi6 sing1", "tau3 ming4 ge2 seoi6 sing1"]
        若无粤拼结果则返回 "[]"。
    """
    if not text or not text.strip():
        return "[]"

    logger.info("chinese_to_jyutping: text=%r", text[:60])
    candidates = await _call_api(text.strip())

    jyutping_candidates = [c for c in candidates if _is_jyutping(c)]

    # Sort: full Jyutping readings (all tokens are syllable+tone) first
    def _is_full_jp(s: str) -> bool:
        tokens = s.strip().split()
        return bool(tokens and all(_JP_SYLLABLE_RE.fullmatch(t.lower()) for t in tokens))

    full_jp    = [c for c in jyutping_candidates if _is_full_jp(c)]
    partial_jp = [c for c in jyutping_candidates if not _is_full_jp(c)]
    ordered    = full_jp + partial_jp

    logger.info(
        "chinese_to_jyutping: %d 个粤拼候选（%d 完整 + %d 部分）",
        len(ordered), len(full_jp), len(partial_jp),
    )
    return json.dumps(ordered, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 3: Chinese text → numeric tone code
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_tone_code(text: str) -> str:
    """
    获取中文文本在 0243.hk 系统中对应的数字声调码。

    调用 0243.hk API（Mode 2），从响应中提取纯数字码。
    这些数字码代表该文本的粤语声调模式编码，
    可反向用于 find_words_by_tone_code 工具查找同声调码的词语。

    参数
    ----
    text : str
        输入的中文文本，例如 "透明嘅彗星"。

    返回
    ----
    str
        JSON 数组字符串，包含所有数字码。
        示例：["40423", "40923"]
        若无数字码结果则返回 "[]"。
    """
    if not text or not text.strip():
        return "[]"

    logger.info("get_tone_code: text=%r", text[:60])
    candidates = await _call_api(text.strip())

    codes = [c for c in candidates if _is_numeric_code(c)]
    logger.info("get_tone_code: 找到 %d 个数字码", len(codes))
    return json.dumps(codes, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 4: Chinese text → space-separated tone digit sequence
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_tone_pattern(text: str) -> str:
    """
    返回中文文本每个音节的粤语声调数字序列（空格分隔）。

    调用 0243.hk API（Mode 2），选取最佳粤拼候选，
    提取每个音节的声调数字（1-6），以空格分隔返回。

    参数
    ----
    text : str
        输入的中文文本，例如 "透明嘅彗星"。

    返回
    ----
    str
        空格分隔的声调数字序列，例如 "3 4 3 6 1"。
        声调含义：1=高平, 2=高升, 3=中平, 4=低降, 5=低升, 6=低平。
        若无法确定声调则返回空字符串 ""。
    """
    if not text or not text.strip():
        return ""

    logger.info("get_tone_pattern: text=%r", text[:60])
    candidates = await _call_api(text.strip())

    best_jp = _best_jyutping(candidates)
    if not best_jp:
        logger.warning("get_tone_pattern: 找不到粤拼候选，text=%r", text[:60])
        return ""

    tones = _extract_tones_from_jyutping(best_jp)
    if not tones:
        logger.warning("get_tone_pattern: 无法从粤拼中提取声调，jp=%r", best_jp)
        return ""

    result = " ".join(str(t) for t in tones)
    logger.info("get_tone_pattern: %r → %r → %r", text[:40], best_jp[:40], result)
    return result


# ---------------------------------------------------------------------------
# Tool 5: Numeric tone code → Chinese words
# ---------------------------------------------------------------------------

@mcp.tool()
async def find_words_by_tone_code(code: str | int) -> str:
    """
    根据数字声调码查找具有该声调模式的中文词语。

    调用 0243.hk API（Mode 1）。0243.hk 是一个粤语输入法引擎，
    数字码代表粤语声调序列——输入 "43" 即可找到所有
    第一个音节为声调4、第二个音节为声调3的双音节词语。

    这是歌词创作的核心工具：当旋律要求某位置的字必须具有特定声调模式时，
    用此工具快速找到符合要求的候选词语。

    参数
    ----
    code : str
        数字字符串，每位数字代表一个音节的声调。
        例如：
        - "1"  → 所有声调1（高平）的单音节字
        - "43" → 声调4后接声调3的双音节词
        - "0243" → 四音节词，声调依次为 0-2-4-3

    返回
    ----
    str
        JSON 数组字符串，包含具有该声调码的中文词语。
        示例（code="0243"）：["羅曼蒂克", "權力鬥爭", "年事已高", ...]
        若无结果则返回 "[]"。
    """
    if code is None:
        return "[]"

    code = str(code).strip()
    if not code:
        return "[]"
    if not _NUMERIC_RE.match(code):
        logger.warning("find_words_by_tone_code: 非数字输入 %r", code)
        return json.dumps({"error": f"输入必须为纯数字字符串，收到：{code!r}"})

    logger.info("find_words_by_tone_code: code=%r", code)
    candidates = await _call_api(code)

    # Return only Chinese word results (filter out Jyutping and numeric responses)
    chinese_words = [c for c in candidates if _is_chinese(c)]

    # If no Chinese-only results, return everything (some codes return mixed)
    results = chinese_words if chinese_words else candidates

    logger.info(
        "find_words_by_tone_code: code=%r → %d 个结果", code, len(results)
    )
    return json.dumps(results, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 6: Tone-constrained continuation (mixed mode)
# ---------------------------------------------------------------------------

@mcp.tool()
async def find_tone_continuation(chinese_prefix: str, tone_digits: str | int) -> str:
    """
    在已有粤语文本后面，查找具有指定声调模式的续接词语。

    调用 0243.hk API（Mode 3：中文前缀 + 数字后缀的混合查询）。
    这是歌词创作最强大的工具：在已填入的歌词之后，
    找出声调模式符合旋律要求的候选汉字/词语。

    工作原理
    --------
    0243.hk 的混合查询模式（如 "我要43"）会将中文前缀作为上下文，
    返回在该前缀之后、声调序列与数字后缀匹配的词语候选。
    这让歌词创作代理能够做到"上下文感知的声调约束补全"。

    参数
    ----
    chinese_prefix : str
        已确定的中文歌词文本，作为查询前缀。
        例如："青山" 或 "縱然" 或 "我要"

    tone_digits : str
        续接词语所需的声调数字序列（纯数字字符串）。
        例如：
        - "1"  → 下一个字为声调1（高平）
        - "13" → 接下来两个字声调依次为1、3
        - "141"→ 接下来三个字声调依次为1、4、1

    返回
    ----
    str
        JSON 数组字符串，包含可续接在前缀之后、且声调符合要求的中文词语。
        示例（prefix="我要", tone_digits="43"）：
        ["那麼", "有多", "這麼", "更多", "唱歌", ...]
        若无结果则返回 "[]"。

    使用示例
    --------
    歌词已写到 "青山依"，旋律下一个音节要求声调6：
        find_tone_continuation("青山依", "6")
        → 找到声调6的字填入第4个位置

    已写 "縱然"，接下来两音节需要声调1、4：
        find_tone_continuation("縱然", "14")
        → 找到适合续接的声调1+4双音节词
    """
    if not chinese_prefix or tone_digits is None:
        return "[]"

    chinese_prefix = chinese_prefix.strip()
    tone_digits    = str(tone_digits).strip()
    if not tone_digits:
        return "[]"

    if not _NUMERIC_RE.match(tone_digits):
        return json.dumps(
            {"error": f"tone_digits 必须为纯数字字符串，收到：{tone_digits!r}"}
        )

    # Combine into the mixed query format: Chinese text + digit suffix
    query = f"{chinese_prefix}{tone_digits}"
    logger.info(
        "find_tone_continuation: prefix=%r + tones=%r → query=%r",
        chinese_prefix[:30], tone_digits, query[:40],
    )

    candidates = await _call_api(query)

    # The continuation mode typically returns Chinese words; filter accordingly.
    # If response contains mixed types, prefer Chinese results.
    chinese_results = [c for c in candidates if _is_chinese(c)]
    results = chinese_results if chinese_results else candidates

    logger.info(
        "find_tone_continuation: %r → %d 个候选词",
        query[:30], len(results),
    )
    return json.dumps(results, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("粤拼 MCP 服务器启动（stdio 传输）")
    mcp.run(transport="stdio")
