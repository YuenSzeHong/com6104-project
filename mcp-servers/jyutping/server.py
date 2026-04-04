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
import os
import re
import sys
import time
from pathlib import Path
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
_LOCAL_POSTFIX_PATH = Path(__file__).parent / "data" / "postfix_m1_all.json"
_CACHE_TTL_SECONDS = int(os.getenv("JYUTPING_CACHE_TTL_SECONDS", "600"))

# Matches a complete Jyutping syllable: letters + tone digit 1-6
# e.g. "sing1", "tau3", "ming4"
_JP_SYLLABLE_RE = re.compile(r"[a-z]+[1-6]", re.IGNORECASE)

# Matches a purely numeric string (the tone codes returned by the API)
_NUMERIC_RE = re.compile(r"^\d+$")
_API_CACHE: dict[str, tuple[float, list[str]]] = {}


def _cache_get(query: str) -> list[str] | None:
    if _CACHE_TTL_SECONDS <= 0:
        return None
    item = _API_CACHE.get(query)
    if item is None:
        return None
    expires_at, value = item
    if time.time() >= expires_at:
        _API_CACHE.pop(query, None)
        return None
    return value


def _cache_set(query: str, value: list[str]) -> None:
    if _CACHE_TTL_SECONDS <= 0:
        return
    _API_CACHE[query] = (time.time() + _CACHE_TTL_SECONDS, value)


def _load_local_postfix_map() -> dict[str, list[str]]:
    """Load local tone-code fallback lexicon from a JSON snapshot file."""
    if not _LOCAL_POSTFIX_PATH.exists():
        logger.info("Local postfix snapshot not found: %s", _LOCAL_POSTFIX_PATH)
        return {}

    try:
        payload = json.loads(_LOCAL_POSTFIX_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load local postfix snapshot: %s", exc)
        return {}

    if not isinstance(payload, dict):
        logger.warning(
            "Invalid local postfix snapshot format: %s",
            type(payload).__name__,
        )
        return {}

    normalized: dict[str, list[str]] = {}
    for key, value in payload.items():
        tone_code = str(key).strip()
        if not tone_code or not _NUMERIC_RE.match(tone_code):
            continue
        if not isinstance(value, list):
            continue

        words: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and re.search(r"[\u4e00-\u9fff]", text):
                words.append(text)

        if words:
            normalized[tone_code] = list(dict.fromkeys(words))

    logger.info(
        "Loaded local postfix snapshot: %d tone codes from %s",
        len(normalized),
        _LOCAL_POSTFIX_PATH,
    )
    return normalized


def _merge_words(remote_words: list[str], local_words: list[str]) -> list[str]:
    """Merge remote and local candidate lists with stable de-duplication."""
    merged: list[str] = []
    for word in [*remote_words, *local_words]:
        if not word:
            continue
        if word not in merged:
            merged.append(word)
    return merged


_LOCAL_POSTFIX_MAP: dict[str, list[str]] = _load_local_postfix_map()


async def _call_api(nums: str | list[str]) -> list[Any]:
    """
    Wrapper that supports single or batch queries to the 0243.hk API.

    Returns a flat list of strings for single queries, or a list of lists for batch
    queries. The static return type is relaxed to `list[Any]` to simplify downstream
    typing and avoid false-positive static-type errors when callers receive either
    a `list[str]` or `list[list[str]]`.
    """

    async def _single_call(query: str, client: httpx.AsyncClient) -> list[str]:
        cached = _cache_get(query)
        if cached is not None:
            return list(cached)

        payload = {"nums": query}
        last_exc = None
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
                    parsed = [str(item) for item in data if item is not None]
                    _cache_set(query, parsed)
                    return parsed

                logger.warning("0243.hk 返回非数组响应: %s", type(data).__name__)
                return []

            except httpx.HTTPStatusError as exc:
                last_exc = exc
                logger.warning(
                    "0243.hk HTTP 错误（第 %d/%d 次）: %s",
                    attempt, _MAX_RETRIES, getattr(exc.response, "status_code", "?")
                )
            except httpx.RequestError as exc:
                last_exc = exc
                logger.warning(
                    "0243.hk 请求错误（第 %d/%d 次）: %s",
                    attempt, _MAX_RETRIES, exc,
                )
            except json.JSONDecodeError as exc:
                last_exc = exc
                logger.warning(
                    "0243.hk JSON 解析失败（第 %d/%d 次）: %s",
                    attempt, _MAX_RETRIES, exc,
                )

            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_RETRY_DELAY * attempt)

        # All retries exhausted, raise the last exception
        if last_exc is not None:
            raise last_exc
        return []

    # Batch path
    if isinstance(nums, (list, tuple)):
        if not nums:
            return []
        normalized_queries = [str(q) for q in nums]
        cached_results: dict[str, list[str]] = {}
        missing_queries: list[str] = []
        for query in normalized_queries:
            cached = _cache_get(query)
            if cached is not None:
                cached_results[query] = list(cached)
            else:
                missing_queries.append(query)

        fetched_results: dict[str, list[str]] = {}
        if missing_queries:
            semaphore = asyncio.Semaphore(8)
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:

                async def _fetch(q: str) -> tuple[str, list[str]]:
                    async with semaphore:
                        return q, await _single_call(q, client)

                tasks = [_fetch(query) for query in missing_queries]
                fetched_pairs = await asyncio.gather(*tasks)
            fetched_results = {query: payload for query, payload in fetched_pairs}

        return [
            fetched_results.get(query, cached_results.get(query, []))
            for query in normalized_queries
        ]

    # Single path
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        return await _single_call(str(nums), client)


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
# Tool 5: Numeric tone code → Chinese words (single or batch)
# ---------------------------------------------------------------------------

@mcp.tool()
async def find_words_by_tone_code(code: str | int | list[Any]) -> str:
    """
    根据数字声调码查找具有该声调模式的中文词语。

    支持单个 code（str 或 int）或批量输入（list/tuple）。
    - 单个输入返回 JSON 数组字符串：["羅曼蒂克", ...]
    - 批量输入返回 JSON 数组的数组：[ ["羅曼蒂克", ...], ["年事已高", ...], ... ]

    参数
    ----
    code : str | int | list[str]
        数字字符串或数字，或字符串列表。

    返回
    ----
    str
        JSON 字符串（数组或数组的数组）。
    """
    if code is None:
        return "[]"

    # Batch handling
    if isinstance(code, (list, tuple)):
        if not code:
            return "[]"
        # Normalize to string tokens
        normalized = [str(c).strip() for c in code]

        # Validate numeric tokens
        for token in normalized:
            if not token:
                return "[]"
            if not _NUMERIC_RE.match(token):
                logger.warning("find_words_by_tone_code (batch): 非数字输入 %r", token)
                return json.dumps({"error": f"输入必须为纯数字字符串，收到：{token!r}"})

        logger.info("find_words_by_tone_code: batch codes=%r", normalized)
        try:
            candidates_list = await _call_api(normalized)  # Expected: list[list[str]]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "find_words_by_tone_code (batch): remote API failed, fallback to local snapshot: %s",
                exc,
            )
            candidates_list = []

        results: list[list[str]] = []
        for idx, tone_code in enumerate(normalized):
            candidates = candidates_list[idx] if idx < len(candidates_list) else []
            # Filter Chinese results for each sublist
            chinese_words = [c for c in candidates if _is_chinese(c)]
            remote_words = chinese_words if chinese_words else candidates
            local_words = _LOCAL_POSTFIX_MAP.get(tone_code, [])
            results.append(_merge_words(remote_words, local_words))

        logger.info("find_words_by_tone_code: batch → %d items", len(results))
        return json.dumps(results, ensure_ascii=False)

    # Single handling
    code = str(code).strip()
    if not code:
        return "[]"
    if not _NUMERIC_RE.match(code):
        logger.warning("find_words_by_tone_code: 非数字输入 %r", code)
        return json.dumps({"error": f"输入必须为纯数字字符串，收到：{code!r}"})

    logger.info("find_words_by_tone_code: code=%r", code)
    try:
        candidates = await _call_api(code)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "find_words_by_tone_code: remote API failed, fallback to local snapshot: %s",
            exc,
        )
        candidates = []

    # Return only Chinese word results (filter out Jyutping and numeric responses)
    chinese_words = [c for c in candidates if _is_chinese(c)]

    # If no Chinese-only results, return everything (some codes return mixed)
    remote_words = chinese_words if chinese_words else candidates
    local_words = _LOCAL_POSTFIX_MAP.get(code, [])
    results = _merge_words(remote_words, local_words)

    logger.info(
        "find_words_by_tone_code: code=%r → %d 个结果", code, len(results)
    )
    return json.dumps(results, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 6: Tone-constrained continuation (mixed mode; single or batch)
# ---------------------------------------------------------------------------

@mcp.tool()
async def find_tone_continuation(chinese_prefix: str | list[Any], tone_digits: str | int | list[Any]) -> str:
    """
    在已有粤语文本后面，查找具有指定声调模式的续接词语。

    支持单个 prefix+digits，也支持批量（两者为 list/tuple）。
    - 单个输入返回 JSON 数组字符串：["那麼", "有多", ...]
    - 批量输入返回 JSON 数组的数组：[ ["那麼","有多"], ["這麼","更多"], ... ]

    参数
    ----
    chinese_prefix : str | list[str]
        已确定的中文歌词文本，或者文本列表。
    tone_digits : str | int | list[int]
        续接词语所需的声调数字序列（纯数字字符串），或者对应的列表。

    返回
    ----
    str
        JSON 字符串（数组或数组的数组）。
    """
    # Normalize None/empty guard
    if (chinese_prefix is None) or (tone_digits is None):
        return "[]"

    # Batch / mixed scenarios
    # Case A: both lists -> pairwise combination
    if isinstance(chinese_prefix, (list, tuple)) or isinstance(tone_digits, (list, tuple)):
        # Normalize prefixes to list
        if isinstance(chinese_prefix, (list, tuple)):
            prefixes = [str(p).strip() for p in chinese_prefix]
        else:
            prefixes = [str(chinese_prefix).strip()]

        # Normalize tone digits to list of strings
        if isinstance(tone_digits, (list, tuple)):
            tones = [str(t).strip() for t in tone_digits]
        else:
            tones = [str(tone_digits).strip()]

        # Determine pairing strategy:
        # - If lengths equal -> zip pairwise
        # - If one is length 1 -> broadcast
        # - Else if both >1 and lengths unequal -> error
        if len(prefixes) == len(tones):
            pairs = list(zip(prefixes, tones))
        elif len(prefixes) == 1 and len(tones) >= 1:
            pairs = [(prefixes[0], t) for t in tones]
        elif len(tones) == 1 and len(prefixes) >= 1:
            pairs = [(p, tones[0]) for p in prefixes]
        else:
            logger.warning("find_tone_continuation: prefixes and tones length mismatch")
            return json.dumps({"error": "prefix list and tone_digits list must have equal length or one must be length 1"})

        # Validate and build queries
        queries: list[str] = []
        for p, t in pairs:
            if not p:
                queries.append(t)  # fall back to just digits although odd; keeps behavior consistent
                continue
            if not t:
                queries.append(p)
                continue
            if not _NUMERIC_RE.match(t):
                logger.warning("find_tone_continuation (batch): 非数字 tone_digits %r", t)
                return json.dumps({"error": f"tone_digits 必须为纯数字字符串，收到：{t!r}"})
            queries.append(f"{p}{t}")

        logger.info("find_tone_continuation: batch queries=%r", queries)
        candidates_list = await _call_api(queries)  # expected: list[list[str]]

        results: list[list[str]] = []
        for candidates in candidates_list:
            chinese_results = [c for c in candidates if _is_chinese(c)]
            results.append(chinese_results if chinese_results else candidates)

        logger.info("find_tone_continuation: batch → %d items", len(results))
        return json.dumps(results, ensure_ascii=False)

    # Single path (both single scalars)
    chinese_prefix = str(chinese_prefix).strip()
    tone_digits = str(tone_digits).strip()
    if not chinese_prefix or not tone_digits:
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
