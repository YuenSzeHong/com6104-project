"""
jyutping_batch.py

Helper utilities to build and send batched queries to the 0243.hk-backed jyutping
MCP server. These helpers are async and intentionally accept a user-provided
`call_api` coroutine (or async callable) so they do not depend on any
particular network client implementation.

Primary functions:
- batch_find_words_by_tone_code(...)
- batch_find_tone_continuation(...)

Both functions return a dict with:
- results: list of lists; each entry corresponds to one input query and is a list
  of candidate strings returned by the upstream API.
- tool_calls_made: list[str] describing which batch calls were issued (useful
  for auditing / embedding in agent outputs).

Usage example
-------------
async def my_call_api(arg):
    # wrapper that delegates to the MCP client (provided by the caller)
    return await my_mcp_call(arg)

out = await batch_find_words_by_tone_code(["0","43","14"], call_api=my_call_api)
# out["results"] -> [["..."], ["..."], ["..."]]
# out["tool_calls_made"] -> ['find_words_by_tone_code(["0","43","14"])']
"""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Sequence, Tuple

# Regex helper to validate numeric tone digit strings (e.g. "43", "0243")
_NUMERIC_RE = re.compile(r"^\d+$")


# Type for the provided API caller: it should accept either a single str or a list[str]
# and return awaitable result which is typically list[str] (for single) or list[list[str]] (for batch).
ApiCaller = Callable[[Any], Awaitable[Any]]


def _chunked(iterable: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    """Yield successive chunks of `size` from `iterable` (preserves order)."""
    if size <= 0:
        raise ValueError("chunk size must be > 0")
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def _ensure_str_list(x: Any) -> List[str]:
    """Coerce an iterable of scalars to list[str]."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    try:
        return [str(i) for i in list(x)]
    except Exception:
        return [str(x)]


def _normalize_api_batch_response(resp: Any, expected_len: int) -> List[List[str]]:
    """
    Normalize the upstream API response into a list-of-lists of strings.

    The upstream `_call_api` may return:
      - list[str]              (for single query)
      - list[list[str]]        (for batch query)
      - other (coerce as best-effort)

    This function ensures the returned structure has exactly `expected_len`
    items; if only a single list is returned for a batch, it will replicate it
    per expected_len to avoid downstream index errors.
    """
    if resp is None:
        return [[] for _ in range(expected_len)]

    # If resp is a list of lists -> likely a proper batch response
    if isinstance(resp, list) and resp and all(isinstance(i, list) for i in resp):
        # Coerce inner elements to str lists
        out: List[List[str]] = []
        for sub in resp:
            out.append([str(s) for s in sub if s is not None])
        # If resp length differs from expected_len, try to adapt:
        if len(out) == expected_len:
            return out
        if len(out) < expected_len:
            # pad with empty lists
            out.extend([[] for _ in range(expected_len - len(out))])
            return out
        # if more than expected, truncate
        return out[:expected_len]

    # If resp is a flat list (list[str]) and only one response expected, wrap it.
    if isinstance(resp, list) and all(not isinstance(i, list) for i in resp):
        if expected_len == 1:
            return [[str(i) for i in resp if i is not None]]
        # Broadcast the same candidate list for each expected element
        out = [[str(i) for i in resp if i is not None] for _ in range(expected_len)]
        return out

    # If resp is a scalar or otherwise, coerce into repeated singletons
    str_list = [str(resp)]
    return [str_list for _ in range(expected_len)]


async def _call_in_chunks(
    queries: List[str],
    call_api: ApiCaller,
    batch_size: int,
    tool_label: str,
    record_calls: List[str],
) -> List[List[str]]:
    """
    Helper: call `call_api` with chunks of `queries`, each chunk of size up to
    `batch_size`. `call_api` is expected to accept the chunk (list[str]) and
    return a batch-shaped response.

    - `tool_label` is used to record the textual description of the call.
    - `record_calls` is an output list to append human-readable call descriptions.

    Returns a flattened list-of-lists of candidates in the same order as `queries`.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    results: List[List[str]] = []
    # Process chunks sequentially to avoid overloading upstream; calling code can
    # parallelize if desired.
    for chunk in _chunked(queries, batch_size):
        # Prepare human-readable call record
        try:
            label = f"{tool_label}({json.dumps(chunk, ensure_ascii=False)})"
        except Exception:
            label = f"{tool_label}({str(chunk)})"
        record_calls.append(label)

        # Call the provided API; callers may implement retries/backoff around their client.
        raw = await call_api(list(chunk))
        normalized = _normalize_api_batch_response(raw, len(chunk))
        results.extend(normalized)
    return results


# Public helpers -------------------------------------------------------------


async def batch_find_words_by_tone_code(
    codes: Sequence[str],
    call_api: ApiCaller,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Batch query numeric tone code(s) via the underlying API.

    Parameters
    ----------
    codes
        Sequence of numeric-tone-code strings (e.g. ["0","43","14"]). Each entry
        must be purely numeric (checked by regex).
    call_api
        Async callable that accepts a list[str] (batch queries) and returns the
        upstream response (usually list[list[str]]).
    batch_size
        Maximum number of queries to include per upstream request.

    Returns
    -------
    dict with keys:
      - results: list[list[str]] (same length as `codes`)
      - tool_calls_made: list[str] descriptions of the issued batch calls
    """
    if not isinstance(codes, (list, tuple)):
        raise TypeError("codes must be a list or tuple of strings")

    # Normalize and validate
    normalized = [str(c).strip() for c in codes]
    for token in normalized:
        if not token:
            raise ValueError("empty tone code provided")
        if not _NUMERIC_RE.match(token):
            raise ValueError(f"invalid numeric tone code: {token!r}")

    tool_calls: List[str] = []
    results = await _call_in_chunks(normalized, call_api, batch_size, "find_words_by_tone_code", tool_calls)

    return {"results": results, "tool_calls_made": tool_calls}


async def batch_find_tone_continuation(
    prefixes: Sequence[str] | str,
    tone_digits: Sequence[str] | str,
    call_api: ApiCaller,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Batch perform mixed prefix+tone continuation queries.

    Supports the following calling conventions:
    - Pairwise: prefixes and tone_digits are lists of equal length -> queries are
      zip(prefixes[i] + tone_digits[i]).
    - Broadcast (single tone_digits): prefixes is a list, tone_digits is a single
      string -> the same tone_digits are applied to every prefix.
    - Single-single: both scalars -> a single query sent.

    Parameters
    ----------
    prefixes
        Either a single prefix string or a sequence of prefix strings.
    tone_digits
        Either a single tone_digits string (e.g. "43") or a sequence of strings.
    call_api
        Async callable that accepts a list[str] and returns upstream responses.
    batch_size
        Maximum number of queries to issue per batch.

    Returns
    -------
    dict:
      - results: list[list[str]] aligned with the prefixes order
      - tool_calls_made: list[str] descriptions of the issued batch calls
    """
    # Normalize inputs to lists
    if isinstance(prefixes, str):
        prefixes_list = [prefixes.strip()]
    else:
        prefixes_list = [str(p).strip() for p in list(prefixes)]

    if isinstance(tone_digits, str):
        tones_list = [tone_digits.strip()]
    else:
        tones_list = [str(t).strip() for t in list(tone_digits)]

    if not prefixes_list:
        raise ValueError("prefixes must contain at least one element")
    if not tones_list:
        raise ValueError("tone_digits must contain at least one element")

    # Determine pairing: pairwise, broadcast, or error
    if len(prefixes_list) == len(tones_list):
        pairs = list(zip(prefixes_list, tones_list))
    elif len(tones_list) == 1:
        pairs = [(p, tones_list[0]) for p in prefixes_list]
    elif len(prefixes_list) == 1:
        pairs = [(prefixes_list[0], t) for t in tones_list]
    else:
        raise ValueError(
            "When both prefixes and tone_digits are sequences, their lengths must match "
            "or one must be length 1 (broadcast)."
        )

    # Validate tone strings
    queries: List[str] = []
    for p, t in pairs:
        if not t:
            raise ValueError("empty tone_digits value")
        if not _NUMERIC_RE.match(t):
            raise ValueError(f"invalid numeric tone_digits: {t!r}")
        # If prefix is empty, we still allow the query (it becomes purely digits)
        queries.append(f"{p}{t}")

    tool_calls: List[str] = []
    results = await _call_in_chunks(queries, call_api, batch_size, "find_tone_continuation", tool_calls)

    return {"results": results, "tool_calls_made": tool_calls}


# Convenience wrapper: collect queries from strong-beat/rhyme position maps -----


async def prepare_and_batch_strong_and_rhyme_queries(
    strong_beats: Sequence[Tuple[int, str]],
    rhyme_positions: Sequence[Tuple[int, str]],
    call_api: ApiCaller,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """
    Given lists of (position_index, tone_code) pairs for strong beats and rhyme
    positions, issue batched `find_words_by_tone_code` calls as efficiently as
    possible and return a mapping from the original positions to candidate lists.

    Parameters
    ----------
    strong_beats
        Sequence of (index, tone_code) for strong-beat positions.
    rhyme_positions
        Sequence of (index, tone_code) for rhyme positions.
    call_api
        Async callable that accepts list[str] (batch) and returns batch results.
    batch_size
        Batch size for upstream calls.

    Returns
    -------
    dict with keys:
      - by_position: dict[int, list[str]] mapping original position index -> candidates
      - tool_calls_made: list[str]
      - results_list: list[list[str]] (flat list of all returned candidate-lists in the order queried)
    """
    # Collect unique tone codes to avoid duplicate queries
    codes_ordered: List[str] = []
    pos_to_code_index: Dict[int, int] = {}

    # Helper to add code and record mapping
    def _add(code: str, pos_idx: int):
        code = str(code).strip()
        if not code:
            return
        if code in codes_ordered:
            idx = codes_ordered.index(code)
        else:
            idx = len(codes_ordered)
            codes_ordered.append(code)
        pos_to_code_index[pos_idx] = idx

    for pos_idx, code in list(strong_beats) + list(rhyme_positions):
        if not _NUMERIC_RE.match(str(code).strip()):
            raise ValueError(f"invalid numeric tone code in inputs: {code!r}")
        _add(code, pos_idx)

    # If no codes to query, return empty structure
    if not codes_ordered:
        return {"by_position": {}, "tool_calls_made": [], "results_list": []}

    # Perform batched calls
    batch_result = await batch_find_words_by_tone_code(codes_ordered, call_api, batch_size=batch_size)
    results_list: List[List[str]] = batch_result["results"]
    tool_calls: List[str] = batch_result["tool_calls_made"]

    # Map back to positions
    by_position: Dict[int, List[str]] = {}
    for pos_idx, code_index in pos_to_code_index.items():
        # defensive bounds check
        if code_index < len(results_list):
            by_position[pos_idx] = results_list[code_index]
        else:
            by_position[pos_idx] = []

    return {"by_position": by_position, "tool_calls_made": tool_calls, "results_list": results_list}
