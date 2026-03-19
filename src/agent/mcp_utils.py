"""
Helpers for normalizing MCP / LangChain tool payloads into plain Python values.
"""

from __future__ import annotations

import json
from typing import Any


def unwrap_mcp_payload(raw: Any) -> Any:
    """
    Unwrap MCP tool results into plain Python values.

    Common wrappers seen in this repo:
    - {"type": "text", "text": "...", "id": "..."}
    - [{"type": "text", "text": "...", "id": "..."}]
    - ToolMessage / AIMessage-like objects with a ``content`` attribute
    """
    if raw is None:
        return None

    content = getattr(raw, "content", None)
    if content is not None and content is not raw:
        return unwrap_mcp_payload(content)

    if isinstance(raw, dict) and "type" in raw and "text" in raw:
        if raw.get("type") == "text":
            return raw.get("text", "")

    if isinstance(raw, list):
        if not raw:
            return []

        if all(isinstance(item, dict) and "type" in item and "text" in item for item in raw):
            text_blocks = [
                str(item.get("text", ""))
                for item in raw
                if item.get("type") == "text"
            ]
            if len(text_blocks) == 1:
                return text_blocks[0]
            return "\n".join(block for block in text_blocks if block)

        return [unwrap_mcp_payload(item) for item in raw]

    return raw


def normalize_mcp_result(raw: Any, *, parse_json: bool = True) -> Any:
    """
    Normalize MCP tool results and optionally parse JSON-like text payloads.
    """
    raw = unwrap_mcp_payload(raw)

    if not parse_json:
        return raw

    if isinstance(raw, (dict, list)):
        return raw

    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw

    return raw
