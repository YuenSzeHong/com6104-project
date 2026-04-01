"""
Test word-selector MCP server via stdio transport.

Uses the official MCP Python SDK (ClientSession) for clean, high-level testing.
"""

from __future__ import annotations

import json
import os

import pytest


@pytest.mark.skipif(
    os.getenv("SKIP_NETWORK_TESTS", "").lower() == "true",
    reason="Set SKIP_NETWORK_TESTS=true to skip tests that require internet access.",
)
@pytest.mark.asyncio
async def test_find_words_returns_candidates(mcp_session_jyutping):
    """Test that find_words_by_tone_code returns candidate words."""
    async with mcp_session_jyutping() as session:
        result = await session.call_tool(
            "find_words_by_tone_code",
            {"code": "43"},
        )
        candidates = json.loads(result.content[0].text)

        assert isinstance(candidates, list)
        assert len(candidates) > 0
        # All results should be Chinese words
        for word in candidates[:5]:  # Check first 5
            assert any("\u4e00" <= c <= "\u9fff" for c in word)


@pytest.mark.skipif(
    os.getenv("SKIP_NETWORK_TESTS", "").lower() == "true",
    reason="Set SKIP_NETWORK_TESTS=true to skip tests that require internet access.",
)
@pytest.mark.asyncio
async def test_find_tone_continuation_returns_contextual_words(mcp_session_jyutping):
    """Test that find_tone_continuation returns contextually appropriate words."""
    async with mcp_session_jyutping() as session:
        result = await session.call_tool(
            "find_tone_continuation",
            {
                "chinese_prefix": "我要",
                "tone_digits": "43",
            },
        )
        candidates = json.loads(result.content[0].text)

        assert isinstance(candidates, list)
        assert len(candidates) > 0
        # All results should be Chinese words
        for word in candidates[:5]:
            assert any("\u4e00" <= c <= "\u9fff" for c in word)
