"""
Test jyutping MCP server via stdio transport.

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
async def test_mcp_server_initialize(mcp_session_jyutping):
    """Test that the MCP server can be initialized via stdio."""
    async with mcp_session_jyutping() as session:
        server_caps = session.get_server_capabilities()
        assert server_caps is not None


@pytest.mark.skipif(
    os.getenv("SKIP_NETWORK_TESTS", "").lower() == "true",
    reason="Set SKIP_NETWORK_TESTS=true to skip tests that require internet access.",
)
@pytest.mark.asyncio
async def test_mcp_server_list_tools(mcp_session_jyutping):
    """Test that the MCP server lists all expected tools."""
    async with mcp_session_jyutping() as session:
        tools_result = await session.list_tools()
        tool_names = [t.name for t in tools_result.tools]

        expected_tools = [
            "query_raw",
            "chinese_to_jyutping",
            "get_tone_code",
            "get_tone_pattern",
            "find_words_by_tone_code",
            "find_tone_continuation",
        ]
        for name in expected_tools:
            assert name in tool_names, f"Tool '{name}' not found"


@pytest.mark.skipif(
    os.getenv("SKIP_NETWORK_TESTS", "").lower() == "true",
    reason="Set SKIP_NETWORK_TESTS=true to skip tests that require internet access.",
)
@pytest.mark.asyncio
async def test_find_words_by_tone_code_accepts_int_str_and_batch(mcp_session_jyutping):
    """Test find_words_by_tone_code tool with int, str, and batch inputs."""
    async with mcp_session_jyutping() as session:
        # single int
        result = await session.call_tool("find_words_by_tone_code", {"code": 0})
        result_from_int = json.loads(result.content[0].text)
        assert isinstance(result_from_int, list)
        assert len(result_from_int) > 0

        # single str
        result = await session.call_tool("find_words_by_tone_code", {"code": "43"})
        result_from_str = json.loads(result.content[0].text)
        assert isinstance(result_from_str, list)
        assert len(result_from_str) > 0

        # batch
        result = await session.call_tool(
            "find_words_by_tone_code", {"code": ["0", "43"]}
        )
        result_from_batch = json.loads(result.content[0].text)
        assert isinstance(result_from_batch, list)
        assert len(result_from_batch) == 2
        assert all(
            isinstance(item, list) and len(item) > 0 for item in result_from_batch
        )


@pytest.mark.skipif(
    os.getenv("SKIP_NETWORK_TESTS", "").lower() == "true",
    reason="Set SKIP_NETWORK_TESTS=true to skip tests that require internet access.",
)
@pytest.mark.asyncio
async def test_find_tone_continuation_accepts_int_digits_and_batch(
    mcp_session_jyutping,
):
    """Test find_tone_continuation tool with single and batch inputs."""
    async with mcp_session_jyutping() as session:
        # single
        result = await session.call_tool(
            "find_tone_continuation",
            {"chinese_prefix": "我要", "tone_digits": 43},
        )
        payload = json.loads(result.content[0].text)
        assert isinstance(payload, list)
        assert len(payload) > 0

        # batch
        result = await session.call_tool(
            "find_tone_continuation",
            {"chinese_prefix": ["我要", "青山"], "tone_digits": [43, 1]},
        )
        batch_result = json.loads(result.content[0].text)
        assert isinstance(batch_result, list)
        assert len(batch_result) == 2
        assert all(isinstance(item, list) and len(item) > 0 for item in batch_result)


@pytest.mark.skipif(
    os.getenv("SKIP_NETWORK_TESTS", "").lower() == "true",
    reason="Set SKIP_NETWORK_TESTS=true to skip tests that require internet access.",
)
@pytest.mark.asyncio
async def test_find_tone_continuation_pairwise_and_broadcast_batching(
    mcp_session_jyutping,
):
    """Test find_tone_continuation pairwise and broadcast batching modes."""
    async with mcp_session_jyutping() as session:
        # Pairwise: prefixes and tones lists of same length
        result = await session.call_tool(
            "find_tone_continuation",
            {"chinese_prefix": ["我要", "青山"], "tone_digits": [43, 1]},
        )
        pairwise_result = json.loads(result.content[0].text)
        assert isinstance(pairwise_result, list)
        assert len(pairwise_result) == 2
        assert all(isinstance(item, list) and len(item) > 0 for item in pairwise_result)

        # Broadcast: multiple prefixes with a single tone_digits
        result = await session.call_tool(
            "find_tone_continuation",
            {"chinese_prefix": ["青山", "縱然"], "tone_digits": "43"},
        )
        broadcast_result = json.loads(result.content[0].text)
        assert isinstance(broadcast_result, list)
        assert len(broadcast_result) == 2
        assert all(
            isinstance(item, list) and len(item) > 0 for item in broadcast_result
        )


@pytest.mark.skipif(
    os.getenv("SKIP_NETWORK_TESTS", "").lower() == "true",
    reason="Set SKIP_NETWORK_TESTS=true to skip tests that require internet access.",
)
@pytest.mark.asyncio
async def test_query_raw_live_smoke(mcp_session_jyutping):
    """Test query_raw tool with live API call."""
    async with mcp_session_jyutping() as session:
        result = await session.call_tool("query_raw", {"nums": "43"})
        payload = json.loads(result.content[0].text)

        assert isinstance(payload, list)
        assert payload
