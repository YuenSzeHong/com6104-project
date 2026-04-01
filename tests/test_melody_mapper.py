"""
Test melody-mapper MCP server via stdio transport.

Uses the official MCP Python SDK (ClientSession) for clean, high-level testing.
"""

from __future__ import annotations

import json

import pytest


@pytest.mark.asyncio
async def test_mcp_server_initialize(mcp_session_melody_mapper):
    """Test that the MCP server can be initialized via stdio."""
    async with mcp_session_melody_mapper() as session:
        server_caps = session.get_server_capabilities()
        assert server_caps is not None


@pytest.mark.asyncio
async def test_mcp_server_list_tools(mcp_session_melody_mapper):
    """Test that the MCP server lists all expected tools."""
    async with mcp_session_melody_mapper() as session:
        tools_result = await session.list_tools()
        tool_names = [t.name for t in tools_result.tools]

        expected_tools = [
            "analyze_melody_contour",
            "get_tone_requirements",
            "find_words_by_melody",
            "suggest_tone_sequence",
            "find_phrase_words",
        ]
        for name in expected_tools:
            assert name in tool_names, f"Tool '{name}' not found"


@pytest.mark.asyncio
async def test_analyze_melody_contour_returns_lean_0243_sequence(
    mcp_session_melody_mapper, doraemon_midi
):
    """Test analyze_melody_contour returns expected 0243 sequence."""
    async with mcp_session_melody_mapper() as session:
        result = await session.call_tool(
            "analyze_melody_contour",
            {"file_path": str(doraemon_midi)},
        )
        payload = json.loads(result.content[0].text)

        assert "error" not in payload
        assert payload["melody_channel"] == 0
        assert payload["syllable_count"] > 0
        assert len(payload["tone_sequence"]) == payload["syllable_count"]
        assert set(payload["tone_sequence"]).issubset({0, 2, 3, 4})
        assert payload["strong_beats"]
        assert payload["phrase_ends"]


@pytest.mark.asyncio
async def test_suggest_tone_sequence_matches_contour_result(
    mcp_session_melody_mapper, doraemon_midi
):
    """Test suggest_tone_sequence matches contour result."""
    async with mcp_session_melody_mapper() as session:
        # First get contour
        result = await session.call_tool(
            "analyze_melody_contour",
            {"file_path": str(doraemon_midi)},
        )
        contour = json.loads(result.content[0].text)

        # Then get tone sequence summary
        result = await session.call_tool(
            "suggest_tone_sequence",
            {"file_path": str(doraemon_midi)},
        )
        summary = json.loads(result.content[0].text)

        assert "error" not in summary
        assert summary["syllable_count"] == contour["syllable_count"]
        assert summary["tone_sequence_str"].split()[:8] == [
            str(x) for x in contour["tone_sequence"][:8]
        ]
