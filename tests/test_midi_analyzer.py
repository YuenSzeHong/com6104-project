"""
Test midi-analyzer MCP server via stdio transport.

Uses the official MCP Python SDK (ClientSession) for clean, high-level testing.
"""

from __future__ import annotations

import json

import pytest


@pytest.mark.asyncio
async def test_mcp_server_initialize(mcp_session_midi_analyzer):
    """Test that the MCP server can be initialized via stdio."""
    async with mcp_session_midi_analyzer() as session:
        server_caps = session.get_server_capabilities()
        assert server_caps is not None


@pytest.mark.asyncio
async def test_mcp_server_list_tools(mcp_session_midi_analyzer):
    """Test that the MCP server lists all expected tools."""
    async with mcp_session_midi_analyzer() as session:
        tools_result = await session.list_tools()
        tool_names = [t.name for t in tools_result.tools]

        expected_tools = [
            "analyze_midi",
            "extract_embedded_lyrics",
            "get_syllable_durations",
            "suggest_rhyme_positions",
        ]
        for name in expected_tools:
            assert name in tool_names, f"Tool '{name}' not found"


@pytest.mark.asyncio
async def test_analyze_midi_returns_expected_metadata(
    mcp_session_midi_analyzer, doraemon_midi
):
    """Test analyze_midi tool returns expected metadata."""
    async with mcp_session_midi_analyzer() as session:
        result = await session.call_tool(
            "analyze_midi",
            {"file_path": str(doraemon_midi)},
        )
        payload = json.loads(result.content[0].text)

        assert "error" not in payload
        assert payload["syllable_count"] == 98
        assert payload["effective_syllable_count"] == 98
        assert payload["effective_syllable_count_source"] == "embedded_lyrics"
        assert payload["embedded_lyrics_source"] == "lyrics_meta"
        assert payload["embedded_lyric_unit_count"] == 98
        assert payload["melody_channel"] == 0
        assert payload["track_count"] == 4
        assert payload["bpm"] > 0
        assert payload["strong_beat_positions"]


@pytest.mark.asyncio
async def test_extract_embedded_lyrics_returns_units(
    mcp_session_midi_analyzer, doraemon_midi
):
    """Test extract_embedded_lyrics tool returns units."""
    async with mcp_session_midi_analyzer() as session:
        result = await session.call_tool(
            "extract_embedded_lyrics",
            {"file_path": str(doraemon_midi)},
        )
        payload = json.loads(result.content[0].text)

        assert "error" not in payload
        assert payload["source"] == "lyrics_meta"
        assert payload["unit_count"] == 98
        assert payload["units"][:4] == ["こん", "な", "こ", "と"]
