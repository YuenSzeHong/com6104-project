from __future__ import annotations

import sys
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Callable

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def doraemon_midi(repo_root: Path) -> Path:
    return repo_root / "test" / "midi" / "ドラえもんのうた.mid"


def _session_factory(
    server_script: Path,
) -> Callable[[], AbstractAsyncContextManager[ClientSession]]:
    """Return an async context manager that creates an MCP client session."""

    @asynccontextmanager
    async def _make_session() -> AsyncIterator[ClientSession]:
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(server_script)],
        )
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session

    return _make_session


@pytest.fixture
def mcp_session_jyutping():
    """Fixture: factory for MCP sessions connected to jyutping server.

    Usage:
        async with mcp_session_jyutping() as session:
            result = await session.call_tool(...)
    """
    server_script = REPO_ROOT / "mcp-servers" / "jyutping" / "server.py"
    return _session_factory(server_script)


@pytest.fixture
def mcp_session_melody_mapper():
    """Fixture: factory for MCP sessions connected to melody-mapper server."""
    server_script = REPO_ROOT / "mcp-servers" / "melody-mapper" / "server.py"
    return _session_factory(server_script)


@pytest.fixture
def mcp_session_midi_analyzer():
    """Fixture: factory for MCP sessions connected to midi-analyzer server."""
    server_script = REPO_ROOT / "mcp-servers" / "midi-analyzer" / "server.py"
    return _session_factory(server_script)
