"""
Agent and MCP Server registries.

These registries act as the central catalogue for the multi-agent framework:
  - AgentRegistry   : tracks every BaseAgent subclass added to the workflow
  - MCPServerRegistry: tracks every MCP server config and its live client

Usage
-----
Both registries are singleton instances (AGENT_REGISTRY, MCP_REGISTRY) that
are imported wherever needed.  The orchestrator calls them to wire everything
together at startup.

Adding a new agent
------------------
1. Subclass BaseAgent and decorate the class:

       @AGENT_REGISTRY.register("my-agent")
       class MyAgent(BaseAgent): ...

   OR call it manually after class definition:

       AGENT_REGISTRY.register_instance(MyAgent(config, llm, memory))

Adding a new MCP server
-----------------------
1. Add an MCPServerConfig entry to config.AGENTS in config.py – that's it.
   MCPServerRegistry.build_from_config() is called by the orchestrator at
   startup and populates the registry automatically.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Type

if TYPE_CHECKING:
    from .base_agent import BaseAgent
    from .config import AgentConfig, MCPServerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent Registry
# ---------------------------------------------------------------------------

class AgentRegistry:
    """
    Catalogue of agent *classes* and live *instances*.

    Class catalogue
    ~~~~~~~~~~~~~~~
    Agent classes are registered (usually via the @register decorator) so the
    orchestrator can instantiate them on demand with the correct config / LLM /
    memory objects.

    Instance catalogue
    ~~~~~~~~~~~~~~~~~~
    Once the orchestrator instantiates an agent it calls ``add_instance`` so
    other components can look up a running agent by name without going through
    the orchestrator.
    """

    def __init__(self) -> None:
        # name → agent class
        self._classes: dict[str, Type["BaseAgent"]] = {}
        # name → live instance
        self._instances: dict[str, "BaseAgent"] = {}

    # ------------------------------------------------------------------
    # Class registration
    # ------------------------------------------------------------------

    def register(self, name: str) -> Callable[[Type["BaseAgent"]], Type["BaseAgent"]]:
        """
        Class decorator that registers an agent class under *name*.

        Example
        -------
        @AGENT_REGISTRY.register("lyrics-composer")
        class LyricsComposerAgent(BaseAgent):
            ...
        """
        def decorator(cls: Type["BaseAgent"]) -> Type["BaseAgent"]:
            self._register_class(name, cls)
            return cls
        return decorator

    def _register_class(self, name: str, cls: Type["BaseAgent"]) -> None:
        if name in self._classes:
            logger.warning(
                "AgentRegistry: overwriting existing class registration for '%s' "
                "(old=%s, new=%s)",
                name, self._classes[name].__name__, cls.__name__,
            )
        self._classes[name] = cls
        logger.debug("AgentRegistry: registered class '%s' → %s", name, cls.__name__)

    def register_class(self, name: str, cls: Type["BaseAgent"]) -> None:
        """Programmatic alternative to the @register decorator."""
        self._register_class(name, cls)

    # ------------------------------------------------------------------
    # Instance management
    # ------------------------------------------------------------------

    def add_instance(self, agent: "BaseAgent") -> None:
        """Store a live agent instance (called by the orchestrator)."""
        name = agent.name
        if name in self._instances:
            logger.warning(
                "AgentRegistry: replacing existing instance for '%s'", name
            )
        self._instances[name] = agent
        logger.debug("AgentRegistry: stored instance for '%s'", name)

    def get_instance(self, name: str) -> "BaseAgent":
        """Return the live instance for *name* or raise KeyError."""
        if name not in self._instances:
            raise KeyError(
                f"AgentRegistry: no live instance for '{name}'. "
                f"Available: {list(self._instances)}"
            )
        return self._instances[name]

    def get_class(self, name: str) -> Type["BaseAgent"]:
        """Return the registered class for *name* or raise KeyError."""
        if name not in self._classes:
            raise KeyError(
                f"AgentRegistry: no class registered for '{name}'. "
                f"Available: {list(self._classes)}"
            )
        return self._classes[name]

    def remove_instance(self, name: str) -> None:
        """Remove a live instance (e.g. during teardown)."""
        self._instances.pop(name, None)
        logger.debug("AgentRegistry: removed instance for '%s'", name)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def registered_classes(self) -> list[str]:
        """Sorted list of all registered class names."""
        return sorted(self._classes)

    @property
    def live_agents(self) -> list[str]:
        """Sorted list of all live instance names."""
        return sorted(self._instances)

    def has_class(self, name: str) -> bool:
        return name in self._classes

    def has_instance(self, name: str) -> bool:
        return name in self._instances

    def all_instances(self) -> list["BaseAgent"]:
        """Return all live instances in insertion order."""
        return list(self._instances.values())

    def clear_instances(self) -> None:
        """Remove all live instances (used between test runs)."""
        self._instances.clear()
        logger.debug("AgentRegistry: all instances cleared")

    def __repr__(self) -> str:
        return (
            f"AgentRegistry("
            f"classes={self.registered_classes}, "
            f"instances={self.live_agents})"
        )


# ---------------------------------------------------------------------------
# MCP Server Registry
# ---------------------------------------------------------------------------

class MCPServerRegistry:
    """
    Catalogue of MCP server configurations and their runtime state.

    Configs are loaded from ``config.MCP_SERVERS`` at startup.
    The orchestrator sets the live ``MultiServerMCPClient`` after it
    establishes the stdio connections.
    """

    def __init__(self) -> None:
        # name → MCPServerConfig
        self._configs: dict[str, "MCPServerConfig"] = {}
        # name → list of tool names exposed by this server (filled after connect)
        self._tool_names: dict[str, list[str]] = {}
        # Flat name → tool callable (filled after connect)
        self._tools: dict[str, Any] = {}
        # Whether the client has been connected
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Config management
    # ------------------------------------------------------------------

    def register_config(self, config: "MCPServerConfig") -> None:
        """Register one MCP server config."""
        if config.name in self._configs:
            logger.warning(
                "MCPServerRegistry: overwriting config for '%s'", config.name
            )
        self._configs[config.name] = config
        logger.debug(
            "MCPServerRegistry: registered config '%s' (enabled=%s)",
            config.name, config.enabled,
        )

    def build_from_config(self) -> None:
        """
        Populate the registry from the central ``config.MCP_SERVERS`` list.

        Called once at orchestrator startup before connecting.
        """
        from .config import MCP_SERVERS  # local import to avoid circulars

        for srv in MCP_SERVERS:
            self.register_config(srv)

        logger.info(
            "MCPServerRegistry: loaded %d server config(s): %s",
            len(self._configs),
            [s.name for s in MCP_SERVERS],
        )

    # ------------------------------------------------------------------
    # Tool bookkeeping (populated after the MCP client connects)
    # ------------------------------------------------------------------

    def set_server_tools(self, server_name: str, tool_names: list[str]) -> None:
        """Record which tool names are exposed by a connected server."""
        self._tool_names[server_name] = tool_names
        logger.debug(
            "MCPServerRegistry: server '%s' exposes tools: %s",
            server_name, tool_names,
        )

    def register_tool(self, name: str, tool: Any) -> None:
        """Store a resolved LangChain tool object under its fully-qualified name."""
        self._tools[name] = tool

    def get_tools_for_servers(self, server_names: list[str]) -> list[Any]:
        """
        Return LangChain tool objects that belong to any of the given servers.

        If *server_names* is empty, returns all registered tools.
        """
        if not server_names:
            return list(self._tools.values())

        allowed_tools: list[Any] = []
        for srv_name in server_names:
            for tool_name in self._tool_names.get(srv_name, []):
                if tool_name in self._tools:
                    allowed_tools.append(self._tools[tool_name])
                else:
                    logger.warning(
                        "MCPServerRegistry: tool '%s' listed for server '%s' "
                        "but not found in tool map",
                        tool_name, srv_name,
                    )
        return allowed_tools

    def get_all_tools(self) -> list[Any]:
        """Return every registered LangChain tool object."""
        return list(self._tools.values())

    def mark_connected(self) -> None:
        self._connected = True
        logger.info("MCPServerRegistry: all servers connected")

    def mark_disconnected(self) -> None:
        self._connected = False
        self._tools.clear()
        self._tool_names.clear()

    # ------------------------------------------------------------------
    # Config introspection
    # ------------------------------------------------------------------

    def get_config(self, name: str) -> "MCPServerConfig":
        if name not in self._configs:
            raise KeyError(
                f"MCPServerRegistry: no config for '{name}'. "
                f"Available: {list(self._configs)}"
            )
        return self._configs[name]

    @property
    def enabled_servers(self) -> list["MCPServerConfig"]:
        """All server configs where ``enabled=True``."""
        return [c for c in self._configs.values() if c.enabled]

    @property
    def all_configs(self) -> list["MCPServerConfig"]:
        return list(self._configs.values())

    @property
    def is_connected(self) -> bool:
        return self._connected

    def langchain_server_params(self) -> dict[str, Any]:
        """
        Build the ``servers`` dict expected by
        ``langchain_mcp_adapters.MultiServerMCPClient``.

        Only enabled servers are included.
        """
        return {
            srv.name: srv.to_langchain_params()
            for srv in self.enabled_servers
        }

    def has_server(self, name: str) -> bool:
        return name in self._configs

    def __repr__(self) -> str:
        return (
            f"MCPServerRegistry("
            f"servers={[s.name for s in self.all_configs]}, "
            f"connected={self._connected})"
        )


# ---------------------------------------------------------------------------
# Module-level singleton instances
# ---------------------------------------------------------------------------

#: Global agent registry – import this wherever agents are registered or looked up.
AGENT_REGISTRY = AgentRegistry()

#: Global MCP server registry – import this for tool access and server management.
MCP_REGISTRY = MCPServerRegistry()
