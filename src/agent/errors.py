from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentRuntimeError(Exception):
    """Base runtime error carrying structured context for diagnostics."""

    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if not self.context:
            return self.message
        return f"{self.message} | context={self.context}"


@dataclass
class ToolInvokeError(AgentRuntimeError):
    """Raised when an MCP tool invocation fails or is unavailable."""


@dataclass
class ParseError(AgentRuntimeError):
    """Raised when structured parsing fails on strict parse paths."""


@dataclass
class ConstraintViolation(AgentRuntimeError):
    """Raised when required hard constraints are violated."""
