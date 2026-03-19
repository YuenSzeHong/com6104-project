"""
Short-term memory module for the Cantonese Lyrics Agent.

Implements a sliding-window conversation buffer that is shared across all
agents in the pipeline.  Each agent reads from and writes to the same
ShortTermMemory instance so that context (MIDI analysis results, Jyutping
mappings, draft lyrics, validator feedback …) flows naturally between steps.

Design
------
- Conversation turns are stored as LangChain BaseMessage objects so they can
  be passed directly to any ChatModel / chain.
- A configurable ``max_turns`` limit keeps memory bounded.  When the window
  is full the oldest *non-system* message pair is evicted.
- A structured ``context`` dict (separate from the message list) holds
  typed pipeline artefacts (e.g. MidiAnalysis, tone patterns) that agents
  pass to each other without polluting the chat history.
- The class is intentionally NOT async so it can be used freely in both sync
  and async code paths.
"""

from __future__ import annotations

import copy
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """A single conversation turn (one or more messages at the same step)."""

    index: int                          # monotonically increasing turn number
    timestamp: float = field(default_factory=time.time)
    messages: list[BaseMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "messages": [_message_to_dict(m) for m in self.messages],
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _message_to_dict(msg: BaseMessage) -> dict[str, Any]:
    """Serialise a LangChain message to a plain dict."""
    return {
        "type": msg.__class__.__name__,
        "content": msg.content,
        "additional_kwargs": msg.additional_kwargs,
    }


def _message_from_dict(d: dict[str, Any]) -> BaseMessage:
    """Deserialise a plain dict back to a LangChain message."""
    type_map: dict[str, type[BaseMessage]] = {
        "HumanMessage": HumanMessage,
        "AIMessage": AIMessage,
        "SystemMessage": SystemMessage,
        "ToolMessage": ToolMessage,
    }
    cls = type_map.get(d["type"], HumanMessage)
    kwargs: dict[str, Any] = {"content": d["content"]}
    if d.get("additional_kwargs"):
        kwargs["additional_kwargs"] = d["additional_kwargs"]
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class ShortTermMemory:
    """
    Sliding-window short-term memory shared across all pipeline agents.

    Parameters
    ----------
    max_turns:
        Maximum number of *conversation turns* to keep.  When exceeded, the
        oldest non-system turn is silently dropped.
    system_prompt:
        Optional persistent system prompt.  It is always the first message
        returned by ``get_messages()`` and is never evicted.
    session_id:
        Human-readable label for this run (used in serialisation output).

    Quick-start
    -----------
    >>> mem = ShortTermMemory(max_turns=10, system_prompt="You are a Cantonese lyrics expert.")
    >>> mem.add_user_message("Analyse this MIDI file.")
    >>> mem.add_ai_message("I found 32 syllables at 120 BPM.")
    >>> mem.set_context("midi_analysis", {"syllable_count": 32, "bpm": 120})
    >>> for msg in mem.get_messages():
    ...     print(msg)
    """

    def __init__(
        self,
        max_turns: int = 20,
        system_prompt: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self._max_turns = max_turns
        self._session_id: str = session_id or f"session-{int(time.time())}"
        self._turn_counter: int = 0

        # The sliding window of conversation turns
        self._turns: deque[Turn] = deque()

        # Persistent system message (never evicted)
        self._system_message: SystemMessage | None = (
            SystemMessage(content=system_prompt) if system_prompt else None
        )

        # Structured runtime state
        self._pipeline_state: dict[str, Any] = {}
        self._attempt_state: dict[str, Any] = {}
        self._artifacts: dict[str, Any] = {}

    _PIPELINE_KEYS = {
        "midi_analysis",
        "melody_analysis",
        "jyutping_map",
        "draft_lyrics",
        "validation_result",
        "best_result",
        "final_result",
        "run_status",
    }

    _ATTEMPT_KEYS = {
        "attempt_index",
        "last_score",
        "last_feedback",
        "revision_instructions",
    }

    _ARTIFACT_KEYS = {
        "source_text",
        "embedded_lyrics",
        "raw_tool_outputs",
    }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def max_turns(self) -> int:
        return self._max_turns

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def system_prompt(self) -> str | None:
        return self._system_message.content if self._system_message else None  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # System prompt management
    # ------------------------------------------------------------------

    def set_system_prompt(self, prompt: str) -> None:
        """Replace (or set) the persistent system prompt."""
        self._system_message = SystemMessage(content=prompt)

    def load_system_prompt_from_file(self, path: str | Path) -> None:
        """Load system prompt text from a Markdown / text file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"System prompt file not found: {p}")
        self.set_system_prompt(p.read_text(encoding="utf-8").strip())

    # ------------------------------------------------------------------
    # Adding messages
    # ------------------------------------------------------------------

    def add_user_message(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> Turn:
        """Append a human/user message and return the new Turn."""
        return self._append_turn(
            [HumanMessage(content=content)], metadata=metadata or {}
        )

    def add_ai_message(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> Turn:
        """Append an AI/assistant message and return the new Turn."""
        return self._append_turn(
            [AIMessage(content=content)], metadata=metadata or {}
        )

    def add_tool_result(
        self,
        tool_name: str,
        result: Any,
        tool_call_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Turn:
        """
        Append a tool-result message.

        The result is JSON-serialised if it is not already a string so it
        renders cleanly in the conversation history.
        """
        content = (
            result
            if isinstance(result, str)
            else json.dumps(result, ensure_ascii=False, indent=2)
        )
        msg = ToolMessage(
            content=content,
            tool_call_id=tool_call_id or f"{tool_name}-{int(time.time())}",
        )
        return self._append_turn([msg], metadata={"tool_name": tool_name, **(metadata or {})})

    def add_messages(
        self,
        messages: list[BaseMessage],
        metadata: dict[str, Any] | None = None,
    ) -> Turn:
        """Append an arbitrary list of LangChain messages as a single turn."""
        return self._append_turn(messages, metadata=metadata or {})

    # ------------------------------------------------------------------
    # Retrieving messages
    # ------------------------------------------------------------------

    def get_messages(self, include_system: bool = True) -> list[BaseMessage]:
        """
        Return the full message list ready to pass to a ChatModel.

        Layout
        ------
        [SystemMessage?] + [all messages from sliding window turns]
        """
        result: list[BaseMessage] = []
        if include_system and self._system_message:
            result.append(self._system_message)
        for turn in self._turns:
            result.extend(turn.messages)
        return result

    def get_last_n_messages(
        self, n: int, include_system: bool = True
    ) -> list[BaseMessage]:
        """Return only the last *n* individual messages (plus optional system)."""
        all_msgs = self.get_messages(include_system=False)
        recent = all_msgs[-n:] if n < len(all_msgs) else all_msgs
        if include_system and self._system_message:
            return [self._system_message] + recent
        return recent

    def get_last_human_message(self) -> HumanMessage | None:
        """Return the most recent HumanMessage, or None."""
        for turn in reversed(self._turns):
            for msg in reversed(turn.messages):
                if isinstance(msg, HumanMessage):
                    return msg
        return None

    def get_last_ai_message(self) -> AIMessage | None:
        """Return the most recent AIMessage, or None."""
        for turn in reversed(self._turns):
            for msg in reversed(turn.messages):
                if isinstance(msg, AIMessage):
                    return msg
        return None

    def iter_turns(self) -> Iterator[Turn]:
        """Iterate over all turns in chronological order."""
        yield from self._turns

    # ------------------------------------------------------------------
    # Structured runtime state
    # ------------------------------------------------------------------

    def set_pipeline_value(self, key: str, value: Any) -> None:
        self._pipeline_state[key] = value

    def get_pipeline_value(self, key: str, default: Any = None) -> Any:
        return self._pipeline_state.get(key, default)

    def update_pipeline_state(self, data: dict[str, Any]) -> None:
        self._pipeline_state.update(data)

    def set_attempt_value(self, key: str, value: Any) -> None:
        self._attempt_state[key] = value

    def get_attempt_value(self, key: str, default: Any = None) -> Any:
        return self._attempt_state.get(key, default)

    def update_attempt_state(self, data: dict[str, Any]) -> None:
        self._attempt_state.update(data)

    def set_artifact(self, key: str, value: Any) -> None:
        self._artifacts[key] = value

    def get_artifact(self, key: str, default: Any = None) -> Any:
        return self._artifacts.get(key, default)

    def update_artifacts(self, data: dict[str, Any]) -> None:
        self._artifacts.update(data)

    def set_current_draft(self, draft: dict[str, Any]) -> None:
        self._pipeline_state["draft_lyrics"] = draft

    def get_current_draft(self) -> dict[str, Any]:
        draft = self._pipeline_state.get("draft_lyrics")
        return draft if isinstance(draft, dict) else {}

    def set_validation_result(self, result: dict[str, Any]) -> None:
        self._pipeline_state["validation_result"] = result

    def get_validation_result(self) -> dict[str, Any]:
        result = self._pipeline_state.get("validation_result")
        return result if isinstance(result, dict) else {}

    def set_best_result(
        self,
        *,
        draft: dict[str, Any] | None,
        validation: dict[str, Any] | None,
        score: float,
    ) -> None:
        self._pipeline_state["best_result"] = {
            "draft": draft or {},
            "validation": validation or {},
            "score": float(score),
        }

    def get_best_result(self) -> dict[str, Any]:
        result = self._pipeline_state.get("best_result")
        return result if isinstance(result, dict) else {}

    def set_final_result(self, result: dict[str, Any]) -> None:
        self._pipeline_state["final_result"] = result

    def get_final_result(self) -> dict[str, Any]:
        result = self._pipeline_state.get("final_result")
        return result if isinstance(result, dict) else {}

    def set_run_status(self, **status: Any) -> None:
        current = self._pipeline_state.get("run_status", {})
        if not isinstance(current, dict):
            current = {}
        current.update(status)
        self._pipeline_state["run_status"] = current

    def get_run_status(self) -> dict[str, Any]:
        status = self._pipeline_state.get("run_status")
        return status if isinstance(status, dict) else {}

    def set_context(self, key: str, value: Any) -> None:
        """Compatibility wrapper around the structured runtime stores."""
        if key in self._PIPELINE_KEYS:
            self._pipeline_state[key] = value
        elif key in self._ATTEMPT_KEYS:
            self._attempt_state[key] = value
        elif key in self._ARTIFACT_KEYS:
            self._artifacts[key] = value
        else:
            self._pipeline_state[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Compatibility wrapper around the structured runtime stores."""
        if key in self._pipeline_state:
            return self._pipeline_state.get(key, default)
        if key in self._attempt_state:
            return self._attempt_state.get(key, default)
        if key in self._artifacts:
            return self._artifacts.get(key, default)
        return default

    def update_context(self, data: dict[str, Any]) -> None:
        """Compatibility wrapper around the structured runtime stores."""
        for key, value in data.items():
            self.set_context(key, value)

    def has_context(self, key: str) -> bool:
        return (
            key in self._pipeline_state
            or key in self._attempt_state
            or key in self._artifacts
        )

    def clear_context(self, key: str | None = None) -> None:
        """Remove one key, or wipe all runtime state if key is None."""
        if key is None:
            self._pipeline_state.clear()
            self._attempt_state.clear()
            self._artifacts.clear()
        else:
            self._pipeline_state.pop(key, None)
            self._attempt_state.pop(key, None)
            self._artifacts.pop(key, None)

    @property
    def context(self) -> dict[str, Any]:
        """Read-only merged snapshot of all runtime state."""
        merged: dict[str, Any] = {}
        merged.update(self._pipeline_state)
        merged.update(self._attempt_state)
        merged.update(self._artifacts)
        return copy.deepcopy(merged)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear(self, keep_system: bool = True) -> None:
        """
        Wipe the conversation history and context.

        Parameters
        ----------
        keep_system:
            When True the system prompt is preserved; only the turn
            window and context dict are cleared.
        """
        self._turns.clear()
        self._pipeline_state.clear()
        self._attempt_state.clear()
        self._artifacts.clear()
        self._turn_counter = 0
        if not keep_system:
            self._system_message = None

    def trim_to(self, n_turns: int) -> None:
        """Keep only the most recent *n_turns* turns."""
        while len(self._turns) > n_turns:
            self._turns.popleft()

    # ------------------------------------------------------------------
    # Serialisation (for debugging / persistence)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full memory state to a plain dict."""
        return {
            "session_id": self._session_id,
            "max_turns": self._max_turns,
            "system_prompt": self._system_message.content if self._system_message else None,
            "turns": [t.to_dict() for t in self._turns],
            "pipeline_state": self._pipeline_state,
            "attempt_state": self._attempt_state,
            "artifacts": self._artifacts,
            "context": self.context,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShortTermMemory":
        """Restore a ShortTermMemory instance from a serialised dict."""
        mem = cls(
            max_turns=data.get("max_turns", 20),
            system_prompt=data.get("system_prompt"),
            session_id=data.get("session_id"),
        )
        for raw_turn in data.get("turns", []):
            turn = Turn(
                index=raw_turn["index"],
                timestamp=raw_turn.get("timestamp", 0.0),
                messages=[_message_from_dict(m) for m in raw_turn.get("messages", [])],
                metadata=raw_turn.get("metadata", {}),
            )
            mem._turns.append(turn)
            mem._turn_counter = max(mem._turn_counter, turn.index + 1)
        mem._pipeline_state = data.get("pipeline_state", {})
        mem._attempt_state = data.get("attempt_state", {})
        mem._artifacts = data.get("artifacts", {})
        # Backward compatibility for older serialised snapshots
        if not mem._pipeline_state and not mem._attempt_state and not mem._artifacts:
            legacy = data.get("context", {})
            if isinstance(legacy, dict):
                mem._pipeline_state = legacy
        return mem

    @classmethod
    def from_json(cls, json_str: str) -> "ShortTermMemory":
        return cls.from_dict(json.loads(json_str))

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of individual messages (excluding system)."""
        return sum(len(t.messages) for t in self._turns)

    def __repr__(self) -> str:
        return (
            f"ShortTermMemory("
            f"session_id={self._session_id!r}, "
            f"turns={len(self._turns)}/{self._max_turns}, "
            f"messages={len(self)}, "
            f"pipeline_keys={list(self._pipeline_state.keys())!r}, "
            f"attempt_keys={list(self._attempt_state.keys())!r}, "
            f"artifact_keys={list(self._artifacts.keys())!r}"
            f")"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_turn(
        self,
        messages: list[BaseMessage],
        metadata: dict[str, Any],
    ) -> Turn:
        """
        Create a new Turn, append it to the window, and evict the oldest
        turn if the window is now over capacity.
        """
        turn = Turn(
            index=self._turn_counter,
            messages=messages,
            metadata=metadata,
        )
        self._turn_counter += 1
        self._turns.append(turn)

        # Evict oldest turn if window exceeded
        while len(self._turns) > self._max_turns:
            self._turns.popleft()

        return turn
