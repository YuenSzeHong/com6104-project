"""
BaseAgent – abstract base class for every agent in the pipeline.

All concrete agents (MidiAnalyserAgent, JyutpingMapperAgent, LyricsComposerAgent,
ValidatorAgent, …) inherit from this class.

Responsibilities
----------------
- Hold a reference to the shared ShortTermMemory and the LLM instance.
- Expose a standardised async ``run()`` entry-point that the orchestrator calls.
- Provide helpers for invoking LLM calls with the current memory context.
- Emit structured logging so every agent step is traceable.
- Optionally bind a filtered subset of MCP tools (those allowed for this agent).

Implementing a new agent
------------------------
1. Subclass BaseAgent.
2. Override ``_execute()`` with the agent's core logic.
3. (Optional) Override ``_build_prompt()`` to customise how the task is
   framed before being sent to the LLM.
4. Register the class with the AgentRegistry:

       from agent.registry import AGENT_REGISTRY

       @AGENT_REGISTRY.register("my-agent")
       class MyAgent(BaseAgent):
           async def _execute(self, task: str, **kwargs) -> AgentResult:
               ...
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from .memory import ShortTermMemory
from .config import AgentConfig, PROVIDER, WORKFLOW_CONFIG

# Lazy import for current LangChain agent API
create_agent: Callable[..., Any] | None
try:
    _agents_module = import_module("langchain.agents")
    create_agent = cast(
        Callable[..., Any] | None, getattr(_agents_module, "create_agent", None)
    )
except ImportError:
    create_agent = None

logger = logging.getLogger(__name__)


# Sentinel returned when structured output is intentionally skipped.
STRUCTURED_OUTPUT_SKIPPED = object()


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    """
    Standardised return value from every agent's ``run()`` call.

    Fields
    ------
    agent_name   : name of the agent that produced this result
    success      : True if the agent completed without errors
    output       : primary string output (e.g. generated lyrics, analysis JSON)
    data         : optional structured data to be stored in ShortTermMemory context
    error        : error message if success is False
    duration_s   : wall-clock seconds the agent took
    metadata     : any extra key/value pairs the agent wants to surface
    """

    agent_name: str
    success: bool
    output: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    duration_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # Convenience ----------------------------------------------------------------

    @property
    def failed(self) -> bool:
        return not self.success

    def raise_if_failed(self) -> None:
        """Raise RuntimeError if the agent did not succeed."""
        if not self.success:
            raise RuntimeError(
                f"Agent '{self.agent_name}' failed: {self.error}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "success": self.success,
            "output": self.output,
            "data": self.data,
            "error": self.error,
            "duration_s": round(self.duration_s, 3),
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        status = "OK" if self.success else f"FAIL({self.error[:60]})"
        return (
            f"AgentResult(agent={self.agent_name!r}, "
            f"status={status}, "
            f"duration={self.duration_s:.2f}s)"
        )


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """
    Abstract base for all pipeline agents.

    Parameters
    ----------
    config  : AgentConfig instance from config.py describing this agent.
    llm     : A LangChain ChatModel (e.g. ChatOllama) shared across agents.
    memory  : The shared ShortTermMemory instance for the current session.
    tools   : LangChain tool objects this agent is allowed to call (pre-filtered
              by the orchestrator based on config.allowed_mcp_servers).
    """

    def __init__(
        self,
        config: AgentConfig,
        llm: BaseChatModel,
        memory: ShortTermMemory,
        tools: list[BaseTool] | None = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._memory = memory
        self._tools: list[BaseTool] = tools or []
        self._executor: Any | None = None

        # Configure per-agent logger
        self._log = logging.getLogger(f"agent.{self.name}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique identifier for this agent (from AgentConfig)."""
        return self._config.name

    @property
    def description(self) -> str:
        return self._config.description

    @property
    def config(self) -> AgentConfig:
        return self._config

    @property
    def memory(self) -> ShortTermMemory:
        return self._memory

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools)

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self._tools]

    # ------------------------------------------------------------------
    # Public entry-point  (called by the orchestrator)
    # ------------------------------------------------------------------

    async def run(self, task: str, **kwargs: Any) -> AgentResult:
        """
        Execute the agent on *task* and return an AgentResult.

        This method wraps ``_execute()`` with:
        - Timing
        - Structured logging (start / success / failure)
        - Automatic memory update on success

        Subclasses should override ``_execute()``, not this method.
        """
        self._log.info("▶ 开始执行 | task=%r", task[:120])
        t0 = time.perf_counter()

        try:
            result = await self._execute(task, **kwargs)
        except Exception as exc:
            duration = time.perf_counter() - t0
            self._log.exception("✗ 未捕获异常，已运行 %.2fs", duration)
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_s=duration,
            )

        result.duration_s = time.perf_counter() - t0

        if result.success:
            self._log.info("✔ 执行完成，耗时 %.2fs", result.duration_s)
            # 将输出写入共享记忆
            self._memory.add_ai_message(
                result.output,
                metadata={"agent": self.name, "duration_s": result.duration_s},
            )
            # 将结构化数据存入上下文
            if result.data:
                self._memory.update_context(result.data)
        else:
            self._log.warning(
                "✗ 执行失败，耗时 %.2fs | error=%s", result.duration_s, result.error
            )

        return result

    # ------------------------------------------------------------------
    # Abstract method – subclasses implement this
    # ------------------------------------------------------------------

    @abstractmethod
    async def _execute(self, task: str, **kwargs: Any) -> AgentResult:
        """
        Core agent logic.

        Subclasses receive the *task* string (written by the orchestrator or
        a previous agent) and should return an AgentResult.

        Tips
        ----
        - Call ``self._invoke_llm(messages)`` for a plain LLM call.
        - Call ``self._invoke_with_tools(task)`` to let the LLM call MCP tools.
        - Read structured runtime state with ``self.memory.get_pipeline_value("key")``.
        - Write pipeline context via the returned AgentResult.data dict.
        """

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    async def _invoke_llm(
        self,
        messages: list[BaseMessage] | None = None,
        extra_user_message: str | None = None,
    ) -> str:
        """
        Call the LLM with the current memory context (+ optional extra message).

        Parameters
        ----------
        messages          : If provided, use these messages instead of memory.
        extra_user_message: Appended as a HumanMessage after the base message list.

        Returns
        -------
        The text content of the LLM's reply.
        """
        if messages is None:
            # Keep only recent turns to avoid overflowing context when prior
            # tool traces are verbose.
            messages = self._memory.get_last_n_messages(24)

        if extra_user_message:
            messages = list(messages) + [HumanMessage(content=extra_user_message)]

        self._log.debug("_invoke_llm | 发送 %d 条消息到 LLM", len(messages))

        response: AIMessage = await self._llm.ainvoke(messages)
        return str(response.content)

    async def _invoke_llm_structured(
        self,
        schema: type[Any],
        messages: list[BaseMessage] | None = None,
        extra_user_message: str | None = None,
    ) -> Any | None:
        """
        Call the LLM with schema-constrained output when supported.

        Returns
        -------
        Parsed structured payload (usually a Pydantic model instance or dict),
        or None if structured output is not supported / failed.
        """
        if messages is None:
            messages = self._memory.get_messages()

        if extra_user_message:
            messages = list(messages) + [HumanMessage(content=extra_user_message)]

        if PROVIDER == "ollama-cloud":
            self._memory.add_ai_message(
                f"[{self.name}] ollama-cloud 结构化输出不稳定，直接使用文本生成。",
                metadata={"agent": self.name, "event": "structured_skipped"},
            )
            return STRUCTURED_OUTPUT_SKIPPED

        if not hasattr(self._llm, "with_structured_output"):
            self._log.debug("当前模型不支持 with_structured_output，回退到文本解析")
            return None

        try:
            structured_llm = self._llm.with_structured_output(schema)
            return await structured_llm.ainvoke(messages)
        except Exception as exc:  # noqa: BLE001
            self._log.warning("结构化输出调用失败，回退到文本解析：%s", exc)
            # Mirror fallback reason into shared memory so GUI can show live status.
            self._memory.add_ai_message(
                f"[结构化输出回退] {type(exc).__name__}: {str(exc)[:240]}",
                metadata={"agent": self.name, "event": "structured_fallback"},
            )
            return None

    async def _invoke_with_tools(
        self,
        task: str,
        extra_context: str | None = None,
    ) -> str:
        """
        Run a LangChain agent loop: the LLM decides which tools to call,
        calls them (via MCP), and produces a final answer.

        After execution, extracts all intermediate messages (tool calls,
        tool results, final response) and records them to shared memory
        so the GUI can display the full creative process.

        Parameters
        ----------
        task          : The task description sent as the human turn.
        extra_context : Optional additional context prepended to the task.

        Returns
        -------
        The final text answer produced by the LLM after all tool calls.
        """
        if not self._tools:
            self._log.warning(
                "代理 '%s' 调用了 _invoke_with_tools 但未绑定任何工具，"
                "回退为普通 LLM 调用。",
                self.name,
            )
            return await self._invoke_llm(extra_user_message=task)

        # Build or reuse the cached agent graph
        if self._executor is None:
            self._executor = self._build_executor()

        full_task = f"{extra_context}\n\n{task}" if extra_context else task

        self._log.debug(
            "_invoke_with_tools | task=%r | 工具=%s",
            full_task[:80], self.tool_names,
        )

        if self._executor is None:
            raise RuntimeError("Agent graph not initialized")

        # Use a compact rolling window for tool-agent turns.
        messages = list(self._memory.get_last_n_messages(24, include_system=False))
        messages.append(HumanMessage(content=full_task))

        # Record a marker to memory so GUI shows the agent is working
        self._memory.add_ai_message(
            f"🔄 正在处理任务... (使用工具: {', '.join(self.tool_names[:6])}{'...' if len(self.tool_names) > 6 else ''})",
            metadata={"agent": self.name, "event": "processing"},
        )

        result = await self._executor.ainvoke({"messages": messages})

        # Extract ALL messages from the agent execution and record to memory
        # This includes tool calls, tool results, and the final AI response
        self._record_agent_messages(result)

        output_str = self._extract_agent_output(result)
        if not output_str.strip():
            # Recover from runs where the model only emitted tool calls and
            # never produced a final assistant text.
            output_str = self._extract_lyrics_from_tool_calls(result)
        self._log.debug("_invoke_with_tools | extracted_output=%r", output_str[:200])

        return output_str

    def _record_agent_messages(self, result: Any) -> None:
        """
        Extract all messages from the agent execution result and record
        them to shared memory for GUI visibility.

        This captures the full creative process:
        - LLM tool calls (with arguments)
        - Tool execution results
        - Final AI text response
        """
        if not isinstance(result, dict):
            return

        result_messages = result.get("messages", [])
        if not isinstance(result_messages, list):
            return

        # Build a set of content hashes from existing memory to avoid duplicates
        existing_content_hashes = set()
        for turn in self._memory._turns:
            for msg in turn.messages:
                content = getattr(msg, "content", "")
                if content:
                    existing_content_hashes.add(hash(str(content)[:200]))

        # Process each message from the agent execution
        for msg in result_messages:
            # Skip HumanMessage inputs - those are already added by the caller
            if isinstance(msg, HumanMessage):
                continue

            content = getattr(msg, "content", "")
            content_hash = hash(str(content)[:200]) if content else None

            # Skip duplicates
            if content_hash and content_hash in existing_content_hashes:
                continue

            # Record this message to memory
            if isinstance(msg, AIMessage):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    # This is a tool call message
                    for tc in msg.tool_calls:
                        # Log to agent-specific logger
                        self._log.info(
                            "📝 [GUI] Agent → Tool: %s(%s)",
                            tc.get("name", "?"),
                            str(tc.get("args", {}))[:100],
                        )
                        # Also log to main logger for console visibility
                        logger = logging.getLogger("main")
                        logger.info(
                            "[AGENT TOOL CALL] %s → %s(%s)",
                            self.name,
                            tc.get("name", "?"),
                            str(tc.get("args", {}))[:100],
                        )
                        # Emit pipeline event for UI/activity panel if orchestrator event callback is present
                        event_callback = getattr(self, "_event_callback", None)
                        if callable(event_callback):
                            try:
                                maybe_awaitable = event_callback(
                                    {
                                        "type": "agent_tool_call",
                                        "agent": self.name,
                                        "tool": tc.get("name", "?"),
                                        "args": tc.get("args", {}),
                                    }
                                )
                                if hasattr(maybe_awaitable, "__await__"):
                                    import asyncio

                                    asyncio.create_task(maybe_awaitable)
                            except Exception as exc:
                                logger.warning(
                                    "Failed to emit agent tool call event: %s", exc
                                )
                elif content:
                    # This is a text response
                    self._log.info("📝 [GUI] Agent response: %r", content[:100])
            elif isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "unknown")
                self._log.info(
                    "📝 [GUI] Tool result (%s): %r",
                    tool_name,
                    str(content)[:100],
                )

            # Add to memory with truncation to avoid token explosion.
            compact_msg: BaseMessage | None = None
            if isinstance(msg, AIMessage):
                text = str(content) if content is not None else ""
                if text.strip():
                    compact_msg = AIMessage(content=text[:800])
            elif isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "tool")
                compact_msg = AIMessage(content=f"[tool:{tool_name}] result received")

            if compact_msg is not None:
                self._memory._append_turn([compact_msg], metadata={"agent": self.name})
            if content_hash:
                existing_content_hashes.add(content_hash)

    @staticmethod
    def _extract_lyrics_from_tool_calls(result: Any) -> str:
        """Best-effort recovery when agent output is empty but tool calls contain lyrics."""
        if not isinstance(result, dict):
            return ""
        messages = result.get("messages", [])
        if not isinstance(messages, list):
            return ""

        for msg in reversed(messages):
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                continue
            for tc in reversed(tool_calls):
                if str(tc.get("name", "")) != "count_syllables":
                    continue
                args = tc.get("args", {})
                if isinstance(args, dict):
                    lyrics = str(args.get("lyrics", "")).strip()
                    if lyrics:
                        return lyrics
        return ""

    def _build_executor(self) -> Any:
        """
        构建带工具调用支持的 LangChain create_agent 图。

        系统提示优先级（从高到低）：
        1. 代理专属提示文件（prompts/<agent-name>.md）
        2. 共享系统提示（memory.system_prompt，来自 prompts/system.md）
        3. 硬编码的中文默认提示

        可在子类中重写此方法以自定义执行器行为
        （例如修改 max_iterations、添加自定义输出解析器等）。
        """
        # 1. 优先使用代理专属提示文件
        system_prompt = self._load_prompt_file()

        # 2. 回退到共享系统提示
        if not system_prompt:
            system_prompt = self._memory.system_prompt

        # 3. 最终回退：硬编码中文默认提示
        if not system_prompt:
            system_prompt = (
                "你是一位专业的粤语作词人和语言学家，"
                "擅长根据 MIDI 旋律生成符合粤语声调规律的歌词。"
                "请始终用中文思考和回答，并以 JSON 格式输出结构化结果。"
            )
            self._log.warning(
                "代理 '%s' 没有专属提示文件，且共享系统提示未设置，使用默认提示。",
                self.name,
            )
        else:
            self._log.debug(
                "代理 '%s' 使用系统提示（来源：%s）",
                self.name,
                "专属文件" if self._load_prompt_file() else "共享提示",
            )

        if create_agent is None:
            raise RuntimeError(
                "langchain.agents.create_agent not available in the current "
                "LangChain installation."
            )

        return create_agent(
            model=self._llm,
            tools=self._tools,
            system_prompt=system_prompt,
            debug=WORKFLOW_CONFIG["stream_output"],
            name=self.name,
        )

    @staticmethod
    def _extract_agent_output(result: Any) -> str:
        """Extract the final assistant text from a create_agent result payload."""
        if isinstance(result, dict):
            if "output" in result:
                return str(result.get("output", ""))

            messages = result.get("messages")
            if isinstance(messages, list) and messages:
                # Walk backwards to find the last AIMessage with actual text content
                for msg in reversed(messages):
                    if not hasattr(msg, "content"):
                        continue
                    content = msg.content
                    if isinstance(content, list):
                        parts: list[str] = []
                        for item in content:
                            if isinstance(item, str):
                                parts.append(item)
                            elif isinstance(item, dict):
                                text = item.get("text")
                                if text:
                                    parts.append(str(text))
                        joined = "\n".join(p for p in parts if p).strip()
                        if joined:
                            return joined
                    elif isinstance(content, str) and content.strip():
                        return content.strip()

                # Fallback: return whatever the last message had
                last = messages[-1]
                last_content = getattr(last, "content", "")
                if isinstance(last_content, str):
                    return last_content
                if isinstance(last_content, list):
                    parts = []
                    for item in last_content:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict):
                            text = item.get("text")
                            if text:
                                parts.append(str(text))
                    return "\n".join(p for p in parts if p).strip()
                return str(last_content)

        return str(result or "")

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, task: str, **context_vars: Any) -> str:
        """
        构建代理的任务提示，可选择注入流水线上下文变量
        （如 midi_analysis、jyutping_map）。

        子类可重写此方法，使用从 prompts/<agent-name>.md 加载的模板。
        """
        lines: list[str] = [f"## 任务\n{task}"]

        if context_vars:
            lines.append("\n## 上下文")
            for key, value in context_vars.items():
                lines.append(f"\n### {key}\n{value}")

        # 从共享记忆上下文中注入相关键值
        midi     = self._memory.get_pipeline_value("midi_analysis")
        jyutping = self._memory.get_pipeline_value("jyutping_map")
        draft    = self._memory.get_current_draft()

        if midi and "midi_analysis" not in context_vars:
            lines.append(f"\n### midi_analysis\n{midi}")
        if jyutping and "jyutping_map" not in context_vars:
            lines.append(f"\n### jyutping_map\n{jyutping}")
        if draft and "draft_lyrics" not in context_vars:
            lines.append(f"\n### draft_lyrics\n{draft}")

        return "\n".join(lines)

    def _load_prompt_file(self) -> str | None:
        """
        加载代理的专属提示文件（若已配置）。

        查找顺序：
        1. self._config.prompt_file（在 config.py 中显式配置的路径）
        2. prompts/<agent-name>.md（约定路径，自动发现）

        若文件不存在或未配置，返回 None。
        """
        # 优先使用 config 中显式配置的路径
        pf = self._config.prompt_file
        if pf is not None and pf.exists():
            content = pf.read_text(encoding="utf-8").strip()
            self._log.debug("已加载专属提示文件：%s", pf)
            return content

        # 按约定路径自动发现（prompts/<name>.md）
        from .config import PROMPTS_AGENTS_DIR

        convention_path = PROMPTS_AGENTS_DIR / f"{self.name}.md"
        if convention_path.exists():
            content = convention_path.read_text(encoding="utf-8").strip()
            self._log.debug("已按约定路径加载提示文件：%s", convention_path)
            return content

        self._log.debug("代理 '%s' 没有找到专属提示文件。", self.name)
        return None

    # ------------------------------------------------------------------
    # Tool management
    # ------------------------------------------------------------------

    def set_tools(self, tools: list[BaseTool]) -> None:
        """
        替换工具列表并使缓存的执行器失效。

        由 orchestrator 在 MCP 客户端连接并解析工具后调用。
        """
        self._tools   = tools
        self._executor = None  # 强制下次重建
        self._log.debug("工具已更新：%s", [t.name for t in tools])

    def add_tool(self, tool: BaseTool) -> None:
        """追加单个工具并使执行器缓存失效。"""
        self._tools.append(tool)
        self._executor = None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"tools={self.tool_names})"
        )
