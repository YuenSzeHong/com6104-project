"""
WordSelectorAgent – 选字 Agent.

Responsibilities
----------------
- 从 0243.hk API 返回的大量候选词中筛选最合适的词语
- 根据上下文、语义、韵律等因素进行智能选择
- 支持批量选词，保持歌词整体连贯性

Context keys read
~~~~~~~~~~~~~~~~~
- ``candidate_words`` (list) – 0243.hk API 返回的候选词列表
- ``selection_context`` (dict) – 选字上下文，包括：
    - position: 词语在歌词中的位置
    - melody_tone: 旋律声调要求
    - surrounding_words: 上下文词语
    - semantic_field: 语义场/主题
    - rhyme_final: 押韵要求

Context keys written
~~~~~~~~~~~~~~~~~~~~
- ``selected_words`` (dict) – 选中的词语及其理由
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agent.base_agent import BaseAgent, AgentResult, STRUCTURED_OUTPUT_SKIPPED
from agent.config import PROMPTS_AGENTS_DIR
from agent.registry import AGENT_REGISTRY

logger = logging.getLogger(__name__)


class WordSelectionSchema(BaseModel):
    """Structured output schema for word selection."""

    word: str = Field(default="", description="选中的词，必须来自候选词列表")
    reason: str = Field(default="", description="选择该词的简短理由")
    alternatives: list[str] = Field(
        default_factory=list,
        description="备选词列表（最多 3 个）",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="选择置信度，范围 0.0-1.0",
    )


@AGENT_REGISTRY.register("word-selector")
class WordSelectorAgent(BaseAgent):
    """
    从候选词列表中选择最合适的词语。

    该 Agent 使用 LLM 的语义理解能力，从 0243.hk API 返回的候选词中
    选择最符合以下条件的词语：

    1. 语义连贯性：与上下文语义相符
    2. 声调匹配：符合旋律的声调要求
    3. 韵律协调：与押韵位置的字协调
    4. 主题一致：符合歌词整体主题和情感
    5. 用词自然：避免生僻字和不自然的搭配
    """

    async def _execute(self, task: str, **kwargs: Any) -> AgentResult:
        """
        执行选字任务。

        Parameters
        ----------
        task : str
            选字任务描述，包含候选词列表和选择要求
        kwargs : dict
            可选参数：
            - candidates: list[str] – 候选词列表
            - context: dict – 选字上下文
            - count: int – 需要选择的词语数量（默认 1）
        """
        # 解析输入参数
        candidates: list[str] = kwargs.get("candidates", [])
        context: dict[str, Any] = kwargs.get("context", {})
        count: int = kwargs.get("count", 1)

        # 如果候选词为空，直接返回
        if not candidates:
            return AgentResult(
                agent_name=self.name,
                success=True,
                output="[]",
                data={"selected_words": []},
                metadata={"reason": "no_candidates"},
            )

        # 限制候选词数量（避免超出 LLM 上下文限制）
        max_candidates = 50
        if len(candidates) > max_candidates:
            self._log.warning(
                "候选词过多 (%d 个)，截断至 %d 个",
                len(candidates), max_candidates,
            )
            candidates = candidates[:max_candidates]

        self._log.info(
            "开始选字 | 候选词=%d | 需选=%d | 位置=%s",
            len(candidates), count, context.get("position", "?"),
        )

        # 构建选字提示
        prompt = self._build_selection_prompt(
            candidates=candidates,
            context=context,
            count=count,
        )

        # 调用 LLM
        self._memory.add_user_message(prompt)
        selected = await self._select_with_schema(prompt, candidates)

        self._log.info(
            "选字完成 | 选中=%s | 理由=%s",
            selected.get("word", "?"),
            selected.get("reason", "?")[:60],
        )

        return AgentResult(
            agent_name=self.name,
            success=True,
            output=selected.get("word", ""),
            data={"selected_words": [selected]},
            metadata={
                "candidates_count": len(candidates),
                "selection_reason": selected.get("reason", ""),
                "alternatives": selected.get("alternatives", []),
            },
        )

    async def _select_with_schema(
        self,
        prompt: str,
        candidates: list[str],
    ) -> dict[str, Any]:
        """
        Prefer schema-constrained output; fall back to legacy text parsing.
        """
        structured = await self._invoke_llm_structured(
            schema=WordSelectionSchema,
            extra_user_message=prompt,
        )

        if structured is STRUCTURED_OUTPUT_SKIPPED:
            structured = None

        if structured is not None:
            parsed = (
                structured.model_dump()
                if hasattr(structured, "model_dump")
                else dict(structured)
                if isinstance(structured, dict)
                else {}
            )
            return self._normalize_selection(parsed, candidates)

        raw_response = await self._invoke_llm(extra_user_message=prompt)
        return self._parse_response(raw_response, candidates)

    def _build_selection_prompt(
        self,
        candidates: list[str],
        context: dict[str, Any],
        count: int,
    ) -> str:
        """构建选字提示。"""
        # 上下文信息
        position = context.get("position", "未知位置")
        melody_tone = context.get("melody_tone", "未知")
        surrounding_before = context.get("surrounding_before", "")
        surrounding_after = context.get("surrounding_after", "")
        semantic_field = context.get("semantic_field", "")
        rhyme_final = context.get("rhyme_final", "")
        theme = context.get("theme", "通用主题")

        # 构建候选词列表
        candidates_str = "\n".join(f"  {i + 1}. {word}" for i, word in enumerate(candidates))

        # 构建上下文描述
        context_parts = []
        if surrounding_before:
            context_parts.append(f"前文：{surrounding_before}")
        if surrounding_after:
            context_parts.append(f"后文：{surrounding_after}")
        if semantic_field:
            context_parts.append(f"语义场：{semantic_field}")
        if theme:
            context_parts.append(f"主题：{theme}")
        if melody_tone:
            context_parts.append(f"旋律声调：{melody_tone}")
        if rhyme_final:
            context_parts.append(f"押韵要求：{rhyme_final}")

        context_str = "\n".join(context_parts) if context_parts else "无明显限制"

        return self._render_prompt_template(
            "word-selector-task.md",
            position=position,
            candidates_str=candidates_str,
            context_str=context_str,
            count=count,
            candidates_json=json.dumps(candidates, ensure_ascii=False),
        )

    def _parse_response(
        self,
        raw: str,
        candidates: list[str],
    ) -> dict[str, Any]:
        """解析 LLM 返回的选字结果。"""
        # 尝试解析 JSON
        json_str = self._extract_json_block(raw)
        parsed: dict[str, Any] = {}

        if json_str:
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                self._log.warning("无法解析选字 JSON，使用启发式提取")
                parsed = self._extract_selection_heuristically(raw, candidates)
        else:
            parsed = self._extract_selection_heuristically(raw, candidates)

        # 验证选中的词是否在候选列表中
        selected_word = str(parsed.get("word", "")).strip()
        if selected_word and selected_word not in candidates:
            # 尝试模糊匹配
            for c in candidates:
                if c.strip() == selected_word:
                    selected_word = c
                    break
            else:
                self._log.warning(
                    "选中的词不在候选列表中：%r，使用第一个候选",
                    selected_word,
                )
                selected_word = candidates[0] if candidates else ""

        if not selected_word and candidates:
            selected_word = candidates[0]

        parsed["word"] = selected_word
        return self._normalize_selection(parsed, candidates)

    def _normalize_selection(
        self,
        parsed: dict[str, Any],
        candidates: list[str],
    ) -> dict[str, Any]:
        """Normalize and validate selection payload to a stable shape."""
        selected_word = str(parsed.get("word", "")).strip()
        if selected_word and selected_word not in candidates:
            for candidate in candidates:
                if candidate.strip() == selected_word:
                    selected_word = candidate
                    break
            else:
                self._log.warning(
                    "选中的词不在候选列表中：%r，使用第一个候选",
                    selected_word,
                )
                selected_word = candidates[0] if candidates else ""

        if not selected_word and candidates:
            selected_word = candidates[0]

        alternatives_raw = parsed.get("alternatives", [])
        alternatives = (
            [str(item) for item in alternatives_raw if str(item).strip()][:3]
            if isinstance(alternatives_raw, list)
            else []
        )

        confidence_raw = parsed.get("confidence", 0.8)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.8
        confidence = max(0.0, min(1.0, confidence))

        return {
            "word": selected_word,
            "reason": str(parsed.get("reason", "未提供理由")),
            "alternatives": alternatives,
            "confidence": confidence,
        }

    @staticmethod
    def _extract_selection_heuristically(
        text: str,
        candidates: list[str],
    ) -> dict[str, Any]:
        """启发式提取选字结果。"""
        result: dict[str, Any] = {}

        # 提取选中的词
        for key in ("word", "selected", "选择", "推荐"):
            pattern = rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                result["word"] = match.group(1).replace('\\"', '"')
                break

        # 如果没有找到 JSON，尝试从纯文本中提取
        if "word" not in result:
            # 查找第一个出现在候选列表中的词
            for c in candidates:
                if c in text:
                    result["word"] = c
                    break

        # 提取理由
        for key in ("reason", "理由", "explanation"):
            pattern = rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                result["reason"] = match.group(1).replace('\\"', '"')
                break

        return result

    @staticmethod
    def _extract_json_block(text: str) -> str:
        """从文本中提取 JSON 块。"""
        # 处理 markdown 代码块
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fenced:
            return fenced.group(1).strip()

        # 查找平衡的 { ... }
        start = text.find("{")
        if start == -1:
            return ""

        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        return ""

    @staticmethod
    def _render_prompt_template(template_name: str, **kwargs: Any) -> str:
        """渲染提示模板。"""
        template_path: Path = PROMPTS_AGENTS_DIR / template_name
        if template_path.exists():
            return template_path.read_text(encoding="utf-8").format(**kwargs).strip()

        # 默认模板
        return f"""
# 选字任务

## 位置
{kwargs.get('position', '未知位置')}

## 候选词列表
{kwargs.get('candidates_str', '')}

## 上下文
{kwargs.get('context_str', '无明显限制')}

## 任务
请从候选词列表中选择{kwargs.get('count', 1)}个最合适的词语。

## 输出格式
请以 JSON 格式输出：
```json
{{
    "word": "选中的词",
    "reason": "选择理由",
    "alternatives": ["备选词 1", "备选词 2"],
    "confidence": 0.9
}}
```
""".strip()
