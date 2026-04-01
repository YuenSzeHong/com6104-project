# 选字 Agent (WordSelectorAgent)

## 概述

选字 Agent 用于从 0243.hk API 返回的大量候选词中智能筛选最合适的词语。

当 0243.hk API 返回过多候选词时（例如某个声调码可能返回几十上百个词），直接让 LLM 从大量候选中选择会超出上下文限制或导致选择不一致。选字 Agent 通过以下方式解决此问题：

1. **候选词截断**：限制最多 50 个候选词送入 LLM
2. **上下文感知选择**：考虑前后文、语义场、主题等因素
3. **结构化输出**：返回选中的词、选择理由、备选词和置信度

## 架构设计

### 集成方式

选字 Agent **不直接暴露为 MCP 工具**，而是作为内部 Agent 被 LyricsComposerAgent 调用：

```
┌─────────────────────────────────────────────────────────────┐
│                   LyricsComposerAgent                        │
│                                                               │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐ │
│  │ 0243.hk API │───▶│ 候选词过滤   │───▶│ WordSelectorAgent│ │
│  │ (MCP 工具)    │    │ (>10 个候选)  │    │ (内部调用)       │ │
│  └─────────────┘    └──────────────┘    └─────────────────┘ │
│                                                               │
│  上下文：前后文、主题、韵律、声调                              │
└─────────────────────────────────────────────────────────────┘
```

### 为什么这样设计？

1. **减少工具调用上下文**：选字 Agent 可以看到完整的歌词上下文，而不是每次单独调用
2. **更高效的选择**：基于前后文、主题、韵律等完整信息做选择
3. **避免重复调用**：只有候选词超过阈值（10 个）时才调用选字 Agent

## 使用方式

### 1. LyricsComposerAgent 自动调用（推荐）

在 LyricsComposerAgent 创作歌词时，会自动调用选字 Agent 处理候选词过多的位置：

```python
from agent.agents import LyricsComposerAgent

composer = LyricsComposerAgent(...)

# 创作歌词时，如果有位置候选词超过 10 个，会自动调用选字 Agent
result = await composer.run(
    task="根据旋律创作粤语歌词",
)

# 内部流程：
# 1. LLM 生成初始歌词
# 2. 检查 candidate_words_map（来自 0243.hk API）
# 3. 对候选词>10 的位置，调用 WordSelectorAgent 选字
# 4. 更新歌词并返回
```

### 2. 直接调用选字 Agent（高级用法）

```python
from agent.agents import WordSelectorAgent
from agent.config import AgentConfig, LLM_CONFIG
from agent.memory import ShortTermMemory
from langchain_openai import ChatOpenAI

# 初始化
config = AgentConfig(
    name="word-selector",
    description="选字代理",
)
llm = ChatOpenAI(
    model=LLM_CONFIG["model"],
    base_url=LLM_CONFIG["base_url"],
    api_key=LLM_CONFIG["api_key"],
    temperature=LLM_CONFIG["temperature"],
)
memory = ShortTermMemory()

agent = WordSelectorAgent(
    config=config,
    llm=llm,
    memory=memory,
)

# 调用
result = await agent.run(
    task="从候选词中选择最合适的词语",
    candidates=["世界", "唱歌", "快乐", "美好", ...],  # 0243.hk 返回的候选列表
    context={
        "position": "第 3 句第 2 字",
        "melody_tone": "43",
        "surrounding_before": "我要",
        "surrounding_after": "明天",
        "semantic_field": "时间、未来",
        "theme": "希望与梦想",
    },
    count=1,  # 需要选择的词语数量
)

# 结果
selected_word = result.output  # 选中的词
reason = result.metadata["selection_reason"]  # 选择理由
alternatives = result.metadata["alternatives"]  # 备选词
```

### 2. 在 LyricsComposerAgent 中集成

选字 Agent 可集成到歌词创作流程中，在以下场景自动调用：

1. **候选词过多时**：当 0243.hk 返回超过 10 个候选词
2. **关键位置选字**：押韵位置、强拍位置
3. **语义模糊时**：多个候选词语义差异较大

```python
# 在 LyricsComposerAgent 内部
async def _select_best_word(
    self,
    candidates: list[str],
    position: int,
    context: dict,
) -> str:
    if len(candidates) <= 3:
        # 候选词少，直接让 LLM 在创作时选择
        return candidates[0] if candidates else ""
    
    # 候选词多，调用选字 Agent
    selector = WordSelectorAgent(...)
    result = await selector.run(
        task=f"为位置 {position} 选择最佳词语",
        candidates=candidates,
        context=context,
    )
    return result.output
```

## 选择标准

选字 Agent 根据以下标准评估候选词：

| 标准 | 权重 | 说明 |
|------|------|------|
| 语义连贯 | 30% | 与上下文语义自然衔接 |
| 声调匹配 | 25% | 符合旋律的声调要求 |
| 韵律协调 | 20% | 如处押韵位置，需考虑韵脚 |
| 主题一致 | 15% | 符合歌词整体主题和情感基调 |
| 用词自然 | 10% | 优先选择常用、自然的表达 |

## 输出格式

选字 Agent 返回 JSON 格式的结构化结果：

```json
{
    "word": "选中的词语",
    "reason": "详细的选择理由，解释为何该词最符合上述标准",
    "alternatives": ["备选词 1", "备选词 2", "备选词 3"],
    "confidence": 0.85
}
```

## 配置

在 `src/agent/config.py` 中已配置选字 Agent：

```python
AgentConfig(
    name        = "word-selector",
    description = "选字代理（LLM 驱动）...",
    allowed_mcp_servers = ["jyutping"],
    prompt_file = PROMPTS_DIR / "word-selector-task.md",
)
```

## 测试

运行选字相关测试：

```bash
# 运行所有测试
pytest tests/test_word_selector.py -v

# 运行特定测试
pytest tests/test_word_selector.py::test_find_words_returns_candidates -v
```

## 示例

### 示例 1：基本选字

**输入**：
- 候选词：["世界", "唱歌", "快乐", "美好", "天地", "风雨", ...]
- 位置：副歌第 1 句第 3 字
- 前文："我要"
- 后文："明天"
- 主题："希望与梦想"

**输出**：
```json
{
    "word": "世界",
    "reason": "「世界」与「我要」「明天」形成完整语义链「我要世界明天」，表达对未来广阔天地的向往，符合希望与梦想主题。声调 43 与旋律要求匹配。",
    "alternatives": ["天地", "美好"],
    "confidence": 0.92
}
```

### 示例 2：押韵位置选字

**输入**：
- 候选词：["唱歌", "快乐", "灯火", "你我", ...]
- 位置：押韵位置（韵脚：-o）
- 前文："一起"
- 韵脚要求："-o"

**输出**：
```json
{
    "word": "唱歌",
    "reason": "「唱歌」韵脚为 -o，与押韵要求完全匹配。语义上「一起唱歌」自然流畅，符合歌词场景。",
    "alternatives": ["你我", "快乐"],
    "confidence": 0.95
}
```

## 注意事项

1. **候选词数量限制**：超过 50 个候选词会被截断，可能遗漏最佳选择
2. **LLM 上下文限制**：大量候选词会占用较多 token，注意上下文窗口
3. **选择一致性**：相同输入可能因 LLM 温度设置产生不同输出，建议设置较低温度（0.3-0.5）
4. **性能考虑**：选字过程涉及 LLM 调用，耗时约 1-3 秒，建议仅在必要时调用
