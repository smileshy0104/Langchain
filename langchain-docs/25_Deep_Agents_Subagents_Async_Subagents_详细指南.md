# Deep Agents Subagents 与 Async Subagents 详细指南

> 基于 Deep Agents 官方 `Subagents` 与 `Async subagents` 文档整理。本文聚焦如何使用同步子智能体隔离上下文、如何配置自定义子智能体、如何流式观察和追踪子智能体、如何使用结构化输出与 skills，以及如何用 async subagents 启动后台并发任务。

## 目录

1. [核心概念](#核心概念)
2. [为什么需要 Subagents](#为什么需要-subagents)
3. [同步 Subagents 与 Async Subagents 对比](#同步-subagents-与-async-subagents-对比)
4. [同步 Subagents 配置](#同步-subagents-配置)
5. [默认 General-purpose Subagent](#默认-general-purpose-subagent)
6. [自定义 SubAgent](#自定义-subagent)
7. [CompiledSubAgent](#compiledsubagent)
8. [Streaming 与 LangSmith Tracing](#streaming-与-langsmith-tracing)
9. [Structured Output](#structured-output)
10. [Skills 继承与隔离](#skills-继承与隔离)
11. [Context Management](#context-management)
12. [Async Subagents 概览](#async-subagents-概览)
13. [Async Subagents 配置](#async-subagents-配置)
14. [Async Subagent Tools 与生命周期](#async-subagent-tools-与生命周期)
15. [Async State Management](#async-state-management)
16. [Transport 与部署拓扑](#transport-与部署拓扑)
17. [常见模式](#常见模式)
18. [最佳实践](#最佳实践)
19. [故障排查](#故障排查)
20. [快速参考](#快速参考)
21. [资料来源](#资料来源)

---

## 核心概念

Deep Agents 可以创建子智能体，把一部分任务委派出去。主智能体通常负责高层规划、协调和最终合成；子智能体负责具体的多步骤执行、检索、分析、代码、审查等工作。

最核心的价值是：

```text
让复杂工作发生在子智能体自己的上下文里，
主智能体只接收最终结果，
避免主上下文被大量中间工具调用污染。
```

一个典型流程：

```text
User
  -> Main Agent / Supervisor
    -> task tool / async task tool
      -> Subagent
        -> tools / files / model calls
      <- concise final result
  <- final answer
```

Deep Agents 中有两类子智能体：

| 类型 | 执行方式 | 适合场景 |
|------|----------|----------|
| 同步 subagents | 主智能体调用后阻塞等待结果 | 需要马上拿到结果再继续推理 |
| Async subagents | 后台启动任务，立即返回 task ID | 长任务、并发任务、需要中途更新或取消 |

---

## 为什么需要 Subagents

Subagents 主要解决 **context bloat** 问题。

当主智能体直接执行大量工具调用时，主上下文会迅速被填满，例如：

| 工具/任务 | 容易产生的问题 |
|-----------|----------------|
| Web search | 搜索结果、网页全文、引用片段很多 |
| 文件读取 | 大文件内容占满上下文 |
| 数据库查询 | 表格结果和中间 SQL 过程很多 |
| 代码分析 | 多文件、多轮 grep/read/edit 产生大量历史 |
| 多主题研究 | 每个主题都有独立中间过程 |

使用 subagents 后：

```text
主智能体：负责决定要研究什么
子智能体：执行搜索、读取、分析、试错
主智能体：只拿到最终摘要或结构化结果
```

适合使用 subagents：

| 场景 | 原因 |
|------|------|
| 多步骤任务 | 中间过程多，容易污染主上下文 |
| 专门领域 | 需要特定 system prompt、工具或模型 |
| 不同能力模型 | 子任务可用更适合的模型 |
| 高层协调 | 主智能体只需要最终结论 |
| 多工具探索 | 搜索、文件、数据库等工具输出很大 |

不适合使用 subagents：

| 场景 | 原因 |
|------|------|
| 简单单步任务 | 委派开销大于收益 |
| 主智能体必须保留完整中间上下文 | 子智能体隔离反而不方便 |
| 子任务很短 | 直接执行更简单 |
| 需要严格共享逐步推理过程 | 子智能体只返回结果，不返回全部历史 |

---

## 同步 Subagents 与 Async Subagents 对比

| 维度 | 同步 Subagents | Async Subagents |
|------|----------------|-----------------|
| 执行模型 | supervisor 阻塞等待 subagent 完成 | 启动后立即返回 task ID |
| 并发 | 可以并行，但 supervisor 等结果 | 并行且非阻塞 |
| 中途更新 | 不支持 | 支持 `update_async_task` |
| 取消任务 | 不支持 | 支持 `cancel_async_task` |
| 状态 | 单次调用，通常无跨调用持久状态 | 每个任务有自己的 thread，具备状态 |
| 结果获取 | 子智能体完成后直接返回 | supervisor 后续 `check_async_task` |
| 适合 | 子结果是下一步推理的必要输入 | 长任务、后台任务、用户可继续交互 |
| 实现依赖 | Deep Agents 内部同步 subagent middleware | Agent Protocol server / LangSmith Deployment / self-host |

选择建议：

```text
需要马上等结果 -> 同步 SubAgent
任务很久、可并行、需要中途追加要求或取消 -> AsyncSubAgent
```

---

## 同步 Subagents 配置

同步 subagents 通过 `create_deep_agent(..., subagents=...)` 配置。`subagents` 可以包含：

1. 字典形式的 `SubAgent`。
2. `CompiledSubAgent` 对象。
3. AsyncSubAgent 对象，走 async middleware。

简单示例：

```python
import os
from typing import Literal

from deepagents import create_deep_agent
from tavily import TavilyClient

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search."""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


research_subagent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions",
    "system_prompt": "You are a great researcher",
    "tools": [internet_search],
    "model": "openai:gpt-5.5",
}

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    subagents=[research_subagent],
)
```

主智能体会看到一个 `task` 工具，用于委派给可用的同步 subagent。

---

## 默认 General-purpose Subagent

Deep Agents 默认会自动添加一个同步 `general-purpose` subagent，除非你已经提供了同名同步 subagent。

默认 `general-purpose` subagent 的特点：

| 能力 | 说明 |
|------|------|
| 名称 | `general-purpose` |
| 用途 | 通用多步骤任务委派 |
| 文件系统工具 | 默认可用 |
| 模型 | 默认继承主智能体模型 |
| 工具 | 默认访问相同工具 |
| Skills | 当主智能体配置 skills 时自动继承 |
| Prompt | 使用自己的默认 system prompt，并应用 harness profile overlays |

### 替换默认 General-purpose

传入同名 subagent 即可完全替换默认版本：

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[internet_search],
    subagents=[
        {
            "name": "general-purpose",
            "description": "General-purpose agent for research and multi-step tasks",
            "system_prompt": "You are a general-purpose assistant.",
            "tools": [internet_search],
            "model": "openai:gpt-5.5",
        },
    ],
)
```

这表示：

| 项 | 效果 |
|----|------|
| 主 agent | 使用 Gemini |
| general-purpose subagent | 使用 GPT |
| 默认 general-purpose | 不再自动添加 |

### 禁用同步 Subagents

如果希望 agent 没有 `task` 工具，需要同时满足：

1. 在 active harness profile 中设置 `general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)`。
2. `create_deep_agent` 不传任何同步 subagents。

注意：

1. 不要用 `excluded_middleware` 试图移除 `SubAgentMiddleware`。
2. 文档说明 `SubAgentMiddleware` 是必需 scaffolding，列入 excluded 会抛 `ValueError`。
3. Async subagents 不受这个开关影响，它们有自己的 middleware 和工具。

---

## 自定义 SubAgent

字典形式的 `SubAgent` 是最常用配置方式。

字段说明：

| Field | 是否必填 | 说明 | 继承行为 |
|-------|----------|------|----------|
| `name` | 必填 | 子智能体唯一名称；主 agent 用它调用 `task()`；也会出现在 metadata/streaming 中 | 不适用 |
| `description` | 必填 | 描述子智能体能力；主 agent 用它决定是否委派 | 不适用 |
| `system_prompt` | 必填 | 子智能体指令；应包含工具使用和输出格式要求 | 不继承主 agent |
| `tools` | 可选 | 子智能体可用工具 | 默认继承主 agent；指定后完全覆盖 |
| `model` | 可选 | 模型字符串或 LangChain chat model | 默认继承主 agent |
| `middleware` | 可选 | 子智能体额外 middleware | 不继承主 agent；追加到默认 subagent stack |
| `interrupt_on` | 可选 | HITL 配置 | 默认继承主 agent；子配置覆盖 |
| `skills` | 可选 | 子智能体自己的 skills 路径 | 自定义 subagent 默认不继承主 agent skills |
| `response_format` | 可选 | 结构化输出 schema | 不适用 |
| `permissions` | 可选 | 文件系统权限规则 | 默认继承主 agent；指定后完全替换 |

### Description 很关键

主智能体通过 `description` 判断该不该委派，所以描述必须具体、动作化。

推荐：

```python
{
    "name": "research-specialist",
    "description": (
        "Conducts in-depth research on specific topics using web search. "
        "Use when you need detailed information requiring multiple searches."
    ),
}
```

不推荐：

```python
{
    "name": "helper",
    "description": "helps with stuff",
}
```

### System Prompt 应包含什么

自定义 subagent 的 `system_prompt` 不继承主 agent，所以要写完整：

1. 角色和任务边界。
2. 如何拆解任务。
3. 何时使用哪些工具。
4. 输出格式。
5. 返回长度限制。
6. 是否保存原始数据到文件。
7. 是否需要引用来源。

示例：

```python
research_subagent = {
    "name": "research-agent",
    "description": "Conducts in-depth research using web search and synthesizes findings",
    "system_prompt": """You are a thorough researcher. Your job is to:

    1. Break down the research question into searchable queries
    2. Use internet_search to find relevant information
    3. Synthesize findings into a comprehensive but concise summary
    4. Cite sources when making claims

    Output format:
    - Summary (2-3 paragraphs)
    - Key findings (bullet points)
    - Sources (with URLs)

    Keep your response under 500 words to maintain clean context.""",
    "tools": [internet_search],
}
```

---

## CompiledSubAgent

复杂工作流可以用预编译 LangGraph graph 作为 `CompiledSubAgent`。

字段：

| Field | 类型 | 说明 |
|-------|------|------|
| `name` | `str` | 子智能体唯一名称 |
| `description` | `str` | 子智能体能力描述 |
| `runnable` | `Runnable` | 已编译 LangGraph graph，必须先 `.compile()` |

示例：

```python
from deepagents import create_deep_agent, CompiledSubAgent
from langchain.agents import create_agent

custom_graph = create_agent(
    model=your_model,
    tools=specialized_tools,
    prompt="You are a specialized agent for data analysis...",
)

custom_subagent = CompiledSubAgent(
    name="data-analyzer",
    description="Specialized agent for complex data analysis tasks",
    runnable=custom_graph,
)

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[internet_search],
    system_prompt=research_instructions,
    subagents=[custom_subagent],
)
```

注意：

1. 如果自定义 LangGraph graph，需要有 state key `"messages"`。
2. `CompiledSubAgent` 适合已有复杂 LangGraph 流程、专用 agent graph 或复用已有 agent。
3. Declarative `SubAgent` 通常更简单，优先用于普通场景。

---

## Streaming 与 LangSmith Tracing

Deep Agents 支持从 coordinator 和每个 delegated subagent 流式获取更新。

推荐使用：

```python
stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "Research one recent advance in quantum computing."}]},
    version="v3",
)

for name, item in stream.interleave("messages", "subagents"):
    if name == "messages":
        print("[coordinator]", item.text)
    else:
        print(f"[{item.name}] started")
        for message in item.messages:
            print(f"[{item.name}]", message.text)
        print(f"[{item.name}] status: {item.status}")
```

子智能体 handle 暴露：

| 属性 | 说明 |
|------|------|
| `.name` | 子智能体名称 |
| `.messages` | 子智能体消息流 |
| `.tool_calls` | 子智能体工具调用 |
| `.output` | 最终输出 |
| `.status` | 当前状态 |

### LangSmith Tracing

每个 coordinator 或 subagent 运行都会在 metadata 中带上：

```python
{"lc_agent_name": "research-agent"}
```

用途：

| 用途 | 说明 |
|------|------|
| 区分主 agent 与 subagent | 观察不同 agent 的行为 |
| 按 subagent 过滤 traces | 定位某个子智能体的问题 |
| 比较 subagent 表现 | 观察工具调用、token、耗时、错误 |
| 监控生产行为 | 结合 LangSmith tracing / Engine |

### LangSmith UI 过滤

步骤：

1. 打开 LangSmith tracing project。
2. 切到 Runs 视图。
3. 添加 Metadata filter。
4. Key 设置为 `lc_agent_name`。
5. Value 设置为子智能体名称，例如 `research-agent`。

### SDK 过滤

```python
from langsmith import Client

client = Client()

runs = client.list_runs(
    project_name="<your-project>",
    filter='has(metadata, \'{"lc_agent_name": "research-agent"}\')',
)

for run in runs:
    print(run.name, run.start_time, run.status)
```

获取所有 named subagent runs：

```python
runs = client.list_runs(
    project_name="<your-project>",
    filter="has(metadata, 'lc_agent_name')",
)
```

---

## Structured Output

Subagents 支持结构化输出。配置 `response_format` 后，父 agent 收到的是符合 schema 的 JSON，而不是自由文本。

要求：

```text
deepagents>=0.5.3
```

示例：

```python
from pydantic import BaseModel, Field
from deepagents import create_deep_agent


class ResearchFindings(BaseModel):
    """Structured findings from a research task."""
    summary: str = Field(description="Summary of findings")
    confidence: float = Field(description="Confidence score from 0 to 1")
    sources: list[str] = Field(description="List of source URLs")


research_subagent = {
    "name": "researcher",
    "description": "Researches topics and returns structured findings",
    "system_prompt": "Research the given topic thoroughly. Return your findings.",
    "tools": [web_search],
    "response_format": ResearchFindings,
}

agent = create_deep_agent(
    model="claude-sonnet-4-6",
    subagents=[research_subagent],
)
```

父 agent 的 `ToolMessage` 中会收到 JSON 序列化结果：

```json
{
  "summary": "...",
  "confidence": 0.87,
  "sources": ["https://..."]
}
```

适合场景：

| 场景 | 原因 |
|------|------|
| 父 agent 需要程序化处理结果 | JSON 更稳定 |
| 后续工具需要结构化参数 | 可直接解析 |
| 多 subagent 汇总 | schema 一致方便合并 |
| 需要 confidence / sources / scores | 强制结果字段 |

支持的 schema 类型与 LangChain `create_agent` 一致，包括 Pydantic models、`ToolStrategy(...)`、`ProviderStrategy(...)` 或 raw schema type。

---

## Skills 继承与隔离

Skills 在 subagents 中有明确继承规则：

| Agent 类型 | 是否继承主 agent skills |
|------------|-------------------------|
| General-purpose subagent | 是 |
| Custom subagents | 否 |
| 配置了 `skills` 的 custom subagent | 使用自己的 skills |

示例：

```python
research_subagent = {
    "name": "researcher",
    "description": "Research assistant with specialized skills",
    "system_prompt": "You are a researcher.",
    "tools": [web_search],
    "skills": ["/skills/research/", "/skills/web-search/"],
}

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    skills=["/skills/main/"],
    subagents=[research_subagent],
)
```

结果：

| 对象 | 可见 skills |
|------|-------------|
| Main agent | `/skills/main/` |
| General-purpose subagent | `/skills/main/` |
| `researcher` custom subagent | `/skills/research/`, `/skills/web-search/` |

隔离规则：

1. 只有配置了 skills 的 subagent 才会有自己的 `SkillsMiddleware`。
2. Subagent 加载的 skill state 不会传给 parent。
3. Parent 已加载的 skill state 不会传给 custom subagent。
4. Skills 适合按子任务专业化，而不是让所有 agent 共享巨大技能集合。

---

## Context Management

父 agent 调用时传入的 runtime context 会自动传播给所有 subagents。

示例：

```python
from dataclasses import dataclass

from deepagents import create_deep_agent
from langchain.messages import HumanMessage
from langchain.tools import tool, ToolRuntime


@dataclass
class Context:
    user_id: str
    session_id: str


@tool
def get_user_data(query: str, runtime: ToolRuntime[Context]) -> str:
    """Fetch data for the current user."""
    user_id = runtime.context.user_id
    return f"Data for user {user_id}: {query}"


research_subagent = {
    "name": "researcher",
    "description": "Conducts research for the current user",
    "system_prompt": "You are a research assistant.",
    "tools": [get_user_data],
}

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    subagents=[research_subagent],
    context_schema=Context,
)

result = await agent.invoke(
    {"messages": [HumanMessage("Look up my recent activity")]},
    context=Context(user_id="user-123", session_id="abc"),
)
```

### Per-subagent Context

所有 subagents 默认收到同一份 parent context。如果某个 subagent 需要自己的配置，可以：

1. 使用 flat mapping 中的 namespaced keys，例如 `researcher:max_depth`。
2. 在 context schema 中建独立字段，例如 `researcher_max_depth`。

示例：

```python
from dataclasses import dataclass


@dataclass
class Context:
    user_id: str
    researcher_max_depth: int | None = None
    fact_checker_strict_mode: bool | None = None
```

### 判断哪个 Subagent 调用了工具

当 parent 和多个 subagents 共用同一个工具时，可以读取 metadata：

```python
from langchain.tools import tool, ToolRuntime


@tool
def shared_lookup(query: str, runtime: ToolRuntime) -> str:
    """Look up information."""
    agent_name = runtime.config.get("metadata", {}).get("lc_agent_name")
    if agent_name == "fact-checker":
        return strict_lookup(query)
    return general_lookup(query)
```

也可以同时结合 runtime context 和 `lc_agent_name`：

```python
@tool
def flexible_search(query: str, runtime: ToolRuntime[Context]) -> str:
    """Search with agent-specific settings."""
    agent_name = runtime.config.get("metadata", {}).get("lc_agent_name", "unknown")
    ctx = runtime.context
    if agent_name == "researcher":
        max_results = ctx.researcher_max_depth or 5
    else:
        max_results = 5

    return perform_search(query, max_results=max_results, include_raw=False)
```

---

## Async Subagents 概览

Async subagents 允许 supervisor 启动后台任务并立刻继续与用户交互。

特点：

| 能力 | 说明 |
|------|------|
| 非阻塞启动 | 启动后立即返回 task ID |
| 并发执行 | 多个 subagents 可同时工作 |
| 进度查询 | supervisor 可检查任务状态 |
| 中途更新 | 可发送 follow-up instructions |
| 任务取消 | 可取消正在运行的任务 |
| 独立状态 | 每个任务有自己的 thread 状态 |

Preview 状态：

```text
Async subagents 是 deepagents 0.5.0 的 preview feature，API 可能变化。
```

Async subagents 与任何实现 Agent Protocol 的 server 通信，例如：

1. LangSmith Deployments。
2. 自托管 Agent Protocol-compatible server。

每个 async subagent 独立于 supervisor 运行，supervisor 通过 SDK launch、check、update、cancel。

---

## Async Subagents 配置

Async subagents 用 `AsyncSubAgent` spec 定义。

示例：

```python
from deepagents import AsyncSubAgent, create_deep_agent

async_subagents = [
    AsyncSubAgent(
        name="researcher",
        description="Research agent for information gathering and synthesis",
        graph_id="researcher",
    ),
    AsyncSubAgent(
        name="coder",
        description="Coding agent for code generation and review",
        graph_id="coder",
        # url="https://coder-deployment.langsmith.dev"
    ),
]

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    subagents=async_subagents,
)
```

字段：

| Field | 是否必填 | 说明 |
|-------|----------|------|
| `name` | 必填 | 唯一名称；supervisor 用它启动任务 |
| `description` | 必填 | 能力描述；supervisor 用它选择 subagent |
| `graph_id` | 必填 | Agent Protocol server 上的 graph ID 或 assistant ID |
| `url` | 可选 | 不填使用 ASGI in-process；填写后使用 HTTP remote |
| `headers` | 可选 | 远程 server 请求 headers，用于自定义认证 |

LangGraph co-deployed 场景需要在同一个 `langgraph.json` 注册所有 graphs：

```json
{
  "graphs": {
    "supervisor": "./src/supervisor.py:graph",
    "researcher": "./src/researcher.py:graph",
    "coder": "./src/coder.py:graph"
  }
}
```

---

## Async Subagent Tools 与生命周期

配置 async subagents 后，`AsyncSubAgentMiddleware` 会给 supervisor 提供五个工具。

| Tool | 作用 | 返回 |
|------|------|------|
| `start_async_task` | 启动后台任务 | 立即返回 task ID |
| `check_async_task` | 查询任务状态和结果 | 状态 + 完成时的结果 |
| `update_async_task` | 给运行中任务发送新指令 | 确认 + 更新后状态 |
| `cancel_async_task` | 取消运行中任务 | 取消确认 |
| `list_async_tasks` | 列出所有已追踪任务及实时状态 | 任务摘要 |

这些工具由 supervisor 的 LLM 像普通工具一样调用，middleware 负责 thread 创建、run 管理和状态持久化。

### 生命周期

```text
User: Research topic X
Supervisor: start_async_task(researcher, "topic X")
Server: task_id = abc123
Supervisor: 告诉用户任务已启动

User: 进展如何？
Supervisor: check_async_task("abc123")
Server: status + result
Supervisor: 汇报结果
```

细节：

| 操作 | 行为 |
|------|------|
| Launch | 在 server 上创建新 thread，启动 run，并返回 thread ID 作为 task ID |
| Check | 获取 run 当前状态；成功后读取 thread state 提取最终输出 |
| Update | 在同一 thread 上创建新 run，使用 interrupt multitask strategy；旧 run 被中断，subagent 用完整历史 + 新指令重启 |
| Cancel | 调用 `runs.cancel()`，并标记任务为 `cancelled` |
| List | 遍历 tracked tasks；非终态任务并发获取 live status，终态任务从 cache 返回 |

关键约束：

```text
start 后不要立刻循环 check，
否则 async 会退化成 blocking。
```

---

## Async State Management

Async task metadata 存在 supervisor graph 的专用 state channel：

```text
async_tasks
```

它与 message history 分离。

原因：

1. Deep Agents 会在上下文变长时 compact message history。
2. 如果 task ID 只存在 tool messages 中，summarization 后可能丢失。
3. 专用 `async_tasks` channel 能确保 supervisor 即使经历多轮总结，也能通过 `list_async_tasks` 找回任务。

每个 tracked task 记录：

| 字段 | 说明 |
|------|------|
| task ID | 通常是 subagent thread ID |
| agent name | 哪个 async subagent |
| thread ID | 子任务 thread |
| run ID | 当前/最近 run |
| status | 运行状态 |
| `created_at` | 创建时间 |
| `last_checked_at` | 最近检查时间 |
| `last_updated_at` | 最近更新时间 |

---

## Transport 与部署拓扑

Async subagents 支持两类 transport。

### ASGI Transport

当 `AsyncSubAgent` 不设置 `url` 时，使用 ASGI transport。

特点：

| 点 | 说明 |
|----|------|
| 运行方式 | in-process function calls |
| 网络延迟 | 无额外 HTTP 延迟 |
| 认证 | 不需要额外 auth 配置 |
| 状态 | subagent 仍作为独立 thread 运行 |
| 要求 | 对 LangGraph 部署，所有 graph 需在同一个 `langgraph.json` |
| 推荐 | 官方推荐默认起点 |

### HTTP Transport

设置 `url` 后，SDK 通过 HTTP 调用远程 Agent Protocol server。

```python
AsyncSubAgent(
    name="researcher",
    description="Research agent",
    graph_id="researcher",
    url="https://my-research-deployment.langsmith.dev",
)
```

适合场景：

| 场景 | 说明 |
|------|------|
| 独立扩缩容 | subagent 需要单独资源池 |
| 不同资源类型 | 例如 coder 需要更强 CPU/GPU 或更长 timeout |
| 不同团队维护 | supervisor 和 subagent 分属不同服务 |
| 跨部署复用 | 多个 supervisor 共用同一 subagent 服务 |

认证：

1. LangGraph deployments 通常由 SDK 使用 `LANGSMITH_API_KEY` 或 `LANGGRAPH_API_KEY`。
2. Self-hosted Agent Protocol server 可以通过 `headers` 或自定义机制认证。

### 部署拓扑

| 拓扑 | 说明 | 适合 |
|------|------|------|
| Single deployment | supervisor 与所有 subagents 同服，ASGI transport | 起步阶段，低延迟 |
| Split deployment | supervisor 与 subagents 分别部署，HTTP transport | 独立扩缩容、隔离团队 |
| Hybrid | 部分 co-deployed，部分 remote | 混合资源和组织边界 |

Hybrid 示例：

```python
async_subagents = [
    AsyncSubAgent(
        name="researcher",
        description="Research agent",
        graph_id="researcher",
    ),
    AsyncSubAgent(
        name="coder",
        description="Coding agent",
        graph_id="coder",
        url="https://coder-deployment.langsmith.dev",
    ),
]
```

---

## 常见模式

### 多个专业同步 Subagents

```python
subagents = [
    {
        "name": "data-collector",
        "description": "Gathers raw data from various sources",
        "system_prompt": "Collect comprehensive data on the topic",
        "tools": [web_search, api_call, database_query],
    },
    {
        "name": "data-analyzer",
        "description": "Analyzes collected data for insights",
        "system_prompt": "Analyze data and extract key insights",
        "tools": [statistical_analysis],
    },
    {
        "name": "report-writer",
        "description": "Writes polished reports from analysis",
        "system_prompt": "Create professional reports from insights",
        "tools": [format_document],
    },
]

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    system_prompt="You coordinate data analysis and reporting. Use subagents for specialized tasks.",
    subagents=subagents,
)
```

流程：

1. Main agent 创建高层计划。
2. 委派 data-collector 收集数据。
3. 委派 data-analyzer 分析。
4. 委派 report-writer 写报告。
5. Main agent 合成最终输出。

### Sync + Async 混合

可以同时使用：

| 类型 | 用途 |
|------|------|
| 同步 subagent | 当前回答必须依赖的即时子任务 |
| Async subagent | 长时间后台研究、代码生成、批处理 |

例如：

1. 用户要求“先给我一个方案，同时后台跑完整研究”。
2. Supervisor 用同步 general-purpose 快速分析需求。
3. Supervisor 用 async researcher 启动深度研究。
4. 用户继续聊天。
5. 后续通过 task ID 检查研究结果。

### 图片/文档密集任务

适合把重型读取和分析交给 subagent：

```text
主 agent：用户上传一堆 PDF/截图
subagent：逐个 read_file、提取摘要、保存细节
主 agent：只接收结构化结论和文件路径
```

这样可以减少主上下文中的多模态内容和中间工具结果。

---

## 最佳实践

### 同步 Subagents

1. Description 写清“何时使用”，不要泛泛写 helper。
2. System prompt 写清步骤、工具使用、输出格式和长度限制。
3. 子智能体工具集越小越好，只给它需要的工具。
4. 对不同子任务选择不同模型。
5. 让 subagent 返回摘要、结构化结果和必要文件路径，不要返回原始数据。
6. 大数据写入文件系统，主 agent 按需读取。
7. 用 LangSmith 的 `lc_agent_name` 过滤调试单个子智能体。
8. 需要稳定下游处理时使用 `response_format`。

### Async Subagents

1. 长任务才用 async，不要把简单任务都后台化。
2. 启动任务后返回给用户，不要立刻循环 check。
3. 本地开发时增加 worker pool。
4. task ID 必须完整保留，不要截断。
5. 汇报状态前调用 `check_async_task` 或 `list_async_tasks`，不要引用旧消息里的状态。
6. 用 thread ID / task ID 在 LangSmith 中关联 supervisor 和 subagent traces。
7. ASGI co-deployed 作为默认起点；需要独立扩缩容时再用 HTTP remote。

### Worker Pool

本地 `langgraph dev` 时，每个 active run 占用一个 worker slot。

如果 supervisor 同时启动 3 个 async subagents，至少需要：

```text
1 supervisor + 3 subagents = 4 slots
```

可配置：

```bash
langgraph dev --n-jobs-per-worker 10
```

---

## 故障排查

### 同步 Subagent 不被调用

| 可能原因 | 处理 |
|----------|------|
| description 太模糊 | 写清具体能力和适用场景 |
| main agent 没被要求委派 | 在主 system prompt 中强调复杂任务用 `task()` |
| 子任务太简单 | 可能不值得委派，这是正常现象 |
| subagent 名称/描述冲突 | 区分各 subagent 的边界 |

主 prompt 可补充：

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    system_prompt="""...your instructions...

    IMPORTANT: For complex tasks, delegate to your subagents using the task() tool.
    This keeps your context clean and improves results.""",
    subagents=[...],
)
```

### Context 仍然膨胀

| 可能原因 | 处理 |
|----------|------|
| subagent 返回太多原始数据 | 要求只返回 essential summary |
| 工具输出直接塞回主 agent | 大数据写文件，只返回路径和摘要 |
| 主 agent 直接做了太多工具调用 | 强化委派策略 |
| subagent 最终回答太长 | 在 system prompt 中设置字数上限 |

推荐子 agent prompt：

```text
IMPORTANT: Return only the essential summary.
Do NOT include raw data, intermediate search results, or detailed tool outputs.
Your response should be under 500 words.
```

### Wrong Subagent 被选中

| 原因 | 处理 |
|------|------|
| 多个 description 太相似 | 明确区分使用边界 |
| name 太泛 | 使用 `quick-researcher`、`deep-researcher` 等具体名称 |
| 没写任务复杂度 | 在 description 中写 “1-2 searches” 或 “multi-search synthesis” |

示例：

```python
subagents = [
    {
        "name": "quick-researcher",
        "description": "For simple, quick research questions that need 1-2 searches. Use for basic facts or definitions.",
    },
    {
        "name": "deep-researcher",
        "description": "For complex, in-depth research requiring multiple searches, synthesis, and analysis. Use for comprehensive reports.",
    },
]
```

### Async Supervisor 启动后立即轮询

问题：

```text
Supervisor start_async_task 后马上循环 check_async_task，
导致 async 退化成 blocking。
```

处理：

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    system_prompt="""...your instructions...

    After launching an async subagent, ALWAYS return control to the user.
    Never call check_async_task immediately after launch.""",
    subagents=async_subagents,
)
```

### Async Supervisor 汇报旧状态

| 问题 | 处理 |
|------|------|
| 模型引用 conversation history 中旧状态 | 要求汇报前必须 `check` 或 `list` |
| task 状态已变但没刷新 | 调用 `check_async_task(task_id)` |
| 多个 task 混淆 | 用完整 task ID 和 `list_async_tasks` |

### Task ID Lookup 失败

| 可能原因 | 处理 |
|----------|------|
| task ID 被截断 | prompt 中要求永远展示完整 task ID |
| 模型改写 ID | 换模型或强化 system prompt |
| 用户复制了部分 ID | 让 supervisor 用 `list_async_tasks` 查找 |

### Async Subagent 排队不运行

| 可能原因 | 处理 |
|----------|------|
| worker pool 不足 | 提高 `--n-jobs-per-worker` |
| active runs 太多 | 限制并发或取消旧任务 |
| remote server 资源不足 | 分拆部署或扩容 |

---

## 快速参考

### SubAgent 字段速查

| 字段 | 作用 |
|------|------|
| `name` | 唯一名称，供 `task()` 或 metadata 使用 |
| `description` | 主 agent 选择委派目标的依据 |
| `system_prompt` | 子智能体完整行为指令 |
| `tools` | 子智能体工具集 |
| `model` | 子智能体模型 override |
| `middleware` | 子智能体额外 middleware |
| `interrupt_on` | HITL 配置 |
| `skills` | 子智能体专属 skills |
| `response_format` | 结构化输出 |
| `permissions` | 子智能体文件系统权限 |

### AsyncSubAgent 字段速查

| 字段 | 作用 |
|------|------|
| `name` | 唯一名称 |
| `description` | supervisor 选择目标的依据 |
| `graph_id` | Agent Protocol server 上的 graph/assistant ID |
| `url` | 远程 HTTP server URL；不填为 ASGI co-deployed |
| `headers` | 自定义认证 headers |

### Async Tools 速查

| Tool | 何时用 |
|------|--------|
| `start_async_task` | 启动后台任务 |
| `check_async_task` | 用户问进度或需要结果时 |
| `update_async_task` | 用户追加要求、改变方向时 |
| `cancel_async_task` | 用户要求停止或任务不再需要时 |
| `list_async_tasks` | 不确定有哪些任务或需要总体状态时 |

### 选型速查

| 需求 | 推荐 |
|------|------|
| 隔离上下文但要马上拿结果 | 同步 SubAgent |
| 通用复杂任务隔离 | 默认 `general-purpose` |
| 专门领域能力 | 自定义 SubAgent |
| 已有 LangGraph workflow | `CompiledSubAgent` |
| 长时间后台任务 | AsyncSubAgent |
| 需要中途追加指令 | AsyncSubAgent + `update_async_task` |
| 需要取消任务 | AsyncSubAgent + `cancel_async_task` |
| 父 agent 需要 parseable result | `response_format` |
| 子智能体专属 workflow | subagent `skills` |

### 一句话总结

```text
同步 subagents 用于阻塞式委派和上下文隔离；
async subagents 用于后台并发任务、进度查询、中途更新和取消；
description 决定是否会被正确选中；
system_prompt 决定子智能体工作质量；
简洁输出决定主智能体上下文是否干净。
```

---

## 资料来源

- Deep Agents Subagents 文档：附件 `28630eae-caad-4cad-a24e-9e34f2b19ee3/pasted-text.txt`。
- Deep Agents Async subagents 文档：附件 `9cf3d9e6-7a99-40a5-a080-c6b0968c41bd/pasted-text.txt`。
