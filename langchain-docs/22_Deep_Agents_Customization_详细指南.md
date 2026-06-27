# Deep Agents Customization（自定义）详细指南

> 基于官方文档 Customize Deep Agents 整理的中文增强版。本文聚焦 `create_deep_agent` 的主要自定义入口：模型、工具、MCP、系统提示词、Prompt Assembly、Middleware、Interpreters、Subagents、Backends、Sandboxes、Human-in-the-loop、Skills、Memory、Profiles、Structured Output 和高级自定义。

## 目录

1. [概述](#概述)
2. [`create_deep_agent` 参数总览](#create_deep_agent-参数总览)
3. [Model](#model)
4. [Tools 与 MCP](#tools-与-mcp)
5. [System Prompt](#system-prompt)
6. [Prompt Assembly](#prompt-assembly)
7. [Middleware](#middleware)
8. [Interpreters](#interpreters)
9. [Subagents](#subagents)
10. [Backends](#backends)
11. [Sandboxes](#sandboxes)
12. [Human-in-the-loop](#human-in-the-loop)
13. [Skills](#skills)
14. [Memory](#memory)
15. [Profiles](#profiles)
16. [Structured Output](#structured-output)
17. [Advanced](#advanced)
18. [最佳实践](#最佳实践)
19. [快速参考](#快速参考)

---

## 概述

Deep Agents 的核心入口是 `create_deep_agent`。它提供一个 production-ready 的 Agent harness，让你可以围绕具体目标组装模型、工具、文件系统、记忆、技能、子智能体、middleware、人类审批和结构化输出。

最小形态：

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    system_prompt="You are a helpful assistant.",
    tools=[search, fetch_url],
    memory=["./AGENTS.md"],
    skills=["./skills/"],
)
```

可以把 Deep Agents 自定义理解成四层：

```
┌─────────────────────────────────────────────────────────────┐
│               Deep Agents Customization                      │
│                                                             │
│  Behavior                                                    │
│  ├─ model                                                    │
│  ├─ system_prompt                                            │
│  ├─ profiles                                                 │
│  └─ response_format                                          │
│                                                             │
│  Capabilities                                                │
│  ├─ tools / MCP tools                                        │
│  ├─ middleware / interpreters                                │
│  └─ subagents                                                │
│                                                             │
│  Context                                                     │
│  ├─ memory / AGENTS.md                                       │
│  ├─ skills / SKILL.md                                        │
│  └─ backends / sandboxes                                     │
│                                                             │
│  Safety & Runtime                                            │
│  ├─ permissions                                              │
│  ├─ interrupt_on                                             │
│  ├─ checkpointer / store                                     │
│  └─ state_schema / context_schema                            │
└─────────────────────────────────────────────────────────────┘
```

---

## `create_deep_agent` 参数总览

### 常用参数

| 参数 | 作用 |
|------|------|
| `model=` | 指定使用哪个模型 |
| `system_prompt=` | Agent 的自定义行为指令 |
| `tools=` | Agent 可调用的领域工具 |
| `memory=` | 启动时加载的 `AGENTS.md` 文件 |
| `skills=` | 按需加载的 skills 目录 |
| `backend=` | 虚拟文件系统后端，默认是 `StateBackend` |
| `permissions=` | 文件系统路径级读写权限控制 |
| `subagents=` | 可委派任务的自定义子智能体 |
| `middleware=` | 追加到默认 stack 的额外 middleware |
| `interrupt_on=` | 对指定 tool call 触发人工审批 |
| `response_format=` | 结构化输出 schema |
| `state_schema=` | 自定义 graph state schema |
| `context_schema=` | 每次运行的 runtime context schema |
| `checkpointer=` | 持久化 thread 状态，HITL 场景必需 |
| `store=` | 长期存储，常用于 `StoreBackend` |
| `debug=` | 调试开关 |
| `name=` | Agent graph 名称 |
| `cache=` | LangGraph cache |

### 选型口诀

| 想做什么 | 看哪个参数 |
|----------|------------|
| 换模型 | `model` |
| 改 Agent 行为 | `system_prompt` / `profiles` |
| 接入外部 API | `tools` / MCP |
| 增加横切逻辑 | `middleware` |
| 处理复杂子任务 | `subagents` |
| 持久保存文件或记忆 | `backend` / `store` / `memory` |
| 加人类审批 | `interrupt_on` / `checkpointer` |
| 返回 Pydantic 结构 | `response_format` |

---

## Model

`model` 可以传两类值：

1. `provider:model` 字符串。
2. 已初始化的 chat model 实例。

官方推荐用 `provider:model` 字符串快速切换模型，例如：

```python
from deepagents import create_deep_agent

agent = create_deep_agent(model="openai:gpt-5.5")
```

如果需要设置更细模型参数，可以先用 `init_chat_model` 或具体模型类初始化，再传给 `create_deep_agent`。

### 三种模型配置方式

| 方式 | 示例 | 适合场景 |
|------|------|----------|
| 默认参数字符串 | `model="openai:gpt-5.5"` | 快速切换 provider / model |
| `init_chat_model` | `model = init_chat_model(...)` | 需要统一 LangChain 模型初始化方式 |
| 具体模型类 | `ChatOpenAI(model="gpt-5.5")` | 需要 provider 特定参数 |

### 支持的 Provider 示例

| Provider | 安装包 / 依赖 | 示例 |
|----------|---------------|------|
| OpenAI | `langchain[openai]` | `openai:gpt-5.5` |
| Anthropic | `langchain[anthropic]` | `anthropic:claude-sonnet-4-6` |
| Azure OpenAI | `langchain[openai]` | `azure_openai:gpt-5.5` |
| Google Gemini | `langchain[google-genai]` | `google_genai:gemini-3.5-flash` |
| AWS Bedrock | `langchain[aws]` | `bedrock_converse` |
| HuggingFace | `langchain[huggingface]` | `model_provider="huggingface"` |
| OpenRouter / Fireworks / Baseten / Ollama | 对应集成包 | `provider:model-name` |

### 重试与超时

Chat models 会自动重试瞬时 API 失败，通常使用指数退避。要调 `max_retries`、`timeout` 等连接稳定性参数，应参考 LangChain Models 的 connection resilience 配置。

---

## Tools 与 MCP

Deep Agents 有内置工具用于规划、文件管理、subagent spawning；你也可以通过 `tools=` 添加业务工具。

### 自定义工具

工具可以是普通 Python 函数，也可以是 LangChain Tool。

```python
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

agent = create_deep_agent(
    model="openai:gpt-5.5",
    tools=[internet_search],
)
```

工具设计建议：

- 函数名要短且明确。
- docstring 要描述什么时候使用。
- 参数类型尽量明确。
- 对高风险工具配合 `interrupt_on`。
- 对外部 API 工具考虑 rate limit、错误处理和超时。

### MCP Tools

Deep Agents 完整支持 Model Context Protocol，可以从任意 MCP server 加载工具，例如数据库、API、文件系统等。

安装：

```bash
pip install langchain-mcp-adapters
```

示例：

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagents import create_deep_agent

async def main():
    async with MultiServerMCPClient(
        {
            "my_server": {
                "transport": "http",
                "url": "http://localhost:8000/mcp",
            }
        }
    ) as client:
        tools = await client.get_tools()

        agent = create_deep_agent(
            model="openai:gpt-5.5",
            tools=tools,
        )

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Use the MCP server to help me."}]},
            config={"configurable": {"thread_id": "1"}},
        )

asyncio.run(main())
```

MCP 相关高级能力：

- stdio servers。
- OAuth authentication。
- tool filtering。
- stateful sessions。

---

## System Prompt

Deep Agents 自带内置 system prompt。这个内置 prompt 会告诉模型如何使用 Deep Agents harness 提供的能力，例如 planning、virtual filesystem tools、subagents 等。

因此官方建议：

- 不要复制内置 prompt 再手动维护。
- 用 `system_prompt=` 添加你的业务指令。
- 用 Harness Profile 调整模型相关 prompt 行为。

示例：

```python
from deepagents import create_deep_agent

research_instructions = """\
You are an expert researcher. Your job is to conduct \
thorough research, and then write a polished report.
"""

agent = create_deep_agent(
    model="openai:gpt-5.5",
    system_prompt=research_instructions,
)
```

Middleware 也可能向 system prompt 追加说明。例如 filesystem tools 的 middleware 会添加相关工具使用说明。

---

## Prompt Assembly

Deep Agents 会把 system prompt 组装成多个命名部分，保证调用方指令、SDK 默认指令和 profile 特定指令可以稳定共存。

### 四个 Prompt Slot

| 名称 | 来源 | 说明 |
|------|------|------|
| `USER` | `create_deep_agent(system_prompt=...)` | 调用者业务指令 |
| `BASE` | SDK 默认 `BASE_AGENT_PROMPT` | 默认 Deep Agents harness 指令 |
| `CUSTOM` | `HarnessProfile.base_system_prompt` | 替换 `BASE` |
| `SUFFIX` | `HarnessProfile.system_prompt_suffix` | 追加到最后 |

组装顺序固定：

```text
USER -> (BASE or CUSTOM) -> SUFFIX
```

中间用空行连接。

### 两个不变量

| 不变量 | 含义 |
|--------|------|
| `USER` 总是在最前 | 你的业务 persona / 指令优先 |
| `SUFFIX` 总是在最后 | profile 的模型调优提示靠近 conversation history |

### 组合结果

| `system_prompt` | profile `base_system_prompt` | profile `system_prompt_suffix` | 最终 prompt |
|-----------------|------------------------------|--------------------------------|-------------|
| 无 | 无 | 无 | `BASE` |
| 无 | 无 | 有 | `BASE + SUFFIX` |
| 无 | 有 | 无 | `CUSTOM` |
| 无 | 有 | 有 | `CUSTOM + SUFFIX` |
| 有 | 无 | 无 | `USER + BASE` |
| 有 | 无 | 有 | `USER + BASE + SUFFIX` |
| 有 | 有 | 无 | `USER + CUSTOM` |
| 有 | 有 | 有 | `USER + CUSTOM + SUFFIX` |

### `SystemMessage` 的特殊情况

如果传入的是 `SystemMessage` 而不是字符串，Deep Agents 会把右侧 assembly 作为额外文本 content block 追加到消息的 `content_blocks`，并保留调用方已有 block 的 `cache_control` 标记。这对 Anthropic prompt caching breakpoint 很有用。

### Subagent Prompt Assembly

Subagent 也会根据自己的 model 重新解析 profile。

区别：

- subagent 没有 `USER` 段。
- subagent spec 中写的 `system_prompt` 类似 `BASE`。
- profile 的 `base_system_prompt` 可以替换 subagent authored prompt。
- profile 的 `system_prompt_suffix` 会追加在后面。

### General-purpose Subagent Prompt

自动添加的 general-purpose subagent 有额外优先级：

```text
general_purpose_subagent.system_prompt
-> HarnessProfile.base_system_prompt
-> SDK general-purpose default
```

如果同时设置 `base_system_prompt` 和 `general_purpose_subagent.system_prompt`，后者对 general-purpose subagent 优先。

---

## Middleware

Deep Agents 支持 LangChain middleware，包括：

- Deep Agents 内置 middleware。
- LangChain 预构建 middleware。
- provider-specific middleware。
- 自定义 middleware。

通过 `middleware=` 传入的自定义 middleware 会追加到默认 stack 中。

### Default Stack：Main Agent

主 Agent 默认 middleware 顺序：

| 顺序 | Middleware | 作用 |
|------|------------|------|
| 1 | `TodoListMiddleware` | 管理 todo list，组织任务 |
| 2 | `SkillsMiddleware` | 当传 `skills` 时启用，加载 skill metadata |
| 3 | `FilesystemMiddleware` | 文件系统读写、目录导航；包含权限 enforcement |
| 4 | `SubAgentMiddleware` | 创建和协调 subagents |
| 5 | `SummarizationMiddleware` | 压缩过长消息历史 |
| 6 | `PatchToolCallsMiddleware` | 恢复中断或 malformed tool args 时修复 dangling tool calls |
| 7 | `AsyncSubAgentMiddleware` | 配置 async subagents 时启用 |
| 8 | `middleware` 参数 | 用户传入的额外 middleware |
| 9 | Harness profile extras | provider-specific profile middleware |
| 10 | Excluded-tool filtering | 根据 profile 隐藏工具 |
| 11 | Prompt caching | Anthropic / Bedrock prompt caching middleware |
| 12 | `MemoryMiddleware` | 传 `memory` 时启用 |
| 13 | `HumanInTheLoopMiddleware` | 传 `interrupt_on` 时启用 |

### Stack Ordering 重点

- `SkillsMiddleware` 在 main agent 中位于 filesystem 前，使 skill metadata 先可用。
- `PatchToolCallsMiddleware` 在 prompt caching 前运行，保证缓存前缀对应实际发送内容。
- Prompt caching middleware 总是注册，但不支持的模型会 no-op。
- `MemoryMiddleware` 放在 profile extras 和 prompt caching 后，减少 memory 注入更新对缓存前缀的影响。

### Default Stack：Synchronous Subagents

同步 subagents 的 stack 大体类似主 Agent，但有两点区别：

1. Skills 在 subagents 中运行在 `PatchToolCallsMiddleware` 后。
2. Subagent graph 内部没有 `SubAgentMiddleware`，只有父 Agent 暴露 `task` tool。

如果 declarative subagent 设置 `interrupt_on`，会传递给该 subagent 的 `create_agent`，为对应 tool calls 接入 HITL。

### Prebuilt Middleware

LangChain 还提供预构建 middleware，例如：

- retry。
- fallback。
- PII detection。
- 其他 provider / safety / observability 能力。

Deep Agents 还暴露 `create_summarization_tool_middleware`，允许 Agent 在合适时机主动触发 summarization，而不是只按固定 token 间隔压缩。

### Custom Middleware

示例：拦截并记录每次 tool call。

```python
from langchain.agents.middleware import wrap_tool_call
from langchain.tools import tool
from deepagents import create_deep_agent

@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."

@wrap_tool_call
def log_tool_calls(request, handler):
    print(f"[Middleware] Tool call: {request.name}")
    result = handler(request)
    print("[Middleware] Tool call completed")
    return result

agent = create_deep_agent(
    model="openai:gpt-5.5",
    tools=[get_weather],
    middleware=[log_tool_calls],
)
```

### 自定义 Middleware 注意事项

不要在 middleware 实例属性上做并发可变状态，例如 `self.x += 1`。Deep Agents 可能同时运行：

- subagents。
- parallel tools。
- 不同 threads 上的并发 invocation。

如果要跨 hook 保存计数或状态，应使用 graph state。

推荐：

```python
from langchain.agents.middleware import AgentMiddleware

class CustomMiddleware(AgentMiddleware):
    def before_agent(self, state, runtime):
        return {"x": state.get("x", 0) + 1}
```

避免：

```python
class CustomMiddlewareBad(AgentMiddleware):
    def __init__(self):
        self.x = 1

    def before_agent(self, state, runtime):
        self.x += 1
```

---

## Interpreters

Interpreters 通过 `CodeInterpreterMiddleware` 添加 `eval` 工具，运行在 scoped QuickJS runtime 中。

适合场景：

- 让 Agent 用代码组合工具调用。
- 批处理多个任务。
- 用代码处理错误恢复。
- 对结构化数据做确定性转换。
- 不需要完整 shell 环境。

示例：

```python
from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware

agent = create_deep_agent(
    model="openai:gpt-5.5",
    middleware=[CodeInterpreterMiddleware()],
)
```

Interpreter 与 Sandbox 区别：

| 能力 | Interpreter | Sandbox |
|------|-------------|---------|
| JS eval | 支持 | 不一定 |
| Shell 命令 | 不支持 | 支持 |
| 安装依赖 | 不支持 | 支持 |
| 文件系统 / 网络 | 不提供完整 OS 访问 | 取决于 sandbox |
| 适合 | 轻量程序化控制 | 编程、测试、CLI、依赖安装 |

---

## Subagents

Subagents 用于隔离复杂工作，避免主上下文膨胀。

示例：

```python
research_subagent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions",
    "system_prompt": "You are a great researcher",
    "tools": [internet_search],
    "model": "openai:gpt-5.5",  # optional override
}

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    subagents=[research_subagent],
)
```

### Subagent 字段

| 字段 | 说明 |
|------|------|
| `name` | 子智能体名称 |
| `description` | 何时使用该子智能体 |
| `system_prompt` | 子智能体行为指令 |
| `tools` | 子智能体可用工具 |
| `model` | 可选，覆盖主 Agent model |

### 使用建议

- 子任务较长或需要大量上下文时使用 subagent。
- 子任务可并行时使用 subagent。
- 需要专门工具或不同模型时使用自定义 subagent。
- 如果只是简单工具调用，不必拆 subagent。

---

## Backends

Backends 是 Deep Agents 虚拟文件系统的存储与执行后端。默认是 `StateBackend`。

如果使用 `skills` 或 `memory`，需要确保对应 skill / memory 文件在 backend 中可见。

### Backend 类型

| Backend | 说明 | 适合场景 |
|---------|------|----------|
| `StateBackend` | 文件存在线程 state 中 | 默认、本地开发、线程内持久 |
| `FilesystemBackend` | 使用本机文件系统 | 需要直接读写项目文件 |
| `LocalShellBackend` | 本机文件系统 + host shell execution | 本地强力自动化，风险高 |
| `StoreBackend` | 基于 LangGraph store，跨线程持久 | 用户/租户长期记忆 |
| `ContextHubBackend` | LangSmith Hub repo 中的持久文件系统 | LangSmith Context Hub |
| `CompositeBackend` | 按路径路由到不同 backend | 混合临时文件和长期记忆 |

### StateBackend

默认后端，文件存储在线程 state 中。

特点：

- 文件可在同一 thread 的多 turn 中持久。
- 依赖 checkpointer 持久化 thread。
- 不跨 thread 共享。

```python
from deepagents import create_deep_agent
from deepagents.backends import StateBackend

agent = create_deep_agent(
    model="openai:gpt-5.5",
    backend=StateBackend(),
)
```

### FilesystemBackend

使用本地机器文件系统。

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    model="openai:gpt-5.5",
    backend=FilesystemBackend(root_dir=".", virtual_mode=True),
)
```

注意：

- 它赋予 Agent 直接文件读写能力。
- 应谨慎使用，只在合适环境中启用。
- 官方建议必要时用 `CompositeBackend` 包装，避免内部 Agent 数据写入项目目录。

### LocalShellBackend

本地文件系统 + unrestricted shell execution。

```python
from deepagents.backends import LocalShellBackend

agent = create_deep_agent(
    model="openai:gpt-5.5",
    backend=LocalShellBackend(
        root_dir=".",
        virtual_mode=True,
        env={"PATH": "/usr/bin:/bin"},
    ),
)
```

高风险：它允许 Agent 直接读写本机文件并执行 shell 命令，只应在受控环境使用。

### StoreBackend

提供跨 threads 持久化文件系统。

```python
from deepagents import create_deep_agent
from deepagents.backends import StoreBackend
from langgraph.store.memory import InMemoryStore

agent = create_deep_agent(
    model="openai:gpt-5.5",
    backend=StoreBackend(
        namespace=lambda rt: (rt.server_info.user.identity,),
    ),
    store=InMemoryStore(),
)
```

生产建议：

- 多用户部署必须设置 namespace factory，按 user / tenant 隔离数据。
- LangSmith Deployment 中可省略 `store`，平台会自动 provision。

### ContextHubBackend

把持久文件系统放在 LangSmith Hub repo 中。

```python
from deepagents.backends import ContextHubBackend

agent = create_deep_agent(
    model="openai:gpt-5.5",
    backend=ContextHubBackend("my-agent"),
)
```

### CompositeBackend

按路径把文件路由到不同 backend。

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

agent = create_deep_agent(
    model="openai:gpt-5.5",
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(namespace=lambda _rt: ("memories",)),
        },
    ),
    store=InMemoryStore(),
)
```

适合：

- 默认临时文件走 state。
- 长期记忆走 store。
- 项目文件走 filesystem。
- 不同路径有不同权限和生命周期。

---

## Sandboxes

Sandboxes 是特殊 backend，提供隔离环境、独立文件系统和 `execute` 工具。

适合场景：

- Agent 需要写文件。
- 安装依赖。
- 运行命令。
- 跑测试。
- 不希望影响本地机器。

### 支持的 Sandbox Provider

| Provider | 安装包 | 生命周期清理 |
|----------|--------|--------------|
| LangSmith | `langsmith[sandbox]` | `client.delete_sandbox(...)` |
| Daytona | `langchain-daytona` | `sandbox.stop()` |
| E2B | `langchain-e2b` | `e2b_sandbox.kill()` |
| Modal | `langchain-modal` | `modal_sandbox.terminate()` |
| Runloop | `langchain-runloop` | `devbox.shutdown()` |
| Vercel | `langchain-vercel-sandbox` | `sandbox.stop()` |

示例模式：

```python
from deepagents import create_deep_agent
from deepagents.backends import LangSmithSandbox
from langchain_anthropic import ChatAnthropic
from langsmith.sandbox import SandboxClient

client = SandboxClient()
ls_sandbox = client.create_sandbox()
backend = LangSmithSandbox(sandbox=ls_sandbox)

agent = create_deep_agent(
    model=ChatAnthropic(model="claude-sonnet-4-6"),
    system_prompt="You are a Python coding assistant with sandbox access.",
    backend=backend,
)

try:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Create a small Python package and run pytest"}]}
    )
finally:
    client.delete_sandbox(ls_sandbox.name)
```

---

## Human-in-the-loop

某些工具操作敏感，需要执行前由人审批。通过 `interrupt_on` 配置。

### 基本示例

```python
from langchain.tools import tool
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver

@tool
def remove_file(path: str) -> str:
    """Delete a file from the filesystem."""
    return f"Deleted {path}"

@tool
def fetch_file(path: str) -> str:
    """Read a file from the filesystem."""
    return f"Contents of {path}"

@tool
def notify_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Sent email to {to}"

checkpointer = MemorySaver()

agent = create_deep_agent(
    model="openai:gpt-5.5",
    tools=[remove_file, fetch_file, notify_email],
    interrupt_on={
        "remove_file": True,
        "fetch_file": False,
        "notify_email": {"allowed_decisions": ["approve", "reject"]},
    },
    checkpointer=checkpointer,
)
```

### 关键点

| 配置 | 含义 |
|------|------|
| `"remove_file": True` | 默认支持 approve / edit / reject / respond |
| `"fetch_file": False` | 不需要中断 |
| `"notify_email": {"allowed_decisions": ["approve", "reject"]}` | 只允许批准或拒绝，不允许编辑 |

重要：Human-in-the-loop 需要 `checkpointer`，因为中断和恢复依赖持久化 thread 状态。

HITL 可以配置在：

- Agent tool call。
- Subagent tool call。
- tool 内部触发的 interrupt。

---

## Skills

Skills 给 Deep Agent 提供新能力和专业知识。

与 tools 相比：

| 对比 | Tools | Skills |
|------|-------|--------|
| 作用 | 低层动作，如 API、文件、搜索、计划 | 高层任务流程、领域知识、参考资料、模板 |
| 加载方式 | 作为 tool schema 暴露 | Agent 判断有用时按需加载 |
| 内容 | 函数或工具 | `SKILL.md`、脚本、模板、引用资料 |

Skills 使用 progressive disclosure：

- 启动时不把所有内容塞进上下文。
- 只有当 Agent 判断某个 skill 对当前任务有用时才读取。
- 减少启动 token 和上下文负担。

### 使用 Skills

使用 `skills=["/skills/"]` 指向 skill 目录。

如果使用 `StateBackend`，需要在 invocation 的 `files` 中 seed skill 文件：

```python
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.utils import create_file_data
from langgraph.checkpoint.memory import MemorySaver

backend = StateBackend()
checkpointer = MemorySaver()

skills_files = {
    "/skills/langgraph-docs/SKILL.md": create_file_data(skill_content),
}

agent = create_deep_agent(
    model="openai:gpt-5.5",
    backend=backend,
    skills=["/skills/"],
    checkpointer=checkpointer,
)

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "What is langgraph?"}],
        "files": skills_files,
    },
    config={"configurable": {"thread_id": "12345"}},
)
```

如果使用 `StoreBackend`，则应先把 skill 文件写入 store。使用 `FilesystemBackend` 时，直接指定本地 skills 目录。

---

## Memory

Memory 使用 `AGENTS.md` 文件给 Deep Agent 提供额外上下文，例如项目规范、偏好、长期指令。

通过 `memory=[...]` 传入一个或多个路径。

### StateBackend 示例模式

```python
from deepagents import create_deep_agent
from deepagents.backends.utils import create_file_data
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

agent = create_deep_agent(
    model="openai:gpt-5.5",
    memory=["/AGENTS.md"],
    checkpointer=checkpointer,
)

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Please tell me what's in your memory files."}],
        "files": {"/AGENTS.md": create_file_data(agents_md)},
    },
    config={"configurable": {"thread_id": "123456"}},
)
```

### 不同 Backend 下的 Memory

| Backend | Memory 文件来源 |
|---------|-----------------|
| `StateBackend` | 通过 invocation 的 `files` seed 到虚拟路径 |
| `StoreBackend` | 先 `store.put(...)` 写入 store，再通过路径加载 |
| `FilesystemBackend` | 直接读取本地 `AGENTS.md` 路径 |

### Memory 与 Skills 区别

| 对比 | Memory | Skills |
|------|--------|--------|
| 文件标准 | `AGENTS.md` | `SKILL.md` |
| 加载方式 | 启动时加载 | 按需加载 |
| 用途 | 持久偏好、项目规范、长期上下文 | 专项流程、领域知识、模板 |
| 典型内容 | “本项目使用 FastAPI...” | “如何执行发布流程...” |

---

## Profiles

Profiles 用于把“跟模型选择绑定”的配置打包成可复用 bundle。Deep Agents 文档里主要指 Harness Profile。

适合放进 profile 的配置：

- Claude 特定 instruction style。
- GPT 特定 tool descriptions。
- 只对某 provider 注入的 middleware。
- 某模型不适合使用的工具。
- 某模型需要重写的 general-purpose subagent prompt。

示例：

```python
from deepagents import HarnessProfile, register_harness_profile

register_harness_profile(
    "openai:gpt-5.5",
    HarnessProfile(system_prompt_suffix="Respond in under 100 words."),
)
```

Provider profiles 是更窄的 companion API，用来打包模型构造参数，如 API keys、timeouts、retry settings 等。

更完整的 Harness Profiles 内容见：

[20_Deep_Agents_Overview_详细指南.md](/Users/aiyer/Applications/GolandProjects/Langchain/langchain-docs/20_Deep_Agents_Overview_详细指南.md)

---

## Structured Output

Deep Agents 支持结构化输出。通过 `response_format=` 传入 schema，模型生成的结构化数据会被捕获、校验，并返回到 agent state 的 `structured_response` key 中。

### 示例

```python
from pydantic import BaseModel, Field
from deepagents import create_deep_agent

class WeatherReport(BaseModel):
    """A structured weather report with current conditions and forecast."""
    location: str = Field(description="The location for this weather report")
    temperature: float = Field(description="Current temperature in Celsius")
    condition: str = Field(description="Current weather condition")
    humidity: int = Field(description="Humidity percentage")
    wind_speed: float = Field(description="Wind speed in km/h")
    forecast: str = Field(description="Brief forecast for the next 24 hours")

agent = create_deep_agent(
    model=model,
    response_format=WeatherReport,
    tools=[internet_search],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather like in San Francisco?"}]}
)

print(result["structured_response"])
```

适合：

- API 响应。
- UI 渲染。
- 工作流下一步输入。
- 数据抽取。
- 报告结构化。
- 分类和决策结果。

---

## Advanced

`create_deep_agent` 本质上是在 `create_agent` 之上预组装了一套 middleware stack。

如果你想完全控制 harness：

- 自己选择包含哪些能力。
- 自定义 middleware 顺序。
- 替换默认 stack。
- 构建非常特殊的 Agent workflow。

可以参考 LangChain 的 Configure the harness，或者从零构建 deep agent 的指南。

---

## 最佳实践

### 自定义顺序建议

1. 先选模型。
2. 写业务 `system_prompt`。
3. 接入最少必要 tools。
4. 根据需要加 backend。
5. 对危险工具加 `interrupt_on`。
6. 再加 skills / memory。
7. 最后才加自定义 middleware、subagents、profiles。

### 模型

- 快速切换用 `provider:model` 字符串。
- 需要 provider-specific 参数时传 model instance。
- 针对模型行为差异用 Harness Profile，不要在多个调用点复制条件判断。

### 工具

- 工具名和 docstring 要清晰。
- 参数类型要明确。
- 高风险工具必须配合 HITL 或权限。
- 外部 API 工具要处理异常和限流。

### Middleware

- 不要在 middleware 实例属性里存可变运行状态。
- 横切关注点适合 middleware，例如 logging、retry、guardrail、PII。
- 真正业务数据应放 graph state。

### Backends

- 默认用 `StateBackend`。
- 项目文件操作用 `FilesystemBackend`，但要谨慎。
- 需要 shell 执行优先用 sandbox，而不是直接用 `LocalShellBackend`。
- 多用户长期记忆用 `StoreBackend`，并设置 namespace factory。
- 不同路径生命周期不同，用 `CompositeBackend`。

### Skills 与 Memory

- 稳定偏好和项目规范放 `AGENTS.md`。
- 专项任务流程、模板和领域知识放 skill。
- Skills 目录和 Memory 文件必须在 backend 中可见。

### Human-in-the-loop

- 删除、写文件、发邮件、付款、部署等工具应启用审批。
- HITL 必须配置 checkpointer。
- 对简单只读工具可关闭 interrupt。

---

## 快速参考

### 参数速查

| 参数 | 何时使用 |
|------|----------|
| `model` | 指定模型或传入模型实例 |
| `tools` | 添加业务工具或 MCP tools |
| `system_prompt` | 写业务角色和任务要求 |
| `middleware` | 添加横切能力 |
| `subagents` | 添加专门子智能体 |
| `backend` | 控制文件系统和执行环境 |
| `permissions` | 限制文件读写路径 |
| `interrupt_on` | 对敏感工具加人工审批 |
| `skills` | 添加按需加载的专项能力 |
| `memory` | 添加启动时加载的长期上下文 |
| `response_format` | 要求结构化输出 |
| `state_schema` | 扩展 graph state |
| `context_schema` | 注入 runtime context |
| `checkpointer` | 持久化 thread / 支持 HITL |
| `store` | 长期存储 |

### Backend 速查

| Backend | 是否跨 thread | 是否访问本地文件 | 是否 shell |
|---------|---------------|------------------|------------|
| `StateBackend` | 否 | 否 | 否 |
| `FilesystemBackend` | 取决于磁盘 | 是 | 否 |
| `LocalShellBackend` | 取决于磁盘 | 是 | 是 |
| `StoreBackend` | 是 | 否 | 否 |
| `ContextHubBackend` | 是 | 否 | 否 |
| Sandbox backend | 取决于 provider | 隔离文件系统 | 是 |

### 一句话总结

- `create_deep_agent` 是 Deep Agents 的总装配入口。
- `system_prompt` 定义业务行为，内置 prompt 保留 harness 指令。
- tools / MCP 提供行动能力，skills / memory 提供知识和长期上下文。
- backend 决定文件和执行环境，sandbox 用于安全运行命令。
- middleware 适合横切逻辑，subagents 适合隔离复杂子任务。
- `interrupt_on` + `checkpointer` 是高风险操作的人类审批基础。
- `response_format` 让 Agent 输出可验证结构化结果。
