# Deep Agents Models、Tools 与 Context Engineering 详细指南

> 基于 Deep Agents 官方文档中 `Models`、`Tools`、`Multimodal inputs and outputs`、`Context engineering in Deep Agents` 四部分整理。本文重点说明如何选择和配置模型、如何接入工具与 MCP、如何处理图像/音频/视频/文档等多模态输入输出，以及如何设计 Deep Agents 在长任务中的上下文生命周期。

## 目录

1. [整体理解](#整体理解)
2. [Models：模型配置](#models模型配置)
3. [Tools：工具接入](#tools工具接入)
4. [Multimodal：多模态输入与输出](#multimodal多模态输入与输出)
5. [Context Engineering：上下文工程](#context-engineering上下文工程)
6. [四者如何协同](#四者如何协同)
7. [常见配置模式](#常见配置模式)
8. [最佳实践](#最佳实践)
9. [排查清单](#排查清单)
10. [快速参考](#快速参考)
11. [资料来源](#资料来源)

---

## 整体理解

Deep Agents 是一个 agent harness。它不是只负责调用模型，而是把模型、工具、文件系统、子智能体、上下文压缩、记忆和人工审批组织成一个可运行的长任务系统。

这几类能力可以这样理解：

| 主题 | 解决的问题 | 典型配置入口 |
|------|------------|--------------|
| Models | Agent 用哪个 LLM、如何设置模型参数、如何动态切换模型 | `model=`, `ProviderProfile`, middleware |
| Tools | Agent 能做什么动作、能访问哪些外部系统 | `tools=`, MCP tools, built-in harness tools |
| Multimodal | Agent 如何接收和产出图片、音频、视频、PDF/PPT 等非文本内容 | content blocks, `read_file`, custom tool outputs |
| Context Engineering | Agent 在长任务中看见什么、记住什么、压缩什么、隔离什么 | `system_prompt`, `memory`, `skills`, `context_schema`, `state_schema`, backend, subagents |

一个 Deep Agent 通常由下面几层组成：

```text
Deep Agent
├─ Model
│  ├─ provider:model 字符串
│  ├─ 已配置好的 LangChain chat model
│  └─ 运行时动态模型选择
├─ Tools
│  ├─ 自定义 Python 函数
│  ├─ LangChain @tool / StructuredTool
│  ├─ MCP server 暴露的工具
│  └─ 内置 harness tools
├─ Multimodal
│  ├─ 用户消息中的标准 content blocks
│  ├─ read_file 读取图片/音频/视频/PDF/PPT
│  └─ 自定义工具返回多模态 content blocks
└─ Context
   ├─ 输入上下文：system prompt / memory / skills / tool prompts
   ├─ 运行时上下文：user_id / api_key / role / feature flags
   ├─ 可变状态：graph state / checkpoint
   ├─ 压缩机制：offloading / summarization
   ├─ 隔离机制：subagents
   └─ 长期记忆：CompositeBackend + StoreBackend
```

最小示例：

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search, fetch_url, run_query],
    system_prompt="You are a helpful research assistant.",
)
```

---

## Models：模型配置

Deep Agents 支持任何具备 tool calling 能力的 LangChain chat model。模型既可以用 `provider:model` 字符串指定，也可以传入已经实例化和配置好的模型对象。

### 模型字符串格式

官方推荐使用：

```text
provider:model
```

示例：

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
)
```

常见示例：

| Provider | 示例 |
|----------|------|
| Google | `google_genai:gemini-3.5-flash` |
| OpenAI | `openai:gpt-5.5` |
| Anthropic | `anthropic:claude-sonnet-4-6` |
| Baseten | `baseten:zai-org/GLM-5.2` |
| OpenRouter | `openrouter:z-ai/glm-5.1` |
| Fireworks | `fireworks:accounts/fireworks/models/glm-5p1` |
| Ollama | `ollama:minimax-m2.7:cloud` |

含义：

| 部分 | 含义 |
|------|------|
| `provider` | 选择 LangChain 的模型集成 |
| `model` | 传给对应 provider 的真实模型 ID |

注意：不同 provider 对模型 ID 的格式要求不同。有些是简单名字，有些是 namespaced ID，有些是 deployment path。实际使用时应以 provider 的模型目录或集成文档为准。

### 推荐模型与评估含义

官方文档列出了一批在 Deep Agents eval suite 中表现较好的模型。这些 eval 会测试基础 agent 操作，例如文件操作、检索、工具调用、记忆、对话和总结。

重要理解：

| 结论 | 说明 |
|------|------|
| 通过 eval 是必要条件 | 基础 agent 能力至少要稳定 |
| 通过 eval 不是充分条件 | 长任务、复杂工具链、真实业务环境还需要单独评测 |
| 不同模型强项不同 | 有的文件操作强，有的工具调用强，有的总结强 |
| 选型要结合任务 | 不能只看 overall，需要看你的任务主要依赖哪类能力 |

官方建议的模型类别：

| Provider | 示例模型 |
|----------|----------|
| Google | `gemini-3.1-pro-preview`, `gemini-3.5-flash` |
| OpenAI | `gpt-5.5`, `gpt-5.4` |
| Anthropic | `claude-opus-4-8`, `claude-opus-4-7`, `claude-opus-4-6` |
| Open-weight | `GLM-5.2`, `Kimi-K2.7 Code`, `MiniMax-M3` |

开源权重或开放模型通常可以通过 Baseten、Fireworks、OpenRouter、Ollama 等 provider 使用。

### 配置模型参数

如果只传字符串：

```python
agent = create_deep_agent(
    model="openai:gpt-5.5",
)
```

Deep Agents 内部会通过 LangChain 的 `init_chat_model` 解析模型。

如果需要配置 provider 专属参数，可以先创建模型实例：

```python
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

model = init_chat_model(
    model="google_genai:gemini-3.5-flash",
    thinking_level="medium",
)

agent = create_deep_agent(model=model)
```

也可以直接使用 provider 包：

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from deepagents import create_deep_agent

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro-preview",
    thinking_level="medium",
)

agent = create_deep_agent(model=model)
```

适用建议：

| 方式 | 适合场景 |
|------|----------|
| `provider:model` 字符串 | 快速开发、统一配置、配合 ProviderProfile |
| `init_chat_model(...)` | 需要设置模型参数，但仍希望保留 LangChain 统一入口 |
| provider class | 需要完整控制 provider 专属能力 |

### Provider Profiles

`ProviderProfile` 用来给 `provider:model` 字符串形式的模型注入初始化参数。

它只适用于：

```python
create_deep_agent(model="provider:model")
```

不适用于：

```python
model = init_chat_model(...)
create_deep_agent(model=model)
```

因为后者模型已经实例化完成。

`ProviderProfile` 可以注册两个层级：

| 层级 | 示例 | 作用 |
|------|------|------|
| Provider level | `"openai"` | 应用于该 provider 下所有模型 |
| Model level | `"openai:gpt-5.5"` | 只应用于某个具体模型，并覆盖或合并 provider 级配置 |

示例：

```python
from deepagents import ProviderProfile, register_provider_profile

register_provider_profile(
    "openai",
    ProviderProfile(init_kwargs={"temperature": 0}),
)

register_provider_profile(
    "openai:gpt-5.5",
    ProviderProfile(init_kwargs={"reasoning_effort": "medium"}),
)
```

在这个例子中：

| 模型 | 获得的配置 |
|------|------------|
| `openai:gpt-5.4` | `temperature=0` |
| `openai:gpt-5.5` | `temperature=0` + `reasoning_effort="medium"` |

### Provider Profile 与 Harness Profile 的区别

| 类型 | 控制对象 | 典型用途 |
|------|----------|----------|
| Provider Profile | 模型如何被初始化 | temperature、reasoning effort、provider 参数 |
| Harness Profile | Deep Agent harness 如何适配模型 | base system prompt、system prompt suffix、工具描述覆盖、隐藏工具 |

一句话：

```text
ProviderProfile 管模型本身，HarnessProfile 管 agent harness 如何把 prompt 和 tools 呈现给模型。
```

### 运行时动态选择模型

如果应用允许用户在 UI 中选择模型，不一定要重建 agent。可以通过 runtime context + middleware 在每次模型调用前替换模型。

核心思路：

1. 用 `context_schema` 定义运行时上下文结构。
2. 调用 `agent.invoke(..., context=...)` 时传入用户选择。
3. 用 `@wrap_model_call` middleware 读取 `request.runtime.context`。
4. 在 middleware 中创建并替换模型。

示例：

```python
from dataclasses import dataclass
from typing import Callable

from langchain.chat_models import init_chat_model
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from deepagents import create_deep_agent


@dataclass
class Context:
    model: str


@wrap_model_call
def configurable_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    model_name = request.runtime.context.model
    model = init_chat_model(model_name)
    return handler(request.override(model=model))


agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    middleware=[configurable_model],
    context_schema=Context,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Hello!"}]},
    context=Context(model="openai:gpt-5.5"),
)
```

适合场景：

| 场景 | 说明 |
|------|------|
| 用户手动选模型 | UI dropdown / settings |
| 成本优化 | 简单任务用便宜模型，复杂任务用强模型 |
| 按任务路由 | 检索、代码、总结分别用不同模型 |
| 灰度测试 | 对比新模型表现 |

---

## Tools：工具接入

Deep Agents 可以调用三类工具：

1. 你自己定义的工具。
2. LangChain 工具。
3. 任意 MCP server 暴露的工具。

这些工具通过 `tools=` 传给 `create_deep_agent`，并与 Deep Agents 内置 harness tools 一起工作。

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search, fetch_url, run_query],
)
```

### 自定义工具

Deep Agents 支持直接传入：

| 工具形式 | 说明 |
|----------|------|
| 普通 Python 函数 | 根据函数签名和 docstring 推断 schema |
| `@tool` 装饰的函数 | LangChain 标准工具形式 |
| tool dict | 手动提供 schema |
| `StructuredTool` | 更细粒度的 LangChain 工具配置 |

普通函数示例：

```python
import os
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
    """Run a web search."""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[internet_search],
)
```

### 工具描述为什么重要

模型通过工具名、描述、参数 schema 来判断：

| 模型需要知道 | 对应写法 |
|--------------|----------|
| 什么时候用这个工具 | 在 docstring 中写清触发场景 |
| 参数是什么意思 | 为每个参数写清含义 |
| 参数有哪些可选值 | 使用 `Literal` 或清晰描述 |
| 返回值是什么 | 说明返回内容结构 |
| 工具限制是什么 | 写出约束、错误场景、成本或副作用 |

推荐写法：

```python
from langchain.tools import tool


@tool(parse_docstring=True)
def search_orders(
    user_id: str,
    status: str,
    limit: int = 10,
) -> str:
    """Search for user orders by status.

    Use this when the user asks about order history or wants to check
    order status. Always filter by the provided status.

    Args:
        user_id: Unique identifier for the user.
        status: Order status: 'pending', 'shipped', or 'delivered'.
        limit: Maximum number of results to return.
    """
    ...
```

工具描述写得模糊，会导致模型：

| 问题 | 表现 |
|------|------|
| 不知道什么时候调用 | 明明该用工具却直接回答 |
| 参数填错 | 传入错误 ID、枚举值或空参数 |
| 工具误用 | 把查询工具当写入工具 |
| 成本失控 | 频繁调用昂贵 API |
| 上下文膨胀 | 返回大量不必要原始数据 |

### MCP Tools

MCP 即 Model Context Protocol，是用于把 agent 连接到外部服务的开放协议。

MCP 的价值：

| 能力 | 说明 |
|------|------|
| 标准化连接 | 不必为每个服务写一套集成代码 |
| 工具自动发现 | Deep Agents 从 MCP server 获取工具列表 |
| 外部系统接入 | 数据库、API、文件系统、浏览器等 |
| 可组合 | MCP tools 可和自定义工具、内置工具一起使用 |

安装适配器：

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
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Use the MCP server to help me.",
                    }
                ]
            },
            config={"configurable": {"thread_id": "1"}},
        )


asyncio.run(main())
```

MCP 常见配置项包括：

| 配置 | 说明 |
|------|------|
| `transport` | 连接方式，例如 HTTP 或 stdio |
| `url` | HTTP MCP server 地址 |
| stdio command | 本地进程形式的 MCP server |
| OAuth | 需要认证的 MCP 服务 |
| tool filtering | 只暴露部分工具 |
| stateful sessions | 有状态会话 |

### 内置 Harness Tools

除了你传入的工具，每个 Deep Agent 还带有一组内置 harness tools。

| Tool | 作用 |
|------|------|
| `ls` | 列出目录文件 |
| `read_file` | 读取文件内容，支持分页和多模态文件 |
| `write_file` | 创建新文件 |
| `edit_file` | 对文件执行精确字符串替换 |
| `glob` | 用 glob pattern 查找文件 |
| `grep` | 搜索文件内容 |
| `execute` | 执行 shell 命令，仅 sandbox backend 可用 |
| `task` | 启动子智能体处理委派任务 |
| `write_todos` | 管理结构化 todo list |

这些工具不是“附属品”，而是 Deep Agents 长任务能力的核心：

| 工具组 | 对应能力 |
|--------|----------|
| `write_todos` | 任务规划 |
| 文件系统工具 | 上下文外置、产物保存、检索恢复 |
| `execute` | 沙箱中的代码执行和命令运行 |
| `task` | 子智能体委派和上下文隔离 |

### 多模态工具输出

自定义工具可以返回多模态内容，例如图片、音频、视频或文档引用：

| 返回类型 | 场景 |
|----------|------|
| 字符串 | 文本结果 |
| content blocks | 文本、图片、音频、视频、文件等多模态结果 |
| 有序 content block 列表 | 文本和媒体交错输出 |

示例：

```python
from langchain.tools import tool


@tool
def capture_screenshot() -> list[dict]:
    """Capture a screenshot of the current page."""
    return [
        {"type": "text", "text": "Screenshot of the current page:"},
        {"type": "image", "url": "https://example.com/page.png"},
    ]
```

这类返回值会被转换成模型下一轮可读取的 `ToolMessage`。如果需要访问标准化后的内容，可以查看消息上的 `content_blocks`。

注意：

1. 只有所选模型支持多模态 tool results 时，多模态输出才有意义。
2. 大型媒体内容最好保存到 backend 或对象存储，工具只返回简短描述加路径或 URL。
3. 多模态输入与输出的压缩策略和纯文本不同，图片分辨率压缩、视觉 embedding 等不是内置 context compression 的目标。

### 工具设计建议

| 建议 | 原因 |
|------|------|
| 工具粒度适中 | 太粗不好控，太细调用次数多 |
| 名称动词化 | 让模型快速理解动作，例如 `search_orders` |
| 描述包含使用时机 | 模型需要知道何时调用 |
| 参数 schema 明确 | 减少无效调用 |
| 返回结构稳定 | 便于模型解析和后续步骤使用 |
| 控制返回长度 | 避免上下文膨胀 |
| 大结果写文件 | 让模型按需 `read_file` / `grep` |
| 有副作用的工具加审批 | 配合 human-in-the-loop |

---

## Multimodal：多模态输入与输出

Deep Agents 支持多模态工作流，但前提是你选择的模型本身支持对应的多模态输入、tool results 或多模态输出。

多模态能力主要覆盖：

| 能力 | 说明 |
|------|------|
| 多模态用户输入 | 用户消息中包含文本、图片、音频、视频、文档等 content blocks |
| 内置 `read_file` | 从虚拟文件系统读取受支持的非文本文件，并返回标准 content blocks |
| 自定义工具输出 | 工具返回图片、音频、视频、文件等多模态 content blocks |
| 多模态上下文管理 | 长任务中需要用路径、URL、backend 和 subagents 控制上下文膨胀 |

一句话：

```text
Deep Agents 可以传递和读取多模态内容，但多模态理解能力来自底层模型。
```

### 多模态用户输入

调用 agent 时，可以在 `messages` 中使用 LangChain 标准 content blocks。

示例：

```python
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this screenshot?"},
            {"type": "image", "url": "https://example.com/screenshot.png"},
        ],
    }],
})
```

这种方式适合：

| 场景 | 示例 |
|------|------|
| 图片理解 | 截图、图表、照片 |
| 文档理解 | PDF、PPT、报告 |
| 音频理解 | 录音、语音片段 |
| 视频理解 | 短视频、操作录屏 |

注意事项：

1. 不同 provider 支持的 content block 类型、字段和 MIME type 可能不同。
2. 长会话中不建议反复传大体积 base64 内容。
3. 对于长期任务，优先传 URL、文件路径或对象存储引用。

### 内置 `read_file` 读取多模态文件

Deep Agents 的内置 `read_file` 工具不仅能读文本文件，也能对支持的多模态文件返回标准 content blocks。

当模型支持对应模态时，agent 可以通过虚拟文件系统检查图片、文档和媒体文件。

支持的文件类型包括：

| 类型 | 扩展名 |
|------|--------|
| Image | `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.heic`, `.heif` |
| Video | `.mp4`, `.mpeg`, `.mov`, `.avi`, `.flv`, `.mpg`, `.webm`, `.wmv`, `.3gpp` |
| Audio | `.wav`, `.mp3`, `.aiff`, `.aac`, `.ogg`, `.flac` |
| File | `.pdf`, `.ppt`, `.pptx` |

典型流程：

```text
工具或用户把媒体文件保存到 backend
-> agent 使用 read_file 读取文件
-> read_file 返回标准 content blocks
-> 多模态模型理解内容
-> agent 返回文本结论或进一步操作
```

### 自定义工具返回多模态内容

自定义工具可以直接返回多模态 content blocks。

例如截图工具：

```python
from langchain.tools import tool


@tool
def capture_screenshot() -> list[dict]:
    """Capture a screenshot of the current page."""
    return [
        {"type": "text", "text": "Screenshot of the current page:"},
        {"type": "image", "url": "https://example.com/page.png"},
    ]
```

返回内容会被转换成 `ToolMessage`，模型在下一轮读取。对于复杂工具，建议同时返回：

| 内容 | 作用 |
|------|------|
| 简短文本描述 | 让模型快速知道媒体是什么 |
| URL 或文件路径 | 便于后续重新读取 |
| MIME type 或扩展名 | 帮助 provider 正确解析 |
| 必要元数据 | 时间、来源、尺寸、页码等 |

### 多模态与 MCP Tools

MCP tools 也可能返回多模态 tool content。关键仍然是两点：

1. MCP server 返回的内容需要能被 LangChain / Deep Agents 转成标准 content blocks。
2. 底层模型必须支持读取对应的 tool result 模态。

适合 MCP 的多模态场景：

| MCP Server 类型 | 可能返回 |
|-----------------|----------|
| 浏览器 MCP | 截图、页面 PDF、DOM 摘要 |
| 文件系统 MCP | 图片、PDF、PPT、音频文件 |
| 设计工具 MCP | 画板截图、设计稿导出 |
| 数据分析 MCP | 图表、可视化图片、报告文件 |

安全提醒：MCP 不是 sandbox。多模态 MCP tool 是否能访问本地文件、浏览器、对象存储或内部系统，取决于 MCP server 的运行权限和自身限制。

### 多模态内容的上下文压缩限制

Deep Agents 内置 context compression 主要面向文本和消息历史。多模态内容要单独规划。

| 机制 | 对多模态内容的行为 |
|------|--------------------|
| Offloading | 只测量文本 token；非文本 blocks 通常会保留在替换消息里，不会因为图片体积大而单独触发 offloading |
| Summarization | 会把旧消息压缩成文本摘要；旧的图片、音频、视频、文件 blocks 不会被带入摘要后的活跃上下文 |

也就是说：

```text
旧媒体内容被 summarization 覆盖后，
模型只看见总结器写出来的文字，
不会继续看见原始图片/音频/视频/file block。
```

如果旧媒体仍然重要，应确保它们被保存为可重新读取的文件、URL 或对象存储路径。

### Summarization 前后示意

压缩前，旧消息中可能仍包含图片：

```python
[
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What trends do you see in this chart?"},
            {"type": "image", "base64": "...", "mime_type": "image/png"},
        ],
    },
    {
        "role": "tool",
        "content": [
            {"type": "text", "text": "Updated chart:"},
            {"type": "image", "base64": "...", "mime_type": "image/png"},
        ],
    },
]
```

压缩后，活跃上下文只保留文本摘要：

```python
{
    "content": (
        "User asked about trends in a chart screenshot. "
        "Tool returned an updated chart. Agent identified Q3 revenue growth."
    )
}
```

因此，多模态长任务要避免只依赖“历史消息里曾经有图片”。更可靠的做法是：

```text
保存媒体文件 -> 返回路径/URL -> 在需要时 read_file 重新读取
```

### 多模态工作流建议

| 建议 | 原因 |
|------|------|
| 优先保存媒体到 backend 或对象存储 | 避免大媒体长期塞在消息历史里 |
| 消息中优先传 URL / path | 比 base64 更适合长任务 |
| 工具返回描述 + 引用 | 让模型快速理解，同时可恢复细节 |
| 图片密集任务交给 subagents | 主 agent 只接收简洁文本结论 |
| 检查 provider MIME 支持 | 不同模型支持的图片、音频、视频、文档类型不同 |
| 必要时自定义 token counter | 有些 provider 对图片计费或 token 估算较特殊 |
| 不把 compression 当图片压缩器 | 内置 compression 不降分辨率、不做视觉 embedding |

---

## Context Engineering：上下文工程

Context engineering 的目标是：给 agent 提供正确的信息和工具，并以正确格式组织它们，让 agent 在长任务中仍然可靠。

Deep Agents 的上下文不是单一 prompt，而是多个来源的组合：

| Context Type | 你控制什么 | 范围 |
|--------------|------------|------|
| Input context | 启动时进入 prompt 的内容，例如 system prompt、memory、skills | 静态，每次运行应用 |
| Runtime context | invoke 时传入的静态配置，例如用户信息、API key、连接对象 | 每次运行，传播到 subagents |
| Context compression | 内置 offloading 和 summarization | 自动，在接近上下文限制时触发 |
| Context isolation | 用 subagents 隔离重型任务 | 每个 subagent 单独上下文 |
| Long-term memory | 通过虚拟文件系统跨 thread 持久化信息 | 跨会话持久 |

### Input Context

Input context 是 agent 启动时提供的信息，会成为系统提示词的一部分。

主要来源：

| 来源 | 说明 |
|------|------|
| System prompt | 你定义的角色、行为、领域约束 |
| Memory | 总是加载的 `AGENTS.md` 文件 |
| Skills | 按需加载的能力说明，使用 progressive disclosure |
| Tool prompts | 工具 schema、工具描述，以及内置工具的使用说明 |

### System Prompt

`system_prompt` 用于定义 agent 的角色、行为和知识边界。

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    system_prompt=(
        "You are a research assistant specializing in scientific literature. "
        "Always cite sources. Use subagents for parallel research on different topics."
    ),
)
```

关键点：

| 点 | 说明 |
|----|------|
| 静态 | `system_prompt` 不会随每次 invoke 自动变化 |
| 会与内置 prompt 组合 | Deep Agents 会追加规划、文件系统、子智能体等内置指导 |
| 适合放稳定行为 | 角色、输出风格、领域原则 |
| 不适合放 per-run 数据 | 用户 ID、权限、API key 应放 runtime context |

如果 prompt 需要根据 context 或 `runtime.store` 动态变化，可以使用 middleware 中的 `@dynamic_prompt`。工具本身如果只需要读取 context 或 store，则不一定要 middleware，因为工具可以直接接收 `ToolRuntime`。

### Memory

Memory 文件通常是 `AGENTS.md`，它们会被始终注入系统提示词。

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    memory=["/project/AGENTS.md", "~/.deepagents/preferences.md"],
)
```

适合放入 memory 的内容：

| 内容 | 示例 |
|------|------|
| 项目约定 | 代码风格、目录结构、测试命令 |
| 用户偏好 | 输出语言、格式偏好、常用约束 |
| 关键长期规则 | 安全要求、业务边界 |

不适合放入 memory 的内容：

| 内容 | 原因 |
|------|------|
| 大量参考资料 | 每次都加载，浪费上下文 |
| 只在特定任务用的流程 | 应放入 skills |
| 临时运行参数 | 应放入 runtime context |

### Skills

Skills 是按需加载的能力包。Agent 启动时只读取每个 `SKILL.md` 的 frontmatter；当判断某个 skill 相关时，才加载完整内容。

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    skills=["/skills/research/", "/skills/web-search/"],
)
```

Skills 的核心价值是 progressive disclosure：

```text
启动时只看摘要 -> 判断相关性 -> 需要时加载详细说明
```

适合放入 skills 的内容：

| 内容 | 示例 |
|------|------|
| 专门工作流 | 写报告、做代码审查、跑评测 |
| 领域流程 | 金融分析、法律检索、医学文献处理 |
| 可执行脚本说明 | 如何调用 `scripts/` 中的工具 |
| 详细参考资料索引 | 指向 `references/` 中的文件 |

Memory 与 Skills 的区别：

| 维度 | Memory | Skills |
|------|--------|--------|
| 加载时机 | 每次都加载 | 相关时才加载 |
| 内容类型 | 总是相关的背景和规则 | 任务相关的流程和知识 |
| 上下文成本 | 固定成本 | 按需成本 |
| 推荐规模 | 小而关键 | 聚焦、可拆分 |

### Tool Prompts

工具提示词来自两部分：

1. 你传入的工具 schema 和 description。
2. Deep Agents 内置 middleware 对 harness tools 添加的使用说明。

内置 tool prompts 通常包括：

| Prompt | 说明 |
|--------|------|
| Planning prompt | 指导如何用 `write_todos` 维护任务列表 |
| Filesystem prompt | 说明 `ls`、`read_file`、`write_file`、`edit_file`、`glob`、`grep` 的用法 |
| Execute prompt | 使用 sandbox backend 时说明 `execute` |
| Subagent prompt | 说明如何用 `task` 委派 |
| HITL prompt | 设置 `interrupt_on` 时说明如何触发人工审批 |
| Local context prompt | CLI 场景下的当前目录和项目信息 |

对自定义工具而言，工具描述本身就是 prompt 的一部分。它应该告诉模型“什么时候用、怎么用、输入输出是什么”。

### 完整 System Prompt 的组成

Deep Agents 最终给模型的系统消息通常由以下部分组成：

1. 你提供的 custom `system_prompt`。
2. Deep Agents base agent prompt。
3. To-do list prompt。
4. Memory prompt：`AGENTS.md` 和 memory 使用指南。
5. Skills prompt：skills 位置、frontmatter 信息和使用指南。
6. Virtual filesystem prompt：文件系统和必要时的 execute 工具文档。
7. Subagent prompt：`task` 工具使用说明。
8. 用户自定义 middleware 添加的 prompts。
9. Human-in-the-loop prompt：设置 `interrupt_on` 时出现。

可以理解成：

```text
custom system_prompt
+ base agent prompt
+ planning instructions
+ memory instructions
+ skills instructions
+ filesystem instructions
+ subagent instructions
+ custom middleware prompts
+ HITL instructions
= final system message
```

### Runtime Context

Runtime context 是每次调用 agent 时传入的配置。它不会自动进入模型 prompt，除非工具、middleware 或其他逻辑读取它并写入消息或系统提示词。

适合放入 runtime context：

| 内容 | 示例 |
|------|------|
| 用户元数据 | `user_id`, `tenant_id`, `role` |
| 鉴权信息 | API key, access token |
| 外部连接 | DB connection, client object |
| 功能开关 | feature flags |
| 本次运行配置 | 用户选择的模型、语言、权限 |

定义方式：

```python
from dataclasses import dataclass


@dataclass
class Context:
    user_id: str
    api_key: str
```

在工具里读取：

```python
from deepagents import create_deep_agent
from langchain.tools import tool, ToolRuntime


@tool
def fetch_user_data(query: str, runtime: ToolRuntime[Context]) -> str:
    """Fetch data for the current user."""
    user_id = runtime.context.user_id
    return f"Data for user {user_id}: {query}"


agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[fetch_user_data],
    context_schema=Context,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Get my recent activity"}]},
    context=Context(user_id="user-123", api_key="sk-..."),
)
```

重要点：

| 点 | 说明 |
|----|------|
| Runtime context 不自动暴露给模型 | 更安全，避免泄露敏感数据 |
| 工具可直接读取 | 通过 `ToolRuntime` |
| middleware 可读取 | 用于动态 prompt、动态模型、权限控制 |
| 会传播到 subagents | 子智能体默认接收同一份 runtime context |

### Custom State Schema

`state_schema` 用于扩展 agent 的可变 graph state。

适合使用 `state_schema` 的情况：

| 场景 | 原因 |
|------|------|
| 数据需要随 thread checkpoint | state 会被持久化 |
| 数据会在 agent 运行中变化 | context 更适合不可变 per-run 配置 |
| 工具或 middleware 需要访问共享状态 | 可通过 `runtime.state` 读取 |
| 子图需要统一状态字段 | declarative subagents 可继承 parent state schema |

不适合使用 `state_schema` 的情况：

| 场景 | 应使用 |
|------|--------|
| 用户 ID | runtime context |
| API key | runtime context |
| feature flag | runtime context |
| 静态角色配置 | runtime context 或 system prompt |

示例：

```python
from deepagents import DeepAgentState, create_deep_agent
from langchain.tools import ToolRuntime, tool


class ResearchState(DeepAgentState):
    page_url: str
    file_urls: list[str]


@tool
def cite_page(runtime: ToolRuntime) -> str:
    """Return the current page URL."""
    return runtime.state["page_url"]


agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[cite_page],
    state_schema=ResearchState,
)

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Cite the current page"}],
        "page_url": "https://example.com/report",
        "file_urls": [],
    }
)
```

注意：

1. 自定义 state schema 需要继承 `DeepAgentState`。
2. 这样可以保留 Deep Agents 内置的 `messages` reducer 行为。
3. 文档指出该能力需要 `deepagents>=0.6.6`。

### Context Compression

每次 `create_deep_agent` 都包含内置 context compression。你不需要额外添加 middleware，offloading 和 summarization 就会生效。

长任务会产生两类上下文压力：

| 压力来源 | 示例 |
|----------|------|
| 大型工具输入或结果 | 大文件、搜索结果、数据库查询结果 |
| 很长的消息历史 | 多轮规划、工具调用、子任务结果 |

Deep Agents 通过两种机制解决：

| 机制 | 作用 |
|------|------|
| Offloading | 把大型工具输入或结果写入文件系统，用引用替代原文 |
| Summarization | 接近上下文限制时，把旧消息总结成结构化摘要 |

### Offloading

Deep Agents 会使用内置文件系统工具自动 offload 大内容，并在需要时搜索和读取。

默认触发阈值：

```text
工具调用输入或结果超过 20,000 tokens
```

两类 offloading：

| 类型 | 行为 |
|------|------|
| 工具调用输入超过阈值 | 当上下文达到模型窗口约 85% 时，旧工具调用中的大输入会被替换为文件路径引用 |
| 工具调用结果超过阈值 | 结果写入 backend，当前消息只保留文件路径和前 10 行 preview |

好处：

| 好处 | 说明 |
|------|------|
| 保留细节 | 原始内容仍在文件系统中 |
| 降低上下文占用 | prompt 中只放引用和预览 |
| 支持按需恢复 | Agent 可用 `read_file` / `grep` 找回细节 |
| 支持长任务 | 避免每次调用都携带所有历史大内容 |

需要注意：

1. 内置 compression 不会压缩图片分辨率。
2. 不会为图片生成视觉 embedding。
3. Offloading 主要按文本 token 触发，图片等非文本 block 不会仅因二进制体积大而触发。
4. 多模态内容需要用 backend、URL、路径和 subagents 单独设计。

### Summarization

Deep Agents 默认 middleware stack 中包含 `SummarizationMiddleware`。当上下文接近模型窗口限制，且没有更多可 offload 内容时，会自动总结消息历史。

总结机制包括两部分：

| 部分 | 说明 |
|------|------|
| In-context summary | 模型生成结构化摘要，保留 session intent、已创建产物、下一步 |
| Filesystem preservation | 原始消息的文本渲染写入文件系统，作为可检索记录 |

配置特征：

| 项 | 默认行为 |
|----|----------|
| 触发比例 | 模型 `max_input_tokens` 的 85% |
| 最近上下文保留 | 约 10% tokens |
| 无模型 profile 时 fallback | 170,000-token trigger / 保留 6 条消息 |
| 溢出错误处理 | 遇到 `ContextOverflowError` 时立即总结并重试 |
| 总结者 | 使用模型总结旧消息 |

这是一种“双轨”策略：

```text
摘要保留任务意识
文件系统保留原始细节
```

对于多模态内容，还要额外注意：

| 内容 | Summarization 后的表现 |
|------|------------------------|
| 旧图片 block | 不会原样保留到活跃上下文 |
| 旧音频 / 视频 block | 不会原样保留到活跃上下文 |
| 旧 PDF / PPT file block | 不会原样保留到活跃上下文 |
| 总结中的文字描述 | 会保留，质量取决于 summarizer |
| 文件系统中的引用 | 如果原始媒体已保存，可后续重新读取 |

因此，多模态任务中应把“媒体原件”保存到文件系统 backend 或外部对象存储，而不是只放在历史消息里。

### 过滤 Summarization 流式 Token

如果使用 streaming，summarization 步骤生成的 token 也可能出现在流中。可以根据 metadata 过滤：

```python
for chunk in agent.stream(
    {"messages": [...]},
    stream_mode="messages",
    version="v2",
):
    token, metadata = chunk["data"]
    if metadata.get("lc_source") == "summarization":
        continue
    else:
        ...
```

### 按需 Compaction Tool

默认 summarization 会在达到阈值时自动运行。

如果希望 agent 能主动压缩对话，例如一个阶段完成后主动整理上下文，可以添加 `compact_conversation` 工具。

示例：

```python
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware.summarization import (
    create_summarization_tool_middleware,
)

backend = StateBackend
model = "google_genai:gemini-3.5-flash"

agent = create_deep_agent(
    model=model,
    middleware=[
        create_summarization_tool_middleware(model, backend),
    ],
)
```

注意：

1. 添加 compaction tool 不会关闭自动 summarization。
2. 手动 compaction 和自动 summarization 共享同一套总结引擎和状态。
3. 适合阶段性任务、超长会话、复杂研究工作流。

### Context Isolation with Subagents

Subagents 用于解决 context bloat。

当主 agent 直接执行大量搜索、读取文件、查询数据库时，主上下文会快速变大。使用 subagents 可以把这些重型工作隔离出去。

工作机制：

1. 主 agent 通过 `task` 工具委派任务。
2. Subagent 在自己的新上下文中运行。
3. Subagent 自主调用工具直到完成。
4. Subagent 只把最终报告返回给主 agent。
5. 主 agent 的上下文保持干净。

适用场景：

| 场景 | 为什么适合 subagent |
|------|--------------------|
| 多步研究 | 搜索和网页读取会产生大量中间结果 |
| 多主题并行 | 每个主题可独立处理 |
| 大文件分析 | 子智能体读取和筛选，主智能体只看结论 |
| 数据库探索 | 子智能体执行多次查询并总结 |
| 代码库扫描 | 子智能体负责局部调查 |

推荐配置：

```python
research_subagent = {
    "name": "researcher",
    "description": "Conducts research on a topic",
    "system_prompt": """You are a research assistant.
    IMPORTANT: Return only the essential summary (under 500 words).
    Do NOT include raw search results or detailed tool outputs.""",
    "tools": [web_search],
}
```

核心原则：

| 原则 | 说明 |
|------|------|
| 主 agent 负责规划和合成 | 不要让主上下文塞满中间细节 |
| subagent 负责重型探索 | 搜索、读取、查询、试错 |
| subagent 返回摘要 | 不返回原始搜索列表或完整日志 |
| 大结果写文件 | 主 agent 按需读取 |

### Long-term Memory

默认文件系统中的工作记忆通常只在单个 thread 内持久。Long-term memory 用于跨 thread、跨对话保存信息。

典型用途：

| 类型 | 示例 |
|------|------|
| 用户偏好 | 输出语言、格式、偏好的工具 |
| 累积知识 | 某项目背景、业务规则 |
| 研究进展 | 已读文献、已完成实验 |
| 长期任务状态 | 多次会话持续推进的工作 |

实现方式：

1. 使用 `CompositeBackend`。
2. 默认路径仍走 `StateBackend`。
3. 指定 `/memories/` 等路径走 `StoreBackend`。
4. 提供 LangGraph Store 作为持久存储。
5. 在 system prompt 中告诉 agent 何时保存、保存到哪里。

示例：

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore


def make_backend(runtime):
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={"/memories/": StoreBackend(runtime)},
    )


agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    store=InMemoryStore(),
    backend=make_backend,
    system_prompt="""When users tell you their preferences, save them to
    /memories/user_preferences.txt so you remember them in future conversations.""",
)
```

注意：

1. 不需要预先创建 `/memories/` 文件。
2. Agent 会根据 prompt 指示用 `write_file` / `edit_file` 创建和更新。
3. 部署到 LangSmith 时，可通过 Store API 预置 memory。

---

## 四者如何协同

Models、Tools、Multimodal、Context Engineering 不是独立配置项，而是互相影响。

### 模型影响工具表现

即使工具完全相同，不同模型的 tool calling 稳定性也不同。

| 模型能力 | 影响 |
|----------|------|
| tool calling | 是否能正确选择工具和填参数 |
| 长上下文 | 能否承受长任务历史 |
| 总结能力 | summarization 质量 |
| 多模态能力 | 是否能处理多模态 tool outputs |
| 指令遵循 | 是否遵守 tool prompt、system prompt、HITL 规则 |

### 多模态能力影响工具设计

如果模型不支持某类多模态输入，工具返回对应 content block 也没有实际价值。

| 模型能力 | 工具设计建议 |
|----------|--------------|
| 支持图片理解 | 工具可返回 screenshot、chart、image URL |
| 支持 PDF / 文件理解 | 工具可返回文档 file block 或路径 |
| 支持音频 | 工具可返回录音或音频 URL |
| 不支持多模态 | 工具应先转换为文本摘要、OCR、转写或结构化数据 |

### 多模态影响上下文策略

| 多模态内容 | 风险 | 建议 |
|------------|------|------|
| base64 图片 | 消息历史变大，summary 后原图丢失 | 存文件，传路径或 URL |
| 视频 / 音频 | provider 支持差异大，成本不可控 | 先切片、转写、抽帧或摘要 |
| PDF / PPT | 页数多，内容混合文本和图像 | 按页处理，保存中间结果 |
| 图表截图 | summary 后只剩文字描述 | 保存原图和分析结论 |

### 工具影响上下文压力

工具返回越大，context engineering 越重要。

| 工具类型 | 上下文风险 | 建议 |
|----------|------------|------|
| 搜索工具 | 返回大量结果 | 返回摘要或写文件 |
| 数据库查询 | 表格很大 | 限制行数，分页，保存 CSV |
| 文件读取 | 大文件塞满上下文 | 用 `grep` / 分页读取 |
| 浏览器工具 | 页面内容很长 | 提取关键段落 |
| 代码执行 | 日志很长 | 截断输出，保存完整日志 |
| 截图工具 | 图片不能被文本 offloading 压缩 | 返回 URL/path 和简短说明 |

### 上下文设计影响模型和工具成本

| 设计 | 效果 |
|------|------|
| memory 过大 | 每次调用都贵、容易稀释指令 |
| skills 聚焦 | 降低启动上下文，按需加载 |
| subagent 隔离 | 主 agent 保持清爽 |
| offloading | 保留细节但不占 prompt |
| summarization | 长会话可继续推进 |
| runtime context | 让工具拿到必要配置但不泄露给模型 |

---

## 常见配置模式

### 模式一：基础工具型 Agent

适合简单业务助手。

```python
agent = create_deep_agent(
    model="openai:gpt-5.5",
    tools=[search_orders, fetch_user_profile],
    system_prompt="You are a customer support assistant.",
)
```

特点：

| 项 | 配置 |
|----|------|
| 模型 | 固定模型 |
| 工具 | 少量业务 API |
| 上下文 | 简短 system prompt |
| 适合 | 简单问答、业务查询 |

### 模式二：带 Runtime Context 的多租户 Agent

适合 SaaS、内部系统、用户隔离场景。

```python
from dataclasses import dataclass


@dataclass
class Context:
    tenant_id: str
    user_id: str
    role: str


agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[query_account_data],
    context_schema=Context,
)
```

特点：

| 项 | 配置 |
|----|------|
| 用户数据 | 放 runtime context |
| 权限控制 | 工具读取 role / tenant_id |
| 模型可见性 | 默认不可见，除非主动注入 |
| 适合 | 多用户、多租户、权限敏感系统 |

### 模式三：研究型 Agent

适合长时间搜索、阅读、合成。

```python
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[web_search, fetch_url],
    system_prompt=(
        "You are a research assistant. Use subagents for parallel research. "
        "Write large notes to files and return concise synthesized conclusions."
    ),
    subagents=[research_subagent],
)
```

特点：

| 项 | 配置 |
|----|------|
| 工具 | 搜索、网页抓取 |
| 子智能体 | 并行研究和上下文隔离 |
| 文件系统 | 保存原文、笔记、报告 |
| 压缩 | 依赖 offloading 和 summarization |

### 模式四：可动态切换模型的 Agent

适合产品化 UI 或成本优化。

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    middleware=[configurable_model],
    context_schema=Context,
)
```

特点：

| 项 | 配置 |
|----|------|
| 默认模型 | 便宜或稳定模型 |
| 动态选择 | middleware 替换 |
| 用户输入 | `context=Context(model="...")` |
| 适合 | 模型选择器、A/B test、智能路由 |

### 模式五：长期记忆 Agent

适合跨会话持续个性化。

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    store=InMemoryStore(),
    backend=make_backend,
    system_prompt=(
        "When users share durable preferences, save them under /memories/."
    ),
)
```

特点：

| 项 | 配置 |
|----|------|
| 短期文件 | `StateBackend` |
| 长期记忆 | `/memories/` -> `StoreBackend` |
| 指令 | 告诉 agent 保存什么和保存到哪里 |
| 适合 | 用户偏好、长期项目、持续研究 |

---

## 最佳实践

### 模型选择

1. 先确认模型支持 tool calling。
2. 不只看 overall eval，要看你的任务依赖的能力项。
3. 长任务优先关注工具调用、总结、长上下文和指令遵循。
4. 对成本敏感时，用 middleware 做动态路由。
5. 对 provider 参数统一管理时，用 `ProviderProfile`。
6. 对 prompt/tool 呈现适配时，用 `HarnessProfile`。

### 工具设计

1. 工具名清晰，尽量表达动作和对象。
2. docstring 写清“什么时候用”。
3. 参数类型和可选值尽量结构化。
4. 返回值控制长度，大结果写文件。
5. 有副作用或高风险工具配合 HITL。
6. MCP tools 接入后考虑过滤不必要工具，避免工具列表过大。
7. 工具错误信息要可理解，方便模型恢复。

### 多模态设计

1. 先确认模型是否支持目标模态和对应 MIME type。
2. 长任务中优先传 URL 或文件路径，不要反复传 base64。
3. 工具产出图片、音频、视频或文档时，优先保存到 backend，然后返回简短说明和引用。
4. 对图片密集或文档密集任务使用 subagents，主 agent 只接收文本结论。
5. 不要假设 summarization 后还能看到旧媒体 block。
6. 对 PDF/PPT 这类复杂文件，最好按页或按章节处理，并保存中间分析。
7. 如果 provider 对图片计费或 token 估算特殊，考虑自定义 token counter 或调整总结阈值。

### 上下文设计

1. Memory 只放总是相关的内容。
2. Skills 放按需使用的工作流和领域知识。
3. Runtime context 放每次运行的用户、权限和连接配置。
4. State schema 放需要 checkpoint 的可变状态。
5. 大型中间结果交给文件系统和 offloading。
6. 重型探索交给 subagents。
7. 长期记忆用 `CompositeBackend` 显式划分路径。

### Subagent 输出控制

Subagent 最容易导致主 agent 上下文被污染，所以要在 subagent prompt 里明确：

```text
Return only the essential summary.
Do not include raw search results.
Write large artifacts to files.
Keep final response under N words.
Include file paths for details.
```

### Memory 与 Skills 的边界

判断口诀：

```text
每次都必须知道 -> memory
只在某类任务需要 -> skill
每次调用传入 -> runtime context
运行中会变化且需 checkpoint -> state_schema
跨会话保存 -> long-term memory backend
```

---

## 排查清单

### 模型相关

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| 工具调用不稳定 | 模型 tool calling 能力弱 | 换模型或优化工具描述 |
| 参数经常填错 | schema/docstring 不清楚 | 加类型、枚举和参数说明 |
| 长任务中断 | 上下文窗口不足或总结质量差 | 检查 summarization、减少 memory、用 subagents |
| 动态模型不生效 | middleware 没读取 runtime context | 检查 `context_schema` 和 `request.runtime.context` |
| ProviderProfile 不生效 | 传入的是已实例化模型 | 改用字符串或直接在模型实例中配置 |

### 工具相关

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| Agent 不调用工具 | 工具描述没写使用时机 | 在 docstring 中明确 “Use this when...” |
| 工具调用过多 | 工具粒度过细或 prompt 缺少约束 | 合并工具或加调用策略 |
| 返回内容太长 | 工具返回原始数据 | 截断、分页、写文件 |
| MCP 工具太多 | MCP server 暴露范围过大 | tool filtering |
| 高风险操作未审批 | 没配置 HITL | 对相关 tool 设置 `interrupt_on` |

### 多模态相关

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| 图片传了但模型看不懂 | 模型不支持图片或 MIME type 不支持 | 更换多模态模型，检查 provider 文档 |
| 长会话后看不到旧图 | summarization 后媒体 block 不保留 | 保存图片到 backend，返回路径并重新 `read_file` |
| 消息历史迅速变大 | 反复传 base64 媒体 | 改用 URL / 文件路径 / 对象存储 |
| 工具返回图片无效 | tool result 格式不是标准 content blocks | 改成标准 content block 结构 |
| 多模态任务成本过高 | 图片/视频 token 或计费高 | 抽帧、压缩业务输入、用 subagent 隔离 |

### 上下文相关

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| 回答忽略项目约定 | memory 没配置或内容太长被稀释 | 精简并检查 `AGENTS.md` |
| Agent 看不到用户 ID | runtime context 不自动进 prompt | 工具读取 `ToolRuntime` 或 middleware 注入 |
| 对话越来越慢 | memory / messages / tool results 太大 | 使用 skills、subagents、offloading |
| 总结后丢细节 | 原始内容没写文件或不可检索 | 大内容写入文件，返回路径 |
| 跨会话记不住 | 只用了默认 StateBackend | 配置 `CompositeBackend` + `StoreBackend` |

---

## 快速参考

### Models 速查

| 需求 | 推荐做法 |
|------|----------|
| 快速指定模型 | `model="provider:model"` |
| 配置 provider 参数 | `init_chat_model(...)` 或 provider class |
| 全局 provider 默认值 | `ProviderProfile("provider")` |
| 单模型覆盖 | `ProviderProfile("provider:model")` |
| 运行时切换模型 | `context_schema` + `@wrap_model_call` |
| agent 行为适配模型 | Harness Profile |

### Tools 速查

| 需求 | 推荐做法 |
|------|----------|
| 接入普通函数 | 直接放入 `tools=[fn]` |
| 更清晰 schema | 使用 `@tool(parse_docstring=True)` |
| 接入外部服务生态 | MCP + `MultiServerMCPClient` |
| 文件操作 | 使用内置 filesystem tools |
| 代码执行 | 使用 sandbox backend 下的 `execute` |
| 子任务委派 | 使用内置 `task` |
| 任务规划 | 使用内置 `write_todos` |
| 多模态返回 | 返回 content blocks |

### Multimodal 速查

| 需求 | 推荐做法 |
|------|----------|
| 用户上传图片 | 在 `messages.content` 中传 image content block |
| 用户上传 PDF/PPT | 传 file block 或先保存到 backend 后 `read_file` |
| 工具返回截图 | 返回文本说明 + image URL/path content block |
| 长任务保留媒体 | 保存到 backend / object store，消息里放引用 |
| 图片密集分析 | 用 subagents 分析，主 agent 接收摘要 |
| summary 后恢复细节 | 用保存的 path/URL 重新 `read_file` |
| 模型不支持多模态 | 先 OCR、转写、抽帧或转换成文本 |

### Context 速查

| 数据类型 | 放在哪里 |
|----------|----------|
| 稳定角色和行为 | `system_prompt` |
| 总是相关的项目规则 | `memory=["AGENTS.md"]` |
| 按需工作流 | `skills=[...]` |
| 用户 ID / API key / 权限 | `runtime context` |
| 运行中可变且需 checkpoint 的数据 | `state_schema` |
| 大型工具输出 | 文件系统 / offloading |
| 旧消息历史 | summarization |
| 重型探索过程 | subagents |
| 跨会话记忆 | `CompositeBackend` + `StoreBackend` |

### 一句话总结

```text
Models 决定 agent 的推理与工具调用能力；
Tools 决定 agent 能对外部世界做什么；
Multimodal 决定 agent 能否处理图片、音频、视频和文档；
Context engineering 决定 agent 在长任务中看见什么、保留什么、压缩什么、隔离什么。
```

---

## 资料来源

- Deep Agents Models 文档：附件 `a5c098d5-6e53-48c9-ab89-1f1b8886c237/pasted-text.txt` 与 `35f075ee-276c-4ddb-ac9c-a5bbb51b72b0/pasted-text.txt`，两份内容相同。
- Deep Agents Context Engineering 文档：附件 `d9ba7a72-6455-410c-a5a3-029044a5517a/pasted-text.txt`。
- Deep Agents Multimodal inputs and outputs 文档：附件 `e46e3d6b-ff70-40e6-9ed2-70c70d9ae34a/pasted-text.txt`。
- Deep Agents Tools 官方页面：https://docs.langchain.com/oss/python/deepagents/tools
