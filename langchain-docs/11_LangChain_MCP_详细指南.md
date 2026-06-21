# LangChain MCP (Model Context Protocol) 详细指南

> 本文基于 `langchain-mcp-adapters` 与 LangChain v1.x 官方 MCP 文档整理，面向需要在 LangChain Agent 中接入本地或远程 MCP 服务器的 Python 开发者。

## 目录

1. [概述](#概述)
2. [核心概念](#核心概念)
3. [快速开始](#快速开始)
4. [传输机制](#传输机制)
5. [工具 Tools](#工具-tools)
6. [资源 Resources](#资源-resources)
7. [提示词 Prompts](#提示词-prompts)
8. [会话管理](#会话管理)
9. [工具拦截器](#工具拦截器)
10. [回调系统](#回调系统)
11. [创建 MCP 服务器](#创建-mcp-服务器)
12. [实战案例](#实战案例)
13. [最佳实践](#最佳实践)
14. [快速参考](#快速参考)

---

## 概述

### 什么是 MCP

**MCP (Model Context Protocol)** 是一个开放协议，用于标准化应用程序如何向 LLM 提供**工具**与**上下文**。它把外部能力抽象为统一的协议层，使 Agent 能够以一致的方式：

- 发现并调用外部工具（Tools）
- 读取外部数据资源（Resources）
- 获取可复用的提示词模板（Prompts）

```text
┌─────────────────────────────────────────────────────────────┐
│                         LangChain Agent                     │
│                 create_agent(model, tools)                  │
└───────────────────────────────┬─────────────────────────────┘
                                │ LangChain Tool / Blob / Message
┌───────────────────────────────▼─────────────────────────────┐
│                 langchain-mcp-adapters                       │
│        MultiServerMCPClient / load_mcp_tools 等              │
└───────────────────────────────┬─────────────────────────────┘
                                │ MCP ClientSession
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼───────┐       ┌───────▼───────┐       ┌───────▼───────┐
│ MCP Server A  │       │ MCP Server B  │       │ MCP Server C  │
│ stdio / local │       │ HTTP / remote │       │ SSE / legacy  │
└───────┬───────┘       └───────┬───────┘       └───────┬───────┘
        │                       │                       │
  Tools / Resources / Prompts   Tools / Resources / Prompts
```

### 为什么使用 MCP

| 优势 | 说明 |
|------|------|
| **标准化** | 一套协议暴露工具、资源和提示词，降低集成成本 |
| **解耦** | Agent 与外部服务分离，MCP Server 可独立开发、部署、复用 |
| **多服务聚合** | 一个 LangChain Agent 可同时使用多个 MCP Server 的能力 |
| **生态兼容** | MCP Server 可被不同客户端复用，不限于 LangChain |
| **运行时可控** | 可通过拦截器、回调和会话管理控制工具执行流程 |

### LangChain 的 MCP 支持

LangChain 通过 `langchain-mcp-adapters` 接入 MCP：

```bash
pip install langchain-mcp-adapters
```

核心能力：

- `MultiServerMCPClient`：连接一个或多个 MCP 服务器
- `client.get_tools()`：把 MCP Tools 转换为 LangChain Tools
- `client.get_resources()`：把 MCP Resources 转换为 LangChain `Blob`
- `client.get_prompt()`：把 MCP Prompts 转换为 LangChain messages
- `client.session(server_name)`：显式管理 MCP `ClientSession` 生命周期
- `tool_interceptors`：在 MCP 工具执行前后访问运行时上下文、修改请求、处理结果
- `callbacks`：处理进度、日志、征询（elicitation）等协议通知

> 注意：`MultiServerMCPClient` **默认是无状态的**。每次工具调用会创建新的 MCP `ClientSession`，执行工具后清理。需要跨工具调用保持状态时，应显式使用 `client.session(...)`。

---

## 核心概念

### MCP 三大组件

| 组件 | LangChain 中的对应形态 | 典型用途 |
|------|------------------------|----------|
| **Tools** | LangChain Tools | 查询数据库、调用 API、执行本地脚本、操作浏览器等 |
| **Resources** | `Blob` | 暴露文件、数据库记录、API 响应、二进制数据等可读内容 |
| **Prompts** | LangChain messages | 复用摘要、代码审查、问答等提示词模板 |

### 一次工具调用的基本流程

```text
用户输入
  ↓
LangChain Agent 决定调用某个 LangChain Tool
  ↓
Tool 由 langchain-mcp-adapters 包装
  ↓
适配器打开 MCP ClientSession
  ↓
MCP Server 执行工具并返回结果
  ↓
适配器转换为 ToolMessage / artifact / content_blocks
  ↓
Agent 继续推理或返回最终答案
```

### 默认无状态与显式有状态

- **默认无状态**：适合独立查询、幂等工具、无需共享上下文的调用。
- **显式有状态**：适合登录态、事务、长期任务、需要复用服务器端会话状态的工具。

---

## 快速开始

### 1. 安装依赖

```bash
pip install langchain-mcp-adapters fastmcp
```

如果使用 `uv`：

```bash
uv add langchain-mcp-adapters fastmcp
```

### 2. 创建本地 MCP Server

```python
# math_server.py
from fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### 3. 在 LangChain Agent 中使用

```python
import asyncio

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",
                "command": "python",
                # 生产中建议使用绝对路径，避免工作目录变化导致找不到文件
                "args": ["/absolute/path/to/math_server.py"],
            }
        }
    )

    tools = await client.get_tools()
    agent = create_agent("claude-opus-4-8", tools)

    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    )
    print(response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
```

### 4. 同时连接多个 MCP Server

```python
client = MultiServerMCPClient(
    {
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": ["/absolute/path/to/math_server.py"],
        },
        "weather": {
            "transport": "http",
            "url": "http://localhost:8000/mcp",
        },
    }
)

tools = await client.get_tools()
agent = create_agent("claude-opus-4-8", tools)

response = await agent.ainvoke(
    {
        "messages": [
            {"role": "user", "content": "what is the weather in nyc? and what's 123 * 456?"}
        ]
    }
)
```

> 模型字符串由 LangChain 的模型集成解析。示例使用当前 Claude Opus 模型；如果项目使用 OpenAI、Google 或其他供应商，也可以替换为对应的 LangChain 模型标识或模型实例。

---

## 传输机制

MCP 支持多种客户端-服务器通信方式。`langchain-mcp-adapters` 常用配置如下。

### stdio

客户端启动本地子进程，通过标准输入/输出通信。适合本地工具、开发环境、桌面自动化和文件系统类能力。

```python
client = MultiServerMCPClient(
    {
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": ["/absolute/path/to/math_server.py"],
        }
    }
)
```

特点：

- 简单，无需单独部署 HTTP 服务
- 子进程生命周期由客户端会话管理
- stdio 连接本身天然有状态，但在 `MultiServerMCPClient` 默认模式下，每次工具调用仍会创建新 session；需要持久连接时使用显式会话

### HTTP / streamable HTTP

HTTP 传输适合远程 MCP 服务。官方文档中 `transport="http"` 也对应 MCP 的 streamable HTTP 传输。

```python
client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "http",
            "url": "http://localhost:8000/mcp",
        }
    }
)
```

#### 传递请求头

HTTP/SSE 场景可以通过 `headers` 传递鉴权、租户、追踪等信息：

```python
client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "http",
            "url": "http://localhost:8000/mcp",
            "headers": {
                "Authorization": "Bearer YOUR_TOKEN",
                "X-Custom-Header": "custom-value",
            },
        }
    }
)
```

#### 自定义认证

`langchain-mcp-adapters` 底层使用官方 MCP Python SDK；HTTP 认证可以通过实现 `httpx.Auth` 并在连接配置中传入 `auth`：

```python
client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "http",
            "url": "http://localhost:8000/mcp",
            "auth": auth,
        }
    }
)
```

### SSE

SSE 是 MCP 早期常见传输方式，当前 MCP 规范中已更推荐 streamable HTTP。维护旧服务时仍可能遇到：

```python
client = MultiServerMCPClient(
    {
        "legacy_service": {
            "transport": "sse",
            "url": "http://localhost:8000/sse",
        }
    }
)
```

### 选择建议

| 场景 | 推荐传输 | 说明 |
|------|----------|------|
| 本地开发、本地工具 | `stdio` | 最简单，无需部署服务 |
| 远程服务、生产环境 | `http` | 易扩展，便于鉴权和观测 |
| 历史 SSE 服务 | `sse` | 兼容旧实现；新项目优先考虑 HTTP |
| 需要跨调用状态 | 显式 `client.session(...)` | 传输方式之外还要管理会话生命周期 |

---

## 工具 Tools

Tools 允许 MCP Server 暴露可执行函数。LangChain 会把 MCP Tools 转换为 LangChain Tools，随后可直接传给 `create_agent`。

### 加载工具

```python
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({...})
tools = await client.get_tools()

for tool in tools:
    print(tool.name)
    print(tool.description)
    print(tool.args_schema)

agent = create_agent("claude-opus-4-8", tools)
```

### 工具错误处理

默认情况下，如果 MCP 工具返回 `CallToolResult(isError=True)`，适配器会把错误作为 `status="error"` 的工具消息返回给模型，让 Agent 有机会读取错误并重试或调整参数。

如果希望工具执行错误直接抛异常，可以设置：

```python
client = MultiServerMCPClient({...}, handle_tool_errors=False)
```

也可以在直接使用 `load_mcp_tools(session, handle_tool_errors=False)` 时配置。

> 注意：该设置只影响 MCP 工具执行层面的错误。传输错误、session 错误和内容转换错误仍会抛异常。

### 结构化内容 structuredContent

MCP 工具可以同时返回人类可读文本和结构化数据。适配器会把 `structuredContent` 包装到 `MCPToolArtifact` 中，并放到 `ToolMessage.artifact`。

```python
from langchain.messages import ToolMessage

result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Get data from the server"}]}
)

for message in result["messages"]:
    if isinstance(message, ToolMessage) and message.artifact:
        structured_content = message.artifact["structured_content"]
        print(structured_content)
```

如果希望结构化内容也进入对话历史、让模型可见，可用拦截器把结构化数据追加为文本内容：

```python
import json

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from mcp.types import TextContent

async def append_structured_content(request: MCPToolCallRequest, handler):
    result = await handler(request)
    if result.structuredContent:
        result.content += [
            TextContent(type="text", text=json.dumps(result.structuredContent, ensure_ascii=False))
        ]
    return result

client = MultiServerMCPClient({...}, tool_interceptors=[append_structured_content])
```

### 多模态工具内容

MCP 工具可返回文本、图片等多模态内容。适配器会转换为 LangChain 标准 content blocks，可通过 `ToolMessage.content_blocks` 访问：

```python
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Take a screenshot of the current page"}]}
)

for message in result["messages"]:
    if message.type == "tool":
        print("Raw content:", message.content)
        for block in message.content_blocks:
            if block["type"] == "text":
                print("Text:", block["text"])
            elif block["type"] == "image":
                print("Image URL:", block.get("url"))
                print("Image base64 prefix:", block.get("base64", "")[:50])
```

---

## 资源 Resources

Resources 允许 MCP Server 暴露可读取数据，例如文件、数据库记录或 API 响应。LangChain 将 MCP Resources 转换为 `Blob` 对象，统一处理文本和二进制内容。

### 加载某个服务器的全部资源

```python
blobs = await client.get_resources("server_name")

for blob in blobs:
    print(f"URI: {blob.metadata['uri']}")
    print(f"MIME type: {blob.mimetype}")
    print(blob.as_string())
```

### 按 URI 加载指定资源

```python
blobs = await client.get_resources(
    "server_name",
    uris=["file:///path/to/file.txt"],
)
```

### 在显式 session 中加载资源

```python
from langchain_mcp_adapters.resources import load_mcp_resources

async with client.session("server_name") as session:
    blobs = await load_mcp_resources(session)
    selected = await load_mcp_resources(session, uris=["file:///path/to/file.txt"])
```

---

## 提示词 Prompts

Prompts 允许 MCP Server 暴露可复用的提示词模板。LangChain 会把 MCP Prompts 转换为 messages，便于直接传给 Agent 或 LangGraph 工作流。

### 加载提示词

```python
messages = await client.get_prompt("server_name", "summarize")

for message in messages:
    print(message.type, message.content)
```

### 带参数的提示词

```python
messages = await client.get_prompt(
    "server_name",
    "code_review",
    arguments={"language": "python", "focus": "security"},
)

response = await agent.ainvoke({"messages": messages})
```

### 在显式 session 中加载提示词

```python
from langchain_mcp_adapters.prompts import load_mcp_prompt

async with client.session("server_name") as session:
    messages = await load_mcp_prompt(session, "summarize")
    review_messages = await load_mcp_prompt(
        session,
        "code_review",
        arguments={"language": "python", "focus": "security"},
    )
```

---

## 会话管理

### 默认无状态模式

`MultiServerMCPClient` 默认在每次工具调用时创建新的 MCP `ClientSession`，调用完成后清理：

```python
client = MultiServerMCPClient({...})
tools = await client.get_tools()
agent = create_agent("claude-opus-4-8", tools)

await agent.ainvoke({"messages": [{"role": "user", "content": "first call"}]})
await agent.ainvoke({"messages": [{"role": "user", "content": "second call"}]})
```

这适合大多数无状态工具，但不适合需要保留登录态、事务状态或服务器上下文的工具。

### 显式有状态会话

```python
from langchain.agents import create_agent
from langchain_mcp_adapters.tools import load_mcp_tools

client = MultiServerMCPClient({...})

async with client.session("server_name") as session:
    tools = await load_mcp_tools(session)
    agent = create_agent("claude-opus-4-8", tools)

    await agent.ainvoke({"messages": [{"role": "user", "content": "Start a workflow"}]})
    await agent.ainvoke({"messages": [{"role": "user", "content": "Continue the previous step"}]})
```

### 何时使用有状态会话

| 场景 | 推荐模式 |
|------|----------|
| 独立查询、幂等 API | 默认无状态 |
| 需要登录态或临时令牌 | 显式有状态 |
| 数据库事务、购物车、长任务 | 显式有状态 |
| 工具内部维护上下文 | 显式有状态 |
| 单次计算、简单搜索 | 默认无状态 |

---

## 工具拦截器

MCP Server 运行在独立进程或远程服务中，无法直接访问 LangGraph/LangChain 运行时的 `context`、`store`、`state` 等信息。**工具拦截器（tool interceptors）**用于桥接这个边界：在 MCP 工具调用前后读取运行时上下文、修改请求、处理返回、重试或短路执行。

### 基本模式

```python
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

async def logging_interceptor(request: MCPToolCallRequest, handler):
    print(f"Calling tool: {request.name}, args={request.args}")
    result = await handler(request)
    print(f"Tool {request.name} returned: {result}")
    return result

client = MultiServerMCPClient({...}, tool_interceptors=[logging_interceptor])
```

多个拦截器按“洋葱模型”执行：列表中第一个是最外层。

```python
client = MultiServerMCPClient(
    {...},
    tool_interceptors=[outer_interceptor, inner_interceptor],
)

# outer before -> inner before -> tool execution -> inner after -> outer after
```

### 修改工具参数

使用 `request.override(...)` 创建修改后的请求，避免直接修改原对象：

```python
async def inject_user_context(request: MCPToolCallRequest, handler):
    runtime = request.runtime
    user_id = runtime.context.user_id

    modified_request = request.override(
        args={**request.args, "user_id": user_id}
    )
    return await handler(modified_request)
```

### 访问 store 与 state

```python
async def personalize_search(request: MCPToolCallRequest, handler):
    runtime = request.runtime
    user_id = runtime.context.user_id
    store = runtime.store

    prefs = store.get(("preferences",), user_id)
    if prefs and request.name == "search":
        request = request.override(
            args={
                **request.args,
                "language": prefs.value.get("language", "en"),
                "limit": prefs.value.get("result_limit", 10),
            }
        )

    return await handler(request)
```

```python
from langchain.messages import ToolMessage

async def require_authentication(request: MCPToolCallRequest, handler):
    runtime = request.runtime
    is_authenticated = runtime.state.get("authenticated", False)

    sensitive_tools = {"delete_file", "update_settings", "export_data"}
    if request.name in sensitive_tools and not is_authenticated:
        return ToolMessage(
            content="Authentication required. Please log in first.",
            tool_call_id=runtime.tool_call_id,
        )

    return await handler(request)
```

### 动态修改 HTTP headers

```python
async def auth_header_interceptor(request: MCPToolCallRequest, handler):
    token = get_token_for_tool(request.name)
    request = request.override(headers={"Authorization": f"Bearer {token}"})
    return await handler(request)
```

### 返回 Command 更新状态或控制图流

拦截器可以返回 LangGraph `Command`，用于更新 Agent 状态或跳转节点：

```python
from langchain.messages import ToolMessage
from langgraph.types import Command

async def handle_task_completion(request: MCPToolCallRequest, handler):
    result = await handler(request)

    if request.name == "submit_order":
        return Command(
            update={
                "messages": [result] if isinstance(result, ToolMessage) else [],
                "task_status": "completed",
            },
            goto="summary_agent",
        )

    return result
```

也可以用 `goto="__end__"` 提前结束图执行。

### 错误处理与重试

```python
import asyncio

async def retry_interceptor(
    request: MCPToolCallRequest,
    handler,
    max_retries: int = 3,
    delay: float = 1.0,
):
    last_error = None
    for attempt in range(max_retries):
        try:
            return await handler(request)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)
                print(f"{request.name} failed, retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
    raise last_error
```

> 默认工具执行错误会以错误 ToolMessage 返回，不一定抛异常。要让工具执行错误进入 `except` 分支，需设置 `handle_tool_errors=False`。传输、session、内容转换错误仍会抛异常。

---

## 回调系统

`Callbacks` 用于处理 MCP Server 的协议级通知，包括进度、日志和征询。

### 进度通知

```python
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext
from langchain_mcp_adapters.client import MultiServerMCPClient

async def on_progress(
    progress: float,
    total: float | None,
    message: str | None,
    context: CallbackContext,
):
    percent = (progress / total * 100) if total else progress
    tool_info = f" ({context.tool_name})" if context.tool_name else ""
    print(f"[{context.server_name}{tool_info}] Progress: {percent:.1f}% - {message}")

client = MultiServerMCPClient(
    {...},
    callbacks=Callbacks(on_progress=on_progress),
)
```

`CallbackContext` 常用字段：

- `server_name`：MCP Server 名称
- `tool_name`：当前工具名（工具调用期间可用）

### 日志通知

```python
from mcp.types import LoggingMessageNotificationParams

async def on_logging_message(
    params: LoggingMessageNotificationParams,
    context: CallbackContext,
):
    print(f"[{context.server_name}] {params.level}: {params.data}")

client = MultiServerMCPClient(
    {...},
    callbacks=Callbacks(on_logging_message=on_logging_message),
)
```

### 征询 Elicitation

Elicitation 允许 MCP Server 在工具执行期间向客户端请求额外输入。

#### Server 端

```python
from pydantic import BaseModel
from mcp.server.fastmcp import Context, FastMCP

server = FastMCP("Profile")

class UserDetails(BaseModel):
    email: str
    age: int

@server.tool()
async def create_profile(name: str, ctx: Context) -> str:
    result = await ctx.elicit(
        message=f"Please provide details for {name}'s profile:",
        schema=UserDetails,
    )
    if result.action == "accept" and result.data:
        return f"Created profile for {name}: email={result.data.email}, age={result.data.age}"
    if result.action == "decline":
        return f"User declined. Created minimal profile for {name}."
    return "Profile creation cancelled."

if __name__ == "__main__":
    server.run(transport="http")
```

#### Client 端

```python
from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult

async def on_elicitation(
    mcp_context: RequestContext,
    params: ElicitRequestParams,
    context: CallbackContext,
) -> ElicitResult:
    # 实际应用中应在 UI 中展示 params.message 与 params.requestedSchema
    return ElicitResult(
        action="accept",
        content={"email": "user@example.com", "age": 25},
    )

client = MultiServerMCPClient(
    {
        "profile": {
            "transport": "http",
            "url": "http://localhost:8000/mcp",
        }
    },
    callbacks=Callbacks(on_elicitation=on_elicitation),
)
```

Elicitation 的响应动作：

| Action | 含义 |
|--------|------|
| `accept` | 用户提供了有效输入，`content` 包含数据 |
| `decline` | 用户拒绝提供信息 |
| `cancel` | 用户取消整个操作 |

---

## 创建 MCP 服务器

### stdio Server

```python
from fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### Streamable HTTP Server

```python
from fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for a location."""
    return "It's always sunny in New York"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

LangChain 客户端配置：

```python
client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "http",
            "url": "http://localhost:8000/mcp",
        }
    }
)
```

### Resources 与 Prompts 示例

```python
from fastmcp import FastMCP

mcp = FastMCP("Docs")

@mcp.resource("docs://intro")
def intro() -> str:
    return "This is the project introduction."

@mcp.prompt()
def summarize(text: str) -> str:
    return f"Summarize the following text:\n\n{text}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

---

## 实战案例

### 案例 1：聚合数学与天气服务

```python
import asyncio

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",
                "command": "python",
                "args": ["/absolute/path/to/math_server.py"],
            },
            "weather": {
                "transport": "http",
                "url": "http://localhost:8000/mcp",
                "headers": {"Authorization": "Bearer WEATHER_TOKEN"},
            },
        }
    )

    tools = await client.get_tools()
    agent = create_agent("claude-opus-4-8", tools)

    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in NYC? Also calculate (3 + 5) * 12.",
                }
            ]
        }
    )
    print(response["messages"][-1].content)

asyncio.run(main())
```

### 案例 2：基于上下文注入用户信息

```python
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

@dataclass
class AppContext:
    user_id: str
    api_key: str

async def inject_user_context(request: MCPToolCallRequest, handler):
    runtime = request.runtime
    request = request.override(
        args={
            **request.args,
            "user_id": runtime.context.user_id,
        },
        headers={
            "Authorization": f"Bearer {runtime.context.api_key}",
        },
    )
    return await handler(request)

client = MultiServerMCPClient(
    {
        "orders": {
            "transport": "http",
            "url": "http://orders-service:8000/mcp",
        }
    },
    tool_interceptors=[inject_user_context],
)

tools = await client.get_tools()
agent = create_agent("claude-opus-4-8", tools, context_schema=AppContext)

response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Search my orders"}]},
    context=AppContext(user_id="user_123", api_key="sk-..."),
)
```

### 案例 3：敏感操作审批

```python
from langchain.messages import ToolMessage
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langgraph.types import Command

SENSITIVE_TOOLS = {"delete_file", "send_email", "make_payment"}

async def approval_interceptor(request: MCPToolCallRequest, handler):
    if request.name in SENSITIVE_TOOLS:
        approved = request.runtime.state.get("approved_tools", set())
        if request.name not in approved:
            return Command(
                update={
                    "pending_approval": {
                        "tool": request.name,
                        "args": request.args,
                    }
                },
                interrupt="Waiting for human approval",
            )

    return await handler(request)

client = MultiServerMCPClient({...}, tool_interceptors=[approval_interceptor])
```

---

## 最佳实践

### 1. 明确无状态与有状态边界

- 默认使用无状态模式，减少连接泄漏和服务端状态污染。
- 只有在确实需要登录态、事务或连续上下文时使用 `client.session(...)`。
- stdio 进程虽然在连接期间有状态，但 `MultiServerMCPClient` 默认每次工具调用仍会新建 session。

### 2. 工具错误要对 Agent 友好

- 对可恢复错误，返回明确错误信息，让模型可以修正参数后重试。
- 对传输、认证、内部错误，记录详细日志，但向模型返回简洁可操作的信息。
- 需要异常语义时设置 `handle_tool_errors=False`。

### 3. 结构化数据与可见文本分层

- `structuredContent` 适合机器读取，默认进入 `ToolMessage.artifact`。
- 如果模型需要看到结构化数据，使用拦截器显式追加到 `content`。
- 大型结构化结果不要无控制地塞进上下文，可先筛选、分页或返回摘要。

### 4. 安全处理外部工具

- 不要把长期密钥硬编码在 MCP Server 或文档示例中；使用环境变量、密钥管理或动态 header 注入。
- 对文件路径、URL、命令参数做白名单或根目录约束，防止路径遍历与越权访问。
- 删除、付款、发邮件、修改配置等高影响操作应加审批、认证或二次确认。
- 拦截器日志中要脱敏密码、token、cookie 等敏感字段。

### 5. 性能与生命周期

- 应用启动时创建可复用的 `MultiServerMCPClient`，避免每个请求重复构造。
- 对远程 HTTP MCP Server 设置合理超时、重试和鉴权刷新策略。
- 工具多、schema 大时，关注模型上下文占用；只连接当前任务需要的 MCP Server。
- 长任务使用 progress callbacks 给用户反馈。

### 6. 可观测性

- 使用 `Callbacks(on_progress=..., on_logging_message=...)` 记录协议级事件。
- 使用拦截器记录工具名、耗时、错误类型、请求 ID，但避免记录敏感参数。
- 结合 LangSmith 可追踪 MCP 工具调用与 Agent 推理步骤。

### 7. 测试策略

- 单测 MCP Server 工具函数本身。
- 用 mock MCP Client 测试 LangChain 侧工具加载。
- 单测拦截器对认证、重试、脱敏、审批的行为。
- 集成测试至少覆盖 stdio 和 HTTP 两种传输之一。

---

## 快速参考

### 安装

```bash
pip install langchain-mcp-adapters fastmcp
```

### MultiServerMCPClient 配置

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "stdio_server": {
            "transport": "stdio",
            "command": "python",
            "args": ["/absolute/path/to/server.py"],
        },
        "http_server": {
            "transport": "http",
            "url": "http://localhost:8000/mcp",
            "headers": {"Authorization": "Bearer TOKEN"},
        },
    },
    tool_interceptors=[...],
    callbacks=Callbacks(...),
)
```

### 核心 API

```python
# Tools
tools = await client.get_tools()

# Resources
blobs = await client.get_resources("server_name")
blobs = await client.get_resources("server_name", uris=["file:///path/to/file.txt"])

# Prompts
messages = await client.get_prompt("server_name", "prompt_name")
messages = await client.get_prompt("server_name", "prompt_name", arguments={...})

# Stateful session
async with client.session("server_name") as session:
    tools = await load_mcp_tools(session)
```

### 拦截器模板

```python
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

async def interceptor(request: MCPToolCallRequest, handler):
    request = request.override(args={**request.args, "extra": "value"})
    result = await handler(request)
    return result
```

### 回调模板

```python
from langchain_mcp_adapters.callbacks import Callbacks

client = MultiServerMCPClient(
    {...},
    callbacks=Callbacks(
        on_progress=on_progress,
        on_logging_message=on_logging_message,
        on_elicitation=on_elicitation,
    ),
)
```

### FastMCP 模板

```python
from fastmcp import FastMCP

mcp = FastMCP("ServiceName")

@mcp.tool()
def my_tool(arg: str) -> str:
    """Tool description."""
    return "result"

@mcp.resource("resource://{name}")
def my_resource(name: str) -> str:
    return f"data for {name}"

@mcp.prompt()
def my_prompt(topic: str) -> str:
    return f"Explain {topic}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

---

## 总结

LangChain 通过 `langchain-mcp-adapters` 将 MCP 的 Tools、Resources、Prompts 接入 Agent 与工作流：

1. **Tools**：MCP 工具会转换为 LangChain Tools，可直接传给 `create_agent`。
2. **Resources**：MCP 资源会转换为 `Blob`，适合统一处理文本和二进制数据。
3. **Prompts**：MCP 提示词会转换为 LangChain messages，适合复用模板。
4. **默认无状态**：`MultiServerMCPClient` 每次工具调用默认创建并清理 session。
5. **显式有状态**：需要服务端上下文时使用 `client.session(...)`。
6. **拦截器**：用于注入上下文、鉴权、动态 header、重试、状态更新和审批。
7. **回调**：用于接收进度、日志和 elicitation 请求。

使用 MCP 可以把外部系统能力与 LangChain Agent 解耦，既保留标准化协议带来的复用性，也能通过 LangChain 的运行时上下文、状态管理和可观测性能力进行工程化控制。