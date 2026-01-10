# LangChain MCP (Model Context Protocol) 详细指南

## 目录

1. [概述](#概述)
2. [核心概念](#核心概念)
3. [快速开始](#快速开始)
4. [传输机制](#传输机制)
5. [工具 (Tools)](#工具-tools)
6. [资源 (Resources)](#资源-resources)
7. [提示词 (Prompts)](#提示词-prompts)
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

**MCP (Model Context Protocol)** 是一个开放协议（模型上下文协议），用于标准化应用程序如何向 LLM 提供**工具**和**上下文**。它定义了一种统一的方式，让 AI 应用能够：

- 发现和调用外部工具
- 访问数据资源
- 使用可重用的提示词模板

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP 生态系统                              │
│                                                             │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐ │
│  │ LangChain   │      │   MCP       │      │   MCP       │ │
│  │   Agent     │ ←──→ │  Client     │ ←──→ │  Server     │ │
│  │             │      │             │      │             │ │
│  └─────────────┘      └─────────────┘      └─────────────┘ │
│                                                   │         │
│                                           ┌───────┴───────┐ │
│                                           │               │ │
│                                     ┌─────┴─────┐   ┌─────┴─────┐
│                                     │  Tools    │   │ Resources │
│                                     │  工具     │   │  资源     │
│                                     └───────────┘   └───────────┘
└─────────────────────────────────────────────────────────────┘
```

### 为什么使用 MCP

| 优势 | 说明 |
|------|------|
| **标准化** | 统一的协议，一次实现，到处使用 |
| **解耦** | AI 应用与工具/服务分离，独立开发和部署 |
| **可复用** | MCP 服务器可被多个 AI 应用共享 |
| **生态丰富** | 社区已有大量现成的 MCP 服务器 |
| **类型安全** | 强类型定义，减少运行时错误 |

### LangChain 的 MCP 支持

LangChain 通过 `langchain-mcp-adapters` 库提供 MCP 支持：

```bash
pip install langchain-mcp-adapters
```

**核心能力**：
- 连接多个 MCP 服务器
- 加载 MCP 工具并与 Agent 集成
- 访问 MCP 资源和提示词
- 工具拦截器和回调系统
- 支持有状态和无状态会话

---

## 核心概念

### MCP 架构

```
┌─────────────────────────────────────────────────────────────┐
│                      MCP 服务器                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Tools (工具)                                        │   │
│  │  • 可执行的函数                                      │   │
│  │  • 输入参数和返回值有明确定义                        │   │
│  │  • 例如: add(), search(), send_email()              │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Resources (资源)                                    │   │
│  │  • 可访问的数据                                      │   │
│  │  • 文件、数据库记录、API 响应等                      │   │
│  │  • 通过 URI 标识                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Prompts (提示词)                                    │   │
│  │  • 可重用的提示词模板                                │   │
│  │  • 支持参数化                                        │   │
│  │  • 例如: summarize, code_review, translate          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↑
                            │ MCP 协议
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      MCP 客户端                              │
│  • 发现服务器能力                                           │
│  • 调用工具                                                 │
│  • 获取资源                                                 │
│  • 加载提示词                                               │
└─────────────────────────────────────────────────────────────┘
```

### 三大核心组件

| 组件 | 说明 | 示例 |
|------|------|------|
| **Tools** | 可执行的函数 | `add(a, b)`, `search(query)` |
| **Resources** | 可访问的数据 | 文件内容、数据库记录 |
| **Prompts** | 提示词模板 | 摘要模板、代码审查模板 |

---

## 快速开始

### 1. 安装依赖

```bash
pip install langchain-mcp-adapters
```

### 2. 创建 MCP 服务器

使用 FastMCP 创建一个简单的数学服务器：

```python
# math_server.py
from fastmcp import FastMCP

# 创建 MCP 服务器实例
mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """两数相加"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """两数相乘"""
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """两数相除"""
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### 3. 在 LangChain 中使用

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

# 创建 MCP 客户端，连接到数学服务器
client = MultiServerMCPClient({
    "math": {
        "transport": "stdio",
        "command": "python",
        "args": ["math_server.py"],
    }
})

# 获取工具并创建 Agent
tools = await client.get_tools()
agent = create_agent("gpt-4o", tools)

# 使用 Agent
response = await agent.ainvoke({
    "messages": [{"role": "user", "content": "计算 (3 + 5) × 12 等于多少？"}]
})

print(response["messages"][-1].content)
# 输出: (3 + 5) × 12 = 8 × 12 = 96
```

### 4. 连接多个服务器

```python
client = MultiServerMCPClient({
    # 本地数学服务器（stdio 传输）
    "math": {
        "transport": "stdio",
        "command": "python",
        "args": ["math_server.py"],
    },
    # 远程天气服务器（HTTP 传输）
    "weather": {
        "transport": "http",
        "url": "http://localhost:8000/mcp",
    },
    # 文件系统服务器
    "filesystem": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-server-filesystem", "/path/to/directory"],
    }
})

# 获取所有服务器的工具
tools = await client.get_tools()
print(f"总共加载了 {len(tools)} 个工具")

# 创建 Agent
agent = create_agent("gpt-4o", tools)

# 可以同时使用多个服务器的工具
response = await agent.ainvoke({
    "messages": [{"role": "user", "content": "今天北京天气怎么样？顺便帮我算一下 123 × 456"}]
})
```

---

## 传输机制

MCP 支持多种传输方式，用于客户端和服务器之间的通信。

### 1. stdio 传输

通过标准输入/输出与子进程通信，适用于本地工具。

```python
client = MultiServerMCPClient({
    "local_tool": {
        "transport": "stdio",
        "command": "python",           # 执行命令
        "args": ["server.py"],         # 命令参数
        "env": {                       # 可选：环境变量
            "API_KEY": "xxx",
            "DEBUG": "true",
        },
        "cwd": "/path/to/directory",   # 可选：工作目录
    }
})
```

**优点**：
- 简单易用
- 无需网络配置
- 适合本地开发

**缺点**：
- 只能在本地使用
- 每次连接启动新进程

### 2. HTTP 传输

使用 HTTP 请求进行通信，适用于远程服务。

```python
client = MultiServerMCPClient({
    "remote_service": {
        "transport": "http",
        "url": "http://localhost:8000/mcp",
        "headers": {                   # 可选：自定义请求头
            "Authorization": "Bearer YOUR_TOKEN",
            "X-Custom-Header": "custom-value",
        },
        "timeout": 30,                 # 可选：超时时间（秒）
    }
})
```

**优点**：
- 支持远程服务
- 易于扩展和负载均衡
- 支持身份验证

**缺点**：
- 需要网络配置
- 可能有延迟

### 3. SSE 传输 (Server-Sent Events)

单向流式传输，适用于实时更新。

```python
client = MultiServerMCPClient({
    "streaming_service": {
        "transport": "sse",
        "url": "http://localhost:8000/sse",
    }
})
```

### 传输方式选择指南

| 场景 | 推荐传输 | 原因 |
|------|----------|------|
| 本地开发 | stdio | 简单、无需网络 |
| 生产环境 | HTTP | 可扩展、支持认证 |
| 实时更新 | SSE | 低延迟流式传输 |
| 微服务架构 | HTTP | 服务独立部署 |

---

## 工具 (Tools)

### 加载工具

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient({...})

# 获取所有服务器的工具
tools = await client.get_tools()

# 查看工具信息
for tool in tools:
    print(f"工具名: {tool.name}")
    print(f"描述: {tool.description}")
    print(f"参数: {tool.args_schema}")
    print("---")

# 创建 Agent
agent = create_agent("gpt-4o", tools)
```

### 结构化内容

MCP 工具可以返回结构化数据（JSON）和人类可读文本。

**服务器端**：
```python
from fastmcp import FastMCP

mcp = FastMCP("DataService")

@mcp.tool()
def get_user_info(user_id: str) -> dict:
    """获取用户信息"""
    # 返回结构化数据
    return {
        "content": f"用户 {user_id} 的信息已获取",  # 人类可读
        "structuredContent": {                       # 机器可解析
            "user_id": user_id,
            "name": "张三",
            "email": "zhangsan@example.com",
            "created_at": "2024-01-01",
        }
    }
```

**客户端访问**：
```python
from langchain.messages import ToolMessage

result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "获取用户 U001 的信息"}]
})

# 遍历消息，找到工具消息
for message in result["messages"]:
    if isinstance(message, ToolMessage):
        # 人类可读内容
        print(f"内容: {message.content}")

        # 结构化数据（通过 artifact 访问）
        if message.artifact:
            print(f"结构化数据: {message.artifact}")
```

### 多模态内容

MCP 工具可以返回文本、图像等多种内容类型。

```python
result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "截取当前页面的截图"}]
})

# 访问多模态内容
for message in result["messages"]:
    if message.type == "tool":
        # 原始内容（提供商特定格式）
        print(f"原始内容: {message.content}")

        # 标准化内容块
        for block in message.content_blocks:
            if block["type"] == "text":
                print(f"文本: {block['text']}")
            elif block["type"] == "image":
                print(f"图片 URL: {block.get('url')}")
                print(f"图片 Base64: {block.get('base64', '')[:50]}...")
```

---

## 资源 (Resources)

MCP 服务器可以暴露数据资源，如文件、数据库记录、API 响应等。

### 加载资源

```python
# 加载服务器的所有资源
blobs = await client.get_resources("server_name")

for blob in blobs:
    print(f"URI: {blob.metadata['uri']}")
    print(f"MIME 类型: {blob.mimetype}")
    print(f"内容: {blob.as_string()[:100]}...")
    print("---")
```

### 按 URI 加载特定资源

```python
# 加载特定资源
blobs = await client.get_resources(
    "server_name",
    uris=[
        "file:///path/to/file.txt",
        "db://users/123",
    ]
)

for blob in blobs:
    print(f"资源: {blob.metadata['uri']}")
    print(f"内容: {blob.as_string()}")
```

### 资源类型示例

| URI 格式 | 说明 | 示例 |
|----------|------|------|
| `file://` | 文件系统 | `file:///home/user/doc.txt` |
| `db://` | 数据库记录 | `db://users/123` |
| `http://` | HTTP 资源 | `http://api.example.com/data` |
| `custom://` | 自定义资源 | `custom://my-resource` |

---

## 提示词 (Prompts)

MCP 服务器可以暴露可重用的提示词模板。

### 加载提示词

```python
# 加载简单提示词
messages = await client.get_prompt("server_name", "summarize")

# 查看返回的消息
for msg in messages:
    print(f"角色: {msg['role']}")
    print(f"内容: {msg['content']}")
```

### 带参数的提示词

```python
# 加载带参数的提示词
messages = await client.get_prompt(
    "server_name",
    "code_review",
    arguments={
        "language": "python",
        "focus": "security",
        "code": "def login(password): ...",
    }
)

# 使用提示词
agent = create_agent("gpt-4o", tools=[])
response = await agent.ainvoke({"messages": messages})
```

### 服务器端定义提示词

```python
from fastmcp import FastMCP

mcp = FastMCP("PromptService")

@mcp.prompt()
def summarize(text: str, style: str = "concise") -> str:
    """生成摘要提示词"""
    if style == "concise":
        return f"请用一句话总结以下内容：\n\n{text}"
    else:
        return f"请详细总结以下内容，包括要点和结论：\n\n{text}"

@mcp.prompt()
def code_review(code: str, language: str, focus: str = "general") -> list:
    """生成代码审查提示词"""
    return [
        {
            "role": "system",
            "content": f"你是一位资深的 {language} 开发者，专注于 {focus} 方面的代码审查。"
        },
        {
            "role": "user",
            "content": f"请审查以下代码：\n\n```{language}\n{code}\n```"
        }
    ]
```

---

## 会话管理

### 无状态模式（默认）

默认情况下，每次工具调用都会创建新会话、执行、然后清理。

```python
client = MultiServerMCPClient({...})

# 无状态调用：每次都是独立的
tools = await client.get_tools()
agent = create_agent("gpt-4o", tools)

# 多次调用之间没有共享状态
await agent.ainvoke({"messages": [...]})
await agent.ainvoke({"messages": [...]})  # 独立的会话
```

### 有状态会话

对于需要跨工具调用维护上下文的场景，可以使用有状态会话。

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent

client = MultiServerMCPClient({...})

# 显式创建有状态会话
async with client.session("server_name") as session:
    # 在同一会话中加载工具
    tools = await load_mcp_tools(session)

    agent = create_agent("gpt-4o", tools)

    # 这些调用共享同一个会话状态
    await agent.ainvoke({"messages": [{"role": "user", "content": "开始新任务"}]})
    await agent.ainvoke({"messages": [{"role": "user", "content": "继续上一步"}]})
    # 会话状态在整个 with 块中保持

# 退出 with 块后，会话自动关闭
```

### 何时使用有状态会话

| 场景 | 推荐模式 |
|------|----------|
| 独立工具调用 | 无状态（默认） |
| 多步骤工作流 | 有状态 |
| 需要保持登录状态 | 有状态 |
| 数据库事务 | 有状态 |
| 简单查询 | 无状态（默认） |

---

## 工具拦截器

拦截器提供中间件式的控制，可以在（MCP）工具执行前后进行拦截和修改。

### 基本结构

```python
async def my_interceptor(
    request: MCPToolCallRequest,
    handler,  # 下一个拦截器或实际工具
):
    # 1. 前置处理（工具执行前）
    print(f"即将调用工具: {request.name}")

    # 2. 调用下一个处理器
    result = await handler(request)

    # 3. 后置处理（工具执行后）
    print(f"工具执行完成")

    return result
```

### 访问运行时上下文

```python
# 导入 MCP 工具调用请求类型
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

async def inject_user_context(
    request: MCPToolCallRequest,  # MCP 工具调用请求对象，包含工具名、参数、运行时等信息
    handler,  # 处理函数，实际的工具执行逻辑
):
    """注入用户上下文到工具参数
    
    这个拦截器函数演示了如何在工具调用前自动注入用户上下文信息，
    如用户 ID 和 API 密钥。这样工具就不需要手动传递这些
    重复的上下文信息，实现自动化的上下文注入。
    """
    # 从请求中获取运行时对象，用于访问上下文信息
    runtime = request.runtime

    # 从 Runtime Context 获取用户相关的上下文信息
    user_id = runtime.context.user_id  # 用户唯一标识符
    api_key = runtime.context.api_key  # API 密钥，用于身份验证

    # 修改请求参数，注入用户上下文信息
    # 使用 ** 解包原有参数，然后添加上下文信息
    modified_args = {
        **request.args,  # 保留原有的工具参数
        "user_id": user_id,  # 注入用户 ID
        "api_key": api_key,  # 注入 API 密钥
    }

    # 使用 override 创建修改后的请求对象
    # 这会创建一个新的请求对象，包含修改后的参数列表
    modified_request = request.override(args=modified_args)

    # 调用处理函数执行实际的工具调用，并传递修改后的请求
    return await handler(modified_request)

# 使用拦截器创建 MCP 客户端
# 拦截器会在每次工具调用前自动执行
client = MultiServerMCPClient(
    {...},  # MCP 服务器配置
    tool_interceptors=[inject_user_context],  # 注册上下文注入拦截器
)
```

### 认证检查

```python
# 导入工具消息类型，用于返回工具执行结果
from langchain.messages import ToolMessage

async def require_authentication(
    request: MCPToolCallRequest,  # MCP 工具调用请求对象
    handler,  # 处理函数，实际的工具执行逻辑
):
    """敏感工具需要认证
    
    这个拦截器函数演示了如何实现基于认证状态的工具访问控制。
    敏感工具只有在用户已认证的情况下才能执行，未认证用户
    尝试调用敏感工具时会收到错误提示。
    """
    # 从请求中获取运行时对象
    runtime = request.runtime
    # 从运行时状态中获取认证状态
    state = runtime.state
    # 检查用户是否已认证，默认为未认证
    is_authenticated = state.get("authenticated", False)

    # 定义敏感工具列表，这些工具需要认证才能使用
    sensitive_tools = ["delete_file", "update_settings", "export_data"]

    # 检查是否为敏感工具且用户未认证
    if request.name in sensitive_tools and not is_authenticated:
        # 未认证用户尝试访问敏感工具：返回错误消息
        return ToolMessage(
            content="需要认证。请先登录。",  # 错误提示信息
            tool_call_id=runtime.tool_call_id,  # 关联工具调用 ID
        )

    # 已认证用户或非敏感工具：正常执行
    return await handler(request)

# 使用拦截器创建 MCP 客户端
# 拦截器会在每次工具调用前自动执行认证检查
client = MultiServerMCPClient(
    {...},  # MCP 服务器配置
    tool_interceptors=[require_authentication],  # 注册认证检查拦截器
)
```

### 错误处理与重试

```python
import asyncio

async def retry_interceptor(
    request: MCPToolCallRequest,
    handler,
    max_retries: int = 3,
    base_delay: float = 1.0,
):
    """失败时自动重试"""
    last_error = None

    for attempt in range(max_retries):
        try:
            return await handler(request)
        except Exception as e:
            last_error = e
            print(f"工具 {request.name} 失败 (尝试 {attempt + 1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                # 指数退避
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)

    # 所有重试都失败
    raise last_error
```

### 日志记录

```python
import time

async def logging_interceptor(
    request: MCPToolCallRequest,
    handler,
):
    """记录工具调用日志"""
    start_time = time.time()

    print(f"[LOG] 开始调用工具: {request.name}")
    print(f"[LOG] 参数: {request.args}")

    try:
        result = await handler(request)
        elapsed = time.time() - start_time
        print(f"[LOG] 工具 {request.name} 成功，耗时 {elapsed:.2f}s")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[LOG] 工具 {request.name} 失败，耗时 {elapsed:.2f}s，错误: {e}")
        raise
```

### 状态更新

```python
from langgraph.types import Command

async def handle_task_completion(
    request: MCPToolCallRequest,
    handler,
):
    """处理任务完成后的状态更新"""
    result = await handler(request)

    # 特定工具完成后更新状态
    if request.name == "submit_order":
        return Command(
            update={
                "messages": [result],
                "task_status": "completed",
                "order_submitted": True,
            },
            goto="summary_agent",  # 跳转到下一个节点
        )

    return result
```

### 拦截器组合

拦截器按"洋葱"模型组合——列表中的第一个是最外层：

```python
async def outer_interceptor(request, handler):
    print("outer: 开始")
    result = await handler(request)
    print("outer: 结束")
    return result

async def inner_interceptor(request, handler):
    print("inner: 开始")
    result = await handler(request)
    print("inner: 结束")
    return result

client = MultiServerMCPClient(
    {...},
    tool_interceptors=[outer_interceptor, inner_interceptor],
)

# 执行顺序:
# outer: 开始
#   inner: 开始
#     [实际工具执行]
#   inner: 结束
# outer: 结束
```

---

## 回调系统

MCP 客户端支持多种回调，用于处理进度、日志、用户输入请求等。

### 进度回调

监控长时间运行的工具的进度：

```python
# 导入 MCP 客户端和回调相关的模块
from langchain_mcp_adapters.client import MultiServerMCPClient  # MCP 客户端类
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext  # 回调类型和上下文

async def on_progress(
    progress: float,  # 当前进度值
    total: float | None,  # 总进度值，可能为 None（未知总数）
    message: str | None,  # 进度描述消息，可能为 None
    context: CallbackContext,  # 回调上下文，包含服务器和工具信息
):
    """处理进度更新
    
    这个回调函数演示了如何监控长时间运行工具的执行进度。
    可以显示百分比进度、状态消息和工具信息，提供用户
    友好的执行反馈。
    """
    # 计算进度百分比
    if total:
        # 如果有总数，计算精确的百分比
        percent = progress / total * 100
    else:
        # 如果没有总数，直接使用当前进度值作为百分比
        percent = progress

    # 构建工具信息字符串，用于标识当前执行的工具
    tool_info = f" ({context.tool_name})" if context.tool_name else ""
    # 格式化并输出进度信息，包含服务器名、工具名、百分比和消息
    print(f"[{context.server_name}{tool_info}] 进度: {percent:.1f}% - {message}")

# 创建 MCP 客户端并注册进度回调
# 回调会在工具执行过程中自动触发
client = MultiServerMCPClient(
    {...},  # MCP 服务器配置
    callbacks=Callbacks(on_progress=on_progress),  # 注册进度回调函数
)
```

### 日志回调

接收 MCP 服务器的日志消息：

```python
async def on_logging_message(
    params,  # 日志参数对象，包含级别、数据等信息
    context: CallbackContext,  # 回调上下文，包含服务器和工具信息
):
    """处理服务器日志
    
    这个回调函数演示了如何接收和处理 MCP 服务器的日志消息。
    可以根据日志级别进行不同的处理，如记录、过滤或告警。
    """
    # 获取日志级别，用于区分不同重要性的日志
    level = params.level  # 可选值: "debug", "info", "warning", "error"
    # 获取日志数据内容
    data = params.data

    # 格式化并输出日志信息，包含服务器名、级别和数据
    print(f"[{context.server_name}] {level.upper()}: {data}")

# 创建 MCP 客户端并注册日志回调
# 回调会在服务器发送日志时自动触发
client = MultiServerMCPClient(
    {...},  # MCP 服务器配置
    callbacks=Callbacks(on_logging_message=on_logging_message),  # 注册日志回调函数
)
```

### 征询回调 (Elicitation)

处理服务器请求用户输入的场景：

**服务器端**：

```python
# 导入 FastMCP 框架和上下文类型
from fastmcp import FastMCP, Context  # FastMCP 服务器和上下文
from pydantic import BaseModel  # 数据模型基类，用于数据验证

# 创建 MCP 服务器实例
mcp = FastMCP("ProfileService")  # 服务器名称

# 定义用户详细信息的数据模型
class UserDetails(BaseModel):
    email: str  # 用户邮箱地址
    age: int  # 用户年龄

@mcp.tool()  # 装饰器：注册为 MCP 工具
async def create_profile(name: str, ctx: Context) -> str:
    """创建用户资料，需要额外信息
    
    这个工具演示了 MCP 的征询（elicitation）机制。
    当工具需要额外信息时，可以向客户端请求数据输入，
    实现交互式的数据收集。
    """
    # 向客户端请求额外信息，使用征询机制
    result = await ctx.elicit(
        message=f"请提供 {name} 的详细信息：",  # 向用户显示的提示消息
        schema=UserDetails,  # 要求数据的结构定义
    )

    # 处理征询结果的不同情况
    if result.action == "accept" and result.data:
        # 用户接受并提供了数据
        return f"已为 {name} 创建资料: {result.data}"
    elif result.action == "decline":
        # 用户拒绝提供信息
        return f"用户拒绝提供信息"
    else:
        # 用户取消操作
        return f"操作已取消"
```

**客户端**：

```python
# 导入征询结果类型
from mcp.types import ElicitResult  # 征询操作的返回结果类型

async def on_elicitation(
    mcp_context,  # MCP 上下文对象
    params,  # 征询参数，包含消息等信息
    context: CallbackContext,  # 回调上下文，包含服务器和工具信息
):
    """处理征询请求
    
    这个回调函数演示了客户端如何处理服务器的征询请求。
    在实际应用中，这里会显示用户界面并收集输入，
    然后将结果返回给服务器。
    """
    # 在实际应用中，这里会提示用户输入
    # 这里只是示例，实际应该显示 UI 并收集用户输入
    print(f"服务器请求输入: {params.message}")

    # 返回用户输入的征询结果
    return ElicitResult(
        action="accept",  # 操作类型："accept", "decline", "cancel"
        content={  # 用户提供的具体数据
            "email": "user@example.com",  # 用户邮箱
            "age": 25,  # 用户年龄
        },
    )

# 创建 MCP 客户端并注册征询回调
# 回调会在服务器请求用户输入时自动触发
client = MultiServerMCPClient(
    {...},  # MCP 服务器配置
    callbacks=Callbacks(on_elicitation=on_elicitation),  # 注册征询回调函数
)
```

### 组合多个回调

```python
# 创建 MCP 客户端并组合多个回调
# 这样可以同时监控进度、处理日志和响应征询
client = MultiServerMCPClient(
    {...},  # MCP 服务器配置
    callbacks=Callbacks(
        on_progress=on_progress,  # 注册进度回调，监控长时间运行的工具
        on_logging_message=on_logging_message,  # 注册日志回调，接收服务器日志
        on_elicitation=on_elicitation,  # 注册征询回调，处理用户输入请求
    ),
)
```

---

## 创建 MCP 服务器

### 使用 FastMCP (Python)

FastMCP 是创建 MCP 服务器最简单的方式：

```python
from fastmcp import FastMCP

# 创建服务器
mcp = FastMCP("MyService")

# 定义工具
@mcp.tool()
def greet(name: str) -> str:
    """向用户问好"""
    return f"你好，{name}！"

@mcp.tool()
def calculate(expression: str) -> float:
    """计算数学表达式"""
    return eval(expression)  # 注意：生产环境请使用安全的表达式解析

# 定义资源
@mcp.resource("file://{path}")
def read_file(path: str) -> str:
    """读取文件内容"""
    with open(path, "r") as f:
        return f.read()

# 定义提示词
@mcp.prompt()
def summarize(text: str) -> str:
    """生成摘要提示词"""
    return f"请总结以下内容：\n\n{text}"

# 运行服务器
if __name__ == "__main__":
    mcp.run(transport="stdio")  # 或 "http"
```

### 完整的服务器示例

```python
from fastmcp import FastMCP, Context
from pydantic import BaseModel
from typing import Optional
import json

mcp = FastMCP("TodoService")

# 模拟数据库
todos = {}

class Todo(BaseModel):
    id: str
    title: str
    completed: bool = False
    priority: str = "medium"

# 工具：创建待办
@mcp.tool()
def create_todo(
    title: str,
    priority: str = "medium",
    ctx: Context = None,
) -> str:
    """创建新的待办事项"""
    import uuid
    todo_id = str(uuid.uuid4())[:8]

    todos[todo_id] = Todo(
        id=todo_id,
        title=title,
        priority=priority,
    )

    # 发送进度通知
    if ctx:
        ctx.report_progress(1, 1, f"已创建待办: {title}")

    return json.dumps({"id": todo_id, "message": f"已创建待办: {title}"})

# 工具：列出待办
@mcp.tool()
def list_todos(
    completed: Optional[bool] = None,
) -> str:
    """列出待办事项"""
    result = []
    for todo in todos.values():
        if completed is None or todo.completed == completed:
            result.append(todo.model_dump())

    return json.dumps(result, ensure_ascii=False)

# 工具：完成待办
@mcp.tool()
def complete_todo(todo_id: str) -> str:
    """标记待办为已完成"""
    if todo_id not in todos:
        return json.dumps({"error": f"未找到待办: {todo_id}"})

    todos[todo_id].completed = True
    return json.dumps({"message": f"待办 {todo_id} 已完成"})

# 工具：删除待办
@mcp.tool()
def delete_todo(todo_id: str) -> str:
    """删除待办事项"""
    if todo_id not in todos:
        return json.dumps({"error": f"未找到待办: {todo_id}"})

    del todos[todo_id]
    return json.dumps({"message": f"待办 {todo_id} 已删除"})

# 资源：获取所有待办
@mcp.resource("todo://all")
def get_all_todos() -> str:
    """获取所有待办的资源"""
    return json.dumps([t.model_dump() for t in todos.values()], ensure_ascii=False)

# 提示词：待办摘要
@mcp.prompt()
def todo_summary() -> str:
    """生成待办摘要的提示词"""
    pending = sum(1 for t in todos.values() if not t.completed)
    completed = sum(1 for t in todos.values() if t.completed)

    return f"""请根据以下待办统计生成摘要：
- 待完成: {pending} 项
- 已完成: {completed} 项
- 总计: {len(todos)} 项

请给出时间管理建议。"""

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### HTTP 服务器

```python
from fastmcp import FastMCP

mcp = FastMCP("HTTPService")

@mcp.tool()
def ping() -> str:
    """健康检查"""
    return "pong"

if __name__ == "__main__":
    # 运行 HTTP 服务器
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8000,
    )
```

---

## 实战案例

### 案例 1：多服务集成 Agent

```python
from dataclasses import dataclass
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.callbacks import Callbacks
from langchain.agents import create_agent

@dataclass
class AppContext:
    user_id: str
    api_keys: dict

# 进度回调
async def on_progress(progress, total, message, context):
    percent = (progress / total * 100) if total else progress
    print(f"[{context.server_name}] {percent:.0f}% - {message}")

# 认证拦截器
async def auth_interceptor(request, handler):
    ctx = request.runtime.context
    server = request.server_name

    # 注入对应服务的 API Key
    if server in ctx.api_keys:
        modified_args = {
            **request.args,
            "api_key": ctx.api_keys[server],
        }
        request = request.override(args=modified_args)

    return await handler(request)

# 创建客户端
client = MultiServerMCPClient(
    {
        "weather": {
            "transport": "http",
            "url": "http://weather-service:8000/mcp",
        },
        "calendar": {
            "transport": "http",
            "url": "http://calendar-service:8000/mcp",
        },
        "email": {
            "transport": "http",
            "url": "http://email-service:8000/mcp",
        },
    },
    tool_interceptors=[auth_interceptor],
    callbacks=Callbacks(on_progress=on_progress),
)

# 获取工具并创建 Agent
tools = await client.get_tools()
agent = create_agent(
    "gpt-4o",
    tools,
    context_schema=AppContext,
)

# 使用
response = await agent.ainvoke(
    {
        "messages": [{
            "role": "user",
            "content": "查看明天北京的天气，如果是晴天就帮我安排户外会议并发邮件通知参会者"
        }]
    },
    context=AppContext(
        user_id="user-123",
        api_keys={
            "weather": "weather-api-key",
            "calendar": "calendar-api-key",
            "email": "email-api-key",
        },
    ),
)
```

### 案例 2：带审批的敏感操作

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain.messages import ToolMessage
from langgraph.types import Command

# 定义敏感操作
SENSITIVE_TOOLS = {
    "delete_file": "删除文件",
    "send_email": "发送邮件",
    "make_payment": "执行付款",
}

# 审批状态
pending_approvals = {}

async def approval_interceptor(request: MCPToolCallRequest, handler):
    """敏感操作需要人工审批"""
    tool_name = request.name

    if tool_name in SENSITIVE_TOOLS:
        # 检查是否已批准
        approval_key = f"{request.runtime.thread_id}:{tool_name}"

        if approval_key not in pending_approvals:
            # 创建审批请求
            pending_approvals[approval_key] = {
                "tool": tool_name,
                "args": request.args,
                "status": "pending",
            }

            # 返回中断，等待审批
            return Command(
                update={
                    "pending_approval": {
                        "tool": tool_name,
                        "description": SENSITIVE_TOOLS[tool_name],
                        "args": request.args,
                    }
                },
                interrupt="等待人工审批",
            )

        approval = pending_approvals.pop(approval_key)

        if approval["status"] != "approved":
            return ToolMessage(
                content=f"操作已被拒绝: {SENSITIVE_TOOLS[tool_name]}",
                tool_call_id=request.runtime.tool_call_id,
            )

    # 执行工具
    return await handler(request)

client = MultiServerMCPClient(
    {...},
    tool_interceptors=[approval_interceptor],
)
```

### 案例 3：文件系统 Agent

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

# 使用官方文件系统 MCP 服务器
client = MultiServerMCPClient({
    "filesystem": {
        "transport": "stdio",
        "command": "npx",
        "args": [
            "-y",
            "@anthropic/mcp-server-filesystem",
            "/Users/username/Documents",  # 允许访问的目录
        ],
    }
})

tools = await client.get_tools()
agent = create_agent("gpt-4o", tools)

# 文件操作
response = await agent.ainvoke({
    "messages": [{
        "role": "user",
        "content": "列出 Documents 目录下的所有 Python 文件，并统计总行数"
    }]
})
```

---

## 最佳实践

### 1. 传输选择

```python
# ✅ 好的做法：根据场景选择传输
config = {
    # 本地工具用 stdio
    "local_tool": {"transport": "stdio", ...},

    # 远程服务用 HTTP
    "remote_service": {"transport": "http", ...},
}

# ❌ 不好的做法：所有服务都用同一种传输
```

### 2. 错误处理

```python
# ✅ 好的做法：在拦截器中处理错误
async def error_handling_interceptor(request, handler):
    try:
        return await handler(request)
    except ConnectionError:
        return ToolMessage(
            content="服务暂时不可用，请稍后重试",
            tool_call_id=request.runtime.tool_call_id,
        )
    except Exception as e:
        # 记录错误但不暴露内部细节
        logger.error(f"工具错误: {e}")
        return ToolMessage(
            content="操作失败，请联系管理员",
            tool_call_id=request.runtime.tool_call_id,
        )
```

### 3. 安全最佳实践

```python
# ✅ 好的做法：验证和清理输入
async def security_interceptor(request, handler):
    # 检查敏感参数
    if "password" in request.args:
        # 不要记录敏感信息
        sanitized_args = {**request.args, "password": "***"}
        logger.info(f"调用 {request.name}: {sanitized_args}")
    else:
        logger.info(f"调用 {request.name}: {request.args}")

    # 验证文件路径（防止路径遍历）
    if "path" in request.args:
        path = request.args["path"]
        if ".." in path or path.startswith("/"):
            return ToolMessage(
                content="无效的文件路径",
                tool_call_id=request.runtime.tool_call_id,
            )

    return await handler(request)
```

### 4. 性能优化

```python
# ✅ 好的做法：复用客户端连接
# 在应用启动时创建客户端
client = MultiServerMCPClient({...})

# 在请求处理中复用
async def handle_request(user_message):
    tools = await client.get_tools()
    agent = create_agent("gpt-4o", tools)
    return await agent.ainvoke({"messages": [user_message]})

# ❌ 不好的做法：每次请求都创建新客户端
async def handle_request_bad(user_message):
    client = MultiServerMCPClient({...})  # 每次都创建
    tools = await client.get_tools()
    ...
```

### 5. 测试 MCP 集成

```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_mcp_tool_integration():
    # 创建模拟的 MCP 客户端
    mock_client = AsyncMock()
    mock_client.get_tools.return_value = [
        MockTool(name="add", description="Add two numbers"),
    ]

    # 测试工具加载
    tools = await mock_client.get_tools()
    assert len(tools) == 1
    assert tools[0].name == "add"

@pytest.mark.asyncio
async def test_interceptor():
    # 测试拦截器逻辑
    request = MockMCPToolCallRequest(
        name="sensitive_tool",
        args={"data": "test"},
    )

    result = await my_interceptor(request, mock_handler)
    assert result is not None
```

---

## 快速参考

### 客户端配置

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "server_name": {
        # 传输配置
        "transport": "stdio" | "http" | "sse",

        # stdio 特定
        "command": "python",
        "args": ["server.py"],
        "env": {"KEY": "value"},
        "cwd": "/path",

        # http 特定
        "url": "http://localhost:8000/mcp",
        "headers": {"Authorization": "Bearer xxx"},
        "timeout": 30,
    }
})
```

### 核心 API

```python
# 获取工具
tools = await client.get_tools()

# 获取资源
blobs = await client.get_resources("server_name")
blobs = await client.get_resources("server_name", uris=["uri1", "uri2"])

# 获取提示词
messages = await client.get_prompt("server_name", "prompt_name")
messages = await client.get_prompt("server_name", "prompt_name", arguments={...})

# 有状态会话
async with client.session("server_name") as session:
    tools = await load_mcp_tools(session)
```

### 拦截器模板

```python
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

async def interceptor(request: MCPToolCallRequest, handler):
    # 前置处理
    # ...

    # 修改请求
    request = request.override(args={...})

    # 调用下一个处理器
    result = await handler(request)

    # 后置处理
    # ...

    return result
```

### 回调模板

```python
from langchain_mcp_adapters.callbacks import Callbacks, CallbackContext

callbacks = Callbacks(
    on_progress=async def(progress, total, message, context): ...,
    on_logging_message=async def(params, context): ...,
    on_elicitation=async def(mcp_context, params, context): ...,
)

client = MultiServerMCPClient({...}, callbacks=callbacks)
```

### FastMCP 服务器模板

```python
from fastmcp import FastMCP, Context

mcp = FastMCP("ServiceName")

@mcp.tool()
def my_tool(arg1: str, arg2: int = 0) -> str:
    """工具描述"""
    return "result"

@mcp.resource("uri://{param}")
def my_resource(param: str) -> str:
    """资源描述"""
    return "data"

@mcp.prompt()
def my_prompt(arg: str) -> str:
    """提示词描述"""
    return f"Prompt with {arg}"

if __name__ == "__main__":
    mcp.run(transport="stdio")  # 或 "http"
```

---

## 总结

**MCP (Model Context Protocol)** 是标准化 AI 应用与外部工具/服务集成的开放协议。

### 核心要点

1. **三大组件**
   - Tools：可执行的函数
   - Resources：可访问的数据
   - Prompts：可重用的提示词模板

2. **传输方式**
   - stdio：本地工具，简单易用
   - HTTP：远程服务，可扩展
   - SSE：实时流式更新

3. **会话管理**
   - 无状态（默认）：每次调用独立
   - 有状态：跨调用保持上下文

4. **拦截器**
   - 中间件式控制
   - 可访问运行时上下文
   - 支持认证、日志、重试等

5. **回调系统**
   - 进度监控
   - 日志接收
   - 用户输入征询

### 最佳实践

- 根据场景选择传输方式
- 使用拦截器处理横切关注点
- 复用客户端连接
- 做好错误处理和安全验证
- 充分测试 MCP 集成

通过 MCP，你可以轻松地将各种外部服务和工具集成到 LangChain Agent 中，构建功能强大的 AI 应用！
