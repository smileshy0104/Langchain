# LangChain Context Engineering（上下文工程）详细指南

## 目录

1. [概述](#概述)
2. [为什么需要上下文工程](#为什么需要上下文工程)
3. [Agent 循环与可控点](#agent-循环与可控点)
4. [三种上下文类型](#三种上下文类型)
5. [三类数据源](#三类数据源)
6. [Model Context：控制模型调用](#model-context控制模型调用)
   - [System Prompt](#system-prompt)
   - [Messages](#messages)
   - [Tools](#tools)
   - [Model](#model)
   - [Response Format](#response-format)
7. [Tool Context：工具读写上下文](#tool-context工具读写上下文)
8. [Life-cycle Context：生命周期中间件](#life-cycle-context生命周期中间件)
9. [瞬态与持久化更新](#瞬态与持久化更新)
10. [实战案例](#实战案例)
11. [最佳实践](#最佳实践)
12. [快速参考](#快速参考)

---

## 概述

**Context Engineering（上下文工程）** 是指：在正确的时间，以正确的格式，向 LLM 提供正确的信息和工具，让它能够可靠地完成任务。

很多 Agent 原型看起来能跑，但进入真实业务后会频繁失败。失败通常不是因为“Agent 框架不够复杂”，而是因为模型在关键调用中没有拿到正确上下文：缺少用户身份、权限、当前状态、长期记忆、可用工具、输出格式或业务约束。

> 上下文工程是 AI 工程师最核心的工作之一：你不是简单地“把所有内容都塞给模型”，而是设计哪些内容该出现、何时出现、以什么形式出现，以及哪些内容应该被持久化。

```text
┌─────────────────────────────────────────────────────────────┐
│                    Context Engineering                       │
│                                                             │
│   正确信息 + 正确工具 + 正确格式 + 正确时机 = 更可靠的 Agent │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ System Prompt│  │   Messages   │  │    Tools     │       │
│  │   行为指令    │  │   对话历史    │  │   行动能力    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│            ┌──────────────┐  ┌──────────────┐               │
│            │    Model     │  │ResponseFormat│               │
│            │   模型选择    │  │  输出结构     │               │
│            └──────────────┘  └──────────────┘               │
│                            ↓                                │
│                           LLM                               │
└─────────────────────────────────────────────────────────────┘
```

LangChain 的优势在于：它通过 **middleware（中间件）** 把上下文工程变成可组合的工程能力。你可以在模型调用前、模型调用后、工具调用前后、Agent 开始和结束时动态修改上下文、读写状态、注入长期记忆、过滤工具、选择模型或改变输出格式。

---

## 为什么需要上下文工程

### Agent 失败的两大原因

当 Agent 没有按预期执行时，通常有两个原因：

| 原因 | 说明 | 解决方向 |
|---|---|---|
| LLM 能力不足 | 模型本身无法推理或执行该任务 | 换更强模型、拆分任务、增加工具 |
| 上下文不正确 | 模型没有拿到完成任务所需的信息 | 做上下文工程 |

在真实系统中，第二类更常见。比如：

- 模型不知道用户是否已登录，却调用了私有数据工具。
- 模型不知道当前是生产环境，执行了危险操作。
- 模型没有看到用户上传文件摘要，回答偏离文件内容。
- 工具太多，模型选错工具或误用危险工具。
- 对话太长，关键信息被挤出上下文窗口。
- 输出格式没有约束，后续程序无法解析。

### 上下文工程的目标

上下文工程不是“给模型更多内容”，而是：

1. **减少无关内容**：避免工具、历史消息、背景资料过载。
2. **补齐关键内容**：让模型知道身份、权限、状态、约束、记忆。
3. **结构化呈现内容**：用清晰格式表达文件、规则、偏好和工具说明。
4. **控制持久化边界**：区分只影响本次调用的瞬态上下文和会影响后续回合的状态更新。
5. **在生命周期中介入**：摘要、护栏、日志、工具审批等逻辑不应全部塞进 prompt。

---

## Agent 循环与可控点

典型 Agent 循环由两步组成：

1. **Model call（模型调用）**：把提示词、消息历史、工具列表、响应格式等传给模型；模型返回最终回答或工具调用请求。
2. **Tool execution（工具执行）**：执行模型请求的工具，把工具结果返回给 Agent，继续下一轮模型调用。

```text
┌──────────────────────────────────────────────┐
│                 Agent Loop                   │
│                                              │
│      ┌───────────────┐                       │
│      │  Model Call   │                       │
│      │  模型调用      │                       │
│      └───────┬───────┘                       │
│              │                               │
│   ┌──────────┴──────────┐                    │
│   │                     │                    │
│   ↓                     ↓                    │
│ Final Response      Tool Calls               │
│ 最终响应             工具调用请求             │
│                         │                    │
│                         ↓                    │
│                 ┌───────────────┐            │
│                 │Tool Execution │            │
│                 │ 工具执行       │            │
│                 └───────┬───────┘            │
│                         │                    │
│                         ↓                    │
│                    Tool Results              │
│                    工具结果                   │
│                         │                    │
│                         └──── 回到模型调用     │
└──────────────────────────────────────────────┘
```

在这个循环里，你可以控制：

| 上下文类型 | 你控制什么 | 瞬态还是持久化 |
|---|---|---|
| **Model Context** | 进入模型调用的内容：系统提示词、消息、工具、模型、响应格式 | 瞬态为主 |
| **Tool Context** | 工具能读取什么、写入什么、返回什么 | 持久化为主 |
| **Life-cycle Context** | 模型调用和工具调用之间发生什么：摘要、护栏、日志、状态更新 | 持久化为主 |

---

## 三种上下文类型

### 1. Model Context（模型上下文）

Model Context 是单次模型调用看到的内容，通常是**瞬态**的：你可以临时修改它，而不改变保存到 State 中的内容。

| 组件 | 说明 | 示例 |
|---|---|---|
| System Prompt | 开发者给模型的基础行为指令 | “你是专业客服助手” |
| Messages | 发送给模型的消息列表 | 对话历史、文件摘要、临时规则 |
| Tools | 模型可用工具 | 搜索、订单查询、发邮件 |
| Model | 实际调用的模型和配置 | `gpt-5.5`、`claude-sonnet-4-6` |
| Response Format | 结构化输出 schema | Pydantic model、JSON schema |

适合在 `@dynamic_prompt` 或 `@wrap_model_call` 中动态控制。

### 2. Tool Context（工具上下文）

Tool Context 是工具执行时可以读取和写入的上下文。工具不只接收模型提供的参数，也可以访问 Runtime、State 和 Store。

| 能力 | 数据源 | 示例 |
|---|---|---|
| 读取用户身份 | Runtime Context | `runtime.context.user_id` |
| 读取当前会话状态 | State | `runtime.state.get("authenticated")` |
| 读取长期偏好 | Store | `runtime.store.get(("preferences",), user_id)` |
| 写入当前状态 | State | 返回 `Command(update={...})` |
| 写入长期记忆 | Store | `runtime.store.put(...)` |

### 3. Life-cycle Context（生命周期上下文）

Life-cycle Context 控制 Agent 生命周期中模型调用和工具执行之间发生的事情。它通常通过 middleware 钩子实现，并且会持久化影响后续回合。

常见用途：

- 长对话摘要（Summarization）
- Guardrails 护栏检查
- 日志、审计、指标埋点
- 工具调用前审批
- 状态字段更新
- 根据条件跳转到生命周期其他步骤

---

## 三类数据源

上下文工程经常围绕三个数据源展开：

| 数据源 | 又称 | 作用域 | 典型内容 |
|---|---|---|---|
| **Runtime Context** | 静态配置 | 单次调用 / 会话级 | 用户 ID、API Key、数据库连接、权限、环境配置 |
| **State** | 短期记忆 | 当前对话 / thread | messages、上传文件、认证状态、工具结果 |
| **Store** | 长期记忆 | 跨对话 | 用户偏好、历史洞察、长期记忆、行为模式 |

### Runtime Context

Runtime Context 是调用时注入的静态信息，适合放“本次调用期间稳定不变”的依赖。

```python
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent


@dataclass
class AppContext:
    user_id: str
    user_role: str
    deployment_env: str
    database: Any = None


agent = create_agent(
    model="gpt-5.5",
    tools=[...],
    context_schema=AppContext,
)

agent.invoke(
    {"messages": [{"role": "user", "content": "查询我的订单"}]},
    context=AppContext(
        user_id="user-123",
        user_role="admin",
        deployment_env="production",
        database=db_conn,
    ),
)
```

### State

State 是当前对话的短期记忆。最重要的字段通常是 `messages`，也可以包含业务自定义字段。

```python
state = {
    "messages": [...],
    "uploaded_files": [
        {"name": "report.pdf", "type": "pdf", "summary": "年度财务报告"},
    ],
    "authenticated": True,
    "current_order_id": "ORD-123",
}
```

### Store

Store 是跨对话的长期记忆，适合保存用户偏好、长期洞察、历史数据索引。

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

store.put(
    ("preferences",),
    "user-123",
    {"communication_style": "concise", "language": "zh"},
)

prefs = store.get(("preferences",), "user-123")
print(prefs.value)
```

---

## Model Context：控制模型调用

Model Context 决定模型在某一次调用中看到什么。它对可靠性、成本和延迟影响极大。

### System Prompt

System Prompt 决定模型的行为基线。它可以从 State、Store、Runtime Context 中读取信息动态生成。

#### 从 State 读取：对话长度感知

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt


@dynamic_prompt
def state_aware_prompt(request: ModelRequest) -> str:
    message_count = len(request.messages)

    base = "你是一个有帮助的助手。"

    if message_count > 10:
        base += "\n这是一个较长对话，请更加简洁，并优先引用已有上下文。"

    return base


agent = create_agent(
    model="gpt-5.5",
    tools=[...],
    middleware=[state_aware_prompt],
)
```

#### 从 Store 读取：用户偏好感知

```python
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langgraph.store.memory import InMemoryStore


@dataclass
class Context:
    user_id: str


@dynamic_prompt
def store_aware_prompt(request: ModelRequest) -> str:
    user_id = request.runtime.context.user_id
    store = request.runtime.store
    user_prefs = store.get(("preferences",), user_id)

    base = "你是一个有帮助的助手。"

    if user_prefs:
        style = user_prefs.value.get("communication_style", "balanced")
        base += f"\n用户偏好 {style} 风格的回复。"

    return base


agent = create_agent(
    model="gpt-5.5",
    tools=[...],
    middleware=[store_aware_prompt],
    context_schema=Context,
    store=InMemoryStore(),
)
```

#### 从 Runtime Context 读取：角色和环境感知

```python
from dataclasses import dataclass

from langchain.agents.middleware import ModelRequest, dynamic_prompt


@dataclass
class Context:
    user_role: str
    deployment_env: str


@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    ctx = request.runtime.context

    base = "你是一个有帮助的助手。"

    if ctx.user_role == "admin":
        base += "\n用户是管理员，可以执行管理操作，但高风险操作仍需确认。"
    elif ctx.user_role == "viewer":
        base += "\n用户只有只读权限，不要建议写入或删除操作。"

    if ctx.deployment_env == "production":
        base += "\n当前是生产环境，任何数据修改都必须谨慎。"

    return base
```

### Messages

Messages 是传给模型的消息列表。消息管理的关键是：**只把当前模型调用需要的信息放进去**。

常见策略：

- 注入上传文件摘要
- 注入用户写作风格
- 注入合规规则
- 注入当前任务上下文
- 修剪或摘要过长历史

#### 瞬态注入文件上下文

`wrap_model_call` 中使用 `request.override(messages=...)` 可以只影响当前模型调用，不把注入消息保存到 State。

```python
from typing import Callable

from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call


@wrap_model_call
def inject_file_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """临时注入本会话上传文件摘要。"""
    uploaded_files = request.state.get("uploaded_files", [])

    if uploaded_files:
        file_descriptions = [
            f"- {file['name']} ({file['type']}): {file['summary']}"
            for file in uploaded_files
        ]

        file_context = f"""本次对话可用文件：
{chr(10).join(file_descriptions)}

回答文件相关问题时，请优先参考这些文件。"""

        messages = [
            *request.messages,
            {"role": "user", "content": file_context},
        ]
        request = request.override(messages=messages)

    return handler(request)
```

#### 注入 Store 中的写作风格

```python
@wrap_model_call
def inject_writing_style(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    user_id = request.runtime.context.user_id
    writing_style = request.runtime.store.get(("writing_style",), user_id)

    if writing_style:
        style = writing_style.value
        style_context = f"""用户写作风格：
- 语气：{style.get('tone', 'professional')}
- 常用开头：{style.get('greeting', 'Hi')}
- 常用结尾：{style.get('sign_off', 'Best')}
- 示例邮件：
{style.get('example_email', '')}"""

        request = request.override(
            messages=[*request.messages, {"role": "user", "content": style_context}]
        )

    return handler(request)
```

#### 注入 Runtime Context 中的合规规则

```python
from dataclasses import dataclass


@dataclass
class Context:
    user_jurisdiction: str
    industry: str
    compliance_frameworks: list[str]


@wrap_model_call
def inject_compliance_rules(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    ctx = request.runtime.context

    rules = []
    if "GDPR" in ctx.compliance_frameworks:
        rules.append("- 处理个人数据前必须获得明确同意")
        rules.append("- 用户有权要求删除个人数据")
    if "HIPAA" in ctx.compliance_frameworks:
        rules.append("- 未授权不得分享患者健康信息")
    if ctx.industry == "finance":
        rules.append("- 不得在缺少免责声明时提供金融建议")

    if rules:
        compliance_context = f"""{ctx.user_jurisdiction} 的合规要求：
{chr(10).join(rules)}"""
        request = request.override(
            messages=[*request.messages, {"role": "user", "content": compliance_context}]
        )

    return handler(request)
```

> 注意：把动态信息追加到消息尾部通常比修改系统提示词更适合“本次调用临时上下文”。如果某条信息需要长期影响对话，应通过 State 或 Store 持久化，而不是每次临时拼接。

### Tools

工具定义本身也是上下文。工具的名称、描述、参数名和参数描述会直接影响模型是否会正确使用工具。

#### 定义清晰工具

```python
from langchain.tools import tool


@tool(parse_docstring=True)
def search_orders(user_id: str, status: str, limit: int = 10) -> str:
    """Search for user orders by status.

    Use this when the user asks about order history or wants to check
    order status. Always filter by the provided status.

    Args:
        user_id: Unique identifier for the user
        status: Order status: 'pending', 'shipped', or 'delivered'
        limit: Maximum number of results to return
    """
    ...
```

工具描述建议包含：

- 这个工具做什么
- 什么时候应该调用
- 什么时候不该调用
- 参数含义和允许值
- 高风险工具的约束条件

#### 基于 State 过滤工具

```python
@wrap_model_call
def state_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    is_authenticated = request.state.get("authenticated", False)
    message_count = len(request.state["messages"])

    if not is_authenticated:
        tools = [t for t in request.tools if t.name.startswith("public_")]
        request = request.override(tools=tools)
    elif message_count < 5:
        tools = [t for t in request.tools if t.name != "advanced_search"]
        request = request.override(tools=tools)

    return handler(request)
```

#### 基于 Runtime Context 过滤工具

```python
@dataclass
class Context:
    user_role: str


@wrap_model_call
def context_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    user_role = request.runtime.context.user_role

    if user_role == "admin":
        return handler(request)

    if user_role == "editor":
        tools = [t for t in request.tools if t.name != "delete_data"]
    else:
        tools = [t for t in request.tools if t.name.startswith("read_")]

    return handler(request.override(tools=tools))
```

#### 基于 Store 过滤工具

```python
@wrap_model_call
def store_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    user_id = request.runtime.context.user_id
    feature_flags = request.runtime.store.get(("features",), user_id)

    if feature_flags:
        enabled_tools = set(feature_flags.value.get("enabled_tools", []))
        tools = [t for t in request.tools if t.name in enabled_tools]
        request = request.override(tools=tools)

    return handler(request)
```

### Model

不同模型有不同成本、速度、上下文窗口和能力。上下文工程可以根据任务动态选择模型。

```python
from typing import Callable

from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.chat_models import init_chat_model

large_model = init_chat_model("claude-sonnet-4-6")
standard_model = init_chat_model("gpt-5.5")
efficient_model = init_chat_model("gpt-5.4-mini")


@wrap_model_call
def state_based_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    message_count = len(request.messages)

    if message_count > 20:
        model = large_model
    elif message_count > 10:
        model = standard_model
    else:
        model = efficient_model

    return handler(request.override(model=model))
```

基于 Runtime Context 的成本层级选择：

```python
@dataclass
class Context:
    cost_tier: str
    environment: str


premium_model = init_chat_model("claude-sonnet-4-6")
standard_model = init_chat_model("gpt-5.5")
budget_model = init_chat_model("gpt-5.4-mini")


@wrap_model_call
def context_based_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    ctx = request.runtime.context

    if ctx.environment == "production" and ctx.cost_tier == "premium":
        model = premium_model
    elif ctx.cost_tier == "budget":
        model = budget_model
    else:
        model = standard_model

    return handler(request.override(model=model))
```

### Response Format

结构化输出让模型最终回答符合指定 schema，便于下游系统解析。

```python
from pydantic import BaseModel, Field


class CustomerSupportTicket(BaseModel):
    """Structured ticket information extracted from customer message."""

    category: str = Field(description="Issue category: billing, technical, account, or product")
    priority: str = Field(description="Urgency level: low, medium, high, or critical")
    summary: str = Field(description="One-sentence summary of the customer's issue")
    customer_sentiment: str = Field(description="Customer's emotional tone")
```

动态选择响应格式：

```python
class SimpleResponse(BaseModel):
    answer: str = Field(description="A brief answer")


class DetailedResponse(BaseModel):
    answer: str = Field(description="A detailed answer")
    reasoning: str = Field(description="Explanation of reasoning")
    confidence: float = Field(description="Confidence score 0-1")


@wrap_model_call
def state_based_output(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    if len(request.messages) < 3:
        request = request.override(response_format=SimpleResponse)
    else:
        request = request.override(response_format=DetailedResponse)

    return handler(request)
```

---

## Tool Context：工具读写上下文

工具是上下文工程中最容易被低估的部分。工具不仅返回结果给模型，还可以读取 Runtime / State / Store，并写入 State / Store。

### 读取 State

```python
from langchain.tools import ToolRuntime, tool


@tool
def check_authentication(runtime: ToolRuntime) -> str:
    """Check if user is authenticated."""
    is_authenticated = runtime.state.get("authenticated", False)
    return "User is authenticated" if is_authenticated else "User is not authenticated"
```

### 读取 Store

```python
from dataclasses import dataclass


@dataclass
class Context:
    user_id: str


@tool
def get_preference(preference_key: str, runtime: ToolRuntime[Context]) -> str:
    """Get user preference from Store."""
    user_id = runtime.context.user_id
    existing_prefs = runtime.store.get(("preferences",), user_id)

    if not existing_prefs:
        return "No preferences found"

    value = existing_prefs.value.get(preference_key)
    return f"{preference_key}: {value}" if value else f"No preference set for {preference_key}"
```

### 读取 Runtime Context

```python
@dataclass
class Context:
    user_id: str
    api_key: str
    db_connection: str


@tool
def fetch_user_data(query: str, runtime: ToolRuntime[Context]) -> str:
    """Fetch data using Runtime Context configuration."""
    ctx = runtime.context
    results = perform_database_query(ctx.db_connection, query, ctx.api_key)
    return f"Found {len(results)} results for user {ctx.user_id}"
```

### 写入 State

工具可以返回 `Command(update={...})` 来更新 State。

```python
from langgraph.types import Command


@tool
def authenticate_user(password: str, runtime: ToolRuntime) -> Command:
    """Authenticate user and update State."""
    return Command(update={"authenticated": password == "correct"})
```

### 写入 Store

```python
@tool
def save_preference(
    preference_key: str,
    preference_value: str,
    runtime: ToolRuntime[Context],
) -> str:
    """Save user preference to Store."""
    user_id = runtime.context.user_id
    store = runtime.store

    existing_prefs = store.get(("preferences",), user_id)
    prefs = existing_prefs.value if existing_prefs else {}
    prefs[preference_key] = preference_value

    store.put(("preferences",), user_id, prefs)
    return f"Saved preference: {preference_key} = {preference_value}"
```

---

## Life-cycle Context：生命周期中间件

Life-cycle Context 控制核心 Agent 步骤之间发生的事情。它是跨切面能力的主要承载方式：摘要、护栏、日志、状态更新、跳转生命周期等。

```text
┌─────────────────────────────────────────────────────────────┐
│                     Middleware Hooks                         │
│                                                             │
│ before_agent                                                │
│      ↓                                                      │
│ before_model → wrap_model_call → after_model                │
│      ↓                         ↑                            │
│ wrap_tool_call  ← Tool execution                             │
│      ↓                                                      │
│ after_agent                                                 │
└─────────────────────────────────────────────────────────────┘
```

### SummarizationMiddleware

长对话最常见的生命周期模式是自动摘要。与 `wrap_model_call` 中临时裁剪消息不同，摘要会**持久化更新 State**：旧消息会被摘要替换，后续回合看到的是摘要后的历史。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="gpt-5.5",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger={"tokens": 4000},
            keep={"messages": 20},
        ),
    ],
)
```

摘要流程：

```text
对话超过触发条件
        ↓
使用独立模型总结旧消息
        ↓
用摘要消息替换旧消息（写入 State）
        ↓
保留最近消息，继续后续对话
```

适合摘要的场景：

- 多轮客服对话
- 长时间代码助手会话
- 多步骤数据分析
- 需要保留早期关键信息但不需要逐字历史的任务

### 生命周期中间件的其他用途

| 用途 | 推荐钩子 | 示例 |
|---|---|---|
| 输入护栏 | `before_agent` | 鉴权、速率限制、恶意请求阻断 |
| 模型调用日志 | `before_model` / `after_model` | 记录消息数、token、模型名 |
| 动态模型策略 | `wrap_model_call` | 按任务复杂度选择模型 |
| 工具审批 | `wrap_tool_call` | 删除、发送、转账前人工确认 |
| 最终审计 | `after_agent` | 记录最终响应、合规检查 |

---

## 瞬态与持久化更新

### 核心区别

| 类型 | 实现方式 | 是否写入 State | 影响范围 | 典型用途 |
|---|---|---:|---|---|
| 瞬态更新 | `wrap_model_call` + `request.override(...)` | 否 | 单次模型调用 | 临时注入消息、过滤工具、切换模型 |
| 持久化更新 | 生命周期钩子或工具返回 `Command` | 是 | 后续回合 | 摘要、认证状态、计数器、上传文件记录 |
| 长期持久化 | Store 写入 | 是，跨对话 | 跨会话 | 用户偏好、长期记忆、历史洞察 |

### 瞬态更新示例

```python
@wrap_model_call
def transient_modification(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    messages = [
        *request.messages,
        {"role": "user", "content": "临时要求：本次请用三句话以内回答。"},
    ]
    tools = [t for t in request.tools if t.name != "dangerous_tool"]

    return handler(request.override(messages=messages, tools=tools))
```

### 持久化更新示例

```python
from langchain.agents.middleware import AgentState, after_model
from langgraph.runtime import Runtime


@after_model
def count_model_calls(state: AgentState, runtime: Runtime) -> dict | None:
    call_count = state.get("model_call_count", 0) + 1
    return {"model_call_count": call_count}
```

选择指南：

```text
这个上下文是否应该影响后续回合？
  ├─ 否 → 用 wrap_model_call 做瞬态修改
  ├─ 是，但只在当前 thread 内有效 → 写入 State
  └─ 是，而且跨会话有效 → 写入 Store
```

---

## 实战案例

### 案例 1：智能客服系统

```python
from dataclasses import dataclass
from typing import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    SummarizationMiddleware,
    dynamic_prompt,
    wrap_model_call,
)
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field


@dataclass
class CustomerContext:
    customer_id: str
    customer_name: str
    subscription_tier: str  # free / pro / enterprise
    language: str = "zh"


@tool
def get_customer_orders(limit: int, runtime: ToolRuntime[CustomerContext]) -> str:
    """Get recent orders for the current customer."""
    customer_id = runtime.context.customer_id
    return f"客户 {customer_id} 最近 {limit} 个订单：[...]"


@tool
def create_support_ticket(issue: str, priority: str, runtime: ToolRuntime[CustomerContext]) -> str:
    """Create a customer support ticket."""
    ctx = runtime.context
    if ctx.subscription_tier == "enterprise" and priority == "normal":
        priority = "high"
    return f"已为 {ctx.customer_name} 创建工单，优先级：{priority}，问题：{issue}"


@dynamic_prompt
def customer_service_prompt(request: ModelRequest) -> str:
    ctx = request.runtime.context
    tier_guide = {
        "free": "提供基础支持，复杂问题建议升级套餐。",
        "pro": "提供专业技术支持。",
        "enterprise": "提供最高级别支持，优先处理，可承诺 SLA。",
    }
    language = "中文" if ctx.language == "zh" else "English"

    return f"""你是专业客服助手。

客户信息：
- 姓名：{ctx.customer_name}
- ID：{ctx.customer_id}
- 套餐：{ctx.subscription_tier}

服务指南：{tier_guide.get(ctx.subscription_tier, tier_guide['free'])}
请使用{language}回复。"""


enterprise_model = init_chat_model("gpt-5.5")
default_model = init_chat_model("gpt-5.4-mini")


@wrap_model_call
def tier_based_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    if request.runtime.context.subscription_tier == "enterprise":
        request = request.override(model=enterprise_model)
    else:
        request = request.override(model=default_model)
    return handler(request)


class TicketSummary(BaseModel):
    category: str = Field(description="billing / technical / account / product")
    priority: str = Field(description="low / medium / high / critical")
    summary: str = Field(description="One-sentence summary")


@wrap_model_call
def support_output_format(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    last_message = request.messages[-1].content if request.messages else ""
    if "工单" in str(last_message) or "ticket" in str(last_message).lower():
        request = request.override(response_format=TicketSummary)
    return handler(request)


customer_service_agent = create_agent(
    model="gpt-5.4-mini",
    tools=[get_customer_orders, create_support_ticket],
    middleware=[
        customer_service_prompt,
        tier_based_model,
        support_output_format,
        SummarizationMiddleware(
            model="gpt-5.4-mini",
            trigger={"tokens": 4000},
            keep={"messages": 15},
        ),
    ],
    context_schema=CustomerContext,
    checkpointer=InMemorySaver(),
    store=InMemoryStore(),
)
```

### 案例 2：自适应学习助手

```python
from dataclasses import dataclass
from typing import Literal

from langchain.agents.middleware import dynamic_prompt
from langchain.tools import ToolRuntime, tool


@dataclass
class LearnerContext:
    learner_id: str
    learner_name: str
    expertise_level: Literal["beginner", "intermediate", "expert"]
    learning_style: Literal["visual", "auditory", "reading", "kinesthetic"]
    preferred_language: str = "zh"


@tool
def save_learning_progress(
    topic: str,
    mastery_level: float,
    runtime: ToolRuntime[LearnerContext],
) -> str:
    """Save learning progress to long-term Store."""
    learner_id = runtime.context.learner_id
    runtime.store.put(
        ("learning_progress", learner_id),
        topic,
        {"mastery": mastery_level},
    )
    return f"已记录 {topic} 的学习进度：{mastery_level * 100:.0f}%"


@tool
def get_learning_history(topic: str, runtime: ToolRuntime[LearnerContext]) -> str:
    """Get previous learning progress from Store."""
    learner_id = runtime.context.learner_id
    progress = runtime.store.get(("learning_progress", learner_id), topic)

    if progress:
        return f"{topic} 历史掌握度：{progress.value['mastery'] * 100:.0f}%"
    return f"没有 {topic} 的学习记录"


@dynamic_prompt
def adaptive_learning_prompt(request: ModelRequest) -> str:
    ctx = request.runtime.context

    level_guide = {
        "beginner": "使用简单语言，避免术语，多举例说明。",
        "intermediate": "可以使用专业术语，但需要适当解释。",
        "expert": "可以进行深入技术讨论。",
    }
    style_guide = {
        "visual": "多使用图表、流程图、示意图描述。",
        "auditory": "使用对话式讲解，强调重点。",
        "reading": "提供详细文字说明和参考资料。",
        "kinesthetic": "设计动手练习和实践任务。",
    }

    return f"""你是 {ctx.learner_name} 的个人学习助手。

专业水平：{level_guide[ctx.expertise_level]}
学习风格：{style_guide[ctx.learning_style]}
回复语言：{ctx.preferred_language}

如果用户询问已学内容，优先调用学习历史工具。"""
```

---

## 最佳实践

1. **从简单开始**：先使用静态 prompt 和固定工具，只有出现明确问题时再加入动态上下文。
2. **区分瞬态与持久化**：临时注入用 `wrap_model_call`，需要后续回合记住的写入 State，需要跨会话记住的写入 Store。
3. **工具说明要写“何时调用”**：工具描述不是文档注释，而是模型选择工具的重要依据。
4. **不要给模型过多工具**：工具过多会增加上下文负担和误用概率，应按权限、阶段、任务动态过滤。
5. **动态 prompt 不要无限膨胀**：从 Store 读取记忆时要筛选，只注入与当前任务相关的内容。
6. **摘要是持久化操作**：Summarization 会替换历史消息，摘要质量会影响后续所有回合。
7. **监控上下文决策**：记录工具数量、消息长度、选择的模型、注入的上下文来源。
8. **测试每个上下文策略**：为角色、权限、长对话、未认证、不同输出格式写测试。
9. **保护敏感信息**：API Key 和数据库连接可以存在 Runtime Context，但不要注入到 messages 或模型可见文本中。
10. **文档化策略**：明确哪些上下文来自 State、Store、Runtime，哪些是瞬态，哪些会持久化。

---

## 快速参考

### 数据源访问方式

| 数据源 | 中间件访问 | 工具访问 | 生命周期 |
|---|---|---|---|
| Runtime Context | `request.runtime.context` | `runtime.context` | 单次调用 / 会话级 |
| State | `request.state` 或 hook 参数 `state` | `runtime.state` | 当前 thread |
| Store | `request.runtime.store` | `runtime.store` | 跨会话 |
| Messages | `request.messages` | 通常不直接访问 | 当前模型调用 |

### 常用中间件装饰器

```python
from langchain.agents.middleware import (
    after_agent,
    after_model,
    before_agent,
    before_model,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)
```

### `request.override` 可修改内容

```python
request = request.override(
    messages=new_messages,
    tools=filtered_tools,
    model=selected_model,
    response_format=MyResponseFormat,
)
```

### SummarizationMiddleware

```python
from langchain.agents.middleware import SummarizationMiddleware

SummarizationMiddleware(
    model="gpt-5.4-mini",
    trigger={"tokens": 4000},
    keep={"messages": 20},
)
```

### 工具读写上下文

```python
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command


@tool
def read_context(runtime: ToolRuntime) -> str:
    user_id = runtime.context.user_id
    authenticated = runtime.state.get("authenticated")
    prefs = runtime.store.get(("preferences",), user_id)
    return "..."


@tool
def write_state(runtime: ToolRuntime) -> Command:
    return Command(update={"authenticated": True})


@tool
def write_store(runtime: ToolRuntime) -> str:
    runtime.store.put(("preferences",), runtime.context.user_id, {"style": "concise"})
    return "saved"
```

---

## 总结

上下文工程是构建可靠 Agent 的核心能力。LangChain 将它拆成了清晰的工程面：

- **Model Context**：控制每次模型调用看到什么。
- **Tool Context**：控制工具能读取和写入什么。
- **Life-cycle Context**：控制模型和工具之间的生命周期逻辑。
- **Runtime Context / State / Store**：分别承载静态配置、短期记忆和长期记忆。
- **Middleware**：把这些能力连接起来，让上下文工程可组合、可测试、可维护。

最重要的原则是：不要把所有东西都塞进 prompt。把临时信息、短期状态、长期记忆、工具能力和生命周期逻辑放在合适的位置，Agent 才会更可靠、更可控，也更容易演进。
