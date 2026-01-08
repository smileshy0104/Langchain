# LangChain Context Engineering（上下文工程）详细指南

## 目录

1. [概述](#概述)
2. [为什么需要上下文工程](#为什么需要上下文工程)
3. [三种上下文类型](#三种上下文类型)
4. [数据源详解](#数据源详解)
5. [System Prompt 动态化](#system-prompt-动态化)
6. [Messages 消息注入](#messages-消息注入)
7. [Tools 动态工具选择](#tools-动态工具选择)
8. [Model 动态模型选择](#model-动态模型选择)
9. [Response Format 结构化输出](#response-format-结构化输出)
10. [工具的上下文读写](#工具的上下文读写)
11. [生命周期中间件](#生命周期中间件)
12. [瞬态与持久化更新](#瞬态与持久化更新)
13. [实战案例](#实战案例)
14. [最佳实践](#最佳实践)
15. [快速参考](#快速参考)

---

## 概述

### 什么是 Context Engineering（上下文工程）

**Context Engineering（上下文工程）** 是指"在正确的时间，以正确的格式，提供正确的信息和工具，使 LLM 能够成功完成任务"的过程。

这是 AI 工程师最核心的工作之一。

```
┌─────────────────────────────────────────────────────────────┐
│                    Context Engineering                       │
│                                                             │
│    正确的信息 + 正确的格式 + 正确的时机 = LLM 成功完成任务   │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Prompt    │  │  Messages   │  │   Tools     │         │
│  │  系统提示词  │  │   对话历史   │  │   可用工具   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                           ↓                                 │
│                    ┌───────────┐                           │
│                    │    LLM    │                           │
│                    └───────────┘                           │
│                           ↓                                 │
│                    ┌───────────┐                           │
│                    │   输出    │                           │
│                    └───────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### 核心理念

上下文工程的本质是：**让 LLM 在每次调用时都能获得完成任务所需的最佳上下文**。

这包括：
- **动态调整 System Prompt** - 根据用户角色、场景定制提示词
- **智能管理消息历史** - 修剪、总结、注入相关信息
- **按需提供工具** - 根据权限和场景动态选择工具
- **选择合适的模型** - 根据任务复杂度选择最佳模型
- **定义输出格式** - 根据需求定制响应结构

---

## 为什么需要上下文工程

### Agent 失败的两大原因

当 AI Agent 表现不佳时，通常是以下两个原因之一：

| 原因 | 说明 | 解决方案 |
|------|------|----------|
| **LLM 能力不足** | 模型本身无法完成任务 | 换用更强大的模型 |
| **上下文不正确** | 没有传递"正确"的信息给 LLM | **上下文工程** |

**重要发现**：大多数 Agent 失败是因为上下文问题，而非模型能力问题。

### 典型的 Agent 循环

```
┌─────────────────────────────────────────────────────────────┐
│                      Agent 循环                              │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  步骤 1: 模型调用                                    │   │
│  │                                                     │   │
│  │  输入:                                              │   │
│  │  • System Prompt (系统提示词)                       │   │
│  │  • Messages (消息历史)                              │   │
│  │  • Tools (可用工具)                                 │   │
│  │  • Model Config (模型配置)                          │   │
│  │                                                     │   │
│  │  输出:                                              │   │
│  │  • 直接响应 或 工具调用请求                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  步骤 2: 工具执行 (如果需要)                         │   │
│  │                                                     │   │
│  │  • 执行 LLM 请求的工具                              │   │
│  │  • 返回工具结果                                     │   │
│  │  • 继续循环或结束                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│                    继续循环 或 返回最终响应                   │
└─────────────────────────────────────────────────────────────┘
```

**上下文工程的目标**：在这个循环的每个阶段，确保 LLM 获得最佳上下文。

---

## 三种上下文类型

LangChain 将上下文分为三种类型：

### 1. Model Context（模型上下文）- 瞬态

每次模型调用时发送给 LLM 的内容，是**瞬态的**（临时的）。

| 组件 | 说明 | 示例 |
|------|------|------|
| **System Prompt** | 开发者给 LLM 的基础指令 | "你是一个专业的客服助手" |
| **Messages** | 完整的对话历史 | 用户消息、AI 响应、工具结果 |
| **Tools** | Agent 可访问的工具列表 | 搜索、计算、发邮件等 |
| **Model** | 调用的模型及配置 | gpt-4o、claude-sonnet-4-5-20250929 等 |
| **Response Format** | 响应的结构定义 | JSON Schema、Pydantic Model |

```python
# 模型上下文示例
model_context = {
    "system_prompt": "你是一个专业的客服助手。",
    "messages": [
        {"role": "user", "content": "我想查询订单"},
        {"role": "assistant", "content": "好的，请提供订单号"},
    ],
    "tools": [search_order, track_shipping],
    "model": "gpt-4o",
    "response_format": OrderResponse,
}
```

### 2. Tool Context（工具上下文）- 持久化

工具执行时可以读取和写入的内容，是**持久化的**。

**读取来源**：
- State（当前会话状态）
- Store（跨会话存储）
- Runtime Context（运行时配置）

**写入目标**：
- State（更新当前状态）
- Store（保存长期数据）

```python
from langchain.tools import tool, ToolRuntime

@tool
def get_user_orders(runtime: ToolRuntime) -> str:
    """工具可以读取上下文"""
    # 从 Runtime Context 读取用户 ID
    user_id = runtime.context.user_id

    # 从 Store 读取用户偏好
    prefs = runtime.store.get(("preferences",), user_id)

    # 执行业务逻辑...
    return "订单列表..."

@tool
def save_preference(key: str, value: str, runtime: ToolRuntime) -> str:
    """工具可以写入上下文"""
    user_id = runtime.context.user_id

    # 写入 Store（持久化）
    runtime.store.put(("preferences",), user_id, {key: value})

    return f"已保存: {key}={value}"
```

### 3. Life-cycle Context（生命周期上下文）- 持久化

控制模型调用和工具执行之间发生的事情，是**持久化的**。

**主要用途**：
- 消息摘要总结
- 护栏检查
- 日志记录
- 状态转换

```python
from langchain.agents.middleware import SummarizationMiddleware

# 生命周期中间件示例：自动总结长对话
agent = create_agent(
    model="gpt-4o",
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),  # 超过 4000 tokens 时触发
            keep=("messages", 20),      # 保留最近 20 条消息
        ),
    ],
)
```

### 三种上下文对比

| 类型 | 生命周期 | 作用范围 | 典型操作 |
|------|----------|----------|----------|
| Model Context | 瞬态 | 单次模型调用 | 动态 prompt、工具过滤、模型选择 |
| Tool Context | 持久化 | 工具执行 | 读写 State/Store |
| Life-cycle Context | 持久化 | 调用之间 | 摘要、护栏、日志 |

---

## 数据源详解

上下文工程涉及三个核心数据源：

### 1. Runtime Context（运行时上下文）

**别名**：静态配置
**作用域**：会话范围（调用期间不变）
**用途**：存储静态的配置和依赖

```python
from dataclasses import dataclass

@dataclass
class AppContext:
    """运行时上下文定义"""
    user_id: str           # 用户标识
    user_name: str         # 用户名称
    user_role: str         # 用户角色（admin/user/guest）
    api_key: str           # API 密钥
    database: any          # 数据库连接
    environment: str       # 环境（production/staging/dev）

# 调用时传入
agent.invoke(
    {"messages": [...]},
    context=AppContext(
        user_id="user-123",
        user_name="张三",
        user_role="admin",
        api_key="sk-xxx",
        database=db_conn,
        environment="production",
    )
)
```

**典型内容**：
- 用户 ID、用户名、角色
- API 密钥、认证令牌
- 数据库连接、缓存连接
- 环境配置、功能开关
- 权限列表

### 2. State（状态）

**别名**：短期记忆
**作用域**：会话范围
**用途**：存储当前会话的动态数据

```python
# State 内容示例
state = {
    "messages": [           # 对话历史
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
    ],
    "uploaded_files": [     # 用户上传的文件
        {"name": "report.pdf", "type": "pdf", "summary": "年度报告"},
    ],
    "authenticated": True,  # 认证状态
    "current_order": "ORD-123",  # 当前操作的订单
}
```

**典型内容**：
- 消息历史（messages）
- 上传的文件和图片
- 认证状态
- 工具执行结果
- 临时计算数据

### 3. Store（存储）

**别名**：长期记忆
**作用域**：跨会话
**用途**：持久化存储用户数据和洞察

```python
# Store 使用示例
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# 保存用户偏好
store.put(
    ("users", "user-123", "preferences"),
    "theme",
    {"value": "dark", "updated_at": "2024-01-01"}
)

# 保存用户历史见解
store.put(
    ("users", "user-123", "insights"),
    "communication_style",
    {"value": "formal", "confidence": 0.9}
)

# 读取数据
prefs = store.get(("users", "user-123", "preferences"), "theme")
print(prefs.value)  # {"value": "dark", "updated_at": "2024-01-01"}
```

**典型内容**：
- 用户偏好设置
- 提取的用户见解
- 历史交互记录
- 学习到的用户特征
- 跨会话的上下文

### 数据源对比

| 数据源 | 别名 | 生命周期 | 可变性 | 典型用途 |
|--------|------|----------|--------|----------|
| Runtime Context | 静态配置 | 单次调用 | 不可变 | 用户 ID、API 密钥、连接 |
| State | 短期记忆 | 会话范围 | 可变 | 消息历史、临时数据 |
| Store | 长期记忆 | 跨会话 | 可变 | 用户偏好、历史见解 |

---

## System Prompt 动态化

### 为什么需要动态 Prompt

静态的 System Prompt 无法适应所有场景：

| 场景 | 静态 Prompt 问题 | 动态 Prompt 解决方案 |
|------|------------------|----------------------|
| 多角色用户 | 同一提示词服务所有用户 | 根据角色定制提示词 |
| 多语言支持 | 写死一种语言 | 根据用户语言切换 |
| 上下文感知 | 不知道对话进展 | 根据对话长度调整 |
| 个性化 | 无法个性化 | 从 Store 读取偏好 |

### 使用 @dynamic_prompt 装饰器

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dataclass
class Context:
    user_name: str
    user_role: str
    language: str = "zh"

@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    """根据用户信息生成个性化提示词"""
    ctx = request.runtime.context

    # 基础提示词
    base = f"你是一个专业的助手。用户名是 {ctx.user_name}。"

    # 根据角色添加指令
    if ctx.user_role == "admin":
        base += "\n用户是管理员，可以执行所有操作。"
    elif ctx.user_role == "guest":
        base += "\n用户是访客，只能进行查询操作。"
    else:
        base += "\n用户是普通用户。"

    # 根据语言调整
    if ctx.language == "en":
        base += "\nPlease respond in English."
    else:
        base += "\n请用中文回复。"

    return base

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[personalized_prompt],
    context_schema=Context,
)

# 使用
agent.invoke(
    {"messages": [{"role": "user", "content": "帮我查询订单"}]},
    context=Context(user_name="张三", user_role="admin", language="zh"),
)
```

### 从不同数据源读取

#### 从 State 读取

```python
@dynamic_prompt
def state_aware_prompt(request: ModelRequest) -> str:
    """根据对话状态调整提示词"""
    messages = request.messages  # 等同于 request.state["messages"]
    message_count = len(messages)

    base = "你是一个专业的助手。"

    if message_count > 20:
        base += "\n这是一个长对话，请保持简洁。"
    elif message_count > 10:
        base += "\n对话已有一定进展，可以引用之前的上下文。"
    else:
        base += "\n这是对话开始，请友好地打招呼。"

    # 检查是否有上传文件
    uploaded_files = request.state.get("uploaded_files", [])
    if uploaded_files:
        file_names = [f["name"] for f in uploaded_files]
        base += f"\n用户上传了以下文件: {', '.join(file_names)}"

    return base
```

#### 从 Store 读取

```python
@dynamic_prompt
def store_aware_prompt(request: ModelRequest) -> str:
    """从长期记忆读取用户偏好"""
    user_id = request.runtime.context.user_id
    store = request.runtime.store

    base = "你是一个专业的助手。"

    # 读取用户偏好
    user_prefs = store.get(("preferences",), user_id)
    if user_prefs:
        style = user_prefs.value.get("communication_style", "balanced")
        tone = user_prefs.value.get("tone", "professional")
        base += f"\n用户偏好 {style} 风格和 {tone} 语气。"

    # 读取历史见解
    insights = store.get(("insights",), user_id)
    if insights:
        expertise = insights.value.get("expertise_level", "intermediate")
        base += f"\n用户专业水平: {expertise}。"

    return base
```

#### 从 Runtime Context 读取

```python
@dataclass
class Context:
    user_role: str
    deployment_env: str
    feature_flags: dict

@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    """根据运行时上下文调整"""
    ctx = request.runtime.context

    base = "你是一个专业的助手。"

    # 根据用户角色
    if ctx.user_role == "admin":
        base += "\n你有管理员权限，可以执行敏感操作。"
    elif ctx.user_role == "viewer":
        base += "\n你只有查看权限，请引导用户使用只读操作。"

    # 根据部署环境
    if ctx.deployment_env == "production":
        base += "\n这是生产环境，请格外小心数据操作。"
    elif ctx.deployment_env == "staging":
        base += "\n这是测试环境，可以自由测试。"

    # 根据功能开关
    if ctx.feature_flags.get("enable_experimental"):
        base += "\n实验功能已启用。"

    return base
```

---

## Messages 消息注入

### 为什么需要消息注入

有时我们需要在发送给模型的消息中临时注入额外信息，但不想永久保存到对话历史中。

**典型场景**：
- 注入用户上传文件的摘要
- 注入用户的写作风格
- 注入相关的背景知识
- 注入实时数据

### 使用 @wrap_model_call 注入消息

```python
# 导入必要的模块
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse  # 模型包装装饰器和类型
from typing import Callable  # 可调用对象类型注解

@wrap_model_call  # 装饰器：包装模型调用，在调用前后执行自定义逻辑
def inject_file_context(
    request: ModelRequest,  # 模型请求对象，包含消息、状态、上下文等信息
    handler: Callable[[ModelRequest], ModelResponse]  # 处理函数，实际的模型调用
) -> ModelResponse:
    """注入用户上传文件的上下文
    
    这个中间件函数演示了如何从请求状态中获取用户上传的文件信息，
    并在模型调用前将文件上下文注入到消息中。这样可以让模型
    基于用户上传的文件内容来回答问题，实现文件驱动的对话。
    """

    # 从请求状态中获取用户上传的文件列表
    # 使用 get 方法提供默认值，避免键不存在时的错误
    uploaded_files = request.state.get("uploaded_files", [])

    # 检查是否有上传的文件
    if uploaded_files:
        # 构建文件描述列表，为每个文件创建可读的描述信息
        file_descriptions = []
        for file in uploaded_files:
            # 为每个文件创建格式化的描述字符串
            file_descriptions.append(
                f"- {file['name']} ({file['type']}): {file['summary']}"
                # 文件名、文件类型、文件摘要的格式化显示
            )

        # 构建文件上下文提示词，告知模型可以访问的文件信息
        file_context = f"""你可以访问以下用户上传的文件:
{chr(10).join(file_descriptions)}  # 使用换行符连接文件描述

请根据这些文件内容回答用户的问题。"""

        # 创建新的消息列表：保留原有消息 + 注入文件上下文
        messages = [
            *request.messages,  # 解包原有消息，保持对话历史
            {"role": "user", "content": file_context},  # 添加用户级文件上下文信息
        ]

        # 使用 override 创建新的请求对象（瞬态修改）
        # 这会创建一个新的请求对象，包含修改后的消息列表
        request = request.override(messages=messages)

    # 调用处理函数执行实际的模型调用
    # 如果没有上传文件，直接使用原始请求
    return handler(request)
```

### 从 Store 注入用户偏好

```python
@wrap_model_call  # 装饰器：包装模型调用，在调用前后执行自定义逻辑
def inject_writing_style(
    request: ModelRequest,  # 模型请求对象，包含消息、上下文等信息
    handler: Callable[[ModelRequest], ModelResponse]  # 处理函数，实际的模型调用
) -> ModelResponse:
    """从 Store 读取并注入用户的写作风格
    
    这个中间件函数演示了如何从持久化存储中读取用户偏好，
    并在模型调用前动态注入个性化的写作风格指令。
    这样可以让每个用户都获得符合其偏好的个性化体验。
    """

    # 从运行时上下文获取用户 ID - 确定当前会话的用户身份
    user_id = request.runtime.context.user_id
    # 从运行时获取 Store 组件 - 用于访问持久化存储
    store = request.runtime.store

    # 从 Store 读取用户的写作风格设置
    # 使用元组 ("writing_style",) 作为键，user_id 作为值进行查询
    writing_style = store.get(("writing_style",), user_id)

    # 检查是否找到了用户的写作风格设置
    if writing_style:
        # 获取存储的风格配置对象
        style = writing_style.value
        # 构建个性化的写作风格指令字符串
        style_context = f"""请使用以下写作风格回复:
- 语气: {style.get('tone', '专业')}  # 语气：专业、友好、幽默等
- 开场白: "{style.get('greeting', '你好')}"  # 个性化开场白
- 结束语: "{style.get('sign_off', '祝好')}"  # 个性化结束语
- 正式程度: {style.get('formality', '中等')}"""  # 正式程度：低、中等、高

        # 构建新的消息列表：保留原有消息 + 添加系统级风格指令
        messages = [
            *request.messages,  # 解包原有消息，保持对话历史
            {"role": "system", "content": style_context}  # 添加系统级风格指令
        ]

        # 使用 override 创建新的请求对象（瞬态修改）
        # 这会创建一个新的请求对象，包含修改后的消息列表
        request = request.override(messages=messages)

    # 调用处理函数执行实际的模型调用
    # 如果没有找到用户风格设置，直接使用原始请求
    return handler(request)
```

### 注入实时数据

```python
# 导入日期时间模块
import datetime  # 用于获取和处理实时时间信息

@wrap_model_call  # 装饰器：包装模型调用，在调用前后执行自定义逻辑
def inject_realtime_context(
    request: ModelRequest,  # 模型请求对象，包含消息、状态、上下文等信息
    handler: Callable[[ModelRequest], ModelResponse]  # 处理函数，实际的模型调用
) -> ModelResponse:
    """注入实时上下文信息
    
    这个中间件函数演示了如何在模型调用前注入实时的上下文信息，
    如当前时间、星期、工作时段等。这样可以让模型根据时间
    提供更相关、更个性化的响应。
    """

    # 获取当前时间对象，包含完整的日期和时间信息
    now = datetime.datetime.now()

    # 构建实时上下文信息字符串，为模型提供时间相关的背景
    realtime_context = f"""当前上下文信息:
- 当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}  # 格式化显示年月日时分秒
- 星期: {['周一', '周二', '周三', '周四', '周五', '周六', '周日'][now.weekday()]}  # 将数字星期转换为中文
- 时段: {'工作时间' if 9 <= now.hour < 18 else '非工作时间'}  # 根据小时判断工作时段"""

    # 构建新的消息列表：实时上下文 + 原有消息
    messages = [
        {"role": "system", "content": realtime_context},  # 添加系统级实时上下文信息
        *request.messages,  # 解包原有消息，保持对话历史
    ]

    # 使用 override 创建新的请求对象（瞬态修改）
    # 这会创建一个新的请求对象，包含修改后的消息列表
    request = request.override(messages=messages)
    
    # 调用处理函数执行实际的模型调用
    return handler(request)
```

---

## Tools 动态工具选择

### 为什么需要动态工具

不同场景下，Agent 需要访问不同的工具：

| 场景 | 工具选择策略 |
|------|--------------|
| 未认证用户 | 只提供公开工具 |
| 管理员用户 | 提供所有工具 |
| 特定任务 | 只提供相关工具 |
| 敏感操作 | 隐藏危险工具 |

### 基于 State 选择工具

```python
@wrap_model_call
def state_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """根据认证状态选择工具"""

    state = request.state
    is_authenticated = state.get("authenticated", False)

    if not is_authenticated:
        # 未认证：只提供公开工具
        public_tools = [t for t in request.tools if t.name.startswith("public_")]
        request = request.override(tools=public_tools)
    else:
        # 已认证：提供所有工具
        pass

    return handler(request)
```

### 基于 Runtime Context 选择工具

```python
@dataclass
class Context:
    user_role: str  # "admin", "editor", "viewer"

@wrap_model_call
def role_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """根据用户角色选择工具"""

    user_role = request.runtime.context.user_role
    all_tools = request.tools

    if user_role == "admin":
        # 管理员：所有工具
        tools = all_tools
    elif user_role == "editor":
        # 编辑者：排除删除工具
        tools = [t for t in all_tools if not t.name.startswith("delete_")]
    else:
        # 查看者：只有读取工具
        tools = [t for t in all_tools if t.name.startswith("read_") or t.name.startswith("get_")]

    request = request.override(tools=tools)
    return handler(request)
```

### 基于对话内容选择工具

```python
@wrap_model_call
def context_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """根据对话内容智能选择工具"""

    # 获取最后一条用户消息
    last_user_msg = ""
    for msg in reversed(request.messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "").lower()
            break

    all_tools = request.tools

    # 根据关键词选择相关工具
    if "订单" in last_user_msg or "购买" in last_user_msg:
        tools = [t for t in all_tools if "order" in t.name or "purchase" in t.name]
    elif "账户" in last_user_msg or "密码" in last_user_msg:
        tools = [t for t in all_tools if "account" in t.name or "auth" in t.name]
    elif "报告" in last_user_msg or "统计" in last_user_msg:
        tools = [t for t in all_tools if "report" in t.name or "stats" in t.name]
    else:
        # 默认提供所有工具
        tools = all_tools

    request = request.override(tools=tools)
    return handler(request)
```

---

## Model 动态模型选择

### 为什么需要动态模型选择

不同任务需要不同能力的模型：

| 场景 | 推荐模型 | 原因 |
|------|----------|------|
| 简单问答 | gpt-4o-mini | 成本低、速度快 |
| 复杂推理 | gpt-4o / claude-sonnet-4-5-20250929 | 能力强 |
| 长对话 | claude-sonnet-4-5-20250929 | 上下文窗口大 |
| 代码生成 | gpt-4o | 编程能力强 |

### 基于对话长度选择模型

```python
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

# 预先初始化模型（避免每次调用都初始化）
large_model = init_chat_model("claude-sonnet-4-5-20250929")  # 大上下文
standard_model = init_chat_model("gpt-4o")             # 标准能力
efficient_model = init_chat_model("gpt-4o-mini")       # 高效低成本

@wrap_model_call
def conversation_length_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """根据对话长度选择模型"""

    message_count = len(request.messages)

    if message_count > 20:
        # 长对话：使用大上下文模型
        model = large_model
        print(f"使用大上下文模型 (消息数: {message_count})")
    elif message_count > 10:
        # 中等对话：使用标准模型
        model = standard_model
        print(f"使用标准模型 (消息数: {message_count})")
    else:
        # 短对话：使用高效模型
        model = efficient_model
        print(f"使用高效模型 (消息数: {message_count})")

    request = request.override(model=model)
    return handler(request)
```

### 基于用户层级选择模型

```python
@dataclass
class Context:
    cost_tier: str       # "premium", "standard", "budget"
    environment: str     # "production", "staging"

@wrap_model_call
def tier_based_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """根据用户订阅层级选择模型"""

    ctx = request.runtime.context
    cost_tier = ctx.cost_tier
    environment = ctx.environment

    if environment == "production" and cost_tier == "premium":
        # 生产环境 + 高级用户：最强模型
        model = init_chat_model("claude-sonnet-4-5-20250929")
    elif cost_tier == "budget":
        # 预算层级：经济模型
        model = init_chat_model("gpt-4o-mini")
    else:
        # 标准层级
        model = init_chat_model("gpt-4o")

    request = request.override(model=model)
    return handler(request)
```

### 基于任务类型选择模型

```python
@wrap_model_call
def task_based_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """根据任务类型选择最适合的模型"""

    # 分析最后一条用户消息
    last_msg = request.messages[-1].get("content", "").lower()

    # 代码相关任务
    if any(kw in last_msg for kw in ["代码", "编程", "函数", "debug", "code"]):
        model = init_chat_model("gpt-4o")  # GPT-4o 编程能力强

    # 创意写作任务
    elif any(kw in last_msg for kw in ["写作", "故事", "诗", "创意", "文案"]):
        model = init_chat_model("claude-sonnet-4-5-20250929")  # Claude 创意能力强

    # 数学/逻辑任务
    elif any(kw in last_msg for kw in ["计算", "数学", "推理", "分析"]):
        model = init_chat_model("gpt-4o")

    # 简单问答
    else:
        model = init_chat_model("gpt-4o-mini")

    request = request.override(model=model)
    return handler(request)
```

---

## Response Format 结构化输出

### 为什么需要结构化输出

让 LLM 返回结构化数据而非自由文本，便于：
- 程序解析和处理
- 数据验证
- 界面展示
- 后续流程集成

### 定义输出格式

使用 Pydantic 定义结构化输出：

```python
from pydantic import BaseModel, Field
from typing import Literal

class CustomerSupportTicket(BaseModel):
    """客户支持工单结构"""

    category: Literal["billing", "technical", "account", "product"] = Field(
        description="问题类别"
    )
    priority: Literal["low", "medium", "high", "critical"] = Field(
        description="紧急程度"
    )
    summary: str = Field(
        description="问题摘要（一句话）"
    )
    customer_sentiment: Literal["frustrated", "neutral", "satisfied"] = Field(
        description="客户情绪"
    )
    suggested_actions: list[str] = Field(
        description="建议的处理步骤"
    )

class OrderAnalysis(BaseModel):
    """订单分析结构"""

    order_id: str = Field(description="订单编号")
    status: str = Field(description="订单状态")
    issues: list[str] = Field(description="发现的问题")
    recommendations: list[str] = Field(description="处理建议")
    estimated_resolution_time: str = Field(description="预计解决时间")
```

### 动态选择输出格式

```python
class SimpleResponse(BaseModel):
    """简单响应（对话初期）"""
    answer: str = Field(description="简要回答")

class DetailedResponse(BaseModel):
    """详细响应（深入对话）"""
    answer: str = Field(description="详细回答")
    reasoning: str = Field(description="推理过程")
    confidence: float = Field(description="置信度 (0-1)")
    follow_up_questions: list[str] = Field(description="追问建议")

@wrap_model_call
def adaptive_output_format(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """根据对话阶段选择输出格式"""

    message_count = len(request.messages)

    if message_count < 3:
        # 对话初期：简单响应
        request = request.override(response_format=SimpleResponse)
    else:
        # 深入对话：详细响应
        request = request.override(response_format=DetailedResponse)

    return handler(request)
```

### 根据任务类型选择格式

```python
class SearchResult(BaseModel):
    """搜索结果格式"""
    results: list[dict] = Field(description="搜索结果列表")
    total_count: int = Field(description="总数")
    query_interpretation: str = Field(description="查询理解")

class AnalysisResult(BaseModel):
    """分析结果格式"""
    findings: list[str] = Field(description="发现")
    conclusion: str = Field(description="结论")
    data_quality: str = Field(description="数据质量评估")

class ActionResult(BaseModel):
    """操作结果格式"""
    success: bool = Field(description="是否成功")
    action_taken: str = Field(description="执行的操作")
    next_steps: list[str] = Field(description="后续步骤")

@wrap_model_call
def task_based_format(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """根据任务类型选择输出格式"""

    last_msg = request.messages[-1].get("content", "").lower()

    if any(kw in last_msg for kw in ["搜索", "查找", "查询"]):
        request = request.override(response_format=SearchResult)
    elif any(kw in last_msg for kw in ["分析", "评估", "总结"]):
        request = request.override(response_format=AnalysisResult)
    elif any(kw in last_msg for kw in ["执行", "操作", "处理"]):
        request = request.override(response_format=ActionResult)

    return handler(request)
```

---

## 工具的上下文读写

### 工具读取上下文

工具可以通过 `ToolRuntime` 访问各种上下文数据：

```python
from langchain.tools import tool, ToolRuntime

@tool
def get_user_preferences(
    category: str,
    runtime: ToolRuntime
) -> str:
    """从 Store 读取用户偏好"""

    # 从 Runtime Context 获取用户 ID
    user_id = runtime.context.user_id

    # 从 Store 读取偏好
    store = runtime.store
    prefs = store.get(("preferences", category), user_id)

    if prefs:
        return f"用户 {category} 偏好: {prefs.value}"
    else:
        return f"未找到 {category} 偏好设置"

@tool
def search_with_context(
    query: str,
    runtime: ToolRuntime
) -> str:
    """带上下文的搜索"""

    # 从 Runtime Context 读取配置
    api_key = runtime.context.api_key
    search_region = runtime.context.get("search_region", "cn")

    # 从 State 读取当前上下文
    current_topic = runtime.state.get("current_topic", "")

    # 执行搜索（带上下文增强）
    enhanced_query = f"{query} {current_topic}" if current_topic else query

    # 实际搜索逻辑...
    return f"搜索结果: {enhanced_query}"
```

### 工具写入 State

使用 `Command` 更新 State：

```python
from langgraph.types import Command

@tool
def authenticate_user(
    username: str,
    password: str,
    runtime: ToolRuntime
) -> Command:
    """认证用户并更新 State"""

    # 验证逻辑
    is_valid = verify_credentials(username, password)

    if is_valid:
        # 返回 Command 更新 State
        return Command(
            update={
                "authenticated": True,
                "username": username,
                "login_time": datetime.now().isoformat(),
            }
        )
    else:
        return Command(
            update={
                "authenticated": False,
                "auth_error": "用户名或密码错误",
            }
        )

@tool
def upload_file(
    file_name: str,
    file_content: str,
    runtime: ToolRuntime
) -> Command:
    """上传文件并更新 State"""

    # 处理文件
    file_summary = summarize_file(file_content)

    # 获取现有文件列表
    existing_files = runtime.state.get("uploaded_files", [])

    # 添加新文件
    new_file = {
        "name": file_name,
        "summary": file_summary,
        "uploaded_at": datetime.now().isoformat(),
    }

    return Command(
        update={
            "uploaded_files": existing_files + [new_file],
            "last_upload": file_name,
        }
    )
```

### 工具写入 Store

直接操作 Store 进行持久化：

```python
@tool
def save_user_preference(
    key: str,
    value: str,
    runtime: ToolRuntime
) -> str:
    """保存用户偏好到 Store（长期记忆）"""

    user_id = runtime.context.user_id
    store = runtime.store

    # 读取现有偏好
    existing = store.get(("preferences",), user_id)
    prefs = existing.value if existing else {}

    # 更新偏好
    prefs[key] = value
    prefs["updated_at"] = datetime.now().isoformat()

    # 保存到 Store
    store.put(("preferences",), user_id, prefs)

    return f"已保存偏好: {key} = {value}"

@tool
def record_interaction_insight(
    insight_type: str,
    insight_value: str,
    confidence: float,
    runtime: ToolRuntime
) -> str:
    """记录从交互中学到的见解"""

    user_id = runtime.context.user_id
    store = runtime.store

    # 保存见解
    store.put(
        ("insights", user_id),
        insight_type,
        {
            "value": insight_value,
            "confidence": confidence,
            "recorded_at": datetime.now().isoformat(),
        }
    )

    return f"已记录见解: {insight_type}"
```

---

## 生命周期中间件

### 消息摘要总结（SummarizationMiddleware）

长对话会超出模型的上下文窗口，需要自动总结压缩。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",           # 用于总结的模型
            trigger=("tokens", 4000),       # 触发条件：超过 4000 tokens
            keep=("messages", 20),          # 保留最近 20 条消息
        ),
    ],
    checkpointer=InMemorySaver(),  # 需要 checkpointer 持久化
)
```

**工作原理**：

```
对话历史超过 4000 tokens
          ↓
┌─────────────────────────────┐
│  消息 1 (旧)                │
│  消息 2 (旧)                │  ← 这些会被总结
│  消息 3 (旧)                │
│  ...                        │
├─────────────────────────────┤
│  [摘要消息]                 │  ← 用摘要替换
├─────────────────────────────┤
│  消息 N-19                  │
│  消息 N-18                  │  ← 保留最近 20 条
│  ...                        │
│  消息 N (最新)              │
└─────────────────────────────┘
```

**配置参数详解**：

```python
SummarizationMiddleware(
    # 用于生成摘要的模型
    model="gpt-4o-mini",

    # 触发条件（满足任一条件即触发）
    trigger=("tokens", 4000),        # 超过 4000 tokens
    # 或
    trigger=("messages", 50),        # 超过 50 条消息
    # 或
    trigger=("fraction", 0.8),       # 超过上下文窗口的 80%
    # 或多个条件
    trigger=[("tokens", 4000), ("messages", 50)],  # OR 逻辑

    # 保留策略
    keep=("messages", 20),           # 保留最近 20 条消息
    # 或
    keep=("tokens", 2000),           # 保留最近 2000 tokens
    # 或
    keep=("fraction", 0.3),          # 保留上下文窗口的 30%

    # 可选：自定义摘要提示词
    summary_prompt="请总结以下对话的要点:\n{messages}",

    # 可选：摘要前缀
    summary_prefix="[对话摘要]",
)
```

**完整示例**：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20),
        )
    ],
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "conversation-1"}}

# 多轮对话
agent.invoke({"messages": "你好，我叫张三"}, config)
agent.invoke({"messages": "我在北京工作"}, config)
agent.invoke({"messages": "我喜欢编程和摄影"}, config)
# ... 很多轮对话后 ...

# 即使早期消息被总结，模型仍能记住关键信息
result = agent.invoke({"messages": "我叫什么名字？"}, config)
print(result["messages"][-1].content)  # "你叫张三"
```

### 消息修剪（Message Trimming）

简单的消息修剪，不进行总结：

```python
from langchain.agents.middleware import before_model, AgentState
from langgraph.runtime import Runtime

@before_model
def trim_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """简单修剪：保留最近 N 条消息"""

    messages = state["messages"]
    MAX_MESSAGES = 20

    if len(messages) > MAX_MESSAGES:
        # 保留系统消息 + 最近的消息
        system_msgs = [m for m in messages if m.get("role") == "system"]
        recent_msgs = messages[-MAX_MESSAGES:]

        return {"messages": system_msgs + recent_msgs}

    return None

# 基于 token 的修剪
@before_model
def trim_by_tokens(state: AgentState, runtime: Runtime) -> dict | None:
    """基于 token 数量修剪消息"""

    messages = state["messages"]
    MAX_TOKENS = 4000

    # 简单估算 token 数（每个字符约 0.5 token）
    def estimate_tokens(msg):
        return len(str(msg.get("content", ""))) // 2

    total_tokens = sum(estimate_tokens(m) for m in messages)

    if total_tokens > MAX_TOKENS:
        # 从旧消息开始删除，直到低于限制
        trimmed = []
        current_tokens = 0

        for msg in reversed(messages):
            msg_tokens = estimate_tokens(msg)
            if current_tokens + msg_tokens <= MAX_TOKENS:
                trimmed.insert(0, msg)
                current_tokens += msg_tokens
            elif msg.get("role") == "system":
                # 始终保留系统消息
                trimmed.insert(0, msg)

        return {"messages": trimmed}

    return None
```

---

## 瞬态与持久化更新

### 核心区别

| 类型 | 实现方式 | 影响范围 | 持久化 | 典型用途 |
|------|----------|----------|--------|----------|
| **瞬态** | `wrap_model_call` | 单次调用 | 否 | 临时注入、过滤 |
| **持久化** | 生命周期钩子 | 所有后续调用 | 是 | 摘要、状态更新 |

### 瞬态更新

使用 `@wrap_model_call` 进行瞬态修改：

```python
@wrap_model_call
def transient_modification(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """瞬态修改：只影响当前调用"""

    # 1. 临时注入消息（不保存到 State）
    messages = [
        *request.messages,
        {"role": "system", "content": "临时指令：本次请简洁回答"}
    ]

    # 2. 临时过滤工具
    tools = [t for t in request.tools if t.name != "dangerous_tool"]

    # 3. 临时切换模型
    from langchain.chat_models import init_chat_model
    model = init_chat_model("gpt-4o-mini")

    # 使用 override 创建修改后的请求
    request = request.override(
        messages=messages,
        tools=tools,
        model=model,
    )

    # 这些修改只影响本次调用，不会保存到 State
    return handler(request)
```

### 持久化更新

使用生命周期钩子进行持久化修改：

```python
from langchain.agents.middleware import before_model, after_model

@before_model
def persistent_before(state: AgentState, runtime: Runtime) -> dict | None:
    """持久化修改：影响所有后续调用"""

    # 返回的 dict 会合并到 State 中（持久化）
    return {
        "messages": modified_messages,  # 永久修改消息历史
        "custom_field": "value",        # 添加自定义字段
    }

@after_model
def persistent_after(state: AgentState, runtime: Runtime) -> dict | None:
    """模型调用后的持久化修改"""

    # 例如：记录调用次数
    call_count = state.get("model_call_count", 0) + 1

    return {
        "model_call_count": call_count,
        "last_call_time": datetime.now().isoformat(),
    }
```

### 选择指南

```
需要修改上下文？
      │
      ├── 只影响本次调用 ──────→ 使用 @wrap_model_call（瞬态）
      │   • 临时注入信息
      │   • 临时过滤工具
      │   • 临时切换模型
      │
      └── 影响所有后续调用 ────→ 使用生命周期钩子（持久化）
          • 消息摘要
          • 状态更新
          • 计数器累加
```

---

## 实战案例

### 案例 1：智能客服系统

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import (
    dynamic_prompt,
    wrap_model_call,
    SummarizationMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import tool, ToolRuntime
from pydantic import BaseModel, Field

# 1. 定义上下文
@dataclass
class CustomerContext:
    customer_id: str
    customer_name: str
    subscription_tier: str  # "free", "pro", "enterprise"
    language: str = "zh"

# 2. 定义工具
@tool
def get_customer_orders(
    limit: int,
    runtime: ToolRuntime[CustomerContext]
) -> str:
    """获取客户订单"""
    customer_id = runtime.context.customer_id
    # 实际查询逻辑...
    return f"客户 {customer_id} 的最近 {limit} 个订单: [...]"

@tool
def create_support_ticket(
    issue: str,
    priority: str,
    runtime: ToolRuntime[CustomerContext]
) -> str:
    """创建支持工单"""
    ctx = runtime.context

    # 企业客户自动升级优先级
    if ctx.subscription_tier == "enterprise" and priority == "normal":
        priority = "high"

    return f"已为 {ctx.customer_name} 创建工单 (优先级: {priority})"

# 3. 动态 System Prompt
@dynamic_prompt
def customer_service_prompt(request: ModelRequest) -> str:
    ctx = request.runtime.context

    tier_guide = {
        "free": "提供基础支持",
        "pro": "提供专业技术支持",
        "enterprise": "提供最高级别支持，承诺 SLA",
    }

    return f"""你是专业客服助手。

客户信息:
- 姓名: {ctx.customer_name}
- ID: {ctx.customer_id}
- 套餐: {ctx.subscription_tier}

服务指南: {tier_guide.get(ctx.subscription_tier, tier_guide['free'])}

请用{ctx.language == 'zh' and '中文' or 'English'}回复。"""

# 4. 动态模型选择
@wrap_model_call
def tier_based_model(request: ModelRequest, handler) -> ModelResponse:
    tier = request.runtime.context.subscription_tier

    if tier == "enterprise":
        from langchain.chat_models import init_chat_model
        request = request.override(model=init_chat_model("gpt-4o"))

    return handler(request)

# 5. 创建 Agent
from langgraph.checkpoint.memory import InMemorySaver

customer_service_agent = create_agent(
    model="gpt-4o-mini",  # 默认模型
    tools=[get_customer_orders, create_support_ticket],
    middleware=[
        customer_service_prompt,
        tier_based_model,
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 15),
        ),
    ],
    context_schema=CustomerContext,
    checkpointer=InMemorySaver(),
)

# 6. 使用
result = customer_service_agent.invoke(
    {"messages": [{"role": "user", "content": "我想查看最近的订单"}]},
    context=CustomerContext(
        customer_id="C001",
        customer_name="张三",
        subscription_tier="enterprise",
    ),
    config={"configurable": {"thread_id": "support-123"}},
)
```

### 案例 2：自适应学习助手

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class LearnerContext:
    learner_id: str
    learner_name: str
    expertise_level: Literal["beginner", "intermediate", "expert"]
    learning_style: Literal["visual", "auditory", "reading", "kinesthetic"]
    preferred_language: str = "zh"

# 工具：保存学习进度
@tool
def save_learning_progress(
    topic: str,
    mastery_level: float,
    runtime: ToolRuntime[LearnerContext]
) -> str:
    """保存学习进度到长期记忆"""
    learner_id = runtime.context.learner_id
    store = runtime.store

    store.put(
        ("learning_progress", learner_id),
        topic,
        {
            "mastery": mastery_level,
            "last_studied": datetime.now().isoformat(),
        }
    )

    return f"已记录 {topic} 的学习进度: {mastery_level*100:.0f}%"

# 工具：获取学习历史
@tool
def get_learning_history(
    topic: str,
    runtime: ToolRuntime[LearnerContext]
) -> str:
    """从长期记忆获取学习历史"""
    learner_id = runtime.context.learner_id
    store = runtime.store

    progress = store.get(("learning_progress", learner_id), topic)

    if progress:
        return f"{topic} 学习历史: 掌握度 {progress.value['mastery']*100:.0f}%, 最后学习: {progress.value['last_studied']}"
    else:
        return f"没有 {topic} 的学习记录"

# 自适应提示词
@dynamic_prompt
def adaptive_learning_prompt(request: ModelRequest) -> str:
    ctx = request.runtime.context
    store = request.runtime.store

    # 基础指令
    base = f"你是 {ctx.learner_name} 的个人学习助手。"

    # 根据专业水平调整
    level_guide = {
        "beginner": "使用简单语言，避免术语，多举例说明",
        "intermediate": "可以使用专业术语，但需要适当解释",
        "expert": "可以进行深入的技术讨论",
    }
    base += f"\n\n教学风格: {level_guide[ctx.expertise_level]}"

    # 根据学习风格调整
    style_guide = {
        "visual": "多使用图表、流程图、示意图描述",
        "auditory": "使用对话式讲解，强调重点",
        "reading": "提供详细的文字说明和参考资料",
        "kinesthetic": "设计动手练习和实践项目",
    }
    base += f"\n学习风格偏好: {style_guide[ctx.learning_style]}"

    # 从 Store 读取学习历史
    # (简化示例)

    return base

# 动态响应格式
class BeginnerResponse(BaseModel):
    explanation: str = Field(description="通俗易懂的解释")
    examples: list[str] = Field(description="生活中的例子")
    key_takeaway: str = Field(description="一句话总结")

class ExpertResponse(BaseModel):
    explanation: str = Field(description="技术解释")
    technical_details: str = Field(description="深入技术细节")
    advanced_topics: list[str] = Field(description="相关进阶主题")
    references: list[str] = Field(description="参考资料")

@wrap_model_call
def adaptive_response_format(request: ModelRequest, handler) -> ModelResponse:
    level = request.runtime.context.expertise_level

    if level == "beginner":
        request = request.override(response_format=BeginnerResponse)
    elif level == "expert":
        request = request.override(response_format=ExpertResponse)

    return handler(request)

# 创建学习助手
learning_assistant = create_agent(
    model="gpt-4o",
    tools=[save_learning_progress, get_learning_history],
    middleware=[
        adaptive_learning_prompt,
        adaptive_response_format,
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 6000),
            keep=("messages", 30),
        ),
    ],
    context_schema=LearnerContext,
    checkpointer=InMemorySaver(),
    store=InMemoryStore(),
)
```

---

## 最佳实践

### 1. 从简单开始

```python
# ✅ 好的做法：先用静态配置
agent = create_agent(
    model="gpt-4o",
    tools=[tool1, tool2],
    prompt="你是一个助手。",  # 先用静态 prompt
)

# 只在需要时添加动态功能
# ❌ 不好的做法：一开始就过度工程化
```

### 2. 增量添加上下文工程

```python
# 第 1 步：基础 Agent
agent = create_agent(model="gpt-4o", tools=[...])

# 第 2 步：添加动态 prompt
agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[dynamic_prompt_middleware],
)

# 第 3 步：添加消息管理
agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        dynamic_prompt_middleware,
        SummarizationMiddleware(...),
    ],
)

# 第 4 步：添加动态工具/模型选择
# ...
```

### 3. 监控和调试

```python
@wrap_model_call
def debug_middleware(request: ModelRequest, handler) -> ModelResponse:
    """调试中间件：记录上下文信息"""

    print(f"=== 模型调用 ===")
    print(f"消息数: {len(request.messages)}")
    print(f"工具数: {len(request.tools)}")
    print(f"模型: {request.model}")

    response = handler(request)

    print(f"响应长度: {len(str(response))}")
    print(f"================")

    return response
```

### 4. 测试上下文工程

```python
def test_dynamic_prompt():
    """测试动态 prompt 是否正确生成"""

    # 模拟不同上下文
    contexts = [
        Context(user_role="admin"),
        Context(user_role="viewer"),
        Context(user_role="guest"),
    ]

    for ctx in contexts:
        # 创建模拟请求
        mock_request = MockModelRequest(runtime=MockRuntime(context=ctx))

        # 调用动态 prompt 函数
        prompt = my_dynamic_prompt(mock_request)

        # 验证 prompt 内容
        if ctx.user_role == "admin":
            assert "管理员" in prompt
        elif ctx.user_role == "viewer":
            assert "只读" in prompt
```

### 5. 文档化上下文策略

```python
"""
上下文工程策略文档
==================

1. System Prompt 策略
   - 根据用户角色调整权限说明
   - 根据订阅层级调整服务水平
   - 根据语言偏好切换语言

2. 消息管理策略
   - 触发条件: 超过 4000 tokens
   - 保留策略: 最近 20 条消息
   - 摘要模型: gpt-4o-mini

3. 工具选择策略
   - 未认证: 只提供公开工具
   - 普通用户: 排除管理工具
   - 管理员: 所有工具

4. 模型选择策略
   - 短对话 (<10 消息): gpt-4o-mini
   - 中等对话: gpt-4o
   - 长对话 (>20 消息): claude-sonnet-4-5-20250929
"""
```

---

## 快速参考

### 数据源访问方式

| 数据源 | 在中间件中访问 | 在工具中访问 |
|--------|----------------|--------------|
| Runtime Context | `request.runtime.context` | `runtime.context` |
| State | `request.state` 或 `state` | `runtime.state` |
| Store | `request.runtime.store` | `runtime.store` |
| Messages | `request.messages` | - |

### 常用装饰器

```python
from langchain.agents.middleware import (
    dynamic_prompt,      # 动态 System Prompt
    wrap_model_call,     # 包装模型调用（瞬态）
    before_model,        # 模型调用前（持久化）
    after_model,         # 模型调用后（持久化）
    before_agent,        # Agent 开始前
    after_agent,         # Agent 结束后
    wrap_tool_call,      # 包装工具调用
)
```

### 请求修改方法

```python
# 在 @wrap_model_call 中使用 override
request = request.override(
    messages=new_messages,      # 修改消息
    tools=filtered_tools,       # 修改工具
    model=different_model,      # 修改模型
    response_format=MyFormat,   # 修改输出格式
)
```

### SummarizationMiddleware 配置

```python
SummarizationMiddleware(
    model="gpt-4o-mini",

    # 触发条件
    trigger=("tokens", 4000),      # token 数
    trigger=("messages", 50),      # 消息数
    trigger=("fraction", 0.8),     # 上下文占比

    # 保留策略
    keep=("messages", 20),         # 保留消息数
    keep=("tokens", 2000),         # 保留 token 数
    keep=("fraction", 0.3),        # 保留占比
)
```

### 工具上下文读写

```python
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

# 读取上下文
@tool
def read_context(runtime: ToolRuntime) -> str:
    user_id = runtime.context.user_id      # Runtime Context
    data = runtime.state.get("key")        # State
    prefs = runtime.store.get(ns, key)     # Store
    return "..."

# 写入 State
@tool
def write_state(runtime: ToolRuntime) -> Command:
    return Command(update={"key": "value"})

# 写入 Store
@tool
def write_store(runtime: ToolRuntime) -> str:
    runtime.store.put(namespace, key, value)
    return "saved"
```

---

## 总结

**Context Engineering（上下文工程）** 是构建优秀 AI Agent 的核心技能。

### 核心要点

1. **三种上下文类型**
   - Model Context（瞬态）：每次调用的输入
   - Tool Context（持久化）：工具的读写
   - Life-cycle Context（持久化）：调用之间的处理

2. **三种数据源**
   - Runtime Context：静态配置（用户 ID、API 密钥）
   - State：短期记忆（消息历史、临时数据）
   - Store：长期记忆（用户偏好、历史见解）

3. **五大上下文工程技术**
   - 动态 System Prompt
   - 消息注入与管理
   - 动态工具选择
   - 动态模型选择
   - 结构化输出格式

4. **瞬态 vs 持久化**
   - `wrap_model_call`：瞬态修改，只影响单次调用
   - 生命周期钩子：持久化修改，影响所有后续调用

### 最佳实践

- 从简单开始，按需添加复杂性
- 使用内置中间件（SummarizationMiddleware 等）
- 监控和记录上下文决策
- 充分测试不同场景
- 文档化上下文策略

通过掌握上下文工程，你可以让 LLM 在每次调用时都获得最佳上下文，从而构建出更智能、更可靠的 AI Agent！
