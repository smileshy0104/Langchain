# LangChain Agents 详细总结

> 基于官方文档 [https://docs.langchain.com/oss/python/langchain/agents](https://docs.langchain.com/oss/python/langchain/agents) 的完整中文总结（在原有内容基础上补充当前 v1.x 文档中的 Model + Harness、调用方式、Streaming 与 Harness 中间件体系）

---

## 📋 目录

- [核心概念](#核心概念)
- [create_agent 详解](#create_agent-详解)
- [Agent 调用与流式输出](#agent-调用与流式输出)
- [工具（Tools）](#工具tools)
- [中间件（Middleware）](#中间件middleware)
- [短期记忆（Short-term Memory）](#短期记忆short-term-memory)
- [上下文工程（Context Engineering）](#上下文工程context-engineering)
- [人机协作（Human-in-the-Loop）](#人机协作human-in-the-loop)
- [结构化输出](#结构化输出)
- [Harness 配置与 Deep Agents](#harness-配置与-deep-agents)
- [最佳实践](#最佳实践)

---

## 核心概念

### 什么是 Agent？

**定义**: Agents 将语言模型与工具结合，创建能够推理任务、决定使用哪些工具并迭代地朝着解决方案努力的系统。

**当前官方文档的核心表述**: **Agent = Model + Harness**。

- **Model**: 负责推理、决定是否调用工具以及如何继续任务
- **Harness**: 围绕模型循环的运行时外壳，包括提示词、工具、上下文、状态、中间件、检查点与观测能力
- Harness 的职责是：**在正确的时间，把正确的上下文交给模型**

**核心特征**:

- LLM Agent 在循环中运行工具以实现目标
- Agent 持续运行直到满足停止条件（模型输出最终答案或达到迭代限制）

### Agent 执行流程

```
┌─────────────┐
│   输入查询   │
│   (input)   │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   模型推理   │
│   (model)   │
└──────┬──────┘
       │
       ├──→ action ──→ ┌─────────────┐
       │               │  工具执行    │
       │               │  (tools)    │
       │               └──────┬──────┘
       │                      │
       │ ←─── observation ────┘
       │
       │ (循环继续直到完成)
       │
       ↓
┌─────────────┐
│   最终输出   │
│   (output)  │
└─────────────┘
```

**执行步骤**:

1. **输入** → 用户查询进入 Agent
2. **模型** → LLM 分析并决定是否调用工具
3. **工具** → 执行工具并返回观察结果（observation）
4. **循环** → 模型根据观察结果继续推理
5. **完成** → 模型输出最终答案

### 基于图的架构

`create_agent` 构建基于 **LangGraph** 的图状运行时：

**图的组成**:

- **节点（Nodes）**: 执行步骤
  - `model` 节点: 调用模型
  - `tools` 节点: 执行工具
  - 中间件节点: 自定义逻辑
- **边（Edges）**: 连接，定义信息流

**优势**:

- 可视化执行流程
- 灵活的控制流
- 持久化执行状态
- 支持复杂的条件逻辑

---

## create_agent 详解

### 基础用法

`create_agent` 是 **LangChain v1.x** 构建 Agent 的标准方式。它是一个高度可配置的 Harness，最基础只需要配置 `model` 和 `tools`，复杂能力则通过 `middleware`、`checkpointer`、`context_schema`、`state_schema` 等扩展。

当前官方文档推荐的最简写法是直接传入模型标识字符串：

```python
from langchain.agents import create_agent

agent = create_agent("openai:gpt-5.4", tools=tools)
```

也可以继续传入已初始化的模型实例，适合需要自定义 provider 参数、超时、温度等配置的场景。

**最简示例**:

```python
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic

# 定义工具
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """获取指定位置的天气信息"""
    return f"{location} 的天气是晴朗的，温度 20°C"

# 创建 Agent
model = ChatAnthropic(model="claude-sonnet-4-6")
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是一个有帮助的 AI 助手"
)

# 执行
result = agent.invoke({
    "messages": [{"role": "user", "content": "北京天气如何？"}]
})

print(result["messages"][-1]["content"])
# 输出: 北京的天气是晴朗的，温度 20°C
```

### 核心参数

#### 1. `model` - 模型配置

**类型**: `BaseChatModel` 或模型标识字符串（推荐格式：`"provider:model"`）

**支持的模型**:

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# 方式 1: 直接传入模型实例
model = ChatAnthropic(model="claude-sonnet-4-6")

# 方式 2: 传入模型名称（简写）
agent = create_agent(
    model="claude-sonnet-4-6",  # 简化写法
    tools=tools
)

# 方式 3: 使用不同提供商
models = [
    ChatOpenAI(model="gpt-4o"),
    ChatAnthropic(model="claude-sonnet-4-6"),
    ChatGoogleGenerativeAI(model="gemini-pro")
]
```

**动态模型选择** (通过中间件):

```python
from langchain.agents.middleware import wrap_model_call
from langchain_openai import ChatOpenAI

basic_model = ChatOpenAI(model="gpt-4o-mini")
advanced_model = ChatOpenAI(model="gpt-4o")

@wrap_model_call
def dynamic_model_selection(request, handler):
    """根据对话复杂度选择模型"""
    message_count = len(request.messages)

    if message_count > 10:
        request = request.override(model=advanced_model)
    else:
        request = request.override(model=basic_model)

    return handler(request)

agent = create_agent(
    model="gpt-4o-mini",
    tools=tools,
    middleware=[dynamic_model_selection]
)
```

#### 2. `tools` - 工具列表

**类型**: `List[BaseTool]`

**定义工具的方式**:

**方式 1: 使用 `@tool` 装饰器**

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    """搜索输入参数"""
    query: str = Field(description="搜索查询词")
    limit: int = Field(default=10, description="返回结果数量")

@tool(args_schema=SearchInput)
def search_web(query: str, limit: int = 10) -> str:
    """
    在网络上搜索信息

    Args:
        query: 搜索关键词
        limit: 最多返回多少条结果

    Returns:
        搜索结果摘要
    """
    # 实现搜索逻辑
    return f"找到 {limit} 条关于 '{query}' 的结果"

tools = [search_web]
```

**方式 2: 从 Retriever 创建工具**

```python
from langchain_classic.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever=vectorstore.as_retriever(),
    name="search_docs",
    description="搜索文档库以获取相关信息"
)

tools = [retriever_tool]
```

#### 3. `system_prompt` - 系统提示词

**类型**: `str` (可选)

**作用**: 设置 Agent 的行为和能力

**静态提示词**:

```python
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="""你是一个专业的数据分析助手。

# 核心能力
- 数据查询和分析
- 生成可视化图表
- 提供业务洞察

# 工作流程
1. 理解用户需求
2. 选择合适的工具
3. 分析数据结果
4. 生成清晰的报告

# 限制
- 不要编造数据
- 不确定时明确说明
- 超出能力范围时建议人工介入
"""
)
```

**动态提示词** (通过中间件):

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from dataclasses import dataclass

@dataclass
class Context:
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """根据用户角色生成系统提示"""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "你是一个有帮助的助手。"

    if user_role == "expert":
        return f"{base_prompt} 提供详细的技术响应。"
    elif user_role == "beginner":
        return f"{base_prompt} 简单解释概念，避免术语。"

    return base_prompt

agent = create_agent(
    model="gpt-4o",
    tools=[search_web],
    middleware=[user_role_prompt],
    context_schema=Context
)

# 使用时传入上下文
result = agent.invoke(
    {"messages": [{"role": "user", "content": "解释机器学习"}]},
    context={"user_role": "expert"}
)
```

#### 4. `middleware` - 中间件

**类型**: `List[Middleware]` (可选)

**作用**: 在 Agent 执行的不同阶段插入自定义逻辑

详见 [中间件章节](#中间件middleware)

#### 5. `checkpointer` - 检查点器

**类型**: `BaseCheckpointSaver` (可选)

**作用**: 启用短期记忆（对话历史持久化）

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=checkpointer
)

# 使用 thread_id 管理会话
agent.invoke(
    {"messages": [{"role": "user", "content": "我叫张三"}]},
    {"configurable": {"thread_id": "user-123"}}
)

# 在同一会话中继续对话
agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
    {"configurable": {"thread_id": "user-123"}}
)
# 输出: 你叫张三
```

#### 6. `state_schema` - 状态模式

**类型**: `Type[TypedDict]` (可选)

**作用**: 扩展 Agent 状态以存储额外信息

```python
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import MemorySaver

class CustomAgentState(AgentState):
    user_id: str
    preferences: dict

agent = create_agent(
    model="gpt-5",
    tools=[get_user_info],
    state_schema=CustomAgentState,
    checkpointer=MemorySaver()
)

# 传入自定义状态
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "你好"}],
        "user_id": "user_123",
        "preferences": {"theme": "dark"}
    },
    {"configurable": {"thread_id": "1"}}
)
```

#### 7. `context_schema` - 上下文模式

**类型**: `Type` (可选)

**作用**: 定义运行时上下文的类型，用于依赖注入

```python
from dataclasses import dataclass

@dataclass
class Context:
    user_id: str
    db_connection: Any

agent = create_agent(
    model=model,
    tools=tools,
    context_schema=Context
)

# 使用时注入上下文
from sqlalchemy import create_engine

db = create_engine("postgresql://...")
agent.invoke(
    {"messages": [...]},
    context={"user_id": "123", "db_connection": db}
)
```

#### 8. `response_format` - 响应格式（结构化输出）

**类型**: `ToolStrategy` 或 `ProviderStrategy` (可选)

**作用**: 强制 Agent 返回结构化数据

```python
from pydantic import BaseModel, Field
from langchain.agents.structured_output import ToolStrategy

class WeatherResponse(BaseModel):
    """天气响应结构"""
    location: str = Field(description="位置")
    temperature: float = Field(description="温度（摄氏度）")
    condition: str = Field(description="天气状况")
    humidity: int = Field(description="湿度百分比")

agent = create_agent(
    model=model,
    tools=[get_weather],
    response_format=ToolStrategy(
        schema=WeatherResponse,
        tool_message_content="天气信息已获取！"
    )
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "北京天气如何？"}]
})

weather_data: WeatherResponse = result["structured_response"]
print(f"温度: {weather_data.temperature}°C")
```

#### 9. `name` - Agent 名称

**类型**: `str` (可选)

**作用**: 为 Agent 设置标识符。当 Agent 被嵌入多 Agent 系统或作为 LangGraph 子图使用时，`name` 会作为节点名，便于追踪和调试。

```python
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=tools,
    name="research_assistant"
)
```

---

## Agent 调用与流式输出

### 1. 基础调用

Agent 接收的是对状态（State）的更新。所有 Agent 默认都包含 `messages` 序列，因此最常见的调用方式是传入一条新的用户消息：

```python
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]
})

print(result["messages"][-1].content)
```

### 2. 使用 `thread_id` 保持会话历史

如果需要 Agent 记住同一段对话中的上下文，需要同时满足两个条件：

1. 创建 Agent 时配置 `checkpointer`
2. 调用时在 `config` 中传入稳定的 `thread_id`

当前官方示例使用 `InMemorySaver`：

```python
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": str(uuid7())}}

agent.invoke(
    {"messages": [{"role": "user", "content": "我叫张三"}]},
    config=config,
)

# 复用同一个 thread_id，Agent 可以恢复同一会话历史
result = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
    config=config,
)
```

> `thread_id` 标识一段对话；`checkpointer` 负责保存和恢复该对话的状态。部署到 LangSmith 时通常会自动配置检查点；本地运行则需要显式传入。

### 3. `context` 与 `thread_id` 的区别


| 概念          | 用途                           | 生命周期  |
| ----------- | ---------------------------- | ----- |
| `thread_id` | 标识会话，用于保存/恢复消息历史和 checkpoint | 会话级   |
| `context`   | 每次调用传入的运行时数据，供工具和中间件读取       | 单次调用级 |


适合放入 `context` 的内容包括：`user_id`、角色、权限、API key、feature flags、租户信息、数据库连接等。

```python
from dataclasses import dataclass

@dataclass
class Context:
    user_id: str
    role: str

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=tools,
    context_schema=Context,
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "查询我的订单"}]},
    config={"configurable": {"thread_id": "conversation-1"}},
    context=Context(user_id="user-123", role="customer"),
)
```

### 4. Streaming 流式输出

当 Agent 执行多步推理或多个工具调用时，`invoke` 只能拿到最终结果。使用 `stream` 可以展示中间状态、调试工具调用顺序或在 UI 中显示进度。

```python
from langchain.messages import AIMessage, HumanMessage

for chunk in agent.stream(
    {
        "messages": [
            {"role": "user", "content": "Search for AI news and summarize the findings"}
        ]
    },
    stream_mode="values",
):
    # 每个 chunk 包含当前完整状态
    latest_message = chunk["messages"][-1]

    if latest_message.content:
        if isinstance(latest_message, HumanMessage):
            print(f"User: {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            print(f"Agent: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
```

---

## 工具（Tools）

### 工具定义规范

#### 1. 基础工具定义

```python
from langchain_core.tools import tool

@tool
def calculate(expression: str) -> float:
    """
    计算数学表达式

    Args:
        expression: 要计算的数学表达式，如 "2 + 2"

    Returns:
        计算结果

    Examples:
        >>> calculate("10 * 5")
        50.0
    """
    return eval(expression)
```

**关键要素**:

- ✅ **函数名**: 描述性名称（模型会看到）
- ✅ **文档字符串**: 详细说明工具用途
- ✅ **参数类型**: 明确的类型注解
- ✅ **返回类型**: 清晰的返回值

#### 2. 带参数验证的工具

```python
from pydantic import BaseModel, Field, validator

class DatabaseQueryInput(BaseModel):
    """数据库查询输入"""
    query: str = Field(description="SQL 查询语句")
    limit: int = Field(default=100, ge=1, le=1000, description="最大返回行数")

    @validator("query")
    def validate_query(cls, v):
        # 安全检查：防止 SQL 注入
        forbidden = ["DROP", "DELETE", "UPDATE", "INSERT"]
        if any(word in v.upper() for word in forbidden):
            raise ValueError("只允许 SELECT 查询")
        return v

@tool(args_schema=DatabaseQueryInput)
def query_database(query: str, limit: int = 100) -> list:
    """
    在数据库中执行只读查询

    安全限制:
    - 仅允许 SELECT 语句
    - 最多返回 1000 行
    """
    # 执行查询
    results = db.execute(query).fetchmany(limit)
    return results
```

#### 3. 工具错误处理

```python
from langchain_core.tools import ToolException

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送电子邮件"""
    try:
        # 验证邮箱格式
        if "@" not in to:
            raise ToolException("无效的邮箱地址格式")

        # 发送邮件
        result = email_service.send(to, subject, body)
        return f"邮件已发送至 {to}"

    except EmailServiceError as e:
        # 转换为 ToolException，Agent 可以理解
        raise ToolException(
            f"邮件发送失败: {e}. 请检查邮箱地址或稍后重试。"
        )
    except Exception as e:
        raise ToolException(f"未知错误: {e}")
```

**通过中间件处理工具错误**:

```python
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    """统一处理工具执行错误"""
    try:
        return handler(request)
    except Exception as e:
        # 返回友好的错误消息给模型
        return ToolMessage(
            content=f"工具执行失败: 请检查输入参数。详情: {str(e)}",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="gpt-4o",
    tools=[search, send_email],
    middleware=[handle_tool_errors]
)
```

### 特殊类型的工具

#### 1. Retriever 工具（RAG）

```python
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 创建向量存储
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings()
)

# 创建 retriever 工具
retriever_tool = create_retriever_tool(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    name="search_company_docs",
    description="搜索公司文档库以查找相关政策、流程和指南"
)

agent = create_agent(
    model=model,
    tools=[retriever_tool]
)
```

#### 2. 异步工具

```python
import asyncio
import aiohttp

@tool
async def async_web_search(query: str) -> str:
    """异步网络搜索"""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://api.example.com/search?q={query}"
        ) as resp:
            data = await resp.json()
            return data["results"]

# Agent 会自动并行执行多个异步工具调用
```

---

## 中间件（Middleware）

### 核心概念

**中间件** 是 `create_agent` 的定义性功能，提供高度可定制的入口点。

**主要用途**:

- 📝 动态提示词
- 💬 对话摘要
- 🔧 选择性工具访问
- 📊 状态管理
- 🛡️ 安全防护

### 执行流程

中间件在 Agent 执行的不同阶段提供钩子（hooks）：

```
Agent 开始
    │
    ├─→ before_agent()      # Agent 开始前
    │
    ├─→ 循环开始
    │   │
    │   ├─→ before_model()  # 模型调用前
    │   │
    │   ├─→ wrap_model_call()  # 包装模型调用
    │   │       │
    │   │       └─→ 实际模型调用
    │   │
    │   ├─→ after_model()   # 模型调用后
    │   │
    │   ├─→ wrap_tool_call()  # 包装工具调用
    │   │       │
    │   │       └─→ 实际工具执行
    │   │
    │   └─→ 循环继续或结束
    │
    └─→ after_agent()       # Agent 完成后
```

### 中间件钩子（Hooks）


| 钩子                | 执行时机        | 用途         |
| ----------------- | ----------- | ---------- |
| `before_agent`    | Agent 开始前   | 加载记忆、验证输入  |
| `before_model`    | 每次 LLM 调用前  | 更新提示、修剪消息  |
| `wrap_model_call` | 每次 LLM 调用周围 | 拦截和修改请求/响应 |
| `after_model`     | 每次 LLM 响应后  | 验证输出、应用防护  |
| `wrap_tool_call`  | 每次工具调用周围    | 拦截和修改工具执行  |
| `after_agent`     | Agent 完成后   | 保存结果、清理资源  |


### 装饰器风格中间件

#### 1. `@before_model` - 模型调用前

```python
from langchain.agents.middleware import before_model
from langchain.agents.middleware import AgentState
from langgraph.runtime import Runtime

@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict | None:
    """记录模型调用前的状态"""
    print(f"准备调用模型，当前有 {len(state['messages'])} 条消息")
    return None  # 不修改状态

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[log_before_model]
)
```

**修剪消息示例**:

```python
from langchain.agents.middleware import before_model, trim_messages

@before_model
async def trim_message_history(state: AgentState, runtime: Runtime):
    """修剪过长的消息历史"""
    trimmed = await trim_messages(
        state["messages"],
        max_tokens=384,
        strategy="last",  # 保留最后的消息
        start_on="human",
        end_on=["human", "tool"]
    )
    return {"messages": trimmed}
```

#### 2. `@after_model` - 模型调用后

```python
from langchain.agents.middleware import after_model
from langchain.messages import AIMessage

@after_model(can_jump_to=["end"])
def validate_output(state: AgentState, runtime: Runtime):
    """验证模型输出并应用内容过滤"""
    last_message = state["messages"][-1]

    # 检查是否包含禁止内容
    if "BLOCKED" in last_message.content:
        return {
            "messages": [AIMessage("抱歉，我无法回应该请求。")],
            "jump_to": "end"  # 提前结束 Agent
        }

    return None

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[validate_output]
)
```

#### 3. `@wrap_model_call` - 包装模型调用

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """模型调用失败时自动重试"""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            return handler(request)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"重试 {attempt + 1}/{max_retries}，错误: {e}")
            time.sleep(2 ** attempt)  # 指数退避
```

**动态模型选择**:

```python
@wrap_model_call
def smart_model_routing(request: ModelRequest, handler):
    """根据任务复杂度选择模型"""
    # 简单问题使用快速模型
    if len(request.messages) <= 3:
        request = request.override(model=ChatOpenAI(model="gpt-4o-mini"))
    # 复杂问题使用强大模型
    else:
        request = request.override(model=ChatOpenAI(model="gpt-4o"))

    return handler(request)
```

#### 4. `@wrap_tool_call` - 包装工具调用

```python
from langchain.agents.middleware import wrap_tool_call

@wrap_tool_call
def log_tool_execution(request, handler):
    """记录工具执行"""
    tool_name = request.tool_call["name"]
    print(f"🔧 执行工具: {tool_name}")

    start_time = time.time()
    result = handler(request)
    elapsed = time.time() - start_time

    print(f"✅ 工具 {tool_name} 完成，耗时: {elapsed:.2f}s")
    return result
```

#### 5. `@dynamic_prompt` - 动态提示词

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    """基于上下文生成动态提示"""
    # 访问消息数量
    message_count = len(request.messages)

    base = "你是一个有帮助的助手。"

    # 长对话时要求简洁
    if message_count > 10:
        base += "\n这是一个长对话 - 请保持回答简洁。"

    # 访问运行时上下文
    user_role = request.runtime.context.get("user_role", "user")
    if user_role == "admin":
        base += "\n你有管理员权限，可以执行所有操作。"

    return base
```

### 类风格中间件

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from dataclasses import dataclass
from typing import Callable

@dataclass
class Context:
    user_expertise: str = "beginner"

class ExpertiseBasedToolMiddleware(AgentMiddleware):
    """基于用户专业水平动态选择工具"""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        user_level = request.runtime.context.user_expertise

        if user_level == "expert":
            # 专家用户：强大模型 + 高级工具
            model = ChatOpenAI(model="gpt-5")
            tools = [advanced_search, data_analysis, ml_training]
        else:
            # 初学者：简单模型 + 基础工具
            model = ChatOpenAI(model="gpt-5-nano")
            tools = [simple_search, basic_calculator]

        request = request.override(model=model, tools=tools)
        return handler(request)

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[simple_search, advanced_search, basic_calculator, data_analysis],
    middleware=[ExpertiseBasedToolMiddleware()],
    context_schema=Context
)
```

### 预构建中间件

LangChain 提供了常用的预构建中间件：

#### 1. PII 中间件（敏感信息脱敏）

```python
from langchain.agents import pii_redaction_middleware

agent = create_agent(
    model=model,
    tools=tools,
    middleware=[
        pii_redaction_middleware(patterns=["email", "phone", "ssn"])
    ]
)
```

#### 2. 摘要中间件

```python
from langchain.agents import summarization_middleware

agent = create_agent(
    model=model,
    tools=tools,
    middleware=[
        summarization_middleware(
            model="claude-sonnet-4-6",
            max_tokens_before_summary=500
        )
    ]
)
```

#### 3. 人机协作中间件

```python
from langchain.agents import human_in_the_loop_middleware

agent = create_agent(
    model=model,
    tools=[send_email, delete_file],
    middleware=[
        human_in_the_loop_middleware(
            interrupt_on={
                "send_email": {"allowed_decisions": ["approve", "edit", "reject"]},
                "delete_file": True  # 默认：approve, edit, reject
            }
        )
    ]
)
```

### 中间件执行顺序

当使用多个中间件时，理解执行顺序很重要：

```python
middleware=[middleware1, middleware2, middleware3]
```

**执行顺序**:

1. **Before 钩子**: 按顺序执行
  ```
   middleware1.before_agent()
   middleware2.before_agent()
   middleware3.before_agent()
  ```
2. **Wrap 钩子**: 嵌套执行（洋葱模型）
  ```
   middleware1.wrap_model_call(
       middleware2.wrap_model_call(
           middleware3.wrap_model_call(
               → 实际模型调用
           )
       )
   )
  ```
3. **After 钩子**: 逆序执行
  ```
   middleware3.after_model()
   middleware2.after_model()
   middleware1.after_model()
  ```

---

## 短期记忆（Short-term Memory）

### 概念

**短期记忆** = 线程级持久化，跟踪单个会话中的对话历史。

**核心机制**:

- LangChain 将短期记忆作为 Agent **状态（State）** 的一部分管理
- 使用 **Checkpointer** 将状态持久化到数据库（或内存）
- 通过 `thread_id` 区分不同的对话会话

### 启用短期记忆

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# 创建检查点器
checkpointer = MemorySaver()

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[],
    checkpointer=checkpointer  # 启用记忆
)

# 第一轮对话
agent.invoke(
    {"messages": [{"role": "user", "content": "你好！我叫张三。"}]},
    {"configurable": {"thread_id": "conversation-1"}}
)

# 第二轮对话 - Agent 记得之前的内容
agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
    {"configurable": {"thread_id": "conversation-1"}}
)
# 输出: 你叫张三。
```

### 自定义状态模式

**默认状态**: `AgentState` 只包含 `messages` 字段

**扩展状态**:

```python
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import MemorySaver

class CustomAgentState(AgentState):
    """扩展的 Agent 状态"""
    user_id: str  # 用户 ID
    preferences: dict  # 用户偏好
    session_data: dict  # 会话数据

agent = create_agent(
    model="gpt-5",
    tools=[get_user_info],
    state_schema=CustomAgentState,
    checkpointer=MemorySaver()
)

# 传入自定义状态
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "推荐一部电影"}],
        "user_id": "user_123",
        "preferences": {"genre": "科幻", "rating": ">8.0"}
    },
    {"configurable": {"thread_id": "1"}}
)
```

### 在中间件中访问状态

```python
from langchain.agents.middleware import before_model

@before_model
def check_authentication(state: AgentState, runtime: Runtime):
    """检查用户是否已认证"""
    is_authenticated = state.get("authenticated", False)

    if not is_authenticated:
        raise ValueError("用户未认证，请先登录")

    return None
```

### 消息修剪

长对话会消耗大量 token，需要修剪消息历史：

```python
from langchain.agents.middleware import before_model, trim_messages

@before_model
async def trim_long_conversations(state: AgentState, runtime: Runtime):
    """修剪过长的对话历史"""
    if len(state["messages"]) > 20:
        trimmed = await trim_messages(
            state["messages"],
            max_tokens=2000,
            strategy="last",  # 保留最后的消息
            start_on="human",  # 从人类消息开始
            end_on=["human", "tool"],  # 以人类或工具消息结束
            include_system=True  # 保留系统消息
        )
        return {"messages": trimmed}

    return None
```

---

## 上下文工程（Context Engineering）

### 概念

**上下文工程** = 在正确的时间将正确的信息提供给模型

**三个关键来源**:

1. **State（状态）**: 当前会话的数据
2. **Runtime Context（运行时上下文）**: 每次调用的配置
3. **Store（存储）**: 跨会话的长期记忆

### 1. 基于状态的动态提示

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def state_aware_prompt(request: ModelRequest) -> str:
    """根据状态生成提示"""
    # 访问消息数量
    message_count = len(request.messages)

    base = "你是一个有帮助的助手。"

    if message_count > 10:
        base += "\n这是一个长对话 - 请保持简洁。"

    # 访问自定义状态字段
    state = request.state
    if state.get("authenticated"):
        base += "\n用户已认证，可以访问敏感功能。"

    return base
```

### 2. 基于运行时上下文的动态提示

```python
from dataclasses import dataclass

@dataclass
class Context:
    user_role: str
    deployment_env: str

@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    """根据运行时上下文生成提示"""
    # 访问运行时上下文
    user_role = request.runtime.context.user_role
    env = request.runtime.context.deployment_env

    base = "你是一个有帮助的助手。"

    if user_role == "admin":
        base += "\n你有管理员权限，可以执行所有操作。"
    elif user_role == "viewer":
        base += "\n你只有只读权限，仅引导用户进行读操作。"

    if env == "production":
        base += "\n请特别小心数据修改操作。"

    return base

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[context_aware_prompt],
    context_schema=Context
)

# 使用时传入上下文
agent.invoke(
    {"messages": [...]},
    context={"user_role": "admin", "deployment_env": "production"}
)
```

### 3. 基于 Store 的长期记忆

```python
from langgraph.store.memory import InMemoryStore

@dynamic_prompt
async def store_aware_prompt(request: ModelRequest) -> str:
    """从长期存储读取用户偏好"""
    user_id = request.runtime.context.user_id

    # 从 Store 读取用户偏好
    store = request.runtime.store
    user_prefs = await store.get(("preferences",), user_id)

    base = "你是一个有帮助的助手。"

    if user_prefs:
        style = user_prefs.value.get("communication_style", "balanced")
        base += f"\n用户偏好 {style} 风格的响应。"

    return base

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[store_aware_prompt],
    context_schema=Context,
    store=InMemoryStore()
)
```

### 4. 动态工具选择

#### 基于状态选择工具

```python
from langchain.agents.middleware import wrap_model_call

@wrap_model_call
def state_based_tools(request: ModelRequest, handler):
    """根据认证状态过滤工具"""
    state = request.state
    is_authenticated = state.get("authenticated", False)
    message_count = len(state["messages"])

    # 未认证：仅公开工具
    if not is_authenticated:
        tools = [t for t in request.tools if t.name.startswith("public_")]
        request = request.override(tools=tools)
    # 对话初期：限制高级工具
    elif message_count < 5:
        tools = [t for t in request.tools if t.name != "advanced_search"]
        request = request.override(tools=tools)

    return handler(request)
```

#### 基于上下文选择工具

```python
@dataclass
class Context:
    user_role: str

@wrap_model_call
def context_based_tools(request: ModelRequest, handler):
    """根据用户角色过滤工具"""
    user_role = request.runtime.context.user_role

    if user_role == "admin":
        # 管理员：所有工具
        pass
    elif user_role == "editor":
        # 编辑者：不能删除
        tools = [t for t in request.tools if t.name != "delete_data"]
        request = request.override(tools=tools)
    else:
        # 查看者：仅只读工具
        tools = [t for t in request.tools if t.name.startswith("read_")]
        request = request.override(tools=tools)

    return handler(request)

agent = create_agent(
    model="gpt-4o",
    tools=[read_data, write_data, delete_data],
    middleware=[context_based_tools],
    context_schema=Context
)
```

---

## 人机协作（Human-in-the-Loop）

### 概念

**Human-in-the-Loop (HITL)** 中间件允许你为 Agent 工具调用添加人工监督。

**工作原理**:

1. 模型提议一个可能需要审查的操作（如删除文件、发送邮件）
2. 中间件根据配置的策略检查工具调用
3. 如果需要干预，中间件发出 **中断（interrupt）**，暂停执行
4. 图状态通过 LangGraph 的持久化层保存
5. 人类做出决定：批准（approve）、编辑（edit）或拒绝（reject）
6. 执行恢复

### 基础配置

```python
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents import human_in_the_loop_middleware
from langgraph.checkpoint.memory import MemorySaver

@tool
def delete_file(path: str) -> str:
    """删除文件"""
    os.remove(path)
    return f"已删除 {path}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件"""
    email_service.send(to, subject, body)
    return f"已发送邮件至 {to}"

# ⚠️ HITL 需要 checkpointer
checkpointer = MemorySaver()

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[delete_file, send_email],
    middleware=[
        human_in_the_loop_middleware(
            interrupt_on={
                "delete_file": True,  # 默认：approve, edit, reject
                "send_email": {"allowed_decisions": ["approve", "reject"]},  # 不允许编辑
            }
        )
    ],
    checkpointer=checkpointer  # 必需！
)
```

### 决策类型


| 决策类型          | 说明                 | 示例用例          |
| ------------- | ------------------ | ------------- |
| ✅ **approve** | 操作按原样批准并执行，无更改     | 按原样发送邮件草稿     |
| ✏️ **edit**   | 工具调用被修改后执行         | 发送邮件前更改收件人    |
| ❌ **reject**  | 工具调用被拒绝，并将解释添加到对话中 | 拒绝邮件草稿并说明如何重写 |


### 执行流程

```python
# 1. 启动 Agent
thread_id = "thread-123"
result = agent.invoke(
    {"messages": [{"role": "user", "content": "删除 report.pdf"}]},
    {"configurable": {"thread_id": thread_id}}
)

# 2. Agent 遇到需要审批的工具调用，暂停执行

# 3. 获取中断请求
from langgraph.types import Command

state = agent.get_state({"configurable": {"thread_id": thread_id}})
interrupt_request = state.values.get("hitl_request")

print(interrupt_request)
# HITLRequest(
#     action_requests=[
#         ActionRequest(
#             tool_call={"name": "delete_file", "args": {"path": "report.pdf"}},
#             decision=None
#         )
#     ]
# )

# 4. 人类做出决定
from langchain.agents.human_in_the_loop import HITLResponse, Decision

# 选项 1: 批准
response = HITLResponse(
    decisions=[Decision(type="approve")]
)

# 选项 2: 编辑
response = HITLResponse(
    decisions=[
        Decision(
            type="edit",
            tool_call={"name": "delete_file", "args": {"path": "backup/report.pdf"}}
        )
    ]
)

# 选项 3: 拒绝
response = HITLResponse(
    decisions=[
        Decision(
            type="reject",
            explanation="请不要删除 report.pdf，我们还需要它"
        )
    ]
)

# 5. 恢复执行
agent.invoke(
    Command(resume=response),
    {"configurable": {"thread_id": thread_id}}
)
```

### 多工具调用

当多个工具调用同时暂停时，必须为每个操作提供决策，且**顺序必须与中断请求中的顺序一致**。

```python
# Agent 同时调用两个工具
interrupt_request.action_requests
# [
#     ActionRequest(tool_call={"name": "send_email", ...}),
#     ActionRequest(tool_call={"name": "delete_file", ...})
# ]

# 提供决策（顺序匹配）
response = HITLResponse(
    decisions=[
        Decision(type="approve"),  # 对应 send_email
        Decision(type="reject", explanation="不要删除")  # 对应 delete_file
    ]
)
```

---

## 结构化输出

### 概念

强制 Agent 返回符合预定义 schema 的结构化数据，而不是自由文本。

### 直接传入 Schema（当前文档推荐的简洁方式）

对于常见场景，可以直接把 Pydantic schema 传给 `response_format`：

```python
from pydantic import BaseModel
from langchain.agents import create_agent

class Answer(BaseModel):
    summary: str
    confidence: float

agent = create_agent(
    "anthropic:claude-sonnet-4-6",
    tools=tools,
    response_format=Answer,
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "总结一下 AI Agent 的趋势"}]
})

answer: Answer = result["structured_response"]
```

### 使用 ToolStrategy

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ProductReview(BaseModel):
    """产品评论分析"""
    rating: int | None = Field(description="产品评分", ge=1, le=5)
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="情感倾向")
    key_points: list[str] = Field(description="关键点，小写，每个 1-3 词")

agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(
        schema=ProductReview,
        tool_message_content="评论分析完成！"  # 可选：自定义工具消息
    )
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "分析评论: '很棒的产品：5星。发货快，但价格贵'"
    }]
})

review: ProductReview = result["structured_response"]
print(f"评分: {review.rating}")
print(f"情感: {review.sentiment}")
print(f"关键点: {review.key_points}")
# 输出:
# 评分: 5
# 情感: positive
# 关键点: ['发货快', '价格贵']
```

### 支持的 Schema 类型

1. **Pydantic 模型**

```python
class WeatherData(BaseModel):
    temperature: float
    condition: str
```

1. **数据类（Dataclass）**

```python
from dataclasses import dataclass

@dataclass
class WeatherData:
    temperature: float
    condition: str
```

1. **TypedDict**

```python
from typing import TypedDict

class WeatherData(TypedDict):
    temperature: float
    condition: str
```

1. **JSON Schema**

```python
schema = {
    "type": "object",
    "properties": {
        "temperature": {"type": "number"},
        "condition": {"type": "string"}
    },
    "required": ["temperature", "condition"]
}
```

1. **Union 类型**（多个 schema 选项）

```python
from typing import Union

class EmailAction(BaseModel):
    type: Literal["email"]
    to: str
    subject: str

class SlackAction(BaseModel):
    type: Literal["slack"]
    channel: str
    message: str

# 模型会根据上下文选择合适的 schema
response_format = ToolStrategy(schema=Union[EmailAction, SlackAction])
```

### 错误处理

```python
from langchain.agents.structured_output import ToolStrategy

def custom_error_handler(e: Exception) -> str:
    """自定义错误处理"""
    return f"结构化输出验证失败: {e}. 请检查数据格式。"

agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(
        schema=ProductReview,
        handle_errors=custom_error_handler  # 自定义错误处理函数
    )
)
```

**内置错误处理选项**:

- `True`: 捕获所有错误，使用默认错误模板
- `str`: 捕获所有错误，使用自定义消息
- `type[Exception]`: 只捕获特定异常类型
- `tuple[type[Exception], ...]`: 捕获多种异常类型
- `Callable[[Exception], str]`: 自定义错误处理函数
- `False`: 不重试，让异常传播

---

## Harness 配置与 Deep Agents

当前官方文档把 `create_agent` 描述为一个可配置的 Harness。中间件是扩展 Harness 的主要方式：每个中间件处理一个关注点，并在 Agent Loop 的合适阶段介入。

### 1. Execution Environment（执行环境）

Agent 需要能执行动作，而不仅是生成文本。执行环境中间件可以提供文件系统、沙箱、解释器或代码执行环境。

```python
from langchain.agents import create_agent
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search],
    middleware=[FilesystemMiddleware(backend=StateBackend())],
)
```

### 2. Context Management（上下文管理）

长任务会不断累积消息、工具结果和中间步骤，容易填满上下文窗口。上下文管理中间件用于摘要历史、加载记忆、按需加载技能，并减少无关上下文。

```python
from deepagents.backends import StateBackend
from deepagents.middleware import (
    FilesystemMiddleware,
    MemoryMiddleware,
    SkillsMiddleware,
    SummarizationMiddleware,
)

backend = StateBackend()
model = "anthropic:claude-sonnet-4-6"

agent = create_agent(
    model=model,
    tools=[search],
    middleware=[
        FilesystemMiddleware(backend=backend),
        SummarizationMiddleware(model=model, backend=backend),
        MemoryMiddleware(backend=backend, sources=["./AGENTS.md"]),
        SkillsMiddleware(backend=backend, sources=["./skills/"]),
    ],
)
```

### 3. Planning and Delegation（规划与委派）

复杂任务可以由主 Agent 规划，再委派给子 Agent 处理。这样可以保持主 Agent 上下文简洁，并让子任务并行或隔离执行。

```python
from langchain.agents.middleware import TodoListMiddleware
from deepagents import SubAgent
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware, SubAgentMiddleware

backend = StateBackend()

researcher: SubAgent = {
    "name": "researcher",
    "description": "Searches and returns a structured summary.",
    "tools": [search],
}

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search],
    middleware=[
        FilesystemMiddleware(backend=backend),
        TodoListMiddleware(),
        SubAgentMiddleware(backend=backend, subagents=[researcher]),
    ],
)
```

### 4. Fault Tolerance（容错）

生产环境中常见限流、超时、模型调用失败、工具 API 抖动等问题。推荐使用中间件统一处理重试和回退逻辑。

```python
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search],
    middleware=[
        ModelRetryMiddleware(max_retries=3),
        ToolRetryMiddleware(max_retries=2),
    ],
)
```

### 5. Guardrails（护栏）

某些策略不应只依赖 prompt，而应通过确定性逻辑强制执行。例如 PII 脱敏、内容策略、输入/输出合规检查等。

```python
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search],
    middleware=[PIIMiddleware()],
)
```

### 6. Steering（人机协作）

对删除、写入、发送邮件、付款、调用高成本 API 等高影响操作，建议使用 Human-in-the-loop 让 Agent 暂停并等待人工审批。

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[write_file, send_email],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": True,
                "send_email": True,
            }
        )
    ],
    checkpointer=InMemorySaver(),
)
```

### 7. Deep Agents

如果要构建长时间运行的编码、研究或复杂任务 Agent，官方建议关注 `create_deep_agent`。它预先组合了常见能力，例如：

- 文件系统
- 摘要
- 子 Agent
- prompt caching

它适合比普通工具调用 Agent 更复杂、更长周期的任务。

---

## 最佳实践

### 1. Agent 设计模式

#### ✅ 单一职责原则

```python
# 好的设计
customer_service_agent = create_agent(
    model=model,
    tools=[search_kb, create_ticket, escalate],
    system_prompt="你是客服专员，专注于解决客户问题"
)

sales_agent = create_agent(
    model=model,
    tools=[search_products, calculate_price, create_order],
    system_prompt="你是销售专员，专注于产品推荐和订单"
)

# ❌ 不好的设计
everything_agent = create_agent(
    model=model,
    tools=[...100个工具],  # 太多职责
    system_prompt="你什么都能做"
)
```

#### ✅ 分层 Agent 架构

```python
# 顶层：路由 Agent
router_agent = create_agent(
    model=model,
    tools=[
        delegate_to_customer_service,
        delegate_to_sales,
        delegate_to_technical
    ],
    system_prompt="根据用户需求路由到专门的 Agent"
)

# 底层：专业 Agent
@tool
def delegate_to_customer_service(query: str) -> str:
    """委托给客服 Agent"""
    result = customer_service_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1]["content"]
```

### 2. 工具设计最佳实践

#### ✅ 详细的文档字符串

```python
@tool
def search_database(query: str, limit: int = 10, filters: dict = {}) -> list:
    """
    在数据库中搜索内容

    使用场景：
    - 查找用户信息
    - 搜索历史记录
    - 检索产品数据

    Args:
        query: 搜索关键词，支持模糊匹配
        limit: 最多返回多少条结果，默认 10
        filters: 额外的过滤条件，如 {"status": "active"}

    Returns:
        匹配的记录列表

    Examples:
        >>> search_database("张三", limit=5)
        [{"name": "张三", "id": 1}, ...]
    """
    results = db.query(query, limit=limit, **filters)
    return results
```

#### ✅ 输入验证

```python
from pydantic import BaseModel, Field, validator

class SearchInput(BaseModel):
    query: str = Field(description="搜索查询")
    limit: int = Field(default=10, ge=1, le=100)

    @validator("query")
    def validate_query(cls, v):
        if len(v) < 2:
            raise ValueError("查询至少需要 2 个字符")
        return v
```

#### ✅ 错误处理

```python
@tool
def risky_operation(param: str) -> str:
    """可能失败的操作"""
    try:
        result = external_api_call(param)
        return result
    except ExternalAPIError as e:
        raise ToolException(f"API 调用失败: {e}. 请稍后重试。")
```

### 3. 提示词工程

#### ✅ 结构化系统提示

```python
SYSTEM_PROMPT = """
# 角色定义
你是一个专业的数据分析助手。

# 核心能力
- 数据查询和分析
- 生成可视化图表
- 提供业务洞察

# 工作流程
1. 理解用户需求
2. 选择合适的工具
3. 分析数据结果
4. 生成清晰的报告

# 输出格式
- 使用 Markdown 格式
- 数据用表格展示
- 结论要简明扼要

# 限制
- 不要编造数据
- 不确定时明确说明
- 超出能力范围时建议人工介入
"""
```

### 4. 性能优化

#### ✅ 使用异步工具

```python
@tool
async def async_search(query: str) -> str:
    """异步搜索"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/search?q={query}") as resp:
            return await resp.text()

# Agent 会自动并行调用多个异步工具
```

#### ✅ 缓存策略

```python
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache

# 生产环境：持久化缓存
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# 相同的调用会从缓存读取
agent.invoke({"messages": [...]})  # 调用 API
agent.invoke({"messages": [...]})  # 从缓存读取
```

### 5. 安全最佳实践

#### ✅ 输入验证

```python
@validator("user_query")
def validate_query(cls, v):
    # 检查 SQL 注入
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT"]
    if any(word in v.upper() for word in forbidden):
        raise ValueError("检测到潜在的 SQL 注入")

    # 长度限制
    if len(v) > 1000:
        raise ValueError("查询过长")

    return v
```

#### ✅ 权限控制

```python
def require_permission(permission: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, config=None, **kwargs):
            user_permissions = config.get("context", {}).get("permissions", [])
            if permission not in user_permissions:
                raise PermissionError(f"需要权限: {permission}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@tool
@require_permission("database.write")
def delete_record(record_id: str) -> str:
    """删除记录（需要写权限）"""
    db.delete(record_id)
    return "已删除"
```

#### ✅ 敏感信息过滤

```python
import re

class SensitiveDataMiddleware(Middleware):
    PATTERNS = {
        "phone": r"\d{11}",
        "email": r"[\w\.-]+@[\w\.-]+\.\w+",
        "id_card": r"\d{17}[\dXx]"
    }

    def after_model(self, state, response, config):
        content = response.content

        # 脱敏处理
        for data_type, pattern in self.PATTERNS.items():
            content = re.sub(pattern, f"[已隐藏的{data_type}]", content)

        response.content = content
        return response
```

---

## 总结

### 核心要点

1. **Agent = Model + Harness**
  - Agent 通过循环调用模型和工具来完成任务
  - Harness 负责提示词、工具、上下文、状态、中间件、检查点和观测
  - 基于 LangGraph 的图状运行时提供灵活控制
2. **create_agent 是标准方式**
  - 简单易用（<10 行代码）
  - 高度可定制（通过中间件）
  - 生产就绪（持久化、监控等）
3. **中间件是核心扩展点**
  - 在执行的各个阶段插入自定义逻辑
  - 支持动态提示、工具选择、错误处理等
  - 可组合、可复用
4. **上下文工程很关键**
  - State: 会话数据
  - Runtime Context: 运行时配置
  - Store: 长期记忆
  - 在正确的时间提供正确的信息
5. **记忆管理分两种**
  - 短期记忆: Checkpointer + thread_id
  - 长期记忆: Store + 自定义命名空间
6. **人机协作提升安全性**
  - 敏感操作需要人工批准
  - 支持批准、编辑、拒绝三种决策
  - 需要 Checkpointer 支持

### 推荐学习路径

1. **入门** (1-2 天)
  - 创建第一个 Agent
  - 定义简单工具
  - 理解基本执行流程
2. **进阶** (1 周)
  - 使用中间件自定义行为
  - 添加短期记忆
  - 实现动态提示和工具选择
3. **高级** (2-4 周)
  - 构建多 Agent 系统
  - 实现人机协作
  - 优化性能和安全性
  - 集成长期记忆

### 参考资源

- **官方文档**: [https://docs.langchain.com/oss/python/langchain/agents](https://docs.langchain.com/oss/python/langchain/agents)
- **API 参考**: [https://api.python.langchain.com/](https://api.python.langchain.com/)
- **LangGraph 文档**: [https://docs.langchain.com/langgraph](https://docs.langchain.com/langgraph)
- **示例项目**: [https://github.com/langchain-ai/langchain/tree/master/templates](https://github.com/langchain-ai/langchain/tree/master/templates)

---

**文档版本**: 1.1
**最后更新**: 2026-05-31
**基于**: LangChain v1.0 官方文档

如有问题或建议，欢迎反馈！