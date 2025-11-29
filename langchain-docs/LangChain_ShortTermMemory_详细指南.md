# LangChain Short-term Memory 详细指南

> 基于官方文档 https://docs.langchain.com/oss/python/langchain/short-term-memory 的完整中文总结

---

## 📋 目录

- [核心概念](#核心概念)
- [为什么需要短期记忆](#为什么需要短期记忆)
- [短期记忆的实现方式](#短期记忆的实现方式)
- [常见的记忆管理模式](#常见的记忆管理模式)
- [消息修剪 (Trim Messages)](#消息修剪-trim-messages)
- [消息删除 (Delete Messages)](#消息删除-delete-messages)
- [消息总结 (Summarize Messages)](#消息总结-summarize-messages)
- [自定义 Agent 记忆](#自定义-agent-记忆)
- [Checkpointer 的使用](#checkpointer-的使用)
- [最佳实践](#最佳实践)
- [性能优化](#性能优化)

---

## 核心概念

### 什么是短期记忆 (Short-term Memory)？

**短期记忆**是一种让应用程序在单个线程或对话中记住之前交互的系统。它是 AI Agent 记忆系统的重要组成部分，使 Agent 能够：

- 📝 记住之前的对话内容
- 🔄 从用户反馈中学习
- 🎯 适应用户偏好
- 💬 维持连贯的多轮对话

### 短期记忆 vs 长期记忆

| 特性 | 短期记忆 (Short-term) | 长期记忆 (Long-term) |
|------|---------------------|---------------------|
| **作用域** | 单个对话线程 (Thread) | 跨多个会话 |
| **生命周期** | 会话期间 | 永久存储 |
| **存储位置** | Graph State | Store |
| **典型内容** | 对话历史、临时数据 | 用户偏好、历史交互 |
| **管理方式** | Checkpointer | BaseStore |
| **更新时机** | 每次 invoke/step | 按需更新 |

### 线程 (Thread) 的概念

**Thread（线程）** 组织单个会话中的多个交互，类似于电子邮件中的对话线程。

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
agent = create_agent(model="gpt-5.0", tools=[], checkpointer=checkpointer)

# 同一线程的多次交互
config = {"configurable": {"thread_id": "conversation_1"}}

# 第一轮对话
agent.invoke({"messages": "你好，我叫 Alice"}, config)

# 第二轮对话 - Agent 记住之前的对话
agent.invoke({"messages": "我叫什么名字?"}, config)
# 输出: "你叫 Alice"
```

---

## 为什么需要短期记忆

### 1. 上下文窗口的挑战

大多数 LLM 都有最大上下文窗口限制：

- **gpt-5.0**: 400K tokens
- **Claude 4.5 Sonnet**: 200K tokens  
- **Gemini 3.0 Pro**: 1M tokens

长对话可能超出这些限制，导致：
- ❌ 上下文丢失或错误
- ❌ 响应时间变慢
- ❌ API 成本增加

### 2. 长上下文性能问题

即使模型支持长上下文，也存在问题：

- **注意力分散**: 被过时或无关内容干扰
- **性能下降**: 处理时间和质量都会下降
- **成本增加**: Token 使用量激增

### 3. 实际案例

```python
# 问题场景：100 轮对话后
messages = [
    SystemMessage("你是助手"),
    HumanMessage("问题1"),
    AIMessage("答案1"),
    # ... 200+ 条消息
    HumanMessage("最新问题")
]

# 可能遇到的问题：
# - 超出上下文窗口限制
# - 模型被早期无关对话干扰
# - Token 成本过高
```

---

## 短期记忆的实现方式

### 1. 使用 MessagesState

LangChain 提供了预构建的 `MessagesState` 来管理对话历史。

```python
from langgraph.graph import MessagesState

# MessagesState 包含一个 messages 键
class State(MessagesState):
    # 可以添加其他字段
    documents: list[str]
```

**MessagesState 的特点**:
- 自动包含 `messages` 键
- 使用 `add_messages` reducer
- 支持消息的添加、更新和删除

### 2. 使用 add_messages Reducer

`add_messages` 是一个智能 reducer，能够：

```python
from langgraph.graph import add_messages
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# add_messages 的功能：
# 1. 添加新消息到列表
# 2. 更新具有相同 ID 的消息
# 3. 保持消息顺序
```

**示例**:

```python
from langgraph.graph import StateGraph, MessagesState
from langchain_core.messages import HumanMessage, AIMessage

def chatbot(state: MessagesState):
    return {"messages": [AIMessage(content="你好！")]}

graph = StateGraph(MessagesState)
graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")

compiled = graph.compile()

# 调用
result = compiled.invoke({
    "messages": [HumanMessage(content="你好")]
})

print(result["messages"])
# [HumanMessage("你好"), AIMessage("你好！")]
```

### 3. 启用持久化

使用 **Checkpointer** 在多次调用之间保持状态：

```python
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

# 创建内存保存器
checkpointer = MemorySaver()

# 创建 Agent
agent = create_agent(
    model="gpt-5.0",
    tools=[],
    checkpointer=checkpointer
)

# 配置线程 ID
config = {"configurable": {"thread_id": "1"}}

# 第一次调用
agent.invoke({"messages": "我喜欢 Python"}, config)

# 第二次调用 - 会记住之前的对话
agent.invoke({"messages": "我喜欢什么?"}, config)
# 输出: "你喜欢 Python"
```

---

## 常见的记忆管理模式

当对话变长时，需要采用策略管理消息历史：

### 1. 消息修剪 (Trim Messages)

保留最近的 N 条消息，删除较早的消息。

**优点**:
- ✅ 简单直接
- ✅ 性能开销小
- ✅ 可预测的 token 使用

**缺点**:
- ❌ 可能丢失重要的早期信息
- ❌ 上下文可能不完整

### 2. 消息删除 (Delete Messages)

永久从状态中删除特定消息。

**优点**:
- ✅ 精确控制保留内容
- ✅ 可以删除敏感信息

**缺点**:
- ❌ 不可恢复
- ❌ 需要仔细管理消息有效性

### 3. 消息总结 (Summarize Messages)

使用模型总结早期消息，用摘要替换原消息。

**优点**:
- ✅ 保留关键信息
- ✅ 支持更长的有效对话
- ✅ 平衡了上下文和长度

**缺点**:
- ❌ 额外的 LLM 调用成本
- ❌ 可能丢失细节

### 4. 自定义策略

根据业务需求实现特定的过滤或管理逻辑。

**示例**: 只保留包含特定关键词的消息、基于重要性评分等。

---

## 消息修剪 (Trim Messages)

### 基本修剪策略

使用 `trim_messages` 函数按 token 数量修剪消息：

```python
from langchain_core.messages import trim_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="你是一个有帮助的助手"),
    HumanMessage(content="你好"),
    AIMessage(content="你好！有什么可以帮你的吗？"),
    HumanMessage(content="我想了解 AI"),
    AIMessage(content="AI 是人工智能的缩写..."),
    HumanMessage(content="告诉我更多"),
]

# 修剪到最多 1000 tokens
trimmed = trim_messages(
    messages,
    max_tokens=1000,
    strategy="last",  # 保留最后的消息
    token_counter=len,  # 使用简单的计数器
    include_system=True  # 始终保留系统消息
)
```

### 在 Agent 中使用修剪

#### 方法 1: 使用 @before_model 中间件

```python
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """保留最近的几条消息以适应上下文窗口"""
    messages = state["messages"]
    
    if len(messages) <= 3:
        return None  # 不需要修剪
    
    # 保留第一条消息（通常是系统消息）
    first_msg = messages[0]
    
    # 保留最近的消息
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages
    
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

agent = create_agent(
    model="gpt-5.0",
    tools=[],
    middleware=[trim_messages],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "你好，我叫 Bob"}, config)
agent.invoke({"messages": "写一首关于猫的诗"}, config)
agent.invoke({"messages": "现在写一首关于狗的诗"}, config)
result = agent.invoke({"messages": "我叫什么名字?"}, config)

result["messages"][-1].pretty_print()
# 输出: "你叫 Bob。你之前告诉过我。"
```

#### 方法 2: 使用 LangGraph 手动修剪

```python
from langchain_core.messages import trim_messages
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import MemorySaver

model = ChatAnthropic(model="claude-sonnet-4-5-20250929")

def call_model(state: MessagesState):
    # 在调用模型前修剪消息
    trimmed = trim_messages(
        state["messages"],
        max_tokens=128,
        strategy="last",
        token_counter=model,  # 使用模型的 token 计数器
        start_on="human",
        end_on=["human", "tool"],
    )
    response = model.invoke(trimmed)
    return {"messages": [response]}

checkpointer = MemorySaver()
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": [{"role": "user", "content": "你好，我叫 Bob"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "写一首关于猫的诗"}]}, config)
result = graph.invoke({"messages": [{"role": "user", "content": "我叫什么名字?"}]}, config)

print(result["messages"][-1].content)
```

### trim_messages 参数详解

```python
trimmed = trim_messages(
    messages,
    
    # 基本参数
    max_tokens=1000,        # 最大 token 数
    strategy="last",         # 策略: "first" 或 "last"
    
    # Token 计数
    token_counter=model,    # 使用模型的计数器，或自定义函数
    
    # 消息选择
    include_system=True,    # 始终包含系统消息
    start_on="human",       # 从哪种消息类型开始
    end_on=["human", "tool"], # 在哪种消息类型结束
    
    # 其他选项
    allow_partial=False,    # 是否允许部分消息
)
```

---

## 消息删除 (Delete Messages)

### 使用 RemoveMessage

`RemoveMessage` 允许从状态中永久删除消息。

#### 删除特定消息

```python
from langchain.messages import RemoveMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """删除旧消息以保持对话可管理"""
    messages = state["messages"]
    
    if len(messages) > 2:
        # 删除最早的两条消息
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    
    return None

agent = create_agent(
    model="gpt-5.0",
    tools=[],
    middleware=[delete_old_messages],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "你好，我叫 Bob"}, config)
agent.invoke({"messages": "我喜欢 Python"}, config)
agent.invoke({"messages": "我叫什么名字?"}, config)
```

#### 删除所有消息

```python
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.messages import RemoveMessage

def clear_conversation(state):
    """清空整个对话历史"""
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```

#### 在 LangGraph 中删除消息

```python
from langchain.messages import RemoveMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import MemorySaver

model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

def delete_messages(state: MessagesState):
    messages = state["messages"]
    if len(messages) > 2:
        # 删除最早的两条消息
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return {}

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

checkpointer = MemorySaver()
builder = StateGraph(MessagesState)
builder.add_node("call_model", call_model)
builder.add_node("delete_messages", delete_messages)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", "delete_messages")

graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": [{"role": "user", "content": "你好，我叫 Bob"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "我叫什么名字?"}]}, config)
```

### 删除消息的注意事项

⚠️ **重要**: 删除消息时确保结果消息历史有效：

1. **某些提供商期望消息从用户消息开始**
2. **大多数提供商要求工具调用后必须有对应的工具结果消息**

```python
# ✅ 有效的消息序列
[
    HumanMessage("问题"),
    AIMessage("", tool_calls=[...]),
    ToolMessage("结果", tool_call_id="..."),
    AIMessage("回答")
]

# ❌ 无效的消息序列
[
    AIMessage("", tool_calls=[...]),  # 缺少对应的 ToolMessage
    AIMessage("回答")
]
```

---

## 消息总结 (Summarize Messages)

### 为什么使用总结？

修剪和删除消息会丢失信息，而总结可以：
- ✅ 保留关键信息
- ✅ 压缩对话历史
- ✅ 支持更长的有效对话

### 使用内置的 SummarizationMiddleware

#### 在 Agent 中使用

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

agent = create_agent(
    model="gpt-5.0",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.0-mini",              # 用于总结的模型
            max_tokens_before_summary=4000,   # 触发总结的阈值
            messages_to_keep=20,              # 总结后保留的消息数
        )
    ],
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "你好，我叫 Bob"}, config)
agent.invoke({"messages": "写一首关于猫的诗"}, config)
agent.invoke({"messages": "现在写一首关于狗的诗"}, config)
result = agent.invoke({"messages": "我叫什么名字?"}, config)

result["messages"][-1].pretty_print()
# 输出: "你叫 Bob！"
```

### SummarizationMiddleware 配置选项

```python
SummarizationMiddleware(
    model="gpt-5.0-mini",              # 总结模型
    max_tokens_before_summary=4000,   # Token 阈值
    messages_to_keep=20,              # 保留的最近消息数
    token_counter=None,               # 自定义 token 计数器
    summary_prompt=None,              # 自定义总结提示词
)
```

### 在 LangGraph 中手动实现总结

```python
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage, HumanMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver

model = init_chat_model("claude-3-5-sonnet-20241022")

# 扩展状态以包含总结
class State(MessagesState):
    summary: str = ""

def call_model(state: State):
    # 如果有总结，添加为系统消息
    summary = state.get("summary", "")
    messages = state["messages"]
    
    if summary:
        system_msg = SystemMessage(content=f"之前对话的总结: {summary}")
        messages = [system_msg] + messages
    
    response = model.invoke(messages)
    return {"messages": [response]}

def should_continue(state: State):
    """决定是否需要总结"""
    if len(state["messages"]) > 6:
        return "summarize"
    return END

def summarize_conversation(state: State):
    """总结对话历史"""
    summary = state.get("summary", "")
    messages = state["messages"]
    
    # 创建总结提示
    if summary:
        summary_msg = (
            f"这是到目前为止的对话总结: {summary}\n\n"
            "考虑上面的新消息，扩展总结:"
        )
    else:
        summary_msg = "创建上述对话的总结:"
    
    all_messages = messages + [HumanMessage(content=summary_msg)]
    response = model.invoke(all_messages)
    
    # 删除除最后两条外的所有消息
    delete_msgs = [RemoveMessage(id=m.id) for m in messages[:-2]]
    
    return {"summary": response.content, "messages": delete_msgs}

# 构建图
checkpointer = InMemorySaver()
builder = StateGraph(State)
builder.add_node("call_model", call_model)
builder.add_node("summarize", summarize_conversation)
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", should_continue)
builder.add_edge("summarize", END)

graph = builder.compile(checkpointer=checkpointer)

# 使用
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "你好，我叫 Alice"}, config)
graph.invoke({"messages": "我喜欢编程"}, config)
graph.invoke({"messages": "特别是 Python"}, config)
graph.invoke({"messages": "我还喜欢机器学习"}, config)
result = graph.invoke({"messages": "我叫什么名字，我喜欢什么?"}, config)

print(result["messages"][-1].content)
print(f"\n总结: {result['summary']}")
```

### 使用 SummarizationNode (langmem)

对于更高级的总结功能，可以使用 `langmem` 库：

```python
from langmem.short_term import SummarizationNode, RunningSummary
from langchain_core.messages.utils import count_tokens_approximately
from typing import TypedDict

model = init_chat_model("claude-3-5-sonnet-20241022")
summarization_model = model.bind(max_tokens=128)

class State(MessagesState):
    context: dict[str, RunningSummary]

class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, RunningSummary]

summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=256,
    max_tokens_before_summary=256,
    max_summary_tokens=128,
)

def call_model(state: LLMInputState):
    response = model.invoke(state["summarized_messages"])
    return {"messages": [response]}

checkpointer = InMemorySaver()
builder = StateGraph(State)
builder.add_node("call_model", call_model)
builder.add_node("summarize", summarization_node)
builder.add_edge(START, "summarize")
builder.add_edge("summarize", "call_model")
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "你好，我叫 Bob"}, config)
graph.invoke({"messages": "写一首关于猫的诗"}, config)
result = graph.invoke({"messages": "我叫什么名字?"}, config)

print(result["messages"][-1].content)
print(f"\n总结: {result['context']['running_summary'].summary}")
```

---

## 自定义 Agent 记忆

### 扩展 AgentState

默认情况下，Agent 使用 `AgentState` 管理短期记忆。你可以扩展它以添加自定义字段。

#### 方法 1: 使用 state_schema (传统方式)

```python
from langchain.agents import create_agent, AgentState
from typing import TypedDict
from langchain.messages import AnyMessage

class CustomState(AgentState):
    user_id: str
    preferences: dict[str, str]

agent = create_agent(
    model="gpt-5.0",
    tools=[],
    state_schema=CustomState,
)

# 使用自定义状态
result = agent.invoke({
    "messages": "你好",
    "user_id": "user_123",
    "preferences": {"theme": "dark"}
})
```

#### 方法 2: 使用 Middleware (推荐)

```python
from langchain.agents import create_agent, create_middleware
from langgraph.checkpoint.memory import MemorySaver
import z from "zod"

# 定义自定义状态 schema
custom_state_schema = {
    "user_id": str,
    "preferences": dict[str, Any],
}

# 创建状态扩展中间件
state_extension_middleware = create_middleware(
    name="StateExtension",
    state_schema=custom_state_schema,
)

checkpointer = MemorySaver()
agent = create_agent(
    model="gpt-5.0",
    tools=[],
    middleware=[state_extension_middleware],
    checkpointer=checkpointer,
)

# 调用时传递自定义状态
result = agent.invoke({
    "messages": [{"role": "user", "content": "你好"}],
    "user_id": "user_123",
    "preferences": {"theme": "dark"},
})
```

### 在工具中访问自定义状态

```python
from langchain_core.tools import tool
from langchain.tools import ToolRuntime

@tool
def get_user_preference(
    preference_name: str,
    runtime: ToolRuntime
) -> str:
    """获取用户偏好设置"""
    preferences = runtime.state.get("preferences", {})
    return preferences.get(preference_name, "未设置")

agent = create_agent(
    model="gpt-5.0",
    tools=[get_user_preference],
    middleware=[state_extension_middleware],
    checkpointer=checkpointer,
)
```

---

## Checkpointer 的使用

### 什么是 Checkpointer？

**Checkpointer** 负责持久化 Agent 的状态，使对话能够跨多次调用保持连续。

### 内存 Checkpointer

用于开发和测试，数据存储在内存中：

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
```

**特点**:
- ✅ 快速、简单
- ✅ 无需外部依赖
- ❌ 进程结束时数据丢失

### SQLite Checkpointer

将状态持久化到 SQLite 数据库：

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
```

**特点**:
- ✅ 持久化存储
- ✅ 轻量级
- ❌ 不适合高并发

### PostgreSQL Checkpointer

用于生产环境的强大选择：

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:password@localhost/dbname"
)
```

**特点**:
- ✅ 生产就绪
- ✅ 支持高并发
- ✅ 可扩展

### 使用 Checkpointer

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

agent = create_agent(
    model="gpt-5.0",
    tools=[],
    checkpointer=checkpointer
)

# 使用线程 ID 进行对话
config_1 = {"configurable": {"thread_id": "conversation_1"}}
config_2 = {"configurable": {"thread_id": "conversation_2"}}

# 对话 1
agent.invoke({"messages": "我叫 Alice"}, config_1)
agent.invoke({"messages": "我叫什么?"}, config_1)  # "Alice"

# 对话 2 (独立的线程)
agent.invoke({"messages": "我叫 Bob"}, config_2)
agent.invoke({"messages": "我叫什么?"}, config_2)  # "Bob"
```

### 查看状态历史

```python
# 获取当前状态
state = graph.get_state(config)
print(state.values)
print(state.next)  # 下一个要执行的节点

# 获取状态历史
history = list(graph.get_state_history(config))
for snapshot in history:
    print(f"Step {snapshot.metadata['step']}")
    print(f"Messages: {len(snapshot.values['messages'])}")
```

---

## 最佳实践

### 1. 选择合适的记忆管理策略

```python
# 短对话 (< 10 轮) - 不需要特殊处理
agent = create_agent(model="gpt-5.0", tools=[], checkpointer=checkpointer)

# 中等对话 (10-50 轮) - 使用消息修剪
agent = create_agent(
    model="gpt-5.0",
    tools=[],
    middleware=[trim_messages_middleware],
    checkpointer=checkpointer,
)

# 长对话 (> 50 轮) - 使用消息总结
agent = create_agent(
    model="gpt-5.0",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-5.0-mini",
            max_tokens_before_summary=4000,
            messages_to_keep=20,
        )
    ],
    checkpointer=checkpointer,
)
```

### 2. 始终保留系统消息

```python
@before_model
def trim_messages(state: AgentState, runtime: Runtime):
    messages = state["messages"]
    
    # ✅ 保留系统消息
    system_msg = messages[0]
    recent_msgs = messages[-10:]
    
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            system_msg,
            *recent_msgs
        ]
    }
```

### 3. 使用适当的 Token 计数器

```python
from langchain_core.messages import trim_messages
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-sonnet-4-5-20250929")

# ✅ 使用模型的 token 计数器
trimmed = trim_messages(
    messages,
    max_tokens=1000,
    token_counter=model,  # 准确的 token 计数
)

# ⚠️ 使用简单计数器（快速但不准确）
trimmed = trim_messages(
    messages,
    max_tokens=1000,
    token_counter=len,  # 字符数，不是 token 数
)
```

### 4. 监控和记录记忆使用

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@before_model
def trim_with_logging(state: AgentState, runtime: Runtime):
    messages = state["messages"]
    original_count = len(messages)
    
    # 执行修剪
    if original_count > 10:
        trimmed_messages = messages[-10:]
        logger.info(f"修剪消息: {original_count} -> {len(trimmed_messages)}")
        
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *trimmed_messages
            ]
        }
    
    return None
```

### 5. 处理多模态内容

```python
from langchain_core.messages import trim_messages

# 多模态消息可能消耗大量 tokens
trimmed = trim_messages(
    messages,
    max_tokens=2000,  # 为图像等留出更多空间
    token_counter=model,
    include_system=True,
)
```

### 6. 定期清理过期线程

```python
from datetime import datetime, timedelta

def cleanup_old_threads(checkpointer, max_age_days=30):
    """清理超过指定天数的线程"""
    cutoff_date = datetime.now() - timedelta(days=max_age_days)
    
    # 实现取决于 checkpointer 类型
    # 这是一个概念示例
    for thread_id in checkpointer.list_threads():
        last_update = checkpointer.get_last_update(thread_id)
        if last_update < cutoff_date:
            checkpointer.delete_thread(thread_id)

# 定期运行
cleanup_old_threads(checkpointer)
```

### 7. 测试不同的策略

```python
import pytest
from langchain.agents import create_agent

@pytest.fixture
def agent_with_trim():
    return create_agent(
        model="gpt-5.0",
        tools=[],
        middleware=[trim_middleware],
        checkpointer=MemorySaver(),
    )

@pytest.fixture
def agent_with_summary():
    return create_agent(
        model="gpt-5.0",
        tools=[],
        middleware=[SummarizationMiddleware(...)],
        checkpointer=MemorySaver(),
    )

def test_long_conversation_trim(agent_with_trim):
    config = {"configurable": {"thread_id": "test"}}
    
    # 模拟长对话
    for i in range(20):
        agent_with_trim.invoke({"messages": f"消息 {i}"}, config)
    
    # 验证记忆管理
    state = agent_with_trim.get_state(config)
    assert len(state.values["messages"]) <= 10

def test_long_conversation_summary(agent_with_summary):
    # 类似的测试...
    pass
```

---

## 性能优化

### 1. 使用轻量级模型进行总结

```python
# ✅ 好的做法
SummarizationMiddleware(
    model="gpt-5.0-mini",  # 快速、便宜的模型
    max_tokens_before_summary=4000,
)

# ❌ 避免
SummarizationMiddleware(
    model="gpt-5.0",  # 慢、贵
    max_tokens_before_summary=4000,
)
```

### 2. 批量处理消息删除

```python
# ✅ 一次性删除多条消息
@after_model
def batch_delete(state: AgentState, runtime: Runtime):
    messages = state["messages"]
    if len(messages) > 20:
        to_delete = messages[:10]  # 删除最早的 10 条
        return {"messages": [RemoveMessage(id=m.id) for m in to_delete]}
    return None

# ❌ 避免频繁的小批量删除
@after_model
def frequent_delete(state: AgentState, runtime: Runtime):
    messages = state["messages"]
    if len(messages) > 5:
        return {"messages": [RemoveMessage(id=messages[0].id)]}  # 每次只删一条
    return None
```

### 3. 缓存 Token 计数

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def count_tokens(text: str) -> int:
    """缓存的 token 计数"""
    return len(text) // 4  # 简化示例

# 使用缓存的计数器
trimmed = trim_messages(
    messages,
    max_tokens=1000,
    token_counter=lambda msgs: sum(count_tokens(m.content) for m in msgs),
)
```

### 4. 异步处理总结

```python
import asyncio

async def async_summarize(messages, model):
    """异步总结消息"""
    summary_prompt = "总结以下对话:"
    all_msgs = messages + [HumanMessage(content=summary_prompt)]
    response = await model.ainvoke(all_msgs)
    return response.content

# 在后台总结，不阻塞主流程
```

### 5. 监控 Token 使用

```python
from langchain_core.messages import count_tokens_approximately

def monitor_token_usage(state: AgentState):
    """监控当前对话的 token 使用"""
    messages = state["messages"]
    total_tokens = count_tokens_approximately(messages)
    
    logger.info(f"当前 tokens: {total_tokens}")
    
    if total_tokens > 8000:
        logger.warning("Token 使用接近上限，考虑总结")
    
    return {"token_usage": total_tokens}
```

---

## 🎯 快速参考

### 记忆管理策略对比

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **修剪** | 简单、快速、可预测 | 丢失早期信息 | 短期对话 (< 20 轮) |
| **删除** | 精确控制 | 不可恢复 | 删除敏感信息 |
| **总结** | 保留关键信息 | 额外成本 | 长期对话 (> 50 轮) |
| **混合** | 平衡各方面 | 复杂度高 | 复杂应用 |

### 常用代码片段

```python
# 1. 启用基本记忆
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
agent = create_agent(model="gpt-5.0", tools=[], checkpointer=checkpointer)

# 2. 修剪消息
from langchain.agents.middleware import before_model
@before_model
def trim(state, runtime):
    messages = state["messages"]
    return {"messages": [messages[0]] + messages[-10:]} if len(messages) > 10 else None

# 3. 总结消息
from langchain.agents.middleware import SummarizationMiddleware
middleware = [SummarizationMiddleware(model="gpt-5.0-mini", max_tokens_before_summary=4000)]

# 4. 删除消息
from langchain.messages import RemoveMessage
return {"messages": [RemoveMessage(id=m.id) for m in old_messages]}

# 5. 查看状态
state = graph.get_state(config)
history = list(graph.get_state_history(config))
```

---

## 🔗 相关资源

- **官方文档**: https://docs.langchain.com/oss/python/langchain/short-term-memory
- **配套文档**:
  - [LangChain Messages 详细指南](./LangChain_Messages_详细指南.md)
  - [LangChain Models 详细指南](./LangChain_Models_详细指南.md)
  - [LangChain Agents 详细总结](./LangChain_Agents_详细总结.md)

---

**文档版本**: 1.0  
**最后更新**: 2025年1月  
**基于**: LangChain v0.3+, Python 3.9+

本文档涵盖了 LangChain Short-term Memory 的核心概念、实现方式、管理策略和最佳实践，包含 60+ 实用代码示例。
