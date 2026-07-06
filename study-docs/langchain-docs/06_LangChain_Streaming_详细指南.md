# LangChain Streaming 详细指南

> 基于官方文档 https://docs.langchain.com/oss/python/langchain/streaming 与 https://docs.langchain.com/oss/python/langchain/event-streaming 的完整中文总结。本文在原有内容基础上补充并优化当前 LangChain v1.x 的 `stream_events(..., version="v3")` typed projections、`stream(..., version="v2")` chunk format、Agent progress、Reasoning tokens、Tool calls、Human-in-the-loop 与 Sub-agents 等内容。

---

## 目录

1. [概述](#概述)
2. [为什么需要 Streaming](#为什么需要-streaming)
3. [核心概念](#核心概念)
4. [LangChain 中的 Streaming](#langchain-中的-streaming)
   - [Model Streaming](#model-streaming)
   - [Agent Streaming](#agent-streaming)
   - [自动流式传输](#自动流式传输)
5. [LangGraph 中的 Streaming](#langgraph-中的-streaming)
   - [Stream 模式详解](#stream-模式详解)
   - [流式传输图状态](#流式传输图状态)
   - [流式传输 LLM Tokens](#流式传输-llm-tokens)
   - [流式传输自定义数据](#流式传输自定义数据)
   - [流式传输子图输出](#流式传输子图输出)
   - [调试模式](#调试模式)
6. [v1.x 官方补充：常见 Streaming 模式](#v1x-官方补充常见-streaming-模式)
7. [Event Streaming（stream_events）](#event-streamingstream_events)
8. [高级特性](#高级特性)
   - [多模式组合](#多模式组合)
   - [禁用 Streaming](#禁用-streaming)
   - [过滤特定节点](#过滤特定节点)
9. [消息块处理](#消息块处理)
10. [实际应用场景](#实际应用场景)
11. [最佳实践](#最佳实践)
12. [性能优化](#性能优化)
13. [快速参考](#快速参考)

---

## 概述

LangChain 实现了一个强大的流式传输系统，用于实时显示更新。流式传输对于增强基于 LLM 的应用程序的响应性至关重要。通过逐步显示输出（即使在完整响应准备好之前），流式传输显著改善了用户体验（UX），特别是在处理 LLM 的延迟时。

**关键优势：**
- 🚀 实时反馈 - 用户立即看到进展
- ⚡ 改善感知性能 - 即使总时间相同，流式传输让应用感觉更快
- 📊 进度可视化 - 可以显示中间步骤和状态
- 🎯 更好的 UX - 特别是对于长响应

---

## 为什么需要 Streaming

### 1. **LLM 延迟问题**
大型语言模型生成响应需要时间，特别是对于长输出：
- GPT-4 生成 500 字可能需要 10-20 秒
- 用户期望即时反馈
- 流式传输让等待过程更加可控

### 2. **用户体验**
```python
# 非流式：用户等待 15 秒，然后看到完整响应
response = model.invoke("写一篇关于 AI 的文章")
print(response.content)  # 15 秒后显示

# 流式：用户立即看到文字逐渐出现
for chunk in model.stream("写一篇关于 AI 的文章"):
    print(chunk.text, end="", flush=True)  # 实时显示
```

### 3. **Agent 可观察性**
在复杂的 Agent 系统中，流式传输让你看到：
- Agent 正在调用哪个工具
- 工具执行的进度
- 中间思考过程

---

## 核心概念

### 两套 Streaming API

当前 LangChain 文档中有两套常见的流式接口，定位不同：

| API | 推荐场景 | 消费方式 | 典型版本 |
|-----|----------|----------|----------|
| `stream_events()` / `astream_events()` | 应用层、前端、Agent UI，想分别消费 messages、tool calls、state、subagents | typed projections，例如 `stream.messages`、`stream.tool_calls`、`stream.values` | `version="v3"` |
| `stream()` / `astream()` | LangGraph/Pregel 低层流式模式，按 `stream_mode` 消费 updates、messages、custom 等 chunk | `chunk["type"]`、`chunk["ns"]`、`chunk["data"]` | `version="v2"` |

简单选择：

| 需求 | 推荐 |
|------|------|
| 新应用要构建实时 Agent UI | 优先用 `stream_events(..., version="v3")` |
| 只想看 Agent 每一步进度 | 用 `stream(..., stream_mode="updates", version="v2")` |
| 想逐 token 展示模型输出 | `stream_events().messages` 或 `stream_mode="messages"` |
| 想展示工具执行生命周期 | `stream_events().tool_calls` |
| 想保留已有 LangGraph streaming 代码 | 继续使用 `stream()` / `astream()` |

### Stream Mode（流式模式）

`stream()` / `astream()` 支持多种流式模式，每种模式提供不同级别的信息：

| 模式 | 描述 | 用途 |
|------|------|------|
| `values` | 每步后的完整状态 | 查看完整的图状态 |
| `updates` | 每步的状态更新（增量） | 只看变化部分 |
| `messages` | LLM token 流 + 元数据 | 流式显示 LLM 输出 |
| `custom` | 自定义用户数据 | 进度更新、日志等 |
| `debug` | 详细的执行信息 | 调试和故障排除 |

> 注意：官方 LangChain Streaming 文档的主线示例目前重点展示 `updates`、`messages`、`custom`；`values` 和 `debug` 更多见于 LangGraph streaming 语境。

### 流式输出类型

1. **Token-level streaming** - 逐个 token 输出
2. **Step-level streaming** - 每个节点/步骤的输出
3. **Projection-level streaming** - 用 `stream_events()` 分别消费 messages、tool calls、values、subagents 等语义投影
4. **Raw event streaming** - 直接遍历底层事件 envelope，用于调试或构建自定义投影

---

## LangChain 中的 Streaming

### Model Streaming

#### 基本用法

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(model="glm-4.5-air")

# 流式输出 tokens
for chunk in model.stream("什么颜色是天空？"):
    print(chunk.text, end="", flush=True)
```

**输出：**
```
天
天空
天空通常
天空通常是
天空通常是蓝色
...
```

#### 累积消息块

```python
from langchain_core.messages import AIMessageChunk

# 累积完整消息
full = None
for chunk in model.stream("你好"):
    full = chunk if full is None else full + chunk
    print(full.text)

# 最终的 full 是一个完整的 AIMessage
print(full.content_blocks)
# [{"type": "text", "text": "你好！有什么可以帮你的吗？"}]
```

#### 流式工具调用

```python
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city} 今天晴天"

model_with_tools = model.bind_tools([get_weather])

# 工具调用在流式传输中逐步构建
for chunk in model_with_tools.stream("波士顿天气如何？"):
    if chunk.tool_call_chunks:
        for tool_chunk in chunk.tool_call_chunks:
            print(f"Tool: {tool_chunk.get('name', '')}")
            print(f"Args: {tool_chunk.get('args', '')}")
```

**输出：**
```
Tool: get_weather
Args:
Tool:
Args: {"city
Tool:
Args: ": "Boston"}
```

### Agent Streaming

#### 流式 Agent 进度

```python
from langchain.agents import create_agent

agent = create_agent(
    model="glm-4.5-air",
    tools=[get_weather],
)

# stream_mode="updates" 显示每个步骤后的更新
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "SF 的天气"}]},
    stream_mode="updates"
):
    print(chunk)
```

**输出示例：**
```python
# LLM 节点输出（带工具调用）
{'model': {'messages': [AIMessage(tool_calls=[...])]}}

# 工具节点输出
{'tools': {'messages': [ToolMessage(content="晴天")]}}

# LLM 最终响应
{'model': {'messages': [AIMessage(content="旧金山今天晴天")]}}
```

### 自动流式传输（`invoke()`）

LangChain 的一个强大特性是**自动流式传输**：即使在节点内使用 `invoke()`，如果整个应用程序处于流式模式，LangChain 也会自动切换到流式传输。

```python
from langgraph.graph import StateGraph, START

def my_node(state):
    # 使用 invoke()，但在流式上下文中会自动流式传输
    response = model.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(MessagesState).add_node("model", my_node).compile()

# graph.stream() 会自动触发 model 的流式传输
for chunk in graph.stream(
    {"messages": [{"role": "user", "content": "你好"}]},
    stream_mode="messages"  # 流式传输 LLM tokens
):
    print(chunk)
```

**工作原理：**
- 当检测到整体应用程序在流式模式时，`invoke()` 自动切换到内部流式模式
- LangChain 触发 `on_llm_new_token` 回调事件
- LangGraph 的 `stream()` / `astream()` 或 LangChain 的 `stream_events()` / `astream_events()` 将这些输出暴露给调用方

---

## LangGraph 中的 Streaming

LangGraph 提供更强大的流式传输功能，适用于复杂的 Agent 工作流。

### Stream 模式详解

#### 1. Values Mode - 完整状态

每个超步骤后流式传输完整的图状态。

```python
from langgraph.graph import StateGraph, START
from typing import TypedDict

class State(TypedDict):
    topic: str
    joke: str

def generate_joke(state: State):
    return {"joke": f"关于 {state['topic']} 的笑话"}

graph = StateGraph(State).add_node("generate", generate_joke).compile()

for chunk in graph.stream(
    {"topic": "冰淇淋"},
    stream_mode="values"  # 完整状态
):
    print(chunk)
```

**输出：**
```python
{'topic': '冰淇淋', 'joke': ''}  # 初始状态
{'topic': '冰淇淋', 'joke': '关于 冰淇淋 的笑话'}  # 执行后
```

#### 2. Updates Mode - 增量更新

只流式传输状态的更新部分。

```python
for chunk in graph.stream(
    {"topic": "冰淇淋"},
    stream_mode="updates"  # 只有更新
):
    print(chunk)
```

**输出：**
```python
{'generate': {'joke': '关于 冰淇淋 的笑话'}}  # 只显示更新
```

### 流式传输图状态

```python
from typing import Annotated
from langgraph.graph import StateGraph, START, END

class GraphState(TypedDict):
    count: Annotated[int, lambda x, y: x + y]  # reducer
    data: str

def node_a(state: GraphState):
    return {"count": 1, "data": "A"}

def node_b(state: GraphState):
    return {"count": 1, "data": "B"}

graph = (
    StateGraph(GraphState)
    .add_node("a", node_a)
    .add_node("b", node_b)
    .add_edge(START, "a")
    .add_edge("a", "b")
    .compile()
)

# 流式传输状态更新
for chunk in graph.stream(
    {"count": 0, "data": ""},
    stream_mode="updates"
):
    print(chunk)
```

**输出：**
```python
{'a': {'count': 1, 'data': 'A'}}
{'b': {'count': 1, 'data': 'B'}}
```

### 流式传输 LLM Tokens

使用 `stream_mode="messages"` 实时流式传输 LLM 生成的 tokens。

```python
from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START

@dataclass
class MyState:
    topic: str
    joke: str = ""

model = init_chat_model(model="glm-4.5-air-mini")

def call_model(state: MyState):
    """调用 LLM 生成笑话"""
    # 即使使用 invoke，在 messages 模式下也会流式传输
    response = model.invoke([
        {"role": "user", "content": f"生成一个关于 {state.topic} 的笑话"}
    ])
    return {"joke": response.content}

graph = (
    StateGraph(MyState)
    .add_node("call_model", call_model)
    .add_edge(START, "call_model")
    .compile()
)

# 流式传输 LLM tokens
for message_chunk, metadata in graph.stream(
    {"topic": "冰淇淋"},
    stream_mode="messages"  # Token 流式传输
):
    if message_chunk.content:
        print(message_chunk.content, end="|", flush=True)
```

**输出：**
```
为|什么|冰淇淋|从不|感到|孤独|？|因为|它|总是|和|它的|朋友|在|一起|！|
```

**元数据包含：**
```python
{
    'langgraph_node': 'call_model',  # 节点名称
    'langgraph_triggers': [...],     # 触发器
    'langgraph_path': [...],         # 执行路径
}
```

#### 过滤特定节点的 Tokens

```python
from langgraph.graph import StateGraph, START

def write_joke(state):
    response = model.invoke([
        {"role": "user", "content": f"写一个关于 {state['topic']} 的笑话"}
    ])
    return {"joke": response.content}

def write_poem(state):
    response = model.invoke([
        {"role": "user", "content": f"写一首关于 {state['topic']} 的诗"}
    ])
    return {"poem": response.content}

graph = (
    StateGraph(State)
    .add_node("write_joke", write_joke)
    .add_node("write_poem", write_poem)
    .add_edge(START, "write_joke")
    .add_edge(START, "write_poem")
    .compile()
)

# 只流式传输 write_poem 节点的输出
for msg, metadata in graph.stream(
    {"topic": "猫"},
    stream_mode="messages"
):
    # 按节点过滤
    if msg.content and metadata.get("langgraph_node") == "write_poem":
        print(msg.content, end="|", flush=True)
```

### 流式传输自定义数据

从工具或节点内部发送自定义进度更新。

#### 使用 get_stream_writer

```python
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START

class State(TypedDict):
    query: str
    answer: str

def node(state: State):
    # 获取流写入器
    writer = get_stream_writer()
    
    # 发送自定义进度更新
    writer({"progress": "开始处理查询..."})
    
    # 模拟处理
    import time
    time.sleep(1)
    
    writer({"progress": "正在生成答案..."})
    time.sleep(1)
    
    writer({"progress": "完成！"})
    
    return {"answer": "答案内容"}

graph = (
    StateGraph(State)
    .add_node("process", node)
    .add_edge(START, "process")
    .compile()
)

# stream_mode="custom" 接收自定义数据
for chunk in graph.stream(
    {"query": "示例查询"},
    stream_mode="custom"
):
    print(chunk)
```

**输出：**
```python
{'progress': '开始处理查询...'}
{'progress': '正在生成答案...'}
{'progress': '完成！'}
```

#### 在工具中使用流式传输

```python
from langchain.tools import tool
from langgraph.config import get_stream_writer

@tool
def query_database(query: str) -> str:
    """查询数据库"""
    writer = get_stream_writer()
    
    # 发送进度更新
    writer({"data": "已检索 0/100 条记录", "type": "progress"})
    
    # 模拟查询
    import time
    for i in range(0, 101, 20):
        time.sleep(0.5)
        writer({"data": f"已检索 {i}/100 条记录", "type": "progress"})
    
    return "查询结果"

# 在 agent 中使用
agent = create_agent(model="glm-4.5-air", tools=[query_database])

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "查询数据库"}]},
    stream_mode="custom"
):
    print(chunk)
```

**输出：**
```python
{'data': '已检索 0/100 条记录', 'type': 'progress'}
{'data': '已检索 20/100 条记录', 'type': 'progress'}
{'data': '已检索 40/100 条记录', 'type': 'progress'}
...
```

#### Python < 3.11 异步注意事项

在 Python < 3.11 的异步代码中，`get_stream_writer` 不可用，需要直接使用 `StreamWriter`：

```python
from langgraph.types import StreamWriter

async def async_node(state: State, writer: StreamWriter):
    writer({"status": "开始"})
    # ... 异步处理
    writer({"status": "完成"})
    return {"result": "data"}
```

### 流式传输子图输出

当使用嵌套子图时，可以流式传输来自父图和子图的输出。

```python
from langgraph.graph import START, StateGraph
from typing import TypedDict

# 定义子图
class SubgraphState(TypedDict):
    foo: str  # 与父图共享的键
    bar: str

def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}

def subgraph_node_2(state: SubgraphState):
    return {"foo": state["foo"] + state["bar"]}

subgraph = (
    StateGraph(SubgraphState)
    .add_node("subgraph_node_1", subgraph_node_1)
    .add_node("subgraph_node_2", subgraph_node_2)
    .add_edge(START, "subgraph_node_1")
    .add_edge("subgraph_node_1", "subgraph_node_2")
    .compile()
)

# 定义父图
class ParentState(TypedDict):
    foo: str

def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}

graph = (
    StateGraph(ParentState)
    .add_node("node_1", node_1)
    .add_node("node_2", subgraph)  # 子图作为节点
    .add_edge(START, "node_1")
    .add_edge("node_1", "node_2")
    .compile()
)

# 流式传输时包含子图输出
for chunk in graph.stream(
    {"foo": "foo"},
    stream_mode="updates",
    subgraphs=True  # 启用子图流式传输
):
    print(chunk)
```

**输出：**
```python
((), {'node_1': {'foo': 'hi! foo'}})  # 父图
(('node_2:UUID',), {'subgraph_node_1': {'bar': 'bar'}})  # 子图节点 1
(('node_2:UUID',), {'subgraph_node_2': {'foo': 'hi! foobar'}})  # 子图节点 2
((), {'node_2': {'foo': 'hi! foobar'}})  # 父图
```

**命名空间说明：**
- `()` - 父图
- `('node_2:UUID',)` - 子图路径
- 可以通过命名空间识别来自哪个图/子图

### 调试模式

`debug` 模式流式传输尽可能多的执行信息，包括完整状态和所有中间步骤。

```python
for chunk in graph.stream(
    {"topic": "冰淇淋"},
    stream_mode="debug"
):
    print(chunk)
```

**输出包含：**
- 节点名称
- 完整状态
- 执行元数据
- 时间戳
- 所有中间结果

---

## v1.x 官方补充：常见 Streaming 模式

当前官方 Streaming 文档把 Agent 流式输出归纳为几个常用维度：Agent 进度、LLM tokens、Reasoning tokens、Tool calls、HITL 暂停/恢复、子 Agent。下面对照官方文档对原文做的补充。

### 1. Agent progress（Agent 进度）

`stream_mode="updates"` 是最直观的进度流，每完成一个节点会推送一次状态更新。这适合在 UI 中展示 Agent 当前所处步骤。

```python
from langchain.agents import create_agent

agent = create_agent("anthropic:claude-sonnet-4-6", tools=[get_weather])

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "旧金山天气如何？"}]},
    stream_mode="updates",
):
    print(chunk)
```

### 2. LLM tokens（模型 token 流）

`stream_mode="messages"` 会返回 `(AIMessageChunk, metadata)` 元组，适合实时显示模型输出的文字。

```python
for chunk, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "讲个长故事"}]},
    stream_mode="messages",
):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

可以根据 `metadata["langgraph_node"]` 过滤特定节点的输出，比如只显示主模型节点而忽略子 Agent。

### 3. Streaming thinking / reasoning tokens（推理内容）

部分模型支持显式的 reasoning 输出。当模型返回 reasoning 内容块时，流式输出中也会逐步出现 `reasoning` 字段，应用可以单独显示“思考过程”和“最终回答”。

```python
for chunk, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "证明素数有无穷多个"}]},
    stream_mode="messages",
):
    for block in chunk.content_blocks or []:
        if block.get("type") == "reasoning":
            print("[思考]", block.get("reasoning"))
        elif block.get("type") == "text":
            print(block.get("text"), end="", flush=True)
```

### 4. Streaming tool calls（工具调用流）

模型生成工具调用时，参数 JSON 会以 `tool_call_chunk` 的形式逐步到达。可以在客户端实时构建工具调用进度，比如显示“正在准备调用 search…”。

```python
gathered = None
for chunk, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "搜索 LangChain 介绍"}]},
    stream_mode="messages",
):
    gathered = chunk if gathered is None else gathered + chunk
    if gathered.tool_calls:
        print(gathered.tool_calls[-1])
```

#### 访问已完成的消息

如果应用更关心一条完整的 `AIMessage` 或完整的 `ToolMessage`，可以监听 `stream_mode="updates"` 中模型节点和工具节点的最终输出，而不是逐 token 拼接。

### 5. Streaming with human-in-the-loop（人机协作）

当 Agent 中配置了 `HumanInTheLoopMiddleware` 时，工具调用前可能会触发 interrupt。流式输出会在中断点暂停；应用收到 interrupt 后展示给用户审批，再通过 `Command(resume=...)` 恢复执行，这时流式输出从中断点继续。

```python
from langgraph.types import Command

state = agent.get_state(config)
if state.tasks and state.tasks[0].interrupts:
    decision = ask_human(state.tasks[0].interrupts)
    for chunk in agent.stream(
        Command(resume=decision),
        config=config,
        stream_mode="updates",
    ):
        print(chunk)
```

### 6. Streaming from sub-agents（子 Agent 流式输出）

当主 Agent 委派任务给子 Agent 时，开启 `subgraphs=True` 可以接收子 Agent 内部节点的流式输出，便于展示子任务进度。

```python
for namespace, chunk in agent.stream(
    {"messages": [{"role": "user", "content": "并行研究三个主题"}]},
    stream_mode="updates",
    subgraphs=True,
):
    if namespace == ():
        print("[主 Agent]", chunk)
    else:
        print(f"[子 Agent {namespace}]", chunk)
```

如果不希望子 Agent 的中间步骤暴露给用户，也可以只保留 `namespace == ()` 的事件。

---

## Event Streaming（stream_events）

官方 Event Streaming 文档现在推荐在大多数应用和前端场景中使用 `stream_events(..., version="v3")`。它返回一个 run stream 对象，并提供 typed projections，让应用可以分别消费 messages、tool calls、state、subagents 和自定义扩展，而不是手动解析底层事件或 `stream_mode` tuple。

### 1. 基本用法

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
)

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    version="v3",
)

for message in stream.messages:
    for delta in message.text:
        print(delta, end="", flush=True)

final_state = stream.output
```

这个 API 更适合直接接 UI：

```text
stream.messages     -> 聊天正文 / reasoning
stream.tool_calls   -> 工具调用面板
stream.values       -> 状态快照
stream.subagents    -> 子 Agent 卡片
stream.output       -> 最终状态
stream.extensions   -> 自定义投影
```

### 2. What you can stream（可观测内容）

| Projection | 含义 |
|------------|------|
| `for event in stream` | 原始协议事件，包含完整 envelope，可访问所有 channel |
| `stream.messages` | 每次 LLM 调用对应的消息流 |
| `message.text` | 文本 delta 和最终文本 |
| `message.reasoning` | 支持 reasoning 的模型输出的 reasoning delta |
| `message.tool_calls` | 模型生成工具调用时的参数 chunk 和最终 tool calls |
| `message.output` | 该次模型调用完成后的最终 message 对象 |
| `stream.tool_calls` | 工具执行生命周期，包括 input、output delta、最终 output、error |
| `stream.values` | Agent state snapshots |
| `stream.output` | 最终 Agent state |
| `stream.subagents` | 命名子 Agent 的 nested run |
| `stream.subgraphs` | nested graph runs，包括 sub-agents 和普通 subgraphs |
| `stream.extensions` | 自定义 transformer projection |

`stream.messages` 产出的是 `ChatModelStream` 对象。每个 message stream 暴露 `.text`、`.reasoning`、`.tool_calls`、`.output`。同步 projection 既可以实时迭代 delta，也可以在结束后读取最终值，例如 `str(message.text)` 或 `message.tool_calls.get()`。

### 3. Agent messages

`stream.messages` 用于读取每次模型调用的输出。相比 `stream_mode="messages"` 返回 `(token, metadata)`，这里把单次模型调用封装成 message stream，更适合按 message 维度处理。

```python
stream = agent.stream_events(input, version="v3")

for message in stream.messages:
    print(f"[{message.node}] ", end="")
    for delta in message.text:
        print(delta, end="", flush=True)

    full_message = message.output
    usage = full_message.usage_metadata
    if usage:
        print(usage)
```

`message.output` 是最终的 `AIMessage`，包含 provider-specific content blocks。Python 中 token usage 通常从 `message.output.usage_metadata` 读取。

### 4. Reasoning content

如果模型支持 reasoning，`message.reasoning` 会像 `message.text` 一样逐步输出 reasoning delta。

```python
stream = agent.stream_events(input, version="v3")

for message in stream.messages:
    for delta in message.reasoning:
        print(f"[thinking] {delta}", end="", flush=True)

    for delta in message.text:
        print(delta, end="", flush=True)
```

注意：

| 注意点 | 说明 |
|--------|------|
| reasoning 需要模型支持 | 不同 provider 的配置方式不同 |
| LangChain 会做标准化 | Anthropic thinking、OpenAI reasoning summaries 等会被标准化为 reasoning content blocks |
| UI 应区分 reasoning 与 final text | reasoning 更适合放到可折叠区域或开发者视图 |

### 5. Tool calls

Event Streaming 有两类工具调用 projection：

| Projection | 关注点 |
|------------|--------|
| `message.tool_calls` | 模型正在生成工具调用时的参数 chunk，以及 finalized tool calls |
| `stream.tool_calls` | 工具真正开始执行后的生命周期、输入、输出 delta、最终结果、错误 |

示例：

```python
stream = agent.stream_events(input, version="v3")

for message in stream.messages:
    for chunk in message.tool_calls:
        print(f"tool call chunk: {chunk}")

    finalized = message.tool_calls.get()
    if finalized:
        print(f"finalized tool calls: {finalized}")

for call in stream.tool_calls:
    print(f"{call.tool_name}({call.input})")
    for delta in call.output_deltas:
        print(delta, end="", flush=True)
    print(call.output, call.error)
```

可以这样理解：

```text
message.tool_calls = 模型“正在写出我要调用什么工具、参数是什么”
stream.tool_calls  = 工具“已经开始执行，执行结果是什么”
```

### 6. Streaming sub-agents

当一个 `create_agent` 调用另一个命名的 `create_agent` 时，内部 agent 的事件会出现在 nested namespace 中。只要创建 agent 时传入 `name=`，就可以通过 `stream.subagents` 获取专门的子 Agent projection。

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

weather_agent = create_agent(
    model=init_chat_model("openai:gpt-5.5"),
    tools=[get_weather],
    name="weather_agent",
)

def call_weather(query: str) -> str:
    """Query the weather agent."""
    result = weather_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].text

supervisor = create_agent(
    model=init_chat_model("openai:gpt-5.5"),
    tools=[call_weather],
    name="supervisor",
)

stream = supervisor.stream_events(
    {"messages": [{"role": "user", "content": "What's the weather in Boston?"}]},
    version="v3",
)

for subagent in stream.subagents:
    print(f"{subagent.name}: ", end="")
    for message in subagent.messages:
        for token in message.text:
            print(token, end="", flush=True)
    print()
```

`stream.subagents` 与 `stream.subgraphs` 的区别：

| Projection | 范围 | 适合场景 |
|------------|------|----------|
| `stream.subagents` | 命名的 `create_agent` 子 Agent | 用户可见 UI，按 agent name 展示 |
| `stream.subgraphs` | 所有 nested graph runs，包括普通 StateGraph subgraphs | 调试 graph 结构或处理非 agent 子图 |

如果使用 `agent.stream(..., subgraphs=True, version="v2")`，则需要通过 `chunk["ns"]` 或 metadata 中的 `lc_agent_name` 区分来源；`stream_events().subagents` 则直接给出更聚焦的子 Agent 视图。

### 7. State and final output

`stream.values` 用于读取运行中的 state snapshots，`stream.output` 用于读取最终 agent state。

```python
stream = agent.stream_events(input, version="v3")

for snapshot in stream.values:
    print(snapshot)

final_state = stream.output
```

适合用途：

| 数据 | 用途 |
|------|------|
| `stream.values` | 实时状态面板、调试、保存中间状态 |
| `stream.output` | 页面最终结果、服务端返回值、持久化最终状态 |

### 8. Multiple projections

多个 projection 可以并发消费。

异步代码使用 `astream_events()` 和 `asyncio.gather()`：

```python
import asyncio

stream = await agent.astream_events(input, version="v3")

async def consume_messages():
    async for message in stream.messages:
        print(await message.text)

async def consume_tool_calls():
    async for call in stream.tool_calls:
        print(call.tool_name, call.input)

await asyncio.gather(consume_messages(), consume_tool_calls())
```

同步代码使用 `stream.interleave(...)`：

```python
stream = agent.stream_events(input, version="v3")

for name, item in stream.interleave("messages", "tool_calls", "values"):
    if name == "messages":
        print(item.text)
    elif name == "tool_calls":
        print(item.tool_name, item.input)
    elif name == "values":
        print(item)
```

如果 projection 没有覆盖你需要的 channel，或者要查看完整事件 envelope，可以直接遍历 raw protocol events：

```python
for event in stream:
    print(event["method"], event["params"]["namespace"], event["params"]["data"])
```

### 9. Projection 的惰性消费

Event Streaming 的 projection 是按需消费的。你访问哪个 projection，就读取哪个 projection 的事件；不需要为了展示工具调用去遍历 messages，也不需要为了拿最终状态去解析所有 raw events。

```python
stream = agent.stream_events(input, version="v3")

# 只消费工具执行生命周期
for call in stream.tool_calls:
    print(call.tool_name, call.output)

# 最后读取最终状态
final_state = stream.output
```

这只表示客户端消费层面的“按需读取”，不表示 agent 运行时没有发生其他步骤。模型调用、工具执行、子 Agent 执行仍会按任务正常发生。

### 10. Custom updates 和 Transformers

当应用需要内置 projection 之外的结构，例如 retrieval progress、artifact 生成、业务事件、审计事件时，可以注册 custom stream transformers。

```python
stream = agent.stream_events(
    input,
    version="v3",
    transformers=[ToolActivityTransformer],
)

for activity in stream.extensions["tool_activity"]:
    print(activity)
```

中间件也可以声明 transformer factories。官方文档中提到，middleware-registered transformers 需要 `langchain>=1.3.2`。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware

class ToolActivityMiddleware(AgentMiddleware):
    transformers = (ToolActivityTransformer,)

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
    middleware=[ToolActivityMiddleware()],
)
```

编译时 transformer 顺序：

1. 内置 `ToolCallTransformer`。
2. middleware 注册的 transformers，按 middleware 顺序。
3. `create_agent(..., transformers=...)` 调用方传入的 transformers。

这个机制也用于输出安全处理。例如 `PIIMiddleware(apply_to_output=True)` 可以在流式输出离开 run 前对 text deltas、tool-call args、tool outputs、state snapshots 做脱敏，避免 live readers 看到未经处理的 PII。

### 11. Raw `astream_events` 作为底层补充

旧示例里常见的 raw `astream_events` 仍然有价值，尤其适合调试或兼容历史代码。它直接暴露 `on_chat_model_stream`、`on_tool_start`、`on_tool_end` 等事件：

```python
async for event in agent.astream_events({"messages": [...]}):
    if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"]
        print(chunk.text, end="", flush=True)
    elif event["event"] == "on_tool_start":
        print(f"调用工具：{event['name']} args={event['data'].get('input')}")
    elif event["event"] == "on_tool_end":
        print(f"工具结果：{event['data'].get('output')}")
```

建议把 raw events 理解为底层协议层，把 `stream_events(..., version="v3")` 的 typed projections 理解为应用层消费接口。新 UI 优先使用 projections；需要自定义协议处理、调试完整 envelope 或兼容旧代码时再直接遍历 raw events。

---

## 高级特性

### 多模式组合

可以同时使用多个流式模式：

```python
from langchain.agents import create_agent
from langgraph.config import get_stream_writer

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    writer = get_stream_writer()
    writer(f"正在查找 {city} 的数据")
    writer(f"已获取 {city} 的数据")
    return f"{city} 总是晴天！"

agent = create_agent(
    model="glm-4.5-air-mini",
    tools=[get_weather],
)

# 同时使用 updates、messages 和 custom 模式
for stream_mode, chunk in agent.stream(
    {"messages": [{"role": "user", "content": "旧金山天气如何？"}]},
    stream_mode=["updates", "messages", "custom"]
):
    print(f"{stream_mode}: {chunk}\n")
```

**输出：**
```
updates: {'model': {'messages': [AIMessage(tool_calls=[...])]}}

custom: 正在查找 San Francisco 的数据

custom: 已获取 San Francisco 的数据

updates: {'tools': {'messages': [ToolMessage(...)]}}

messages: (AIMessageChunk(content='旧金山'), {...})

updates: {'model': {'messages': [AIMessage(content='...')]}}
```

### v2 streaming format 与禁用 Streaming

当前 LangChain 文档中需要区分两个版本参数语境：

| 语境 | 推荐版本 | 说明 |
|------|----------|------|
| `stream_events()` / `astream_events()` typed projections | `version="v3"` | 返回 run stream 对象，通过 `stream.messages`、`stream.tool_calls`、`stream.values` 等 projection 消费 |
| `stream()` / `astream()` stream modes | `version="v2"` | 返回统一 `StreamPart` dict，形如 `{"type": ..., "ns": ..., "data": ...}` |
| raw `astream_events` 历史示例 | 可能出现 v1/v2 | 常见 `on_chat_model_stream`、`on_tool_start` 等底层事件命名 |

因此，新应用通常可以这样选择：

```text
应用/前端语义化消费 -> stream_events(..., version="v3")
LangGraph stream_mode chunk 消费 -> stream(..., version="v2")
底层事件调试或旧代码兼容 -> raw astream_events events
```

`stream()` / `astream()` 的 v2 chunk 统一格式：

```python
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode=["updates", "custom"],
    version="v2",
):
    print(chunk["type"])  # "updates" or "custom"
    print(chunk["ns"])    # namespace, subgraphs=True 时用于区分子图
    print(chunk["data"])  # payload
```

这比旧式 tuple unpacking 更适合多模式、多子图场景。

### 禁用 Streaming

在某些场景（如多 Agent 系统），可能需要禁用特定模型的流式传输。

#### Python

```python
from langchain_openai import ChatOpenAI

# 禁用流式传输
model = ChatOpenAI(
    model="o1-preview",
    disable_streaming=True  # 显式禁用
)
```

或使用 `init_chat_model`：

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "anthropic:claude-sonnet-4-6",
    disable_streaming=True,
)
```

#### JavaScript

```javascript
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({
  model: "o1-preview",
  streaming: false  // 禁用流式传输
});
```

**使用场景：**
- 某些模型不支持流式传输（如 OpenAI o1 系列）
- 多 Agent 系统中控制哪些 Agent 流式传输
- 需要等待完整响应再处理

### 过滤特定节点

```python
# 只显示特定节点的流式输出
for msg, metadata in graph.stream(
    inputs,
    stream_mode="messages"
):
    # 过滤指定节点
    if msg.content and metadata.get("langgraph_node") == "target_node":
        print(msg.content, end="", flush=True)
```

---

## 消息块处理

### AIMessageChunk

在流式传输过程中，你会收到 `AIMessageChunk` 对象，可以累积成完整消息。

#### Python

```python
from langchain_core.messages import AIMessageChunk

chunks = []
full_message = None

for chunk in model.stream("你好"):
    chunks.append(chunk)
    print(chunk.text)
    
    # 累积完整消息
    full_message = chunk if full_message is None else full_message + chunk

# full_message 现在是完整的 AIMessage
print(full_message.content_blocks)
```

#### JavaScript

```javascript
import { AIMessageChunk } from "langchain";

let finalChunk = undefined;

for (const chunk of chunks) {
  finalChunk = finalChunk ? finalChunk.concat(chunk) : chunk;
}

console.log(finalChunk.contentBlocks);
```

### 流式传输工具调用

```python
# 工具调用逐步构建
full = None
for chunk in model_with_tools.stream("波士顿天气？"):
    full = chunk if full is None else full + chunk
    print(full.content_blocks)
```

**输出示例：**
```python
[{"type": "tool_call_chunk", "name": "get_weather", "args": ""}]
[{"type": "tool_call_chunk", "name": "get_weather", "args": "{\"city"}]
[{"type": "tool_call_chunk", "name": "get_weather", "args": "\": \"Boston\"}"}]
# ... 最终变为完整的 tool_call
```

---

## 实际应用场景

### 1. 聊天应用

```python
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

model = init_chat_model("glm-4.5-air")
agent = create_agent(model=model, tools=[])

def chat_stream(user_message: str):
    """流式聊天响应"""
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        stream_mode="messages"
    ):
        msg, metadata = chunk
        if msg.content:
            yield msg.content

# 使用
for text in chat_stream("介绍一下 LangChain"):
    print(text, end="", flush=True)
```

### 2. 进度跟踪的文档处理

```python
@tool
def process_document(file_path: str) -> str:
    """处理文档"""
    writer = get_stream_writer()
    
    writer({"stage": "reading", "progress": 0})
    # 读取文档
    
    writer({"stage": "analyzing", "progress": 30})
    # 分析内容
    
    writer({"stage": "summarizing", "progress": 70})
    # 生成摘要
    
    writer({"stage": "complete", "progress": 100})
    return "处理完成"

agent = create_agent(model="glm-4.5-air", tools=[process_document])

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "处理 report.pdf"}]},
    stream_mode="custom"
):
    # 更新 UI 进度条
    update_progress_bar(chunk["progress"])
```

### 3. 多步骤 Agent 工作流

```python
from langgraph.graph import StateGraph, START, END

class WorkflowState(TypedDict):
    query: str
    search_results: list
    analysis: str
    summary: str

def search(state: WorkflowState):
    writer = get_stream_writer()
    writer({"step": "搜索中..."})
    # 执行搜索
    return {"search_results": [...]}

def analyze(state: WorkflowState):
    writer = get_stream_writer()
    writer({"step": "分析结果..."})
    # 分析
    return {"analysis": "..."}

def summarize(state: WorkflowState):
    writer = get_stream_writer()
    writer({"step": "生成摘要..."})
    # 总结
    return {"summary": "..."}

graph = (
    StateGraph(WorkflowState)
    .add_node("search", search)
    .add_node("analyze", analyze)
    .add_node("summarize", summarize)
    .add_edge(START, "search")
    .add_edge("search", "analyze")
    .add_edge("analyze", "summarize")
    .compile()
)

# 同时流式传输步骤和自定义进度
for mode, chunk in graph.stream(
    {"query": "LangChain 是什么？"},
    stream_mode=["updates", "custom"]
):
    if mode == "custom":
        print(f"进度: {chunk['step']}")
    elif mode == "updates":
        print(f"完成节点: {list(chunk.keys())[0]}")
```

### 4. 实时数据处理

```python
@tool
def fetch_realtime_data(source: str) -> str:
    """获取实时数据"""
    writer = get_stream_writer()
    
    for i in range(10):
        # 模拟实时数据流
        data = fetch_batch(i)
        writer({
            "batch": i,
            "data": data,
            "timestamp": datetime.now()
        })
        time.sleep(0.5)
    
    return "数据获取完成"

# 实时显示数据
for chunk in agent.stream(..., stream_mode="custom"):
    plot_data(chunk["data"])  # 实时绘图
```

---

## 最佳实践（重点）

### 1. **选择合适的流式模式**

```python
# 新应用 / 前端 - 优先使用 Event Streaming projections
stream = agent.stream_events(..., version="v3")
for message in stream.messages:
    for token in message.text:
        display_to_user(token)

# 调试 - 使用 debug 模式
for chunk in graph.stream(..., stream_mode="debug", version="v2"):
    log_to_console(chunk["data"])

# 进度跟踪 - 使用 custom 模式
for chunk in agent.stream(..., stream_mode="custom", version="v2"):
    update_progress_bar(chunk["data"])

# 状态监控 - 使用 updates 模式
for chunk in graph.stream(..., stream_mode="updates", version="v2"):
    update_state_display(chunk["data"])
```

### 2. **正确处理消息块累积**

```python
# ✅ 正确：累积块
full = None
for chunk in model.stream("长文本"):
    full = chunk if full is None else full + chunk
    # 可以安全地使用 full

# ❌ 错误：不累积，丢失上下文
for chunk in model.stream("长文本"):
    print(chunk.text)  # 只显示部分内容
```

### 3. **使用 flush 立即显示**

```python
# ✅ 正确：立即刷新输出
for chunk in model.stream("你好"):
    print(chunk.text, end="", flush=True)

# ❌ 错误：可能被缓冲
for chunk in model.stream("你好"):
    print(chunk.text, end="")  # 没有 flush
```

### 4. **错误处理**

```python
from langgraph.errors import GraphRecursionError

try:
    for chunk in graph.stream(
        inputs,
        stream_mode="updates",
        config={"recursion_limit": 10}
    ):
        process_chunk(chunk)
except GraphRecursionError:
    print("达到递归限制，但获得了部分结果")
except Exception as e:
    print(f"流式传输错误: {e}")
```

### 5. **组合多种模式**

```python
# 同时获取状态更新和自定义进度
for mode, chunk in agent.stream(
    inputs,
    stream_mode=["updates", "custom", "messages"]
):
    if mode == "updates":
        log_state_change(chunk)
    elif mode == "custom":
        update_ui_progress(chunk)
    elif mode == "messages":
        display_llm_output(chunk[0].content)
```

### 6. **避免阻塞**

```python
import asyncio

# ✅ 异步流式传输（非阻塞）
async def async_stream():
    async for chunk in agent.astream(inputs):
        await process_chunk(chunk)

# ❌ 同步流式传输（阻塞）
def sync_stream():
    for chunk in agent.stream(inputs):
        process_chunk(chunk)  # 阻塞主线程
```

### 7. **内存管理**

```python
# 对于长时间运行的流，定期清理
chunks_buffer = []
MAX_BUFFER_SIZE = 100

for chunk in model.stream("超长文本"):
    chunks_buffer.append(chunk)
    
    # 定期处理并清空缓冲区
    if len(chunks_buffer) >= MAX_BUFFER_SIZE:
        process_chunks(chunks_buffer)
        chunks_buffer = []
```

### 8. **子图流式传输**

```python
# ✅ 启用子图流式传输以获得完整可见性
for chunk in graph.stream(
    inputs,
    stream_mode="updates",
    subgraphs=True  # 包含子图输出
):
    process_chunk(chunk)

# ❌ 不启用子图流式传输可能丢失信息
for chunk in graph.stream(inputs, stream_mode="updates"):
    # 只能看到父图输出
    process_chunk(chunk)
```

---

## 性能优化

### 1. **使用流式传输减少感知延迟**

```python
import time

# 非流式：用户等待完整响应
start = time.time()
response = model.invoke("写一篇长文章")
print(response.content)  # 15 秒后显示
print(f"总时间: {time.time() - start}s")

# 流式：立即开始显示
start = time.time()
first_token_time = None
for chunk in model.stream("写一篇长文章"):
    if first_token_time is None:
        first_token_time = time.time()
        print(f"首个 token 时间: {first_token_time - start}s")  # ~0.5s
    print(chunk.text, end="", flush=True)
print(f"\n总时间: {time.time() - start}s")  # 总时间相同，但体验更好
```

### 2. **批量处理**

```python
# ✅ 批量处理流式块
batch = []
BATCH_SIZE = 10

for chunk in model.stream("长文本"):
    batch.append(chunk)
    if len(batch) >= BATCH_SIZE:
        process_batch(batch)
        batch = []

# 处理剩余
if batch:
    process_batch(batch)
```

### 3. **异步并发**

```python
import asyncio

async def stream_multiple_agents():
    """并发运行多个 agent 流"""
    tasks = [
        agent1.astream(input1, stream_mode="messages"),
        agent2.astream(input2, stream_mode="messages"),
        agent3.astream(input3, stream_mode="messages"),
    ]
    
    # 并发处理所有流
    results = await asyncio.gather(*tasks)
    return results
```

### 4. **选择性流式传输**

```python
# 只流式传输需要的节点
for msg, metadata in graph.stream(
    inputs,
    stream_mode="messages"
):
    # 只处理特定节点
    if metadata.get("langgraph_node") in ["important_node_1", "important_node_2"]:
        process_message(msg)
    # 忽略其他节点，减少处理开销
```

### 5. **缓存和复用**

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_stream_result(query: str):
    """缓存流式结果"""
    result = []
    for chunk in model.stream(query):
        result.append(chunk)
    return result

# 复用缓存的流
cached = get_cached_stream_result("常见问题")
for chunk in cached:
    print(chunk.text, end="")
```

---

## 快速参考

### API 选择表

| 目标 | 推荐 API |
|------|----------|
| 新应用构建实时 Agent UI | `agent.stream_events(..., version="v3")` |
| 分别消费文本、工具调用、状态、子 Agent | `stream.messages`、`stream.tool_calls`、`stream.values`、`stream.subagents` |
| 只看 Agent 步骤进度 | `agent.stream(..., stream_mode="updates", version="v2")` |
| 逐 token 读取所有 LLM 输出 | `stream_events().messages` 或 `stream_mode="messages"` |
| 发送工具/节点自定义进度 | `get_stream_writer()` + `stream_mode="custom"` 或 custom transformer |
| 观察普通 subgraph 内部输出 | `subgraphs=True` + `chunk["ns"]` |
| 观察命名子 Agent | `stream.subagents` 或 metadata `lc_agent_name` |
| 调试底层协议事件 | 直接遍历 raw `astream_events` / `stream_events` events |

### Event Streaming projections

| Projection | 说明 |
|------------|------|
| `stream.messages` | LLM message streams |
| `message.text` | 文本 delta / 最终文本 |
| `message.reasoning` | reasoning delta |
| `message.tool_calls` | 模型生成工具调用时的 chunk / finalized tool calls |
| `message.output` | 完整 AI message |
| `stream.tool_calls` | 工具执行生命周期 |
| `stream.values` | state snapshots |
| `stream.output` | final agent state |
| `stream.subagents` | 命名子 Agent runs |
| `stream.subgraphs` | nested graph runs |
| `stream.extensions` | custom transformer projections |

### Stream 模式对比表

| 模式 | 返回内容 | 使用场景 | 示例输出 |
|------|---------|---------|----------|
| `values` | 完整状态 | 查看所有状态 | `{"topic": "AI", "joke": "..."}` |
| `updates` | 增量更新 | 只关心变化 | `{"node": {"field": "value"}}` |
| `messages` | LLM tokens + 元数据 | 流式聊天 | `(AIMessageChunk("Hi"), {...})` |
| `custom` | 自定义数据 | 进度/日志 | `{"progress": 50, "status": "..."}` |
| `debug` | 详细信息 | 调试 | `{node, state, metadata, ...}` |

### 常用代码片段

#### 基本流式传输

```python
# Model streaming
for chunk in model.stream("query"):
    print(chunk.text, end="", flush=True)

# Agent streaming
for chunk in agent.stream({"messages": [...]}, stream_mode="updates"):
    print(chunk)

# Graph streaming
for chunk in graph.stream(inputs, stream_mode="values"):
    print(chunk)

# Event streaming
stream = agent.stream_events({"messages": [...]}, version="v3")
for message in stream.messages:
    for token in message.text:
        print(token, end="", flush=True)
```

#### 累积消息

```python
full = None
for chunk in model.stream("query"):
    full = chunk if full is None else full + chunk
```

#### 自定义进度

```python
from langgraph.config import get_stream_writer

def my_node(state):
    writer = get_stream_writer()
    writer({"progress": "开始"})
    # ... 处理
    writer({"progress": "完成"})
    return state

# 流式传输
for chunk in graph.stream(inputs, stream_mode="custom"):
    print(chunk["progress"])
```

#### 多模式

```python
for chunk in agent.stream(
    inputs,
    stream_mode=["updates", "messages", "custom"],
    version="v2",
):
    if chunk["type"] == "updates":
        print(f"State: {chunk['data']}")
    elif chunk["type"] == "messages":
        token, metadata = chunk["data"]
        print(f"Token: {token.text}")
    elif chunk["type"] == "custom":
        print(f"Progress: {chunk['data']}")
```

#### 子图流式传输

```python
for chunk in graph.stream(
    inputs,
    stream_mode="updates",
    subgraphs=True,  # 包含子图
    version="v2",
):
    namespace = chunk["ns"]
    data = chunk["data"]
    if namespace == ():
        print("父图:", data)
    else:
        print(f"子图 {namespace}:", data)
```

### 事件类型（astream_events）

```python
async for event in model.astream_events("Hello"):
    if event["event"] == "on_chat_model_start":
        print(f"开始: {event['data']['input']}")
    
    elif event["event"] == "on_chat_model_stream":
        print(f"Token: {event['data']['chunk'].text}")
    
    elif event["event"] == "on_chat_model_end":
        print(f"完整消息: {event['data']['output'].text}")
```

### 禁用流式传输

```python
# Python
model = ChatOpenAI(model="o1-preview", disable_streaming=True)

# JavaScript
const model = new ChatOpenAI({ model: "o1-preview", streaming: false });
```

---

## 总结

LangChain 和 LangGraph 的流式传输系统提供了强大而灵活的实时数据传输能力：

**核心优势：**
✅ 改善用户体验 - 即时反馈和进度可视化  
✅ 多种流式模式 - 适应不同场景需求  
✅ 自动流式传输 - 简化开发复杂度  
✅ 子图支持 - 复杂工作流的完整可见性  
✅ 自定义数据 - 灵活的进度和日志传输  

**关键要点：**
- 新应用优先考虑 `stream_events(..., version="v3")`，用 typed projections 直接消费 messages、tool calls、state、subagents
- 使用 `stream()` 或 `astream()` 时，通过 `stream_mode` 参数选择合适的流式模式
- `stream(..., version="v2")` 返回统一 `{"type", "ns", "data"}` chunk，更适合多模式和子图流
- 使用 `get_stream_writer()` 发送自定义进度更新
- 正确累积 `AIMessageChunk` 以获得完整消息
- 对于复杂工作流，启用 `subgraphs=True`
- 使用多模式组合获得全面的可观察性

通过合理使用流式传输，你可以构建响应迅速、用户体验优秀的 LLM 应用程序！

---

## 相关资源

- 官方 Streaming 文档：<https://docs.langchain.com/oss/python/langchain/streaming>
- 官方 Event Streaming 文档：<https://docs.langchain.com/oss/python/langchain/event-streaming>
- 配套文档：
  - [LangChain Agents 详细总结](01_LangChain_Agents_详细总结.md)
  - [LangChain Models 详细指南](02_LangChain_Models_详细指南.md)
  - [LangChain Tools 详细指南](04_LangChain_Tools_详细指南.md)
  - [LangChain Messages 详细指南](03_LangChain_Messages_详细指南.md)
  - [LangChain ShortTermMemory 详细指南](05_LangChain_ShortTermMemory_详细指南.md)

---

**文档版本**: 1.2  
**最后更新**: 2026-07-03  
**基于**: LangChain v1.x 官方文档
