# LangChain Streaming 详细指南

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
6. [高级特性](#高级特性)
   - [多模式组合](#多模式组合)
   - [禁用 Streaming](#禁用-streaming)
   - [过滤特定节点](#过滤特定节点)
7. [消息块处理](#消息块处理)
8. [实际应用场景](#实际应用场景)
9. [最佳实践](#最佳实践)
10. [性能优化](#性能优化)
11. [快速参考](#快速参考)

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

### Stream Mode（流式模式）

LangChain 支持多种流式模式，每种模式提供不同级别的信息：

| 模式 | 描述 | 用途 |
|------|------|------|
| `values` | 每步后的完整状态 | 查看完整的图状态 |
| `updates` | 每步的状态更新（增量） | 只看变化部分 |
| `messages` | LLM token 流 + 元数据 | 流式显示 LLM 输出 |
| `custom` | 自定义用户数据 | 进度更新、日志等 |
| `debug` | 详细的执行信息 | 调试和故障排除 |

### 流式输出类型

1. **Token-level streaming** - 逐个 token 输出
2. **Step-level streaming** - 每个节点/步骤的输出
3. **Event streaming** - 语义事件（开始、流式、结束）

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
- LangGraph 的 `stream()` 和 `astream_events()` 实时显示输出

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
    "claude-sonnet-4-5-20250929",
    disable_streaming=True
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
# 聊天应用 - 使用 messages 模式
for msg, _ in agent.stream(..., stream_mode="messages"):
    display_to_user(msg.content)

# 调试 - 使用 debug 模式
for chunk in graph.stream(..., stream_mode="debug"):
    log_to_console(chunk)

# 进度跟踪 - 使用 custom 模式
for progress in agent.stream(..., stream_mode="custom"):
    update_progress_bar(progress)

# 状态监控 - 使用 updates 模式
for update in graph.stream(..., stream_mode="updates"):
    update_state_display(update)
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
for mode, chunk in agent.stream(
    inputs,
    stream_mode=["updates", "messages", "custom"]
):
    if mode == "updates":
        print(f"State: {chunk}")
    elif mode == "messages":
        print(f"Token: {chunk[0].content}")
    elif mode == "custom":
        print(f"Progress: {chunk}")
```

#### 子图流式传输

```python
for chunk in graph.stream(
    inputs,
    stream_mode="updates",
    subgraphs=True  # 包含子图
):
    namespace, data = chunk
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
- 使用 `stream()` 或 `astream()` 方法启动流式传输
- 通过 `stream_mode` 参数选择合适的流式模式
- 使用 `get_stream_writer()` 发送自定义进度更新
- 正确累积 `AIMessageChunk` 以获得完整消息
- 对于复杂工作流，启用 `subgraphs=True`
- 使用多模式组合获得全面的可观察性

通过合理使用流式传输，你可以构建响应迅速、用户体验优秀的 LLM 应用程序！
