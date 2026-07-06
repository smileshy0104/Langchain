# LangGraph Event Streaming 与 Streaming 详细指南

> 基于 LangGraph 官方 Event streaming、Streaming 文档整理。本文聚焦两套流式接口的定位差异：`stream_events(..., version="v3")` 提供 typed projections，适合应用层消费；`stream(..., version="v2")` 提供 stream-mode chunks，适合低层 graph runtime 事件、调试和精细控制。

## 目录

1. [整体理解](#整体理解)
2. [两套 Streaming API](#两套-streaming-api)
3. [Event Streaming Quickstart](#event-streaming-quickstart)
4. [Event Streaming 架构](#event-streaming-架构)
5. [Typed Projections](#typed-projections)
6. [Stream Messages](#stream-messages)
7. [Stream Subgraphs](#stream-subgraphs)
8. [Stream State 与 Output](#stream-state-与-output)
9. [Multiple Projections](#multiple-projections)
10. [Interrupt Resume](#interrupt-resume)
11. [Raw Protocol Events](#raw-protocol-events)
12. [Channels 与 Event Lifecycle](#channels-与-event-lifecycle)
13. [Build Your Own Projection](#build-your-own-projection)
14. [StreamTransformer](#streamtransformer)
15. [StreamChannel](#streamchannel)
16. [ToolCallTransformer](#toolcalltransformer)
17. [Streaming Quickstart](#streaming-quickstart)
18. [Stream Output Format v2](#stream-output-format-v2)
19. [Stream Modes](#stream-modes)
20. [Graph State：values 与 updates](#graph-statevalues-与-updates)
21. [LLM Tokens：messages](#llm-tokensmessages)
22. [Custom Data](#custom-data)
23. [Subgraph Outputs](#subgraph-outputs)
24. [Checkpoints、Tasks 与 Debug](#checkpointstasks-与-debug)
25. [Multiple Modes](#multiple-modes)
26. [Advanced Streaming](#advanced-streaming)
27. [Migrate to v2](#migrate-to-v2)
28. [Async with Python < 3.11](#async-with-python--311)
29. [最佳实践](#最佳实践)
30. [故障排查](#故障排查)
31. [快速参考](#快速参考)
32. [资料来源](#资料来源)

---

## 整体理解

LangGraph 有两层流式能力：

```text
Pregel engine
  -> Streaming
       原始 graph execution stream modes
       updates / values / messages / custom / checkpoints / tasks / debug
  -> Event streaming
       在 Streaming 之上规范化事件
       经过 transformers
       暴露 typed projections
       stream.messages / stream.values / stream.subgraphs / stream.output / stream.extensions
```

一句话：

```text
Streaming 是底层运行时事件；
Event streaming 是面向应用代码的 typed projection 层。
```

官方建议：

```text
新应用优先使用 Event streaming；
需要低层 stream_mode、debug、tasks、checkpoints 时使用 Streaming。
```

---

## 两套 Streaming API

| 维度 | Event Streaming | Streaming |
|------|-----------------|-----------|
| API | `stream_events()` / `astream_events()` | `stream()` / `astream()` |
| 推荐版本 | `version="v3"` | `version="v2"` |
| 抽象层级 | 应用层 typed projections | Pregel/graph runtime stream modes |
| 消费方式 | `stream.messages`、`stream.values`、`stream.subgraphs` 等 | `chunk["type"]`、`chunk["ns"]`、`chunk["data"]` |
| 适合 | 前端/UI、应用代码、并发消费多个 projection | runtime 调试、低层事件、checkpoints/tasks/debug |
| 扩展方式 | `StreamTransformer` + `StreamChannel` | `get_stream_writer()`、stream modes |
| 子图 | `stream.subgraphs` | `subgraphs=True` + `chunk["ns"]` |
| 自定义数据 | `stream.extensions` | `stream_mode="custom"` |

选择建议：

| 需求 | 推荐 |
|------|------|
| 新应用展示 LLM token、state、subgraphs | `stream_events(..., version="v3")` |
| 多个消费者独立读取 messages/state/output | Event streaming |
| 需要自己构建 projection | Event streaming transformer |
| 只想看 state updates | `stream(..., stream_mode="updates", version="v2")` |
| 需要 checkpoints / tasks / debug | Streaming |
| 需要低层 `StreamPart` 类型分支 | Streaming v2 |
| 需要 raw protocol envelope | 直接遍历 `stream_events` run object |

---

## Event Streaming Quickstart

基础用法：

```python
stream = graph.stream_events(
    {
        "messages": [
            {"role": "user", "content": "What is 42 * 17?"}
        ],
    },
    version="v3",
)

for message in stream.messages:
    for token in message.text:
        print(token, end="", flush=True)

final_state = stream.output
```

核心特点：

| 特点 | 说明 |
|------|------|
| typed projections | 不需要手动按 chunk type 分支 |
| multiple consumers | `messages`、`values`、`subgraphs` 可独立消费 |
| extensions | 自定义 projection 放在 `stream.extensions` |
| raw events still available | 直接遍历 `stream` 可读协议事件 |

如果 graph 部署在 Agent Server 后面：

```text
使用 LangSmith Streaming API。
```

---

## Event Streaming 架构

官方把 streaming stack 分成两层：

```text
1. Streaming
   Pregel engine 产生原始 graph execution events

2. Event streaming
   规范化事件
   经过 stream transformers
   暴露 typed projections
```

流程：

```text
Pregel engine
  -> Raw Pregel events
       updates / values / messages / custom / checkpoints / tasks / debug
  -> Event router
  -> Stream transformers
       built-in transformers
       custom transformers
  -> Event Stream projections
```

Event router 的职责：

| 职责 | 说明 |
|------|------|
| 接收 normalized Pregel events | 从底层 stream modes 进入 |
| 调用 transformer pipeline | 每个 transformer 观察/处理事件 |
| 生成 built-in projections | `messages`、`values`、`subgraphs`、`output` |
| 挂载 custom projections | 放到 `stream.extensions` |

---

## Typed Projections

`stream_events()` 返回的 run stream 暴露以下 projections：

| Projection | 用途 |
|------------|------|
| `stream` | 直接迭代所有 protocol events |
| `stream.messages` | chat model messages 和 token deltas |
| `stream.values` | state snapshots |
| `stream.output` | final output |
| `stream.subgraphs` | nested graph executions |
| `stream.interrupts` | human-in-the-loop interrupt payloads |
| `stream.interrupted` | 是否因 human input 暂停 |
| `stream.extensions` | custom transformer projections |

重要特性：

```text
多个 projection 可以同时消费；
读取 stream.messages 不会消耗 stream.values / stream.subgraphs / stream.output 所需事件。
```

这就是 projection 的按需/独立消费能力。

---

## Stream Messages

使用 `stream.messages` 读取 chat model 输出。

```python
stream = graph.stream_events(input, version="v3")

for message in stream.messages:
    text = str(message.text)
    usage = message.output.usage_metadata

    print(text)
    print(usage)
```

`message` 常用 projection：

| Projection | 说明 |
|------------|------|
| `message.text` | 文本 token deltas，可迭代；`str()` 得完整文本 |
| `message.reasoning` | reasoning deltas |
| `message.tool_calls` | tool-call argument chunks |
| `message.output` | 完整最终 message |

注意：

```text
如果需要 text、reasoning、tool-call chunks 的精确到达顺序，
应迭代 message stream 的 raw events，
而不是分别消费 message.text / reasoning / tool_calls。
```

---

## Stream Subgraphs

使用 `stream.subgraphs` 观察嵌套 graph work。

```python
stream = graph.stream_events(input, version="v3")

for subgraph in stream.subgraphs:
    print(subgraph.graph_name, subgraph.path)

    for message in subgraph.messages:
        print(message.text)
```

字段：

| 字段 | 说明 |
|------|------|
| `graph_name` | compiled graph 或 agent 的 `name` |
| `path` | 从 root 到该 subgraph 的 path |
| `messages` | subgraph 内 LLM messages |
| `values` | subgraph state snapshots |
| `output` | subgraph final output |

和 Streaming 的区别：

| Event streaming | Streaming |
|-----------------|-----------|
| `stream.subgraphs` 直接给 subgraph handle | `subgraphs=True` 后看 `chunk["ns"]` |
| 不必解析 namespace string | 需要按 namespace 判断来源 |
| 更适合 UI/应用层 | 更适合低层调试 |

---

## Stream State 与 Output

使用 `stream.values` 读取 full state snapshots：

```python
stream = graph.stream_events(input, version="v3")

for snapshot in stream.values:
    print(snapshot)

final_state = stream.output
```

区别：

| Projection | 说明 |
|------------|------|
| `stream.values` | 运行过程中每一步的 state snapshots |
| `stream.output` | graph 最终输出 |

适合：

| 需求 | 使用 |
|------|------|
| 实时状态面板 | `stream.values` |
| 最终结果 | `stream.output` |
| 调试 state 变化 | `stream.values` |

---

## Multiple Projections

Async 并发消费：

```python
import asyncio

stream = await graph.astream_events(input, version="v3")

async def consume_messages():
    async for message in stream.messages:
        print(f"[llm] node={message.node}")

async def consume_subgraphs():
    async for subgraph in stream.subgraphs:
        print(f"[subgraph] path={subgraph.path}")

await asyncio.gather(consume_messages(), consume_subgraphs())
```

Sync 严格到达顺序：

```python
stream = graph.stream_events(input, version="v3")

for name, item in stream.interleave("values", "messages", "subgraphs"):
    if name == "values":
        print(f"[state] keys={list(item)}")
    elif name == "messages":
        print(f"[llm] node={item.node}")
    elif name == "subgraphs":
        print(f"[subgraph] path={item.path}")
```

选择：

| 需求 | 推荐 |
|------|------|
| 各 UI 区域独立更新 | async gather |
| 严格按照到达顺序渲染 | `interleave()` |
| 只关心最终结果 | `stream.output` |

---

## Interrupt Resume

当 graph 因 human input 暂停：

1. 查看 `stream.interrupted`。
2. 读取 `stream.interrupts`。
3. 用 `Command(resume=...)` 再次调用 `stream_events()`。

示例：

```python
from langgraph.types import Command

stream = graph.stream_events(input, version="v3")

for message in stream.messages:
    print(message.text)

if stream.interrupted:
    print(stream.interrupts)

stream = graph.stream_events(
    Command(resume={"decisions": [{"type": "approve"}]}),
    version="v3",
)
final_state = stream.output
```

前提：

```text
graph 必须配置 checkpointer；
config 必须带 thread_id。
```

---

## Raw Protocol Events

直接遍历 run object 可以读取所有 protocol events：

```python
stream = graph.stream_events(
    {
        "messages": [
            {"role": "user", "content": "What is 42 * 17?"}
        ],
    },
    version="v3",
)

for event in stream:
    namespace = event["params"]["namespace"]
    print(namespace, event["method"], event["params"]["data"])
```

ProtocolEvent shape：

```python
class ProtocolEvent(TypedDict):
    seq: int
    method: str
    params: ProtocolEventParams

class ProtocolEventParams(TypedDict):
    namespace: list[str]
    timestamp: int
    data: Any
```

字段说明：

| 字段 | 说明 |
|------|------|
| `seq` | run 内严格递增，用于排序 |
| `method` | channel name，如 `messages`、`values`、`custom` |
| `namespace` | 从 root graph 到当前 scope 的路径 |
| `timestamp` | wall-clock milliseconds，可能漂移，不建议用于排序 |
| `data` | channel-specific payload |

Namespace 示例：

```text
[]                                  root
["researcher:6f4d"]                 nested graph
["researcher:6f4d", "tools:91ac"]   subgraph 内工具调用
```

---

## Channels 与 Event Lifecycle

Raw events 的 channel 由 `event["method"]` 表示。

| Channel | Purpose |
|---------|---------|
| `values` | full graph state snapshots |
| `updates` | per-node state deltas |
| `messages` | content-block-centric chat model output |
| `tools` | tool start/output/finish/error |
| `lifecycle` | run、subgraph、subagent status changes |
| `checkpoints` | lightweight checkpoint envelopes |
| `input` | human-in-the-loop input requests/responses |
| `tasks` | Pregel task creation/result events |
| `custom` | user-defined payloads |
| `custom:<name>` | custom transformer output |

### Messages lifecycle

`messages` channel 的 data event：

| Event | 说明 |
|-------|------|
| `message-start` | 一条模型消息开始 |
| `content-block-start` | 一个 content block 开始 |
| `content-block-delta` | content block 增量 |
| `content-block-finish` | content block 结束 |
| `message-finish` | 一条消息结束，可包含 usage |

Content block 可以表示：

```text
text token
reasoning block
tool-call block
multimodal content
```

Raw 消费 text/reasoning：

```python
for event in stream:
    if event["method"] != "messages":
        continue

    data = event["params"]["data"][0]
    if not isinstance(data, dict):
        continue
    if data.get("event") != "content-block-delta":
        continue

    block = data.get("delta") or {}
    if block.get("type") == "text-delta":
        print(block.get("text", ""), end="", flush=True)
    elif block.get("type") == "reasoning-delta":
        print(f"[thinking]{block.get('reasoning', '')}", end="", flush=True)
```

### Tools lifecycle

`tools` channel event：

| Event | 说明 |
|-------|------|
| `tool-started` | 工具开始 |
| `tool-output-delta` | 工具输出增量 |
| `tool-finished` | 工具完成 |
| `tool-error` | 工具错误 |

Tool events 可通过 tool call ID 与 `messages` channel 中的 tool-call content block 关联。

### Lifecycle channel

`lifecycle` event：

| Event | 说明 |
|-------|------|
| `started` | run/scope 开始 |
| `running` | 运行中 |
| `completed` | 完成 |
| `failed` | 失败 |
| `interrupted` | 被 human input 暂停 |

可能包含：

```text
graph_name
error
cause
```

---

## Build Your Own Projection

当内置 projections 不符合应用需要时，可以写 stream transformer。

适合：

| 需求 | 示例 |
|------|------|
| 工具活动面板 | tool started/error/finished |
| token 统计 | total output tokens |
| 业务进度 | retrieval progress、artifact progress |
| 协议转换 | 转成前端自定义事件 |
| 审计日志 | 特定事件筛选 |

Transformer 工作流：

```text
Pregel modes
  -> Protocol events
  -> Built-in projections
  -> User transformers
  -> Run projections
```

Stream handler 对每个 protocol event：

1. 调用每个 transformer 的 `process(event)`。
2. 把 named `StreamChannel` push 接回主 event stream。
3. 存储事件，除非 transformer suppress。
4. run 结束时调用 `finalize()` 或 `fail()`。

---

## StreamTransformer

接口：

```python
from langgraph.stream import ProtocolEvent, StreamTransformer

class MyTransformer(StreamTransformer):
    def init(self) -> dict:
        ...

    def process(self, event: ProtocolEvent) -> bool:
        ...

    def finalize(self) -> None:
        ...

    def fail(self, err: BaseException) -> None:
        ...
```

方法说明：

| 方法 | 说明 |
|------|------|
| `init()` | 创建 projection object，返回到 `stream.extensions` |
| `process(event)` | 观察每个 protocol event；返回 false 可 suppress 原事件 |
| `finalize()` | 成功结束时关闭/resolve projection |
| `fail(err)` | 失败时把错误传播给 projection |

### required_stream_modes

Transformer 必须声明自己需要哪些底层 stream modes：

```python
class CustomTransformer(StreamTransformer):
    required_stream_modes = ("custom",)

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] == "custom":
            ...
        return True
```

关键点：

```text
required_stream_modes 打开上游 Pregel emission；
process() 仍会收到所有 emitted events，需要自己按 method 过滤。
```

有效值：

```text
messages
tools
custom
values
updates
checkpoints
tasks
debug
```

---

## StreamChannel

`StreamChannel` 是 transformer 发布 streaming projection 的基础。

| Need | Use |
|------|-----|
| 只作为 side-channel projection | `StreamChannel()` |
| 同时写回主 protocol event stream | `StreamChannel(name)` |

Named channel：

```text
stream.extensions["tool_activity"]
同时 raw stream 中出现 custom:tool_activity event
payload 必须可序列化
```

Unnamed channel：

```text
只在 stream.extensions 中可见
不会进入 raw protocol stream
适合 promise、async iterable、class instance 等不可序列化对象
```

### Named channel 示例

```python
from typing import TypedDict
from langgraph.stream import ProtocolEvent, StreamChannel, StreamTransformer

class ToolActivity(TypedDict):
    name: str
    status: str

class ToolActivityTransformer(StreamTransformer):
    required_stream_modes = ("tools",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self.activity = StreamChannel[ToolActivity]("tool_activity")

    def init(self) -> dict:
        return {"tool_activity": self.activity}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] != "tools":
            return True

        data = event["params"]["data"]
        if isinstance(data, dict) and data.get("tool_name") and data.get("event"):
            status = "error" if data["event"] == "tool-error" else "started"
            self.activity.push({"name": data["tool_name"], "status": status})
        return True
```

### Unnamed channel 示例

```python
class CustomTransformer(StreamTransformer):
    required_stream_modes = ("custom",)

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self.log = StreamChannel()

    def init(self) -> dict:
        return {"custom": self.log}

    def process(self, event: ProtocolEvent) -> bool:
        if event["method"] == "custom":
            self.log.push(event["params"]["data"])
        return True
```

注册：

```python
stream = graph.stream_events(
    input,
    version="v3",
    transformers=[CustomTransformer],
)

for item in stream.extensions["custom"]:
    print(item)
```

---

## ToolCallTransformer

LangGraph 内置 `ToolCallTransformer`，可在普通 `StateGraph` 上暴露 `stream.tool_calls`。

```python
from langgraph.prebuilt import ToolCallTransformer

stream = graph.stream_events(
    input,
    version="v3",
    transformers=[ToolCallTransformer],
)

for tool_call in stream.tool_calls:
    print(tool_call.tool_name, tool_call.input)
```

适合：

```text
你使用 plain StateGraph，但希望像 LangChain agent 那样消费工具调用 projection。
```

---

## Streaming Quickstart

Streaming API 通过 `graph.stream()` / `graph.astream()` 使用。

基本示例：

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode=["updates", "custom"],
    version="v2",
):
    if chunk["type"] == "updates":
        for node_name, state in chunk["data"].items():
            print(f"Node {node_name} updated: {state}")
    elif chunk["type"] == "custom":
        print(f"Status: {chunk['data']['status']}")
```

特点：

| 特点 | 说明 |
|------|------|
| `stream_mode` | 指定要接收哪些 runtime events |
| v2 format | 所有 chunk 统一为 dict |
| low-level | 更靠近 graph execution |
| 多模式 | 可传 list，如 `["updates", "custom"]` |

---

## Stream Output Format v2

`version="v2"` 后，每个 chunk 都是统一 `StreamPart`：

```python
{
    "type": "values" | "updates" | "messages" | "custom" | "checkpoints" | "tasks" | "debug",
    "ns": (),
    "data": ...,
}
```

字段：

| 字段 | 说明 |
|------|------|
| `type` | stream mode |
| `ns` | namespace tuple，subgraph events 时非空 |
| `data` | 当前 mode 的 payload |

v1 vs v2：

| 场景 | v1 | v2 |
|------|----|----|
| 单 mode | raw data | `StreamPart` dict |
| 多 mode | `(mode, data)` | `StreamPart` dict |
| subgraphs | `(namespace, data)` | `StreamPart` dict，读 `chunk["ns"]` |
| 多 mode + subgraphs | nested tuples/triples | 统一 dict |

建议：

```text
新代码统一使用 version="v2"。
```

---

## Stream Modes

| Mode | Type | Description |
|------|------|-------------|
| `values` | `ValuesStreamPart` | 每步后的 full state |
| `updates` | `UpdatesStreamPart` | 每步后的 state deltas |
| `messages` | `MessagesStreamPart` | LLM token + metadata |
| `custom` | `CustomStreamPart` | `get_stream_writer()` 发出的自定义数据 |
| `checkpoints` | `CheckpointStreamPart` | checkpoint events，需要 checkpointer |
| `tasks` | `TasksStreamPart` | task start/finish、result、error，需要 checkpointer |
| `debug` | `DebugStreamPart` | 尽可能多的调试信息 |

选择：

| 需求 | Mode |
|------|------|
| 看完整状态 | `values` |
| 看节点更新 | `updates` |
| 看 LLM token | `messages` |
| 看业务进度 | `custom` |
| 看 checkpoint | `checkpoints` |
| 看 task 生命周期 | `tasks` |
| 全量调试 | `debug` |

---

## Graph State：values 与 updates

`updates`：

```text
每个节点执行后返回的 state update。
包含 node name 和 update。
```

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="updates",
    version="v2",
):
    if chunk["type"] == "updates":
        for node_name, state in chunk["data"].items():
            print(f"Node `{node_name}` updated: {state}")
```

`values`：

```text
每一步后的完整 state。
```

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="values",
    version="v2",
):
    if chunk["type"] == "values":
        print(chunk["data"])
```

对比：

| Mode | 数据量 | 适合 |
|------|--------|------|
| `updates` | 小 | 进度、节点变更 |
| `values` | 大 | 完整 state 可视化、调试 |

---

## LLM Tokens：messages

`messages` mode 会从 graph 中任何 LLM 调用流式输出 token。

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        message_chunk, metadata = chunk["data"]
        if message_chunk.content:
            print(message_chunk.content, end="|", flush=True)
```

`data` 结构：

```text
(message_chunk, metadata)
```

metadata 常用：

| 字段 | 用途 |
|------|------|
| `langgraph_node` | 判断 token 来自哪个节点 |
| `tags` | 判断 token 来自哪个 LLM invocation |

过滤 by tags：

```python
if metadata["tags"] == ["joke"]:
    print(msg.content, end="|", flush=True)
```

过滤 by node：

```python
if msg.content and metadata["langgraph_node"] == "write_poem":
    print(msg.content, end="|", flush=True)
```

### Omit messages

给模型 invocation 加 `nostream` tag，可以不让其 token 出现在 `messages` mode。

适合：

| 场景 | 原因 |
|------|------|
| 内部 structured output | 不希望流给用户 |
| 私有 notes | 不进入客户端 |
| 已通过 custom channel 输出 | 避免重复 |

---

## Custom Data

节点或工具里使用 `get_stream_writer()`：

```python
from langgraph.config import get_stream_writer

def node(state: State):
    writer = get_stream_writer()
    writer({"custom_key": "Generating custom data inside node"})
    return {"answer": "some data"}
```

消费：

```python
for chunk in graph.stream(inputs, stream_mode="custom", version="v2"):
    if chunk["type"] == "custom":
        print(chunk["data"])
```

工具中也可以：

```python
@tool
def query_database(query: str) -> str:
    writer = get_stream_writer()
    writer({"data": "Retrieved 0/100 records", "type": "progress"})
    writer({"data": "Retrieved 100/100 records", "type": "progress"})
    return "some-answer"
```

注意：

```text
Python < 3.11 async 环境中，get_stream_writer 在 async node/tool 中不可用；
需要通过 writer 参数手动传入。
```

---

## Subgraph Outputs

Streaming API 中开启 subgraph 输出：

```python
for chunk in graph.stream(
    {"foo": "foo"},
    subgraphs=True,
    stream_mode="updates",
    version="v2",
):
    print(chunk["type"])
    print(chunk["ns"])
    print(chunk["data"])
```

`chunk["ns"]`：

| 值 | 含义 |
|----|------|
| `()` | root graph |
| `("node_name:<task_id>",)` | 某个 subgraph |
| 多段 tuple | nested subgraphs |

示例输出概念：

```text
Root: {'node_1': {'foo': 'hi! foo'}}
Subgraph ('node_2:<id>',): {'subgraph_node_1': {'bar': 'bar'}}
Subgraph ('node_2:<id>',): {'subgraph_node_2': {'foo': 'hi! foobar'}}
Root: {'node_2': {'foo': 'hi! foobar'}}
```

---

## Checkpoints、Tasks 与 Debug

### checkpoints

接收 checkpoint events，格式类似 `get_state()`。

要求：

```text
graph 需要 checkpointer。
```

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    config=config,
    stream_mode="checkpoints",
    version="v2",
):
    if chunk["type"] == "checkpoints":
        print(chunk["data"])
```

### tasks

接收 task start/finish events，包括运行节点、结果和错误。

要求：

```text
graph 需要 checkpointer。
```

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    config=config,
    stream_mode="tasks",
    version="v2",
):
    if chunk["type"] == "tasks":
        print(chunk["data"])
```

### debug

尽可能多地输出执行信息。

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="debug",
    version="v2",
):
    if chunk["type"] == "debug":
        print(chunk["data"])
```

注意：

```text
debug mode 组合了 checkpoints 和 tasks，并带额外 metadata。
如果只需要其中一部分，优先用 checkpoints 或 tasks。
```

---

## Multiple Modes

可以同时传多个 stream modes：

```python
for chunk in graph.stream(
    inputs,
    stream_mode=["updates", "custom"],
    version="v2",
):
    if chunk["type"] == "updates":
        ...
    elif chunk["type"] == "custom":
        ...
```

v2 的好处：

```text
无论单 mode、多 mode、subgraphs、多 mode + subgraphs，
都统一用 chunk["type"] / chunk["ns"] / chunk["data"]。
```

---

## Advanced Streaming

### Use with any LLM

如果 LLM 不是 LangChain chat model integration，也可以用 `custom` mode 输出 token。

```python
from langgraph.config import get_stream_writer

def call_arbitrary_model(state):
    writer = get_stream_writer()
    for chunk in your_custom_streaming_client(state["topic"]):
        writer({"custom_llm_chunk": chunk})
    return {"result": "completed"}

for chunk in graph.stream(
    {"topic": "cats"},
    stream_mode="custom",
    version="v2",
):
    if chunk["type"] == "custom":
        print(chunk["data"])
```

### Disable streaming for specific chat models

某些模型不支持 streaming，或不想输出其 token。

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "claude-sonnet-4-6",
    streaming=False,
)
```

或：

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="o1-preview", streaming=False)
```

如果集成不支持 `streaming` 参数：

```python
disable_streaming=True
```

---

## Migrate to v2

v2 Streaming format 提供统一输出。

| Scenario | v1 | v2 |
|----------|----|----|
| Single stream mode | raw data | `StreamPart` dict |
| Multiple stream modes | `(mode, data)` tuples | `StreamPart` dict |
| Subgraph streaming | `(namespace, data)` tuples | `StreamPart` dict with `ns` |
| Multiple modes + subgraphs | `(namespace, mode, data)` triples | `StreamPart` dict |
| `invoke()` return type | plain dict | `GraphOutput` |
| Interrupt location in stream | `__interrupt__` in state dict | `interrupts` field |
| Interrupt location in invoke | `__interrupt__` in result dict | `.interrupts` |
| Pydantic/dataclass output | plain dict | coerced model/dataclass instance |

### v2 invoke format

```python
from langgraph.types import GraphOutput

result = graph.invoke(inputs, version="v2")

assert isinstance(result, GraphOutput)
result.value
result.interrupts
```

旧式 dict access 仍兼容但已 deprecated：

```text
result["key"]
"key" in result
result["__interrupt__"]
```

应迁移到：

```text
result.value
result.interrupts
```

### Pydantic/dataclass state coercion

`version="v2"` 的 `values` mode 会把 state coerces 回对应类型。

```python
for chunk in graph.stream(
    {"value": "x", "items": []},
    stream_mode="values",
    version="v2",
):
    print(type(chunk["data"]))
```

---

## Async with Python < 3.11

Python < 3.11 的 async context propagation 有限制。

影响：

1. async LLM 调用必须显式传 `RunnableConfig`。
2. async nodes/tools 不能使用 `get_stream_writer()`，要用 `writer` 参数。

### async LLM 手动传 config

```python
async def call_model(state, config):
    joke_response = await model.ainvoke(
        [{"role": "user", "content": f"Write a joke about {state['topic']}"}],
        config,
    )
    return {"joke": joke_response.content}
```

### async custom streaming 用 writer 参数

```python
from langgraph.types import StreamWriter

async def generate_joke(state: State, writer: StreamWriter):
    writer({"custom_key": "Streaming custom data while generating a joke"})
    return {"joke": f"This is a joke about {state['topic']}"}
```

建议：

```text
能升级就使用 Python 3.11+；
否则显式传 config/writer，避免 streaming 丢失。
```

---

## 最佳实践

1. 新应用优先用 `stream_events(..., version="v3")`。
2. 需要低层 runtime modes 时用 `stream(..., version="v2")`。
3. 新代码统一使用 v2 `StreamPart`，避免 v1 tuple 兼容复杂度。
4. UI 分区展示时，用 typed projections，不要手写 namespace 解析。
5. 调试 subgraph 时，用 `subgraphs=True` 和 `chunk["ns"]`。
6. 自定义业务进度用 `get_stream_writer()` + `custom` mode。
7. 应用级 projection 用 `StreamTransformer`。
8. named `StreamChannel` payload 必须可序列化。
9. 需要 strict arrival order 时使用 `interleave()` 或 raw events。
10. checkpoints/tasks/debug 需要 checkpointer。
11. 内部 LLM 不想暴露 token 时用 `nostream` tag。
12. Python < 3.11 async 环境显式传 config/writer。

---

## 故障排查

| 问题 | 可能原因 | 处理方式 |
|------|----------|----------|
| `stream.messages` 有内容但 state 没更新 | 只消费了 message projection | 同时消费 `stream.values` 或读 `stream.output` |
| Streaming chunk 格式不一致 | 没传 `version="v2"` | 新代码统一传 `version="v2"` |
| 子图事件看不到 | 未设置 `subgraphs=True` 或未用 `stream.subgraphs` | Streaming 用 `subgraphs=True`，Event streaming 用 `stream.subgraphs` |
| custom 数据收不到 | 没有 `stream_mode="custom"` 或 transformer 未声明 `required_stream_modes` | 打开 custom mode 或声明 `("custom",)` |
| transformer 收不到某类事件 | 忘记声明 required mode | 设置 `required_stream_modes` |
| async custom writer 不工作 | Python < 3.11 context 限制 | 使用 `writer: StreamWriter` 参数 |
| LLM token 没有流出 | 模型 streaming disabled 或 tag 为 `nostream` | 检查模型配置和 tags |
| checkpoints/tasks 没输出 | graph 未配置 checkpointer | compile 时加 checkpointer，调用带 thread_id |
| raw events 顺序混乱 | 用 timestamp 排序 | 用 `seq` 排序，timestamp 不可靠 |
| named StreamChannel 报序列化问题 | payload 不可序列化 | 改用 unnamed channel 或换 JSON-serializable payload |

---

## 快速参考

### API 选择

| 目标 | API |
|------|-----|
| typed projections | `graph.stream_events(..., version="v3")` |
| low-level stream modes | `graph.stream(..., version="v2")` |
| async typed projections | `await graph.astream_events(..., version="v3")` |
| async stream modes | `graph.astream(..., version="v2")` |
| deployed Agent Server | LangSmith Streaming API |

### Event Streaming projections

| Projection | 说明 |
|------------|------|
| `stream.messages` | chat model messages |
| `stream.values` | state snapshots |
| `stream.output` | final output |
| `stream.subgraphs` | nested graph executions |
| `stream.interrupts` | HITL interrupt payloads |
| `stream.interrupted` | 是否暂停 |
| `stream.extensions` | custom projections |

### Streaming modes

| Mode | 说明 |
|------|------|
| `values` | full state |
| `updates` | per-node updates |
| `messages` | LLM token + metadata |
| `custom` | custom writer data |
| `checkpoints` | checkpoint events |
| `tasks` | task lifecycle |
| `debug` | all debug info |

### v2 StreamPart

```python
{
    "type": "...",
    "ns": (),
    "data": ...,
}
```

### ProtocolEvent

```python
{
    "seq": 1,
    "method": "messages",
    "params": {
        "namespace": [],
        "timestamp": 1234567890,
        "data": ...
    }
}
```

---

## 资料来源

- LangGraph Event streaming 官方文档：<https://docs.langchain.com/oss/python/langgraph/event-streaming>
- LangGraph Streaming 官方文档：<https://docs.langchain.com/oss/python/langgraph/streaming>
