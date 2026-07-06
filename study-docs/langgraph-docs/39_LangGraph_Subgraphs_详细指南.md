# LangGraph Subgraphs 详细指南

> 基于 LangGraph 官方 Subgraphs 文档整理。本文聚焦 subgraph 的两种接入方式、父图与子图的 state 通信、三种 subgraph persistence 模式、subgraph state 查看，以及如何用 Event Streaming 观察嵌套执行。

## 目录

1. [整体理解](#整体理解)
2. [Subgraph 是什么](#subgraph-是什么)
3. [适用场景](#适用场景)
4. [安装](#安装)
5. [Subgraph Communication](#subgraph-communication)
6. [两种接入模式对比](#两种接入模式对比)
7. [Call a Subgraph Inside a Node](#call-a-subgraph-inside-a-node)
8. [不同 State Schema 示例](#不同-state-schema-示例)
9. [多层 Subgraphs](#多层-subgraphs)
10. [Add a Subgraph as a Node](#add-a-subgraph-as-a-node)
11. [共享 State Schema 示例](#共享-state-schema-示例)
12. [Subgraph Persistence](#subgraph-persistence)
13. [三种 Checkpointer 模式](#三种-checkpointer-模式)
14. [Per-invocation](#per-invocation)
15. [Per-thread](#per-thread)
16. [Stateless](#stateless)
17. [Checkpointer Reference](#checkpointer-reference)
18. [Namespace Isolation](#namespace-isolation)
19. [View Subgraph State](#view-subgraph-state)
20. [Per-invocation State Inspection](#per-invocation-state-inspection)
21. [Per-thread State Inspection](#per-thread-state-inspection)
22. [Stream Subgraph Outputs](#stream-subgraph-outputs)
23. [Raw Protocol Events](#raw-protocol-events)
24. [与 Interrupt、Memory、Time Travel 的关系](#与-interruptmemorytime-travel-的关系)
25. [最佳实践](#最佳实践)
26. [故障排查](#故障排查)
27. [快速参考](#快速参考)
28. [资料来源](#资料来源)

---

## 整体理解

Subgraph 是被另一个 graph 当作节点使用的 graph。

你可以把 LangGraph 应用拆成多个可组合的子流程：

```text
Parent graph
  -> normal node
  -> subgraph node
       -> subgraph node A
       -> subgraph node B
  -> normal node
```

一句话：

```text
Subgraph 让复杂 graph 可以模块化、复用、分团队开发，并支持多 agent 分工。
```

它的关键设计问题有两个：

1. 父图和子图如何交换 state？
2. 子图内部状态是否要在多次调用之间保留？

---

## Subgraph 是什么

官方定义：

```text
A subgraph is a graph that is used as a node in another graph.
```

也就是说，subgraph 本身仍然是一个完整的 LangGraph graph：

| 特性 | 说明 |
|------|------|
| 有自己的 state schema | 可以和父图相同，也可以不同 |
| 有自己的 nodes/edges | 内部可以是复杂 workflow |
| 可以被 compile | 得到 compiled graph |
| 可以作为父图 node | 直接 `add_node("name", subgraph)` |
| 也可以在父节点函数内 invoke | 手动做输入输出转换 |

---

## 适用场景

Subgraphs 常见用途：

| 场景 | 说明 |
|------|------|
| Multi-agent systems | 每个 agent 是一个 subgraph |
| 复用流程 | 同一组 nodes 在多个 graph 中复用 |
| 分团队开发 | 不同团队维护不同 subgraph |
| 隔离状态 | 每个 agent 有自己的私有 state |
| 复杂工作流拆分 | 把大 graph 拆成多个子流程 |
| 独立调试 | 单独测试某个 subgraph |

多团队协作时，关键是保持 subgraph interface 稳定：

```text
只要 input/output schema 不变，
父图不需要知道 subgraph 内部细节。
```

---

## 安装

```bash
pip install -U langgraph
```

或：

```bash
uv add langgraph
```

---

## Subgraph Communication

添加 subgraph 时，需要定义父图和子图如何通信。

核心取决于 state schema 是否共享 key。

| 模式 | 何时使用 | 通信方式 |
|------|----------|----------|
| Call a subgraph inside a node | 父图和子图 state schema 不同，或需要转换 state | 写 wrapper node，手动把 parent state 转为 subgraph input，再把 output 转回 parent state |
| Add a subgraph as a node | 父图和子图共享 state keys | 直接把 compiled subgraph 传给 `add_node` |

选择原则：

```text
state key 不共享或需要隔离：在 node 内 invoke subgraph；
state key 共享且想让子图直接读写父图 channel：直接 add_node。
```

---

## 两种接入模式对比

| 维度 | Inside a Node | As a Node |
|------|---------------|-----------|
| API | `subgraph.invoke(...)` | `builder.add_node("x", subgraph)` |
| 是否需要 wrapper | 需要 | 不需要 |
| state schema | 通常不同 | 必须有共享 key 才有意义 |
| state 转换 | 手动转换 | 自动读写共享 channels |
| 私有 state | 更容易隔离 | 子图可有私有 key，但输出到父图需走共享 key |
| 典型场景 | 多 agent 私有消息、接口适配 | 多 agent 共享 messages、模块化子流程 |
| namespace | 由调用路径/顺序影响 | 节点名提供稳定 namespace |

---

## Call a Subgraph Inside a Node

当父图和子图 state schema 不同，推荐在父图节点函数中调用 subgraph。

基本结构：

```python
from typing_extensions import TypedDict
from langgraph.graph.state import START, StateGraph


class SubgraphState(TypedDict):
    bar: str


def subgraph_node_1(state: SubgraphState):
    return {"bar": "hi! " + state["bar"]}


subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()


class State(TypedDict):
    foo: str


def call_subgraph(state: State):
    subgraph_output = subgraph.invoke({"bar": state["foo"]})
    return {"foo": subgraph_output["bar"]}


builder = StateGraph(State)
builder.add_node("node_1", call_subgraph)
builder.add_edge(START, "node_1")
graph = builder.compile()
```

这里 wrapper node 做了两次转换：

| 步骤 | 说明 |
|------|------|
| Parent -> Subgraph | `{"bar": state["foo"]}` |
| Subgraph -> Parent | `{"foo": subgraph_output["bar"]}` |

适合：

| 场景 | 说明 |
|------|------|
| 子 agent 有私有消息历史 | 父图不直接暴露子图 messages |
| 子流程有自己的 schema | 父图只关心最终结果 |
| 需要做输入输出适配 | 字段名、结构、类型都可转换 |
| 多层 subgraph | 每一层都显式转换接口 |

---

## 不同 State Schema 示例

子图：

```python
class SubgraphState(TypedDict):
    bar: str
    baz: str


def subgraph_node_1(state: SubgraphState):
    return {"baz": "baz"}


def subgraph_node_2(state: SubgraphState):
    return {"bar": state["bar"] + state["baz"]}


subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()
```

父图：

```python
class ParentState(TypedDict):
    foo: str


def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}


def node_2(state: ParentState):
    response = subgraph.invoke({"bar": state["foo"]})
    return {"foo": response["bar"]}


builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()
```

事件流中的 namespace 能看出嵌套执行：

```python
stream = graph.stream_events({"foo": "foo"}, version="v3")

for event in stream:
    if event["method"] == "updates":
        print(event["params"]["namespace"], event["params"]["data"])
```

示例输出：

```text
[] {'node_1': {'foo': 'hi! foo'}}
['node_2:...'] {'subgraph_node_1': {'baz': 'baz'}}
['node_2:...'] {'subgraph_node_2': {'bar': 'hi! foobaz'}}
[] {'node_2': {'foo': 'hi! foobaz'}}
```

---

## 多层 Subgraphs

Subgraphs 可以嵌套多层：

```text
parent
  -> child
       -> grandchild
```

每一层 state schema 可以不同。

关键规则：

```text
父图 key 不会自动出现在子图中；
子图 key 也不会自动出现在父图中。
只有 wrapper 显式传入/返回的字段会跨层流动。
```

例如：

```python
def call_grandchild_graph(state: ChildState) -> ChildState:
    grandchild_graph_input = {
        "my_grandchild_key": state["my_child_key"]
    }
    grandchild_graph_output = grandchild_graph.invoke(grandchild_graph_input)
    return {
        "my_child_key": grandchild_graph_output["my_grandchild_key"] + " today?"
    }
```

父图调用 child：

```python
def call_child_graph(state: ParentState) -> ParentState:
    child_graph_input = {"my_child_key": state["my_key"]}
    child_graph_output = child_graph.invoke(child_graph_input)
    return {"my_key": child_graph_output["my_child_key"]}
```

事件 namespace 可能显示为：

```text
['child:...', 'child_1:...'] {'grandchild_1': {...}}
['child:...'] {'child_1': {...}}
[] {'child': {...}}
```

这种模式适合复杂系统中的多层 agent 或多级业务流程。

---

## Add a Subgraph as a Node

当父图和子图共享 state keys 时，可以直接把 compiled subgraph 作为节点添加到父图。

最小示例：

```python
from typing_extensions import TypedDict
from langgraph.graph.state import START, StateGraph


class State(TypedDict):
    foo: str


def subgraph_node_1(state: State):
    return {"foo": "hi! " + state["foo"]}


subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()


builder = StateGraph(State)
builder.add_node("node_1", subgraph)
builder.add_edge(START, "node_1")
graph = builder.compile()
```

特点：

| 特点 | 说明 |
|------|------|
| 无需 wrapper | compiled graph 直接是 node |
| 共享 state channels | 子图可以读写父图共享 key |
| 子图仍可有私有 key | 但父图只接收共享 key 的更新 |
| namespace 稳定 | 基于父图 node name |

---

## 共享 State Schema 示例

子图 state：

```python
class SubgraphState(TypedDict):
    foo: str
    bar: str
```

其中：

| key | 作用 |
|-----|------|
| `foo` | 与父图共享 |
| `bar` | 子图私有 |

子图内部：

```python
def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}


def subgraph_node_2(state: SubgraphState):
    return {"foo": state["foo"] + state["bar"]}
```

父图：

```python
class ParentState(TypedDict):
    foo: str


def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}


builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", subgraph)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()
```

如果只打印父图级 updates：

```python
stream = graph.stream_events({"foo": "foo"}, version="v3")

for event in stream:
    if event["method"] == "updates" and not event["params"]["namespace"]:
        print(event["params"]["data"])
```

输出类似：

```text
{'node_1': {'foo': 'hi! foo'}}
{'node_2': {'foo': 'hi! foobar'}}
```

---

## Subgraph Persistence

Subgraph persistence 决定子图内部 state 在多次调用之间如何保存。

想象一个客服 bot 委托给 billing expert：

```text
billing expert 每次都从空白开始？
还是记住这个 customer 之前问过的问题？
```

由 subgraph `.compile(checkpointer=...)` 控制。

核心前提：

```text
父图必须 compile with checkpointer，
subgraph 的 interrupts、state inspection、per-thread memory 等能力才可用。
```

---

## 三种 Checkpointer 模式

| 模式 | `checkpointer=` | 行为 |
|------|-----------------|------|
| Per-invocation | `None` 默认 | 每次调用 fresh；单次调用内继承父图 checkpointer，支持 interrupt 和 durable execution |
| Per-thread | `True` | 同一个 thread 内多次调用会累积 state |
| Stateless | `False` | 完全无 checkpoint；像普通函数调用，不支持 interrupt/durable execution |

选择建议：

| 需求 | 推荐 |
|------|------|
| 大多数 multi-agent subagent 工具调用 | Per-invocation |
| 子 agent 需要跨多次调用记忆 | Per-thread |
| 纯函数式子流程，无暂停恢复需求 | Stateless |
| 需要 interrupt | Per-invocation 或 Per-thread |
| 需要并行多次调用同一个 subgraph | Per-invocation 或 Stateless |

---

## Per-invocation

默认模式：

```python
subgraph = subgraph_builder.compile()
# or
subgraph = subgraph_builder.compile(checkpointer=None)
```

行为：

| 行为 | 说明 |
|------|------|
| 每次调用 fresh | 不保留上次调用的 subgraph state |
| 单次调用内可持久化 | 继承父图 checkpointer |
| 支持 interrupt | 可暂停并 resume |
| 支持 durable execution | 单次调用中可恢复 |
| 支持 parallel calls | 每次 invocation 有独立 checkpoint namespace |

适合：

```text
subagent 处理独立请求，例如：
look up order、summarize document、answer one fruit question。
```

示例：

```python
fruit_agent = create_agent(
    model="gpt-5.4-mini",
    tools=[fruit_info],
    prompt="You are a fruit expert.",
)


@tool
def ask_fruit_expert(question: str) -> str:
    response = fruit_agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
    )
    return response["messages"][-1].content
```

外层 agent 配置 checkpointer：

```python
agent = create_agent(
    model="gpt-5.4-mini",
    tools=[ask_fruit_expert],
    prompt="Always delegate fruit questions.",
    checkpointer=MemorySaver(),
)
```

如果子图工具里触发 interrupt：

```python
@tool
def fruit_info(fruit_name: str) -> str:
    interrupt("continue?")
    return f"Info about {fruit_name}"
```

恢复：

```python
stream = agent.stream_events(
    {"messages": [{"role": "user", "content": "Tell me about apples"}]},
    config={"configurable": {"thread_id": "1"}},
    version="v3",
)

_ = stream.output

resumed = agent.stream_events(
    Command(resume=True),
    config={"configurable": {"thread_id": "1"}},
    version="v3",
)
```

多轮行为：

```text
第一次问 apples：subagent message count = 4
第二次问 bananas：subagent fresh，message count 仍是 4
```

---

## Per-thread

Per-thread 模式：

```python
subgraph = subgraph_builder.compile(checkpointer=True)
```

或对 `create_agent`：

```python
fruit_agent = create_agent(
    model="gpt-5.4-mini",
    tools=[fruit_info],
    prompt="You are a fruit expert.",
    checkpointer=True,
)
```

行为：

| 行为 | 说明 |
|------|------|
| 同一 thread 内累积 state | 每次调用接着上次 subgraph state |
| 支持 interrupt | 可以暂停恢复 |
| 支持 state inspection | 可查看累积 subgraph state |
| 不支持同一个 subgraph 并行多次调用 | 会产生 checkpoint namespace 冲突 |

适合：

| 场景 | 说明 |
|------|------|
| research assistant | 多次交互累积研究上下文 |
| coding assistant | 记住已经编辑的文件 |
| 专家 agent | 长期处理同一 thread 下相关问题 |
| 子会话 | 子 agent 有自己的多轮会话 |

并行调用警告：

```text
Per-thread subgraph 不支持 parallel tool calls。
如果同一个 subagent tool 被 LLM 同时调用多次，会写同一个 checkpoint namespace，
导致 checkpoint conflicts。
```

LangChain agent 可以用 `ToolCallLimitMiddleware` 限制：

```python
from langchain.agents.middleware import ToolCallLimitMiddleware


agent = create_agent(
    model="gpt-5.4-mini",
    tools=[ask_fruit_expert],
    middleware=[
        ToolCallLimitMiddleware(
            tool_name="ask_fruit_expert",
            run_limit=1,
        ),
    ],
    checkpointer=MemorySaver(),
)
```

多轮行为：

```text
第一次问 apples：subagent message count = 4
第二次问 bananas：subagent remembers apples，message count = 8
```

---

## Stateless

Stateless 模式：

```python
subgraph = subgraph_builder.compile(checkpointer=False)
```

行为：

| 行为 | 说明 |
|------|------|
| 无 checkpoint | 不保存子图状态 |
| 不支持 interrupt | 不能暂停/恢复 |
| 不支持 durable execution | 崩溃后必须从头重跑 |
| 不支持 state inspection | 没有子图 checkpoint 可看 |
| 开销低 | 像普通函数调用 |

适合：

| 场景 | 说明 |
|------|------|
| 纯计算子流程 | 不需要恢复 |
| 短小 deterministic workflow | 失败重跑成本低 |
| 不需要人工中断 | 没有 HITL |
| 性能敏感且可接受无持久化 | 减少 checkpoint overhead |

---

## Checkpointer Reference

| Feature | Per-invocation | Per-thread | Stateless |
|---------|----------------|------------|-----------|
| `checkpointer=` | `None` | `True` | `False` |
| Interrupts | 支持 | 支持 | 不支持 |
| Multi-turn memory | 不支持 | 支持 | 不支持 |
| Different subgraph multiple calls | 支持 | 需注意 namespace 隔离 | 支持 |
| Same subgraph multiple parallel calls | 支持 | 不支持 | 支持 |
| State inspection | 当前 invocation 内可看 | 可看累积 state | 不支持 |
| Durable execution | 单次 invocation 内支持 | 支持 | 不支持 |

解释：

| Feature | 说明 |
|---------|------|
| Interrupts | 子图可调用 `interrupt()` 暂停，等待 `Command(resume=...)` |
| Multi-turn memory | 子图 state 在同一 thread 多次调用之间保留 |
| State inspection | 通过 `graph.get_state(config, subgraphs=True)` 查看 |
| Durable execution | 进程失败后可从 checkpoint 恢复 |

---

## Namespace Isolation

当有多个 per-thread subgraphs 时，必须考虑 checkpoint namespace 隔离。

问题：

```text
如果在同一个 node 内按调用顺序调用多个 subgraph，
namespace 可能基于 call order。
一旦调用顺序改变，可能把 A 的 state 读到 B 上。
```

推荐做法：给每个 subagent 包一层带唯一 node name 的 StateGraph。

```python
from langgraph.graph import MessagesState, StateGraph


def create_sub_agent(model, *, name, **kwargs):
    agent = create_agent(model=model, name=name, **kwargs)

    return (
        StateGraph(MessagesState)
        .add_node(name, agent)
        .add_edge("__start__", name)
        .compile()
    )
```

使用：

```python
fruit_agent = create_sub_agent(
    "gpt-5.4-mini",
    name="fruit_agent",
    tools=[fruit_info],
    prompt="...",
    checkpointer=True,
)

veggie_agent = create_sub_agent(
    "gpt-5.4-mini",
    name="veggie_agent",
    tools=[veggie_info],
    prompt="...",
    checkpointer=True,
)
```

这样：

```text
fruit_agent 和 veggie_agent 拥有稳定、独立的 namespace。
```

注意：

```text
通过 add_node 添加的 subgraphs 已经自动获得基于节点名的 namespace，
通常不需要额外 wrapper。
```

---

## View Subgraph State

开启 persistence 后，可以查看 subgraph state：

```python
state = graph.get_state(config, subgraphs=True)
```

限制：

```text
LangGraph 必须能静态发现 subgraph。
```

可以查看的情况：

| 情况 | 是否支持 |
|------|----------|
| subgraph 通过 `add_node` 添加 | 支持 |
| subgraph 在 node 中直接调用 | 支持 |
| subgraph 在 tool 函数或其他间接层中调用 | 不支持 state inspection |
| stateless subgraph | 不支持 |

注意：

```text
即使 subgraph state inspection 不可用，interrupt 仍会传播到顶层 graph。
```

---

## Per-invocation State Inspection

Per-invocation 模式只能查看当前 invocation 的 subgraph state。

示例：

```python
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    foo: str


def subgraph_node_1(state: State):
    value = interrupt("Provide value:")
    return {"foo": state["foo"] + value}


subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()


builder = StateGraph(State)
builder.add_node("node_1", subgraph)
builder.add_edge(START, "node_1")

graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "1"}}

graph.invoke({"foo": ""}, config)

subgraph_state = graph.get_state(
    config,
    subgraphs=True,
).tasks[0].state

graph.invoke(Command(resume="bar"), config)
```

特点：

| 特点 | 说明 |
|------|------|
| 当前 invocation 内可查看 | 尤其是 interrupt 暂停时 |
| invocation 完成后不累积 | 下一次调用 fresh |
| 适合 debug 单次子流程 | 看当前子图卡在哪里 |

---

## Per-thread State Inspection

Per-thread 模式可以查看同一 thread 下累积的 subgraph state。

示例：

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph


subgraph_builder = StateGraph(MessagesState)
# add nodes and edges...
subgraph = subgraph_builder.compile(checkpointer=True)


builder = StateGraph(MessagesState)
builder.add_node("agent", subgraph)
builder.add_edge(START, "agent")

graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "1"}}

graph.invoke({"messages": [{"role": "user", "content": "hi"}]}, config)
graph.invoke({"messages": [{"role": "user", "content": "what did I say?"}]}, config)

subgraph_state = graph.get_state(
    config,
    subgraphs=True,
).tasks[0].state
```

特点：

| 特点 | 说明 |
|------|------|
| state 跨 invocation 累积 | 同一个 thread 下保留子图 state |
| 适合子 agent 多轮记忆 | 子 agent 记住自己之前对话 |
| 需要避免并行冲突 | 同一个 subgraph 不要并行多次调用 |

---

## Stream Subgraph Outputs

推荐用 Event Streaming 观察嵌套 graph 执行：

```python
stream = graph.stream_events({"foo": "foo"}, version="v3")

for subgraph in stream.subgraphs:
    print(subgraph.graph_name, subgraph.path)

    for snapshot in subgraph.values:
        print(subgraph.path, snapshot)
```

`stream.subgraphs` 提供 typed projection，不需要手动解析 namespace 字符串。

常用字段：

| 字段 | 说明 |
|------|------|
| `subgraph.graph_name` | 子图名称 |
| `subgraph.path` | 嵌套路径 |
| `subgraph.messages` | 子图内 message stream |
| `subgraph.values` | 子图内 state snapshots |

适合：

| 场景 | 说明 |
|------|------|
| UI 展示 nested agent | 显示哪个 agent 正在工作 |
| Debug 子流程 | 查看子图 state 更新 |
| 观察多层 graph | 展示 path |
| 分析工具/agent 调用 | 看子图内部输出 |

---

## Raw Protocol Events

如果需要底层事件，也可以直接遍历 `stream_events` 并解析 namespace：

```python
stream = graph.stream_events({"foo": "foo"}, version="v3")

for event in stream:
    if event["method"] == "updates":
        print(
            event["params"]["namespace"],
            event["params"]["data"],
        )
```

示例输出：

```text
[] {'node_1': {'foo': 'hi! foo'}}
['node_2:...'] {'subgraph_node_1': {'bar': 'bar'}}
['node_2:...'] {'subgraph_node_2': {'foo': 'hi! foobar'}}
[] {'node_2': {'foo': 'hi! foobar'}}
```

namespace 的含义：

| namespace | 说明 |
|-----------|------|
| `[]` | 父图层级事件 |
| `['node_2:...']` | `node_2` 子图内部事件 |
| 多个元素 | 多层嵌套 subgraph |

建议：

```text
应用层优先用 stream.subgraphs；
需要底层协议或自定义日志时再解析 raw events。
```

---

## 与 Interrupt、Memory、Time Travel 的关系

### Interrupt

Subgraph 内可以调用 `interrupt()`。

```text
只要 subgraph 有 checkpointer 支持，
interrupt 会暂停整个顶层 graph，并通过顶层 resume 恢复。
```

Per-invocation 和 per-thread 支持 interrupt。

Stateless 不支持 interrupt。

### Memory

Subgraph persistence 决定子图记忆行为：

| 模式 | 子图记忆 |
|------|----------|
| Per-invocation | 每次调用 fresh |
| Per-thread | 同 thread 多次调用累积 |
| Stateless | 无记忆 |

父图 checkpointer 是前提。

### Time Travel

Subgraph checkpoint 粒度影响 time travel：

| 配置 | Time Travel 粒度 |
|------|------------------|
| 默认继承父图 | 父图把整个 subgraph 当作一个 super-step |
| `checkpointer=True` | 子图内部也有 checkpoint history |
| `checkpointer=False` | 无子图 checkpoint 可 time travel |

---

## 最佳实践

1. 先决定父图和子图是否共享 state key。

```text
共享 key：直接 add_node；
不共享 key：在 node 中 invoke，并显式转换。
```

2. 子图接口要稳定。

```text
把 input/output schema 当作模块接口；
父图不依赖子图内部节点。
```

3. 大多数 subagent 工具调用使用 per-invocation。

```text
每次调用 fresh，避免不必要的记忆污染和 checkpoint 冲突。
```

4. 只有确实需要子 agent 多轮记忆时使用 per-thread。

```text
per-thread 会累积状态，也要处理并行调用冲突。
```

5. Per-thread subgraph 避免 parallel tool calls。

```text
用 middleware、模型配置或业务逻辑限制同一 subagent 并发调用。
```

6. 多个 per-thread subgraphs 要做 namespace isolation。

```text
用唯一 node name 包装 subagent，避免调用顺序变化造成状态混淆。
```

7. 需要查看子图状态时，保证 subgraph 可被静态发现。

```text
通过 add_node 或直接在 node 中调用；
不要藏在 tool 函数深处还期待 get_state(subgraphs=True) 可见。
```

8. 子图输出观察优先用 Event Streaming。

```python
for subgraph in stream.subgraphs:
    ...
```

9. Stateless 只用于可重跑、无中断、无持久化需求的子流程。

10. 复杂系统中建议单独测试 subgraph。

```text
先验证子图 input/output，再集成到父图。
```

---

## 故障排查

| 问题 | 常见原因 | 处理方式 |
|------|----------|----------|
| 子图读不到父图字段 | schema 不共享或没有传入 | 使用 wrapper 显式映射，或共享 state key |
| 父图拿不到子图私有字段 | 子图私有 key 不会自动回写父图 | 返回共享 key，或 wrapper 转换输出 |
| 子 agent 每次都忘记之前内容 | 使用了 per-invocation 默认模式 | 如果需要累积记忆，使用 `checkpointer=True` |
| per-thread 子图并行调用冲突 | 同一 namespace 被并行写入 | 禁止 parallel tool calls，或改用 per-invocation |
| 多个子 agent 状态混淆 | namespace 基于调用顺序不稳定 | 用唯一 node name 包装每个 subagent |
| `get_state(..., subgraphs=True)` 看不到子图 | 子图藏在 tool 函数或间接调用中 | 通过 `add_node` 或直接 node 调用让 LangGraph 静态发现 |
| stateless 子图无法 interrupt | `checkpointer=False` | 改为 `None` 或 `True` |
| 子图崩溃后不能恢复 | stateless 无 durable execution | 使用 stateful persistence |
| stream 中 namespace 难解析 | 直接读 raw events | 使用 `stream.subgraphs` typed projection |
| subgraph time travel 粒度太粗 | 子图没有自己的 checkpoint history | 使用 `compile(checkpointer=True)` |

---

## 快速参考

### Inside a node

```python
def call_subgraph(state: ParentState):
    subgraph_output = subgraph.invoke({"bar": state["foo"]})
    return {"foo": subgraph_output["bar"]}


builder.add_node("node_1", call_subgraph)
```

### As a node

```python
subgraph = subgraph_builder.compile()

builder.add_node("node_1", subgraph)
```

### Per-invocation

```python
subgraph = subgraph_builder.compile()
# or
subgraph = subgraph_builder.compile(checkpointer=None)
```

### Per-thread

```python
subgraph = subgraph_builder.compile(checkpointer=True)
```

### Stateless

```python
subgraph = subgraph_builder.compile(checkpointer=False)
```

### View subgraph state

```python
state = graph.get_state(config, subgraphs=True)
subgraph_state = state.tasks[0].state
```

### Stream subgraphs

```python
stream = graph.stream_events(input_data, version="v3")

for subgraph in stream.subgraphs:
    print(subgraph.graph_name, subgraph.path)
    for snapshot in subgraph.values:
        print(snapshot)
```

### Raw updates

```python
for event in graph.stream_events(input_data, version="v3"):
    if event["method"] == "updates":
        print(event["params"]["namespace"], event["params"]["data"])
```

### 模式选择

| 需求 | 推荐 |
|------|------|
| 父子 state schema 不同 | Inside a node |
| 父子共享 messages/foo 等 key | As a node |
| 每次调用独立 | Per-invocation |
| 子 agent 需要多轮记忆 | Per-thread |
| 无 interrupt / 无持久化 | Stateless |
| 需要观察嵌套输出 | `stream.subgraphs` |

---

## 资料来源

- LangGraph Subgraphs 官方文档：<https://docs.langchain.com/oss/python/langgraph/use-subgraphs>
- LangGraph Graph API 官方文档：<https://docs.langchain.com/oss/python/langgraph/graph-api>
- LangGraph Persistence 官方文档：<https://docs.langchain.com/oss/python/langgraph/persistence>
- LangGraph Interrupts 官方文档：<https://docs.langchain.com/oss/python/langgraph/interrupts>
- LangGraph Event Streaming 官方文档：<https://docs.langchain.com/oss/python/langgraph/event-streaming>
