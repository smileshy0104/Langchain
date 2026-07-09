# LangGraph Use Time Travel 详细指南

> 基于 LangGraph 官方 Use time-travel 文档整理。本文聚焦如何基于 checkpoints 回放历史执行、从历史 checkpoint 分叉探索替代路径，以及 time travel 与 interrupts、subgraphs、checkpointer 粒度之间的关系。

## 目录

1. [整体理解](#整体理解)
2. [Time Travel 是什么](#time-travel-是什么)
3. [Replay 与 Fork 对比](#replay-与-fork-对比)
4. [核心依赖：Checkpoints](#核心依赖checkpoints)
5. [Replay](#replay)
6. [Replay 示例](#replay-示例)
7. [Fork](#fork)
8. [Fork 示例](#fork-示例)
9. [update_state 与 as_node](#update_state-与-as_node)
10. [Time Travel 与 Interrupts](#time-travel-与-interrupts)
11. [Multiple Interrupts](#multiple-interrupts)
12. [Time Travel 与 Subgraphs](#time-travel-与-subgraphs)
13. [Inherited Checkpointer](#inherited-checkpointer)
14. [Subgraph Own Checkpointer](#subgraph-own-checkpointer)
15. [常见使用场景](#常见使用场景)
16. [最佳实践](#最佳实践)
17. [故障排查](#故障排查)
18. [快速参考](#快速参考)
19. [资料来源](#资料来源)

---

## 整体理解

LangGraph Time Travel 允许你基于过去的 checkpoint 重新执行 graph。

它主要有两种能力：

| 能力 | 说明 |
|------|------|
| Replay | 从历史 checkpoint 重新运行后续节点 |
| Fork | 从历史 checkpoint 创建一个带修改 state 的新分支 |

一句话：

```text
Replay 是“从过去某点再跑一次”；
Fork 是“从过去某点改一点状态，然后开一条新路线继续跑”。
```

Time Travel 的基本流程：

```text
thread execution
  -> checkpoint A
  -> checkpoint B
  -> checkpoint C

Replay from B:
  A/B 之前已保存的结果不重跑
  B 之后的节点重新执行

Fork from B:
  基于 B 写入一个新的 checkpoint
  原始历史不变
  从新 checkpoint 继续执行
```

重要提醒：

```text
Time Travel 不是简单读取缓存。
checkpoint 之后的节点会重新执行，包括 LLM 调用、API 请求和 interrupt。
```

---

## Time Travel 是什么

Time Travel 是 LangGraph 基于 persistence/checkpointer 提供的执行历史能力。

它允许你：

1. 查看某个 thread 的历史 checkpoints。
2. 选择一个历史 checkpoint。
3. 从该 checkpoint 重新执行后续节点。
4. 或者先修改该 checkpoint 的 state，再继续执行新分支。

适合：

| 场景 | 说明 |
|------|------|
| 调试 | 从错误前的 checkpoint 重新跑 |
| 对比实验 | 修改中间 state 后比较结果 |
| 人工修正 | 人工改掉某一步输出，再继续执行 |
| 替代路径探索 | 改 prompt、改输入、改 state 看后续行为 |
| HITL 回溯 | 从某个 interrupt 前后重新收集输入 |
| 测试 | 构造中间状态，跳过前置节点 |

---

## Replay 与 Fork 对比

| 维度 | Replay | Fork |
|------|--------|------|
| 目标 | 从旧 checkpoint 重新执行后续节点 | 从旧 checkpoint 分叉出新路径 |
| 是否修改 state | 不修改 | 可以通过 `update_state` 修改 |
| 原历史是否保留 | 保留 | 保留 |
| 是否回滚 thread | 否 | 否 |
| 后续节点是否重跑 | 是 | 是 |
| 典型 API | `graph.invoke(None, checkpoint.config)` | `graph.update_state(...)` + `graph.invoke(None, fork_config)` |
| 适合 | 重试、调试、重新生成 | 人工修改、A/B 路径、替代结果 |

关键区别：

```text
Replay 使用已有 checkpoint config 直接继续；
Fork 先创建一个新的 checkpoint，再从新 checkpoint 继续。
```

---

## 核心依赖：Checkpoints

Time Travel 依赖 LangGraph checkpoints。

每个 checkpoint 记录 graph 在某个时间点的状态，包括：

| 信息 | 说明 |
|------|------|
| state values | 当前 graph state |
| next | 接下来要执行的节点 |
| config | 包含 `thread_id`、`checkpoint_id` 等恢复信息 |
| metadata | checkpoint 相关元信息 |
| tasks | 当前待执行任务或子图状态 |

通过 `get_state_history(config)` 可以拿到某个 thread 的历史：

```python
history = list(graph.get_state_history(config))
```

注意：

```text
state history 通常按倒序返回，也就是最新 checkpoint 在前。
```

每个历史 state 中常用字段：

| 字段 | 作用 |
|------|------|
| `state.next` | 这个 checkpoint 后面要执行的节点 |
| `state.config` | 可用于 replay/fork 的 checkpoint config |
| `state.values` | checkpoint 中保存的 state |
| `state.tasks` | task/subgraph 相关信息 |

理解 `state.next` 很关键：

```text
state.next 不是“刚刚执行过的节点”，
而是“从这个 checkpoint 继续时，下一步要执行的节点”。
```

例如：

```text
ask_name -> ask_age -> final
```

如果某个 checkpoint 的：

```python
state.next == ("ask_age",)
```

它表示：

```text
ask_name 已经执行完成；
ask_age 还没有开始执行；
从这个 checkpoint replay/fork，会从 ask_age 继续。
```

因此，选择 checkpoint 时常见写法是：

```python
before_age = next(
    s for s in history
    if s.next == ("ask_age",)
)
```

这类变量名通常可以理解为：

```text
before_age = “ask_age 之前的 checkpoint”
```

注意 `get_state_history(config)` 通常按倒序返回：

```text
history[0] 是最新 checkpoint；
history[-1] 是最早 checkpoint。
```

如果同一个节点在循环或多轮调用中出现多次，`[0]` 和 `[-1]` 的语义不同：

| 写法 | 含义 |
|------|------|
| `matches[0]` | 最近一次 next 为目标节点的 checkpoint |
| `matches[-1]` | 最早一次 next 为目标节点的 checkpoint |
| `next(...)` | 等价于取第一个匹配，通常是最近一次 |

---

## Replay

Replay 指从一个历史 checkpoint 重新运行 graph。

调用方式：

```python
replay_result = graph.invoke(None, checkpoint.config)
```

语义：

| 部分 | 是否重新执行 |
|------|--------------|
| checkpoint 之前的节点 | 不重新执行 |
| checkpoint 中已保存的结果 | 直接使用 |
| checkpoint 之后的节点 | 重新执行 |
| LLM calls | 会重新调用 |
| API requests | 会重新请求 |
| interrupts | 会重新触发 |

重要警告：

```text
Replay re-executes nodes.
它不是 cache read，也不是只把历史结果返回。
```

如果从最终 checkpoint replay：

```text
如果 checkpoint 没有 next nodes，replay 是 no-op。
```

也就是说，最终 checkpoint 后没有待执行节点，所以不会发生新的执行。

---

## Replay 示例

定义 graph：

```python
from typing_extensions import NotRequired, TypedDict

from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph


class State(TypedDict):
    topic: NotRequired[str]
    joke: NotRequired[str]


def generate_topic(state: State):
    return {"topic": "socks in the dryer"}


def write_joke(state: State):
    return {
        "joke": f"Why do {state['topic']} disappear? They elope!"
    }


checkpointer = InMemorySaver()

graph = (
    StateGraph(State)
    .add_node("generate_topic", generate_topic)
    .add_node("write_joke", write_joke)
    .add_edge(START, "generate_topic")
    .add_edge("generate_topic", "write_joke")
    .compile(checkpointer=checkpointer)
)
```

运行一次：

```python
config = {
    "configurable": {
        "thread_id": str(uuid7())
    }
}

result = graph.invoke({}, config)
```

查看历史 checkpoint：

```python
history = list(graph.get_state_history(config))

for state in history:
    print(
        "next=",
        state.next,
        "checkpoint_id=",
        state.config["configurable"]["checkpoint_id"],
    )
```

找到 `write_joke` 前的 checkpoint：

```python
before_joke = next(
    s for s in history
    if s.next == ("write_joke",)
)
```

从该 checkpoint replay：

```python
replay_result = graph.invoke(None, before_joke.config)
```

执行效果：

```text
generate_topic 不重新执行；
write_joke 会重新执行。
```

---

## Fork

Fork 指从历史 checkpoint 创建一个新的分支。

核心 API：

```python
fork_config = graph.update_state(
    checkpoint.config,
    values={...},
)

fork_result = graph.invoke(None, fork_config)
```

语义：

| 操作 | 说明 |
|------|------|
| `update_state` | 在指定 checkpoint 基础上创建新 checkpoint |
| `values` | 写入或覆盖部分 state |
| `fork_config` | 新分支的 checkpoint config |
| `invoke(None, fork_config)` | 从新分支继续执行 |

最重要的点：

```text
update_state 不会回滚原 thread。
它会从指定 checkpoint 创建一个新的分支 checkpoint。
原始执行历史仍然保留。
```

---

## Fork 示例

继续使用前面的 graph：

```python
history = list(graph.get_state_history(config))

before_joke = next(
    s for s in history
    if s.next == ("write_joke",)
)
```

从 `write_joke` 前分叉，并修改 topic：

```python
fork_config = graph.update_state(
    before_joke.config,
    values={"topic": "chickens"},
)
```

继续执行新分支：

```python
fork_result = graph.invoke(None, fork_config)

print(fork_result["joke"])
```

执行效果：

```text
write_joke 使用新的 topic = "chickens" 重新执行；
原始 socks joke 历史不会被删除。
```

Fork 适合：

| 场景 | 说明 |
|------|------|
| 修改中间变量 | 改 `topic`、`plan`、`tool_args` |
| 人工修正 LLM 输出 | 改掉错误草稿后继续 |
| A/B 测试 | 对比不同 state 对后续结果的影响 |
| 跳过昂贵前置节点 | 保留前面结果，只重跑后面节点 |

---

## update_state 与 as_node

`update_state` 不只是简单把值塞进 state。

它会把这次更新视为某个节点产生的输出，并使用该节点的 writers，包括 reducers。

```python
fork_config = graph.update_state(
    before_joke.config,
    values={"topic": "chickens"},
    as_node="generate_topic",
)
```

`as_node` 的作用：

```text
告诉 LangGraph：这次 state update 应该被当作哪个节点写入的结果。
执行会从该节点的 successors 继续。
```

例如 graph 是：

```text
generate_topic -> write_joke
```

如果：

```python
graph.update_state(
    before_joke.config,
    values={"topic": "chickens"},
    as_node="generate_topic",
)
```

LangGraph 会认为 `generate_topic` 已经产生了新的 topic，然后从它的后继节点 `write_joke` 继续。

默认情况下，LangGraph 会根据 checkpoint 的版本历史推断 `as_node`。

大多数从历史 checkpoint fork 的场景不需要显式传 `as_node`。

需要显式指定 `as_node` 的情况：

| 场景 | 原因 |
|------|------|
| Parallel branches | 同一步多个节点更新 state，无法判断最后写入者 |
| No execution history | 新 thread 上手动设置 state，缺少历史 |
| Skipping nodes | 想让 graph 认为某个后面的节点已经运行过 |

如果推断失败，可能出现：

```text
InvalidUpdateError
```

---

## Time Travel 与 Interrupts

如果 graph 中使用 `interrupt()` 做 HITL，time travel 时 interrupts 会重新触发。

原因：

```text
time travel 会重新执行 checkpoint 之后的节点；
如果这些节点包含 interrupt()，就会再次暂停并等待新的 Command(resume=...)。
```

示例：

```python
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    value: list[str]


def ask_human(state: State):
    answer = interrupt("What is your name?")
    return {"value": [f"Hello, {answer}!"]}


def final_step(state: State):
    return {"value": ["Done"]}


graph = (
    StateGraph(State)
    .add_node("ask_human", ask_human)
    .add_node("final_step", final_step)
    .add_edge(START, "ask_human")
    .add_edge("ask_human", "final_step")
    .compile(checkpointer=InMemorySaver())
)
```

首次运行：

```python
config = {"configurable": {"thread_id": "1"}}

graph.invoke({"value": []}, config)
graph.invoke(Command(resume="Alice"), config)
```

从 `ask_human` 前 replay：

```python
history = list(graph.get_state_history(config))

before_ask = [
    s for s in history
    if s.next == ("ask_human",)
][-1]

replay_result = graph.invoke(None, before_ask.config)
```

结果：

```text
ask_human 重新执行；
interrupt("What is your name?") 再次暂停；
需要新的 Command(resume=...)。
```

Fork 时也是一样：

```python
fork_config = graph.update_state(
    before_ask.config,
    {"value": ["forked"]},
)

fork_result = graph.invoke(None, fork_config)

graph.invoke(Command(resume="Bob"), fork_config)
```

最终可能得到：

```python
{"value": ["forked", "Hello, Bob!", "Done"]}
```

---

## Multiple Interrupts

如果 graph 有多个 interrupt，例如多步表单：

```text
ask_name -> ask_age -> final
```

你可以从两个 interrupt 中间 fork，从而保留前一个回答，只重新询问后一个问题。

示例：

```python
def ask_name(state):
    name = interrupt("What is your name?")
    return {"value": [f"name:{name}"]}


def ask_age(state):
    age = interrupt("How old are you?")
    return {"value": [f"age:{age}"]}
```

完成两个 interrupt 后，找到中间 checkpoint：

```python
history = list(graph.get_state_history(config))

between = [
    s for s in history
    if s.next == ("ask_age",)
][-1]
```

这段代码的作用是：

```text
从历史 checkpoint 中找出“下一步要执行 ask_age”的状态。
```

也就是：

```text
ask_name 已经完成；
ask_age 尚未执行；
这个 checkpoint 位于两个 interrupt 中间。
```

变量名 `between` 的含义可以理解为：

```text
between = ask_name 和 ask_age 之间的 checkpoint
```

为什么过滤 `s.next == ("ask_age",)`：

| 条件 | 表示 |
|------|------|
| `s.next == ("ask_name",)` | 还没问名字 |
| `s.next == ("ask_age",)` | 已经问完名字，下一步要问年龄 |
| `s.next == ("final",)` | 已经问完年龄，下一步要进入 final |
| `s.next == ()` | graph 已完成，没有后续节点 |

为什么这里用了 `[-1]`：

```text
get_state_history(config) 通常按倒序返回。
如果 ask_age 这个节点在历史中出现多次，[-1] 会取最早一次匹配。
```

在简单的一次性表单里通常只有一个 `ask_age` checkpoint，`[0]`、`[-1]`、`next(...)` 得到的效果相同。

如果是循环、多轮表单或同一个节点可能多次出现，需要明确你想要哪一个：

```python
matches = [
    s for s in history
    if s.next == ("ask_age",)
]

latest_between = matches[0]   # 最近一次到达 ask_age 前
earliest_between = matches[-1]  # 最早一次到达 ask_age 前
```

更推荐在业务代码里写得更明确：

```python
ask_age_checkpoints = [
    s for s in history
    if s.next == ("ask_age",)
]

between = ask_age_checkpoints[0]  # 最近一次
```

或者：

```python
between = next(
    s for s in history
    if s.next == ("ask_age",)
)
```

从中间 fork：

```python
fork_config = graph.update_state(
    between.config,
    {"value": ["modified"]},
)

result = graph.invoke(None, fork_config)
```

执行效果：

```text
ask_name 的结果已在 checkpoint 中保存；
ask_age 会重新触发 interrupt，等待新的年龄回答。
```

适合：

| 场景 | 说明 |
|------|------|
| 多步表单 | 保留前面字段，只改后面字段 |
| 审批链 | 保留前面审批，只重做后续审批 |
| 人工标注 | 从某个问题重新标注 |
| Agent 交互 | 保留前面 user answer，重试后续 tool/LLM |

---

## Time Travel 与 Subgraphs

Subgraph 的 time travel 粒度取决于 subgraph 是否有自己的 checkpointer。

| Subgraph 配置 | Time travel 粒度 |
|---------------|------------------|
| 默认继承父图 checkpointer | 父图把整个 subgraph 当成一个 super-step |
| `compile(checkpointer=True)` | subgraph 内部每一步都有自己的 checkpoint history |

关键差异：

```text
默认 subgraph 只能从父图层面的 subgraph_node 前后 time travel；
有独立 checkpointer 的 subgraph 可以从 subgraph 内部某个节点之间 time travel。
```

---

## Inherited Checkpointer

默认情况下，subgraph 没有自己的 checkpointer，会继承父图的 checkpointer。

示例：

```python
subgraph = (
    StateGraph(State)
    .add_node("step_a", step_a)
    .add_node("step_b", step_b)
    .add_edge(START, "step_a")
    .add_edge("step_a", "step_b")
    .compile()
)

graph = (
    StateGraph(State)
    .add_node("subgraph_node", subgraph)
    .add_edge(START, "subgraph_node")
    .compile(checkpointer=InMemorySaver())
)
```

这种情况下：

```text
父图把整个 subgraph 执行视为一个 super-step。
```

限制：

| 限制 | 说明 |
|------|------|
| 不能 time travel 到 subgraph 内部两个节点之间 | 父图没有这一级 checkpoint |
| 从 subgraph 前 replay/fork | 整个 subgraph 会从头重新执行 |
| subgraph 内多个 interrupt | 不能只保留 step_a、重跑 step_b |

示例语义：

```python
history = list(graph.get_state_history(config))

before_sub = [
    s for s in history
    if s.next == ("subgraph_node",)
][-1]

fork_config = graph.update_state(
    before_sub.config,
    {"value": ["forked"]},
)

result = graph.invoke(None, fork_config)
```

结果：

```text
整个 subgraph 从 scratch 重新执行。
```

---

## Subgraph Own Checkpointer

如果给 subgraph 配置自己的 checkpoint history：

```python
subgraph = (
    StateGraph(State)
    .add_node("step_a", step_a)
    .add_node("step_b", step_b)
    .add_edge(START, "step_a")
    .add_edge("step_a", "step_b")
    .compile(checkpointer=True)
)
```

父图：

```python
graph = (
    StateGraph(State)
    .add_node("subgraph_node", subgraph)
    .add_edge(START, "subgraph_node")
    .compile(checkpointer=InMemorySaver())
)
```

此时 subgraph 内部也有 checkpoint。

可以用：

```python
parent_state = graph.get_state(config, subgraphs=True)
sub_config = parent_state.tasks[0].state.config
```

拿到 subgraph 自己的 checkpoint config。

然后从 subgraph checkpoint fork：

```python
fork_config = graph.update_state(
    sub_config,
    {"value": ["forked"]},
)

result = graph.invoke(None, fork_config)
```

执行效果：

```text
可以从 subgraph 内部的某个点继续；
例如保留 step_a 的结果，只重新执行 step_b。
```

适合：

| 场景 | 说明 |
|------|------|
| subgraph 内有多个 HITL interrupt | 希望从中间某个问题重新问 |
| subgraph 内步骤昂贵 | 不想从头重跑整个 subgraph |
| 子流程复杂 | 希望细粒度调试和分叉 |
| agent 子任务 | 子任务内部也需要回放/分叉 |

---

## 常见使用场景

### 重新生成后续结果

当中间 state 没问题，但后续 LLM 输出不满意：

```text
找到后续 LLM 节点前的 checkpoint
-> replay
-> 让 LLM 重新生成
```

### 人工修正中间状态

当某个中间字段错了：

```text
找到错误字段产生后的 checkpoint
-> update_state 修正
-> fork
-> 从新分支继续执行
```

### 对比替代路径

用于评估不同输入或策略：

```text
原路径: topic = socks
fork A: topic = chickens
fork B: topic = robots
对比后续结果
```

### 跳过前置昂贵步骤

如果前置步骤包括长时间检索、复杂规划或外部调用：

```text
保留前置 checkpoint
-> 只重跑后续节点
```

### HITL 回溯

如果用户想修改后面的问题回答：

```text
fork from between interrupts
-> 保留前面回答
-> 重新触发后面 interrupt
```

---

## 最佳实践

1. 为需要 time travel 的 graph 配置 checkpointer。

```python
graph = builder.compile(checkpointer=checkpointer)
```

2. 用稳定的 `thread_id` 管理历史。

```python
config = {"configurable": {"thread_id": run_id}}
```

3. 用 `get_state_history(config)` 查找 checkpoint。

```python
history = list(graph.get_state_history(config))
```

4. 根据 `state.next` 找到正确回放点。

```python
before_node = next(s for s in history if s.next == ("target_node",))
```

这里的 `before_node` 表示：

```text
target_node 还没执行，但从该 checkpoint 继续时将执行 target_node。
```

如果同一个 `target_node` 可能出现多次，不要随手使用 `[-1]`，先明确你需要最近一次还是最早一次：

```python
matches = [
    s for s in history
    if s.next == ("target_node",)
]

latest_before_node = matches[0]
earliest_before_node = matches[-1]
```

5. Replay 前确认后续节点可以安全重跑。

```text
LLM、API、工具、interrupt 都会再次执行。
```

6. Fork 时记住原历史不会被回滚。

```text
update_state 创建新 checkpoint，不是修改旧 checkpoint。
```

7. 并行分支或测试初始化时显式传 `as_node`。

```python
graph.update_state(config, values={...}, as_node="some_node")
```

8. 有复杂 subgraph 时，提前决定 checkpoint 粒度。

```text
只需要父图级别：subgraph 继承 checkpointer。
需要子图内部 time travel：subgraph.compile(checkpointer=True)。
```

9. 对包含副作用的节点保持幂等。

```text
Replay/Fork 会重跑后续节点，非幂等副作用可能重复发生。
```

10. 与 interrupts 结合时，UI 要准备好再次收集输入。

```text
time travel 后 interrupt 会重新触发。
```

---

## 故障排查

| 问题 | 常见原因 | 处理方式 |
|------|----------|----------|
| `get_state_history` 没有历史 | graph 没有 checkpointer 或 thread_id 不对 | compile 加 checkpointer，使用正确 config |
| Replay 后没发生任何事 | 选中了最终 checkpoint | 选择 `next` 非空的 checkpoint |
| Replay 后 LLM/API 又调用了一次 | 这是预期行为 | 确认后续节点可重跑，必要时做幂等保护 |
| Fork 后原历史还在 | `update_state` 不回滚 thread | 这是预期行为，Fork 创建新 checkpoint |
| `update_state` 报 `InvalidUpdateError` | 多分支或无法推断 `as_node` | 显式传 `as_node` |
| Fork 后从错误节点继续 | `as_node` 推断不符合预期 | 指定正确 `as_node` |
| Time travel 后 interrupt 又暂停 | interrupt 所在节点被重跑 | 使用新的 `Command(resume=...)` 恢复 |
| 想从 subgraph 内部中间点恢复但找不到 | subgraph 没有自己的 checkpointer | subgraph 使用 `compile(checkpointer=True)` |
| 从默认 subgraph 前 fork 后全部重跑 | 默认 subgraph 被父图视为一个 super-step | 使用 subgraph own checkpointer 提高粒度 |
| 副作用重复发生 | 从 checkpoint 后重跑了含副作用节点 | 将副作用设计为幂等或移到更合适节点 |

---

## 快速参考

### 查看历史

```python
history = list(graph.get_state_history(config))

for state in history:
    print(state.next, state.config["configurable"]["checkpoint_id"])
```

### Replay

```python
checkpoint = next(
    s for s in history
    if s.next == ("target_node",)
)

result = graph.invoke(None, checkpoint.config)
```

### 选择目标节点前的 checkpoint

```python
# get_state_history 通常倒序：最新 checkpoint 在前
history = list(graph.get_state_history(config))

matches = [
    s for s in history
    if s.next == ("ask_age",)
]

latest_before_ask_age = matches[0]
earliest_before_ask_age = matches[-1]
```

语义：

| 选择 | 含义 |
|------|------|
| `s.next == ("ask_age",)` | 从该 checkpoint 继续会执行 `ask_age` |
| `matches[0]` | 最近一次到达 `ask_age` 前 |
| `matches[-1]` | 最早一次到达 `ask_age` 前 |

### Fork

```python
fork_config = graph.update_state(
    checkpoint.config,
    values={"key": "new value"},
)

result = graph.invoke(None, fork_config)
```

### Fork with as_node

```python
fork_config = graph.update_state(
    checkpoint.config,
    values={"topic": "chickens"},
    as_node="generate_topic",
)
```

### Interrupt resume after time travel

```python
result = graph.invoke(None, before_interrupt.config)

# graph pauses again
resumed = graph.invoke(
    Command(resume="new answer"),
    before_interrupt.config,
)
```

### Subgraph own checkpoint

```python
subgraph = subgraph_builder.compile(checkpointer=True)

parent_state = graph.get_state(config, subgraphs=True)
sub_config = parent_state.tasks[0].state.config

fork_config = graph.update_state(
    sub_config,
    {"value": ["forked"]},
)
```

### API 对照

| API | 作用 |
|-----|------|
| `get_state_history(config)` | 获取 thread 的 checkpoint 历史 |
| `get_state(config, subgraphs=True)` | 获取当前 state，包括 subgraph 状态 |
| `invoke(None, checkpoint.config)` | 从 checkpoint replay |
| `update_state(checkpoint.config, values)` | 从 checkpoint 创建 fork |
| `update_state(..., as_node="node")` | 指定这次 state update 属于哪个节点 |

### 核心规则

| 规则 | 说明 |
|------|------|
| Replay 会重跑后续节点 | 不是读缓存 |
| Fork 不会回滚原历史 | 创建新 checkpoint |
| 最终 checkpoint replay 是 no-op | 没有 next nodes |
| Interrupt 会重新触发 | 需要新的 resume |
| 默认 subgraph 只能父图级 time travel | 子图内部无独立历史 |
| `checkpointer=True` 让 subgraph 有内部 checkpoint | 可从子图中间 fork |

---

## 资料来源

- LangGraph Use time-travel 官方文档：<https://docs.langchain.com/oss/python/langgraph/use-time-travel>
- LangGraph Checkpointers 官方文档：<https://docs.langchain.com/oss/python/langgraph/checkpointers>
- LangGraph Interrupts 官方文档：<https://docs.langchain.com/oss/python/langgraph/interrupts>
- LangGraph Use subgraphs 官方文档：<https://docs.langchain.com/oss/python/langgraph/use-subgraphs>
