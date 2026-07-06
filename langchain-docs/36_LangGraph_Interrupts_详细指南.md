# LangGraph Interrupts 详细指南

> 基于 LangGraph 官方 Interrupts 文档整理。本文聚焦如何用 `interrupt()` 动态暂停 graph、通过 checkpoint 和 `thread_id` 保存执行位置、用 `Command(resume=...)` 恢复运行，以及多 interrupt、工具审批、人工校验、静态断点和常见规则。

## 目录

1. [整体理解](#整体理解)
2. [Interrupts 是什么](#interrupts-是什么)
3. [动态 Interrupt 与静态 Breakpoint](#动态-interrupt-与静态-breakpoint)
4. [核心机制](#核心机制)
5. [使用前提](#使用前提)
6. [Pause using interrupt](#pause-using-interrupt)
7. [Resume interrupts](#resume-interrupts)
8. [Event Streaming 中处理 Interrupt](#event-streaming-中处理-interrupt)
9. [invoke API 中处理 Interrupt](#invoke-api-中处理-interrupt)
10. [Human-in-the-loop 循环](#human-in-the-loop-循环)
11. [Handling Multiple Interrupts](#handling-multiple-interrupts)
12. [Approve or Reject](#approve-or-reject)
13. [Review and Edit State](#review-and-edit-state)
14. [Interrupts in Tools](#interrupts-in-tools)
15. [Validating Human Input](#validating-human-input)
16. [Rules of Interrupts](#rules-of-interrupts)
17. [不要用 try/except 包住 interrupt](#不要用-tryexcept-包住-interrupt)
18. [不要重排同一节点内的 interrupt](#不要重排同一节点内的-interrupt)
19. [Interrupt payload 必须可序列化](#interrupt-payload-必须可序列化)
20. [Interrupt 前的副作用必须幂等](#interrupt-前的副作用必须幂等)
21. [Subgraphs Called as Functions](#subgraphs-called-as-functions)
22. [Debugging with Static Interrupts](#debugging-with-static-interrupts)
23. [LangSmith Studio](#langsmith-studio)
24. [与 Persistence、Checkpointer、Streaming 的关系](#与-persistencecheckpointerstreaming-的关系)
25. [最佳实践](#最佳实践)
26. [故障排查](#故障排查)
27. [快速参考](#快速参考)
28. [资料来源](#资料来源)

---

## 整体理解

LangGraph Interrupts 用来让 graph 在运行过程中暂停，并等待外部输入后继续。

典型场景：

| 场景 | 说明 |
|------|------|
| 人工审批 | 执行危险动作前暂停，让人确认 |
| 人工编辑 | 让人修改 LLM 生成内容或工具参数 |
| 表单补充 | 缺字段时暂停收集用户输入 |
| 工具调用审核 | 工具真正执行前暂停，允许批准、修改或取消 |
| 多分支审批 | 并行分支同时暂停，之后一次性恢复多个 interrupt |
| 调试断点 | 在节点前后设置静态断点，逐步观察状态 |

核心心智模型：

```text
node 执行到 interrupt(payload)
  -> LangGraph 抛出内部暂停信号
  -> checkpointer 保存当前 graph state
  -> 调用方拿到 interrupt payload
  -> graph 无限期等待
  -> 调用方用 Command(resume=value) 再次调用 graph
  -> 节点从头重新执行
  -> interrupt(...) 返回 resume value
  -> 节点继续往下运行
```

最关键的一点：

```text
恢复时不是从 interrupt 那一行继续；
而是从触发 interrupt 的节点开头重新执行。
```

这会直接影响副作用、多个 interrupt 的顺序、循环校验方式和 subgraph 行为。

---

## Interrupts 是什么

`interrupt()` 是 LangGraph 的动态暂停机制。

它允许你在节点函数里的任意位置暂停 graph：

```python
from langgraph.types import interrupt

def approval_node(state: State):
    approved = interrupt("Do you approve this action?")
    return {"approved": approved}
```

`interrupt()` 接收一个 JSON-serializable payload，这个 payload 会返回给调用方，用于 UI 展示或外部系统处理。

恢复时：

```python
from langgraph.types import Command

graph.stream_events(Command(resume=True), config=config, version="v3").output
```

`Command(resume=True)` 中的 `True` 会成为节点内部 `interrupt(...)` 的返回值：

```python
approved = interrupt("Do you approve this action?")
# resumed 后 approved == True
```

---

## 动态 Interrupt 与静态 Breakpoint

LangGraph 里有两类暂停机制：

| 类型 | 机制 | 适合场景 |
|------|------|----------|
| Dynamic interrupt | 节点代码中调用 `interrupt()` | Human-in-the-loop、审批、编辑、输入收集 |
| Static interrupt / breakpoint | `interrupt_before` / `interrupt_after` | 调试、测试、逐节点观察 |

区别：

| 维度 | Dynamic interrupt | Static breakpoint |
|------|-------------------|-------------------|
| 定义位置 | 节点代码内部 | compile 或 invoke 参数 |
| 是否可条件触发 | 可以，基于业务逻辑 | 固定在节点前后 |
| payload | 可以传业务数据 | 主要用于调试状态 |
| 恢复输入 | `Command(resume=...)` | 通常 `graph.invoke(None, config=config)` |
| 推荐用途 | HITL 生产流程 | Debugging |

官方建议：

```text
Human-in-the-loop 工作流使用 interrupt()；
调试和单步运行使用 static interrupts。
```

---

## 核心机制

`interrupt()` 依赖 LangGraph persistence 层。

当节点执行到 `interrupt()`：

1. graph 执行被暂停。
2. 当前状态通过 checkpointer 保存。
3. payload 返回给调用方。
4. graph 等待外部 resume。
5. resume payload 被传回节点。
6. 节点从开头重新运行。
7. 当重新执行到同一个 `interrupt()` 时，它返回 resume payload。

简化流程：

```text
initial input
  -> node starts
  -> interrupt(payload)
  -> checkpoint saved under thread_id
  -> caller sees payload
  -> Command(resume=value)
  -> same thread_id loads checkpoint
  -> node restarts
  -> interrupt(...) returns value
  -> graph continues
```

---

## 使用前提

使用动态 interrupts 需要三个条件：

| 条件 | 说明 |
|------|------|
| checkpointer | 用于持久化 graph state |
| thread_id | 用于定位要恢复的 checkpoint |
| JSON payload | `interrupt()` 传出的值要可序列化 |

示例：

```python
from langgraph.checkpoint.memory import InMemorySaver

graph = builder.compile(checkpointer=InMemorySaver())

config = {
    "configurable": {
        "thread_id": "approval-123"
    }
}
```

`thread_id` 是恢复游标：

| 行为 | 结果 |
|------|------|
| resume 时复用同一个 `thread_id` | 恢复同一个暂停点 |
| 使用新的 `thread_id` | 启动一个全新的 thread |
| 不传 `thread_id` | checkpointer 无法知道恢复哪条执行 |

生产环境建议：

```text
不要使用纯内存 checkpointer 作为生产持久层；
生产应使用数据库、Redis、Postgres、SQLite 等更持久的 checkpointer。
```

---

## Pause using interrupt

最小示例：

```python
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


class State(TypedDict):
    action_details: str
    approved: bool | None


def approval_node(state: State):
    approved = interrupt({
        "question": "Approve this action?",
        "details": state["action_details"],
    })
    return {"approved": approved}


builder = StateGraph(State)
builder.add_node("approval", approval_node)
builder.add_edge(START, "approval")
builder.add_edge("approval", END)
```

payload 可以是字符串：

```python
answer = interrupt("What is your age?")
```

也可以是结构化对象：

```python
decision = interrupt({
    "type": "approval",
    "title": "Transfer approval",
    "amount": 500,
    "currency": "USD",
})
```

建议用结构化 payload，因为更适合 UI 渲染：

| 字段 | 作用 |
|------|------|
| `type` | 让前端知道要渲染哪类交互 |
| `question` | 给用户看的问题 |
| `details` | 审批上下文 |
| `fields` | 需要编辑或填写的字段 |
| `current_values` | 当前状态中的已有值 |

---

## Resume interrupts

暂停后，用 `Command(resume=...)` 恢复：

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "thread-1"}}

initial = graph.stream_events(
    {"action_details": "Transfer $500", "approved": None},
    config=config,
    version="v3",
)

_ = initial.output

if initial.interrupted:
    print(initial.interrupts)

resumed = graph.stream_events(
    Command(resume=True),
    config=config,
    version="v3",
)

final_state = resumed.output
```

恢复时的关键点：

| 规则 | 说明 |
|------|------|
| 必须使用相同 `thread_id` | 否则不会恢复同一个暂停点 |
| `Command(resume=...)` 是输入 | 传给 `invoke` / `stream` / `stream_events` |
| resume 值成为 `interrupt()` 返回值 | 节点继续执行 |
| 节点从头重跑 | `interrupt()` 前面的代码会再次执行 |
| resume 值也应可序列化 | 尤其是跨进程、跨服务时 |

特别注意：

```text
作为 graph input 时，只推荐使用 Command(resume=...)。
Command(update=...)、Command(goto=...)、Command(graph=...) 主要用于节点函数 return，
不要把它们当作继续多轮对话的输入。
```

---

## Event Streaming 中处理 Interrupt

官方推荐用 `graph.stream_events(..., version="v3")` 驱动可能暂停的 graph。

原因是 Event Streaming 提供 typed projections：

| Projection | 作用 |
|------------|------|
| `stream.messages` | 读取 LLM token / message chunks |
| `stream.values` | 读取 state snapshots |
| `stream.interrupts` | 读取 interrupt payloads |
| `stream.interrupted` | 判断本轮是否暂停 |
| `stream.output` | 读取最终状态或驱动 stream 完成 |

示例：

```python
stream = graph.stream_events(
    {"input": "data"},
    config=config,
    version="v3",
)

final = stream.output

if stream.interrupted:
    interrupt_payloads = stream.interrupts
    print(interrupt_payloads)

resumed = graph.stream_events(
    Command(resume=True),
    config=config,
    version="v3",
)

final = resumed.output
```

注意：

```text
需要消费 stream，graph 才会实际向前运行。
常见做法是读取 stream.output，或遍历 messages / values 等 projection。
```

---

## invoke API 中处理 Interrupt

如果不需要流式输出，可以使用 `graph.invoke(...)`。

默认 `invoke()` 会把 interrupts 放在结果中的 `__interrupt__` 下：

```python
result = graph.invoke(
    {"action_details": "Transfer $500", "approved": None},
    config=config,
)

if "__interrupt__" in result:
    print(result["__interrupt__"])

resumed = graph.invoke(
    Command(resume=True),
    config=config,
)
```

选择建议：

| 需求 | 推荐 |
|------|------|
| UI 里要展示 token 流 | `stream_events(..., version="v3")` |
| 需要同时处理 messages、state、interrupt | `stream_events(..., version="v3")` |
| 简单同步流程，不需要流式 | `invoke()` |
| 需要底层 stream modes | `stream(..., version="v2")` |

---

## Human-in-the-loop 循环

交互式 agent 常见模式是循环调用 graph：

1. 用普通 input 启动。
2. 消费 messages 或 output。
3. 如果 `stream.interrupted`，读取 `stream.interrupts`。
4. 把 payload 渲染给用户。
5. 用 `Command(resume=user_response)` 恢复。
6. 直到不再 interrupted。

示例：

```python
from langgraph.types import Command

stream_input: dict | Command = {
    "messages": [
        {"role": "user", "content": "Please book the trip"}
    ]
}

while True:
    stream = graph.stream_events(
        stream_input,
        config=config,
        version="v3",
    )

    for message in stream.messages:
        for token in message.text:
            display_streaming_content(token)

    if not stream.interrupted:
        final_state = stream.output
        break

    interrupt_info = stream.interrupts[0].value
    user_response = get_user_input(interrupt_info)
    stream_input = Command(resume=user_response)
```

这个模式适合：

| 场景 | 说明 |
|------|------|
| Chat UI | 一边展示 token，一边等待审批 |
| Agent workflow | 工具执行前暂停 |
| 表单向导 | 多轮补充字段 |
| 审核后台 | 审核员修改并恢复任务 |

---

## Handling Multiple Interrupts

如果 graph 有并行分支，多个节点可能同时触发 interrupt。

例如：

```text
START
  -> node_a -> END
  -> node_b -> END
```

两个节点都调用 `interrupt()` 时，本轮会产生多个 pending interrupts。

这时恢复时应传入一个 map：

```python
resume_map = {
    interrupt.id: f"answer for {interrupt.value}"
    for interrupt in stream.interrupts
}

resumed = graph.stream_events(
    Command(resume=resume_map),
    config=config,
    version="v3",
)
```

完整示例：

```python
from typing import Annotated, TypedDict
import operator

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    vals: Annotated[list[str], operator.add]


def node_a(state):
    answer = interrupt("question_a")
    return {"vals": [f"a:{answer}"]}


def node_b(state):
    answer = interrupt("question_b")
    return {"vals": [f"b:{answer}"]}


graph = (
    StateGraph(State)
    .add_node("a", node_a)
    .add_node("b", node_b)
    .add_edge(START, "a")
    .add_edge(START, "b")
    .add_edge("a", END)
    .add_edge("b", END)
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "multi-1"}}

stream = graph.stream_events({"vals": []}, config=config, version="v3")
_ = stream.output

resume_map = {
    i.id: f"answer for {i.value}"
    for i in stream.interrupts
}

resumed = graph.stream_events(
    Command(resume=resume_map),
    config=config,
    version="v3",
)

print(resumed.output)
```

为什么要用 `id -> value`：

| 原因 | 说明 |
|------|------|
| 并行分支顺序不适合作为业务语义 | 用 id 更稳 |
| 每个 interrupt 都有唯一 id | 可以精确匹配 resume value |
| 一次恢复多个暂停点 | 避免把 A 的回答传给 B |

---

## Approve or Reject

审批是最常见的 interrupt 用法。

节点里暂停：

```python
from typing import Literal

from langgraph.types import Command, interrupt


def approval_node(state: State) -> Command[Literal["proceed", "cancel"]]:
    is_approved = interrupt({
        "question": "Do you want to proceed with this action?",
        "details": state["action_details"],
    })

    if is_approved:
        return Command(goto="proceed")

    return Command(goto="cancel")
```

恢复时传布尔值：

```python
# approve
graph.stream_events(Command(resume=True), config=config, version="v3").output

# reject
graph.stream_events(Command(resume=False), config=config, version="v3").output
```

完整 graph 结构：

```python
from typing import Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ApprovalState(TypedDict):
    action_details: str
    status: Literal["pending", "approved", "rejected"] | None


def approval_node(state: ApprovalState) -> Command[Literal["proceed", "cancel"]]:
    decision = interrupt({
        "question": "Approve this action?",
        "details": state["action_details"],
    })
    return Command(goto="proceed" if decision else "cancel")


def proceed_node(state: ApprovalState):
    return {"status": "approved"}


def cancel_node(state: ApprovalState):
    return {"status": "rejected"}


builder = StateGraph(ApprovalState)
builder.add_node("approval", approval_node)
builder.add_node("proceed", proceed_node)
builder.add_node("cancel", cancel_node)
builder.add_edge(START, "approval")
builder.add_edge("proceed", END)
builder.add_edge("cancel", END)

graph = builder.compile(checkpointer=InMemorySaver())
```

---

## Review and Edit State

有时不是简单批准/拒绝，而是让人编辑 graph state 中的某个字段。

示例：

```python
from langgraph.types import interrupt


def review_node(state: State):
    edited_content = interrupt({
        "instruction": "Review and edit this content",
        "content": state["generated_text"],
    })

    return {"generated_text": edited_content}
```

恢复：

```python
graph.stream_events(
    Command(resume="The edited and improved text"),
    config=config,
    version="v3",
).output
```

适合：

| 场景 | 示例 |
|------|------|
| 文案审核 | 人工修改 LLM draft |
| 数据补全 | 人工填入缺失字段 |
| 工具参数修改 | 修改 email subject/body、SQL 条件等 |
| 合同/报告审批 | 修改条款后继续生成 |

设计建议：

```python
edited = interrupt({
    "type": "edit",
    "field": "generated_text",
    "instruction": "Review and edit this content",
    "current_value": state["generated_text"],
})
```

这样 UI 可以根据 `type="edit"` 渲染文本框，而不是只显示一段字符串。

---

## Interrupts in Tools

`interrupt()` 可以直接放在 tool 函数内部。

这适合把审批逻辑和工具绑定在一起：

```python
from langchain.tools import tool
from langgraph.types import interrupt


@tool
def send_email(to: str, subject: str, body: str):
    """Send an email to a recipient."""

    response = interrupt({
        "action": "send_email",
        "to": to,
        "subject": subject,
        "body": body,
        "message": "Approve sending this email?",
    })

    if response.get("action") == "approve":
        final_to = response.get("to", to)
        final_subject = response.get("subject", subject)
        final_body = response.get("body", body)

        return f"Email sent to {final_to} with subject '{final_subject}'"

    return "Email cancelled by user"
```

恢复时可以批准并修改参数：

```python
resumed = graph.stream_events(
    Command(resume={
        "action": "approve",
        "subject": "Updated subject",
    }),
    config=config,
    version="v3",
)

print(resumed.output["messages"][-1])
```

这个模式的优点：

| 优点 | 说明 |
|------|------|
| 工具自带安全闸门 | 不依赖调用方记得审批 |
| 可复用 | 同一个 tool 在多个 graph 中都能暂停 |
| 支持参数编辑 | resume payload 可以覆盖原始 tool args |
| 适合高风险工具 | 发邮件、付款、数据库修改、外部 API 操作 |

注意：

```text
tool 中 interrupt 后，恢复时 tool 所在节点也会从头执行。
tool 真正的不可逆副作用应放在 interrupt 之后。
```

---

## Validating Human Input

人工输入可能无效，例如年龄不是正整数、审批字段缺失、邮箱格式错误。

官方推荐模式：

1. 每次节点调用只执行一次 `interrupt()`。
2. 如果输入无效，把新的问题写回 state。
3. 用 conditional edge 回到同一个节点。
4. 下一轮重新发起 interrupt。

不要在同一个节点里写：

```python
while True:
    answer = interrupt("...")
    if valid(answer):
        break
```

原因：

```text
每次 resume 都会从节点开头重跑。
如果在同一个节点内循环调用 interrupt，会导致之前迭代也被重放，
使执行次数越来越多，甚至出现指数级重复执行。
```

推荐写法：

```python
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


class FormState(TypedDict):
    age: int | None
    pending_question: str | None


def get_age_node(state: FormState):
    question = state.get("pending_question") or "What is your age?"
    answer = interrupt(question)

    if isinstance(answer, int) and answer > 0:
        return {"age": answer, "pending_question": None}

    return {
        "pending_question": (
            f"'{answer}' is not a valid age. "
            "Please enter a positive number."
        )
    }


def route(state: FormState):
    return END if state.get("age") is not None else "collect_age"


builder = StateGraph(FormState)
builder.add_node("collect_age", get_age_node)
builder.add_edge(START, "collect_age")
builder.add_conditional_edges("collect_age", route)
```

恢复流程：

```python
first = graph.stream_events(
    {"age": None, "pending_question": None},
    config=config,
    version="v3",
)
_ = first.output
print(first.interrupts)

retry = graph.stream_events(
    Command(resume="thirty"),
    config=config,
    version="v3",
)
_ = retry.output
print(retry.interrupts)

final = graph.stream_events(
    Command(resume=30),
    config=config,
    version="v3",
)
print(final.output["age"])
```

这个模式的关键收益：

| 收益 | 说明 |
|------|------|
| 每次 resume 只执行一次 interrupt | 避免重复 replay |
| 错误信息保存在 state | graph 状态清晰 |
| 校验逻辑可测试 | route 和 node 可单独测试 |
| UI 简单 | 每次只展示当前 pending question |

---

## Rules of Interrupts

Interrupts 有几条非常重要的规则。

根本原因：

```text
interrupt() 是通过抛出内部异常来通知 LangGraph runtime 暂停。
恢复后，runtime 会从节点开头重新执行，而不是从暂停行继续。
```

所以要遵守：

| 规则 | 原因 |
|------|------|
| 不要用裸 `try/except` 包住 `interrupt()` | 会吞掉内部暂停异常 |
| 不要重排同一节点里的多个 `interrupt()` | resume 值按 index 匹配 |
| payload 要可序列化 | checkpointer / server 需要持久化和传输 |
| `interrupt()` 前的副作用要幂等 | 恢复时前置代码会重跑 |
| 避免同一节点中非确定性循环 interrupt | 恢复时 interrupt 序列会错乱或重复 |

---

## 不要用 try/except 包住 interrupt

`interrupt()` 内部会抛出一个特殊异常，让 runtime 捕获后暂停 graph。

如果你写了裸 `try/except Exception`，可能会把这个异常吞掉：

```python
def node_a(state: State):
    try:
        name = interrupt("What's your name?")
    except Exception as e:
        print(e)

    return state
```

推荐：

```python
def node_a(state: State):
    name = interrupt("What's your name?")

    try:
        fetch_data()
    except NetworkException as e:
        print(e)

    return {"name": name}
```

或者只捕获具体业务异常：

```python
def node_a(state: State):
    try:
        name = interrupt("What's your name?")
        fetch_data()
    except NetworkException as e:
        print(e)

    return {"name": name}
```

对比：

| 写法 | 是否推荐 | 原因 |
|------|----------|------|
| `try: interrupt(...) except Exception` | 否 | 可能吞掉 interrupt 信号 |
| interrupt 后再 try 外部调用 | 是 | 暂停信号不会被业务异常处理挡住 |
| 捕获具体异常类型 | 是 | 避免捕获 LangGraph 内部控制流异常 |

---

## 不要重排同一节点内的 interrupt

同一个节点里可以有多个 `interrupt()`，但顺序必须稳定。

原因：

```text
LangGraph 为当前 task 维护 resume values 列表；
恢复时按 interrupt 调用顺序和 index 匹配。
```

推荐：

```python
def node_a(state: State):
    name = interrupt("What's your name?")
    age = interrupt("What's your age?")
    city = interrupt("What's your city?")

    return {
        "name": name,
        "age": age,
        "city": city,
    }
```

不推荐条件性跳过：

```python
def node_a(state: State):
    name = interrupt("What's your name?")

    if state.get("needs_age"):
        age = interrupt("What's your age?")

    city = interrupt("What's your city?")

    return {"name": name, "city": city}
```

如果 `needs_age` 在不同执行中变化，resume value 会错位。

不推荐基于动态列表循环：

```python
def node_a(state: State):
    results = []
    for item in state.get("dynamic_list", []):
        result = interrupt(f"Approve {item}?")
        results.append(result)

    return {"results": results}
```

更稳的做法：

| 需求 | 推荐方式 |
|------|----------|
| 多字段收集 | 固定顺序多个 interrupt，或拆成多个节点 |
| 动态列表审批 | fan-out 到多个并行节点，每个节点一个 interrupt |
| 输入校验循环 | 每次节点调用一个 interrupt + conditional edge |
| 可选字段 | 用 state 预先确定流程，然后进入不同节点 |

---

## Interrupt payload 必须可序列化

`interrupt(payload)` 的 payload 要能被 checkpointer 持久化。

推荐传简单类型：

```python
name = interrupt("What's your name?")
count = interrupt(42)
approved = interrupt(True)
```

推荐传简单结构：

```python
response = interrupt({
    "question": "Enter user details",
    "fields": ["name", "email", "age"],
    "current_values": state.get("user", {}),
})
```

不推荐传函数：

```python
def validate_input(value):
    return len(value) > 0


def node_a(state: State):
    response = interrupt({
        "question": "What's your name?",
        "validator": validate_input,
    })
    return {"name": response}
```

不推荐传类实例：

```python
class DataProcessor:
    def __init__(self, config):
        self.config = config


def node_a(state: State):
    processor = DataProcessor({"mode": "strict"})

    response = interrupt({
        "question": "Enter data to process",
        "processor": processor,
    })
    return {"result": response}
```

建议：

```text
把函数、校验器、处理器留在代码里；
payload 只传 type、字段名、当前值、约束描述等可序列化元数据。
```

---

## Interrupt 前的副作用必须幂等

由于恢复时节点会从开头重跑，`interrupt()` 前的代码可能执行多次。

如果在 `interrupt()` 前做了不可逆副作用，就可能重复执行：

| 非幂等副作用 | 风险 |
|--------------|------|
| 创建数据库记录 | resume 后重复插入 |
| 发送邮件 | 发送多封 |
| 扣款/下单 | 重复交易 |
| append 日志或历史 | 重复记录 |
| 调用外部 mutation API | 状态被重复修改 |

推荐做法一：使用幂等操作。

```python
def node_a(state: State):
    db.upsert_user(
        user_id=state["user_id"],
        status="pending_approval",
    )

    approved = interrupt("Approve this change?")

    return {"approved": approved}
```

推荐做法二：把副作用放在 interrupt 之后。

```python
def node_a(state: State):
    approved = interrupt("Approve this change?")

    if approved:
        db.create_audit_log(
            user_id=state["user_id"],
            action="approved",
        )

    return {"approved": approved}
```

推荐做法三：拆分成不同节点。

```python
def approval_node(state: State):
    approved = interrupt("Approve this change?")
    return {"approved": approved}


def notification_node(state: State):
    if state["approved"]:
        send_notification(
            user_id=state["user_id"],
            status="approved",
        )

    return state
```

不推荐：

```python
def node_a(state: State):
    audit_id = db.create_audit_log({
        "user_id": state["user_id"],
        "action": "pending_approval",
    })

    approved = interrupt("Approve this change?")

    return {"approved": approved, "audit_id": audit_id}
```

---

## Subgraphs Called as Functions

如果在父图节点里把 subgraph 当函数调用，而 subgraph 内部触发 interrupt：

```python
def node_in_parent_graph(state: State):
    some_code()
    subgraph_result = subgraph.invoke(some_input)
    return {"result": subgraph_result}


def node_in_subgraph(state: State):
    some_other_code()
    result = interrupt("What's your name?")
    return {"name": result}
```

恢复时：

| 层级 | 行为 |
|------|------|
| 父图 | 从调用 subgraph 的父节点开头重新执行 |
| 子图 | 从触发 interrupt 的子图节点开头重新执行 |

也就是说：

```text
some_code() 会重新执行；
some_other_code() 也会重新执行。
```

设计建议：

| 问题 | 建议 |
|------|------|
| 父节点里有昂贵计算 | 缓存结果或拆到独立节点 |
| 父节点里有副作用 | 放到 interrupt 后或改成幂等 |
| 子图内有多处 interrupt | 保持顺序稳定 |
| 想在 UI 中展示子图事件 | 配合 Event Streaming 的 `stream.subgraphs` |

---

## Debugging with Static Interrupts

静态 interrupts 是调试断点。

可以在 compile 时配置：

```python
graph = builder.compile(
    interrupt_before=["node_a"],
    interrupt_after=["node_b", "node_c"],
    checkpointer=checkpointer,
)

config = {
    "configurable": {
        "thread_id": "some_thread"
    }
}

graph.invoke(inputs, config=config)

graph.invoke(None, config=config)
```

也可以在运行时配置：

```python
graph.invoke(
    inputs,
    interrupt_before=["node_a"],
    interrupt_after=["node_b", "node_c"],
    config=config,
)

graph.invoke(None, config=config)
```

静态断点特点：

| 参数 | 说明 |
|------|------|
| `interrupt_before` | 在节点执行前暂停 |
| `interrupt_after` | 在节点执行后暂停 |
| `checkpointer` | 需要 checkpoint 才能恢复 |
| resume 输入 | 一般传 `None` |

用途：

| 用途 | 说明 |
|------|------|
| 单步调试 | 一次运行到下一个断点 |
| 检查 state | 在节点前后观察状态变化 |
| 测试路由 | 验证 conditional edges |
| 排查节点问题 | 找出哪个节点产生异常状态 |

注意：

```text
静态 interrupts 不推荐用于生产 HITL 流程。
生产 HITL 应使用节点内部的 interrupt()。
```

---

## LangSmith Studio

LangSmith Studio 可以在 UI 中设置静态 interrupts，并查看 graph 执行过程中的状态。

适合：

| 场景 | 说明 |
|------|------|
| 本地/远程调试 | 不必在代码里临时加断点 |
| 观察状态变化 | 看每个节点前后的 state |
| 分析执行路径 | 检查节点和边的实际运行顺序 |
| 调试 HITL | 查看 interrupt 前后的 graph 状态 |

与代码内 `interrupt()` 的关系：

```text
LangSmith Studio 的 static interrupt 更像调试器断点；
代码里的 interrupt() 更像业务流程里的人工等待点。
```

---

## 与 Persistence、Checkpointer、Streaming 的关系

Interrupts 不是孤立能力，它和 LangGraph 的 persistence、checkpointer、streaming 强相关。

### Persistence

`interrupt()` 能够暂停并恢复，是因为 LangGraph 会保存 state。

```text
没有持久化状态，就无法知道恢复到哪里。
```

### Checkpointer

checkpointer 负责保存 checkpoint。

| 环境 | 建议 |
|------|------|
| demo / 单进程测试 | `InMemorySaver` / `MemorySaver` |
| 本地持久化 | SQLite checkpointer |
| 生产服务 | 数据库型 durable checkpointer |

### thread_id

`thread_id` 是恢复游标。

```python
config = {"configurable": {"thread_id": "user-123-session-456"}}
```

建议：

| 场景 | thread_id 设计 |
|------|----------------|
| chat 会话 | conversation id |
| 审批任务 | approval request id |
| 工作流实例 | workflow run id |
| 用户表单 | form session id |

### Streaming

Event Streaming 是处理 interrupt 的推荐接口：

```python
stream = graph.stream_events(input, config=config, version="v3")

if stream.interrupted:
    payloads = stream.interrupts
```

好处：

| 好处 | 说明 |
|------|------|
| typed projection | 不用手动解析底层 chunks |
| 同时看 token 和 interrupt | UI 更自然 |
| `stream.output` | 统一拿最终状态 |
| `stream.subgraphs` | 可以处理子图消息 |

---

## 最佳实践

1. 生产环境使用 durable checkpointer。

```text
HITL 任务可能等待几分钟、几小时甚至几天；
不要依赖进程内存保存暂停点。
```

2. 为每个业务实例设计稳定 `thread_id`。

```text
thread_id = workflow_run_id / conversation_id / approval_id
```

3. interrupt payload 用结构化 JSON。

```python
interrupt({
    "type": "approval",
    "question": "Approve payment?",
    "details": {...},
})
```

4. 恢复时优先使用 `Command(resume=...)`。

```python
graph.stream_events(Command(resume=value), config=config, version="v3")
```

5. HITL UI 优先用 `stream_events(..., version="v3")`。

```text
messages、values、interrupts、output 都能以 typed projection 读取。
```

6. 每个校验节点每次只调用一次 `interrupt()`。

```text
输入无效时，用 state + conditional edge 重新进入节点。
```

7. 把不可逆副作用放在 interrupt 后。

```text
先问，再做；或者先写幂等 pending 状态，再等审批。
```

8. 同一节点内多个 interrupt 顺序必须稳定。

```text
不要用会改变 interrupt 数量和顺序的条件或循环。
```

9. 并行多个 interrupt 用 id map 恢复。

```python
Command(resume={interrupt.id: answer for interrupt in stream.interrupts})
```

10. 调试用 static interrupts，不要把它当生产审批机制。

---

## 故障排查

| 问题 | 常见原因 | 处理方式 |
|------|----------|----------|
| resume 后像是重新开始了 | 使用了新的 `thread_id` | 复用暂停时的 `thread_id` |
| interrupt 没返回给调用方 | 没有 checkpointer 或 stream 未被消费 | compile 加 checkpointer，并读取 `stream.output` 或遍历 stream |
| `interrupt()` 没暂停 | 被 `try/except Exception` 捕获 | 不要用裸 except 包住 interrupt |
| resume 后副作用重复执行 | 副作用在 `interrupt()` 前 | 改成幂等、移到 interrupt 后、拆节点 |
| 多个回答对应错了 | 同一节点内 interrupt 顺序变化 | 保持顺序稳定，或拆节点 |
| 并行多个 interrupt 恢复错乱 | 没用 interrupt id map | 使用 `Command(resume={id: value})` |
| payload 序列化失败 | 传了函数、类实例、不可 JSON 化对象 | 只传字符串、数字、布尔、列表、dict 等 |
| 输入校验越来越慢 | 在节点里 `while True + interrupt()` | 改成一次 interrupt + conditional edge |
| static breakpoint 无法恢复 | 没有 checkpointer 或 thread_id | compile 加 checkpointer，config 加 thread_id |
| 子图恢复后重复执行父节点代码 | subgraph 在父节点内触发 interrupt | 父节点会从头重跑，需保证前置代码幂等 |

---

## 快速参考

### 最小动态 interrupt

```python
from langgraph.types import interrupt


def node(state):
    answer = interrupt("Need approval?")
    return {"answer": answer}
```

### 使用 Event Streaming 启动

```python
stream = graph.stream_events(
    input_data,
    config={"configurable": {"thread_id": "thread-1"}},
    version="v3",
)

_ = stream.output
```

### 判断是否暂停

```python
if stream.interrupted:
    print(stream.interrupts)
```

### 恢复单个 interrupt

```python
from langgraph.types import Command

resumed = graph.stream_events(
    Command(resume=True),
    config=config,
    version="v3",
)

final = resumed.output
```

### 恢复多个 interrupts

```python
resume_map = {
    i.id: get_answer(i.value)
    for i in stream.interrupts
}

graph.stream_events(
    Command(resume=resume_map),
    config=config,
    version="v3",
)
```

### invoke API

```python
result = graph.invoke(input_data, config=config)

if "__interrupt__" in result:
    print(result["__interrupt__"])

graph.invoke(Command(resume=value), config=config)
```

### 静态断点

```python
graph = builder.compile(
    interrupt_before=["node_a"],
    interrupt_after=["node_b"],
    checkpointer=checkpointer,
)

graph.invoke(input_data, config=config)
graph.invoke(None, config=config)
```

### 核心规则

| 规则 | 记忆方式 |
|------|----------|
| 需要 checkpointer | 没保存就没法恢复 |
| 需要同一个 `thread_id` | thread_id 是恢复游标 |
| payload 要可序列化 | 要能持久化和传输 |
| resume 后节点从头重跑 | interrupt 前的代码会再执行 |
| 不要裸 try/except | interrupt 靠内部异常暂停 |
| 多 interrupt 顺序稳定 | resume 值按 index 匹配 |
| 副作用要放后面或幂等 | 避免重复执行 |

---

## 资料来源

- LangGraph Interrupts 官方文档：<https://docs.langchain.com/oss/python/langgraph/interrupts>
- LangGraph Persistence 官方文档：<https://docs.langchain.com/oss/python/langgraph/persistence>
- LangGraph Event Streaming 官方文档：<https://docs.langchain.com/oss/python/langgraph/event-streaming>
