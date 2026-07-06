# LangGraph Overview、Quickstart、Thinking 与 Workflows/Agents 详细指南

> 基于 LangGraph 官方 LangGraph overview、Quickstart、Thinking in LangGraph、Workflows and agents 文档整理。本文聚焦 LangGraph 的定位、核心价值、Graph API 与 Functional API 的快速上手方式、如何用 LangGraph 思维拆解 Agent，以及常见 workflow/agent 架构模式。

## 目录

1. [整体理解](#整体理解)
2. [LangGraph 是什么](#langgraph-是什么)
3. [LangChain 生态中的位置](#langchain-生态中的位置)
4. [安装与 Hello World](#安装与-hello-world)
5. [核心收益](#核心收益)
6. [Graph API Quickstart](#graph-api-quickstart)
7. [Functional API Quickstart](#functional-api-quickstart)
8. [Graph API vs Functional API](#graph-api-vs-functional-api)
9. [Thinking in LangGraph](#thinking-in-langgraph)
10. [Step 1：把流程拆成节点](#step-1把流程拆成节点)
11. [Step 2：识别每个节点的职责](#step-2识别每个节点的职责)
12. [Step 3：设计 State](#step-3设计-state)
13. [Step 4：实现 Nodes 与错误处理](#step-4实现-nodes-与错误处理)
14. [Step 5：连接图并运行](#step-5连接图并运行)
15. [Workflows 与 Agents 的区别](#workflows-与-agents-的区别)
16. [LLM Augmentations](#llm-augmentations)
17. [Workflow Pattern：Prompt Chaining](#workflow-patternprompt-chaining)
18. [Workflow Pattern：Parallelization](#workflow-patternparallelization)
19. [Workflow Pattern：Routing](#workflow-patternrouting)
20. [Workflow Pattern：Orchestrator-Worker](#workflow-patternorchestrator-worker)
21. [Workflow Pattern：Evaluator-Optimizer](#workflow-patternevaluator-optimizer)
22. [Agent Pattern](#agent-pattern)
23. [ToolNode](#toolnode)
24. [实践选择建议](#实践选择建议)
25. [最佳实践](#最佳实践)
26. [故障排查](#故障排查)
27. [快速参考](#快速参考)
28. [资料来源](#资料来源)

---

## 整体理解

LangGraph 是一个面向长时间运行、有状态、可恢复 Agent 和 Workflow 的低层编排框架与运行时。它不试图替你决定 prompt、agent 架构或业务流程，而是提供构建这些系统所需的基础能力：

```text
有状态执行
  + 节点/边/条件路由
  + 持久化与恢复
  + streaming
  + human-in-the-loop
  + retry / error handling
  + observability / deployment
= 可控、可调试、可上线的 Agent 编排 runtime
```

一句话：

```text
LangChain 帮你接模型和工具；
LangGraph 帮你把多步骤、有状态、可恢复的 agent/workflow 编排起来；
LangSmith 帮你观察、评估、部署和调试这些系统。
```

这四篇文档可以连成一条学习路径：

| 文档 | 关注点 | 读完后应该理解 |
|------|--------|----------------|
| LangGraph overview | 定位与能力边界 | LangGraph 是低层 orchestration runtime |
| Quickstart | 快速构建计算器 Agent | Graph API 与 Functional API 的基本写法 |
| Thinking in LangGraph | 设计思维 | 如何从业务流程拆节点、设计状态、处理错误和连接图 |
| Workflows and agents | 架构模式 | prompt chaining、parallelization、routing、orchestrator-worker、evaluator-optimizer、agent loop |

---

## LangGraph 是什么

官方对 LangGraph 的定位是：

```text
low-level orchestration framework and runtime
for building, managing, and deploying
long-running, stateful agents
```

关键词：

| 关键词 | 含义 |
|--------|------|
| low-level | 不内置固定 agent 架构，给开发者更细粒度控制 |
| orchestration | 负责节点、边、状态、路由、恢复、并行、循环等流程控制 |
| runtime | 不只是构图 DSL，还负责实际执行、持久化、streaming、interrupt |
| long-running | 支持长任务、跨时间恢复、失败后继续 |
| stateful | 每个节点通过共享 state 协作 |
| agents | 支持动态工具调用、循环、自主决策 |

LangGraph 不抽象：

| 不抽象的部分 | 说明 |
|--------------|------|
| Prompt 架构 | 你自己决定系统提示词、节点提示词、上下文格式 |
| Agent 架构 | 可以构建 ReAct、router、planner、multi-agent、自定义流程 |
| 业务语义 | 节点职责、state schema、错误策略由你定义 |

LangGraph 抽象：

| 抽象的部分 | 说明 |
|------------|------|
| Graph execution | 按节点和边执行 |
| State update | 节点读取 state，返回 partial updates |
| Conditional routing | 根据 state 决定下一步 |
| Persistence | 在节点边界 checkpoint，支持恢复 |
| Interrupt | 暂停并等待人类输入 |
| Streaming | 实时输出状态、消息、事件 |
| Retry / error handler | 对节点配置重试和补偿 |

---

## LangChain 生态中的位置

官方文档把 LangChain 产品栈分成不同层次：

| 产品 | 角色 | 典型用途 |
|------|------|----------|
| LangChain | Agent framework | 模型、工具、消息、agent loop、常见抽象 |
| LangGraph | Orchestration runtime | 有状态 workflow/agent、持久化、streaming、HITL |
| Deep Agents | Agent harness | 基于 LangGraph 的规划、subagents、文件系统工具、上下文管理 |
| LangSmith | Observability / evaluation / deployment platform | tracing、评估、prompt、部署 |
| LangSmith Engine | Trace-driven issue detection | 从 LangGraph trace 中发现问题并提出修复 |
| LangSmith Fleet | No-code agent builder | 模板、集成、自动化工作流 |

选择建议：

| 需求 | 推荐 |
|------|------|
| 刚开始做简单工具调用 Agent | LangChain agents |
| 需要完全控制 agent loop、状态、路由和恢复 | LangGraph |
| 需要开箱即用的深度 Agent 能力 | Deep Agents |
| 需要 trace、评估、部署、监控 | LangSmith |

---

## 安装与 Hello World

安装：

```bash
pip install -U langgraph
```

或：

```bash
uv add langgraph
```

最小 Hello World：

```python
from langgraph.graph import StateGraph, MessagesState, START, END

def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

graph = StateGraph(MessagesState)
graph.add_node(mock_llm)
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)
graph = graph.compile()

graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
```

这段代码体现了 LangGraph 的基本结构：

```text
StateGraph(State)
  -> add_node(function)
  -> add_edge(START, node)
  -> add_edge(node, END)
  -> compile()
  -> invoke(input_state)
```

---

## 核心收益

LangGraph 为 long-running stateful workflows/agents 提供以下核心能力：

| 能力 | 说明 |
|------|------|
| Persistence | 任务失败、中断或长时间运行后可从 checkpoint 恢复 |
| Human-in-the-loop | 可在任意节点检查和修改 state，等待人类审批或输入 |
| Memory | 支持短期工作记忆和跨会话长期记忆 |
| Streaming | 实时输出 token、状态、事件、进度 |
| Debugging with LangSmith | 可视化执行路径、状态变化、runtime metrics |
| Deployment | 面向 stateful long-running workflow 的生产部署能力 |

核心设计哲学：

```text
不要只构建一个“会回答”的 LLM 调用；
要构建一个“可观察、可恢复、可审查、可部署”的执行系统。
```

---

## Graph API Quickstart

Quickstart 用一个计算器 Agent 展示 Graph API。Agent 可用工具做加法、乘法、除法。

### 1. 定义工具与模型

```python
from langchain.tools import tool
from langchain.chat_models import init_chat_model

model = init_chat_model("claude-sonnet-4-6", temperature=0)

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

@tool
def divide(a: int, b: int) -> float:
    """Divide a and b."""
    return a / b

tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)
```

### 2. 定义 State

```python
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
```

关键点：

| State 字段 | 作用 |
|------------|------|
| `messages` | 保存对话、工具调用和工具结果 |
| `Annotated[..., operator.add]` | 新消息会 append，而不是替换整个列表 |
| `llm_calls` | 记录 LLM 调用次数，便于观测或限制 |

### 3. 定义模型节点

```python
from langchain.messages import SystemMessage

def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not."""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }
```

### 4. 定义工具节点

```python
from langchain.messages import ToolMessage

def tool_node(state: MessagesState):
    """Performs the tool call."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}
```

### 5. 定义条件路由

```python
from typing import Literal
from langgraph.graph import END

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tool_node"

    return END
```

### 6. 构建并编译图

```python
from langgraph.graph import StateGraph, START, END

agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END],
)
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()
```

执行逻辑：

```text
START
  -> llm_call
      -> 如果 LLM 生成 tool_calls：tool_node
      -> 否则：END
  -> tool_node
      -> 把 ToolMessage 写回 messages
      -> 回到 llm_call
```

---

## Functional API Quickstart

Functional API 用普通 Python 控制流表达同样的 Agent。它不显式定义节点和边，而是用 `@entrypoint` 和 `@task` 标记可执行单元。

### 1. 定义 task

```python
from langgraph.func import entrypoint, task

@task
def call_llm(messages: list[BaseMessage]):
    return model_with_tools.invoke(
        [
            SystemMessage(
                content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
            )
        ]
        + messages
    )

@task
def call_tool(tool_call: ToolCall):
    tool = tools_by_name[tool_call["name"]]
    return tool.invoke(tool_call)
```

### 2. 定义 entrypoint

```python
from langgraph.graph import add_messages

@entrypoint()
def agent(messages: list[BaseMessage]):
    model_response = call_llm(messages).result()

    while True:
        if not model_response.tool_calls:
            break

        tool_result_futures = [
            call_tool(tool_call) for tool_call in model_response.tool_calls
        ]
        tool_results = [fut.result() for fut in tool_result_futures]
        messages = add_messages(messages, [model_response, *tool_results])
        model_response = call_llm(messages).result()

    messages = add_messages(messages, model_response)
    return messages
```

执行并 stream：

```python
messages = [HumanMessage(content="Add 3 and 4.")]
stream = agent.stream_events(messages, version="v3")
for snapshot in stream.values:
    print(snapshot)
```

Functional API 的关键：

| 概念 | 说明 |
|------|------|
| `@task` | 标记可被 LangGraph runtime 管理的任务 |
| `@entrypoint` | 定义 workflow/agent 的入口 |
| `.result()` | 等待 task 结果 |
| 普通 Python 控制流 | 用 while/if/list comprehension 表达循环、分支、并行 |

---

## Graph API vs Functional API

| 维度 | Graph API | Functional API |
|------|-----------|----------------|
| 表达方式 | 显式节点和边 | 一个函数中的普通控制流 |
| 适合人群 | 喜欢画图、显式路由、可视化结构 | 喜欢写 Python 逻辑、控制流自然表达 |
| 节点 | `add_node(name, fn)` | `@task` |
| 入口 | `START` edge | `@entrypoint()` |
| 条件分支 | `add_conditional_edges()` 或 `Command(goto=...)` | `if/else` |
| 循环 | 边回到前面节点 | `while` |
| 并行 | 多条边或 `Send` | 多个 task future |
| 可视化 | 天然可画 graph | 更像程序流，但仍由 runtime 管理 |

选择建议：

| 需求 | 推荐 |
|------|------|
| 需要清晰展示节点和边 | Graph API |
| 需要条件路由、动态 worker、HITL、状态检查点可视化 | Graph API |
| 工作流本来就是普通程序逻辑 | Functional API |
| 想快速把现有函数包装进 LangGraph runtime | Functional API |
| 团队需要产品/运营也能看懂流程图 | Graph API |

---

## Thinking in LangGraph

官方建议用五步来思考 LangGraph Agent：

```text
1. Map out workflow as discrete steps
2. Identify what each step needs to do
3. Design your state
4. Build your nodes
5. Wire it together
```

换成中文：

```text
先画业务流程；
再把流程拆成节点；
然后设计节点共享的状态；
接着实现每个节点的输入、输出、错误处理；
最后用边、条件边或 Command 连接起来。
```

官方示例是客服邮件 Agent，需要完成：

| 需求 | 对应节点 |
|------|----------|
| 读取客户邮件 | `Read Email` |
| 分类紧急程度和主题 | `Classify Intent` |
| 搜索相关文档 | `Doc Search` |
| 创建/更新 bug issue | `Bug Track` |
| 起草回复 | `Draft Reply` |
| 复杂情况升级人工 | `Human Review` |
| 发送回复 | `Send Reply` |

---

## Step 1：把流程拆成节点

节点应该是“做一件明确事情”的函数。

客服邮件 Agent 的流程：

```text
START
  -> Read Email
  -> Classify Intent
      -> Doc Search
      -> Bug Track
      -> Human Review
  -> Draft Reply
      -> Human Review
      -> Send Reply
  -> END
```

节点类型：

| 节点 | 职责 |
|------|------|
| `Read Email` | 提取、解析邮件内容 |
| `Classify Intent` | 用 LLM 判断意图、紧急程度、主题，并决定路由 |
| `Doc Search` | 查询知识库 |
| `Bug Track` | 创建/更新 issue |
| `Draft Reply` | 生成回复草稿 |
| `Human Review` | 暂停并等待人工审批 |
| `Send Reply` | 调用邮件服务发送 |

注意：

```text
有些节点负责做事；
有些节点还负责决定下一步去哪。
```

例如：

| 节点 | 是否决策 |
|------|----------|
| `Read Email` | 通常固定进入 `Classify Intent` |
| `Classify Intent` | 根据分类结果路由 |
| `Draft Reply` | 根据复杂度决定是否人工审核 |
| `Send Reply` | 固定结束 |

---

## Step 2：识别每个节点的职责

官方把节点大致分成四类。

| 类型 | 适用场景 | 示例 |
|------|----------|------|
| LLM steps | 理解、分析、生成、推理决策 | 分类意图、起草回复 |
| Data steps | 从外部系统检索信息 | 文档搜索、客户历史查询 |
| Action steps | 执行外部副作用 | 发邮件、建 bug ticket |
| User input steps | 等待人类输入或审批 | 人工 review、补充客户 ID |

### LLM steps

LLM 节点要明确三件事：

| 内容 | 示例 |
|------|------|
| Static context | 分类类别、语气规范、输出格式 |
| Dynamic context | 邮件内容、发送人、历史记录、搜索结果 |
| Desired outcome | 结构化分类、回复草稿、路由决策 |

### Data steps

Data 节点要考虑：

| 内容 | 示例 |
|------|------|
| 参数 | 搜索 query、customer id |
| retry strategy | 网络失败时重试 |
| caching | 高频查询是否缓存 |
| fallback | 数据不可用时是否降级 |

### Action steps

Action 节点要尤其谨慎，因为它们通常有副作用：

| 内容 | 示例 |
|------|------|
| 何时执行 | 审批通过后发邮件 |
| 是否可重试 | 网络问题可重试，但要避免重复发送 |
| 是否可缓存 | 一般不能缓存 action 结果替代执行 |
| 幂等性 | 建 ticket、付款、发送通知要设计幂等键 |

### User input steps

User input 节点需要定义：

| 内容 | 示例 |
|------|------|
| 展示给人的上下文 | 原始邮件、草稿、紧急程度 |
| 人类输入格式 | approve/edit/reject |
| 触发条件 | 高紧急、复杂问题、质量风险 |

---

## Step 3：设计 State

State 是所有节点共享的 memory。每个节点读取 state，并返回 state 的部分更新。

官方强调：

```text
State should store raw data, not formatted text.
Format prompts inside nodes when needed.
```

也就是：

| 应该放进 state | 不建议放进 state |
|----------------|------------------|
| 原始 email 内容 | prompt 模板 |
| sender email / email id | 拼好的 prompt 字符串 |
| LLM 结构化分类结果 | 临时格式化的上下文 |
| 搜索结果 raw chunks | 可以从已有数据重新推导的文本 |
| 客户历史 raw dict | 节点私有中间变量 |
| 回复草稿 | 一次性日志或无需恢复的数据 |

客服邮件 Agent 的 state 示例：

```python
from typing import TypedDict, Literal

class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str

class EmailAgentState(TypedDict):
    email_content: str
    sender_email: str
    email_id: str
    classification: EmailClassification | None
    search_results: list[str] | None
    customer_history: dict | None
    draft_response: str | None
    messages: list[str] | None
```

这样做的好处：

| 好处 | 说明 |
|------|------|
| prompt 可独立演进 | 改 prompt 不需要改 state schema |
| 节点可复用同一数据 | 不同节点按需格式化 |
| 调试更清晰 | trace 中看到的是原始事实 |
| 恢复更可靠 | checkpoint 保存的是业务数据，不是临时字符串 |

---

## Step 4：实现 Nodes 与错误处理

LangGraph 中 node 本质上是函数：

```text
node(state) -> partial state update
```

如果节点还要决定下一步，可以返回 `Command(update=..., goto=...)`。

错误处理策略：

| Error Type | 谁来修 | 策略 | 使用场景 |
|------------|--------|------|----------|
| Transient errors | 系统自动 | RetryPolicy | 网络、rate limit、临时服务失败 |
| LLM-recoverable errors | LLM | 把错误写入 state 并路由回 LLM | 工具失败、解析失败、可让模型改参重试 |
| User-fixable errors | 人类 | `interrupt()` 暂停等待输入 | 缺 customer ID、指令不清、需要审批 |
| Recoverable failure after retries | 开发者声明 | `error_handler` 补偿/恢复分支 | 重试耗尽后进入补偿流程 |
| Unexpected errors | 开发者调试 | 直接抛出 | 未知错误，不要吞掉 |

Retry 示例：

```python
from langgraph.types import RetryPolicy

workflow.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0),
)
```

LLM-recoverable 示例：

```python
from langgraph.types import Command

def execute_tool(state: State) -> Command[Literal["agent", "execute_tool"]]:
    try:
        result = run_tool(state["tool_call"])
        return Command(update={"tool_result": result}, goto="agent")
    except ToolError as e:
        return Command(
            update={"tool_result": f"Tool error: {str(e)}"},
            goto="agent",
        )
```

Human input 示例：

```python
from langgraph.types import Command, interrupt

def lookup_customer_history(state: State) -> Command[Literal["draft_response"]]:
    if not state.get("customer_id"):
        user_input = interrupt({
            "message": "Customer ID needed",
            "request": "Please provide the customer's account ID",
        })
        return Command(
            update={"customer_id": user_input["customer_id"]},
            goto="lookup_customer_history",
        )

    customer_data = fetch_customer_history(state["customer_id"])
    return Command(update={"customer_history": customer_data}, goto="draft_response")
```

注意：

```text
interrupt() 所在节点在 resume 后会重新执行。
如果同一个节点里有 interrupt()，应尽量让 interrupt() 前面没有不可重复副作用。
```

---

## Step 5：连接图并运行

当节点自己用 `Command(goto=...)` 做路由时，图结构可以很简洁。

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

workflow = StateGraph(EmailAgentState)

workflow.add_node("read_email", read_email)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=3),
)
workflow.add_node("bug_tracking", bug_tracking)
workflow.add_node("draft_response", draft_response)
workflow.add_node("human_review", human_review)
workflow.add_node("send_reply", send_reply)

workflow.add_edge(START, "read_email")
workflow.add_edge("read_email", "classify_intent")
workflow.add_edge("send_reply", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```

为什么需要 checkpointer：

| 场景 | 作用 |
|------|------|
| human-in-the-loop | 暂停后保存 state，之后 resume |
| 长时间运行任务 | 中途失败可恢复 |
| 多轮会话 | 用 `thread_id` 绑定一次 conversation |
| 调试和回放 | 能看到每个 checkpoint 的 state |

运行 HITL 示例：

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "customer_123"}}

stream = app.stream_events(initial_state, config, version="v3")
_ = stream.output
print(stream.interrupts)

human_response = Command(
    resume={
        "approved": True,
        "edited_response": "We sincerely apologize...",
    }
)

resumed = app.stream_events(human_response, config, version="v3")
final_state = resumed.output
```

---

## Workflows 与 Agents 的区别

官方定义：

| 类型 | 特点 |
|------|------|
| Workflows | 代码路径预先确定，按既定顺序运行 |
| Agents | 动态决定自己的流程和工具使用 |

对比：

| 维度 | Workflow | Agent |
|------|----------|-------|
| 控制权 | 开发者定义流程 | LLM 在约束内决定下一步 |
| 路径 | 通常可预先枚举 | 动态、不可完全预知 |
| 适合任务 | 明确流程、稳定业务规则 | 问题空间开放、工具选择不确定 |
| 可控性 | 高 | 中等，需要 guardrails |
| 灵活性 | 中等 | 高 |
| 可测试性 | 更容易 | 更依赖 trace、eval、限制循环 |

经验判断：

```text
如果你能画出稳定流程，优先 workflow；
如果你无法预先知道需要哪些工具和步骤，再使用 agent loop。
```

---

## LLM Augmentations

Workflow 和 Agent 都建立在 LLM 及其增强能力之上。

常见增强：

| 增强 | 作用 |
|------|------|
| Tool calling | 让模型请求调用外部函数/API |
| Structured output | 让模型输出符合 schema 的结构化结果 |
| Short-term memory | 保存当前会话工作上下文 |

结构化输出示例：

```python
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query optimized for web search.")
    justification: str = Field(None, description="Why this query is relevant.")

structured_llm = llm.with_structured_output(SearchQuery)
output = structured_llm.invoke("How does Calcium CT score relate to high cholesterol?")
```

工具调用示例：

```python
def multiply(a: int, b: int) -> int:
    return a * b

llm_with_tools = llm.bind_tools([multiply])
msg = llm_with_tools.invoke("What is 2 times 3?")
msg.tool_calls
```

---

## Workflow Pattern：Prompt Chaining

Prompt chaining 指多个 LLM 调用串联，后一步处理前一步输出。

适合：

| 场景 | 说明 |
|------|------|
| 文档翻译 | 翻译、校对、风格润色 |
| 内容生成 | 初稿、改写、最终润色 |
| 一步一步验证 | 先生成，再检查，再修订 |

示例流程：

```text
generate_joke
  -> check_punchline
      -> Pass: END
      -> Fail: improve_joke
            -> polish_joke
            -> END
```

Graph API 核心写法：

```python
workflow.add_edge(START, "generate_joke")
workflow.add_conditional_edges(
    "generate_joke",
    check_punchline,
    {"Fail": "improve_joke", "Pass": END},
)
workflow.add_edge("improve_joke", "polish_joke")
workflow.add_edge("polish_joke", END)
```

Functional API 核心写法：

```python
@entrypoint()
def prompt_chaining_workflow(topic: str):
    original_joke = generate_joke(topic).result()
    if check_punchline(original_joke) == "Pass":
        return original_joke

    improved_joke = improve_joke(original_joke).result()
    return polish_joke(improved_joke).result()
```

---

## Workflow Pattern：Parallelization

Parallelization 指同时运行多个独立 LLM 任务，最后聚合结果。

适合：

| 场景 | 说明 |
|------|------|
| 拆分任务 | 文档不同部分并行处理 |
| 多角度评审 | 准确性、引用、风格分别评分 |
| 多候选生成 | 同一任务生成多个版本，提高选择质量 |
| 提升速度 | 独立任务并行执行 |

示例流程：

```text
START
  ├─ call_llm_1 -> joke
  ├─ call_llm_2 -> story
  └─ call_llm_3 -> poem
       ↓
    aggregator
       ↓
      END
```

Graph API 核心：

```python
parallel_builder.add_edge(START, "call_llm_1")
parallel_builder.add_edge(START, "call_llm_2")
parallel_builder.add_edge(START, "call_llm_3")
parallel_builder.add_edge("call_llm_1", "aggregator")
parallel_builder.add_edge("call_llm_2", "aggregator")
parallel_builder.add_edge("call_llm_3", "aggregator")
```

Functional API 核心：

```python
@entrypoint()
def parallel_workflow(topic: str):
    joke_fut = call_llm_1(topic)
    story_fut = call_llm_2(topic)
    poem_fut = call_llm_3(topic)
    return aggregator(
        topic,
        joke_fut.result(),
        story_fut.result(),
        poem_fut.result(),
    ).result()
```

---

## Workflow Pattern：Routing

Routing 先分析输入，再把请求分发到不同专用流程。

适合：

| 场景 | 路由维度 |
|------|----------|
| 客服 | pricing / refunds / returns / bugs |
| 内容生成 | story / joke / poem |
| 数据处理 | extract / summarize / classify |
| RAG | policy docs / product docs / billing docs |

核心是用结构化输出做 routing logic：

```python
from typing_extensions import Literal
from pydantic import BaseModel, Field

class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(
        None,
        description="The next step in the routing process",
    )

router = llm.with_structured_output(Route)
```

Graph API 条件路由：

```python
def route_decision(state: State):
    if state["decision"] == "story":
        return "llm_call_1"
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "poem":
        return "llm_call_3"

router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {
        "llm_call_1": "llm_call_1",
        "llm_call_2": "llm_call_2",
        "llm_call_3": "llm_call_3",
    },
)
```

Functional API：

```python
@entrypoint()
def router_workflow(input_: str):
    next_step = llm_call_router(input_)
    if next_step == "story":
        llm_call = llm_call_1
    elif next_step == "joke":
        llm_call = llm_call_2
    elif next_step == "poem":
        llm_call = llm_call_3

    return llm_call(input_).result()
```

---

## Workflow Pattern：Orchestrator-Worker

Orchestrator-worker 模式中，orchestrator 负责：

1. 把任务拆成子任务。
2. 把子任务分配给 workers。
3. 汇总 worker 输出形成最终结果。

适合：

| 场景 | 原因 |
|------|------|
| 子任务数量无法预先确定 | 需要 LLM 先规划 |
| 写报告 | orchestrator 生成章节，workers 写每章 |
| 修改多文件代码 | planner 找文件，workers 分别处理 |
| 内容批处理 | 每个 worker 处理一个 section/item |

Functional API 示例：

```python
@task
def orchestrator(topic: str):
    report_sections = planner.invoke([
        SystemMessage(content="Generate a plan for the report."),
        HumanMessage(content=f"Here is the report topic: {topic}"),
    ])
    return report_sections.sections

@task
def llm_call(section: Section):
    result = llm.invoke([
        SystemMessage(content="Write a report section."),
        HumanMessage(content=f"Section: {section.name}, {section.description}"),
    ])
    return result.content

@entrypoint()
def orchestrator_worker(topic: str):
    sections = orchestrator(topic).result()
    section_futures = [llm_call(section) for section in sections]
    return synthesizer([f.result() for f in section_futures]).result()
```

### Send API

LangGraph Graph API 提供 `Send` 动态创建 worker：

```python
from langgraph.types import Send

def assign_workers(state: State):
    return [Send("llm_call", {"section": s}) for s in state["sections"]]
```

配合 shared reducer：

```python
class State(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list, operator.add]
    final_report: str
```

含义：

| 设计 | 作用 |
|------|------|
| `Send("llm_call", {"section": s})` | 为每个 section 创建 worker 输入 |
| `WorkerState` | 每个 worker 有自己的局部 state |
| `completed_sections: Annotated[list, operator.add]` | 多个 worker 并行写入同一个聚合字段 |
| `synthesizer` | 汇总所有 worker 输出 |

---

## Workflow Pattern：Evaluator-Optimizer

Evaluator-optimizer 模式由一个生成器和一个评估器组成。

流程：

```text
generator
  -> evaluator
      -> accepted: END
      -> rejected + feedback: generator
```

适合：

| 场景 | 说明 |
|------|------|
| 翻译 | 需要语义一致但可能多次修订 |
| 代码生成 | 生成后由测试/评审反馈 |
| 文案生成 | 不满足风格或标准时改写 |
| 信息抽取 | 不符合 schema 或质量要求时重试 |

结构化评估：

```python
class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(
        description="Decide if the joke is funny or not."
    )
    feedback: str = Field(
        description="If not funny, provide feedback on how to improve it."
    )

evaluator = llm.with_structured_output(Feedback)
```

Graph API 路由：

```python
def route_joke(state: State):
    if state["funny_or_not"] == "funny":
        return "Accepted"
    elif state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"

optimizer_builder.add_conditional_edges(
    "llm_call_evaluator",
    route_joke,
    {
        "Accepted": END,
        "Rejected + Feedback": "llm_call_generator",
    },
)
```

Functional API：

```python
@entrypoint()
def optimizer_workflow(topic: str):
    feedback = None
    while True:
        joke = llm_call_generator(topic, feedback).result()
        feedback = llm_call_evaluator(joke).result()
        if feedback.grade == "funny":
            break
    return joke
```

与普通 Agent 的区别：

| Evaluator-Optimizer | Agent |
|---------------------|-------|
| 循环目标明确 | 循环目标可能开放 |
| evaluator 决定是否继续 | LLM 自己决定工具和步骤 |
| 适合有清晰验收标准 | 适合过程不可预知的问题 |

---

## Agent Pattern

Agent 通常是一个 LLM 通过工具执行动作的循环。

典型循环：

```text
messages
  -> LLM with tools
      -> 如果有 tool_calls
            -> 执行 tools
            -> ToolMessage 写回 messages
            -> 再次调用 LLM
      -> 如果没有 tool_calls
            -> 返回最终回答
```

Graph API 的核心节点：

| 节点 | 职责 |
|------|------|
| `llm_call` | 调用绑定工具的 LLM，决定是否调用工具 |
| `tool_node` | 执行模型请求的工具调用 |
| `should_continue` | 如果最后一条消息有 tool_calls，就去工具节点，否则结束 |

Agent 与 workflow 的差异：

| 维度 | Workflow | Agent |
|------|----------|-------|
| 谁决定下一步 | 代码 / 路由函数 | LLM 根据上下文决定 |
| 工具使用 | 通常固定在某节点 | 动态选择工具 |
| 循环 | 预定义 | 根据 tool_calls 继续 |
| 适合问题 | 结构清晰 | 过程不确定 |

但 Agent 仍然需要边界：

| 边界 | 示例 |
|------|------|
| 工具集 | 只暴露允许使用的 tools |
| 系统提示词 | 明确行为规范 |
| 最大步数 | 防止无限循环 |
| human review | 对高风险工具审批 |
| tracing/eval | 用 LangSmith 观察质量 |

---

## ToolNode

`ToolNode` 是 LangGraph 预构建的工具执行节点。

它处理：

| 能力 | 说明 |
|------|------|
| parallel tool execution | 多个 tool calls 可并行执行 |
| error handling | 内置工具执行错误处理能力 |
| state injection | 工具可读取图 state |
| context injection | 工具可读取 run-scoped context |

基础用法：

```python
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState, StateGraph

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

builder = StateGraph(MessagesState)
builder.add_node("tools", ToolNode([search, calculator]))
graph = builder.compile()
```

### 从工具读取 State 与 Context

Python 中工具可以通过 `ToolRuntime` 读取当前 graph state 和 run context。

```python
from dataclasses import dataclass
from langchain.tools import ToolRuntime, tool
from langgraph.graph import MessagesState

class State(MessagesState):
    user_id: str

@dataclass
class Context:
    organization_id: str

@tool
def get_user_info(runtime: ToolRuntime[Context, State]) -> str:
    user_id = runtime.state["user_id"]
    organization_id = runtime.context.organization_id
    return f"User {user_id} in organization {organization_id}"
```

注意：

```text
ToolNode 只能访问传给它的 state。
如果你手动从另一个节点调用 ToolNode，并且工具需要自定义 state 字段，
要传完整 state，而不是只传 {"messages": state["messages"]}。
```

---

## 实践选择建议

### API 选择

| 你的问题 | 推荐 |
|----------|------|
| 我想画出明确流程图 | Graph API |
| 我想用普通 Python 写循环和分支 | Functional API |
| 我需要动态创建 worker | Graph API + `Send` |
| 我需要工具执行节点 | `ToolNode` |
| 我需要人工审批/暂停恢复 | `interrupt()` + checkpointer |
| 我需要状态持久化 | compile 时配置 checkpointer |
| 我需要实时 UI | `stream_events(..., version="v3")` |
| 我需要 trace/debug/eval | LangSmith |

### 模式选择

| 任务特征 | 推荐模式 |
|----------|----------|
| 步骤固定，后一步依赖前一步 | Prompt chaining |
| 多个独立任务可同时做 | Parallelization |
| 先判断类型，再走专门流程 | Routing |
| 子任务数量由模型规划 | Orchestrator-worker |
| 有验收标准，需要反复改 | Evaluator-optimizer |
| 工具和路径无法预先确定 | Agent |

---

## 最佳实践

1. 从业务流程开始，而不是从代码开始。
2. 每个节点只做一件清晰的事情。
3. State 存原始数据，不存 prompt 字符串。
4. 在节点内按需格式化 prompt。
5. 对 LLM routing 使用 structured output，而不是自由文本解析。
6. 对外部 API 节点配置 `RetryPolicy`。
7. 对高风险 action 节点设计幂等性。
8. 用 `interrupt()` 处理需要人类信息或审批的问题。
9. 使用 checkpointer 支持 HITL、恢复和长任务。
10. 能用 workflow 解决时，不要过早上 fully dynamic agent。
11. 用 `ToolNode` 简化工具执行，而不是重复手写工具循环。
12. 用 LangSmith trace 检查状态变化、路由、工具调用和失败点。

---

## 故障排查

| 问题 | 可能原因 | 处理方式 |
|------|----------|----------|
| state 字段被覆盖 | reducer 没设置，默认替换 | 对 list 使用 `Annotated[list, operator.add]` 或合适 reducer |
| message 没累积 | `messages` 没配置 append reducer | 使用 `MessagesState` 或 `add_messages` |
| graph 不知道下一步去哪 | 条件边返回值和 mapping 不匹配 | 检查 `add_conditional_edges` mapping |
| interrupt 后无法恢复 | 没有 checkpointer 或 thread_id | compile 配 checkpointer，调用传 `configurable.thread_id` |
| resume 后副作用重复执行 | `interrupt()` 前有不可重复代码 | 把 `interrupt()` 放到节点开头，副作用放 resume 后 |
| 工具读不到 state | 手动调用 ToolNode 时只传了 messages | 传完整 state 给 ToolNode |
| agent 无限循环 | 没有限制迭代或工具反馈不清 | 加最大步数、改 prompt、加入 evaluator/HITL |
| LLM 路由不稳定 | 自由文本分类 | 使用 `with_structured_output()` |
| 并行 worker 输出丢失 | 聚合字段 reducer 不对 | 使用 `Annotated[list, operator.add]` |
| 外部服务偶发失败导致终止 | 未配置 retry | 对节点加 `RetryPolicy` |

---

## 快速参考

### LangGraph 基本构件

| 构件 | 说明 |
|------|------|
| `StateGraph(State)` | 定义一个基于 state 的图 |
| `START` / `END` | 图入口和出口 |
| `add_node()` | 添加节点函数 |
| `add_edge()` | 添加固定边 |
| `add_conditional_edges()` | 添加条件路由 |
| `compile()` | 编译成可执行 graph |
| `invoke()` | 同步执行 |
| `stream_events(..., version="v3")` | typed event streaming |
| `Command(update=..., goto=...)` | 节点同时更新 state 并指定下一跳 |
| `interrupt()` | 暂停执行，等待外部输入 |
| `RetryPolicy` | 节点重试策略 |
| `Send` | 动态创建 worker |
| `ToolNode` | 预构建工具执行节点 |

### Graph API Agent Skeleton

```python
builder = StateGraph(MessagesState)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)
builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")
agent = builder.compile()
```

### Functional API Agent Skeleton

```python
@task
def call_llm(messages):
    return model_with_tools.invoke(messages)

@task
def call_tool(tool_call):
    return tools_by_name[tool_call["name"]].invoke(tool_call)

@entrypoint()
def agent(messages):
    response = call_llm(messages).result()
    while response.tool_calls:
        results = [call_tool(tc).result() for tc in response.tool_calls]
        messages = add_messages(messages, [response, *results])
        response = call_llm(messages).result()
    return add_messages(messages, response)
```

### Pattern Cheatsheet

| Pattern | 核心结构 |
|---------|----------|
| Prompt chaining | A -> B -> C |
| Parallelization | A/B/C in parallel -> aggregate |
| Routing | classify -> branch |
| Orchestrator-worker | plan -> dynamic workers -> synthesize |
| Evaluator-optimizer | generate -> evaluate -> revise loop |
| Agent | LLM -> tools -> LLM loop |

---

## 资料来源

- LangGraph overview 官方文档：<https://docs.langchain.com/oss/python/langgraph/overview>
- LangGraph Quickstart 官方文档：<https://docs.langchain.com/oss/python/langgraph/quickstart>
- Thinking in LangGraph 官方文档：<https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph>
- Workflows and agents 官方文档：<https://docs.langchain.com/oss/python/langgraph/workflows-agents>
