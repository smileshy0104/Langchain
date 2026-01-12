# LangChain Multi-Agent（多智能体）详细指南

## 目录

1. [概述](#概述)
2. [核心概念](#核心概念)
3. [多智能体模式对比](#多智能体模式对比)
4. [Subagents（监督者/子智能体模式）](#subagents监督者子智能体模式)
5. [Handoffs（交接模式）](#handoffs交接模式)
6. [Skills（技能加载模式）](#skills技能加载模式)
7. [Router（路由模式）](#router路由模式)
8. [Custom Workflow（自定义工作流）](#custom-workflow自定义工作流)
9. [性能与成本分析](#性能与成本分析)
10. [选型指南](#选型指南)
11. [实战案例](#实战案例)
12. [最佳实践](#最佳实践)
13. [快速参考](#快速参考)

---

## 概述

### 什么是 Multi-Agent

**Multi-Agent（多智能体系统）** 是一种架构模式，通过多个专业化 Agent 协作来完成复杂任务。每个 Agent 专注于特定领域或功能，通过协调机制实现整体目标。

**核心思想**：不是"堆多个模型"，而是**用更清晰的职责分工和上下文管理**完成复杂任务。

```
┌─────────────────────────────────────────────────────────────┐
│                   Multi-Agent 系统                          │
│                                                             │
│    用户请求                                                 │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────┐                                               │
│  │ 协调层  │ ← Subagents / Handoffs / Router              │
│  └────┬────┘                                               │
│       │                                                     │
│       ├──────────┬──────────┬──────────┐                    │
│       ▼          ▼          ▼          ▼                    │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐              │
│  │ Agent 1│ │ Agent 2│ │ Agent 3│ │ Agent 4│              │
│  │ 日程   │ │ 邮件   │ │ 销售   │ │ 支持   │              │
│  └────────┘ └────────┘ └────────┘ └────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 为什么需要 Multi-Agent

| 单 Agent 问题 | Multi-Agent 解决方案 |
|---------------|---------------------|
| 所有功能集中，难以维护 | 每个 Agent 专注单一领域 |
| 复杂任务需处理多领域知识 | 专业 Agent 处理各自领域 |
| Prompt 过长，难以管理 | 每个 Agent 有简洁的 Prompt |
| 上下文膨胀，成本高昂 | 按需加载上下文和技能 |
| 难以扩展新功能 | 添加新 Agent 即可 |

### 何时使用 Multi-Agent

只有在以下需求明显时，才值得使用多智能体：

- **上下文管理**：需要在不同领域之间切换、避免上下文膨胀
- **并行化**：子任务可并行执行，降低总时延
- **团队协作（分布式开发）**：不同团队维护不同能力，组合成完整系统
- **专业化分工**：任务需要不同类型的专业知识

**重要提醒**：很多复杂任务，一个"单智能体 + 正确工具/提示词"也能完成。

---

## 核心概念

### 基础构件

LangChain Multi-Agent 的核心构件：

| 构件 | 说明 | 来源 |
|------|------|------|
| `create_agent` | Agent 工厂函数 | `langchain.agents` |
| `AgentState` | 基础状态结构 | `langchain.agents` |
| `ToolRuntime` | 工具运行时上下文 | `langchain.tools` |
| `Command` | 状态更新和跳转 | `langgraph.types` |
| `StateGraph` | 工作流图构建 | `langgraph.graph` |
| `checkpointer` | 状态持久化 | `langgraph.checkpoint` |

### 消息类型

```python
# 消息类型关系
AIMessage      # AI 生成的消息
├── tool_calls # 工具调用列表
└── content    # 文本内容

ToolMessage    # 工具执行结果
├── content    # 人类可读结果
└── artifact   # 结构化数据（可选）

HumanMessage  # 用户输入消息
SystemMessage # 系统提示词
```

### 共享状态设计

```python
from typing import Literal
from langchain.agents import AgentState
from typing_extensions import NotRequired

class MultiAgentState(AgentState):
    """多智能体系统的共享状态"""

    # 消息历史（继承自 AgentState）
    messages: list

    # 当前活跃的 Agent
    active_agent: NotRequired[str]

    # 工作流控制
    current_step: NotRequired[str]
    next_step: NotRequired[str]

    # 上下文数据
    user_data: NotRequired[dict]
    session_data: NotRequired[dict]

    # 技能状态
    skills_loaded: NotRequired[list[str]]

    # 元数据
    metadata: NotRequired[dict]
```

---

## 多智能体模式对比

官方文档提供的五种核心模式：

### 模式对比表

| 模式 | 核心机制 | 控制流 | 典型场景 | 子Agent直面用户 |
|------|----------|--------|----------|----------------|
| **Subagents** | 监督者调用子Agent工具(子代理作为工具) | 集中式 | 多域、多工具协调 | 否 |
| **Handoffs** | 状态驱动切换配置/Agent | 分散式 | 顺序对话流程 | 是 |
| **Skills** | 按需加载提示词/知识 | 单Agent | 大量技能、渐进式暴露 | 是 |
| **Router** | 路由器分类后并行派发 | 分发式 | 多垂直领域查询 | 部分 |
| **Custom Workflow** | LangGraph自定义流程 | 自由 | 复杂流程、需定制 | 可变 |

### 详细对比

| 维度 | Subagents | Handoffs | Skills | Router |
|------|-----------|----------|--------|--------|
| **分工开发** | 高 | 低 | 高 | 中 |
| **并行能力** | 高 | 低 | 中 | 高 |
| **多跳链路** | 高 | 高 | 高 | 低 |
| **上下文隔离** | 强 | 弱 | 中 | 强 |
| **对话连贯性** | 中 | 高 | 高 | 低 |
| **实现复杂度** | 中 | 中 | 低 | 高 |

### 性能对比

**场景 1：一次性请求（One-shot）**
- Subagents：4 次调用
- Handoffs：3 次调用
- Skills：3 次调用
- Router：3 次调用

**场景 2：重复请求（Repeat）**
- Subagents：8 次调用（4 + 4）
- Handoffs：5 次调用（3 + 2）
- Skills：5 次调用（3 + 2）
- Router：6 次调用（3 + 3）

**场景 3：多领域任务（Multi-domain）**
- Subagents：5 次调用 / ~9K tokens
- Handoffs：7+ 次调用 / ~14K+ tokens
- Skills：3 次调用 / ~15K tokens
- Router：5 次调用 / ~9K tokens

---

## Subagents（监督者/子智能体模式）

### 概述

**Supervisor Pattern** 使用中央 Agent（监督者）协调多个专业化 Agent（工作者）。

```
┌─────────────────────────────────────────────────────────────┐
│                    Supervisor 架构                           │
│                                                             │
│  用户 ──→ Supervisor Agent                                   │
│              │                                               │
│              │ 分解任务                                      │
│              ▼                                               │
│    ┌─────────┼─────────┐                                    │
│    ▼         ▼         ▼                                    │
│ ┌──────┐ ┌──────┐ ┌──────┐                                 │
│ │日 程 │ │ 邮 件 │ │文 档 │  ← 子 Agent                    │
│ │Agent │ │Agent │ │Agent │                                 │
│ └──────┘ └──────┘ └──────┘                                 │
│    │         │         │                                    │
│    └─────────┼─────────┘                                    │
│              ▼                                               │
│        汇总结果                                             │
│              │                                               │
│              ▼                                               │
│  用户 ←─ Supervisor Agent → 返回                             │
└─────────────────────────────────────────────────────────────┘
```

### 特点

- ✅ **集中式控制**：统一调度，主 Agent 决定何时调用谁
- ✅ **上下文隔离**：子 Agent 无状态，避免上下文污染
- ✅ **并行执行**：支持同时调用多个子 Agent
- ✅ **可组合性**：子 Agent 返回结果供主 Agent 使用
- ❌ **间接交互**：子 Agent 不直接与用户对话

### 基本实现

```python
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

# ============================================================================
# 步骤 1: 定义底层 API 工具
# ============================================================================

@tool
def create_calendar_event(
    title: str,
    start_time: str,    # ISO 格式: "2024-01-15T14:00:00"
    end_time: str,      # ISO 格式: "2024-01-15T15:00:00"
    attendees: list[str],
    location: str = ""
) -> str:
    """创建日历事件。需要精确的 ISO 日期时间格式。"""
    return f"事件已创建: {title}，从 {start_time} 到 {end_time}"

@tool
def send_email(
    to: list[str],
    subject: str,
    body: str,
    cc: list[str] = []
) -> str:
    """通过邮件 API 发送邮件。"""
    return f"邮件已发送至 {', '.join(to)}"

@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,
    duration_minutes: int
) -> list[str]:
    """检查指定日期参会者的可用时间段。"""
    return ["09:00", "14:00", "16:00"]

# ============================================================================
# 步骤 2: 创建专门的子 Agent
# ============================================================================

llm = init_chat_model("gpt-4o")

# 日程 Agent
calendar_agent = create_agent(
    llm,
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=(
        "你是一个日程安排助手。"
        "将自然语言的日程请求解析为正确的 ISO 日期时间格式。"
        "需要时使用 get_available_time_slots 检查可用性。"
        "使用 create_calendar_event 安排事件。"
    ),
)

# 邮件 Agent
email_agent = create_agent(
    llm,
    tools=[send_email],
    system_prompt=(
        "你是一个邮件助手。"
        "根据自然语言请求撰写专业邮件。"
        "使用 send_email 发送邮件。"
    ),
)

# ============================================================================
# 步骤 3: 将子 Agent 包装为工具
# ============================================================================

@tool
def schedule_event(request: str) -> str:
    """使用自然语言安排日历事件。

    当用户想要创建、修改或查看日历预约时使用此工具。
    处理日期/时间解析、可用性检查和事件创建。

    输入: 自然语言日程请求（如'下周二下午2点和设计团队开会'）
    """
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].content

@tool
def manage_email(request: str) -> str:
    """使用自然语言发送邮件。

    当用户想要发送通知、提醒或任何邮件通信时使用此工具。
    处理收件人提取、主题生成和邮件撰写。

    输入: 自然语言邮件请求（如'给他们发一个关于会议的提醒'）
    """
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].content

# ============================================================================
# 步骤 4: 创建主管 Agent
# ============================================================================

SUPERVISOR_PROMPT = (
    "你是一个有用的个人助手。"
    "你可以安排日历事件和发送邮件。"
    "将用户请求分解为适当的工具调用并协调结果。"
    "当请求涉及多个操作时，按顺序使用多个工具。"
)

supervisor_agent = create_agent(
    llm,
    tools=[schedule_event, manage_email],
    system_prompt=SUPERVISOR_PROMPT,
)

# ============================================================================
# 步骤 5: 使用示例
# ============================================================================

query = (
    "安排下周二下午2点和设计团队开1小时的会，"
    "然后给他们发一封提醒邮件，让他们审核新的原型设计。"
)

for step in supervisor_agent.stream(
    {"messages": [{"role": "user", "content": query}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            message.pretty_print()
```

### 添加人工审核

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

supervisor_agent = create_agent(
    llm,
    tools=[schedule_event, manage_email],
    system_prompt=SUPERVISOR_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "schedule_event": {
                    "allow_accept": True,
                    "allow_edit": True,
                    "allow_respond": True,
                },
                "manage_email": {
                    "allow_accept": True,
                    "allow_edit": True,
                    "allow_respond": True,
                },
            }
        ),
    ],
    checkpointer=InMemorySaver(),
)

# 使用时需要人工确认
config = {"configurable": {"thread_id": "session-123"}}

result = supervisor_agent.invoke(
    {"messages": [{"role": "user", "content": query}]},
    config=config
)

# 检查是否需要人工确认
if result.get("__interrupt__"):
    # 用户确认后恢复
    from langgraph.types import Command
    result = supervisor_agent.invoke(
        Command(resume={"decisions": [{"type": "accept"}]}),
        config=config
    )
```

### 适用场景

- ✅ 多领域、多工具，需要统一调度
- ✅ 子 Agent 不需要与用户直接对话
- ✅ 需要并行处理多个子任务
- ✅ 需要集中式工作流控制

### 设计决策

**1) 同步 vs 异步**
- 同步：主 Agent 等待子 Agent 结果
- 异步：后台执行，需要 job_id、status 工具

**2) 工具模式**
- 一个子 Agent 对应一个工具：灵活可控
- 单一 dispatch 工具：更易扩展，定制能力弱

**3) 上下文工程**
- 子 Agent 只看到必要上下文
- 输出要能被主 Agent 直接使用
- 可用 ToolRuntime 读取主 Agent state

**4) 子 Agent 发现**
- 对小规模：系统 prompt 列出可用子 Agent
- 对大规模：提供 list_agents / search_agents 工具

---

## Handoffs（交接模式）

### 概述

**Handoffs Pattern** 允许 Agent 直接将控制权转移给另一个 Agent。

```
┌─────────────────────────────────────────────────────────────┐
│                     Handoffs 流程                            │
│                                                             │
│  用户 ──→ Agent A (销售)                                     │
│              │                                               │
│              │ 发现需要技术支持                               │
│              ▼                                               │
│           转移到 Agent B                                     │
│              │                                               │
│              ▼                                               │
│  用户 ←─ Agent B (支持) ──→ 用户                            │
│              │                                               │
│              │ 需要处理付款                                   │
│              ▼                                               │
│           转移回 Agent A                                     │
│              │                                               │
│              ▼                                               │
│  用户 ──→ Agent A (销售) → 完成                              │
└─────────────────────────────────────────────────────────────┘
```

### 特点

- ✅ **分散式控制**：Agent 自主决策转移
- ✅ **直接交互**：用户与活跃 Agent 直接对话
- ✅ **连贯体验**：对话自然流畅
- ✅ **专业化**：专家 Agent 接管对话
- ❌ **状态复杂**：需要管理活跃 Agent 状态

### 基本实现

```python
from typing import Literal
from langchain.agents import AgentState, create_agent
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing_extensions import NotRequired

# 1. 定义共享状态
class MultiAgentState(AgentState):
    """多智能体系统的共享状态"""
    active_agent: NotRequired[str]  # 当前活跃的 Agent

# 2. 创建交接工具
@tool
def transfer_to_sales(runtime: ToolRuntime) -> Command:
    """转移到销售 Agent

    当用户询问产品定价、购买相关问题时使用此工具。
    """
    # 获取最后一条 AI 消息（保留上下文）
    last_ai_message = next(
        msg for msg in reversed(runtime.state["messages"])
        if isinstance(msg, AIMessage)
    )

    # 创建转移消息
    transfer_message = ToolMessage(
        content="已从支持 Agent 转移到销售 Agent",
        tool_call_id=runtime.tool_call_id,
    )

    # 返回 Command，指定跳转目标和状态更新
    return Command(
        goto="sales_agent",           # 跳转到销售 Agent 节点
        update={
            "active_agent": "sales_agent",
            "messages": [last_ai_message, transfer_message],  # 保留消息上下文
        },
        graph=Command.PARENT,        # 在父图层面执行跳转
    )

@tool
def transfer_to_support(runtime: ToolRuntime) -> Command:
    """转移到支持 Agent

    当用户遇到技术问题、产品使用相关问题时使用此工具。
    """
    last_ai_message = next(
        msg for msg in reversed(runtime.state["messages"])
        if isinstance(msg, AIMessage)
    )

    transfer_message = ToolMessage(
        content="已从销售 Agent 转移到支持 Agent",
        tool_call_id=runtime.tool_call_id,
    )

    return Command(
        goto="support_agent",
        update={
            "active_agent": "support_agent",
            "messages": [last_ai_message, transfer_message],
        },
        graph=Command.PARENT,
    )

# 3. 创建专门的 Agent
sales_agent = create_agent(
    model="gpt-4o",
    tools=[transfer_to_support],
    system_prompt=(
        "你是一个销售专家，帮助用户了解产品、定价和购买流程。"
        "当用户询问技术问题时，使用 transfer_to_support 工具转移。"
    ),
)

support_agent = create_agent(
    model="gpt-4o",
    tools=[transfer_to_sales],
    system_prompt=(
        "你是一个技术支持专家，帮助用户解决产品使用问题。"
        "当用户询问定价或购买相关问题时，使用 transfer_to_sales 工具转移。"
    ),
)

# 4. 创建 Agent 节点
def call_sales_agent(state: MultiAgentState):
    """调用销售 Agent 的节点函数"""
    return sales_agent.invoke(state)

def call_support_agent(state: MultiAgentState):
    """调用支持 Agent 的节点函数"""
    return support_agent.invoke(state)

# 5. 创建路由函数
def route_after_agent(
    state: MultiAgentState,
) -> Literal["sales_agent", "support_agent", "__end__"]:
    """Agent 执行后的路由逻辑"""
    messages = state.get("messages", [])

    # 检查最后一条消息
    if messages:
        last_msg = messages[-1]
        # 如果是 AI 消息且没有工具调用，表示完成
        if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
            return "__end__"

    # 否则，继续使用活跃的 Agent
    active = state.get("active_agent", "sales_agent")
    return active if active else "sales_agent"

def route_initial(state: MultiAgentState) -> Literal["sales_agent", "support_agent"]:
    """初始路由：根据状态中的 active_agent 决定起始 Agent"""
    return state.get("active_agent") or "sales_agent"

# 6. 构建图
builder = StateGraph(MultiAgentState)
builder.add_node("sales_agent", call_sales_agent)
builder.add_node("support_agent", call_support_agent)

# 初始条件路由
builder.add_conditional_edges(START, route_initial, ["sales_agent", "support_agent"])

# Agent 后续路由
builder.add_conditional_edges(
    "sales_agent",
    route_after_agent,
    ["sales_agent", "support_agent", END]
)
builder.add_conditional_edges(
    "support_agent",
    route_after_agent,
    ["sales_agent", "support_agent", END]
)

# 编译图
graph = builder.compile()

# 7. 使用
result = graph.invoke({
    "messages": [{
        "role": "user",
        "content": "你好，我的手机屏幕碎了，请问保修期内可以免费维修吗？"
    }]
})

for msg in result["messages"]:
    msg.pretty_print()
```

### 客户支持工作流

更复杂的多步骤工作流：

```python
from typing import Literal
from langchain.agents import AgentState, create_agent
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing_extensions import NotRequired

# 1. 定义工作流状态
class SupportState(AgentState):
    """客户支持工作流状态"""
    current_step: NotRequired[Literal[
        "warranty_collector",   # 保修信息收集
        "issue_classifier",     # 问题分类
        "resolution_specialist", # 解决方案专家
    ]]
    warranty_status: NotRequired[Literal["in_warranty", "out_of_warranty"]]
    issue_type: NotRequired[Literal["hardware", "software"]]

# 2. 创建工作流转移工具
@tool
def record_warranty_status(
    status: Literal["in_warranty", "out_of_warranty"],
    runtime: ToolRuntime,
) -> Command:
    """记录保修状态并转移到问题分类"""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"保修状态已记录: {status}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "warranty_status": status,
            "current_step": "issue_classifier",
        }
    )

@tool
def record_issue_type(
    issue_type: Literal["hardware", "software"],
    runtime: ToolRuntime,
) -> Command:
    """记录问题类型并转移到解决方案专家"""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"问题类型已记录: {issue_type}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "issue_type": issue_type,
            "current_step": "resolution_specialist",
        }
    )

@tool
def escalate_to_human(reason: str) -> str:
    """升级到人工客服"""
    return f"正在转接到人工客服，原因: {reason}"

@tool
def provide_solution(solution: str) -> str:
    """提供解决方案"""
    return f"解决方案: {solution}"

# 3. 创建各步骤的 Agent
warranty_collector = create_agent(
    model="gpt-4o",
    tools=[record_warranty_status],
    system_prompt=(
        "你是保修信息收集专员。"
        "询问用户产品的保修状态。"
        "确认后使用 record_warranty_status 工具记录状态。"
    ),
)

issue_classifier = create_agent(
    model="gpt-4o",
    tools=[record_issue_type],
    system_prompt=(
        "你是问题分类专家。"
        "根据用户的描述，判断问题是硬件问题还是软件问题。"
        "使用 record_issue_type 工具记录问题类型。"
    ),
)

resolution_specialist = create_agent(
    model="gpt-4o",
    tools=[escalate_to_human, provide_solution],
    system_prompt=(
        "你是解决方案专家。"
        "根据保修状态和问题类型，提供相应的解决方案。"
        "如果问题复杂，使用 escalate_to_human 转人工。"
    ),
)

# 4. 构建工作流图
builder = StateGraph(SupportState)
builder.add_node("warranty_collector", lambda s: warranty_collector.invoke(s))
builder.add_node("issue_classifier", lambda s: issue_classifier.invoke(s))
builder.add_node("resolution_specialist", lambda s: resolution_specialist.invoke(s))

def route_by_step(state: SupportState) -> str:
    """根据当前步骤路由"""
    return state.get("current_step", "warranty_collector")

builder.add_conditional_edges(START, lambda s: "warranty_collector", ["warranty_collector"])
builder.add_conditional_edges("warranty_collector", route_by_step, ["issue_classifier"])
builder.add_conditional_edges("issue_classifier", route_by_step, ["resolution_specialist"])
builder.add_conditional_edges("resolution_specialist", route_by_step, [END])

agent = builder.compile()
```

### 适用场景

- ✅ 多领域对话，需要专家 Agent 接管
- ✅ 顺序对话流程
- ✅ 用户需要与活跃 Agent 直接交互
- ✅ 需要专业的领域知识

### 关键细节

- **ToolMessage 必须与 tool_call_id 配对**
- 跨 Agent 时只传关键消息，避免上下文污染
- 使用 `Command.PARENT` 在父图层面跳转
- 结束时确保最后一条消息是 `AIMessage`

---

## Skills（技能加载模式）

### 概述

**Skills Pattern** 允许 Agent 按需加载特定技能，避免一次性加载所有上下文。

```
┌─────────────────────────────────────────────────────────────┐
│                   Skills Loading 流程                        │
│                                                             │
│  用户 ──→ Agent                                             │
│              │                                               │
│              │ 发现需要特定技能                               │
│              ▼                                               │
│         load_skill                                         │
│              │                                               │
│              ▼                                               │
│         注入技能内容                                        │
│              │                                               │
│              ▼                                               │
│         执行任务                                            │
└─────────────────────────────────────────────────────────────┘
```

### 特点

- ✅ **按需加载**：避免一次性加载所有上下文
- ✅ **渐进式暴露**：类似"检索提示词"
- ✅ **支持大量技能**：适合大规模技能集
- ✅ **单线程对话**：保持对话连贯性

### 基本实现

```python
from langchain.tools import tool
from langchain.agents import create_agent

# 定义技能库
SKILLS = {
    "sales_analytics": {
        "name": "销售分析",
        "description": "分析销售数据、生成报告、识别趋势",
        "schema": "包含 orders、customers、revenues 表",
    },
    "inventory_management": {
        "name": "库存管理",
        "description": "管理库存水平、补货建议、库存优化",
        "schema": "包含 products、inventory、suppliers 表",
    },
}

@tool
def load_skill(skill_name: str) -> str:
    """加载特定技能的详细内容

    使用此工具获取特定领域的完整知识、schema 和业务规则。

    Args:
        skill_name: 技能名称，如 'sales_analytics' 或 'inventory_management'
    """
    if skill_name not in SKILLS:
        return f"错误：未知技能 '{skill_name}'。可用技能: {list(SKILLS.keys())}"

    skill = SKILLS[skill_name]
    return f"""技能: {skill['name']}

描述: {skill['description']}

数据库 Schema:
{skill['schema']}

业务规则:
- 使用标准 SQL 语法
- 日期使用 ISO 格式
- 货币金额保留两位小数
"""

# 创建 Agent
agent = create_agent(
    model="gpt-4o",
    tools=[load_skill],
    system_prompt="""你是一个 SQL 分析助手。

你可以访问以下技能：
- sales_analytics: 销售分析
- inventory_management: 库存管理

当需要处理特定领域的请求时，使用 load_skill 工具加载该领域的详细信息。
""",
)

# 使用
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "帮我分析上个月的销售额趋势"
    }]
})
```

### 进阶：强约束

```python
from langchain.tools import tool, ToolRuntime
from typing import Literal

# 定义支持的技能
SUPPORTED_SKILLS = ["sales_analytics", "inventory_management"]

@tool
def write_sql_query(
    query: str,
    vertical: Literal["sales_analytics", "inventory_management"],
    runtime: ToolRuntime,
) -> str:
    """编写并验证 SQL 查询

    此工具帮助格式化和验证 SQL 查询。
    你必须先加载相应的技能以了解数据库 schema。

    Args:
        query: SQL 查询语句
        vertical: 业务领域（sales_analytics 或 inventory_management）
    """
    skills_loaded = runtime.state.get("skills_loaded", [])

    if vertical not in skills_loaded:
        return (
            f"错误：你必须先加载 '{vertical}' 技能以了解数据库 schema。"
            f"使用 load_skill('{vertical}') 加载 schema。"
        )

    return (
        f"SQL 查询已验证 ({vertical}):\n\n"
        f"```sql\n{query}\n```\n\n"
        f"✓ 查询已验证，准备执行。"
    )

# 创建带状态管理的 Agent
class AgentState(AgentState):
    skills_loaded: list[str]

agent = create_agent(
    model="gpt-4o",
    tools=[load_skill, write_sql_query],
    state_schema=AgentState,
)
```

### 自定义中间件

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import SystemMessage
from typing import Callable

class SkillMiddleware(AgentMiddleware):
    """动态注入技能描述的中间件"""

    # 注册 load_skill 工具
    tools = [load_skill]

    def __init__(self, skills: dict):
        super().__init__()
        # 构建技能提示词
        skills_list = []
        for name, info in skills.items():
            skills_list.append(f"- **{name}**: {info['description']}")
        self.skills_prompt = "\n".join(skills_list)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """注入技能描述到系统提示词"""
        skills_addendum = (
            f"\n\n## 可用技能\n\n{self.skills_prompt}\n\n"
            "使用 load_skill 工具获取特定技能的详细信息。"
        )

        # 追加到系统消息
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        new_system_message = SystemMessage(content=new_content)

        modified_request = request.override(system_message=new_system_message)
        return handler(modified_request)

# 使用
SKILLS = {
    "sales_analytics": {"description": "分析销售数据"},
    "inventory_management": {"description": "管理库存"},
}

agent = create_agent(
    model="gpt-4o",
    tools=[load_skill],
    middleware=[SkillMiddleware(SKILLS)],
)
```

### 适用场景

- ✅ 大量知识/技能场景
- ✅ 需要渐进式暴露能力
- ✅ 避免上下文膨胀
- ✅ 类似"按需文档暴露"的理念

---

## Router（路由模式）

### 概述

**Router Pattern** 根据请求类型将查询分发到专门的 Agent。

```
┌─────────────────────────────────────────────────────────────┐
│                      Router 架构                             │
│                                                             │
│  用户 ──→ Router                                            │
│              │                                               │
│              │ 分析请求类型                                  │
│              ▼                                               │
│    ┌─────────┼─────────┬─────────┐                          │
│    ▼         ▼         ▼         ▼                          │
│ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                       │
│ │销 售 │ │支 持 │ │财 务 │ │技 术 │  ← 专家 Agent           │
│ │Agent │ │Agent │ │Agent │ │Agent │                       │
│ └──────┘ └──────┘ └──────┘ └──────┘                       │
│    │         │         │         │                          │
│    └─────────┼─────────┴─────────┘                          │
│              ▼                                               │
│        返回结果                                             │
└─────────────────────────────────────────────────────────────┘
```

### 特点

- ✅ **并行处理**：可同时查询多个领域
- ✅ **独立运行**：各 Agent 互不影响
- ✅ **高效路由**：一次性分类决定分发
- ❌ **分类准确性**：依赖路由器分类质量

### 基本实现

```python
from typing import Literal
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# 1. 定义专门的 Agent
research_agent = create_agent(
    model="gpt-4o",
    tools=[],
    system_prompt="你是一个研究专家，提供详细的信息和分析。"
)

code_agent = create_agent(
    model="gpt-4o",
    tools=[],
    system_prompt="你是一个编程专家，帮助解决代码问题。"
)

writing_agent = create_agent(
    model="gpt-4o",
    tools=[],
    system_prompt="你是一个写作专家，帮助改进文本质量。"
)

# 2. 路由逻辑
def classify_query(state) -> Literal["research", "code", "writing"]:
    """分类用户查询"""
    last_message = state["messages"][-1]
    query = last_message.content.lower()

    if any(kw in query for kw in ["code", "programming", "function", "bug"]):
        return "code"
    elif any(kw in query for kw in ["write", "edit", "improve", "grammar"]):
        return "writing"
    else:
        return "research"

# 3. Agent 节点
def call_research(state):
    return research_agent.invoke(state)

def call_code(state):
    return code_agent.invoke(state)

def call_writing(state):
    return writing_agent.invoke(state)

# 4. 构建路由图
builder = StateGraph(AgentState)
builder.add_node("research", call_research)
builder.add_node("code", call_code)
builder.add_node("writing", call_writing)

# 路由到对应 Agent
builder.add_conditional_edges(
    START,
    classify_query,
    ["research", "code", "writing"]
)

# 每个 Agent 完成后结束
for node in ["research", "code", "writing"]:
    builder.add_edge(node, END)

router_agent = builder.compile()

# 使用
result = router_agent.invoke({
    "messages": [{"role": "user", "content": "如何用 Python 实现快速排序？"}]
})
```

### 并行路由（Send）

```python
from langgraph.types import Send

def classify_and_route(state):
    """分类并并行路由"""
    query = state["messages"][-1].content

    # 分类结果
    classifications = [
        {"source": "github", "query": query},
        {"source": "notion", "query": query},
        {"source": "slack", "query": query},
    ]

    # 使用 Send 并行派发
    return [
        Send(c["source"], {"messages": [{"role": "user", "content": c["query"]}]})
        for c in classifications
    ]

# 构建并行路由图
builder = StateGraph(AgentState)
builder.add_node("github", lambda s: github_agent.invoke(s))
builder.add_node("notion", lambda s: notion_agent.invoke(s))
builder.add_node("slack", lambda s: slack_agent.invoke(s))
builder.add_node("summarize", summarize_results)

builder.add_conditional_edges(START, classify_and_route, ["github", "notion", "slack"])

# 每个 Agent 完成后到汇总
for node in ["github", "notion", "slack"]:
    builder.add_edge(node, "summarize")

builder.add_edge("summarize", END)

parallel_router = builder.compile()
```

### 适用场景

- ✅ 多垂直领域查询
- ✅ 可并行的独立任务
- ✅ 各领域互不干扰
- ❌ 需要多轮对话的场景

### Stateless vs Stateful

| 类型 | 说明 | 适用场景 |
|------|------|----------|
| **Stateless** | 每次独立分类、并行调用 | 一次性查询 |
| **Stateful** | 需持久化历史 | 多轮对话（需包装成工具） |

---

## Custom Workflow（自定义工作流）

### 概述

**Custom Workflow** 使用 LangGraph 构建完全自定义的流程图。

```
┌─────────────────────────────────────────────────────────────┐
│                   Custom Workflow 示例                         │
│                                                             │
│    ┌──────────┐                                             │
│    │  START   │                                             │
│    └────┬─────┘                                             │
│         │                                                   │
│         ▼                                                   │
│    ┌──────────┐     ┌──────────┐     ┌──────────┐           │
│    │   检索   │ ───→│   Agent  │ ───→│  总结    │           │
│    └──────────┘     └──────────┘     └──────────┘           │
│         │                                    │               │
│         └────────────────────────────────────┘               │
│                         │                                   │
│                         ▼                                   │
│                    ┌──────────┐                               │
│                    │   END    │                               │
│                    └──────────┘                               │
└─────────────────────────────────────────────────────────────┘
```

### 特点

- ✅ **完全控制**：自定义每个步骤的逻辑
- ✅ **复杂分支**：支持条件、循环、并行
- ✅ **组合模式**：可嵌入其他 Multi-Agent 模式
- ❌ **复杂度高**：需要设计和维护完整流程图

### 示例：RAG + Agent

```python
from langgraph.graph import StateGraph, START, END

# 1. 定义各步骤节点
def rewrite_query(state):
    """重写查询以提高检索质量"""
    original_query = state["messages"][-1].content
    rewritten = query_rewriter.invoke(original_query)
    return {"messages": [rewritten]}

def retrieve_docs(state):
    """检索相关文档"""
    query = state["messages"][-1].content
    docs = vector_store.search(query, k=5)
    return {"documents": docs}

def generate_answer(state):
    """基于文档生成答案"""
    agent_response = rag_agent.invoke(state)
    return agent_response

# 2. 构建图
builder = StateGraph(AgentState)
builder.add_node("rewrite", rewrite_query)
builder.add_node("retrieve", retrieve_docs)
builder.add_node("generate", generate_answer)

# 定义流程
builder.add_edge(START, "rewrite")
builder.add_edge("rewrite", "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

rag_workflow = builder.compile()
```

### 混合模式

```python
# 在自定义工作流中使用其他 Multi-Agent 模式

builder = StateGraph(AgentState)

# 添加 Supervisor 模式作为节点
builder.add_node("supervisor", supervisor_node)

# 添加 Handoffs 模式作为子图
builder.add_subgraph("handoffs", handoff_graph)

# 添加 Router 模式作为节点
builder.add_node("router", router_node)

# 自定义路由逻辑
builder.add_conditional_edges(START, route_by_intent, [
    "supervisor", "handoffs", "router"
])

custom_workflow = builder.compile()
```

---

## 性能与成本分析

### 关键指标

| 指标 | 说明 | 影响 |
|------|------|------|
| **Model calls** | 模型调用次数 | 延迟、成本 |
| **Tokens processed** | 总 token 处理量 | 成本 |
| **Latency** | 响应延迟 | 用户体验 |
| **Parallelism** | 并行能力 | 总时延 |

### 性能对比

**场景 1：一次性请求（One-shot）**

| 模式 | 调用次数 | 相对成本 |
|------|----------|----------|
| Subagents | 4 | 中 |
| Handoffs | 3 | 低 |
| Skills | 3 | 低 |
| Router | 3 | 低 |

**场景 2：重复请求（Repeat）**

| 模式 | 调用次数 | Token 效率 |
|------|----------|-----------|
| Subagents | 8 (4+4) | 中 |
| Handoffs | 5 (3+2) | 高 |
| Skills | 5 (3+2) | 高 |
| Router | 6 (3+3) | 中 |

**场景 3：多领域任务（Multi-domain）**

| 模式 | 调用次数 | Tokens | 成本 |
|------|----------|--------|------|
| Subagents | 5 | ~9K | 低 |
| Handoffs | 7+ | ~14K+ | 高 |
| Skills | 3 | ~15K | 高 |
| Router | 5 | ~9K | 低 |

### 结论

- **并行模式（Subagents/Router）**：最适合多领域任务，总时延更低
- **状态型模式（Handoffs/Skills）**：重复任务中更省调用，但上下文膨胀更快
- **Skills**：调用次数少但 token 成本高，适合真正大规模技能集

---

## 选型指南

### 快速决策树

```
需要多智能体？
    │
    ├─→ 否 → 使用单 Agent + 工具/提示词工程
    │
    └─→ 是 → 继续判断
              │
              ├─→ 需要多领域并行查询？
              │       ├─→ 是 → Router 或 Subagents
              │       └─→ 否 → 继续判断
              │
              ├─→ 需要严格顺序流程 + 用户对话？
              │       ├─→ 是 → Handoffs
              │       └─→ 否 → 继续判断
              │
              ├─→ 有大量技能/知识按需加载？
              │       ├─→ 是 → Skills
              │       └─→ 否 → 继续判断
              │
              └─→ 流程复杂且非线性？
                      ├─→ 是 → Custom Workflow
                      └─→ 否 → Subagents
```

### 模式选择表

| 需求 | 推荐模式 | 替代方案 |
|------|----------|----------|
| 多域并行 | Subagents / Router | - |
| 顺序对话流程 | Handoffs | Custom Workflow |
| 大量技能集 | Skills | Subagents（分域） |
| 集中调度 | Subagents | Router + 汇总 |
| 用户直接交互 | Handoffs | - |
| 复杂工作流 | Custom Workflow | 混合模式 |

### 混合模式

实际应用中，可以组合多种模式：

```python
# Supervisor 调用 Router 处理并行检索
supervisor_tools = [
    parallel_search_router,  # Router 模式
    schedule_event,          # 直接工具
]

supervisor = create_agent(model, tools=supervisor_tools)

# Handoffs 某一步用 Skills 加载知识
handoff_step_tools = [
    load_skill,              # Skills 模式
    transfer_to_next,        # Handoffs 转移
]
```

---

## 实战案例

### 案例 1：完整的多领域客服系统

```python
from typing import Literal
from langchain.agents import AgentState, create_agent
from langchain.tools import tool, ToolRuntime
from langchain.messages import AIMessage, ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import NotRequired

# 1. 定义状态
class CustomerServiceState(AgentState):
    active_agent: NotRequired[str]
    customer_id: NotRequired[str]
    issue_category: NotRequired[str]

# 2. 创建交接工具
@tool
def transfer_to_billing(runtime: ToolRuntime) -> Command:
    """转移到账单专家"""
    last_ai = next(
        m for m in reversed(runtime.state["messages"])
        if isinstance(m, AIMessage)
    )
    return Command(
        goto="billing_agent",
        update={
            "active_agent": "billing_agent",
            "messages": [last_ai, ToolMessage(
                content="已转接到账单专家",
                tool_call_id=runtime.tool_call_id
            )],
        },
        graph=Command.PARENT,
    )

@tool
def transfer_to_technical(runtime: ToolRuntime) -> Command:
    """转移到技术支持"""
    last_ai = next(
        m for m in reversed(runtime.state["messages"])
        if isinstance(m, AIMessage)
    )
    return Command(
        goto="technical_agent",
        update={
            "active_agent": "technical_agent",
            "messages": [last_ai, ToolMessage(
                content="已转接到技术支持",
                tool_call_id=runtime.tool_call_id
            )],
        },
        graph=Command.PARENT,
    )

# 3. 创建专门工具
@tool
def check_invoice(invoice_id: str) -> str:
    """查询发票详情"""
    return f"发票 {invoice_id}: 金额 $299.00，状态已支付"

@tool
def troubleshoot_device(device: str, issue: str) -> str:
    """设备故障排查"""
    return f"对于 {device} 的 {issue} 问题，建议尝试重启设备"

# 4. 创建 Agent
billing_agent = create_agent(
    model="gpt-4o",
    tools=[check_invoice, transfer_to_technical],
    system_prompt="你是账单专家..."
)

technical_agent = create_agent(
    model="gpt-4o",
    tools=[troubleshoot_device, transfer_to_billing],
    system_prompt="你是技术支持专家..."
)

# 5. 构建图
builder = StateGraph(CustomerServiceState)
builder.add_node("billing_agent", lambda s: billing_agent.invoke(s))
builder.add_node("technical_agent", lambda s: technical_agent.invoke(s))

def route_after_agent(state):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and not last.tool_calls:
        return "__end__"
    return state.get("active_agent", "billing_agent")

builder.add_conditional_edges(
    START,
    lambda s: s.get("active_agent") or "billing_agent",
    ["billing_agent", "technical_agent"]
)

builder.add_conditional_edges(
    "billing_agent",
    route_after_agent,
    ["billing_agent", "technical_agent", "__end__"]
)

builder.add_conditional_edges(
    "technical_agent",
    route_after_agent,
    ["billing_agent", "technical_agent", "__end__"]
)

agent = builder.compile(checkpointer=InMemorySaver())

# 6. 使用
config = {"configurable": {"thread_id": "customer-123"}}

result = agent.invoke(
    {"messages": [HumanMessage("我的发票 #12345 金额是多少？")]},
    config=config
)
```

### 案例 2：智能文档分析系统

```python
# Supervisor + Skills 组合

from langchain.agents import create_agent
from langchain.tools import tool

# 技能定义
ANALYSIS_SKILLS = {
    "financial": {
        "description": "财务文档分析",
        "prompts": "...",
        "tools": [extract_financial_data, calculate_ratios],
    },
    "legal": {
        "description": "法律文档分析",
        "prompts": "...",
        "tools": [extract_clauses, identify_risks],
    },
}

# 技能加载工具
@tool
def load_analysis_skill(domain: str) -> str:
    """加载特定领域的分析技能"""
    skill = ANALYSIS_SKILLS.get(domain)
    if not skill:
        return f"错误：未知领域 '{domain}'"
    return f"""技能: {skill['description']}

{skill['prompts']}
"""

# 文档处理工具
@tool
def extract_text(document_path: str) -> str:
    """从文档中提取文本"""
    ...

# 创建主管 Agent
supervisor = create_agent(
    model="gpt-4o",
    tools=[load_analysis_skill, extract_text],
    system_prompt=(
        "你是文档分析主管。"
        "根据文档类型加载相应的分析技能。"
        "使用 extract_text 提取文档内容。"
    ),
)
```

---

## 最佳实践

### 1. 上下文边界管理

```python
# ✅ 好的做法：明确的上下文边界
sub_agent_prompt = (
    "你只能看到以下信息：\n"
    "- 用户当前问题\n"
    "- 相关的工具结果\n"
    "不要假设其他上下文。"
)

# ❌ 不好的做法：上下文不清晰
sub_agent_prompt = (
    "你可能需要各种信息，自己判断..."
)
```

### 2. 状态持久化

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

# 开发环境
checkpointer = InMemorySaver()

# 生产环境
checkpointer = PostgresSaver(connection_string="postgresql://...")

agent = create_agent(
    model,
    tools=[...],
    checkpointer=checkpointer,  # 必须配置
)
```

### 3. 长对话管理

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model,
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 6000),
            keep=("messages", 20),
        ),
    ],
    checkpointer=checkpointer,
)
```

### 4. 转移历史记录

```python
from datetime import datetime

class MultiAgentState(AgentState):
    handoff_history: NotRequired[list[dict]]

@tool
def transfer_with_log(runtime: ToolRuntime) -> Command:
    """带日志记录的转移"""
    from_agent = runtime.state.get("active_agent")
    to_agent = "target_agent"

    return Command(
        goto=to_agent,
        update={
            "active_agent": to_agent,
            "handoff_history": runtime.state.get("handoff_history", []) + [{
                "from": from_agent,
                "to": to_agent,
                "timestamp": datetime.now().isoformat(),
            }]
        },
        graph=Command.PARENT,
    )
```

### 5. 防止转移循环

```python
def prevent_transfer_loops(state: MultiAgentState) -> bool:
    """防止无限转移循环"""
    chain = state.get("handoff_chain", [])

    # 如果同一个 Agent 出现超过 2 次
    if chain.count(chain[-1]) > 2:
        return True  # 阻止转移

    return False

# 在路由函数中使用
def route_with_loop_check(state):
    if prevent_transfer_loops(state):
        return "__end__"
    return route_after_agent(state)
```

### 6. 结构化输出

```python
from pydantic import BaseModel, Field

class QueryClassification(BaseModel):
    """查询分类结果"""
    domain: Literal["sales", "support", "billing"]
    confidence: float
    reasoning: str

# 在 Router 中使用
@tool
def classify_query_structured(query: str) -> QueryClassification:
    """分类用户查询（结构化输出）"""
    ...
```

---

## 快速参考

### Subagents 模式

```python
# 1. 创建子 Agent
sub_agent = create_agent(model, tools=[...])

# 2. 包装为工具
@tool
def call_sub_agent(request: str) -> str:
    result = sub_agent.invoke({"messages": [{"role": "user", "content": request}]})
    return result["messages"][-1].content

# 3. 创建主管
supervisor = create_agent(model, tools=[call_sub_agent])
```

### Handoffs 模式

```python
# 1. 定义状态
class State(AgentState):
    active_agent: NotRequired[str]

# 2. 创建转移工具
@tool
def transfer_to_x(runtime: ToolRuntime) -> Command:
    last_ai = next(
        m for m in reversed(runtime.state["messages"])
        if isinstance(m, AIMessage)
    )
    return Command(
        goto="agent_x",
        update={
            "active_agent": "agent_x",
            "messages": [last_ai, ToolMessage(...)],
        },
        graph=Command.PARENT,
    )

# 3. 构建图
builder.add_conditional_edges(START, lambda s: s.get("active_agent") or "agent_x", [...])
```

### Skills 模式

```python
# 1. 定义技能库
SKILLS = {"skill_name": {...}}

# 2. 创建加载工具
@tool
def load_skill(skill_name: str) -> str:
    return SKILLS[skill_name]

# 3. 创建 Agent
agent = create_agent(
    model,
    tools=[load_skill],
    system_prompt="你可以加载以下技能：..."
)
```

### Router 模式

```python
# 1. 分类函数
def classify_query(state) -> Literal["a", "b", "c"]:
    ...

# 2. 构建图
builder.add_conditional_edges(START, classify_query, ["a", "b", "c"])
builder.add_edge("a", END)
builder.add_edge("b", END)
builder.add_edge("c", END)
```

### Command 对象

```python
from langgraph.types import Command

# 基本跳转
Command(goto="next_agent")

# 带状态更新
Command(
    goto="next_agent",
    update={"key": "value"},
)

# 跳转到父图
Command(
    goto="other_agent",
    graph=Command.PARENT,
)
```

---

## 总结

**Multi-Agent（多智能体）** 通过专业化 Agent 协作完成复杂任务。

### 五大核心模式

| 模式 | 核心机制 | 控制流 | 适用场景 |
|------|----------|--------|----------|
| **Subagents** | 监督者调用子 Agent 工具 | 集中式 | 多域并行、集中调度 |
| **Handoffs** | 状态驱动切换 Agent | 分散式 | 顺序对话流程 |
| **Skills** | 按需加载技能 | 单 Agent | 大量技能、渐进式暴露 |
| **Router** | 路由器分类分发 | 分发式 | 多垂直领域查询 |
| **Custom Workflow** | LangGraph 自定义图 | 自由 | 复杂流程 |

### 核心要点

1. **不是堆模型**：核心是上下文工程和信息流设计
2. **选择合适的模式**：根据场景选择最合适的架构
3. **管理好状态**：使用 checkpointer 持久化状态
4. **控制上下文**：避免上下文膨胀和污染
5. **添加监控**：使用 LangSmith 观察真实 trace

### 选型决策树

```
需要多领域并行？ → Router / Subagents
    ↓ 否
需要顺序对话流程？ → Handoffs
    ↓ 否
大量技能按需加载？ → Skills
    ↓ 否
复杂非线性流程？ → Custom Workflow
    ↓ 否
使用单 Agent + 工具/提示词工程
```

通过合理使用 Multi-Agent 模式，你可以构建出更专业、更可控、更易维护的 AI 应用！
