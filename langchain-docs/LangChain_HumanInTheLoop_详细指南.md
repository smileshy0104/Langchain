# LangChain Human-in-the-Loop (人机协作) 详细指南

## 目录

1. [概述](#概述)
2. [核心概念](#核心概念)
3. [快速开始](#快速开始)
4. [决策类型](#决策类型)
5. [配置中断](#配置中断)
6. [响应中断](#响应中断)
7. [流式处理](#流式处理)
8. [执行生命周期](#执行生命周期)
9. [持久化与恢复](#持久化与恢复)
10. [多智能体集成](#多智能体集成)
11. [实战案例](#实战案例)
12. [最佳实践](#最佳实践)
13. [快速参考](#快速参考)

---

## 概述

### 什么是 Human-in-the-Loop

**Human-in-the-Loop (HITL)** 是一种人机协作模式，允许在 Agent 执行敏感操作前插入人工审核环节。当模型提议执行需要审查的操作（如写入文件、执行 SQL、发送邮件等）时，中间件会暂停执行并等待人工决策。

```
┌─────────────────────────────────────────────────────────────┐
│              Human-in-the-Loop 工作流程                        │
│                                                             │
│  用户请求                                                     │
│     │                                                       │
│     ▼                                                       │
│  Agent 规划 ──→ 生成工具调用                                   │
│     │                                                       │
│     ▼                                                       │
│  HITL 中间件                                                 │
│     │                                                       │
│     ├─→ 敏感操作？ ──→ [是] ──→ 暂停执行，等待人工决策           │
│     │                                │                      │
│     │                                ├─→ 批准 (approve)     │
│     │                                ├─→ 修改 (edit)        │
│     │                                └─→ 拒绝 (reject)     │
│     │                                │                      │
│     │                                └─→ 继续执行            │
│     │                                                       │
│     └─→ [否] ──→ 直接执行                                     │
│     │                                                       │
│     ▼                                                       │
│  返回结果                                                     │
└─────────────────────────────────────────────────────────────┘
```

### 为什么使用 HITL

| 优势 | 说明 | 典型场景 |
|------|------|----------|
| **安全控制** | 防止误执行危险操作 | 删除数据、执行 SQL |
| **合规要求** | 满足人工审批的合规需求 | 金融交易、数据导出 |
| **质量保证** | 人工审核提高输出质量 | 发送邮件、生成报告 |
| **逐步引导** | 通过反馈指导 Agent 行为 | 复杂任务、多步操作 |

### HITL 的应用场景

- **高风险操作**：数据库写入、金融交易、文件删除
- **合规工作流**：必须有人工审批的业务流程
- **长期对话**：需要人工反馈引导 Agent 的场景
- **敏感通信**：发送外部邮件、消息通知

---

## 核心概念

### 三大决策类型

```
┌─────────────────────────────────────────────────────────────┐
│                    HITL 决策类型                              │
│                                                             │
│  1. 批准 (approve) ──→ 按原样执行工具调用                      │
│     ├─ 用途：确认操作无误，直接执行                            │
│     └─ 示例：发送已审核的邮件                                 │
│                                                             │
│  2. 修改 (edit) ──→ 修改参数后执行                             │
│     ├─ 用途：调整操作参数，修正错误                            │
│     └─ 示例：修改邮件收件人、调整 SQL 查询                     │
│                                                             │
│  3. 拒绝 (reject) ──→ 拒绝执行并提供反馈                        │
│     ├─ 用途：阻止不当操作，指导正确行为                         │
│     └─ 示例：拒绝危险的删除操作，说明原因                       │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 说明 | 来源 |
|------|------|------|
| `HumanInTheLoopMiddleware` | HITL 中间件 | `langchain.agents.middleware` |
| `interrupt` | 中断原语 | `langgraph.types` |
| `Command` | 命令对象（用于恢复执行） | `langgraph.types` |
| `checkpointer` | 持久化层（必需） | `langgraph.checkpoint` |

### 中断机制

HITL 使用 LangGraph 的 `interrupt` 原语来暂停执行：

```python
# interrupt 会保存当前图状态
# 执行可以在任何时候恢复
interrupt({
    "action_requests": [...],  # 需要审核的操作
    "review_configs": [...]     # 每个操作的审核配置
})
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install langchain langgraph
```

### 2. 基本配置

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

# 定义工具
from langchain_core.tools import tool

@tool
def write_file(path: str, content: str) -> str:
    """写入文件"""
    with open(path, "w") as f:
        f.write(content)
    return f"已写入文件: {path}"

@tool
def execute_sql(query: str) -> str:
    """执行 SQL 查询"""
    # 模拟 SQL 执行
    return f"已执行: {query}"

@tool
def read_data(table: str) -> str:
    """读取数据（安全操作）"""
    return f"从 {table} 读取的数据"

# 创建带 HITL 的 Agent
agent = create_agent(
    model="gpt-4o",
    tools=[write_file, execute_sql, read_data],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # 所有敏感操作需要审核
                "write_file": True,       # 允许所有决策类型
                "execute_sql": {
                    "allowed_decisions": ["approve", "reject"],  # 不允许修改
                },
                # 安全操作自动批准
                "read_data": False,
            },
            description_prefix="工具执行需要审批",
        ),
    ],
    # HITL 必须使用 checkpointer
    checkpointer=InMemorySaver(),
)
```

### 3. 使用示例

```python
from langgraph.types import Command

# 配置会话 ID（必需）
config = {"configurable": {"thread_id": "session-123"}}

# 执行 Agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "删除所有旧记录"}]},
    config=config
)

# 检查是否需要人工审核
if "__interrupt__" in result:
    print("需要人工审核:")
    print(result["__interrupt__"])

    # 批准操作
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config
    )
```

---

## 决策类型

### Approve（批准）

直接执行工具调用，不做任何修改。

```python
# 场景：确认邮件内容无误，直接发送
agent.invoke(
    Command(resume={
        "decisions": [
            {"type": "approve"}
        ]
    }),
    config=config
)
```

**适用场景**：
- 操作完全符合预期
- 参数正确无误
- 已验证操作安全性

### Edit（修改）

修改工具调用参数后执行。

```python
# 场景：修改邮件收件人
agent.invoke(
    Command(resume={
        "decisions": [
            {
                "type": "edit",
                "edited_action": {
                    "name": "send_email",
                    "args": {
                        "to": "correct@example.com",  # 修改后的值
                        "subject": "项目更新",          # 保留原值
                        "body": "以下是项目进度...",     # 保留原值
                    }
                }
            }
        ]
    }),
    config=config
)
```

**适用场景**：
- 操作意图正确但参数有误
- 需要调整部分参数值
- 修正拼写错误或格式问题

### Reject（拒绝）

拒绝执行工具调用，并提供反馈指导 Agent。

```python
# 场景：拒绝危险的删除操作
agent.invoke(
    Command(resume={
        "decisions": [
            {
                "type": "reject",
                "message": "不能删除所有数据！请只删除超过 30 天的旧记录，并使用 WHERE 条件限制范围。"
            }
        ]
    }),
    config=config
)
```

**适用场景**：
- 操作存在安全风险
- 不符合业务规则
- 需要提供正确的操作指导

### 多决策处理

当多个操作同时需要审核时，按顺序提供决策：

```python
agent.invoke(
    Command(resume={
        "decisions": [
            {"type": "approve"},  # 第一个操作：批准
            {
                "type": "edit",   # 第二个操作：修改
                "edited_action": {
                    "name": "send_email",
                    "args": {"to": "updated@example.com"}
                }
            },
            {
                "type": "reject",  # 第三个操作：拒绝
                "message": "此操作不符合公司政策"
            }
        ]
    }),
    config=config
)
```

---

## 配置中断

### interrupt_on 配置选项

`interrupt_on` 是一个字典，映射工具名称到审核配置：

| 配置值 | 说明 | 示例 |
|--------|------|------|
| `True` | 启用所有决策类型 | `"write_file": True` |
| `False` | 自动批准，无需审核 | `"read_data": False` |
| `dict` | 自定义配置 | `{"allowed_decisions": [...]}` |

### 基本配置

```python
HumanInTheLoopMiddleware(
    interrupt_on={
        # 简洁配置：所有决策类型
        "send_email": True,

        # 禁用审核：自动批准
        "read_email": False,

        # 自定义配置
        "execute_sql": {
            "allowed_decisions": ["approve", "reject"],  # 不允许编辑
            "description": "SQL 执行需要 DBA 审批",
        }
    }
)
```

### InterruptOnConfig 选项

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

HumanInTheLoopMiddleware(
    interrupt_on={
        "delete_file": {
            # 允许的决策类型
            "allowed_decisions": ["approve", "edit", "reject"],

            # 自定义描述（静态字符串）
            "description": "危险操作：删除文件",

            # 或使用动态描述（函数）
            "description": lambda args: (
                f"即将删除文件: {args['path']}\n"
                f"此操作不可恢复！"
            ),
        }
    }
)
```

### description_prefix

为所有中断消息添加统一前缀：

```python
HumanInTheLoopMiddleware(
    interrupt_on={...},
    description_prefix="工具执行需要审批",  # 默认值
)
```

最终消息格式：
```
工具执行需要审批

工具: execute_sql
参数: {'query': 'DELETE FROM users WHERE ...'}
```

### 完整配置示例

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4o",
    tools=[write_file, execute_sql, read_data, send_email],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # 高风险操作：需要审批
                "write_file": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "写入文件操作",
                },
                "execute_sql": {
                    "allowed_decisions": ["approve", "reject"],  # SQL 不允许修改
                    "description": "SQL 执行操作",
                },
                "send_email": True,  # 邮件允许所有决策

                # 安全操作：自动批准
                "read_data": False,
            },
            description_prefix="审批流程",
        ),
    ],
    checkpointer=InMemorySaver(),
)
```

---

## 响应中断

### 检测中断

```python
config = {"configurable": {"thread_id": "session-123"}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "执行敏感操作"}]},
    config=config
)

# 检查是否有中断
if "__interrupt__" in result:
    interrupt_data = result["__interrupt__"]

    # 遍历需要审核的操作
    for action_request in interrupt_data["action_requests"]:
        print(f"工具: {action_request['name']}")
        print(f"参数: {action_request['arguments']}")
        print(f"描述: {action_request['description']}")
        print("---")

    # 查看每个操作的审核配置
    for review_config in interrupt_data["review_configs"]:
        print(f"工具: {review_config['action_name']}")
        print(f"允许的决策: {review_config['allowed_decisions']}")
```

### 中断数据结构

```python
{
    "__interrupt__": [
        {
            "value": {
                "action_requests": [
                    {
                        "name": "execute_sql",
                        "arguments": {"query": "DELETE FROM ..."},
                        "description": "工具执行需要审批\n\n工具: execute_sql\n参数: {...}"
                    }
                ],
                "review_configs": [
                    {
                        "action_name": "execute_sql",
                        "allowed_decisions": ["approve", "reject"]
                    }
                ]
            }
        }
    ]
}
```

### 响应决策

```python
from langgraph.types import Command

# 方式 1: 批准
agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config
)

# 方式 2: 修改
agent.invoke(
    Command(resume={
        "decisions": [{
            "type": "edit",
            "edited_action": {
                "name": "execute_sql",
                "args": {"query": "SELECT * FROM users"}
            }
        }]
    }),
    config=config
)

# 方式 3: 拒绝
agent.invoke(
    Command(resume={
        "decisions": [{
            "type": "reject",
            "message": "不能删除数据，请使用 SELECT 查询"
        }]
    }),
    config=config
)
```

### 审核工作流示例

```python
def review_workflow(agent, user_message, config):
    """完整的审核工作流"""
    # 第一步：执行 Agent
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config
    )

    # 第二步：检查是否需要审核
    while "__interrupt__" in result:
        interrupt = result["__interrupt__"]

        # 第三步：展示待审核操作
        print("\n=== 需要审核的操作 ===")
        for i, action in enumerate(interrupt["value"]["action_requests"]):
            print(f"\n[{i+1}] {action['name']}")
            print(f"参数: {action['arguments']}")
            print(f"描述: {action['description']}")

        # 第四步：收集决策
        decisions = []
        for i, action in enumerate(interrupt["value"]["action_requests"]):
            review_config = interrupt["value"]["review_configs"][i]
            allowed = review_config["allowed_decisions"]

            print(f"\n操作 {i+1}: {action['name']}")
            print(f"允许的决策: {allowed}")

            decision_type = input("选择决策 (approve/edit/reject): ").strip()

            if decision_type == "approve":
                decisions.append({"type": "approve"})
            elif decision_type == "edit":
                print("当前参数:", action['arguments'])
                new_args = input("输入新参数 (JSON 格式): ")
                import json
                decisions.append({
                    "type": "edit",
                    "edited_action": {
                        "name": action['name'],
                        "args": json.loads(new_args)
                    }
                })
            elif decision_type == "reject":
                message = input("拒绝原因: ")
                decisions.append({
                    "type": "reject",
                    "message": message
                })

        # 第五步：恢复执行
        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=config
        )

    # 返回最终结果
    return result
```

---

## 流式处理

### 使用 stream() 处理中断

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "session-123"}}

# 收集中断
interrupts = []

# 流式执行，收集 tokens 和中断
for mode, chunk in agent.stream(
    {"messages": [{"role": "user", "content": "执行敏感操作"}]},
    config=config,
    stream_mode=["updates", "messages"],
):
    if mode == "messages":
        # LLM token 流式输出
        token, metadata = chunk
        if hasattr(token, 'content') and token.content:
            print(token.content, end="", flush=True)

    elif mode == "updates":
        # 检查中断
        if "__interrupt__" in chunk:
            print("\n\n检测到中断:")
            interrupts.append(chunk["__interrupt__"])
            print(chunk["__interrupt__"])

# 处理中断
if interrupts:
    for interrupt in interrupts:
        decisions = collect_user_decisions(interrupt)
        # 继续流式执行
        for mode, chunk in agent.stream(
            Command(resume={"decisions": decisions}),
            config=config,
            stream_mode=["updates", "messages"],
        ):
            if mode == "messages":
                token, metadata = chunk
                if hasattr(token, 'content') and token.content:
                    print(token.content, end="", flush=True)
```

### 完整的流式处理示例

```python
async def streaming_with_hitl(agent, user_message, config):
    """带 HITL 的流式处理"""
    import asyncio

    config = {"configurable": {"thread_id": "session-123"}}

    while True:
        interrupts = []

        # 流式执行直到中断或完成
        async for mode, chunk in agent.astream(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config,
            stream_mode=["updates", "messages"],
        ):
            if mode == "messages":
                token, metadata = chunk
                if hasattr(token, 'content') and token.content:
                    print(token.content, end="", flush=True)
            elif mode == "updates":
                if "__interrupt__" in chunk:
                    print("\n\n--- 需要审核 ---")
                    interrupts.append(chunk["__interrupt__"])

        # 如果没有中断，完成
        if not interrupts:
            break

        # 收集所有决策
        all_decisions = []
        for interrupt in interrupts:
            decisions = await collect_decisions_async(interrupt)
            all_decisions.extend(decisions)

        # 用决策恢复执行
        user_message = Command(resume={"decisions": all_decisions})

async def collect_decisions_async(interrupt):
    """异步收集决策"""
    decisions = []
    for action in interrupt["value"]["action_requests"]:
        # 在实际应用中，这里会显示 UI 收集用户输入
        print(f"\n需要审核: {action['name']}")
        print(f"参数: {action['arguments']}")

        # 模拟异步等待用户输入
        await asyncio.sleep(0.1)
        decision = input("决策 (approve/edit/reject): ").strip()

        if decision == "approve":
            decisions.append({"type": "approve"})
        elif decision == "reject":
            message = input("拒绝原因: ")
            decisions.append({"type": "reject", "message": message})
        # ... edit 处理

    return decisions
```

### 流模式说明

| 流模式 | 说明 | 用途 |
|--------|------|------|
| `messages` | LLM token 流式输出 | 实时显示生成内容 |
| `updates` | 图状态更新 | 检测中断、查看进度 |
| `values` | 完整状态值 | 获取当前状态 |

---

## 执行生命周期

### HITL 中间件执行流程

```
┌─────────────────────────────────────────────────────────────┐
│              HITL 执行生命周期                                │
│                                                             │
│  1. Agent 调用模型                                          │
│     │                                                       │
│     ▼                                                       │
│  2. 模型生成响应（可能包含工具调用）                            │
│     │                                                       │
│     ▼                                                       │
│  3. after_model 钩子                                         │
│     │                                                       │
│     ├─→ 检查响应中的工具调用                                   │
│     │                                                       │
│     ├─→ 有需要审核的调用？                                     │
│     │     │                                                 │
│     │     ├─→ [是]                                          │
│     │     │     │                                           │
│     │     │     ├─→ 构建 HITLRequest                        │
│     │     │     │   - action_requests: 待审核操作            │
│     │     │     │   - review_configs: 审核配置               │
│     │     │     │                                           │
│     │     │     ├─→ 调用 interrupt()                        │
│     │     │     │   - 暂停执行                               │
│     │     │     │   - 保存状态到 checkpointer                │
│     │     │     │                                           │
│     │     │     └─→ 等待人工决策                             │
│     │     │           │                                     │
│     │     │           └─→ 收集 HITLResponse 决策            │
│     │     │                   │                             │
│     │     │                   ├─→ approve: 执行工具         │
│     │     │                   ├─→ edit: 修改后执行          │
│     │     │                   └─→ reject: 生成 ToolMessage │
│     │     │                                                 │
│     │     └─→ [否] ──→ 直接执行工具                           │
│     │                                                       │
│     ▼                                                       │
│  4. 继续执行或返回结果                                        │
└─────────────────────────────────────────────────────────────┘
```

### after_model 钩子

HITL 中间件在 `after_model` 钩子中工作：

```python
from langchain.agents.middleware import AgentMiddleware

class HumanInTheLoopMiddleware(AgentMiddleware):
    async def after_model(
        self,
        request: ModelRequest,
        response: ModelResponse,
    ) -> ModelResponse:
        """模型生成后、工具执行前"""
        # 1. 检查响应中的工具调用
        tool_calls = response.message.tool_calls

        if not tool_calls:
            return response

        # 2. 筛选需要审核的工具调用
        needs_review = []
        for call in tool_calls:
            if self.should_interrupt(call["name"]):
                needs_review.append(call)

        if not needs_review:
            return response

        # 3. 构建中断请求
        hitl_request = {
            "action_requests": [
                {
                    "name": call["name"],
                    "arguments": call["args"],
                    "description": self.get_description(call),
                }
                for call in needs_review
            ],
            "review_configs": [
                self.get_review_config(call["name"])
                for call in needs_review
            ]
        }

        # 4. 触发中断
        interrupt(hitl_request)

        return response
```

---

## 持久化与恢复

### Checkpointer 的重要性

HITL 必须使用 checkpointer 来：
1. 保存中断时的图状态
2. 支持暂停和恢复执行
3. 维护会话连续性

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import AsyncPostgresSaver

# 开发环境：内存存储
checkpointer = InMemorySaver()

# 生产环境：PostgreSQL 存储
checkpointer = AsyncPostgresSaver.from_conn_string(
    "postgresql://user:password@localhost/dbname"
)

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[HumanInTheLoopMiddleware(...)],
    checkpointer=checkpointer,  # 必需！
)
```

### Thread ID 管理

每次执行都需要唯一的 thread_id：

```python
import uuid

# 生成唯一会话 ID
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

# 执行 Agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "..."}]},
    config=config
)

# 同一会话可以恢复
result = agent.invoke(
    Command(resume={"decisions": [...]}),
    config=config  # 使用相同的 thread_id
)
```

### 查看历史状态

```python
# 获取会话的所有检查点
checkpointer.get(config)  # 获取最新状态
checkpointer.list(config)  # 列出所有历史状态

# 恢复到特定检查点
state_snapshot = checkpointer.get_tuple(config)

# 时间旅行：回到之前的检查点
for checkpoint in checkpointer.list(config):
    print(f"Checkpoint ID: {checkpoint.config['checkpoint_id']}")
    print(f"Step: {checkpoint.metadata.get('step')}")
    print(f"State: {checkpoint.values}")
```

### 生产环境配置

```python
import asyncio
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def setup_production_agent():
    """配置生产环境 Agent"""
    # PostgreSQL 连接池
    checkpointer = AsyncPostgresSaver.from_conn_string(
        "postgresql+asyncpg://user:password@localhost/langchain"
    )

    # 初始化表
    await checkpointer.setup()

    # 创建 Agent
    agent = create_agent(
        model="gpt-4o",
        tools=[...],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "write_file": True,
                    "execute_sql": {
                        "allowed_decisions": ["approve", "reject"]
                    },
                }
            )
        ],
        checkpointer=checkpointer,
    )

    return agent

# 使用
agent = asyncio.run(setup_production_agent())
```

---

## 多智能体集成

### Subagents 模式

在 Subagents 模式中，可以为子 Agent 的工具添加 HITL：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool

# 子 Agent 工具
@tool
def create_calendar_event(title: str, start: str, end: str) -> str:
    """创建日历事件"""
    return f"已创建事件: {title}"

@tool
def send_email(to: list[str], subject: str, body: str) -> str:
    """发送邮件"""
    return f"已发送邮件至 {to}"

# 创建子 Agent（带 HITL）
calendar_agent = create_agent(
    model="gpt-4o",
    tools=[create_calendar_event],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "create_calendar_event": True,  # 需要审核
            }
        )
    ],
)

email_agent = create_agent(
    model="gpt-4o",
    tools=[send_email],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True,  # 需要审核
            }
        )
    ],
)

# 包装为工具
@tool
def schedule_event(request: str) -> str:
    """安排日程"""
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].content

@tool
def manage_email(request: str) -> str:
    """发送邮件"""
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].content

# 创建主管 Agent（只需要 checkpointer）
supervisor_agent = create_agent(
    model="gpt-4o",
    tools=[schedule_event, manage_email],
    checkpointer=InMemorySaver(),  # 只在顶层添加
)
```

### 处理多级中断

```python
config = {"configurable": {"thread_id": "multi-agent-123"}}

query = (
    "安排下周二下午2点和设计团队开会，"
    "并发送邮件提醒他们审核新原型"
)

interrupts = []

# 流式执行，收集所有中断
for step in supervisor_agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    config=config,
):
    for update in step.values():
        if isinstance(update, dict):
            # 正常消息
            for message in update.get("messages", []):
                message.pretty_print()
        else:
            # 中断
            interrupt_ = update[0]
            interrupts.append(interrupt_)
            print(f"\nINTERRUPTED: {interrupt_.id}")

# 处理所有中断
for interrupt_ in interrupts:
    for request in interrupt_.value["action_requests"]:
        print(f"\n待审核: {request['name']}")
        print(f"描述: {request['description']}")

    # 收集决策
    decisions = collect_decisions_for_interrupt(interrupt_)

    # 恢复执行
    result = supervisor_agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config
    )
```

### Router 模式

在 Router 模式中，可以统一配置所有路由目标的 HITL：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

# 定义敏感工具
SENSITIVE_TOOLS = {"execute_sql", "write_file", "delete_data"}

# 为所有 Agent 添加统一的 HITL
hitl_middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        tool: True for tool in SENSITIVE_TOOLS
    }
)

# 创建各个 Agent
research_agent = create_agent(
    model="gpt-4o",
    tools=[safe_search, read_data],
)

database_agent = create_agent(
    model="gpt-4o",
    tools=[execute_sql, write_file],
    middleware=[hitl_middleware],  # 只给需要的 Agent 添加
)

email_agent = create_agent(
    model="gpt-4o",
    tools=[send_email, read_email],
    middleware=[hitl_middleware],
)
```

---

## 实战案例

### 案例 1：数据库操作审批系统

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.postgres import AsyncPostgresSaver
from langchain_core.tools import tool
import asyncio

@tool
def execute_sql(query: str) -> str:
    """执行 SQL 查询"""
    # 验证 SQL 安全性
    dangerous_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER"]
    query_upper = query.upper()

    for keyword in dangerous_keywords:
        if keyword in query_upper:
            return f"错误：检测到危险关键字 {keyword}"

    # 执行查询
    return f"已执行: {query}"

@tool
def export_data(table: str, format: str = "csv") -> str:
    """导出数据"""
    return f"已导出 {table} 为 {format} 格式"

async def create_db_agent():
    """创建数据库 Agent"""
    checkpointer = AsyncPostgresSaver.from_conn_string(
        "postgresql://user:pass@localhost/db"
    )
    await checkpointer.setup()

    agent = create_agent(
        model="gpt-4o",
        tools=[execute_sql, export_data],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "execute_sql": {
                        "allowed_decisions": ["approve", "edit"],
                        "description": lambda args: (
                            f"SQL 审批:\n"
                            f"查询: {args['query']}\n"
                            f"请确认查询正确且安全"
                        )
                    },
                    "export_data": {
                        "allowed_decisions": ["approve", "reject"],
                        "description": "数据导出审批",
                    }
                }
            )
        ],
        checkpointer=checkpointer,
    )

    return agent

# 使用
async def main():
    agent = await create_db_agent()
    config = {"configurable": {"thread_id": "db-session-001"}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "导出所有用户数据"}]},
        config=config
    )

    if "__interrupt__" in result:
        # 审批流程
        print(result["__interrupt__"])

        # 批准
        result = agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config
        )

asyncio.run(main())
```

### 案例 2：邮件发送审核系统

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool

@tool
def send_email(
    to: list[str],
    subject: str,
    body: str,
    attachments: list[str] = []
) -> str:
    """发送邮件"""
    # 验证收件人
    allowed_domains = ["company.com", "trusted-partner.com"]
    for email in to:
        domain = email.split("@")[-1]
        if domain not in allowed_domains:
            return f"错误：不允许发送到 {domain}"

    return f"已发送邮件至 {', '.join(to)}"

@tool
def schedule_meeting(
    attendees: list[str],
    title: str,
    start_time: str,
    duration_minutes: int
) -> str:
    """安排会议"""
    return f"已安排会议: {title}, 参与者: {len(attendees)} 人"

# 创建 Agent
agent = create_agent(
    model="gpt-4o",
    tools=[send_email, schedule_meeting],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": lambda args: (
                        f"邮件审核:\n"
                        f"收件人: {', '.join(args['to'])}\n"
                        f"主题: {args['subject']}\n"
                        f"正文预览: {args['body'][:100]}..."
                    )
                },
                "schedule_meeting": {
                    "allowed_decisions": ["approve", "edit"],
                    "description": "会议安排审核",
                }
            }
        )
    ],
    checkpointer=InMemorySaver(),
)

# 审核界面
def email_review_interface(agent, user_request, config):
    """邮件审核界面"""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_request}]},
        config=config
    )

    while "__interrupt__" in result:
        interrupt = result["__interrupt__"]

        print("\n" + "="*60)
        print("待审核操作")
        print("="*60)

        for i, action in enumerate(interrupt["value"]["action_requests"]):
            print(f"\n[{i+1}] {action['name']}")
            print(action['description'])

            # 收件人检查
            if action['name'] == 'send_email':
                recipients = action['arguments']['to']
                external = [e for e in recipients if not e.endswith("company.com")]
                if external:
                    print(f"⚠️  警告：包含外部收件人: {external}")

        print("\n" + "-"*60)

        # 收集决策
        decisions = []
        for i, action in enumerate(interrupt["value"]["action_requests"]):
            print(f"\n操作 {i+1}/{len(interrupt['value']['action_requests'])}")

            choice = input("操作 (a=批准/e=编辑/r=拒绝/v=查看详情): ").lower()

            if choice == 'a':
                decisions.append({"type": "approve"})
            elif choice == 'e':
                print("当前参数:", action['arguments'])
                if action['name'] == 'send_email':
                    new_to = input("收件人 (当前: " + str(action['arguments']['to']) + "): ")
                    if new_to:
                        action['arguments']['to'] = new_to.split(',')

                    new_subject = input(f"主题 (当前: {action['arguments']['subject']}): ")
                    if new_subject:
                        action['arguments']['subject'] = new_subject

                decisions.append({
                    "type": "edit",
                    "edited_action": {
                        "name": action['name'],
                        "args": action['arguments']
                    }
                })
            elif choice == 'r':
                reason = input("拒绝原因: ")
                decisions.append({
                    "type": "reject",
                    "message": reason
                })
            elif choice == 'v':
                # 显示详情
                import json
                print(json.dumps(action['arguments'], indent=2, ensure_ascii=False))
                # 重新询问
                decisions.extend(email_review_interface(agent, user_request, config))
                return decisions

        # 恢复执行
        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=config
        )

    return result
```

### 案例 3：金融交易审批系统

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from datetime import datetime
from typing import Literal

@tool
def transfer_funds(
    from_account: str,
    to_account: str,
    amount: float,
    currency: str = "USD"
) -> str:
    """转账"""
    # 金额限制
    if amount > 100000:
        return f"错误：单笔转账限额为 $100,000"

    return f"已从 {from_account} 转账 ${amount} {currency} 到 {to_account}"

@tool
def approve_invoice(
    invoice_id: str,
    approval_amount: float,
    notes: str = ""
) -> str:
    """审批发票"""
    return f"已审批发票 {invoice_id}，金额: ${approval_amount}"

# 创建 Agent
agent = create_agent(
    model="gpt-4o",
    tools=[transfer_funds, approve_invoice],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "transfer_funds": {
                    "allowed_decisions": ["approve", "reject"],  # 不允许修改金额
                    "description": lambda args: (
                        f"转账审批:\n"
                        f"从: {args['from_account']}\n"
                        f"到: {args['to_account']}\n"
                        f"金额: {args['amount']} {args.get('currency', 'USD')}\n"
                        f"⚠️  此操作不可撤销！"
                    )
                },
                "approve_invoice": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": lambda args: (
                        f"发票审批:\n"
                        f"发票号: {args['invoice_id']}\n"
                        f"审批金额: ${args['approval_amount']}"
                    )
                }
            }
        )
    ],
    checkpointer=InMemorySaver(),
)

# 审批工作流
def finance_approval_workflow(agent, request, config):
    """金融审批工作流"""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": request}]},
        config=config
    )

    while "__interrupt__" in result:
        interrupt = result["__interrupt__"]

        print("\n" + "="*60)
        print("金融交易审批")
        print("="*60)

        for action in interrupt["value"]["action_requests"]:
            print(f"\n操作: {action['name']}")
            print(action['description'])

            # 风险检查
            if action['name'] == 'transfer_funds':
                amount = action['arguments']['amount']
                if amount > 50000:
                    print("⚠️  高风险交易：金额超过 $50,000")
                    print("需要二级审批")

                to_account = action['arguments']['to_account']
                if to_account.startswith("EXTERNAL"):
                    print("⚠️  外部账户转账")

        print("\n" + "-"*60)

        # 收集决策
        decisions = []
        for i, action in enumerate(interrupt["value"]["action_requests"]):
            config_info = interrupt["value"]["review_configs"][i]
            allowed = config_info["allowed_decisions"]

            print(f"\n操作 {i+1}: {action['name']}")
            print(f"允许的决策: {', '.join(allowed)}")

            if "approve" in allowed:
                print("1. 批准 (approve)")
            if "edit" in allowed:
                print("2. 修改 (edit)")
            if "reject" in allowed:
                print("3. 拒绝 (reject)")

            choice = input("选择: ").strip().lower()

            if choice == "1" and "approve" in allowed:
                # 二级审批检查
                if action['name'] == 'transfer_funds' and action['arguments']['amount'] > 50000:
                    approver_2 = input("二级审批人 ID: ")
                    if not approver_2:
                        print("需要二级审批人批准")
                        continue

                decisions.append({"type": "approve"})

            elif choice == "2" and "edit" in allowed:
                if action['name'] == 'approve_invoice':
                    new_amount = input(f"新金额 (当前: {action['arguments']['approval_amount']}): ")
                    if new_amount:
                        try:
                            action['arguments']['approval_amount'] = float(new_amount)
                        except ValueError:
                            print("无效金额")
                            continue

                decisions.append({
                    "type": "edit",
                    "edited_action": {
                        "name": action['name'],
                        "args": action['arguments']
                    }
                })

            elif choice == "3" and "reject" in allowed:
                reason = input("拒绝原因: ")
                decisions.append({
                    "type": "reject",
                    "message": reason or "审批被拒绝"
                })

        # 恢复执行
        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=config
        )

    return result
```

### 案例 4：带日志和审计的 HITL

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from datetime import datetime
import json

class AuditMiddleware:
    """审计日志中间件"""

    def __init__(self, log_file="hitl_audit.log"):
        self.log_file = log_file

    def log_decision(self, interrupt, decisions, user_id):
        """记录审批决策"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "actions": [
                {
                    "tool": action["name"],
                    "arguments": action["arguments"]
                }
                for action in interrupt["value"]["action_requests"]
            ],
            "decisions": decisions,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

# 创建带审计的 Agent
audit = AuditMiddleware()

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "sensitive_operation": True
            }
        )
    ],
    checkpointer=InMemorySaver(),
)

# 审批流程
def audited_approval_workflow(agent, request, config, user_id):
    """带审计的审批流程"""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": request}]},
        config=config
    )

    while "__interrupt__" in result:
        interrupt = result["__interrupt__"]

        # 收集决策
        decisions = collect_user_decisions(interrupt)

        # 记录审计日志
        audit.log_decision(interrupt, decisions, user_id)

        # 恢复执行
        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=config
        )

    return result
```

---

## 最佳实践

### 1. 分层审批策略

```python
# 根据风险等级设置不同的审批策略
HumanInTheLoopMiddleware(
    interrupt_on={
        # 低风险：仅记录
        "read_data": False,

        # 中风险：允许批准和拒绝
        "update_record": {
            "allowed_decisions": ["approve", "reject"]
        },

        # 高风险：允许批准、修改和拒绝
        "delete_record": {
            "allowed_decisions": ["approve", "edit", "reject"]
        },

        # 极高风险：只允许批准（需在 Agent 外部做额外检查）
        "drop_table": {
            "allowed_decisions": ["approve"],  # 还需要额外验证
            "description": "极端危险操作，需要 DBA 亲自确认"
        }
    }
)
```

### 2. 动态描述生成

```python
def generate_hitl_description(tool_name: str, args: dict) -> str:
    """生成友好的审批描述"""
    descriptions = {
        "send_email": (
            f"准备发送邮件:\n"
            f"收件人: {', '.join(args.get('to', []))}\n"
            f"主题: {args.get('subject', '无主题')}\n"
            f"正文长度: {len(args.get('body', ''))} 字符"
        ),
        "execute_sql": (
            f"准备执行 SQL:\n"
            f"查询: {args.get('query', '')}\n"
            f"⚠️ 请检查 SQL 安全性"
        ),
        "transfer_funds": (
            f"准备转账:\n"
            f"金额: {args.get('amount', 0)} {args.get('currency', 'USD')}\n"
            f"收款账户: {args.get('to_account', '未知')}\n"
            f"⚠️  此操作不可撤销"
        )
    }
    return descriptions.get(tool_name, f"操作: {tool_name}")

HumanInTheLoopMiddleware(
    interrupt_on={
        tool_name: {
            "description": lambda args: generate_hitl_description(tool_name, args)
        }
        for tool_name in ["send_email", "execute_sql", "transfer_funds"]
    }
)
```

### 3. 批量审批处理

```python
def batch_approval(agent, requests, config):
    """批量审批多个请求"""
    all_results = []

    for request in requests:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": request}]},
            config=config
        )

        # 收集中断
        interrupts = []
        while "__interrupt__" in result:
            interrupts.append(result["__interrupt__"])

            # 批量收集所有决策
            all_decisions = []
            for interrupt in interrupts:
                for action in interrupt["value"]["action_requests"]:
                    # 自动决策规则
                    if is_safe_action(action):
                        all_decisions.append({"type": "approve"})
                    elif is_rejected_action(action):
                        all_decisions.append({
                            "type": "reject",
                            "message": "不符合安全策略"
                        })
                    else:
                        # 需要人工决策
                        decision = get_manual_decision(action)
                        all_decisions.append(decision)

            # 批量恢复
            result = agent.invoke(
                Command(resume={"decisions": all_decisions}),
                config=config
            )

        all_results.append(result)

    return all_results
```

### 4. 审批超时处理

```python
import time
from threading import Thread

def approval_with_timeout(agent, request, config, timeout_seconds=300):
    """带超时的审批流程"""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": request}]},
        config=config
    )

    if "__interrupt__" not in result:
        return result

    # 启动超时定时器
    timeout_occurred = False

    def timeout_handler():
        nonlocal timeout_occurred
        time.sleep(timeout_seconds)
        timeout_occurred = True
        print(f"\n审批超时 ({timeout_seconds}秒)，自动拒绝")

    timeout_thread = Thread(target=timeout_handler, daemon=True)
    timeout_thread.start()

    # 收集决策
    decisions = []
    for action in result["__interrupt__"]["value"]["action_requests"]:
        if timeout_occurred:
            decisions.append({
                "type": "reject",
                "message": "审批超时，操作被自动拒绝"
            })
        else:
            decision = input(f"审批 {action['name']}: ")
            if decision.lower() == "approve":
                decisions.append({"type": "approve"})
            else:
                decisions.append({
                    "type": "reject",
                    "message": "用户拒绝"
                })

    # 恢复执行
    return agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config
    )
```

### 5. 条件性启用 HITL

```python
class ConditionalHITL:
    """条件性启用 HITL"""

    def __init__(self, always_interrupt_tools=None, conditional_rules=None):
        self.always_interrupt = set(always_interrupt_tools or [])
        self.conditional_rules = conditional_rules or []

    def should_interrupt(self, tool_name: str, args: dict) -> bool:
        """判断是否需要中断"""
        # 始终中断的工具
        if tool_name in self.always_interrupt:
            return True

        # 条件规则
        for rule in self.conditional_rules:
            if rule["tool"] == tool_name:
                if rule["condition"](args):
                    return True

        return False

# 使用规则
conditional_hitl = ConditionalHITL(
    always_interrupt_tools=["delete_file", "drop_table"],
    conditional_rules=[
        {
            "tool": "execute_sql",
            "condition": lambda args: any(
                kw in args["query"].upper()
                for kw in ["DELETE", "UPDATE", "DROP"]
            )
        },
        {
            "tool": "send_email",
            "condition": lambda args: any(
                not e.endswith("company.com")
                for e in args["to"]
            )
        }
    ]
)

# 在中间件中使用
class SmartHITLMiddleware(HumanInTheLoopMiddleware):
    def __init__(self, conditional_hitl):
        self.conditional_hitl = conditional_hitl
        super().__init__(interrupt_on={})

    def should_interrupt(self, tool_name: str, args: dict) -> bool:
        return self.conditional_hitl.should_interrupt(tool_name, args)
```

### 6. 测试 HITL 工作流

```python
import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

@pytest.fixture
def hitl_agent():
    """创建测试用的 HITL Agent"""
    @tool
    def sensitive_operation(param: str) -> str:
        return f"执行: {param}"

    @tool
    def safe_operation(param: str) -> str:
        return f"安全: {param}"

    agent = create_agent(
        model="gpt-4o",
        tools=[sensitive_operation, safe_operation],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "sensitive_operation": True,
                    "safe_operation": False,
                }
            )
        ],
        checkpointer=InMemorySaver(),
    )

    return agent

def test_safe_operation_auto_approves(hitl_agent):
    """测试安全操作自动批准"""
    config = {"configurable": {"thread_id": "test-1"}}

    result = hitl_agent.invoke(
        {"messages": [{"role": "user", "content": "执行 safe_operation"}]},
        config=config
    )

    assert "__interrupt__" not in result

def test_sensitive_operation_requires_approval(hitl_agent):
    """测试敏感操作需要审批"""
    config = {"configurable": {"thread_id": "test-2"}}

    result = hitl_agent.invoke(
        {"messages": [{"role": "user", "content": "执行 sensitive_operation"}]},
        config=config
    )

    assert "__interrupt__" in result
    assert result["__interrupt__"]["value"]["action_requests"][0]["name"] == "sensitive_operation"

def test_approve_flow(hitl_agent):
    """测试批准流程"""
    from langgraph.types import Command

    config = {"configurable": {"thread_id": "test-3"}}

    # 第一次调用：触发中断
    result = hitl_agent.invoke(
        {"messages": [{"role": "user", "content": "执行 sensitive_operation"}]},
        config=config
    )

    assert "__interrupt__" in result

    # 第二次调用：批准
    result = hitl_agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config
    )

    assert "__interrupt__" not in result

def test_reject_flow(hitl_agent):
    """测试拒绝流程"""
    from langgraph.types import Command

    config = {"configurable": {"thread_id": "test-4"}}

    # 第一次调用：触发中断
    result = hitl_agent.invoke(
        {"messages": [{"role": "user", "content": "执行 sensitive_operation"}]},
        config=config
    )

    assert "__interrupt__" in result

    # 第二次调用：拒绝
    result = hitl_agent.invoke(
        Command(resume={
            "decisions": [{
                "type": "reject",
                "message": "测试拒绝"
            }]
        }),
        config=config
    )

    # 拒绝后应继续执行（不执行工具）
    assert "__interrupt__" not in result

def test_edit_flow(hitl_agent):
    """测试修改流程"""
    from langgraph.types import Command

    config = {"configurable": {"thread_id": "test-5"}}

    # 第一次调用：触发中断
    result = hitl_agent.invoke(
        {"messages": [{"role": "user", "content": "执行 sensitive_operation"}]},
        config=config
    )

    assert "__interrupt__" in result

    # 第二次调用：修改参数
    result = hitl_agent.invoke(
        Command(resume={
            "decisions": [{
                "type": "edit",
                "edited_action": {
                    "name": "sensitive_operation",
                    "args": {"param": "修改后的参数"}
                }
            }]
        }),
        config=config
    )

    assert "__interrupt__" not in result
```

---

## 快速参考

### 基本配置

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "sensitive_tool": True,     # 启用所有决策
                "safe_tool": False,          # 自动批准
                "restricted_tool": {         # 自定义配置
                    "allowed_decisions": ["approve", "reject"],
                    "description": "自定义描述"
                }
            },
            description_prefix="工具执行需要审批"
        )
    ],
    checkpointer=InMemorySaver(),  # 必需
)
```

### 检测和处理中断

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "session-123"}}

# 执行并检测中断
result = agent.invoke(
    {"messages": [{"role": "user", "content": "..."}]},
    config=config
)

if "__interrupt__" in result:
    interrupt = result["__interrupt__"]

    # 批准
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config
    )

    # 修改
    result = agent.invoke(
        Command(resume={
            "decisions": [{
                "type": "edit",
                "edited_action": {
                    "name": "tool_name",
                    "args": {"key": "new_value"}
                }
            }]
        }),
        config=config
    )

    # 拒绝
    result = agent.invoke(
        Command(resume={
            "decisions": [{
                "type": "reject",
                "message": "拒绝原因"
            }]
        }),
        config=config
    )
```

### 流式处理

```python
config = {"configurable": {"thread_id": "session-123"}}

for mode, chunk in agent.stream(
    {"messages": [{"role": "user", "content": "..."}]},
    config=config,
    stream_mode=["updates", "messages"],
):
    if mode == "messages":
        token, metadata = chunk
        if hasattr(token, 'content') and token.content:
            print(token.content, end="", flush=True)
    elif mode == "updates":
        if "__interrupt__" in chunk:
            # 处理中断
            decisions = collect_decisions(chunk["__interrupt__"])

            # 继续流式处理
            for mode, chunk in agent.stream(
                Command(resume={"decisions": decisions}),
                config=config,
                stream_mode=["updates", "messages"],
            ):
                # ...
```

### 生产环境配置

```python
import asyncio
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def create_production_agent():
    checkpointer = AsyncPostgresSaver.from_conn_string(
        "postgresql+asyncpg://user:password@localhost/db"
    )
    await checkpointer.setup()

    agent = create_agent(
        model="gpt-4o",
        tools=[...],
        middleware=[HumanInTheLoopMiddleware(...)],
        checkpointer=checkpointer,
    )

    return agent
```

### 决策类型

| 类型 | 用法 | 说明 |
|------|------|------|
| approve | `{"type": "approve"}` | 按原样执行 |
| edit | `{"type": "edit", "edited_action": {...}}` | 修改后执行 |
| reject | `{"type": "reject", "message": "..."}` | 拒绝并提供反馈 |

---

## 总结

**Human-in-the-Loop (HITL)** 是实现人机协作的关键模式，通过在敏感操作前插入人工审核，提高 AI 应用的安全性和可靠性。

### 核心要点

1. **三种决策类型**
   - approve：批准执行
   - edit：修改后执行
   - reject：拒绝并反馈

2. **必需组件**
   - HumanInTheLoopMiddleware：中间件
   - checkpointer：持久化层
   - thread_id：会话标识

3. **配置方式**
   - True：启用所有决策
   - False：自动批准
   - dict：自定义配置

4. **执行流程**
   - 模型生成响应
   - 中间件检查工具调用
   - 需要审核则触发中断
   - 等待人工决策
   - 根据决策继续执行

5. **应用场景**
   - 高风险操作（数据库、金融交易）
   - 合规工作流
   - 敏感通信
   - 长期对话引导

### 最佳实践

- 根据风险等级设置分层审批策略
- 使用动态描述提高审批体验
- 实现批量审批提高效率
- 添加审批超时处理
- 实现条件性启用 HITL
- 充分测试 HITL 工作流

通过合理使用 HITL，你可以构建既强大又安全的 AI 应用，在自动化和人工控制之间找到最佳平衡！
