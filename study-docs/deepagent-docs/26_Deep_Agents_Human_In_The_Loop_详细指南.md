# Deep Agents Human-in-the-loop 与 Permissions 详细指南

> 基于 Deep Agents 官方 Human-in-the-loop 与 Permissions 文档整理。本文聚焦如何为敏感工具调用配置人工审批、如何处理 approve/edit/reject/respond 四类决策、如何使用条件中断、如何恢复被中断的运行，以及如何用 filesystem permissions 做路径级 allow/deny/interrupt 控制。

## 目录

1. [核心概念](#核心概念)
2. [Human-in-the-loop 是什么](#human-in-the-loop-是什么)
3. [基本配置](#基本配置)
4. [Decision Types：四种人工决策](#decision-types四种人工决策)
5. [Conditional Interrupts：条件中断](#conditional-interrupts条件中断)
6. [Handle Interrupts：处理中断与恢复](#handle-interrupts处理中断与恢复)
7. [Multiple Tool Calls：批量工具审批](#multiple-tool-calls批量工具审批)
8. [Rejection Messages：拒绝信息](#rejection-messages拒绝信息)
9. [Edit Tool Arguments：编辑工具参数](#edit-tool-arguments编辑工具参数)
10. [Subagent Interrupts：子智能体中断](#subagent-interrupts子智能体中断)
11. [Filesystem Permission Interrupts](#filesystem-permission-interrupts)
12. [Permissions：路径级访问控制](#permissions路径级访问控制)
13. [Permission 示例模式](#permission-示例模式)
14. [Subagent Permissions](#subagent-permissions)
15. [CompositeBackend 与 Permissions](#compositebackend-与-permissions)
16. [与 Backends / Permissions / Sandbox 的关系](#与-backends--permissions--sandbox-的关系)
17. [常见配置模式](#常见配置模式)
18. [最佳实践](#最佳实践)
19. [故障排查](#故障排查)
20. [快速参考](#快速参考)
21. [资料来源](#资料来源)

---

## 核心概念

Human-in-the-loop，简称 HITL，是让 agent 在执行敏感工具操作前暂停，由人类审查并决定是否继续。

典型流程：

```text
Agent 准备调用工具
  -> HumanInTheLoopMiddleware 判断是否需要中断
    -> 不需要：直接执行工具
    -> 需要：暂停并返回 action_requests
      -> 人类 approve / edit / reject / respond
        -> 使用 Command(resume=...) 恢复运行
```

适合配置 HITL 的操作：

| 操作 | 原因 |
|------|------|
| 删除文件 | 不可逆或高风险 |
| 写入/修改文件 | 可能破坏数据或代码 |
| 发送邮件 | 有外部副作用 |
| 调用支付/下单/交易 API | 资金或业务风险 |
| 部署生产环境 | 影响线上系统 |
| 写入 secrets / policies | 高安全风险 |
| 执行 shell 命令 | 可能影响宿主机或环境 |

Deep Agents 通过 LangGraph interrupt 能力实现 HITL。配置 `interrupt_on` 后，`HumanInTheLoopMiddleware` 会加入默认 middleware stack。

如果运行在工具返回前被取消或中断，同一 stack 中的 `PatchToolCallsMiddleware` 会自动修复 message history，避免留下不完整 tool call。

---

## Human-in-the-loop 是什么

HITL 不是让人类参与每一步推理，而是只在关键动作前暂停审批。

可以把它理解成：

```text
模型负责提出动作；
人类负责审核动作；
系统负责在同一个 thread 上恢复执行。
```

HITL 解决的问题：

| 问题 | HITL 如何解决 |
|------|---------------|
| Agent 可能误删文件 | 删除前暂停，等待 approve/reject |
| Agent 可能发错邮件 | 发送前允许 edit recipient/body |
| Agent 可能写入敏感路径 | filesystem permission interrupt |
| Agent 可能执行危险工具 | 高风险工具配置 `interrupt_on` |
| 审批后要继续上下文 | checkpointer + same thread ID 恢复 |

HITL 不等于 sandbox：

| 概念 | 作用 |
|------|------|
| HITL | 执行前人工审批 |
| Permissions | 路径级 allow/deny/interrupt |
| Sandbox | 隔离执行环境 |
| Backend | 文件系统工具背后的实际存储/执行层 |

它们可以组合使用，而不是互相替代。

---

## 基本配置

`interrupt_on` 参数接收一个字典，key 是工具名，value 是中断配置。

配置形式：

| 配置 | 含义 |
|------|------|
| `True` | 对该工具启用默认中断，允许 `approve`、`edit`、`reject`、`respond` |
| `False` | 对该工具禁用中断 |
| `InterruptOnConfig` / dict | 自定义配置，例如 `allowed_decisions` 或 `when` |

基础示例：

```python
from langchain.tools import tool
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver


@tool
def remove_file(path: str) -> str:
    """Delete a file from the filesystem."""
    return f"Deleted {path}"


@tool
def fetch_file(path: str) -> str:
    """Read a file from the filesystem."""
    return f"Contents of {path}"


@tool
def notify_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Sent email to {to}"


checkpointer = MemorySaver()

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[remove_file, fetch_file, notify_email],
    interrupt_on={
        "remove_file": True,
        "fetch_file": False,
        "notify_email": {"allowed_decisions": ["approve", "reject"]},
    },
    checkpointer=checkpointer,
)
```

这个配置表示：

| 工具 | 审批策略 |
|------|----------|
| `remove_file` | 需要审批，允许默认所有决策 |
| `fetch_file` | 不需要审批 |
| `notify_email` | 需要审批，但只允许 approve/reject，不允许 edit/respond |

关键要求：

```text
HITL 必须配置 checkpointer。
```

没有 checkpointer，agent 无法在 interrupt 与 resume 之间保存状态。

---

## Decision Types：四种人工决策

`allowed_decisions` 控制审查时人类可以做什么。

| Decision | 含义 | 典型场景 |
|----------|------|----------|
| `approve` | 使用 agent 原始参数执行工具 | 邮件草稿没问题，直接发送 |
| `edit` | 修改工具参数后再执行 | 修改邮件收件人、文件路径、正文 |
| `reject` | 跳过该工具调用，并把拒绝反馈返回给 agent | 拒绝删除文件并说明原因 |
| `respond` | 不执行工具，直接把人类消息作为 synthetic tool result 返回 | `ask_user` 工具请求用户回答 |

### `reject` 与 `respond` 的区别

| 类型 | 是否表示工具成功 | 用法 |
|------|------------------|------|
| `reject` | 否 | 拒绝敏感/有副作用工具 |
| `respond` | 更像工具返回结果 | 人类充当工具，例如回答 `ask_user` |

重要注意：

```text
不要用 respond 来拒绝有副作用工具。
```

因为模型可能把 `respond` 的内容当作成功的工具结果，从而继续按“工具已完成”推理。

### 按风险定制决策

```python
interrupt_on = {
    # 高风险：允许审批、编辑、拒绝
    "delete_file": {"allowed_decisions": ["approve", "edit", "reject"]},

    # 中风险：只允许审批或拒绝，不允许修改参数
    "write_file": {"allowed_decisions": ["approve", "reject"]},

    # 关键操作：只能审批，不能拒绝或编辑
    "critical_operation": {"allowed_decisions": ["approve"]},
}
```

编辑参数时应保守。大幅改动参数可能让模型重新评估策略，导致重复调用工具或产生意外行为。

---

## Conditional Interrupts：条件中断

默认情况下，只要工具在 `interrupt_on` 中配置为中断，每次调用都会暂停。

如果只想对某些参数组合中断，可以使用 `when` predicate。

要求：

```text
Conditional interrupts require langchain>=1.3.3
```

示例：只有写入 `/workspace/` 外部路径时才暂停。

```python
from deepagents import create_deep_agent
from langchain.agents.middleware import ToolCallRequest
from langgraph.checkpoint.memory import MemorySaver


def writes_outside_workspace(request: ToolCallRequest) -> bool:
    """Pause writes to paths outside the workspace directory."""
    path = request.tool_call["args"].get("file_path", "")
    return not path.startswith("/workspace/")


agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    interrupt_on={
        "write_file": {
            "allowed_decisions": ["approve", "edit", "reject"],
            "when": writes_outside_workspace,
        },
    },
    checkpointer=MemorySaver(),
)
```

行为：

| `when` 返回 | 行为 |
|-------------|------|
| `False` | 自动通过，不进入 interrupt batch |
| `True` | 暂停，进入人工审批 |
| 未配置 `when` | 每次调用都暂停 |

适合使用条件中断的场景：

| 场景 | 条件 |
|------|------|
| 文件写入 | 只审批 workspace 外写入 |
| 邮件发送 | 只审批外部域名收件人 |
| 数据库操作 | 只审批 UPDATE/DELETE |
| API 调用 | 只审批金额超过阈值 |
| Shell 执行 | 只审批包含危险命令的调用 |

---

## Handle Interrupts：处理中断与恢复

当工具调用触发 interrupt 时，agent 会暂停并返回控制权。调用方需要：

1. 检查 `result.interrupts`。
2. 读取 `action_requests` 和 `review_configs`。
3. 展示给人类审查。
4. 收集 decisions。
5. 用 `Command(resume={"decisions": decisions})` 恢复。
6. 使用相同 `config`，尤其是相同 `thread_id`。

示例：

```python
from langchain_core.utils.uuid import uuid7
from langgraph.types import Command

config = {"configurable": {"thread_id": str(uuid7())}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Delete the file temp.txt"}]},
    config=config,
    version="v2",
)

if result.interrupts:
    interrupt_value = result.interrupts[0].value
    action_requests = interrupt_value["action_requests"]
    review_configs = interrupt_value["review_configs"]

    config_map = {cfg["action_name"]: cfg for cfg in review_configs}

    for action in action_requests:
        review_config = config_map[action["name"]]
        print(f"Tool: {action['name']}")
        print(f"Arguments: {action['args']}")
        print(f"Allowed decisions: {review_config['allowed_decisions']}")

    decisions = [
        {
            "type": "reject",
            "message": "User rejected deleting temp.txt. Do not retry deletion.",
        }
    ]

    result = agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config,
        version="v2",
    )

print(result.value["messages"][-1].content)
```

中断结果中的关键字段：

| 字段 | 说明 |
|------|------|
| `result.interrupts` | 中断列表 |
| `interrupts[0].value` | 中断 payload |
| `action_requests` | 待审批工具调用列表 |
| `review_configs` | 每个 action 的允许决策配置 |

恢复时的关键点：

| 要点 | 原因 |
|------|------|
| 使用 `Command(resume=...)` | 告诉 LangGraph 从 interrupt 处继续 |
| 使用相同 `config` | 找回同一个 thread/checkpoint |
| 使用相同 `thread_id` | 否则无法恢复原运行状态 |
| `version="v2"` | 文档示例要求用于 interrupt result |

---

## Multiple Tool Calls：批量工具审批

当 agent 一次调用多个需要审批的工具时，Deep Agents 会把它们合并到一个 interrupt 中。

你必须按 `action_requests` 顺序提供同样数量的 decisions。

示例：

```python
from langchain_core.utils.uuid import uuid7
from langgraph.types import Command

config = {"configurable": {"thread_id": str(uuid7())}}

result = agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "Delete temp.txt and send an email to admin@example.com",
        }]
    },
    config=config,
    version="v2",
)

if result.interrupts:
    interrupt_value = result.interrupts[0].value
    action_requests = interrupt_value["action_requests"]

    assert len(action_requests) == 2

    decisions = [
        {"type": "approve"},
        {
            "type": "reject",
            "message": "User rejected this action. Do not retry this tool call.",
        },
    ]

    result = agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config,
        version="v2",
    )
```

原则：

```text
decisions[i] 对应 action_requests[i]
```

如果顺序错了，可能会把审批应用到错误工具调用上。

---

## Rejection Messages：拒绝信息

当人类选择 `reject` 时，Deep Agents 会跳过该工具调用，并把拒绝反馈作为工具结果返回给 agent。

如果省略 `message`，系统会提供默认反馈：工具未执行，不要重试同一个工具调用，除非用户要求。

对于敏感或有副作用工具，推荐显式写 domain-specific message：

```python
decisions = [
    {
        "type": "reject",
        "message": (
            "User rejected deleting this file. "
            "Do not retry deletion. Ask which file to archive instead."
        ),
    }
]
```

好的 rejection message 应包含：

| 内容 | 示例 |
|------|------|
| 明确未执行 | “The file was not deleted.” |
| 不要重试 | “Do not retry this deletion.” |
| 下一步建议 | “Ask which file to archive instead.” |
| 原因 | “The path contains production data.” |

---

## Edit Tool Arguments：编辑工具参数

当工具允许 `edit` 时，人类可以修改工具参数再执行。

示例：修改邮件收件人。

```python
from langgraph.types import Command

if result.interrupts:
    interrupt_value = result.interrupts[0].value
    action_request = interrupt_value["action_requests"][0]

    print(action_request["args"])

    decisions = [{
        "type": "edit",
        "edited_action": {
            "name": action_request["name"],
            "args": {
                "to": "team@company.com",
                "subject": "...",
                "body": "...",
            },
        },
    }]

    result = agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config,
        version="v2",
    )
```

编辑要求：

| 字段 | 要求 |
|------|------|
| `type` | `"edit"` |
| `edited_action.name` | 必须包含原工具名 |
| `edited_action.args` | 修改后的完整工具参数 |

编辑建议：

1. 做最小必要修改。
2. 不要把工具语义改成另一个任务。
3. 不要大幅改变参数导致模型误判工具结果。
4. 对高风险操作，必要时禁止 edit，只允许 approve/reject。

---

## Subagent Interrupts：子智能体中断

Subagents 中有两类中断方式：

1. 对 subagent 的工具调用配置 `interrupt_on`。
2. 在 subagent 工具内部直接调用 `interrupt()`。

### Interrupts on Tool Calls

每个 subagent 可以配置自己的 `interrupt_on`，并覆盖主 agent 的设置。

示例：

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[delete_file, read_file],
    interrupt_on={
        "delete_file": True,
        "read_file": False,
    },
    subagents=[{
        "name": "file-manager",
        "description": "Manages file operations",
        "system_prompt": "You are a file management assistant.",
        "tools": [delete_file, read_file],
        "interrupt_on": {
            "delete_file": True,
            "read_file": True,
        },
    }],
    checkpointer=checkpointer,
)
```

这里主 agent 与 subagent 的策略不同：

| 工具 | 主 agent | `file-manager` subagent |
|------|----------|--------------------------|
| `delete_file` | 需要审批 | 需要审批 |
| `read_file` | 不需要审批 | 需要审批 |

当 subagent 触发 interrupt 时，处理方式与主 agent 相同：检查 `result.interrupts`，然后用 `Command(resume=...)` 恢复。

### Interrupts within Tool Calls

工具内部可以直接调用 LangGraph 的 `interrupt()` primitive。

适合场景：

| 场景 | 说明 |
|------|------|
| 工具内部需要审批 | 例如部署、付款、发送外部请求 |
| 审批逻辑不是简单按工具名判断 | 工具运行中发现风险再暂停 |
| CompiledSubAgent 内部 workflow | 子 graph 自己决定何时 interrupt |

简化示例：

```python
from langchain.tools import tool
from langgraph.types import interrupt


@tool(description="Request human approval before proceeding with an action.")
def request_approval(action_description: str) -> str:
    """Request human approval using the interrupt() primitive."""
    approval = interrupt({
        "type": "approval_request",
        "action": action_description,
        "message": f"Please approve or reject: {action_description}",
    })

    if approval.get("approved"):
        return f"Action '{action_description}' was APPROVED. Proceeding..."
    else:
        return (
            f"Action '{action_description}' was REJECTED. "
            f"Reason: {approval.get('reason', 'No reason provided')}"
        )
```

恢复时可以传任意与工具约定匹配的 payload：

```python
result = parent_agent.invoke(
    Command(resume={"approved": True}),
    config=config,
    version="v2",
)
```

这类中断不一定使用 `decisions` 列表，因为它是工具内部自定义 `interrupt()` 的 payload。

---

## Filesystem Permission Interrupts

除了 `interrupt_on`，Deep Agents 还支持在 filesystem permissions 中把规则设置为：

```python
mode="interrupt"
```

要求：

```text
Filesystem permission interrupts require deepagents>=0.6.8
```

示例：写入 `/secrets/**` 前暂停。

```python
from deepagents import FilesystemPermission, create_deep_agent
from langgraph.checkpoint.memory import MemorySaver


agent = create_deep_agent(
    model=model,
    permissions=[
        FilesystemPermission(
            operations=["write"],
            paths=["/secrets/**"],
            mode="interrupt",
        ),
    ],
    checkpointer=MemorySaver(),
)
```

当 agent 调用 `write_file` 或 `edit_file` 写入匹配路径时，会触发与工具调用相同形式的人类审批 interrupt。

处理方式：

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "fs-thread-1"}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Save the API key to /secrets/key.txt"}]},
    config=config,
    version="v2",
)

if result.interrupts:
    action = result.interrupts[0].value["action_requests"][0]
    print(f"Approve {action['name']} on {action['args']}?")

    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config,
        version="v2",
    )
```

特点：

1. 适用于内置 filesystem tools。
2. 与 `interrupt_on` 合并到同一个 review step。
3. 一个审批步骤可以同时覆盖自定义工具和受保护文件路径。
4. 仍然需要 checkpointer。

---

## Permissions：路径级访问控制

Permissions 用于控制 Deep Agents 内置文件系统工具可以读写哪些路径。

要求：

```text
Permissions require deepagents>=0.5.2
```

Permissions 只作用于内置 filesystem tools：

| Operation | 覆盖工具 |
|-----------|----------|
| `read` | `ls`, `read_file`, `glob`, `grep` |
| `write` | `write_file`, `edit_file` |

不覆盖：

| 不覆盖对象 | 原因 |
|------------|------|
| Custom tools | 自定义工具内部访问文件不经过内置 filesystem permission |
| MCP tools | MCP server 有自己的运行权限 |
| Sandbox `execute` | shell 命令可绕过路径级规则访问文件 |
| Backend policy hooks | 这是另一层自定义策略机制 |

基本用法：

```python
from deepagents import FilesystemPermission, create_deep_agent


agent = create_deep_agent(
    model=model,
    backend=backend,
    permissions=[
        FilesystemPermission(
            operations=["write"],
            paths=["/**"],
            mode="deny",
        ),
    ],
)
```

这个例子表示：拒绝所有写操作，让 agent 变成只读。

### Rule Structure

每个 `FilesystemPermission` 有三个字段：

| Field | 类型 | 说明 |
|-------|------|------|
| `operations` | `list["read" | "write"]` | 规则适用的操作类型 |
| `paths` | `list[str]` | glob path patterns，例如 `["/workspace/**"]` |
| `mode` | `"allow" | "deny" | "interrupt"` | 允许、拒绝或暂停人工审批；默认是 `allow` |

Path pattern 支持：

| Pattern | 含义 |
|---------|------|
| `/workspace/**` | `/workspace/` 下所有文件和子目录 |
| `/workspace/.env` | 精确匹配某个文件 |
| `/projects/*/secrets/**` | 匹配一层项目名下的 secrets 目录 |
| `/{docs,policies}/**` | 使用 alternation 匹配多个目录 |

### First-match-wins

Permissions 按声明顺序评估：

```text
第一条同时匹配 operation 和 path 的规则获胜。
如果没有任何规则匹配，默认允许。
```

这点非常重要。更具体的 deny 规则应放在更宽泛的 allow 规则之前。

正确：

```python
correct_permissions = [
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/workspace/.env"],
        mode="deny",
    ),
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/workspace/**"],
        mode="allow",
    ),
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/**"],
        mode="deny",
    ),
]
```

错误：

```python
incorrect_permissions = [
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/workspace/**"],
        mode="allow",
    ),
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/workspace/.env"],
        mode="deny",
    ),
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/**"],
        mode="deny",
    ),
]
```

错误原因：

```text
/workspace/** 先匹配了 /workspace/.env，
后面的 deny 永远不会触发。
```

### Permission Mode

| Mode | 行为 | 适合场景 |
|------|------|----------|
| `allow` | 允许匹配操作 | 明确开放 workspace |
| `deny` | 直接拒绝匹配操作 | 禁止读 secrets、禁止写 policies |
| `interrupt` | 暂停等待人工审批 | 写入敏感路径前让人类审核 |

`interrupt` 与 HITL 的关系：

1. `mode="interrupt"` 会自动接入 human-in-the-loop middleware。
2. 触发后处理方式与 `interrupt_on` 工具审批相同。
3. 需要 checkpointer。
4. 可以与 `interrupt_on` 合并到同一次 review。

### Interrupt Pattern 建议

文档建议：interrupt pattern 最好以明确的 literal leading segment 开头。

推荐：

```text
/secrets/**
/projects/*/secrets/**
```

不推荐：

```text
/**/secrets
```

原因是 `ls`、`glob`、`grep` 这类 bulk tools 在搜索子树可能与规则重叠时会保守触发 interrupt。完全不 anchored 的 pattern 可能导致过度触发。

---

## Permission 示例模式

### 只允许 Workspace 读写

允许读写 `/workspace/`，拒绝其他路径：

```python
agent = create_deep_agent(
    model=model,
    backend=backend,
    permissions=[
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/workspace/**"],
            mode="allow",
        ),
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/**"],
            mode="deny",
        ),
    ],
)
```

适合：

| 场景 | 说明 |
|------|------|
| 本地项目工作区 | 只让 agent 操作项目目录 |
| 多租户隔离 | 每个 tenant 一个路径 |
| CI 任务 | 限制在 workspace 内 |

### 保护特定文件

禁止访问 `.env` 和 examples，但允许 workspace 其他内容：

```python
agent = create_deep_agent(
    model=model,
    backend=backend,
    permissions=[
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/workspace/.env", "/workspace/examples/**"],
            mode="deny",
        ),
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/workspace/**"],
            mode="allow",
        ),
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/**"],
            mode="deny",
        ),
    ],
)
```

注意：特定 deny 必须放在 workspace allow 前面。

### Read-only Memory

让 agent 可以读取 memory，但不能修改 memory 或 policies：

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend


agent = create_deep_agent(
    model=model,
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(
                namespace=lambda rt: (rt.server_info.user.identity,),
            ),
            "/policies/": StoreBackend(
                namespace=lambda rt: (rt.context.org_id,),
            ),
        },
    ),
    permissions=[
        FilesystemPermission(
            operations=["write"],
            paths=["/memories/**", "/policies/**"],
            mode="deny",
        ),
    ],
)
```

适合：

| 路径 | 策略 |
|------|------|
| `/memories/**` | agent 可读，但不可写 |
| `/policies/**` | 组织策略只读 |
| 其他路径 | 未匹配时默认允许，按实际需要可再加 deny baseline |

### Deny All Access

阻止所有读写，作为最严格 baseline：

```python
agent = create_deep_agent(
    model=model,
    backend=backend,
    permissions=[
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/**"],
            mode="deny",
        ),
    ],
)
```

如果要在严格 baseline 上开放路径，需要把 allow 规则放在 deny all 之前：

```python
permissions = [
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/workspace/**"],
        mode="allow",
    ),
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/**"],
        mode="deny",
    ),
]
```

---

## Subagent Permissions

Subagents 默认继承 parent agent 的 permissions。

如果在 subagent spec 中设置 `permissions` 字段，它会：

```text
完全替换 parent permissions，
不是 merge。
```

示例：父 agent 可读写 workspace，但 auditor 子智能体只读 workspace。

```python
agent = create_deep_agent(
    model=model,
    backend=backend,
    permissions=[
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/workspace/**"],
            mode="allow",
        ),
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/**"],
            mode="deny",
        ),
    ],
    subagents=[
        {
            "name": "auditor",
            "description": "Read-only code reviewer",
            "system_prompt": "Review the code for issues.",
            "permissions": [
                FilesystemPermission(
                    operations=["write"],
                    paths=["/**"],
                    mode="deny",
                ),
                FilesystemPermission(
                    operations=["read"],
                    paths=["/workspace/**"],
                    mode="allow",
                ),
                FilesystemPermission(
                    operations=["read"],
                    paths=["/**"],
                    mode="deny",
                ),
            ],
        }
    ],
)
```

使用建议：

| Subagent 类型 | 推荐权限 |
|---------------|----------|
| reviewer / auditor | 只读 |
| writer / editor | 只写指定 workspace |
| deployment agent | 对执行工具配置 HITL 或 sandbox |
| memory curator | 只允许写 `/memories/`，禁止写 `/policies/` |
| data analyst | 只读数据目录，结果写 `/reports/` |

注意：

1. Subagent permissions 替换 parent permissions，配置时要写完整规则集。
2. 如果忘记加 deny baseline，未匹配路径会默认允许。
3. 对审查类 subagent，建议显式 deny write。

---

## CompositeBackend 与 Permissions

当 backend 是 `CompositeBackend` 时，permissions 通常按路径前缀与 route 对齐。

例如：

```python
from deepagents.backends import CompositeBackend


composite = CompositeBackend(
    default=sandbox,
    routes={"/memories/": memories_backend},
)
```

如果 default 是 sandbox，权限 path 必须限定在已知 route prefix 下。

可行：

```python
agent = create_deep_agent(
    model=model,
    backend=composite,
    permissions=[
        FilesystemPermission(
            operations=["write"],
            paths=["/memories/**"],
            mode="deny",
        ),
    ],
)
```

不可行：

```python
agent = create_deep_agent(
    model=model,
    backend=composite,
    permissions=[
        FilesystemPermission(
            operations=["write"],
            paths=["/workspace/**"],
            mode="deny",
        ),
    ],
)
```

如果 `/workspace/**` 没有 route，会落到 sandbox default，文档说明会抛 `NotImplementedError`。

同样不可行：

```python
agent = create_deep_agent(
    model=model,
    backend=composite,
    permissions=[
        FilesystemPermission(
            operations=["read"],
            paths=["/**"],
            mode="deny",
        ),
    ],
)
```

原因：

```text
/** 同时覆盖 route 和 sandbox default。
而 sandbox 支持任意 execute 命令，
路径级权限无法可靠限制 shell 对文件系统的访问。
```

关键结论：

| 情况 | 建议 |
|------|------|
| Composite default 是 `StateBackend` | 可用宽泛 baseline，例如 `/**` |
| Composite default 是 sandbox | permission path 必须 scoped 到 route prefix |
| 需要限制 sandbox shell | 不靠 permissions，使用 sandbox provider 的隔离策略或不要暴露 execute |
| 需要自定义策略 | 用 backend policy hooks 或工具内部校验 |

---

## 与 Backends / Permissions / Sandbox 的关系

HITL 常与其他安全机制组合使用。

| 机制 | 解决什么问题 | 与 HITL 的关系 |
|------|--------------|----------------|
| `interrupt_on` | 按工具名审批敏感工具 | HITL 的主要入口 |
| `FilesystemPermission(mode="interrupt")` | 按文件路径审批读写 | 路径级 HITL |
| `FilesystemPermission(mode="deny")` | 直接拒绝某些文件操作 | 比 HITL 更硬，不给审批机会 |
| Backend | 决定工具实际读写哪里 | HITL 只管审批，不改变存储位置 |
| Sandbox backend | 隔离命令执行环境 | HITL 不等于隔离，可搭配 sandbox |
| `LocalShellBackend` | 本机 shell 执行 | 强烈建议给 `execute` 配 HITL 或改用 sandbox |
| MCP tools | 外部工具服务 | 需要在 tool 或 server 层单独做审批/权限 |
| Backend policy hooks | 自定义校验、审计、内容检查 | 当 permissions 不够表达时使用 |

关键结论：

```text
HITL 是执行前审批，不是执行隔离；
Sandbox 是执行环境隔离，不是人工审批；
Permissions 是路径规则，只覆盖内置 filesystem tools；
Custom tools / MCP tools / sandbox execute 需要额外控制。
```

---

## 常见配置模式

### 模式一：删除文件前审批

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[delete_file],
    interrupt_on={
        "delete_file": {"allowed_decisions": ["approve", "reject"]},
    },
    checkpointer=MemorySaver(),
)
```

适合：

| 场景 | 说明 |
|------|------|
| 删除文件 | 防止误删 |
| 归档数据 | 确认路径和目标 |
| 清理资源 | 操作不可逆 |

### 模式二：发邮件前可编辑

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[send_email],
    interrupt_on={
        "send_email": {
            "allowed_decisions": ["approve", "edit", "reject"],
        },
    },
    checkpointer=MemorySaver(),
)
```

适合编辑：

| 参数 | 例子 |
|------|------|
| 收件人 | 修正为团队邮箱 |
| 主题 | 加上标签 |
| 正文 | 删除敏感内容 |

### 模式三：只审批 workspace 外写入

```python
def writes_outside_workspace(request: ToolCallRequest) -> bool:
    path = request.tool_call["args"].get("file_path", "")
    return not path.startswith("/workspace/")


agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    interrupt_on={
        "write_file": {
            "allowed_decisions": ["approve", "edit", "reject"],
            "when": writes_outside_workspace,
        },
    },
    checkpointer=MemorySaver(),
)
```

### 模式四：secrets 路径写入前审批

```python
agent = create_deep_agent(
    model=model,
    permissions=[
        FilesystemPermission(
            operations=["write"],
            paths=["/secrets/**"],
            mode="interrupt",
        ),
    ],
    checkpointer=MemorySaver(),
)
```

### 模式五：Subagent 更严格审批

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[delete_file, read_file],
    interrupt_on={
        "delete_file": True,
        "read_file": False,
    },
    subagents=[{
        "name": "file-manager",
        "description": "Manages file operations",
        "system_prompt": "You are a file management assistant.",
        "tools": [delete_file, read_file],
        "interrupt_on": {
            "delete_file": True,
            "read_file": True,
        },
    }],
    checkpointer=MemorySaver(),
)
```

### 模式六：Workspace 隔离 + secrets 保护

```python
permissions = [
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/workspace/.env", "/workspace/secrets/**"],
        mode="deny",
    ),
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/workspace/**"],
        mode="allow",
    ),
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/**"],
        mode="deny",
    ),
]
```

这个模式同时实现：

| 目标 | 规则 |
|------|------|
| 保护 `.env` 和 secrets | specific deny 放最前 |
| 允许 workspace 读写 | allow `/workspace/**` |
| 拒绝其他路径 | deny `/**` |

### 模式七：只读 Auditor Subagent

```python
auditor = {
    "name": "auditor",
    "description": "Read-only code reviewer",
    "system_prompt": "Review the code for issues.",
    "permissions": [
        FilesystemPermission(
            operations=["write"],
            paths=["/**"],
            mode="deny",
        ),
        FilesystemPermission(
            operations=["read"],
            paths=["/workspace/**"],
            mode="allow",
        ),
        FilesystemPermission(
            operations=["read"],
            paths=["/**"],
            mode="deny",
        ),
    ],
}
```

注意：subagent 的 `permissions` 是替换父规则，不是追加。

---

## 最佳实践

### Always use a checkpointer

HITL 必须有 checkpointer：

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[...],
    interrupt_on={...},
    checkpointer=checkpointer,
)
```

### Use the same thread ID

第一次调用和恢复调用必须使用相同 `config`：

```python
config = {"configurable": {"thread_id": "my-thread"}}

result = agent.invoke(input, config=config, version="v2")

result = agent.invoke(
    Command(resume={...}),
    config=config,
    version="v2",
)
```

### Match decision order to actions

多工具审批时：

```text
len(decisions) == len(action_requests)
decisions[i] 对应 action_requests[i]
```

### Tailor configurations by risk

按风险配置不同策略：

```python
interrupt_on = {
    # 高风险：全控制
    "delete_file": {"allowed_decisions": ["approve", "edit", "reject"]},
    "send_email": {"allowed_decisions": ["approve", "edit", "reject"]},

    # 中风险：不允许编辑
    "write_file": {"allowed_decisions": ["approve", "reject"]},

    # 低风险：不中断
    "read_file": False,
    "ls": False,
}
```

### Order permission rules carefully

Permissions 是 first-match-wins。把更具体的 deny/interrupt 放在更宽泛的 allow 前面。

推荐顺序：

```text
specific deny / interrupt
-> scoped allow
-> broad deny baseline
```

示例：

```python
permissions = [
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/workspace/.env"],
        mode="deny",
    ),
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/workspace/**"],
        mode="allow",
    ),
    FilesystemPermission(
        operations=["read", "write"],
        paths=["/**"],
        mode="deny",
    ),
]
```

### 其他建议

1. 对 `LocalShellBackend` 的 `execute` 保持极高谨慎。
2. 有副作用工具默认至少允许 `reject`。
3. 对删除、转账、发送外部消息等操作提供清晰 rejection message。
4. 对路径类工具优先使用条件中断或 permission interrupt，减少无意义审批。
5. 不要让人类审批界面隐藏工具参数，审查必须看到完整 args。
6. 审批 UI 中展示 allowed decisions，避免用户做出系统不支持的选择。
7. 对高风险工具不要随意允许 `respond`。
8. Permissions 默认是 permissive：没有匹配规则时允许操作。
9. 对生产路径控制，建议加 broad deny baseline。
10. Permissions 不覆盖 custom tools 和 MCP tools，必要时用 backend policy hooks 或工具内部校验。

---

## 故障排查

### 没有触发中断

| 可能原因 | 处理 |
|----------|------|
| 工具名不匹配 | `interrupt_on` key 必须与工具名一致 |
| `when` 返回 `False` | 检查 predicate 逻辑 |
| 工具没有被调用 | 检查模型是否选择了该工具 |
| 使用的是外部 MCP/custom tool 内部操作 | 在外部工具自身实现审批或暴露可审批工具 |

### 无法恢复执行

| 可能原因 | 处理 |
|----------|------|
| 没有 checkpointer | 添加 `checkpointer=MemorySaver()` 或持久 checkpointer |
| thread ID 不一致 | 恢复时使用同一个 `config` |
| resume payload 格式错误 | 工具调用中断用 `{"decisions": [...]}` |
| 自定义 `interrupt()` payload 不匹配 | 按工具内部约定传 resume 值 |

### 批量审批行为错乱

| 可能原因 | 处理 |
|----------|------|
| decisions 顺序错了 | 按 `action_requests` 顺序生成 |
| decisions 数量不一致 | 每个 action request 一个 decision |
| UI 重新排序了 actions | 保留原始顺序或存 index |

### Reject 后模型又重试

| 可能原因 | 处理 |
|----------|------|
| rejection message 太含糊 | 明确写 “Do not retry this tool call.” |
| 模型认为只是暂时失败 | 指明用户拒绝且工具未执行 |
| 需要替代方案 | 在 message 中说明下一步，例如询问用户 |

### Edit 后行为不符合预期

| 可能原因 | 处理 |
|----------|------|
| 参数改动太大 | 只做保守修改 |
| 缺少工具名 | `edited_action.name` 必填 |
| args 不完整 | 提供工具所需完整参数 |
| 修改后语义不一致 | 选择 reject 并让 agent 重新规划 |

### Permissions 没生效

| 可能原因 | 处理 |
|----------|------|
| 没有匹配规则 | 默认允许，添加 broad deny baseline |
| 规则顺序错了 | specific deny 放在 broad allow 前 |
| 操作类型写错 | `read` 覆盖 `ls/read_file/glob/grep`，`write` 覆盖 `write_file/edit_file` |
| 工具不是内置 filesystem tool | custom tool / MCP tool 内部自行控制 |
| sandbox `execute` 绕过路径规则 | 使用 sandbox provider 隔离或不要暴露 execute |

### CompositeBackend + Sandbox 报 NotImplementedError

| 可能原因 | 处理 |
|----------|------|
| permission path 落到 sandbox default | 把 path 限定到已知 route prefix |
| 使用 `/**` 覆盖 route 和 sandbox default | 避免对 sandbox default 使用宽泛路径规则 |
| 想限制 shell 文件访问 | permissions 不适合，使用 sandbox 隔离策略 |

---

## 快速参考

### `interrupt_on` 配置速查

| 配置 | 含义 |
|------|------|
| `"tool": True` | 默认中断，允许默认决策 |
| `"tool": False` | 不对该工具中断 |
| `"tool": {"allowed_decisions": [...]}` | 控制可选决策 |
| `"tool": {"when": predicate}` | 条件中断 |
| subagent `"interrupt_on"` | 子智能体覆盖主 agent 设置 |

### Decision 速查

| Decision | 是否执行工具 | 用途 |
|----------|--------------|------|
| `approve` | 是，原参数 | 直接批准 |
| `edit` | 是，修改后参数 | 修正工具参数 |
| `reject` | 否 | 拒绝有副作用操作 |
| `respond` | 否，以人工回复作为工具结果 | 人类充当工具回答问题 |

### Resume 速查

| 中断类型 | Resume payload |
|----------|----------------|
| 工具审批中断 | `Command(resume={"decisions": decisions})` |
| 多工具审批 | `decisions` 与 `action_requests` 同顺序 |
| 自定义 `interrupt()` | `Command(resume=<custom_payload>)` |
| filesystem permission interrupt | 同工具审批中断 |

### Permission 配置速查

| 配置 | 含义 |
|------|------|
| `operations=["read"]` | 控制 `ls`, `read_file`, `glob`, `grep` |
| `operations=["write"]` | 控制 `write_file`, `edit_file` |
| `paths=["/workspace/**"]` | 匹配 workspace 下全部路径 |
| `mode="allow"` | 允许匹配操作 |
| `mode="deny"` | 拒绝匹配操作 |
| `mode="interrupt"` | 暂停人工审批 |
| 无匹配规则 | 默认允许 |

### Permission 顺序速查

| 顺序 | 示例 | 目的 |
|------|------|------|
| 1 | deny `/workspace/.env` | 先保护特定敏感文件 |
| 2 | allow `/workspace/**` | 再开放工作区 |
| 3 | deny `/**` | 最后拒绝其他路径 |

### 权限边界速查

| 对象 | Permissions 是否覆盖 |
|------|----------------------|
| `ls` | 是 |
| `read_file` | 是 |
| `glob` | 是 |
| `grep` | 是 |
| `write_file` | 是 |
| `edit_file` | 是 |
| custom tools | 否 |
| MCP tools | 否 |
| sandbox `execute` | 否 |
| backend policy hooks | 另一层机制 |

### 风险分层速查

| 风险 | 示例 | 推荐 |
|------|------|------|
| 高 | delete、send_email、execute、deploy | `approve/edit/reject` 或 `approve/reject` |
| 中 | write_file、edit_file、database update | 条件中断或 permission interrupt |
| 低 | read_file、ls、pure search | 通常不中断 |
| 特殊 | ask_user | 可用 `respond` |

### 一句话总结

```text
Human-in-the-loop 让 Deep Agents 在敏感工具调用前暂停；
checkpointer 和相同 thread_id 是恢复执行的前提；
approve/edit/reject/respond 的语义要区分清楚；
条件中断和 filesystem permission interrupt 可以减少无意义审批；
permissions 用路径规则限制内置文件工具；
HITL 是审批机制，不是 sandbox。
```

---

## 资料来源

- Deep Agents Human-in-the-loop 文档：附件 `3ec55931-e691-4211-924b-b390bdb08464/pasted-text.txt`。
- Deep Agents Permissions 文档：附件 `f814e122-c082-4b31-8c3d-4bf9d2b66de0/pasted-text.txt`。
