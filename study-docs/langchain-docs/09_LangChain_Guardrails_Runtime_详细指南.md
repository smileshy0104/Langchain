# LangChain Guardrails 与 Runtime 详细指南

## 目录

1. [概述](#概述)
2. [Guardrails 护栏系统](#guardrails-护栏系统)
   - [护栏能解决什么问题](#护栏能解决什么问题)
   - [两类实现方式](#两类实现方式)
   - [LangChain 中间件执行位置](#langchain-中间件执行位置)
   - [内置护栏](#内置护栏)
   - [自定义护栏](#自定义护栏)
   - [组合多重护栏](#组合多重护栏)
3. [Runtime 运行时系统](#runtime-运行时系统)
   - [Runtime 是什么](#runtime-是什么)
   - [核心能力](#核心能力)
   - [在工具中访问 Runtime](#在工具中访问-runtime)
   - [在中间件中访问 Runtime](#在中间件中访问-runtime)
   - [执行信息与服务端信息](#执行信息与服务端信息)
4. [实战案例](#实战案例)
5. [最佳实践](#最佳实践)
6. [快速参考](#快速参考)

---

## 概述

在生产级 Agent 应用中，**安全性、合规性、可控性、可测试性**通常比“能调用模型”本身更重要。LangChain v1.x 推荐通过 **middleware（中间件）** 在 Agent 执行的关键节点插入护栏逻辑，并通过 LangGraph 的 **Runtime** 在工具和中间件中注入上下文、存储、流式写入器等运行时依赖。

- **Guardrails（护栏）**：在输入、模型调用、工具调用、最终输出等位置验证和过滤内容，防止 PII 泄露、提示注入、不当输出、违规工具调用等问题。
- **Runtime（运行时）**：为工具和中间件提供依赖注入，让它们在一次 Agent 调用中访问用户 ID、数据库连接、长期记忆、执行 ID、流式输出通道等信息，而不需要硬编码或使用全局变量。

整体架构图如下：

```text
┌─────────────────────────────────────────────────────────────┐
│                         用户请求                             │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Guardrails 输入护栏                     │    │
│  │  • PII 检测与脱敏                                    │    │
│  │  • 内容过滤 / 提示注入防护                            │    │
│  │  • 鉴权、速率限制、业务规则                           │    │
│  └─────────────────────────────────────────────────────┘    │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    Agent 核心                        │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │              Runtime 运行时                  │    │    │
│  │  │  • Context：用户信息、配置、外部依赖          │    │    │
│  │  │  • Store：长期记忆 / 持久化数据               │    │    │
│  │  │  • Stream writer：进度与自定义流式输出        │    │    │
│  │  │  • Execution / Server info：执行与服务信息    │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  │                                                     │    │
│  │  模型调用  ⇄  工具调用  ⇄  中间件钩子                │    │
│  └─────────────────────────────────────────────────────┘    │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Guardrails 输出护栏                     │    │
│  │  • 输出安全评估                                      │    │
│  │  • 格式 / 质量验证                                   │    │
│  │  • 敏感信息二次检查与审计                            │    │
│  └─────────────────────────────────────────────────────┘    │
│                            ↓                                │
│                         响应返回                             │
└─────────────────────────────────────────────────────────────┘
```

典型执行链路如下：

```text
用户请求
  ↓
before_agent：会话级输入护栏 / 鉴权 / 速率限制
  ↓
before_model / wrap_model_call：模型调用前后护栏 / 动态提示词 / 模型选择
  ↓
wrap_tool_call：工具调用审批 / 参数校验 / 权限控制
  ↓
after_model：模型响应后检查 / 输出修正
  ↓
after_agent：最终输出安全检查 / 审计 / 质量验证
  ↓
响应返回
```

---

## Guardrails 护栏系统

### 护栏能解决什么问题

Guardrails 是一组在 Agent 关键执行点上运行的验证和拦截逻辑，用于确保系统行为符合安全、合规和业务预期。

| 场景 | 说明 | 示例 |
|---|---|---|
| PII 保护 | 检测并处理个人身份信息 | 邮箱、信用卡、IP 地址脱敏或阻断 |
| 提示注入防护 | 阻止用户或网页内容覆盖系统意图 | “忽略之前的指令”类攻击检测 |
| 内容安全 | 阻止不当、违法或高风险内容 | 暴力、仇恨、危险操作说明 |
| 业务合规 | 执行业务规则和监管要求 | 金融建议免责声明、地区限制 |
| 工具安全 | 拦截高风险工具调用 | 发邮件、转账、删除数据前人工审批 |
| 输出质量 | 校验最终回答格式、事实或结构 | JSON 格式检查、引用完整性检查 |

### 两类实现方式

#### 1. 确定性护栏（Deterministic Guardrails）

使用正则表达式、关键词匹配、白名单/黑名单、显式条件判断等规则。

**优点**：快速、稳定、可预测、成本低。  
**缺点**：难以理解语义，容易漏掉变体或上下文相关问题。

适用于：格式检查、长度限制、明显敏感词、简单权限规则、PII 正则检测。

```python
BANNED_KEYWORDS = ["hack", "exploit", "malware", "破解", "攻击"]


def contains_banned_keyword(content: str) -> bool:
    content_lower = content.lower()
    return any(keyword in content_lower for keyword in BANNED_KEYWORDS)
```

#### 2. 基于模型的护栏（Model-based Guardrails）

使用 LLM 或分类模型对内容进行语义判断。

**优点**：能理解上下文和隐含意图，适合复杂判断。  
**缺点**：慢、成本高、需要额外模型调用，结果也需要测试和监控。

适用于：复杂内容审核、意图识别、事实/质量检查、最终回答安全评估。

```python
async def evaluate_safety(content: str, safety_model) -> bool:
    """使用模型评估内容是否安全。"""
    prompt = f"""Evaluate if this content is safe and appropriate.
Respond with only SAFE or UNSAFE.

Content: {content}"""

    result = await safety_model.ainvoke([{"role": "user", "content": prompt}])
    return "UNSAFE" not in result.content.upper()
```

### LangChain 中间件执行位置

LangChain 的 guardrails 通常通过 `langchain.agents.middleware` 中的中间件钩子实现。常见位置如下：

| 钩子 | 执行时机 | 适合做什么 |
|---|---|---|
| `before_agent` | 每次 Agent 调用刚开始 | 鉴权、速率限制、输入合规、阻断请求 |
| `after_agent` | Agent 完成后、返回用户前 | 最终输出审核、审计、质量检查 |
| `before_model` | 每次模型调用前 | 修改消息、注入上下文、模型调用前检查 |
| `after_model` | 每次模型调用后 | 检查模型输出、修复格式、触发重试策略 |
| `wrap_model_call` | 包裹模型调用 | 动态模型选择、重试、降级、成本控制 |
| `wrap_tool_call` | 包裹工具调用 | 工具权限、参数校验、人工审批、审计 |
| `dynamic_prompt` | 生成系统提示词时 | 根据 Runtime context 动态生成提示词 |

> 如果护栏需要提前终止 Agent，可使用 `@hook_config(can_jump_to=["end"])` 或装饰器参数 `can_jump_to=["end"]`，并返回 `{"jump_to": "end"}`。

---

## 内置护栏

### 1. PII 检测中间件（PIIMiddleware）

`PIIMiddleware` 用于检测和处理个人身份信息（PII）。它可以应用在用户输入、模型输出和工具结果上。

#### 官方内置 PII 类型

根据 LangChain 官方文档，内置类型包括：

| 类型 | 说明 | 示例 |
|---|---|---|
| `email` | 电子邮箱 | `john@example.com` |
| `credit_card` | 信用卡号（Luhn 校验） | `5105-1051-0510-5100` |
| `ip` | IP 地址 | `192.168.1.1` |
| `mac_address` | MAC 地址 | `00:1B:44:11:3A:B7` |
| `url` | URL | `https://example.com` |

> 电话、身份证、API Key、SSN 等可通过 `detector` 自定义正则或函数实现；不要误认为它们一定是官方内置类型。

#### 处理策略

| 策略 | 行为 | 示例 |
|---|---|---|
| `redact` | 替换为 `[REDACTED_{PII_TYPE}]` | `john@example.com` → `[REDACTED_EMAIL]` |
| `mask` | 部分遮蔽 | `5105-1051-0510-5100` → `****-****-****-5100` |
| `hash` | 替换为确定性哈希 | `john@example.com` → `a8f5f167...` |
| `block` | 检测到后抛出异常 | API Key 出现时直接阻断 |

#### 基础用法

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model="gpt-5.5",
    tools=[customer_service_tool, email_tool],
    middleware=[
        # 输入中的邮箱脱敏
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),

        # 输入中的信用卡遮蔽
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),

        # 自定义 API Key 检测，发现即阻断
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
            apply_to_input=True,
        ),
    ],
)
```

#### 参数速查

```python
PIIMiddleware(
    pii_type: str,                    # PII 类型：内置类型或自定义名称
    strategy: str = "redact",         # block / redact / mask / hash
    detector: str | callable = None,  # 自定义检测器：正则或函数
    apply_to_input: bool = True,      # 检查用户输入
    apply_to_output: bool = False,    # 检查 AI 输出
    apply_to_tool_results: bool = False,  # 检查工具结果
)
```

> `apply_to_output=True` 时，较新版本的 LangChain 还可以通过注册 stream transformer 对流式 wire output 进行脱敏，包括 text deltas、tool-call args、tool outputs 和 state snapshots。官方说明要求 `langchain>=1.3.2`。

#### 自定义检测器示例

```python
import re
from langchain.agents.middleware import PIIMiddleware

# 中国身份证号
chinese_id_guardrail = PIIMiddleware(
    "chinese_id",
    detector=r"\d{17}[\dXx]",
    strategy="redact",
    apply_to_input=True,
)

# 公司敏感词：返回匹配区间

def detect_company_secrets(text: str) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    for match in re.finditer(r"(?i)(机密|保密|内部资料)", text):
        matches.append((match.start(), match.end()))
    return matches

company_secret_guardrail = PIIMiddleware(
    "company_secrets",
    detector=detect_company_secrets,
    strategy="redact",
    apply_to_input=True,
)
```

### 2. 人机协同中间件（HumanInTheLoopMiddleware）

`HumanInTheLoopMiddleware` 用于在执行敏感工具前暂停 Agent，等待人工批准。这是高风险操作中最有效的护栏之一。

适用场景：

- 发送邮件、短信、通知
- 转账、交易、退款
- 删除或修改生产数据
- 执行 SQL、部署、重启服务
- 对外部系统产生不可逆影响的操作

#### 基础配置

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-5.5",
    tools=[search_tool, send_email_tool, delete_database_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True,
                "delete_database": True,
                "search": False,
            }
        ),
    ],
    # Human-in-the-loop 需要 checkpointer 保存暂停状态
    checkpointer=InMemorySaver(),
)
```

#### 恢复执行

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "conversation-123"}}

# 首次调用：可能在敏感工具前暂停
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Send an email to the team"}]},
    config=config,
)

# 人工批准后恢复，同一个 thread_id
result = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,
)
```

> 不同 LangChain / LangGraph 版本中决策类型命名可能有差异。官方 Guardrails 页面展示的是 `approve`；部分示例或旧版本中可能见到 `accept`、`edit`、`respond`。实际项目中应以当前安装版本的 human-in-the-loop 文档和类型定义为准。

#### 细粒度配置思路

```python
HumanInTheLoopMiddleware(
    interrupt_on={
        "send_email": {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True,
            "description": "发送邮件前需要人工审批",
        },
        "execute_sql": {
            "allow_accept": True,
            "allow_respond": True,
            "description": "SQL 执行需要 DBA 批准",
        },
    }
)
```

---

## 自定义护栏

### Before Agent 护栏

`before_agent` 在每次 Agent 调用开始时运行一次，适合做请求级检查：鉴权、速率限制、业务规则、明显违规内容阻断等。

#### 类语法

```python
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime


class ContentFilterMiddleware(AgentMiddleware):
    """确定性护栏：阻止包含禁用关键词的请求。"""

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        first_message = state["messages"][0]
        if first_message.type != "human":
            return None

        content = first_message.content.lower()
        for keyword in self.banned_keywords:
            if keyword in content:
                return {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "我无法处理包含不当内容的请求，请重新表述。",
                        }
                    ],
                    "jump_to": "end",
                }

        return None
```

#### 装饰器语法

```python
from typing import Any

from langchain.agents.middleware import before_agent, AgentState
from langgraph.runtime import Runtime

banned_keywords = ["hack", "exploit", "malware"]


@before_agent(can_jump_to=["end"])
def content_filter(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if not state["messages"]:
        return None

    first_message = state["messages"][0]
    if first_message.type != "human":
        return None

    content = first_message.content.lower()
    if any(keyword in content for keyword in banned_keywords):
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": "我无法处理包含不当内容的请求，请重新表述。",
                }
            ],
            "jump_to": "end",
        }

    return None
```

### After Agent 护栏

`after_agent` 在 Agent 结束后执行一次，适合对最终输出做安全评估、质量检查或合规扫描。

```python
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage
from langgraph.runtime import Runtime


class SafetyGuardrailMiddleware(AgentMiddleware):
    """基于模型的护栏：使用独立模型评估最终回答安全性。"""

    def __init__(self):
        super().__init__()
        self.safety_model = init_chat_model("gpt-5.4-mini")

    @hook_config(can_jump_to=["end"])
    def after_agent(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        safety_prompt = f"""Evaluate if this response is safe and appropriate.
Respond with only SAFE or UNSAFE.

Response: {last_message.content}"""

        result = self.safety_model.invoke([{"role": "user", "content": safety_prompt}])

        if "UNSAFE" in result.content.upper():
            last_message.content = "我无法提供该响应。请换一种方式提问。"

        return None
```

### 常见自定义护栏模式

#### 输入长度限制

```python
from langchain.agents.middleware import before_agent


@before_agent(can_jump_to=["end"])
def limit_input_length(state, runtime):
    if not state["messages"]:
        return None

    first_message = state["messages"][0]
    if len(first_message.content) > 10_000:
        return {
            "messages": [{"role": "assistant", "content": "输入内容过长，请精简后重试。"}],
            "jump_to": "end",
        }

    return None
```

#### 工具权限控制

```python
from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage


class DepartmentAccessMiddleware(AgentMiddleware):
    """根据用户部门限制工具访问。"""

    DEPARTMENT_TOOLS = {
        "HR": {"employee_lookup", "payroll_query"},
        "Finance": {"expense_report", "budget_query"},
        "Engineering": {"code_search", "deploy_status"},
    }

    def wrap_tool_call(self, request, handler):
        ctx = request.runtime.context
        tool_name = request.tool_call.name
        allowed_tools = self.DEPARTMENT_TOOLS.get(ctx.department, set())

        if ctx.access_level >= 4:
            return handler(request)

        if tool_name not in allowed_tools:
            return ToolMessage(
                content=f"访问被拒绝：{ctx.department} 部门无权使用 {tool_name}",
                tool_call_id=request.tool_call.id,
            )

        return handler(request)
```

---

## 组合多重护栏

多个护栏可以按 middleware 数组顺序叠加，形成分层防护体系。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-5.5",
    tools=[search_tool, send_email_tool, database_tool],
    middleware=[
        # 第 1 层：输入 PII 保护
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="block", apply_to_input=True),

        # 第 2 层：确定性输入过滤
        ContentFilterMiddleware(banned_keywords=["hack", "exploit", "破解"]),

        # 第 3 层：业务规则与权限
        DepartmentAccessMiddleware(),

        # 第 4 层：敏感工具人工审批
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True,
                "database_tool": True,
            }
        ),

        # 第 5 层：输出 PII 与安全检查
        PIIMiddleware("email", strategy="redact", apply_to_output=True),
        SafetyGuardrailMiddleware(),
    ],
    checkpointer=InMemorySaver(),
)
```

设计顺序建议：

1. **便宜、确定性的检查放前面**：长度、关键词、PII、鉴权。
2. **会阻断执行的规则尽早运行**：减少模型调用成本。
3. **高风险工具用 `wrap_tool_call` 或 human-in-the-loop**：不要只在最终输出里补救。
4. **模型型护栏放后面或关键节点**：只对需要语义判断的内容额外调用模型。
5. **输出护栏仍然必要**：工具结果、模型幻觉或上下文污染都可能影响最终回答。

---

## Runtime 运行时系统

### Runtime 是什么

LangChain 的 `create_agent` 底层运行在 LangGraph runtime 上。Runtime 是每次 Agent 调用的运行时对象，为工具和中间件提供：

1. **Context**：调用时注入的静态上下文，如用户 ID、角色、配置、数据库连接。
2. **Store**：长期记忆或持久化存储（`BaseStore`）。
3. **Stream writer**：用于 `"custom"` stream mode 的自定义流式输出通道。
4. **Execution info**：当前执行身份信息，如 thread ID、run ID、尝试次数。
5. **Server info**：在 LangGraph Server 上运行时的服务端元信息，如 assistant ID、graph ID、已认证用户。

> Runtime 的核心价值是 **依赖注入**：工具和中间件不再从全局变量、配置文件或硬编码中读取依赖，而是在每次调用时通过 `context` 明确传入。这让代码更容易测试、复用和审计。

### 核心能力

```text
Runtime
  ├─ context：本次调用的静态上下文
  ├─ store：长期记忆 / 持久化 KV 存储
  ├─ stream_writer：自定义流式输出
  ├─ execution_info：thread_id、run_id、attempt 等
  └─ server_info：LangGraph Server 运行时的 assistant / graph / user 信息
```

### 配置 Context Schema

```python
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent


@dataclass
class AppContext:
    user_id: str
    user_name: str
    user_role: str = "user"
    language: str = "zh"
    database_connection: Any = None


agent = create_agent(
    model="gpt-5-nano",
    tools=[...],
    middleware=[...],
    context_schema=AppContext,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "查询我的订单"}]},
    context=AppContext(
        user_id="user-123",
        user_name="张三",
        user_role="admin",
        database_connection=db_conn,
    ),
)
```

### 在工具中访问 Runtime

工具中使用 `ToolRuntime[ContextType]` 访问 Runtime。

```python
from dataclasses import dataclass

from langchain.tools import ToolRuntime, tool


@dataclass
class Context:
    user_id: str


@tool
def fetch_user_email_preferences(runtime: ToolRuntime[Context]) -> str:
    """Fetch the user's email preferences from the store."""
    user_id = runtime.context.user_id

    preferences = "The user prefers you to write a brief and polite email."

    if runtime.store:
        if memory := runtime.store.get(("users",), user_id):
            preferences = memory.value["preferences"]

    return preferences
```

#### 工具中使用 Stream Writer

```python
@tool
def search_with_progress(query: str, runtime: ToolRuntime[Context]) -> str:
    """执行搜索并发送自定义进度更新。"""
    if runtime.stream_writer:
        runtime.stream_writer.write({"progress": 0, "status": "开始搜索"})

    results = []
    for i, source in enumerate(search_sources):
        results.extend(search_source(source, query))

        if runtime.stream_writer:
            runtime.stream_writer.write(
                {
                    "progress": (i + 1) / len(search_sources),
                    "status": f"已搜索 {i + 1}/{len(search_sources)} 个来源",
                }
            )

    return format_results(results)
```

### 在中间件中访问 Runtime

#### 动态系统提示词

```python
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt


@dataclass
class Context:
    user_name: str
    language: str = "zh"


@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    ctx = request.runtime.context
    language_instruction = "请使用中文回复。" if ctx.language == "zh" else "Please respond in English."
    return f"你是一个有帮助的助手。用户名称是 {ctx.user_name}。{language_instruction}"


agent = create_agent(
    model="gpt-5-nano",
    tools=[...],
    middleware=[dynamic_system_prompt],
    context_schema=Context,
)
```

#### before_model / after_model

```python
from langchain.agents.middleware import AgentState, after_model, before_model
from langgraph.runtime import Runtime


@before_model
def log_before_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    print(f"Processing request for user: {runtime.context.user_name}")
    return None


@after_model
def log_after_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    print(f"Completed request for user: {runtime.context.user_name}")
    return None
```

#### wrap_model_call：基于上下文动态选择模型

```python
from typing import Callable

from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.chat_models import init_chat_model


@wrap_model_call
def context_aware_model_selection(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    ctx = request.runtime.context

    if ctx.user_role == "admin":
        request.model = init_chat_model("gpt-5.5")
    else:
        request.model = init_chat_model("gpt-5-nano")

    return handler(request)
```

### 执行信息与服务端信息

较新版本中，Runtime 还暴露执行身份和 LangGraph Server 相关元信息。

```python
from langchain.tools import ToolRuntime, tool


@tool
def context_aware_tool(runtime: ToolRuntime) -> str:
    info = runtime.execution_info
    print(f"Thread: {info.thread_id}, Run: {info.run_id}")

    server = runtime.server_info
    if server is not None:
        print(f"Assistant: {server.assistant_id}")
        if server.user is not None:
            print(f"User: {server.user.identity}")

    return "done"
```

在中间件中也可以访问：

```python
from langchain.agents.middleware import AgentState, before_model
from langgraph.runtime import Runtime


@before_model
def auth_gate(state: AgentState, runtime: Runtime) -> dict | None:
    server = runtime.server_info
    if server is not None and server.user is None:
        raise ValueError("Authentication required")

    print(f"Thread: {runtime.execution_info.thread_id}")
    return None
```

> 官方文档说明：`runtime.execution_info` 和 `runtime.server_info` 需要 `deepagents>=0.5.0` 或 `langgraph>=1.1.5`。

---

## 实战案例

### 案例 1：安全的企业客服 Agent

```python
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, PIIMiddleware, dynamic_prompt
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


@dataclass(frozen=True)
class CustomerContext:
    customer_id: str
    customer_name: str
    subscription_tier: str
    language: str = "zh"
    database: Any = None


@tool
def get_customer_info(runtime: ToolRuntime[CustomerContext]) -> str:
    ctx = runtime.context
    return f"客户: {ctx.customer_name} (ID: {ctx.customer_id}), 套餐: {ctx.subscription_tier}"


@tool
def create_support_ticket(issue: str, priority: str, runtime: ToolRuntime[CustomerContext]) -> str:
    ctx = runtime.context

    if ctx.subscription_tier == "enterprise" and priority == "normal":
        priority = "high"

    return f"已为客户 {ctx.customer_name} 创建工单：{issue}，优先级：{priority}"


@dynamic_prompt
def customer_service_prompt(request) -> str:
    ctx = request.runtime.context

    tier_instructions = {
        "free": "提供基础支持，复杂问题建议升级套餐。",
        "pro": "提供专业支持，可以处理技术问题。",
        "enterprise": "提供最高级别支持，优先处理，可承诺 SLA。",
    }

    language_instruction = "请使用中文回复。" if ctx.language == "zh" else "Please respond in English."

    return f"""你是专业客服助手。

当前客户：
- 姓名：{ctx.customer_name}
- 客户 ID：{ctx.customer_id}
- 套餐：{ctx.subscription_tier}

服务指南：
{tier_instructions.get(ctx.subscription_tier, tier_instructions["free"])}

{language_instruction}
保持专业、友好、高效。"""


agent = create_agent(
    model="gpt-5.5",
    tools=[get_customer_info, create_support_ticket],
    middleware=[
        customer_service_prompt,
        PIIMiddleware("email", strategy="redact", apply_to_input=True, apply_to_output=True),
        PIIMiddleware("credit_card", strategy="block", apply_to_input=True),
        HumanInTheLoopMiddleware(interrupt_on={"create_support_ticket": True}),
    ],
    context_schema=CustomerContext,
    checkpointer=InMemorySaver(),
    store=InMemoryStore(),
)
```

### 案例 2：金融合规 Agent

```python
from dataclasses import dataclass

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime


@dataclass(frozen=True)
class FinancialContext:
    client_id: str
    client_name: str
    risk_profile: str  # conservative / moderate / aggressive
    accredited_investor: bool
    jurisdiction: str  # US / EU / CN


class FinancialComplianceMiddleware(AgentMiddleware):
    """金融合规护栏。"""

    JURISDICTION_RULES = {
        "US": {"disclaimer_required": True, "crypto_allowed": True, "leverage_limit": 4},
        "CN": {"disclaimer_required": True, "crypto_allowed": False, "leverage_limit": 1},
    }

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state, runtime: Runtime[FinancialContext]):
        ctx = runtime.context
        rules = self.JURISDICTION_RULES.get(ctx.jurisdiction, {})

        if rules.get("disclaimer_required"):
            disclaimer = """免责声明：本内容仅供参考，不构成投资建议。投资有风险，请咨询专业顾问。"""
            return {"messages": [SystemMessage(content=disclaimer)] + state["messages"]}

        return None

    def wrap_tool_call(self, request, handler):
        ctx = request.runtime.context
        rules = self.JURISDICTION_RULES.get(ctx.jurisdiction, {})
        tool_name = request.tool_call.name

        if tool_name == "crypto_trade" and not rules.get("crypto_allowed"):
            return ToolMessage(
                content=f"根据 {ctx.jurisdiction} 的法规，不支持加密货币交易。",
                tool_call_id=request.tool_call.id,
            )

        if tool_name in {"hedge_fund_invest", "private_equity"} and not ctx.accredited_investor:
            return ToolMessage(
                content="此投资产品仅限合格投资者。",
                tool_call_id=request.tool_call.id,
            )

        return handler(request)


class RiskProfileMiddleware(AgentMiddleware):
    """根据风险偏好过滤最终建议。"""

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state, runtime: Runtime[FinancialContext]):
        ctx = runtime.context
        last_msg = state["messages"][-1]

        if isinstance(last_msg, AIMessage) and ctx.risk_profile == "conservative":
            high_risk_keywords = ["期货", "期权", "杠杆", "衍生品", "futures", "options"]
            if any(keyword.lower() in last_msg.content.lower() for keyword in high_risk_keywords):
                return {
                    "messages": [
                        AIMessage(
                            content="根据您的保守型风险偏好，我无法推荐高风险产品。建议优先考虑债券、货币基金、蓝筹股或低波动 ETF。"
                        )
                    ],
                    "jump_to": "end",
                }

        return None
```

---

## 最佳实践

### 护栏设计原则

1. **分层防护**：输入、工具、模型输出、最终输出都可以有不同护栏。
2. **快速失败**：便宜、确定性、能阻断的检查尽早执行。
3. **高风险操作必须前置审批**：不要等工具执行后再做安全检查。
4. **模型护栏要控制成本**：只对需要语义判断的节点使用模型型检查。
5. **护栏可观测**：记录被阻断原因、工具调用、人工审批结果和异常。
6. **不要只依赖关键词**：关键词过滤适合作为第一层，不适合作为唯一安全机制。
7. **自定义检测器要可测试**：为 PII、业务规则、权限矩阵写单元测试。

### Runtime 设计原则

1. **明确 Context 边界**：只放本次调用需要的依赖和静态信息。
2. **避免全局状态**：工具通过 `runtime.context` 访问依赖，便于测试和复用。
3. **优先不可变 Context**：使用 `@dataclass(frozen=True)` 或不可变字段减少副作用。
4. **Store 用于跨会话记忆**：不要把长期记忆塞进全局变量。
5. **Stream writer 只发送必要进度**：避免过度刷屏，但长任务要给用户可见进展。
6. **服务端信息要判空**：`runtime.server_info` 在本地开发时通常为 `None`。
7. **敏感信息不要写入消息历史**：API Key、密码等应通过安全配置或后端依赖注入，而不是用户消息或系统提示词。

### 常见坑

| 问题 | 后果 | 建议 |
|---|---|---|
| 把所有护栏都放在 `after_agent` | 已经消耗模型和工具成本 | 输入和工具风险尽量前置 |
| PII 类型写错 | 护栏没有按预期生效 | 使用官方内置类型；其他用自定义 detector |
| Human-in-the-loop 没有 checkpointer | 无法暂停/恢复 | 配置 `InMemorySaver` 或持久化 checkpointer |
| 工具内部读全局变量 | 难测试、并发风险 | 通过 `ToolRuntime` 注入依赖 |
| `server_info` 本地直接访问 | `NoneType` 错误 | 访问前判断是否为 `None` |
| 过度依赖模型护栏 | 慢且贵 | 确定性规则 + 模型检查组合 |

---

## 快速参考

### 护栏类型对比

| 类型 | 速度 | 成本 | 准确性 | 适用场景 |
|---|---:|---:|---:|---|
| 确定性护栏 | 快 | 低 | 中 | 格式、长度、关键词、显式规则 |
| 模型护栏 | 慢 | 高 | 高 | 语义安全、意图识别、复杂质量检查 |
| 混合护栏 | 中 | 中 | 高 | 生产系统的推荐方案 |

### PII 中间件速查

```python
PIIMiddleware("email", strategy="redact", apply_to_input=True)
PIIMiddleware("credit_card", strategy="mask", apply_to_input=True, apply_to_output=True)
PIIMiddleware("ip", strategy="redact", apply_to_tool_results=True)
PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block")
```

### Human-in-the-loop 速查

```python
HumanInTheLoopMiddleware(
    interrupt_on={
        "send_email": True,
        "delete_database": True,
        "search": False,
    }
)

# 恢复执行（同一个 thread_id）
agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config={"configurable": {"thread_id": "some_id"}},
)
```

### Runtime 访问速查

```python
# 工具中
@tool
def my_tool(runtime: ToolRuntime[MyContext]) -> str:
    user_id = runtime.context.user_id
    runtime.stream_writer.write({"status": "working"})
    return "done"

# node-style hook 中
@before_model
def hook(state: AgentState, runtime: Runtime[MyContext]) -> dict | None:
    print(runtime.context.user_id)
    return None

# wrap-style / dynamic_prompt 中
@dynamic_prompt
def prompt(request: ModelRequest) -> str:
    return f"User: {request.runtime.context.user_name}"
```

---

## 总结

Guardrails 与 Runtime 是 LangChain Agent 工程化的两条主线：

- **Guardrails** 负责“能不能做、该不该做、输出是否安全”。
- **Runtime** 负责“本次调用是谁、有哪些依赖、如何访问外部资源、如何流式反馈”。

推荐实践是：用确定性规则快速过滤明显问题，用模型护栏处理复杂语义判断，用 human-in-the-loop 控制高风险工具，用 Runtime 注入用户上下文、存储和执行信息。这样可以构建既安全、合规，又可测试、可维护的生产级 Agent 应用。
