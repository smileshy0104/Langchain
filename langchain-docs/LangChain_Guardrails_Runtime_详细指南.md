# LangChain Guardrails 与 Runtime 详细指南

## 目录

1. [概述](#概述)
2. [Guardrails 护栏系统](#guardrails-护栏系统)
   - [什么是护栏](#什么是护栏)
   - [两种护栏类型](#两种护栏类型)
   - [内置护栏](#内置护栏)
   - [自定义护栏](#自定义护栏)
   - [组合多重护栏](#组合多重护栏)
3. [Runtime 运行时系统](#runtime-运行时系统)
   - [什么是 Runtime](#什么是-runtime)
   - [核心组件](#核心组件)
   - [配置与使用](#配置与使用)
   - [依赖注入](#依赖注入)
4. [实战案例](#实战案例)
5. [最佳实践](#最佳实践)
6. [快速参考](#快速参考)

---

## 概述

在构建生产级 AI 应用时，**安全性**和**可控性**是两个核心挑战：

- **Guardrails（护栏）**：确保 AI 系统的输入和输出符合安全、合规要求
- **Runtime（运行时）**：提供依赖注入和上下文管理，让 Agent 能够访问外部资源

这两个系统协同工作，让你能够构建既安全又灵活的 AI 应用。

```
┌─────────────────────────────────────────────────────────────┐
│                      用户请求                                │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Guardrails 输入护栏                        │   │
│  │  • PII 检测与脱敏                                    │   │
│  │  • 内容过滤                                          │   │
│  │  • 提示注入防护                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Agent 核心                              │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │           Runtime 运行时                     │   │   │
│  │  │  • Context 上下文（用户信息、配置）          │   │   │
│  │  │  • Store 长期记忆存储                        │   │   │
│  │  │  • Stream 流式输出                           │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Guardrails 输出护栏                        │   │
│  │  • 安全性评估                                        │   │
│  │  • 内容审核                                          │   │
│  │  • 人工审批                                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│                      响应返回                                │
└─────────────────────────────────────────────────────────────┘
```

---

## Guardrails 护栏系统

### 什么是护栏

**Guardrails（护栏）** 是一套安全机制，用于在 AI 应用的关键执行点验证和过滤内容。就像高速公路上的护栏防止车辆偏离道路一样，AI 护栏确保系统行为不会偏离预期轨道。

**核心应用场景：**

| 场景 | 说明 | 示例 |
|------|------|------|
| **PII 保护** | 防止个人身份信息泄露 | 脱敏邮箱、电话、身份证号 |
| **提示注入防护** | 检测和阻止恶意提示 | 阻止"忽略之前的指令"类攻击 |
| **内容过滤** | 阻止不当或有害内容 | 过滤暴力、色情、仇恨言论 |
| **业务合规** | 强制执行业务规则 | 限制讨论竞品、法律建议 |
| **输出验证** | 确保响应质量和准确性 | 检查格式、事实核查 |

### 两种护栏类型

#### 1. 确定性护栏（Deterministic Guardrails）

使用基于规则的逻辑进行检查，如正则表达式、关键词匹配或显式条件判断。

**特点：**
- ✅ **快速** - 毫秒级响应
- ✅ **可预测** - 结果一致，易于测试
- ✅ **成本低** - 不需要额外的 API 调用
- ❌ **局限性** - 可能遗漏细微或变体的违规

**适用场景：**
- 敏感关键词过滤
- 格式验证（邮箱、电话等）
- 长度限制检查
- 简单的黑名单/白名单

```python
# 确定性护栏示例：关键词过滤
BANNED_KEYWORDS = ["hack", "exploit", "malware", "破解", "攻击"]

def check_banned_keywords(content: str) -> bool:
    """检查是否包含禁用关键词"""
    content_lower = content.lower()
    for keyword in BANNED_KEYWORDS:
        if keyword in content_lower:
            return True  # 发现违规
    return False  # 内容安全
```

#### 2. 基于模型的护栏（Model-based Guardrails）

使用 LLM 或专门的分类器进行语义理解和评估。

**特点：**
- ✅ **智能** - 能理解上下文和语义
- ✅ **灵活** - 可以捕获规则遗漏的细微问题
- ✅ **适应性** - 无需频繁更新规则
- ❌ **速度较慢** - 需要额外的模型调用
- ❌ **成本较高** - 消耗额外的 API 配额

**适用场景：**
- 复杂的内容审核
- 意图识别
- 上下文相关的安全检查
- 输出质量评估

```python
# 基于模型的护栏示例：安全性评估
async def evaluate_safety(content: str, model) -> bool:
    """使用 LLM 评估内容安全性"""
    prompt = f"""评估以下内容是否安全和适当。
    只回复 'SAFE' 或 'UNSAFE'。

    内容: {content}"""

    result = await model.invoke([{"role": "user", "content": prompt}])
    return "UNSAFE" not in result.content
```

### 内置护栏

LangChain 提供了开箱即用的护栏中间件。

#### 1. PII 检测中间件（PIIMiddleware）

自动检测和处理个人身份信息（Personally Identifiable Information）。

**支持的 PII 类型：**

| 类型 | 说明 | 示例 |
|------|------|------|
| `email` | 电子邮箱 | user@example.com |
| `phone_number` | 电话号码 | 138-xxxx-xxxx |
| `credit_card` | 信用卡号 | 4111-1111-1111-1111 |
| `ssn` | 社会保险号 | 123-45-6789 |
| `ip_address` | IP 地址 | 192.168.1.1 |
| `mac_address` | MAC 地址 | 00:1B:44:11:3A:B7 |
| `api_key` | API 密钥 | sk-xxxxxx |

**处理策略：**

| 策略 | 行为 | 输入 | 输出 |
|------|------|------|------|
| `redact` | 替换为标记 | `联系我 test@email.com` | `联系我 [REDACTED_EMAIL]` |
| `mask` | 部分遮蔽 | `卡号 4111111111111111` | `卡号 ****-****-****-1111` |
| `hash` | 替换为哈希 | `用户 test@email.com` | `用户 a8f5f167f44f...` |
| `block` | 抛出异常 | `API密钥是 sk-xxx` | 🚫 抛出错误 |

**基础用法：**

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[customer_service_tool, email_tool],
    middleware=[
        # 脱敏用户输入中的邮箱地址
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True, # 检查输入
        ),

        # 遮蔽信用卡号
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
            apply_to_output=True,  # 同时检查输出
        ),

        # 阻止 API 密钥泄露
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",  # 自定义正则
            strategy="block",
            apply_to_input=True,    # 检查输入
            apply_to_output=True,   # 同时检查输出
        ),
    ],
)
```

**配置参数详解：**

```python
PIIMiddleware(
    pii_type: str,              # PII 类型（内置类型或自定义名称）
    strategy: str = "redact",   # 处理策略：redact/mask/hash/block
    detector: str | callable = None,  # 自定义检测器（正则或函数）
    apply_to_input: bool = True,      # 是否检查用户输入
    apply_to_output: bool = False,    # 是否检查 AI 输出
    apply_to_tool_results: bool = False,  # 是否检查工具返回结果
)
```

**自定义检测器示例：**

```python
import re

# 使用正则表达式检测中国身份证号
PIIMiddleware(
    "chinese_id",
    detector=r"\d{17}[\dXx]",
    strategy="redact",
    apply_to_input=True,
)

# 使用函数检测敏感词
def detect_company_secrets(text: str) -> list[tuple[int, int]]:
    """返回敏感词的位置列表 [(start, end), ...]"""
    pattern = r"(?i)(机密|保密|内部资料)"
    matches = []
    for match in re.finditer(pattern, text):
        matches.append((match.start(), match.end()))
    return matches

PIIMiddleware(
    "company_secrets",
    detector=detect_company_secrets,
    strategy="redact",
    apply_to_input=True,
)
```

#### 2. 人机协同中间件（HumanInTheLoopMiddleware）

在执行敏感操作前要求人工批准，提供关键决策点的人工监督。

**工作流程：**

```
用户请求 → Agent 决定调用工具 → 中间件拦截 → 暂停等待
                                              ↓
                                         人工审核
                                              ↓
                              ┌───────────────┼───────────────┐
                              ↓               ↓               ↓
                           批准            编辑             拒绝
                              ↓               ↓               ↓
                           执行工具      修改后执行      返回反馈
```

**基础配置：**

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, send_email_tool, delete_database_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # 配置哪些工具需要人工批准
                "send_email": True,           # 需要批准，允许所有决策类型（批准、编辑、拒绝）
                "delete_database": True,      # 需要批准，危险操作必须人工审核
                "search": False,              # 自动批准，无需人工干预（安全操作）
            }
        ),
    ],
    # 必须配置 checkpointer 以支持暂停和恢复
    checkpointer=InMemorySaver(),
)
```

**高级配置 - 细粒度控制：**

```python
HumanInTheLoopMiddleware(
    interrupt_on={
        "send_email": {
            "allow_accept": True,    # 允许直接批准
            "allow_edit": True,      # 允许编辑参数
            "allow_respond": True,   # 允许拒绝并反馈
            "description": "发送邮件需要审批",  # 自定义提示
        },
        "execute_sql": {
            "allow_accept": True,
            "allow_respond": True,
            # 不允许编辑 SQL，只能批准或拒绝
            "description": "SQL 执行需要 DBA 批准",
        },
        "delete_file": {
            "allow_accept": True,
            "allow_respond": True,
            # 危险操作，不允许编辑
        },
    },
    description_prefix="工具执行待批准",  # 全局提示前缀
)
```

**完整使用示例：**

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# 1. 创建带有人工审核的 Agent
agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, send_email_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": True}
        ),
    ],
    # 必须配置 checkpointer 以支持暂停和恢复
    checkpointer=InMemorySaver(),
)

# 2. 发起请求（需要线程 ID 以支持暂停/恢复）
config = {"configurable": {"thread_id": "conversation-123"}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "给团队发送项目更新邮件"}]},
    config=config
)

# 3. 检查是否被中断
if result.get("__interrupt__"):
    interrupt_info = result["__interrupt__"]
    print(f"等待批准: {interrupt_info}")

    # 显示待批准的操作详情
    for action in interrupt_info:
        print(f"工具: {action['tool_name']}")
        print(f"参数: {action['tool_args']}")

# 4. 人工做出决策后恢复执行

# 方式 A: 批准执行
result = agent.invoke(
    Command(resume={"decisions": [{"type": "accept"}]}),
    config=config
)

# 方式 B: 编辑参数后执行
result = agent.invoke(
    Command(resume={"decisions": [{
        "type": "edit",
        "args": {"to": "team@company.com", "subject": "修改后的主题"}
    }]}),
    config=config
)

# 方式 C: 拒绝并提供反馈
result = agent.invoke(
    Command(resume={"decisions": [{
        "type": "respond",
        "response": "请先确认收件人列表是否正确"
    }]}),
    config=config
)
```

### 自定义护栏

除了内置护栏，你可以创建完全自定义的护栏来满足特定需求。

#### Before Agent 护栏

在 Agent 开始执行前进行一次性验证，适用于会话级检查。

```python
# 导入必要的类型和类
from typing import Any  # 用于类型注解
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config  # 中间件基类和装饰器
from langchain.messages import AIMessage  # AI 消息类型
from langgraph.runtime import Runtime  # 运行时环境

class ContentFilterMiddleware(AgentMiddleware):
    """确定性护栏：基于关键词的内容过滤
    
    这是一个确定性护栏中间件，用于在 Agent 执行前检查用户输入
    是否包含预定义的禁用关键词。如果发现违规内容，会立即
    阻止执行并返回预设的响应消息。
    """

    def __init__(self, banned_keywords: list[str]):
        """初始化内容过滤中间件
        
        Args:
            banned_keywords: 禁用关键词列表，支持中英文
        """
        super().__init__()  # 调用父类初始化
        # 将所有关键词转换为小写，实现大小写不敏感的匹配
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])  # 装饰器：允许直接跳转到执行结束
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """在 Agent 执行前检查用户输入内容
        
        这是一个 before_agent 钩子方法，会在 Agent 开始处理用户请求
        之前被调用。如果检测到违规内容，可以立即终止执行流程。
        
        Args:
            state: Agent 状态，包含消息历史等信息
            runtime: 运行时环境，提供上下文和资源访问
            
        Returns:
            - None: 内容安全，继续正常执行
            - dict: 包含新消息和跳转指令的字典，用于阻止执行
        """
        # 检查状态中是否有消息
        if not state["messages"]:
            return None  # 没有消息，直接继续

        # 获取第一条用户消息（通常是用户的初始请求）
        first_message = state["messages"][0]
        # 确保这是人类用户的消息，而不是系统消息或 AI 响应
        if first_message.type != "human":
            return None

        # 将消息内容转换为小写，实现大小写不敏感的关键词匹配
        content = first_message.content.lower()

        # 遍历所有禁用关键词进行检查
        for keyword in self.banned_keywords:
            if keyword in content:
                # 发现违规内容，返回阻止响应
                return {
                    "messages": [AIMessage(
                        content="抱歉，我无法处理包含不当内容的请求。"
                    )],
                    "jump_to": "end"  # 直接跳转到结束，不调用 LLM 模型
                }

        return None  # 内容安全，继续执行 Agent

# 使用示例：创建带内容过滤的 Agent
agent = create_agent(
    model="gpt-4o",  # 使用 GPT-4o 模型
    tools=[search_tool, calculator_tool],  # 提供搜索和计算工具
    middleware=[
        # 添加内容过滤中间件
        ContentFilterMiddleware(
            banned_keywords=["hack", "exploit", "破解", "攻击"]  # 禁用的关键词列表
        ),
    ],
)
```

#### After Agent 护栏

在返回给用户前验证最终输出，适用于输出质量和安全检查。

```python
# 导入必要的模块
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config  # 中间件基类和装饰器
from langchain.messages import AIMessage  # AI 消息类型
from langchain_openai import ChatOpenAI  # OpenAI 模型接口
from langgraph.runtime import Runtime  # 运行时环境

class SafetyGuardrailMiddleware(AgentMiddleware):
    """基于模型的护栏：使用 LLM 评估响应安全性
    
    这是一个基于模型的护栏中间件，使用独立的 LLM 模型来评估
    Agent 生成的响应是否安全、适当。相比确定性护栏，它能更好地
    理解上下文和语义，但需要额外的 API 调用。
    """

    def __init__(self, safety_model: str = "gpt-4o-mini"):
        """初始化安全护栏中间件
        
        Args:
            safety_model: 用于安全评估的模型名称，默认使用 gpt-4o-mini
                           选择轻量级模型以降低成本和延迟
        """
        super().__init__()  # 调用父类初始化
        # 创建专门用于安全评估的模型实例
        self.safety_model = ChatOpenAI(model=safety_model)

    @hook_config(can_jump_to=["end"])  # 装饰器：允许直接跳转到执行结束
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """在 Agent 执行完成后评估最终响应的安全性
        
        这是一个 after_agent 钩子方法，会在 Agent 完成处理并生成
        响应后被调用。用于对最终输出进行安全性和适当性评估。
        
        Args:
            state: Agent 状态，包含完整的消息历史
            runtime: 运行时环境，提供上下文和资源访问
            
        Returns:
            - None: 响应安全，正常返回给用户
            - dict: 包含安全提示消息和跳转指令的字典，用于替换不安全响应
        """
        # 检查状态中是否有消息
        if not state["messages"]:
            return None  # 没有消息，直接继续

        # 获取最后一条消息（通常是 Agent 的响应）
        last_message = state["messages"][-1]
        # 确保这是 AI 生成的消息，而不是用户输入或系统消息
        if not isinstance(last_message, AIMessage):
            return None

        # 构建安全评估提示词，指导模型进行安全性检查
        safety_prompt = f"""请评估以下 AI 响应是否安全和适当。

考虑以下方面：
1. 是否包含有害、危险或不当内容
2. 是否泄露敏感信息
3. 是否可能误导用户
4. 是否符合道德标准

只回复 'SAFE' 或 'UNSAFE'。

响应内容:
{last_message.content}"""

        # 调用安全模型进行评估
        result = self.safety_model.invoke([
            {"role": "user", "content": safety_prompt}
        ])

        # 检查评估结果，使用大小写不敏感的匹配
        if "UNSAFE" in result.content.upper():
            # 响应不安全，替换为安全的默认响应
            return {
                "messages": [AIMessage(
                    content="抱歉，我无法提供该响应。请换一种方式提问。"
                )],
                "jump_to": "end"  # 直接跳转到结束，不返回原始响应
            }

        return None  # 响应安全，正常返回给用户

# 使用示例：创建带安全护栏的 Agent
agent = create_agent(
    model="gpt-4o",  # 主要的 Agent 模型
    tools=[...],  # 提供的工具列表
    middleware=[
        # 添加安全护栏中间件，使用轻量级模型进行评估
        SafetyGuardrailMiddleware(safety_model="gpt-4o-mini"),
    ],
)
```

#### 装饰器语法

对于简单的护栏逻辑，可以使用装饰器语法：

```python
# 导入装饰器和必要的类型
from langchain.agents.middleware import before_agent, after_agent, hook_config  # 护栏装饰器
from langchain.messages import AIMessage  # AI 消息类型

# 输入护栏：在 Agent 执行前检查输入
@before_agent(can_jump_to=["end"])  # 装饰器：允许直接跳转到执行结束
def input_length_check(state: AgentState, runtime: Runtime) -> dict | None:
    """限制输入长度 - 防止过长的输入导致性能问题或超出模型限制
    
    这是一个简单的确定性护栏，用于检查用户输入的长度。
    如果输入过长，会立即阻止执行并返回提示信息。
    
    Args:
        state: Agent 状态，包含消息历史
        runtime: 运行时环境
        
    Returns:
        - None: 输入长度正常，继续执行
        - dict: 包含错误消息和跳转指令，阻止执行
    """
    # 检查状态中是否有消息
    if state["messages"]:
        # 获取第一条消息（通常是用户的输入）
        first_msg = state["messages"][0]
        # 检查消息内容长度是否超过限制（10000 字符）
        if len(first_msg.content) > 10000:
            # 输入过长，返回错误信息并结束执行
            return {
                "messages": [AIMessage(content="输入内容过长，请精简后重试。")],
                "jump_to": "end"  # 直接跳转到结束，不调用模型
            }
    return None  # 输入长度正常，继续执行

# 输出护栏：在 Agent 执行后检查输出
@after_agent(can_jump_to=["end"])  # 装饰器：允许直接跳转到执行结束
def output_format_check(state: AgentState, runtime: Runtime) -> dict | None:
    """确保输出格式正确 - 自动修复常见的格式问题
    
    这是一个输出格式检查护栏，用于检测和修复 AI 响应中的
    格式问题。例如未正确关闭的代码块等。
    
    Args:
        state: Agent 状态，包含完整的消息历史
        runtime: 运行时环境
        
    Returns:
        - None: 输出格式正确，正常返回
        - dict: 包含修复后消息的字典，替换原始输出
    """
    # 获取最后一条消息（通常是 AI 的响应）
    last_msg = state["messages"][-1]
    # 确保这是 AI 生成的消息
    if isinstance(last_msg, AIMessage):
        # 检查是否包含未正确关闭的代码块
        if "```" in last_msg.content and not last_msg.content.endswith("```"):
            # 发现未关闭的代码块，自动添加结束标记
            corrected = last_msg.content + "\n```"
            # 返回修复后的消息，替换原始输出
            return {"messages": [AIMessage(content=corrected)]}
    return None  # 格式正确，正常返回

# 使用示例：创建带装饰器护栏的 Agent
agent = create_agent(
    model="gpt-4o",  # 使用 GPT-4o 模型
    middleware=[input_length_check, output_format_check],  # 添加输入和输出护栏
    tools=[...],  # 提供的工具列表
)
```

### 组合多重护栏

通过 middleware 数组堆叠多个护栏，构建分层防护体系：

```python
# 导入必要的模块
from langchain.agents import create_agent  # Agent 创建函数
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware  # 内置中间件
from langgraph.checkpoint.memory import InMemorySaver  # 内存检查点，支持暂停/恢复

# 创建带有多层防护体系的 Agent
agent = create_agent(
    model="gpt-4o",  # 使用 GPT-4o 模型作为主要推理引擎
    tools=[search_tool, send_email_tool, database_tool],  # 提供的工具集
    middleware=[
        # ========== 第 1 层：输入清理 ==========
        # 在任何处理之前，先清理输入中的敏感信息
        # 这一层确保敏感数据不会进入后续处理流程
        PIIMiddleware("email", strategy="redact", apply_to_input=True),  # 脱敏邮箱地址
        PIIMiddleware("phone_number", strategy="redact", apply_to_input=True),  # 脱敏电话号码
        PIIMiddleware("credit_card", strategy="block", apply_to_input=True),  # 阻止信用卡号

        # ========== 第 2 层：确定性输入过滤 ==========
        # 快速过滤明显的违规内容，使用规则进行第一道防线
        ContentFilterMiddleware(
            banned_keywords=["hack", "exploit", "破解"]  # 禁用的关键词列表
        ),

        # ========== 第 3 层：业务规则 ==========
        # 应用特定的业务逻辑，确保请求符合业务要求
        BusinessRulesMiddleware(
            max_query_complexity=10,  # 限制查询复杂度，防止过度消耗资源
            allowed_domains=["company.com", "partner.com"],  # 允许访问的域名白名单
        ),

        # ========== 第 4 层：人工审批 ==========
        # 敏感操作需要人工确认，提供关键决策点的人工监督
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True,  # 发送邮件需要人工批准
                "database_tool": {"allow_accept": True, "allow_respond": True},  # 数据库操作可批准或拒绝
            }
        ),

        # ========== 第 5 层：输出安全检查 ==========
        # 确保最终输出安全，防止敏感信息泄露
        PIIMiddleware("email", strategy="redact", apply_to_output=True),  # 输出中脱敏邮箱
        SafetyGuardrailMiddleware(),  # 使用模型评估输出安全性

        # ========== 第 6 层：监控和日志 ==========
        # 记录所有操作用于审计，满足合规要求
        AuditLogMiddleware(),  # 记录所有交互和决策
    ],
    checkpointer=InMemorySaver(),  # 必须配置检查点以支持人工审批的暂停/恢复功能
)
```

**护栏执行顺序：**

```
用户输入
    ↓
[第1层] PII 脱敏 (输入)
    ↓
[第2层] 关键词过滤 (before_agent)
    ↓
[第3层] 业务规则检查 (before_agent)
    ↓
    ↓ ← 如果被拒绝，直接返回
    ↓
[Agent 执行]
    ↓
[第4层] 人工审批 (wrap_tool_call)
    ↓
    ↓ ← 暂停等待批准
    ↓
[工具执行]
    ↓
[第5层] PII 脱敏 (输出) + 安全检查 (after_agent)
    ↓
[第6层] 审计日志 (after_agent)
    ↓
最终响应
```

---

## Runtime 运行时系统

### 什么是 Runtime

**Runtime（运行时）** 是 LangChain Agent 的执行环境，提供了依赖注入机制，让工具和中间件能够访问外部资源和配置，而无需硬编码或使用全局状态。

**为什么需要 Runtime：**

| 传统方式 | Runtime 方式 |
|---------|-------------|
| 硬编码数据库连接 | 通过 context 注入 |
| 全局变量传递用户 ID | 通过 context 访问 |
| 工具内部读取配置文件 | 通过 context 传递配置 |
| 难以测试和模拟 | 依赖注入，易于测试 |

### 核心组件

Runtime 包含三个核心组件：

```
┌─────────────────────────────────────────────────┐
│                    Runtime                       │
│  ┌──────────────────────────────────────────┐   │
│  │           Context（上下文）              │   │
│  │  • 用户信息 (user_id, user_name)        │   │
│  │  • 数据库连接                            │   │
│  │  • API 密钥和配置                        │   │
│  │  • 任何静态依赖                          │   │
│  └──────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────┐   │
│  │            Store（存储）                 │   │
│  │  • 长期记忆存储                          │   │
│  │  • 跨会话持久化数据                      │   │
│  │  • 用户偏好和历史                        │   │
│  └──────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────┐   │
│  │        Stream Writer（流写入器）         │   │
│  │  • 实时传输自定义数据                    │   │
│  │  • 进度更新                              │   │
│  │  • 中间结果                              │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### 配置与使用

#### 1. 定义 Context Schema

使用 dataclass 定义上下文结构：
```python
# 导入必要的模块
from dataclasses import dataclass  # 数据类装饰器，用于定义结构化数据
from typing import Any  # 通用类型注解

@dataclass  # 装饰器：自动生成初始化方法、比较方法等
class AppContext:
    """应用上下文 - 定义运行时可用的信息
    
    这个数据类定义了 Agent 运行时可以访问的所有上下文信息。
    通过依赖注入机制，工具和中间件可以安全地访问这些资源，
    而无需使用全局变量或硬编码。
    """

    # ========== 用户信息 ==========
    user_id: str  # 用户唯一标识符，必需字段
    user_name: str  # 用户显示名称，必需字段
    user_role: str = "user"  # 用户角色，默认为普通用户
                                # 可选值: "user" (普通用户), "admin" (管理员), "guest" (访客)

    # ========== 数据库连接 ==========
    database_connection: Any = None  # 数据库连接对象，默认为空
                                     # 可以是任何类型的数据库连接（MySQL、PostgreSQL等）

    # ========== 配置选项 ==========
    max_tokens: int = 4000  # 最大令牌数限制，防止过度消耗资源
    language: str = "zh"  # 界面语言设置，默认中文

    # ========== API 密钥 ==========
    api_key: str = ""  # API 密钥，建议通过环境变量传入，避免硬编码
```

#### 2. 创建使用 Context 的 Agent

```python
# 导入 Agent 创建函数
from langchain.agents import create_agent

# 创建使用上下文的 Agent
agent = create_agent(
    model="gpt-4o",  # 使用 GPT-4o 模型作为主要推理引擎
    tools=[...],  # 提供的工具列表，这些工具可以访问上下文
    middleware=[...],  # 中间件列表，这些中间件也可以访问上下文
    context_schema=AppContext,  # 指定上下文结构，启用依赖注入功能
)
```

#### 3. 调用时传递 Context

```python
# 创建上下文实例 - 为特定用户会话配置运行时环境
context = AppContext(
    user_id="user-12345",  # 用户唯一标识符
    user_name="张三",  # 用户显示名称
    user_role="admin",  # 用户角色：管理员权限，可以执行所有操作
    database_connection=db_conn,  # 数据库连接对象，需要在调用前已建立连接
    language="zh",  # 界面语言设置
)

# 调用 Agent 时传递上下文 - 实现依赖注入
result = agent.invoke(
    {"messages": [{"role": "user", "content": "查询我的订单"}]},  # 用户请求
    context=context,  # 传递运行时配置，工具和中间件可以访问这些信息
)
```

### 依赖注入

#### 在工具中访问 Runtime

使用 `ToolRuntime` 类型注解让工具可以访问运行时信息：

```python
# 导入工具相关的模块
from langchain.tools import tool, ToolRuntime  # 工具装饰器和运行时类型

@tool  # 装饰器：将函数注册为 Agent 可用的工具
def get_user_orders(
    limit: int,  # 查询限制数量，由用户指定
    runtime: ToolRuntime[AppContext]  # 声明需要访问运行时，类型注解指定上下文类型
) -> str:
    """获取用户订单列表
    
    这个工具演示了如何通过依赖注入访问运行时信息，
    包括用户身份和数据库连接，而无需硬编码或全局变量。
    """

    # 从上下文获取用户 ID - 自动从当前会话中获取用户身份
    user_id = runtime.context.user_id

    # 从上下文获取数据库连接 - 使用已建立的数据库连接
    db = runtime.context.database_connection

    # 执行数据库查询，使用参数化查询防止 SQL 注入
    orders = db.query(f"SELECT * FROM orders WHERE user_id = ? LIMIT ?",
                      [user_id, limit])

    # 格式化查询结果并返回
    return format_orders(orders)

@tool  # 装饰器：注册为工具
def save_user_preference(
    key: str,  # 偏好设置键名
    value: str,  # 偏好设置值
    runtime: ToolRuntime[AppContext]  # 运行时访问
) -> str:
    """保存用户偏好设置
    
    演示如何使用 Runtime 的 Store 组件进行持久化存储，
    实现跨会话的用户偏好管理。
    """

    # 从上下文获取用户 ID，确定偏好设置的所有者
    user_id = runtime.context.user_id

    # 使用 Store 进行持久化存储 - 跨会话保存数据
    if runtime.store:
        # 使用元组作为键，构建层次化存储结构
        runtime.store.put(
            ("users", user_id, "preferences"),  # 存储路径：用户/偏好设置
            key,  # 偏好键名
            {"value": value}  # 偏好值，使用字典结构便于扩展
        )
        return f"已保存偏好设置: {key}={value}"

    # 如果存储服务不可用，返回错误信息
    return "存储服务不可用"

@tool  # 装饰器：注册为工具
def search_with_progress(
    query: str,  # 搜索查询字符串
    runtime: ToolRuntime[AppContext]  # 运行时访问
) -> str:
    """带进度更新的搜索
    
    演示如何使用 Runtime 的 Stream Writer 组件实现实时进度更新，
    提升用户体验。
    """

    # 使用 Stream Writer 发送进度更新 - 实时向用户反馈执行状态
    if runtime.stream_writer:
        runtime.stream_writer.write({
            "progress": 0, 
            "status": "开始搜索..."
        })

    results = []
    # 遍历多个搜索源，模拟分布式搜索
    for i, source in enumerate(search_sources):
        # 在单个源上执行搜索
        partial_results = search_source(source, query)
        results.extend(partial_results)

        # 更新进度 - 向用户实时反馈搜索进展
        if runtime.stream_writer:
            progress = (i + 1) / len(search_sources) * 100
            runtime.stream_writer.write({
                "progress": progress,
                "status": f"已搜索 {i+1}/{len(search_sources)} 个来源"
            })

    # 格式化并返回搜索结果
    return format_results(results)
```

#### 在中间件中访问 Runtime

中间件可以通过 `runtime` 参数或 `request.runtime` 访问运行时：

```python
from langchain.agents.middleware import (
    dynamic_prompt,
    before_model,
    after_model,
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    AgentState
)
from langgraph.runtime import Runtime
from typing import Callable

# 1. 动态提示词 - 根据用户信息个性化
@dynamic_prompt
def personalized_system_prompt(request: ModelRequest) -> str:
    """生成个性化系统提示词"""
    ctx = request.runtime.context

    base_prompt = "你是一个智能助手。"

    # 根据用户角色调整提示词
    if ctx.user_role == "admin":
        role_prompt = "用户是管理员，可以执行所有操作。"
    elif ctx.user_role == "guest":
        role_prompt = "用户是访客，只能进行查询操作。"
    else:
        role_prompt = "用户是普通用户。"

    # 根据语言调整
    if ctx.language == "zh":
        lang_prompt = "请使用中文回复。"
    else:
        lang_prompt = "Please respond in English."

    return f"{base_prompt}\n{role_prompt}\n{lang_prompt}\n用户名: {ctx.user_name}"

# 2. 前置钩子 - 记录请求
@before_model
def log_request(state: AgentState, runtime: Runtime[AppContext]) -> dict | None:
    """记录用户请求"""
    user_id = runtime.context.user_id
    user_name = runtime.context.user_name

    print(f"[{user_id}] {user_name} 发起请求")
    print(f"消息数: {len(state['messages'])}")

    return None  # 不修改状态

# 3. 后置钩子 - 保存对话
@after_model
def save_conversation(state: AgentState, runtime: Runtime[AppContext]) -> dict | None:
    """保存对话到数据库"""
    db = runtime.context.database_connection
    user_id = runtime.context.user_id

    if db:
        last_msg = state["messages"][-1]
        db.save_message(user_id, last_msg.content)

    return None

# 4. 包装钩子 - 根据上下文选择模型
@wrap_model_call
def context_aware_model_selection(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """根据用户角色选择模型"""
    ctx = request.runtime.context

    if ctx.user_role == "admin":
        # 管理员使用更强大的模型
        from langchain_openai import ChatOpenAI
        request.model = ChatOpenAI(model="gpt-4o")

    return handler(request)
```

#### 在类式中间件中使用 Runtime

```python
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime
from typing import Any

class UserContextMiddleware(AgentMiddleware):
    """用户上下文管理中间件"""

    def before_agent(self, state: AgentState, runtime: Runtime[AppContext]) -> dict | None:
        """Agent 开始前加载用户数据"""
        ctx = runtime.context

        # 从存储加载用户历史
        if runtime.store:
            user_history = runtime.store.get(("users",), ctx.user_id)
            if user_history:
                print(f"已加载用户 {ctx.user_name} 的历史记录")

        return None

    def after_agent(self, state: AgentState, runtime: Runtime[AppContext]) -> dict | None:
        """Agent 完成后保存用户数据"""
        ctx = runtime.context

        # 保存对话摘要到存储
        if runtime.store:
            runtime.store.put(
                ("users", ctx.user_id),
                "last_conversation",
                {
                    "messages_count": len(state["messages"]),
                    "timestamp": datetime.now().isoformat()
                }
            )

        return None
```

### Runtime 完整示例

```python
from dataclasses import dataclass
from typing import Any
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, before_model, after_model, ModelRequest
from langchain.tools import tool, ToolRuntime
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

# 1. 定义上下文结构
@dataclass
class CustomerServiceContext:
    """客服系统上下文"""
    customer_id: str
    customer_name: str
    subscription_tier: str  # "free", "pro", "enterprise"
    language: str = "zh"

# 2. 定义工具
@tool
def get_customer_info(runtime: ToolRuntime[CustomerServiceContext]) -> str:
    """获取当前客户信息"""
    ctx = runtime.context
    return f"客户: {ctx.customer_name} (ID: {ctx.customer_id}), 套餐: {ctx.subscription_tier}"

@tool
def get_order_history(
    limit: int,
    runtime: ToolRuntime[CustomerServiceContext]
) -> str:
    """获取客户订单历史"""
    customer_id = runtime.context.customer_id
    # 实际应用中这里会查询数据库
    return f"客户 {customer_id} 最近 {limit} 个订单: [订单列表...]"

@tool
def create_support_ticket(
    issue: str,
    priority: str,
    runtime: ToolRuntime[CustomerServiceContext]
) -> str:
    """创建支持工单"""
    ctx = runtime.context

    # 企业客户自动升级优先级
    if ctx.subscription_tier == "enterprise" and priority == "normal":
        priority = "high"

    return f"已为客户 {ctx.customer_name} 创建工单\n问题: {issue}\n优先级: {priority}"

# 3. 定义中间件
@dynamic_prompt
def customer_service_prompt(request: ModelRequest) -> str:
    """客服系统提示词"""
    ctx = request.runtime.context

    tier_instructions = {
        "free": "提供基础支持，复杂问题建议升级套餐。",
        "pro": "提供专业支持，可以处理技术问题。",
        "enterprise": "提供最高级别支持，优先处理，可承诺 SLA。",
    }

    return f"""你是一个专业的客服助手。

当前客户信息:
- 姓名: {ctx.customer_name}
- 客户 ID: {ctx.customer_id}
- 套餐: {ctx.subscription_tier}

服务指南:
{tier_instructions.get(ctx.subscription_tier, tier_instructions["free"])}

请使用{ctx.language == "zh" and "中文" or "English"}回复。
保持专业、友好、高效。"""

@before_model
def log_customer_interaction(state, runtime: Runtime[CustomerServiceContext]):
    """记录客户交互"""
    print(f"[客服] 处理客户 {runtime.context.customer_id} 的请求")
    return None

# 4. 创建 Agent
agent = create_agent(
    model="gpt-4o",
    tools=[get_customer_info, get_order_history, create_support_ticket],
    middleware=[
        customer_service_prompt,
        log_customer_interaction,
    ],
    context_schema=CustomerServiceContext,
    checkpointer=InMemorySaver(),
    store=InMemoryStore(),
)

# 5. 使用 Agent
def handle_customer_request(customer_id: str, customer_name: str, tier: str, message: str):
    """处理客户请求"""

    # 创建上下文
    context = CustomerServiceContext(
        customer_id=customer_id,
        customer_name=customer_name,
        subscription_tier=tier,
        language="zh",
    )

    # 调用 Agent
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        context=context,
        config={"configurable": {"thread_id": f"customer-{customer_id}"}}
    )

    return result["messages"][-1].content

# 使用示例
response = handle_customer_request(
    customer_id="C001",
    customer_name="李明",
    tier="enterprise",
    message="我想查看最近的订单，并创建一个退款工单"
)
print(response)
```

---

## 实战案例

### 案例 1：安全的企业聊天机器人

结合护栏和运行时，构建企业级安全聊天机器人：

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import (
    PIIMiddleware,
    HumanInTheLoopMiddleware,
    SummarizationMiddleware,
)
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

@dataclass
class EnterpriseContext:
    employee_id: str
    employee_name: str
    department: str
    access_level: int  # 1-5, 5 最高
    db_connection: any

# 自定义护栏：部门访问控制
class DepartmentAccessMiddleware(AgentMiddleware):
    """确保员工只能访问其部门的数据"""

    DEPARTMENT_TOOLS = {
        "HR": ["employee_lookup", "payroll_query"],
        "Finance": ["expense_report", "budget_query"],
        "Engineering": ["code_search", "deploy_status"],
    }

    def wrap_tool_call(self, request, handler):
        ctx = request.runtime.context
        tool_name = request.tool_call.name
        allowed_tools = self.DEPARTMENT_TOOLS.get(ctx.department, [])

        # 高访问级别可以访问所有工具
        if ctx.access_level >= 4:
            return handler(request)

        if tool_name not in allowed_tools:
            from langchain.messages import ToolMessage
            return ToolMessage(
                content=f"访问被拒绝: {ctx.department} 部门无权使用 {tool_name}",
                tool_call_id=request.tool_call.id
            )

        return handler(request)

# 创建安全的企业 Agent
enterprise_agent = create_agent(
    model="gpt-4o",
    tools=[
        employee_lookup,
        payroll_query,
        expense_report,
        budget_query,
        code_search,
        deploy_status,
        send_notification,
    ],
    middleware=[
        # 第 1 层：PII 保护
        PIIMiddleware("ssn", strategy="block"),
        PIIMiddleware("credit_card", strategy="block"),
        PIIMiddleware("email", strategy="redact", apply_to_output=True),

        # 第 2 层：对话管理
        SummarizationMiddleware(
            model="gpt-4o-mini",
            max_tokens_before_summary=5000,
        ),

        # 第 3 层：访问控制
        DepartmentAccessMiddleware(),

        # 第 4 层：敏感操作审批
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_notification": True,
                "payroll_query": {"allow_accept": True, "allow_respond": True},
            }
        ),

        # 第 5 层：审计日志
        AuditLogMiddleware(),
    ],
    context_schema=EnterpriseContext,
    checkpointer=PostgresSaver(connection_string="postgresql://..."),
    store=PostgresStore(connection_string="postgresql://..."),
)
```

### 案例 2：合规的金融顾问 Agent

```python
from dataclasses import dataclass

@dataclass
class FinancialContext:
    client_id: str
    client_name: str
    risk_profile: str  # "conservative", "moderate", "aggressive"
    accredited_investor: bool
    jurisdiction: str  # "US", "EU", "CN"

# 合规护栏
class FinancialComplianceMiddleware(AgentMiddleware):
    """金融合规护栏"""

    # 不同司法管辖区的限制
    JURISDICTION_RULES = {
        "US": {
            "disclaimer_required": True,
            "crypto_allowed": True,
            "leverage_limit": 4,
        },
        "CN": {
            "disclaimer_required": True,
            "crypto_allowed": False,
            "leverage_limit": 1,
        },
    }

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state, runtime: Runtime[FinancialContext]):
        """添加合规免责声明"""
        ctx = runtime.context
        rules = self.JURISDICTION_RULES.get(ctx.jurisdiction, {})

        if rules.get("disclaimer_required"):
            disclaimer = """⚠️ 免责声明：
本内容仅供参考，不构成投资建议。
投资有风险，请在做出任何投资决策前咨询专业顾问。
过往业绩不代表未来表现。"""

            # 将免责声明添加到对话开始
            from langchain.messages import SystemMessage
            return {
                "messages": [SystemMessage(content=disclaimer)] + state["messages"]
            }

        return None

    def wrap_tool_call(self, request, handler):
        """检查工具调用的合规性"""
        ctx = request.runtime.context
        rules = self.JURISDICTION_RULES.get(ctx.jurisdiction, {})
        tool_name = request.tool_call.name

        # 检查加密货币限制
        if tool_name == "crypto_trade" and not rules.get("crypto_allowed"):
            from langchain.messages import ToolMessage
            return ToolMessage(
                content=f"根据 {ctx.jurisdiction} 的法规，不支持加密货币交易。",
                tool_call_id=request.tool_call.id
            )

        # 检查非合格投资者限制
        if tool_name in ["hedge_fund_invest", "private_equity"]:
            if not ctx.accredited_investor:
                from langchain.messages import ToolMessage
                return ToolMessage(
                    content="此投资产品仅限合格投资者。",
                    tool_call_id=request.tool_call.id
                )

        return handler(request)

# 风险适配护栏
class RiskProfileMiddleware(AgentMiddleware):
    """根据风险偏好过滤建议"""

    RISK_LIMITS = {
        "conservative": {"max_volatility": 10, "allowed_assets": ["bonds", "blue_chip", "etf"]},
        "moderate": {"max_volatility": 20, "allowed_assets": ["bonds", "stocks", "etf", "reits"]},
        "aggressive": {"max_volatility": 50, "allowed_assets": ["all"]},
    }

    @hook_config(can_jump_to=["end"])
    def after_model(self, state, runtime: Runtime[FinancialContext]):
        """检查建议是否符合客户风险偏好"""
        ctx = runtime.context
        limits = self.RISK_LIMITS.get(ctx.risk_profile, self.RISK_LIMITS["conservative"])

        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage):
            # 检查是否推荐了不适合的产品
            high_risk_keywords = ["期货", "期权", "杠杆", "衍生品", "futures", "options"]

            if ctx.risk_profile == "conservative":
                for keyword in high_risk_keywords:
                    if keyword.lower() in last_msg.content.lower():
                        return {
                            "messages": [AIMessage(
                                content=f"根据您的风险偏好（{ctx.risk_profile}），"
                                        f"我无法推荐高风险产品。以下是更适合您的选择：\n"
                                        f"[适合保守型投资者的产品建议]"
                            )],
                            "jump_to": "end"
                        }

        return None

# 创建合规金融顾问
financial_advisor = create_agent(
    model="gpt-4o",
    tools=[
        market_research,
        portfolio_analysis,
        stock_trade,
        bond_trade,
        crypto_trade,
        hedge_fund_invest,
    ],
    middleware=[
        # 合规层
        FinancialComplianceMiddleware(),
        RiskProfileMiddleware(),

        # PII 保护
        PIIMiddleware("ssn", strategy="block"),
        PIIMiddleware("account_number", strategy="mask"),

        # 所有交易需要确认
        HumanInTheLoopMiddleware(
            interrupt_on={
                "stock_trade": True,
                "bond_trade": True,
                "crypto_trade": True,
            }
        ),
    ],
    context_schema=FinancialContext,
)
```

---

## 最佳实践

### 护栏设计原则

#### 1. 分层防护

```python
# ✅ 好的做法：多层防护，各司其职
middleware=[
    # 第 1 层：快速、确定性检查
    InputLengthCheck(),
    BannedKeywordsFilter(),

    # 第 2 层：PII 保护
    PIIMiddleware(...),

    # 第 3 层：业务规则
    BusinessRulesMiddleware(),

    # 第 4 层：智能检查（更耗时）
    SafetyGuardrailMiddleware(),
]

# ❌ 不好的做法：单一护栏做所有事情
middleware=[
    DoEverythingMiddleware(),  # 难以维护和测试
]
```

#### 2. 快速失败

```python
# ✅ 好的做法：尽早拒绝不合规请求
@before_agent(can_jump_to=["end"])
def quick_reject(state, runtime):
    """在 Agent 开始前就拒绝不合规请求"""
    if is_clearly_malicious(state["messages"][0].content):
        return {
            "messages": [AIMessage(content="请求被拒绝")],
            "jump_to": "end"  # 立即结束，不调用模型
        }
    return None

# ❌ 不好的做法：让不合规请求走完整个流程
@after_agent
def late_reject(state, runtime):
    """在 Agent 完成后才检查"""
    # 此时已经消耗了模型调用的成本
    pass
```

#### 3. 优雅降级

```python
# ✅ 好的做法：护栏失败时优雅处理
@wrap_model_call
def resilient_guardrail(request, handler):
    try:
        # 尝试执行护栏检查
        safety_check(request)
    except Exception as e:
        # 记录错误但不阻塞
        logger.error(f"护栏检查失败: {e}")
        # 可以选择继续执行或返回安全默认值

    return handler(request)

# ❌ 不好的做法：护栏失败导致整个系统崩溃
@wrap_model_call
def fragile_guardrail(request, handler):
    safety_check(request)  # 如果抛出异常，整个请求失败
    return handler(request)
```

### Runtime 设计原则

#### 1. 明确的上下文边界

```python
# ✅ 好的做法：清晰定义上下文结构
@dataclass
class AppContext:
    """应用上下文 - 所有必需的运行时信息"""
    user_id: str           # 必需
    user_name: str         # 必需
    permissions: list[str] # 必需

    # 可选配置
    language: str = "zh"
    timezone: str = "Asia/Shanghai"

    # 外部依赖（可选，便于测试）
    database: Any = None
    cache: Any = None

# ❌ 不好的做法：在工具内部获取全局状态
@tool
def bad_tool():
    user_id = global_state["user_id"]  # 隐式依赖
    db = get_global_db_connection()    # 难以测试
```

#### 2. 不可变上下文

```python
# ✅ 好的做法：上下文在调用期间保持不变
@dataclass(frozen=True)  # 使用 frozen 防止修改
class ImmutableContext:
    user_id: str
    permissions: tuple[str, ...]  # 使用不可变类型

# ❌ 不好的做法：在执行过程中修改上下文
def some_middleware(state, runtime):
    runtime.context.user_id = "new_id"  # 可能导致不可预测的行为
```

#### 3. 测试友好

```python
# ✅ 好的做法：依赖注入使测试简单
def test_agent_with_mock_db():
    # 创建模拟依赖
    mock_db = MockDatabase()
    mock_db.add_user("user-1", "Test User")

    # 注入到上下文
    context = AppContext(
        user_id="user-1",
        user_name="Test User",
        database=mock_db,  # 注入模拟对象
    )

    result = agent.invoke(
        {"messages": [...]},
        context=context,
    )

    assert mock_db.query_count == 1
```

---

## 快速参考

### 护栏类型对比

| 类型 | 速度 | 成本 | 准确性 | 适用场景 |
|------|------|------|--------|----------|
| 确定性护栏 | 快 | 低 | 中 | 格式检查、关键词过滤 |
| 模型护栏 | 慢 | 高 | 高 | 语义理解、复杂判断 |
| 混合护栏 | 中 | 中 | 高 | 先快速过滤，再深度检查 |

### PII 中间件配置速查

```python
# 邮箱脱敏
PIIMiddleware("email", strategy="redact", apply_to_input=True)

# 信用卡遮蔽
PIIMiddleware("credit_card", strategy="mask", apply_to_input=True, apply_to_output=True)

# 阻止 SSN
PIIMiddleware("ssn", strategy="block")

# 自定义检测器
PIIMiddleware("custom", detector=r"your_regex", strategy="redact")
```

### 人工审批配置速查

```python
HumanInTheLoopMiddleware(
    interrupt_on={
        "tool_name": True,  # 简单配置：允许所有决策类型
        "another_tool": {
            "allow_accept": True,
            "allow_edit": False,
            "allow_respond": True,
            "description": "自定义提示",
        },
    }
)

# 恢复执行
agent.invoke(Command(resume={"decisions": [{"type": "accept"}]}), config)
agent.invoke(Command(resume={"decisions": [{"type": "edit", "args": {...}}]}), config)
agent.invoke(Command(resume={"decisions": [{"type": "respond", "response": "..."}]}), config)
```

### Runtime 访问方式

```python
# 在工具中
@tool
def my_tool(runtime: ToolRuntime[MyContext]):
    user_id = runtime.context.user_id
    data = runtime.store.get(...)
    runtime.stream_writer.write(...)

# 在装饰器中间件中
@before_model
def hook(state, runtime: Runtime[MyContext]):
    user_id = runtime.context.user_id

@dynamic_prompt
def prompt(request: ModelRequest):
    user_id = request.runtime.context.user_id

# 在类中间件中
class MyMiddleware(AgentMiddleware):
    def before_model(self, state, runtime: Runtime[MyContext]):
        user_id = runtime.context.user_id
```

### 常见护栏模式

```python
# 1. 输入长度限制
@before_agent(can_jump_to=["end"])
def limit_input(state, runtime):
    if len(state["messages"][0].content) > 10000:
        return {"messages": [AIMessage("输入过长")], "jump_to": "end"}
    return None

# 2. 输出格式验证
@after_agent
def validate_json_output(state, runtime):
    try:
        json.loads(state["messages"][-1].content)
    except:
        return {"messages": [AIMessage("输出格式错误，请重试")]}
    return None

# 3. 速率限制
class RateLimitMiddleware(AgentMiddleware):
    def __init__(self, max_requests_per_minute=10):
        self.requests = {}
        self.limit = max_requests_per_minute

    def before_agent(self, state, runtime):
        user_id = runtime.context.user_id
        # 实现速率限制逻辑
        ...

# 4. 重试逻辑
@wrap_model_call
def retry_on_failure(request, handler):
    for i in range(3):
        try:
            return handler(request)
        except Exception:
            if i == 2: raise
            time.sleep(2 ** i)
```

---

## 总结

**Guardrails（护栏）** 和 **Runtime（运行时）** 是构建生产级 AI 应用的两大核心系统：

### Guardrails 核心要点

- **确定性护栏**：快速、可预测，适合格式检查和关键词过滤
- **模型护栏**：智能、灵活，适合复杂的语义理解
- **内置护栏**：PIIMiddleware、HumanInTheLoopMiddleware 开箱即用
- **分层防护**：多层护栏协同工作，构建纵深防御

### Runtime 核心要点

- **Context**：通过依赖注入传递配置和资源
- **Store**：长期记忆存储，跨会话持久化
- **Stream Writer**：实时传输进度和中间结果
- **测试友好**：依赖注入使单元测试更简单

### 最佳实践

1. **快速失败**：尽早拒绝不合规请求
2. **分层防护**：多层护栏，各司其职
3. **优雅降级**：护栏失败时不阻塞系统
4. **明确边界**：清晰定义上下文结构
5. **不可变上下文**：避免执行过程中修改
6. **测试友好**：使用依赖注入便于测试

通过合理使用这两个系统，你可以构建出既安全又灵活的 AI 应用！
