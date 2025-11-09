# LangChain Middleware 详细指南

## 目录

1. [概述](#概述)
2. [为什么需要 Middleware](#为什么需要-middleware)
3. [核心概念](#核心概念)
4. [Middleware 生命周期钩子](#middleware-生命周期钩子)
5. [执行顺序](#执行顺序)
6. [装饰器式 Middleware](#装饰器式-middleware)
7. [类式 Middleware](#类式-middleware)
8. [预置 Middleware](#预置-middleware)
9. [自定义 Middleware 实战](#自定义-middleware-实战)
10. [自定义状态管理](#自定义状态管理)
11. [Runtime 访问](#runtime-访问)
12. [实际应用场景](#实际应用场景)
13. [最佳实践](#最佳实践)
14. [性能优化](#性能优化)
15. [快速参考](#快速参考)

---

## 概述

Middleware（中间件）是 LangChain `create_agent` 的核心特性，提供了一种在 Agent 执行的不同阶段精确控制行为的方法。通过 Middleware，你可以拦截并修改 Agent 执行流程中的数据，而无需改变核心 Agent 逻辑。

**核心 Agent 循环：**
```
开始 → 调用模型 → 模型选择工具 → 执行工具 → 继续或结束
```

**Middleware 提供的钩子：**
- ✅ **before_agent** - Agent 开始前
- ✅ **before_model** - 每次调用模型前
- ✅ **wrap_model_call** - 包装模型调用
- ✅ **after_model** - 每次模型响应后
- ✅ **wrap_tool_call** - 包装工具调用
- ✅ **after_agent** - Agent 完成后

---

## 为什么需要 Middleware

### 1. **上下文工程 (Context Engineering)**

优秀的 Agent 需要在正确的时间获得正确的信息。Middleware 帮助你：
- 动态调整 prompt
- 总结对话历史
- 选择性工具访问
- 状态管理

### 2. **可组合性**

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    PIIMiddleware,
    HumanInTheLoopMiddleware
)

agent = create_agent(
    model="gpt-4o",
    tools=[read_email, send_email],
    middleware=[
        PIIMiddleware("email", strategy="redact"),           # PII 保护
        SummarizationMiddleware(                              # 对话总结
            model="gpt-4o-mini",
            max_tokens_before_summary=500
        ),
        HumanInTheLoopMiddleware(                             # 人工审核
            interrupt_on={"send_email": {"allowed_decisions": ["approve", "edit", "reject"]}}
        ),
    ]
)
```

### 3. **关注点分离**

Middleware 让你将横切关注点（日志、监控、安全）从核心业务逻辑中分离出来：
- 📊 监控 - 追踪 Agent 行为、日志、分析
- 🔄 修改 - 转换 prompts、工具选择、输出格式
- 🎮 控制 - 添加重试、回退、提前终止逻辑
- 🛡️ 执行 - 应用速率限制、护栏、PII 检测

---

## 核心概念

### Agent 执行流程

```
┌─────────────────────────────────────────────────────────┐
│ before_agent()                                          │
│ ↓                                                       │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Agent Loop                                          │ │
│ │                                                     │ │
│ │ before_model()                                      │ │
│ │ ↓                                                   │ │
│ │ wrap_model_call() → 模型调用                         │ │
│ │ ↓                                                   │ │
│ │ after_model()                                       │ │
│ │ ↓                                                   │ │
│ │ wrap_tool_call() → 工具执行 (如果有工具调用)         │ │
│ │ ↓                                                   │ │
│ │ (循环直到无工具调用)                                 │ │
│ └─────────────────────────────────────────────────────┘ │
│ ↓                                                       │
│ after_agent()                                           │
└─────────────────────────────────────────────────────────┘
```

### Middleware 类型

#### 1. **装饰器式 Middleware**
快速添加单一钩子功能：

```python
from langchain.agents.middleware import before_model, after_model

@before_model
def log_before_model(state, runtime):
    print(f"调用模型，消息数: {len(state['messages'])}")
    return None
```

#### 2. **类式 Middleware**
实现多个钩子的复杂逻辑：

```python
from langchain.agents.middleware import AgentMiddleware

class CustomMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        # 模型调用前的逻辑
        return None
    
    def after_model(self, state, runtime):
        # 模型响应后的逻辑
        return None
```

---

## Middleware 生命周期钩子

### 1. before_agent

**时机：** Agent 开始执行前运行一次  
**用途：** 加载内存、验证输入、初始化资源

```python
from langchain.agents.middleware import before_agent
from langgraph.runtime import Runtime

@before_agent
def load_user_memory(state: AgentState, runtime: Runtime):
    """从数据库加载用户历史"""
    user_id = runtime.context.user_id
    history = database.load_history(user_id)
    
    return {"messages": history + state["messages"]}
```

### 2. before_model

**时机：** 每次调用 LLM 前执行  
**用途：** 更新 prompts、修剪消息、注入上下文

```python
from langchain.agents.middleware import before_model

@before_model
def trim_messages(state: AgentState, runtime: Runtime):
    """保留最后几条消息以适应上下文窗口"""
    messages = state["messages"]
    
    if len(messages) <= 5:
        return None  # 无需修改
    
    # 保留系统消息和最近的消息
    system_msg = messages[0]
    recent_msgs = messages[-4:]
    
    return {"messages": [system_msg] + recent_msgs}
```

**返回值：**
- `None` - 不修改状态
- `dict` - 更新状态（与现有状态合并）
- `{"jump_to": "node_name"}` - 跳转到指定节点

### 3. wrap_model_call

**时机：** 包装模型调用  
**用途：** 拦截请求/响应、重试逻辑、动态模型选择

```python
from langchain.agents.middleware import wrap_model_call
from langchain.agents.middleware import ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """添加重试逻辑"""
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"重试 {attempt + 1}/3，错误: {e}")
```

**动态模型选择：**
```python
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler):
    """根据消息数量选择模型"""
    message_count = len(request.messages)
    
    if message_count > 10:
        # 复杂对话使用强大模型
        request.model = ChatOpenAI(model="gpt-4o")
    else:
        # 简单对话使用轻量模型
        request.model = ChatOpenAI(model="gpt-4o-mini")
    
    return handler(request)
```

### 4. after_model

**时机：** 每次 LLM 响应后执行  
**用途：** 验证输出、应用护栏、内容过滤

```python
from langchain.agents.middleware import after_model
from langchain.messages import AIMessage

@after_model(can_jump_to=["end"])
def validate_output(state: AgentState, runtime: Runtime):
    """验证模型输出并应用内容过滤"""
    last_message = state["messages"][-1]
    
    # 检查被阻止的内容
    if "BLOCKED" in last_message.content:
        return {
            "messages": [AIMessage("抱歉，我无法响应该请求。")],
            "jump_to": "end"
        }
    
    return None
```

### 5. wrap_tool_call

**时机：** 包装工具执行  
**用途：** 错误处理、权限检查、日志记录

```python
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    """自定义工具错误处理"""
    try:
        return handler(request)
    except Exception as e:
        # 返回友好的错误消息
        return ToolMessage(
            content=f"工具错误：请检查输入并重试。({e})",
            tool_call_id=request.tool_call.id
        )
```

### 6. after_agent

**时机：** Agent 完成执行后运行一次  
**用途：** 保存结果、清理资源、记录分析

```python
from langchain.agents.middleware import after_agent

@after_agent
def save_results(state: AgentState, runtime: Runtime):
    """保存对话历史到数据库"""
    user_id = runtime.context.user_id
    database.save_conversation(user_id, state["messages"])
    
    # 不修改状态
    return None
```

---

## 执行顺序

### 多个 Middleware 的执行顺序

当使用多个 Middleware 时，执行顺序遵循特定规则：

```python
agent = create_agent(
    model="gpt-4o",
    middleware=[middleware1, middleware2, middleware3],
    tools=[...]
)
```

**执行流程：**

```
Before 钩子（按顺序）:
  middleware1.before_agent()
  middleware2.before_agent()
  middleware3.before_agent()

Agent 循环开始:
  middleware1.before_model()
  middleware2.before_model()
  middleware3.before_model()

Wrap 钩子（嵌套调用）:
  middleware1.wrap_model_call() →
    middleware2.wrap_model_call() →
      middleware3.wrap_model_call() →
        模型调用

After 钩子（反向顺序）:
  middleware3.after_model()
  middleware2.after_model()
  middleware1.after_model()

Agent 循环结束:
  middleware3.after_agent()
  middleware2.after_agent()
  middleware1.after_agent()
```

**关键规则：**
- ✅ `before_*` 钩子：从第一个到最后一个
- ✅ `after_*` 钩子：从最后一个到第一个（反向）
- ✅ `wrap_*` 钩子：嵌套调用（第一个包装所有其他的）

**示例：**

```python
from langchain.agents.middleware import before_model, after_model

@before_model
def middleware_a(state, runtime):
    print("A: before")
    return None

@before_model
def middleware_b(state, runtime):
    print("B: before")
    return None

@after_model
def middleware_c(state, runtime):
    print("C: after")
    return None

agent = create_agent(
    model="gpt-4o",
    middleware=[middleware_a, middleware_b, middleware_c],
    tools=[]
)
```

**输出：**
```
A: before
B: before
(模型调用)
C: after
```

---

## 装饰器式 Middleware

装饰器提供最快捷的方式添加单一钩子功能。

### 1. @before_model - 节点风格

```python
from langchain.agents.middleware import before_model, AgentState
from langgraph.runtime import Runtime
from typing import Any

@before_model
def log_and_trim(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """日志记录并修剪消息"""
    print(f"将要调用模型，消息数: {len(state['messages'])}")
    
    # 修剪消息
    if len(state["messages"]) > 10:
        return {"messages": state["messages"][-10:]}
    
    return None
```

### 2. @after_model - 节点风格

```python
from langchain.agents.middleware import after_model

@after_model(can_jump_to=["end"])
def check_safety(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """安全检查"""
    last_msg = state["messages"][-1]
    
    if contains_unsafe_content(last_msg.content):
        return {
            "messages": [AIMessage("内容已被过滤")],
            "jump_to": "end"
        }
    
    return None
```

### 3. @wrap_model_call - 包装风格

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def add_caching(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """添加缓存层"""
    cache_key = hash_request(request)
    
    # 检查缓存
    if cache_key in cache:
        return cache[cache_key]
    
    # 调用模型
    response = handler(request)
    
    # 存入缓存
    cache[cache_key] = response
    return response
```

### 4. @dynamic_prompt - 动态提示词

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    """根据用户角色生成动态提示词"""
    user_role = request.runtime.context.get("user_role", "user")
    
    base_prompt = "你是一个有帮助的助手。"
    
    if user_role == "expert":
        return f"{base_prompt} 提供详细的技术回答。"
    elif user_role == "beginner":
        return f"{base_prompt} 用简单的语言解释概念，避免术语。"
    
    return base_prompt
```

### 5. @wrap_tool_call - 工具包装

```python
from langchain.agents.middleware import wrap_tool_call

@wrap_tool_call
def log_tool_usage(request, handler):
    """记录工具使用情况"""
    tool_name = request.tool_call.name
    
    print(f"执行工具: {tool_name}")
    start_time = time.time()
    
    try:
        result = handler(request)
        elapsed = time.time() - start_time
        print(f"工具 {tool_name} 成功，耗时 {elapsed:.2f}s")
        return result
    except Exception as e:
        print(f"工具 {tool_name} 失败: {e}")
        raise
```

---

## 类式 Middleware

类式 Middleware 适用于需要实现多个钩子或维护状态的复杂场景。

### 基本结构

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from typing import Any, Callable

class CustomMiddleware(AgentMiddleware):
    """自定义中间件模板"""
    
    def __init__(self, config: dict):
        self.config = config
    
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Agent 开始前"""
        return None
    
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """模型调用前"""
        return None
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """包装模型调用"""
        return handler(request)
    
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """模型响应后"""
        return None
    
    def wrap_tool_call(self, request, handler):
        """包装工具调用"""
        return handler(request)
    
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Agent 完成后"""
        return None
```

### 实例：请求计数中间件

```python
from dataclasses import dataclass

@dataclass
class Context:
    user_id: str
    max_requests: int = 10

class RequestCounterMiddleware(AgentMiddleware):
    """限制每个用户的请求次数"""
    
    def __init__(self):
        self.request_counts = {}
    
    def before_agent(self, state: AgentState, runtime: Runtime[Context]) -> dict | None:
        """检查请求限制"""
        user_id = runtime.context.user_id
        max_req = runtime.context.max_requests
        
        # 增加计数
        count = self.request_counts.get(user_id, 0) + 1
        self.request_counts[user_id] = count
        
        # 检查是否超限
        if count > max_req:
            raise ValueError(f"用户 {user_id} 超过请求限制 ({max_requests})")
        
        print(f"用户 {user_id} 请求 {count}/{max_req}")
        return None
    
    def after_agent(self, state: AgentState, runtime: Runtime[Context]) -> dict | None:
        """记录完成"""
        user_id = runtime.context.user_id
        print(f"用户 {user_id} 请求完成")
        return None

# 使用
agent = create_agent(
    model="gpt-4o",
    middleware=[RequestCounterMiddleware()],
    tools=[],
    context_schema=Context
)

agent.invoke(
    {"messages": [{"role": "user", "content": "你好"}]},
    context=Context(user_id="alice", max_requests=5)
)
```

---

## 预置 Middleware

LangChain 提供了常用模式的预置 Middleware。

### 1. SummarizationMiddleware - 对话总结

当对话历史过长时，自动总结并压缩。

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",              # 用于总结的模型
            max_tokens_before_summary=4000,   # 触发总结的阈值
            messages_to_keep=20,              # 总结后保留的消息数
            summary_prompt="自定义总结提示词...",  # 可选
        )
    ]
)
```

**工作原理：**
1. 监控消息历史的 token 数量
2. 超过阈值时，使用单独的 LLM 调用生成总结
3. 用总结消息**永久替换**旧消息（持久化更新状态）
4. 保留最近的消息以维持上下文

**示例：**

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            max_tokens_before_summary=4000,
            messages_to_keep=20,
        )
    ],
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "你好，我叫 Bob"}, config)
agent.invoke({"messages": "写一首关于猫的诗"}, config)
agent.invoke({"messages": "现在写一首关于狗的"}, config)
# ... 许多轮对话后 ...
result = agent.invoke({"messages": "我叫什么名字？"}, config)

print(result["messages"][-1].content)
# "你的名字是 Bob！"
```

### 2. PIIMiddleware - PII 保护

自动检测和处理个人身份信息（PII）。

```python
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[send_email],
    middleware=[
        # 编辑电子邮件地址
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        
        # 阻止包含电话号码的请求
        PIIMiddleware(
            "phone_number",
            detector=r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{4}",
            strategy="block"
        ),
    ]
)
```

**策略：**
- `redact` - 编辑 PII（替换为 `[REDACTED]`）
- `block` - 阻止包含 PII 的请求

### 3. HumanInTheLoopMiddleware - 人工审核

要求人工批准敏感的工具调用。

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[read_email, send_email, delete_file],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                },
                "delete_file": {
                    "allowed_decisions": ["approve", "reject"]
                }
            }
        )
    ]
)
```

**工作流程：**
1. Agent 提议调用工具（如 `send_email`）
2. Middleware 拦截并发出中断（interrupt）
3. 执行暂停，等待人工决策
4. 人工做出决策：
   - `approve` - 批准执行
   - `edit` - 修改参数后执行
   - `reject` - 拒绝并提供反馈
5. 根据决策恢复执行

### 4. ContextEditingMiddleware - 上下文编辑

管理对话上下文，定期清理工具使用记录。

```python
from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(trigger=1000),  # 每 1000 tokens 清理工具使用
            ]
        )
    ]
)
```

---

## 自定义 Middleware 实战

### 实例 1：基于专业知识的动态工具选择

根据用户专业水平动态选择工具和模型。

```python
from dataclasses import dataclass
from typing import Callable
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

@dataclass
class Context:
    user_expertise: str = "beginner"  # "beginner" 或 "expert"

class ExpertiseBasedMiddleware(AgentMiddleware):
    """根据用户专业水平选择工具和模型"""
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        user_level = request.runtime.context.user_expertise
        
        if user_level == "expert":
            # 专家级用户：强大模型 + 高级工具
            request.model = ChatOpenAI(model="gpt-4o")
            request.tools = [advanced_search, data_analysis, code_execution]
        else:
            # 初学者：轻量模型 + 基础工具
            request.model = ChatOpenAI(model="gpt-4o-mini")
            request.tools = [simple_search, basic_calculator]
        
        return handler(request)

# 使用
agent = create_agent(
    model="gpt-4o",  # 基础模型
    tools=[simple_search, advanced_search, basic_calculator, data_analysis, code_execution],
    middleware=[ExpertiseBasedMiddleware()],
    context_schema=Context
)

# 初学者模式
result_beginner = agent.invoke(
    {"messages": [{"role": "user", "content": "解释机器学习"}]},
    context=Context(user_expertise="beginner")
)

# 专家模式
result_expert = agent.invoke(
    {"messages": [{"role": "user", "content": "解释 Transformer 架构"}]},
    context=Context(user_expertise="expert")
)
```

### 实例 2：智能重试与回退

添加重试逻辑和模型回退策略。

```python
import time
from langchain.agents.middleware import wrap_model_call

@wrap_model_call
def retry_with_fallback(request: ModelRequest, handler):
    """重试失败的调用，并在必要时降级到备用模型"""
    primary_model = request.model
    fallback_model = ChatOpenAI(model="gpt-4o-mini")
    
    # 尝试主模型（最多 3 次）
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                # 主模型失败，尝试备用模型
                print(f"主模型失败，切换到备用模型")
                request.model = fallback_model
                try:
                    return handler(request)
                except Exception as fallback_e:
                    raise Exception(f"主模型和备用模型都失败: {e}, {fallback_e}")
            
            # 指数退避
            wait_time = 2 ** attempt
            print(f"重试 {attempt + 1}/3，{wait_time}秒后重试...")
            time.sleep(wait_time)
```

### 实例 3：内容审核与护栏

实施内容审核和安全护栏。

```python
from langchain.agents.middleware import before_model, after_model
from langchain.messages import AIMessage

# 输入审核
@before_model
def moderate_input(state: AgentState, runtime: Runtime):
    """审核用户输入"""
    last_user_msg = None
    for msg in reversed(state["messages"]):
        if msg.role == "user":
            last_user_msg = msg
            break
    
    if last_user_msg and is_harmful_content(last_user_msg.content):
        return {
            "messages": [AIMessage("抱歉，我无法处理该请求。")],
            "jump_to": "end"
        }
    
    return None

# 输出审核
@after_model(can_jump_to=["end"])
def moderate_output(state: AgentState, runtime: Runtime):
    """审核模型输出"""
    last_ai_msg = state["messages"][-1]
    
    if is_harmful_content(last_ai_msg.content):
        return {
            "messages": [AIMessage("响应已被过滤。")],
            "jump_to": "end"
        }
    
    return None

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[moderate_input, moderate_output]
)
```

### 实例 4：性能监控和日志

全面的性能监控。

```python
import time
from langchain.agents.middleware import AgentMiddleware

class PerformanceMonitorMiddleware(AgentMiddleware):
    """监控 Agent 性能指标"""
    
    def __init__(self):
        self.metrics = {
            "model_calls": 0,
            "tool_calls": 0,
            "total_time": 0,
            "model_time": 0,
            "tool_time": 0,
        }
        self.start_time = None
        self.model_start = None
        self.tool_start = None
    
    def before_agent(self, state, runtime):
        """记录开始时间"""
        self.start_time = time.time()
        print("🚀 Agent 开始执行")
        return None
    
    def before_model(self, state, runtime):
        """记录模型调用"""
        self.model_start = time.time()
        self.metrics["model_calls"] += 1
        print(f"🤖 模型调用 #{self.metrics['model_calls']}")
        return None
    
    def after_model(self, state, runtime):
        """计算模型耗时"""
        elapsed = time.time() - self.model_start
        self.metrics["model_time"] += elapsed
        print(f"✅ 模型完成，耗时 {elapsed:.2f}s")
        return None
    
    def wrap_tool_call(self, request, handler):
        """监控工具执行"""
        self.tool_start = time.time()
        self.metrics["tool_calls"] += 1
        tool_name = request.tool_call.name
        
        print(f"🔧 执行工具: {tool_name}")
        
        try:
            result = handler(request)
            elapsed = time.time() - self.tool_start
            self.metrics["tool_time"] += elapsed
            print(f"✅ 工具完成，耗时 {elapsed:.2f}s")
            return result
        except Exception as e:
            print(f"❌ 工具失败: {e}")
            raise
    
    def after_agent(self, state, runtime):
        """打印总结报告"""
        self.metrics["total_time"] = time.time() - self.start_time
        
        print("\n📊 性能报告:")
        print(f"  总耗时: {self.metrics['total_time']:.2f}s")
        print(f"  模型调用: {self.metrics['model_calls']} 次")
        print(f"  模型耗时: {self.metrics['model_time']:.2f}s")
        print(f"  工具调用: {self.metrics['tool_calls']} 次")
        print(f"  工具耗时: {self.metrics['tool_time']:.2f}s")
        
        return None

# 使用
agent = create_agent(
    model="gpt-4o",
    tools=[web_search, calculator],
    middleware=[PerformanceMonitorMiddleware()]
)
```

---

## 自定义状态管理

Middleware 可以扩展 Agent 的状态，添加自定义字段。

### 方式 1：通过 Middleware 的 state_schema

```python
from langchain.agents.middleware import AgentState, AgentMiddleware
from typing_extensions import NotRequired
from typing import Any

class CustomState(AgentState):
    """扩展状态"""
    model_call_count: NotRequired[int]
    user_id: NotRequired[str]
    preferences: NotRequired[dict]

class CallCounterMiddleware(AgentMiddleware[CustomState]):
    """追踪模型调用次数"""
    state_schema = CustomState
    
    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        """检查调用次数"""
        count = state.get("model_call_count", 0)
        
        if count > 10:
            print("达到调用限制")
            return {"jump_to": "end"}
        
        return None
    
    def after_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        """增加计数"""
        current_count = state.get("model_call_count", 0)
        return {"model_call_count": current_count + 1}

# 使用
agent = create_agent(
    model="gpt-4o",
    middleware=[CallCounterMiddleware()],
    tools=[]
)

# 调用时提供自定义状态
result = agent.invoke({
    "messages": [{"role": "user", "content": "你好"}],
    "model_call_count": 0,
    "user_id": "user-123",
    "preferences": {"language": "zh"}
})

print(f"调用次数: {result['model_call_count']}")
```

### 方式 2：添加工具到 Middleware

```python
from langchain.tools import tool

class UserPreferencesMiddleware(AgentMiddleware):
    """管理用户偏好"""
    state_schema = CustomState
    
    # 添加专属工具
    tools = [
        tool(
            name="get_user_preference",
            description="获取用户偏好设置",
        )(lambda key: state.get("preferences", {}).get(key)),
        
        tool(
            name="set_user_preference",
            description="设置用户偏好",
        )(lambda key, value: {"preferences": {**state.get("preferences", {}), key: value}})
    ]
    
    def before_model(self, state: CustomState, runtime):
        """注入用户偏好到提示词"""
        prefs = state.get("preferences", {})
        
        if prefs:
            system_msg = f"用户偏好: {prefs}"
            # 添加系统消息
            return {"messages": [SystemMessage(content=system_msg)] + state["messages"]}
        
        return None
```

---

## Runtime 访问

Runtime 对象提供依赖注入和上下文信息访问。

### Runtime 包含的信息

```python
from langgraph.runtime import Runtime

# Runtime 包含:
# - Context: 静态信息（用户 ID、数据库连接等）
# - Store: 长期记忆存储
# - Stream writer: 流式传输自定义数据
```

### 在 Middleware 中访问 Runtime

```python
from dataclasses import dataclass
from langchain.agents.middleware import dynamic_prompt, before_model, after_model
from langchain.agents.middleware import ModelRequest

@dataclass
class Context:
    user_name: str
    user_id: str
    database_connection: Any

# 1. 动态提示词中访问
@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    """根据用户信息生成提示词"""
    user_name = request.runtime.context.user_name
    return f"你是 {user_name} 的助手。请友好且简洁地回答。"

# 2. before_model 中访问
@before_model
def log_request(state: AgentState, runtime: Runtime[Context]):
    """记录请求"""
    user_id = runtime.context.user_id
    print(f"处理用户 {user_id} 的请求")
    return None

# 3. after_model 中访问
@after_model
def save_to_db(state: AgentState, runtime: Runtime[Context]):
    """保存到数据库"""
    db = runtime.context.database_connection
    user_id = runtime.context.user_id
    
    # 保存最后一条消息
    last_msg = state["messages"][-1]
    db.save_message(user_id, last_msg)
    
    return None

# 创建 Agent
agent = create_agent(
    model="gpt-4o",
    middleware=[personalized_prompt, log_request, save_to_db],
    tools=[],
    context_schema=Context
)

# 调用时提供 context
agent.invoke(
    {"messages": [{"role": "user", "content": "你好"}]},
    context=Context(
        user_name="Alice",
        user_id="user-123",
        database_connection=db_conn
    )
)
```

### 在工具中访问 Runtime

```python
from langchain.tools import tool, ToolRuntime

@tool
def get_user_data(query: str, runtime: ToolRuntime) -> str:
    """从数据库获取用户数据"""
    user_id = runtime.context.user_id
    db = runtime.context.database_connection
    
    data = db.query(user_id, query)
    return data
```

---

## 实际应用场景

### 场景 1：企业级聊天机器人

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    PIIMiddleware,
    HumanInTheLoopMiddleware
)
from langgraph.checkpoint.postgres import PostgresSaver

# 配置
checkpointer = PostgresSaver(connection_string="postgresql://...")

agent = create_agent(
    model="gpt-4o",
    tools=[
        search_knowledge_base,
        create_ticket,
        send_email,
        query_database
    ],
    middleware=[
        # PII 保护
        PIIMiddleware("email", strategy="redact"),
        PIIMiddleware("ssn", strategy="block"),
        
        # 对话总结
        SummarizationMiddleware(
            model="gpt-4o-mini",
            max_tokens_before_summary=3000,
            messages_to_keep=15
        ),
        
        # 敏感操作需要人工审核
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {"allowed_decisions": ["approve", "edit", "reject"]},
                "query_database": {"allowed_decisions": ["approve", "reject"]}
            }
        ),
        
        # 性能监控
        PerformanceMonitorMiddleware(),
    ],
    checkpointer=checkpointer
)
```

### 场景 2：多租户 SaaS 应用

```python
@dataclass
class TenantContext:
    tenant_id: str
    subscription_tier: str  # "free", "pro", "enterprise"
    api_key: str
    rate_limit: int

class TenantMiddleware(AgentMiddleware):
    """多租户隔离和配额管理"""
    
    def __init__(self):
        self.usage = {}
    
    def before_agent(self, state, runtime: Runtime[TenantContext]):
        """验证和速率限制"""
        tenant_id = runtime.context.tenant_id
        tier = runtime.context.subscription_tier
        
        # 检查使用配额
        usage = self.usage.get(tenant_id, 0)
        limit = runtime.context.rate_limit
        
        if usage >= limit:
            raise ValueError(f"租户 {tenant_id} 超过速率限制")
        
        self.usage[tenant_id] = usage + 1
        return None
    
    def wrap_model_call(self, request, handler):
        """根据订阅层级选择模型"""
        tier = request.runtime.context.subscription_tier
        
        if tier == "enterprise":
            request.model = ChatOpenAI(model="gpt-4o")
            request.tools = all_tools
        elif tier == "pro":
            request.model = ChatOpenAI(model="gpt-4o-mini")
            request.tools = pro_tools
        else:  # free
            request.model = ChatOpenAI(model="gpt-3.5-turbo")
            request.tools = basic_tools
        
        return handler(request)

agent = create_agent(
    model="gpt-4o",
    tools=all_tools,
    middleware=[TenantMiddleware()],
    context_schema=TenantContext
)
```

### 场景 3：教育辅导系统

```python
@dataclass
class StudentContext:
    student_id: str
    grade_level: int
    learning_style: str  # "visual", "auditory", "kinesthetic"
    strengths: list[str]
    weaknesses: list[str]

class AdaptiveLearningMiddleware(AgentMiddleware):
    """自适应学习中间件"""
    
    def __init__(self, student_db):
        self.student_db = student_db
    
    def before_agent(self, state, runtime: Runtime[StudentContext]):
        """加载学生历史"""
        student_id = runtime.context.student_id
        history = self.student_db.get_learning_history(student_id)
        
        if history:
            # 注入学习历史
            context_msg = f"学生学习历史: {history}"
            return {"messages": [SystemMessage(content=context_msg)] + state["messages"]}
        
        return None
    
    def wrap_model_call(self, request, handler):
        """根据学习风格调整提示词"""
        style = request.runtime.context.learning_style
        grade = request.runtime.context.grade_level
        
        # 添加自适应指导
        adaptive_prompt = self._generate_adaptive_prompt(style, grade)
        
        # 修改系统消息
        messages = request.messages
        messages[0] = SystemMessage(content=adaptive_prompt)
        request.messages = messages
        
        return handler(request)
    
    def after_agent(self, state, runtime: Runtime[StudentContext]):
        """保存学习进度"""
        student_id = runtime.context.student_id
        self.student_db.save_session(student_id, state["messages"])
        return None
```

---

## 最佳实践

### 1. **选择合适的 Middleware 类型**

```python
# ✅ 单一钩子 - 使用装饰器
@before_model
def simple_logging(state, runtime):
    print("调用模型")
    return None

# ✅ 多个钩子或维护状态 - 使用类
class ComplexMiddleware(AgentMiddleware):
    def __init__(self):
        self.state = {}
    
    def before_model(self, state, runtime):
        # ...
        return None
    
    def after_model(self, state, runtime):
        # ...
        return None
```

### 2. **明确返回值**

```python
# ✅ 正确：明确返回 None 或 dict
@before_model
def good_middleware(state, runtime):
    if condition:
        return {"messages": modified_messages}
    return None  # 明确返回

# ❌ 错误：隐式返回
@before_model
def bad_middleware(state, runtime):
    if condition:
        return {"messages": modified_messages}
    # 隐式返回 None - 可能引起困惑
```

### 3. **避免副作用**

```python
# ✅ 正确：通过返回值修改状态
@before_model
def modify_state_correctly(state, runtime):
    new_messages = process(state["messages"])
    return {"messages": new_messages}

# ❌ 错误：直接修改状态
@before_model
def modify_state_incorrectly(state, runtime):
    state["messages"].append(new_msg)  # 不要这样做！
    return None
```

### 4. **合理使用 jump_to**

```python
# ✅ 正确：声明可跳转的节点
@after_model(can_jump_to=["end", "retry"])
def controlled_jump(state, runtime):
    if should_end:
        return {"jump_to": "end"}
    elif should_retry:
        return {"jump_to": "retry"}
    return None

# ❌ 错误：跳转到未声明的节点
@after_model(can_jump_to=["end"])
def bad_jump(state, runtime):
    return {"jump_to": "unknown"}  # 错误！
```

### 5. **异常处理**

```python
# ✅ 正确：捕获并处理异常
@wrap_model_call
def safe_wrapper(request, handler):
    try:
        return handler(request)
    except RateLimitError as e:
        # 等待并重试
        time.sleep(60)
        return handler(request)
    except Exception as e:
        # 记录并重新抛出
        logger.error(f"模型调用失败: {e}")
        raise

# ❌ 错误：吞掉异常
@wrap_model_call
def unsafe_wrapper(request, handler):
    try:
        return handler(request)
    except:
        return None  # 不要这样做！
```

### 6. **注意执行顺序**

```python
# 按功能分组和排序 Middleware
agent = create_agent(
    model="gpt-4o",
    middleware=[
        # 1. 输入验证和清理
        PIIMiddleware(...),
        InputValidationMiddleware(),
        
        # 2. 上下文管理
        SummarizationMiddleware(...),
        ContextInjectionMiddleware(),
        
        # 3. 业务逻辑
        DynamicModelSelectionMiddleware(),
        ToolSelectionMiddleware(),
        
        # 4. 安全和合规
        ContentModerationMiddleware(),
        HumanInTheLoopMiddleware(...),
        
        # 5. 监控和日志
        PerformanceMonitorMiddleware(),
        AuditLogMiddleware(),
    ],
    tools=[...]
)
```

### 7. **文档化自定义 Middleware**

```python
class CustomMiddleware(AgentMiddleware):
    """
    自定义中间件说明
    
    功能:
    - 功能 1 描述
    - 功能 2 描述
    
    参数:
    - param1: 参数 1 说明
    - param2: 参数 2 说明
    
    示例:
        >>> middleware = CustomMiddleware(param1="value")
        >>> agent = create_agent(middleware=[middleware], ...)
    """
    
    def __init__(self, param1: str, param2: int = 10):
        self.param1 = param1
        self.param2 = param2
```

### 8. **测试 Middleware**

```python
import pytest
from langchain.agents import create_agent

def test_middleware_behavior():
    """测试 Middleware 行为"""
    
    # 创建带有 Middleware 的 Agent
    agent = create_agent(
        model="gpt-4o-mini",
        middleware=[CustomMiddleware()],
        tools=[]
    )
    
    # 测试特定行为
    result = agent.invoke({"messages": [{"role": "user", "content": "测试"}]})
    
    # 验证结果
    assert result is not None
    assert "messages" in result
```

---

## 性能优化

### 1. **缓存昂贵的操作**

```python
from functools import lru_cache

class CachingMiddleware(AgentMiddleware):
    """缓存模型响应"""
    
    @lru_cache(maxsize=100)
    def _compute_cache_key(self, messages_tuple):
        """计算缓存键"""
        return hash(messages_tuple)
    
    def wrap_model_call(self, request, handler):
        """使用缓存"""
        # 将消息转换为可哈希的元组
        messages_tuple = tuple(
            (m.role, m.content) for m in request.messages
        )
        
        cache_key = self._compute_cache_key(messages_tuple)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        response = handler(request)
        self.cache[cache_key] = response
        return response
```

### 2. **延迟加载**

```python
class LazyLoadMiddleware(AgentMiddleware):
    """延迟加载资源"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._config = None
    
    @property
    def config(self):
        """延迟加载配置"""
        if self._config is None:
            self._config = load_config(self.config_path)
        return self._config
    
    def before_model(self, state, runtime):
        # 只在需要时加载配置
        setting = self.config.get("some_setting")
        # ...
        return None
```

### 3. **批处理**

```python
class BatchProcessingMiddleware(AgentMiddleware):
    """批处理消息"""
    
    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
        self.batch = []
    
    def before_model(self, state, runtime):
        """收集批次"""
        self.batch.append(state["messages"][-1])
        
        if len(self.batch) >= self.batch_size:
            # 批量处理
            processed = batch_process(self.batch)
            self.batch = []
            return {"messages": processed}
        
        return None
```

### 4. **异步操作**

```python
import asyncio

class AsyncMiddleware(AgentMiddleware):
    """异步操作支持"""
    
    async def _fetch_data_async(self, user_id):
        """异步获取数据"""
        # 异步数据库查询
        return await database.fetch_async(user_id)
    
    def before_agent(self, state, runtime):
        """同步包装异步操作"""
        user_id = runtime.context.user_id
        
        # 运行异步操作
        data = asyncio.run(self._fetch_data_async(user_id))
        
        return {"custom_data": data}
```

### 5. **限制消息历史**

```python
@before_model
def limit_history_size(state: AgentState, runtime: Runtime):
    """限制消息历史大小以提高性能"""
    messages = state["messages"]
    MAX_MESSAGES = 20
    
    if len(messages) > MAX_MESSAGES:
        # 保留系统消息和最近的消息
        system_msgs = [m for m in messages if m.role == "system"]
        recent_msgs = messages[-MAX_MESSAGES:]
        
        return {"messages": system_msgs + recent_msgs}
    
    return None
```

---

## 快速参考

### Middleware 钩子对比

| 钩子 | 时机 | 返回类型 | 主要用途 |
|------|------|----------|----------|
| `before_agent` | Agent 开始前 | `dict \| None` | 加载内存、验证输入 |
| `before_model` | 每次模型调用前 | `dict \| None` | 修剪消息、注入上下文 |
| `wrap_model_call` | 包装模型调用 | `ModelResponse` | 重试、缓存、动态模型 |
| `after_model` | 每次模型响应后 | `dict \| None` | 验证输出、护栏 |
| `wrap_tool_call` | 包装工具调用 | `ToolMessage` | 错误处理、权限检查 |
| `after_agent` | Agent 完成后 | `dict \| None` | 保存结果、清理 |

### 装饰器快速参考

```python
from langchain.agents.middleware import (
    before_agent,
    before_model,
    wrap_model_call,
    after_model,
    wrap_tool_call,
    after_agent,
    dynamic_prompt
)

# 节点风格 (返回 dict | None)
@before_agent
def hook(state: AgentState, runtime: Runtime) -> dict | None:
    return None

@before_model
def hook(state: AgentState, runtime: Runtime) -> dict | None:
    return None

@after_model(can_jump_to=["end"])
def hook(state: AgentState, runtime: Runtime) -> dict | None:
    return None

@after_agent
def hook(state: AgentState, runtime: Runtime) -> dict | None:
    return None

# 包装风格 (返回响应对象)
@wrap_model_call
def hook(request: ModelRequest, handler: Callable) -> ModelResponse:
    return handler(request)

@wrap_tool_call
def hook(request, handler):
    return handler(request)

# 动态提示词
@dynamic_prompt
def hook(request: ModelRequest) -> str:
    return "系统提示词"
```

### 常用模式

#### 1. 简单日志

```python
@before_model
def log(state, runtime):
    print(f"消息: {len(state['messages'])}")
    return None
```

#### 2. 消息修剪

```python
@before_model
def trim(state, runtime):
    if len(state["messages"]) > 10:
        return {"messages": state["messages"][-10:]}
    return None
```

#### 3. 重试逻辑

```python
@wrap_model_call
def retry(request, handler):
    for i in range(3):
        try:
            return handler(request)
        except Exception as e:
            if i == 2: raise
            time.sleep(2 ** i)
```

#### 4. 动态提示词

```python
@dynamic_prompt
def prompt(request: ModelRequest) -> str:
    role = request.runtime.context.role
    return f"你是一个 {role} 助手"
```

#### 5. 错误处理

```python
@wrap_tool_call
def handle_errors(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"错误: {e}",
            tool_call_id=request.tool_call.id
        )
```

### 预置 Middleware 配置

```python
# 总结
SummarizationMiddleware(
    model="gpt-4o-mini",
    max_tokens_before_summary=4000,
    messages_to_keep=20,
    summary_prompt="可选自定义提示词"
)

# PII 保护
PIIMiddleware(
    "email",                    # PII 类型
    strategy="redact",          # "redact" 或 "block"
    apply_to_input=True,        # 应用于输入
    detector=r"regex_pattern"   # 可选自定义正则
)

# 人工审核
HumanInTheLoopMiddleware(
    interrupt_on={
        "tool_name": {
            "allowed_decisions": ["approve", "edit", "reject"]
        }
    }
)

# 上下文编辑
ContextEditingMiddleware(
    edits=[
        ClearToolUsesEdit(trigger=1000)
    ]
)
```

---

## 总结

LangChain Middleware 提供了强大而灵活的 Agent 行为控制能力：

**核心优势：**
✅ **精确控制** - 在 Agent 执行的每个阶段插入逻辑  
✅ **可组合性** - 多个 Middleware 可以组合使用  
✅ **关注点分离** - 横切关注点与业务逻辑分离  
✅ **预置功能** - 总结、PII保护、人工审核等开箱即用  
✅ **灵活扩展** - 轻松创建自定义 Middleware  

**关键要点：**
- 使用 `@装饰器` 快速添加单一钩子功能
- 使用 `类` 实现复杂的多钩子逻辑
- 注意执行顺序：before 顺序、after 反向、wrap 嵌套
- 通过 Runtime 访问上下文和依赖注入
- 利用预置 Middleware 处理常见场景
- 自定义状态扩展 Agent 能力

通过合理使用 Middleware，你可以构建高度定制化、可维护、安全的 LLM Agent 应用！
