# LangChain 短期记忆（Short-Term Memory）完整指南

> 基于官方文档：https://docs.langchain.com/oss/python/langchain/short-term-memory

## 📖 概述

短期记忆（Short-Term Memory）是 LangChain Agent 的核心功能之一，它允许 Agent 在单个会话（thread）中记住之前的交互。通过合理使用短期记忆，可以构建具有上下文感知能力的智能对话系统。

## 🎯 核心概念

### 什么是短期记忆？

- **定义**：在单个线程（thread）或会话中记住之前交互的能力
- **管理方式**：通过 LangGraph 的状态（State）和检查点（Checkpointer）管理
- **持久化**：使用 `MemorySaver`（内存）或 `PostgresSaver`（数据库）

### 为什么需要短期记忆？

1. **多轮对话**：记住用户之前说的内容
2. **上下文连贯性**：基于历史信息做出回应
3. **状态管理**：跟踪任务进度和中间结果
4. **会话隔离**：不同用户的对话相互独立

## 📦 安装依赖

```bash
# 基础依赖
pip install langchain langgraph langchain-community

# 使用智谱 AI（GLM）
pip install zhipuai

# 生产环境使用 PostgreSQL
pip install langgraph-checkpoint-postgres
```

## 🚀 快速开始

### 最简单的短期记忆示例

```python
from langchain.agents import create_agent
from langchain_community.chat_models import ChatZhipuAI
from langgraph.checkpoint.memory import MemorySaver

# 创建模型和检查点器
model = ChatZhipuAI(model="glm-4.6")
checkpointer = MemorySaver()

# 创建 Agent
agent = create_agent(
    model=model,
    tools=[],
    checkpointer=checkpointer,  # 启用短期记忆
    system_prompt="你是一个助手"
)

# 会话配置（thread_id 标识唯一会话）
config = {"configurable": {"thread_id": "user-1"}}

# 第一轮对话
agent.invoke(
    {"messages": [{"role": "user", "content": "我叫张三"}]},
    config
)

# 第二轮对话 - Agent 记得你的名字
result = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
    config
)
print(result['messages'][-1].content)  # 输出：你叫张三
```

## 📚 完整示例说明

`06_short_term_memory_comprehensive.py` 包含 10 个完整示例：

### 1️⃣ 基础短期记忆

**功能**：使用 `MemorySaver` 实现基本的对话记忆

**核心代码**：
```python
checkpointer = MemorySaver()
agent = create_agent(model, tools=[], checkpointer=checkpointer)
config = {"configurable": {"thread_id": "conversation-1"}}
```

**应用场景**：
- 简单的多轮对话
- 客服机器人
- 个人助理

---

### 2️⃣ 多线程会话管理

**功能**：同时管理多个用户的独立会话

**核心代码**：
```python
config_a = {"configurable": {"thread_id": "user-A"}}
config_b = {"configurable": {"thread_id": "user-B"}}

agent.invoke({"messages": [...]}, config_a)  # 用户A的会话
agent.invoke({"messages": [...]}, config_b)  # 用户B的会话
```

**应用场景**：
- 多用户在线客服
- SaaS 应用
- 聊天室

---

### 3️⃣ 消息修剪（Trim Messages）

**功能**：自动修剪过长的对话历史，控制上下文窗口大小

**核心代码**：
```python
@before_model
def trim_messages_middleware(state: AgentState, runtime: Runtime):
    max_messages = 6
    messages = state["messages"]

    if len(messages) > max_messages:
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                messages[0],  # 保留系统消息
                *messages[-(max_messages-1):]  # 保留最近的消息
            ]
        }
    return None

agent = create_agent(model, middleware=[trim_messages_middleware])
```

**应用场景**：
- 长对话管理
- Token 成本控制
- 避免超出模型上下文限制

---

### 4️⃣ 消息删除（Remove Messages）

**功能**：删除特定的消息或批量删除消息

**核心代码**：
```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

# 删除特定消息
@after_model
def delete_old_messages(state: AgentState, runtime: Runtime):
    messages = state["messages"]
    if len(messages) > 4:
        to_delete = messages[1:3]  # 删除第2和第3条
        return {"messages": [RemoveMessage(id=m.id) for m in to_delete]}
    return None

# 删除所有消息
return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```

**应用场景**：
- 清理敏感信息
- 重置会话
- 定期清理历史

---

### 5️⃣ 消息摘要（Summarization）

**功能**：自动总结对话历史，压缩上下文

**核心代码**：
```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model=model,
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger={"messages": 6},   # 超过6条消息时触发
            keep={"messages": 3},      # 保留最近3条
        )
    ]
)
```

**配置选项**：
- `trigger`: 触发条件
  - `{"messages": 10}` - 消息数量
  - `{"tokens": 4000}` - Token 数量
  - `{"fraction": 0.8}` - 上下文使用率
- `keep`: 保留策略
  - `{"messages": 5}` - 保留消息数
  - `{"tokens": 1000}` - 保留 Token 数
  - `{"fraction": 0.3}` - 保留比例

**应用场景**：
- 超长对话
- 保留完整上下文但压缩存储
- 提升响应速度

---

### 6️⃣ 自定义状态（Custom State）

**功能**：扩展 `AgentState`，添加自定义字段

**核心代码**：
```python
from langchain.agents import AgentState

class UserPreferencesState(AgentState):
    user_id: str = ""
    preferences: dict = {}
    session_count: int = 0

agent = create_agent(
    model=model,
    state_schema=UserPreferencesState
)

# 使用自定义状态
agent.invoke({
    "messages": [...],
    "user_id": "user_123",
    "preferences": {"theme": "dark"},
    "session_count": 1
})
```

**应用场景**：
- 用户偏好管理
- 任务状态跟踪
- 多阶段工作流

---

### 7️⃣ 工具中读取状态

**功能**：工具函数访问当前会话状态

**核心代码**：
```python
from langchain.tools import tool, ToolRuntime

@tool
def get_user_profile(runtime: ToolRuntime) -> str:
    """从状态中读取 user_id"""
    user_id = runtime.state.get("user_id", "unknown")
    # 使用 user_id 查询数据库...
    return f"用户信息: {user_id}"

agent = create_agent(
    model=model,
    tools=[get_user_profile],
    state_schema=CustomState
)
```

**应用场景**：
- 个性化服务
- 上下文相关的工具调用
- 会话级缓存

---

### 8️⃣ 工具中写入状态

**功能**：工具函数修改会话状态

**核心代码**：
```python
from langgraph.types import Command
from langchain.messages import ToolMessage

@tool
def update_user_name(runtime: ToolRuntime) -> Command:
    """更新用户名到状态"""
    user_name = "张三"  # 从数据库查询

    return Command(update={
        "user_name": user_name,
        "messages": [
            ToolMessage(
                f"已更新用户名: {user_name}",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })
```

**应用场景**：
- 工作流状态更新
- 中间结果保存
- 动态数据收集

---

### 9️⃣ 动态提示词

**功能**：根据上下文动态生成系统提示词

**核心代码**：
```python
from langchain.agents.middleware import dynamic_prompt

@dynamic_prompt
def create_dynamic_system_prompt(request) -> str:
    user_name = request.runtime.context["user_name"]
    time_of_day = request.runtime.context["time_of_day"]

    greeting = "早上好" if time_of_day == "morning" else "你好"
    return f"{greeting}，{user_name}！我是你的助手。"

agent = create_agent(
    model=model,
    middleware=[create_dynamic_system_prompt],
    context_schema=CustomContext
)
```

**应用场景**：
- 个性化问候
- 角色扮演
- 多语言支持

---

### 🔟 状态查询与管理

**功能**：查询和管理会话状态

**核心代码**：
```python
config = {"configurable": {"thread_id": "session-1"}}

# 发送消息
agent.invoke({"messages": [...]}, config)

# 查询状态
state = agent.get_state(config)
print(f"消息数量: {len(state.values['messages'])}")
print(f"最新消息: {state.values['messages'][-1].content}")

# 查看所有消息
for msg in state.values['messages']:
    print(f"[{msg.type}] {msg.content}")
```

**应用场景**：
- 调试和监控
- 会话分析
- 状态导出

---

## 🔧 生产环境最佳实践

### 1. 使用持久化存储

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:pass@localhost:5432/mydb"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # 自动创建表

    agent = create_agent(
        model=model,
        checkpointer=checkpointer
    )
```

### 2. 合理配置摘要策略

```python
# 推荐配置：基于 Token 数触发
SummarizationMiddleware(
    model="gpt-4o-mini",  # 使用便宜的模型做摘要
    trigger={"tokens": 4000},  # 接近上下文限制时触发
    keep={"messages": 20},     # 保留足够的上下文
)
```

### 3. 组合使用多种策略

```python
agent = create_agent(
    model=model,
    middleware=[
        trim_messages_middleware,      # 先修剪
        SummarizationMiddleware(...),  # 再摘要
        custom_validation_middleware,  # 最后验证
    ],
    checkpointer=checkpointer
)
```

### 4. 错误处理

```python
try:
    result = agent.invoke({"messages": [...]}, config)
except Exception as e:
    # 记录错误
    logger.error(f"Agent error: {e}")
    # 清理状态
    agent.update_state(
        config,
        {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
    )
```

## 📊 性能优化建议

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **消息修剪** | 快速、无成本 | 丢失信息 | 短对话、实时应用 |
| **消息删除** | 精确控制 | 需手动管理 | 敏感信息、定制需求 |
| **消息摘要** | 保留语义 | 有成本、延迟 | 长对话、知识保留 |
| **自定义状态** | 灵活强大 | 复杂度高 | 复杂业务逻辑 |

## 🐛 常见问题

### Q1: 为什么 Agent 记不住之前的对话？

**A**: 确保添加了 `checkpointer` 并且使用相同的 `thread_id`

```python
checkpointer = MemorySaver()  # ✓ 添加检查点器
agent = create_agent(model, checkpointer=checkpointer)

config = {"configurable": {"thread_id": "same-id"}}  # ✓ 使用相同ID
```

### Q2: 如何清空某个会话的历史？

**A**: 使用 `RemoveMessage` 删除所有消息

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

agent.update_state(
    {"configurable": {"thread_id": "session-1"}},
    {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
)
```

### Q3: 消息摘要会永久修改历史吗？

**A**: 是的，`SummarizationMiddleware` 会永久替换旧消息为摘要。如果需要临时压缩，使用 `@before_model` 修剪。

### Q4: 如何在多个 Agent 之间共享状态？

**A**: 使用相同的 `checkpointer` 和 `thread_id`

```python
checkpointer = MemorySaver()

agent1 = create_agent(model1, checkpointer=checkpointer)
agent2 = create_agent(model2, checkpointer=checkpointer)

config = {"configurable": {"thread_id": "shared"}}
agent1.invoke({...}, config)  # Agent1 写入
agent2.invoke({...}, config)  # Agent2 读取
```

### Q5: InMemorySaver vs PostgresSaver？

| 特性 | InMemorySaver | PostgresSaver |
|------|---------------|---------------|
| **持久化** | ❌ 进程重启丢失 | ✅ 数据库持久化 |
| **性能** | ⚡ 极快 | 🐌 网络延迟 |
| **扩展性** | ❌ 单机 | ✅ 分布式 |
| **适用场景** | 开发测试 | 生产环境 |

## 🎓 学习路径

```
1. 基础入门（1-2小时）
   ├─ 运行示例1：基础短期记忆
   ├─ 运行示例2：多线程会话管理
   └─ 理解 thread_id 和 checkpointer

2. 消息管理（2-3小时）
   ├─ 运行示例3：消息修剪
   ├─ 运行示例4：消息删除
   ├─ 运行示例5：消息摘要
   └─ 对比三种策略的差异

3. 高级功能（3-4小时）
   ├─ 运行示例6：自定义状态
   ├─ 运行示例7-8：工具读写状态
   ├─ 运行示例9：动态提示词
   └─ 运行示例10：状态管理

4. 实战项目（1周+）
   ├─ 构建多轮对话客服系统
   ├─ 实现个性化推荐 Agent
   └─ 部署到生产环境
```

## 🔗 相关资源

- **官方文档**: https://docs.langchain.com/oss/python/langchain/short-term-memory
- **LangGraph 文档**: https://langchain-ai.github.io/langgraph/
- **示例代码**: [06_short_term_memory_comprehensive.py](./06_short_term_memory_comprehensive.py)
- **GitHub 仓库**: https://github.com/langchain-ai/langchain

## 📞 运行示例

```bash
# 设置 API Key
export ZHIPUAI_API_KEY="your-api-key"

# 运行完整示例
python langchain_agents_examples/06_short_term_memory_comprehensive.py

# 选择要运行的示例（1-10）或输入 0 运行全部
```

## 🎉 总结

短期记忆是构建智能对话系统的基础能力。通过合理使用：

- ✅ **基础记忆**：实现多轮对话
- ✅ **消息管理**：控制上下文大小
- ✅ **自定义状态**：扩展业务逻辑
- ✅ **工具集成**：读写会话状态
- ✅ **动态提示**：个性化交互

你可以构建出强大、高效、用户友好的 AI Agent！

---

**Happy Coding! 🚀**
