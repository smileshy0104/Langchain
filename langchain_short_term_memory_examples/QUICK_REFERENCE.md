# 🚀 LangChain 短期记忆快速参考

## 📋 基础用法

### 启用短期记忆

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
agent = create_agent(
    model=model,
    tools=[],
    checkpointer=checkpointer
)

config = {"configurable": {"thread_id": "session-1"}}
agent.invoke({"messages": "..."}, config)
```

### 多用户会话

```python
config_a = {"configurable": {"thread_id": "user-A"}}
config_b = {"configurable": {"thread_id": "user-B"}}

agent.invoke({...}, config_a)  # 用户A
agent.invoke({...}, config_b)  # 用户B
```

---

## 🔧 消息管理

### 方式1：修剪消息

```python
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

@before_model
def trim_messages(state, runtime):
    max_messages = 6
    if len(state["messages"]) > max_messages:
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                state["messages"][0],
                *state["messages"][-(max_messages-1):]
            ]
        }
```

### 方式2：消息摘要

```python
from langchain.agents.middleware import SummarizationMiddleware

middleware=[
    SummarizationMiddleware(
        model=model,
        trigger={"messages": 6},   # 超过6条时触发
        keep={"messages": 3},      # 保留最近3条
    )
]
```

### 删除所有消息

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

agent.update_state(
    config,
    {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
)
```

---

## 🎨 自定义状态

### 定义自定义状态

```python
from langchain.agents import AgentState

class MyState(AgentState):
    user_id: str = ""
    user_name: str = ""
    preferences: dict = {}
    session_count: int = 0

agent = create_agent(
    model=model,
    state_schema=MyState
)
```

### 使用自定义状态

```python
agent.invoke({
    "messages": [...],
    "user_id": "123",
    "user_name": "张三",
    "preferences": {"theme": "dark"}
}, config)
```

---

## 🛠️ 工具与状态

### 工具读取状态

```python
from langchain.tools import tool, ToolRuntime

@tool
def get_info(runtime: ToolRuntime) -> str:
    user_id = runtime.state.get("user_id")
    return f"用户: {user_id}"
```

### 工具写入状态

```python
from langgraph.types import Command
from langchain.messages import ToolMessage

@tool
def update_info(runtime: ToolRuntime) -> Command:
    return Command(update={
        "user_name": "张三",
        "messages": [
            ToolMessage(
                "已更新",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })
```

---

## 📊 状态查询

### 获取当前状态

```python
state = agent.get_state(config)
print(state.values)
print(f"消息数: {len(state.values['messages'])}")
```

### 更新状态

```python
agent.update_state(config, {
    "user_name": "新名字",
    "session_count": 5
})
```

---

## 🔄 中间件类型

### @before_model

在模型调用**前**执行

```python
@before_model
def my_middleware(state, runtime):
    # 修改消息
    return {"messages": [...]}
```

### @after_model

在模型调用**后**执行

```python
@after_model
def my_middleware(state, runtime):
    # 处理响应
    return {"messages": [...]}
```

### @dynamic_prompt

动态生成提示词

```python
@dynamic_prompt
def my_prompt(request):
    user = request.runtime.context["user_name"]
    return f"你好，{user}！"
```

---

## 💾 持久化选项

### 内存存储（开发）

```python
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
```

### PostgreSQL（生产）

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:pass@localhost:5432/db"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    agent = create_agent(model, checkpointer=checkpointer)
```

---

## ⚙️ 摘要配置选项

### 触发条件

```python
# 按消息数
trigger={"messages": 10}

# 按 Token 数
trigger={"tokens": 4000}

# 按比例（上下文的80%）
trigger={"fraction": 0.8}

# 组合条件（AND）
trigger={"messages": 10, "tokens": 5000}

# 多条件（OR）
trigger=[
    {"messages": 10},
    {"tokens": 5000}
]
```

### 保留策略

```python
# 保留消息数
keep={"messages": 5}

# 保留 Token 数
keep={"tokens": 1000}

# 保留比例
keep={"fraction": 0.3}
```

---

## 🐛 常见错误

### 错误1：记忆不工作

```python
# ❌ 错误
agent = create_agent(model, tools=[])

# ✅ 正确
checkpointer = MemorySaver()
agent = create_agent(model, tools=[], checkpointer=checkpointer)
```

### 错误2：会话混乱

```python
# ❌ 错误 - 没有 thread_id
agent.invoke({"messages": "..."})

# ✅ 正确 - 指定 thread_id
config = {"configurable": {"thread_id": "session-1"}}
agent.invoke({"messages": "..."}, config)
```

### 错误3：状态未保存

```python
# ❌ 错误 - 直接修改状态
state.values["user_name"] = "新名字"

# ✅ 正确 - 使用 update_state
agent.update_state(config, {"user_name": "新名字"})
```

---

## 📈 性能优化

### 1. 选择合适的模型

```python
# 摘要使用便宜的模型
SummarizationMiddleware(
    model="gpt-4o-mini",  # 便宜
    # ...
)

# 主 Agent 使用强大的模型
agent = create_agent(
    model="gpt-4o",  # 强大
    # ...
)
```

### 2. 合理设置触发条件

```python
# 避免过早触发
trigger={"tokens": 4000}  # 接近上下文限制

# 避免过晚触发
trigger={"tokens": 8000}  # 可能已经超限
```

### 3. 组合使用策略

```python
middleware=[
    trim_messages,           # 先快速修剪
    SummarizationMiddleware(...),  # 再智能摘要
]
```

---

## 📚 示例代码片段

### 完整示例

```python
from langchain.agents import create_agent, AgentState
from langchain_community.chat_models import ChatZhipuAI
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import MemorySaver

# 自定义状态
class MyState(AgentState):
    user_id: str = ""
    user_name: str = ""

# 创建 Agent
model = ChatZhipuAI(model="glm-4.6")
checkpointer = MemorySaver()

agent = create_agent(
    model=model,
    tools=[],
    state_schema=MyState,
    checkpointer=checkpointer,
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger={"messages": 6},
            keep={"messages": 3}
        )
    ],
    system_prompt="你是一个助手"
)

# 使用
config = {"configurable": {"thread_id": "session-1"}}

result = agent.invoke({
    "messages": [{"role": "user", "content": "你好"}],
    "user_id": "user_123",
    "user_name": "张三"
}, config)

print(result['messages'][-1].content)
```

---

## 🔍 调试技巧

### 查看消息历史

```python
state = agent.get_state(config)
for i, msg in enumerate(state.values['messages']):
    print(f"{i}. [{msg.type}] {msg.content[:50]}...")
```

### 查看状态值

```python
state = agent.get_state(config)
print(f"用户ID: {state.values.get('user_id')}")
print(f"用户名: {state.values.get('user_name')}")
```

### 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 📞 快速帮助

### 问题排查清单

- [ ] 是否设置了 `checkpointer`？
- [ ] 是否指定了 `thread_id`？
- [ ] `thread_id` 在多次调用中是否一致？
- [ ] 状态字段名是否正确？
- [ ] 是否设置了 API Key？

### 获取帮助

1. 查看 [README.md](README.md)
2. 阅读 [LEARNING_GUIDE.md](LEARNING_GUIDE.md)
3. 查看官方文档
4. 运行示例代码

---

**快速参考卡片版本：v1.0**

**最后更新：2024-11-29**
