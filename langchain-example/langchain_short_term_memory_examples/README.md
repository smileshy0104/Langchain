��# LangChain 短期记忆（Short-Term Memory）示例集合

> 基于官方文档：https://docs.langchain.com/oss/python/langchain/short-term-memory
>
> 使用 GLM-4.6 模型实现

## 📖 项目简介

本项目提供了完整的 LangChain 短期记忆功能示例代码，涵盖从基础到高级的所有核心功能。每个示例都可以独立运行，包含详细的注释和说明。

## 🎯 什么是短期记忆？

**短期记忆（Short-Term Memory）** 是指在单个会话（thread）或对话中记住之前交互的能力。它允许 AI Agent：

- ✅ 记住用户在对话中说过的内容
- ✅ 基于历史上下文做出回应
- ✅ 跟踪任务进度和状态
- ✅ 提供连贯的多轮对话体验

## 📦 安装依赖

```bash
# 基础依赖
pip install langchain langgraph langchain-community

# 智谱 AI（GLM）
pip install zhipuai

# 可选：生产环境使用 PostgreSQL
pip install langgraph-checkpoint-postgres
```

## 🗂️ 项目结构

```
langchain_short_term_memory_examples/
├── README.md                          # 本文件
├── 01_basic_memory.py                 # 基础短期记忆
├── 02_multi_thread.py                 # 多线程会话管理
├── 03_trim_messages.py                # 消息修剪
├── 04_summarization.py                # 消息摘要
├── 05_custom_state.py                 # 自定义状态
├── 06_tool_state_access.py            # 工具读写状态
└── requirements.txt                   # 依赖列表
```

## 📚 示例说明

### 1️⃣ 基础短期记忆 ([01_basic_memory.py](01_basic_memory.py))

**功能**：最基本的短期记忆实现

**核心代码**：
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
agent = create_agent(model, checkpointer=checkpointer)
config = {"configurable": {"thread_id": "session-1"}}
```

**学到什么**：
- ✅ 如何启用短期记忆
- ✅ 使用 `InMemorySaver` 存储对话
- ✅ 通过 `thread_id` 标识会话

**运行**：
```bash
python 01_basic_memory.py
```

---

### 2️⃣ 多线程会话管理 ([02_multi_thread.py](02_multi_thread.py))

**功能**：同时管理多个用户的独立会话

**核心代码**：
```python
config_a = {"configurable": {"thread_id": "user-A"}}
config_b = {"configurable": {"thread_id": "user-B"}}

agent.invoke({...}, config_a)  # 用户A的会话
agent.invoke({...}, config_b)  # 用户B的会话
```

**学到什么**：
- ✅ 会话隔离机制
- ✅ 多用户场景处理
- ✅ 独立的对话上下文

**运行**：
```bash
python 02_multi_thread.py
```

---

### 3️⃣ 消息修剪 ([03_trim_messages.py](03_trim_messages.py))

**功能**：自动修剪过长的对话历史

**核心代码**：
```python
@before_model
def trim_messages_middleware(state: AgentState, runtime: Runtime):
    max_messages = 6
    if len(state["messages"]) > max_messages:
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                state["messages"][0],  # 保留系统消息
                *state["messages"][-(max_messages-1):]  # 保留最近的
            ]
        }
```

**学到什么**：
- ✅ 使用 `@before_model` 中间件
- ✅ `RemoveMessage` 的使用
- ✅ 控制上下文窗口大小

**运行**：
```bash
python 03_trim_messages.py
```

---

### 4️⃣ 消息摘要 ([04_summarization.py](04_summarization.py))

**功能**：自动总结对话历史，压缩上下文

**核心代码**：
```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model=model,
    middleware=[
        SummarizationMiddleware(
            model=model,                       # 用于生成摘要的模型
            max_tokens_before_summary=1000,    # Token数超过1000时触发摘要
            messages_to_keep=3,                # 摘要后保留最近3条消息
        )
    ]
)
```

**学到什么**：
- ✅ `SummarizationMiddleware` 的使用
- ✅ 基于 Token 数量的触发条件
- ✅ 控制保留消息的数量

**运行**：
```bash
python 04_summarization.py
```

---

### 5️⃣ 自定义状态 ([05_custom_state.py](05_custom_state.py))

**功能**：扩展 AgentState 添加业务字段

**核心代码**：
```python
class UserProfileState(AgentState):
    user_id: str = ""
    user_name: str = ""
    preferences: dict = {}
    session_count: int = 0

agent = create_agent(
    model=model,
    state_schema=UserProfileState
)
```

**学到什么**：
- ✅ 继承 `AgentState`
- ✅ 添加自定义字段
- ✅ 状态自动持久化

**运行**：
```bash
python 05_custom_state.py
```

---

### 6️⃣ 工具读写状态 ([06_tool_state_access.py](06_tool_state_access.py))

**功能**：工具函数访问和修改会话状态

**核心代码**：
```python
# 读取状态
@tool
def get_info(runtime: ToolRuntime) -> str:
    user_id = runtime.state.get("user_id")
    return f"用户ID: {user_id}"

# 写入状态
@tool
def update_info(runtime: ToolRuntime) -> Command:
    return Command(update={
        "user_name": "张三",
        "messages": [ToolMessage("已更新")]
    })
```

**学到什么**：
- ✅ 使用 `ToolRuntime` 访问状态
- ✅ 使用 `Command` 更新状态
- ✅ 工具与状态的集成

**运行**：
```bash
python 06_tool_state_access.py
```

---

## 🚀 快速开始

### 1. 设置环境变量

```bash
export ZHIPUAI_API_KEY="your-api-key-here"
```

或在代码中设置：
```python
import os
os.environ["ZHIPUAI_API_KEY"] = "your-api-key"
```

### 2. 运行示例

```bash
# 运行任意示例
python 01_basic_memory.py
python 02_multi_thread.py
# ...
```

### 3. 查看输出

每个示例都会打印详细的执行过程和结果，包括：
- 👤 用户输入
- 🤖 AI 回复
- 📊 统计信息
- 💡 说明提示

## 📊 功能对比

| 示例 | 功能 | 复杂度 | 适用场景 |
|------|------|--------|----------|
| 01 | 基础记忆 | ⭐ | 简单对话 |
| 02 | 多线程 | ⭐⭐ | 多用户系统 |
| 03 | 消息修剪 | ⭐⭐⭐ | Token 控制 |
| 04 | 消息摘要 | ⭐⭐⭐⭐ | 长对话 |
| 05 | 自定义状态 | ⭐⭐⭐ | 复杂业务 |
| 06 | 工具状态 | ⭐⭐⭐⭐ | 高级集成 |

## 🔧 核心概念

### Thread ID（会话ID）

```python
config = {"configurable": {"thread_id": "unique-session-id"}}
```

- 唯一标识一个会话
- 相同 `thread_id` = 共享记忆
- 不同 `thread_id` = 隔离记忆

### Checkpointer（检查点器）

```python
# 开发环境：内存存储
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# 生产环境：数据库存储
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string("postgresql://...")
```

### Middleware（中间件）

```python
@before_model  # 模型调用前
@after_model   # 模型调用后
@dynamic_prompt  # 动态提示词
```

## 💡 最佳实践

### 1. 选择合适的存储方式

| 场景 | 推荐方案 |
|------|---------|
| 开发测试 | `MemorySaver` |
| 生产部署 | `PostgresSaver` |
| 分布式系统 | `PostgresSaver` + 连接池 |

### 2. 控制上下文大小

```python
# 方式1：消息修剪（快速，无成本）
middleware=[trim_messages_middleware]

# 方式2：消息摘要（保留语义，有成本）
middleware=[SummarizationMiddleware(...)]

# 方式3：组合使用
middleware=[trim_messages, SummarizationMiddleware(...)]
```

### 3. 合理设置触发条件

```python
SummarizationMiddleware(
    model=model,                      # 使用与主模型相同或更便宜的模型
    max_tokens_before_summary=1500,   # 接近上下文限制的70-80%
    messages_to_keep=10,              # 保留足够上下文
)
```

## 🐛 常见问题

### Q1: Agent 为什么记不住之前的对话？

**A**: 检查两点：
1. 是否添加了 `checkpointer`
2. 是否使用了相同的 `thread_id`

```python
# ❌ 错误示例
agent = create_agent(model, tools=[])  # 缺少 checkpointer

# ✅ 正确示例
checkpointer = MemorySaver()
agent = create_agent(model, tools=[], checkpointer=checkpointer)
```

### Q2: 如何清空会话历史？

**A**: 使用 `RemoveMessage`

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

agent.update_state(
    {"configurable": {"thread_id": "session-1"}},
    {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
)
```

### Q3: 消息摘要会永久修改历史吗？

**A**: 是的，`SummarizationMiddleware` 会永久替换旧消息。如果需要临时压缩，使用 `@before_model` 修剪。

### Q4: 如何查看当前状态？

**A**: 使用 `get_state()`

```python
state = agent.get_state({"configurable": {"thread_id": "session-1"}})
print(state.values)
```

## 📖 学习路径

```
第1步：基础入门（1小时）
  └─ 运行 01_basic_memory.py 和 02_multi_thread.py
  └─ 理解 thread_id 和 checkpointer

第2步：消息管理（2小时）
  └─ 运行 03_trim_messages.py 和 04_summarization.py
  └─ 对比两种策略的差异

第3步：高级功能（2小时）
  └─ 运行 05_custom_state.py 和 06_tool_state_access.py
  └─ 理解状态扩展和工具集成

第4步：实战项目（1周+）
  └─ 构建多轮对话客服系统
  └─ 实现个性化推荐 Agent
```

## 🔗 相关资源

- **官方文档**: https://docs.langchain.com/oss/python/langchain/short-term-memory
- **LangGraph 文档**: https://langchain-ai.github.io/langgraph/
- **GitHub 仓库**: https://github.com/langchain-ai/langchain
- **智谱 AI**: https://open.bigmodel.cn/

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

---

**Happy Coding! 🚀**

如有问题，请查阅官方文档或提交 Issue。
