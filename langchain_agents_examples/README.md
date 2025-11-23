# LangChain Agents 完整示例集 (GLM 模型版本)

本项目包含基于 LangChain Agents 官方文档的完整示例代码，使用智谱 AI 的 GLM 模型实现。

## 📋 目录

- [01_basic_agent.py](01_basic_agent.py) - 基础 Agent 示例
- [02_middleware_examples.py](02_middleware_examples.py) - 中间件示例
- [03_memory_management.py](03_memory_management.py) - 记忆管理示例
- [04_structured_output.py](04_structured_output.py) - 结构化输出示例
- [05_human_in_the_loop.py](05_human_in_the_loop.py) - 人机协作示例

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install langchain langchain-community zhipuai langgraph
```

### 2. 设置 API Key

在运行示例之前，需要设置智谱 AI 的 API Key：

```bash
export ZHIPUAI_API_KEY="your-api-key-here"
```

或者在代码中直接设置（不推荐用于生产环境）：

```python
os.environ["ZHIPUAI_API_KEY"] = "your-api-key-here"
```

### 3. 运行示例

```bash
# 基础示例
python 01_basic_agent.py

# 中间件示例
python 02_middleware_examples.py

# 记忆管理示例
python 03_memory_management.py

# 结构化输出示例
python 04_structured_output.py

# 人机协作示例
python 05_human_in_the_loop.py
```

## 📚 示例说明

### 01_basic_agent.py - 基础 Agent 示例

**包含内容:**
- 基础工具定义 (`@tool` 装饰器)
- 带参数验证的工具 (Pydantic)
- 创建简单 Agent
- 结构化系统提示词
- 多工具组合使用

**示例工具:**
- `get_weather` - 获取天气信息
- `calculate` - 数学计算
- `search_web` - 网络搜索

**运行示例:**
```bash
python 01_basic_agent.py
```

---

### 02_middleware_examples.py - 中间件示例

**包含内容:**
- 工具错误处理 (`@wrap_tool_call`)
- 动态模型选择 (`@wrap_model_call`)
- 动态提示词 (`@dynamic_prompt`)
- Before/After 钩子 (`@before_model`, `@after_model`)
- 工具执行日志
- 多个中间件组合使用

**核心中间件:**
- `handle_tool_errors` - 统一工具错误处理
- `dynamic_model_selection` - 根据对话复杂度选择模型
- `context_aware_prompt` - 基于上下文生成动态提示
- `log_tool_execution` - 记录工具执行过程

**运行示例:**
```bash
python 02_middleware_examples.py
```

---

### 03_memory_management.py - 记忆管理示例

**包含内容:**
- 基础短期记忆 (Checkpointer)
- 多会话管理 (thread_id)
- 自定义状态模式 (CustomAgentState)
- 消息修剪 (trim_messages)
- 状态访问和管理

**核心概念:**
- `MemorySaver` - 内存检查点器
- `thread_id` - 会话 ID
- `AgentState` - Agent 状态模式
- 消息历史修剪

**运行示例:**
```bash
python 03_memory_management.py
```

---

### 04_structured_output.py - 结构化输出示例

**包含内容:**
- Pydantic 模型作为 Schema
- Dataclass 作为 Schema
- Union 类型 - 多种可能输出
- 嵌套结构
- 产品评论分析
- 联系人信息提取
- 事件提取

**核心 Schema:**
- `WeatherResponse` - 天气信息
- `ProductReview` - 产品评论
- `ContactInfo` - 联系人信息
- `Event` - 事件信息
- `Union[EmailAction, SlackAction, TodoAction]` - 多种操作类型

**运行示例:**
```bash
python 04_structured_output.py
```

---

### 05_human_in_the_loop.py - 人机协作示例

**包含内容:**
- 基础人机协作流程
- 三种决策类型 (approve, edit, reject)
- 多工具调用审批
- 选择性审批
- 自定义审批逻辑

**敏感操作工具:**
- `delete_file` - 删除文件
- `send_email` - 发送邮件
- `transfer_money` - 转账

**决策类型:**
- ✅ `approve` - 批准操作
- ✏️ `edit` - 编辑后执行
- ❌ `reject` - 拒绝操作

**运行示例:**
```bash
python 05_human_in_the_loop.py
```

## 🔧 代码结构

### 通用模式

所有示例都遵循以下模式：

```python
# 1. 导入依赖
from langchain.agents import create_agent
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.tools import tool

# 2. 定义工具
@tool
def my_tool(param: str) -> str:
    """工具描述"""
    return "结果"

# 3. 创建 Agent
model = ChatZhipuAI(model="glm-4-plus", temperature=0.5)
agent = create_agent(
    model=model,
    tools=[my_tool],
    system_prompt="你是一个助手"
)

# 4. 调用 Agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "问题"}]
})
```

## 📖 核心概念

### 1. Agent = 模型 + 工具 + 循环

Agent 通过循环调用模型和工具来完成任务：

```
输入 → 模型推理 → 工具执行 → 观察结果 → 继续推理 → 最终输出
```

### 2. 工具定义

使用 `@tool` 装饰器定义工具：

```python
@tool
def search(query: str) -> str:
    """搜索信息 - 这个描述会被模型看到"""
    return f"搜索结果: {query}"
```

### 3. 中间件

在 Agent 执行的不同阶段插入自定义逻辑：

- `@before_model` - 模型调用前
- `@after_model` - 模型调用后
- `@wrap_model_call` - 包装模型调用
- `@wrap_tool_call` - 包装工具调用
- `@dynamic_prompt` - 动态提示词

### 4. 记忆管理

- **短期记忆**: 使用 `MemorySaver` + `thread_id`
- **自定义状态**: 扩展 `AgentState`
- **消息修剪**: 控制对话历史长度

### 5. 结构化输出

使用 `response_format` 强制返回结构化数据：

```python
class MySchema(BaseModel):
    field1: str
    field2: int

agent = create_agent(
    model=model,
    response_format=ToolStrategy(schema=MySchema)
)
```

### 6. 人机协作

为敏感操作添加人工审批：

```python
agent = create_agent(
    model=model,
    tools=[delete_file],
    middleware=[
        human_in_the_loop_middleware(
            interrupt_on={"delete_file": True}
        )
    ],
    checkpointer=MemorySaver()
)
```

## 🎯 使用场景

### 场景 1: 客服助手

```python
# 使用基础 Agent + 工具
tools = [search_kb, create_ticket, escalate]
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="你是客服专员"
)
```

### 场景 2: 数据分析助手

```python
# 使用结构化输出
agent = create_agent(
    model=model,
    tools=[query_database],
    response_format=ToolStrategy(schema=AnalysisResult)
)
```

### 场景 3: 审批工作流

```python
# 使用人机协作
agent = create_agent(
    model=model,
    tools=[approve_expense, reject_expense],
    middleware=[human_in_the_loop_middleware(...)],
    checkpointer=MemorySaver()
)
```

## 🔑 GLM 模型说明

### 可用模型

- `glm-4-plus` - 推荐，性能强大
- `glm-4-flash` - 快速响应，成本低
- `glm-4` - 标准版本

### 模型配置

```python
model = ChatZhipuAI(
    model="glm-4-plus",
    temperature=0.5,  # 控制随机性
    max_tokens=1000,  # 最大输出长度
)
```

## ⚠️ 注意事项

1. **API Key 安全**
   - 不要在代码中硬编码 API Key
   - 使用环境变量或配置文件
   - 不要提交 API Key 到版本控制

2. **错误处理**
   - 所有示例都包含基础错误处理
   - 生产环境需要更完善的错误处理

3. **成本控制**
   - 使用 `temperature` 控制输出随机性
   - 使用 `max_tokens` 限制输出长度
   - 考虑使用 `glm-4-flash` 降低成本

4. **性能优化**
   - 使用消息修剪减少 token 消耗
   - 异步工具提升并发性能
   - 缓存重复查询结果

## 🐛 常见问题

### Q1: 提示 "API Key 未设置"

**解决方案:**
```bash
export ZHIPUAI_API_KEY="your-api-key-here"
```

### Q2: 工具没有被调用

**原因:**
- 工具描述不够清晰
- 系统提示词没有提到工具

**解决方案:**
- 改进工具的文档字符串
- 在系统提示词中说明可用工具

### Q3: 结构化输出格式错误

**原因:**
- Schema 定义不够清晰
- 模型不支持结构化输出

**解决方案:**
- 使用更详细的 Field 描述
- 使用 `glm-4-plus` 模型

### Q4: 记忆功能不工作

**原因:**
- 没有设置 `checkpointer`
- `thread_id` 不一致

**解决方案:**
```python
checkpointer = MemorySaver()
agent = create_agent(..., checkpointer=checkpointer)
# 使用相同的 thread_id
agent.invoke(..., {"configurable": {"thread_id": "same-id"}})
```

## 📚 参考资源

- [LangChain 官方文档](https://docs.langchain.com/oss/python/langchain/agents)
- [智谱 AI 文档](https://open.bigmodel.cn/dev/api)
- [LangGraph 文档](https://docs.langchain.com/langgraph)
- [原始总结文档](../langchain-docs/LangChain_Agents_详细总结.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

---

**作者**: 基于 LangChain 官方文档改编
**版本**: 1.0
**更新日期**: 2025-01-23
