# LangChain Messages 示例集

这是一个全面的 LangChain Messages 使用示例集合,涵盖了从基础到高级的各种使用场景。

## 📚 目录

1. [基础消息类型](#01-基础消息类型)
2. [多模态内容](#02-多模态内容)
3. [消息元数据](#03-消息元数据)
4. [消息操作](#04-消息操作)
5. [消息历史管理](#05-消息历史管理)
6. [最佳实践](#06-最佳实践)

## 📖 示例文件说明

### 01. 基础消息类型
**文件**: `01_basic_messages.py`

涵盖内容:
- ✅ HumanMessage 基础用法
- ✅ AIMessage 基础用法
- ✅ SystemMessage 系统提示
- ✅ 多轮对话实现
- ✅ 消息属性访问
- ✅ 消息拷贝和修改
- ✅ 消息序列化
- ✅ 消息列表操作
- ✅ 条件消息构建
- ✅ 消息内容格式化

**适用场景**:
- 刚开始学习 LangChain Messages
- 需要了解基本消息类型
- 构建简单的对话系统

**运行示例**:
```bash
python 01_basic_messages.py
```

---

### 02. 多模态内容
**文件**: `02_multimodal_content.py`

涵盖内容:
- ✅ 图像 URL 输入
- ✅ Base64 编码图像
- ✅ 多图像输入
- ✅ 图像详细级别控制
- ✅ 文本与图像交错
- ✅ 视频内容 (标准格式)
- ✅ 音频内容 (标准格式)
- ✅ 文档内容 (标准格式)
- ✅ 混合多模态内容
- ✅ Content Blocks 最佳实践
- ✅ 实用工具函数
- ✅ 错误处理和验证

**适用场景**:
- 处理图像、视频、音频等多模态数据
- 需要使用 GLM-4V 等视觉模型
- 构建多模态 AI 应用

**注意事项**:
- GLM-4V 支持图像理解
- 需要确保模型支持对应的模态
- 注意 token 消耗和成本

**运行示例**:
```bash
python 02_multimodal_content.py
```

---

### 03. 消息元数据
**文件**: `03_message_metadata.py`

涵盖内容:
- ✅ Tool Calls 基础
- ✅ Tool Call 完整流程
- ✅ Usage Metadata (token 使用统计)
- ✅ Response Metadata (响应元数据)
- ✅ Additional Kwargs (自定义字段)
- ✅ Message ID 使用
- ✅ Message Name 使用
- ✅ ToolMessage 详解
- ✅ 自定义元数据
- ✅ 元数据过滤和查询
- ✅ 元数据继承和传播
- ✅ 元数据最佳实践

**适用场景**:
- 需要跟踪工具调用
- 监控 token 使用情况
- 实现复杂的消息管理
- 多用户/会话场景

**运行示例**:
```bash
python 03_message_metadata.py
```

---

### 04. 消息操作
**文件**: `04_message_operations.py`

涵盖内容:
- ✅ add_messages 基础
- ✅ add_messages 更新现有消息
- ✅ RemoveMessage 删除消息
- ✅ 批量删除消息
- ✅ trim_messages 基础
- ✅ trim_messages 保留系统消息
- ✅ 按 Token 数量修剪
- ✅ 消息窗口滑动
- ✅ 消息摘要和压缩
- ✅ 消息去重
- ✅ 消息过滤
- ✅ 消息操作最佳实践

**适用场景**:
- 管理长对话历史
- 控制上下文长度
- 实现消息增删改查
- 优化 token 使用

**关键函数**:
- `add_messages()`: 合并和更新消息
- `trim_messages()`: 修剪消息历史
- `RemoveMessage`: 删除特定消息

**运行示例**:
```bash
python 04_message_operations.py
```

---

### 05. 消息历史管理
**文件**: `05_message_history.py`

涵盖内容:
- ✅ 基础消息历史
- ✅ 使用历史进行对话
- ✅ 会话管理
- ✅ 历史限制和修剪
- ✅ 历史持久化 (JSON)
- ✅ 历史摘要
- ✅ 历史搜索
- ✅ 历史分支
- ✅ 历史统计
- ✅ 历史最佳实践

**适用场景**:
- 实现持久化对话
- 多用户会话管理
- 对话历史分析
- 长期记忆系统

**存储方案**:
- `InMemoryChatMessageHistory`: 内存存储
- JSON 文件持久化
- 可扩展到 Redis/数据库

**运行示例**:
```bash
python 05_message_history.py
```

---

### 06. 最佳实践
**文件**: `06_best_practices.py`

涵盖内容:
- ✅ 清晰的系统提示
- ✅ 结构化消息内容
- ✅ 合理使用消息类型
- ✅ 上下文管理
- ✅ 错误处理
- ✅ 性能优化
- ✅ 安全和隐私
- ✅ 可维护性
- ✅ 测试友好
- ✅ 综合最佳实践清单

**适用场景**:
- 构建生产级应用
- 代码审查参考
- 团队开发规范
- 性能优化指南

**重点内容**:
- 系统提示设计原则
- 上下文长度控制
- 输入验证和清理
- 工厂函数和模板

**运行示例**:
```bash
python 06_best_practices.py
```

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- LangChain 相关包
- 智谱 AI API Key

### 安装依赖

```bash
pip install langchain langchain-community langgraph
```

### 设置 API Key

```bash
export ZHIPUAI_API_KEY="your-api-key-here"
```

或在代码中设置:
```python
os.environ["ZHIPUAI_API_KEY"] = "your-api-key-here"
```

### 运行示例

```bash
# 运行基础示例
python 01_basic_messages.py

# 运行特定功能
python 03_message_metadata.py

# 运行所有示例
for file in 0*.py; do python "$file"; done
```

---

## 📋 核心概念

### 消息类型

| 类型 | 用途 | 示例 |
|------|------|------|
| `SystemMessage` | 设置 AI 角色和行为 | "你是一个 Python 专家" |
| `HumanMessage` | 用户输入 | "如何读取文件?" |
| `AIMessage` | AI 回复 | "使用 open() 函数..." |
| `ToolMessage` | 工具执行结果 | "天气:晴朗,22°C" |
| `RemoveMessage` | 删除标记 | 用于删除特定消息 |

### 消息属性

```python
message = HumanMessage(
    content="消息内容",           # 必需:消息文本
    name="用户名",                # 可选:发送者名称
    id="msg-001",                # 可选:唯一标识符
    additional_kwargs={           # 可选:自定义元数据
        "user_id": "123",
        "priority": "high"
    }
)
```

### Content Blocks (多模态)

```python
message = HumanMessage(
    content=[
        {"type": "text", "text": "描述这张图"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/image.jpg",
                "detail": "high"
            }
        }
    ]
)
```

---

## 🎯 使用场景

### 场景 1: 简单问答
```python
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage

model = ChatZhipuAI(model="glm-4.6")
response = model.invoke([HumanMessage(content="你好")])
print(response.content)
```

### 场景 2: 带上下文的对话
```python
from langchain_core.chat_history import InMemoryChatMessageHistory

history = InMemoryChatMessageHistory()
history.add_user_message("我叫张三")
history.add_ai_message("你好,张三!")
history.add_user_message("我叫什么名字?")

response = model.invoke(history.messages)
print(response.content)  # 应该记得名字
```

### 场景 3: 工具调用
```python
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

@tool
def get_weather(city: str) -> str:
    """获取天气"""
    return f"{city}:晴朗,22°C"

model_with_tools = model.bind_tools([get_weather])
ai_response = model_with_tools.invoke([
    HumanMessage(content="北京天气如何?")
])

# 执行工具
if ai_response.tool_calls:
    tool_call = ai_response.tool_calls[0]
    result = get_weather.invoke(tool_call['args'])

    # 返回结果
    tool_msg = ToolMessage(
        content=result,
        tool_call_id=tool_call['id'],
        name='get_weather'
    )
```

### 场景 4: 图像理解
```python
from langchain_community.chat_models import ChatZhipuAI

model = ChatZhipuAI(model="glm-4v")  # 使用视觉模型

response = model.invoke([
    HumanMessage(content=[
        {"type": "text", "text": "这是什么?"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/cat.jpg"}
        }
    ])
])
```

---

## 🔧 常用操作

### 限制上下文长度
```python
from langchain_core.messages import trim_messages

trimmed = trim_messages(
    messages,
    max_tokens=10,      # 最多保留 10 条
    strategy="last",    # 保留最后的
    token_counter=len,  # 计数器
    include_system=True # 保留系统消息
)
```

### 删除特定消息
```python
from langgraph.graph.message import add_messages
from langchain_core.messages import RemoveMessage

# 删除 ID 为 'msg-1' 的消息
result = add_messages(
    messages,
    [RemoveMessage(id='msg-1')]
)
```

### 更新消息
```python
# 使用相同 ID 更新消息
updated = add_messages(
    existing_messages,
    [HumanMessage(content="新内容", id="msg-1")]
)
```

### 消息序列化
```python
# 转为字典
msg_dict = message.model_dump()

# 从字典重建
reconstructed = HumanMessage(**msg_dict)

# JSON 序列化
import json
json_str = json.dumps(msg_dict, ensure_ascii=False)
```

---

## ⚡ 性能优化

### 1. Token 优化
- 简化系统提示
- 使用 `trim_messages()` 限制历史
- 删除不必要的消息

### 2. 上下文管理
- 维护滑动窗口 (最近 N 条)
- 定期生成摘要压缩历史
- 只保留关键消息

### 3. 批量处理
```python
import asyncio

async def batch_invoke(messages_list):
    tasks = [model.ainvoke(msgs) for msgs in messages_list]
    return await asyncio.gather(*tasks)
```

### 4. 缓存
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_response(question: str):
    return model.invoke([HumanMessage(content=question)])
```

---

## 🔒 安全建议

1. **输入验证**
   - 清理用户输入
   - 防止注入攻击
   - 验证消息格式

2. **隐私保护**
   - 不记录敏感信息
   - 加密存储历史
   - 遵守 GDPR/CCPA

3. **访问控制**
   - 会话隔离
   - 用户认证
   - 权限检查

4. **错误处理**
   - Try-except 包装
   - 友好错误信息
   - 日志记录

---

## 🐛 常见问题

### Q1: 消息历史太长导致 token 超限?
**A**: 使用 `trim_messages()` 限制长度,或定期生成摘要。

```python
trimmed = trim_messages(messages, max_tokens=4000, strategy="last")
```

### Q2: 如何实现多用户会话隔离?
**A**: 为每个用户/会话分配唯一 ID,分别存储历史。

```python
sessions = {}
sessions[user_id] = InMemoryChatMessageHistory()
```

### Q3: Tool Call 的 tool_call_id 从哪来?
**A**: 由模型自动生成,包含在 `ai_message.tool_calls[0]['id']` 中。

```python
tool_call_id = ai_response.tool_calls[0]['id']
tool_msg = ToolMessage(content=result, tool_call_id=tool_call_id)
```

### Q4: 如何处理多模态内容?
**A**: 使用 Content Blocks 格式,确保模型支持对应模态。

```python
content = [
    {"type": "text", "text": "..."},
    {"type": "image_url", "image_url": {"url": "..."}}
]
```

### Q5: 消息 ID 是必需的吗?
**A**: 不必需,但在需要更新/删除特定消息时必须提供。

---

## 📚 参考资料

- [LangChain Messages 官方文档](https://docs.langchain.com/oss/python/langchain/messages)
- [LangGraph Human-in-the-Loop](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/)
- [智谱 AI 文档](https://open.bigmodel.cn/dev/api)
- [Pydantic V2 迁移指南](https://docs.pydantic.dev/latest/migration/)

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request!

---

## 📄 许可

MIT License

---

## 📧 联系方式

如有问题,请提交 Issue。

---

## 🎓 学习路径

### 初学者
1. 从 `01_basic_messages.py` 开始
2. 理解基本消息类型
3. 学习简单的对话流程

### 中级
1. 学习 `03_message_metadata.py` 理解元数据
2. 掌握 `04_message_operations.py` 的操作技巧
3. 实现 `05_message_history.py` 的历史管理

### 高级
1. 深入 `02_multimodal_content.py` 多模态
2. 实践 `06_best_practices.py` 的所有建议
3. 构建生产级应用

---

## ✨ 示例特点

- ✅ **完整性**: 涵盖所有常见场景
- ✅ **实用性**: 可直接运行的代码
- ✅ **清晰性**: 详细的中文注释
- ✅ **渐进性**: 从简单到复杂
- ✅ **规范性**: 遵循最佳实践

---

## 🎯 下一步

1. **运行示例**: 从简单的开始
2. **修改实验**: 改变参数观察效果
3. **构建应用**: 结合自己的需求
4. **分享经验**: 帮助其他开发者

---

**祝学习愉快! 🚀**
