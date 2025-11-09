# LangChain Messages 详细指南

> 基于官方文档 https://docs.langchain.com/oss/python/langchain/messages 的完整中文总结

---

## 📋 目录

- [核心概念](#核心概念)
- [消息类型](#消息类型)
- [Message Content (消息内容)](#message-content-消息内容)
- [多模态内容](#多模态内容)
- [Content Blocks (内容块)](#content-blocks-内容块)
- [消息属性和元数据](#消息属性和元数据)
- [消息操作](#消息操作)
- [消息在 Agent 中的应用](#消息在-agent-中的应用)
- [消息历史管理](#消息历史管理)
- [最佳实践](#最佳实践)

---

## 核心概念

### 什么是 Messages？

**Messages（消息）** 是 LangChain 中用于表示对话和交互的核心数据结构。消息封装了对话中的不同角色（用户、AI、系统等）的内容和元数据。

### 为什么使用 Messages？

现代 LLM 提供商都使用聊天模型接口，接受消息列表作为输入。LangChain 的 `ChatModel` 接受 `Message` 对象列表作为输入，这些消息有多种形式：

- **HumanMessage**: 用户输入
- **AIMessage**: LLM 响应  
- **SystemMessage**: 系统指令
- **ToolMessage**: 工具执行结果
- **FunctionMessage**: 函数调用结果（已废弃，推荐使用 ToolMessage）

### Messages 的核心特征

1. **角色区分**: 通过不同的消息类型标识发送者
2. **内容封装**: 支持文本、图像、音频等多模态内容
3. **元数据支持**: 包含 token 使用、工具调用等信息
4. **序列化**: 可以轻松转换为 JSON 等格式

---

## 消息类型

### HumanMessage (用户消息)

表示用户输入和交互，可以包含文本、图像、音频、文件等多模态内容。

```python
from langchain_core.messages import HumanMessage

# 简单文本消息
human_msg = HumanMessage(content="你好，我是 Bob")

# 带 ID 的消息
human_msg = HumanMessage(
    content="帮我分析这个数据",
    id="msg_123"
)

# 带元数据的消息
human_msg = HumanMessage(
    content="这是我的问题",
    metadata={"user_id": "user_456", "session": "abc"}
)
```

**使用场景**:
- 用户输入的问题
- 用户提供的指令
- 用户上传的文件或图片

### AIMessage (AI 响应消息)

表示模型调用的输出，可以包含多模态数据、工具调用和提供商特定的元数据。

```python
from langchain_core.messages import AIMessage

# 简单 AI 响应
ai_msg = AIMessage(content="你好 Bob！我能帮你什么吗？")

# 包含工具调用的响应
ai_msg = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_weather",
            "args": {"location": "北京"},
            "id": "call_123"
        }
    ]
)

# 手动创建 AI 消息（用于对话历史）
ai_msg = AIMessage(content="我很乐意帮助你！")
```

**AIMessage 的特殊属性**:
- `tool_calls`: 工具调用列表
- `usage_metadata`: Token 使用信息
- `response_metadata`: 提供商特定的元数据

**使用场景**:
- 模型生成的回答
- 工具调用请求
- 手动插入对话历史

### SystemMessage (系统消息)

表示初始指令，用于设定模型的行为、角色和响应准则。

```python
from langchain_core.messages import SystemMessage

# 设定助手角色
system_msg = SystemMessage(content="你是一个专业的编程助手")

# 详细的系统提示词
system_msg = SystemMessage(content="""
你是一个客户服务助手。

职责:
- 礼貌、专业地回答问题
- 如果不确定，诚实说明
- 使用简洁的语言

限制:
- 不提供医疗建议
- 不分享个人信息
""")
```

**使用场景**:
- 设定 AI 的角色和行为
- 定义回答的风格和规则
- 提供上下文和背景信息

### ToolMessage (工具消息)

表示工具执行的结果，用于将工具输出返回给模型。

```python
from langchain_core.messages import ToolMessage

# 工具执行结果
tool_msg = ToolMessage(
    content="北京的天气是晴朗，温度 22°C",
    tool_call_id="call_123",
    name="get_weather"
)

# 包含错误的工具消息
tool_msg = ToolMessage(
    content="Error: API 调用失败",
    tool_call_id="call_456",
    name="search_database",
    status="error"
)
```

**使用场景**:
- 返回工具执行结果
- 报告工具执行错误
- 提供额外的工具元数据

### RemoveMessage (删除消息)

用于从对话历史中删除特定消息，常用于内存管理。

```python
from langchain_core.messages import RemoveMessage

# 删除特定消息
remove_msg = RemoveMessage(id="msg_123")

# 删除所有消息（配合 REMOVE_ALL_MESSAGES）
from langgraph.graph.message import REMOVE_ALL_MESSAGES
remove_all = RemoveMessage(id=REMOVE_ALL_MESSAGES)
```

**使用场景**:
- 清理对话历史
- 删除敏感信息
- 管理上下文窗口

---

## Message Content (消息内容)

### 内容类型

消息的 `content` 属性是松散类型的，支持多种格式：

1. **字符串**: 简单的文本内容
2. **提供商原生格式**: 如 OpenAI 格式的内容块列表
3. **LangChain 标准内容块**: 跨提供商的统一格式

```python
from langchain_core.messages import HumanMessage

# 1. 字符串内容
msg1 = HumanMessage(content="你好，世界！")

# 2. 提供商原生格式 (OpenAI)
msg2 = HumanMessage(content=[
    {"type": "text", "text": "描述这张图片"},
    {
        "type": "image_url", 
        "image_url": {"url": "https://example.com/image.jpg"}
    }
])

# 3. LangChain 标准内容块
msg3 = HumanMessage(content_blocks=[
    {"type": "text", "text": "描述这张图片"},
    {"type": "image", "url": "https://example.com/image.jpg"}
])
```

### content vs content_blocks

- **content**: 松散类型，支持字符串和任意对象列表
- **content_blocks**: 类型安全的接口，使用 LangChain 标准内容块

```python
from langchain_core.messages import HumanMessage

# 使用 content_blocks（推荐）
msg = HumanMessage(content_blocks=[
    {"type": "text", "text": "这是文本"},
    {"type": "image", "url": "https://example.com/image.jpg"}
])

# content_blocks 会自动填充 content
print(msg.content)  # 自动包含内容块数据
```

---

## 多模态内容

### 图像内容

LangChain 支持三种方式传递图像：

#### 1. 使用 URL

```python
from langchain_core.messages import HumanMessage

msg = HumanMessage(content=[
    {"type": "text", "text": "这张图片里有什么?"},
    {
        "type": "image_url",
        "image_url": {"url": "https://example.com/image.jpg"}
    }
])
```

#### 2. 使用 Base64 编码

```python
from langchain_core.messages import HumanMessage
import base64

# 读取并编码图像
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

msg = HumanMessage(content=[
    {"type": "text", "text": "分析这张图片"},
    {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
    }
])
```

#### 3. 使用提供商文件 ID

```python
from langchain_core.messages import HumanMessage

msg = HumanMessage(content=[
    {"type": "text", "text": "描述图片内容"},
    {"type": "image", "source_type": "id", "id": "file-abc123"}
])
```

### LangChain 标准多模态格式

```python
from langchain_core.messages import HumanMessage

# 图像
msg = HumanMessage(content_blocks=[
    {"type": "text", "text": "分析这张图片"},
    {
        "type": "image",
        "source_type": "url",
        "url": "https://example.com/image.jpg",
        "mime_type": "image/jpeg"
    }
])

# Base64 图像
msg = HumanMessage(content_blocks=[
    {"type": "text", "text": "这是什么?"},
    {
        "type": "image",
        "source_type": "base64",
        "data": "base64_encoded_data_here...",
        "mime_type": "image/png"
    }
])
```

### 视频内容

```python
from langchain_core.messages import HumanMessage

# 视频 URL
msg = HumanMessage(content_blocks=[
    {"type": "text", "text": "描述这个视频"},
    {
        "type": "video",
        "source_type": "url",
        "url": "https://example.com/video.mp4"
    }
])

# Base64 视频
msg = HumanMessage(content_blocks=[
    {"type": "text", "text": "分析视频内容"},
    {
        "type": "video",
        "source_type": "base64",
        "data": "base64_video_data...",
        "mime_type": "video/mp4"
    }
])
```

### 音频内容

```python
from langchain_core.messages import HumanMessage

msg = HumanMessage(content_blocks=[
    {"type": "text", "text": "转录这段音频"},
    {
        "type": "audio",
        "source_type": "base64",
        "data": "base64_audio_data...",
        "mime_type": "audio/wav"
    }
])
```

### PDF 和文档

```python
from langchain_core.messages import HumanMessage

# PDF 文件
msg = HumanMessage(content_blocks=[
    {"type": "text", "text": "总结这个 PDF"},
    {
        "type": "document",
        "source_type": "base64",
        "data": "base64_pdf_data...",
        "mime_type": "application/pdf",
        "extras": {"filename": "document.pdf"}  # 某些提供商需要
    }
])
```

### 多个多模态内容

```python
from langchain_core.messages import HumanMessage

msg = HumanMessage(content=[
    {"type": "text", "text": "比较这两张图片的差异"},
    {
        "type": "image_url",
        "image_url": {"url": "https://example.com/image1.jpg"}
    },
    {
        "type": "image_url",
        "image_url": {"url": "https://example.com/image2.jpg"}
    }
])
```

---

## Content Blocks (内容块)

### 标准内容块类型

LangChain 定义了标准的内容块类型，可跨提供商使用：

#### 1. Text Block (文本块)

```python
text_block = {
    "type": "text",
    "text": "这是文本内容"
}
```

#### 2. Image Block (图像块)

```python
# 从 URL
image_block = {
    "type": "image",
    "source_type": "url",
    "url": "https://example.com/image.jpg",
    "mime_type": "image/jpeg"
}

# 从 Base64
image_block = {
    "type": "image",
    "source_type": "base64",
    "data": "base64_encoded_data...",
    "mime_type": "image/png"
}

# 从文件 ID
image_block = {
    "type": "image",
    "source_type": "id",
    "id": "file-abc123"
}
```

#### 3. Video Block (视频块)

```python
video_block = {
    "type": "video",
    "source_type": "url",
    "url": "https://example.com/video.mp4",
    "mime_type": "video/mp4"
}
```

#### 4. Audio Block (音频块)

```python
audio_block = {
    "type": "audio",
    "source_type": "base64",
    "data": "base64_audio_data...",
    "mime_type": "audio/wav"
}
```

#### 5. Document Block (文档块)

```python
document_block = {
    "type": "document",
    "source_type": "base64",
    "data": "base64_pdf_data...",
    "mime_type": "application/pdf",
    "extras": {"filename": "report.pdf"}
}
```

### 内容块的使用

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# 组合多种内容块
msg = HumanMessage(content_blocks=[
    {"type": "text", "text": "分析以下多媒体内容："},
    {
        "type": "image",
        "source_type": "url",
        "url": "https://example.com/chart.png"
    },
    {"type": "text", "text": "并提供详细报告"}
])

response = model.invoke([msg])
print(response.content)
```

---

## 消息属性和元数据

### 核心属性

每个消息都有以下核心属性：

```python
from langchain_core.messages import HumanMessage, AIMessage

msg = HumanMessage(
    content="你好",
    id="msg_123",           # 消息 ID
    name="用户名",           # 发送者名称
    metadata={              # 自定义元数据
        "user_id": "user_456",
        "timestamp": "2025-01-09",
        "source": "web"
    }
)

# AIMessage 的特殊属性
ai_msg = AIMessage(
    content="你好！",
    id="msg_124",
    response_metadata={     # 响应元数据
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5
        }
    },
    usage_metadata={        # Token 使用信息
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15
    }
)
```

### 访问消息属性

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
response = model.invoke([HumanMessage(content="你好")])

# 基本属性
print(f"消息 ID: {response.id}")
print(f"内容: {response.content}")
print(f"类型: {type(response).__name__}")

# 元数据
print(f"模型: {response.response_metadata.get('model')}")
print(f"Stop reason: {response.response_metadata.get('stop_reason')}")

# Token 使用
if hasattr(response, 'usage_metadata'):
    usage = response.usage_metadata
    print(f"输入 tokens: {usage.get('input_tokens')}")
    print(f"输出 tokens: {usage.get('output_tokens')}")
    print(f"总 tokens: {usage.get('total_tokens')}")
```

### Tool Calls 属性

AIMessage 可以包含工具调用信息：

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

@tool
def get_weather(location: str) -> str:
    """获取天气信息"""
    return f"{location}: 晴朗"

model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke([
    HumanMessage(content="北京天气如何?")
])

# 检查工具调用
if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"工具名称: {tool_call['name']}")
        print(f"工具参数: {tool_call['args']}")
        print(f"调用 ID: {tool_call['id']}")
```

---

## 消息操作

### 创建消息

```python
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage
)

# 多种方式创建消息
messages = [
    SystemMessage(content="你是一个助手"),
    HumanMessage(content="你好"),
    AIMessage(content="你好！有什么可以帮你的吗？"),
    HumanMessage(content="今天天气如何?")
]
```

### 添加消息

```python
from langgraph.graph import add_messages
from langchain_core.messages import HumanMessage, AIMessage

# 初始消息
messages = [
    HumanMessage(content="你好")
]

# 添加新消息
new_messages = [
    AIMessage(content="你好！"),
    HumanMessage(content="你叫什么名字?")
]

# 使用 add_messages reducer
messages = add_messages(messages, new_messages)
```

### 更新消息

```python
from langchain_core.messages import AIMessage

# 创建消息
msg = AIMessage(content="初始内容", id="msg_1")

# 更新消息（通过创建新消息）
updated_msg = AIMessage(
    content="更新后的内容",
    id="msg_1"  # 相同 ID 会替换原消息
)
```

### 删除消息

```python
from langchain_core.messages import RemoveMessage, HumanMessage, AIMessage

# 创建消息历史
messages = [
    HumanMessage(content="问题1", id="msg_1"),
    AIMessage(content="答案1", id="msg_2"),
    HumanMessage(content="问题2", id="msg_3"),
    AIMessage(content="答案2", id="msg_4")
]

# 删除特定消息
remove_msg = RemoveMessage(id="msg_1")
messages = add_messages(messages, [remove_msg])

# 删除所有消息
from langgraph.graph.message import REMOVE_ALL_MESSAGES
remove_all = RemoveMessage(id=REMOVE_ALL_MESSAGES)
messages = add_messages(messages, [remove_all])
```

### 消息格式转换

```python
from langchain_core.messages import HumanMessage

# 从字典创建
msg_dict = {
    "role": "user",
    "content": "你好"
}
msg = HumanMessage(**msg_dict)

# 转换为字典
msg_dict = {
    "role": "user",
    "content": msg.content
}

# OpenAI 格式
openai_format = {
    "role": "user",
    "content": msg.content
}
```

---

## 消息在 Agent 中的应用

### 基本对话流程

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# 构建对话
conversation = [
    SystemMessage(content="你是一个有帮助的助手"),
    HumanMessage(content="你好"),
    AIMessage(content="你好！有什么可以帮你吗？"),
    HumanMessage(content="什么是量子计算?")
]

# 调用模型
response = model.invoke(conversation)
print(response.content)
```

### 在 Agent 中使用消息

```python
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

agent = create_agent(
    model=model,
    tools=[],
    system_prompt="你是一个专业的助手"
)

# 使用消息调用 Agent
result = agent.invoke({
    "messages": [HumanMessage(content="帮我计算 25 * 4")]
})

# 查看消息历史
for msg in result["messages"]:
    print(f"{type(msg).__name__}: {msg.content}")
```

### 工具调用消息流

```python
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

@tool
def multiply(a: int, b: int) -> int:
    """将两个数相乘"""
    return a * b

model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
agent = create_agent(model=model, tools=[multiply])

result = agent.invoke({
    "messages": [HumanMessage(content="25 乘以 4 等于多少?")]
})

# 消息流:
# 1. HumanMessage: "25 乘以 4 等于多少?"
# 2. AIMessage: (with tool_calls for multiply)
# 3. ToolMessage: "100"
# 4. AIMessage: "25 乘以 4 等于 100"

for msg in result["messages"]:
    if isinstance(msg, HumanMessage):
        print(f"用户: {msg.content}")
    elif isinstance(msg, AIMessage):
        if msg.tool_calls:
            print(f"AI 工具调用: {msg.tool_calls}")
        else:
            print(f"AI: {msg.content}")
    elif isinstance(msg, ToolMessage):
        print(f"工具结果: {msg.content}")
```

---

## 消息历史管理

### 短期记忆 (Short-term Memory)

短期记忆在单个对话线程中维护消息历史。

```python
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver

model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# 使用内存保存器
memory = MemorySaver()

agent = create_agent(
    model=model,
    tools=[],
    checkpointer=memory
)

# 配置线程 ID
config = {"configurable": {"thread_id": "conversation_1"}}

# 第一轮对话
result1 = agent.invoke({
    "messages": [{"role": "user", "content": "我叫 Bob"}]
}, config)

# 第二轮对话 - Agent 会记住之前的对话
result2 = agent.invoke({
    "messages": [{"role": "user", "content": "我叫什么名字?"}]
}, config)

print(result2["messages"][-1].content)  # "你叫 Bob"
```

### 消息修剪 (Trimming)

当对话过长时，需要修剪消息以适应上下文窗口。

```python
from langchain_core.messages import trim_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

messages = [
    SystemMessage(content="你是一个助手"),
    HumanMessage(content="问题 1"),
    AIMessage(content="答案 1"),
    HumanMessage(content="问题 2"),
    AIMessage(content="答案 2"),
    HumanMessage(content="问题 3"),
    AIMessage(content="答案 3"),
]

# 保留最近的 4 条消息 + 系统消息
trimmed = trim_messages(
    messages,
    max_tokens=1000,
    strategy="last",
    token_counter=len,  # 简化的 token 计数
    include_system=True  # 始终保留系统消息
)
```

### 消息总结 (Summarization)

使用模型总结旧消息，压缩对话历史。

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# 原始消息历史
messages = [
    HumanMessage(content="我叫 Bob"),
    AIMessage(content="你好 Bob！"),
    HumanMessage(content="我喜欢 Python"),
    AIMessage(content="Python 是很棒的语言！"),
    # ... 更多消息
]

# 总结旧消息
summary_prompt = "总结以下对话内容："
summary_messages = messages[:-2] + [
    HumanMessage(content=summary_prompt)
]

summary = model.invoke(summary_messages)

# 使用总结替换旧消息
new_messages = [
    SystemMessage(content=f"对话总结: {summary.content}"),
    *messages[-2:]  # 保留最近的消息
]
```

### 删除消息

```python
from langchain_core.messages import RemoveMessage
from langgraph.graph import add_messages

# 删除特定消息
messages = [
    HumanMessage(content="消息1", id="msg_1"),
    AIMessage(content="消息2", id="msg_2"),
    HumanMessage(content="消息3", id="msg_3")
]

# 删除 msg_1
remove = RemoveMessage(id="msg_1")
messages = add_messages(messages, [remove])

# 清空所有消息
from langgraph.graph.message import REMOVE_ALL_MESSAGES
clear_all = RemoveMessage(id=REMOVE_ALL_MESSAGES)
messages = add_messages(messages, [clear_all])
```

---

## 最佳实践

### 1. 始终使用适当的消息类型

```python
# ✅ 好的做法
messages = [
    SystemMessage(content="你是助手"),
    HumanMessage(content="用户问题"),
    AIMessage(content="AI 回答")
]

# ❌ 避免
messages = [
    {"role": "system", "content": "..."},  # 应使用 SystemMessage
    {"role": "user", "content": "..."}      # 应使用 HumanMessage
]
```

### 2. 为消息添加 ID

```python
# ✅ 好的做法 - 便于追踪和删除
msg = HumanMessage(content="你好", id="msg_123")

# ❌ 避免 - 难以管理
msg = HumanMessage(content="你好")
```

### 3. 使用元数据存储上下文信息

```python
# ✅ 好的做法
msg = HumanMessage(
    content="帮我预订航班",
    metadata={
        "user_id": "user_123",
        "session_id": "session_456",
        "timestamp": "2025-01-09T10:00:00"
    }
)
```

### 4. 合理管理消息历史

```python
# ✅ 好的做法 - 定期清理或总结
from langchain_core.messages import trim_messages

# 限制消息数量
trimmed_messages = messages[-10:]  # 只保留最近 10 条

# 或使用 trim_messages
trimmed = trim_messages(
    messages,
    max_tokens=4000,
    strategy="last",
    include_system=True
)
```

### 5. 处理多模态内容时指定 MIME 类型

```python
# ✅ 好的做法
msg = HumanMessage(content_blocks=[
    {"type": "text", "text": "分析图片"},
    {
        "type": "image",
        "source_type": "base64",
        "data": image_data,
        "mime_type": "image/jpeg"  # 明确指定类型
    }
])
```

### 6. 使用 content_blocks 获得类型安全

```python
# ✅ 好的做法 - 类型安全
msg = HumanMessage(content_blocks=[
    {"type": "text", "text": "你好"},
    {"type": "image", "url": "https://example.com/img.jpg"}
])

# ⚠️ 可用但不够类型安全
msg = HumanMessage(content=[
    {"type": "text", "text": "你好"},
    {"type": "image_url", "image_url": {"url": "..."}}
])
```

### 7. 工具消息必须包含 tool_call_id

```python
# ✅ 好的做法
tool_msg = ToolMessage(
    content="结果",
    tool_call_id="call_123",  # 必需
    name="tool_name"
)

# ❌ 避免 - 缺少 tool_call_id
tool_msg = ToolMessage(content="结果")
```

### 8. 使用 AnyMessage 进行序列化

```python
from langchain_core.messages import AnyMessage
from typing import List
from pydantic import BaseModel

# ✅ 好的做法 - 用于序列化
class ChatState(BaseModel):
    messages: List[AnyMessage]  # 支持序列化/反序列化

# ❌ 避免 - 可能导致序列化问题
class ChatState(BaseModel):
    messages: List[BaseMessage]
```

---

## 🎯 消息类型快速参考

| 消息类型 | 用途 | 主要属性 |
|---------|------|---------|
| **HumanMessage** | 用户输入 | content, id, metadata |
| **AIMessage** | AI 响应 | content, tool_calls, usage_metadata |
| **SystemMessage** | 系统指令 | content |
| **ToolMessage** | 工具结果 | content, tool_call_id, name |
| **RemoveMessage** | 删除消息 | id |

## 📊 Content Block 类型

| Block 类型 | 用途 | 必需字段 |
|-----------|------|---------|
| **text** | 文本内容 | type, text |
| **image** | 图像 | type, source_type, url/data/id |
| **video** | 视频 | type, source_type, url/data |
| **audio** | 音频 | type, source_type, data |
| **document** | 文档 (PDF等) | type, source_type, data, mime_type |

---

## 🔗 相关资源

- **官方文档**: https://docs.langchain.com/oss/python/langchain/messages
- **配套文档**:
  - [LangChain Models 详细指南](./LangChain_Models_详细指南.md)
  - [LangChain Agents 详细总结](./LangChain_Agents_详细总结.md)
  - [LangChain Tools 详细指南](./LangChain_Tools_详细指南.md)

---

**文档版本**: 1.0  
**最后更新**: 2025年1月  
**基于**: LangChain v0.3+, Python 3.9+

本文档涵盖了 LangChain Messages 的核心概念、所有消息类型、多模态支持和最佳实践，包含 80+ 实用代码示例。
