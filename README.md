# GLM-4.6 + LangChain 集成示例

本项目展示了如何使用 LangChain 框架集成智谱AI的GLM-4.6大语言模型，提供了从基础调用到高级应用的完整示例。

## 📋 目录

- [项目简介](#项目简介)
- [环境准备](#环境准备)
- [快速开始](#快速开始)
- [示例说明](#示例说明)
- [API参考](#api参考)
- [常见问题](#常见问题)
- [相关资源](#相关资源)

## 🚀 项目简介

本项目基于官方文档创建，展示了GLM-4.6模型与LangChain框架的深度集成。包含以下核心功能：

- **基础调用**：简单的文本生成
- **消息管理**：系统提示、用户消息、AI回复的处理
- **提示词模板**：动态构建复杂的提示词
- **链式调用**：组合多个组件完成复杂任务
- **多轮对话**：维护对话历史的上下文理解
- **流式输出**：实时生成响应内容
- **代码生成**：智能编程辅助

## 🛠️ 环境准备

### 系统要求

- Python 3.8+
- 智谱AI API密钥

### 安装依赖

```bash
pip install langchain langchain-community python-dotenv
```

### 环境变量配置

1. 在项目根目录创建 `.env` 文件
2. 添加你的智谱AI API密钥：

```env
ZHIPUAI_API_KEY=your-zhipu-api-key-here
```

> 💡 **获取API密钥**：访问 [智谱AI开放平台](https://open.bigmodel.cn/) 注册并获取API密钥

## 🏃‍♂️ 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd langchain-glm-examples
```

### 2. 配置API密钥

编辑 `.env` 文件，将 `your-zhipu-api-key-here` 替换为你的实际API密钥。

### 3. 运行示例

```bash
python glm_official_example.py
```

程序将依次运行所有示例，展示GLM-4.6的不同功能特性。

## 📚 示例说明

### 1. 基础示例 (`basic_example`)

最简单的GLM-4调用方式：

```python
from langchain_community.chat_models import ChatZhipuAI

chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.5,
    api_key=api_key
)

response = chat.invoke("你好，请简单介绍一下GLM-4模型")
```

**特点**：
- 直接文本输入输出
- 适合简单的问答场景
- 可调节temperature控制输出随机性

### 2. 消息示例 (`message_example`)

使用结构化消息进行交互：

```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="你是一个专业的Python编程助手"),
    HumanMessage(content="请解释一下Python中的装饰器")
]
response = chat.invoke(messages)
```

**特点**：
- 支持系统提示设定角色
- 结构化消息传递
- 更好的上下文控制

### 3. 提示词模板示例 (`prompt_template_example`)

动态构建复杂提示词：

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，擅长{expertise}。"),
    ("human", "请{task}：{topic}")
])

formatted_prompt = prompt.format_messages(
    role="科技博主",
    expertise="用简单易懂的语言解释复杂概念",
    task="解释什么是人工智能",
    topic="机器学习的基本原理"
)
```

**特点**：
- 参数化提示词模板
- 动态内容填充
- 高度可复用

### 4. 链式调用示例 (`chain_example`)

使用LCEL（LangChain Expression Language）构建处理链：

```python
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "请为主题'{topic}'写一个{style}的{length}。要求：{requirements}"
)

chain = prompt | chat | StrOutputParser()
result = chain.invoke({
    "topic": "人工智能与人类的关系",
    "style": "富有想象力",
    "length": "短诗",
    "requirements": "语言优美，意境深远"
})
```

**特点**：
- 声明式编程风格
- 组件化设计
- 易于扩展和维护

### 5. 多轮对话示例 (`conversation_example`)

维护对话历史的上下文理解：

```python
messages = [
    SystemMessage(content="你是一个友善的AI助手，能够记住对话内容。"),
    HumanMessage(content="你好，我叫小明，我是一名大学生。"),
    AIMessage(content="你好小明！很高兴认识你。你正在学习什么专业呢？"),
    HumanMessage(content="我在学习计算机科学。"),
    AIMessage(content="计算机科学是一个很棒的专业！你对哪个方向最感兴趣呢？"),
    HumanMessage(content="我对人工智能最感兴趣，特别是自然语言处理。")
]
response = chat.invoke(messages)
```

**特点**：
- 完整对话历史管理
- 上下文理解能力
- 连贯的多轮交互

### 6. 流式输出示例 (`streaming_example`)

实时生成响应内容：

```python
chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.7,
    api_key=api_key,
    streaming=True
)
```

**特点**：
- 实时响应用户
- 更好的交互体验
- 适合长文本生成

### 7. 代码生成示例 (`code_generation_example`)

智能编程辅助功能：

```python
code_prompt = """请用Python写一个函数，实现以下功能：
1. 计算斐波那契数列的第n项
2. 包含错误处理
3. 添加注释说明
4. 提供使用示例

要求代码清晰易懂，适合初学者理解。"""
response = chat.invoke(code_prompt)
```

**特点**：
- 智能代码生成
- 包含错误处理
- 详细注释说明
- 适合编程学习

## 📖 API参考

### ChatZhipuAI 参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| model | str | 否 | "glm-4" | 模型名称 |
| temperature | float | 否 | 0.7 | 控制输出随机性 (0-1) |
| api_key | str | 是 | - | 智谱AI API密钥 |
| streaming | bool | 否 | False | 是否启用流式输出 |

### 支持的模型

- `glm-4`：GLM-4.6 通用大模型
- `glm-4-flash`：GLM-4.6 快速版本
- `glm-3-turbo`：GLM-3 Turbo版本

### 消息类型

- `SystemMessage`：系统提示消息
- `HumanMessage`：用户消息
- `AIMessage`：AI回复消息

## ❓ 常见问题

### Q: 如何获取智谱AI API密钥？

A: 访问 [智谱AI开放平台](https://open.bigmodel.cn/)，注册账号后在控制台创建API密钥。

### Q: 调用失败怎么办？

A: 检查以下几点：
1. API密钥是否正确设置
2. 网络连接是否正常
3. API余额是否充足
4. 模型名称是否正确

### Q: 如何控制输出长度？

A: 可以通过以下方式：
1. 在提示词中明确要求字数限制
2. 调整temperature参数（值越低输出越简洁）
3. 使用output_parser进行后处理

### Q: 流式输出如何实现？

A: 设置 `streaming=True` 参数，然后使用适当的流式处理方法。具体实现请参考官方文档。

### Q: 如何处理长文本？

A: GLM-4.6支持较长的上下文，但仍建议：
1. 合理分段处理长文本
2. 保持提示词简洁明了
3. 必要时使用摘要或分块处理

## 🔗 相关资源

- [LangChain官方文档](https://python.langchain.com/)
- [智谱AI开放平台](https://open.bigmodel.cn/)
- [LangChain中文文档](https://python.langchain.ac.cn/)
- [GLM-4.6模型介绍](https://open.bigmodel.cn/model/glm4)

## 📄 许可证

本项目仅用于学习和演示目的，请遵循相关平台的API使用条款。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

---

**注意**：使用本示例前请确保已正确配置API密钥，并遵守智谱AI的使用条款。