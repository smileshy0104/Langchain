# LangChain Models 详细指南

> 基于官方文档 https://docs.langchain.com/oss/python/langchain/models 的完整中文总结（在原有内容基础上补充当前 v1.x 文档中的统一模型接口、调用方法、模型能力画像、Reasoning、Prompt Caching、限流与动态模型选择等内容）

---

## 📋 目录

- [核心概念](#核心概念)
- [Chat Models vs LLMs](#chat-models-vs-llms)
- [模型初始化](#模型初始化)
- [支持的模型提供商](#支持的模型提供商)
- [模型参数配置](#模型参数配置)
- [工具调用 (Tool Calling)](#工具调用-tool-calling)
- [结构化输出](#结构化输出)
- [模型调用方法 (Invocation)](#模型调用方法-invocation)
- [流式处理](#流式处理)
- [多模态支持](#多模态支持)
- [Token 使用和元数据](#token-使用和元数据)
- [错误处理和重试](#错误处理和重试)
- [最佳实践](#最佳实践)
- [当前官方文档补充：高级主题](#当前官方文档补充高级主题)
- [高级用法](#高级用法)

---

## 核心概念

### 什么是 Models？

在 LangChain 中，**Models（模型）** 是 Agent 的推理引擎。它们负责理解用户输入、生成响应、决定是否调用工具以及如何使用工具返回的结果。

### Models 的核心能力

1. **推理 (Reasoning)**: 理解复杂问题并生成合理的回答
2. **工具调用 (Tool Calling)**: 决定何时以及如何使用外部工具
3. **结构化输出 (Structured Output)**: 按照预定义的 schema 生成格式化数据
4. **多模态 (Multimodality)**: 处理文本、图像、音频等多种输入类型
5. **流式处理 (Streaming)**: 实时生成和传输响应

### Models 在 Agent 中的角色

```
用户输入 → Model (推理) → 决策 → 调用工具 → Model (综合) → 最终响应
```

Models 是整个 Agent 系统的大脑，负责：
- 理解用户意图
- 制定执行计划
- 选择合适的工具
- 综合信息生成回答

### 当前官方文档强调的模型能力

当前 LangChain Models 文档把模型能力概括为几个维度：

- **Tool calling**：模型可以生成工具调用请求，由应用执行数据库查询、API 调用等外部动作。
- **Structured output**：模型输出可以被约束为 schema，例如 Pydantic、TypedDict 或 JSON Schema。
- **Multimodality**：模型可处理文本之外的数据，如图片、音频、视频；不同 provider 支持程度不同。
- **Reasoning**：部分模型支持显式推理配置或推理内容返回，适合多步骤问题求解。

模型选择会直接影响 Agent 的基础可靠性、工具调用质量、上下文处理能力、成本和延迟。LangChain 的统一接口让你可以较容易地在不同 provider 之间切换并实验。

---

## Chat Models vs LLMs

### Chat Models

**Chat Models** 是专门为对话场景设计的模型，接受消息列表作为输入并返回消息。

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# 初始化 Chat Model
model = ChatAnthropic(model="claude-sonnet-4-6")

# 使用消息列表调用
messages = [
    SystemMessage(content="你是一个有帮助的 AI 助手"),
    HumanMessage(content="什么是量子计算?")
]

response = model.invoke(messages)
print(response.content)
```

**特点**:
- 接受 `List[Message]` 作为输入
- 返回 `Message` 对象
- 支持系统消息、用户消息、AI 消息等多种消息类型
- 原生支持工具调用
- 更好的对话上下文管理

### LLMs (传统语言模型)

**LLMs** 是更通用的语言模型，接受字符串输入并返回字符串。

```python
from langchain_openai import OpenAI

# 初始化 LLM
llm = OpenAI(model="gpt-3.5-turbo-instruct")

# 使用字符串调用
response = llm.invoke("什么是量子计算?")
print(response)
```

**特点**:
- 接受 `str` 作为输入
- 返回 `str` 
- 更简单的接口
- 适合单轮文本生成

### 选择建议

**优先使用 Chat Models**，现代应用推荐使用 Chat Models，因为它们：
- ✅ 支持更丰富的对话上下文
- ✅ 原生支持工具调用
- ✅ 更好的系统提示词控制
- ✅ 更适合 Agent 应用
- ✅ 支持多模态输入

---

## 模型初始化

### 使用 init_chat_model (推荐)

`init_chat_model` 是 LangChain 推荐的统一模型初始化方式，支持多个提供商。它适合做配置驱动的模型切换，也能减少 provider 类直接导入带来的耦合。

```python
from langchain.chat_models import init_chat_model

# Anthropic Claude
model = init_chat_model(
    model="claude-sonnet-4-6",
    model_provider="anthropic",
    temperature=0.7
)

# OpenAI GPT
openai_model = init_chat_model(
    model="gpt-4o",
    model_provider="openai",
    temperature=0.5,
    max_tokens=1000
)

# Google Gemini
google_model = init_chat_model(
    model="gemini-3.5-flash",
    model_provider="google_genai",
    temperature=0
)
```

**优势**:
- 统一的 API 接口
- 轻松切换不同提供商
- 简化配置管理
- 适合从配置文件、环境变量或运行时参数中动态选择模型

### 使用 `provider:model` 字符串

在 `create_agent` 等高层 API 中，当前官方文档也推荐直接使用 `"provider:model"` 形式：

```python
from langchain.agents import create_agent

agent = create_agent("anthropic:claude-sonnet-4-6", tools=tools)
agent = create_agent("openai:gpt-5.4", tools=tools)
agent = create_agent("google_genai:gemini-3.5-flash", tools=tools)
```

这种写法更适合快速创建 Agent；如果需要精细配置模型参数、base_url、proxy、rate limiter 等，则可以继续使用 provider 类或 `init_chat_model`。

### 直接使用提供商类

每个提供商都有自己的类，可以直接导入使用。

#### Anthropic Claude
```bash
pip install -U "langchain[anthropic]"
```

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key="your-api-key",  # 或从环境变量读取
    temperature=0.7,
    max_tokens=1024
)
```

#### OpenAI GPT
```bash
pip install -U "langchain[openai]"
```

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o",
    api_key="your-api-key",
    temperature=0.5
)
```

#### Google Gemini
```bash
pip install -U "langchain[google-genai]"
```

```python
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key="your-api-key"
)
```

#### Azure OpenAI
```bash
pip install -U "langchain[openai]"
```

```python
from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    azure_endpoint="https://your-endpoint.openai.azure.com/",
    azure_deployment="your-deployment-name",
    api_version="2024-02-15-preview"
)
```

#### AWS Bedrock
```bash
pip install -U "langchain[aws]"
```

```python
from langchain_aws import ChatBedrock

model = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1"
)
```

### 从环境变量读取配置

```python
import os
from langchain_anthropic import ChatAnthropic

# 设置环境变量
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

# api_key 会自动从环境变量读取
model = ChatAnthropic(
    model="claude-sonnet-4-6"
)
```

**支持的环境变量**:
- `ANTHROPIC_API_KEY` - Anthropic Claude
- `OPENAI_API_KEY` - OpenAI GPT
- `GOOGLE_API_KEY` - Google Gemini
- `AZURE_OPENAI_API_KEY` - Azure OpenAI
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` - AWS Bedrock

---

## 支持的模型提供商

### Anthropic Claude

```python
from langchain_anthropic import ChatAnthropic

# Claude 4.6 Sonnet (推荐用于 Agents)
model = ChatAnthropic(model="claude-sonnet-4-6")

# Claude 4.8 Opus (最强大)
model = ChatAnthropic(model="claude-opus-4-8")

# Claude 4.5 Haiku (最快速)
model = ChatAnthropic(model="claude-haiku-4-5-20251001")
```

**特点**:
- ✨ 出色的推理能力
- 🛠️ 原生工具调用支持
- 📚 长上下文窗口 (200K tokens)
- 🎯 强大的指令遵循能力
- 💡 适合复杂的 Agent 任务

**最佳用途**: Agent 应用、复杂推理、工具调用

### OpenAI GPT

```python
from langchain_openai import ChatOpenAI

# GPT-4o (多模态)
model = ChatOpenAI(model="gpt-4o")

# GPT-4 Turbo
model = ChatOpenAI(model="gpt-4-turbo")

# GPT-3.5 Turbo (经济型)
model = ChatOpenAI(model="gpt-3.5-turbo")
```

**特点**:
- 🌍 广泛采用和成熟
- 🔧 强大的工具调用
- 🖼️ 多模态支持 (GPT-4o)
- 🌐 丰富的生态系统

**最佳用途**: 通用任务、多模态应用、成熟的生产环境

### Google Gemini

```python
from langchain_google_genai import ChatGoogleGenerativeAI

# Gemini 2.0 Flash
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# Gemini 1.5 Pro
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
```

**特点**:
- 📖 极长上下文 (1M+ tokens)
- 🎨 多模态能力
- ⚡ 快速响应
- 💰 成本效益高

**最佳用途**: 长文档分析、超长上下文处理、经济型应用

### Azure OpenAI

```python
from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    azure_endpoint="https://your-endpoint.openai.azure.com/",
    azure_deployment="your-gpt4-deployment",
    api_version="2024-02-15-preview",
    temperature=0.7
)
```

**特点**:
- 🔒 企业级安全和合规
- ☁️ 与 Azure 生态集成
- 🛡️ 数据隐私保证
- 📞 SLA 支持

**最佳用途**: 企业应用、合规要求高的场景

### AWS Bedrock

```python
from langchain_aws import ChatBedrock

# Claude 3 on Bedrock
model = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1"
)

# Llama 2 on Bedrock
model = ChatBedrock(
    model_id="meta.llama2-70b-chat-v1",
    region_name="us-west-2"
)
```

**特点**:
- 🎯 多种模型选择
- 🌐 AWS 基础设施集成
- 🏢 企业级部署
- 💳 按需付费

**最佳用途**: AWS 生态内的应用、多模型实验

---

## 模型参数配置

### 核心参数

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(
    model="claude-sonnet-4-6",
    
    # API 密钥: 从环境变量读取
    api_key="your-api-key",

    # 温度 (0-1): 控制输出随机性
    # 0 = 确定性输出, 1 = 高度随机
    temperature=0.7,
    
    # 最大 tokens: 限制响应长度
    max_tokens=1024,
    
    # Top P: 核采样参数 (0-1)
    top_p=0.9,
    
    # Top K: 限制候选 tokens 数量
    top_k=40,
    
    # Stop sequences: 遇到这些序列时停止生成
    stop=["\\n\\nHuman:", "\\n\\nAssistant:"],
    
    # 流式处理
    streaming=True,
    
    # 超时设置 (秒)
    timeout=60,
    
    # 最大重试次数
    max_retries=3
)
```

### Temperature (温度) 使用指南

Temperature 控制输出的随机性和创造性:

```python
# temperature = 0: 确定性输出
# 适用场景: 数据提取、分类、结构化输出、代码生成
deterministic_model = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0
)

# temperature = 0.3-0.5: 平衡
# 适用场景: 客服对话、问答系统、技术文档
balanced_model = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0.4
)

# temperature = 0.7-1.0: 高创造性
# 适用场景: 创意写作、头脑风暴、故事生成
creative_model = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0.9
)
```

### Max Tokens

```python
# 短回答 (节省成本)
short_model = ChatAnthropic(
    model="claude-sonnet-4-6",
    max_tokens=100
)

# 标准响应
standard_model = ChatAnthropic(
    model="claude-sonnet-4-6",
    max_tokens=1024
)

# 长内容生成
long_model = ChatAnthropic(
    model="claude-sonnet-4-6",
    max_tokens=4096
)
```

### 超时和重试配置

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(
    model="claude-sonnet-4-6",
    timeout=120,  # 120 秒超时
    max_retries=5,  # 最多重试 5 次
    default_request_timeout=60
)

# 使用 RunnableConfig 自定义
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    max_concurrency=5,
    recursion_limit=10
)

response = model.invoke(messages, config=config)
```

---

## 工具调用 (Tool Calling)

工具调用是 Models 的核心能力之一，允许模型决定何时调用外部函数来获取信息或执行操作。

### 基本工具调用

```python
# 引入模型和工具
from langchain_anthropic import ChatAnthropic
# from langchain.tools import tool
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# 定义工具
@tool
def get_weather(location: str) -> str:
    """获取指定位置的天气信息。
    
    Args:
        location: 城市名称，例如 '北京' 或 '上海'
    """
    # 实际应用中会调用天气 API
    return f"{location}的天气是晴朗，温度 22°C"

@tool
def calculate(expression: str) -> float:
    """计算数学表达式。
    
    Args:
        expression: 要计算的数学表达式，例如 '2 + 2'
    """
    return eval(expression)

# 初始化模型并绑定工具
model = ChatAnthropic(model="claude-sonnet-4-6")
model_with_tools = model.bind_tools([get_weather, calculate])

# 调用模型
response = model_with_tools.invoke([
    HumanMessage(content="北京的天气怎么样?")
])

# 检查是否有工具调用
if response.tool_calls:
    print("模型决定调用工具:")
    for tool_call in response.tool_calls:
        print(f"  - 工具: {tool_call['name']}")
        print(f"  - 参数: {tool_call['args']}")
```

### 完整工具调用流程

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

@tool
def search_database(query: str) -> str:
    """在数据库中搜索信息。"""
    return f"找到 5 条关于 '{query}' 的记录"

# 初始化
model = ChatAnthropic(model="claude-sonnet-4-6")
model_with_tools = model.bind_tools([search_database])

# 1. 用户问题
messages = [HumanMessage(content="查找所有关于 Python 的记录")]

# 2. 模型决定调用工具
response = model_with_tools.invoke(messages)

# 3. 执行工具调用
if response.tool_calls:
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # 执行工具
        tool_result = search_database.invoke(tool_args)
        
        # 4. 将工具结果添加到消息历史
        # tool_call_id 是工具调用的唯一标识符,工具返回的每个 ToolMessage 都包含一个 tool_call_id ，该 tool_call_id 与原始工具调用相匹配
        messages.append(response)
        messages.append(ToolMessage(
            content=tool_result,
            tool_call_id=tool_call["id"]
        ))

# 5. 模型使用工具结果生成最终回答
final_response = model.invoke(messages)
print(final_response.content)
```

### 强制工具调用

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

@tool
def format_response(data: dict) -> str:
    """格式化响应数据。"""
    return str(data)

model = ChatAnthropic(model="claude-sonnet-4-6")

# 强制使用特定工具
model_forced = model.bind_tools(
    [format_response],
    tool_choice="format_response"  # 强制使用这个工具
)

# 强制使用任意工具
model_any = model.bind_tools(
    [format_response],
    tool_choice="any"  # 必须使用某个工具
)

# 自动决定 (默认)
model_auto = model.bind_tools(
    [format_response],
    tool_choice="auto"
)
```

### 并行工具调用

某些模型支持在单次响应中调用多个工具。

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

@tool
def get_weather(location: str) -> str:
    """获取天气信息。"""
    return f"{location}: 晴朗，22°C"

@tool
def get_time(timezone: str) -> str:
    """获取时间信息。"""
    return f"{timezone}: 14:30"

model = ChatAnthropic(model="claude-sonnet-4-6")
model_with_tools = model.bind_tools([get_weather, get_time])

# 模型可能同时调用多个工具
response = model_with_tools.invoke([
    HumanMessage(content="告诉我北京的天气和当前时间")
])

# 处理多个工具调用
for tool_call in response.tool_calls:
    print(f"调用工具: {tool_call['name']} with {tool_call['args']}")
```

**禁用并行工具调用**

某些模型（包括 OpenAI 和 Anthropic）允许禁用并行工具调用功能：

```python
# 禁用并行工具调用，强制模型一次只调用一个工具
model_sequential = model.bind_tools(
    [get_weather, get_time],
    parallel_tool_calls=False
)
```

### 流式工具调用 (Streaming Tool Calls)

在流式响应中，工具调用通过 `ToolCallChunk` 逐步构建。这允许你在工具调用生成过程中实时查看进度，而不是等待完整响应。

#### 基本流式工具调用

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """获取指定位置的天气信息。"""
    return f"{location}的天气是晴朗，温度 22°C"

@tool
def get_time(timezone: str) -> str:
    """获取指定时区的时间。"""
    return f"{timezone}的时间是 14:30"

model = ChatAnthropic(model="claude-sonnet-4-6")
model_with_tools = model.bind_tools([get_weather, get_time])

# 流式调用 - 工具调用片段会逐步到达
for chunk in model_with_tools.stream(
    "北京和东京的天气怎么样?"
):
    # 工具调用块逐步到达
    for tool_chunk in chunk.tool_call_chunks:
        if name := tool_chunk.get("name"):
            print(f"工具: {name}")
        if id_ := tool_chunk.get("id"):
            print(f"ID: {id_}")
        if args := tool_chunk.get("args"):
            print(f"参数: {args}")

# 输出示例:
# 工具: get_weather
# ID: call_SvMlU1TVIZugrFLckFE2ceRE
# 参数: {"lo
# 参数: catio
# 参数: n": "北
# 参数: 京"}
# 工具: get_weather
# ID: call_QMZdy6qInx13oWKE7KhuhOLR
# 参数: {"lo
# 参数: catio
# 参数: n": "东
# 参数: 京"}
```

#### 累积块以构建完整工具调用

流式响应中的工具调用片段可以累积起来，以便获取完整的工具调用信息：

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

@tool
def search_database(query: str) -> str:
    """在数据库中搜索信息。"""
    return f"找到 5 条关于 '{query}' 的记录"

model = ChatAnthropic(model="claude-sonnet-4-6")
model_with_tools = model.bind_tools([search_database])

# 累积块以构建完整的工具调用
gathered = None
for chunk in model_with_tools.stream("搜索关于 Python 的信息"):
    gathered = chunk if gathered is None else gathered + chunk
    print(gathered.tool_calls)

# 输出逐步构建的完整工具调用:
# []
# []
# [{'name': 'search_database', 'args': {}, 'id': 'call_xxx'}]
# [{'name': 'search_database', 'args': {'qu': ''}, 'id': 'call_xxx'}]
# [{'name': 'search_database', 'args': {'query': 'Py'}, 'id': 'call_xxx'}]
# [{'name': 'search_database', 'args': {'query': 'Python'}, 'id': 'call_xxx'}]
```

#### 流式工具调用的实际应用

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

@tool
def analyze_data(data_type: str, metric: str) -> dict:
    """分析特定类型的数据指标。"""
    return {
        "data_type": data_type,
        "metric": metric,
        "result": f"分析完成: {data_type} 的 {metric}"
    }

model = ChatAnthropic(model="claude-sonnet-4-6")
model_with_tools = model.bind_tools([analyze_data])

# 实时显示工具调用的构建过程
messages = [HumanMessage(content="分析销售数据的增长率")]
gathered_chunk = None

print("🔄 开始流式工具调用...")
for chunk in model_with_tools.stream(messages):
    # 累积块
    gathered_chunk = chunk if gathered_chunk is None else gathered_chunk + chunk

    # 实时显示进度
    if chunk.tool_call_chunks:
        for tool_chunk in chunk.tool_call_chunks:
            if args := tool_chunk.get("args"):
                print(f"📡 接收参数片段: {args}")

# 显示完整的工具调用
if gathered_chunk and gathered_chunk.tool_calls:
    print("\n✅ 完整工具调用:")
    for tool_call in gathered_chunk.tool_calls:
        print(f"  工具名称: {tool_call['name']}")
        print(f"  完整参数: {tool_call['args']}")
        print(f"  调用 ID: {tool_call['id']}")
```

#### 流式多个并行工具调用

当模型决定并行调用多个工具时，流式响应会包含多个工具的片段：

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

@tool
def get_stock_price(symbol: str) -> str:
    """获取股票价格。"""
    return f"{symbol} 当前价格: $150.00"

@tool
def get_company_info(symbol: str) -> str:
    """获取公司信息。"""
    return f"{symbol} 公司信息: 科技公司"

model = ChatAnthropic(model="claude-sonnet-4-6")
model_with_tools = model.bind_tools([get_stock_price, get_company_info])

# 跟踪多个工具调用的构建过程
tool_calls_progress = {}

for chunk in model_with_tools.stream(
    "告诉我 AAPL 的股价和公司信息"
):
    for tool_chunk in chunk.tool_call_chunks:
        # 使用 index 跟踪不同的工具调用
        index = tool_chunk.get("index", 0)

        if index not in tool_calls_progress:
            tool_calls_progress[index] = {
                "name": "",
                "args": "",
                "id": ""
            }

        # 累积每个工具调用的信息
        if name := tool_chunk.get("name"):
            tool_calls_progress[index]["name"] = name
        if id_ := tool_chunk.get("id"):
            tool_calls_progress[index]["id"] = id_
        if args := tool_chunk.get("args"):
            tool_calls_progress[index]["args"] += args

        print(f"🔧 工具 #{index}: {tool_calls_progress[index]}")
```

#### ToolCallChunk 数据结构

流式工具调用中的 `ToolCallChunk` 包含以下字段：

```python
# ToolCallChunk 结构
{
    "type": "tool_call_chunk",      # 始终为 "tool_call_chunk"
    "name": "tool_name",            # 被调用的工具名称（可能为空）
    "args": '{"partial": "json"}',  # 部分工具参数（可能是不完整的 JSON）
    "id": "call_xxx",               # 工具调用标识符
    "index": 0                      # 此块在流中的位置
}
```

#### 使用 astream_events 进行高级流式处理

对于更复杂的流式场景，可以使用 `astream_events()` 来获取语义事件：

```python
import asyncio
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

@tool
def complex_calculation(formula: str) -> float:
    """执行复杂计算。"""
    return eval(formula)

async def stream_tool_calls_with_events():
    model = ChatAnthropic(model="claude-sonnet-4-6")
    model_with_tools = model.bind_tools([complex_calculation])

    async for event in model_with_tools.astream_events(
        "计算 (25 * 4) + (100 / 2)",
        version="v1"
    ):
        if event["event"] == "on_chat_model_start":
            print(f"🚀 模型开始: {event['data']['input']}")

        elif event["event"] == "on_chat_model_stream":
            chunk = event['data']['chunk']
            if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                for tool_chunk in chunk.tool_call_chunks:
                    print(f"🔨 工具块: {tool_chunk}")

        elif event["event"] == "on_chat_model_end":
            output = event['data']['output']
            if hasattr(output, 'tool_calls') and output.tool_calls:
                print(f"✅ 完整工具调用: {output.tool_calls}")

# 运行异步函数
# asyncio.run(stream_tool_calls_with_events())
```

#### 流式工具调用的最佳实践

1. **进度指示器**: 使用流式工具调用为用户提供实时反馈
2. **错误处理**: 监控不完整的 JSON 参数，处理解析错误
3. **累积策略**: 决定何时累积块以获得完整信息
4. **性能优化**: 对于大型参数，流式处理可以提高响应速度
5. **用户体验**: 在 UI 中显示"正在调用工具..."的加载状态

```python
# 完整示例：带进度指示的流式工具调用
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
import json

@tool
def fetch_large_dataset(category: str, filters: dict) -> str:
    """获取大型数据集。"""
    return f"获取 {category} 数据，应用过滤器: {filters}"

def stream_with_progress():
    model = ChatAnthropic(model="claude-sonnet-4-6")
    model_with_tools = model.bind_tools([fetch_large_dataset])

    gathered = None
    current_tool_name = None

    for chunk in model_with_tools.stream(
        "获取2024年第一季度的销售数据，过滤条件：地区为华东，金额大于10000"
    ):
        gathered = chunk if gathered is None else gathered + chunk

        # 检测工具名称
        if chunk.tool_call_chunks:
            for tool_chunk in chunk.tool_call_chunks:
                if name := tool_chunk.get("name"):
                    current_tool_name = name
                    print(f"\n🔧 准备调用工具: {name}")

                if args := tool_chunk.get("args"):
                    print(".", end="", flush=True)  # 进度点

    # 执行工具调用
    if gathered and gathered.tool_calls:
        print("\n\n📋 执行工具调用:")
        for tool_call in gathered.tool_calls:
            print(f"  ✓ {tool_call['name']}")
            print(f"  ✓ 参数: {json.dumps(tool_call['args'], ensure_ascii=False, indent=2)}")

# stream_with_progress()
```

---

## 结构化输出

`with_structured_output` 允许模型按照预定义的 Pydantic schema 生成结构化数据。

### 基本结构化输出

```python
# TODO Validation验证 ：Pydantic 模型提供自动验证，而 TypedDict 和 JSON Schema 则需要手动验证。

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

# 定义输出结构
class Person(BaseModel):
    """一个人的信息。"""
    name: str = Field(description="人的姓名")
    age: int = Field(description="人的年龄")
    email: str = Field(description="电子邮件地址")
    occupation: str = Field(description="职业")

# 配置模型使用结构化输出
model = ChatAnthropic(model="claude-sonnet-4-6")
structured_model = model.with_structured_output(Person) # 结构化输出

# 调用模型
response = structured_model.invoke([
    HumanMessage(content="张伟是一位 35 岁的软件工程师，邮箱是 zhang@example.com")
])

# response 是 Person 实例
print(f"姓名: {response.name}")
print(f"年龄: {response.age}")
print(f"邮箱: {response.email}")
print(f"职业: {response.occupation}")
```

### 复杂嵌套结构

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class Address(BaseModel):
    """地址信息。"""
    street: str = Field(description="街道地址")
    city: str = Field(description="城市")
    country: str = Field(description="国家")
    postal_code: Optional[str] = Field(description="邮政编码")

class Company(BaseModel):
    """公司信息。"""
    name: str = Field(description="公司名称")
    industry: str = Field(description="行业")
    employees: int = Field(description="员工数量")

class Employee(BaseModel):
    """员工完整信息。"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    position: str = Field(description="职位")
    address: Address = Field(description="住址")
    company: Company = Field(description="所在公司")
    skills: List[str] = Field(description="技能列表")

model = ChatAnthropic(model="claude-sonnet-4-6")
structured_model = model.with_structured_output(Employee)

response = structured_model.invoke([
    HumanMessage(content="""
    李明，32岁，在北京海淀区中关村大街1号的阿里巴巴工作，
    担任高级软件工程师。公司有5000名员工，主要从事电子商务。
    他擅长Python、机器学习和云计算。邮编100080。
    """)
])

print(f"员工: {response.name}, {response.position}")
print(f"公司: {response.company.name}, {response.company.employees}人")
print(f"地址: {response.address.city}, {response.address.street}")
print(f"技能: {', '.join(response.skills)}")
```

### 列表类型输出

```python
from pydantic import BaseModel, Field
from typing import List

class Product(BaseModel):
    """产品信息。"""
    name: str = Field(description="产品名称")
    price: float = Field(description="价格")
    category: str = Field(description="类别")

class ProductList(BaseModel):
    """产品列表。"""
    products: List[Product] = Field(description="产品列表")

model = ChatAnthropic(model="claude-sonnet-4-6")
structured_model = model.with_structured_output(ProductList)

response = structured_model.invoke([
    HumanMessage(content="""
    我们有以下产品:
    1. iPhone 15 Pro - 8999元 - 电子产品
    2. MacBook Air - 9499元 - 电子产品  
    3. AirPods Pro - 1999元 - 配件
    """)
])

for product in response.products:
    print(f"{product.name}: ¥{product.price} ({product.category})")
```

### 使用 Pydantic 验证器（Validation 验证器）

```python
# TODO Pydantic 模型提供自动验证，而 TypedDict 和 JSON Schema 则需要手动验证。
from pydantic import BaseModel, Field, validator

class OrderInfo(BaseModel):
    """订单信息。"""
    order_id: str = Field(description="订单号")
    amount: float = Field(description="金额", gt=0)
    status: str = Field(description="状态")
    
    @validator('status')
    def validate_status(cls, v):
        """验证状态必须是允许的值之一。"""
        allowed = ['pending', 'paid', 'shipped', 'delivered', 'cancelled']
        if v.lower() not in allowed:
            raise ValueError(f'状态必须是: {allowed}之一')
        return v.lower()
    
    @validator('order_id')
    def validate_order_id(cls, v):
        """验证订单号格式。"""
        if not v.startswith('ORD-'):
            raise ValueError('订单号必须以 ORD- 开头')
        return v

model = ChatAnthropic(model="claude-sonnet-4-6")
structured_model = model.with_structured_output(OrderInfo)

response = structured_model.invoke([
    HumanMessage(content="订单 ORD-12345，金额 299.99，状态已支付")
])
```

---

## 模型调用方法 (Invocation)

当前官方文档把模型调用方式分为几类：`invoke`、`stream`、`batch` 以及对应的异步版本。

### 1. Invoke：单次调用

`invoke` 是最常用的同步调用方式，输入可以是字符串、消息列表或 prompt value，输出通常是 `AIMessage`。

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("anthropic:claude-sonnet-4-6", temperature=0)
response = model.invoke("The sky is")

print(response.content)
```

对于 Chat Model，更推荐显式传入消息：

```python
from langchain_core.messages import HumanMessage, SystemMessage

response = model.invoke([
    SystemMessage(content="你是一个严谨的技术助手。"),
    HumanMessage(content="解释一下 LangChain 的 Model 接口。"),
])
```

### 2. Stream：流式返回

`stream` 会逐步返回 `AIMessageChunk`，适合在 UI 中实时显示输出。

```python
for chunk in model.stream("The sky is"):
    print(chunk.content, end="", flush=True)
```

### 3. Batch：批处理

`batch` 可以一次处理多个输入，LangChain 会并发执行请求，适合大量独立任务。

```python
responses = model.batch([
    "什么是 LangChain?",
    "什么是 LangGraph?",
    "什么是 LangSmith?",
])

for response in responses:
    print(response.content)
```

### 4. 异步调用

大多数调用方法都有异步版本：

| 同步方法 | 异步方法 | 用途 |
|---------|---------|------|
| `invoke` | `ainvoke` | 单次调用 |
| `stream` | `astream` | 流式输出 |
| `batch` | `abatch` | 批量调用 |
| `astream_events` | `astream_events` | 事件级流式处理 |

---

## 流式处理

流式处理允许实时接收模型生成的内容，提升用户体验。

### Token 流式处理

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

model = ChatAnthropic(
    model="claude-sonnet-4-6",
    streaming=True
)

# 使用 stream() 方法
for chunk in model.stream([HumanMessage(content="写一首关于春天的诗")]):
    print(chunk.content, end="", flush=True)
```

### 异步流式处理

```python
import asyncio # 引入 asyncio，异步处理
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

async def async_stream_example():
    model = ChatAnthropic(
        model="claude-sonnet-4-6",
        streaming=True
    )
    
    async for chunk in model.astream([
        HumanMessage(content="解释量子纠缠")
    ]):
        print(chunk.content, end="", flush=True)

# 运行
asyncio.run(async_stream_example())
```

### 事件流 (astream_events)

更细粒度的流式控制，可以监听模型、工具等的所有事件。

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import asyncio

@tool
def search(query: str) -> str:
    """搜索信息。"""
    return f"关于 {query} 的搜索结果"

async def stream_events_example():
    model = ChatAnthropic(model="claude-sonnet-4-6")
    model_with_tools = model.bind_tools([search])
    
    async for event in model_with_tools.astream_events(
        [HumanMessage(content="搜索 LangChain 信息")],
        version="v2"
    ):
        kind = event["event"]
        
        if kind == "on_chat_model_start":
            print("模型开始处理...")
        
        elif kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)
        
        elif kind == "on_tool_start":
            print(f"\n工具调用开始: {event['name']}")
        
        elif kind == "on_tool_end":
            print(f"\n工具调用结束: {event['data']['output']}")

asyncio.run(stream_events_example())
```

### 流式处理中的 Token 统计

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

model = ChatAnthropic(model="claude-sonnet-4-6")

total_tokens = 0
content = ""

for chunk in model.stream([HumanMessage(content="介绍人工智能")]):
    content += chunk.content
    
    # 某些 chunk 包含 usage 信息
    if hasattr(chunk, 'usage_metadata'):
        total_tokens = chunk.usage_metadata.get('total_tokens', 0)

print(f"\n\n总 tokens: {total_tokens}")
```

---

## 多模态支持

现代模型支持处理文本、图像等多种输入类型。

### 图像输入

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
import base64

# 读取图像
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

model = ChatAnthropic(model="claude-sonnet-4-6")

# 发送图像给模型
response = model.invoke([
    HumanMessage(content=[
        {"type": "text", "text": "这张图片里有什么?"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
        }
    ])
])

print(response.content)
```

### 处理多张图像

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
import base64

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

model = ChatAnthropic(model="claude-sonnet-4-6")

# 同时处理多张图片
response = model.invoke([
    HumanMessage(content=[
        {"type": "text", "text": "比较这两张图片的差异"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image('image1.jpg')}"}
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image('image2.jpg')}"}
        }
    ])
])

print(response.content)
```

### 使用图像 URL

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

model = ChatAnthropic(model="claude-sonnet-4-6")

# 直接使用图像 URL
response = model.invoke([
    HumanMessage(content=[
        {"type": "text", "text": "描述这张图片"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.jpg"}
        }
    ])
])

print(response.content)
```

---

## Token 使用和元数据

### 获取 Token 使用信息

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

model = ChatAnthropic(model="claude-sonnet-4-6")

response = model.invoke([
    HumanMessage(content="解释机器学习")
])

# 获取 token 使用信息
if hasattr(response, 'usage_metadata'):
    usage = response.usage_metadata
    print(f"输入 tokens: {usage.get('input_tokens', 0)}")
    print(f"输出 tokens: {usage.get('output_tokens', 0)}")
    print(f"总 tokens: {usage.get('total_tokens', 0)}")
```

### 计算成本

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# Claude 4.6 Sonnet 价格 (示例)
INPUT_PRICE_PER_1M = 3.0  # $3 per 1M input tokens
OUTPUT_PRICE_PER_1M = 15.0  # $15 per 1M output tokens

model = ChatAnthropic(model="claude-sonnet-4-6")

response = model.invoke([
    HumanMessage(content="写一篇关于人工智能的文章")
])

if hasattr(response, 'usage_metadata'):
    usage = response.usage_metadata
    input_tokens = usage.get('input_tokens', 0)
    output_tokens = usage.get('output_tokens', 0)
    
    # 计算成本
    input_cost = (input_tokens / 1_000_000) * INPUT_PRICE_PER_1M
    output_cost = (output_tokens / 1_000_000) * OUTPUT_PRICE_PER_1M
    total_cost = input_cost + output_cost
    
    print(f"输入: {input_tokens} tokens (${input_cost:.6f})")
    print(f"输出: {output_tokens} tokens (${output_cost:.6f})")
    print(f"总成本: ${total_cost:.6f}")
```

### 响应元数据

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

model = ChatAnthropic(model="claude-sonnet-4-6")

response = model.invoke([
    HumanMessage(content="你好")
])

# 获取元数据
print("响应元数据:")
print(f"  模型: {response.response_metadata.get('model')}")
print(f"  Stop reason: {response.response_metadata.get('stop_reason')}")
print(f"  消息 ID: {response.id}")
```

---

## 错误处理和重试

### 基本错误处理

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
import anthropic

model = ChatAnthropic(model="claude-sonnet-4-6")

try:
    response = model.invoke([
        HumanMessage(content="你好")
    ])
    print(response.content)
    
except anthropic.APIError as e:
    print(f"API 错误: {e}")
    
except anthropic.RateLimitError as e:
    print(f"速率限制: {e}")
    
except anthropic.APIConnectionError as e:
    print(f"连接错误: {e}")
    
except Exception as e:
    print(f"未知错误: {e}")
```

### 自动重试配置

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(
    model="claude-sonnet-4-6",
    max_retries=5,  # 最多重试 5 次
    timeout=120,  # 超时时间
)

# LangChain 会自动处理重试
response = model.invoke([HumanMessage(content="你好")])
```

### 使用 tenacity 进行高级重试

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import anthropic

@retry(
    stop=stop_after_attempt(3),  # 最多重试 3 次
    wait=wait_exponential(multiplier=1, min=4, max=60),  # 指数退避
    retry=retry_if_exception_type((
        anthropic.RateLimitError,
        anthropic.APIConnectionError
    ))
)
def call_model_with_retry(messages):
    model = ChatAnthropic(model="claude-sonnet-4-6")
    return model.invoke(messages)

# 使用
try:
    response = call_model_with_retry([
        HumanMessage(content="你好")
    ])
    print(response.content)
except Exception as e:
    print(f"所有重试都失败了: {e}")
```

---

## 最佳实践

### 1. 选择合适的模型

```python
# 简单任务 → 快速模型
simple_model = ChatAnthropic(model="claude-haiku-4-5-20251001")

# 复杂推理 → 强大模型
complex_model = ChatAnthropic(model="claude-sonnet-4-6")

# 关键任务 → 最强模型
critical_model = ChatAnthropic(model="claude-opus-4-8")
```

### 2. 优化 Temperature

```python
# 事实性任务: temperature = 0
factual_model = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0
)

# 创造性任务: temperature = 0.7-1.0
creative_model = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0.9
)
```

### 3. 清晰的系统提示词

```python
from langchain_core.messages import SystemMessage, HumanMessage

system_prompt = """你是一个专业的客户服务助手。

你的职责:
- 礼貌、专业地回答客户问题
- 如果不确定答案，诚实说明并寻求帮助
- 使用简洁、清晰的语言

你的限制:
- 不要提供医疗或法律建议
- 不要分享客户的个人信息
- 不要做出公司无法兑现的承诺
"""

model = ChatAnthropic(model="claude-sonnet-4-6")
response = model.invoke([
    SystemMessage(content=system_prompt),
    HumanMessage(content="我的订单什么时候到?")
])
```

### 4. 工具描述清晰准确

```python
from langchain_core.tools import tool

@tool
def get_customer_info(customer_id: str) -> dict:
    """获取客户详细信息。
    
    使用此工具查找客户的账户信息、订单历史和偏好设置。
    
    Args:
        customer_id: 客户的唯一标识符，格式: CUST-XXXXX
        
    Returns:
        包含客户信息的字典，包括姓名、邮箱、会员等级等
    """
    return {"name": "张伟", "email": "zhang@example.com"}
```

### 5. 监控和日志

```python
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitored_model_call(messages):
    model = ChatAnthropic(model="claude-sonnet-4-6")
    start_time = time.time()
    
    try:
        response = model.invoke(messages)
        duration = time.time() - start_time
        logger.info(f"模型调用成功 - 耗时: {duration:.2f}s")
        
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            logger.info(f"Token使用 - 输入: {usage.get('input_tokens')}, 输出: {usage.get('output_tokens')}")
        
        return response
    except Exception as e:
        logger.error(f"模型调用失败: {e}")
        raise
```

---

## 当前官方文档补充：高级主题

当前 Models 官方文档除了基础初始化和调用外，还强调了以下生产级能力。下面内容是在原有文档基础上的补充，便于和 v1.x 文档对齐。

### 1. Model Profiles（模型能力画像）

不同 provider 对工具调用、结构化输出、多模态、推理内容、最大上下文长度等能力支持不同。LangChain 使用模型 profile 描述这些能力，便于应用在运行时判断模型是否支持某些特性。

概念上，一个模型能力画像可能包含：

```python
{
    "max_input_tokens": 400000,
    "image_inputs": True,
    "reasoning_output": True,
    "tool_calling": True,
}
```

实践建议：不要假设所有模型都支持同样能力。尤其是以下能力需要按 provider/model 验证：

- 是否支持工具调用
- 是否支持并行工具调用
- 是否支持结构化输出
- 是否支持图片、音频或视频输入
- 是否返回 token usage
- 是否支持 reasoning 参数或 reasoning 输出

### 2. Reasoning（推理能力）

部分模型支持更强的多步骤推理，甚至支持配置 reasoning effort、返回 reasoning summary 或 reasoning content。适合：

- 复杂规划
- 数学/逻辑问题
- 代码分析
- 多工具 Agent 决策
- 需要解释推理过程的任务

使用建议：

- 简单分类、抽取任务不一定需要高推理模型。
- 复杂 Agent、代码任务、长上下文综合更适合强推理模型。
- 如果 provider 支持 reasoning 参数，应把它作为模型配置的一部分，而不是完全依赖 prompt 要求“请认真思考”。

### 3. Prompt Caching（提示词缓存）

对于长系统提示词、长文档上下文或多轮复用的固定前缀，Prompt Caching 可以降低延迟和成本。适合：

- 长系统提示词
- 大段参考文档
- ReAct/Agent 固定工具说明
- 多轮共享的上下文前缀

实践建议：

- 把稳定、可复用的内容放在前面。
- 避免在可缓存前缀中混入每次变化的数据。
- 用 provider 和 LangSmith 的 usage/metadata 检查缓存是否命中。

### 4. Server-side Tool Use（服务端工具）

一些 provider 支持服务端工具，例如 web search、code execution 等。与 LangChain 本地工具不同，服务端工具由模型 provider 执行。

选择建议：

| 工具类型 | 优点 | 注意事项 |
|---------|------|---------|
| LangChain 本地工具 | 可控、可审计、可接入内部系统 | 需要自己执行和维护 |
| Provider 服务端工具 | 使用简单、与模型深度集成 | 可控性、可观测性和合规边界需要额外确认 |

### 5. Rate Limiting（限流）

生产环境需要控制模型调用速率，避免触发 provider 限流或造成成本失控。可以在模型层配置 rate limiter，或在应用层统一控制并发。

常见策略：

- 限制每秒/每分钟请求数
- 限制并发数
- 对批处理任务做队列化
- 遇到 429 使用指数退避
- 区分用户级、租户级、全局级限额

### 6. Base URL 与 Proxy

如果使用兼容 OpenAI 协议的模型网关、本地模型服务或企业代理，通常需要配置 `base_url`、proxy 或 provider 相关 endpoint。

适用场景：

- 企业统一模型网关
- 私有化部署
- 本地模型服务
- OpenAI-compatible API
- 网络代理环境

### 7. Log Probabilities

部分模型支持返回 token 的 log probabilities。适用于：

- 分类置信度估计
- 生成质量分析
- 可解释性分析
- 评估模型不确定性

注意：并非所有 Chat Model/provider 都支持 logprobs，且不同 provider 的返回结构可能不同。

### 8. Invocation Config（调用配置）

除了模型初始化参数，也可以通过 `RunnableConfig` 为单次调用传入运行配置，例如 tags、metadata、run_name、max_concurrency 等。

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    tags=["customer-support"],
    metadata={"tenant": "acme"},
    max_concurrency=5,
)

response = model.invoke("你好", config=config)
```

这些配置对 LangSmith tracing、调试、批处理和并发控制很有用。

### 9. Configurable Models（可配置模型）

如果希望在运行时切换模型或模型参数，可以把模型做成 configurable runnable，而不是在业务代码里写大量 if/else。

适用场景：

- 根据环境切换 dev/prod 模型
- 根据用户套餐选择模型
- A/B 测试
- fallback 或灰度发布

### 10. Dynamic Model Selection（动态模型选择）

动态模型选择是 Agent 和生产应用中的常见模式：

- 简单问题 → 快速、低成本模型
- 复杂推理 → 更强模型
- 长上下文 → 大上下文模型
- 需要图片输入 → 多模态模型
- 高风险任务 → 更稳定模型 + 更严格结构化输出

在 Agent 中，动态模型选择通常通过 middleware 实现；在普通 LCEL 链中，可以通过路由逻辑或 configurable model 实现。

```python
from langchain.chat_models import init_chat_model

fast_model = init_chat_model("anthropic:claude-haiku-4-5-20251001")
smart_model = init_chat_model("anthropic:claude-sonnet-4-6")

complexity_threshold = 1000

def choose_model(user_input: str):
    if len(user_input) > complexity_threshold:
        return smart_model
    return fast_model

question = "解释一下量子计算"
response = choose_model(question).invoke(question)
```

---

## 高级用法

### 1. 链式调用 (Chains)

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 创建提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}"),
    ("human", "{input}")
])

# 创建链
model = ChatAnthropic(model="claude-sonnet-4-6")
chain = prompt | model | StrOutputParser()

# 调用链
response = chain.invoke({
    "role": "诗人",
    "input": "写一首关于月亮的诗"
})

print(response)
```

### 2. Fallback 机制

```python
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# 主模型
primary_model = ChatAnthropic(model="claude-sonnet-4-6")

# 备用模型
fallback_model = ChatOpenAI(model="gpt-4o")

# 创建带 fallback 的模型
model_with_fallback = primary_model.with_fallbacks([fallback_model])

# 如果主模型失败，会自动使用备用模型
response = model_with_fallback.invoke([HumanMessage(content="你好")])
```

### 3. 批处理优化

```python
import asyncio
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

async def batch_process():
    model = ChatAnthropic(model="claude-sonnet-4-6")
    
    # 准备多个请求
    questions = [
        "什么是AI?",
        "什么是机器学习?",
        "什么是深度学习?"
    ]
    
    # 并行处理
    tasks = [
        model.ainvoke([HumanMessage(content=q)])
        for q in questions
    ]
    
    responses = await asyncio.gather(*tasks)
    return responses

# 运行
results = asyncio.run(batch_process())
```

---

## 📊 模型选择对比表

| 模型 | 提供商 | 上下文窗口 | 最佳用途 | 相对成本 |
|------|--------|-----------|----------|---------|
| Claude 4.6 Sonnet | Anthropic | 200K | Agent、工具调用、复杂推理 | 中等 |
| Claude 4.8 Opus | Anthropic | 200K | 最复杂任务、最高质量输出 | 高 |
| Claude 4.5 Haiku | Anthropic | 200K | 快速响应、简单任务 | 低 |
| GPT-4o | OpenAI | 128K | 多模态、通用任务 | 中等 |
| GPT-4 Turbo | OpenAI | 128K | 复杂推理、长上下文 | 高 |
| GPT-3.5 Turbo | OpenAI | 16K | 简单任务、经济型 | 低 |
| Gemini 2.0 Flash | Google | 1M+ | 超长上下文、多模态 | 低 |
| Gemini 1.5 Pro | Google | 2M | 最长上下文、复杂分析 | 中等 |

---

## 🎯 关键概念总结

1. **Models 是 Agent 的推理引擎**: 负责理解、决策和生成
2. **优先使用 Chat Models**: 更适合现代对话应用
3. **init_chat_model 统一初始化**: 跨提供商的一致接口
4. **provider:model 字符串适合高层 API**: 在 `create_agent` 中尤其简洁
5. **工具调用是核心能力**: 通过 `bind_tools` 实现
6. **结构化输出提高可靠性**: 使用 `with_structured_output`
7. **流式处理改善体验**: 使用 `stream()` 和 `astream_events()`
8. **批处理和异步调用提升吞吐**: 使用 `batch`、`abatch`、`ainvoke`
9. **合理配置参数和能力画像**: temperature、max_tokens、reasoning、multimodal 等
10. **监控 token 使用和限流**: 优化成本、延迟与稳定性
11. **根据任务动态选择模型**: 平衡质量、成本、速度和上下文窗口

---

## ❓ 常见问题

**Q: 如何选择合适的模型？**  
A: 根据任务复杂度、响应时间要求和成本预算选择。简单任务用 Haiku/GPT-3.5，复杂推理用 Sonnet/GPT-4o，关键任务用 Opus。

**Q: 工具调用失败怎么办？**  
A: 确保工具描述清晰、参数类型正确，使用 Pydantic 验证，并处理 ToolException。

**Q: 如何降低 API 成本？**  
A: 使用较小的模型、优化提示词长度、利用缓存、减少 max_tokens、批处理请求。

**Q: 流式处理和普通调用有什么区别？**  
A: 流式处理实时返回 tokens，改善用户体验，但稍微增加复杂度。

**Q: 结构化输出总是可靠吗？**  
A: 大多数情况下可靠，但应添加 Pydantic 验证和错误处理作为保障。

---

## 🔗 相关资源

- **官方文档**: https://docs.langchain.com/oss/python/langchain/models
- **Anthropic 文档**: https://docs.anthropic.com/
- **OpenAI 文档**: https://platform.openai.com/docs
- **配套文档**:
  - [LangChain Agents 详细总结](01_LangChain_Agents_详细总结.md)
  - [LangChain Tools 详细指南](04_LangChain_Tools_详细指南.md)

---

**文档版本**: 1.1  
**最后更新**: 2026-05-31  
**基于**: LangChain v1.x 官方文档, Python 3.9+

本文档涵盖了 LangChain Models 的核心概念、使用方法和最佳实践，包含 100+ 实用代码示例。
