# LangChain Structured Output 详细指南

> 基于官方文档 https://docs.langchain.com/oss/python/langchain/structured-output 的完整中文总结（在原有内容基础上补充当前 v1.x 文档中的 Response format、ProviderStrategy/ToolStrategy 自动选择、tool_message_content、错误处理与重试等内容）

---

## 目录

1. [概述](#概述)
2. [为什么需要 Structured Output](#为什么需要-structured-output)
3. [核心概念](#核心概念)
4. [在 Model 中使用 Structured Output](#在-model-中使用-structured-output)
5. [在 Agent 中使用 Structured Output](#在-agent-中使用-structured-output)
5.1. [Response format 与策略自动选择](#response-format-与策略自动选择)
6. [Schema 类型](#schema-类型)
7. [生成策略](#生成策略)
7.1. [自定义 Tool Message Content](#自定义-tool-message-content)
7.2. [错误处理与重试](#错误处理与重试)
8. [高级特性](#高级特性)
9. [实际应用场景](#实际应用场景)
10. [最佳实践](#最佳实践)
11. [常见问题](#常见问题)
12. [快速参考](#快速参考)

---

## 概述

Structured Output（结构化输出）允许 LLM 返回**特定、可预测格式**的数据，而不是自然语言文本。你可以获得经过验证的结构化数据（JSON 对象、Pydantic 模型、数据类），可以直接在应用程序中使用。

**传统方式 vs Structured Output：**

```python
# ❌ 传统方式：需要解析自然语言
response = model.invoke("提取联系信息：John Doe, john@example.com, (555) 123-4567")
print(response.content)
# "联系人姓名是 John Doe，邮箱是 john@example.com..."
# 需要编写复杂的解析逻辑 😫

# ✅ Structured Output：直接获得结构化数据
class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

model_with_structure = model.with_structured_output(ContactInfo)
response = model_with_structure.invoke("提取联系信息：John Doe, john@example.com, (555) 123-4567")
print(response)
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
# 直接使用！😊
```

---

## 为什么需要 Structured Output

### 1. **可靠的数据提取**

传统的文本解析容易出错：
- 格式变化导致解析失败
- 需要编写复杂的正则表达式
- 边缘情况难以处理

Structured Output 保证输出符合 schema：
```python
# 保证总是获得正确的数据结构
response: ContactInfo  # 类型安全！
print(response.email)  # 总是有效的
```

### 2. **与下游系统集成**

当需要将 LLM 输出传递给其他系统时：
```python
# 直接存入数据库
db.insert(response.dict())

# 直接调用 API
api.create_contact(
    name=response.name,
    email=response.email,
    phone=response.phone
)

# 直接序列化
json_data = response.json()
```

### 3. **类型安全和验证**

Pydantic 提供自动验证：
```python
from pydantic import BaseModel, Field, EmailStr, validator

class ContactInfo(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr  # 自动验证邮箱格式
    phone: str = Field(..., pattern=r'^\(\d{3}\) \d{3}-\d{4}$')
    
    @validator('name')
    def name_must_be_capitalized(cls, v):
        if not v[0].isupper():
            raise ValueError('名字必须大写开头')
        return v
```

### 4. **更好的开发体验**

- IDE 自动补全
- 类型检查
- 清晰的 API 契约
- 易于测试

---

## 核心概念

### Structured Output vs Tools

| 特性 | Structured Output | Tools |
|------|-------------------|-------|
| 选择性 | 总是响应这个格式 | LLM 可以选择调用或不调用 |
| 数量 | 只生成一个响应 | 可以选择多个工具 |
| 用途 | 数据提取、格式化输出 | 执行动作、获取外部数据 |
| 实现 | 可能使用工具调用底层实现 | 独立的功能调用 |

### Schema 和 Method

**Schema** - 定义输出的结构：
- Pydantic Model (Python)——Python 对象（重点）
- TypedDict (Python)
- JSON Schema (通用)
- Zod Schema (JavaScript)

**Method** - 生成结构化输出的方式：
- `json_schema` - 提供商原生支持（最可靠）—— 通过工具调用实现
- `function_calling` - 通过工具调用实现
- `json_mode` - 生成有效 JSON（需在 prompt 中描述 schema）

---

## 在 Model 中使用 Structured Output

### 方式 1: 使用 Pydantic Model（推荐）

```python
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

class Movie(BaseModel):
    """电影详情"""
    title: str = Field(..., description="电影标题")
    year: int = Field(..., description="上映年份")
    director: str = Field(..., description="导演")
    rating: float = Field(..., description="评分（满分 10）")

model = init_chat_model("glm-4.6")
model_with_structure = model.with_structured_output(Movie)

response = model_with_structure.invoke("提供《盗梦空间》的详细信息")
print(response)
# Movie(title="Inception", year=2010, director="Christopher Nolan", rating=8.8)

# 直接访问字段
print(f"标题: {response.title}")
print(f"年份: {response.year}")
```

**优势：**
- ✅ 自动类型验证
- ✅ 字段描述和约束
- ✅ 嵌套结构支持
- ✅ IDE 自动补全

### 方式 2: 使用 TypedDict

```python
from typing_extensions import Annotated, TypedDict

class MovieDict(TypedDict):
    """电影详情"""
    title: Annotated[str, ..., "电影标题"]
    year: Annotated[int, ..., "上映年份"]
    director: Annotated[str, ..., "导演"]
    rating: Annotated[float, ..., "评分（满分 10）"]

model_with_structure = model.with_structured_output(MovieDict)
response = model_with_structure.invoke("提供《盗梦空间》的详细信息")
print(response)
# {'title': 'Inception', 'year': 2010, 'director': 'Christopher Nolan', 'rating': 8.8}
```

**使用场景：**
- 不需要运行时验证
- 更简单的用例
- 与现有代码集成

### 方式 3: 使用 JSON Schema

```python
json_schema = {
    "title": "Movie",
    "description": "电影详情",
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "电影标题"
        },
        "year": {
            "type": "integer",
            "description": "上映年份"
        },
        "director": {
            "type": "string",
            "description": "导演"
        },
        "rating": {
            "type": "number",
            "description": "评分（满分 10）"
        }
    },
    "required": ["title", "year", "director", "rating"]
}

model_with_structure = model.with_structured_output(
    json_schema,
    method="json_schema"
)

response = model_with_structure.invoke("提供《盗梦空间》的详细信息")
print(response)
# {'title': 'Inception', 'year': 2010, ...}
```

**使用场景：**
- 最大控制和互操作性
- 需要手动验证
- 跨语言使用

### 嵌套结构

```python
from pydantic import BaseModel
from typing import List, Optional

class Actor(BaseModel):
    """演员信息"""
    name: str
    role: str

class MovieDetails(BaseModel):
    """详细电影信息"""
    title: str
    year: int
    cast: List[Actor]  # 嵌套列表
    genres: List[str]
    budget: Optional[float] = Field(None, description="预算（百万美元）")

model_with_structure = model.with_structured_output(MovieDetails)

response = model_with_structure.invoke("提供《盗梦空间》的完整信息，包括演员阵容")
print(response)
# MovieDetails(
#     title="Inception",
#     year=2010,
#     cast=[
#         Actor(name="Leonardo DiCaprio", role="Dom Cobb"),
#         Actor(name="Joseph Gordon-Levitt", role="Arthur"),
#         ...
#     ],
#     genres=["Sci-Fi", "Thriller"],
#     budget=160.0
# )
```

### 获取原始响应

使用 `include_raw=True` 同时获取解析后的数据和原始 AIMessage：

```python
model_with_structure = model.with_structured_output(
    Movie,
    include_raw=True
)

response = model_with_structure.invoke("提供《盗梦空间》的详细信息")
print(response)
# {
#     'raw': AIMessage(content='...', usage_metadata={...}),
#     'parsed': Movie(title="Inception", ...)
# }

# 访问 token 使用情况
print(response['raw'].usage_metadata)
# {'input_tokens': 42, 'output_tokens': 28, ...}

# 访问解析后的数据
movie = response['parsed']
print(movie.title)
```

---

## 在 Agent 中使用 Structured Output

LangChain 的 `create_agent` 自动处理结构化输出。

### 基本用法

```python
from langchain.agents import create_agent
from pydantic import BaseModel

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model="glm-4.6",
    tools=[search_tool],
    response_format=ContactInfo  # ⚠️ v1.0 后需要使用策略
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "提取联系信息：John Doe, john@example.com, (555) 123-4567"}
    ]
})

# 结构化响应在 'structured_response' 键中
print(result["structured_response"])
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

### 工作流程

1. Agent 运行模型/工具调用循环
2. 当模型完成工具调用后
3. 最终响应被强制转换为提供的格式
4. 验证并返回在 `structured_response` 中

**关键优势：**
- ✅ 在主循环中生成（无额外 LLM 调用）
- ✅ 降低成本
- ✅ 自动验证

---

## Response format 与策略自动选择

`create_agent(response_format=...)` 当前接受四种形态：

| `response_format` | 含义 |
|------------------|------|
| `ProviderStrategy[Schema]` | 使用 provider 原生结构化输出能力 |
| `ToolStrategy[Schema]` | 使用工具调用回退实现结构化输出 |
| `type[Schema]`（如 `ContactInfo`） | 自动选择策略；模型支持原生时用 ProviderStrategy，否则用 ToolStrategy |
| `None` | 不显式要求结构化输出 |

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent

class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="Person's name")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")

# 直接传 schema：自动选择最佳策略
agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    response_format=ContactInfo,
)
```

> 当前文档说明：从 `langchain>=1.1` 起，原生结构化输出能力会从模型 profile 中动态读取；如果 profile 不可用，可以手动指定。如果同时使用 tools 与 structured output，模型必须支持二者并存。

最终结构化结果保存在 Agent 状态的 `structured_response` 键下：

```python
result = agent.invoke({
    "messages": [{"role": "user", "content": "提取联系：John Doe, john@example.com, (555) 123-4567"}]
})
print(result["structured_response"])
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

### 何时手动指定策略？

- **强制使用 provider 原生结构化输出**：`response_format=ProviderStrategy(Schema)`
- **强制使用工具调用回退**：`response_format=ToolStrategy(Schema)`
- **模型 profile 不准确或希望明确语义**：手动指定，避免自动判断带来的不确定性
- **需要自定义错误处理或 tool message**：用 `ToolStrategy(...)` 并配置 `handle_errors`、`tool_message_content`

---

## Schema 类型

### Python: Pydantic Model

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum

class Priority(str, Enum):
    """优先级枚举"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class Task(BaseModel):
    """任务信息"""
    title: str = Field(..., min_length=1, max_length=200, description="任务标题")
    description: Optional[str] = Field(None, description="任务描述")
    priority: Priority = Field(default=Priority.MEDIUM, description="优先级")
    tags: List[str] = Field(default_factory=list, description="标签列表")
    estimated_hours: float = Field(..., gt=0, le=1000, description="预估工时")
    
    @validator('tags')
    def validate_tags(cls, v):
        """验证标签"""
        if len(v) > 10:
            raise ValueError('标签不能超过 10 个')
        return [tag.lower() for tag in v]  # 统一转小写

model_with_structure = model.with_structured_output(Task)
```

**特性：**
- 字段验证（min_length, max_length, gt, le 等）
- 自定义验证器
- 默认值
- 枚举类型
- 可选字段

### Python: TypedDict

```python
from typing_extensions import Annotated, TypedDict, NotRequired

class TaskDict(TypedDict):
    title: Annotated[str, ..., "任务标题"]
    description: NotRequired[Annotated[str, ..., "任务描述"]]  # 可选字段
    priority: Annotated[str, ..., "优先级：high/medium/low"]
    tags: Annotated[List[str], ..., "标签列表"]
    estimated_hours: Annotated[float, ..., "预估工时"]

model_with_structure = model.with_structured_output(TaskDict)
```

### JavaScript: Zod Schema

```javascript
import * as z from "zod";

const Movie = z.object({
  title: z.string().describe("电影标题"),
  year: z.number().describe("上映年份"),
  director: z.string().describe("导演"),
  rating: z.number().describe("评分（满分 10）"),
});

const modelWithStructure = model.withStructuredOutput(Movie);

const response = await modelWithStructure.invoke("提供《盗梦空间》的详细信息");
console.log(response);
// {
//   title: "Inception",
//   year: 2010,
//   director: "Christopher Nolan",
//   rating: 8.8
// }
```

**Zod 特性：**
- 自动验证
- 类型推断
- 丰富的验证方法

### JavaScript: JSON Schema

```javascript
const jsonSchema = {
  "title": "Movie",
  "description": "电影详情",
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "year": {"type": "integer"},
    "director": {"type": "string"},
    "rating": {"type": "number"}
  },
  "required": ["title", "year", "director", "rating"]
};

const modelWithStructure = model.withStructuredOutput(
  jsonSchema,
  { method: "jsonSchema" }
);
```

---

## 生成策略

LangChain v1.0 引入了两种策略来生成结构化输出。

### 1. ToolStrategy - 人工工具调用

通过强制工具调用来生成结构化输出，适用于**任何支持工具调用的模型**。

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel

class Weather(BaseModel):
    temperature: float
    condition: str

agent = create_agent(
    model="glm-4.6",
    tools=[weather_tool],
    response_format=ToolStrategy(Weather)  # 使用 ToolStrategy
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "旧金山的天气如何？"}]
})

print(result["structured_response"])
# Weather(temperature=70.0, condition='sunny')
```

**优势：**
- ✅ 兼容所有支持工具调用的模型
- ✅ 更广泛的模型支持

**劣势：**
- ❌ 可能不如原生支持可靠

### 2. ProviderStrategy - 提供商原生支持

使用模型提供商的原生结构化输出功能（如 OpenAI 的 Structured Outputs）。

```python
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="gpt-4o",
    tools=[weather_tool],
    response_format=ProviderStrategy(Weather)  # 使用 ProviderStrategy
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "旧金山的天气如何？"}]
})

print(result["structured_response"])
# Weather(temperature=70.0, condition='sunny')
```

**优势：**
- ✅ 更可靠
- ✅ 通常性能更好
- ✅ 严格模式支持（如 OpenAI）

**劣势：**
- ❌ 只支持特定提供商（OpenAI, Anthropic 等）

### 选择策略的建议

```python
# 如果模型支持原生结构化输出（如 GPT-4, Claude）
response_format = ProviderStrategy(Schema)  # ✅ 推荐

# 如果模型只支持工具调用
response_format = ToolStrategy(Schema)  # ✅ 回退选项

# 直接传入 schema，由 LangChain 自动选择策略（推荐）
response_format = Schema  # ✅ v1.1+：模型支持原生 → ProviderStrategy；否则 → ToolStrategy
```

### 错误处理

使用 `handle_errors` 参数控制错误处理：

```python
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model="glm-4.6",
    tools=[],
    response_format=ToolStrategy(
        Weather,
        handle_errors="raise"  # "raise" | "return_none" | "return_partial"
    )
)
```

**错误类型：**
1. **解析错误** - 模型生成的数据不匹配 schema
2. **多次工具调用** - 模型为结构化输出 schema 生成多个工具调用

**处理选项：**
- `raise` - 抛出异常（默认）
- `return_none` - 返回 None
- `return_partial` - 返回部分解析的数据

---

## 自定义 Tool Message Content

使用 `ToolStrategy` 时，模型会以工具调用形式生成结构化结果，结构化数据会附带一条 `ToolMessage`。`tool_message_content` 让你自定义这条 `ToolMessage` 的内容，便于在对话历史中提供更友好的语义。

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class MeetingAction(BaseModel):
    """Action items extracted from a meeting transcript."""
    task: str = Field(description="The specific task to be completed")
    assignee: str = Field(description="Person responsible for the task")
    priority: Literal["low", "medium", "high"] = Field(description="Priority level")

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[],
    response_format=ToolStrategy(
        schema=MeetingAction,
        tool_message_content="Action item captured and added to meeting notes!",
    ),
)

agent.invoke({
    "messages": [{
        "role": "user",
        "content": "From our meeting: Sarah needs to update the project timeline as soon as possible",
    }]
})
```

如果不指定 `tool_message_content`，默认 `ToolMessage` 内容形如 `Returning structured response: {...}`。在对话日志、UI 展示或多 Agent 协作场景中，自定义这段文本可以让历史更清晰。

---

## 错误处理与重试

`ToolStrategy` 内置了模型生成错误的智能重试机制，由 `handle_errors` 参数控制。

```python
class ToolStrategy(Generic[SchemaT]):
    schema: type[SchemaT]
    tool_message_content: str | None
    handle_errors: Union[
        bool,
        str,
        type[Exception],
        tuple[type[Exception], ...],
        Callable[[Exception], str],
    ]
```

### 1. `handle_errors` 选项

| 取值 | 行为 |
|-----|------|
| `True`（默认） | 捕获所有错误，使用默认错误模板，要求模型重试 |
| `str` | 捕获所有错误，使用自定义错误消息 |
| `type[Exception]` | 仅捕获指定异常类型 |
| `tuple[type[Exception], ...]` | 仅捕获多个异常类型 |
| `Callable[[Exception], str]` | 自定义错误信息生成函数 |
| `False` | 不重试，让异常向上抛出 |

### 2. 多次结构化输出错误（Multiple structured outputs error）

当模型在一次响应中错误地调用了多个结构化输出工具时，Agent 会在工具消息中追加错误说明并要求模型重新调用：

```python
from pydantic import BaseModel, Field
from typing import Union
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ContactInfo(BaseModel):
    name: str = Field(description="Person's name")
    email: str = Field(description="Email address")

class EventDetails(BaseModel):
    event_name: str = Field(description="Name of the event")
    date: str = Field(description="Event date")

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[],
    response_format=ToolStrategy(Union[ContactInfo, EventDetails]),
)
```

模型如果同时返回 `ContactInfo` 和 `EventDetails`，工具消息会提示：

```
Error: Model incorrectly returned multiple structured responses (ContactInfo, EventDetails) when only one is expected.
Please fix your mistakes.
```

模型会基于这条提示重试，最终只输出一个 schema。

### 3. Schema 验证错误（Schema validation error）

当模型生成的字段不满足 schema（例如 `rating=10` 不在 `1..=5`），Agent 会把 Pydantic 验证错误转换成结构化的工具反馈，让模型修正：

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ProductRating(BaseModel):
    rating: int | None = Field(description="Rating from 1-5", ge=1, le=5)
    comment: str = Field(description="Review comment")

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[],
    response_format=ToolStrategy(ProductRating),
    system_prompt="你是一个解析产品评论的助手，禁止编造字段或数值。",
)
```

如果模型返回 `rating=10`，工具消息会包含 Pydantic 校验失败的详细原因，模型会被引导重新生成符合约束的输出。

### 4. 自定义错误处理

```python
from langchain.agents.structured_output import ToolStrategy

def to_user_message(exc: Exception) -> str:
    return f"结构化输出失败：{exc}. 请重新输出符合 schema 的 JSON。"

response_format = ToolStrategy(
    schema=ProductRating,
    handle_errors=to_user_message,
)
```

也可以选择只捕获部分异常类型：

```python
from pydantic import ValidationError

response_format = ToolStrategy(
    schema=ProductRating,
    handle_errors=ValidationError,  # 只捕获 schema 校验错误
)
```

或者完全关闭重试，让上层应用自己处理：

```python
response_format = ToolStrategy(
    schema=ProductRating,
    handle_errors=False,
)
```

### 5. 何时关闭自动重试？

- 调试期希望直接看到原始错误
- 业务上要求结构化错误必须由上层流程处理
- 想要把错误送到外部告警/监控

> 默认情况下保持 `handle_errors=True` 即可获得稳定的结构化输出体验；只有在需要更强可观测性或自定义重试逻辑时才修改。

---

## 高级特性

### 1. 动态响应格式选择

根据对话状态、用户偏好或上下文动态选择格式。

#### 基于对话状态

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from pydantic import BaseModel, Field
from typing import Callable

class SimpleResponse(BaseModel):
    """简单响应（早期对话）"""
    answer: str = Field(description="简短答案")

class DetailedResponse(BaseModel):
    """详细响应（深入对话）"""
    answer: str = Field(description="详细答案")
    reasoning: str = Field(description="推理过程")
    confidence: float = Field(description="置信度 0-1")

@wrap_model_call
def state_based_output(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """根据对话历史选择格式"""
    message_count = len(request.messages)
    
    if message_count < 3:
        # 早期对话 - 简单格式
        request = request.override(response_format=SimpleResponse)
    else:
        # 深入对话 - 详细格式
        request = request.override(response_format=DetailedResponse)
    
    return handler(request)

agent = create_agent(
    model="glm-4.6",
    tools=[],
    middleware=[state_based_output]
)
```

#### 基于用户角色

```python
from dataclasses import dataclass

@dataclass
class Context:
    user_role: str  # "admin" | "user"
    environment: str  # "production" | "development"

class AdminResponse(BaseModel):
    answer: str
    debug_info: dict
    system_status: str

class UserResponse(BaseModel):
    answer: str

@wrap_model_call
def context_based_output(request: ModelRequest, handler):
    """根据用户角色选择格式"""
    user_role = request.runtime.context.user_role
    environment = request.runtime.context.environment
    
    if user_role == "admin" and environment == "production":
        response_format = AdminResponse
    else:
        response_format = UserResponse
    
    request = request.override(response_format=response_format)
    return handler(request)

agent = create_agent(
    model="glm-4.6",
    middleware=[context_based_output],
    context_schema=Context
)

# 管理员调用
admin_result = agent.invoke(
    {"messages": [{"role": "user", "content": "系统状态？"}]},
    context=Context(user_role="admin", environment="production")
)
```

### 2. 多个响应格式

某些场景需要支持多种输出格式。

```python
from typing import Union

# 定义多个格式
class ShortAnswer(BaseModel):
    answer: str

class DetailedAnswer(BaseModel):
    answer: str
    explanation: str
    sources: List[str]

# Agent 可以选择返回哪种格式
agent = create_agent(
    model="glm-4.6",
    response_format=[ShortAnswer, DetailedAnswer]  # 多个格式
)
```

### 3. Strict Mode（严格模式）

OpenAI 支持严格模式，确保输出**严格**遵循 schema。

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="glm-4.6")

model_with_structure = model.with_structured_output(
    Weather,
    method="json_schema",
    strict=True  # 启用严格模式
)
```

**注意事项：**
- 某些 schema 特性在严格模式下不受支持
- 推理模型（如 o1）使用 `z.nullable()` 而不是 `z.optional()`

```javascript
// JavaScript - 不推荐
const schema = z.object({
  color: z.string().optional()  // ❌ 在 o1 模型中不工作
});

// JavaScript - 推荐
const schema = z.object({
  color: z.string().nullable()  // ✅ 正确方式
});
```

### 4. 与 Routing 结合

使用结构化输出进行路由决策。

```python
from typing_extensions import Literal
from pydantic import BaseModel, Field

class Route(BaseModel):
    """路由决策"""
    step: Literal["poem", "story", "joke"] = Field(
        description="下一步路由"
    )

# 创建路由器
router = model.with_structured_output(Route)

def route_request(user_input: str) -> str:
    """根据用户输入路由到不同节点"""
    decision = router.invoke([
        {"role": "system", "content": "根据用户请求路由到 story、joke 或 poem"},
        {"role": "user", "content": user_input}
    ])
    
    if decision.step == "story":
        return write_story(user_input)
    elif decision.step == "joke":
        return write_joke(user_input)
    elif decision.step == "poem":
        return write_poem(user_input)

# 使用
result = route_request("给我讲个笑话")
```

---

## 实际应用场景

### 场景 1：数据提取

从非结构化文本中提取结构化信息。

```python
from pydantic import BaseModel, EmailStr
from typing import List, Optional

class ExtractedInfo(BaseModel):
    """提取的联系信息"""
    name: str
    email: EmailStr
    phone: Optional[str] = None
    company: Optional[str] = None
    position: Optional[str] = None

model_with_structure = model.with_structured_output(ExtractedInfo)

# 从邮件签名提取信息
text = """
Best regards,
Jane Smith
Senior Software Engineer
Tech Corp Inc.
jane.smith@techcorp.com
+1 (555) 987-6543
"""

result = model_with_structure.invoke(f"从以下文本提取联系信息：\n{text}")
print(result)
# ExtractedInfo(
#     name="Jane Smith",
#     email="jane.smith@techcorp.com",
#     phone="+1 (555) 987-6543",
#     company="Tech Corp Inc.",
#     position="Senior Software Engineer"
# )

# 直接存入数据库
db.contacts.insert(result.dict())
```

### 场景 2：内容分类

```python
from enum import Enum
from pydantic import BaseModel, Field
from typing import List

class Category(str, Enum):
    TECH = "technology"
    BUSINESS = "business"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    POLITICS = "politics"

class Classification(BaseModel):
    """文章分类结果"""
    primary_category: Category = Field(description="主要分类")
    secondary_categories: List[Category] = Field(description="次要分类")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度")
    keywords: List[str] = Field(description="关键词")
    summary: str = Field(max_length=200, description="摘要")

classifier = model.with_structured_output(Classification)

article = "Apple announces new iPhone with AI features..."
result = classifier.invoke(f"分类以下文章：\n{article}")
print(result)
# Classification(
#     primary_category=Category.TECH,
#     secondary_categories=[Category.BUSINESS],
#     confidence=0.95,
#     keywords=["Apple", "iPhone", "AI"],
#     summary="Apple 发布具有 AI 功能的新款 iPhone..."
# )
```

### 场景 3：表单填充

```python
from pydantic import BaseModel, Field, validator
from datetime import date
from typing import Optional

class JobApplication(BaseModel):
    """职位申请表单"""
    first_name: str = Field(..., min_length=1)
    last_name: str = Field(..., min_length=1)
    email: EmailStr
    phone: str = Field(..., pattern=r'^\+?1?\d{9,15}$')
    position: str
    years_experience: int = Field(..., ge=0, le=50)
    start_date: Optional[date] = None
    cover_letter: str = Field(..., min_length=100, max_length=1000)
    
    @validator('start_date')
    def validate_start_date(cls, v):
        if v and v < date.today():
            raise ValueError('开始日期不能是过去')
        return v

form_filler = model.with_structured_output(JobApplication)

user_input = """
我叫 John Smith，邮箱是 john.smith@email.com，电话 +1234567890。
我想申请软件工程师职位，有 5 年经验，希望下个月开始。
我对这个职位非常感兴趣，因为...（详细 cover letter）
"""

application = form_filler.invoke(f"根据以下信息填写申请表：\n{user_input}")

# 自动验证通过，可以直接提交
submit_application(application)
```

### 场景 4：代码生成

```python
from pydantic import BaseModel, Field
from typing import List

class FunctionDefinition(BaseModel):
    """函数定义"""
    name: str = Field(description="函数名（snake_case）")
    description: str = Field(description="函数功能描述")
    parameters: List[dict] = Field(description="参数列表")
    return_type: str = Field(description="返回类型")
    code: str = Field(description="完整函数代码")
    test_cases: List[str] = Field(description="测试用例")

code_generator = model.with_structured_output(FunctionDefinition)

request = "创建一个函数，计算列表中所有偶数的和"
func_def = code_generator.invoke(request)

print(func_def.code)
# def sum_even_numbers(numbers: List[int]) -> int:
#     """计算列表中所有偶数的和"""
#     return sum(n for n in numbers if n % 2 == 0)

# 自动生成测试
for test in func_def.test_cases:
    print(test)
```

### 场景 5：评估和打分

```python
from pydantic import BaseModel, Field
from typing import Annotated

class EssayGrade(BaseModel):
    """作文评分"""
    explanation: Annotated[str, Field(description="评分理由")]
    grammar_score: Annotated[int, Field(ge=0, le=100, description="语法分数")]
    content_score: Annotated[int, Field(ge=0, le=100, description="内容分数")]
    structure_score: Annotated[int, Field(ge=0, le=100, description="结构分数")]
    overall_score: Annotated[int, Field(ge=0, le=100, description="总分")]
    feedback: Annotated[str, Field(description="改进建议")]

grader = model.with_structured_output(EssayGrade, method="json_schema", strict=True)

essay = "学生的作文内容..."
grade = grader.invoke(f"评分以下作文：\n{essay}")

print(f"总分: {grade.overall_score}")
print(f"反馈: {grade.feedback}")

# 存储评分
db.grades.insert(grade.dict())
```

---

## 最佳实践

### 1. **提供清晰的字段描述**

```python
# ❌ 不好：缺少描述
class Person(BaseModel):
    name: str
    age: int

# ✅ 好：清晰的描述
class Person(BaseModel):
    name: str = Field(description="全名（名和姓）")
    age: int = Field(description="年龄（整数）", ge=0, le=150)
```

### 2. **使用验证器**

```python
from pydantic import BaseModel, Field, validator

class Product(BaseModel):
    name: str
    price: float = Field(gt=0, description="价格必须大于 0")
    quantity: int = Field(ge=0, description="库存数量")
    
    @validator('price')
    def round_price(cls, v):
        """价格保留两位小数"""
        return round(v, 2)
    
    @validator('quantity')
    def check_stock(cls, v):
        """检查库存"""
        if v == 0:
            # 警告但允许
            print("警告：库存为 0")
        return v
```

### 3. **使用枚举限制选项**

```python
from enum import Enum

class Status(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class Task(BaseModel):
    title: str
    status: Status  # 只允许这些值
```

### 4. **合理使用可选字段**

```python
from typing import Optional

class User(BaseModel):
    # 必填字段
    username: str
    email: EmailStr
    
    # 可选字段
    full_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
```

### 5. **嵌套结构要适度**

```python
# ✅ 好：适度嵌套
class Address(BaseModel):
    street: str
    city: str
    country: str

class Company(BaseModel):
    name: str
    address: Address  # 一层嵌套

# ❌ 避免：过度嵌套
class A(BaseModel):
    b: 'B'

class B(BaseModel):
    c: 'C'

class C(BaseModel):
    d: 'D'  # 多层嵌套，LLM 可能难以正确生成
```

### 6. **处理错误**

```python
from pydantic import ValidationError

try:
    result = model_with_structure.invoke(user_input)
except ValidationError as e:
    print("验证失败:")
    for error in e.errors():
        print(f"  字段: {error['loc']}")
        print(f"  错误: {error['msg']}")
        print(f"  类型: {error['type']}")
```

### 7. **测试你的 Schema**

```python
import pytest

def test_contact_info_schema():
    """测试 ContactInfo schema"""
    # 有效数据
    valid_data = {
        "name": "John Doe",
        "email": "john@example.com",
        "phone": "(555) 123-4567"
    }
    contact = ContactInfo(**valid_data)
    assert contact.name == "John Doe"
    
    # 无效数据
    with pytest.raises(ValidationError):
        ContactInfo(name="", email="invalid-email", phone="123")
```

### 8. **文档化你的 Schema**

```python
class ComplexSchema(BaseModel):
    """
    复杂数据结构
    
    用于从非结构化文本中提取结构化数据。
    
    示例:
        >>> schema = ComplexSchema(
        ...     field1="value1",
        ...     field2=123
        ... )
    """
    field1: str = Field(description="字段 1 的描述")
    field2: int = Field(description="字段 2 的描述")
    
    class Config:
        schema_extra = {
            "examples": [
                {
                    "field1": "示例值",
                    "field2": 42
                }
            ]
        }
```

---

## 常见问题

### 1. 为什么模型没有遵循 schema？

**可能原因：**
- Schema 太复杂
- 字段描述不清晰
- 模型不支持该 method

**解决方案：**
```python
# ✅ 简化 schema
# ✅ 添加清晰的描述
# ✅ 使用 ProviderStrategy（如果支持）
# ✅ 尝试更强大的模型
```

### 2. 如何处理可选字段？

```python
from typing import Optional
from pydantic import Field

class Schema(BaseModel):
    required_field: str
    optional_field: Optional[str] = None  # 可以是 None
    optional_with_default: str = "默认值"  # 有默认值
```

### 3. 如何在 JavaScript 中使用？

```javascript
import * as z from "zod";
import { ChatOpenAI } from "@langchain/openai";

const ContactInfo = z.object({
  name: z.string(),
  email: z.string().email(),
  phone: z.string()
});

const model = new ChatOpenAI({ model: "glm-4.6" });
const modelWithStructure = model.withStructuredOutput(ContactInfo);

const result = await modelWithStructure.invoke(
  "Extract: John Doe, john@example.com, (555) 123-4567"
);
console.log(result);
```

### 4. 性能优化

```python
# ✅ 使用 ProviderStrategy（更快更可靠）
response_format = ProviderStrategy(Schema)

# ✅ 缓存模型实例
model_with_structure = model.with_structured_output(Schema)  # 只创建一次

# ✅ 简化 schema（减少 tokens）
# ✅ 批量处理
```

### 5. 如何调试？

```python
# 1. 使用 include_raw 查看原始响应
result = model_with_structure.invoke(input, include_raw=True)
print("原始响应:", result['raw'].content)
print("解析结果:", result['parsed'])

# 2. 检查 token 使用
print("Tokens:", result['raw'].usage_metadata)

# 3. 尝试不同的 method
for method in ["json_schema", "function_calling", "json_mode"]:
    try:
        model_test = model.with_structured_output(Schema, method=method)
        result = model_test.invoke(input)
        print(f"{method}: 成功")
    except Exception as e:
        print(f"{method}: 失败 - {e}")
```

---

## 快速参考

### Model 使用

```python
# Pydantic
model.with_structured_output(Schema)

# TypedDict
model.with_structured_output(TypedDictSchema)

# JSON Schema
model.with_structured_output(json_schema, method="json_schema")

# 包含原始响应
model.with_structured_output(Schema, include_raw=True)

# 严格模式
model.with_structured_output(Schema, method="json_schema", strict=True)
```

### Agent 使用

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy

# ToolStrategy
agent = create_agent(
    model="glm-4.6",
    response_format=ToolStrategy(Schema)
)

# ProviderStrategy
agent = create_agent(
    model="gpt-4o",
    response_format=ProviderStrategy(Schema)
)

# 访问结果
result = agent.invoke({"messages": [...]})
structured_data = result["structured_response"]
```

### Schema 定义

```python
from pydantic import BaseModel, Field, EmailStr, validator
from typing import List, Optional
from enum import Enum

class MyEnum(str, Enum):
    OPTION1 = "option1"
    OPTION2 = "option2"

class NestedModel(BaseModel):
    field: str

class CompleteSchema(BaseModel):
    """Schema 描述"""
    
    # 基本类型
    string_field: str = Field(description="字符串字段")
    int_field: int = Field(ge=0, le=100, description="整数字段")
    float_field: float = Field(gt=0.0, description="浮点字段")
    bool_field: bool = Field(description="布尔字段")
    
    # 复杂类型
    list_field: List[str] = Field(description="列表字段")
    optional_field: Optional[str] = None
    enum_field: MyEnum = Field(description="枚举字段")
    
    # 嵌套
    nested: NestedModel
    
    # 验证
    @validator('string_field')
    def validate_string(cls, v):
        return v.strip().lower()
```

### Method 对比

| Method | 描述 | 支持提供商 | 可靠性 |
|--------|------|-----------|--------|
| `json_schema` | 原生支持 | OpenAI, Anthropic | ⭐⭐⭐⭐⭐ |
| `function_calling` | 工具调用 | 大多数模型 | ⭐⭐⭐⭐ |
| `json_mode` | JSON 模式 | 部分模型 | ⭐⭐⭐ |

### 常用模式

```python
# 数据提取
response = model.with_structured_output(ExtractedData).invoke(text)

# 分类
category = model.with_structured_output(Classification).invoke(text)

# 评分
grade = model.with_structured_output(Grade).invoke(essay)

# 路由
route = model.with_structured_output(RouteDecision).invoke(query)

# 表单填充
form = model.with_structured_output(FormSchema).invoke(user_input)
```

---

## 总结

LangChain Structured Output 提供了强大而灵活的方式来获取可预测、可验证的 LLM 输出：

**核心优势：**
✅ **类型安全** - 自动验证和类型检查  
✅ **易于集成** - 直接用于下游系统  
✅ **可靠输出** - 保证符合 schema  
✅ **丰富验证** - Pydantic 提供强大的验证能力  
✅ **多种选择** - 支持多种 schema 类型和生成方法  

**关键要点：**
- 使用 Pydantic Model 获得最佳体验（Python）
- 使用 Zod Schema 获得最佳体验（JavaScript）
- 在 Agent 中使用 `ToolStrategy` 或 `ProviderStrategy`
- 提供清晰的字段描述
- 使用验证器确保数据质量
- 处理验证错误
- 根据需求选择合适的 method

通过合理使用 Structured Output，你可以构建更可靠、更易维护的 LLM 应用程序！

---

## 相关资源

- 官方 Structured Output 文档：<https://docs.langchain.com/oss/python/langchain/structured-output>
- 配套文档：
  - [LangChain Models 详细指南](./LangChain_Models_详细指南.md)
  - [LangChain Agents 详细总结](./LangChain_Agents_详细总结.md)
  - [LangChain Tools 详细指南](./LangChain_Tools_详细指南.md)
  - [LangChain Messages 详细指南](./LangChain_Messages_详细指南.md)
  - [LangChain ShortTermMemory 详细指南](./LangChain_ShortTermMemory_详细指南.md)

---

**文档版本**: 1.1  
**最后更新**: 2026-06-01  
**基于**: LangChain v1.x 官方文档
