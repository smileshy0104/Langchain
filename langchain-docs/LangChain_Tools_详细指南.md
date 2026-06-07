# LangChain Tools 详细指南

> 基于官方文档 https://docs.langchain.com/oss/python/langchain/tools 的完整中文总结（在原有内容基础上补充当前 v1.x 文档中的上下文访问、状态/Store 注入、StreamWriter、工具返回 Command、动态工具选择、预构建工具与服务端工具等内容）

---

## 📋 目录

- [核心概念](#核心概念)
- [工具定义方式](#工具定义方式)
- [参数验证与Schema](#参数验证与schema)
- [访问上下文](#访问上下文)
- [错误处理](#错误处理)
- [异步工具](#异步工具)
- [流式输出](#流式输出)
- [特殊类型工具](#特殊类型工具)
- [工具集成](#工具集成)
- [高级用法](#高级用法)
- [官方 v1.x 补充：工具执行与返回值](#官方-v1x-补充工具执行与返回值)
- [最佳实践](#最佳实践)

---

## 核心概念

### 什么是 Tool？

**定义**: Tools（工具）是 Agent 或 Chat Model 调用以执行操作的组件。它们通过明确定义的输入和输出扩展模型能力，使模型能够查询数据库、调用 API、读写系统、检索知识库或执行业务动作。

**核心特征**:
- 封装一个可调用函数
- 定义输入 Schema（输入模式）
- 可以传递给兼容的聊天模型或 Agent
- 模型决定是否调用工具以及使用什么参数
- 工具可以访问运行时上下文、短期状态、长期 Store、stream writer 和执行配置
- 工具结果可以是字符串、对象，或用于更新图状态的 `Command`

### Tool 的作用

Tools 主要用于两种方式：

#### 1. 定义输入 Schema

将 "输入 Schema" 或 "参数 Schema" 传递给聊天模型的工具调用功能，使模型能够生成符合指定输入 Schema 的 "工具调用"。

```python
from langchain_core.tools import tool

@tool
def calculate(expression: str) -> float:
    """计算数学表达式"""
    return eval(expression)

# 工具的 Schema 会传递给模型
# 模型可以生成: {"name": "calculate", "args": {"expression": "10 * 5"}}
```

#### 2. 执行工具调用

接收模型生成的"工具调用"，采取某些操作并返回响应，该响应可以作为 `ToolMessage` 传递回模型。

```python
# 模型生成的工具调用
tool_call = {"name": "calculate", "args": {"expression": "10 * 5"}}

# 执行工具
result = calculate.invoke(tool_call["args"])

# 返回 ToolMessage
ToolMessage(content=str(result), tool_call_id="call_123")
```

### 工具的组成

```python
from langchain_core.tools import BaseTool

class MyTool(BaseTool):
    # 3 个必需属性
    name: str                    # 工具名称
    description: str             # 工具描述
    args_schema: Type[BaseModel] # 参数 Schema

    # 2 个必需方法
    def _run(self, *args, **kwargs):        # 同步执行
        pass

    async def _arun(self, *args, **kwargs): # 异步执行
        pass
```

---

## 工具定义方式

### 方式 1: @tool 装饰器（推荐）

**最简单的方式**，使用 `@tool` 装饰器定义工具。

#### 基础示例

```python
from langchain_core.tools import tool

@tool(name="search_database", description="搜索客户数据库以查找匹配查询的记录。")  # 自定义工具名称和描述
def search_database(query: str, limit: int = 10) -> str:
    """搜索客户数据库以查找匹配查询的记录。

    Args:
        query: 要查找的搜索词
        limit: 返回的最大结果数
    """
    return f"找到 {limit} 条关于 '{query}' 的结果"
```

**关键要点**:
- ✅ **类型提示是必需的** - 它们定义工具的输入 Schema
- ✅ **文档字符串** - 成为工具的描述，帮助模型理解何时使用它
- ✅ **函数名** - 成为工具名称
- ✅ **返回类型** - 定义输出类型

#### 使用工具

```python
# 1. 直接调用
result = search_database.invoke({"query": "张三", "limit": 5})
print(result)  # 输出: 找到 5 条关于 '张三' 的结果

# 2. 传递给 Agent
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[search_database],
    system_prompt="你是一个有帮助的助手"
)

# 3. 查看工具的 Schema（工具架构）
print(search_database.args_schema.schema())
# {
#     'type': 'object',
#     'properties': {
#         'query': {'type': 'string'},
#         'limit': {'type': 'integer', 'default': 10}
#     },
#     'required': ['query']
# }
```

#### 自定义工具名称和描述

默认情况下，函数名会作为工具名，docstring 会作为工具描述。你也可以显式指定：

```python
from langchain_core.tools import tool

@tool("web_search", description="Search the web for up-to-date information.")
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"
```

工具名称和描述会直接影响模型是否正确选择工具。建议：

- 名称短、明确、动词开头，例如 `search_docs`、`get_user`、`create_ticket`
- 描述说明“什么时候使用这个工具”
- 参数描述说明格式、限制和示例

#### 详细的文档字符串

```python
@tool
def send_email(to: str, subject: str, body: str, cc: list[str] = None) -> str:
    """
    发送电子邮件给指定收件人。

    此工具会发送电子邮件并返回发送状态。

    使用场景：
    - 向客户发送通知
    - 发送报告和摘要
    - 转发重要信息

    Args:
        to: 收件人邮箱地址，必须是有效的邮箱格式
        subject: 邮件主题，应简洁明了
        body: 邮件正文，支持 HTML 格式
        cc: 抄送列表，可选参数

    Returns:
        发送状态消息

    Examples:
        >>> send_email("user@example.com", "会议通知", "明天下午 2 点会议")
        "邮件已发送至 user@example.com"

    注意:
        - 请确保收件人地址正确
        - 敏感信息应加密发送
    """
    # 实现邮件发送逻辑
    if cc is None:
        cc = []

    email_service.send(to=to, subject=subject, body=body, cc=cc)
    return f"邮件已发送至 {to}"
```

### 方式 2: 使用 args_schema 参数

对于复杂的参数验证，可以使用 Pydantic 模型定义 `args_schema`。

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field, validator

## Advanced schema definition 高级架构定义（参数验证模型）
class DatabaseQueryInput(BaseModel):
    """数据库查询输入参数"""

    query: str = Field(
        description="SQL 查询语句",
        min_length=5,
        max_length=1000
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="最大返回行数"
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="跳过的行数"
    )

    @validator("query") # Pydantic 验证器
    def validate_query(cls, v):
        """验证查询安全性"""
        # 禁止危险操作
        forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE"]
        v_upper = v.upper()

        if any(word in v_upper for word in forbidden):
            raise ValueError("只允许 SELECT 查询")

        # 必须以 SELECT 开头
        if not v_upper.strip().startswith("SELECT"):
            raise ValueError("查询必须以 SELECT 开始")

        return v

## 工具定义，调用 query_database 工具，参数使用 DatabaseQueryInput 校验模型
@tool(args_schema=DatabaseQueryInput)
def query_database(query: str, limit: int = 100, offset: int = 0) -> list:
    """
    在数据库中执行只读查询。

    安全限制:
    - 仅允许 SELECT 语句
    - 最多返回 1000 行
    - 自动超时保护
    """
    results = db.execute(query).fetchmany(limit)
    return results[offset:]
```

### 保留参数名（Reserved argument names）

工具函数可以通过特殊参数名接收 LangChain 运行时注入的信息。这些参数通常不应暴露给模型作为普通输入。

常见保留/特殊参数包括：

| 参数/注解 | 用途 |
|----------|------|
| `config` / `RunnableConfig` | 访问调用配置、metadata、tags、configurable 等 |
| `InjectedState` | 注入 Agent/LangGraph 短期状态 |
| `InjectedStore` | 注入长期 Store |
| `ToolRuntime` | 统一访问 state、context、store、stream_writer、config、tool_call_id 等运行时信息 |

实践建议：业务参数尽量避免命名为 `config`、`runtime`、`state`、`store`，以免和运行时注入概念混淆。

### 方式 3: 继承 BaseTool 类

对于需要完全控制的复杂工具。

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type

# 自定义搜索工具输入模型
class SearchInput(BaseModel):
    """搜索工具输入"""
    query: str = Field(description="搜索查询")
    filters: dict = Field(default={}, description="过滤条件")

# 自定义搜索工具类，继承 BaseTool
class AdvancedSearchTool(BaseTool):
    """高级搜索工具"""

    name: str = "advanced_search"
    description: str = "在知识库中进行高级搜索，支持复杂过滤"
    args_schema: Type[BaseModel] = SearchInput

    # 可选：工具特定的配置
    api_key: str = Field(default="", description="API 密钥")
    max_results: int = Field(default=10, description="最大结果数")

    def _run(self, query: str, filters: dict = None) -> str:
        """同步执行搜索"""
        if filters is None:
            filters = {}

        # 实现搜索逻辑
        results = self._search_api(query, filters)
        return self._format_results(results)

    async def _arun(self, query: str, filters: dict = None) -> str:
        """异步执行搜索"""
        if filters is None:
            filters = {}

        results = await self._async_search_api(query, filters)
        return self._format_results(results)

    def _search_api(self, query: str, filters: dict) -> list:
        """调用搜索 API"""
        # 实现同步搜索
        pass

    async def _async_search_api(self, query: str, filters: dict) -> list:
        """调用异步搜索 API"""
        # 实现异步搜索
        pass

    def _format_results(self, results: list) -> str:
        """格式化结果"""
        return "\n".join([f"- {r['title']}: {r['snippet']}" for r in results])

# 使用
search_tool = AdvancedSearchTool(api_key="your-api-key", max_results=20)
```

---

## 参数验证与Schema

### Pydantic 验证

> 如果项目使用 Pydantic v2，推荐使用 `@field_validator` / `@model_validator`；本文原有示例中的 `@validator` / `@root_validator` 仍常见于旧项目或兼容模式。


#### 1. 字段验证

```python
from pydantic import BaseModel, Field, validator
from typing import Literal
# 邮件输入参数校验
class EmailInput(BaseModel):
    """邮件输入参数"""

    to: str = Field(
        description="收件人邮箱",
        pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    )
    subject: str = Field(
        description="邮件主题",
        min_length=1,
        max_length=200
    )
    body: str = Field(
        description="邮件正文",
        min_length=1
    )
    priority: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="优先级"
    )

    @validator("to")
    def validate_email(cls, v):
        """验证邮箱域名"""
        allowed_domains = ["example.com", "company.com"]
        domain = v.split("@")[1]

        if domain not in allowed_domains:
            raise ValueError(f"只允许发送至: {', '.join(allowed_domains)}")

        return v

    @validator("body")
    def sanitize_body(cls, v):
        """清理邮件正文"""
        # 移除潜在的脚本注入
        import re
        v = re.sub(r'<script.*?</script>', '', v, flags=re.DOTALL)
        return v

## 创建工具 使用参数验证模型
@tool(args_schema=EmailInput)
def send_email(to: str, subject: str, body: str, priority: str = "medium") -> str:
    """发送邮件"""
    # 实现邮件发送
    return f"邮件已发送至 {to}"
```

#### 2. 模型级验证

```python
from pydantic import BaseModel, Field, root_validator

# 转账输入参数校验BaseModel
class TransferInput(BaseModel):
    """转账输入参数"""

    from_account: str = Field(description="源账户")
    to_account: str = Field(description="目标账户")
    amount: float = Field(gt=0, description="转账金额")
    currency: str = Field(default="CNY", description="货币")

    @root_validator
    def validate_transfer(cls, values):
        """验证转账请求"""
        from_account = values.get("from_account")
        to_account = values.get("to_account")
        amount = values.get("amount")

        # 防止自转
        if from_account == to_account:
            raise ValueError("不能转账到相同账户")

        # 金额限制
        if amount > 50000:
            raise ValueError("单笔转账不能超过 50,000")

        return values

# 调用工具 使用参数验证模型args_schema
@tool(args_schema=TransferInput)
def transfer_money(from_account: str, to_account: str, amount: float, currency: str = "CNY") -> str:
    """执行转账"""
    # 实现转账逻辑
    return f"已从 {from_account} 转账 {amount} {currency} 至 {to_account}"
```

### 复杂类型支持

#### 1. 列表和字典

```python
from typing import List, Dict, Optional

class DataProcessInput(BaseModel):
    """数据处理输入"""

    records: List[Dict[str, any]] = Field(
        description="要处理的记录列表"
    )
    operations: List[str] = Field(
        description="要执行的操作列表",
        min_items=1
    )
    config: Optional[Dict[str, any]] = Field(
        default=None,
        description="可选配置"
    )

# 调用工具DataProcessInput 使用参数验证模型args_schema
@tool(args_schema=DataProcessInput)
def process_data(
    records: List[Dict[str, any]],
    operations: List[str],
    config: Optional[Dict[str, any]] = None
) -> str:
    """批量处理数据记录"""
    processed = 0
    for record in records:
        for operation in operations:
            # 执行操作
            processed += 1

    return f"已处理 {processed} 条记录，执行了 {len(operations)} 个操作"
```

#### 2. 嵌套模型

```python
class Address(BaseModel):
    """地址信息"""
    street: str
    city: str
    country: str
    postal_code: str

class Person(BaseModel):
    """人员信息"""
    name: str = Field(min_length=1)
    age: int = Field(ge=0, le=150)
    email: str
    address: Address  # 嵌套模型

class CreateUserInput(BaseModel):
    """创建用户输入"""
    person: Person
    role: Literal["admin", "user", "guest"] = "user"
    active: bool = True

# 调用工具CreateUserInput 使用参数验证模型args_schema
@tool(args_schema=CreateUserInput)
def create_user(person: Person, role: str = "user", active: bool = True) -> str:
    """在系统中创建新用户"""
    user_data = {
        "name": person.name,
        "age": person.age,
        "email": person.email,
        "address": {
            "street": person.address.street,
            "city": person.address.city,
            "country": person.address.country,
            "postal_code": person.address.postal_code
        },
        "role": role,
        "active": active
    }

    # 保存用户
    user_id = database.create_user(user_data)
    return f"用户 {person.name} 已创建，ID: {user_id}"
```

#### 3. Union 类型

```python
from typing import Union

class TextContent(BaseModel):
    """文本内容"""
    type: Literal["text"] = "text"
    text: str

class ImageContent(BaseModel):
    """图像内容"""
    type: Literal["image"] = "image"
    url: str
    caption: Optional[str] = None

class VideoContent(BaseModel):
    """视频内容"""
    type: Literal["video"] = "video"
    url: str
    duration: int  # 秒

class PublishInput(BaseModel):
    """发布内容输入"""
    content: Union[TextContent, ImageContent, VideoContent] = Field(
        description="要发布的内容",
        discriminator="type"  # 使用 type 字段区分
    )
    platform: str = Field(description="发布平台")

# 调用工具PublishInput 使用参数验证模型args_schema
@tool(args_schema=PublishInput)
def publish_content(
    content: Union[TextContent, ImageContent, VideoContent],
    platform: str
) -> str:
    """发布内容到指定平台"""
    if isinstance(content, TextContent):
        return f"已发布文本至 {platform}: {content.text[:50]}..."
    elif isinstance(content, ImageContent):
        return f"已发布图像至 {platform}: {content.url}"
    elif isinstance(content, VideoContent):
        return f"已发布视频至 {platform}: {content.url} (时长: {content.duration}s)"
```

---

## 访问上下文

当前官方文档强调：工具不仅能接收模型生成的参数，还可以访问 Agent/Runtime 中的上下文信息。这样可以避免把用户 ID、权限、数据库连接、长期记忆等敏感或运行时数据暴露给模型。

### 1. 访问短期记忆（State）

在 Agent 中，短期状态通常包含 `messages` 和自定义 state 字段。工具可以通过注入方式读取这些状态。

```python
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

@tool
def get_conversation_length(
    state: Annotated[dict, InjectedState]
) -> str:
    """获取当前对话消息数量。"""
    return f"当前对话包含 {len(state['messages'])} 条消息"
```

#### 访问自定义 State 字段

```python
from typing import Annotated, TypedDict
from langchain.agents import AgentState, create_agent
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

class CustomState(AgentState):
    user_id: str
    plan: list[str]

@tool
def get_current_user(
    state: Annotated[CustomState, InjectedState]
) -> str:
    """获取当前用户 ID。"""
    return state["user_id"]

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[get_current_user],
    state_schema=CustomState,
)
```

### 2. 更新 State：返回 Command

工具不仅可以读取状态，也可以通过返回 `Command(update=...)` 更新图状态。适合保存任务进度、表单字段、用户选择等。

```python
from typing import Annotated
from langchain_core.tools import tool
from langgraph.types import Command
from langgraph.prebuilt import InjectedToolCallId
from langchain_core.messages import ToolMessage

@tool
def update_user_name(
    name: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """更新当前用户姓名。"""
    return Command(update={
        "user_name": name,
        "messages": [ToolMessage(
            content=f"已将用户姓名更新为 {name}",
            tool_call_id=tool_call_id,
        )],
    })
```

### 3. 访问 Context（运行时上下文）

`context` 适合存放单次调用相关的数据，例如用户 ID、权限、租户、feature flags、数据库连接等。它不应该进入模型上下文。

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain_core.tools import tool, ToolException

@dataclass
class Context:
    user_id: str
    permissions: list[str]

@tool
def delete_resource(resource_id: str, runtime) -> str:
    """删除资源，需要 resource.delete 权限。"""
    permissions = runtime.context.permissions
    if "resource.delete" not in permissions:
        raise ToolException("权限不足：需要 resource.delete")
    return f"资源 {resource_id} 已删除"

agent = create_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[delete_resource],
    context_schema=Context,
)
```

> 不同版本中运行时注入 API 可能有差异。核心思想是：**模型只生成业务参数，用户身份、权限、连接等运行时数据由应用注入**。

### 4. 访问长期记忆（Store）

Store 用于跨会话保存长期数据，例如用户偏好、历史资料、画像等。

```python
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedStore

@tool
def save_user_preference(
    key: str,
    value: str,
    store: Annotated[object, InjectedStore()],
) -> str:
    """保存用户偏好。"""
    store.put(("preferences",), key, value)
    return f"已保存偏好 {key}"
```

### 5. StreamWriter：工具内流式进度

长时间运行的工具可以通过 stream writer 输出进度，让 UI 或调用方实时显示状态。

```python
from langchain_core.tools import tool
from langgraph.types import StreamWriter

@tool
async def generate_report(topic: str, writer: StreamWriter) -> str:
    """生成报告，并流式输出进度。"""
    writer({"status": "start", "message": f"开始生成 {topic} 报告"})
    # ... 执行任务
    writer({"status": "progress", "message": "完成大纲"})
    # ... 执行任务
    writer({"status": "done", "message": "报告生成完成"})
    return "报告内容..."
```

### 6. 执行信息与服务端信息

工具还可以访问执行时信息，例如：

- 当前 tool call ID
- RunnableConfig 中的 tags/metadata/configurable
- 当前线程或调用 ID
- 运行环境、服务信息或注入依赖

这类信息适合用于审计、追踪、权限校验和调试，不建议让模型直接生成。

---

## 错误处理

### 1. 使用 ToolException

```python
from langchain_core.tools import tool, ToolException

@tool
def divide(a: float, b: float) -> float:
    """除法运算"""
    if b == 0:
        raise ToolException("错误: 除数不能为零。请提供非零的除数。")
    return a / b

@tool
def fetch_user_data(user_id: str) -> dict:
    """获取用户数据"""
    try:
        user = database.get_user(user_id)
        if user is None:
            raise ToolException(f"未找到用户 ID: {user_id}。请检查用户 ID 是否正确。")
        return user
    except DatabaseError as e:
        raise ToolException(f"数据库错误: {e}. 请稍后重试或联系管理员。")
    except Exception as e:
        raise ToolException(f"获取用户数据时发生未知错误: {e}")
```

**ToolException 的优势**:
- Agent 可以理解错误消息
- 错误消息会返回给模型，模型可以重试或采取其他行动
- 提供用户友好的错误信息

### 2. 在中间件中统一处理错误

```python
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

@wrap_tool_call # 使用 wrap_tool_call 装饰器，拦截所有工具调用 
def handle_tool_errors(request, handler):
    """统一的工具错误处理中间件"""
    try:
        return handler(request)

    except ToolException as e:
        # ToolException 已经有友好的消息
        return ToolMessage(
            content=str(e),
            tool_call_id=request.tool_call["id"]
        )

    except ValueError as e:
        # 参数验证错误
        return ToolMessage(
            content=f"参数错误: {e}. 请检查输入参数。",
            tool_call_id=request.tool_call["id"]
        )

    except ConnectionError as e:
        # 网络错误
        return ToolMessage(
            content=f"连接失败: {e}. 请检查网络连接或稍后重试。",
            tool_call_id=request.tool_call["id"]
        )

    except TimeoutError as e:
        # 超时错误
        return ToolMessage(
            content=f"请求超时: {e}. 服务响应缓慢，请稍后重试。",
            tool_call_id=request.tool_call["id"]
        )

    except Exception as e:
        # 未知错误
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"工具执行错误: {error_trace}")

        return ToolMessage(
            content=f"工具执行失败: 发生未知错误。请联系技术支持。",
            tool_call_id=request.tool_call["id"]
        )

# 使用中间件
agent = create_agent(
    model=model,
    tools=[divide, fetch_user_data],
    middleware=[handle_tool_errors] # 添加中间件，在其中调用handle_tool_errors的工具，捕获错误并返回 ToolMessage
)
```

### 3. 重试逻辑

```python
from langchain.agents.middleware import wrap_tool_call
import time

@wrap_tool_call # 使用 wrap_tool_call 装饰器，拦截所有工具调用
def retry_on_failure(request, handler):
    """失败时自动重试的中间件"""
    max_retries = 3
    retry_delay = 1  # 秒

    for attempt in range(max_retries):
        try:
            return handler(request)

        except (ConnectionError, TimeoutError) as e:
            if attempt < max_retries - 1:
                # 指数退避
                wait_time = retry_delay * (2 ** attempt)
                print(f"重试 {attempt + 1}/{max_retries}，等待 {wait_time}s...")
                time.sleep(wait_time)
            else:
                # 最后一次重试失败
                raise ToolException(
                    f"重试 {max_retries} 次后仍然失败: {e}"
                )

        except Exception as e:
            # 其他错误不重试
            raise

agent = create_agent(
    model=model,
    tools=[api_call_tool],
    middleware=[retry_on_failure, handle_tool_errors]
)
```

### 4. 错误日志记录

```python
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@tool
def critical_operation(param: str) -> str:
    """执行关键操作"""
    start_time = datetime.now()

    try:
        logger.info(f"开始执行 critical_operation，参数: {param}")

        # 执行操作
        result = perform_critical_task(param)

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"critical_operation 成功完成，耗时: {duration}s")

        return result

    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()

        # 记录详细错误
        logger.error(
            f"critical_operation 失败: {e}",
            extra={
                "param": param,
                "duration": duration,
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat()
            },
            exc_info=True  # 包含堆栈跟踪
        )

        # 发送告警（可选）
        send_alert(f"Critical operation failed: {e}")

        raise ToolException(f"操作失败: {e}")
```

---

## 异步工具

### 基础异步工具

```python
import asyncio
import aiohttp
from langchain_core.tools import tool

@tool
async def async_web_search(query: str, limit: int = 10) -> str:
    """异步执行网络搜索"""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://api.example.com/search",
            params={"q": query, "limit": limit}
        ) as response:
            data = await response.json()
            results = data.get("results", [])
            return "\n".join([f"- {r['title']}: {r['url']}" for r in results[:limit]])

@tool
async def async_database_query(query: str) -> list:
    """异步数据库查询"""
    # 使用异步数据库驱动
    async with async_db_pool.acquire() as conn:
        results = await conn.fetch(query)
        return [dict(row) for row in results]
```

### 同时支持同步和异步

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class WeatherInput(BaseModel):
    location: str = Field(description="位置名称")

class WeatherTool(BaseTool):
    """天气查询工具，支持同步和异步"""

    name: str = "get_weather"
    description: str = "获取指定位置的天气信息"
    args_schema: Type[BaseModel] = WeatherInput

    def _run(self, location: str) -> str:
        """同步执行"""
        import requests
        response = requests.get(
            f"https://api.weather.com/v1/current",
            params={"location": location}
        )
        data = response.json()
        return f"{location} 的天气: {data['condition']}, 温度: {data['temp']}°C"

    async def _arun(self, location: str) -> str:
        """异步执行"""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://api.weather.com/v1/current",
                params={"location": location}
            ) as response:
                data = await response.json()
                return f"{location} 的天气: {data['condition']}, 温度: {data['temp']}°C"
```

### 并行执行异步工具

```python
import asyncio
from langchain_core.tools import tool

@tool
async def fetch_user_data(user_id: str) -> dict:
    """异步获取用户数据"""
    await asyncio.sleep(0.5)  # 模拟 API 调用
    return {"id": user_id, "name": f"User {user_id}"}

@tool
async def fetch_order_data(user_id: str) -> list:
    """异步获取订单数据"""
    await asyncio.sleep(0.5)  # 模拟 API 调用
    return [{"order_id": "001", "amount": 100}]

@tool
async def fetch_activity_log(user_id: str) -> list:
    """异步获取活动日志"""
    await asyncio.sleep(0.5)  # 模拟 API 调用
    return [{"action": "login", "time": "2025-01-01"}]

# Agent 会自动并行执行这些异步工具
# 当模型决定调用多个工具时，它们会同时执行
agent = create_agent(
    model=model,
    tools=[fetch_user_data, fetch_order_data, fetch_activity_log]
)

# 如果模型生成 3 个工具调用，它们会并行执行
# 总耗时约 0.5 秒，而不是 1.5 秒
```

---

## 流式输出

### 工具内部流式输出

```python
from langgraph.types import StreamWriter
from langchain_core.tools import tool

@tool
async def generate_report(topic: str, config=None) -> str:
    """生成报告并流式输出进度"""
    writer: StreamWriter = config.get("writer") if config else None

    # 流式输出进度
    if writer:
        writer({"status": "starting", "message": f"开始生成关于 '{topic}' 的报告..."})

    # 模拟生成过程
    sections = ["引言", "主要内容", "分析", "结论"]
    report_parts = []

    for i, section in enumerate(sections):
        await asyncio.sleep(1)  # 模拟处理时间

        section_content = f"## {section}\n这是关于 {topic} 的 {section} 部分。"
        report_parts.append(section_content)

        if writer:
            writer({
                "status": "progress",
                "message": f"已完成 {section} ({i+1}/{len(sections)})",
                "progress": (i + 1) / len(sections)
            })

    full_report = "\n\n".join(report_parts)

    if writer:
        writer({"status": "complete", "message": "报告生成完成！"})

    return full_report

# 使用时启用流式模式
agent = create_agent(model=model, tools=[generate_report])

async for chunk in agent.astream(
    {"messages": [{"role": "user", "content": "生成关于 AI 的报告"}]},
    stream_mode="custom"  # 启用自定义流式输出
):
    print(chunk)
    # 输出: {"status": "starting", "message": "开始生成..."}
    # 输出: {"status": "progress", "message": "已完成 引言 (1/4)", "progress": 0.25}
    # ...
```

### 流式工具调用

```python
from langchain_core.tools import tool
@tool
async def stream_search_results(query: str, config=None) -> str:
    """流式返回搜索结果"""
    writer = config.get("writer") if config else None

    # 模拟搜索多个来源
    sources = ["Wikipedia", "News", "Academic Papers", "Blogs"]
    all_results = []

    for source in sources:
        if writer:
            writer(f"正在搜索 {source}...")

        await asyncio.sleep(0.5)
        results = await search_source(source, query)
        all_results.extend(results)

        if writer:
            writer(f"从 {source} 找到 {len(results)} 条结果")

    return format_search_results(all_results)

# 在 Agent 中使用流式事件
async for event in agent.astream_events(
    {"messages": [{"role": "user", "content": "搜索关于量子计算的信息"}]},
    version="v2"
):
    if event["event"] == "on_tool_start":
        print(f"🔧 开始执行工具: {event['name']}")

    elif event["event"] == "on_tool_stream":
        print(f"📊 进度: {event['data']}")

    elif event["event"] == "on_tool_end":
        print(f"✅ 工具完成: {event['name']}")
```

---

## 特殊类型工具

### 1. Retriever 工具（RAG）

```python
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 创建向量存储
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings()
)

# 创建 retriever 工具
retriever_tool = create_retriever_tool(
    retriever=vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    ),
    name="search_company_docs",
    description="""
        搜索公司文档库以查找相关政策、流程和指南。

        使用场景:
        - 查找公司政策
        - 搜索操作流程
        - 检索技术文档

        最适合: 需要从公司知识库获取信息的问题
    """
)

agent = create_agent(
    model=model,
    tools=[retriever_tool],
    system_prompt="你是公司知识助手，帮助员工查找信息"
)
```

### 2. 代码解释器工具

```python
from langchain_core.tools import tool
import subprocess
import tempfile
import os

@tool
def execute_python_code(code: str) -> str:
    """
    在沙盒环境中执行 Python 代码。

    安全限制:
    - 无网络访问
    - 无文件系统写入
    - 30 秒超时
    - 内存限制 100MB

    Args:
        code: 要执行的 Python 代码

    Returns:
        代码执行的输出或错误信息
    """
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        # 在隔离环境中执行
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=30,  # 30 秒超时
            env={
                'PYTHONPATH': '',  # 限制导入
                'HOME': tempfile.gettempdir()
            }
        )

        if result.returncode == 0:
            return f"执行成功:\n{result.stdout}"
        else:
            return f"执行错误:\n{result.stderr}"

    except subprocess.TimeoutExpired:
        return "错误: 代码执行超时（超过 30 秒）"

    except Exception as e:
        return f"错误: {e}"

    finally:
        # 清理临时文件
        os.unlink(temp_file)
```

### 3. 数据库工具

```python
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool

# 连接数据库
db = SQLDatabase.from_uri("sqlite:///chinook.db")

@tool
def list_tables() -> str:
    """列出数据库中所有可用的表"""
    return db.get_table_names()

@tool
def get_table_schema(table_name: str) -> str:
    """获取指定表的 Schema"""
    return db.get_table_info_no_throw([table_name])

@tool
def execute_sql(query: str) -> str:
    """
    执行 SQL 查询（仅限 SELECT）。

    安全检查:
    - 仅允许 SELECT 语句
    - 自动添加 LIMIT 子句
    - 禁止子查询
    """
    # 安全验证
    query_upper = query.upper().strip()

    if not query_upper.startswith("SELECT"):
        raise ToolException("仅允许 SELECT 查询")

    if any(word in query_upper for word in ["DROP", "DELETE", "UPDATE", "INSERT"]):
        raise ToolException("禁止的操作")

    # 添加限制
    if "LIMIT" not in query_upper:
        query += " LIMIT 100"

    try:
        result = db.run(query)
        return result
    except Exception as e:
        raise ToolException(f"SQL 执行错误: {e}")

# SQL Agent
sql_tools = [list_tables, get_table_schema, execute_sql]
sql_agent = create_agent(
    model=model,
    tools=sql_tools,
    system_prompt="""
    你是一个 SQL 专家。请遵循以下步骤:

    1. 使用 list_tables 查看可用的表
    2. 使用 get_table_schema 了解表结构
    3. 构建 SQL 查询
    4. 使用 execute_sql 执行查询
    5. 解释结果
    """
)
```

### 4. API 调用工具

```python
import requests
from typing import Optional

@tool
def call_rest_api(
    method: str,
    url: str,
    headers: Optional[dict] = None,
    body: Optional[dict] = None
) -> str:
    """
    调用 REST API。

    Args:
        method: HTTP 方法 (GET, POST, PUT, DELETE)
        url: API 端点 URL
        headers: HTTP 头部
        body: 请求体（仅用于 POST/PUT）

    Returns:
        API 响应
    """
    if headers is None:
        headers = {}

    # 安全检查: 仅允许特定域名
    allowed_domains = ["api.example.com", "api.internal.com"]
    from urllib.parse import urlparse
    domain = urlparse(url).netloc

    if domain not in allowed_domains:
        raise ToolException(f"不允许访问域名: {domain}")

    try:
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            json=body,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        raise ToolException("API 请求超时")

    except requests.exceptions.HTTPError as e:
        raise ToolException(f"HTTP 错误: {e}")

    except Exception as e:
        raise ToolException(f"API 调用失败: {e}")
```

---

## 工具集成

### 预构建工具

LangChain 提供了许多预构建的工具集成。

#### 1. 搜索工具

```python
# DuckDuckGo 搜索
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

# Tavily 搜索（更强大）
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_search = TavilySearchResults(
    max_results=5,
    search_depth="advanced"
)

# Google 搜索
from langchain_community.tools import GoogleSearchRun
from langchain_community.utilities import GoogleSearchAPIWrapper

google_search = GoogleSearchRun(api_wrapper=GoogleSearchAPIWrapper())
```

#### 2. Wikipedia 工具

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=3,
        doc_content_chars_max=4000
    )
)
```

#### 3. 文件操作工具

```python
from langchain_community.tools import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool
)

# 文件读取
read_tool = ReadFileTool()

# 文件写入
write_tool = WriteFileTool()

# 目录列表
list_tool = ListDirectoryTool()

file_tools = [read_tool, write_tool, list_tool]
```

### 工具包（Toolkits）

工具包是一组相关工具的集合。

```python
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits import GmailToolkit

# 文件管理工具包
file_toolkit = FileManagementToolkit(
    root_dir="/path/to/workspace"
)
file_tools = file_toolkit.get_tools()

# SQL 数据库工具包
db_toolkit = SQLDatabaseToolkit(db=db, llm=model)
db_tools = db_toolkit.get_tools()

# Gmail 工具包
gmail_toolkit = GmailToolkit()
gmail_tools = gmail_toolkit.get_tools()

# 组合使用
all_tools = file_tools + db_tools + gmail_tools
agent = create_agent(model=model, tools=all_tools)
```

---

## 官方 v1.x 补充：工具执行与返回值

### 1. Tool return values

工具返回值会被转换成 `ToolMessage` 传回模型。返回值可以是字符串、对象，或用于更新状态的 `Command`。

#### 返回字符串

```python
@tool
def get_weather(city: str) -> str:
    """获取城市天气。"""
    return f"{city}: 晴，22°C"
```

#### 返回对象

```python
@tool
def get_user(user_id: str) -> dict:
    """获取用户信息。"""
    return {"id": user_id, "name": "Alice", "level": "gold"}
```

对象返回值会被序列化后提供给模型。建议返回结构清晰、字段少而明确的数据；大对象应先摘要或分页。

#### 返回 Command

如果工具需要更新 LangGraph/Agent 状态，可以返回 `Command(update=...)`。这适合在工具执行后写入状态字段、追加消息或记录任务进度。

```python
from langgraph.types import Command

@tool
def mark_task_done(task_id: str) -> Command:
    """标记任务完成。"""
    return Command(update={"completed_tasks": [task_id]})
```

### 2. State injection

State injection 可以让工具读取 Agent 状态，而不要求模型生成这些参数。典型用途：读取历史消息、当前用户、任务上下文、计划等。

```python
from typing import Annotated
from langgraph.prebuilt import InjectedState

@tool
def summarize_current_state(state: Annotated[dict, InjectedState]) -> str:
    """总结当前状态。"""
    return f"当前有 {len(state['messages'])} 条消息"
```

### 3. Dynamic tool selection

生产环境中不应总是把所有工具暴露给模型。可以根据用户权限、任务阶段、环境、成本限制动态选择工具。

```python
from langchain.agents.middleware import wrap_model_call

@wrap_model_call
def select_tools_by_role(request, handler):
    role = request.runtime.context.get("role", "viewer")

    if role == "viewer":
        request = request.override(
            tools=[t for t in request.tools if t.name.startswith("read_")]
        )

    return handler(request)
```

### 4. Prebuilt tools

LangChain 生态提供了大量预构建工具和工具包，例如搜索、文件系统、SQL、检索、Gmail、Slack 等。使用预构建工具前应检查：

- 是否需要 API key 或 OAuth
- 是否会执行写操作或外部副作用
- 是否需要人工审批
- 返回内容是否可能过长
- 是否需要限制访问目录、域名或数据库表

### 5. Server-side tool use

部分模型 provider 支持服务端工具，例如 web search、code execution 等。它们由 provider 在服务端执行，不同于 LangChain 本地工具。

| 类型 | 优点 | 注意事项 |
|-----|------|---------|
| LangChain 本地工具 | 可控、可审计、可接入内部系统 | 需要自己维护执行环境 |
| Provider 服务端工具 | 与模型深度集成、配置简单 | 可观测性、权限边界、数据合规需要额外确认 |

选择建议：内部系统、敏感数据、强审计需求优先使用 LangChain 本地工具；公共搜索或 provider 原生能力可考虑服务端工具。

---

## 高级用法

### 1. 动态工具选择

根据上下文动态启用/禁用工具。

```python
from langchain.agents.middleware import wrap_model_call

@wrap_model_call
def dynamic_tool_selection(request, handler):
    """根据用户权限动态选择工具"""
    # 获取用户权限
    user_role = request.runtime.context.get("user_role", "guest")

    # 根据角色过滤工具
    if user_role == "admin":
        # 管理员: 所有工具
        pass
    elif user_role == "editor":
        # 编辑者: 不能删除
        filtered_tools = [
            t for t in request.tools
            if t.name not in ["delete_file", "drop_table"]
        ]
        request = request.override(tools=filtered_tools)
    else:
        # 访客: 仅只读工具
        filtered_tools = [
            t for t in request.tools
            if t.name.startswith("read_") or t.name.startswith("get_")
        ]
        request = request.override(tools=filtered_tools)

    return handler(request)

agent = create_agent(
    model=model,
    tools=[read_file, write_file, delete_file, read_db, write_db, drop_table],
    middleware=[dynamic_tool_selection],
    context_schema=Context
)
```

### 2. 工具链（Tool Chaining）

一个工具的输出作为另一个工具的输入。

```python
@tool
def search_products(query: str) -> list:
    """搜索产品"""
    results = product_db.search(query)
    return [{"id": p.id, "name": p.name} for p in results]

@tool
def get_product_details(product_id: str) -> dict:
    """获取产品详情"""
    product = product_db.get(product_id)
    return {
        "id": product.id,
        "name": product.name,
        "price": product.price,
        "description": product.description,
        "reviews": product.reviews
    }

@tool
def compare_products(product_ids: list[str]) -> str:
    """比较多个产品"""
    products = [product_db.get(pid) for pid in product_ids]

    comparison = "产品对比:\n"
    for p in products:
        comparison += f"\n{p.name}:\n"
        comparison += f"  价格: ¥{p.price}\n"
        comparison += f"  评分: {p.rating}/5\n"

    return comparison

# Agent 会自动链接这些工具:
# 1. search_products("笔记本电脑")
# 2. get_product_details(product_id="123")
# 3. compare_products(["123", "456", "789"])
```

### 3. 条件工具执行

```python
@tool
def check_inventory(product_id: str) -> dict:
    """检查库存"""
    inventory = inventory_db.get(product_id)
    return {
        "product_id": product_id,
        "in_stock": inventory.quantity > 0,
        "quantity": inventory.quantity
    }

@tool
def reserve_product(product_id: str, quantity: int = 1) -> str:
    """预留产品（仅当有库存时）"""
    # 首先检查库存
    inventory = inventory_db.get(product_id)

    if inventory.quantity < quantity:
        raise ToolException(
            f"库存不足。可用: {inventory.quantity}，需要: {quantity}"
        )

    # 预留
    reservation_id = inventory_db.reserve(product_id, quantity)
    return f"已预留 {quantity} 件，预留 ID: {reservation_id}"

# Agent 会先调用 check_inventory，
# 然后根据结果决定是否调用 reserve_product
```

### 4. 工具回调

```python
from langchain.callbacks import BaseCallbackHandler

class ToolCallbackHandler(BaseCallbackHandler):
    """工具调用回调处理器"""

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        """工具开始执行时"""
        tool_name = serialized.get("name", "unknown")
        print(f"🔧 开始执行工具: {tool_name}")
        print(f"📥 输入: {input_str}")

    def on_tool_end(self, output: str, **kwargs):
        """工具执行完成时"""
        print(f"✅ 工具输出: {output[:100]}...")

    def on_tool_error(self, error: Exception, **kwargs):
        """工具执行错误时"""
        print(f"❌ 工具错误: {error}")

# 使用回调
agent = create_agent(
    model=model,
    tools=[search, calculate],
    callbacks=[ToolCallbackHandler()]
)
```

---

## 最佳实践

### 1. 工具设计原则

#### ✅ 单一职责

```python
# ✅ 好的设计 - 每个工具专注一个任务
@tool
def get_user(user_id: str) -> dict:
    """获取用户信息"""
    return db.get_user(user_id)

@tool
def update_user(user_id: str, data: dict) -> str:
    """更新用户信息"""
    db.update_user(user_id, data)
    return f"用户 {user_id} 已更新"

# ❌ 不好的设计 - 一个工具做太多事情
@tool
def manage_user(action: str, user_id: str, data: dict = None) -> any:
    """管理用户（获取、创建、更新、删除）"""
    if action == "get":
        return db.get_user(user_id)
    elif action == "create":
        return db.create_user(data)
    # ...太复杂
```

#### ✅ 清晰的文档

```python
@tool
def process_payment(
    order_id: str,
    amount: float,
    payment_method: str,
    currency: str = "CNY"
) -> str:
    """
    处理订单支付。

    此工具会验证订单、处理支付并更新订单状态。

    使用场景:
    - 完成订单支付
    - 处理退款（金额为负数）

    Args:
        order_id: 订单 ID，格式: ORD-XXXXXX
        amount: 支付金额，必须大于 0（退款时为负数）
        payment_method: 支付方式，可选: credit_card, alipay, wechat
        currency: 货币代码，默认 CNY

    Returns:
        支付确认消息，包含交易 ID

    Raises:
        ToolException: 当订单不存在或支付失败时

    Examples:
        >>> process_payment("ORD-123456", 99.99, "alipay")
        "支付成功！交易 ID: TXN-789"

    注意:
        - 确保订单状态为 "待支付"
        - 支付金额会四舍五入到 2 位小数
        - 大额支付（>10000）需要额外验证
    """
    # 实现...
```

#### ✅ 输入验证

```python
from pydantic import BaseModel, Field, validator

class PaymentInput(BaseModel):
    """支付输入验证"""

    order_id: str = Field(
        pattern=r"^ORD-\d{6}$",
        description="订单 ID"
    )
    amount: float = Field(
        gt=0,
        le=100000,
        description="支付金额"
    )
    payment_method: Literal["credit_card", "alipay", "wechat"]

    @validator("amount")
    def round_amount(cls, v):
        """金额四舍五入到 2 位小数"""
        return round(v, 2)

@tool(args_schema=PaymentInput)
def process_payment(
    order_id: str,
    amount: float,
    payment_method: str,
    currency: str = "CNY"
) -> str:
    """处理支付"""
    # 实现...
```

### 2. 性能优化

#### ✅ 缓存

```python
from functools import lru_cache
import time

@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    """获取汇率（带缓存）"""
    return _get_exchange_rate_cached(from_currency, to_currency)

@lru_cache(maxsize=100)
def _get_exchange_rate_cached(from_currency: str, to_currency: str) -> float:
    """缓存的汇率查询"""
    # 调用外部 API
    response = requests.get(f"https://api.exchangerate.com/{from_currency}/{to_currency}")
    return response.json()["rate"]

# 带过期时间的缓存
from datetime import datetime, timedelta

_cache = {}
_cache_timeout = timedelta(minutes=5)

@tool
def get_stock_price(symbol: str) -> float:
    """获取股票价格（5 分钟缓存）"""
    now = datetime.now()

    # 检查缓存
    if symbol in _cache:
        cached_time, cached_price = _cache[symbol]
        if now - cached_time < _cache_timeout:
            return cached_price

    # 获取新数据
    price = fetch_stock_price(symbol)
    _cache[symbol] = (now, price)
    return price
```

#### ✅ 批量处理

```python
@tool
def get_user_details(user_ids: list[str]) -> list[dict]:
    """批量获取用户详情（而非逐个查询）"""
    # ✅ 好 - 一次数据库查询
    users = db.get_users_batch(user_ids)
    return users

# ❌ 不好 - 多次查询
# for user_id in user_ids:
#     user = db.get_user(user_id)
```

#### ✅ 异步操作

```python
@tool
async def fetch_multiple_sources(query: str) -> dict:
    """并行从多个来源获取数据"""
    # 并行执行多个 API 调用
    results = await asyncio.gather(
        fetch_from_source_a(query),
        fetch_from_source_b(query),
        fetch_from_source_c(query),
        return_exceptions=True
    )

    return {
        "source_a": results[0] if not isinstance(results[0], Exception) else None,
        "source_b": results[1] if not isinstance(results[1], Exception) else None,
        "source_c": results[2] if not isinstance(results[2], Exception) else None,
    }
```

### 3. 安全实践

#### ✅ 输入清理

```python
import re

@tool
def execute_command(command: str) -> str:
    """执行系统命令（受限）"""
    # 白名单检查
    allowed_commands = ["ls", "pwd", "echo", "cat"]

    cmd_parts = command.split()
    if not cmd_parts or cmd_parts[0] not in allowed_commands:
        raise ToolException(f"不允许的命令: {cmd_parts[0]}")

    # 防止命令注入
    if any(char in command for char in [";", "&", "|", "`", "$"]):
        raise ToolException("命令包含非法字符")

    # 执行
    result = subprocess.run(
        cmd_parts,
        capture_output=True,
        text=True,
        timeout=5
    )
    return result.stdout
```

#### ✅ 权限检查

```python
from dataclasses import dataclass

@dataclass
class Context:
    user_id: str
    permissions: list[str]

@tool
def delete_resource(resource_id: str, config=None) -> str:
    """删除资源（需要权限）"""
    # 从上下文获取权限
    if config:
        context = config.get("context", {})
        permissions = context.get("permissions", [])

        if "resource.delete" not in permissions:
            raise ToolException("权限不足: 需要 'resource.delete' 权限")

    # 执行删除
    db.delete_resource(resource_id)
    return f"资源 {resource_id} 已删除"
```

#### ✅ 审计日志

```python
import logging
from datetime import datetime

audit_logger = logging.getLogger("audit")

@tool
def sensitive_operation(param: str, config=None) -> str:
    """敏感操作（记录审计日志）"""
    # 记录审计日志
    user_id = config.get("context", {}).get("user_id", "unknown") if config else "unknown"

    audit_logger.info({
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "operation": "sensitive_operation",
        "params": {"param": param},
        "ip_address": config.get("context", {}).get("ip_address") if config else None
    })

    # 执行操作
    result = perform_sensitive_task(param)

    # 记录结果
    audit_logger.info({
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "operation": "sensitive_operation",
        "status": "success",
        "result_summary": result[:100]
    })

    return result
```

### 4. 测试

#### 单元测试

```python
import pytest
from unittest.mock import Mock, patch

@pytest.fixture
def mock_db():
    """模拟数据库"""
    db = Mock()
    db.get_user.return_value = {"id": "123", "name": "测试用户"}
    return db

def test_get_user_tool(mock_db):
    """测试获取用户工具"""
    with patch('your_module.db', mock_db):
        result = get_user.invoke({"user_id": "123"})

        assert result["id"] == "123"
        assert result["name"] == "测试用户"
        mock_db.get_user.assert_called_once_with("123")

def test_tool_error_handling():
    """测试错误处理"""
    with pytest.raises(ToolException) as exc_info:
        divide.invoke({"a": 10, "b": 0})

    assert "除数不能为零" in str(exc_info.value)

def test_tool_validation():
    """测试参数验证"""
    with pytest.raises(ValueError):
        send_email.invoke({
            "to": "invalid-email",
            "subject": "Test",
            "body": "Content"
        })
```

---

## 总结

### 核心要点

1. **Tools 扩展 Agent 能力**
   - 封装函数 + 输入 Schema
   - 模型决定何时调用
   - 返回结果供模型继续推理

2. **多种定义方式**
   - `@tool` 装饰器（最简单）
   - `args_schema` 参数（复杂验证）
   - 继承 `BaseTool`（完全控制）

3. **参数验证很重要**
   - 使用 Pydantic 模型
   - 添加字段验证
   - 清理和验证输入

4. **错误处理要友好**
   - 使用 `ToolException`
   - 提供清晰的错误消息
   - 在中间件中统一处理

5. **工具可以访问运行时上下文**
   - State: 短期对话状态
   - Context: 单次调用数据、权限、租户等
   - Store: 长期记忆
   - StreamWriter: 工具内流式进度

6. **支持多种返回值**
   - 字符串：简单结果
   - 对象：结构化结果
   - Command：更新 Agent/LangGraph 状态

7. **支持异步操作**
   - 使用 `async def`
   - 并行执行提高性能
   - 同时支持同步和异步

8. **安全第一**
   - 输入验证和清理
   - 权限检查
   - 审计日志
   - 限制危险操作

### 推荐工具设计模式

```python
from langchain_core.tools import tool, ToolException
from pydantic import BaseModel, Field, validator
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class MyToolInput(BaseModel):
    """工具输入验证"""
    param1: str = Field(description="参数1说明")
    param2: int = Field(gt=0, description="参数2说明")

    @validator("param1")
    def validate_param1(cls, v):
        # 自定义验证
        return v

@tool(args_schema=MyToolInput)
async def my_tool(param1: str, param2: int, config=None) -> str:
    """
    工具的详细描述。

    使用场景:
    - 场景1
    - 场景2

    Args:
        param1: 详细说明
        param2: 详细说明

    Returns:
        返回值说明

    Examples:
        >>> my_tool("test", 10)
        "结果"
    """
    try:
        # 记录审计日志
        user_id = config.get("context", {}).get("user_id") if config else "unknown"
        logger.info(f"User {user_id} called my_tool with {param1}, {param2}")

        # 执行工具逻辑
        result = await perform_operation(param1, param2)

        return result

    except ValueError as e:
        raise ToolException(f"参数错误: {e}")
    except Exception as e:
        logger.error(f"Tool error: {e}", exc_info=True)
        raise ToolException(f"操作失败: {e}")
```

### 参考资源

- **官方文档**: https://docs.langchain.com/oss/python/langchain/tools
- **工具集成**: https://docs.langchain.com/oss/python/integrations/tools
- **API 参考**: https://api.python.langchain.com/en/latest/tools/langchain_core.tools.html

---

**文档版本**: 1.1
**最后更新**: 2026-06-01
**基于**: LangChain v1.x 官方文档

如有问题或建议，欢迎反馈！
