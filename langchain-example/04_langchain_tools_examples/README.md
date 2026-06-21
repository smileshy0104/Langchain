# LangChain Tools 完整示例集 (GLM 模型版本)

本项目包含基于 LangChain Tools 官方文档的完整示例代码，使用智谱 AI 的 GLM 模型实现。

## 📋 目录

- [01_tool_definition.py](01_tool_definition.py) - 工具定义的多种方式
- [02_validation_and_errors.py](02_validation_and_errors.py) - 参数验证和错误处理
- [03_async_and_special_tools.py](03_async_and_special_tools.py) - 异步工具和特殊类型工具

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install langchain langchain-community langchain-core zhipuai pydantic
```

### 2. 设置 API Key

```bash
export ZHIPUAI_API_KEY="your-api-key-here"
```

### 3. 运行示例

```bash
# 工具定义方式示例
python 01_tool_definition.py

# 参数验证和错误处理
python 02_validation_and_errors.py

# 异步工具和特殊类型工具
python 03_async_and_special_tools.py
```

## 📚 示例说明

### 01_tool_definition.py - 工具定义方式

**包含内容:**
- @tool 装饰器 - 最简单的工具定义方式
- 详细的文档字符串 (Docstring)
- args_schema 参数 - 使用 Pydantic 定义参数
- 继承 BaseTool 类 - 完全控制
- 自定义工具名称和描述
- 复杂类型参数 (嵌套模型)
- 多个工具组合使用

**核心定义方式:**

**方式 1: @tool 装饰器 (最简单)**
```python
@tool
def search_database(query: str, limit: int = 10) -> str:
    """搜索客户数据库以查找匹配查询的记录。

    Args:
        query: 要查找的搜索词
        limit: 返回的最大结果数
    """
    return f"找到 {limit} 条关于 '{query}' 的结果"
```

**方式 2: 使用 args_schema (推荐用于复杂验证)**
```python
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

    @validator("query")
    def validate_query(cls, v):
        """验证查询安全性"""
        forbidden = ["DROP", "DELETE", "UPDATE"]
        if any(word in v.upper() for word in forbidden):
            raise ValueError("只允许 SELECT 查询")
        return v

@tool(args_schema=DatabaseQueryInput)
def query_database(query: str, limit: int = 100, offset: int = 0) -> str:
    """在数据库中执行只读查询"""
    return f"查询结果: 返回 {limit} 行数据"
```

**方式 3: 继承 BaseTool (完全控制)**
```python
class AdvancedSearchTool(BaseTool):
    """高级搜索工具"""

    name: str = "advanced_search"
    description: str = "在知识库中进行高级搜索"
    args_schema: Type[BaseModel] = SearchInput

    # 工具特定的配置
    api_key: str = Field(default="", description="API 密钥")
    max_results: int = Field(default=10, description="最大结果数")

    def _run(self, query: str, filters: Dict = None) -> str:
        """同步执行搜索"""
        # 实现逻辑
        return "搜索结果..."

    async def _arun(self, query: str, filters: Dict = None) -> str:
        """异步执行搜索"""
        return self._run(query, filters)
```

---

### 02_validation_and_errors.py - 参数验证和错误处理

**包含内容:**
- 字段级验证 (Field Validators)
- 模型级验证 (Root Validators)
- ToolException 使用
- 错误处理中间件
- 重试机制和指数退避
- 复杂验证场景
- 与模型集成的错误处理
- 错误日志和监控

**字段级验证:**
```python
class EmailInput(BaseModel):
    """邮件发送参数验证"""

    to: str = Field(
        description="收件人邮箱地址",
        min_length=5,
        max_length=100
    )

    @validator("to")
    def validate_email(cls, v):
        """验证邮箱格式"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError(f"无效的邮箱地址: {v}")
        return v.lower()
```

**模型级验证:**
```python
class TransferInput(BaseModel):
    """转账参数验证"""

    from_account: str
    to_account: str
    amount: float = Field(gt=0)

    @root_validator
    def validate_transfer(cls, values):
        """确保不是自己转给自己"""
        if values.get("from_account") == values.get("to_account"):
            raise ValueError("不能转账给自己")

        if values.get("amount") > 50000:
            raise ValueError("单笔转账金额不能超过 50000")

        return values
```

**使用 ToolException:**
```python
@tool
def delete_file(file_path: str, force: bool = False) -> str:
    """删除文件"""
    dangerous_paths = ["/", "/usr", "/etc", "/System"]

    if not force and any(file_path.startswith(path) for path in dangerous_paths):
        raise ToolException(
            "不能删除系统目录！请使用 force=True 参数（不推荐）"
        )

    return f"文件 {file_path} 已删除"
```

**重试机制:**
```python
class ErrorHandlingTool(BaseTool):
    """带错误处理的工具基类"""

    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=1.0)

    def _run_with_retry(self, func, *args, **kwargs):
        """带重试的执行"""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # 指数退避
                    time.sleep(wait_time)
                else:
                    raise ToolException(f"操作失败，已重试 {self.max_retries} 次")
```

---

### 03_async_and_special_tools.py - 异步工具和特殊类型工具

**包含内容:**
- 异步工具基础 (async/await)
- 混合同步异步工具
- 流式工具 (Streaming)
- 数据库工具
- API 调用工具
- 文件处理工具
- 缓存工具
- 批处理工具
- 与模型集成
- StructuredTool 使用

**异步工具:**
```python
@tool
async def async_fetch_data(url: str) -> str:
    """异步获取数据"""
    await asyncio.sleep(1)  # 模拟网络请求
    return f"从 {url} 获取的数据"

# 使用
result = await async_fetch_data.ainvoke({"url": "https://api.example.com/data"})
```

**混合同步异步:**
```python
class DataProcessor(BaseTool):
    """数据处理工具（支持同步和异步）"""

    def _run(self, data: str) -> str:
        """同步处理"""
        return f"同步结果: {data.upper()}"

    async def _arun(self, data: str) -> str:
        """异步处理"""
        await asyncio.sleep(0.5)
        return f"异步结果: {data.upper()}"
```

**流式工具:**
```python
class StreamingTool(BaseTool):
    """流式数据生成工具"""

    def _run(self, topic: str, count: int = 5) -> Iterator[str]:
        """同步流式生成"""
        for i in range(count):
            time.sleep(0.2)
            yield f"[{i+1}/{count}] {topic} 的内容片段 {i+1}\n"

    async def _arun(self, topic: str, count: int = 5) -> AsyncIterator[str]:
        """异步流式生成"""
        for i in range(count):
            await asyncio.sleep(0.2)
            yield f"[{i+1}/{count}] {topic} 的内容片段 {i+1}\n"
```

**数据库工具:**
```python
class DatabaseTool(BaseTool):
    """数据库查询工具"""

    # 模拟数据库
    _database: Dict[str, List[Dict]] = {
        "users": [
            {"id": 1, "name": "张三", "email": "zhang@example.com"},
            {"id": 2, "name": "李四", "email": "li@example.com"}
        ]
    }

    def _run(self, table: str, filter_field: str = None, filter_value: str = None) -> str:
        """查询数据库"""
        data = self._database.get(table, [])

        if filter_field and filter_value:
            data = [row for row in data if str(row.get(filter_field)) == filter_value]

        return f"从 {table} 表找到 {len(data)} 条记录"
```

**缓存工具:**
```python
class CachedTool(BaseTool):
    """带缓存的工具"""

    _cache: Dict[str, str] = {}

    def _run(self, key: str, compute: bool = False) -> str:
        """执行操作（带缓存）"""
        # 检查缓存
        if key in self._cache and not compute:
            return f"缓存结果: {self._cache[key]}"

        # 计算结果
        result = f"{key.upper()}_COMPUTED"
        self._cache[key] = result

        return f"新计算结果: {result}"
```

**批处理工具:**
```python
class BatchProcessTool(BaseTool):
    """批处理工具"""

    async def _arun(self, items: List[str], operation: str = "process") -> str:
        """异步批处理"""
        async def process_item(item: str) -> str:
            await asyncio.sleep(0.2)
            return f"{operation}: {item} -> 完成"

        # 并行处理
        tasks = [process_item(item) for item in items]
        results = await asyncio.gather(*tasks)

        return "\n".join(results)
```

**StructuredTool:**
```python
def simple_function(text: str, count: int = 1) -> str:
    """简单的处理函数"""
    return f"{text} " * count

# 从函数创建工具
tool = StructuredTool.from_function(
    func=simple_function,
    name="repeat_text",
    description="重复文本指定次数"
)
```

## 💡 核心概念

### 1. 工具定义的三种方式对比

| 方式 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| @tool 装饰器 | 简单工具 | 代码简洁 | 验证能力有限 |
| args_schema | 需要复杂验证 | 强大的验证能力 | 代码稍多 |
| BaseTool | 需要完全控制 | 灵活性最高 | 代码最多 |

### 2. 参数验证层次

```
1. Pydantic 字段验证 (@validator)
   ↓
2. Pydantic 模型验证 (@root_validator)
   ↓
3. 工具内部逻辑验证
   ↓
4. ToolException 抛出
```

### 3. 同步 vs 异步工具

**何时使用异步:**
- 网络 I/O 操作 (API 调用)
- 文件 I/O 操作
- 数据库查询
- 需要并行执行多个操作

**何时使用同步:**
- 简单计算
- 本地数据处理
- 不涉及 I/O 的操作

### 4. 错误处理最佳实践

```python
# 1. 输入验证 - 使用 Pydantic
class Input(BaseModel):
    field: str = Field(...)
    @validator("field")
    def validate_field(cls, v):
        # 验证逻辑
        return v

# 2. 业务逻辑错误 - 使用 ToolException
if not valid_operation:
    raise ToolException("操作不允许")

# 3. 外部错误 - 使用重试机制
def _run_with_retry(self, func):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

# 4. 日志记录
logger.info(f"工具调用: {operation}")
logger.error(f"工具失败: {error}")
```

## 🎯 使用场景

### 场景 1: 数据提取和验证

**适用工具定义方式:** args_schema

```python
class ContactInfo(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    phone: str = Field(pattern=r'^\d{11}$')
    email: str = Field(pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

@tool(args_schema=ContactInfo)
def save_contact(name: str, phone: str, email: str) -> str:
    """保存联系人信息"""
    return f"已保存联系人 {name}"
```

### 场景 2: API 集成

**适用工具定义方式:** BaseTool + 异步

```python
class WeatherAPI(BaseTool):
    """天气 API 工具"""

    async def _arun(self, city: str) -> str:
        """异步获取天气"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.weather.com/{city}") as resp:
                data = await resp.json()
                return f"{city}: {data['weather']}, {data['temp']}°C"
```

### 场景 3: 数据库操作

**适用工具定义方式:** BaseTool + 验证

```python
class SafeDatabaseTool(BaseTool):
    """安全的数据库工具"""

    def _run(self, query: str) -> str:
        """执行查询"""
        # SQL 注入检查
        if any(word in query.upper() for word in ["DROP", "DELETE"]):
            raise ToolException("不允许修改操作")

        # 执行查询
        return execute_query(query)
```

### 场景 4: 批量处理

**适用工具定义方式:** 异步 + 批处理

```python
async def process_documents(file_paths: List[str]) -> str:
    """批量处理文档"""
    tasks = [process_single_doc(path) for path in file_paths]
    results = await asyncio.gather(*tasks)
    return f"处理了 {len(results)} 个文档"
```

## ⚠️ 注意事项

### 1. API Key 安全

```bash
# 使用环境变量
export ZHIPUAI_API_KEY="your-api-key-here"

# 不要硬编码
# ❌ 错误
api_key = "sk-xxx..."

# ✅ 正确
api_key = os.getenv("ZHIPUAI_API_KEY")
```

### 2. 参数验证

- 始终验证用户输入
- 使用 Pydantic Field 约束
- 添加自定义 validator
- 提供清晰的错误消息

### 3. 错误处理

- 使用 ToolException 而不是普通 Exception
- 提供详细的错误信息
- 实现重试机制
- 记录错误日志

### 4. 性能优化

- 对 I/O 操作使用异步
- 实现缓存机制
- 批量处理多个请求
- 使用连接池

### 5. 安全考虑

```python
# 文件操作 - 路径验证
if ".." in file_path or file_path.startswith("/"):
    raise ToolException("非法路径")

# SQL 查询 - 防止注入
if not query.upper().startswith("SELECT"):
    raise ToolException("只允许 SELECT 查询")

# API 调用 - 超时设置
async with timeout(10):
    result = await api_call()
```

## 🐛 常见问题

### Q1: 工具参数没有被正确解析

**原因:** args_schema 与函数签名不匹配

**解决方案:**
```python
# ✅ 正确 - 参数名称和类型一致
class Input(BaseModel):
    query: str
    limit: int

@tool(args_schema=Input)
def search(query: str, limit: int) -> str:
    pass
```

### Q2: 异步工具调用报错

**原因:** 在同步上下文中调用异步方法

**解决方案:**
```python
# ❌ 错误
result = async_tool.ainvoke({"url": "..."})

# ✅ 正确
result = await async_tool.ainvoke({"url": "..."})

# 或在同步环境中
result = asyncio.run(async_tool.ainvoke({"url": "..."}))
```

### Q3: ToolException 没有被捕获

**原因:** 使用了普通 Exception

**解决方案:**
```python
# ❌ 错误
raise Exception("错误")

# ✅ 正确
from langchain_core.tools import ToolException
raise ToolException("错误")
```

### Q4: 验证器没有生效

**原因:** validator 装饰器位置错误

**解决方案:**
```python
# ✅ 正确 - 在 @validator 之后定义方法
@validator("field")
def validate_field(cls, v):
    return v
```

### Q5: 模型不调用工具

**原因:** 工具描述不够清晰

**解决方案:**
```python
# ❌ 模糊的描述
"""处理数据"""

# ✅ 清晰的描述
"""
从数据库中搜索用户信息。

使用场景:
- 查找特定用户
- 获取用户列表
- 验证用户存在

Args:
    query: 搜索关键词，可以是用户名、邮箱或ID
    limit: 返回结果数量，默认10条
"""
```

## 📖 参考资源

- [LangChain 官方文档 - Tools](https://docs.langchain.com/oss/python/langchain/tools)
- [智谱 AI 文档](https://open.bigmodel.cn/dev/api)
- [Pydantic 文档](https://docs.pydantic.dev/)
- [Python asyncio 文档](https://docs.python.org/3/library/asyncio.html)
- [原始总结文档](../langchain-docs/LangChain_Tools_详细指南.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request!

## 📄 许可证

MIT License

---

**作者**: 基于 LangChain 官方文档改编
**版本**: 1.0
**更新日期**: 2025-01-23
