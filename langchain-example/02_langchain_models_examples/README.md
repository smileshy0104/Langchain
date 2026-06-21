# LangChain Models 完整示例集 (GLM 模型版本)

本项目包含基于 LangChain Models 官方文档的完整示例代码，使用智谱 AI 的 GLM 模型实现。

## 📋 目录

- [01_model_initialization.py](01_model_initialization.py) - 模型初始化和参数配置
- [02_tool_calling.py](02_tool_calling.py) - 工具调用示例
- [03_structured_output.py](03_structured_output.py) - 结构化输出示例
- [04_streaming_and_advanced.py](04_streaming_and_advanced.py) - 流式处理和高级用法

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
# 模型初始化示例
python 01_model_initialization.py

# 工具调用示例
python 02_tool_calling.py

# 结构化输出示例
python 03_structured_output.py

# 流式处理和高级用法
python 04_streaming_and_advanced.py
```

## 📚 示例说明

### 01_model_initialization.py - 模型初始化

**包含内容:**
- Chat Models 基本使用
- 模型参数配置 (temperature, max_tokens 等)
- Temperature 使用指南 (0-1)
- Max Tokens 配置
- 系统提示词使用
- 对话历史管理
- 不同模型选择
- 错误处理
- 响应元数据获取
- 流式处理预览

**核心参数:**
```python
model = ChatZhipuAI(
    model="glm-4-plus",
    temperature=0.7,    # 控制随机性 0-1
    max_tokens=1024,    # 最大输出长度
    top_p=0.9,          # 核采样
    timeout=60,         # 超时时间
    max_retries=3       # 最大重试次数
)
```

---

### 02_tool_calling.py - 工具调用

**包含内容:**
- 基本工具调用
- 完整工具调用流程
- 并行工具调用
- 顺序工具调用 (禁用并行)
- 强制工具调用
- 工具调用决策模式

**工具定义:**
```python
@tool
def get_weather(location: str) -> str:
    """获取指定位置的天气信息。

    Args:
        location: 城市名称，例如 '北京' 或 '上海'
    """
    return f"{location}的天气是晴朗，温度 22°C"
```

**使用工具:**
```python
model = ChatZhipuAI(model="glm-4-plus")
model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke([
    HumanMessage(content="北京的天气怎么样?")
])
```

---

### 03_structured_output.py - 结构化输出

**包含内容:**
- 基本结构化输出
- 复杂嵌套结构
- 列表类型输出
- Pydantic 验证器
- 数据提取示例
- 情感分析示例
- 事件提取示例

**定义结构:**
```python
class Person(BaseModel):
    """一个人的信息。"""
    name: str = Field(description="人的姓名")
    age: int = Field(description="人的年龄")
    email: str = Field(description="电子邮件地址")
    occupation: str = Field(description="职业")
```

**使用结构化输出:**
```python
model = ChatZhipuAI(model="glm-4-plus")
structured_model = model.with_structured_output(Person)

response = structured_model.invoke([
    HumanMessage(content="张伟是一位 35 岁的软件工程师")
])

# response 是 Person 实例
print(response.name, response.age)
```

---

### 04_streaming_and_advanced.py - 流式处理和高级用法

**包含内容:**
- Token 流式处理
- 异步流式处理
- 流式工具调用
- 链式调用 (Chains)
- 批处理优化
- Fallback 机制
- 重试配置
- Token 使用统计
- 监控和日志

**流式处理:**
```python
model = ChatZhipuAI(model="glm-4-plus", streaming=True)

for chunk in model.stream([HumanMessage(content="写一首诗")]):
    print(chunk.content, end="", flush=True)
```

**链式调用:**
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}"),
    ("human", "{input}")
])

chain = prompt | model | StrOutputParser()

response = chain.invoke({
    "role": "诗人",
    "input": "写一首诗"
})
```

## 🔑 GLM 模型说明

### 可用模型

- **glm-4-plus** - 推荐，性能强大，适合复杂任务
- **glm-4-flash** - 快速响应，成本低，适合简单任务
- **glm-4** - 标准版本

### 模型选择建议

| 场景 | 推荐模型 | Temperature |
|------|---------|------------|
| 数据提取、分类 | glm-4-plus | 0 |
| 客服对话、问答 | glm-4-plus | 0.3-0.5 |
| 创意写作 | glm-4-plus | 0.7-0.9 |
| 简单任务 | glm-4-flash | 0.5 |

## 💡 核心概念

### 1. Chat Models vs LLMs

**Chat Models** (推荐):
- 接受消息列表
- 支持系统消息、用户消息、AI 消息
- 原生支持工具调用
- 更好的对话管理

**LLMs**:
- 接受字符串输入
- 返回字符串输出
- 适合单轮文本生成

### 2. Temperature 控制

```python
# temperature = 0: 确定性输出
# 适用: 数据提取、分类、代码生成
model = ChatZhipuAI(temperature=0)

# temperature = 0.5: 平衡模式
# 适用: 客服对话、技术文档
model = ChatZhipuAI(temperature=0.5)

# temperature = 0.9: 高创造性
# 适用: 创意写作、头脑风暴
model = ChatZhipuAI(temperature=0.9)
```

### 3. 工具调用流程

```
用户输入 → Model推理 → 决定调用工具 → 执行工具 →
返回结果 → Model综合 → 最终响应
```

### 4. 结构化输出优势

- ✅ 保证输出格式一致
- ✅ 自动类型验证
- ✅ 易于解析和处理
- ✅ 减少错误处理代码

### 5. 流式处理好处

- ⚡ 实时响应，改善用户体验
- 📊 可以显示进度
- 🔄 支持大量文本生成

## 🎯 使用场景

### 场景 1: 数据提取

```python
class ContactInfo(BaseModel):
    name: str
    phone: str
    email: str

model = ChatZhipuAI(model="glm-4-plus", temperature=0)
structured_model = model.with_structured_output(ContactInfo)

response = structured_model.invoke([
    HumanMessage(content="张三，电话13812345678，邮箱zhang@example.com")
])
```

### 场景 2: 智能客服

```python
@tool
def search_kb(query: str) -> str:
    """搜索知识库"""
    return "找到相关文档..."

model = ChatZhipuAI(model="glm-4-plus", temperature=0.5)
model_with_tools = model.bind_tools([search_kb])

response = model_with_tools.invoke([
    SystemMessage(content="你是客服助手"),
    HumanMessage(content="如何退款?")
])
```

### 场景 3: 内容生成

```python
model = ChatZhipuAI(
    model="glm-4-plus",
    temperature=0.8,
    streaming=True
)

for chunk in model.stream([
    SystemMessage(content="你是一个创意作家"),
    HumanMessage(content="写一个科幻故事")
]):
    print(chunk.content, end="", flush=True)
```

## ⚠️ 注意事项

1. **API Key 安全**
   - 使用环境变量存储
   - 不要硬编码在代码中
   - 不要提交到版本控制

2. **成本控制**
   - 使用 `max_tokens` 限制输出
   - 选择合适的 temperature
   - 考虑使用 glm-4-flash 降低成本

3. **错误处理**
   - 使用 `max_retries` 配置重试
   - 捕获异常并处理
   - 添加超时设置

4. **性能优化**
   - 使用流式处理改善体验
   - 批处理多个请求
   - 使用异步调用提升并发

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
- Field 描述不详细

**解决方案:**
- 使用更详细的 Field 描述
- 添加示例和约束
- 使用 Pydantic 验证器

### Q4: 流式处理中断

**原因:**
- 网络问题
- 超时设置过短

**解决方案:**
- 增加 timeout 参数
- 添加重试逻辑
- 处理异常

## 📖 参考资源

- [LangChain 官方文档](https://docs.langchain.com/oss/python/langchain/models)
- [智谱 AI 文档](https://open.bigmodel.cn/dev/api)
- [Pydantic 文档](https://docs.pydantic.dev/)
- [原始总结文档](../langchain-docs/LangChain_Models_详细指南.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request!

## 📄 许可证

MIT License

---

**作者**: 基于 LangChain 官方文档改编
**版本**: 1.0
**更新日期**: 2025-01-23
