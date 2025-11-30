# LangChain Structured Output 示例集合

> 基于官方文档：https://docs.langchain.com/oss/python/langchain/structured-output
>
> 使用 GLM-4.5-air 模型实现
>
> ⚠️ **重要说明**: ChatZhipuAI 不支持 `ToolStrategy`，Agent 示例使用后处理方式实现。详见 [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)

## 📖 项目简介

本项目提供了完整的 LangChain Structured Output（结构化输出）功能示例代码，涵盖从基础到高级的所有核心功能。每个示例都可以独立运行，包含详细的注释和说明。

**实现特色**:
- ✅ Pydantic V2 兼容
- ✅ 适配 ChatZhipuAI 限制
- ✅ 完整的错误处理
- ✅ 生产环境可用

## 🎯 什么是 Structured Output？

**Structured Output（结构化输出）** 允许 LLM 返回**特定、可预测格式**的数据，而不是自然语言文本。你可以获得经过验证的结构化数据（JSON 对象、Pydantic 模型），可以直接在应用程序中使用。

**核心优势：**
- ✅ **类型安全** - 自动验证和类型检查
- ✅ **易于集成** - 直接用于下游系统
- ✅ **可靠输出** - 保证符合 schema
- ✅ **丰富验证** - Pydantic 提供强大的验证能力

## 📦 安装依赖

```bash
# 基础依赖
pip install langchain langgraph langchain-community

# 智谱 AI（GLM）
pip install zhipuai

# Pydantic（数据验证）
pip install pydantic email-validator
```

或使用 requirements.txt:

```bash
pip install -r requirements.txt
```

## 🗂️ 项目结构

```
langchain_structured_output_examples/
├── README.md                          # 本文件
├── 01_basic_model_usage.py            # Model 基础用法
├── 02_agent_usage.py                  # Agent 中使用
├── 03_real_world_applications.py      # 实际应用场景
├── 04_advanced_features.py            # 高级特性
├── 05_comprehensive_demo.py           # 综合演示
├── QUICK_REFERENCE.md                 # 快速参考
├── LEARNING_GUIDE.md                  # 学习指南
└── requirements.txt                   # 依赖列表
```

## 📚 示例说明

### 1️⃣ Model 基础用法 ([01_basic_model_usage.py](01_basic_model_usage.py))

**功能**：演示如何在 LangChain Model 中使用结构化输出

**包含示例**：
- ✅ 基础 Pydantic Model
- ✅ 嵌套结构
- ✅ 使用验证器
- ✅ 获取原始响应
- ✅ 提取多个实例

**运行**：
```bash
python 01_basic_model_usage.py
```

---

### 2️⃣ Agent 用法 ([02_agent_usage.py](02_agent_usage.py))

**功能**：演示如何在 LangChain Agent 中使用结构化输出

⚠️ **注意**: 由于 ChatZhipuAI 不支持 ToolStrategy，本文件使用**后处理方式**实现结构化输出。

**包含示例**：
- ✅ 基础 Agent 结构化输出（后处理方式）
- ✅ 复杂查询
- ✅ 多工具协作
- ✅ 带记忆的 Agent
- ✅ Pydantic 验证错误处理

**运行**：
```bash
python 02_agent_usage.py
```

**实现说明**: Agent 执行任务后，使用 `model.with_structured_output()` 对响应进行结构化提取

---

### 3️⃣ 实际应用场景 ([03_real_world_applications.py](03_real_world_applications.py))

**功能**：真实场景中的应用

**包含场景**：
- 📧 数据提取（邮件签名解析）
- 🏷️ 内容分类（新闻文章分类）
- 📝 表单填充（求职申请）
- 📊 评分系统（作文评分）
- 🛍️ 产品信息提取（电商描述）

**运行**：
```bash
python 03_real_world_applications.py
```

---

### 4️⃣ 高级特性 ([04_advanced_features.py](04_advanced_features.py))

**功能**：高级功能演示

**包含功能**：
- 🔀 动态响应格式选择
- 🛣️ 路由决策
- 🔄 多格式支持
- ⚙️ 自定义验证
- 🎯 复杂嵌套结构

**运行**：
```bash
python 04_advanced_features.py
```

---

### 5️⃣ 综合演示 ([05_comprehensive_demo.py](05_comprehensive_demo.py))

**功能**：完整的综合示例

**特点**：
- 集成所有功能
- 交互式菜单
- 完整工作流程

**运行**：
```bash
python 05_comprehensive_demo.py
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
python 01_basic_model_usage.py
python 02_agent_usage.py
# ...
```

### 3. 查看输出

每个示例都会打印详细的执行过程和结果，包括：
- 📝 输入数据
- 🤖 结构化响应
- 📊 字段详情
- 💡 说明提示

## 📊 功能对比

| 示例 | 难度 | 内容 | 适用场景 |
|------|------|------|----------|
| 01 | ⭐ | Model 基础 | 简单数据提取 |
| 02 | ⭐⭐ | Agent 用法 | 带工具的场景 |
| 03 | ⭐⭐⭐ | 实际应用 | 生产环境 |
| 04 | ⭐⭐⭐⭐ | 高级特性 | 复杂业务 |
| 05 | ⭐⭐⭐⭐⭐ | 综合示例 | 完整系统 |

## 🔧 核心概念

### Schema 类型

```python
# Pydantic Model（推荐）
class Person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")

model_with_structure = model.with_structured_output(Person)
```

### 在 Model 中使用

```python
# 创建结构化输出模型
model = ChatZhipuAI(model="glm-4.6")
model_with_structure = model.with_structured_output(Schema)

# 调用
result = model_with_structure.invoke("提取信息...")
```

### 在 Agent 中使用

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

# 创建 Agent
agent = create_agent(
    model=model,
    tools=[...],
    response_format=ToolStrategy(Schema)
)

# 调用
result = agent.invoke({"messages": [...]})
structured_data = result["structured_response"]
```

## 💡 最佳实践

### 1. 提供清晰的字段描述

```python
class Person(BaseModel):
    name: str = Field(description="全名（名和姓）")
    age: int = Field(description="年龄（整数）", ge=0, le=150)
```

### 2. 使用验证器

```python
from pydantic import validator

class Product(BaseModel):
    price: float = Field(gt=0)

    @validator('price')
    def round_price(cls, v):
        return round(v, 2)
```

### 3. 使用枚举限制选项

```python
from enum import Enum

class Status(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"

class Task(BaseModel):
    status: Status
```

### 4. 适度使用嵌套

```python
# ✅ 好：适度嵌套
class Address(BaseModel):
    city: str
    country: str

class Company(BaseModel):
    name: str
    address: Address  # 一层嵌套
```

## 🐛 常见问题

### Q1: 模型为什么没有遵循 schema？

**A**: 可能原因：
- Schema 太复杂
- 字段描述不清晰
- 需要使用更强大的模型

**解决方案**：
```python
# ✅ 简化 schema
# ✅ 添加清晰的描述
# ✅ 尝试更强大的模型
```

### Q2: 如何处理可选字段？

```python
from typing import Optional

class Schema(BaseModel):
    required_field: str
    optional_field: Optional[str] = None
```

### Q3: 如何调试？

```python
# 使用 include_raw 查看原始响应
result = model_with_structure.invoke(input, include_raw=True)
print("原始响应:", result['raw'].content)
print("解析结果:", result['parsed'])
```

## 📖 学习路径

```
第1步：基础入门（1小时）
  └─ 运行 01_basic_model_usage.py
  └─ 理解 Pydantic Model 和 Field

第2步：Agent 集成（2小时）
  └─ 运行 02_agent_usage.py
  └─ 理解 ToolStrategy 的使用

第3步：实际应用（2小时）
  └─ 运行 03_real_world_applications.py
  └─ 学习不同场景的应用

第4步：高级特性（2小时）
  └─ 运行 04_advanced_features.py
  └─ 掌握动态格式和路由

第5步：综合实战（1周+）
  └─ 运行 05_comprehensive_demo.py
  └─ 构建自己的应用
```

## 🔗 相关资源

- **官方文档**: https://docs.langchain.com/oss/python/langchain/structured-output
- **LangChain 文档**: https://python.langchain.com/docs/
- **Pydantic 文档**: https://docs.pydantic.dev/
- **智谱 AI**: https://open.bigmodel.cn/

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

---

**Happy Coding! 🚀**

如有问题，请查阅官方文档或提交 Issue。
