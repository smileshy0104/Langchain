# 🚀 LangChain Structured Output 快速参考

## 📋 基础用法

### 在 Model 中使用

```python
from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatZhipuAI

class Person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")

model = ChatZhipuAI(model="glm-4.6")
model_with_structure = model.with_structured_output(Person)

result = model_with_structure.invoke("提取：张三，28岁")
print(result)
# Person(name='张三', age=28)
```

### 在 Agent 中使用

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model=model,
    tools=[search_tool],
    response_format=ToolStrategy(Person)
)

result = agent.invoke({"messages": [...]})
person = result["structured_response"]
```

---

## 🔧 Schema 定义

### 基础类型

```python
from pydantic import BaseModel, Field

class BasicSchema(BaseModel):
    string_field: str = Field(description="字符串")
    int_field: int = Field(description="整数", ge=0, le=100)
    float_field: float = Field(description="浮点数", gt=0.0)
    bool_field: bool = Field(description="布尔值")
```

### 复杂类型

```python
from typing import List, Optional
from enum import Enum

class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ComplexSchema(BaseModel):
    # 列表
    tags: List[str] = Field(description="标签列表")

    # 可选
    description: Optional[str] = None

    # 枚举
    priority: Priority = Field(default=Priority.MEDIUM)
```

### 嵌套结构

```python
class Address(BaseModel):
    city: str
    country: str

class Person(BaseModel):
    name: str
    address: Address  # 嵌套
```

### 使用验证器

```python
from pydantic import validator, EmailStr

class Contact(BaseModel):
    name: str
    email: EmailStr  # 自动验证邮箱格式
    phone: str

    @validator('name')
    def name_must_be_capitalized(cls, v):
        if not v[0].isupper():
            raise ValueError('名字首字母必须大写')
        return v

    @validator('phone')
    def standardize_phone(cls, v):
        return ''.join(c for c in v if c.isdigit())
```

---

## 📊 常用模式

### 数据提取

```python
class ExtractedData(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None

extractor = model.with_structured_output(ExtractedData)
result = extractor.invoke("提取：张三，zhangsan@example.com，13800138000")
```

### 分类

```python
class Category(str, Enum):
    TECH = "科技"
    SPORTS = "体育"

class Classification(BaseModel):
    category: Category
    confidence: float = Field(ge=0, le=1)

classifier = model.with_structured_output(Classification)
result = classifier.invoke("这是一篇关于AI的文章")
```

### 评分

```python
class Grade(BaseModel):
    score: int = Field(ge=0, le=100)
    feedback: str

grader = model.with_structured_output(Grade)
result = grader.invoke("评分：这篇作文写得很好...")
```

---

## 🎯 Agent 策略

### ToolStrategy

适用于所有支持工具调用的模型：

```python
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model="glm-4.6",
    tools=[...],
    response_format=ToolStrategy(Schema)
)
```

### ProviderStrategy

使用提供商原生支持（如 OpenAI）：

```python
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    response_format=ProviderStrategy(Schema)
)
```

### 错误处理

```python
agent = create_agent(
    model=model,
    response_format=ToolStrategy(
        Schema,
        handle_errors="raise"  # "raise" | "return_none"
    )
)
```

---

## ⚙️ 高级特性

### 获取原始响应

```python
model_with_structure = model.with_structured_output(
    Schema,
    include_raw=True
)

result = model_with_structure.invoke(input)
print(result['parsed'])  # 解析后的数据
print(result['raw'])     # 原始响应
```

### 多个实例提取

```python
class Person(BaseModel):
    name: str
    age: int

class People(BaseModel):
    persons: List[Person]

extractor = model.with_structured_output(People)
result = extractor.invoke("Alice 28岁，Bob 35岁")
```

---

## 💡 最佳实践

### ✅ 好的做法

```python
# 清晰的描述
class Good(BaseModel):
    name: str = Field(description="全名（名和姓）")
    age: int = Field(description="年龄（整数）", ge=0, le=150)

# 使用枚举
class Status(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

# 适度验证
@validator('email')
def validate_email(cls, v):
    if '@' not in v:
        raise ValueError('无效邮箱')
    return v.lower()
```

### ❌ 避免的做法

```python
# 缺少描述
class Bad(BaseModel):
    name: str  # ❌ 没有描述
    age: int   # ❌ 没有约束

# 过度嵌套
class A(BaseModel):
    b: 'B'  # ❌ 多层嵌套难以生成

class B(BaseModel):
    c: 'C'

# 过于宽松
class Loose(BaseModel):
    data: dict  # ❌ 太宽松，失去类型安全
```

---

## 🐛 常见错误处理

### 验证错误

```python
from pydantic import ValidationError

try:
    result = model_with_structure.invoke(input)
except ValidationError as e:
    for error in e.errors():
        print(f"字段: {error['loc']}")
        print(f"错误: {error['msg']}")
```

### 调试技巧

```python
# 1. 查看原始响应
result = model_with_structure.invoke(input, include_raw=True)
print("原始内容:", result['raw'].content)

# 2. 简化 Schema
# 从简单开始，逐步增加复杂度

# 3. 添加详细描述
# 描述越清晰，模型遵循得越好
```

---

## 📈 性能优化

### 选择合适的策略

```python
# 优先使用 ProviderStrategy（如果支持）
response_format = ProviderStrategy(Schema)  # ✅ 更可靠

# 回退到 ToolStrategy
response_format = ToolStrategy(Schema)  # ✅ 更广泛支持
```

### 缓存模型实例

```python
# ✅ 好：只创建一次
model_with_structure = model.with_structured_output(Schema)

# 重复使用
result1 = model_with_structure.invoke(input1)
result2 = model_with_structure.invoke(input2)

# ❌ 避免：每次都创建
result = model.with_structured_output(Schema).invoke(input)
```

### 简化 Schema

```python
# ✅ 好：简单清晰
class Simple(BaseModel):
    title: str
    score: int

# ❌ 避免：过度复杂
class Complex(BaseModel):
    nested: dict[str, list[dict[str, Any]]]
```

---

## 🔍 快速诊断

### 问题：模型不遵循 Schema

**检查清单：**
- [ ] Schema 是否太复杂？
- [ ] 字段描述是否清晰？
- [ ] 是否使用了合适的模型？
- [ ] 是否尝试了不同的 method？

### 问题：验证失败

**检查清单：**
- [ ] 字段约束是否太严格？
- [ ] 是否使用了正确的类型？
- [ ] 验证器逻辑是否正确？

### 问题：性能慢

**检查清单：**
- [ ] 是否缓存了模型实例？
- [ ] Schema 是否可以简化？
- [ ] 是否选择了合适的策略？

---

## 📚 完整示例模板

```python
import os
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum
from langchain_community.chat_models import ChatZhipuAI

# 设置 API Key
os.environ["ZHIPUAI_API_KEY"] = "your-key"

# 定义 Schema
class MySchema(BaseModel):
    """Schema 描述"""
    field1: str = Field(description="字段1描述")
    field2: int = Field(description="字段2描述", ge=0)

    @validator('field1')
    def validate_field1(cls, v):
        return v.strip().lower()

# 创建模型
model = ChatZhipuAI(model="glm-4.6")
model_with_structure = model.with_structured_output(MySchema)

# 调用
result = model_with_structure.invoke("你的输入")
print(result)
```

---

**快速参考版本：v1.0**

**最后更新：2024-11-30**
