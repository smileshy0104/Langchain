# LangChain Structured Output 实现说明

## 📌 项目状态

**创建日期**: 2024-11-30
**当前版本**: v1.0
**兼容性**: Pydantic V2, LangChain >= 0.3.0

---

## ⚠️ 重要发现：ChatZhipuAI 限制

### 问题描述

在实现过程中发现 **ChatZhipuAI 模型不支持 `ToolStrategy`**，原因是：

```
❌ 错误: ChatZhipuAI currently only supports `auto` tool choice
```

### 技术原因

- `ToolStrategy` 需要模型支持强制工具调用（forced tool calling）
- ChatZhipuAI 只支持 `auto` 模式的工具选择
- 这意味着模型可以自行决定是否调用工具，而不能强制使用特定工具返回结构化输出

### 解决方案

本项目采用了 **后处理方式（Post-processing Approach）** 实现 Agent 的结构化输出：

```python
# ❌ 不可用：直接使用 ToolStrategy
agent = create_agent(
    model=model,
    tools=[...],
    response_format=ToolStrategy(Schema)  # ChatZhipuAI 不支持
)

# ✅ 可用：后处理方式
agent = create_agent(
    model=model,
    tools=[...]
)

# 获取 Agent 响应
result = agent.invoke({"messages": [...]})
agent_response = result['messages'][-1].content

# 使用 Model 的 structured output 提取结构化数据
model_with_structure = model.with_structured_output(Schema)
structured_data = model_with_structure.invoke(f"提取以下内容的结构化信息：\n{agent_response}")
```

---

## 📂 文件更新记录

### 1. `01_basic_model_usage.py`

**更新内容**:
- ✅ 修复 Pydantic V2 兼容性（`@validator` → `@field_validator`）
- ✅ 嵌套结构字段改为可选（`default_factory=list`）
- ✅ API Key 配置改为从环境变量读取

**关键修改**:
```python
# Before
@validator('name')
def name_must_be_capitalized(cls, v):
    ...

# After
@field_validator('name')
@classmethod
def name_must_be_capitalized(cls, v):
    ...
```

```python
# Before
cast: List[Actor] = Field(description="主要演员阵容")

# After
cast: List[Actor] = Field(default_factory=list, description="主要演员阵容，至少列出2-3位主演")
```

### 2. `02_agent_usage.py`

**重大重构**:
- ⚠️ 移除了 `ToolStrategy` 的使用
- ✅ 改为后处理方式实现结构化输出
- ✅ 添加了文件头部说明，解释限制原因
- ✅ 更新了所有 5 个示例

**新增说明**:
```python
"""
注意：ChatZhipuAI 模型目前不支持 ToolStrategy，因为它只支持 'auto' 工具选择。
本示例使用直接的 Model.with_structured_output() 方法演示结构化输出。
如需在 Agent 中使用 ToolStrategy，请使用 OpenAI 等支持该功能的模型。
"""
```

### 3. `03_real_world_applications.py`

**更新内容**:
- ✅ 修复 Pydantic V2 兼容性
- ✅ API Key 配置改为从环境变量读取
- ✅ 更新验证器语法

---

## 🔧 技术细节

### Pydantic V2 迁移

| 特性 | Pydantic V1 | Pydantic V2 |
|------|-------------|-------------|
| 验证器装饰器 | `@validator` | `@field_validator` |
| 方法签名 | `def validate(cls, v)` | `@classmethod`<br>`def validate(cls, v)` |
| 导入 | `from pydantic import validator` | `from pydantic import field_validator` |

### 嵌套结构处理

**问题**: LLM 模型难以一次性生成复杂的嵌套结构

**解决方案**: 使用可选字段 + 默认值

```python
# 严格模式（可能失败）
cast: List[Actor] = Field(description="演员列表")

# 宽松模式（推荐）
cast: List[Actor] = Field(
    default_factory=list,
    description="演员列表，至少列出2-3位主演"
)
```

**优势**:
- ✅ 避免验证错误
- ✅ 通过描述引导模型填充数据
- ✅ 即使模型未返回也不会报错

---

## 🎯 使用建议

### 选择合适的方法

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 单次结构化输出 | `model.with_structured_output()` | 简单直接，完全支持 |
| Agent + 结构化输出（GLM） | 后处理方式 | GLM 不支持 ToolStrategy |
| Agent + 结构化输出（OpenAI） | `ToolStrategy` | 原生支持，更可靠 |

### 模型选择

**支持 ToolStrategy 的模型**:
- ✅ OpenAI (gpt-4, gpt-3.5-turbo)
- ✅ Anthropic Claude
- ✅ Google Gemini

**不支持 ToolStrategy 的模型**:
- ❌ ChatZhipuAI (GLM-4.x)
- 需要使用后处理方式

---

## 📊 示例概览

### Model 示例（01_basic_model_usage.py）

| 示例 | 功能 | 状态 |
|------|------|------|
| 1.1 | 基础 Pydantic Model | ✅ |
| 1.2 | 嵌套结构 | ✅ 已修复 |
| 1.3 | 字段验证器 | ✅ 已更新到 V2 |
| 1.4 | 获取原始响应 | ✅ |
| 1.5 | 提取多个实例 | ✅ |

### Agent 示例（02_agent_usage.py）

| 示例 | 原实现 | 新实现 | 状态 |
|------|--------|--------|------|
| 2.1 | ToolStrategy | 后处理 | ✅ 已重构 |
| 2.2 | ToolStrategy | 后处理 | ✅ 已重构 |
| 2.3 | ToolStrategy | 后处理 | ✅ 已重构 |
| 2.4 | ToolStrategy | 后处理 | ✅ 已重构 |
| 2.5 | ToolStrategy 错误处理 | Pydantic 验证 | ✅ 已重构 |

### 实际应用（03_real_world_applications.py）

| 场景 | 功能 | 状态 |
|------|------|------|
| 3.1 | 数据提取 | ✅ |
| 3.2 | 内容分类 | ✅ |
| 3.3 | 表单填充 | ✅ 已更新验证器 |
| 3.4 | 评分系统 | ✅ |
| 3.5 | 产品信息提取 | ✅ |

---

## 🐛 已知问题

### 1. API 认证问题

**症状**: `401 Unauthorized` 错误

**原因**: API Key 配置问题

**解决方案**:
```bash
# 设置环境变量
export ZHIPUAI_API_KEY="your-actual-api-key"

# 或在代码中（不推荐）
os.environ["ZHIPUAI_API_KEY"] = "your-actual-api-key"
```

### 2. 嵌套结构验证失败

**症状**: `Field required` 错误

**原因**: 模型未返回所有必需字段

**解决方案**: 使用 `default_factory` 或 `Optional`

---

## 📝 开发日志

### 2024-11-30

**初始创建**:
- ✅ 创建项目结构
- ✅ 实现 15 个示例
- ✅ 编写完整文档

**问题修复**:
- 🔧 Pydantic V1 → V2 迁移
- 🔧 ChatZhipuAI ToolStrategy 限制
- 🔧 嵌套结构验证错误
- 🔧 API Key 配置

**重构**:
- 🔄 Agent 示例全部改为后处理方式
- 🔄 更新所有验证器语法
- 🔄 优化嵌套结构定义

---

## 🚀 未来改进

### 短期

- [ ] 添加 OpenAI 模型的 ToolStrategy 示例（对比）
- [ ] 添加更多错误处理示例
- [ ] 添加性能对比测试

### 中期

- [ ] 支持流式输出（Streaming）
- [ ] 添加批量处理示例
- [ ] 创建交互式 Notebook

### 长期

- [ ] 支持多模态输入（图片+文本）
- [ ] 添加缓存优化
- [ ] 集成向量数据库

---

## 📚 参考资料

- [LangChain Structured Output 官方文档](https://docs.langchain.com/oss/python/langchain/structured-output)
- [Pydantic V2 迁移指南](https://docs.pydantic.dev/latest/migration/)
- [智谱 AI 开放平台](https://open.bigmodel.cn/)

---

## 💡 最佳实践总结

1. **优先使用 Model.with_structured_output()**
   - 所有模型都支持
   - 实现简单
   - 效果稳定

2. **Schema 设计原则**
   - 清晰的字段描述
   - 适度的验证约束
   - 嵌套层级不超过 2-3 层

3. **错误处理**
   - 使用 try-except 捕获 ValidationError
   - 为可选字段提供默认值
   - 记录失败案例便于调试

4. **性能优化**
   - 缓存 model_with_structure 实例
   - 简化 Schema 结构
   - 使用合适的模型温度参数

---

**文档版本**: v1.0
**最后更新**: 2024-11-30
**维护者**: LangChain Structured Output 项目组
