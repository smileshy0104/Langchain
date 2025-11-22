# 修复说明

## 问题描述

运行 `02_plan_and_solve.py` 时出现 HTTP 400 错误：
```
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/400
```

## 根本原因

智谱AI GLM-4 API 对 `ChatPromptTemplate.from_messages()` 格式的支持有限制。

使用以下格式会导致 400 错误：
```python
ChatPromptTemplate.from_messages([
    ("system", "系统提示词"),
    ("human", "用户提示词")
])
```

## 解决方案

### ✅ 修复方法

将所有 `ChatPromptTemplate.from_messages()` 改为 `ChatPromptTemplate.from_template()`：

**修改前（有问题）:**
```python
self.prompt = ChatPromptTemplate.from_messages([
    ("system", "你是AI助手..."),
    ("human", "问题: {question}")
])
```

**修改后（正确）:**
```python
self.prompt = ChatPromptTemplate.from_template("""你是AI助手...

问题: {question}

请回答:""")
```

### 📝 已修复的文件

1. **02_plan_and_solve.py**
   - `Planner` 类的提示词（第63-74行）
   - `Executor` 类的提示词（第145-167行）

2. **utils.py**
   - 模型属性访问：`llm.model` → `llm.model_name`（第136行）

## 验证修复

运行测试脚本：
```bash
python test_plan_solve.py
```

## GLM-4 API 兼容性建议

### ✅ 推荐做法

1. **使用 `from_template()`**
   ```python
   prompt = ChatPromptTemplate.from_template("提示词内容\n\n{variable}")
   ```

2. **简单的变量插值**
   ```python
   prompt = ChatPromptTemplate.from_template("""
   你是AI助手。

   问题: {question}
   历史: {history}

   请回答:
   """)
   ```

### ❌ 避免使用

1. **`from_messages()` 格式**
   ```python
   # 可能导致 400 错误
   ChatPromptTemplate.from_messages([
       ("system", "..."),
       ("human", "...")
   ])
   ```

2. **复杂的消息角色**
   ```python
   # 避免使用
   ("assistant", "..."),
   ("function", "...")
   ```

## 其他 LLM 兼容性

### OpenAI (ChatOpenAI)

- ✅ 完全支持 `from_messages()`
- ✅ 支持所有消息角色

### 智谱AI (ChatZhipuAI)

- ⚠️ `from_messages()` 支持有限
- ✅ 推荐使用 `from_template()`
- ⚠️ 某些角色可能不支持

### 通用建议

为了**最大兼容性**，建议：
1. 优先使用 `ChatPromptTemplate.from_template()`
2. 将所有内容放在一个模板字符串中
3. 使用变量 `{variable}` 进行插值

## 完整示例

### 规划器示例

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# 定义解析器
parser = JsonOutputParser(pydantic_object=Plan)

# 创建提示词（兼容 GLM-4）
prompt = ChatPromptTemplate.from_template("""你是规划专家。

{format_instructions}

问题: {question}

请输出 JSON 格式的计划:""")

# 创建链
chain = prompt.partial(
    format_instructions=parser.get_format_instructions()
) | llm | parser

# 调用
result = chain.invoke({"question": "如何做一道菜？"})
```

### 执行器示例

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 创建提示词（兼容 GLM-4）
prompt = ChatPromptTemplate.from_template("""你是执行专家。

原始问题: {question}
完整计划: {plan}
历史结果: {history}
当前步骤: {current_step}

请执行当前步骤:""")

# 创建链
chain = prompt | llm | StrOutputParser()

# 调用
result = chain.invoke({
    "question": "...",
    "plan": "...",
    "history": "...",
    "current_step": "..."
})
```

## 测试清单

- [x] 修复 `utils.py` 中的 `model` 属性问题
- [x] 修复 `02_plan_and_solve.py` 的 Planner 提示词
- [x] 修复 `02_plan_and_solve.py` 的 Executor 提示词
- [x] 创建测试脚本 `test_plan_solve.py`
- [ ] 运行完整的 `02_plan_and_solve.py` 验证
- [ ] 检查其他文件是否有类似问题

## 下一步

1. **运行测试**:
   ```bash
   python test_plan_solve.py
   ```

2. **验证修复**:
   ```bash
   python 02_plan_and_solve.py
   ```

3. **检查其他文件**:
   - `01_react_agent.py` - 检查是否使用了 `from_messages()`
   - `03_reflection_agent.py` - 检查是否使用了 `from_messages()`

---

**修复日期**: 2025-11-22
**问题**: HTTP 400 - GLM-4 API 不支持某些提示词格式
**解决方案**: 使用 `from_template()` 替代 `from_messages()`
