# 5分钟快速开始

> 快速上手 Hello-Agents LangChain v1.0 实现

## 🚀 三步开始

### 步骤 1: 安装依赖（1分钟）

```bash
# 进入项目目录
cd /Users/yuyansong/AiProject/Langchain/hello-agents-langchain-v1

# 安装依赖
pip install -r requirements.txt
```

### 步骤 2: 配置 API 密钥（2分钟）

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件
# 填入你的智谱AI API密钥
```

**获取 API 密钥**: https://open.bigmodel.cn/

`.env` 文件内容:
```bash
ZHIPUAI_API_KEY=你的API密钥
```

### 步骤 3: 运行示例（2分钟）

```bash
# 运行 ReAct 示例
python 01_react_agent.py

# 运行 Plan-and-Solve 示例
python 02_plan_and_solve.py

# 运行 Reflection 示例
python 03_reflection_agent.py
```

---

## 📝 使用你自己的问题

### ReAct - 动态工具调用

```python
from utils import get_llm
from tools import get_weather, calculator
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

# 1. 创建 LLM
llm = get_llm(provider="zhipuai", model="glm-4")

# 2. 定义工具
tools = [get_weather, calculator]

# 3. 创建 Agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是智能助手，可以调用工具帮助用户。"
)

# 4. 提问
result = agent.invoke({
    "messages": [HumanMessage(content="北京天气如何？")]
})

print(result["messages"][-1].content)
```

### Plan-and-Solve - 结构化规划

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

class Plan(BaseModel):
    steps: List[str] = Field(description="步骤列表")

# 规划链
parser = JsonOutputParser(pydantic_object=Plan)
prompt = ChatPromptTemplate.from_messages([
    ("system", "{format_instructions}"),
    ("human", "问题: {question}")
])

chain = prompt.partial(
    format_instructions=parser.get_format_instructions()
) | llm | parser

# 生成计划
plan = chain.invoke({"question": "你的问题"})
print(plan["steps"])
```

### Reflection - 迭代优化

```python
# 初始生成
initial_chain = initial_prompt | llm | StrOutputParser()
code = initial_chain.invoke({"task": "编写函数"})

# 反思
reflect_chain = reflect_prompt | llm | StrOutputParser()
feedback = reflect_chain.invoke({"task": "编写函数", "code": code})

# 优化
refine_chain = refine_prompt | llm | StrOutputParser()
better_code = refine_chain.invoke({
    "task": "编写函数",
    "last_code": code,
    "feedback": feedback
})
```

---

## 🔧 自定义工具

创建你自己的工具:

```python
from langchain_core.tools import tool

@tool
def my_custom_tool(input: str) -> str:
    """工具描述（会被 LLM 看到）

    Args:
        input: 参数描述
    """
    # 实现你的逻辑
    result = f"处理结果: {input}"
    return result

# 使用
tools = [my_custom_tool]
agent = create_agent(model=llm, tools=tools, ...)
```

---

## 💡 下一步

- 📖 阅读 [README.md](README.md) 了解详细信息
- 🎓 查看 [转换指南](../agent-docs/原始代码到LangChain_v1.0_转换指南.md) 学习原理
- 🔨 修改代码示例，解决你自己的问题
- 🌟 探索更多 LangChain 功能

---

## ❓ 常见问题

### Q: 如何切换到 GPT-4？

修改 `.env`:
```bash
LLM_MODEL_ID=gpt-4
LLM_API_KEY=your_openai_key
LLM_BASE_URL=https://api.openai.com/v1
```

在代码中:
```python
llm = get_llm(provider="openai", model="gpt-4")
```

### Q: 如何启用调试模式？

```python
# ReAct
agent = ReActAgent(debug=True)

# Plan-and-Solve
agent = PlanAndSolveAgent(debug=True)

# Reflection
agent = ReflectionAgent(debug=True)

# 或使用 create_agent
agent = create_agent(..., debug=True)
```

### Q: 遇到错误怎么办？

1. 检查 API 密钥是否正确
2. 确保网络连接正常
3. 查看错误信息（通常很明确）
4. 参考 [README.md](README.md) 常见问题部分

---

🎉 开始构建你的智能体吧！
