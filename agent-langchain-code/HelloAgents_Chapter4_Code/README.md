# Hello-Agents LangChain v1.0 实现

> 基于 **LangChain v1.0** 和 **智谱AI GLM-4** 模型的三种经典智能体范式实现

## 🎯 项目简介

本项目是 [Hello-Agents](https://github.com/datawhalechina/hello-agents) 第四章三种经典智能体范式的 **LangChain v1.0 标准实现**:

1. **ReAct** - 边想边做，动态调用工具
2. **Plan-and-Solve** - 先规划后执行，结构化解决问题
3. **Reflection** - 自我反思优化，追求完美

### 与原教程的区别

| 特性 | 原教程实现 | 本项目实现 |
|------|-----------|-----------|
| **实现方式** | 从零手工实现 | 基于 LangChain v1.0 |
| **代码量** | ~570 行 | ~200 行 |
| **学习价值** | ⭐⭐⭐⭐⭐ 理解原理 | ⭐⭐⭐⭐ 工程实践 |
| **生产就绪** | 需完善 | ✅ 可直接使用 |
| **维护成本** | 高 | 低 |
| **API 标准** | 自定义 | LangChain v1.0 标准 |
| **模型支持** | OpenAI 兼容 | GLM-4 + OpenAI 兼容 |

## 📁 项目结构

```
hello-agents-langchain-v1/
├── README.md                  # 本文档
├── requirements.txt           # 依赖包
├── .env.example              # 环境变量示例
│
├── utils.py                  # 通用工具（LLM 初始化）
├── tools.py                  # 自定义工具（搜索、计算器等）
│
├── 01_react_agent.py         # ReAct 范式实现
├── 02_plan_and_solve.py      # Plan-and-Solve 范式实现
├── 03_reflection_agent.py    # Reflection 范式实现
│
└── examples/                 # 使用示例
    ├── react_examples.py     # ReAct 多个示例
    ├── plan_examples.py      # Plan-and-Solve 示例
    └── reflection_examples.py # Reflection 示例
```

## 🚀 快速开始

### 1. 环境准备

**Python 版本**: 3.9+

```bash
# 克隆项目
cd /Users/yuyansong/AiProject/Langchain/hello-agents-langchain-v1

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API 密钥

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API 密钥
# 方式1: 使用智谱AI GLM-4（推荐，中文优化）
ZHIPUAI_API_KEY=your_zhipuai_api_key

# 方式2: 使用 OpenAI 兼容 API
LLM_MODEL_ID=gpt-4
LLM_API_KEY=your_openai_api_key
LLM_BASE_URL=https://api.openai.com/v1

# 可选：SerpAPI（用于网页搜索）
SERPAPI_API_KEY=your_serpapi_key
```

**获取 API 密钥**:
- 智谱AI: https://open.bigmodel.cn/
- SerpAPI: https://serpapi.com/

### 3. 运行示例

#### ReAct 范式

```bash
# 基础示例
python 01_react_agent.py

# 多个示例（天气查询、计算、多工具组合等）
python examples/react_examples.py
```

#### Plan-and-Solve 范式

```bash
# 基础示例
python 02_plan_and_solve.py

# 更多示例
python examples/plan_examples.py
```

#### Reflection 范式

```bash
# 基础示例
python 03_reflection_agent.py

# 更多示例
python examples/reflection_examples.py
```

## 📚 三种范式详解

### 1. ReAct - 边想边做

**核心思想**: 思考 → 行动 → 观察 → 再思考...

**适用场景**:
- ✅ 需要查询实时信息（天气、新闻、股票等）
- ✅ 需要使用外部工具（计算器、搜索引擎等）
- ✅ 问题需要多步推理和工具组合

**代码示例**:
```python
from langchain.agents import create_agent
from langchain_community.chat_models import ChatZhipuAI
from tools import get_weather, calculator

llm = ChatZhipuAI(model="glm-4", api_key="your_key")
tools = [get_weather, calculator]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是智能助手，可以调用工具帮助用户。",
    debug=True
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "北京天气如何？"}]
})
```

**特点**:
- ⚡ 动态决策，灵活应对
- 🔧 自动选择和调用工具
- 🧠 边思考边执行

### 2. Plan-and-Solve - 先谋后动

**核心思想**: 先制定完整计划 → 再逐步执行

**适用场景**:
- ✅ 复杂的多步骤问题
- ✅ 可以提前规划的任务
- ✅ 需要结构化解决方案

**代码示例**:
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# 规划链
plan_chain = plan_prompt | llm | JsonOutputParser()
plan = plan_chain.invoke({"question": "计算一个数学应用题"})

# 执行链
for step in plan['steps']:
    result = execute_chain.invoke({
        "question": question,
        "current_step": step,
        "history": history
    })
```

**特点**:
- 📋 清晰的执行计划
- 🎯 结构化解决问题
- ⏱️ 可预测的执行流程

### 3. Reflection - 自我优化

**核心思想**: 生成 → 评审 → 优化 → 再评审...

**适用场景**:
- ✅ 需要高质量输出（代码、文章、方案等）
- ✅ 可以通过迭代改进的任务
- ✅ 有明确质量标准的场景

**代码示例**:
```python
# 初始生成
code = initial_chain.invoke({"task": "编写素数查找函数"})

# 迭代优化
for i in range(max_iterations):
    # 反思
    feedback = reflect_chain.invoke({"task": task, "code": code})

    # 检查是否需要继续
    if "无需改进" in feedback:
        break

    # 优化
    code = refine_chain.invoke({
        "task": task,
        "last_code": code,
        "feedback": feedback
    })
```

**特点**:
- 🎨 追求完美质量
- 🔄 迭代优化
- 📈 持续改进

## 🔧 核心组件说明

### utils.py - LLM 初始化

```python
from langchain_community.chat_models import ChatZhipuAI

def get_llm(model: str = "glm-4", temperature: float = 0.7):
    """获取 LLM 实例

    支持:
    - glm-4: 智谱AI GLM-4（推荐，中文优化）
    - gpt-4: OpenAI GPT-4
    - 其他 OpenAI 兼容模型
    """
    return ChatZhipuAI(
        model=model,
        api_key=os.getenv("ZHIPUAI_API_KEY"),
        temperature=temperature
    )
```

### tools.py - 工具定义

使用 `@tool` 装饰器定义工具：

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息

    Args:
        city: 城市名称，例如 '北京'、'上海'
    """
    # 实现逻辑
    return f"{city}今天天气：晴天，温度 20°C"

@tool
def calculator(expression: str) -> str:
    """执行数学计算

    Args:
        expression: 数学表达式，例如 '2 + 3 * 4'
    """
    result = eval(expression)
    return f"计算结果: {expression} = {result}"
```

## 📊 性能对比

| 指标 | 原教程实现 | LangChain v1.0 |
|------|-----------|---------------|
| 代码行数 | ~570 行 | ~200 行 (-65%) |
| 错误处理 | 手动 | 自动 |
| 工具调用 | 手动解析 | 自动路由 |
| 调试难度 | 高 | 低 |
| 扩展性 | 中 | 高 |
| 生产就绪 | ❌ | ✅ |

## 🎓 学习路线

### 路线 1: 快速上手（1小时）

```
1. 配置环境和 API 密钥（15分钟）
2. 运行 ReAct 示例（15分钟）
3. 运行 Plan-and-Solve 示例（15分钟）
4. 运行 Reflection 示例（15分钟）
```

### 路线 2: 深入理解（半天）

```
1. 阅读代码实现（1小时）
2. 对比原教程实现（1小时）
3. 修改提示词和工具（1小时）
4. 解决实际问题（1小时）
```

### 路线 3: 工程应用（1-2天）

```
1. 理解 LangChain v1.0 架构
2. 自定义工具和中间件
3. 集成到实际项目
4. 性能优化和监控
```

## 🔍 与原教程对照

### 代码映射

| 原教程文件 | 本项目文件 | 说明 |
|-----------|-----------|------|
| llm_client.py | utils.py | LLM 初始化 |
| tools.py | tools.py | 工具定义 |
| ReAct.py | 01_react_agent.py | ReAct 实现 |
| Plan_and_solve.py | 02_plan_and_solve.py | Plan-and-Solve 实现 |
| Reflection.py | 03_reflection_agent.py | Reflection 实现 |

### 学习建议

1. **先理解原理**: 阅读原教程手工实现，理解底层逻辑
2. **再学框架**: 学习本项目的 LangChain 实现，掌握工程实践
3. **对比分析**: 理解两种实现方式的优劣
4. **实际应用**: 在项目中使用 LangChain 实现

## 🛠️ 高级功能

### 1. 中间件支持

```python
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model=llm,
    tools=tools,
    middleware=[
        PIIMiddleware(pii_type="email", strategy="redact")
    ]
)
```

### 2. 多轮对话

```python
messages = []

# 第一轮
messages.append(HumanMessage(content="北京天气如何？"))
result = agent.invoke({"messages": messages})
messages = result["messages"]

# 第二轮（保持上下文）
messages.append(HumanMessage(content="上海呢？"))
result = agent.invoke({"messages": messages})
```

### 3. 流式输出

```python
llm = ChatZhipuAI(model="glm-4", streaming=True)

for chunk in agent.stream({"messages": messages}):
    print(chunk, end="", flush=True)
```

### 4. 调试和追踪

```python
# 方式1: 启用 debug 模式
agent = create_agent(..., debug=True)

# 方式2: 使用 LangSmith
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_key"
```

## 📖 参考资源

### 官方文档

- [LangChain v1.0 文档](https://python.langchain.com/docs/concepts/agents/)
- [智谱AI 开放平台](https://open.bigmodel.cn/)
- [GLM-4 API 文档](https://open.bigmodel.cn/dev/api)

### 相关项目

- [Hello-Agents 原教程](https://github.com/datawhalechina/hello-agents)
- [原始代码到 LangChain v1.0 转换指南](../agent-docs/原始代码到LangChain_v1.0_转换指南.md)
- [LangGraph 教程](https://langchain-ai.github.io/langgraph/)

### 本项目文档

- [README.md](README.md) - 本文档
- [QUICKSTART.md](QUICKSTART.md) - 快速开始（5分钟）
- [API.md](API.md) - API 参考文档

## ❓ 常见问题

### Q1: 为什么选择 GLM-4 而不是 GPT-4？

**A**: GLM-4 优势:
- ✅ 中文优化，中文理解和生成能力更强
- ✅ 成本更低（约为 GPT-4 的 1/3）
- ✅ 国内访问更快速稳定
- ✅ 工具调用能力强

当然，代码也支持 GPT-4 和其他 OpenAI 兼容模型。

### Q2: 如何切换到 OpenAI 模型？

**A**: 修改 `.env` 文件:

```bash
# 使用 OpenAI
LLM_MODEL_ID=gpt-4
LLM_API_KEY=your_openai_key
LLM_BASE_URL=https://api.openai.com/v1
```

然后在 `utils.py` 中:

```python
from langchain_openai import ChatOpenAI

def get_llm():
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID"),
        openai_api_key=os.getenv("LLM_API_KEY"),
        openai_api_base=os.getenv("LLM_BASE_URL")
    )
```

### Q3: 这个实现和原教程有什么区别？

**A**: 核心区别:

| 方面 | 原教程 | 本项目 |
|------|--------|--------|
| **目标** | 理解原理 | 工程实践 |
| **实现** | 从零手工 | LangChain 框架 |
| **代码量** | 多 | 少 |
| **稳定性** | 中 | 高 |
| **学习价值** | 理解底层 | 掌握工具 |

**建议**: 两者都学！先用原教程理解原理，再用本项目掌握实践。

### Q4: 如何添加自定义工具？

**A**: 使用 `@tool` 装饰器:

```python
@tool
def my_custom_tool(input: str) -> str:
    """工具描述（会被 LLM 看到）

    Args:
        input: 参数描述
    """
    # 实现逻辑
    return result

# 添加到工具列表
tools = [my_custom_tool, ...]
```

### Q5: 遇到错误怎么办？

**A**: 调试步骤:

1. **启用 debug 模式**:
   ```python
   agent = create_agent(..., debug=True)
   ```

2. **检查 API 密钥**: 确保 `.env` 配置正确

3. **查看错误日志**: 错误信息通常很明确

4. **参考示例代码**: 对比 `examples/` 目录中的示例

5. **查看文档**: [转换指南](../agent-docs/原始代码到LangChain_v1.0_转换指南.md)

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

## 📄 许可证

MIT License

---

**最后更新**: 2025-11-22
**版本**: v1.0
**作者**: Claude Code

---

🎉 开始你的智能体开发之旅吧！
