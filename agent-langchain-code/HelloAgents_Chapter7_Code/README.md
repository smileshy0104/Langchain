# LangChain v1.0 实现 - Hello Agents 第七章：构建你的 Agent 框架

本目录包含使用 **LangChain v1.0** 重新实现 Hello-Agents 第七章的核心内容，展示如何基于 LangChain 构建自己的 Agent 框架。

## 📁 目录结构

```
agent-langchain-code/HelloAgents_Chapter7_Code/
├── README.md                                    # 本文档
├── .env.example                                 # 环境变量示例
├── quick_test.py                                # 快速测试脚本
│
├── core/                                        # 核心框架层
│   ├── __init__.py
│   ├── tools.py                                 # 工具基类和注册机制
│   ├── agents.py                                # Agent 基类
│   └── utils.py                                 # 工具函数
│
├── tools/                                       # 工具实现
│   ├── __init__.py
│   ├── calculator_tool.py                       # 计算器工具
│   └── search_tool.py                          # 搜索工具
│
├── agents/                                      # Agent 实现
│   ├── __init__.py
│   ├── simple_agent_langchain.py               # 简单对话 Agent
│   ├── react_agent_langchain.py                # ReAct Agent
│   ├── reflection_agent_langchain.py           # 反思 Agent
│   └── plan_solve_agent_langchain.py           # 计划执行 Agent
│
└── examples/                                    # 示例代码
    ├── example_simple_agent.py
    ├── example_react_agent.py
    ├── example_reflection_agent.py
    └── example_plan_solve_agent.py
```

## 🎯 核心特性

### 基于 LangChain v1.0 的实现

本章重新实现了 Hello-Agents 框架，但使用 LangChain 的标准组件：

- ✅ **LangChain Tools**: 使用 `@tool` 装饰器和 `BaseTool`
- ✅ **LCEL 链式调用**: 使用 LangChain Expression Language
- ✅ **智谱AI GLM-4.6**: 通过 `ChatZhipuAI` 集成
- ✅ **统一接口**: 所有 Agent 遵循相同的接口规范
- ✅ **模块化设计**: 工具、Agent、配置分离

### 四种 Agent 实现

| Agent 类型 | 文件 | 特点 | 适用场景 |
|-----------|------|------|----------|
| **SimpleAgent** | `agents/simple_agent_langchain.py` | 基础对话 | 简单问答、知识查询 |
| **ReActAgent** | `agents/react_agent_langchain.py` | 推理-行动循环 | 需要工具辅助的任务 |
| **ReflectionAgent** | `agents/reflection_agent_langchain.py` | 自我反思改进 | 需要高质量输出 |
| **PlanAndSolveAgent** | `agents/plan_solve_agent_langchain.py` | 计划后执行 | 复杂多步骤任务 |

## 🚀 快速开始

### 1. 环境准备

安装依赖：

```bash
# 核心依赖
pip install langchain langchain-community langchain-core
pip install python-dotenv pydantic

# 智谱AI SDK
pip install zhipuai

# 可选：搜索工具依赖
pip install tavily-python serpapi
```

### 2. 配置 API 密钥

复制 `.env.example` 为 `.env` 并填写：

```bash
# 必需：智谱AI API Key
ZHIPUAI_API_KEY=your-zhipuai-api-key

# 可选：搜索 API
TAVILY_API_KEY=your-tavily-key
SERPAPI_API_KEY=your-serpapi-key
```

获取 API 密钥：
- 智谱AI: https://open.bigmodel.cn/
- Tavily: https://tavily.com/
- SerpAPI: https://serpapi.com/

### 3. 快速验证

运行快速测试脚本：

```bash
cd agent-langchain-code/HelloAgents_Chapter7_Code
python quick_test.py
```

### 4. 运行示例

```bash
# SimpleAgent 示例
python examples/example_simple_agent.py

# ReActAgent 示例
python examples/example_react_agent.py

# ReflectionAgent 示例
python examples/example_reflection_agent.py

# PlanAndSolveAgent 示例
python examples/example_plan_solve_agent.py
```

### 5. 完成课后习题

```bash
# 运行所有习题
cd exercises
python run_all_exercises.py

# 或运行单个习题
python exercise_02_new_model_provider.py  # 多模型支持
python exercise_04_custom_tools.py        # 自定义工具
python exercise_05_plugin_system.py       # 插件系统
```

详见 [习题总结](EXERCISES_SUMMARY.md)

## 📖 核心组件说明

### 1. 工具系统 (core/tools.py)

使用 LangChain 的标准工具接口：

```python
from langchain.tools import BaseTool
from langchain_core.tools import tool

# 方式1: 使用装饰器
@tool
def calculator(expression: str) -> str:
    """执行数学计算"""
    return str(eval(expression))

# 方式2: 继承 BaseTool
class SearchTool(BaseTool):
    name = "search"
    description = "搜索互联网信息"

    def _run(self, query: str) -> str:
        # 实现搜索逻辑
        pass
```

### 2. Agent 基类 (core/agents.py)

统一的 Agent 接口：

```python
from abc import ABC, abstractmethod
from langchain_community.chat_models import ChatZhipuAI

class BaseAgent(ABC):
    def __init__(self, llm: ChatZhipuAI, tools: list = None):
        self.llm = llm
        self.tools = tools or []
        self.history = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """执行 Agent（子类必须实现）"""
        pass
```

### 3. SimpleAgent - 基础对话

```python
from core.agents import BaseAgent

class SimpleAgent(BaseAgent):
    def run(self, input_text: str, **kwargs) -> str:
        # 构建消息
        messages = [
            {"role": "system", "content": "你是一个友好的AI助手"},
            {"role": "user", "content": input_text}
        ]

        # 调用 LLM
        response = self.llm.invoke(messages)
        return response.content
```

### 4. ReActAgent - 推理与行动

基于 LangChain 的 `create_react_agent` 或自定义实现：

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

class ReActAgent(BaseAgent):
    def __init__(self, llm, tools):
        super().__init__(llm, tools)

        # 使用 LangChain Hub 的 ReAct 提示词
        prompt = hub.pull("hwchase17/react")

        # 创建 agent
        agent = create_react_agent(llm, tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools)

    def run(self, input_text: str, **kwargs) -> str:
        result = self.agent_executor.invoke({"input": input_text})
        return result["output"]
```

## 🔧 与 Hello-Agents 框架的对应关系

| Hello-Agents 组件 | LangChain v1.0 对应 | 说明 |
|------------------|-------------------|------|
| `HelloAgentsLLM` | `ChatZhipuAI` | LLM 调用封装 |
| `Tool` 基类 | `BaseTool` / `@tool` | 工具定义 |
| `ToolRegistry` | `tools` 列表 | 工具管理 |
| `Agent` 基类 | 自定义 `BaseAgent` | Agent 接口 |
| `SimpleAgent` | 基于 `ChatZhipuAI` 的简单实现 | 基础对话 |
| `ReActAgent` | `create_react_agent` | 推理行动 |
| `Message` 类 | LangChain 消息格式 | 消息系统 |

## 💡 设计思路

### 1. 为什么使用 LangChain？

虽然 Hello-Agents 教程强调从零构建，但在实际生产中：

- ✅ **标准化**：LangChain 提供了行业标准的接口
- ✅ **生态丰富**：大量预构建的工具和集成
- ✅ **社区支持**：活跃的社区和文档
- ✅ **生产就绪**：经过大规模验证

### 2. 保留 Hello-Agents 的核心理念

- ✅ **轻量级**：只使用必要的 LangChain 组件
- ✅ **教学友好**：代码清晰，易于理解
- ✅ **渐进式**：从简单到复杂
- ✅ **统一抽象**：所有 Agent 遵循相同接口

### 3. 两种框架的选择建议

**使用 Hello-Agents 如果你想**：
- 深入理解 Agent 工作原理
- 完全掌控每一行代码
- 学习系统设计
- 构建特定领域的轻量框架

**使用 LangChain 如果你想**：
- 快速开发生产级应用
- 利用丰富的工具生态
- 与其他 LangChain 项目集成
- 遵循行业标准实践

## 🎓 学习路径

### 对于初学者

1. ✅ 先学习 Hello-Agents 第七章原始实现
2. ✅ 理解核心概念（Agent、Tool、Registry）
3. ✅ 对比本目录的 LangChain 实现
4. ✅ 理解两种方式的优缺点

### 对于进阶者

1. ✅ 深入研究 LangChain 源码
2. ✅ 对比不同框架的设计模式
3. ✅ 根据需求选择合适的技术栈
4. ✅ 构建自己的最佳实践

## 🔗 相关资源

- **Hello-Agents 原始代码**: https://github.com/datawhalechina/hello-agents/tree/main/code/chapter7
- **LangChain 文档**: https://python.langchain.com/
- **智谱AI 文档**: https://open.bigmodel.cn/dev/api
- **第六章 LangChain 实现**: ../HelloAgents_Chapter6_Code/

## 📝 核心差异对比

### Hello-Agents 框架

```python
# Hello-Agents 方式
from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import CalculatorTool

llm = HelloAgentsLLM()
registry = ToolRegistry()
registry.register_tool(CalculatorTool())

agent = SimpleAgent(name="助手", llm=llm, tool_registry=registry)
result = agent.run("计算 2 + 3")
```

### LangChain 实现

```python
# LangChain 方式
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.tools import tool
from agents.simple_agent_langchain import SimpleAgent

@tool
def calculator(expression: str) -> str:
    """计算器工具"""
    return str(eval(expression))

llm = ChatZhipuAI(model="glm-4-plus")
agent = SimpleAgent(llm=llm, tools=[calculator])
result = agent.run("计算 2 + 3")
```

**核心区别**：
- Hello-Agents 有自己的工具注册机制
- LangChain 直接使用工具列表
- Hello-Agents 强调教学和理解
- LangChain 强调实用和生态

## ⚠️ 注意事项

1. **API 兼容性**：确保使用 LangChain >= 0.1.0
2. **模型调用**：所有示例默认使用智谱AI GLM-4-Plus
3. **工具依赖**：搜索工具需要额外的 API 密钥
4. **Python 版本**：需要 Python >= 3.10

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**作者**: 基于 Hello-Agents 第七章改编
**版本**: 1.0
**更新日期**: 2025-01-23
**LangChain 版本**: 1.0+
