# LangChain v1.0 实现 - Hello Agents 第七章：构建你的 Agent 框架

本目录包含使用 **LangChain v1.0** 重新实现 Hello-Agents 第七章的核心内容，展示如何基于 LangChain 构建自己的 Agent 框架。

## 📁 目录结构

```
agent-langchain-code/HelloAgents_Chapter7_Code/
├── README.md                                    # 本文档
├── EXERCISES_SUMMARY.md                         # 习题总结文档
├── .env.example                                 # 环境变量示例
├── .env                                         # 环境变量配置（需自行创建）
├── quick_test.py                                # 快速测试脚本
│
├── core/                                        # 核心框架层
│   ├── __init__.py
│   ├── tools.py                                 # 工具基类和注册机制
│   ├── agents.py                                # Agent 基类
│   └── utils.py                                 # 工具函数（含环境变量加载）
│
├── tools/                                       # 工具实现
│   ├── __init__.py
│   ├── calculator_tool.py                       # 计算器工具
│   └── search_tool.py                          # 搜索工具（Mock + 真实搜索）
│
├── agents/                                      # Agent 实现
│   ├── __init__.py
│   ├── simple_agent_langchain.py               # 简单对话 Agent
│   ├── react_agent_langchain.py                # ReAct Agent
│   ├── reflection_agent_langchain.py           # 反思 Agent
│   └── plan_solve_agent_langchain.py           # 计划执行 Agent
│
├── examples/                                    # 示例代码
│   ├── example_simple_agent.py                  # SimpleAgent 示例
│   ├── example_react_agent.py                   # ReActAgent 示例
│   ├── example_reflection_agent.py              # ReflectionAgent 示例
│   ├── example_plan_solve_agent.py              # PlanAndSolveAgent 示例
│   └── example_structured_glm_agent.py          # 结构化输出示例
│
└── exercises/                                   # 课后习题
    ├── __init__.py
    ├── run_all_exercises.py                     # 运行所有习题
    ├── exercise_02_new_model_provider.py        # 习题2：多模型支持
    ├── exercise_04_custom_tools.py              # 习题4：自定义工具
    └── exercise_05_plugin_system.py             # 习题5：插件系统
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

#### 方式一：使用 .env 文件（推荐）

复制 `.env.example` 为 `.env` 并填写：

```bash
# 必需：智谱AI API Key
ZHIPUAI_API_KEY=your-zhipuai-api-key

# 可选：搜索 API
TAVILY_API_KEY=your-tavily-key
SERPAPI_API_KEY=your-serpapi-key
```

#### 方式二：设置系统环境变量

在终端中设置：

```bash
# macOS/Linux
export ZHIPUAI_API_KEY=your-zhipuai-api-key

# Windows PowerShell
$env:ZHIPUAI_API_KEY="your-zhipuai-api-key"

# Windows CMD
set ZHIPUAI_API_KEY=your-zhipuai-api-key
```

#### 环境变量加载说明

本项目的环境变量加载机制（在 `core/utils.py` 中）：

1. **优先尝试加载 .env 文件**：如果安装了 `python-dotenv`，会自动加载 `.env` 文件
2. **回退到系统环境变量**：如果没有安装 `python-dotenv`，使用系统环境变量
3. **提供默认值**：如果环境变量未设置，会使用占位符并在运行时报错提示

```python
# core/utils.py 中的加载逻辑
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # 继续使用系统环境变量

os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")
```

#### 获取 API 密钥

- **智谱AI** (必需): https://open.bigmodel.cn/
- **Tavily** (可选): https://tavily.com/
- **SerpAPI** (可选): https://serpapi.com/

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

## 📚 详细使用示例

### SimpleAgent - 基础对话

```python
from core.utils import setup_llm
from agents.simple_agent_langchain import SimpleAgent

# 初始化 LLM
llm = setup_llm(model="glm-4-flash")

# 创建 Agent
agent = SimpleAgent(
    name="助手",
    llm=llm,
    enable_tool_calling=False
)

# 运行对话
response = agent.run("介绍一下 Python")
print(response)
```

### ReActAgent - 推理与行动

```python
from core.utils import setup_llm
from agents.react_agent_langchain import ReActAgent
from tools.calculator_tool import CalculatorTool
from tools.search_tool import MockSearchTool

llm = setup_llm(model="glm-4-flash")
calculator = CalculatorTool()
search = MockSearchTool()

agent = ReActAgent(
    name="ReAct助手",
    llm=llm,
    tools=[calculator, search],
    max_steps=10
)

result = agent.run("北京的人口是多少？帮我计算一下是上海的几倍")
print(result)
```

### 使用结构化输出

```python
from pydantic import BaseModel, Field
from typing import List

class AnalysisResult(BaseModel):
    summary: str = Field(description="文本摘要")
    sentiment: str = Field(description="情感倾向")
    keywords: List[str] = Field(description="关键词")

llm = setup_llm(model="glm-4-flash")
structured_llm = llm.with_structured_output(AnalysisResult)

result = structured_llm.invoke([
    {"role": "user", "content": "分析这段文本：LangChain 很强大..."}
])

print(f"摘要: {result.summary}")
print(f"情感: {result.sentiment}")
print(f"关键词: {result.keywords}")
```

## 🔧 常见问题 FAQ

### Q1: 运行时提示 "Invalid API key: your-api-key-here"

**原因**：环境变量 `ZHIPUAI_API_KEY` 未正确设置。

**解决方案**：
1. 检查 `.env` 文件是否存在并包含正确的 API Key
2. 确认 `python-dotenv` 已安装：`pip install python-dotenv`
3. 或直接设置系统环境变量（参考上方配置说明）

### Q2: 提示 "未设置 ZHIPUAI_API_KEY 环境变量"

**原因**：API Key 为空或等于默认占位符。

**解决方案**：
```bash
# 1. 检查 .env 文件
cat .env

# 2. 验证环境变量是否加载
python -c "import os; print(os.getenv('ZHIPUAI_API_KEY'))"

# 3. 如果输出为 None 或 your-api-key-here，需要重新配置
```

### Q3: 模型返回 429 错误（请求过于频繁）

**原因**：API 调用频率超限。

**解决方案**：
1. 使用 `glm-4-flash` 而非 `glm-4-plus`（更高的速率限制）
2. 在连续调用之间添加延时：`import time; time.sleep(2)`
3. 检查智谱AI控制台的配额使用情况

### Q4: 如何切换到其他模型提供商？

参考 `exercises/exercise_02_new_model_provider.py`，支持：
- Anthropic Claude
- Moonshot AI
- Ollama（本地模型）

```python
from exercises.exercise_02_new_model_provider import MultiModelLLM

# 使用 Claude
llm = MultiModelLLM(provider="anthropic", model="claude-3-5-sonnet-20241022")

# 使用 Moonshot
llm = MultiModelLLM(provider="moonshot", model="moonshot-v1-8k")

# 使用本地 Ollama
llm = MultiModelLLM(provider="ollama", model="llama2")
```

### Q5: 如何自定义工具？

参考 `exercises/exercise_04_custom_tools.py`：

```python
from langchain_core.tools import BaseTool

class MyCustomTool(BaseTool):
    name: str = "my_tool"
    description: str = "工具描述"

    def _run(self, query: str) -> str:
        # 实现工具逻辑
        return f"处理结果: {query}"

# 或使用装饰器
from langchain_core.tools import tool

@tool
def my_simple_tool(input: str) -> str:
    """工具描述"""
    return f"结果: {input}"
```

### Q6: Agent 无法正确调用工具？

**检查清单**：
1. 工具的 `description` 是否清晰描述了功能和参数
2. 工具是否正确传递给 Agent：`agent = Agent(tools=[tool1, tool2])`
3. 提示词是否引导模型使用工具
4. 检查 Agent 日志输出，查看工具调用过程

### Q7: 如何调试 Agent 执行过程？

启用详细日志：

```python
import logging

# 设置 LangChain 日志级别
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("langchain").setLevel(logging.DEBUG)

# 或在 Agent 中添加打印
agent = SimpleAgent(llm=llm, verbose=True)
```

### Q8: 运行示例时找不到模块？

**解决方案**：
```bash
# 确保在项目根目录运行
cd agent-langchain-code/HelloAgents_Chapter7_Code

# 或设置 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Windows
set PYTHONPATH=%PYTHONPATH%;%cd%
```

## ⚠️ 注意事项

1. **API 兼容性**：确保使用 LangChain >= 0.1.0
2. **模型选择**：
   - 开发/测试：使用 `glm-4-flash`（更快、更便宜）
   - 生产环境：使用 `glm-4-plus` 或 `glm-4.6`（更强大）
3. **工具依赖**：搜索工具需要额外的 API 密钥
4. **Python 版本**：需要 Python >= 3.10
5. **并发限制**：免费 API Key 有速率限制，连续调用需添加延时
6. **安全提醒**：
   - 不要将 `.env` 文件提交到 Git 仓库
   - 不要在代码中硬编码 API Key
   - 使用 `.gitignore` 排除敏感文件

## 🎓 推荐学习路径

### 第一阶段：基础理解（1-2天）

1. **阅读 Hello-Agents 第七章**
   - 理解 Agent 框架的核心概念
   - 学习工具、Agent、注册机制的设计思路

2. **运行快速测试**
   ```bash
   python quick_test.py
   ```
   - 验证环境配置
   - 理解基本组件的工作方式

3. **体验基础示例**
   ```bash
   python examples/example_simple_agent.py
   ```
   - 观察 Agent 如何处理对话
   - 理解消息流转过程

### 第二阶段：深入实践（3-5天）

1. **学习四种 Agent 模式**
   - SimpleAgent：基础对话
   - ReActAgent：推理-行动循环
   - ReflectionAgent：自我反思
   - PlanAndSolveAgent：计划与执行

2. **对比 LangChain 与 Hello-Agents**
   - 工具定义方式的差异
   - Agent 实现的不同思路
   - 理解抽象层次的取舍

3. **完成课后习题**
   ```bash
   python exercises/run_all_exercises.py
   ```
   - 多模型支持
   - 自定义工具开发
   - 插件系统设计

### 第三阶段：实战项目（1-2周）

1. **构建自己的 Agent 应用**
   - 选择一个实际问题场景
   - 设计工具和 Agent 架构
   - 实现并测试

2. **性能优化与调试**
   - 优化提示词
   - 减少 API 调用次数
   - 处理异常情况

3. **部署与监控**
   - 配置生产环境
   - 添加日志和监控
   - 处理并发和限流

## 📊 性能与成本参考

### 智谱AI GLM 模型对比

| 模型 | 速度 | 能力 | 价格 | 推荐场景 |
|------|------|------|------|----------|
| **glm-4-flash** | ⚡⚡⚡ | ⭐⭐⭐ | 💰 | 开发测试、简单任务 |
| **glm-4-plus** | ⚡⚡ | ⭐⭐⭐⭐ | 💰💰 | 生产环境、复杂任务 |
| **glm-4.6** | ⚡⚡ | ⭐⭐⭐⭐⭐ | 💰��💰 | 高质量输出、关键任务 |

### 成本优化建议

1. **开发阶段**：使用 `glm-4-flash`，快速迭代
2. **测试阶段**：混合使用，关键路径用高级模型
3. **生产阶段**：根据任务重要性选择模型
4. **缓存策略**：对相同输入缓存结果
5. **批量处理**：合并多个请求减少调用次数

## 🔗 相关资源

### 官方文档

- **Hello-Agents 原始代码**: https://github.com/datawhalechina/hello-agents/tree/main/code/chapter7
- **LangChain 官方文档**: https://python.langchain.com/
- **智谱AI 开放平台**: https://open.bigmodel.cn/
- **LangChain Tools 文档**: https://python.langchain.com/docs/modules/agents/tools/

### 相关项目

- **第六章 LangChain 实现**: ../HelloAgents_Chapter6_Code/
- **langchain_agents_examples**: ../../langchain_agents_examples/

### 推荐阅读

- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/)
- [Building Agents with LangChain](https://python.langchain.com/docs/modules/agents/)
- [智谱AI GLM-4 技术报告](https://open.bigmodel.cn/)

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 如何贡献

1. **报告问题**
   - 在 Issues 中描述遇到的问题
   - 提供复现步骤和环境信息
   - 附上错误日志和截图

2. **提交改进**
   - Fork 本仓库
   - 创建特性分支：`git checkout -b feature/your-feature`
   - 提交更改：`git commit -m "Add: your feature"`
   - 推送到分支：`git push origin feature/your-feature`
   - 创建 Pull Request

3. **完善文档**
   - 修正文档中的错误
   - 添加更多使用示例
   - 翻译文档到其他语言

### 代码规范

- 遵循 PEP 8 编码规范
- 添加必要的注释和文档字符串
- 确保所有测试通过
- 更新相关文档

## 📄 许可证

本项目基于 MIT 许可证开源。

## 🙏 致谢

- **Datawhale Hello-Agents 团队**：提供优秀的 Agent 开发教程
- **LangChain 社区**：构建强大的 LLM 应用框架
- **智谱AI**：提供高质量的中文大语言模型

---

## 📋 版本信息

**项目**: LangChain v1.0 实现 - Hello Agents 第七章
**版本**: 2.0
**更新日期**: 2025-01-29
**作者**: 基于 Hello-Agents 第七章改编
**LangChain 版本**: >= 1.0
**Python 版本**: >= 3.10

### 更新日志

**v2.0** (2025-01-29)
- ✅ 优化环境变量加载机制，支持 .env 文件和系统环境变量
- ✅ 添加详细的 FAQ 常见问题解答
- ✅ 完善示例代码和使用说明
- ✅ 添加性能优化和成本控制建议
- ✅ 增强错误处理和用户提示

**v1.0** (2025-01-23)
- ✅ 基于 LangChain v1.0 实现四种 Agent 模式
- ✅ 完成核心工具系统和 Agent 基类
- ✅ 添加完整示例和课后习题
- ✅ 编写项目文档和使用指南

---

**如有问题或建议，欢迎提交 Issue！** 💬
