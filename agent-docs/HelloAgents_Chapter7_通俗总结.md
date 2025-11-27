# Hello Agents 第七章：构建你的 Agent 框架（通俗总结）

> **本章核心思想**：从零开始构建一个轻量级、易学习的智能体框架 HelloAgents，让你从"使用者"变成"构建者"。

---

## 📖 目录

- [1. 为什么要自己造轮子？](#1-为什么要自己造轮子)
- [2. HelloAgents 框架设计理念](#2-helloagents-框架设计理念)
- [3. 核心组件实现](#3-核心组件实现)
- [4. Agent 范式框架化](#4-agent-范式框架化)
- [5. 工具系统设计](#5-工具系统设计)
- [6. 本章总结](#6-本章总结)

---

## 1. 为什么要自己造轮子？

### 🤔 市面上不是有很多 Agent 框架吗？

是的，有 LangChain、AutoGen、AgentScope 等很多成熟框架。但它们都有一些问题：

**问题一：太复杂了** 😵
- 要学习一堆概念：Chain、Agent、Tool、Memory、Retriever...
- 新手学习曲线陡峭
- 一个简单任务可能需要理解十几个类

**问题二：更新太快** 🔄
- API 经常变更，代码升级后跑不动
- 维护成本高

**问题三：黑盒子** 📦
- 内部实现看不清楚
- 出了问题不知道怎么调试
- 难以深度定制

**问题四：依赖多** 📚
- 安装包很大
- 可能和其他项目冲突

### 💡 自己造轮子的好处

1. **深度理解**：真正搞懂 Agent 工作原理
2. **完全掌控**：每一行代码都在你手里
3. **培养能力**：提升系统设计能力
4. **按需定制**：想加什么功能就加什么

> 💭 **类比**：就像学做菜，用外卖很方便（成熟框架），但自己做才能真正学会（自建框架）

---

## 2. HelloAgents 框架设计理念

### 🎯 四大核心理念

#### 1. 轻量级 + 教学友好

```
传统框架：📦 大黑箱（看不懂）
HelloAgents：📖 透明玻璃箱（一目了然）
```

- 核心代码按章节组织
- 除了 OpenAI SDK，几乎不引入额外依赖
- 任何有编程基础的人都能看懂

#### 2. 基于标准 API

```python
# 不重新发明轮子，基于 OpenAI 标准接口
llm = HelloAgentsLLM()  # 兼容所有支持 OpenAI API 的模型
```

**为什么选择 OpenAI API？**
- 已经是行业标准
- 大部分模型都支持
- 学习一次，到处可用

#### 3. 渐进式学习

```
第4章 -> 第7章 -> 第8章 -> ...
基础 Agent -> 框架化 -> 加入记忆 -> ...
```

每一步都是自然升级，没有概念跳跃

#### 4. 万物皆工具

```
传统框架：
- Agent 类
- Memory 类
- RAG 类
- Tool 类
- ...（好多类要学）

HelloAgents：
- Agent 类
- Tool 类（统一抽象）
```

**核心思想**：Memory、RAG、MCP 都当成"工具"，统一处理

> 💡 **类比**：就像手机 APP 统一管理，而不是每个功能一个入口

---

## 3. 核心组件实现

### 3.1 框架整体架构

```
hello-agents/
├── core/                     # 核心框架层
│   ├── agent.py             # Agent 基类
│   ├── llm.py               # LLM 统一接口
│   ├── message.py           # 消息系统
│   ├── config.py            # 配置管理
│   └── exceptions.py        # 异常处理
│
├── agents/                  # Agent 实现层
│   ├── simple_agent.py      # 简单对话
│   ├── react_agent.py       # ReAct 范式
│   ├── reflection_agent.py  # 反思范式
│   └── plan_solve_agent.py  # 计划执行范式
│
└── tools/                   # 工具系统
    ├── base.py              # 工具基类
    ├── registry.py          # 工具注册
    └── builtin/             # 内置工具
```

### 3.2 HelloAgentsLLM - 多模型支持

#### 🎯 设计目标

让你的 Agent 能轻松切换各种 LLM：

```python
# OpenAI
llm = HelloAgentsLLM(provider="openai")

# 本地 Ollama
llm = HelloAgentsLLM(provider="ollama")

# 智谱 AI
llm = HelloAgentsLLM(provider="zhipu")

# 自动检测
llm = HelloAgentsLLM()  # 根据环境变量自动选择
```

#### 🔧 自动检测机制

**优先级顺序**：

1. **检查特定 API Key** （最高优先级）
   ```bash
   MODELSCOPE_API_KEY="xxx"  # 优先检测
   OPENAI_API_KEY="xxx"
   ```

2. **检查 Base URL**
   ```bash
   LLM_BASE_URL="http://localhost:11434/v1"  # 识别为 Ollama
   ```

3. **检查 API Key 格式**（辅助判断）
   ```python
   if api_key.startswith("ms-"):  # ModelScope
       return "modelscope"
   ```

#### 📝 实现示例

```python
# my_llm.py
from hello_agents import HelloAgentsLLM

class MyLLM(HelloAgentsLLM):
    def __init__(self, provider="auto", **kwargs):
        if provider == "my_custom_provider":
            # 自定义配置
            self.api_key = os.getenv("MY_API_KEY")
            self.base_url = "https://my-api.com/v1"
            # ... 初始化
        else:
            # 使用父类逻辑
            super().__init__(provider=provider, **kwargs)
```

#### 🏠 本地模型支持

**方式一：VLLM（高性能）**

```bash
# 1. 安装
pip install vllm

# 2. 启动服务
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen1.5-0.5B-Chat \
    --port 8000

# 3. 使用
llm = HelloAgentsLLM(
    provider="vllm",
    base_url="http://localhost:8000/v1"
)
```

**方式二：Ollama（最简单）**

```bash
# 1. 安装 Ollama
# 访问 https://ollama.com 下载

# 2. 运行模型
ollama run llama3

# 3. 使用
llm = HelloAgentsLLM(provider="ollama")
```

### 3.3 Message - 消息系统

#### 🎯 为什么需要 Message 类？

```python
# ❌ 不好的方式：直接用字典
messages = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"}
]

# ✅ 好的方式：使用 Message 类
messages = [
    Message("你好", "user"),
    Message("你好！", "assistant")
]
```

**好处**：
- 类型安全（不会写错 role）
- 自带时间戳
- 可扩展元数据
- 统一格式转换

#### 📝 Message 实现

```python
from pydantic import BaseModel
from datetime import datetime

class Message(BaseModel):
    content: str
    role: Literal["user", "assistant", "system", "tool"]
    timestamp: datetime = None
    metadata: Optional[Dict] = None

    def to_dict(self):
        """转为 OpenAI API 格式"""
        return {
            "role": self.role,
            "content": self.content
        }
```

### 3.4 Config - 配置管理

#### 🎯 集中管理配置

```python
class Config(BaseModel):
    # LLM 配置
    default_model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    # 系统配置
    debug: bool = False
    log_level: str = "INFO"

    # Agent 配置
    max_history_length: int = 100

    @classmethod
    def from_env(cls):
        """从环境变量读取"""
        return cls(
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )
```

**使用方式**：

```python
# 方式1：使用默认值
config = Config()

# 方式2：从环境变量
config = Config.from_env()

# 方式3：手动指定
config = Config(temperature=0.9, debug=True)
```

### 3.5 Agent 基类

#### 🎯 统一接口设计

```python
class Agent(ABC):
    """所有 Agent 的抽象基类"""

    def __init__(self, name, llm, system_prompt=None, config=None):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """执行 Agent（子类必须实现）"""
        pass

    def add_message(self, message: Message):
        """添加到历史"""
        self._history.append(message)

    def get_history(self) -> list:
        """获取历史"""
        return self._history.copy()
```

**设计精髓**：

1. ✅ **统一入口**：所有 Agent 都有 `run()` 方法
2. ✅ **强制规范**：用 `@abstractmethod` 强制子类实现
3. ✅ **通用功能**：历史管理等功能在基类实现
4. ✅ **灵活扩展**：子类可以添加自己的方法

---

## 4. Agent 范式框架化

### 4.1 SimpleAgent - 基础对话

#### 🎯 最简单的 Agent

```python
from hello_agents import SimpleAgent, HelloAgentsLLM

# 创建 LLM
llm = HelloAgentsLLM()

# 创建 Agent
agent = SimpleAgent(
    name="小助手",
    llm=llm,
    system_prompt="你是一个友好的 AI 助手"
)

# 使用
response = agent.run("你好，介绍一下自己")
print(response)
```

#### 🔧 支持工具调用

```python
from hello_agents.tools import CalculatorTool

# 添加工具
calculator = CalculatorTool()
agent.add_tool(calculator)

# 现在可以计算了
response = agent.run("帮我算一下 2 + 3 * 4")
```

#### 📝 工作流程

```
用户输入
   ↓
构建消息（包含历史）
   ↓
调用 LLM
   ↓
检测工具调用 → 执行工具 → 返回结果
   ↓
生成最终回答
   ↓
保存到历史
```

### 4.2 ReActAgent - 推理与行动

#### 🎯 核心思想

```
Thought（思考） → Action（行动） → Observation（观察）
→ 思考 → 行动 → 观察 → ... → Finish（完成）
```

#### 📝 提示词模板

```python
REACT_PROMPT = """
你是一个具备推理和行动能力的 AI 助手。

## 可用工具
{tools}

## 工作流程
每次回应必须包含：

Thought: 分析当前问题，思考需要什么信息
Action: 选择一个行动
  - tool_name[input] - 调用工具
  - Finish[answer] - 给出最终答案

## 当前任务
Question: {question}

## 执行历史
{history}

现在开始：
"""
```

#### 💡 使用示例

```python
from hello_agents import ReActAgent, ToolRegistry

# 创建工具注册表
registry = ToolRegistry()
registry.register_tool(CalculatorTool())
registry.register_tool(SearchTool())

# 创建 ReAct Agent
agent = ReActAgent(
    name="推理助手",
    llm=llm,
    tool_registry=registry,
    max_steps=5  # 最多执行 5 步
)

# 使用
response = agent.run("北京明天天气怎么样？需要带伞吗？")
```

**执行过程**：

```
第 1 步：
  Thought: 需要查询北京明天的天气
  Action: search[北京明天天气]
  Observation: 明天北京有雨，温度 15-20℃

第 2 步：
  Thought: 既然有雨，需要带伞
  Action: Finish[明天北京有雨，温度 15-20℃，建议带伞]
```

### 4.3 ReflectionAgent - 自我反思

#### 🎯 核心思想

```
初次回答 → 反思 → 改进 → 反思 → 改进 → ... → 最终答案
```

#### 📝 三阶段提示词

```python
# 阶段1：初次回答
INITIAL_PROMPT = """
请完成任务：{task}
"""

# 阶段2：反思
REFLECT_PROMPT = """
原始任务：{task}
当前回答：{content}

请分析回答的质量，指出不足，提出改进建议。
"""

# 阶段3：改进
REFINE_PROMPT = """
原始任务：{task}
上一轮回答：{last_attempt}
反馈意见：{feedback}

请提供改进后的回答。
"""
```

#### 💡 使用示例

```python
from hello_agents import ReflectionAgent

agent = ReflectionAgent(
    name="反思助手",
    llm=llm,
    max_reflections=3  # 最多反思 3 次
)

response = agent.run("写一篇关于 AI 发展的文章")
```

**执行过程**：

```
轮次 1：写初稿
  反思：结构不够清晰，缺少具体例子

轮次 2：改进版本
  反思：内容更好了，但缺少总结

轮次 3：最终版本
  反思：很好，无需继续改进

返回最终版本
```

### 4.4 PlanAndSolveAgent - 计划执行

#### 🎯 核心思想

```
复杂问题 → 分解成步骤 → 逐步执行 → 合并结果
```

#### 📝 两阶段提示词

```python
# 阶段1：规划
PLANNER_PROMPT = """
将问题分解成多个简单步骤：

问题：{question}

输出格式（Python 列表）：
["步骤1", "步骤2", "步骤3"]
"""

# 阶段2：执行
EXECUTOR_PROMPT = """
原始问题：{question}
完整计划：{plan}
历史结果：{history}
当前步骤：{current_step}

请执行当前步骤，只输出答案。
"""
```

#### 💡 使用示例

```python
from hello_agents import PlanAndSolveAgent

agent = PlanAndSolveAgent(
    name="计划助手",
    llm=llm
)

question = """
一个水果店周一卖了15个苹果，
周二卖的是周一的两倍，
周三比周二少5个。
三天总共卖了多少个？
"""

response = agent.run(question)
```

**执行过程**：

```
规划阶段：
  ["计算周一销量", "计算周二销量", "计算周三销量", "求总和"]

执行阶段：
  步骤1: 周一 = 15
  步骤2: 周二 = 15 × 2 = 30
  步骤3: 周三 = 30 - 5 = 25
  步骤4: 总和 = 15 + 30 + 25 = 70

最终答案：70个苹果
```

### 📊 四种 Agent 对比

| Agent 类型 | 适用场景 | 核心特点 | 优势 | 局限 |
|-----------|---------|---------|------|------|
| **SimpleAgent** | 简单对话、知识问答 | 直接回答 | 速度快、成本低 | 无推理能力 |
| **ReActAgent** | 需要工具辅助的任务 | 思考-行动循环 | 推理能力强 | Token 消耗多 |
| **ReflectionAgent** | 需要高质量输出 | 自我反思改进 | 输出质量高 | 耗时长 |
| **PlanAndSolveAgent** | 复杂多步骤任务 | 规划后执行 | 逻辑清晰 | 需要分解能力 |

---

## 5. 工具系统设计

### 5.1 工具基类设计

#### 🎯 统一抽象

```python
from abc import ABC, abstractmethod

class Tool(ABC):
    """工具基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str:
        """执行工具（子类必须实现）"""
        pass

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """返回参数定义"""
        pass
```

#### 📝 参数定义

```python
class ToolParameter(BaseModel):
    """工具参数定义"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
```

### 5.2 工具注册机制

#### 🎯 ToolRegistry - 工具管理中心

```python
class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools = {}  # 存储工具对象
        self._functions = {}  # 存储函数工具

    def register_tool(self, tool: Tool):
        """注册 Tool 对象"""
        self._tools[tool.name] = tool

    def register_function(self, name, description, func):
        """直接注册函数"""
        self._functions[name] = {
            "description": description,
            "func": func
        }

    def execute_tool(self, tool_name, input_data):
        """执行工具"""
        if tool_name in self._tools:
            return self._tools[tool_name].run(input_data)
        elif tool_name in self._functions:
            return self._functions[tool_name]["func"](input_data)
```

### 5.3 自定义工具开发

#### 💡 方式一：函数注册（简单）

```python
def my_calculator(expression: str) -> str:
    """简单计算器"""
    try:
        result = eval(expression)  # 实际应该用安全的解析
        return str(result)
    except:
        return "计算失败"

# 注册
registry = ToolRegistry()
registry.register_function(
    name="calculator",
    description="数学计算工具",
    func=my_calculator
)

# 使用
result = registry.execute_tool("calculator", "2 + 3")
```

#### 💡 方式二：Tool 类（复杂）

```python
class CalculatorTool(Tool):
    """计算器工具"""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="执行数学计算"
        )

    def get_parameters(self):
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="数学表达式",
                required=True
            )
        ]

    def run(self, parameters):
        expression = parameters.get("expression", "")
        try:
            result = eval(expression)
            return str(result)
        except:
            return "计算失败"
```

### 5.4 多源搜索工具

#### 🎯 整合多个搜索引擎

```python
class SearchTool(Tool):
    """智能搜索工具"""

    def __init__(self, backend="hybrid"):
        super().__init__(
            name="search",
            description="智能网页搜索"
        )
        self.backend = backend
        self.available_backends = []
        self._setup_backends()

    def _setup_backends(self):
        """检测可用的搜索源"""
        # 检查 Tavily
        if os.getenv("TAVILY_API_KEY"):
            self.available_backends.append("tavily")

        # 检查 SerpAPI
        if os.getenv("SERPAPI_API_KEY"):
            self.available_backends.append("serpapi")

    def run(self, parameters):
        query = parameters.get("query", "")

        # 混合模式：优先 Tavily，失败则 SerpAPI
        if "tavily" in self.available_backends:
            try:
                return self._search_tavily(query)
            except:
                if "serpapi" in self.available_backends:
                    return self._search_serpapi(query)

        return "没有可用的搜索源"
```

**核心设计思想**：

1. **智能降级**：优先使用最佳源，失败则降级
2. **统一格式**：不同源的结果格式化为统一输出
3. **容错处理**：每个源都有异常处理

### 5.5 高级特性

#### 🔗 工具链（ToolChain）

**场景**：需要串联多个工具

```python
class ToolChain:
    """工具链"""

    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.steps = []

    def add_step(self, tool_name, input_template, output_key):
        """添加步骤"""
        self.steps.append({
            "tool_name": tool_name,
            "input_template": input_template,
            "output_key": output_key
        })

    def execute(self, registry, initial_input, context=None):
        """执行工具链"""
        context = context or {}
        context["input"] = initial_input

        for step in self.steps:
            # 替换模板变量
            tool_input = step["input_template"].format(**context)

            # 执行工具
            result = registry.execute_tool(step["tool_name"], tool_input)

            # 保存结果
            context[step["output_key"]] = result

        return context[self.steps[-1]["output_key"]]
```

**使用示例**：

```python
# 创建工具链：搜索 -> 总结
chain = ToolChain("research", "研究助手")

chain.add_step(
    tool_name="search",
    input_template="{input}",
    output_key="search_result"
)

chain.add_step(
    tool_name="summarizer",
    input_template="总结以下内容：{search_result}",
    output_key="summary"
)

# 执行
result = chain.execute(registry, "人工智能的发展")
```

#### ⚡ 异步执行（AsyncToolExecutor）

**场景**：并行执行多个耗时工具

```python
class AsyncToolExecutor:
    """异步工具执行器"""

    def __init__(self, registry, max_workers=4):
        self.registry = registry
        self.executor = ThreadPoolExecutor(max_workers)

    async def execute_tools_parallel(self, tasks):
        """并行执行多个工具"""
        async_tasks = [
            self.execute_tool_async(task["tool_name"], task["input"])
            for task in tasks
        ]

        results = await asyncio.gather(*async_tasks)
        return results
```

**使用示例**：

```python
executor = AsyncToolExecutor(registry)

tasks = [
    {"tool_name": "search", "input": "Python 编程"},
    {"tool_name": "search", "input": "机器学习"},
    {"tool_name": "calculator", "input": "2 + 2"}
]

# 并行执行
results = await executor.execute_tools_parallel(tasks)
```

---

## 6. 本章总结

### 🎯 你学到了什么？

#### 1. 框架设计思想

✅ **分层解耦**
- 核心层、Agent 层、工具层各司其职
- 修改一个地方不影响其他部分

✅ **统一抽象**
- Agent 统一继承基类
- 工具统一实现接口
- 降低学习成本

✅ **渐进式扩展**
- 从简单到复杂
- 每章在上一章基础上迭代

#### 2. 核心技术点

📚 **LLM 调用封装**
```python
HelloAgentsLLM
  ├── 多模型支持
  ├── 自动检测
  └── 本地部署
```

🤖 **四种 Agent 范式**
```python
SimpleAgent      # 基础对话
ReActAgent       # 推理行动
ReflectionAgent  # 自我反思
PlanAndSolveAgent # 计划执行
```

🔧 **工具系统**
```python
Tool 基类
  ├── 工具注册
  ├── 工具链
  └── 异步执行
```

### 📈 对比第四章的进步

| 维度 | 第四章 | 第七章 |
|-----|--------|--------|
| **代码组织** | 单文件实现 | 模块化框架 |
| **可扩展性** | 难以扩展 | 易于扩展 |
| **可维护性** | 代码耦合 | 职责分离 |
| **可复用性** | 难以复用 | 组件化复用 |
| **学习曲线** | 一次性理解 | 渐进式学习 |

### 🚀 后续章节预告

```
第 7 章（当前）：框架基础
         ↓
第 8 章：记忆与 RAG
  - Memory 机制
  - 向量数据库
  - RAG 系统
         ↓
第 9 章：上下文工程
  - 消息管理
  - Token 优化
  - 上下文策略
         ↓
第 10 章：智能体协议
  - MCP 协议
  - Agent 通信
  - 多智能体协作
```

### 💡 学习建议

**对于初学者**：
1. ✅ 先体验：`pip install hello-agents` 直接使用
2. ✅ 再理解：阅读源码，理解设计思想
3. ✅ 后实践：跟着教程重新实现

**对于进阶者**：
1. ✅ 深入源码：研究每个设计决策
2. ✅ 扩展框架：添加自己的 Agent 类型
3. ✅ 对比框架：和 LangChain 等框架对比

**对于专业开发者**：
1. ✅ 生产化改造：添加日志、监控、容错
2. ✅ 性能优化：缓存、并发、资源管理
3. ✅ 构建应用：基于框架开发实际项目

### 🔗 相关资源

- **GitHub 仓库**：https://github.com/jjyaoao/helloagents
- **完整测试案例**：[chapter07_basic_setup.py](https://github.com/jjyaoao/HelloAgents/blob/main/examples/chapter07_basic_setup.py)
- **Hello Agents 官方文档**：https://datawhalechina.github.io/hello-agents/

---

## 📝 快速参考

### 安装

```bash
pip install "hello-agents==0.1.1"
```

### 最小示例

```python
from hello_agents import SimpleAgent, HelloAgentsLLM
from dotenv import load_dotenv

load_dotenv()

llm = HelloAgentsLLM()
agent = SimpleAgent(name="助手", llm=llm)

response = agent.run("你好")
print(response)
```

### 添加工具

```python
from hello_agents.tools import CalculatorTool

calculator = CalculatorTool()
agent.add_tool(calculator)

response = agent.run("计算 2 + 3")
```

### 自定义 Agent

```python
from hello_agents import Agent

class MyAgent(Agent):
    def run(self, input_text, **kwargs):
        # 你的实现
        pass
```

---

## 🎓 章节习题提示

1. **框架设计理念**：思考"万物皆工具"的优缺点
2. **多模型支持**：实践添加新的模型供应商
3. **Agent 实现**：对比不同 Agent 的适用场景
4. **工具开发**：实现一个实用的自定义工具
5. **系统扩展**：设计插件系统架构

---

## 📌 核心要点回顾

```
🎯 为什么自建框架？
   → 深度理解 + 完全掌控 + 按需定制

🏗️ 框架设计理念
   → 轻量级 + 标准API + 渐进式 + 万物皆工具

🔧 核心组件
   → LLM封装 + 消息系统 + 配置管理 + Agent基类

🤖 四种Agent
   → Simple + ReAct + Reflection + PlanAndSolve

🔨 工具系统
   → 工具基类 + 注册机制 + 工具链 + 异步执行
```

---

**下一章预告**：第八章将深入探讨如何为 Agent 添加"记忆"和 RAG 能力，让你的 Agent 能够记住对话历史、检索外部知识！

**Happy Coding! 🚀**
