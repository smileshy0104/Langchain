# Hello Agents 第四章：智能体经典范式构建（通俗总结）

> **本章核心思想**：让大语言模型从"只会聊天"进化到"能思考、能规划、能自我改进"的智能体。我们将学习三种经典范式，理解它们的设计思想和实现原理。

---

## 📖 目录

- [1. 为什么需要智能体范式？](#1-为什么需要智能体范式)
- [2. ReAct：推理与行动循环](#2-react推理与行动循环)
- [3. Plan-and-Solve：计划执行范式](#3-plan-and-solve计划执行范式)
- [4. Reflection：自我反思范式](#4-reflection自我反思范式)
- [5. 范式选择与组合](#5-范式选择与组合)
- [6. 本章总结](#6-本章总结)

---

## 1. 为什么需要智能体范式？

### 🤔 大语言模型的三大局限

尽管 LLM 很强大，但它们有三个致命弱点：

**局限1：无法与外界交互** 🚫
```
用户：现在北京天气怎么样？
GPT：我无法实时查询天气...（无能为力）
```

**局限2：缺乏系统性规划** 🎯
```
复杂问题：设计一个完整的用户认证系统
GPT：（直接给方案，缺乏步骤分解，容易遗漏细节）
```

**局限3：不会自我优化** 📈
```
用户：帮我写个算法
GPT：（写了个 O(n²) 的暴力解法就结束了）
```

### 💡 智能体范式的解决方案

三种经典范式就是针对这三大局限的解决方案：

| 范式 | 解决的问题 | 核心能力 | 类比 |
|------|----------|---------|------|
| **ReAct** | 无法交互 | 工具调用 + 推理循环 | 🕵️ 侦探破案 |
| **Plan-and-Solve** | 缺乏规划 | 问题分解 + 步骤执行 | 🏗️ 建筑师盖房 |
| **Reflection** | 不会优化 | 自我评估 + 迭代改进 | 📝 作家修稿 |

### 🎯 一句话概括

**智能体范式 = LLM + 思考方式（范式） + 外部能力（工具）**

---

## 2. ReAct：推理与行动循环

### 🧠 核心思想

**ReAct = Reasoning（推理） + Acting（行动）**

让 AI 像侦探破案一样：**观察线索 → 推理分析 → 采取行动 → 观察结果 → ...**

### 📋 工作流程

```
┌─────────────────────────────────────┐
│  用户问题：华为最新手机是什么？         │
└─────────────┬───────────────────────┘
              │
              ▼
    ┌─────────────────────┐
    │  Thought（思考）      │
    │  分析：需要搜索最新信息 │
    └──────────┬───────────┘
               │
               ▼
    ┌─────────────────────────┐
    │  Action（行动）           │
    │  调用工具：Search["华为最新手机"] │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  Observation（观察）        │
    │  结果：找到 Mate 70 相关信息 │
    └──────────┬────────────────┘
               │
               ▼
    ┌─────────────────────────┐
    │  Thought（再次思考）       │
    │  分析：信息足够，可以回答    │
    └──────────┬───────────────┘
               │
               ▼
    ┌─────────────────────┐
    │  Answer（最终答案）    │
    │  华为最新手机是 Mate 70... │
    └─────────────────────┘
```

### 💻 提示词模板设计

#### 核心提示词结构

```python
REACT_PROMPT_TEMPLATE = """
你是一个具备推理和行动能力的 AI 助手。

## 可用工具
{tools_description}

## 工作流程
请按照以下格式思考和行动：

Thought: [分析当前情况，思考下一步需要什么]
Action: [选择一个行动]
  - tool_name[input] - 调用工具
  - Finish[answer] - 给出最终答案
Observation: [工具返回的结果]

## 重要规则
1. 每次只能执行一个行动
2. 必须等待 Observation 后再继续思考
3. 如果工具调用失败，尝试其他方法
4. 达到目标后使用 Finish[answer] 结束

## 当前任务
Question: {question}

## 执行历史
{scratchpad}

现在开始：
Thought:
"""
```

#### 工具描述格式

```python
def format_tools(tools):
    """格式化工具列表"""
    tool_descriptions = []

    for tool in tools:
        desc = f"""
### {tool.name}
- 描述：{tool.description}
- 用法：{tool.name}[{tool.input_format}]
- 示例：{tool.example}
"""
        tool_descriptions.append(desc)

    return "\n".join(tool_descriptions)
```

### 🛠️ 工具系统实现

#### 工具基类

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class Tool(ABC):
    """工具基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.input_format = "input_string"
        self.example = ""

    @abstractmethod
    def run(self, input_data: str) -> str:
        """执行工具（子类必须实现）"""
        pass

    def format_for_llm(self) -> str:
        """格式化为 LLM 可理解的描述"""
        return f"{self.name}: {self.description}\n用法: {self.name}[{self.input_format}]"
```

#### 常用工具实现

**1. 搜索工具**
```python
import requests

class SearchTool(Tool):
    """网络搜索工具"""

    def __init__(self, api_key: str = None):
        super().__init__(
            name="Search",
            description="搜索互联网获取实时信息"
        )
        self.api_key = api_key or os.getenv("SERPAPI_KEY")
        self.input_format = "搜索关键词"
        self.example = 'Search["Python 最新版本"]'

    def run(self, query: str) -> str:
        """执行搜索"""
        try:
            params = {
                "q": query,
                "api_key": self.api_key,
                "num": 3  # 返回前3个结果
            }
            response = requests.get(
                "https://serpapi.com/search",
                params=params,
                timeout=10
            )
            results = response.json().get("organic_results", [])

            # 格式化结果
            formatted = []
            for i, result in enumerate(results[:3], 1):
                formatted.append(
                    f"{i}. {result['title']}\n{result['snippet']}"
                )

            return "\n\n".join(formatted)
        except Exception as e:
            return f"搜索失败：{str(e)}"
```

**2. 计算器工具**
```python
import ast
import operator

class CalculatorTool(Tool):
    """安全的计算器工具"""

    # 支持的运算符
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def __init__(self):
        super().__init__(
            name="Calculator",
            description="执行数学计算"
        )
        self.input_format = "数学表达式"
        self.example = 'Calculator["2 + 3 * 4"]'

    def run(self, expression: str) -> str:
        """安全执行计算"""
        try:
            # 解析表达式为 AST
            tree = ast.parse(expression, mode='eval')
            result = self._eval_node(tree.body)
            return str(result)
        except Exception as e:
            return f"计算错误：{str(e)}"

    def _eval_node(self, node):
        """递归计算 AST 节点"""
        if isinstance(node, ast.Num):  # 数字
            return node.n
        elif isinstance(node, ast.BinOp):  # 二元运算
            op = self.OPERATORS[type(node.op)]
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):  # 一元运算
            op = self.OPERATORS[type(node.op)]
            operand = self._eval_node(node.operand)
            return op(operand)
        else:
            raise ValueError(f"不支持的操作: {type(node)}")
```

**3. 数据库查询工具**
```python
import sqlite3

class DatabaseTool(Tool):
    """数据库查询工具"""

    def __init__(self, db_path: str):
        super().__init__(
            name="Database",
            description="查询 SQLite 数据库"
        )
        self.db_path = db_path
        self.input_format = "SQL 查询语句"
        self.example = 'Database["SELECT * FROM users LIMIT 5"]'

    def run(self, sql: str) -> str:
        """执行 SQL 查询"""
        # 安全检查：只允许 SELECT
        if not sql.strip().upper().startswith("SELECT"):
            return "错误：只支持 SELECT 查询"

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql)

            # 获取列名和结果
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            conn.close()

            # 格式化为表格
            result = f"列: {', '.join(columns)}\n"
            for row in rows:
                result += f"{row}\n"

            return result
        except Exception as e:
            return f"查询失败：{str(e)}"
```

### 🔄 ReAct Agent 完整实现

```python
import re
from typing import List, Optional

class ReActAgent:
    """ReAct 智能体"""

    def __init__(
        self,
        llm,
        tools: List[Tool],
        max_steps: int = 5,
        verbose: bool = True
    ):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.verbose = verbose

        # 解析 Action 的正则
        self.action_pattern = re.compile(
            r"Action:\s*(\w+)\[(.*?)\]",
            re.DOTALL
        )

    def run(self, question: str) -> str:
        """执行 ReAct 流程"""
        scratchpad = []  # 记录思考和行动历史

        for step in range(1, self.max_steps + 1):
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Step {step}/{self.max_steps}")
                print(f"{'='*50}")

            # 1. 构建提示词
            prompt = self._build_prompt(question, scratchpad)

            # 2. 调用 LLM 思考
            response = self.llm.generate(prompt)

            if self.verbose:
                print(f"\n{response}")

            # 3. 解析 Action
            action_match = self.action_pattern.search(response)

            if not action_match:
                # 没有找到 Action，可能是格式错误
                scratchpad.append(response)
                scratchpad.append("Observation: 格式错误，请使用 Action: ToolName[input] 格式")
                continue

            action_name = action_match.group(1)
            action_input = action_match.group(2).strip()

            # 4. 检查是否完成
            if action_name == "Finish":
                return action_input

            # 5. 执行工具
            if action_name not in self.tools:
                observation = f"错误：工具 {action_name} 不存在"
            else:
                observation = self.tools[action_name].run(action_input)

            # 6. 记录到 scratchpad
            scratchpad.append(response)
            scratchpad.append(f"Observation: {observation}")

            if self.verbose:
                print(f"\nObservation: {observation}")

        # 达到最大步数
        return f"未能在 {self.max_steps} 步内完成任务"

    def _build_prompt(self, question: str, scratchpad: List[str]) -> str:
        """构建提示词"""
        tools_desc = "\n".join([
            tool.format_for_llm()
            for tool in self.tools.values()
        ])

        history = "\n".join(scratchpad) if scratchpad else "无"

        return REACT_PROMPT_TEMPLATE.format(
            tools_description=tools_desc,
            question=question,
            scratchpad=history
        )
```

### 📊 使用示例

```python
from langchain_openai import ChatOpenAI

# 1. 初始化 LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 2. 准备工具
tools = [
    SearchTool(api_key="your_api_key"),
    CalculatorTool(),
    DatabaseTool("users.db")
]

# 3. 创建 Agent
agent = ReActAgent(
    llm=llm,
    tools=tools,
    max_steps=5,
    verbose=True
)

# 4. 执行任务
question = "北京明天天气怎么样？需要带伞吗？"
answer = agent.run(question)
print(f"\n最终答案：{answer}")
```

**执行过程示例**：
```
==================================================
Step 1/5
==================================================

Thought: 我需要搜索北京明天的天气信息
Action: Search["北京明天天气预报"]

Observation: 1. 北京天气预报 - 中国天气网
明天北京有小雨，气温 15-22℃，降雨概率 80%...

==================================================
Step 2/5
==================================================

Thought: 根据搜索结果，明天有雨且降雨概率高，应该建议带伞
Action: Finish[明天北京有小雨，温度 15-22℃，降雨概率 80%，建议带伞]

最终答案：明天北京有小雨，温度 15-22℃，降雨概率 80%，建议带伞
```

### ✅ 优势与局限

**优势**：
- ✅ **灵活性强**：可根据情况动态调整策略
- ✅ **可解释性**：每一步推理过程都清晰可见
- ✅ **容错能力**：工具失败可尝试其他方法
- ✅ **可扩展**：轻松添加新工具

**局限**：
- ❌ **Token 消耗大**：每步都需要调用 LLM
- ❌ **速度慢**：串行执行，无法并行
- ❌ **可能死循环**：缺乏长远规划
- ❌ **成本高**：多次调用 API

### 🎯 适用场景

| 场景 | 推荐度 | 原因 |
|------|-------|------|
| 需要实时信息（天气、新闻、股价） | ⭐⭐⭐⭐⭐ | 必须调用工具获取 |
| 数学计算 | ⭐⭐⭐⭐ | 计算器工具保证准确性 |
| 数据查询 | ⭐⭐⭐⭐ | 需要访问数据库 |
| 简单问答 | ⭐⭐ | 过度设计，用 SimpleAgent |
| 需要规划的任务 | ⭐⭐⭐ | 考虑 Plan-and-Solve |

---

## 3. Plan-and-Solve：计划执行范式

### 🧠 核心思想

**Plan-and-Solve = Planning（规划） + Solving（执行）**

像建筑师盖房子一样：**先画完整图纸（规划），再按图施工（执行）**

### 📋 工作流程

```
┌─────────────────────────────────────┐
│  复杂问题                             │
│  "一个水果店周一卖了15个苹果..."       │
└──────────────┬──────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Phase 1: Planning   │
    │  问题分解为步骤         │
    └──────────┬────────────┘
               │
               ▼
    ┌─────────────────────────────┐
    │  计划列表                     │
    │  1. 计算周一销量              │
    │  2. 计算周二销量（周一×2）     │
    │  3. 计算周三销量（周二-5）     │
    │  4. 求三天总和                │
    └──────────┬──────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Phase 2: Solving    │
    │  逐步执行计划          │
    └──────────┬────────────┘
               │
               ▼
    ┌───────────────────────────┐
    │  执行步骤1                  │
    │  周一销量 = 15个             │
    └──────────┬────────────────┘
               │
               ▼
    ┌───────────────────────────┐
    │  执行步骤2                  │
    │  周二销量 = 15 × 2 = 30个    │
    └──────────┬────────────────┘
               │
               ▼
    ┌───────────────────────────┐
    │  执行步骤3                  │
    │  周三销量 = 30 - 5 = 25个    │
    └──────────┬────────────────┘
               │
               ▼
    ┌───────────────────────────┐
    │  执行步骤4                  │
    │  总和 = 15+30+25 = 70个     │
    └──────────┬────────────────┘
               │
               ▼
    ┌─────────────────────┐
    │  最终答案：70个苹果   │
    └─────────────────────┘
```

### 💻 提示词模板设计

#### Phase 1: Planning 提示词

```python
PLANNER_PROMPT = """
你是一个善于规划的 AI 助手。请将复杂问题分解为简单的执行步骤。

## 规划原则
1. 每个步骤应该是原子操作（不可再分）
2. 步骤之间应该有清晰的依赖关系
3. 步骤描述要具体明确
4. 步骤数量控制在 3-7 个

## 问题
{question}

## 输出格式
请输出 Python 列表格式的计划：
["步骤1的具体描述", "步骤2的具体描述", "步骤3的具体描述", ...]

## 示例
问题：小明有10个苹果，给了小红3个，又买了5个，现在有多少个？
计划：["计算给小红后剩余苹果数", "计算买了苹果后的总数", "得出最终答案"]

现在开始规划：
"""
```

#### Phase 2: Solving 提示词

```python
SOLVER_PROMPT = """
你是一个执行步骤的 AI 助手。请根据计划和历史结果，执行当前步骤。

## 原始问题
{question}

## 完整计划
{plan}

## 已执行的步骤和结果
{execution_history}

## 当前要执行的步骤
步骤 {current_step_number}: {current_step}

## 执行要求
1. 只执行当前步骤，不要跳步
2. 可以参考之前步骤的结果
3. 只输出这一步的结果，不要多余解释
4. 如果需要计算，请给出详细过程

现在执行当前步骤：
"""
```

### 🔄 Plan-and-Solve Agent 完整实现

```python
import json
import re
from typing import List, Dict

class PlanAndSolveAgent:
    """Plan-and-Solve 智能体"""

    def __init__(
        self,
        llm,
        verbose: bool = True,
        max_retries: int = 3
    ):
        self.llm = llm
        self.verbose = verbose
        self.max_retries = max_retries

    def run(self, question: str) -> str:
        """执行 Plan-and-Solve 流程"""

        # Phase 1: Planning
        if self.verbose:
            print(f"\n{'='*50}")
            print("Phase 1: Planning")
            print(f"{'='*50}")

        plan = self._generate_plan(question)

        if self.verbose:
            print(f"\n生成的计划：")
            for i, step in enumerate(plan, 1):
                print(f"  {i}. {step}")

        # Phase 2: Solving
        if self.verbose:
            print(f"\n{'='*50}")
            print("Phase 2: Solving")
            print(f"{'='*50}")

        execution_history = []

        for i, step in enumerate(plan, 1):
            if self.verbose:
                print(f"\n执行步骤 {i}/{len(plan)}: {step}")

            result = self._execute_step(
                question=question,
                plan=plan,
                current_step_number=i,
                current_step=step,
                execution_history=execution_history
            )

            execution_history.append({
                "step": step,
                "result": result
            })

            if self.verbose:
                print(f"结果: {result}")

        # 最后一步的结果就是最终答案
        final_answer = execution_history[-1]["result"]

        return final_answer

    def _generate_plan(self, question: str) -> List[str]:
        """生成执行计划"""
        prompt = PLANNER_PROMPT.format(question=question)

        for attempt in range(self.max_retries):
            response = self.llm.generate(prompt)

            # 尝试解析为列表
            plan = self._parse_plan(response)

            if plan:
                return plan

            # 解析失败，重试
            if self.verbose:
                print(f"解析失败，重试 {attempt + 1}/{self.max_retries}")

        # 全部失败，返回默认计划
        return ["分析问题", "解决问题"]

    def _parse_plan(self, response: str) -> List[str]:
        """解析计划文本为列表"""
        try:
            # 方式1：尝试解析 Python 列表
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                plan_str = match.group(0)
                plan = json.loads(plan_str)
                if isinstance(plan, list) and all(isinstance(s, str) for s in plan):
                    return plan
        except:
            pass

        try:
            # 方式2：按行分割
            lines = response.strip().split('\n')
            plan = []
            for line in lines:
                # 匹配 "1. xxx" 或 "- xxx" 格式
                match = re.match(r'^\s*[\d\-\*]\s*[.、]?\s*(.+)$', line)
                if match:
                    plan.append(match.group(1).strip())

            if plan:
                return plan
        except:
            pass

        return None

    def _execute_step(
        self,
        question: str,
        plan: List[str],
        current_step_number: int,
        current_step: str,
        execution_history: List[Dict]
    ) -> str:
        """执行单个步骤"""

        # 格式化历史记录
        history_text = ""
        for i, record in enumerate(execution_history, 1):
            history_text += f"步骤{i}: {record['step']}\n"
            history_text += f"结果: {record['result']}\n\n"

        if not history_text:
            history_text = "无（这是第一步）"

        # 构建提示词
        prompt = SOLVER_PROMPT.format(
            question=question,
            plan="\n".join([f"{i+1}. {s}" for i, s in enumerate(plan)]),
            execution_history=history_text,
            current_step_number=current_step_number,
            current_step=current_step
        )

        # 执行
        result = self.llm.generate(prompt)

        return result.strip()
```

### 📊 使用示例

```python
from langchain_openai import ChatOpenAI

# 1. 初始化 LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 2. 创建 Agent
agent = PlanAndSolveAgent(
    llm=llm,
    verbose=True
)

# 3. 复杂数学应用题
question = """
一个水果店周一卖了15个苹果，
周二卖的是周一的两倍，
周三比周二少5个。
请问三天总共卖了多少个苹果？
"""

answer = agent.run(question)
print(f"\n最终答案：{answer}")
```

**执行过程示例**：
```
==================================================
Phase 1: Planning
==================================================

生成的计划：
  1. 计算周一卖出的苹果数
  2. 计算周二卖出的苹果数（周一的两倍）
  3. 计算周三卖出的苹果数（周二减5）
  4. 计算三天总共卖出的苹果数

==================================================
Phase 2: Solving
==================================================

执行步骤 1/4: 计算周一卖出的苹果数
结果: 15个

执行步骤 2/4: 计算周二卖出的苹果数（周一的两倍）
结果: 15 × 2 = 30个

执行步骤 3/4: 计算周三卖出的苹果数（周二减5）
结果: 30 - 5 = 25个

执行步骤 4/4: 计算三天总共卖出的苹果数
结果: 15 + 30 + 25 = 70个

最终答案：70个
```

### 🎨 高级变体：Plan-and-Execute

在基础 Plan-and-Solve 上增强，支持工具调用：

```python
class PlanAndExecuteAgent(PlanAndSolveAgent):
    """Plan-and-Execute（支持工具调用）"""

    def __init__(self, llm, tools: List[Tool], **kwargs):
        super().__init__(llm, **kwargs)
        self.tools = {tool.name: tool for tool in tools}

    def _execute_step(self, **kwargs) -> str:
        """执行步骤（支持工具调用）"""
        # 先调用父类方法获取 LLM 输出
        result = super()._execute_step(**kwargs)

        # 检测是否需要调用工具
        tool_match = re.search(r'(\w+)\[(.*?)\]', result)

        if tool_match and tool_match.group(1) in self.tools:
            tool_name = tool_match.group(1)
            tool_input = tool_match.group(2)

            if self.verbose:
                print(f"  → 调用工具: {tool_name}[{tool_input}]")

            # 执行工具
            tool_result = self.tools[tool_name].run(tool_input)

            # 将工具结果作为这一步的输出
            return tool_result

        return result
```

### ✅ 优势与局限

**优势**：
- ✅ **思路清晰**：计划一目了然
- ✅ **稳定性好**：按部就班不易出错
- ✅ **易于调试**：每步结果可验证
- ✅ **可并行**：某些步骤可并行执行

**局限**：
- ❌ **缺乏灵活性**：计划定了就难调整
- ❌ **依赖规划能力**：规划不好全盘皆输
- ❌ **不适合探索性任务**：无法根据中间结果调整
- ❌ **错误传播**：一步错，后面都错

### 🎯 适用场景

| 场景 | 推荐度 | 原因 |
|------|-------|------|
| 数学应用题 | ⭐⭐⭐⭐⭐ | 步骤明确，易分解 |
| 代码生成 | ⭐⭐⭐⭐ | 可先设计架构 |
| 写作任务 | ⭐⭐⭐⭐ | 可先列大纲 |
| 需要实时信息 | ⭐⭐ | 不如 ReAct 灵活 |
| 需要高质量输出 | ⭐⭐⭐ | 考虑 Reflection |

---

## 4. Reflection：自我反思范式

### 🧠 核心思想

**Reflection = Execution（执行） + Evaluation（评估） + Refinement（改进）**

像作家修稿一样：**写初稿 → 自我审查 → 修改 → 再审查 → 再修改 → ...**

### 📋 工作流程

```
┌─────────────────────────────────┐
│  任务：写一个查找素数的函数        │
└──────────────┬──────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Round 1: Execution  │
    │  写初稿               │
    └──────────┬────────────┘
               │
               ▼
    ┌────────────────────────────┐
    │  初稿代码                   │
    │  def is_prime(n):          │
    │    for i in range(2, n):   │
    │      if n % i == 0:        │
    │        return False        │
    │    return True             │
    └──────────┬────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Reflection          │
    │  自我评估             │
    └──────────┬────────────┘
               │
               ▼
    ┌────────────────────────────────┐
    │  反馈：                          │
    │  1. 时间复杂度高 O(n)            │
    │  2. 可优化为 O(√n)               │
    │  3. 未处理 n<=1 的情况            │
    └──────────┬────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Refinement          │
    │  改进代码             │
    └──────────┬────────────┘
               │
               ▼
    ┌────────────────────────────────┐
    │  改进版代码                       │
    │  def is_prime(n):               │
    │    if n <= 1: return False      │
    │    for i in range(2, int(n**0.5)+1): │
    │      if n % i == 0:             │
    │        return False             │
    │    return True                  │
    └──────────┬────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Reflection Again    │
    │  再次评估             │
    └──────────┬────────────┘
               │
               ▼
    ┌─────────────────────────┐
    │  反馈：已优化良好，无需改进 │
    └──────────┬──────────────┘
               │
               ▼
    ┌─────────────────┐
    │  输出最终版本     │
    └─────────────────┘
```

### 💻 提示词模板设计

#### Phase 1: 初次执行

```python
INITIAL_EXECUTION_PROMPT = """
请完成以下任务：

{task}

要求：
1. 直接给出你的最佳方案
2. 不需要解释过程
3. 确保输出完整可用

现在开始：
"""
```

#### Phase 2: 反思评估

```python
REFLECTION_PROMPT = """
你是一位严格的评审员，请评估以下方案的质量。

## 原始任务
{task}

## 当前方案
{current_attempt}

## 评估维度
请从以下维度评估：
1. **正确性**：方案是否正确解决问题？
2. **效率**：是否有性能问题？时间/空间复杂度如何？
3. **完整性**：是否考虑了边界情况？
4. **可读性**：代码/文本是否清晰易懂？
5. **最佳实践**：是否遵循领域最佳实践？

## 输出格式
请按以下格式输出：

### 优点
- [列出做得好的地方]

### 问题
- [列出存在的问题，具体指出]

### 改进建议
- [给出具体的改进方向]

### 结论
[优秀/良好/需改进]

现在开始评估：
"""
```

#### Phase 3: 改进优化

```python
REFINEMENT_PROMPT = """
根据反馈意见，改进你的方案。

## 原始任务
{task}

## 上一版方案
{last_attempt}

## 反馈意见
{feedback}

## 改进要求
1. 针对反馈中的问题逐一解决
2. 保留原方案的优点
3. 输出改进后的完整方案
4. 不要添加多余说明

现在开始改进：
"""
```

### 🔄 Reflection Agent 完整实现

```python
from typing import Optional

class ReflectionAgent:
    """Reflection 智能体"""

    def __init__(
        self,
        llm,
        max_reflections: int = 3,
        quality_threshold: str = "良好",
        verbose: bool = True
    ):
        self.llm = llm
        self.max_reflections = max_reflections
        self.quality_threshold = quality_threshold
        self.verbose = verbose

        # 质量等级
        self.quality_levels = {
            "需改进": 1,
            "良好": 2,
            "优秀": 3
        }

    def run(self, task: str) -> str:
        """执行 Reflection 流程"""

        # Round 0: 初次执行
        if self.verbose:
            print(f"\n{'='*50}")
            print("Round 0: 初次执行")
            print(f"{'='*50}")

        current_attempt = self._execute_task(task)

        if self.verbose:
            print(f"\n初稿:\n{current_attempt}")

        # 反思-改进循环
        for round_num in range(1, self.max_reflections + 1):
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Round {round_num}: 反思与改进")
                print(f"{'='*50}")

            # 反思评估
            feedback = self._reflect(task, current_attempt)

            if self.verbose:
                print(f"\n反馈:\n{feedback}")

            # 检查质量是否达标
            quality = self._extract_quality(feedback)

            if self.verbose:
                print(f"\n当前质量: {quality}")

            if self._is_good_enough(quality):
                if self.verbose:
                    print(f"\n质量已达标，停止反思")
                break

            # 改进
            current_attempt = self._refine(task, current_attempt, feedback)

            if self.verbose:
                print(f"\n改进后:\n{current_attempt}")

        return current_attempt

    def _execute_task(self, task: str) -> str:
        """初次执行任务"""
        prompt = INITIAL_EXECUTION_PROMPT.format(task=task)
        return self.llm.generate(prompt).strip()

    def _reflect(self, task: str, current_attempt: str) -> str:
        """反思评估"""
        prompt = REFLECTION_PROMPT.format(
            task=task,
            current_attempt=current_attempt
        )
        return self.llm.generate(prompt).strip()

    def _refine(self, task: str, last_attempt: str, feedback: str) -> str:
        """改进优化"""
        prompt = REFINEMENT_PROMPT.format(
            task=task,
            last_attempt=last_attempt,
            feedback=feedback
        )
        return self.llm.generate(prompt).strip()

    def _extract_quality(self, feedback: str) -> str:
        """从反馈中提取质量结论"""
        for level in ["优秀", "良好", "需改进"]:
            if level in feedback:
                return level
        return "需改进"

    def _is_good_enough(self, quality: str) -> bool:
        """判断质量是否达标"""
        current_level = self.quality_levels.get(quality, 0)
        threshold_level = self.quality_levels.get(self.quality_threshold, 2)
        return current_level >= threshold_level
```

### 📊 使用示例

```python
from langchain_openai import ChatOpenAI

# 1. 初始化 LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# 2. 创建 Agent
agent = ReflectionAgent(
    llm=llm,
    max_reflections=3,
    quality_threshold="优秀",
    verbose=True
)

# 3. 代码优化任务
task = """
写一个 Python 函数 is_prime(n)，判断一个数是否为素数。
要求：高效、健壮、易读。
"""

result = agent.run(task)
print(f"\n最终代码:\n{result}")
```

**执行过程示例**：
```
==================================================
Round 0: 初次执行
==================================================

初稿:
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

==================================================
Round 1: 反思与改进
==================================================

反馈:
### 优点
- 正确处理了 n < 2 的情况
- 逻辑清晰易懂

### 问题
- 时间复杂度 O(n)，数字很大时效率低
- 可以优化为只检查到 √n

### 改进建议
- 将循环范围改为 range(2, int(n**0.5) + 1)
- 可以先排除偶数，进一步优化

### 结论
需改进

当前质量: 需改进

改进后:
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

==================================================
Round 2: 反思与改进
==================================================

反馈:
### 优点
- 优化了时间复杂度到 O(√n)
- 排除偶数，进一步提升效率
- 边界情况处理完整

### 问题
- 无明显问题

### 改进建议
- 可添加类型提示和文档字符串

### 结论
优秀

当前质量: 优秀

质量已达标，停止反思

最终代码:
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
```

### 🎨 高级变体

#### 1. 多评审员 Reflection

使用多个"评审员"从不同角度评估：

```python
class MultiReflectionAgent(ReflectionAgent):
    """多评审员反思"""

    def _reflect(self, task: str, current_attempt: str) -> str:
        """多角度反思"""
        reviewers = [
            ("性能专家", "重点评估时间复杂度和空间复杂度"),
            ("安全专家", "重点评估安全性和异常处理"),
            ("代码质量专家", "重点评估可读性和最佳实践")
        ]

        feedbacks = []

        for reviewer_name, focus in reviewers:
            prompt = f"""
你是一位{reviewer_name}。{focus}

任务: {task}
方案: {current_attempt}

请给出你的评估意见。
"""
            feedback = self.llm.generate(prompt)
            feedbacks.append(f"## {reviewer_name}\n{feedback}")

        return "\n\n".join(feedbacks)
```

#### 2. 外部验证 Reflection

结合实际测试结果：

```python
class ValidatedReflectionAgent(ReflectionAgent):
    """带验证的反思"""

    def __init__(self, llm, test_cases=None, **kwargs):
        super().__init__(llm, **kwargs)
        self.test_cases = test_cases or []

    def _reflect(self, task: str, current_attempt: str) -> str:
        """反思 + 测试验证"""
        # 先执行测试
        test_results = self._run_tests(current_attempt)

        # 构建包含测试结果的反思提示
        prompt = f"""
评估以下方案：

任务: {task}
方案: {current_attempt}

测试结果:
{test_results}

请基于测试结果给出评估和改进建议。
"""
        return self.llm.generate(prompt)

    def _run_tests(self, code: str) -> str:
        """运行测试用例"""
        results = []

        for test in self.test_cases:
            try:
                # 执行代码（注意：实际应该用安全的沙箱）
                exec(code, globals())

                # 运行测试
                result = eval(test["input"])
                expected = test["expected"]

                if result == expected:
                    results.append(f"✓ {test['name']}: 通过")
                else:
                    results.append(f"✗ {test['name']}: 期望 {expected}, 得到 {result}")
            except Exception as e:
                results.append(f"✗ {test['name']}: 错误 - {str(e)}")

        return "\n".join(results)
```

### ✅ 优势与局限

**优势**：
- ✅ **质量最高**：经过多轮打磨
- ✅ **自我纠错**：能发现并修复问题
- ✅ **持续改进**：从合格到优秀
- ✅ **可解释性强**：每轮改进都有理由

**局限**：
- ❌ **成本高**：每轮 3 次 LLM 调用
- ❌ **耗时长**：多轮迭代时间长
- ❌ **可能过度优化**：反复修改反而变差
- ❌ **需要强模型**：评估能力要求高

### 🎯 适用场景

| 场景 | 推荐度 | 原因 |
|------|-------|------|
| 关键代码生成 | ⭐⭐⭐⭐⭐ | 质量要求高 |
| 重要文章写作 | ⭐⭐⭐⭐⭐ | 需要反复打磨 |
| 算法优化 | ⭐⭐⭐⭐ | 需要性能改进 |
| 快速响应 | ⭐ | 太慢 |
| 简单任务 | ⭐ | 过度设计 |

---

## 5. 范式选择与组合

### 🎯 三种范式对比总结

| 维度 | ReAct | Plan-and-Solve | Reflection |
|------|-------|----------------|-----------|
| **核心思想** | 思考-行动循环 | 规划后执行 | 执行-反思-改进 |
| **调用次数** | 多次（每步1次） | 中等（规划1次+执行N次） | 最多（每轮3次） |
| **速度** | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐ 较快 | ⭐ 很慢 |
| **成本** | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐ 较低 | ⭐ 很高 |
| **质量** | ⭐⭐⭐ 不错 | ⭐⭐⭐⭐ 稳定 | ⭐⭐⭐⭐⭐ 最佳 |
| **灵活性** | ⭐⭐⭐⭐⭐ 最灵活 | ⭐⭐ 不灵活 | ⭐⭐⭐ 中等 |
| **可控性** | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐⭐ 最可控 | ⭐⭐⭐ 中等 |
| **学习难度** | ⭐⭐⭐ 中等 | ⭐⭐ 较简单 | ⭐⭐⭐⭐ 较难 |

### 🔀 范式组合策略

实际应用中，经常需要组合多种范式：

#### 策略1：Plan + ReAct

**场景**：复杂任务需要规划，但执行需要灵活调用工具

```python
class PlanReActAgent:
    """Plan-and-Solve + ReAct 组合"""

    def __init__(self, llm, tools):
        self.planner = PlanAndSolveAgent(llm)
        self.reactor = ReActAgent(llm, tools)

    def run(self, question: str) -> str:
        # 1. 使用 Plan-and-Solve 生成计划
        plan = self.planner._generate_plan(question)

        # 2. 使用 ReAct 执行每个步骤
        results = []
        for step in plan:
            result = self.reactor.run(step)
            results.append(result)

        # 3. 合并结果
        return self._merge_results(results)
```

#### 策略2：ReAct + Reflection

**场景**：需要工具调用，且对质量要求高

```python
class ReActReflectionAgent:
    """ReAct + Reflection 组合"""

    def __init__(self, llm, tools):
        self.reactor = ReActAgent(llm, tools)
        self.reflector = ReflectionAgent(llm, max_reflections=2)

    def run(self, question: str) -> str:
        # 1. ReAct 获取初步答案
        initial_answer = self.reactor.run(question)

        # 2. Reflection 优化答案
        refined_answer = self.reflector.run(
            f"优化以下回答：\n问题：{question}\n回答：{initial_answer}"
        )

        return refined_answer
```

#### 策略3：三者全结合

**场景**：超高质量要求的复杂任务

```python
class HybridAgent:
    """混合智能体"""

    def __init__(self, llm, tools):
        self.planner = PlanAndSolveAgent(llm)
        self.reactor = ReActAgent(llm, tools)
        self.reflector = ReflectionAgent(llm)

    def run(self, question: str) -> str:
        # 1. Plan: 生成计划
        plan = self.planner._generate_plan(question)

        # 2. ReAct: 执行计划（每步可用工具）
        results = []
        for step in plan:
            result = self.reactor.run(step)
            results.append(result)

        # 3. 合并初步答案
        initial_answer = self._merge_results(results)

        # 4. Reflection: 优化答案
        final_answer = self.reflector.run(
            f"优化答案：\n问题：{question}\n初步答案：{initial_answer}"
        )

        return final_answer
```

### 📊 决策树：如何选择范式？

```
                      开始
                       │
                       ▼
              ┌─────────────────┐
              │ 需要调用外部工具？│
              └────┬────────┬───┘
                   │YES     │NO
                   ▼        ▼
          ┌───────────┐  ┌──────────────┐
          │ ReAct系列  │  │能提前规划步骤？│
          └───────────┘  └──┬────────┬──┘
                             │YES     │NO
                             ▼        ▼
                    ┌──────────────┐ ┌─────────────┐
                    │Plan-and-Solve│ │直接LLM回答？│
                    └──────────────┘ └──┬──────┬──┘
                                        │YES   │NO
                                        ▼      ▼
                               ┌──────────┐ ┌──────────┐
                               │SimpleAgent│ │Reflection│
                               └──────────┘ └──────────┘
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │追求极致质量时使用│
                                    └──────────────────┘
```

### 💡 实践建议

#### 新手学习路径

```
第1周：理解概念
  ├─ 阅读本文档
  ├─ 画流程图
  └─ 对比三种范式

第2周：动手实践
  ├─ 从 Plan-and-Solve 开始（最简单）
  ├─ 运行示例代码
  └─ 修改提示词

第3周：进阶学习
  ├─ 实现 ReAct（最实用）
  ├─ 添加自定义工具
  └─ 处理异常情况

第4周：高级挑战
  ├─ 实现 Reflection（最高级）
  ├─ 尝试范式组合
  └─ 构建实际项目
```

#### 生产环境建议

**性能优化**：
```python
# 1. 缓存常见查询
cache = {}

def cached_llm_call(prompt):
    if prompt in cache:
        return cache[prompt]

    result = llm.generate(prompt)
    cache[prompt] = result
    return result

# 2. 设置超时
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("LLM 调用超时")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30秒超时
```

**成本控制**：
```python
# 使用便宜模型做规划，贵模型做执行
planner_llm = ChatOpenAI(model="gpt-3.5-turbo")  # 便宜
executor_llm = ChatOpenAI(model="gpt-4")  # 贵但准确

agent = PlanAndSolveAgent(
    planner_llm=planner_llm,
    executor_llm=executor_llm
)
```

**错误处理**：
```python
class RobustAgent:
    def run(self, question: str) -> str:
        max_retries = 3

        for attempt in range(max_retries):
            try:
                return self._run_internal(question)
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"执行失败：{str(e)}"
                time.sleep(2 ** attempt)  # 指数退避
```

---

## 6. 本章总结

### 🎯 核心要点回顾

**三种范式本质**：
```
ReAct        = 灵活的探索者（侦探）
Plan-Solve   = 严谨的规划者（建筑师）
Reflection   = 追求完美的工匠（作家）
```

**关键设计要素**：
1. ✅ **提示词工程**：每种范式都有特定的提示词模板
2. ✅ **工具系统**：工具是 Agent 的"手脚"
3. ✅ **循环控制**：避免死循环，设置最大步数
4. ✅ **解析机制**：从 LLM 输出中提取结构化信息
5. ✅ **错误处理**：优雅地处理异常情况

### 📈 学习收获

**理论层面**：
- 理解了智能体不是一个模型，而是"模型 + 范式 + 工具"
- 掌握了三种经典范式的设计思想和工作原理
- 明白了不同范式的适用场景和权衡

**实践层面**：
- 学会了如何设计提示词模板
- 掌握了工具系统的实现方式
- 了解了范式组合的策略

**工程层面**：
- 学会了解析 LLM 输出
- 掌握了循环控制和异常处理
- 了解了性能优化和成本控制

### 🚀 下一步学习

```
第4章（当前）：经典范式
         ↓
第5章：Prompt Engineering 深化
  - 如何写出高质量提示词
  - Few-shot 示例设计
  - 思维链技术
         ↓
第7章：构建你的 Agent 框架
  - 框架化设计思想
  - 组件抽象
  - 工具注册机制
         ↓
第8章：记忆与 RAG
  - 长期记忆
  - 向量检索
  - 知识库集成
```

### 💬 常见问题速查

**Q: 新手应该从哪个范式开始学？**
A: Plan-and-Solve → ReAct → Reflection（由简入繁）

**Q: 哪个范式最实用？**
A: ReAct，因为大多数任务都需要调用工具

**Q: 能不能混合使用？**
A: 完全可以，实际项目中经常需要组合

**Q: 如何降低成本？**
A: 使用便宜模型 + 缓存 + 减少步数

**Q: 提示词怎么写？**
A: 参考本文模板，然后根据实际情况调整

### 🔗 相关资源

- **论文**:
  - ReAct: [Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
  - Plan-and-Solve: [Plan-and-Solve Prompting](https://arxiv.org/abs/2305.04091)
  - Reflection: [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)

- **代码示例**:
  - [HelloAgents GitHub](https://github.com/jjyaoao/helloagents)
  - [LangChain Examples](https://github.com/langchain-ai/langchain)

- **在线教程**:
  - [Hello Agents 官方文档](https://datawhalechina.github.io/hello-agents/)

---

## 📝 快速参考

### ReAct 最小实现
```python
llm = ChatOpenAI(model="gpt-3.5-turbo")
tools = [SearchTool(), CalculatorTool()]
agent = ReActAgent(llm, tools, max_steps=5)
result = agent.run("北京明天天气怎么样？")
```

### Plan-and-Solve 最小实现
```python
llm = ChatOpenAI(model="gpt-3.5-turbo")
agent = PlanAndSolveAgent(llm)
result = agent.run("小明有10个苹果...")
```

### Reflection 最小实现
```python
llm = ChatOpenAI(model="gpt-4")
agent = ReflectionAgent(llm, max_reflections=3)
result = agent.run("写一个高效的素数判断函数")
```

---

**最后更新**: 2025-11-29
**版本**: 优化版 v2.0
**适合人群**: 初学者、实践者、架构师

**Happy Learning! 🎉**
