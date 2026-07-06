# 第四章《智能体经典范式构建》课后习题解答

---

## 习题一：三种范式核心对比

### 题目
请对比 ReAct、Plan-and-Solve、Reflection 三种范式的核心思想、适用场景和优缺点。

### 解答

#### 1. 核心思想对比

| 范式 | 核心思想 | 工作流程 | 类比 |
|------|---------|---------|------|
| **ReAct** | 边思考边行动，思维与行动交替进行 | Thought → Action → Observation（循环） | 侦探破案 |
| **Plan-and-Solve** | 先全局规划，再分步执行 | Plan（规划）→ Solve（逐步执行） | 建筑师盖房 |
| **Reflection** | 生成-评审-改进的迭代优化 | Generate → Reflect → Refine（循环） | 匠人打磨 |

#### 2. 适用场景分析

**ReAct 适用场景：**
- 需要调用外部工具（搜索、API、数据库）
- 问题不确定，需要根据中间结果调整策略
- 实时查询类任务（天气、新闻、股价）
- 需要灵活应变的交互式任务

**Plan-and-Solve 适用场景：**
- 数学应用题、逻辑推理题
- 可以明确分步的任务
- 写作（先列大纲再展开）
- 编程（先设计架构再实现）
- 需要稳定、可预测输出的场景

**Reflection 适用场景：**
- 对质量要求极高的任务
- 代码生成（需要检查bug和性能）
- 重要文档撰写（报告、论文）
- 创意性工作（需要不断优化）
- 有充足时间和预算的场景

#### 3. 优缺点详解

| 范式 | 优点 | 缺点 |
|------|------|------|
| **ReAct** | ✅ 灵活性强，能根据反馈调整<br>✅ 能使用外部工具<br>✅ 可解释性好（每步有思考过程） | ❌ 调用次数多，成本较高<br>❌ 可能陷入循环<br>❌ 缺乏全局规划 |
| **Plan-and-Solve** | ✅ 思路清晰，有整体规划<br>✅ 输出稳定可预测<br>✅ 适合结构化问题 | ❌ 灵活性差，计划难以调整<br>❌ 不适合需要外部信息的任务<br>❌ 计划可能不完善 |
| **Reflection** | ✅ 输出质量最高<br>✅ 能自我纠错和优化<br>✅ 持续改进直到满意 | ❌ 速度最慢<br>❌ 成本最高（多轮调用）<br>❌ 实现复杂度高 |

---

## 习题二：ReAct 范式深入理解

### 题目
1. 解释 ReAct 中 Thought、Action、Observation 各自的作用
2. 为什么 ReAct 需要工具？如果没有工具会怎样？
3. 如何防止 ReAct 陷入死循环？

### 解答

#### 1. TAO 三元素详解

**Thought（思考）**
- **作用**：分析当前状态，决定下一步行动
- **内容**：包含推理过程、假设、判断
- **示例**：
  ```
  Thought: 用户问华为最新手机，我需要搜索获取最新信息。
  ```

**Action（行动）**
- **作用**：调用外部工具执行具体操作
- **格式**：通常是 `工具名[参数]` 的形式
- **示例**：
  ```
  Action: Search[华为最新手机 2024]
  ```

**Observation（观察）**
- **作用**：接收工具返回的结果，作为下一轮思考的输入
- **来源**：工具执行后的返回值
- **示例**：
  ```
  Observation: 华为Mate 70系列于2024年11月发布，主打卫星通信和AI功能...
  ```

#### 2. 工具的必要性

**为什么需要工具：**
- LLM 的知识是静态的（训练数据截止时间）
- LLM 不能执行实际操作（计算、查询、写文件）
- 工具让 AI 从"纸上谈兵"变成"能动手做事"

**没有工具的后果：**
```python
# 无工具的 ReAct
用户：今天北京天气怎么样？
AI Thought：我不知道今天的天气...
AI Action：？？？（没有工具可调用）
AI：抱歉，我无法获取实时天气信息。
```

**常用工具类型：**
- 搜索引擎（获取信息）
- 计算器（数学运算）
- API 调用（获取实时数据）
- 数据库查询（获取结构化数据）
- 代码执行（运行代码）

#### 3. 防止死循环的策略

```python
# 策略1：设置最大步数
max_steps = 5
for i in range(max_steps):
    thought, action = agent.step()
    if action == "Finish":
        break
else:
    return "超过最大步数，强制终止"

# 策略2：检测重复行动
action_history = []
for i in range(max_steps):
    action = agent.get_action()
    if action in action_history[-3:]:  # 最近3步有重复
        return "检测到循环，终止"
    action_history.append(action)

# 策略3：超时机制
import time
start = time.time()
while not done:
    agent.step()
    if time.time() - start > 60:  # 60秒超时
        return "超时终止"
```

---

## 习题三：Plan-and-Solve 实践

### 题目
给定数学应用题：
> "小明周一卖了15个苹果，周二卖的是周一的2倍，周三卖的比周二少5个。问三天共卖了多少个苹果？"

请使用 Plan-and-Solve 范式分两阶段解决。

### 解答

#### 阶段1：规划（Planning）

**Prompt 设计**：
```
请将以下问题分解为明确的计算步骤：
"小明周一卖了15个苹果，周二卖的是周一的2倍，周三卖的比周二少5个。问三天共卖了多少个苹果？"

要求：
1. 每个步骤只做一个计算
2. 步骤之间要有依赖关系
3. 最后一步给出最终答案
```

**生成的计划**：
```python
plan = [
    "Step 1: 计算周一销量 = 15个（已知）",
    "Step 2: 计算周二销量 = 周一 × 2",
    "Step 3: 计算周三销量 = 周二 - 5",
    "Step 4: 计算总销量 = 周一 + 周二 + 周三"
]
```

#### 阶段2：执行（Solving）

**逐步执行**：

```python
# Step 1
context = "问题：小明卖苹果问题"
result_1 = "周一销量 = 15个"
# 变量：monday = 15

# Step 2
context += f"\n已知：{result_1}"
prompt = "根据已知信息，计算周二销量（周一的2倍）"
result_2 = "周二销量 = 15 × 2 = 30个"
# 变量：tuesday = 30

# Step 3
context += f"\n已知：{result_2}"
prompt = "根据已知信息，计算周三销量（周二少5个）"
result_3 = "周三销量 = 30 - 5 = 25个"
# 变量：wednesday = 25

# Step 4
context += f"\n已知：{result_3}"
prompt = "计算三天总销量"
result_4 = "总销量 = 15 + 30 + 25 = 70个"
```

**最终答案**：三天共卖了 **70个苹果**

#### LangChain 实现示例

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4", temperature=0)

# 规划阶段
planning_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
将以下问题分解为计算步骤（JSON格式）：
{question}

输出格式：
{{"steps": ["step1", "step2", ...]}}
"""
)

# 执行阶段
solving_prompt = PromptTemplate(
    input_variables=["question", "history", "current_step"],
    template="""
问题：{question}
已完成步骤：{history}
当前步骤：{current_step}

请执行当前步骤并给出结果。
"""
)

# 主流程
question = "小明周一卖了15个苹果..."

# 1. 生成计划
plan_response = llm.invoke(planning_prompt.format(question=question))
steps = parse_json(plan_response)["steps"]

# 2. 逐步执行
history = []
for step in steps:
    result = llm.invoke(solving_prompt.format(
        question=question,
        history=history,
        current_step=step
    ))
    history.append(f"{step}: {result}")

print(f"最终答案: {history[-1]}")
```

---

## 习题四：Reflection 范式实现

### 题目
使用 Reflection 范式实现一个代码优化器：
1. 生成一个查找素数的代码
2. 反思代码的性能问题
3. 优化并生成最终版本

### 解答

#### 第一轮：初始生成

**初稿代码**：
```python
def find_primes(n):
    """找出小于n的所有素数"""
    primes = []
    for num in range(2, n):
        is_prime = True
        for i in range(2, num):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes
```

#### 第一轮：反思

**反思 Prompt**：
```
请评审以下代码，从以下方面分析：
1. 时间复杂度
2. 空间复杂度
3. 可优化的点
4. 潜在bug

代码：
[初稿代码]
```

**反思结果**：
```
评审意见：
1. 时间复杂度：O(n²)，效率很低
   - 内层循环可以只检查到 sqrt(num)

2. 空间复杂度：O(k)，k为素数个数，可接受

3. 优化建议：
   - 使用埃拉托斯特尼筛法，时间复杂度可降至 O(n log log n)
   - 或至少优化内层循环到 O(√n)

4. 潜在问题：
   - 当 n 很大时性能极差
   - 没有输入验证

建议：重写为筛法实现
```

#### 第二轮：优化

**优化后代码**：
```python
def find_primes(n):
    """找出小于n的所有素数（埃拉托斯特尼筛法）"""
    if n < 2:
        return []

    # 初始化筛子
    is_prime = [True] * n
    is_prime[0] = is_prime[1] = False

    # 筛法核心
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            # 将i的倍数标记为非素数
            for j in range(i*i, n, i):
                is_prime[j] = False

    return [i for i in range(n) if is_prime[i]]
```

#### 第二轮：反思

**反思结果**：
```
评审意见：
1. 时间复杂度：O(n log log n)，非常优秀 ✅
2. 空间复杂度：O(n)，需要额外空间，可接受 ✅
3. 代码质量：
   - 添加了输入验证 ✅
   - 使用列表推导，Pythonic ✅
   - 注释清晰 ✅
4. 潜在优化：
   - 可以用 bitarray 进一步优化空间
   - 但对于一般应用已经足够

结论：代码质量良好，无需进一步优化。
```

#### LangChain 实现框架

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# 生成器角色
generator_prompt = PromptTemplate(
    input_variables=["task", "feedback"],
    template="""
任务：{task}
之前的反馈：{feedback}

请根据反馈生成/改进代码。如果没有反馈，请生成初始版本。
"""
)

# 评审员角色
reflector_prompt = PromptTemplate(
    input_variables=["code", "task"],
    template="""
任务：{task}
代码：{code}
{code}
请从以下方面评审代码：
1. 时间/空间复杂度
2. 代码质量和可读性
3. 潜在bug
4. 优化建议

如果代码已经很好，请回复"APPROVED"。
)
```

```python
def reflection_loop(task, max_iterations=3):
    feedback = ""
    code = ""

    for i in range(max_iterations):
        # 生成/优化
        code = llm.invoke(generator_prompt.format(
            task=task,
            feedback=feedback
        ))

        # 反思
        feedback = llm.invoke(reflector_prompt.format(
            code=code,
            task=task
        ))

        if "APPROVED" in feedback:
            print(f"通过评审，共迭代 {i+1} 次")
            return code

    print(f"达到最大迭代次数 {max_iterations}")
    return code

# 使用
final_code = reflection_loop("实现查找素数的高效算法")
```

---

## 习题五：范式选择与组合

### 题目
分析以下场景应该使用哪种范式（或组合）：
1. 智能客服系统
2. 自动代码审查工具
3. 旅行规划助手
4. 实时股票分析系统

### 解答

#### 场景1：智能客服系统

**推荐方案：ReAct + Plan-and-Solve 组合**

**分析**：
- 需要查询订单状态（ReAct + 工具）
- 需要根据问题类型分步处理（Plan-and-Solve）
- 需要灵活应对用户追问（ReAct）

**实现架构**：
```
用户问题
    ↓
Plan-and-Solve（理解意图，规划处理流程）
    ↓
ReAct（执行每个步骤，调用工具查询信息）
    ↓
生成回复
```

#### 场景2：自动代码审查工具

**推荐方案：Reflection 范式**

**分析**：
- 需要高质量的评审意见
- 需要多角度检查（安全、性能、可读性）
- 不追求速度，追求准确性

**实现架构**：
```
提交代码
    ↓
Reflection 循环：
  ├─ 生成评审意见
  ├─ 反思评审是否全面
  ├─ 补充遗漏的问题
  └─ 直到评审完整
    ↓
输出评审报告
```

#### 场景3：旅行规划助手

**推荐方案：Plan-and-Solve + ReAct 组合**

**分析**：
- 需要先做整体规划（行程安排）
- 需要查询实时信息（机票、酒店、天气）
- 需要根据约束调整计划

**实现架构**：
```
用户需求（目的地、时间、预算）
    ↓
Plan-and-Solve（生成行程大纲）
    ↓
ReAct（执行每天的详细规划）
  ├─ 搜索酒店
  ├─ 搜索景点
  ├─ 查询天气
  └─ 调整计划
    ↓
输出完整行程
```

#### 场景4：实时股票分析系统

**推荐方案：ReAct 范式**

**分析**：
- 高度依赖实时数据
- 需要快速响应市场变化
- 需要调用多个数据源API

**实现架构**：
```
分析请求
    ↓
ReAct 循环：
  ├─ Thought: 需要获取哪些数据？
  ├─ Action: 调用股票API
  ├─ Observation: 获取实时价格
  ├─ Thought: 需要分析技术指标
  ├─ Action: 计算MA、MACD等
  ├─ Observation: 指标结果
  └─ Thought: 可以给出分析了
    ↓
输出分析报告
```

#### 综合对比表

| 场景 | 主要范式 | 辅助范式 | 关键需求 |
|------|---------|---------|---------|
| 智能客服 | ReAct | Plan-and-Solve | 灵活+结构化 |
| 代码审查 | Reflection | - | 高质量 |
| 旅行规划 | Plan-and-Solve | ReAct | 规划+查询 |
| 股票分析 | ReAct | - | 实时数据 |

---

## 习题六：工具设计实践

### 题目
为 ReAct Agent 设计以下工具，包括名称、描述和示例用法：
1. 天气查询工具
2. 计算器工具
3. 知识库搜索工具

### 解答

#### 工具1：天气查询

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """
    查询指定城市的当前天气信息。

    输入：城市名称（如：北京、上海、New York）
    输出：天气状况、温度、湿度等信息

    示例：
    输入：北京
    输出：北京当前天气：晴，温度15°C，湿度45%，东北风3级
    """
    import requests

    # 调用天气API（示例）
    api_key = "your_api_key"
    url = f"https://api.weather.com/v1/current?location={location}&key={api_key}"

    try:
        response = requests.get(url)
        data = response.json()
        return f"{location}当前天气：{data['condition']}，温度{data['temp']}°C，湿度{data['humidity']}%"
    except Exception as e:
        return f"获取天气失败：{str(e)}"
```

#### 工具2：计算器

```python
@tool
def calculator(expression: str) -> str:
    """
    执行数学计算表达式。

    输入：数学表达式字符串
    支持：加减乘除、幂运算、开方、三角函数等

    示例：
    输入：2 + 3 * 4
    输出：14

    输入：sqrt(16) + 2**3
    输出：12.0
    """
    import math

    # 安全的数学函数
    safe_dict = {
        'abs': abs, 'round': round,
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'sqrt': math.sqrt, 'log': math.log, 'exp': math.exp,
        'pi': math.pi, 'e': math.e
    }

    try:
        # 只允许数学运算，防止代码注入
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return str(result)
    except Exception as e:
        return f"计算错误：{str(e)}"
```

#### 工具3：知识库搜索

```python
@tool
def search_knowledge_base(query: str) -> str:
    """
    在公司知识库中搜索相关信息。

    输入：搜索关键词或问题
    输出：最相关的知识条目（最多3条）

    适用场景：
    - 产品信息查询
    - 政策规定查询
    - 操作流程查询

    示例：
    输入：退款政策
    输出：
    1. [退款流程] 用户可在购买后7天内申请退款...
    2. [退款条件] 商品未使用且包装完好...
    3. [退款时间] 退款将在3-5个工作日内到账...
    """
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    # 加载向量数据库
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("knowledge_base", embeddings)

    # 相似度搜索
    docs = vectorstore.similarity_search(query, k=3)

    results = []
    for i, doc in enumerate(docs, 1):
        results.append(f"{i}. [{doc.metadata.get('title', '未知')}] {doc.page_content[:100]}...")

    return "\n".join(results) if results else "未找到相关信息"
```

#### 工具注册与使用

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# 创建 Agent
llm = ChatOpenAI(model="gpt-4")
tools = [get_weather, calculator, search_knowledge_base]

agent = create_react_agent(llm, tools)

# 使用示例
response = agent.invoke({
    "messages": [("user", "北京今天天气怎么样？适合户外运动吗？")]
})

print(response["messages"][-1].content)
```

---

## 习题七：性能优化策略

### 题目
针对三种范式，分别提出3种降低成本或提高性能的策略。

### 解答

#### ReAct 优化策略

**1. 减少调用次数**
```python
# 策略：设置合理的最大步数
max_steps = 5  # 大多数任务5步内能完成
early_stop_threshold = 0.9  # 置信度超过90%就停止
```

**2. 使用缓存**
```python
# 策略：缓存工具调用结果
from functools import lru_cache

@lru_cache(maxsize=100)
def search_with_cache(query):
    return search(query)
```

**3. 工具结果压缩**
```python
# 策略：只保留关键信息
def compress_observation(observation, max_length=500):
    if len(observation) > max_length:
        return observation[:max_length] + "...[已截断]"
    return observation
```

#### Plan-and-Solve 优化策略

**1. 计划复用**
```python
# 策略：相似问题复用已有计划
plan_cache = {
    "数学应用题": ["理解题意", "提取变量", "建立方程", "求解"],
    "写作任务": ["确定主题", "列大纲", "分段写作", "检查润色"]
}
```

**2. 并行执行独立步骤**
```python
# 策略：无依赖的步骤并行执行
import asyncio

async def execute_parallel(independent_steps):
    tasks = [execute_step(step) for step in independent_steps]
    return await asyncio.gather(*tasks)
```

**3. 使用小模型做规划**
```python
# 策略：规划用小模型，执行用大模型
planning_llm = ChatOpenAI(model="gpt-3.5-turbo")  # 便宜
solving_llm = ChatOpenAI(model="gpt-4")  # 强大
```

#### Reflection 优化策略

**1. 早停机制**
```python
# 策略：设置质量阈值，达到就停止
def should_stop(feedback, iteration):
    quality_score = extract_score(feedback)
    return quality_score >= 0.85 or iteration >= 3
```

**2. 选择性反思**
```python
# 策略：只对复杂任务使用 Reflection
def need_reflection(task):
    complexity = estimate_complexity(task)
    return complexity > 0.7  # 只有复杂任务才反思
```

**3. 增量反思**
```python
# 策略：只反思修改的部分，而不是整体
def incremental_reflect(original, modified):
    diff = compute_diff(original, modified)
    return reflect_on_diff(diff)  # 只评审变化的部分
```

#### 通用优化策略

| 优化维度 | 策略 | 效果 |
|---------|------|------|
| **成本** | 使用更小的模型（如 gpt-3.5） | 成本降低 10-20x |
| **速度** | 流式输出 + 并行调用 | 响应时间降低 50% |
| **质量** | 更好的 Prompt 设计 | 减少重试次数 |
| **稳定性** | 重试机制 + 降级方案 | 提高成功率 |

---

## 习题八：综合实践

### 题目
设计一个"智能论文助手"，综合使用三种范式：
1. Plan-and-Solve：规划论文大纲
2. ReAct：搜索参考文献
3. Reflection：优化每个章节

### 解答

#### 系统架构

```
用户输入论文主题
        ↓
   ┌─────────────────┐
   │ Plan-and-Solve  │  → 生成论文大纲
   └────────┬────────┘
            ↓
    对每个章节循环：
   ┌─────────────────┐
   │     ReAct       │  → 搜索参考文献
   └────────┬────────┘
            ↓
   ┌─────────────────┐
   │   Reflection    │  → 撰写并优化内容
   └────────┬────────┘
            ↓
    输出完整论文
```

#### 代码实现

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool

# 初始化
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# 工具定义
@tool
def search_papers(query: str) -> str:
    """搜索学术论文，返回相关论文列表"""
    # 调用学术搜索API
    pass

@tool
def get_paper_abstract(paper_id: str) -> str:
    """获取论文摘要"""
    pass

# 阶段1：Plan-and-Solve - 生成大纲
def generate_outline(topic: str) -> list:
    prompt = f"""
    为以下论文主题生成大纲：
    主题：{topic}

    输出JSON格式：
    {{"sections": [
        {{"title": "章节标题", "description": "章节描述", "subsections": [...]}},
        ...
    ]}}
    """
    response = llm.invoke(prompt)
    return parse_outline(response)

# 阶段2：ReAct - 搜索参考文献
def search_references(section_title: str) -> list:
    tools = [search_papers, get_paper_abstract]
    agent = create_react_agent(llm, tools)

    response = agent.invoke({
        "messages": [(
            "user",
            f"为论文章节'{section_title}'搜索3-5篇相关参考文献，返回标题和摘要"
        )]
    })
    return extract_references(response)

# 阶段3：Reflection - 撰写并优化章节
def write_section_with_reflection(section_info: dict, references: list) -> str:
    # 生成初稿
    draft_prompt = f"""
    撰写论文章节：
    标题：{section_info['title']}
    描述：{section_info['description']}
    参考文献：{references}

    要求：学术风格，引用参考文献
    """
    draft = llm.invoke(draft_prompt)

    # 反思循环
    for i in range(3):
        # 反思
        reflect_prompt = f"""
        评审以下论文章节：
        {draft}

        从以下方面评审：
        1. 逻辑性和连贯性
        2. 学术规范性
        3. 论证充分性
        4. 语言表达

        如果质量满意，回复"APPROVED"，否则给出具体改进建议。
        """
        feedback = llm.invoke(reflect_prompt)

        if "APPROVED" in feedback:
            break

        # 优化
        improve_prompt = f"""
        根据以下反馈改进论文章节：
        原文：{draft}
        反馈：{feedback}
        """
        draft = llm.invoke(improve_prompt)

    return draft

# 主流程
def generate_paper(topic: str) -> str:
    print(f"开始生成论文：{topic}")

    # 1. 生成大纲
    print("阶段1：生成大纲...")
    outline = generate_outline(topic)

    # 2. 处理每个章节
    sections = []
    for section in outline:
        print(f"处理章节：{section['title']}")

        # 搜索参考文献
        print("  - 搜索参考文献...")
        references = search_references(section['title'])

        # 撰写并优化
        print("  - 撰写并优化内容...")
        content = write_section_with_reflection(section, references)

        sections.append({
            'title': section['title'],
            'content': content,
            'references': references
        })

    # 3. 组装论文
    paper = assemble_paper(topic, sections)
    print("论文生成完成！")

    return paper

# 使用
paper = generate_paper("大语言模型在智能客服中的应用研究")
```

#### 流程示例

```
用户输入：大语言模型在智能客服中的应用研究

阶段1 - Plan-and-Solve 生成大纲：
├── 1. 引言
│   ├── 1.1 研究背景
│   └── 1.2 研究意义
├── 2. 相关工作
│   ├── 2.1 大语言模型发展
│   └── 2.2 智能客服现状
├── 3. 方法设计
│   ├── 3.1 系统架构
│   └── 3.2 关键技术
├── 4. 实验与分析
└── 5. 结论

阶段2 - ReAct 搜索参考文献（以第2章为例）：
Thought: 需要搜索大语言模型相关论文
Action: search_papers["LLM customer service"]
Observation: 找到5篇相关论文...
Thought: 需要获取摘要
Action: get_paper_abstract["paper_id_1"]
...

阶段3 - Reflection 撰写章节（以第3章为例）：
第1轮：生成初稿 → 评审发现论证不充分
第2轮：补充实验数据 → 评审发现表达不清晰
第3轮：优化语言表达 → APPROVED

输出：完整论文
```

---

## 总结

### 核心要点回顾

1. **ReAct**：思考-行动-观察循环，适合需要工具的动态任务
2. **Plan-and-Solve**：先规划后执行，适合可分解的结构化任务
3. **Reflection**：生成-反思-优化循环，适合高质量要求的任务

### 选择建议

- 需要外部信息 → ReAct
- 可以分步规划 → Plan-and-Solve
- 追求极致质量 → Reflection
- 复杂任务 → 组合使用

### 实践建议

1. 从简单开始，逐步增加复杂度
2. 注意设置超时和最大步数防止死循环
3. 使用缓存和并行优化性能
4. 根据任务特点选择合适的范式组合

---

**最后更新**: 2025-11-21
**版本**: v1.0
