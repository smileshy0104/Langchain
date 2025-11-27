# 习题 3: Agent 实现对比

## 题目

对比不同 Agent 的适用场景

---

## 🤖 四种 Agent 架构对比

### 概览表格

| Agent 类型 | 核心特点 | 执行模式 | 适用场景 | 性能开销 |
|-----------|---------|---------|---------|---------|
| **SimpleAgent** | 直接对话 | 单次调用 | 简单问答 | ⭐ 低 |
| **ReActAgent** | 推理-行动循环 | 多步迭代 | 工具辅助任务 | ⭐⭐⭐ 中 |
| **ReflectionAgent** | 自我反思 | 生成-评估-改进 | 高质量输出 | ⭐⭐⭐⭐ 高 |
| **PlanAndSolveAgent** | 先计划后执行 | 规划-分解-执行 | 复杂多步骤 | ⭐⭐⭐⭐⭐ 很高 |

---

## 1. SimpleAgent - 简单对话 Agent

### 🎯 核心特点

```python
class SimpleAgent:
    """最基础的 Agent,直接调用 LLM"""

    def run(self, input_text: str) -> str:
        # 单次调用,直接返回
        return self.llm.invoke([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text}
        ])
```

### 💡 工作流程

```
用户输入 → LLM → 响应
```

**特点**：
- ✅ 最简单,最快速
- ✅ 成本最低(单次调用)
- ❌ 无法使用工具
- ❌ 无推理能力

### 📊 适用场景

#### ✅ 适合的场景

1. **简单问答**
```python
# 场景1: 知识查询
agent.run("什么是 Python?")

# 场景2: 文本生成
agent.run("写一首关于春天的诗")

# 场景3: 翻译
agent.run("把'Hello World'翻译成中文")
```

2. **无需工具的任务**
```python
# 文本摘要
agent.run("总结这篇文章的要点: ...")

# 情感分析
agent.run("分析这段评论的情感: ...")
```

#### ❌ 不适合的场景

1. **需要计算**
```python
# ❌ LLM 可能算错
agent.run("计算 123456 * 789012")
# 应该使用 ReActAgent + Calculator 工具
```

2. **需要实时信息**
```python
# ❌ LLM 没有最新数据
agent.run("今天北京的天气如何?")
# 应该使用 ReActAgent + Weather API
```

3. **多步推理**
```python
# ❌ 可能出错
agent.run("先搜索Python的信息,然后生成学习计划")
# 应该使用 ReActAgent 或 PlanAndSolveAgent
```

### 📈 性能指标

```python
# 性能测试
平均响应时间: 1-2秒
Token 消耗: 100-500 tokens
成功率: 95% (简单任务)
```

---

## 2. ReActAgent - 推理行动 Agent

### 🎯 核心特点

```python
class ReActAgent:
    """
    ReAct = Reasoning + Acting
    通过 Thought-Action-Observation 循环解决问题
    """

    def run(self, input_text: str) -> str:
        for step in range(self.max_iterations):
            # Thought: 推理下一步做什么
            thought = self._generate_thought()

            # Action: 选择并执行工具
            action, tool_input = self._parse_action(thought)

            if action == "Finish":
                return tool_input  # 完成

            # Observation: 观察工具执行结果
            observation = self._execute_tool(action, tool_input)

            # 更新上下文,继续循环
```

### 💡 工作流程

```
问题 → Thought → Action → Observation → Thought → ... → 答案
      ↑______________|__________________|
              循环反馈
```

**示例对话**：
```
User: 计算 (25 * 4) + 10,然后搜索这个数字的含义

Thought 1: 我需要先计算 25 * 4
Action 1: calculator[25 * 4]
Observation 1: 100

Thought 2: 现在加上 10
Action 2: calculator[100 + 10]
Observation 2: 110

Thought 3: 搜索 110 的含义
Action 3: search[110 的含义]
Observation 3: 110 是紧急求助电话号码...

Thought 4: 我已经得到了所有信息
Action 4: Finish[计算结果是 110,这是紧急求助电话号码]
```

### 📊 适用场景

#### ✅ 适合的场景

1. **需要工具辅助的任务**
```python
# 场景1: 数学计算
agent.run("计算复利: 本金10000,年利率5%,10年后是多少?")
# → 使用 calculator 工具

# 场景2: 信息检索
agent.run("搜索 LangChain 的最新版本并总结新特性")
# → 使用 search 工具

# 场景3: 数据查询
agent.run("查询数据库中销售额最高的产品")
# → 使用 database 工具
```

2. **多步骤推理**
```python
# 先搜索,再计算,再总结
agent.run("""
1. 搜索比特币当前价格
2. 计算购买 0.5 个需要多少钱
3. 总结投资风险
""")
```

3. **动态工具选择**
```python
# Agent 自己决定使用哪个工具
agent.run("帮我规划从上海到北京的出行方案")
# → 可能使用: weather, map, search 等多个工具
```

#### ❌ 不适合的场景

1. **简单对话**
```python
# ❌ 杀鸡用牛刀
agent.run("你好")
# 应该使用 SimpleAgent
```

2. **需要深度思考的任务**
```python
# ❌ ReAct 每步都要调用 LLM,成本高
agent.run("写一篇 3000 字的论文")
# 应该使用 ReflectionAgent 或 PlanAndSolveAgent
```

### 📈 性能指标

```python
# 性能测试
平均步数: 3-5步
平均响应时间: 5-10秒
Token 消耗: 500-2000 tokens
成功率: 80% (复杂任务)
```

---

## 3. ReflectionAgent - 反思 Agent

### 🎯 核心特点

```python
class ReflectionAgent:
    """
    通过自我反思和迭代改进,生成高质量输出
    """

    def run(self, input_text: str) -> str:
        current_output = self._initial_generate(input_text)

        for iteration in range(self.max_reflections):
            # 评估当前输出
            critique = self._reflect(current_output)

            # 如果足够好,停止
            if self._is_good_enough(critique):
                break

            # 基于反思改进输出
            current_output = self._improve(current_output, critique)

        return current_output
```

### 💡 工作流程

```
输入 → 初始生成 → 自我评估 → 改进 → 自我评估 → ... → 最终输出
               ↑_____________|___________|
                      反思循环
```

**示例对话**：
```
User: 写一个 Python 函数计算斐波那契数列

[迭代 1] 生成:
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

[反思 1]: 这个实现正确但效率低,有重复计算

[迭代 2] 改进:
def fib(n):
    cache = {}
    def helper(n):
        if n in cache:
            return cache[n]
        if n <= 1:
            return n
        cache[n] = helper(n-1) + helper(n-2)
        return cache[n]
    return helper(n)

[反思 2]: 很好,但可以更简洁

[迭代 3] 最终版本:
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

[反思 3]: 完美!时间复杂度 O(n),空间复杂度 O(1)
```

### 📊 适用场景

#### ✅ 适合的场景

1. **需要高质量输出的任务**
```python
# 场景1: 代码生成
agent.run("写一个高性能的排序算法")

# 场景2: 文章写作
agent.run("写一篇关于AI伦理的深度文章")

# 场景3: 方案设计
agent.run("设计一个分布式缓存系统")
```

2. **需要多次改进的任务**
```python
# 简历优化
agent.run("优化这份简历: ...")

# 代码重构
agent.run("重构这段代码,提高可读性和性能: ...")
```

3. **创意任务**
```python
# 广告文案
agent.run("为新产品写一个吸引人的广告文案")

# 故事创作
agent.run("写一个科幻短篇小说")
```

#### ❌ 不适合的场景

1. **时效性要求高的任务**
```python
# ❌ 反思需要多次 LLM 调用,太慢
agent.run("快速回答: 1+1=?")
```

2. **简单明确的任务**
```python
# ❌ 浪费资源
agent.run("翻译: Hello")
# 应该使用 SimpleAgent
```

### 📈 性能指标

```python
# 性能测试
平均反思次数: 2-3次
平均响应时间: 10-20秒
Token 消耗: 1000-5000 tokens
输出质量: ⭐⭐⭐⭐⭐
```

---

## 4. PlanAndSolveAgent - 计划执行 Agent

### 🎯 核心特点

```python
class PlanAndSolveAgent:
    """
    先制定详细计划,再逐步执行
    """

    def run(self, input_text: str) -> str:
        # 阶段1: 制定计划
        plan = self._create_plan(input_text)
        # 例如: ["步骤1: 搜索信息", "步骤2: 分析数据", "步骤3: 生成报告"]

        # 阶段2: 执行计划
        results = []
        for step in plan:
            result = self._execute_step(step, results)
            results.append(result)

        # 阶段3: 总结结果
        return self._summarize(results)
```

### 💡 工作流程

```
问题 → 规划阶段 → 执行阶段 → 总结阶段
        ↓            ↓           ↓
     [计划列表]   [执行每步]   [最终答案]
```

**示例对话**：
```
User: 分析2024年AI领域的发展趋势

[规划阶段]
计划:
1. 搜索2024年AI重大事件
2. 搜索各大公司AI产品发布
3. 搜索学术界突破
4. 分析共同趋势
5. 预测未来方向

[执行阶段]
执行步骤1: search[2024年AI重大事件]
→ 结果: GPT-4.5发布,Claude 3推出...

执行步骤2: search[2024年AI产品]
→ 结果: Sora视频生成,Gemini Ultra...

执行步骤3: search[2024年AI学术突破]
→ 结果: 多模态大模型,强化学习新进展...

执行步骤4: 分析趋势
→ 多模态、个性化、开源化

执行步骤5: 预测
→ 2025年将更注重AI安全和可解释性

[总结阶段]
综合报告: 2024年AI发展呈现三大趋势...
```

### 📊 适用场景

#### ✅ 适合的场景

1. **复杂多步骤任务**
```python
# 场景1: 数据分析
agent.run("""
分析这个数据集:
1. 数据清洗
2. 探索性分析
3. 特征工程
4. 模型训练
5. 结果解释
""")

# 场景2: 项目规划
agent.run("为开发一个电商网站制定详细计划")

# 场景3: 研究报告
agent.run("研究量子计算的最新进展并写报告")
```

2. **需要明确步骤的任务**
```python
# 烹饪食谱
agent.run("详细讲解如何做红烧肉")

# 维修指南
agent.run("电脑无法开机的诊断和修复步骤")
```

3. **教学任务**
```python
# 知识讲解
agent.run("教我学习机器学习,从零开始")

# 技能培训
agent.run("如何成为一名优秀的前端工程师?")
```

#### ❌ 不适合的场景

1. **需要灵活应变的任务**
```python
# ❌ 计划可能不适应动态变化
agent.run("和用户进行自然对话")
# 应该使用 SimpleAgent 或 ReActAgent
```

2. **简单任务**
```python
# ❌ 杀鸡用牛刀
agent.run("2+2等于几?")
```

### 📈 性能指标

```python
# 性能测试
平均步骤数: 5-10步
平均响应时间: 15-30秒
Token 消耗: 2000-10000 tokens
任务完成度: 95% (结构化任务)
```

---

## 🎯 选择决策树

```
                    开始
                     |
              是否需要工具?
               /         \
             否            是
             |             |
       是否需要      是否有明确步骤?
       高质量?        /         \
        /   \       是           否
       否    是      |            |
       |     |      |     是否需要多次改进?
  SimpleAgent  |    |        /         \
       |  ReflectionAgent   否          是
       |       |    |        |           |
       |       | PlanAndSolve  |     Reflection
       |       |    Agent   ReActAgent   Agent
       └───────┴────────┴──────┴──────────┘
```

---

## 📊 综合对比表

### 性能对比

| Agent | 响应时间 | Token消耗 | 成本 | 准确率 | 复杂度 |
|-------|---------|----------|------|--------|--------|
| Simple | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $ | 85% | 低 |
| ReAct | ⭐⭐⭐ | ⭐⭐⭐ | $$ | 80% | 中 |
| Reflection | ⭐⭐ | ⭐⭐ | $$$ | 95% | 高 |
| PlanAndSolve | ⭐ | ⭐ | $$$$ | 90% | 很高 |

### 场景适配度

| 场景类型 | Simple | ReAct | Reflection | PlanAndSolve |
|---------|--------|-------|------------|--------------|
| 简单问答 | ✅✅✅ | ❌ | ❌ | ❌ |
| 工具调用 | ❌ | ✅✅✅ | ✅ | ✅✅ |
| 代码生成 | ✅ | ✅ | ✅✅✅ | ✅✅ |
| 数据分析 | ❌ | ✅✅ | ✅ | ✅✅✅ |
| 创意写作 | ✅✅ | ✅ | ✅✅✅ | ✅✅ |
| 规划任务 | ❌ | ✅ | ✅ | ✅✅✅ |
| 实时对话 | ✅✅✅ | ✅✅ | ❌ | ❌ |

---

## 💼 实际案例

### 案例 1: 客服聊天机器人

**需求**：回答用户常见问题

**选择**：**SimpleAgent**

**理由**：
- ✅ 大部分是简单问答
- ✅ 需要快速响应
- ✅ 成本要低
- ❌ 不需要复杂推理

---

### 案例 2: 数据分析助手

**需求**：分析销售数据,生成报表

**选择**：**ReActAgent** 或 **PlanAndSolveAgent**

**理由**：
- ✅ 需要查询数据库
- ✅ 需要计算统计指标
- ✅ 多步骤任务
- 如果步骤固定 → PlanAndSolve
- 如果需要灵活 → ReAct

---

### 案例 3: 代码审查助手

**需求**：审查代码,提出改进建议

**选择**：**ReflectionAgent**

**理由**：
- ✅ 需要多角度评估
- ✅ 需要高质量输出
- ✅ 可以多次迭代改进
- ❌ 不追求极致速度

---

### 案例 4: 学习助手

**需求**：制定学习计划,推荐资源

**选择**：**PlanAndSolveAgent**

**理由**：
- ✅ 典型的规划任务
- ✅ 步骤清晰
- ✅ 需要结构化输出

---

## 🎓 总结

### 核心原则

1. **简单优先**：能用 SimpleAgent 就不用复杂的
2. **需求驱动**：根据任务特点选择
3. **成本考虑**：Token 消耗和响应时间
4. **迭代优化**：先用简单的,不够再升级

### 选择建议

```python
# 1. 快速对话 → Simple
if task == "简单问答" and not need_tools:
    return SimpleAgent

# 2. 工具调用 → ReAct
elif need_tools and not need_perfect:
    return ReActAgent

# 3. 高质量 → Reflection
elif need_high_quality and not time_sensitive:
    return ReflectionAgent

# 4. 复杂规划 → PlanAndSolve
elif complex_steps and structured:
    return PlanAndSolveAgent
```

---

## 🔬 扩展思考

1. **混合模式**：能否结合多种 Agent 的优点?
2. **自动选择**：如何让系统自动选择最合适的 Agent?
3. **性能优化**：如何降低复杂 Agent 的成本?
4. **新的模式**：除了这四种,还有哪些可能的 Agent 架构?

---

## 📚 参考资料

- [ReAct Paper](https://arxiv.org/abs/2210.03629) - ReAct 论文
- [Reflexion Paper](https://arxiv.org/abs/2303.11366) - Reflection 论文
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/) - LangChain 官方文档
- [Agent Patterns](https://eugeneyan.com/writing/llm-patterns/) - LLM 设计模式
