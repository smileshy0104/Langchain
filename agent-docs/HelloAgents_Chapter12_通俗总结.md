# Hello Agents 第十二章：智能体性能评估（通俗总结）

> **本章核心思想**：在工程界有句名言："无法度量，就无法改进"。之前我们造出了能跑能跳的 Agent，现在我们要给它做全身体检，用科学的"体检报告"告诉世界它到底有多强。

---

## 📖 目录

- [1. 为什么要给 Agent 做体检？](#1-为什么要给-agent-做体检)
- [2. 三大体检套餐：BFCL、GAIA、数据生成](#2-三大体检套餐bfclgaia数据生成)
- [3. BFCL：手脚灵活度测试（工具调用）](#3-bfcl手脚灵活度测试工具调用)
- [4. GAIA：综合素质测试（通用能力）](#4-gaia综合素质测试通用能力)
- [5. 数��生成评估：创造力测试](#5-数据生成评估创造力测试)
- [6. 本章总结](#6-本章总结)

---

## 1. 为什么要给 Agent 做体检？

### 🤔 "感觉不错" vs "数据说话"

之前我们测试 Agent 都是："嗯，这个回答看着挺好的"。这种主观感觉在工程中是不靠谱的。
*   **问题 1**：换了个模型，是变强了还是变弱了？
*   **问题 2**：它在简单任务上表现好，复杂任务呢？
*   **问题 3**：怎么向老板/用户证明它比竞品强？

### 💡 评估基准 (Benchmark)

我们需要一套**标准试卷**（数据集）和**评分标准**（Metric），让 Agent 在相同条件下考试。这就是 Benchmark。

---

## 2. 三大体检套餐：BFCL、GAIA、数据生成

本章我们为 Agent 安排了三场不同维度的考试：

| 评估项目 | 核心能力 | 形象比喻 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **BFCL** | **工具调用** | **实操考试**<br>给你个扳手，看你会不会用，螺丝拧得紧不紧。 | API 调用、函数执行 |
| **GAIA** | **通用解决问题** | **综合考试**<br>给你个复杂任务，看你能不能查资料、动脑子搞定。 | 真实世界助手、复杂推理 |
| **数据生成** | **生成质量** | **创作考试**<br>让你出题，看你���的题水平高不高。 | 训练数据合成、内容创作 |

---

## 3. BFCL：手脚灵活度测试（工具调用）

**BFCL (Berkeley Function Calling Leaderboard)** 是专门测 Agent 调用函数能力的。

### 📝 考题类型
*   **Simple**：单一函数调用（如：查北京天气）。
*   **Multiple**：多个函数（如：查天气 + 查航班）。
*   **Parallel**：并行调用（如：同时查北京和上海天气）。
*   **Irrelevance**：干扰项（如：问"你好"，不应该调用任何工具）。

### 🔍 评分标准：AST 匹配
不是简单的字符串对比，而是**抽象语法树 (AST)** 匹配。
*   `get_weather(city="Beijing")` 和 `get_weather(city='Beijing')` 算对。
*   参数顺序换了也算对。
*   这比死板的文本匹配更科学。

### 💻 一键评估

```python
from hello_agents.tools import BFCLEvaluationTool

# 给 Agent 发卷子
bfcl_tool = BFCLEvaluationTool()
results = bfcl_tool.run(
    agent=agent, 
    category="simple_python", 
    max_samples=5
)

print(f"准确率: {results['overall_accuracy']:.2%}")
```

---

## 4. GAIA：综合素质测试（通用能力）

**GAIA (General AI Assistants)** 是测 Agent 在真实世界里解决复杂问题能力的。

### 📝 考题特点
*   **真题**：不是造出来的假数据，是人类真实会遇到的问题。
*   **多模态**：可能有图片、PDF、Excel 表格。
*   **难度分级**：Level 1（简单） -> Level 3（地狱级）。

### 🔍 评分标准：准精确匹配
答案必须非常精准。比如问人口数，答案是 `12,847,521`。
*   归一化：去掉逗号、单位、大小写差异。
*   比如：`$100` 和 `100` 算对，`The Apple` 和 `apple` 算对。

### 💻 一键评估

```python
from hello_agents.tools import GAIAEvaluationTool

# GAIA 需要特殊的 System Prompt
agent = SimpleAgent(..., system_prompt=GAIA_SYSTEM_PROMPT)

gaia_tool = GAIAEvaluationTool()
results = gaia_tool.run(agent=agent, level=1)

print(f"精确匹配率: {results['exact_match_rate']:.2%}")
```

---

## 5. 数据生成评估：创造力测试

除了做题，Agent 还能出题。我们以**生成数学竞赛题 (AIME)** 为例，评估 Agent 的创造力。

### 🤖 三位评委

1.  **LLM Judge**：请 GPT-4 当老师，给生成的题目打分。
    *   维度：正确性、清晰度、难度、完整性。
    *   **Average Score**：平均分高不高？
    *   **Pass Rate**：及格率多少？

2.  **Win Rate**：竞技场模式。
    *   拿生成的题目和真题 PK，让裁判盲测哪个更好。
    *   目标：Win Rate 接近 50%（说明达到真题水平）。

3.  **Human Verification**：人工审核。
    *   最后还得人来看一眼，确保没出大 bug。
    *   我们提供了一个 Web 界面，方便人工打分。

---

## 6. 本章总结

### 🌟 核心收获

1.  **评估是工程化的基石**：有了评估，优化才有方向。
2.  **工具化**：我们将复杂的评估流程封装成了 `BFCLEvaluationTool` 和 `GAIAEvaluationTool`，一键跑分。
3.  **多维视角**：从工具调用准确率、通用任务解决率到生成内容质量，全方位无死角体检。

### 🚀 下一步是什么？

到现在为止，我们已经掌握了 Agent 的构建、增强、训练和评估。
接下来的章节将进入**综合案例进阶**部分，我们将把前面学到的所有武功（Memory, RAG, Protocol, RL, Eval）融会贯通，去解决真实世界的复杂大问题！

下一章（第十三章），我们将打造一个**智能旅行助手**，看看多智能体是如何协作规划一次完美旅行的！

---

### 🔗 快速传送门
- **GitHub 源码**: [hello-agents/chapter12](https://github.com/jjyaoao/helloagents)
- **安装命令**: `pip install "hello-agents[evaluation]"`
