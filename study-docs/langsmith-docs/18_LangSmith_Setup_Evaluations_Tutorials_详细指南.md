# LangSmith Set up Evaluations Tutorials（评估搭建教程）详细指南

> 基于官方文档 Evaluate a chatbot、Evaluate a RAG application、Test a ReAct agent with Pytest/Vitest and LangSmith、Evaluate a complex agent、Run backtests on a new version of an agent 整理的中文增强版。本文聚焦如何从零搭建评估：创建数据集、定义 evaluator、运行实验、比较结果、接入测试框架和 CI/CD，以及用生产 traces 做 backtesting。

## 目录

1. [概述](#概述)
2. [通用评估搭建流程](#通用评估搭建流程)
3. [Chatbot 评估教程](#chatbot-评估教程)
4. [RAG 应用评估教程](#rag-应用评估教程)
5. [ReAct Agent 测试教程](#react-agent-测试教程)
6. [复杂 Agent 评估教程](#复杂-agent-评估教程)
7. [Backtesting 新版本 Agent](#backtesting-新版本-agent)
8. [Evaluator 设计模式](#evaluator-设计模式)
9. [CI/CD 与长期追踪](#cicd-与长期追踪)
10. [实践建议](#实践建议)
11. [快速参考](#快速参考)

---

## 概述

这一组教程展示了 LangSmith Evaluation 的完整实践路径：从一个简单 QA chatbot，到 RAG、ReAct Agent、复杂多路径 Agent，再到使用生产历史数据做 backtesting。

整体脉络可以概括为：

```
┌─────────────────────────────────────────────────────────────┐
│              LangSmith Evaluation Tutorials                  │
│                                                             │
│  1. 构建或选定应用                                           │
│     ├─ Chatbot                                               │
│     ├─ RAG                                                   │
│     ├─ ReAct Agent                                           │
│     └─ Complex Agent                                         │
│                                                             │
│  2. 准备 Dataset                                             │
│     ├─ 手工 golden examples                                  │
│     ├─ reference answers                                     │
│     ├─ expected tool calls / trajectories                    │
│     └─ production traces                                     │
│                                                             │
│  3. 定义 Evaluators                                          │
│     ├─ Code / heuristic                                      │
│     ├─ LLM-as-judge                                          │
│     ├─ trajectory evaluator                                  │
│     └─ testing assertions                                    │
│                                                             │
│  4. 运行与比较 Experiments                                   │
│     ├─ evaluate / aevaluate                                  │
│     ├─ Pytest / Vitest / Jest                                │
│     ├─ comparison view                                       │
│     └─ CI/CD gates                                           │
└─────────────────────────────────────────────────────────────┘
```

### 这些教程分别解决什么

| 教程 | 重点 | 适合场景 |
|------|------|----------|
| Evaluate a chatbot | 从零创建 dataset、metrics、experiment、CI/CD | 最小可行评估体系 |
| Evaluate a RAG application | 同时评估回答质量与检索质量 | RAG 问答、知识库应用 |
| Test a ReAct agent | 用 Pytest/Vitest/Jest 测 Agent 工具调用 | 工具调用 Agent 的自动测试 |
| Evaluate a complex agent | Final response、trajectory、single-step 三类 Agent 评估 | 多路径、多子图 Agent |
| Run backtests | 用生产 traces 比较新旧版本 | 已上线应用的回归和版本升级 |

---

## 通用评估搭建流程

虽然各教程场景不同，但基本流程高度一致。

### 1. 配置环境

常见环境变量：

```bash
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="<Your LangSmith API key>"
export OPENAI_API_KEY="<Your OpenAI API key>"
```

根据应用还可能需要：

| 变量 | 用途 |
|------|------|
| `TAVILY_API_KEY` | 搜索工具 |
| `E2B_API_KEY` | 代码解释器 |
| `POLYGON_API_KEY` | 股票数据 |
| `LANGSMITH_PROJECT` | 指定 tracing project |

### 2. 准备 Dataset

Dataset 是离线评估的核心。每条 example 至少要有 inputs；如果能定义 reference outputs，会更方便做 correctness 类评估。

```python
from langsmith import Client

client = Client()

dataset_name = "QA Example Dataset"
dataset = client.create_dataset(dataset_name)

client.create_examples(
    dataset_id=dataset.id,
    examples=[
        {
            "inputs": {"question": "What is LangChain?"},
            "outputs": {"answer": "A framework for building LLM applications"},
        }
    ],
)
```

数据集建议：

- 一开始不必追求很大，10 到 50 条高质量 examples 就很有价值。
- 先手工标注 10 到 20 条，确保覆盖关键场景和边界条件。
- 后续从真实用户 traces 中不断补充。
- 对 Agent 可额外存 expected tool calls、trajectory、route 等结构化 reference。

### 3. 定义 Target Function

`evaluate` 需要一个 target function，将 dataset example 的 `inputs` 映射到应用输出。

```python
def target(inputs: dict) -> dict:
    return {"response": my_app(inputs["question"])}
```

注意：

- target 输出 key 要和 evaluator 里读取的 key 对齐。
- RAG target 常返回 `answer` 和 `documents`。
- Agent target 可返回 `response`、`trajectory`、`route`、`tool_calls` 等。

### 4. 定义 Evaluators

Evaluator 可以是简单函数，也可以调用 LLM-as-judge。

```python
def concision(outputs: dict, reference_outputs: dict) -> bool:
    return len(outputs["response"]) < 2 * len(reference_outputs["answer"])
```

LLM-as-judge 常用于语义正确性、忠实度、相关性。

### 5. 运行 Evaluation

```python
experiment_results = client.evaluate(
    target,
    data=dataset_name,
    evaluators=[concision, correctness],
    experiment_prefix="my-experiment",
)
```

异步版本：

```python
experiment_results = await client.aevaluate(
    target,
    data=dataset_name,
    evaluators=[my_evaluator],
    experiment_prefix="async-experiment",
    max_concurrency=4,
)
```

### 6. 比较 Experiments

LangSmith UI 支持：

- 在 dataset 的 Experiments tab 查看不同运行的指标。
- 多选 experiments 打开 comparison view。
- 设置 baseline 和 metric，查看 improvement / regression。
- 展开单行查看每个 example 的详细输入、输出、反馈和 trace。

### 7. 接入 CI/CD

把 evaluation 包装成 Pytest/Vitest/Jest 测试，设置通过阈值。

```python
def test_length_score() -> None:
    results = client.evaluate(
        target,
        data=dataset_name,
        evaluators=[concision, correctness],
    )
    feedback = client.list_feedback(
        run_ids=[r.id for r in client.list_runs(project_name=results.experiment_name)],
        feedback_key="concision",
    )
    scores = [f.score for f in feedback]
    assert sum(scores) / len(scores) >= 0.8
```

---

## Chatbot 评估教程

### 教程目标

Chatbot 教程展示最小可行的 LangSmith 评估流程：

1. 创建 golden dataset。
2. 定义 correctness 和 concision 两个指标。
3. 对不同模型或 prompt 运行 experiments。
4. 在 UI 中比较结果。
5. 用测试框架接入 CI/CD。
6. 追踪评估结果随时间变化。

### Dataset 设计

示例应用是简单 QA chatbot，dataset schema 很直接：

| 字段 | 示例 |
|------|------|
| `inputs.question` | `What is LangChain?` |
| `outputs.answer` | `A framework for building LLM applications` |

官方强调三个问题：

| 问题 | 建议 |
|------|------|
| 每条 datapoint 的 schema 是什么 | 至少包含应用 inputs，有能力时加 expected outputs |
| 需要多少 datapoints | 没有硬性规则，先覆盖关键边界，10 到 50 条也有价值 |
| 如何收集 datapoints | 新项目先手工标注 10 到 20 条，后续从真实用户问题补充 |

### Evaluators

教程中定义了两个指标。

#### Correctness

Correctness 使用 LLM-as-judge，将模型回答和参考答案比较。

输入：

- 用户问题。
- reference answer。
- predicted answer。

输出：

- `CORRECT` 或 `INCORRECT`。

适合原因：

- 自然语言答案不需要逐字匹配。
- 可以容忍等价表达。
- 比简单字符串匹配更适合 QA。

#### Concision

Concision 是简单代码 evaluator，检查输出长度是否小于参考答案长度的 2 倍。

```python
def concision(outputs: dict, reference_outputs: dict) -> bool:
    return len(outputs["response"]) < 2 * len(reference_outputs["answer"])
```

这个例子说明：能用确定性规则评估的地方，优先用 code evaluator；需要语义判断时，再用 LLM-as-judge。

### Target Function

应用函数：

```python
def my_app(question: str, model: str, instructions: str) -> str:
    ...
```

LangSmith target wrapper：

```python
def ls_target(inputs: dict) -> dict:
    return {"response": my_app(inputs["question"])}
```

重点：dataset 输入字段和应用函数参数通常不完全一致，需要 wrapper 做映射。

### 多版本实验

教程比较了三个版本：

| 版本 | 变化 |
|------|------|
| v1 | 默认模型和默认简洁 prompt |
| v2 | 换成另一个模型 |
| v3 | 换模型并加强 prompt，限制不超过十个词 |

通过 `experiment_prefix` 给实验命名，方便后续比较。

### UI 比较

可在 dataset 的 Experiments tab 中：

- 看每个 experiment 的指标汇总。
- 选择多个 experiments side-by-side 比较。
- 用颜色查看相对 baseline 的 regression / improvement。
- 通过 Display 控制展示哪些列和指标。
- 展开某条 example 查看详细信息。

### CI/CD

将 evaluation 放进测试文件，用聚合指标做门禁。例如 concision 平均分必须大于 0.8。

### 长期追踪

LangSmith 会在 Experiments tab 中展示指标随时间变化，并自动关联 git 信息，方便定位某次代码变化带来的质量波动。

---

## RAG 应用评估教程

### 教程目标

RAG 教程展示如何同时评估：

- 答案是否正确。
- 答案是否回应问题。
- 答案是否基于检索文档。
- 检索文档是否和问题相关。

示例应用针对 Lilian Weng 的几篇博客做问答。

### RAG 应用结构

```
┌──────────────┐
│ Web documents│
└──────┬───────┘
       ▼
┌──────────────┐
│ Split chunks │
└──────┬───────┘
       ▼
┌──────────────┐
│ Vector store │
└──────┬───────┘
       ▼
┌──────────────┐
│  Retriever   │
└──────┬───────┘
       ▼
┌──────────────┐
│ LLM answer   │
└──────────────┘
```

教程使用：

- 文档加载：requests + BeautifulSoup。
- 分块：RecursiveCharacterTextSplitter。
- 向量库：InMemoryVectorStore / MemoryVectorStore。
- Embeddings：OpenAIEmbeddings。
- LLM：ChatOpenAI。
- tracing：`@traceable()`。

### Dataset

RAG dataset 包含问题和 reference answer。

| 字段 | 说明 |
|------|------|
| `inputs.question` | 用户问题 |
| `outputs.answer` | 参考答案 |

即便没有 reference answer，也可以做 relevance、groundedness、retrieval relevance 等 reference-free 评估。

### 四类 RAG Evaluator

#### 1. Correctness: Response vs Reference Answer

评估问题：

**生成答案相对于标准答案是否事实正确？**

输入：

- `inputs.question`
- `reference_outputs.answer`
- `outputs.answer`

特点：

- 需要 reference answer。
- 适合 offline evaluation。
- 使用 LLM-as-judge。
- 推荐结构化输出，如 `{explanation, correct}`。

#### 2. Relevance: Response vs Input

评估问题：

**回答是否切中了用户问题，是否有帮助？**

输入：

- `inputs.question`
- `outputs.answer`

特点：

- 不需要 reference answer。
- 可用于 offline 或 online。
- 不能判断“事实正确性”，只能判断“是否答题”。

#### 3. Groundedness: Response vs Retrieved Docs

评估问题：

**答案是否被检索文档支持，是否出现幻觉？**

输入：

- `outputs.answer`
- `outputs.documents`

特点：

- 不需要 reference answer。
- 是 RAG 的关键质量指标。
- 要求 target 返回 retrieved documents。

#### 4. Retrieval Relevance: Retrieved Docs vs Input

评估问题：

**检索出来的 documents 是否与问题相关？**

输入：

- `inputs.question`
- `outputs.documents`

特点：

- 不评估答案，只评估检索器。
- 可帮助定位问题来自 retrieval 还是 generation。

### RAG Target Function

target 返回答案和文档：

```python
def target(inputs: dict) -> dict:
    return rag_bot(inputs["question"])
```

`rag_bot` 返回：

```python
{
    "answer": ai_msg.content,
    "documents": docs,
}
```

### 运行 Evaluation

```python
experiment_results = client.evaluate(
    target,
    data=dataset_name,
    evaluators=[
        correctness,
        groundedness,
        relevance,
        retrieval_relevance,
    ],
    experiment_prefix="rag-doc-relevance",
    metadata={"version": "LCEL context, gpt-4-0125-preview"},
)
```

### RAG 诊断方式

| 现象 | 可能原因 | 看哪个指标 |
|------|----------|------------|
| 答案不正确 | 生成错或检索错 | correctness |
| 答非所问 | prompt 或生成逻辑问题 | relevance |
| 回答编造 | 模型没有忠于 context | groundedness |
| 文档不相关 | retriever / chunking / embedding 问题 | retrieval_relevance |

---

## ReAct Agent 测试教程

### 教程目标

该教程展示如何使用 LangSmith 与 Pytest / Vitest / Jest 集成，测试一个股票研究 ReAct Agent。

Agent 使用：

- LangGraph / LangGraph.js 做编排。
- OpenAI 模型。
- Tavily 搜索工具。
- E2B code interpreter。
- Polygon 股票数据工具。

### Agent 工具

| 工具 | 用途 |
|------|------|
| Search tool | 查询新闻、分析师观点等外部信息 |
| Code tool | 执行 Python 代码，处理计算 |
| Ticker / Polygon tool | 查询股票价格和聚合数据 |

### 测试框架集成

Python：

```bash
pip install -U "langsmith[pytest]"
pytest --langsmith-output tests
```

Pytest 中使用：

```python
import pytest
from langsmith import testing as t
```

TypeScript：

```bash
yarn add -D langsmith vitest
yarn add -D langsmith jest
```

Vitest/Jest 中使用：

```typescript
import * as ls from "langsmith/vitest";
// or
import * as ls from "langsmith/jest";
```

### 测试 1：离题问题不调用工具

目标：

**当用户只是打招呼或闲聊时，Agent 不应该调用工具。**

Pytest 思路：

- `t.log_inputs({"query": query})`
- `t.log_reference_outputs({"tool_calls": []})`
- 直接调用 agent model node，而不是完整 ReAct loop。
- 检查 `tool_calls == []`。

价值：

- 测试工具调用克制性。
- 避免 Agent 对无关输入浪费成本和产生副作用。

### 测试 2：简单工具调用

目标：

**用户问 Apple 股价时，Agent 应调用股票工具，并传入 `AAPL`。**

检查点：

- 是否调用 ticker / polygon 工具。
- ticker 参数是否正确。
- TypeScript 例子还检查消息轨迹长度：Human -> AI -> Tool -> AI Final。

价值：

- 单步工具选择和参数构造测试。
- 非常适合 deterministic assertion。

### 测试 3：复杂工具调用

目标：

**当问题需要计算平均收益率时，Agent 应使用 code tool，并给出正确数值。**

检查点：

- tool_calls 中包含 `code_tool`。
- structured_response 中有 numeric_answer。
- 数值答案在容差范围内。
- 记录 `num_steps` feedback，用于长期追踪 Agent 效率。

价值：

- 测试完整 Agent 行为。
- 对多路径问题，用“是否用到必要工具 + 最终答案正确”比精确匹配轨迹更稳。

### 测试 4：LLM-as-judge Groundedness

目标：

**确保 Agent 的回答被搜索结果支持。**

Python 中使用 `t.trace_feedback()`，让 judge LLM 的 run 和 agent run 分开追踪。

TypeScript 中使用 `ls.wrapEvaluator()` 包装 evaluator。

评估流程：

1. 运行 Agent。
2. 提取 search tool outputs。
3. 把 answer 和 documents 交给 judge LLM。
4. judge 输出 groundedness 分数。
5. 使用 `t.log_feedback` 或 wrapped evaluator 记录反馈。

### 测试配置

Vitest 配置要点：

- `include: ["**/*.eval.?(c|m)[jt]s"]`
- `reporters: ["langsmith/vitest/reporter"]`
- `setupFiles: ["dotenv/config"]`
- `testTimeout: 30000`

Jest 配置要点：

- `reporters: ["langsmith/jest/reporter"]`
- `testTimeout: 30000`
- 指定 `.jest.eval.ts` 测试文件。

### ReAct Agent 测试策略

| 测试目标 | 推荐方式 |
|----------|----------|
| 不该调用工具 | 直接检查 tool_calls 为空 |
| 应调用某个工具 | 检查 tool name |
| 参数必须正确 | 检查 tool args |
| 最终数值正确 | 检查 structured response + 容差 |
| 路径效率 | 记录 num_steps feedback |
| 回答忠实于工具输出 | LLM-as-judge groundedness |

---

## 复杂 Agent 评估教程

### 教程目标

复杂 Agent 教程构建了一个数字音乐商店客服机器人，支持两类请求：

| 请求类型 | 示例 |
|----------|------|
| Lookup | 查询歌曲、歌手、专辑 |
| Refund | 帮用户查找购买记录并退款 |

该 Agent 使用 LangGraph 构建：

- Refund subgraph。
- QA / lookup ReAct subgraph。
- Parent router graph。

然后用三类 Agent evaluation 进行评估：

- Final response。
- Trajectory。
- Single step。

### 应用架构

```
┌───────────────────┐
│ intent_classifier │
└───────┬───────────┘
        │
        ├──────────────▶ refund_agent
        │                  ├─ gather_info
        │                  ├─ lookup
        │                  └─ refund
        │
        └──────────────▶ question_answering_agent
                           ├─ lookup_track
                           ├─ lookup_artist
                           └─ lookup_album
```

### Refund Agent

Refund agent 负责：

1. 从对话中提取用户和购买信息。
2. 判断信息是否足够。
3. 如果有 invoice ID 或 invoice line IDs，执行退款。
4. 如果有姓名和电话，查询购买记录。
5. 如果信息不足，要求用户补充。

为了评估安全，refund 函数支持 mock 模式：

```python
mock = config.get("configurable", {}).get("env", "prod") == "test"
```

评估时传入 `config={"env": "test"}`，避免真的修改数据库。

### QA Agent

QA agent 使用 ReAct 架构，提供三个工具：

| 工具 | 用途 |
|------|------|
| `lookup_track` | 查询歌曲 |
| `lookup_album` | 查询专辑 |
| `lookup_artist` | 查询歌手 |

由于 SQL 查询需要精确字符串，教程先为 artists、tracks、albums 建立 vector store，用相似搜索把用户输入归一化到数据库中的真实名称。

### Parent Agent

Parent agent 只负责：

1. 用 structured output 判断用户 intent。
2. 路由到 `refund_agent` 或 `question_answering_agent`。
3. 整理最终 followup。

### Final Response Evaluator

#### Dataset

E2E dataset 同时保存：

| 字段 | 说明 |
|------|------|
| `inputs.question` | 用户问题 |
| `outputs.response` | 期望最终回答 |
| `outputs.trajectory` | 期望路径 |

#### Evaluator

使用 LLM-as-judge 比较：

- QUESTION
- GROUND TRUTH RESPONSE
- STUDENT RESPONSE

输出结构：

- `reasoning`
- `is_correct`

#### Target

```python
async def run_graph(inputs: dict) -> dict:
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": inputs["question"]}]},
        config={"env": "test"},
    )
    return {"response": result["followup"]}
```

#### 使用场景

Final response evaluator 适合判断用户最终是否得到了正确帮助，但失败时定位能力弱。

### Trajectory Evaluator

#### 目标

判断 Agent 是否经过了预期步骤。例如：

```python
["question_answering_agent", "lookup_track"]
```

或：

```python
["refund_agent", "lookup"]
```

#### Evaluator

教程用 subsequence 方式计算部分得分：

```python
def trajectory_subsequence(outputs: dict, reference_outputs: dict) -> float:
    ...
    return matched_steps / len(reference_outputs["trajectory"])
```

这样比 exact match 更宽容：

- 允许中间多走一些步骤。
- 能给部分正确路径打分。
- 适合复杂 Agent 多路径问题。

#### Target

用 LangGraph streaming 记录轨迹：

- `subgraphs=True`：捕捉子图事件。
- `stream_mode="debug"`：捕捉详细事件。
- 当 event type 为 `task` 时记录 node name。
- 如果进入 ToolsNode，还记录具体 tool call name。

### Single Step Evaluator

#### 目标

单独测试 Agent 的关键一步，例如 intent classifier 是否把用户路由到正确子图。

#### Dataset

每条 example 包含：

| 字段 | 说明 |
|------|------|
| `inputs.messages` | 对话消息 |
| `outputs.route` | 期望路由 |

#### Evaluator

```python
def correct(outputs: dict, reference_outputs: dict) -> bool:
    return outputs["route"] == reference_outputs["route"]
```

#### Target

直接调用图中的某个节点：

```python
async def run_intent_classifier(inputs: dict) -> dict:
    command = await graph.nodes["intent_classifier"].ainvoke(inputs)
    return {"route": command.goto}
```

### 三类 Agent 评估对比

| 类型 | 评估对象 | 优点 | 局限 |
|------|----------|------|------|
| Final response | 最终用户响应 | 最贴近用户体验 | 难定位内部失败 |
| Trajectory | 节点和工具路径 | 可定位路径错误，可给部分分 | 需要收集或定义期望路径 |
| Single step | 某个关键节点 | 快、准、适合调试 | 不能代表整体性能 |

---

## Backtesting 新版本 Agent

### 教程目标

Backtesting 教程展示如何用生产历史数据评估新版本应用，判断升级是否真的更好。

基本步骤：

1. 从 production tracing project 选择样本 runs。
2. 将 run inputs 转为 dataset examples。
3. 将原 run outputs 记录为 baseline experiment。
4. 用新系统在同一 dataset 上重跑。
5. 比较 baseline 和 candidate experiments。

### 为什么需要 Backtesting

预生产 dataset 很有用，但不一定代表真实用户行为。Backtesting 使用线上历史 inputs，因此更能反映新版本上线后的真实效果。

适合场景：

- 更换模型。
- 修改 prompt。
- 更换工具。
- 调整 Agent 架构。
- 增加约束或安全策略。

### 示例应用

教程使用 Tweet-writing Agent：

要求：

- 根据主题搜索资料。
- 写一条不超过 280 字的 tweet。
- 至少包含 3 个 emoji。
- 必须使用搜索工具。
- 只基于搜索内容写，不依赖内部知识。
- 需要时附来源链接。

Baseline 使用 GPT-3.5 Turbo，candidate 使用更强模型。

### 模拟生产数据

教程用一组 topic 批量调用 agent，生成 tracing runs。

```python
agent.batch(
    [{"messages": [{"role": "user", "content": topic}]} for topic in fake_inputs]
)
```

### 选择 Runs

通过 `client.list_runs` 从 tracing project 中按时间窗口筛选 root runs。

```python
prod_runs = list(
    client.list_runs(
        project_name=project_name,
        is_root=True,
        filter=run_filter,
    )
)
```

### 转成 Dataset + Baseline Experiment

使用 `convert_runs_to_test`：

```python
from langsmith.beta import convert_runs_to_test

convert_runs_to_test(
    prod_runs,
    dataset_name=dataset_name,
    include_outputs=False,
    load_child_runs=True,
    test_project_name=baseline_experiment_name,
)
```

含义：

| 参数 | 说明 |
|------|------|
| `dataset_name` | 生成的 dataset 名 |
| `include_outputs` | 是否把原 outputs 作为 reference outputs |
| `load_child_runs` | 是否包含完整 child traces |
| `test_project_name` | baseline experiment 名 |

如果没有 ground truth，通常 `include_outputs=False`，然后使用 reference-free evaluators。

### Backtesting Evaluators

Tweet agent 没有标准答案，所以 evaluator 都不依赖 reference outputs。

#### 长度检查

```python
def lt_280_chars(outputs: dict) -> bool:
    return len(final_message_content) <= 280
```

#### Emoji 数量检查

```python
def gte_3_emojis(outputs: dict) -> bool:
    return len(emoji.emoji_list(final_message_content)) >= 3
```

#### Groundedness

从 tool messages 中提取搜索上下文，用 LLM-as-judge 判断最终 tweet 是否被 context 支持。

### 评估 Baseline

baseline experiment 已由历史 runs 转换而来，可以直接对 experiment 名运行 evaluator：

```python
baseline_results = await client.aevaluate(
    baseline_experiment_name,
    evaluators=[lt_280_chars, gte_3_emojis, is_grounded],
)
```

### 评估 Candidate

用新系统在同一个 dataset 上重跑：

```python
candidate_results = await client.aevaluate(
    agent.with_config(model="gpt-5.5"),
    data=dataset_name,
    evaluators=[lt_280_chars, gte_3_emojis, is_grounded],
    experiment_prefix="candidate-gpt-5.5",
)
```

### 比较结果

教程中的发现是一个很典型的 tradeoff：

- 新模型更会遵守格式规则，例如 emoji 数量。
- 但新模型更容易使用内部知识，导致 groundedness 降低。

这说明“升级更强模型”不一定等于“系统整体更好”。Backtesting 能在部署前发现这些问题。

### Backtesting 价值

| 价值 | 说明 |
|------|------|
| 使用真实 inputs | 比手工测试集更贴近生产 |
| 保留 baseline | 能看到当前线上版本表现 |
| 不依赖 ground truth | 可用 reference-free evaluators |
| 支持版本比较 | 直观看到 candidate 的提升和退化 |
| 形成长期数据资产 | 生成的 dataset 可版本化并持续复用 |

---

## Evaluator 设计模式

### Code / Heuristic Evaluator

适合确定性规则。

| 场景 | 示例 |
|------|------|
| 长度 | 不超过 280 字 |
| 格式 | JSON 可解析 |
| 标签 | output label 等于 reference label |
| 工具参数 | ticker 等于 AAPL |
| 数值 | 与 expected 差值小于阈值 |
| 路径 | 是否包含某个工具 |

优点：

- 快。
- 便宜。
- 稳定。
- 适合作为 CI 门禁。

### LLM-as-judge Evaluator

适合语义判断。

| 场景 | 示例 |
|------|------|
| Correctness | 回答与 reference answer 是否一致 |
| Relevance | 回答是否回应问题 |
| Groundedness | 回答是否被上下文支持 |
| Helpfulness | 是否真正有帮助 |
| Final response equivalence | 最终回复是否等价于标准回复 |

建议：

- 使用结构化输出。
- 让模型先解释 reasoning，再给 final score。
- temperature 设为 0。
- 明确评分标准和 True/False 含义。
- 对 judge 结果做人工抽查。

### Trajectory Evaluator

适合 Agent。

| 方法 | 特点 |
|------|------|
| Exact match | 严格要求路径完全一致 |
| Subsequence score | 允许中间多步，给部分分 |
| Expected tools set | 只看是否调用了必要工具 |
| LLM trajectory judge | 判断完整轨迹是否合理 |

教程中推荐的 subsequence 思路更适合复杂 Agent，因为 Agent 可能有多条合理路径。

### Testing Assertion

在 Pytest / Vitest / Jest 中，直接使用断言：

- `assert actual == expected`
- `expect(toolCalls).toContain("code_tool")`
- `expect(abs(actual - expected)).toBeLessThanOrEqual(1)`

同时用 LangSmith logging 记录：

- inputs。
- outputs。
- reference outputs。
- feedback。

这样既能作为测试，又能在 LangSmith 中追踪测试结果。

---

## CI/CD 与长期追踪

### Pytest

```bash
pytest --langsmith-output tests
```

测试中：

```python
import pytest
from langsmith import testing as t

@pytest.mark.langsmith
def test_my_agent():
    t.log_inputs(...)
    t.log_reference_outputs(...)
    t.log_outputs(...)
    t.log_feedback(key="num_steps", score=...)
```

### Vitest / Jest

Vitest reporter：

```typescript
reporters: ["langsmith/vitest/reporter"]
```

Jest reporter：

```typescript
reporters: ["langsmith/jest/reporter"]
```

测试中：

```typescript
import * as ls from "langsmith/vitest";

ls.test("should do something", { inputs, referenceOutputs }, async () => {
  const result = await app.invoke(inputs);
  ls.logOutputs(result);
});
```

### CI 门禁策略

| 门禁类型 | 示例 |
|----------|------|
| 单例断言 | 某个工具参数必须正确 |
| 聚合指标 | correctness 平均分 >= 0.8 |
| 安全指标 | groundedness 必须全部 pass |
| 性能指标 | num_steps 不超过阈值 |
| 回归检查 | candidate 不低于 baseline |

### 长期追踪

LangSmith 可在 Experiments tab 中追踪：

- 各 evaluator 指标随时间变化。
- 不同 experiment 的对比。
- git branch / commit 关联信息。
- 每条 example 的 improvement / regression。

---

## 实践建议

### 从 Chatbot 教程学到的

- 先建立小 dataset，不要一开始追求大而全。
- 一个 code evaluator + 一个 LLM-as-judge 就能形成最小闭环。
- 多跑几个 prompt / model 版本，马上能看到比较价值。
- CI/CD 中要设置明确阈值，而不只是记录结果。

### 从 RAG 教程学到的

- RAG 不应只评估最终答案。
- 至少拆成 retrieval relevance、groundedness、answer relevance、correctness。
- target 必须返回 retrieved documents，否则无法评估 groundedness 和 retrieval。
- 没有 reference answer 也能做很多 reference-free 评估。

### 从 Agent 测试教程学到的

- 工具调用可以被测试得很具体：是否调用、调用哪个、参数是什么。
- 复杂任务可检查“必要工具 + 最终答案”，避免过度约束路径。
- 记录 `num_steps` 是观察 Agent 效率的好办法。
- LLM-as-judge 也可以集成进测试框架。

### 从复杂 Agent 教程学到的

- Final response、trajectory、single step 解决不同问题，最好组合使用。
- 可以直接调用图中的某个节点评估关键决策。
- 评估会产生副作用的流程时，要提供 test/mock config。
- 轨迹评估用部分分比 exact match 更实用。

### 从 Backtesting 教程学到的

- 生产历史 inputs 是非常有价值的回归数据。
- 没有 ground truth 时，也可以用 reference-free evaluator。
- 模型升级可能提升一项指标，同时拉低另一项指标。
- backtesting 能在部署前发现真实 tradeoff。

---

## 快速参考

### 教程场景速查

| 场景 | Dataset | Evaluators | 运行方式 |
|------|---------|------------|----------|
| Chatbot | QA examples + reference answers | correctness、concision | `client.evaluate` |
| RAG | questions + reference answers | correctness、relevance、groundedness、retrieval relevance | `client.evaluate` |
| ReAct Agent | 测试输入 + expected tool calls / answers | assertions、groundedness judge | Pytest / Vitest / Jest |
| Complex Agent | final response + trajectory | final answer judge、trajectory score、route accuracy | `client.aevaluate` |
| Backtesting | production run inputs | length、emoji count、groundedness | `convert_runs_to_test` + `aevaluate` |

### Evaluator 输入速查

| Evaluator | 需要 inputs | 需要 outputs | 需要 reference_outputs |
|-----------|-------------|--------------|------------------------|
| Concision | 否 | 是 | 是 |
| Correctness | 是 | 是 | 是 |
| Relevance | 是 | 是 | 否 |
| Groundedness | 否 | 是，含 documents / tool outputs | 否 |
| Retrieval relevance | 是 | 是，含 documents | 否 |
| Tool arg check | 可选 | 是，含 tool_calls | 是 |
| Trajectory subsequence | 否 | 是，含 trajectory | 是 |
| Route correctness | 否 | 是，含 route | 是 |

### 常用 API

| API | 用途 |
|-----|------|
| `Client()` | LangSmith SDK client |
| `client.create_dataset` | 创建 dataset |
| `client.create_examples` | 创建 examples |
| `client.evaluate` | 同步运行 evaluation |
| `client.aevaluate` | 异步运行 evaluation |
| `traceable` | 标记函数可追踪 |
| `wrappers.wrap_openai` | 包装 OpenAI SDK 以追踪调用 |
| `t.log_inputs` | Pytest 中记录 inputs |
| `t.log_outputs` | Pytest 中记录 outputs |
| `t.log_reference_outputs` | Pytest 中记录 reference outputs |
| `t.log_feedback` | Pytest 中记录 feedback |
| `t.trace_feedback` | 单独追踪 judge feedback |
| `ls.wrapEvaluator` | Vitest/Jest 中包装 evaluator |
| `convert_runs_to_test` | 将 production runs 转为 dataset + baseline experiment |

### 一句话总结

- Chatbot 教程教你搭建最小评估闭环。
- RAG 教程教你把检索和生成分开评估。
- ReAct Agent 教程教你把工具调用写成自动化测试。
- 复杂 Agent 教程教你组合 final response、trajectory、single-step。
- Backtesting 教程教你用生产 traces 评估新版本是否真的更好。
