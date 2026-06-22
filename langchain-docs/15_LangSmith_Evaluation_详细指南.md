# LangSmith Evaluation（评估）详细指南

> 基于官方文档 [Evaluation concepts](https://docs.langchain.com/langsmith/evaluation-concepts) 整理的中文增强版。本文聚焦 LangSmith 中如何定义“好结果”、如何做离线/在线评估、Evaluator 如何工作，以及如何把生产反馈沉淀为可回归的数据集。

## 目录

1. [概述](#概述)
2. [先定义要评估什么](#先定义要评估什么)
3. [离线评估与在线评估](#离线评估与在线评估)
4. [评估生命周期](#评估生命周期)
5. [核心评估对象](#核心评估对象)
6. [Evaluators](#evaluators)
7. [评估技术](#评估技术)
8. [Reference-free 与 Reference-based](#reference-free-与-reference-based)
9. [数据集建设最佳实践](#数据集建设最佳实践)
10. [人工反馈与标注队列](#人工反馈与标注队列)
11. [Evaluation 与 Testing 的区别](#evaluation-与-testing-的区别)
12. [落地工作流](#落地工作流)
13. [快速参考](#快速参考)

---

## 概述

### 为什么需要 Evaluation

LLM 应用的输出通常是非确定性的：同一个输入可能因为模型、温度、提示词、检索结果或工具调用路径不同而产生不同回答。传统测试只能覆盖“是否满足硬性断言”，但很多 LLM 质量问题更接近“好不好”“是否可靠”“有没有更优版本”。

**Evaluation（评估）** 的作用，就是把模糊的质量目标拆成可观测、可比较、可持续追踪的指标。

```
┌─────────────────────────────────────────────────────────────┐
│                  LangSmith Evaluation                        │
│                                                             │
│  目标：把“这个 LLM 应用好不好”变成可持续度量的问题             │
│                                                             │
│  开发阶段：离线评估                                           │
│  ├─ 用人工整理的数据集验证新版本                              │
│  ├─ 对比 prompt、模型、检索策略、工具链                        │
│  └─ 做回归测试和 CI/CD 门禁                                   │
│                                                             │
│  生产阶段：在线评估                                           │
│  ├─ 对真实 trace / run 做监控                                 │
│  ├─ 发现异常、低质回答、安全风险                              │
│  └─ 把问题样本沉淀回离线数据集                                │
└─────────────────────────────────────────────────────────────┘
```

### LangSmith Evaluation 适合解决的问题

| 场景 | 典型问题 | 适合的评估方式 |
|------|----------|----------------|
| RAG 问答 | 检索文档是否相关，答案是否忠于资料 | 离线参考答案评估 + 在线质量监控 |
| Agent | 工具选择是否正确，参数是否规范，执行轨迹是否合理 | 轨迹评估、代码评估、人工标注 |
| Chatbot | 是否有帮助、符合品牌语气、回答是否完整 | LLM-as-judge、人类评估 |
| 结构化输出 | JSON 是否可解析，字段是否完整，分类是否正确 | 代码评估、精确匹配 |
| 生产监控 | 是否出现异常、延迟、敏感信息、低质量回答 | 在线 reference-free 评估 |

---

## 先定义要评估什么

### 从“好结果”开始

官方建议在构建评估前，先明确你的应用里什么叫“好”。不要一上来就写复杂 evaluator，而是先为每个关键组件手工整理少量高质量样例。

建议起步方式：

1. 拆分系统中的关键组件。
2. 为每个组件整理 5 到 10 个“好结果”样例。
3. 根据样例定义质量标准和评估方法。

### 按组件拆解质量标准

| 组件 | 可以评估什么 | 示例指标 |
|------|--------------|----------|
| LLM 调用 | 回答是否准确、完整、风格一致 | correctness、helpfulness、tone |
| Retrieval | 召回文档是否相关，是否漏召关键证据 | relevance、recall、context quality |
| Tool invocation | 工具选择是否正确，参数是否有效 | tool correctness、argument validity |
| Output formatting | 输出是否符合 schema 或 UI 需要 | JSON valid、required fields |
| Agent trajectory | 中间步骤是否合理，有无多余循环 | trajectory quality、step count |

### 不同应用的“好结果”样例

| 应用类型 | 手工样例应该覆盖 |
|----------|------------------|
| RAG 系统 | 好的检索结果、准确且完整的最终答案、无法回答时的拒答策略 |
| Agent | 正确工具选择、参数格式、合理执行路径、失败恢复方式 |
| Chatbot | 能理解用户意图、语气符合品牌、答案有帮助且不越界 |

**核心思路**：先用人工样例定义质量边界，再选择 evaluator 去衡量系统多大概率能产生类似质量的输出。

---

## 离线评估与在线评估

LangSmith 支持两类评估：**Offline Evaluation（离线评估）** 和 **Online Evaluation（在线评估）**。它们不是互相替代关系，而是服务于应用生命周期中的不同阶段。

### Offline Evaluation

离线评估用于发布前测试，目标对象是 **datasets / examples**。因为 examples 通常包含 reference outputs，所以离线评估可以做更精确的正确性判断。

适用场景：

- **Benchmarking**：比较多个版本，选择表现最好的 prompt、模型或链路。
- **Regression testing**：确保新版本没有让质量倒退。
- **Unit testing**：验证某个组件的行为，例如分类器、检索器、输出解析器。
- **Backtesting**：用历史数据回测新版本效果。

### Online Evaluation

在线评估用于生产监控，目标对象是生产环境中的 **runs / threads**。这些数据来自真实 tracing，通常没有 reference outputs，所以更适合做 reference-free 的质量、安全和异常检测。

适用场景：

- **Real-time monitoring**：持续跟踪线上回答质量。
- **Anomaly detection**：发现异常模式、极端案例、长尾问题。
- **Production feedback**：把线上问题样本加入离线数据集，形成后续回归集。

### 核心差异

| 对比项 | Offline Evaluation | Online Evaluation |
|--------|--------------------|-------------------|
| 运行对象 | Dataset / Examples | Tracing Project / Runs / Threads |
| 数据来源 | 人工整理或历史沉淀的测试集 | 生产环境真实流量 |
| 是否有参考答案 | 通常有 reference outputs | 通常没有 reference outputs |
| 使用阶段 | 开发、测试、发布前 | 发布后、生产监控 |
| 主要用途 | benchmark、回归、单元评估、回测 | 实时监控、异常检测、生产反馈 |
| 运行方式 | 批量处理 curated test set | 实时或近实时处理 live trace |
| 数据要求 | 需要维护数据集 | 不需要预先构造数据集 |

---

## 评估生命周期

LangSmith 的评估策略会随着应用从开发走向生产而演进。

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Development │───▶│   Testing   │───▶│ Deployment  │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       ▼                  ▼                  ▼
 Offline Eval       Offline Eval        Online Eval
       │                  │                  │
       └──────────────────┴──────────┬───────┘
                                      ▼
                              ┌─────────────┐
                              │ Monitoring  │
                              └──────┬──────┘
                                     ▼
                              ┌─────────────┐
                              │ Iteration   │
                              └─────────────┘
                                     ▲
                       Online 问题样本沉淀为 Offline 回归集
```

### 1. 开发阶段：离线评估

在发布前，用人工整理的数据集验证功能、比较不同方案，并建立对系统质量的信心。

常见动作：

- 建立小而精的数据集。
- 比较不同 prompt / model / retriever。
- 为关键组件写 evaluator。
- 在 CI 中运行核心回归集。

### 2. 初始发布：在线评估

发布后，用在线评估监控真实行为，发现测试集中没有覆盖的问题。

常见动作：

- 对生产 run 自动打分。
- 检测安全、PII、格式错误、低质量回答。
- 把异常 run 推送到人工标注队列。

### 3. 持续改进：离线与在线闭环

成熟的评估系统不是一次性配置，而是一个反馈循环：

1. 在线评估发现线上问题。
2. 人工复核后把问题 run 转成 dataset example。
3. 离线评估验证修复是否有效。
4. 新版本上线后继续用在线评估确认真实效果。

---

## 核心评估对象

### Datasets

Dataset 是用于评估应用的一组 examples。它是离线评估的基础。

一个 dataset 通常包含：

- 常见输入。
- 边界输入。
- 历史线上问题。
- 不同业务类别的代表性样例。

### Examples

Example 是数据集中的单个测试样本，通常包含：

| 字段 | 说明 |
|------|------|
| `inputs` | 传给应用的输入变量字典 |
| `reference outputs` | 可选，标准答案或期望输出，只给 evaluator 使用，不传给应用 |
| `metadata` | 可选，用于过滤、分组、记录来源或标签 |

Reference outputs 的重要性在于：它们让 evaluator 能判断“实际输出是否接近期望输出”。

### Experiments

Experiment 表示“某个应用版本在某个 dataset 上的评估结果”。

一个 experiment 会记录：

- 每个 example 的应用输出。
- evaluator 分数。
- 执行 trace。
- 版本配置，例如 prompt、模型、检索策略。

同一个 dataset 往往会有多个 experiment，用于比较不同应用版本。

### Runs

Run 是一次应用执行 trace，通常来自生产环境或实验运行。

Run 包含：

- 实际用户输入。
- 应用实际输出。
- 中间步骤，如 LLM 调用、工具调用、检索调用。
- 元数据，如 tags、用户反馈、延迟、错误信息。

在线评估通常直接评估 runs。

### Threads

Thread 是一组相关 runs，通常代表一次多轮对话。

Thread-level evaluation 适合评估会话级质量，例如：

- 多轮上下文是否连贯。
- 是否保持话题。
- 是否逐步解决用户问题。
- 整体用户体验是否满意。

---

## Evaluators

Evaluator 是 LangSmith 中的工作区级资源，用于给应用表现打分。它可以被附加到多个 tracing projects 或 datasets 上复用。

### 可以在哪里运行 Evaluator

| 入口 | 用途 |
|------|------|
| Evaluators 页面 | 将 evaluator 绑定到 tracing project 或 dataset |
| Playground | 在 prompt 调试过程中快速评估 |
| LangSmith SDK | 用 Python / TypeScript 编程运行评估 |
| Rules | 自动在 tracing project 或 dataset 上触发 evaluator |

### Evaluator 输入

离线评估时 evaluator 通常接收：

- `Example`：数据集中的输入、reference outputs、metadata。
- `Run`：应用基于 example input 运行后产生的实际输出和中间步骤。

在线评估时 evaluator 通常接收：

- `Run`：生产 trace，包含输入、输出和中间步骤，但没有标准答案。

### Evaluator 输出

Evaluator 返回 feedback。Feedback 可以是一个字典，也可以是字典列表。

常见字段：

| 字段 | 说明 |
|------|------|
| `key` | 指标名称 |
| `score` | 数值型指标，例如 0 到 1 |
| `value` | 分类或文本型指标，例如 pass / fail |
| `comment` | 可选，解释评分原因 |

示例结构：

```python
{
    "key": "correctness",
    "score": 0.85,
    "comment": "答案覆盖了主要事实，但遗漏了一个限制条件。"
}
```

---

## 评估技术

LangSmith 支持多种评估方法，实际项目中通常会组合使用。

### Human Evaluation

人工评估是很多项目最有效的起点，尤其适合主观质量判断。人可以检查应用输出和完整 trace，包括中间检索结果、工具调用和 Agent 路径。

适合场景：

- 定义早期质量标准。
- 审核 LLM-as-judge 的评分是否可靠。
- 处理品牌语气、专业表达、复杂任务完成度等主观维度。
- 给线上问题样本做高质量标注。

### Code Evaluators

Code evaluator 是确定性的规则函数，适合可程序化判断的问题。

适合场景：

- 响应不能为空。
- JSON 可以解析。
- 必须包含某些字段。
- 生成代码可以编译。
- 分类结果必须精确匹配标签。
- 工具参数符合 schema。

优点是稳定、便宜、可解释；缺点是难以覆盖语义质量。

### LLM-as-judge

LLM-as-judge 使用另一个 LLM 对应用输出打分。评分规则通常写在 grader prompt 中。

可以分为：

| 类型 | 说明 | 示例 |
|------|------|------|
| Reference-free | 不依赖标准答案，只按规则判断输出质量 | 是否有帮助、是否冒犯、是否清晰 |
| Reference-based | 将输出与参考答案比较 | 事实是否正确、是否覆盖标准答案要点 |

使用建议：

- 先人工抽查评分，确认 grader 可靠。
- 对评分 prompt 做迭代。
- 对复杂主观标准加入 few-shot 示例。
- 不要把 LLM-as-judge 当成绝对真理，更适合做趋势、对比和筛查。

### Pairwise Evaluation

Pairwise evaluator 比较两个应用版本的输出，而不是给单个输出绝对打分。

适合场景：

- 总结质量对比。
- A/B 实验。
- 两个 prompt 版本难以绝对评分，但容易判断哪个更好。
- 人类评审两个结果哪个更符合预期。

Pairwise 的优势是：很多时候“哪个更好”比“这个得 0.82 分是否准确”更容易判断。

---

## Reference-free 与 Reference-based

### Reference-free Evaluators

Reference-free evaluator 不需要标准答案，因此既能用于离线评估，也能用于在线评估。

常见用途：

| 类型 | 示例 |
|------|------|
| 安全检查 | 毒性、PII、违规内容 |
| 格式检查 | JSON schema、必填字段、结构有效性 |
| 质量启发式 | 长度、关键词、延迟、步骤数 |
| LLM 主观评分 | 清晰度、连贯性、有帮助程度、语气 |

### Reference-based Evaluators

Reference-based evaluator 需要 reference outputs，因此主要用于离线评估。

常见用途：

| 类型 | 示例 |
|------|------|
| Correctness | 输出是否接近标准答案 |
| Factual accuracy | 是否符合 ground truth |
| Exact match | 分类标签是否完全一致 |
| Reference-based LLM-as-judge | 相对参考答案评估回答质量 |

### 选型建议

| 目标 | 推荐方式 |
|------|----------|
| 生产环境持续监控 | Reference-free |
| 发布前验证正确性 | Reference-based |
| 安全和格式门禁 | Code evaluator + Reference-free |
| 回归测试 | Reference-based + 固定 dataset version |
| 主观质量优化 | Human + LLM-as-judge + Pairwise |

---

## 数据集建设最佳实践

### 手工精选样例

推荐从 10 到 20 个高质量手工样例开始。样例不需要一开始很大，但要足够代表“什么是好结果”。

样例应覆盖：

- 高频用户问题。
- 关键业务路径。
- 边界条件。
- 容易失败的输入。
- 不应该回答或需要拒答的情况。

### 历史 traces

应用上线后，可以从真实 traces 中挑选样本加入数据集。

高价值来源：

- 收到负面用户反馈的 runs。
- 延迟高、报错、重试多的 runs。
- LLM evaluator 标记为低质量的 runs。
- 业务上重要但模型表现不稳定的 conversations。

### Synthetic Data

Synthetic data 可以用已有样例扩增数据集，但最好建立在高质量人工样例基础上。

适合用途：

- 扩展常见输入表达方式。
- 生成不同难度的问题。
- 覆盖更多边界条件。

注意事项：

- 不要只靠合成数据定义质量标准。
- 合成样例需要抽样人工审查。
- 合成数据更适合作为补充，而不是初始真相来源。

### Splits

Splits 是 dataset 的命名子集，用于高层组织样本。

常见 split：

| Split 类型 | 用途 |
|------------|------|
| train / validation / test | 避免只对训练样例过拟合 |
| category-based | 按任务类型分别评估 |
| staged rollout | 将探索性样例与主评估集分开 |

Splits 与 metadata 的区别：

- **Splits**：用于高层评估分组。
- **Metadata**：用于记录每条样例的标签、来源、难度等细粒度信息。

### Versions

LangSmith 会在 examples 改变时自动创建 dataset versions。建议为重要版本打 tag。

适合使用 version / tag 的场景：

- CI 固定评估某个稳定数据集版本。
- 标记发布前基准集。
- 追踪数据集变化对分数的影响。

---

## 人工反馈与标注队列

### Annotation Queues

Annotation queues 用于结构化收集人工反馈。它可以把需要复核的 runs 组织成队列，并按 rubric 进行标注。

它比简单的 inline annotation 更适合团队协作：

- 可以集中管理待审样本。
- 可以设置评分标准。
- 可以分配多个 reviewer。
- 可以启用 reservation，避免多人重复处理同一条 run。
- 可以把已标注 runs 导出为 dataset examples。

### 两类队列

| 队列类型 | 说明 | 适用场景 |
|----------|------|----------|
| Single-run queues | 每次审核一个 run，并按 rubric 打分 | 问题 triage、沉淀数据集 |
| Pairwise queues | 并排比较两个 runs，判断哪个更好 | A/B 对比、实验结果比较 |

### Assertions

Single-run queues 还支持 assertions。Assertions 可以理解为某个样本上的自由文本验收标准，后续离线 evaluator 可以根据这些标准判断新版本是否满足要求。

---

## Evaluation 与 Testing 的区别

Evaluation 和 Testing 相似，但目标不同。

### Evaluation

Evaluation 衡量系统在指标上的表现。指标可能是模糊的、主观的，并且经常用于相对比较。

例如：

- 新 prompt 的 helpfulness 是否高于旧 prompt。
- RAG 回答 factuality 是否有所提升。
- 新模型是否在同一数据集上更稳定。

### Testing

Testing 判断系统是否满足硬性正确性要求。测试更像发布门禁，不通过就不能上线。

例如：

- 输出必须是合法 JSON。
- API 响应不能报错。
- 分类标签必须属于允许集合。
- 回答不得包含 PII。

### 二者如何结合

评估指标可以转化为测试门禁。

例如：

- 新版本平均分必须高于 baseline。
- 关键 split 的 correctness 不能下降。
- 安全 evaluator 必须全部 pass。

对于运行成本较高的 LLM 应用，可以把测试和评估放在同一批执行中完成，提高效率。

---

## 落地工作流

### 推荐从小闭环开始

```
┌──────────────────┐
│ 1. 定义质量标准   │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 2. 整理 10-20 样例│
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 3. 编写 Evaluator │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 4. 跑离线实验     │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 5. 比较版本差异   │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 6. 上线并在线监控 │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 7. 线上问题回流   │
└──────────────────┘
```

### RAG 应用评估建议

| 评估层 | 关注点 | Evaluator 类型 |
|--------|--------|----------------|
| 检索层 | 文档是否相关、是否覆盖关键证据 | LLM-as-judge、人工评审 |
| 生成层 | 答案是否忠于上下文、是否完整 | Reference-based LLM-as-judge |
| 格式层 | 是否按要求输出引用、JSON 或 markdown | Code evaluator |
| 安全层 | 是否泄露敏感信息、是否编造事实 | Reference-free evaluator |

### Agent 应用评估建议

| 评估层 | 关注点 | Evaluator 类型 |
|--------|--------|----------------|
| 工具选择 | 是否选择了正确工具 | Code / LLM-as-judge |
| 参数构造 | 参数字段和类型是否正确 | Code evaluator |
| 执行轨迹 | 是否有多余步骤、是否陷入循环 | Trajectory evaluator、人工评审 |
| 最终结果 | 是否完成用户目标 | Human、LLM-as-judge |

### Chatbot 应用评估建议

| 评估层 | 关注点 | Evaluator 类型 |
|--------|--------|----------------|
| 意图理解 | 是否真正回答用户问题 | LLM-as-judge |
| 语气风格 | 是否符合品牌要求 | Human、LLM-as-judge |
| 对话连续性 | 多轮上下文是否一致 | Thread-level evaluation |
| 安全合规 | 是否违反内容策略 | Reference-free evaluator |

---

## 快速参考

### 核心术语

| 术语 | 含义 |
|------|------|
| Dataset | 用于离线评估的一组 examples |
| Example | 单个测试样本，包含 inputs、可选 reference outputs 和 metadata |
| Experiment | 某个应用版本在 dataset 上运行评估后的结果集合 |
| Run | 一次应用执行 trace，包含输入、输出、中间步骤和元数据 |
| Thread | 多个相关 runs 组成的多轮对话 |
| Evaluator | 给应用输出或 trace 打分的评估器 |
| Feedback | Evaluator 产生的分数、标签或评论 |
| Annotation queue | 用于人工评审 runs 的结构化队列 |

### 什么时候用什么

| 需求 | 推荐方案 |
|------|----------|
| 发布前比较两个 prompt | Offline evaluation + experiments comparison |
| 防止新版本质量倒退 | Regression dataset + fixed version + CI |
| 监控线上安全问题 | Online reference-free evaluator |
| 判断 JSON 是否有效 | Code evaluator |
| 判断回答是否符合参考答案 | Reference-based LLM-as-judge |
| 比较两个版本哪个更好 | Pairwise evaluation |
| 收集人工高质量反馈 | Annotation queues |
| 把线上问题变成回归用例 | Export annotated runs to dataset |

### 最小可行评估体系

一个实用的起步配置：

1. 为核心任务整理 10 到 20 条 examples。
2. 每条 example 包含 inputs 和 reference outputs。
3. 写 1 个 code evaluator 检查格式或结构。
4. 写 1 个 LLM-as-judge evaluator 检查语义质量。
5. 每次修改 prompt / model / retriever 后跑一次 experiment。
6. 上线后对生产 runs 做 reference-free 在线评估。
7. 每周把低分或人工反馈差的 runs 回流到 dataset。

### 关键原则

- 先人工定义“好”，再自动化评估。
- 离线评估负责发布前信心，在线评估负责生产反馈。
- Reference-based 更适合 correctness，reference-free 更适合生产监控。
- Code evaluator 负责硬性规则，LLM-as-judge 负责语义判断，Human 负责校准标准。
- 数据集不只追求数量，更要覆盖真实高价值场景。
- 评估不是一次性任务，而是开发、发布、监控、迭代的闭环。
