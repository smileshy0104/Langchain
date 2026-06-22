# LangSmith Evaluation Approaches（应用专项评估方法）详细指南

> 基于官方文档 [Application-specific evaluation approaches](https://docs.langchain.com/langsmith/evaluation-approaches) 整理的中文增强版。本文聚焦不同 LLM 应用类型如何设计评估方案，包括 Agents、RAG、Summarization、Classification / Tagging，并补充评估对象、数据集要求、Evaluator 选择和落地建议。

## 目录

1. [概述](#概述)
2. [评估方案总览](#评估方案总览)
3. [Agents 评估](#agents-评估)
4. [RAG 评估](#rag-评估)
5. [Summarization 评估](#summarization-评估)
6. [Classification 与 Tagging 评估](#classification-与-tagging-评估)
7. [应用类型对比](#应用类型对比)
8. [落地工作流](#落地工作流)
9. [常见坑与建议](#常见坑与建议)
10. [快速参考](#快速参考)

---

## 概述

上一份 Evaluation concepts 文档讲的是 LangSmith 评估体系的通用概念：离线评估、在线评估、dataset、run、thread、evaluator、reference-free、reference-based 等。

本文进一步回答一个更实战的问题：

**不同类型的 LLM 应用，到底应该评估什么？**

LLM 应用形态不同，评估重点也不同：

| 应用类型 | 核心风险 | 评估重点 |
|----------|----------|----------|
| Agents | 工具选错、参数错误、路径绕远、最终任务没完成 | 最终回答、单步决策、完整轨迹 |
| RAG | 检索不相关、答案幻觉、答案不够有用 | 文档相关性、忠实度、帮助性、正确性 |
| Summarization | 摘要失真、遗漏重点、冗余、不符合用户需求 | 事实准确性、忠实度、帮助性 |
| Classification / Tagging | 标签错误、召回不足、误报过多 | accuracy、precision、recall、规则标签 |

### 核心思想

```
┌─────────────────────────────────────────────────────────────┐
│           Application-specific Evaluation                    │
│                                                             │
│  不同应用类型要评估的不是同一件事：                          │
│                                                             │
│  Agent        → 是否选对工具、走对路径、完成任务              │
│  RAG          → 是否检索到相关材料，回答是否基于材料          │
│  Summary      → 是否忠于原文，是否有用，是否遗漏重点          │
│  Classification → 标签是否正确，误报/漏报是否可控             │
│                                                             │
│  共同方法：                                                   │
│  ├─ Offline evaluation：适合有 reference 的发布前测试         │
│  ├─ Online evaluation：适合 reference-free 的生产监控         │
│  ├─ LLM-as-judge：适合语义、质量、忠实度判断                  │
│  ├─ Code evaluator：适合确定性标签和格式检查                  │
│  └─ Pairwise evaluation：适合比较不同版本优劣                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 评估方案总览

### 先按应用类型选评估对象

| 应用类型 | 输入 | 输出 | 常见 reference | 推荐 evaluator |
|----------|------|------|----------------|----------------|
| Agent 最终回答 | 用户请求、可选工具列表 | Agent 最终响应 | 期望答案或任务完成标准 | LLM-as-judge、人类评审 |
| Agent 单步 | 当前消息、工具列表、可选历史步骤 | LLM 选择的下一步行动或 tool call | 期望工具名和参数 | Code evaluator、启发式规则 |
| Agent 轨迹 | 用户请求、可选工具列表 | 工具调用序列或完整消息轨迹 | 期望工具序列或参考轨迹 | 轨迹匹配、LLM-as-judge |
| RAG | 用户问题、检索文档、回答 | 检索结果或最终答案 | 标准答案或标准文档 | LLM-as-judge |
| 摘要 | 原文、用户需求 | 摘要文本 | 通常不需要固定参考摘要 | Reference-free LLM-as-judge |
| 分类/标签 | 待分类文本 | 标签 | ground truth label | Code evaluator、统计指标 |

### 再判断是否需要 reference

| 是否有 reference | 适合场景 | 典型方法 |
|------------------|----------|----------|
| 有标准答案 | RAG 正确性、分类标签、Agent 期望工具 | Offline evaluation |
| 没有标准答案 | 生产流量、摘要质量、内容安全、RAG 忠实度 | Online / Offline reference-free |
| 有两个候选输出 | prompt A/B、模型 A/B、RAG chain 对比 | Pairwise evaluation |

---

## Agents 评估

### Agent 的基本组成

文档中把 LLM-powered autonomous agents 概括为三个核心能力：

| 能力 | 说明 |
|------|------|
| Tool calling | 模型根据输入决定调用哪个工具，以及传入什么参数 |
| Memory | 记住短期消息历史或其他状态 |
| Planning | 通过提示词、推理或工作流决定下一步行动 |

一个典型 tool-calling agent 的流程如下：

```
┌────────────┐
│ User input │
└─────┬──────┘
      ▼
┌──────────────┐
│ Assistant LLM │
│ 判断是否调用工具│
└─────┬────────┘
      │
      ├── 没有 tool call ──▶ 返回最终回答
      │
      ▼
┌──────────────┐
│ Tool condition│
│ 是否选中工具   │
└─────┬────────┘
      ▼
┌──────────────┐
│  Tool node    │
│  执行工具      │
└─────┬────────┘
      ▼
┌──────────────┐
│ Tool message  │
│ 返回给 LLM     │
└─────┬────────┘
      └──────────────▶ 回到 Assistant LLM
```

这个循环会持续进行，直到 assistant 不再选择工具，而是直接返回最终响应。

### Agent 的三类评估

文档将 Agent 评估分成三类：

| 评估类型 | 评估什么 | 适合解决的问题 |
|----------|----------|----------------|
| Final Response | Agent 最终回答是否完成任务 | 用户是否得到正确结果 |
| Single Step | 某一步是否选对工具、参数是否正确 | 定位具体决策错误 |
| Trajectory | 整个工具调用路径是否合理 | 判断 Agent 是否走了预期路径 |

这三类不是互斥关系。真实项目中，通常需要组合使用。

### 评估 Agent 最终回答

#### 评估目标

把 Agent 当作黑盒，只看它最终是否完成了用户任务。

#### 输入与输出

| 项目 | 内容 |
|------|------|
| 输入 | 用户输入，可选工具列表 |
| 输出 | Agent 的最终响应 |
| reference | 可选，期望答案或任务完成标准 |

有些 Agent 的工具是硬编码的，不需要把工具列表作为输入。有些 Agent 更通用，工具会在运行时传入。

#### Evaluator 选择

如果最终输出是自然语言，LLM-as-judge 通常很有效，因为它可以直接判断：

- 是否完成任务。
- 是否回答了用户问题。
- 是否符合约束。
- 是否输出了足够信息。

#### 优点

- 贴近真实用户体验。
- 可以评估整体任务完成情况。
- 对复杂 Agent 来说实现简单。

#### 缺点

- 运行较慢，因为需要执行完整 Agent。
- 失败时不好定位内部哪一步出错。
- 指标设计可能比较困难。

### 评估 Agent 单步决策

#### 评估目标

只评估 Agent 在某一步做出的行动选择，例如“这一步是否应该调用搜索工具”。

#### 输入与输出

| 项目 | 内容 |
|------|------|
| 输入 | 当前步骤输入，可能包含用户问题、工具列表、历史步骤 |
| 输出 | 当前 LLM 响应，通常包含 tool calls |
| reference | 期望工具名、期望参数或判断规则 |

#### Evaluator 选择

常见做法：

- 二分类评分：是否选中了正确工具。
- 参数启发式检查：工具输入是否包含必需字段、是否格式正确。
- reference tool 可以简单用字符串表示，例如 `"search"`、`"calculator"`。

#### 优点

- 更容易定位问题。
- 运行速度快，因为通常只涉及一次 LLM 调用。
- 可以使用简单的 heuristic evaluator。

#### 缺点

- 只能覆盖单步，不能代表完整 Agent。
- 构造后续步骤的数据集较难，因为需要包含此前的消息、工具结果和状态。
- 更适合评估轨迹早期步骤，后期步骤数据构造复杂度明显上升。

### 评估 Agent 轨迹

#### 评估目标

评估 Agent 完成任务过程中调用了哪些工具、调用顺序是否合理、路径是否接近预期。

#### 输入与输出

| 项目 | 内容 |
|------|------|
| 输入 | 整体 Agent 输入，例如用户请求和可选工具列表 |
| 输出 | 工具调用列表，或完整消息轨迹 |
| reference | 期望工具序列、期望工具集合或完整参考轨迹 |

#### 轨迹评估方式

| 方法 | 说明 | 优点 | 局限 |
|------|------|------|------|
| Exact trajectory match | 工具调用序列必须完全匹配 | 简单、确定性强 | 太严格，无法容忍多条正确路径 |
| Incorrect steps count | 统计错误步骤数量 | 能区分接近正确和完全错误 | 仍需定义什么是错误步骤 |
| Expected tools set | 只检查是否调用了预期工具，不关心顺序 | 适合顺序不重要的任务 | 不评估调用顺序和参数 |
| Full trajectory LLM-as-judge | 把完整轨迹和参考轨迹交给 LLM 评估 | 可评估完整行为和参数质量 | reference 难构造，指标设计复杂 |

#### 重要限制

很多简单轨迹评估只关注“选了哪些工具”，并不检查工具参数是否正确。要评估参数和完整行为，需要把 LLM 响应、tool calls、tool outputs 等完整消息轨迹提供给 LLM-as-judge 或人工评审。

### Agent 评估建议

| 阶段 | 建议 |
|------|------|
| 早期开发 | 先评估 final response，快速判断整体可用性 |
| 调试问题 | 增加 single step evaluator，定位工具选择或参数错误 |
| 生产前回归 | 建立关键任务 trajectory dataset |
| 复杂 Agent | 同时使用 final response、single step、trajectory |
| 多路径任务 | 避免只用 exact match，优先考虑 expected tools set 或 LLM-as-judge |

---

## RAG 评估

### RAG 的评估重点

Retrieval Augmented Generation 通过检索外部文档，并把相关文档交给语言模型生成回答。它的质量问题通常来自两个层面：

1. **Retrieval 层**：有没有检索到相关、充分、正确的文档。
2. **Generation 层**：回答是否基于文档，是否准确、有用、没有幻觉。

```
┌──────────────┐
│ User question│
└──────┬───────┘
       ▼
┌──────────────┐
│  Retriever   │───▶ 评估：Document relevance
└──────┬───────┘
       ▼
┌──────────────┐
│ Retrieved docs│
└──────┬───────┘
       ▼
┌──────────────┐
│     LLM       │───▶ 评估：faithfulness / helpfulness / correctness
└──────┬───────┘
       ▼
┌──────────────┐
│ Final answer  │
└──────────────┘
```

### Dataset 关键问题

评估 RAG 时，最重要的问题是：

**你是否有每个问题对应的 reference answer？**

如果有 reference answer，可以评估 answer correctness。如果没有，也仍然可以通过 reference-free evaluator 评估文档相关性、答案忠实度和帮助性。

### RAG Evaluator

RAG 常用 LLM-as-judge，因为它擅长判断文本之间的事实一致性、语义相关性和回答质量。

RAG evaluator 可以分为两类：

| 类型 | 说明 | 适用评估 |
|------|------|----------|
| 需要 reference output | 把生成答案或检索结果与标准答案/标准文档比较 | correctness |
| 不需要 reference output | 根据问题、文档、答案做自洽性检查 | relevance、faithfulness、helpfulness |

### RAG 常见评估指标

| Evaluator | 评估问题 | 是否需要 reference | 是否适合 LLM-as-judge | 是否适合 Pairwise |
|-----------|----------|--------------------|------------------------|-------------------|
| Document relevance | 检索文档是否与问题相关 | 否 | 是 | 否 |
| Answer faithfulness | 答案是否基于检索文档，没有幻觉 | 否 | 是 | 否 |
| Answer helpfulness | 答案是否有助于解决用户问题 | 否 | 是 | 否 |
| Answer correctness | 答案是否与标准答案一致 | 是 | 是 | 否 |
| Pairwise comparison | 多个 RAG 版本哪个更好 | 否 | 是 | 是 |

### Offline / Online / Pairwise 的使用方式

| 方式 | 适合 RAG 的什么评估 |
|------|--------------------|
| Offline evaluation | 任何依赖 reference answer 的评估，尤其是 answer correctness |
| Online evaluation | reference-free 评估，如相关性、忠实度、帮助性、线上质量监控 |
| Pairwise evaluation | 比较不同 RAG chain、不同 prompt、不同模型或不同检索策略 |

### RAG 评估建议

| 目标 | 建议 |
|------|------|
| 先判断检索是否有问题 | 评估 document relevance |
| 防止模型基于文档编造 | 评估 answer faithfulness |
| 判断回答是否真正解决问题 | 评估 answer helpfulness |
| 有标准答案时做发布门禁 | 评估 answer correctness |
| 比较两个检索方案 | 使用 pairwise evaluation |

---

## Summarization 评估

### 摘要任务的特点

Summarization 是一种自由写作任务。它通常不是只有一个正确答案，而是存在多个合理摘要。因此摘要评估往往不是比较某个固定 reference summary，而是根据一组标准评价摘要质量。

常见质量标准：

- 是否忠于原文。
- 是否事实准确。
- 是否覆盖重点。
- 是否符合用户需求。
- 是否简洁清晰。
- 是否没有幻觉。

### Dataset 来源

摘要评估常见数据来源：

| 数据来源 | 用法 |
|----------|------|
| Developer curated examples | 开发者整理待摘要文本，用于离线评估 |
| Production user logs | 线上摘要应用的真实请求，用于在线评估 |

由于摘要通常使用 reference-free prompt，因此离线和在线评估都可行。

### Evaluator 选择

摘要任务通常使用 LLM-as-judge。原因是摘要质量往往是语义和写作层面的判断，规则 evaluator 难以覆盖。

与 RAG 的 answer correctness 不同，摘要评估不常使用固定 reference summary，因为：

- 同一篇文章可以有多个好摘要。
- 不同用户需求会影响摘要重点。
- 固定参考摘要可能限制模型产生更好的表达。

### 摘要常见评估指标

| Use Case | 评估问题 | 是否需要 reference | 是否适合 LLM-as-judge | 是否适合 Pairwise |
|----------|----------|--------------------|------------------------|-------------------|
| Factual accuracy | 摘要相对原文是否事实准确 | 否 | 是 | 是 |
| Faithfulness | 摘要是否基于源文档，没有幻觉 | 否 | 是 | 是 |
| Helpfulness | 摘要是否满足用户需求 | 否 | 是 | 是 |

### Pairwise 在摘要中的价值

摘要任务非常适合 pairwise evaluation。很多时候，比起给单个摘要打一个绝对分数，人类或 LLM 更容易判断：

- 哪个摘要更准确。
- 哪个摘要更清晰。
- 哪个摘要更符合用户需求。
- 哪个摘要遗漏更少。

### 摘要评估建议

| 目标 | 建议 |
|------|------|
| 检查是否编造 | 使用 faithfulness / hallucination evaluator |
| 检查是否漏掉重点 | 使用 helpfulness evaluator，并明确用户需求 |
| 比较 prompt 版本 | 使用 pairwise evaluation |
| 用于生产监控 | 使用 reference-free online evaluation |
| 用于发布前回归 | 使用开发者精选文本做 offline evaluation |

---

## Classification 与 Tagging 评估

### 分类和打标签任务的特点

Classification / Tagging 是给输入分配标签的任务，例如：

- 情感分类。
- 毒性检测。
- 意图识别。
- 主题标签。
- 风险等级。
- 工单类型。

这类任务的评估关键是：

**是否有 ground truth reference labels？**

### 有 reference labels 的情况

如果数据集中有标准标签，评估目标就是比较模型输出与 ground truth label。

常见指标：

| 指标 | 说明 |
|------|------|
| Accuracy | 所有样本中预测正确的比例 |
| Precision | 被预测为某类的样本中，有多少是真的 |
| Recall | 真实属于某类的样本中，有多少被找出来 |

这种情况通常不需要 LLM-as-judge，使用自定义 heuristic / code evaluator 即可。

### 没有 reference labels 的情况

如果没有标准标签，也可以用 LLM-as-judge 根据指定 criteria 给输入打标签。

适合场景：

- 在线监控用户输入是否有毒性。
- 标记高风险对话。
- 给生产流量自动加标签。
- 根据业务规则对请求做初步分类。

这种方式通常是 reference-free，因此适合 online evaluation。

### 分类/标签评估表

| Use Case | Detail | 是否需要 reference | 是否适合 LLM-as-judge | 是否适合 Pairwise |
|----------|--------|--------------------|------------------------|-------------------|
| Accuracy | 标准准确率 | 是 | 否 | 否 |
| Precision | 标准精确率 | 是 | 否 | 否 |
| Recall | 标准召回率 | 是 | 否 | 否 |

### 分类任务评估建议

| 场景 | 推荐方法 |
|------|----------|
| 有 ground truth labels | Code evaluator + accuracy / precision / recall |
| 没有 ground truth labels | Reference-free LLM-as-judge |
| 线上内容安全标签 | Online evaluation |
| 发布前分类器回归 | Offline evaluation + fixed dataset |
| 类别严重不均衡 | 不要只看 accuracy，还要看 precision / recall |

---

## 应用类型对比

### 是否需要 reference

| 应用类型 | 通常是否需要 reference | 原因 |
|----------|------------------------|------|
| Agent final response | 可选 | 可用任务完成标准，也可用参考答案 |
| Agent single step | 通常需要 | 需要知道期望工具或参数 |
| Agent trajectory | 通常需要 | 需要参考路径或预期工具集合 |
| RAG document relevance | 不需要 | 可以判断问题和文档是否相关 |
| RAG faithfulness | 不需要 | 可以判断答案是否被文档支持 |
| RAG correctness | 需要 | 要与标准答案比较 |
| Summarization | 通常不需要 | 好摘要可能有多个版本 |
| Classification metrics | 需要 | accuracy / precision / recall 需要标准标签 |
| Online tagging | 不需要 | LLM 根据 criteria 直接打标签 |

### Evaluator 选型

| 评估需求 | 优先选择 |
|----------|----------|
| 判断自然语言质量 | LLM-as-judge |
| 判断工具是否选对 | Code evaluator / heuristic |
| 判断工具参数是否有效 | Code evaluator |
| 判断完整 Agent 行为 | Trajectory evaluator + LLM-as-judge |
| 判断 RAG 答案是否忠于文档 | LLM-as-judge |
| 判断摘要是否有用 | LLM-as-judge / Pairwise |
| 判断分类器统计表现 | Code evaluator + metrics |
| 比较两个版本哪个更好 | Pairwise evaluation |

### Offline、Online、Pairwise 选型

| 评估方式 | 最适合 |
|----------|--------|
| Offline evaluation | 有 reference 的发布前测试、回归测试、benchmark |
| Online evaluation | reference-free 的生产监控、线上打标签、异常发现 |
| Pairwise evaluation | 比较 prompt、模型、RAG chain、摘要策略 |

---

## 落地工作流

### 1. 先确定应用类型

不要先问“我要写什么 evaluator”，而是先判断应用属于哪类：

- 是 Agent 吗？
- 是 RAG 吗？
- 是摘要生成吗？
- 是分类/打标签吗？
- 是否混合了多种形态？

很多真实系统是混合型。例如“带工具调用的 RAG Agent”可能需要同时评估：

- 工具选择。
- 检索文档。
- 回答忠实度。
- 最终任务完成度。

### 2. 拆成可评估的层

示例：RAG Agent 可以拆成：

| 层级 | 评估项 |
|------|--------|
| Agent 单步 | 是否调用检索工具 |
| Retrieval | 检索文档是否相关 |
| Generation | 回答是否忠于文档 |
| Final response | 是否完成用户任务 |
| Production monitoring | 是否出现低质量或安全问题 |

### 3. 选择 reference 策略

| 条件 | 策略 |
|------|------|
| 有标准答案 | 使用 reference-based offline evaluation |
| 没有标准答案，但可以判断质量 | 使用 reference-free LLM-as-judge |
| 有业务标签 | 使用 code evaluator 计算指标 |
| 多个版本待比较 | 使用 pairwise evaluation |

### 4. 建立最小数据集

不同应用的起步数据集建议：

| 应用 | 最小数据集内容 |
|------|----------------|
| Agent | 10 到 20 个关键任务，包含期望工具或期望结果 |
| RAG | 10 到 20 个问题，包含标准答案或高质量参考文档 |
| Summarization | 10 到 20 篇代表性原文和用户需求 |
| Classification | 每类至少若干样本，保留 ground truth labels |

### 5. 组合 evaluator

一个实用组合：

| 层 | Evaluator |
|----|-----------|
| 硬规则 | Code evaluator |
| 语义质量 | LLM-as-judge |
| 版本对比 | Pairwise |
| 高风险样本 | Human review |

### 6. 建立反馈闭环

```
┌──────────────┐
│ Offline eval │
│ 发布前验证    │
└──────┬───────┘
       ▼
┌──────────────┐
│  Deployment  │
└──────┬───────┘
       ▼
┌──────────────┐
│ Online eval  │
│ 生产监控      │
└──────┬───────┘
       ▼
┌──────────────┐
│ Human review │
│ 人工复核      │
└──────┬───────┘
       ▼
┌──────────────┐
│ Dataset      │
│ 回流为样本     │
└──────┬───────┘
       ▼
┌──────────────┐
│ Regression   │
│ 下次回归测试   │
└──────────────┘
```

---

## 常见坑与建议

### Agent 评估常见坑

| 问题 | 建议 |
|------|------|
| 只看最终回答，失败时无法定位 | 增加 single step 和 trajectory evaluation |
| 轨迹 exact match 太严格 | 允许 expected tools set 或多条参考路径 |
| 只检查工具名，不检查参数 | 加入参数 schema 或 LLM-as-judge 轨迹评估 |
| 后期步骤样本难构造 | 优先从真实 traces 中沉淀 |

### RAG 评估常见坑

| 问题 | 建议 |
|------|------|
| 只评估最终答案，不评估检索 | 单独评估 document relevance |
| 把 helpfulness 当 correctness | 有标准答案时单独做 correctness |
| 忽略幻觉 | 加 faithfulness evaluator |
| 线上没有 reference 就不评估 | 使用 reference-free online evaluators |

### 摘要评估常见坑

| 问题 | 建议 |
|------|------|
| 强制要求匹配唯一标准摘要 | 改用 criteria-based reference-free 评估 |
| 只看摘要是否流畅 | 同时检查 factual accuracy 和 faithfulness |
| 很难判断新版本是否更好 | 使用 pairwise evaluation |

### 分类评估常见坑

| 问题 | 建议 |
|------|------|
| 类别不均衡时只看 accuracy | 同时看 precision 和 recall |
| 有标签还用 LLM 评分类别是否正确 | 优先使用 code evaluator |
| 没有标签就无法在线评估 | 用 LLM-as-judge 根据 criteria 打标签 |

---

## 快速参考

### Agent 评估速查

| 评估类型 | 输入 | 输出 | 适合 evaluator |
|----------|------|------|----------------|
| Final Response | 用户输入、工具列表 | 最终回答 | LLM-as-judge |
| Single Step | 当前步骤输入、历史、工具列表 | tool call | heuristic / code |
| Trajectory | 整体输入 | 工具调用序列或完整消息轨迹 | trajectory metric / LLM-as-judge |

### RAG 评估速查

| 指标 | 问题 | Reference |
|------|------|-----------|
| Document relevance | 文档是否相关 | 不需要 |
| Answer faithfulness | 答案是否基于文档 | 不需要 |
| Answer helpfulness | 答案是否有帮助 | 不需要 |
| Answer correctness | 答案是否符合标准答案 | 需要 |
| Pairwise comparison | 哪个版本更好 | 不需要 |

### 摘要评估速查

| 指标 | 问题 | 推荐方式 |
|------|------|----------|
| Factual accuracy | 是否事实准确 | LLM-as-judge |
| Faithfulness | 是否忠于源文档 | LLM-as-judge |
| Helpfulness | 是否满足用户需求 | LLM-as-judge |
| Version comparison | 哪个摘要版本更好 | Pairwise |

### 分类/打标签评估速查

| 场景 | 推荐方式 |
|------|----------|
| 有标准标签 | Code evaluator + accuracy / precision / recall |
| 无标准标签 | Reference-free LLM-as-judge |
| 线上输入分类 | Online evaluation |
| 发布前回归 | Offline evaluation |

### 一句话总结

- **Agent**：评估最终结果，也要评估单步和轨迹。
- **RAG**：同时评估检索质量和生成质量。
- **Summarization**：多用 reference-free criteria 和 pairwise。
- **Classification**：有标签用统计指标，没标签用 LLM-as-judge 在线打标。
- **复杂系统**：按组件拆解评估，不要只看最终输出。
