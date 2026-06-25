# Langfuse Overview（平台概览）详细指南

> 基于 Langfuse 官方 Overview 与 Example Project 文档整理的中文总结。本文聚焦 Langfuse 的定位、核心能力、可观测性、Prompt Management、Evaluation、Example Project 的使用方式，以及从 PoC 到生产的落地路径。

## 目录

1. [概述](#概述)
2. [Langfuse 是什么](#langfuse-是什么)
3. [核心能力总览](#核心能力总览)
4. [Observability](#observability)
5. [Prompt Management](#prompt-management)
6. [Evaluation](#evaluation)
7. [Example Project](#example-project)
8. [从哪里开始](#从哪里开始)
9. [为什么选择 Langfuse](#为什么选择-langfuse)
10. [实践建议](#实践建议)
11. [快速参考](#快速参考)

---

## 概述

Langfuse 是一个开源 AI engineering platform，用于帮助团队协作调试、分析和迭代 LLM 应用。它覆盖 LLM 应用开发生命周期中的几个关键环节：

- 线上 tracing 与 observability。
- Prompt 管理、版本控制和发布。
- 质量评估、用户反馈、人工标注和实验。
- 数据集与 prompt/model 对比。
- Dashboard 中的质量、成本、延迟分析。

```
┌─────────────────────────────────────────────────────────────┐
│                         Langfuse                            │
│                                                             │
│  目标：帮助团队构建、调试、评估、迭代 LLM 应用                 │
│                                                             │
│  Observability                                               │
│  ├─ traces / sessions / users                                │
│  ├─ latency / cost / input / output                          │
│  └─ agent graphs / dashboard                                 │
│                                                             │
│  Prompt Management                                           │
│  ├─ prompt 创建、版本、标签、部署                              │
│  ├─ playground 测试                                           │
│  └─ prompts 与 traces / metrics 关联                          │
│                                                             │
│  Evaluation                                                  │
│  ├─ LLM-as-a-judge / code evaluators                         │
│  ├─ user feedback / manual labeling                          │
│  ├─ datasets / experiments                                   │
│  └─ custom scores                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Langfuse 是什么

Langfuse 是一个开源、可自托管、可扩展的 LLM 应用工程平台。它的核心目标不是单独解决某一个点，而是把 observability、prompt management、evaluation 等能力原生集成在一起，让团队能更快完成从调试到上线再到持续改进的闭环。

### 面向的问题

LLM 应用和传统软件不同：

- 输出非确定性，线上问题难以复现。
- 一次用户请求可能包含检索、embedding、工具调用、多个 LLM 调用。
- prompt 修改对质量、成本、延迟的影响需要长期追踪。
- 生产环境需要持续发现低质量回答、安全问题和成本异常。
- 团队需要协作管理 prompts、traces、evaluations 和 datasets。

Langfuse 通过统一平台把这些信息连起来。

### 平台特点

| 特点 | 说明 |
|------|------|
| Open source | 开源，支持通过 public API 扩展 |
| Self-hostable | 可自托管，适合有数据控制要求的团队 |
| Extensible | 支持自定义集成和 custom scores |
| SDK 支持 | 提供 Python / JavaScript 原生 SDK |
| Framework support | 集成 OpenAI SDK、LangChain、LlamaIndex 等常见框架 |
| OpenTelemetry based | 基于 OpenTelemetry，提升兼容性并降低 vendor lock-in |
| Multi-modal | 支持文本、图像等多模态 tracing |

---

## 核心能力总览

### 三大主线

| 能力 | 解决什么问题 | 典型产物 |
|------|--------------|----------|
| Observability | 看清线上 LLM 应用发生了什么 | traces、sessions、users、dashboards |
| Prompt Management | 管理 prompt 生命周期 | prompt versions、labels、playground、metrics |
| Evaluation | 系统化衡量质量 | scores、datasets、experiments、annotation queues |

### 生命周期视角

从 PoC 到生产，Langfuse 能逐步引入：

1. **PoC 阶段**：先接入 tracing，看清输入、输出、成本、延迟。
2. **开发阶段**：把 prompt 从代码里抽出，进行版本管理和 playground 测试。
3. **测试阶段**：建立 datasets，运行 experiments，对比 prompt/model 版本。
4. **上线阶段**：在生产 traces 上运行 evaluations，发现质量问题。
5. **迭代阶段**：将用户反馈和低分 traces 回流到数据集，持续优化。

---

## Observability

Observability 是 Langfuse 的基础能力，用于理解和调试 LLM 应用。LLM 应用通常包含复杂的非确定性调用链，只看最终输出很难定位问题。Langfuse 通过 tracing 展示每一步发生了什么。

### Trace 能捕获什么

Langfuse traces 可以包含：

- LLM calls。
- non-LLM calls。
- retrieval。
- embedding。
- API calls。
- tool calls。
- 复杂 agent workflow。
- 输入、输出、耗时、成本和评分。

### 关键功能

| 功能 | 说明 |
|------|------|
| Trace Details | 查看一次请求中的每个 LLM 调用和业务步骤 |
| Sessions | 跟踪多轮对话或多步骤 agentic workflow |
| Timeline | 分析延迟问题，定位耗时步骤 |
| Users | 按 `userId` 监控每个用户的成本和使用情况 |
| Agent Graphs | 将复杂 Agent 运行流程展示为图 |
| Dashboard | 查看质量、成本、延迟等整体指标 |

### 接入方式

Langfuse 支持多种 trace 捕获方式：

- Python / JS native SDKs。
- 100+ library / framework integrations。
- OpenTelemetry。
- LLM Gateway，例如 LiteLLM。

### 为什么 OpenTelemetry 重要

Langfuse 基于 OpenTelemetry，有两个明显好处：

- 更容易与已有 observability 体系兼容。
- 降低 vendor lock-in，避免 tracing 数据被某个专有协议锁死。

---

## Prompt Management

Prompt Management 用于把 prompt 从代码中解耦出来，并进行版本管理、测试、部署和效果追踪。

### 核心能力

| 功能 | 说明 |
|------|------|
| Create | 通过 UI、SDK 或 API 创建 prompt |
| Version Control | 协作编辑、版本化管理 prompt |
| Deploy | 通过 labels 将 prompt 部署到生产或其他环境，无需改代码 |
| Metrics | 比较不同 prompt 版本的延迟、成本、评估指标 |
| Playground | 在 LLM Playground 中快速测试 prompt |
| Link with Traces | 将 prompt 和 traces 关联，观察生产表现 |
| Track Changes | 查看 prompt 如何随时间演化 |

### 典型价值

| 问题 | Langfuse Prompt Management 的作用 |
|------|-----------------------------------|
| prompt 写死在代码里 | 用平台统一管理 prompt |
| prompt 修改不可追踪 | 通过版本记录变更历史 |
| 不同环境 prompt 不一致 | 用 labels 部署到不同环境 |
| 不知道新 prompt 是否更好 | 对 datasets 跑 experiments |
| 线上某条 trace 不知道用了哪个 prompt | prompt 与 trace 关联 |

### Prompt Experiments

Langfuse 支持在 datasets 上直接运行 experiments，对比新 prompt 版本的表现。这让 prompt 优化从“凭感觉改”变成“基于数据比较”。

---

## Evaluation

Evaluation 用于系统化衡量 LLM 应用质量。Langfuse 的评估能力既支持开发阶段的离线测试，也支持生产阶段的线上质量监控。

### 评估方法

Langfuse 支持多种 evaluation 方法：

| 方法 | 说明 |
|------|------|
| LLM-as-a-judge | 用 LLM 对回答质量、相关性、忠实度等打分 |
| Code evaluators | 用代码规则做确定性评估 |
| User feedback | 收集用户 👍/👎 等反馈 |
| Manual labeling | 人工标注 traces 或 outputs |
| Custom pipelines | 自定义外部评估流程并回写结果 |
| Custom scores | 写入 numeric、boolean、categorical 分数 |

### 生产 Traces 上的评估

文档强调可以直接对 production traces 运行 evaluations，用于尽早发现问题。

适合监控：

- 低质量回答。
- 不相关回答。
- 成本异常。
- 延迟异常。
- 安全风险。
- 工具调用失败。
- 用户负反馈样本。

### Datasets 与 Experiments

Langfuse 支持创建和管理 datasets，用于开发阶段系统测试。

典型流程：

1. 建立 dataset，覆盖关键场景。
2. 选择 prompt / model / application version。
3. 运行 experiment。
4. 查看 scores。
5. 比较不同版本。

### Scores

Langfuse 的 score 可以来自：

- 用户反馈。
- LLM-as-a-judge。
- 代码 evaluator。
- 人工标注。
- 自定义 API / SDK 写入。

自定义 score 示例：

```python
langfuse.score(
    trace_id="123",
    name="my_custom_evaluator",
    value=0.5,
)
```

也可以通过 API 写入：

```bash
POST /api/public/scores
```

### Evaluation UI 能力

| 功能 | 说明 |
|------|------|
| Analytics | 在 Dashboard 中绘制 evaluation results |
| User Feedback | 从前端、服务端 SDK 或 API 收集反馈 |
| LLM-as-a-Judge | 托管式 judge，可应用到应用内任意步骤 |
| Experiments | 在 UI 中对 datasets 测试 prompt 和 model |
| Annotation Queue | 用人工标注建立评估基线 |
| Custom Evals | 支持 numeric、boolean、categorical 评分 |

---

## Example Project

Langfuse Example Project 是一个 live shared project，让用户在正式接入自己的应用前，通过真实数据体验 Langfuse 的功能。

### Example Project 是什么

它可以理解为一个“可浏览的真实示例项目”：

- 提供 view-only access。
- 不需要信用卡。
- 展示真实 chatbot 交互产生的 traces。
- 可以观察用户反馈、scores、prompt、datasets、sessions 等功能。
- 所有用户共享同一个示例项目，所以你会看到其他用户产生的 traces。

### 进入方式

1. 创建免费账号。
2. 打开 Example Project。
3. 默认进入 **Traces** 页面。

### Traces 页面怎么看

打开 Example Project 后，第一眼看到的是 Traces 页面。

每一行代表一次 example chatbot 的交互，通常包括：

- timing。
- costs。
- input / output。
- evaluation scores。
- user feedback。
- 详细执行步骤。

建议操作：

1. 点击任意 trace 查看详情。
2. 查看 graph view，理解 chatbot 组件如何协作。
3. 找到带 scores 的 traces，观察 evaluation 如何工作。

### 可探索的功能区

| 功能区 | 可以看什么 |
|--------|------------|
| Tracing | 单次请求的调用链、输入输出、成本、耗时 |
| Sessions | 多轮对话或多步骤 workflow |
| Prompts | prompt 管理和版本 |
| Scores | 用户反馈和评估分数 |
| Datasets | 用于 experiments 的测试数据 |

### Interactive Demos

Example Project 中的 traces 来自交互式 demos。每次你与 demo app 交互，都会生成新的 trace，可以在 Langfuse 中检查。

文档还提到这些 demo apps 是开源的，可查看 Q&A chatbot 的构建博客了解实现细节。

### Example Project 的价值

| 价值 | 说明 |
|------|------|
| 无需搭建即可体验 | 注册后直接查看真实 traces |
| 学习产品结构 | 通过左侧导航理解 Langfuse 各模块 |
| 观察真实数据 | 能看到多用户、多场景产生的 traces |
| 理解 evaluation | 找带 score 的 traces 查看打分流程 |
| 为接入做准备 | 先知道自己项目接入后会看到什么 |

---

## 从哪里开始

Langfuse 官方建议根据当前需求选择 quickstart。

### 1. 先接入 Tracing

适合：

- 你已经有 LLM 应用。
- 想知道线上请求发生了什么。
- 需要调试成本、延迟、错误和中间步骤。

入口：

- Get Started with Tracing。

### 2. 再接入 Prompt Management

适合：

- prompt 经常修改。
- 多人协作调 prompt。
- 需要版本控制和环境发布。
- 希望不改代码就切换 prompt 版本。

入口：

- Set Up Prompt Management。

### 3. 然后建立 Evaluations

适合：

- 想系统衡量质量。
- 想对 production traces 做在线评估。
- 想用 datasets 做开发阶段实验。
- 想收集用户反馈和人工标注。

入口：

- Create Your First Evaluation。

### 推荐落地顺序

```
┌────────────┐
│  Tracing   │  先看清应用运行
└─────┬──────┘
      ▼
┌────────────┐
│  Prompts   │  再管理可迭代资产
└─────┬──────┘
      ▼
┌────────────┐
│ Evaluation │  最后系统化衡量质量
└─────┬──────┘
      ▼
┌────────────┐
│ Iteration  │  用数据驱动改进
└────────────┘
```

---

## 为什么选择 Langfuse

文档列出的核心理由：

| 理由 | 说明 |
|------|------|
| Open source | 完全开源，支持自定义集成 |
| Production optimized | 低性能开销，面向生产使用 |
| Best-in-class SDKs | 提供 Python 和 JavaScript SDK |
| Framework support | 支持 OpenAI SDK、LangChain、LlamaIndex 等 |
| Multi-modal | 支持文本、图像和其他模态 |
| Full platform | 覆盖完整 LLM 应用开发生命周期 |

---

## 实践建议

### 初次了解 Langfuse

- 先进入 Example Project，不急着接入代码。
- 从 Traces 页面开始，点开几条 trace 看调用链。
- 找有 scores 的 trace，理解 evaluation 和 feedback 如何展示。
- 浏览 Prompts、Datasets、Sessions，建立平台全局认知。

### 接入自己的应用

| 阶段 | 建议 |
|------|------|
| 第一天 | 先接入 tracing，确保每次 LLM 调用和关键业务步骤可见 |
| 第一周 | 增加 userId、session、cost、latency 追踪 |
| Prompt 稳定前 | 将 prompt 纳入 Prompt Management |
| 准备上线 | 建立 datasets 和 experiments |
| 上线后 | 对 production traces 设置 evaluations 和反馈采集 |

### Evaluation 建议

- 先收集用户反馈，建立最真实的问题来源。
- 对 production traces 跑 LLM-as-a-judge，自动发现问题。
- 对关键任务建立 datasets，做离线 experiments。
- 用 annotation queues 做人工校准，不要完全依赖自动评估。
- 将 custom scores 回写 Langfuse，统一展示质量指标。

### Prompt Management 建议

- prompt 不要长期散落在代码里。
- 每次 prompt 修改都应保留版本记录。
- 用 labels 控制部署环境，例如 dev、staging、production。
- 把 prompt 与 traces 关联，观察不同版本真实表现。
- 在 datasets 上跑 prompt experiments，再决定是否发布。

---

## 快速参考

### 核心术语

| 术语 | 含义 |
|------|------|
| Trace | 一次 LLM 应用执行记录，包含调用链、输入输出、成本、延迟 |
| Session | 多轮对话或多步骤 workflow 的集合 |
| User | 按用户维度聚合使用量、成本和行为 |
| Prompt | 可版本化、可部署、可测试的提示词资产 |
| Score | 用户反馈、自动评估或自定义 evaluator 写入的分数 |
| Dataset | 用于 experiments 的测试数据 |
| Experiment | 在 dataset 上测试 prompt/model/application 版本 |
| Annotation Queue | 人工标注和审核工作流 |

### Langfuse 能力速查

| 需求 | 使用能力 |
|------|----------|
| 看清线上调用链 | Observability / Tracing |
| 分析多轮对话 | Sessions |
| 按用户看成本 | Users + userId |
| 分析慢请求 | Timeline |
| 可视化 Agent 流程 | Agent Graphs |
| 管理 prompt 版本 | Prompt Management |
| 测试 prompt | Playground |
| 对比 prompt 版本 | Experiments |
| 收集用户反馈 | User Feedback / Scores |
| 自动评估回答质量 | LLM-as-a-Judge |
| 人工标注 | Annotation Queue |
| 自定义指标 | Custom Scores / API |

### Example Project 使用速查

| 步骤 | 做什么 |
|------|--------|
| 1 | 创建免费账号 |
| 2 | 打开 Example Project |
| 3 | 从 Traces 页面点开一条 trace |
| 4 | 查看 graph view 和执行步骤 |
| 5 | 找有 scores 的 trace 看 evaluation |
| 6 | 浏览 Sessions、Prompts、Scores、Datasets |
| 7 | 决定自己的项目先接入 tracing、prompt management 还是 evaluation |

### 一句话总结

- Langfuse 是开源、可自托管的 LLM 应用工程平台。
- Observability 帮你看清应用发生了什么。
- Prompt Management 帮你管理、测试和部署 prompt。
- Evaluation 帮你系统衡量质量并持续改进。
- Example Project 是理解 Langfuse 最快的入口。
