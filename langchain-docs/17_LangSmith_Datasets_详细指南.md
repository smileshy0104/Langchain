# LangSmith Datasets（数据集）详细指南

> 基于官方文档 Create and manage datasets in the UI、How to create and manage datasets programmatically、Manage datasets、Custom output rendering 整理的中文增强版。本文聚焦 LangSmith Datasets 的创建、导入、样例管理、schema、splits、metadata、版本、共享、导出、SDK 操作，以及自定义输出渲染。

## 目录

1. [概述](#概述)
2. [Dataset 与 Example 核心概念](#dataset-与-example-核心概念)
3. [通过 UI 创建数据集](#通过-ui-创建数据集)
4. [通过 SDK 创建数据集](#通过-sdk-创建数据集)
5. [获取与过滤数据集](#获取与过滤数据集)
6. [管理 Examples](#管理-examples)
7. [Schema 与 Transformations](#schema-与-transformations)
8. [Splits 与 Metadata](#splits-与-metadata)
9. [Dataset Versioning](#dataset-versioning)
10. [按版本、Split、过滤视图评估](#按版本split过滤视图评估)
11. [共享与导出](#共享与导出)
12. [从实验结果回流到数据集](#从实验结果回流到数据集)
13. [自定义输出渲染](#自定义输出渲染)
14. [最佳实践](#最佳实践)
15. [快速参考](#快速参考)

---

## 概述

LangSmith Datasets 用于保存可重复评估的数据。它让你可以在不同时间、不同模型、不同 prompt 或不同应用版本上，使用同一批 examples 进行稳定比较。

```
┌─────────────────────────────────────────────────────────────┐
│                    LangSmith Datasets                        │
│                                                             │
│  Dataset                                                     │
│  ├─ Example 1                                                │
│  │  ├─ inputs                                                │
│  │  ├─ outputs / reference outputs                           │
│  │  └─ metadata                                              │
│  ├─ Example 2                                                │
│  └─ Example N                                                │
│                                                             │
│  用途：                                                       │
│  ├─ 离线评估：对固定样本集做回归测试和版本比较                │
│  ├─ 数据沉淀：把生产 traces、人工标注、实验结果变成样例        │
│  ├─ 质量管理：用 schema、splits、metadata、versions 管理数据   │
│  └─ 团队协作：共享、导出、自定义渲染、人工审核                 │
└─────────────────────────────────────────────────────────────┘
```

### Datasets 解决什么问题

| 问题 | Dataset 的作用 |
|------|----------------|
| LLM 输出非确定，难以判断新版本是否更好 | 用固定 examples 做重复评估 |
| 线上问题难以复现 | 将 traces 转成 dataset examples |
| 人工标注结果容易散落 | 通过 annotation queues 回流到 dataset |
| 样例结构不一致 | 使用 JSON schema 和 transformations |
| CI 中数据集变化导致结果不可控 | 使用 dataset versions / tags 固定版本 |
| 实验结果需要继续分析 | 将 filtered traces 从 experiment 导出回 dataset |

---

## Dataset 与 Example 核心概念

### Dataset

Dataset 是一组 examples 的集合，主要用于离线评估。你可以把它理解为 LLM 应用的测试集、回归集或 benchmark 集。

常见 dataset 来源：

- 手工整理的高质量样例。
- 从 tracing project 中挑选的真实 runs。
- annotation queue 中专家审核后的样例。
- Playground 中创建的 prompt 调试样例。
- CSV / JSONL / DataFrame 文件导入。
- LLM 生成的 synthetic examples。

### Example

Example 是 dataset 中的单条样本。

| 字段 | 说明 |
|------|------|
| `inputs` | 传给应用的输入，例如问题、消息、工具列表 |
| `outputs` / `reference outputs` | 参考输出，通常用于 evaluator，不会传给应用 |
| `metadata` | 附加信息，例如来源、难度、标签、版本、tenant_id |
| `split` / `splits` | 数据集分组，例如 train、validation、test |

### Dataset 在评估中的位置

```
Dataset examples
      │
      ▼
Target app / chain / agent
      │
      ▼
Actual runs / outputs
      │
      ▼
Evaluator 对比 actual outputs 与 reference outputs
      │
      ▼
Experiment results
```

---

## 通过 UI 创建数据集

LangSmith UI 支持多种创建 dataset 和添加 examples 的方式。选择哪一种取决于你的数据来源和工作流。

### 从 Tracing Project 手动添加

这是构建真实评估集的常见方式：从生产或测试 traces 中挑选有价值的 runs，转成 dataset examples。

前提：应用已经配置 LangSmith tracing。

两种入口：

1. 进入 **Tracing Projects**，选择项目，在 **Runs** 表格中多选 runs，点击 **Add to Dataset**。
2. 在某个 run 的详情页，点击右上角 **Add to -> Dataset**。

适合挑选：

- 带有负面用户反馈的 traces。
- 延迟高或报错的 traces。
- 业务关键路径 traces。
- 评估低分或人工认为有代表性的 traces。

添加时如果配置了 transformations 或 schema，UI 会提示转换结果或 schema 校验失败。你也可以在加入 dataset 前编辑 run 内容。

### 从 Tracing Project 自动添加

可以使用 run rules / automation rules，根据条件自动把 traces 加入 dataset。

适合条件：

- 包含特定 tag。
- 用户反馈分数较低。
- run 属于某个 use case。
- evaluator 标记为低质量。

这种方式适合持续收集线上样本，但建议配合人工审核或后续清洗，避免 dataset 被噪声污染。

### 从 Annotation Queue 添加

如果 dataset 需要领域专家参与构建，annotation queues 是更好的入口。

流程：

1. 将 runs 分配到 annotation queue。
2. 审核人员查看 run、修改 inputs / outputs / reference outputs。
3. 选择目标 dataset。
4. 点击 **Add to Dataset**，或使用快捷键 `D`。

特点：

- 修改后的内容会带入 dataset。
- run 的 metadata 也会复制到 dataset example。
- annotation queue 可以设置默认 dataset，也可以通过底部 switcher 切换目标 dataset。
- 可以通过 rules 自动把符合条件的 runs 加入 annotation queue。

### 从 Playground 创建

Playground 适合 prompt 调试阶段快速建立评估数据。

步骤：

1. 在 **Playground** 页面选择 **Set up Evaluation**。
2. 点击 **+New** 创建新 dataset，或选择已有 dataset。
3. 使用 **+Row** 添加样例。
4. 通过右侧 **⋮** 删除样例。
5. 如果是 reference-free dataset，可以删除 **Reference Output** 列。

注意：Playground 内联创建不支持 nested keys 的 dataset。如果 examples 有嵌套字段，需要到 datasets 页面编辑。

### 从 CSV 或 JSONL 导入

在 **Datasets & Experiments** 页面：

1. 点击 **+New Dataset**。
2. 选择 **Import**。
3. 上传 CSV 或 JSONL 文件。

适合已有标注文件、测试集或从其他系统导出的数据。

### 从 Datasets & Experiments 页面从零创建

步骤：

1. 进入 **Datasets & Experiments**。
2. 点击 **+ New Dataset**。
3. 在 **New Dataset** 页面选择 **Create from scratch**。
4. 填写 name 和 description。
5. 可选：创建 dataset schema。
6. 点击 **Create**。
7. 进入 dataset 的 **Examples** tab，点击 **+ Example**。
8. 用 JSON 定义 example 后提交。

### 添加 LLM 生成的 Synthetic Examples

如果 dataset 已有 examples 且定义了 schema，可以使用 UI 里的 **Add AI-Generated Examples** 创建合成样例。

流程：

1. 在 **+ Example** 中选择 **Add AI-Generated Examples**。
2. 设置 OpenAI API key 为 workspace secret；如果 workspace 已配置则跳过。
3. 选择 few-shot examples，可自动选择或手动选择。
4. 输入要生成的 synthetic examples 数量。
5. 点击 **Generate**。
6. 在生成结果页面选择要加入的数据，可先编辑再保存。
7. 每条样例会按 schema 校验，并在 source metadata 中标记为 `synthetic`。

使用建议：

- Synthetic examples 适合扩充覆盖面，不适合单独作为质量标准。
- 建议先有少量人工高质量 examples，再用它们作为 few-shot 模板。
- 生成后要抽样人工检查。

---

## 通过 SDK 创建数据集

LangSmith 支持 Python、TypeScript、Java SDK 管理 datasets。下面以 Python / TypeScript 的核心模式为主整理。

### 从列表创建 Dataset

最灵活的方式是准备一组 examples，每条包含 inputs、outputs 和 metadata。

Python 示例：

```python
from langsmith import Client

client = Client()

examples = [
    {
        "inputs": {"question": "What is the largest mammal?"},
        "outputs": {"answer": "The blue whale"},
        "metadata": {"source": "Wikipedia"},
    },
    {
        "inputs": {"question": "What do mammals and birds have in common?"},
        "outputs": {"answer": "They are both warm-blooded"},
        "metadata": {"source": "Wikipedia"},
    },
]

dataset = client.create_dataset(
    dataset_name="Elementary Animal Questions",
    description="Questions and answers about animal phylogenetics.",
)

client.create_examples(
    dataset_id=dataset.id,
    examples=examples,
)
```

TypeScript 示例：

```typescript
import { Client } from "langsmith";

const client = new Client();

const dataset = await client.createDataset("Elementary Animal Questions", {
  description: "Questions and answers about animal phylogenetics",
});

await client.createExamples({
  inputs: [{ question: "What is the largest mammal?" }],
  outputs: [{ answer: "The blue whale" }],
  metadata: [{ source: "Wikipedia" }],
  datasetId: dataset.id,
});
```

建议：

- 多条样例使用 `create_examples` / `createExamples` 批量创建。
- 单条样例才使用 `create_example` / `createExample`。
- metadata 建议从一开始就设计好，后续分析和过滤会很有用。

### 从 Traces 创建 Dataset

可以先通过 SDK 过滤 runs，再把 run inputs / outputs 转成 examples。

Python 示例：

```python
from langsmith import Client

client = Client()

runs = client.list_runs(
    project_name="my_project",
    is_root=True,
    error=False,
)

dataset = client.create_dataset(
    "Example Dataset",
    description="An example dataset",
)

examples = [
    {"inputs": run.inputs, "outputs": run.outputs}
    for run in runs
]

client.create_examples(
    dataset_id=dataset.id,
    examples=examples,
)
```

适合场景：

- 将生产 traces 批量沉淀为回归集。
- 按 project、error、tag、feedback 等条件筛选 runs。
- 将实验 traces 转成后续测试数据。

### 从 CSV 创建 Dataset

CSV 文件需要有明确列名，用于映射 input keys 和 output keys。

Python 示例：

```python
from langsmith import Client

client = Client()

dataset = client.upload_csv(
    csv_file="path/to/your/csvfile.csv",
    input_keys=["column1", "column2"],
    output_keys=["output1", "output2"],
    name="My CSV Dataset",
    description="Dataset created from a CSV file",
    data_type="kv",
)
```

TypeScript 示例：

```typescript
import { Client } from "langsmith";

const client = new Client();

const dataset = await client.uploadCsv({
  csvFile: "path/to/your/csvfile.csv",
  fileName: "My CSV Dataset",
  inputKeys: ["column1", "column2"],
  outputKeys: ["output1", "output2"],
  description: "Dataset created from a CSV file",
  dataType: "kv",
});
```

### 从 pandas DataFrame 创建 Dataset

Python SDK 支持直接上传 DataFrame，适合从 Parquet 或分析环境中导入。

```python
from langsmith import Client
import pandas as pd

client = Client()

df = pd.read_parquet("path/to/your/myfile.parquet")

dataset = client.upload_dataframe(
    df=df,
    input_keys=["column1", "column2"],
    output_keys=["output1", "output2"],
    name="My Parquet Dataset",
    description="Dataset created from a parquet file",
    data_type="kv",
)
```

---

## 获取与过滤数据集

### 查询所有 Datasets

```python
from langsmith import Client

client = Client()
datasets = client.list_datasets()
```

```typescript
import { Client } from "langsmith";

const client = new Client();
const datasets = await client.listDatasets();
```

### 按精确名称查询

```python
datasets = client.list_datasets(dataset_name="My Test Dataset 1")
```

```typescript
const datasets = await client.listDatasets({
  datasetName: "My Test Dataset 1",
});
```

### 按名称子串查询

适合大小写不敏感的模糊搜索。

```python
datasets = client.list_datasets(dataset_name_contains="some substring")
```

```typescript
const datasets = await client.listDatasets({
  datasetNameContains: "some substring",
});
```

### 按 Data Type 查询

```python
datasets = client.list_datasets(data_type="kv")
```

```typescript
const datasets = await client.listDatasets({
  dataType: "kv",
});
```

---

## 管理 Examples

### 获取 Dataset 中全部 Examples

可以通过 dataset ID 或 dataset name 获取 examples。

```python
examples = client.list_examples(
    dataset_id="c9ace0d8-a82c-4b6c-13d2-83401d68e9ab"
)
```

```python
examples = client.list_examples(
    dataset_name="My Test Dataset"
)
```

```typescript
const examples = await client.listExamples({
  datasetName: "My Test Dataset",
});
```

### 按 Example IDs 获取

```python
examples = client.list_examples(
    example_ids=[
        "734fc6a0-c187-4266-9721-90b7a025751a",
        "d6b4c1b9-6160-4d63-9b61-b034c585074f",
    ]
)
```

### 按 Metadata 获取

如果 example metadata 包含指定 key-value，就会匹配。

```python
examples = client.list_examples(
    dataset_name=dataset_name,
    metadata={"foo": "bar"},
)
```

```typescript
const examples = await client.listExamples({
  datasetName,
  metadata: { foo: "bar" },
});
```

如果 metadata 是 `{"foo": "bar", "baz": "qux"}`，那么以下条件都能匹配：

- `{"foo": "bar"}`
- `{"baz": "qux"}`
- `{"foo": "bar", "baz": "qux"}`

### 使用 Structured Filter 查询 Examples

SDK 支持类似 runs 查询的 structured filter language，但仅支持 metadata 字段。

要求：

- Python SDK v0.1.83+
- TypeScript SDK v0.1.35+

示例：

```python
examples = client.list_examples(
    dataset_name=dataset_name,
    filter="and(not(has(metadata, '{\"foo\": \"bar\"}')), exists(metadata, \"tenant_id\"))",
)
```

含义：

- 排除 metadata 中包含 `{"foo": "bar"}` 的样例。
- 同时要求 metadata 中存在 `tenant_id`。

可用操作：

| 操作 | 说明 |
|------|------|
| `has` | metadata 是否包含指定 key/value |
| `exists` | metadata 是否存在指定 key |
| `and` | 多个条件同时成立 |
| `not` | 取反 |

### 更新单个 Example

```python
client.update_example(
    example_id=example.id,
    inputs={"input": "updated input"},
    outputs={"output": "updated output"},
    metadata={"foo": "bar"},
    split="train",
)
```

```typescript
await client.updateExample(example.id, {
  inputs: { input: "updated input" },
  outputs: { output: "updated output" },
  metadata: { foo: "bar" },
  split: "train",
});
```

### 批量更新 Examples

```python
client.update_examples(
    example_ids=[example.id, example_2.id],
    inputs=[
        {"input": "updated input 1"},
        {"input": "updated input 2"},
    ],
    outputs=[
        {"output": "updated output 1"},
        {"output": "updated output 2"},
    ],
    metadata=[
        {"foo": "baz"},
        {"foo": "qux"},
    ],
    splits=[
        ["training", "foo"],
        "training",
    ],
)
```

注意：splits 可以是数组，也可以是单个字符串。

---

## Schema 与 Transformations

### Dataset Schema

LangSmith datasets 可以存储任意 JSON 对象，但官方建议定义 schema，以保证 examples 的结构一致。

Schema 使用标准 JSON Schema，并额外支持一些 prebuilt types，方便表示常见结构，例如 messages 和 tools。

适合定义 schema 的场景：

- Dataset 用于 CI 或长期回归。
- Examples 由多人维护。
- Inputs / outputs 有复杂嵌套结构。
- 需要用 LLM 生成 synthetic examples。
- 需要统一模型消息格式。

### Transformations

某些 schema 字段支持 **+ Transformations**。Transformations 是添加 examples 时自动执行的预处理步骤。

典型例子：

| Transformation | 作用 |
|----------------|------|
| Convert to OpenAI messages | 将 LangChain messages 等 message-like objects 转成 OpenAI message format |

如果你计划从 LangChain ChatModels 或 LangSmith OpenAI wrapper 收集生产 traces，官方建议使用预置 Chat Model schema。它可以把 messages 和 tools 转换为通用 OpenAI 格式，方便后续用任意模型进行测试。

### Schema 与 Synthetic Examples 的关系

AI-generated examples 依赖 dataset schema：

- Schema 告诉 LLM 应该生成什么结构。
- 生成后的样例会按 schema 校验。
- 不符合 schema 的样例需要修改或丢弃。

---

## Splits 与 Metadata

### Splits

Splits 是 dataset 的命名子集，用于把 examples 分成不同评估集合。

常见 splits：

| Split | 用途 |
-------|------|
| `train` | 用于 prompt / evaluator 调优 |
| `validation` | 用于开发中比较方案 |
| `test` | 用于发布前稳定回归 |
| `edge_cases` | 边界样例 |
| `prod_failures` | 线上失败样例 |

### UI 中管理 Splits

步骤：

1. 在 dataset 中选择 examples。
2. 点击 **Add to Split**。
3. 在弹窗中选择、取消选择或创建 split。

LangSmith 允许一个 example 属于多个 splits。机器学习场景通常建议一个样例只属于一个 split，但评估场景中多 split 对交叉标签很有用。

### Metadata

Metadata 用于给每条 example 添加细粒度信息。

常见 metadata：

| 字段 | 示例 |
|------|------|
| `source` | `production_trace`、`manual`、`synthetic` |
| `difficulty` | `easy`、`medium`、`hard` |
| `topic` | `billing`、`retrieval`、`agent_tool` |
| `tenant_id` | 多租户标识 |
| `version` | 样例来源版本 |
| `reviewer` | 标注人 |

### UI 中编辑 Metadata

步骤：

1. 点击某个 example。
2. 在弹窗右上角点击 **Edit**。
3. 更新、删除或新增 metadata。

Metadata 的作用：

- 在 experiment 分析时 group by metadata。
- 在 SDK 中通过 `list_examples` 过滤。
- 区分 synthetic / human / production 来源。
- 追踪样例质量和业务分布。

### UI 中过滤 Examples

Examples 表格支持：

| 过滤方式 | 说明 |
|----------|------|
| Filter by split | 选择某个 split |
| Filter by metadata | 按 metadata key/value 过滤 |
| Full-text search | 对 examples 做全文搜索 |

可以叠加多个过滤条件，只有同时满足所有条件的 examples 才会展示。

---

## Dataset Versioning

LangSmith datasets 是自动版本化的。每当你添加、更新或删除 examples，都会创建一个新版本。

### 版本如何产生

任何会改变 dataset 内容的操作都会产生新版本：

- 添加 example。
- 修改 example。
- 删除 example。
- 批量更新 examples。

默认版本由变更时间戳定义。进入 dataset 的 **Examples** tab 后，可以点击某个时间点版本，查看该时刻的 dataset 状态。

### 查看历史版本

历史版本中的 examples 是只读的。UI 也会展示该版本与最新版本之间发生过哪些操作。

默认行为：

- **Examples** tab 默认展示最新版本。
- **Tests** tab 会展示不同版本 dataset 上运行过的实验结果。

### 给版本打 Tag

可以给重要 dataset version 打语义化标签，例如：

- `prod`
- `release-2026-06`
- `ci-stable`
- `baseline-v1`

UI 方式：在 **Examples** tab 中点击 **+ Tag this version**。

Python SDK 示例：

```python
from langsmith import Client
from datetime import datetime

client = Client()

initial_time = datetime(2024, 1, 1, 0, 0, 0)

client.update_dataset_tag(
    dataset_name=toxic_dataset_name,
    as_of=initial_time,
    tag="prod",
)
```

### 为什么版本很重要

| 场景 | 版本价值 |
|------|----------|
| CI 回归 | 固定 `prod` 或 `ci-stable`，避免数据变化影响结果 |
| 实验对比 | 明确每次实验使用的数据集状态 |
| 数据治理 | 追踪样例增删改历史 |
| 发布管理 | 将某个版本标记为 release baseline |

---

## 按版本、Split、过滤视图评估

LangSmith 支持把 `list_examples` 返回的可迭代 examples 直接传给 `evaluate` / `aevaluate`。

### 按特定 Dataset Version 评估

使用 `as_of` / `asOf` 指定版本。

Python 示例：

```python
from langsmith import Client

ls_client = Client()

def correct(outputs: dict, reference_outputs: dict) -> bool:
    return outputs["class"] == reference_outputs["label"]

results = ls_client.evaluate(
    lambda inputs: {"class": "Not toxic"},
    data=ls_client.list_examples(
        dataset_name="Toxic Queries",
        as_of="latest",
    ),
    evaluators=[correct],
)
```

TypeScript 示例：

```typescript
import { evaluate } from "langsmith/evaluation";

await evaluate((inputs) => labelText(inputs["input"]), {
  data: langsmith.listExamples({
    datasetName,
    asOf: "latest",
  }),
  evaluators: [correctLabel],
});
```

`as_of` 可以使用：

- `latest`
- 具体时间戳
- 已打 tag 的版本名

### 按 Metadata 过滤视图评估

```python
from langsmith import evaluate

results = evaluate(
    lambda inputs: label_text(inputs["text"]),
    data=client.list_examples(
        dataset_name=dataset_name,
        metadata={"desired_key": "desired_value"},
    ),
    evaluators=[correct_label],
    experiment_prefix="Toxic Queries",
)
```

适合：

- 只评估某个业务类型。
- 只评估某个来源。
- 只评估 hard examples。
- 只评估某个 tenant 的样例。

### 按 Dataset Split 评估

```python
from langsmith import evaluate

results = evaluate(
    lambda inputs: label_text(inputs["text"]),
    data=client.list_examples(
        dataset_name=dataset_name,
        splits=["test", "training"],
    ),
    evaluators=[correct_label],
    experiment_prefix="Toxic Queries",
)
```

TypeScript 示例：

```typescript
import { evaluate } from "langsmith/evaluation";

await evaluate((inputs) => labelText(inputs["input"]), {
  data: langsmith.listExamples({
    datasetName,
    splits: ["test", "training"],
  }),
  evaluators: [correctLabel],
  experimentPrefix: "Toxic Queries",
});
```

---

## 共享与导出

### 公开共享 Dataset

可以从 **Dataset & Experiments** tab 中选择 dataset，点击右上角 **⋮**，选择 **Share Dataset**，复制共享链接。

重要警告：

- 公开共享会让任何拥有链接的人访问 dataset examples。
- 相关 experiments、associated runs、feedback 也会被公开。
- 即使访问者没有 LangSmith 账号也可以访问。
- 该功能仅适用于 cloud-hosted LangSmith。
- 共享前必须确认没有敏感数据。

### 取消共享

两种方式：

1. 在公开 dataset 页面右上角点击 **Public -> Unshare**。
2. 进入 **Settings -> Shared URLs**，找到对应 dataset 后点击 **Unshare**。

### 导出 Dataset

可以从 UI 导出为：

- CSV
- JSONL
- OpenAI fine-tuning format

路径：

1. 进入 **Dataset & Experiments**。
2. 选择 dataset。
3. 点击右上角 **⋮**。
4. 点击 **Download Dataset**。

适合用途：

- 本地备份。
- 迁移到其他环境。
- 进入数据分析流程。
- 准备 fine-tuning 数据。

---

## 从实验结果回流到数据集

离线评估后，经常需要把符合某些条件的 experiment traces 导出回 dataset，用于继续分析和迭代。

### 典型流程

1. 在 experiment 页面点击 experiment 名称旁的箭头。
2. 进入包含该 experiment traces 的 tracing project。
3. 按 evaluation criteria 过滤 traces。
4. 例如筛选 accuracy score 大于 0.5 或小于某阈值的 traces。
5. 多选 filtered runs。
6. 点击 **Add to Dataset**。

### 使用场景

| 场景 | 做法 |
|------|------|
| 收集失败案例 | 过滤低分 traces，加入 `prod_failures` 或 `regression` dataset |
| 构建 hard set | 过滤模型表现差但业务重要的样例 |
| 分析优秀样例 | 过滤高分 traces，作为 few-shot 或参考样例 |
| 比较实验差异 | 将某个版本特有失败样本加入后续评估 |

---

## 自定义输出渲染

Custom output rendering 允许用自定义 HTML 页面展示 run outputs 和 dataset reference outputs。它适合输出不是普通文本，而是更需要可视化或领域格式展示的场景。

### 适用场景

| 场景 | 示例 |
|------|------|
| 领域格式展示 | 医疗记录、法律文书、金融报告 |
| 结构化结果可视化 | 表格、图表、指标面板 |
| 图形化输出 | 流程图、diagram、地图、标注结果 |
| 人工评审体验优化 | annotation queue 中更直观地审核输出 |

### 配置层级

Custom rendering 可以配置在三个层级：

| 层级 | 影响范围 |
|------|----------|
| Tracing project | 该 tracing project 下的 runs |
| Dataset | 与该 dataset 相关的 runs，在 experiments、run detail panes、annotation queues 中生效 |
| Annotation queue | 该 queue 内所有 runs，优先级最高 |

优先级：

```text
annotation queue > dataset > tracing project
```

### 为 Tracing Project 配置

步骤：

1. 进入 **Tracing Projects**。
2. 点击已有项目或创建新项目。
3. 在编辑面板中找到 **Custom Output Rendering**。
4. 开启 **Enable custom output rendering**。
5. 输入网页 URL。
6. 点击 **Save**。

### 为 Dataset 配置

步骤：

1. 进入 **Datasets & Experiments**，打开 dataset。
2. 点击右上角 **⋮**。
3. 选择 **Custom Output Rendering**。
4. 开启 **Enable custom output rendering**。
5. 输入网页 URL。
6. 点击 **Save**。

### 为 Annotation Queue 配置

步骤：

1. 进入 **Annotation Queues**。
2. 点击已有 queue 或创建新 queue。
3. 在设置面板找到 **Custom Output Rendering**。
4. 开启 **Enable custom output rendering**。
5. 输入网页 URL。
6. 点击 **Save** 或 **Create**。

### 自定义 Renderer 接收的数据格式

你的 HTML 页面通过 `postMessage` API 接收 LangSmith 发来的数据。

消息结构：

```typescript
{
  type: "output" | "reference",
  data: {
    // actual output 或 reference output
  },
  metadata: {
    inputs: {
      // 生成该 output 的 inputs
    }
  }
}
```

字段说明：

| 字段 | 说明 |
|------|------|
| `type` | `"output"` 表示实际输出，`"reference"` 表示参考输出 |
| `data` | 输出数据本身，结构由你的应用决定 |
| `metadata.inputs` | 生成该输出的输入，用于上下文展示 |

消息发送机制：

- LangSmith 会用指数退避重试发送 message。
- 最多发送 6 次。
- 延迟为 100ms、200ms、400ms、800ms、1600ms、3200ms。
- 这样可以提高 renderer 页面加载较慢时收到数据的概率。

### Renderer 最小示例

```html
<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8" />
    <title>PostMessage Echo</title>
  </head>
  <body>
    <h1>PostMessage Messages</h1>
    <div id="messages"></div>
    <script>
      let count = 0;
      window.addEventListener("message", (event) => {
        count++;
        const header = document.createElement("h3");
        header.appendChild(document.createTextNode(`Message ${count}`));

        const code = document.createElement("code");
        code.appendChild(
          document.createTextNode(JSON.stringify(event.data, null, 2))
        );

        const pre = document.createElement("pre");
        pre.appendChild(code);

        document.getElementById("messages").appendChild(header);
        document.getElementById("messages").appendChild(pre);
      });
    </script>
  </body>
</html>
```

### Custom Rendering 出现的位置

启用后会替代默认 output view，出现在：

- Experiment comparison view。
- 与 dataset 关联的 run detail panes。
- Annotation queues。

---

## 最佳实践

### 构建 Dataset

| 建议 | 原因 |
|------|------|
| 从少量高质量人工样例开始 | 先定义什么是好结果 |
| 持续从生产 traces 回流 | 覆盖真实长尾问题 |
| 使用 annotation queue 审核重要样例 | 提高 reference outputs 质量 |
| 对 synthetic examples 做人工抽查 | 避免合成数据污染评估标准 |
| 保留 source metadata | 方便区分 manual、production、synthetic |

### 组织 Dataset

| 建议 | 原因 |
|------|------|
| 设计 schema | 保持样例结构一致 |
| 使用 splits | 区分 train、validation、test、edge cases |
| 使用 metadata | 支持过滤、分组、分析 |
| 给稳定版本打 tag | CI 和发布流程可复现 |
| 不直接在 CI 使用易变 latest | 数据变化会导致结果不稳定 |

### 使用 Dataset 做评估

| 建议 | 原因 |
|------|------|
| 按 split 分别看结果 | 避免整体平均值掩盖某类失败 |
| 按 metadata group by 分析 | 找出特定来源、难度、业务线的问题 |
| 对失败 traces 回流 dataset | 形成持续回归集 |
| 对关键 dataset version 固定 as_of | 保证实验可复现 |
| 将 hard cases 单独维护 | 避免模型只在简单样例上表现好 |

### 安全与治理

| 建议 | 原因 |
|------|------|
| 公开共享前脱敏 | 共享链接可被无账号用户访问 |
| 谨慎导出生产数据 | CSV / JSONL 可能包含敏感信息 |
| 多租户样例记录 tenant_id | 方便过滤和隔离分析 |
| 自定义 renderer 不展示敏感字段 | 避免人工审核页面泄露数据 |

---

## 快速参考

### 创建 Dataset 的方式

| 方式 | 适合场景 |
|------|----------|
| UI 从零创建 | 小规模手工样例 |
| 从 Tracing Project 手动添加 | 挑选真实高价值 traces |
| Run rules 自动添加 | 持续收集符合条件的 traces |
| Annotation queue 添加 | 专家审核后沉淀样例 |
| Playground 创建 | prompt 调试阶段快速评估 |
| CSV / JSONL 导入 | 已有标注文件 |
| SDK 创建 | 自动化数据管道 |
| Synthetic examples | 扩充覆盖面 |

### SDK 常用方法

| 方法 | 作用 |
|------|------|
| `create_dataset` / `createDataset` | 创建 dataset |
| `create_examples` / `createExamples` | 批量创建 examples |
| `upload_csv` / `uploadCsv` | 上传 CSV 创建 dataset |
| `upload_dataframe` | 从 pandas DataFrame 创建 dataset |
| `list_datasets` / `listDatasets` | 查询 datasets |
| `list_examples` / `listExamples` | 查询 examples |
| `update_example` / `updateExample` | 更新单个 example |
| `update_examples` / `updateExamples` | 批量更新 examples |
| `update_dataset_tag` | 给 dataset version 打 tag |

### Example 字段速查

| 字段 | 是否必需 | 用途 |
|------|----------|------|
| `inputs` | 是 | 应用运行输入 |
| `outputs` / `reference outputs` | 可选 | evaluator 使用的标准答案 |
| `metadata` | 可选 | 过滤、分组、记录来源 |
| `split` / `splits` | 可选 | 分组评估 |

### 评估视图速查

| 需求 | 做法 |
|------|------|
| 评估最新 dataset | `as_of="latest"` |
| 评估固定版本 | 使用 timestamp 或 tag |
| 只评估 test set | `splits=["test"]` |
| 只评估某类样例 | `metadata={"topic": "billing"}` |
| 复杂 metadata 过滤 | 使用 structured filter |

### 一句话总结

- Dataset 是 LangSmith 离线评估和回归测试的核心资产。
- Examples 应包含稳定的 inputs、可选 reference outputs 和有用 metadata。
- UI 适合人工构建与审核，SDK 适合自动化批量管理。
- Schema、splits、metadata、versions 是 dataset 可维护性的关键。
- 生产 traces、annotation queues、experiment traces 都应持续回流到 dataset。
- 自定义输出渲染可以显著改善复杂结构化输出的评审体验。
