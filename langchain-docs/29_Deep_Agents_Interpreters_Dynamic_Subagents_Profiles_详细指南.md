# Deep Agents Interpreters、Dynamic Subagents 与 Profiles 详细指南

> 基于 Deep Agents 官方 Interpreters、Dynamic subagents、Profiles 文档整理。本文聚焦 QuickJS interpreter 如何让 agent 在模型循环内写代码、如何用 PTC 从代码中调用工具、如何通过 `task()` 动态编排 subagents，以及如何用 HarnessProfile / ProviderProfile 为不同 provider 和 model 打包默认行为。

## 目录

1. [整体理解](#整体理解)
2. [Interpreters 是什么](#interpreters-是什么)
3. [Interpreter vs Sandbox](#interpreter-vs-sandbox)
4. [Interpreter Quickstart](#interpreter-quickstart)
5. [How Interpreters Work](#how-interpreters-work)
6. [Programmatic Tool Calling](#programmatic-tool-calling)
7. [Dynamic Subagents 概览](#dynamic-subagents-概览)
8. [Dynamic Subagents Quickstart](#dynamic-subagents-quickstart)
9. [Dynamic Subagents Patterns](#dynamic-subagents-patterns)
10. [Interpreter Persistence](#interpreter-persistence)
11. [Interpreter Security](#interpreter-security)
12. [Interpreter Configuration](#interpreter-configuration)
13. [Profiles 是什么](#profiles-是什么)
14. [Harness Profiles](#harness-profiles)
15. [Provider Profiles](#provider-profiles)
16. [Registration Keys 与 Merge Semantics](#registration-keys-与-merge-semantics)
17. [Config Files 与 Plugin 分发](#config-files-与-plugin-分发)
18. [三者如何协同](#三者如何协同)
19. [常见架构模式](#常见架构模式)
20. [最佳实践](#最佳实践)
21. [故障排查](#故障排查)
22. [快速参考](#快速参考)
23. [资料来源](#资料来源)

---

## 整体理解

这三部分解决的是 Deep Agents 的高级编排与模型适配问题。

| 主题 | 解决的问题 | 核心入口 |
|------|------------|----------|
| Interpreters | 让 agent 在模型循环内运行轻量代码，做循环、分支、聚合、状态保存 | `CodeInterpreterMiddleware` |
| Dynamic subagents | 让 agent 在 interpreter 代码里批量、并行、递归地调用 subagents | `task()` global |
| Profiles | 为不同 provider/model 打包 harness 行为和模型构造默认值 | `HarnessProfile`, `ProviderProfile` |

关系图：

```text
Deep Agent
├─ Profiles
│  ├─ 调整 prompt / tools / middleware / general-purpose subagent
│  └─ 调整 provider:model 初始化参数
├─ Interpreter
│  ├─ eval tool
│  ├─ JavaScript state
│  ├─ PTC tools.*
│  └─ task() dynamic subagents
└─ Subagents
   ├─ normal task tool path
   └─ dynamic task() from interpreter code
```

一句话：

```text
Interpreters 让 agent 用代码做中间编排；
Dynamic subagents 让这些代码可以批量调用子智能体；
Profiles 让不同模型下的 harness 行为可配置、可复用、可分发。
```

---

## Interpreters 是什么

Interpreters 给 agent 一个轻量级可编程工作区。Agent 写 JavaScript 表达自己的意图，QuickJS runtime 执行代码，并只把相关结果返回给模型。

它适合解决普通 tool calling 的限制：

| 普通 tool calling 的限制 | Interpreter 的解决方式 |
|--------------------------|-------------------------|
| 一轮工具调用 batch 发出后固定 | 代码里可以根据结果继续分支 |
| 需要多轮模型调用才能循环/重试 | 代码中直接 loop / retry |
| 中间结果全部回到模型上下文 | 中间变量留在 interpreter state |
| 大批量任务模型容易漏项 | 代码循环可覆盖每一项 |
| 聚合、排序、过滤不稳定 | 用确定性代码处理 |

适合 interpreter 的任务：

| 任务 | 示例 |
|------|------|
| 数据转换 | parse、sort、group、dedupe、score |
| 批量工具调用 | 对多个 query 并行 search |
| 条件分支 | 根据结果决定下一步 |
| 重试逻辑 | 失败后换参数重试 |
| 中间状态保存 | 多轮对话中保留变量 |
| 子智能体编排 | 批量 task fan-out + synthesize |

Interpreters 当前状态：

```text
Beta API，生命周期行为可能变化。
需要 langchain-quickjs>=0.1.0 和 Python>=3.11。
```

---

## Interpreter vs Sandbox

Interpreters 和 Sandboxes 都与“运行代码”有关，但定位不同。

| 维度 | Interpreter | Sandbox |
|------|-------------|---------|
| 代码位置 | Agent loop 内部 | 隔离操作系统环境 |
| Runtime | QuickJS in-memory JavaScript | Provider sandbox / devbox / container |
| 主要用途 | 编排工具、处理数据、保存中间状态 | shell、安装依赖、跑测试、改文件 |
| 文件系统 | 默认无访问 | 有 sandbox 文件系统 |
| 网络 | 默认无访问 | 取决于 sandbox provider |
| Shell | 不支持 | 支持 `execute` |
| 包管理 | 不支持 | 支持安装依赖 |
| 安全边界 | capability-scoped runtime，不是完整 VM | OS-level isolation，取决于 provider |

选择：

| 需求 | 推荐 |
|------|------|
| 一两个简单外部调用 | 普通 tool calling |
| 小程序：循环、分支、重试、聚合 | Interpreter |
| 大量选定工具调用从代码运行 | Interpreter + PTC |
| 大量独立工作单元、多视角或递归分析 | Interpreter + Dynamic subagents |
| shell、安装包、跑测试、OS 文件系统 | Sandbox |

---

## Interpreter Quickstart

安装：

```bash
pip install -U "deepagents[quickjs]"
```

或：

```bash
uv add "deepagents[quickjs]"
```

配置：

```python
from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware

agent = create_deep_agent(
    model="openai:gpt-5.5",
    middleware=[CodeInterpreterMiddleware()],
)
```

配置后，agent 会获得一个 `eval` 工具。模型在需要时会写 JavaScript 并调用 `eval`。

---

## How Interpreters Work

`CodeInterpreterMiddleware` 会添加一个 `eval` 工具。

Agent 可以写这样的 JavaScript：

```javascript
const rows = [
  { team: "alpha", score: 8 },
  { team: "beta", score: 13 },
  { team: "alpha", score: 21 },
];

const totals = rows.reduce((acc, row) => {
  acc[row.team] = (acc[row.team] ?? 0) + row.score;
  console.log(`${row.team} score: ${acc[row.team]}`);
  return acc;
}, {});

totals;
```

Interpreter 行为：

| 行为 | 说明 |
|------|------|
| 运行语言 | JavaScript |
| Runtime | QuickJS |
| `console.log` | 默认捕获并返回 |
| 返回值 | 最后一个表达式结果 |
| State | 同一 run 内多次 `eval` 使用同一 context |
| 跨 turn state | 默认 snapshot / restore |
| 默认权限 | 无 host FS、无 network、无 shell、无 clock |

默认隔离：

```text
QuickJS code 默认只能计算、保存变量、输出 console.log。
它不能访问宿主文件系统、网络、shell、包管理器或当前时间。
```

只有两个显式桥接能力：

| Bridge | 说明 |
|--------|------|
| PTC | 将 allowlist 工具暴露为 `tools.*` async functions |
| Dynamic subagents | 将 subagent dispatch 暴露为 `task()` global |

---

## Programmatic Tool Calling

Programmatic Tool Calling，简称 PTC，让 interpreter code 可以调用 allowlisted tools。

启用方式：

```python
from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware

agent = create_deep_agent(
    model="openai:gpt-5.5",
    middleware=[
        CodeInterpreterMiddleware(
            ptc=["web_search"],
        )
    ],
)
```

启用后，工具会暴露在 JavaScript 的 `tools` namespace 下。

工具名会转换为 camelCase：

```text
web_search -> tools.webSearch(...)
```

示例：

```javascript
const topics = ["retrieval", "memory", "evaluation"];

const results = await Promise.all(
  topics.map((topic) =>
    tools.webSearch({ query: `${topic} best practices 2025` }),
  ),
);

results.join("\n\n");
```

PTC 适合：

| 场景 | 原因 |
|------|------|
| 多个工具调用要循环执行 | 用 JS loop |
| 调用结果要过滤/聚合 | 中间结果不回模型上下文 |
| 失败要重试 | JS try/catch |
| 并行调用 | `Promise.all` |
| 模型不擅长批量覆盖 | 代码确定性遍历 |

重要限制：

```text
PTC-invoked tool calls 不走普通 tool calling path。
因此 interrupt_on 审批不会逐个 PTC tool call 生效。
```

如果需要审批：

1. 不要把高风险工具放进 PTC allowlist。
2. 对 `eval` 工具本身配置 HITL。
3. 或让敏感操作走普通 tool calling path。

---

## Dynamic Subagents 概览

Dynamic subagents 让 agent 在 interpreter code 里调用 subagents。

前提：

1. Agent 配置了 subagents。
2. Agent 配置了 `CodeInterpreterMiddleware`。
3. `subagents=True`，这是默认值。

当 interpreter 可用且 agent 有 subagents 时，JavaScript 中会有一个全局函数：

```javascript
task({...})
```

它用于从代码中 dispatch subagent。

`task()` 输入：

| 字段 | 说明 |
|------|------|
| `description` | 给 subagent 的任务 prompt |
| `subagentType` | 调用哪个 configured subagent |
| `responseSchema` | 可选，结构化输出 schema |

示例：

```javascript
const review = await task({
  description: "Review src/auth/login.ts for auth issues. Cite line numbers.",
  subagentType: "reviewer",
  responseSchema: {
    type: "object",
    properties: {
      issues: {
        type: "array",
        items: {
          type: "object",
          properties: {
            file: { type: "string" },
            line: { type: "number" },
            severity: { type: "string" },
            description: { type: "string" },
          },
        },
      },
    },
  },
});

const critical = review.issues.filter((issue) => issue.severity === "high");
```

使用 `responseSchema` 时，返回值已经是 typed JavaScript object，不需要 `JSON.parse`。

---

## Dynamic Subagents Quickstart

安装 QuickJS：

```bash
pip install -U "deepagents[quickjs]"
```

配置：

```python
from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware

agent = create_deep_agent(
    model="openai:gpt-5.5",
    middleware=[CodeInterpreterMiddleware()],
)
```

Deep Agents 默认有 `general-purpose` subagent，所以即使不配置自定义 subagents，也可以尝试基础 fan-out。

如果需要专业分工，配置自定义 subagents：

```python
agent = create_deep_agent(
    model="openai:gpt-5.5",
    subagents=[
        {
            "name": "reviewer",
            "description": "Reviews code for security issues, citing lines and severity",
            "system_prompt": "You are a security-focused code reviewer.",
        },
    ],
    middleware=[CodeInterpreterMiddleware()],
)
```

触发建议：

```text
在用户请求中使用 “workflow” 这个词，
可以提示 agent 用 interpreter + dynamic subagents 做编排。
```

例如：

```python
result = await agent.ainvoke({
    "messages": [{
        "role": "user",
        "content": "Run a workflow that reviews every file in src/routes/ and summarizes the top risks.",
    }]
})
```

对单次直接委派，不要强调 workflow，普通 `task` tool 更自然。

---

## Dynamic Subagents Patterns

### Classify and Act

先分类，再把不同类别交给不同 subagent。

适合：

| 场景 | 示例 |
|------|------|
| Support tickets | bug / feature / question |
| Feedback triage | complaint / request / praise |
| Error logs | infra / app / security |

配置：

```python
agent = create_deep_agent(
    model="openai:gpt-5.5",
    subagents=[
        {
            "name": "bug-fixer",
            "description": "Investigates bug reports and provides reproduction steps",
            "system_prompt": "You are a bug triage specialist.",
        },
        {
            "name": "feature-analyst",
            "description": "Evaluates feature requests for feasibility and effort",
            "system_prompt": "You are a product analyst.",
        },
        {
            "name": "support-agent",
            "description": "Answers user questions based on documentation",
            "system_prompt": "You are a support specialist.",
        },
    ],
    middleware=[CodeInterpreterMiddleware()],
)
```

代码形态：

```javascript
const SPECIALIST = {
  bug: "bug-fixer",
  feature: "feature-analyst",
  question: "support-agent",
};

const handled = await Promise.all(
  tickets.map((ticket) =>
    task({
      description: `Handle this ${ticket.category}:\n${ticket.text}`,
      subagentType: SPECIALIST[ticket.category],
    }),
  ),
);

handled;
```

### Fan-out and Synthesize

对许多 items 并行执行同类工作，然后合并结果。

适合：

| 场景 | 示例 |
|------|------|
| 代码审查 | 每个文件一个 reviewer |
| 文档分析 | 每个文档一个 analyst |
| 日志扫描 | 每个服务一个 analyzer |

代码形态：

```javascript
const files = (await tools.glob({ pattern: "src/routes/**/*.ts" }))
  .split("\n")
  .filter(Boolean);

const reviews = await Promise.all(
  files.map((file) =>
    task({
      description: `Review ${file} for authentication issues. Cite line numbers.`,
      subagentType: "reviewer",
      responseSchema: issuesSchema,
    }),
  ),
);

const issues = reviews.flatMap((r) => r.issues);
issues;
```

注意：这个例子使用了 `tools.glob`，需要把 `glob` 加入 PTC allowlist。

### Adversarial Verification

第一轮生成 findings，第二轮由 verifier 独立验证，只保留确认结果。

适合：

| 场景 | 原因 |
|------|------|
| 安全审计 | false positive 成本高 |
| 合规检查 | 需要高置信度 |
| 法务/金融分析 | 需要独立验证 |

代码形态：

```javascript
const { findings } = await task({
  description: "Audit the payments module for vulnerabilities.",
  subagentType: "auditor",
  responseSchema: findingsSchema,
});

const verdicts = await Promise.all(
  findings.map((f) =>
    task({
      description: `Verify ${f.file}:${f.line} (${f.description}). Confirm or refute.`,
      subagentType: "verifier",
      responseSchema: verdictSchema,
    }),
  ),
);

const confirmed = findings.filter((_, i) => verdicts[i]?.confirmed);
confirmed;
```

### Generate and Filter

多个 subagents 独立生成方案，interpreter 负责评分和筛选。

适合：

1. 架构方案。
2. 重构策略。
3. 内容创意。
4. 多实现比较。

```javascript
const proposals = await Promise.all(
  [1, 2, 3].map((n) =>
    task({
      description: `Approach ${n}: redesign the orders schema, with tradeoffs.`,
      subagentType: "architect",
      responseSchema: designSchema,
    }),
  ),
);

const best = proposals.sort((a, b) => score(b) - score(a))[0];
best;
```

### Tournament

多方案两两比较，judge subagent 选择胜者，直到剩一个。

适合：

| 场景 | 示例 |
|------|------|
| 代码可读性选择 | 多个 rewrite 版本 |
| 文案风格选择 | 多种营销文案 |
| 主观质量比较 | 设计方案、解释质量 |

### Loop Until Done

循环发现新结果，去重，直到没有新内容。

适合：

| 场景 | 示例 |
|------|------|
| dead code detection | 不知道范围有多大 |
| exhaustive search | 直到无新增结果 |
| dependency audit | 逐步发现新依赖 |

代码形态：

```javascript
const seen = new Set();
const found = [];

while (true) {
  const { items } = await task({
    description: `Find dead code. Already found: ${[...seen].join(", ") || "(none)"}.`,
    subagentType: "analyzer",
    responseSchema: itemsSchema,
  });

  const fresh = items.filter((i) => !seen.has(i.id));
  if (fresh.length === 0) break;

  for (const i of fresh) {
    seen.add(i.id);
    found.push(i);
  }
}

found;
```

### Disable Dynamic Subagents

默认情况下，只要 agent 有 subagents，interpreter 会暴露 `task()`。

可以关闭：

```python
from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware

agent = create_deep_agent(
    model="openai:gpt-5.5",
    subagents=[
        {
            "name": "reviewer",
            "description": "Reviews code",
            "system_prompt": "Review code.",
        }
    ],
    middleware=[CodeInterpreterMiddleware(subagents=False)],
)
```

---

## Interpreter Persistence

默认情况下，`CodeInterpreterMiddleware` 会在每个 agent run 结束后 snapshot interpreter state，并在下一个 turn 恢复。

生命周期：

1. Turn 开始，middleware 恢复该 thread 的最新 interpreter snapshot。
2. Agent 调用 `eval`，代码读写变量。
3. Agent run 结束，middleware 把最新 JS state snapshot 到 graph state。
4. 下一 turn 从该 snapshot 恢复。

示例：

```python
from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

agent = create_deep_agent(
    model="openai:gpt-5.5",
    checkpointer=checkpointer,
    middleware=[
        CodeInterpreterMiddleware(
            snapshot_between_turns=True,
        )
    ],
)
```

注意：

| 点 | 说明 |
|----|------|
| 同一个 run 内 | 多次 `eval` 使用 live context |
| 跨 turns | 依靠 snapshot / restore |
| 可序列化限制 | 适合数据，不适合函数、class、live objects |
| 不撤销外部副作用 | PTC tool call 造成的外部影响不会被 snapshot undo |
| Time travel | graph checkpoint 恢复时可恢复 interpreter snapshot |

如果不需要跨 turn state：

```python
CodeInterpreterMiddleware(snapshot_between_turns=False)
```

---

## Interpreter Security

Interpreters 使用 QuickJS 运行 JavaScript，默认很窄：

| Capability | 默认可用 | 如何暴露 |
|------------|----------|----------|
| JavaScript execution | 是 | 添加 interpreter middleware |
| Top-level `await` | 是 | 直接使用 promises |
| `console.log` capture | 是 | `capture_console=False` 可关闭 |
| Agent tools | 否 | PTC allowlist |
| Filesystem access | 否 | 通过 PTC 暴露 filesystem tools |
| Network access | 否 | 通过 PTC 暴露特定 network tool |
| Wall-clock / datetime | 否 | 显式暴露 time tool |
| Shell / package install / OS execution | 否 | 使用 sandbox backend |

重要安全点：

```text
QuickJS interpreter 运行在嵌入式 context 中，不是独立 VM 或进程。
它是 capability-scoped execution layer，不是 host-memory isolation boundary。
```

因此：

1. 不要把高风险工具放入 PTC allowlist。
2. 对 PTC allowlist 当作 permission boundary。
3. 对 untrusted/semi-trusted code，agent 运行在隔离 worker process 或 container 中。
4. Shell、包安装、测试和 OS 访问应该使用 sandbox backend。
5. PTC 不逐个执行 `interrupt_on` 审批。
6. Dynamic `task()` 不走 normal tool calling path，parent `interrupt_on` 不逐个生效。

如果需要审批 dynamic orchestration：

```text
Gate the eval tool itself.
```

---

## Interpreter Configuration

`CodeInterpreterMiddleware` 常用参数：

| Kwarg | Default | 作用 |
|-------|---------|------|
| `memory_limit` | `64 * 1024 * 1024` | QuickJS heap memory limit |
| `timeout` | `5.0` | 每次 eval 超时时间 |
| `max_ptc_calls` | `256` | 每次 eval 最多 `tools.*` 调用次数 |
| `tool_name` | `"eval"` | 暴露给模型的 interpreter tool 名称 |
| `max_result_chars` | `4000` | 返回结果和 stdout 最大字符数 |
| `capture_console` | `True` | 是否捕获 console 输出 |
| `subagents` | `True` | 是否暴露 `task()` dynamic subagents |
| `ptc` | `None` | PTC allowlist |
| `snapshot_between_turns` | `True` | 是否跨 turns 保存 interpreter state |
| `max_snapshot_bytes` | `None` | serialized snapshot 最大大小，默认跟 `memory_limit` 相关 |

配置建议：

| 需求 | 建议 |
|------|------|
| 防止无限循环 | 设置较短 `timeout` |
| 防止内存膨胀 | 设置 `memory_limit` |
| 防止工具调用爆炸 | 控制 `max_ptc_calls` |
| 高风险环境 | PTC allowlist 只放低风险工具 |
| 不需要跨 turn state | `snapshot_between_turns=False` |
| 不想让 eval 调 subagents | `subagents=False` |

---

## Profiles 是什么

Profiles 用于按 provider 或 model 打包默认行为。

Deep Agents 有两类 profiles：

| 类型 | 控制什么 | 适用阶段 |
|------|----------|----------|
| HarnessProfile | Deep Agents harness 行为，例如 prompt、tool visibility、middleware、general-purpose subagent | 模型构造之后 |
| ProviderProfile | 模型构造参数，例如 `init_chat_model` kwargs、credential checks | 模型构造时 |

一句话：

```text
HarnessProfile 管 agent harness 怎么对模型说话；
ProviderProfile 管模型实例怎么被创建。
```

---

## Harness Profiles

`HarnessProfile` 用于调整 Deep Agents harness。

示例：

```python
from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    register_harness_profile,
)

register_harness_profile(
    "openai:gpt-5.5",
    HarnessProfile(
        system_prompt_suffix="Respond in under 100 words.",
        excluded_tools={"execute"},
        excluded_middleware={"SummarizationMiddleware"},
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    ),
)
```

字段：

| Field | 作用 |
|-------|------|
| `base_system_prompt` | 替换 Deep Agents base system prompt |
| `system_prompt_suffix` | 追加到 assembled base prompt 后面 |
| `tool_description_overrides` | 按 tool name 覆盖工具描述 |
| `excluded_tools` | 从工具集中移除指定工具 |
| `excluded_middleware` | 从默认 middleware stack 移除指定 middleware |
| `extra_middleware` | 给 stack 追加 middleware |
| `general_purpose_subagent` | 禁用、重命名或改写 general-purpose subagent |

注意：

1. Caller-supplied `system_prompt=` 永远在 assembled prompt 最前。
2. `system_prompt_suffix` 永远在最后。
3. Subagents 会按自己的 model 重新 resolve profile。
4. general-purpose subagent 有自己的 prompt overlay 规则。

### 不要用 excluded_middleware 移除核心 scaffolding

文档明确警告：以下 middleware 不能通过 `excluded_middleware` 移除，否则会抛 `ValueError`：

1. `FilesystemMiddleware`
2. `SubAgentMiddleware`
3. internal permission middleware

如果只是想让模型看不到工具，用：

```python
excluded_tools={"read_file", "write_file", "task"}
```

如果想无 `task` tool：

1. 设置 `general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)`。
2. 不传同步 subagents。

Async subagents 不受这个配置影响。

---

## Provider Profiles

`ProviderProfile` 控制模型构造。

它只适用于：

```python
create_deep_agent(model="provider:model")
```

不适用于已经实例化的模型对象：

```python
model = init_chat_model(...)
create_deep_agent(model=model)
```

示例：

```python
from deepagents import ProviderProfile, register_provider_profile

register_provider_profile(
    "openai",
    ProviderProfile(
        init_kwargs={"temperature": 0},
    ),
)
```

字段：

| Field | 作用 |
|-------|------|
| `init_kwargs` | 静态初始化参数，传给 `init_chat_model` |
| `pre_init` | 构造前副作用，例如 credential validation |
| `init_kwargs_factory` | 每次 `resolve_model` 时动态生成 kwargs |

适合：

| 场景 | 示例 |
|------|------|
| Provider 默认参数 | `temperature=0` |
| Credential 检查 | 构造前验证环境变量 |
| 动态 headers | 从 runtime/env 生成 headers |
| Provider integration plugin | 打包默认模型构造行为 |

---

## Registration Keys 与 Merge Semantics

两类 profile 都使用同样的 key 格式。

| Key 类型 | 示例 | 作用 |
|----------|------|------|
| Provider-level | `"openai"` | 应用于该 provider 下所有模型 |
| Model-level | `"openai:gpt-5.5"` | 只应用于该具体模型 |

当 provider-level 和 model-level 同时存在：

```text
先应用 provider-level；
再应用 model-level；
model-level 显式字段覆盖 provider-level。
```

重新注册同一个 key：

```text
不是替换，而是 merge 到已有 profile 上。
```

没有 wildcard key：

```text
不存在匹配所有 provider 的 profile key（通配key/全能key）。
```

如果想对所有 provider 生效，要分别注册每个 provider key。真正全局的调整更适合放在 `create_deep_agent` 调用处。

### Merge Semantics（合并语义）

| Field | 合并方式 |
|-------|----------|
| `base_system_prompt`, `system_prompt_suffix` | 新值 set 时覆盖，否则继承 |
| `tool_description_overrides` | 按 key 合并，新值覆盖同 key |
| `excluded_tools`, `excluded_middleware` | set union |
| `extra_middleware` | 按 name 合并，新实例替换同名，新增 append |
| `general_purpose_subagent` | field-wise merge |
| Provider `init_kwargs` | dict key-wise merge |
| Provider `pre_init` | callable chain，旧的先运行 |
| Provider `init_kwargs_factory` | factories chain，每次 resolve 合并输出 |

### Preconfigured Model Lookup Order

当传入已配置好的 chat model instance，而不是 `provider:model` 字符串，harness 会尝试合成 canonical key，并按顺序查找：

1. exact `provider:identifier`
2. identifier-only，只有 identifier 已包含 `:` 时
3. provider-only fallback

---

## Config Files 与 Plugin 分发

### YAML / JSON Config（直接在文件中配置）

使用 `HarnessProfileConfig` 加载/保存 declarative profile。

YAML 示例：

```yaml
base_system_prompt: You are helpful.
system_prompt_suffix: Respond briefly.
excluded_tools:
  - execute
  - grep
excluded_middleware:
  - SummarizationMiddleware
  - my_pkg.middleware:TelemetryMiddleware
general_purpose_subagent:
  enabled: false
```

加载（文件内容）：

```python
import yaml
from deepagents import HarnessProfileConfig, register_harness_profile

with open("openai.yaml") as f:
    register_harness_profile(
        "openai",
        HarnessProfileConfig.from_dict(yaml.safe_load(f)),
    )
```

限制：

| 内容 | 是否可序列化 |
|------|--------------|
| prompt text | 是 |
| tool description overrides | 是 |
| excluded tools | 是 |
| excluded middleware string/import ref | 是 |
| general-purpose edits | 是 |
| middleware instances | 否 |
| factories | 否 |
| `__main__` 或函数作用域 middleware class | 否 |

### Plugin 分发

Profiles 可以通过 Python package entry points 自动注册。

`pyproject.toml`：

```toml
[project.entry-points."deepagents.harness_profiles"]
my_provider = "my_pkg.profiles:register_harness"

[project.entry-points."deepagents.provider_profiles"]
my_provider = "my_pkg.profiles:register_provider"
```

注册函数：

```python
from deepagents import (
    HarnessProfile,
    ProviderProfile,
    register_harness_profile,
    register_provider_profile,
)


def register_harness() -> None:
    register_harness_profile(
        "my_provider",
        HarnessProfile(
            system_prompt_suffix="Batch independent tool calls in parallel.",
        ),
    )


def register_provider() -> None:
    register_provider_profile(
        "my_provider",
        ProviderProfile(
            init_kwargs={"temperature": 0},
        ),
    )
```

Load order：

```text
built-ins
-> entry-point plugins
-> direct register_*_profile calls in user code
```

后注册的 profile layer 在前面的基础上。

---

## 三者如何协同

### Interpreter + PTC（JS代码）

```text
Agent 写 JS
-> JS 调 tools.webSearch / tools.glob
-> 中间结果留在 JS variables
-> 模型只看到 compact final result
```

适合批量工具调用、聚合、过滤。

### Interpreter + Dynamic Subagents

```text
Agent 写 JS workflow
-> JS loop 调 task()
-> 多个 subagents 并行处理
-> JS synthesize result
-> 模型看到总结
```

适合批量文件审查、ticket triage、verification workflows。

### Profiles + Interpreters

Profiles 可以针对某些 model/provider：

1. 追加提示：鼓励使用 workflow。
2. 隐藏高风险工具：`excluded_tools={"execute"}`。
3. 添加或移除 middleware。
4. 禁用 general-purpose subagent。
5. 为某个模型修改 tool descriptions，使其更容易正确使用 `eval` 或 subagents。

### Profiles + Dynamic Subagents

可以用 HarnessProfile 针对某些模型调整：

| 调整 | 目的 |
|------|------|
| `system_prompt_suffix` | 鼓励批量任务使用 interpreter workflow |
| `tool_description_overrides` | 改善模型对 `eval` / `task` 的理解 |
| `excluded_tools` | 对能力弱或风险高模型隐藏工具 |
| `general_purpose_subagent` | 禁用或重写默认 subagent |

---

## 常见架构模式

### 模式一：轻量数据处理 Agent

```python
agent = create_deep_agent(
    model="openai:gpt-5.5",
    middleware=[CodeInterpreterMiddleware()],
)
```

适合：

1. 表格聚合。
2. JSON 过滤。
3. 文本列表去重。
4. 多轮保存中间 state。

### 模式二：PTC 批量搜索 Agent

```python
agent = create_deep_agent(
    model="openai:gpt-5.5",
    tools=[web_search],
    middleware=[CodeInterpreterMiddleware(ptc=["web_search"])],
)
```

适合：

1. 多 query 并行搜索。
2. 搜索结果过滤。
3. 批量 retries。

### 模式三：Dynamic Code Review Workflow

```python
agent = create_deep_agent(
    model="openai:gpt-5.5",
    subagents=[
        {
            "name": "reviewer",
            "description": "Reviews code for security issues with line citations",
            "system_prompt": "You are a security-focused code reviewer.",
        },
        {
            "name": "verifier",
            "description": "Verifies whether reported vulnerabilities are real",
            "system_prompt": "Be skeptical and confirm only real issues.",
        },
    ],
    middleware=[
        CodeInterpreterMiddleware(
            ptc=["glob"],
        )
    ],
)
```

用户请求：

```text
Run a workflow to review every file in src/routes for auth vulnerabilities,
verify each high-severity finding, and return only confirmed issues.
```

### 模式四：Provider-specific Harness Tuning

```python
register_harness_profile(
    "openai:gpt-5.5",
    HarnessProfile(
        system_prompt_suffix="For batch workflows, prefer using the eval tool.",
        excluded_tools={"execute"},
    ),
)
```

适合：

1. 某个模型对工具描述敏感。
2. 某 provider 不适合某些 harness tools。
3. 想按模型隐藏 `execute` 或其他工具。

### 模式五：Plugin-packaged Provider Defaults

```python
def register_provider() -> None:
    register_provider_profile(
        "my_provider",
        ProviderProfile(
            init_kwargs={"temperature": 0},
        ),
    )
```

适合打包 provider integration。

---

## 最佳实践

### Interpreters

1. 用 interpreter 做循环、分支、聚合，不要让模型逐步手工调工具。
2. 中间结果留在 JS variables，只把 compact result 返回模型。
3. PTC allowlist 只暴露必要工具。
4. 高风险工具不要放进 PTC。
5. 设置 `timeout`、`memory_limit`、`max_ptc_calls`。
6. 不要把 interpreter 当 OS sandbox；shell/测试/安装包用 sandbox。
7. 长期 state 只保存可序列化数据。

### Dynamic Subagents

1. 对批量任务使用 “workflow” 触发动态编排。
2. 每个 subagent 的 `description` 要清楚，方便模型选择。
3. 使用 `responseSchema` 让 JS 直接处理 typed result。
4. 对 false positive 成本高的任务使用 adversarial verification。
5. 对开放式搜索使用 loop until done。
6. 如果需要审批，gate `eval` tool 本身。
7. 不需要 dynamic task 时设置 `subagents=False`。

### Profiles

1. 用 `HarnessProfile` 调整 agent harness 行为。
2. 用 `ProviderProfile` 调整 model construction。
3. Provider-level profile 放通用默认值。
4. Model-level profile 放具体模型覆盖。
5. 不要用 profile 做真正全局调整；全局逻辑放 `create_deep_agent` 调用处。
6. 不要移除 required scaffolding middleware。
7. Config file 中的 import refs 只加载可信本地配置。

---

## 故障排查

### Agent 不使用 Interpreter

| 可能原因 | 处理 |
|----------|------|
| 没有添加 `CodeInterpreterMiddleware` | 加入 middleware |
| 任务太简单 | 普通 tool call 已足够 |
| 用户没表达 workflow | 对批量编排请求使用 “workflow” |
| eval 工具被 excluded | 检查 profiles / excluded_tools |

### PTC 工具不可用

| 可能原因 | 处理 |
|----------|------|
| 未配置 `ptc` allowlist | `CodeInterpreterMiddleware(ptc=[...])` |
| 工具名 camelCase 不对 | `web_search` -> `tools.webSearch` |
| 工具 schema 参数错误 | 按原工具 schema 传 object |
| 超过 `max_ptc_calls` | 调高或分批处理 |

### HITL 没有拦住 PTC / dynamic task

| 原因 | 处理 |
|------|------|
| PTC 不走普通 tool calling path | 不把敏感工具放入 PTC |
| `task()` 在 eval 内 dispatch | gate `eval` tool |
| parent `interrupt_on` 不逐个生效 | 对 orchestration 前审批 |

### Snapshot 恢复失败

| 可能原因 | 处理 |
|----------|------|
| 保存了 function/class/live object | 只保存可序列化数据 |
| snapshot 太大 | 设置 `max_snapshot_bytes`，清理变量 |
| 不需要跨 turn state | `snapshot_between_turns=False` |

### Dynamic Subagents 结果难处理

| 可能原因 | 处理 |
|----------|------|
| 子智能体返回自由文本 | 使用 `responseSchema` |
| subagent description 不清 | 改写 description |
| 并发结果重复 | JS 中 dedupe |
| false positives 多 | 加 verifier pattern |

### Profile 没生效

| 可能原因 | 处理 |
|----------|------|
| key 不匹配 | 检查 provider/model key |
| 传的是 preconfigured model | 了解 lookup order |
| model-level 覆盖 provider-level | 检查 merge 后字段 |
| 重复注册被 merge | 注意不是替换 |
| wildcard 不存在 | 每个 provider 分别注册 |

---

## 快速参考

### Interpreter 速查

| 需求 | 做法 |
|------|------|
| 启用 interpreter | `middleware=[CodeInterpreterMiddleware()]` |
| 让 JS 调工具 | `CodeInterpreterMiddleware(ptc=["tool_name"])` |
| 禁用 dynamic subagents | `CodeInterpreterMiddleware(subagents=False)` |
| 禁用跨 turn snapshot | `snapshot_between_turns=False` |
| 限制 eval 时间 | `timeout=...` |
| 限制工具调用数 | `max_ptc_calls=...` |

### Dynamic Subagents 速查

| Pattern | 适用 |
|---------|------|
| Classify and act | 混合 items，不同类别不同处理 |
| Fan-out and synthesize | 多文件、多文档、多 tickets 批处理 |
| Adversarial verification | 高置信度审查 |
| Generate and filter | 多方案生成后评分 |
| Tournament | 方案两两比较 |
| Loop until done | 开放式穷尽搜索 |

### Profile 速查

| 类型 | 控制什么 |
|------|----------|
| `HarnessProfile` | prompt、tools、middleware、general-purpose subagent |
| `ProviderProfile` | `init_chat_model` kwargs、pre-init、dynamic kwargs |
| Provider key | `"openai"` |
| Model key | `"openai:gpt-5.5"` |
| Config file | `HarnessProfileConfig` |
| Plugin | entry points under `deepagents.harness_profiles` / `deepagents.provider_profiles` |

### 安全速查

| 风险 | 处理 |
|------|------|
| PTC 调敏感工具 | 不放入 allowlist |
| dynamic task 绕过 per-tool HITL | gate `eval` |
| 需要 shell / package install | 用 sandbox |
| untrusted code | agent 运行在 worker process/container |
| profile import refs | 只加载可信配置 |

### 一句话总结

```text
Interpreters 用 QuickJS 给 agent 一个轻量代码工作区；
PTC 让代码调用 allowlisted tools；
Dynamic subagents 让代码批量编排 subagents；
Profiles 让 provider/model 对应的 harness 行为和模型构造默认值可复用。
```

---

## 资料来源

- Deep Agents Interpreters 文档：附件 `f1e5add8-c194-41b0-9bed-b590a2415ada/pasted-text.txt`。
- Deep Agents Dynamic subagents 文档：附件 `8bd61a85-5661-498b-9e2b-937761c9882b/pasted-text.txt`。
- Deep Agents Profiles 文档：附件 `c72a7148-7671-49cd-b655-e1baa95f43d2/pasted-text.txt`。
