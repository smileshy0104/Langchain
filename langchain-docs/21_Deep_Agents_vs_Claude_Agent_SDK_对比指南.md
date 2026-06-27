# Deep Agents vs Claude Agent SDK 对比指南

> 基于官方文档 Comparison with Claude Agent SDK 整理的中文总结。本文聚焦 LangChain Deep Agents 与 Claude Agent SDK 在执行环境、模型供应商、部署、多租户、生产服务和生态绑定方面的差异，帮助选择适合的 Agent harness。

## 目录

1. [概述](#概述)
2. [一句话结论](#一句话结论)
3. [核心对比表](#核心对比表)
4. [执行环境差异](#执行环境差异)
5. [多租户能力](#多租户能力)
6. [生产 Agent Server](#生产-agent-server)
7. [托管与自托管](#托管与自托管)
8. [模型与供应商绑定](#模型与供应商绑定)
9. [生态系统差异](#生态系统差异)
10. [选型建议](#选型建议)
11. [快速参考](#快速参考)

---

## 概述

Deep Agents 和 Claude Agent SDK 都是用于构建自定义 Agent 的 **agent harness**。它们都不只是简单的模型调用封装，而是帮助 Agent 连接工具、执行任务、管理环境和完成复杂工作流。

但两者的设计取舍不同：

- **Deep Agents** 更强调模型和基础设施灵活性、可插拔执行后端、内置多租户和可托管/自托管部署。
- **Claude Agent SDK** 更强调 Anthropic / Claude 生态内的一体化体验，Agent 运行在 sandbox 内，部署和多租户层需要自己构建。

文档中特别提到，Deep Agents 已被 OpenSWE 和 LangSmith Fleet 用于生产环境。

---

## 一句话结论

如果你希望 **跨模型供应商、跨执行环境、内置多租户、并且能在 managed 与 self-hosted 之间无代码切换**，优先选择 **Deep Agents**。

如果你已经深度投入 **Anthropic / Claude 生态**，并且愿意自己构建 API、认证、流式传输、多租户和部署层，可以选择 **Claude Agent SDK**。

---

## 核心对比表

| 维度 | Deep Agents | Claude Agent SDK |
|------|-------------|------------------|
| Agent 运行位置 | 可在 sandbox 内运行，也可在 sandbox 外运行并远程调用 sandbox | 在 sandbox 内运行 |
| 执行后端 | 可插拔：local、virtual filesystem、remote sandbox、自定义 backend | Agent 所在 sandbox 的本地文件系统 |
| 模型供应商 | 任意 provider：Anthropic、OpenAI、Google 等 100+ | Claude：Anthropic、Bedrock、Vertex、Azure |
| 模型/供应商调优 | Harness profiles，可按 provider 或具体 model 声明 system prompt、tool、middleware、subagent 调整 | 在每个 model call site 通过代码配置 |
| 部署方式 | LangSmith Managed Deep Agents，或通过 `langgraph build` 构建 standalone image 自托管 | 自托管；需要自己构建服务层 |
| Managed 产品 | Managed Deep Agents / LangSmith Fleet 可托管运行 | Claude managed agents 是独立产品，SDK 代码不能直接部署过去 |
| 多租户 | 内置 scoped threads、per-user sandboxes、RBAC | 需要自己实现 |
| Agent Server | 内置 streaming endpoints、thread management、run history、webhooks、auth | 需要自己写 HTTP/WebSocket/SSE server |
| License | MIT | MIT，但 Claude Code 本身是 proprietary |

---

## 执行环境差异

### 两种 Agent 连接 Sandbox 的模式

文档提到 Agent 与 sandbox 的连接通常有两种模式：

1. **Agent 运行在 sandbox 内**
2. **Agent 运行在 sandbox 外，把 sandbox 当作工具远程使用**

Claude Agent SDK 只支持第一种：Agent 运行在 sandbox 内，并使用 sandbox 的本地文件系统执行工具。

Deep Agents 两种都支持：

- 可以像 Claude Agent SDK 一样，让 Agent 运行在 sandbox 内。
- 可以让 Agent 运行在长期存在的容器中，把 remote sandbox 当作工具，通过网络执行命令。
- 可以换成 virtual filesystem 做测试。
- 可以接入自定义 backend 适配自己的基础设施。

### 为什么这个差异重要

| 架构选择 | 影响 |
|----------|------|
| Agent 在 sandbox 内 | 简单直接，但 Agent 生命周期和 sandbox 强绑定 |
| Agent 在 sandbox 外，sandbox 作为工具 | 更适合长期服务、多租户、集中调度和远程执行 |
| 可插拔 backend | 更容易做测试、隔离、迁移和自定义基础设施 |

Deep Agents 的优势在于把 execution backend 抽象出来，让模型、Agent runtime、sandbox 和部署目标可以独立选择。

---

## 多租户能力

生产环境通常不是只服务一个用户，而是面向多个终端用户。因此 Agent 执行环境必须隔离：

- 每个用户的文件系统。
- 每个用户的运行历史。
- 每个用户的认证和权限。
- 每个用户可访问的第三方 API。

### Claude Agent SDK

Claude Agent SDK 将 Agent 和 sandbox 绑定在一起。要实现多租户，你需要自己构建额外服务层：

1. 为每个用户启动 sandbox。
2. 记录哪个 sandbox 属于哪个用户。
3. 管理 sandbox 生命周期。
4. 在用户完成后清理环境。
5. 自己实现认证、授权和隔离。

### Deep Agents

Deep Agents 在 harness 层处理多租户：

- 支持 per-user 或 per-assistant sandbox。
- 支持 scoped threads。
- 支持 run history。
- 支持 RBAC。
- 配合 LangSmith Sandbox 可获得 auth proxy。

LangSmith Sandbox 的 auth proxy 能让终端用户从 sandbox 调用第三方 API，而不需要你为每个用户单独分发和管理凭证。

---

## 生产 Agent Server

### Claude Agent SDK

如果要把自托管 Claude Agent SDK 应用暴露给终端用户，你需要自己实现服务端：

- HTTP / WebSocket / SSE server。
- 调用 Agent 的接口。
- token streaming。
- conversation threads。
- run history。
- authentication。
- authorization。
- 安全和运维。

换句话说，SDK 提供 Agent harness，但生产服务层是你的责任。

### Deep Agents

Deep Agents 部署包含开箱即用的 Agent Server，具备：

- streaming endpoints。
- thread management。
- run history。
- webhooks。
- authentication。

这使它更接近完整生产平台，而不只是本地 SDK。

---

## 托管与自托管

### Claude Agent SDK

Claude Agent SDK 部署方式是自托管。Claude managed agents 是 Anthropic 的另一个独立产品。

关键点：

- SDK 代码不能直接部署到 Claude managed agents。
- 如果使用 SDK，需要自己托管并构建服务层。
- Claude managed agents 限于 Anthropic 生态。

### Deep Agents

Deep Agents 支持两种模式，并且不需要改代码：

| 模式 | 说明 |
|------|------|
| Managed | 在 LangSmith Managed Deep Agents 中创建、运行、运维 |
| Self-hosted | 使用 `langgraph build` 构建 standalone Docker image，部署到任意环境 |

如果想要跨任意模型供应商的托管 Agent 平台，文档建议使用 LangSmith Fleet。

---

## 模型与供应商绑定

### Claude Agent SDK

Claude Agent SDK 将模型、backend 和部署体验绑定在 Claude / Anthropic 生态中。

支持范围：

- Anthropic Claude。
- Bedrock 上的 Claude。
- Vertex 上的 Claude。
- Azure 上的 Claude。

优点是生态内优化更集中，缺点是模型选择相对受限。

### Deep Agents

Deep Agents 将三件事解耦：

1. 模型供应商。
2. 执行 backend。
3. 部署目标。

这意味着你可以选择：

- Anthropic。
- OpenAI。
- Google。
- OpenRouter。
- Fireworks。
- Baseten。
- Ollama。
- 其他 100+ provider。

### Harness Profiles

Deep Agents 提供 Harness Profiles，用于按 provider 或具体 model 声明式配置 Agent harness 行为。它不是普通模型参数配置，而是模型适配层：当选择某个 provider 或 model 时，Deep Agents 会自动应用对应 profile。

可配置内容包括：

| 能力 | 说明 |
|------|------|
| system prompt | 替换基础 prompt，或追加 provider/model 特定后缀 |
| tool descriptions | 按工具名覆盖工具描述，让不同模型更容易正确调用 |
| excluded tools | 隐藏某些 harness tools，例如 `execute`、`grep`、文件系统工具 |
| excluded / extra middleware | 移除或追加 middleware，调整默认 stack |
| general-purpose subagent | 禁用、重命名或重写默认通用 subagent |

注册 key 有两种：

| Key | 示例 | 作用 |
|-----|------|------|
| Provider-level | `"openai"` | 适用于该 provider 下所有模型 |
| Model-level | `"openai:gpt-5.5"` | 只适用于该具体模型 |

当 provider-level 和 model-level 同时存在时，会合并解析：model-level 显式字段覆盖 provider-level，未设置字段继承。重复注册同一个 key 时也会叠加，而不是完全替换。

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
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    ),
)
```

还要区分 `HarnessProfile` 和 `ProviderProfile`：

| 类型 | 调整对象 |
|------|----------|
| `HarnessProfile` | prompt 组装、工具可见性、middleware、subagent 等 harness 行为 |
| `ProviderProfile` | `init_chat_model` 的构造参数、凭证检查、运行时 kwargs |

Claude Agent SDK 中类似调优通常要在每个模型调用位置用代码配置。

---

## 生态系统差异

### Claude Agent SDK

Claude Agent SDK 面向 Claude 和 Anthropic 产品体系设计。

适合：

- 已经围绕 Claude 构建应用。
- 模型策略明确只使用 Claude。
- 愿意自己搭建托管、认证和多租户层。

### Deep Agents

Deep Agents 属于更广泛的 LangChain 生态：

- 与 LangChain tools、MCP、middleware 结合。
- 使用 LangGraph runtime。
- 与 LangSmith 的 observability、evaluation、deployment 集成。
- 支持托管和自托管。
- 支持多模型供应商。

---

## 选型建议

### 选择 Deep Agents，如果你需要

- 模型供应商灵活性。
- 基础设施灵活性。
- remote sandbox / virtual filesystem / custom backend。
- 内置多租户。
- scoped threads、run history、RBAC。
- 托管和自托管都可选。
- 不改代码从 managed 切到 self-hosted。
- LangSmith observability、evaluation、deployment 集成。
- 跨团队、跨用户、面向生产的 Agent 平台能力。

### 选择 Claude Agent SDK，如果你

- 已经深度投入 Anthropic / Claude 生态。
- 明确只使用 Claude 系列模型。
- 偏好 Agent 在 sandbox 内运行的架构。
- 愿意自己构建 API server、auth、streaming、多租户层。
- 不需要 Deep Agents 的 backend 可插拔和跨 provider 部署能力。

### 简化决策树

```
是否只使用 Claude / Anthropic 生态？
├─ 是
│  ├─ 是否愿意自己构建服务层、多租户和 auth？
│  │  ├─ 是：Claude Agent SDK 可考虑
│  │  └─ 否：考虑 Deep Agents / LangSmith Fleet
│
└─ 否
   └─ 需要跨模型、跨 backend、managed/self-hosted 灵活性：Deep Agents
```

---

## 快速参考

### 最大差异

| 问题 | Deep Agents | Claude Agent SDK |
|------|-------------|------------------|
| 能否使用非 Claude 模型 | 可以 | 基本不适合 |
| 能否自带生产 Agent Server | 可以 | 需要自己写 |
| 能否内置多租户 | 可以 | 需要自己做 |
| 能否托管运行 | 可以，Managed Deep Agents / Fleet | SDK 本身不直接部署到 managed agents |
| 能否自托管 | 可以 | 可以 |
| 能否远程使用 sandbox 作为工具 | 可以 | 不支持 |
| 能否使用自定义 backend | 可以 | 不支持或需自行扩展 |

### 生产化关注点

| 生产能力 | Deep Agents | Claude Agent SDK |
|----------|-------------|------------------|
| Streaming endpoints | 内置 | 自建 |
| Thread management | 内置 | 自建 |
| Run history | 内置 | 自建 |
| Webhooks | 内置 | 自建 |
| Authentication | 内置于部署体系 | 自建 |
| RBAC | 内置 | 自建 |
| Per-user sandbox | Harness 支持 | 自建 |

### 一句话总结

- **Deep Agents**：更像跨模型、跨基础设施、面向生产的完整 Agent harness 和部署路径。
- **Claude Agent SDK**：更像 Claude 生态内的 Agent SDK，适合愿意自建服务层的团队。
- 若你看重模型和部署灵活性、内置多租户和 LangSmith 生态，选 Deep Agents。
- 若你只押注 Claude，并且愿意自己搭建生产层，可以考虑 Claude Agent SDK。

> 注：原文说明该对比草拟于 2026-04-16，相关产品能力可能随时间变化；实际选型前建议再查看两边最新文档。
