# Deep Agents Overview（深度智能体）详细指南

> 基于官方文档 Deep Agents overview 整理的中文增强版。本文聚焦 Deep Agents 的定位、Quickstart、核心能力、执行环境、上下文管理、任务委派、人类控制，以及它与 LangChain / LangGraph 的关系。

## 目录

1. [概述](#概述)
2. [Deep Agents 是什么](#deep-agents-是什么)
3. [Quickstart](#quickstart)
4. [核心能力总览](#核心能力总览)
5. [Execution Environment](#execution-environment)
6. [Context Management](#context-management)
7. [Delegation](#delegation)
8. [Steering](#steering)
9. [与 LangChain / LangGraph 的关系](#与-langchain--langgraph-的关系)
10. [适用场景与选型](#适用场景与选型)
11. [最佳实践](#最佳实践)
12. [快速参考](#快速参考)

---

## 概述

Deep Agents 是构建复杂 LLM Agent 和应用的高层 harness。它在基础 tool calling loop 之上，内置了计划、文件系统、子智能体、长期记忆、上下文管理、人类审批等能力，让 Agent 更适合真实复杂任务。

```
┌─────────────────────────────────────────────────────────────┐
│                       Deep Agents                           │
│                                                             │
│  目标：更容易构建能完成复杂、多步骤任务的 Agent                │
│                                                             │
│  Built-in capabilities                                      │
│  ├─ Take actions: tools / files / code execution             │
│  ├─ Connect data: memory / skills / domain knowledge         │
│  ├─ Manage context: summarization / offloading / caching     │
│  ├─ Parallelize tasks: subagents / isolated contexts         │
│  ├─ Stay in loop: human approval / interrupts                │
│  └─ Improve over time: update memory / skills / prompts      │
└─────────────────────────────────────────────────────────────┘
```

一句话理解：

**LangChain 提供 Agent 积木，LangGraph 提供持久运行时，Deep Agents 把常用复杂 Agent 能力打包成开箱即用的 agent harness。**

---

## Deep Agents 是什么

Deep Agents 是一个构建在 LangChain 和 LangGraph 之上的独立库。它适合需要长期运行、多步骤推理、文件操作、上下文管理和子任务委派的 Agent。

### 它解决的问题

普通 Agent 往往只解决“模型选择工具并调用”的问题，但真实任务经常还需要：

- 维护待办事项和任务计划。
- 读写文件、整理中间结果。
- 长任务中压缩上下文，避免超出窗口。
- 把复杂任务拆给子智能体并行完成。
- 对危险操作进行人工审批。
- 在不同会话之间保留偏好、规范和长期记忆。

Deep Agents 将这些能力作为默认 harness 能力集成。

### 内置能力

| 能力 | 说明 |
|------|------|
| Take actions in an environment | 通过 tools、文件读写、代码执行与环境交互 |
| Connect to your data | 按需加载 memories、skills、domain knowledge |
| Manage growing context | 对长历史和大结果做摘要与 offloading |
| Parallelize tasks | 将任务委派给通用或专用 subagents |
| Stay in the loop | 在关键决策点暂停，等待人工批准 |
| Improve over time | 根据真实使用更新 memory、skills、prompts |

---

## Quickstart

Deep Agents 的基本入口是 `create_deep_agent`。

### 安装

根据模型提供商安装对应包：

```bash
pip install -qU deepagents langchain-openai
```

也可以使用：

| Provider | 依赖 |
|----------|------|
| Google | `deepagents langchain-google-genai` |
| OpenAI | `deepagents langchain-openai` |
| Anthropic | `deepagents langchain-anthropic` |
| OpenRouter | `deepagents langchain-openrouter` |
| Fireworks | `deepagents langchain-fireworks` |
| Baseten | `deepagents langchain-baseten` |
| Ollama | `deepagents langchain-ollama` |

### 最小示例

```python
from deepagents import create_deep_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_deep_agent(
    model="openai:gpt-5.5",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

### Provider 示例

| Provider | `model` 示例 |
|----------|--------------|
| Google | `google_genai:gemini-3.5-flash` |
| OpenAI | `openai:gpt-5.5` |
| Anthropic | `anthropic:claude-sonnet-4-6` |
| OpenRouter | `openrouter:anthropic/claude-sonnet-4-6` |
| Fireworks | `fireworks:accounts/fireworks/models/qwen3p5-397b-a17b` |
| Baseten | `baseten:zai-org/GLM-5.2` |
| Ollama | `ollama:devstral-2` |

### 与 LangSmith

官方建议使用 LangSmith 追踪请求、调试 Agent 行为和评估输出。准备生产部署时，可以参考 Deep Agents 的 production 和 LangSmith deployment 相关文档。

---

## 核心能力总览

官方将 Deep Agents 称为 **agent harness**。它使用和其他 Agent 框架类似的核心 tool calling loop，但额外内置了让 Agent 更可靠完成真实任务的能力。

### 四大组件

| 组件 | 能力 |
|------|------|
| Execution environment | tools、virtual filesystem、sandbox、REPL/interpreter |
| Context management | skills、memory、summarization、context offloading、prompt caching |
| Delegation | task planning、subagent spawning |
| Steering | human-in-the-loop approval、interrupts、permissions |

```
┌─────────────────────────────────────────────────────────────┐
│                     Deep Agents Harness                      │
│                                                             │
│  Execution Environment                                       │
│  ├─ Tools / MCP                                              │
│  ├─ Virtual filesystem                                       │
│  ├─ Filesystem permissions                                   │
│  └─ Code execution                                           │
│                                                             │
│  Context Management                                          │
│  ├─ Skills                                                   │
│  ├─ Memory                                                   │
│  ├─ Summarization / offloading                               │
│  └─ Prompt caching                                           │
│                                                             │
│  Delegation                                                  │
│  ├─ write_todos                                              │
│  └─ task / subagents                                         │
│                                                             │
│  Steering                                                    │
│  └─ Human approval / interrupts                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Execution Environment

Execution environment 是 Agent 行动的地方。它包含 tools、virtual filesystem、permissions、code execution 和 streaming。

### Tools and MCP

Deep Agents 支持通过 `tools=` 传入：

- 普通 Python 函数。
- LangChain tools。
- 任意 MCP server 暴露的 tools。

示例：

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[search, fetch_page, run_query],
)
```

MCP 支持让 Deep Agents 可以通过标准接口连接：

- 数据库。
- API。
- 文件系统。
- 内部工具。
- 外部服务。

### Virtual Filesystem

Deep Agents 内置可配置虚拟文件系统，支持不同 backend：

| Backend | 说明 |
|---------|------|
| In-memory state | 临时状态中的文件 |
| Local disk | 本地磁盘 |
| LangGraph store | 持久化 store |
| Composite routing | 按路径路由到不同 backend |
| Custom backend | 自定义存储 |

虚拟文件系统被多个能力复用：

- skills。
- memory。
- code execution。
- context management。
- custom tools / middleware。

### 文件系统工具

| Tool | 作用 |
|------|------|
| `ls` | 列出目录文件和元数据 |
| `read_file` | 读取文件，支持行号、offset/limit、大文件分段和多模态文件 |
| `write_file` | 创建新文件 |
| `edit_file` | 精确字符串替换，支持全局替换 |
| `glob` | 按模式查找文件，例如 `**/*.py` |
| `grep` | 搜索文件内容，支持多种输出模式 |
| `execute` | 在 sandbox backend 中运行 shell 命令 |

### 多模态文件支持

`read_file` 支持返回多模态 content blocks。

| 类型 | 扩展名 |
|------|--------|
| Image | `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.heic`, `.heif` |
| Video | `.mp4`, `.mpeg`, `.mov`, `.avi`, `.flv`, `.mpg`, `.webm`, `.wmv`, `.3gpp` |
| Audio | `.wav`, `.mp3`, `.aiff`, `.aac`, `.ogg`, `.flac` |
| File | `.pdf`, `.ppt`, `.pptx` |

### 隐藏默认文件系统工具

如果不想让模型看到默认 filesystem tools，可以注册 harness profile 并设置 `excluded_tools`。

```python
from deepagents import HarnessProfile, register_harness_profile

register_harness_profile(
    "anthropic:claude-sonnet-4-6",
    HarnessProfile(
        excluded_tools=frozenset(
            {"ls", "read_file", "write_file", "edit_file", "glob", "grep"}
        ),
    ),
)
```

注意：

- 不应通过 `excluded_middleware` 移除 `FilesystemMiddleware`。
- 它是默认 middleware stack 的必要脚手架。
- 要隐藏模型可见工具，用 `excluded_tools`。

### Filesystem Permissions

Deep Agents 支持声明式文件权限规则，用于控制 Agent 能读写哪些路径。

每条规则包含：

| 字段 | 说明 |
|------|------|
| `operations` | `"read"` 和/或 `"write"` |
| `paths` | 文件或目录 glob patterns |
| `mode` | `"allow"` 或 `"deny"` |

规则按声明顺序从上到下匹配，**first-match-wins**。如果没有任何规则匹配，默认允许。

适合用途：

- 限制 Agent 只能访问 `/workspace/`。
- 保护 `.env`、credentials 等敏感文件。
- 给 subagents 更窄的访问权限。

注意：

- permissions 不适用于 sandbox backends，因为 sandbox 可通过 `execute` 执行任意命令。
- 需要更复杂的验证逻辑时，用 backend policy hooks。

### Code Execution

Deep Agents 支持两种代码执行方式。

| 方式 | 工具 | 能力 | 适用场景 |
|------|------|------|----------|
| Sandbox backends | `execute` | shell 命令、依赖安装、测试、CLI、OS 文件系统 | 需要真实环境操作 |
| Interpreters | `eval` | QuickJS 运行 JavaScript，轻量程序化层 | 循环、批处理、确定性转换、程序化工具调用 |

选择建议：

- 需要安装依赖、跑测试、调用 CLI：用 sandbox。
- 只需要轻量数据处理和控制流：用 interpreter。

### Streaming

Deep Agents 通过 event streaming 暴露运行过程，支持 typed projections：

- messages。
- tool calls。
- values。
- output。
- delegated tasks。

Deep Agents 还增加了 `stream.subagents`，每个委派任务都有独立 handle，可以分别查看消息、工具调用和嵌套 subagent streams。

---

## Context Management

Context management 控制 Agent 知道什么、如何在 token 限制内长时间运行，以及跨会话保留什么。

### 四层上下文能力

| 层 | 说明 |
|----|------|
| Skills | 按需加载的领域知识和工作流 |
| Memory | 启动时加载的长期偏好、规范和指令 |
| Summarization / offloading | 自动压缩对话历史和大工具结果 |
| Prompt caching | 缓存静态 prompt 段，降低延迟和成本 |

### Skills

Skills 是专门工作流、领域知识和自定义指令的包。

每个 skill：

- 遵循 Agent Skills standard。
- 位于一个目录。
- 包含 `SKILL.md`。
- 可包含 scripts、templates、reference docs、其他资源。

Deep Agents 使用 progressive disclosure 加载 skills：

1. 启动时只读取 `SKILL.md` frontmatter。
2. 当任务需要时，才读取完整 skill 内容。

好处：

- 启动上下文更小。
- 仍然可以按需获得丰富能力。
- 适合大型技能库。

### Memory

Memory 用于跨会话保存持久上下文，例如：

- coding style。
- 用户偏好。
- 项目规范。
- 团队约定。
- 长期指令。

Deep Agents 使用 `AGENTS.md` 作为 memory 文件，通过 `memory` 参数传入。

与 skills 的区别：

| 对比 | Skills | Memory |
|------|--------|--------|
| 加载方式 | 按需渐进加载 | 启动时总是加载 |
| 内容类型 | 专项能力、流程、知识 | 持久偏好和规则 |
| 存储 | skill 目录资源 | configured backend |

Memory 可以存储在：

- `StateBackend`
- `StoreBackend`
- `FilesystemBackend`

Agent 也可以基于交互和反馈更新 memory，让偏好和模式跨 thread 保留。

### Summarization and Context Offloading

Deep Agents 自动管理上下文，使长任务能在 token 限制内运行。

上下文流包含四部分：

| 部分 | 说明 |
|------|------|
| Input context | system prompt、memory、skills、tool prompts |
| Compression | 对话历史和大型中间结果的摘要与 offloading |
| Isolation | subagents 隔离重型子任务，只返回最终结果 |
| Long-term memory | 虚拟文件系统中的持久化信息跨 thread 保存 |

这套机制能减少手工裁剪上下文的工作，并降低 token 使用。

### Prompt Caching

对于 Anthropic models，`create_deep_agent` 会自动对静态 system prompt 段启用 prompt caching。

可缓存内容包括：

- base agent instructions。
- memory。
- skill content。

作用：

- 避免每次调用重复处理相同 tokens。
- 降低长任务的延迟和成本。

其他 provider 可查看对应 middleware caching 支持。

---

## Delegation

Delegation 让 Agent 能把大任务拆成更小、更容易并行或隔离处理的工作单元。

### Task Planning

Deep Agents 内置 `write_todos` 工具，用于维护结构化任务列表。

任务状态：

- `pending`
- `in_progress`
- `completed`

这些任务会持久保存在 agent state 中，帮助 Agent 管理长期、多步骤任务。

### Subagents

Deep Agents 内置 `task` 工具，让主 Agent 可以创建临时子智能体处理子任务。

Subagent 特点：

| 特点 | 说明 |
|------|------|
| Fresh context | 每次调用创建新的 Agent 实例和上下文 |
| Autonomous execution | 子智能体独立运行直到完成 |
| Single handoff | 只返回一份最终报告给主 Agent |
| Configurable strategy | 可用默认 `general-purpose` subagent，也可自定义 |
| Stateless messaging | 子智能体无状态，不能多次向主 Agent 发消息 |
| Token efficiency | 子任务细节隔离，只把压缩结果带回主上下文 |

### 什么时候用 Subagents

适合：

- 长时间运行的子任务。
- 多步骤研究。
- 可并行任务。
- 需要隔离大量上下文的任务。
- 需要不同专业能力的任务。

不适合：

- 简单工具调用。
- 需要持续双向沟通的任务。
- 对中间步骤强依赖主上下文的任务。

### 禁用 Subagents

如果不想暴露 `task` tool，官方建议通过 harness profile 禁用 auto-added subagent，并且不要传同步 subagents。

注意：

- 不要试图通过 `excluded_middleware` 移除 `SubAgentMiddleware`。
- 这会被有意拒绝。

---

## Steering

Steering 负责运行时控制和安全边界。

### Human-in-the-loop

Deep Agents 集成 LangGraph interrupts，可以在敏感 tool calls 前暂停，等待人工批准。

通过 `create_deep_agent` 的 `interrupt_on` 参数启用。

示例：

```python
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    tools=[...],
    interrupt_on={"edit_file": True},
)
```

效果：

- 每次调用 `edit_file` 前暂停。
- 人类可以批准。
- 人类可以补充指导。
- 人类可以修改 tool inputs。

适用场景：

- 破坏性操作。
- 修改文件。
- 删除数据。
- 调用昂贵 API。
- 部署、发布、生产操作。
- 交互式调试。

---

## 与 LangChain / LangGraph 的关系

### LangChain

LangChain 提供构建 Agent 的核心积木，例如：

- models。
- tools。
- messages。
- middleware。
- `create_agent`。

如果你只需要自定义轻量 Agent，不需要 Deep Agents 的内置 harness 能力，可以直接使用 LangChain `create_agent`。

### LangGraph

LangGraph 提供运行时能力：

- durable execution。
- streaming。
- human-in-the-loop。
- state。
- graph workflows。

Deep Agents 使用 LangGraph runtime 来支撑持久执行、流式事件、人类审批等能力。

### Deep Agents

Deep Agents 是更高层的 agent harness：

- 预装文件系统、skills、memory、subagents、planning 等能力。
- 面向复杂真实任务。
- 减少重复搭建 Agent 基础设施。

### 选型关系

| 需求 | 推荐 |
|------|------|
| 简单 tool calling Agent | LangChain `create_agent` |
| 自定义状态图和工作流 | LangGraph |
| 复杂任务、文件系统、子智能体、长期记忆 | Deep Agents |

---

## 适用场景与选型

### 适合 Deep Agents 的任务

| 场景 | 原因 |
|------|------|
| 复杂研究任务 | 需要多步骤、资料检索、结果整理 |
| 编程助手 | 需要读写文件、执行代码、运行测试 |
| 数据分析 Agent | 需要文件、代码执行、长上下文 |
| 多文档处理 | 需要 offload 大结果和读写中间文件 |
| 多子任务并行 | 可用 subagents 隔离并行处理 |
| 需要人工审批 | 可用 interrupts 控制高风险操作 |
| 长期项目助手 | 可用 memory 保留偏好和规范 |

### 不一定需要 Deep Agents 的任务

| 场景 | 更轻选择 |
|------|----------|
| 单轮问答 | 普通 ChatModel |
| 简单函数调用 | LangChain tools + `create_agent` |
| 固定流程业务编排 | LangGraph 自定义 workflow |
| 不需要文件和长期上下文 | 轻量 Agent 足够 |

---

## 最佳实践

### 安全

- 为文件系统设置 permissions。
- 保护 `.env`、密钥、凭证和生产配置。
- 对 `edit_file`、`execute`、外部 API 等敏感工具启用 `interrupt_on`。
- sandbox backend 下不要只依赖 filesystem permissions。
- 对 subagents 使用更窄权限。

### 上下文管理

- 将长期偏好和项目规范放进 `AGENTS.md` memory。
- 将专项流程和领域知识做成 skills。
- 大结果优先写入虚拟文件系统，而不是塞进对话上下文。
- 复杂子任务交给 subagents，让主上下文只接收总结。
- 使用支持 prompt caching 的模型时，利用静态 prompt 缓存降低成本。

### 任务委派

- 主 Agent 负责规划、整合和决策。
- Subagents 负责隔离的研究、分析、实现子任务。
- 对需要多轮互动的子任务谨慎使用 subagent，因为它只返回最终报告。
- 对路径和工具调用重要的任务，配合 LangSmith tracing 观察行为。

### 文件系统

- 明确哪些目录可读、可写。
- 对大文件使用 `read_file` 的 offset / limit。
- 用 `glob` 和 `grep` 快速定位相关文件。
- 如果不希望模型直接操作文件，使用 `excluded_tools` 隐藏文件工具。

### Observability

- 使用 LangSmith 追踪 Deep Agents 请求。
- 关注 tool calls、subagent streams、latency、cost、final output。
- 对关键任务建立 evaluation 数据集。
- 生产前参考 Going to production 和 LangSmith deployment 选项。

---

## 快速参考

### 核心 API

| API / 参数 | 说明 |
|------------|------|
| `create_deep_agent` | 创建 Deep Agent |
| `model` | 模型标识，例如 `openai:gpt-5.5` |
| `tools` | 自定义函数、LangChain tools、MCP tools |
| `system_prompt` | Agent 系统提示词 |
| `permissions` | 文件系统读写权限规则 |
| `memory` | 加载 `AGENTS.md` 等长期记忆 |
| `interrupt_on` | 对指定 tool 启用人工审批 |
| `subagents` | 自定义子智能体 |

### 内置工具速查

| 工具 | 用途 |
|------|------|
| `ls` | 列目录 |
| `read_file` | 读文件 |
| `write_file` | 写新文件 |
| `edit_file` | 编辑文件 |
| `glob` | 按 pattern 找文件 |
| `grep` | 搜索文件内容 |
| `execute` | sandbox 中运行 shell 命令 |
| `eval` | interpreter 中运行 JavaScript |
| `write_todos` | 维护任务计划 |
| `task` | 创建 subagent 处理子任务 |

### 四大能力速查

| 能力 | 关键点 |
|------|--------|
| Execution environment | tools、filesystem、permissions、code execution、streaming |
| Context management | skills、memory、summarization、offloading、prompt caching |
| Delegation | write_todos、subagents、parallel tasks |
| Steering | human-in-the-loop、interrupts、安全审批 |

### 一句话总结

- Deep Agents 是面向复杂真实任务的 Agent harness。
- 它不是替代 LangChain / LangGraph，而是构建在它们之上的高层封装。
- 它默认提供文件系统、任务计划、子智能体、记忆、上下文管理和人工审批。
- 简单 Agent 用 LangChain 就够；复杂长期任务优先考虑 Deep Agents。
