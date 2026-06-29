# Deep Agents Memory 与 Skills 详细指南

> 基于 Deep Agents 官方 Memory 与 Skills 文档整理。本文聚焦 Deep Agents 如何用 filesystem-backed memory 实现跨会话记忆，如何用 `SKILL.md` 组织按需加载的能力，以及 Memory、Skills、Tools、Backends、Permissions 之间的边界与组合方式。

## 目录

1. [核心概念](#核心概念)
2. [Memory 与 Skills 的区别](#memory-与-skills-的区别)
3. [Memory：长期记忆](#memory长期记忆)
4. [Memory 工作机制](#memory-工作机制)
5. [Scoped Memory：记忆作用域](#scoped-memory记忆作用域)
6. [Advanced Memory：高级用法](#advanced-memory高级用法)
7. [Read-only vs Writable Memory](#read-only-vs-writable-memory)
8. [Skills：按需能力包](#skills按需能力包)
9. [Skills 工作机制](#skills-工作机制)
10. [Skill 目录结构与 SKILL.md](#skill-目录结构与-skillmd)
11. [编写有效 Skills](#编写有效-skills)
12. [Supporting Resources](#supporting-resources)
13. [Backends 与远程加载](#backends-与远程加载)
14. [Subagents 与 Skills](#subagents-与-skills)
15. [Memory、Skills、Tools 对比](#memoryskillstools-对比)
16. [常见架构模式](#常见架构模式)
17. [最佳实践](#最佳实践)
18. [故障排查](#故障排查)
19. [快速参考](#快速参考)
20. [资料来源](#资料来源)

---

## 核心概念

Deep Agents 中，Memory 和 Skills 都是给 agent 提供上下文与能力的机制，但加载时机和使用目的不同。

| 概念 | 核心作用 | 加载方式 |
|------|----------|----------|
| Memory | 让 agent 跨 conversation/session 记住偏好、事实、规则、历史经验 | 通常在启动时加载，或通过文件系统按需读取 |
| Skills | 把特定任务的流程、领域知识、脚本和模板封装成能力包 | progressive disclosure，先看描述，需要时再读完整内容 |

一句话：

```text
Memory 是 agent 总是应该知道或长期保存的信息；
Skills 是 agent 在特定任务出现时才加载的做事方法。
```

Deep Agents 把 memory 和 skills 都建在虚拟文件系统之上：

```text
Agent
  -> memory=["/memories/AGENTS.md"]
  -> skills=["/skills/"]
  -> backend 决定这些文件实际存储在哪里
```

常见存储：

| Backend | 用法 |
|---------|------|
| `StateBackend` | 当前 thread 内短期文件 |
| `StoreBackend` | 跨 thread 的长期 memory / skills |
| `FilesystemBackend` | 本地磁盘上的 memory / skills |
| `CompositeBackend` | `/memories/`、`/skills/`、`/workspace/` 分别路由 |

---

## Memory 与 Skills 的区别

| 维度 | Memory | Skills |
|------|--------|--------|
| 目的 | 持久上下文、偏好、规则、事实、经验 | 可复用工作流、任务流程、领域能力 |
| 常见格式 | `AGENTS.md` 或其他 memory 文件 | 每个 skill 目录下的 `SKILL.md` |
| 加载时机 | 启动时加载到 prompt，或 conversation 中按需读取 | 启动时只加载 name/description，相关时读取完整 `SKILL.md` |
| 上下文成本 | 固定成本较高，适合小而重要的信息 | 按需成本，适合较大的任务说明 |
| 更新方式 | agent 可用 `edit_file` 更新，也可后台 consolidation | 可由开发者维护，也可让 agent 生成/改进 |
| 作用域 | user / agent / organization | user / project / agent / subagent |
| 推荐内容 | 用户偏好、项目约定、组织规则、长期事实 | 处理 PDF、写 SQL、代码审查、跑评测等流程 |

判断口诀：

```text
每次都必须知道 -> Memory
只在某类任务需要 -> Skill
需要执行动作 -> Tool
需要隔离执行过程 -> Subagent
需要跨会话保存文件 -> StoreBackend
```

---

## Memory：长期记忆

Deep Agents 的 Memory 指 long-term memory：跨 conversations 和 sessions 持久保存的信息。

短期记忆与长期记忆的区别：

| 类型 | 范围 | 存储机制 |
|------|------|----------|
| Short-term memory | 单个 thread / conversation | checkpoint + agent state |
| Long-term memory | 跨 thread / conversation | backend，通常是 `StoreBackend` |

Memory 让 agent 可以：

1. 记住用户偏好。
2. 记住 agent 自己学到的经验。
3. 读取组织级政策。
4. 跨对话继续长期任务。
5. 改进自己的工作方式和 skills。

Deep Agents 的特点是 filesystem-backed memory：

```text
Memory 表现为文件；
agent 通过 read_file / edit_file 读写；
backend 控制文件存在哪里、谁能访问。
```

基本配置：

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    memory=["/memories/AGENTS.md"],
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(
                namespace=lambda rt: (rt.server_info.user.identity,),
            ),
        },
    ),
)
```

这里表示：

| 路径 | 作用 |
|------|------|
| `/memories/AGENTS.md` | agent 启动时加载的 memory 文件 |
| `/memories/` route | 由 `StoreBackend` 持久保存 |
| namespace | 按用户隔离 memory |
| default `StateBackend` | 其他临时文件仍按 thread 存储 |

---

## Memory 工作机制

Memory 的流程可以拆成三步。

### 1. 指向 Memory 文件

通过 `memory=` 把文件路径交给 agent：

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    memory=["/memories/AGENTS.md"],
)
```

Memory 文件通常是：

| 文件 | 用途 |
|------|------|
| `/memories/AGENTS.md` | agent/persona/project 级长期规则 |
| `/memories/preferences.md` | 用户偏好 |
| `/policies/compliance.md` | 组织政策 |
| `/memories/research-notes.md` | 长期研究状态 |

### 2. Agent 读取 Memory

Memory 可以：

1. 在启动时注入 system prompt。
2. 在 conversation 中通过文件系统工具读取。
3. 与 skills 搭配：memory 放总是相关的信息，skills 放按需流程。

### 3. Agent 更新 Memory

当 agent 学到新信息时，可以通过内置文件工具更新 memory：

```text
edit_file("/memories/preferences.md", ...)
```

更新策略：

| 策略 | 说明 |
|------|------|
| Hot path | conversation 中直接更新，立即可用，但增加延迟 |
| Background consolidation | 对话后由后台 agent 汇总更新，质量更好但有延迟 |

并非所有 memory 都应该可写。组织政策、合规规则、开发者定义 skills 通常应是只读。

---

## Scoped Memory：记忆作用域

Memory 的关键是 namespace：决定谁能看到同一份 memory。

### Agent-scoped Memory

Agent-scoped memory 让一个 agent 拥有自己的长期身份，所有用户共享。

适合：

| 场景 | 说明 |
|------|------|
| agent 自我改进 | 总结哪些方法有效 |
| 共享知识积累 | 所有用户共用 agent 学到的知识 |
| agent-level skills | agent 可以逐步改进自己的 skills |

配置重点：

```python
StoreBackend(
    namespace=lambda rt: (rt.server_info.assistant_id,),
)
```

示例：

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    memory=["/memories/AGENTS.md"],
    skills=["/skills/"],
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(
                namespace=lambda rt: (rt.server_info.assistant_id,),
            ),
            "/skills/": StoreBackend(
                namespace=lambda rt: (rt.server_info.assistant_id,),
            ),
        },
    ),
)
```

注意：

```text
agent-scoped memory 会被所有用户共享。
如果任何用户可以写入共享 memory，就存在 prompt injection 风险。
```

### User-scoped Memory

User-scoped memory 让每个用户有自己的长期记忆。

适合：

| 场景 | 说明 |
|------|------|
| 用户偏好 | 输出风格、语言、格式 |
| 用户历史上下文 | 长期项目状态、个人资料 |
| 隐私隔离 | Alice 和 Bob 的 memory 不互相泄露 |
| per-user skills | 不同用户有不同技能集 |

配置重点：

```python
StoreBackend(
    namespace=lambda rt: (rt.server_info.user.identity,),
)
```

示例：

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    memory=["/memories/preferences.md"],
    skills=["/skills/"],
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(
                namespace=lambda rt: (rt.server_info.user.identity,),
            ),
            "/skills/": StoreBackend(
                namespace=lambda rt: (rt.server_info.user.identity,),
            ),
        },
    ),
)
```

### Organization-level Memory

Organization-level memory 用于组织共享规则，例如合规政策、品牌语气、内部知识库。

示例：

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    memory=[
        "/memories/preferences.md",
        "/policies/compliance.md",
    ],
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(
                namespace=lambda rt: (rt.server_info.user.identity,),
            ),
            "/policies/": StoreBackend(
                namespace=lambda rt: (rt.context.org_id,),
            ),
        },
    ),
)
```

组织级 memory 通常应为只读，由应用代码或 Store API 更新，而不是让普通 agent 自由修改。

---

## Advanced Memory：高级用法

Memory 可按多个维度设计：

| 维度 | 问题 | 选项 |
|------|------|------|
| Duration | 保存多久 | short-term / long-term |
| Information type | 保存什么 | episodic / procedural / semantic |
| Scope | 谁能看到 | user / agent / organization |
| Update strategy | 何时写入 | hot path / background |
| Retrieval | 如何读取 | startup prompt / on-demand |
| Permissions | agent 能否写 | read-write / read-only |

### Episodic Memory

Episodic memory 保存过去经历：发生了什么、顺序如何、结果如何。

Deep Agents 已经通过 checkpointers 保存 conversation threads。要让过去 conversations 可搜索，可以封装一个工具：

```python
from langgraph_sdk import get_client
from langchain.tools import tool, ToolRuntime

client = get_client(url="<DEPLOYMENT_URL>")


@tool
async def search_past_conversations(query: str, runtime: ToolRuntime) -> str:
    """Search past conversations for relevant context."""
    user_id = runtime.server_info.user.identity
    threads = await client.threads.search(
        metadata={"user_id": user_id},
        limit=5,
    )
    results = []
    for thread in threads:
        history = await client.threads.get_history(thread_id=thread["thread_id"])
        results.append(history)
    return str(results)
```

适合复杂多步骤任务，例如 coding agent 回忆过去 debug 过程，避免重复探索。

### Background Consolidation

默认情况下，agent 可以在 conversation 中直接更新 memory。这叫 hot path。

另一种方式是 background consolidation：对话结束后，由另一个 consolidation agent 汇总近期 conversation，并把重要信息写入 memory。

| 策略 | 优点 | 缺点 |
|------|------|------|
| Hot path | 新 memory 立即可用，用户透明 | 增加对话延迟，agent 要同时处理任务和记忆 |
| Background | 不影响用户延迟，可综合多轮对话 | 新 memory 下一次对话才可用，需要额外 agent |

Consolidation agent 常见职责：

1. 搜索最近 conversations。
2. 提取稳定事实、用户偏好、项目状态。
3. 合并进 memory 文件。
4. 删除过时信息。
5. 保持 memory 简洁。

可以用 cron 定期触发 consolidation agent。Cron schedule 要和查询 lookback window 对齐，避免重复处理或漏处理。

### Concurrent Writes

多个 threads 可并发写 memory，但同一文件可能产生 last-write-wins 冲突。

缓解方式：

| 方法 | 说明 |
|------|------|
| user-scoped memory | 用户通常只有少量并发 conversation，冲突较少 |
| background consolidation | 串行化写入 |
| 按主题拆文件 | 避免所有内容写同一个文件 |
| LangSmith tracing | 审计 memory 写入 |

### Multiple Agents in Same Deployment

如果同一部署里有多个 agents，需要在 namespace 中加入 `assistant_id`：

```python
StoreBackend(
    namespace=lambda rt: (
        rt.server_info.assistant_id,
        rt.server_info.user.identity,
    ),
)
```

如果只需要 per-agent 隔离，不需要 per-user，可只用：

```python
namespace=lambda rt: (rt.server_info.assistant_id,)
```

---

## Read-only vs Writable Memory

默认情况下，如果 backend 和 permissions 允许，agent 可读写 memory 文件。

但共享 memory 通常应该只读，尤其是组织政策、合规规则、开发者定义 skills。

| 类型 | 适合内容 | 更新方式 |
|------|----------|----------|
| Read-write | 用户偏好、agent 自我改进、个人项目状态 | agent 通过 `edit_file` 更新 |
| Read-only | 组织政策、合规规则、共享知识库、开发者定义 skills | 应用代码、Store API、审核流程 |

风险：

```text
如果一个用户能写入其他用户会读取的共享 memory，
恶意用户可能把 prompt injection 写进共享状态。
```

建议：

1. 默认使用 user scope，除非有明确共享需求。
2. 共享政策和组织级 memory 设为 read-only。
3. 对敏感路径写入配置 permissions deny 或 interrupt。
4. 对自定义校验、审计、内容检查使用 backend policy hooks。

只读 memory 示例：

```python
permissions=[
    FilesystemPermission(
        operations=["write"],
        paths=["/policies/**"],
        mode="deny",
    ),
]
```

需要人工审核的 memory 写入：

```python
permissions=[
    FilesystemPermission(
        operations=["write"],
        paths=["/memories/**"],
        mode="interrupt",
    ),
]
```

---

## Skills：按需能力包

Skills 是 Deep Agents 中的 progressive disclosure 能力包。

一个 skill 通常代表：

| 类型 | 示例 |
|------|------|
| 工作流 | 代码审查、写周报、跑评测 |
| 领域能力 | PDF 处理、SQL 编写、金融分析 |
| 工具使用指南 | 何时调用某个 API，如何解释结果 |
| 可执行流程 | 使用脚本抽取数据、验证输出 |
| 风格/格式规范 | 文档模板、输出规范、品牌语气 |

Skills 的核心价值：

```text
不要把所有任务说明都塞进 system prompt；
只在任务相关时加载对应 skill。
```

创建 agent 时传入 skills path：

```python
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    backend=FilesystemBackend(root_dir="/path/to/project", virtual_mode=True),
    skills=["/skills/"],
)
```

如果多个 skill sources 中存在同名 skill，后面的 source 覆盖前面的 source：

```text
last one wins
```

这可以用来实现 base skills + project-specific overrides。

---

## Skills 工作机制

Skills 分三层加载：

| Level | 加载内容 | 何时加载 |
|-------|----------|----------|
| 1. Metadata | `SKILL.md` frontmatter 中的 `name` 和 `description` | Agent startup |
| 2. Instructions | 完整 `SKILL.md` body | Skill 被激活时 |
| 3. Resources | `scripts/`、`references/`、`assets/` 等支持文件 | `SKILL.md` 指示需要时 |

Deep Agents 中：

1. `SkillsMiddleware` 在 agent startup 扫描配置路径。
2. 解析每个 `SKILL.md` 的 frontmatter。
3. 把 name 和 description 注入 system prompt。
4. 当任务匹配 description 时，agent 用 `read_file` 读取完整 `SKILL.md`。
5. Agent 按 `SKILL.md` 指示读取 supporting files 或运行 scripts。

这个机制让 agent 可以拥有很多 skills，但启动 prompt 只承担很少 token 成本。

---

## Skill 目录结构与 SKILL.md

一个 skill 是一个目录，目录中必须有 `SKILL.md`。

推荐结构：

```text
skills/
  langgraph-docs/
    SKILL.md
    scripts/
    references/
    assets/
```

`SKILL.md` 包含：

1. YAML frontmatter。
2. Skill instructions。
3. Supporting resources 的引用说明。

示例：

```markdown
---
name: langgraph-docs
description: Use this skill for requests related to LangGraph in order to fetch relevant documentation to provide accurate, up-to-date guidance.
license: MIT
compatibility: Requires internet access for fetching documentation URLs
metadata:
  author: langchain
  version: "1.0"
allowed-tools: fetch_url
---

# langgraph-docs

Instructions for the agent go here.
```

Frontmatter 字段：

| Field | 必填 | 说明 |
|-------|------|------|
| `name` | 是 | 小写字母数字和 hyphen，1-64 字符，必须匹配父目录名 |
| `description` | 是 | 做什么、什么时候用，最多 1024 字符 |
| `license` | 否 | license 名称或 bundled license 文件 |
| `compatibility` | 否 | 环境要求，最多 500 字符 |
| `metadata` | 否 | 任意 key-value |
| `allowed-tools` | 否 | 预批准工具列表，实验性 |

Deep Agents 限制：

```text
SKILL.md 超过 10 MB 会在 discovery 时被跳过。
```

---

## 编写有效 Skills

### Description 要具体

Discovery 阶段，agent 只看 description 来决定是否加载 skill。

好：

```yaml
description: >-
  Extract text and tables from PDF files, fill PDF forms, and merge
  multiple PDFs. Use when working with PDF documents or when the user
  mentions PDFs, forms, or document extraction.
```

差：

```yaml
description: Helps with PDFs.
```

description 应包含：

| 信息 | 示例 |
|------|------|
| skill 做什么 | Extract text and tables |
| 什么时候用 | Use when user mentions PDFs |
| 关键词 | PDF, forms, extraction, merge |
| 边界 | 与其他相似 skills 区分 |

### Instructions 要可执行

`SKILL.md` body 应写成 agent 能执行的清晰步骤：

1. Step-by-step procedures。
2. Decision criteria。
3. Expected input/output examples。
4. Edge cases。
5. When to read references。
6. When to run scripts。

### 控制体积

建议：

| 项 | 建议 |
|----|------|
| frontmatter | 简洁，避免塞长文 |
| `SKILL.md` body | 尽量小于 5,000 tokens |
| 文件长度 | 遵循 Agent Skills 建议，保持在 500 行以内 |
| 大参考资料 | 放入 `references/` |
| 多个相近 skills | 合并或明确边界 |

### 管理 Skill 数量

少量清晰 skills 通常优于大量重叠 skills。

当 skills 过多且描述相似时，agent 可能：

1. 选错 skill。
2. 犹豫不加载。
3. 加载不必要 skill。
4. 浪费上下文。

优化方式：

| 问题 | 处理 |
|------|------|
| 多个 skills 描述重叠 | 合并 |
| 同一领域多个子任务 | 一个 skill + references 分章节 |
| 主文件过长 | 把细节移动到 supporting files |
| 激活不稳定 | 改写 description，加入关键词和使用时机 |

---

## Supporting Resources

Skill 目录可以包含任意文件，但 Agent Skills spec 定义了三个常用目录。

### `scripts/`

放可执行代码，例如：

1. API clients。
2. 数据转换脚本。
3. 校验脚本。
4. 批处理工具。

脚本建议：

| 建议 | 原因 |
|------|------|
| 自包含 | 减少环境依赖 |
| 明确依赖 | 方便 agent 安装/调用 |
| 错误信息友好 | agent 可根据错误修复 |
| 处理边界情况 | 减少运行失败 |

注意：

```text
Agent 可以从任何 backend 读取 scripts；
但要执行 scripts，通常需要 sandbox backend 或 shell 执行能力。
```

### `references/`

放按需读取的详细文档，例如：

| 文件 | 用途 |
|------|------|
| `REFERENCE.md` | 技术细节 |
| `FORMS.md` | 表单模板 |
| `schema-reference.md` | 数据 schema |
| `error-codes.md` | 错误码 |
| `finance.md` / `legal.md` | 领域指南 |

保持每个 reference 聚焦，避免大而全。

### `assets/`

放静态资源，例如：

1. 文档模板。
2. 配置模板。
3. 示例图片。
4. lookup tables。
5. schemas。

这些资源不是 instructions，只有在 `SKILL.md` 指示时才打开或复制。

### 引用规则

从 `SKILL.md` 引用 supporting files 时，用相对 skill root 的路径：

```markdown
For API details, see [reference guide](references/api-patterns.md).

To extract tables from a PDF, run:
scripts/extract.py
```

建议：

1. 说明每个文件包含什么。
2. 说明什么时候使用。
3. 引用保持一层深。
4. 避免 references 再链到 references，减少多轮读取成本。

---

## Backends 与远程加载

Skills 和 memory 都是文件，因此可以存放在不同 backend 中。

| Backend | 适合场景 |
|---------|----------|
| `StateBackend` | 动态 seed 到当前 thread 的临时 skills |
| `StoreBackend` | 跨 thread / 跨用户 / 长期共享 skills |
| `FilesystemBackend` | 本地项目目录中的 skills |
| `ContextHubBackend` | LangSmith Context Hub 管理 agent repo 和 skill repos |
| Sandbox backend | 需要执行 skill scripts 时 |

### StateBackend seed skill

可以把 skill 文件作为 in-state filesystem seed 传入：

```python
from urllib.request import urlopen
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.utils import create_file_data
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
backend = StateBackend()

skill_content = "...SKILL.md content..."

skills_files = {
    "/skills/langgraph-docs/SKILL.md": create_file_data(skill_content),
}

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    backend=backend,
    skills=["/skills/"],
    checkpointer=checkpointer,
)

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "What is LangGraph?"}],
        "files": skills_files,
    },
    config={"configurable": {"thread_id": "12345"}},
)
```

### SDK 不自动扫描 CLI 目录

官方文档强调：SDK 只加载 `skills=` 中显式传入的 sources。

不会自动扫描：

```text
~/.deepagents/...
~/.agents/...
```

所以如果想复用 CLI 风格目录，需要在 SDK 中显式传入对应 skill paths。

---

## Subagents 与 Skills

Skills 在 subagents 中有继承规则：

| Agent | Skills 行为 |
|-------|-------------|
| Main agent | 使用 `create_deep_agent(skills=[...])` 配置 |
| General-purpose subagent | 自动继承 main agent skills |
| Custom subagent | 默认不继承 main agent skills |
| Custom subagent with `skills` | 使用自己配置的 skills |

示例：

```python
research_subagent = {
    "name": "researcher",
    "description": "Research assistant with specialized skills",
    "system_prompt": "You are a researcher.",
    "tools": [web_search],
    "skills": ["/skills/research/", "/skills/web-search/"],
}

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    skills=["/skills/main/"],
    subagents=[research_subagent],
)
```

结果：

| 对象 | 可见 skills |
|------|-------------|
| Main agent | `/skills/main/` |
| General-purpose subagent | `/skills/main/` |
| `researcher` | `/skills/research/`, `/skills/web-search/` |

---

## Memory、Skills、Tools 对比

| 维度 | Skills | Memory | Tools |
|------|--------|--------|-------|
| Purpose | 按需能力，通过 progressive disclosure 发现 | 启动时加载的持久上下文 | 可调用的程序动作 |
| Loading | 相关时才读完整内容 | agent start 加载 | 每轮都可用 |
| Format | `SKILL.md` in named directories | `AGENTS.md` 或 memory files | Python functions / MCP tools |
| Layering | user/project，last wins | user/project combined | agent 创建时定义 |
| Use when | task-specific 且内容可能较大 | 总是相关的偏好、规则、项目约定 | 需要执行动作或访问外部系统 |

实践中边界不是绝对的：

```text
Skills 也可以作为 progressive-disclosure memory：
agent 把逐渐学到的流程固化成 skill，
以后按需加载，而不是每次都塞进 prompt。
```

---

## 常见架构模式

### 模式一：User-scoped Preferences

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    memory=["/memories/preferences.md"],
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(
                namespace=lambda rt: (rt.server_info.user.identity,),
            ),
        },
    ),
)
```

适合保存每个用户的偏好和长期上下文。

### 模式二：Agent-scoped Self-improvement

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    memory=["/memories/AGENTS.md"],
    skills=["/skills/"],
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(
                namespace=lambda rt: (rt.server_info.assistant_id,),
            ),
            "/skills/": StoreBackend(
                namespace=lambda rt: (rt.server_info.assistant_id,),
            ),
        },
    ),
)
```

适合 agent 在所有用户交互中积累经验，但要谨慎控制写权限。

### 模式三：Org Policies + User Preferences

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    memory=[
        "/memories/preferences.md",
        "/policies/compliance.md",
    ],
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(
                namespace=lambda rt: (rt.server_info.user.identity,),
            ),
            "/policies/": StoreBackend(
                namespace=lambda rt: (rt.context.org_id,),
            ),
        },
    ),
    permissions=[
        FilesystemPermission(
            operations=["write"],
            paths=["/policies/**"],
            mode="deny",
        ),
    ],
)
```

适合组织政策只读、用户偏好可写的产品场景。

### 模式四：Project Skills + Base Skills

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    skills=[
        "/skills/base/",
        "/skills/project/",
    ],
)
```

同名 skill 由后面的 `/skills/project/` 覆盖，实现项目定制。

### 模式五：Skills with Executable Scripts

```text
skills/
  pdf-processing/
    SKILL.md
    scripts/extract_tables.py
    references/pdf-formats.md
```

要求：

1. `SKILL.md` 说明何时运行脚本。
2. Agent 需要有能执行脚本的 backend，例如 sandbox backend。
3. 脚本依赖要清楚。

---

## 最佳实践

### Memory

1. 默认 user-scoped memory，除非确实需要共享。
2. Shared organization memory 应只读。
3. Memory 文件保持简洁，避免把大知识库塞进启动 prompt。
4. 高频变动信息可考虑后台 consolidation。
5. 共享 memory 写入要加 HITL 或 policy hooks。
6. 对并发写入，拆文件或用 consolidation 降低冲突。
7. 用 LangSmith tracing 审计 agent 写入 memory 的行为。

### Skills

1. description 写清 “做什么 + 什么时候用 + 关键词”。
2. 避免多个 skills 描述重叠。
3. `SKILL.md` 只放核心流程，详细资料放 `references/`。
4. 每个 supporting file 都要在 `SKILL.md` 中说明用途。
5. scripts 要自包含，错误信息要可读。
6. 需要执行 scripts 时使用 sandbox backend。
7. 控制 skill 数量，合并相近 skills。
8. 使用 validation tool 检查 frontmatter。

### Memory + Skills

1. 总是相关的规则放 memory。
2. 任务相关流程放 skills。
3. 可写 memory 应 scoped 到 user。
4. 可写 shared skills 要非常谨慎。
5. 让 agent 写 skill 时，最好要求其解释适用场景、输入输出、验证方法。

---

## 故障排查

### Memory 没生效

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| agent 没记住偏好 | memory 路径没传入 `memory=` | 检查 `memory=["..."]` |
| 跨 thread 丢失 | 用了 `StateBackend` | 改用 `StoreBackend` |
| 用户间串记忆 | namespace 没按 user 隔离 | 使用 `rt.server_info.user.identity` |
| shared memory 被污染 | 多用户可写共享路径 | 设置 read-only / HITL / policy hooks |
| 写入冲突 | 多线程写同一文件 | 拆文件或 background consolidation |

### Skill 没被激活

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| agent 不读 `SKILL.md` | description 太模糊 | 加入使用时机和关键词 |
| 选错 skill | 多个 skills 描述重叠 | 区分边界或合并 |
| skill 不在启动 prompt | 没传入 `skills` array | 检查 `skills=[...]` |
| 同名 skill 被覆盖 | later source last wins | 检查 skills source 顺序 |

### Skills Missing at Startup

| 可能原因 | 处理 |
|----------|------|
| path 错误 | 使用正斜杠，路径相对 backend root |
| `SKILL.md` frontmatter 无效 | name 必须匹配父目录 |
| 文件过大 | `SKILL.md` 超过 10 MB 会跳过 |
| StateBackend 未 seed | 用 `invoke(files={...})` 和 `create_file_data()` |

### Supporting Files Not Found

| 可能原因 | 处理 |
|----------|------|
| `SKILL.md` 没引用 | 明确写相对路径和用途 |
| 文件不在 skill 目录 | 保持资源在 skill root 下 |
| sandbox 内没有文件 | 同步 skills 到 sandbox |
| 引用链太深 | 保持一层引用 |

### Scripts Fail to Run

| 可能原因 | 处理 |
|----------|------|
| 只有读权限没有执行环境 | 使用 sandbox backend |
| 依赖未安装 | 在 script 或 skill 中说明依赖 |
| 路径不对 | 使用相对 skill root 的路径 |
| 脚本错误信息不清楚 | 改进脚本异常输出 |

### Subagent Cannot Access Skill

| 可能原因 | 处理 |
|----------|------|
| custom subagent 不继承 main skills | 在 subagent spec 中加 `skills` |
| general-purpose 可访问但 custom 不行 | 这是预期继承规则 |
| skill backend path 不可见 | 检查 subagent 的 backend 和 skills paths |

---

## 快速参考

### Memory 速查

| 需求 | 推荐 |
|------|------|
| 用户偏好 | user-scoped `/memories/preferences.md` |
| agent 自我改进 | agent-scoped `/memories/AGENTS.md` |
| 组织政策 | org-scoped `/policies/compliance.md` + read-only |
| 跨 thread 持久化 | `StoreBackend` |
| 临时 thread 文件 | `StateBackend` |
| 后台更新记忆 | consolidation agent + cron |
| 防止共享 memory 注入 | user scope / read-only / HITL |

### Skills 速查

| 需求 | 推荐 |
|------|------|
| 按需加载流程 | skill directory + `SKILL.md` |
| 发现 skill | frontmatter `name` + `description` |
| 大型参考资料 | `references/` |
| 可执行流程 | `scripts/` + sandbox |
| 模板和静态资源 | `assets/` |
| 覆盖 base skill | later source last wins |
| custom subagent skill | subagent `skills=[...]` |

### Scope 速查

| Scope | Namespace |
|-------|-----------|
| User | `(rt.server_info.user.identity,)` |
| Agent | `(rt.server_info.assistant_id,)` |
| User + Agent | `(rt.server_info.assistant_id, rt.server_info.user.identity)` |
| Organization | `(rt.context.org_id,)` |

### Memory / Skills / Tools 速查

| 问题 | 选什么 |
|------|--------|
| 这个信息每次都相关吗？ | Memory |
| 这是一个任务流程或领域能力吗？ | Skill |
| 需要调用 API、查询 DB、执行动作吗？ | Tool |
| 内容很大但只偶尔需要吗？ | Skill + references |
| 要跨用户共享吗？ | Agent/org-scoped memory，但注意只读和安全 |

### 一句话总结

```text
Memory 让 Deep Agents 跨会话记住重要信息；
Skills 让 Deep Agents 按需加载专业流程；
Memory 适合总是相关的小而关键上下文；
Skills 适合任务相关、可能较大的流程和知识；
Backends 决定它们存在哪里，Permissions 决定 agent 能否修改它们。
```

---

## 资料来源

- Deep Agents Memory 官方页面：https://docs.langchain.com/oss/python/deepagents/memory
- Deep Agents Skills 官方页面：https://docs.langchain.com/oss/python/deepagents/skills
- LangChain Skills 概念附件：`9af8175a-8185-4f56-812e-317a43653437/pasted-text.txt`
- LangChain Long-term memory 附件：`7382fd28-d23b-46e6-ba55-eeabdc9004e3/pasted-text.txt`
