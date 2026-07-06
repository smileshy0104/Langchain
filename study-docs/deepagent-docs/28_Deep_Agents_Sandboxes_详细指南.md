# Deep Agents Sandboxes 详细指南

> 基于 Deep Agents 官方 Sandboxes 文档整理。本文聚焦 sandbox backend 的定位、为什么需要隔离执行环境、如何配置不同 provider、sandbox 生命周期与作用域、两种集成架构、`execute` 工具、文件传输 API、安全边界与最佳实践。

## 目录

1. [核心概念](#核心概念)
2. [Sandboxes 是什么](#sandboxes-是什么)
3. [为什么需要 Sandboxes](#为什么需要-sandboxes)
4. [Sandbox 与 Backend 的关系](#sandbox-与-backend-的关系)
5. [Basic Usage](#basic-usage)
6. [Available Providers](#available-providers)
7. [Lifecycle and Scoping](#lifecycle-and-scoping)
8. [Integration Patterns](#integration-patterns)
9. [How Sandboxes Work](#how-sandboxes-work)
10. [Working with Files](#working-with-files)
11. [Security Considerations](#security-considerations)
12. [与 Permissions / HITL / MCP / LocalShell 的关系](#与-permissions--hitl--mcp--localshell-的关系)
13. [常见架构模式](#常见架构模式)
14. [最佳实践](#最佳实践)
15. [故障排查](#故障排查)
16. [快速参考](#快速参考)
17. [资料来源](#资料来源)

---

## 核心概念

Agents 可能会生成代码、写文件、安装依赖、运行测试、调用 CLI 或访问网络。由于无法完全预测 agent 会执行什么命令，执行环境必须与宿主机隔离。

Sandboxes 提供这种隔离边界：

```text
Agent
  -> sandbox backend
    -> filesystem tools
    -> execute tool
      -> isolated environment
```

与普通 backend 不同，sandbox backend 不只提供文件系统工具，还提供 `execute` 工具。

当配置 sandbox backend 后，agent 可以使用：

| 工具 | 作用 |
|------|------|
| `ls` | 列出 sandbox 内目录 |
| `read_file` | 读取 sandbox 内文件 |
| `write_file` | 写 sandbox 内文件 |
| `edit_file` | 修改 sandbox 内文件 |
| `glob` | 查找 sandbox 内路径 |
| `grep` | 搜索 sandbox 内文件内容 |
| `execute` | 在 sandbox 内运行 shell 命令 |

一句话：

```text
Sandbox 是一种特殊 backend，让 agent 在隔离环境中读写文件和执行命令。
```

---

## Sandboxes 是什么

Sandboxes 是 Deep Agents 中用于安全执行代码的 backend。

它们定义：

1. Agent 的文件系统在哪里。
2. Shell 命令在哪里执行。
3. 执行环境如何与宿主机隔离。
4. 文件如何在宿主应用和 sandbox 之间传输。

对比：

| Backend | 文件操作 | Shell 执行 | 是否隔离宿主机 |
|---------|----------|------------|----------------|
| `StateBackend` | 是 | 否 | 不接触宿主文件系统 |
| `FilesystemBackend` | 是 | 否 | 否，读写真机文件 |
| `LocalShellBackend` | 是 | 是 | 否，命令在宿主机执行 |
| `StoreBackend` | 是 | 否 | 存储层隔离，不执行命令 |
| Sandbox backend | 是 | 是 | 是，在隔离环境执行 |

关键差异：

```text
LocalShellBackend 的 execute 在宿主机运行；
Sandbox backend 的 execute 在隔离环境运行。
```

---

## 为什么需要 Sandboxes

Sandboxes 主要用于安全。

适合场景：

| 场景 | 为什么需要 sandbox |
|------|--------------------|
| Coding agents | 需要 clone repo、安装依赖、运行 tests、调用 git/CLI |
| Data analysis agents | 需要安装 pandas/numpy、运行计算、生成图表或报告 |
| Build/test pipelines | 需要运行不可信代码和脚本 |
| 文件处理 | 需要解压、转换、生成 artifacts |
| 长任务自动执行 | Agent 自主执行命令，风险更高 |

Sandbox 保护宿主系统：

1. Agent 不能读取宿主机本地文件。
2. Agent 不能访问宿主机环境变量。
3. Agent 不能干扰宿主机进程。
4. Agent 命令失败不会破坏宿主机。

但 sandbox 不等于“所有风险消失”。

Sandbox 不能自动防止：

| 风险 | 说明 |
|------|------|
| Context injection | 攻击者可诱导 agent 在 sandbox 内运行恶意命令 |
| Network exfiltration | 如果 sandbox 可联网，数据仍可能被 HTTP/DNS 外发 |
| Sandbox 内 secrets 泄露 | 如果把密钥放进 sandbox，agent 仍可能读到并外传 |
| 恶意输出污染 | Sandbox 产生的文件/输出仍是不可信输入 |

---

## Sandbox 与 Backend 的关系

Deep Agents 把 sandbox 作为 backend 使用：

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    backend=sandbox_backend,
)
```

Sandbox backend 实现 `SandboxBackendProtocol`。

普通 backend 实现：

```text
BackendProtocol
```

Sandbox backend 额外实现：

```text
execute(command: str)
```

Deep Agents harness 会在每次模型调用前检查 backend 是否实现 sandbox protocol：

| 检查结果 | 行为 |
|----------|------|
| backend 实现 `SandboxBackendProtocol` | agent 可见 `execute` 工具 |
| backend 不实现 | `execute` 工具被过滤，agent 看不到 |

这意味着：

```text
execute 工具不是永远存在；
只有 sandbox backend 或 LocalShellBackend 等支持执行的 backend 才会暴露。
```

---

## Basic Usage

基本流程：

1. 用 provider SDK 创建 sandbox/devbox。
2. 包装成 Deep Agents sandbox backend。
3. 把 backend 传给 `create_deep_agent`。
4. 运行 agent。
5. 结束后关闭或删除 sandbox。

以 LangSmith Sandbox 为例：

```python
from deepagents import create_deep_agent
from deepagents.backends import LangSmithSandbox
from langchain_anthropic import ChatAnthropic
from langsmith.sandbox import SandboxClient


client = SandboxClient()
ls_sandbox = client.create_sandbox()
backend = LangSmithSandbox(sandbox=ls_sandbox)

agent = create_deep_agent(
    model=ChatAnthropic(model="claude-sonnet-4-6"),
    system_prompt="You are a Python coding assistant with sandbox access.",
    backend=backend,
)

try:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Create a small Python package and run pytest",
                }
            ]
        }
    )
finally:
    client.delete_sandbox(ls_sandbox.name)
```

注意：

| 步骤 | 说明 |
|------|------|
| 创建 sandbox | 通常由 provider SDK 完成 |
| 包装 backend | 使用 `LangSmithSandbox`、`E2BSandbox` 等 adapter |
| 传给 agent | `backend=backend` |
| 清理资源 | sandbox 会消耗资源和费用，必须关闭 |

---

## Available Providers

官方文档列出可用 provider：

| Provider | Python package / Adapter | 说明 |
|----------|--------------------------|------|
| LangSmith | `langsmith[sandbox]`, `LangSmithSandbox` | LangSmith first-party managed sandbox |
| AgentCore | `langchain-agentcore-codeinterpreter`, `AgentCoreSandbox` | Bedrock AgentCore Code Interpreter |
| Daytona | `langchain-daytona`, `DaytonaSandbox` | Devbox / development environment |
| E2B | `langchain-e2b`, `E2BSandbox` | Code execution sandbox |
| Modal | `langchain-modal`, `ModalSandbox` | Modal sandbox |
| Runloop | `langchain-runloop`, `RunloopSandbox` | Runloop devbox |
| Vercel | `langchain-vercel-sandbox`, `VercelSandbox` | Vercel sandbox |

简化示例：

```python
# Daytona
from daytona import Daytona
from langchain_daytona import DaytonaSandbox

sandbox = Daytona().create()
backend = DaytonaSandbox(sandbox=sandbox)
```

```python
# E2B
from e2b import Sandbox
from langchain_e2b import E2BSandbox

e2b_sandbox = Sandbox.create()
backend = E2BSandbox(sandbox=e2b_sandbox)
```

```python
# Modal
import modal
from langchain_modal import ModalSandbox

app = modal.App.lookup("your-app")
modal_sandbox = modal.Sandbox.create(app=app)
backend = ModalSandbox(sandbox=modal_sandbox)
```

如果官方没有你的 provider，可以实现自己的 sandbox backend。核心是实现 `execute()` 方法。

---

## Lifecycle and Scoping

Sandbox 会消耗资源和成本，必须设计生命周期。

常见作用域：

| Scope | 含义 | 适合 |
|-------|------|------|
| Thread-scoped | 每个 conversation/thread 一个 sandbox | 用户隔离强，适合默认 |
| Assistant-scoped | 同一个 assistant 共享一个 sandbox | 跨对话复用环境、依赖、repo |

### Thread-scoped

每个 conversation 拥有自己的 sandbox。

行为：

1. 第一次运行创建 sandbox。
2. 同一个 thread 后续 turns 复用同一 sandbox。
3. thread 结束或 TTL 到期后 sandbox 被删除/归档。

适合：

| 场景 | 原因 |
|------|------|
| 多用户应用 | 每个 conversation 隔离 |
| 不想跨会话污染状态 | 文件和依赖不会长期积累 |
| 安全优先 | 单个 thread 泄露影响较小 |

示例思路：

```python
async def agent(config):
    thread_id = config["configurable"]["thread_id"]
    sandbox_name = f"thread-{thread_id}"
    ls_sandbox = find_or_create_sandbox(name=sandbox_name, idle_ttl_seconds=3600)
    return create_deep_agent(
        model="google_genai:gemini-3.5-flash",
        backend=LangSmithSandbox(sandbox=ls_sandbox),
    )
```

### Assistant-scoped

同一个 assistant 的所有 threads 复用一个 sandbox。

好处：

1. 安装的包可复用。
2. cloned repositories 可复用。
3. 文件和构建缓存可跨 conversations 保留。

风险：

1. 状态会长期累积。
2. 不同 users/threads 可能互相影响。
3. 磁盘和内存可能无限增长。
4. 如果写入敏感信息，泄露范围更大。

建议：

| 措施 | 说明 |
|------|------|
| 设置 TTL | 空闲后清理 |
| 使用 snapshots | 定期回到干净状态 |
| 实现 cleanup logic | 删除临时文件和缓存 |
| 做用户隔离 | 不要把多用户敏感状态混在一个 sandbox |

---

## Integration Patterns

Deep Agents 有两种 sandbox 集成模式。

### Agent in Sandbox Pattern

Agent 本身运行在 sandbox 内，你从外部通过网络与它通信。

特点：

| 优点 | 说明 |
|------|------|
| 更像本地开发 | Agent 和执行环境紧密耦合 |
| 环境一致性强 | Docker/VM image 中包含 agent framework |

代价：

| 缺点 | 说明 |
|------|------|
| API keys 可能在 sandbox 内 | 安全风险更大 |
| 更新 agent 需要重建 image | 迭代慢 |
| 需要通信层 | WebSocket/HTTP layer |

适合：

1. Provider SDK 自带 agent 通信层。
2. 希望生产环境严格模拟本地开发。
3. Agent 逻辑和环境必须紧密绑定。

### Sandbox as Tool Pattern

Agent 运行在你的服务器或应用中，只有需要执行代码时才通过 sandbox tools 调 provider API。

这是官方文档示例采用的模式。

优点：

| 优点 | 说明 |
|------|------|
| 快速迭代 | 修改 agent 代码不需要重建 image |
| 分离 agent state 和执行环境 | sandbox 失败不丢 agent state |
| API keys 留在 sandbox 外 | 更安全 |
| 可并行多个 sandboxes | 不同任务隔离 |
| 按执行付费 | 只为使用的 sandbox 资源付费 |

代价：

| 缺点 | 说明 |
|------|------|
| 每次 execution 有网络延迟 | 工具调用需要经过 provider API |

选择建议：

```text
需要快速迭代和更安全的密钥边界 -> Sandbox as tool
需要环境高度贴近本地且 provider 管通信 -> Agent in sandbox
```

---

## How Sandboxes Work

### Isolation Boundaries

Sandbox provider 保护宿主系统不受 agent 文件和 shell 操作影响。

Agent 不能：

1. 读取宿主机本地文件。
2. 访问宿主机环境变量。
3. 干扰宿主机上的进程。

但 agent 在 sandbox 内通常拥有较高自由度。

Sandbox 不能防止：

| 风险 | 说明 |
|------|------|
| Context injection | 攻击者诱导 agent 在 sandbox 内执行恶意命令 |
| Network exfiltration | sandbox 联网时可将数据发出 |
| Sandbox 内 secrets 泄露 | agent 可读取 sandbox 内可见 secret |
| 生成恶意 artifact | 输出文件仍需视为不可信 |

### The `execute` Method

Sandbox backend 的核心方法是：

```python
execute(command: str)
```

它运行 shell 命令并返回：

| 返回内容 | 说明 |
|----------|------|
| stdout/stderr | 命令输出 |
| exit code | 成功/失败状态 |
| truncation notice | 输出太大时的截断提示 |

Deep Agents 的 `BaseSandbox` 会基于 `execute()` 构建其他文件系统操作：

| 工具 | 实现方式 |
|------|----------|
| `read_file` | 通过 sandbox 内脚本读取 |
| `write_file` | 通过 sandbox 内脚本写入 |
| `edit_file` | 通过 sandbox 内脚本替换 |
| `ls` | 通过 sandbox 内命令列目录 |
| `glob` | 通过 sandbox 内脚本匹配 |
| `grep` | 通过 sandbox 内命令搜索 |

这意味着实现新 provider 的核心工作通常只是：

```text
实现 execute()，其余工具由 BaseSandbox 处理。
```

如果命令输出很大，结果会自动保存成文件，并提示 agent 使用 `read_file` 分段读取，避免撑爆上下文窗口。

---

## Working with Files

Sandbox 中有两条文件访问通道。

### Agent Filesystem Tools

这些是 LLM 在执行任务时调用的工具：

```text
read_file
write_file
edit_file
ls
glob
grep
execute
```

它们通过 sandbox backend 的 `execute()` 在 sandbox 内运行。

用途：

| 用途 | 示例 |
|------|------|
| Agent 写代码 | `write_file("/src/main.py", "...")` |
| Agent 读文件 | `read_file("/src/main.py")` |
| Agent 跑测试 | `execute("pytest")` |
| Agent 搜索代码 | `grep("TODO", "/src")` |

### File Transfer APIs

这些是应用代码调用的 provider 文件传输 API：

```text
upload_files()
download_files()
```

它们不通过 shell 命令，而是通过 provider native file transfer APIs 跨边界移动文件。

适合：

| API | 用途 |
|-----|------|
| `upload_files()` | agent 运行前 seed 源码、配置、数据 |
| `download_files()` | agent 完成后取回生成代码、报告、构建产物 |

### Seeding the Sandbox

使用 `upload_files()` 预先放入文件。

要求：

```text
paths must be absolute
contents are bytes
```

示例：

```python
backend.upload_files(
    [
        ("/src/index.py", b"print('Hello')\n"),
        ("/pyproject.toml", b"[project]\nname = 'my-app'\n"),
    ]
)
```

适合 seed：

1. 源代码。
2. 测试数据。
3. 配置文件。
4. 依赖锁文件。
5. 模板或资产。

### Retrieving Artifacts

使用 `download_files()` 取回产物：

```python
results = backend.download_files(["/src/index.py", "/output.txt"])
for result in results:
    if result.content is not None:
        print(f"{result.path}: {result.content.decode()}")
    else:
        print(f"Failed to download {result.path}: {result.error}")
```

适合下载：

1. 生成的代码。
2. 测试报告。
3. CSV/JSON 结果。
4. 图表、文档、PPT。
5. 构建产物。

关键区别：

```text
Agent 用 filesystem tools 在 sandbox 内工作；
应用用 upload_files/download_files 跨宿主和 sandbox 边界传文件。
```

---

## Security Considerations

### Sandboxes Protect the Host, Not the Prompt

Sandbox 可以隔离宿主系统，但不能防止 context injection。

攻击者如果控制 agent 输入，仍可能让 agent：

1. 读取 sandbox 内文件。
2. 执行 sandbox 内命令。
3. 通过网络发送 sandbox 内数据。
4. 生成恶意输出文件。

因此：

```text
Sandbox 输出仍然是不可信输入。
```

### Never Put Secrets Inside a Sandbox

官方文档明确建议：不要把 secrets 放进 sandbox。

危险 secrets 包括：

| 类型 | 示例 |
|------|------|
| API keys | OpenAI、GitHub、AWS key |
| Tokens | OAuth token、JWT |
| DB credentials | 数据库用户名密码 |
| Mounted secret files | `.env`、credentials.json |
| Environment variables | `SECRET_KEY`, `AWS_SECRET_ACCESS_KEY` |

原因：

```text
只要 agent 能读取 secret，
context-injected agent 也可能读取并外传。
```

### Handling Secrets Safely

推荐方案一：把 secrets 保留在 sandbox 外部工具中。

```text
Agent 调用工具名；
工具在宿主环境处理认证；
agent 永远看不到 credential。
```

推荐方案二：使用 network proxy 注入凭证。

```text
Sandbox 发普通请求到 proxy；
proxy 添加 Authorization header；
agent 不直接看到 secret。
```

如果必须把 secrets 放入 sandbox，官方也认为这是不推荐的 workaround。最低限度要：

1. 对所有工具调用启用 HITL，而不只是敏感工具。
2. 阻断或限制 sandbox 网络访问。
3. 使用最小权限和最短生命周期 credential。
4. 监控 sandbox 出站网络流量。

即便如此，仍不安全。足够复杂的 context injection 可能绕过输出过滤和 HITL 审查。

### General Security Best Practices

1. Review sandbox outputs before acting on them。
2. 不需要联网时阻断 sandbox 网络访问。
3. 使用 middleware 过滤或脱敏 tool outputs 中的敏感模式。
4. Treat everything produced inside the sandbox as untrusted input。
5. 对高风险 `execute` 行为结合 HITL。
6. 给 sandbox 设置 TTL，避免长期残留状态。
7. 尽量使用 scoped credentials 和外部工具代理。

---

## 与 Permissions / HITL / MCP / LocalShell 的关系

| 概念 | 是否 sandbox | 作用 | 注意 |
|------|--------------|------|------|
| Sandbox backend | 是 | 隔离文件和 shell 执行 | 不防 prompt injection |
| `LocalShellBackend` | 否 | 在宿主机执行 shell | 只用于可信本地开发 |
| Permissions | 否 | 路径级 allow/deny/interrupt | 不适用于 sandbox `execute` |
| HITL | 否 | 工具执行前人工审批 | 审批不是隔离 |
| MCP tools | 否 | 外部工具协议 | 权限取决于 MCP server |
| Backend policy hooks | 否 | 自定义校验/审计 | 可补充 permissions |

关键区别：

```text
Sandbox 解决“在哪里执行”；
Permissions 解决“内置文件工具能访问哪些路径”；
HITL 解决“执行前是否需要人审”；
MCP 解决“如何连接外部工具”；
LocalShell 是宿主机执行，不是 sandbox。
```

注意：Permissions 不适用于 sandbox backends 的 arbitrary command execution，因为 shell 命令可以在 sandbox 内访问文件系统。若需要限制 sandbox 内命令能力，应使用 provider 的隔离能力、网络限制、镜像设计或不要暴露 `execute`。

---

## 常见架构模式

### 模式一：Sandbox as Tool Coding Agent

```python
client = SandboxClient()
ls_sandbox = client.create_sandbox()
backend = LangSmithSandbox(sandbox=ls_sandbox)

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    backend=backend,
    system_prompt=(
        "You are a coding assistant with sandbox access. "
        "You can create and run code in the sandbox."
    ),
)
```

适合：

1. 创建代码。
2. 安装依赖。
3. 运行测试。
4. 生成 artifacts。

### 模式二：Thread-scoped User Sandbox

```python
async def agent(config):
    thread_id = config["configurable"]["thread_id"]
    sandbox_name = f"thread-{thread_id}"
    sandbox = find_or_create_sandbox(name=sandbox_name, idle_ttl_seconds=3600)
    return create_deep_agent(
        model="google_genai:gemini-3.5-flash",
        backend=LangSmithSandbox(sandbox=sandbox),
    )
```

适合多用户应用和隔离 conversation 状态。

### 模式三：Assistant-scoped Persistent Devbox

```python
async def agent(config):
    assistant_id = config["configurable"]["assistant_id"]
    sandbox_name = f"assistant-{assistant_id}"
    sandbox = find_or_create_sandbox(name=sandbox_name)
    return create_deep_agent(
        model="google_genai:gemini-3.5-flash",
        backend=LangSmithSandbox(sandbox=sandbox),
    )
```

适合复用依赖和仓库，但需要 TTL、snapshots 和 cleanup。

### 模式四：Seed Data Then Analyze

```python
backend.upload_files(
    [
        ("/data/input.csv", csv_bytes),
        ("/src/analyze.py", script_bytes),
    ]
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Analyze /data/input.csv and write a report to /out/report.md",
    }]
})

artifacts = backend.download_files(["/out/report.md"])
```

适合数据分析、报告生成、批处理。

### 模式五：Secrets Outside Sandbox

```text
Agent in sandbox
  -> calls host tool: query_private_api
    -> host tool uses API key
    -> returns sanitized result
```

适合需要访问受保护资源但不希望 secret 暴露给 sandbox 的场景。

---

## 最佳实践

### Lifecycle

1. 每个 sandbox 都要有明确关闭/删除逻辑。
2. 使用 TTL 清理 idle sandbox。
3. Thread-scoped 默认更安全。
4. Assistant-scoped 要定期 snapshot/reset/cleanup。
5. 监控资源、费用、磁盘和内存。

### Security

1. 不要把 secrets 放进 sandbox。
2. 需要 authenticated API 时，用 host-side tools 或 network proxy。
3. 不需要网络时禁用 sandbox 网络。
4. Treat sandbox outputs as untrusted。
5. 对高风险命令或外部动作使用 HITL。
6. 对 tool outputs 做敏感信息脱敏。
7. 使用最小权限和短期 credential。

### Files

1. Agent 工作用 filesystem tools。
2. 应用跨边界传文件用 `upload_files()` / `download_files()`。
3. Seed 文件路径用绝对路径。
4. 产物下载后先检查再使用。
5. 大输出让 agent 用 `read_file` 分段读取。

### Architecture

1. 默认选 Sandbox as tool pattern。
2. 需要生产环境完全模拟本地时再考虑 Agent in sandbox。
3. 将 agent state 与 execution environment 分离。
4. 多任务可使用多个 sandbox 并行。
5. provider 选择要考虑生命周期、网络限制、文件传输、成本和观测。

---

## 故障排查

### Agent 看不到 `execute`

| 可能原因 | 处理 |
|----------|------|
| backend 不是 sandbox backend | 使用支持 `SandboxBackendProtocol` 的 backend |
| 使用普通 `StateBackend` / `StoreBackend` | 这些只提供文件工具，不提供 `execute` |
| provider adapter 未正确包装 | 确认使用 `LangSmithSandbox` / `E2BSandbox` 等 |

### Sandbox 中找不到文件

| 可能原因 | 处理 |
|----------|------|
| 没有 seed 文件 | 用 `upload_files()` |
| 上传路径不是绝对路径 | 使用 `/src/file.py` 等绝对路径 |
| agent 写在不同目录 | 用 `ls` / `find` 检查 sandbox 内路径 |
| thread/assistant scope 变了 | 确认复用同一个 sandbox |

### 产物取不回来

| 可能原因 | 处理 |
|----------|------|
| 路径不对 | 让 agent 报告 artifact path |
| 文件未生成 | 检查 `execute` 输出和 exit code |
| sandbox 已关闭 | 在 teardown 前 `download_files()` |
| provider API 返回 error | 查看每个 download result 的 `error` |

### 命令输出太长

| 可能原因 | 处理 |
|----------|------|
| 测试/日志输出巨大 | 输出会被保存到文件，使用 `read_file` 分段读取 |
| agent 仍反复打印 | 在 system prompt 中要求保存日志文件并摘要 |

### Secrets 泄露风险

| 可能原因 | 处理 |
|----------|------|
| secret 放在 sandbox env | 移到 host-side tools |
| sandbox 可联网 | 限制网络或使用 proxy |
| tool output 包含敏感信息 | middleware 脱敏 |
| prompt injection | 不信任 sandbox 内输出，关键动作加 HITL |

### Assistant-scoped Sandbox 状态污染

| 可能原因 | 处理 |
|----------|------|
| 多 thread 共享文件 | 改用 thread-scoped |
| 依赖和缓存长期累积 | 定期 cleanup / snapshot reset |
| 用户数据混在一起 | 分用户 sandbox 或路径隔离 |

---

## 快速参考

### Sandbox 选型

| 需求 | 推荐 |
|------|------|
| LangSmith 原生托管 | `LangSmithSandbox` |
| AWS/Bedrock 生态 | `AgentCoreSandbox` |
| Devbox / git / coding env | `DaytonaSandbox` / `RunloopSandbox` |
| 通用代码执行 | `E2BSandbox` |
| Modal 环境 | `ModalSandbox` |
| Vercel 生态 | `VercelSandbox` |
| 自定义 provider | 实现 `execute()` |

### Scope 速查

| Scope | 优点 | 风险 |
|-------|------|------|
| Thread-scoped | 隔离好，默认安全 | 依赖不能跨 conversation 复用 |
| Assistant-scoped | 复用依赖、repo、缓存 | 状态污染、资源累积、共享风险 |

### 文件通道速查

| 通道 | 谁调用 | 用途 |
|------|--------|------|
| `read_file/write_file/edit_file/ls/glob/grep/execute` | Agent | sandbox 内执行任务 |
| `upload_files()` | 应用代码 | 运行前 seed 文件 |
| `download_files()` | 应用代码 | 运行后取回产物 |

### 安全速查

| 问题 | 结论 |
|------|------|
| Sandbox 能保护宿主机吗 | 能，隔离宿主文件和命令 |
| Sandbox 能防 prompt injection 吗 | 不能 |
| 可以把 secrets 放 sandbox 吗 | 不推荐 |
| Permissions 能限制 sandbox execute 吗 | 不能可靠限制 |
| HITL 是 sandbox 吗 | 不是，是人工审批 |
| MCP tool 是 sandbox 吗 | 不是，取决于 MCP server |
| LocalShellBackend 是 sandbox 吗 | 不是，它在宿主机执行 |

### 一句话总结

```text
Sandboxes 是 Deep Agents 的隔离执行 backend；
它们提供文件系统工具和 execute 工具；
它们保护宿主机，但不防 context injection；
不要把 secrets 放进 sandbox；
应用用 upload_files/download_files 跨边界传文件，
agent 用 filesystem tools 在 sandbox 内工作。
```

---

## 资料来源

- Deep Agents Sandboxes 官方页面：https://docs.langchain.com/oss/python/deepagents/sandboxes
- 已参考本地 Backends / Permissions / Customization 文档中关于 sandbox backend、`execute`、permissions 边界的相关说明。
