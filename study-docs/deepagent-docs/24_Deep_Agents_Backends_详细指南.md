# Deep Agents Backends 详细指南

> 基于 Deep Agents 官方 Backends 文档整理。本文聚焦 Deep Agents 的虚拟文件系统后端：如何选择 backend、如何路由不同路径、如何配置持久化和隔离、如何处理本地文件与 shell 执行、如何设置权限与策略 hooks，以及如何实现自定义 backend。

## 目录

1. [核心概念](#核心概念)
2. [Backend 是什么](#backend-是什么)
3. [Backend 与文件系统工具的关系](#backend-与文件系统工具的关系)
4. [内置 Backend 总览](#内置-backend-总览)
5. [StateBackend](#statebackend)
6. [FilesystemBackend](#filesystembackend)
7. [LocalShellBackend](#localshellbackend)
8. [StoreBackend](#storebackend)
9. [ContextHubBackend](#contexthubbackend)
10. [CompositeBackend](#compositebackend)
11. [Sandbox Backends](#sandbox-backends)
12. [如何指定 Backend](#如何指定-backend)
13. [路径路由与存储分层](#路径路由与存储分层)
14. [自定义 Backend](#自定义-backend)
15. [Permissions：路径权限控制](#permissions路径权限控制)
16. [Policy Hooks：自定义策略控制](#policy-hooks自定义策略控制)
17. [迁移：从 Backend Factory 到 Backend Instance](#迁移从-backend-factory-到-backend-instance)
18. [Backend Protocol 速查](#backend-protocol-速查)
19. [常见架构模式](#常见架构模式)
20. [安全边界与选型建议](#安全边界与选型建议)
21. [排查清单](#排查清单)
22. [快速参考](#快速参考)
23. [资料来源](#资料来源)

---

## 核心概念

Deep Agents 会向 agent 暴露一层虚拟文件系统。Agent 看到的是统一的文件系统工具，但真正的数据读写由 backend 决定。

内置文件系统工具包括：

| 工具 | 作用 |
|------|------|
| `ls` | 列出目录 |
| `read_file` | 读取文件，支持分页；对部分图片文件可返回多模态 content blocks |
| `write_file` | 写入新文件 |
| `edit_file` | 对已有文件做字符串替换 |
| `glob` | 按 glob pattern 查找路径 |
| `grep` | 搜索文件内容 |
| `execute` | 执行 shell 命令；只有 sandbox backend 或 `LocalShellBackend` 提供 |

Backend 决定这些工具实际访问哪里：

```text
Agent
  -> filesystem tools
    -> backend protocol
      -> state / local disk / LangGraph store / Context Hub / sandbox / custom storage
```

一句话：

```text
Deep Agents 的 backend 是文件系统工具背后的存储与执行适配层。
```

---

## Backend 是什么

Backend 在 Deep Agents 里可以理解为：文件系统工具背后的“实际存储/执行层”。

Agent 表面上调用的是这些工具：

```text
ls
read_file
write_file
edit_file
glob
grep
execute
```

但这些工具到底读写哪里，不是工具自己决定的，而是由 backend 决定。

例如：

| Backend | 实际含义 |
|---------|----------|
| `StateBackend` | 文件存在 LangGraph 当前 thread 的 state 里，也是默认方案 |
| `FilesystemBackend` | 文件读写真机本地磁盘目录 |
| `StoreBackend` | 文件存在 LangGraph Store，可跨 thread 持久化 |
| `ContextHubBackend` | 文件存在 LangSmith Context Hub repo |
| `CompositeBackend` | 按路径把不同目录路由到不同 backend |
| `LocalShellBackend` | 本地文件系统 + 本机 shell 执行 |
| Sandbox backend | 隔离环境里的文件系统和命令执行 |

可以这样理解：

```text
Agent 想读文件
  -> 调用 read_file
    -> read_file 问 backend
      -> backend 决定从 state / 本地磁盘 / store / sandbox 哪里读取
```

比如：

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    model="openai:gpt-5.5",
    backend=FilesystemBackend(
        root_dir="/my/project",
        virtual_mode=True,
    ),
)
```

这表示 agent 的 `read_file`、`write_file`、`edit_file`、`glob`、`grep` 等工具会实际操作 `/my/project` 这个本地目录。

如果不指定 backend：

```python
agent = create_deep_agent(
    model="openai:gpt-5.5",
)
```

默认就是：

```python
backend=StateBackend()
```

也就是文件只存在当前会话/thread 的 agent state 里，不会直接写到本地磁盘。

更精确地说：

```text
Backend 是 Deep Agents 的虚拟文件系统后端，
决定 agent 的文件工具实际访问什么存储，
以及是否支持执行命令。
```

---

## Backend 与文件系统工具的关系

Deep Agents 的文件系统能力是“统一接口 + 可插拔实现”：

| 层级 | 说明 |
|------|------|
| Agent | 决定何时读写文件、搜索内容、保存中间结果 |
| Filesystem tools | 提供 `ls/read/write/edit/glob/grep` 等统一工具 |
| Backend | 实现具体存储读写逻辑 |
| Storage / Runtime | 实际的数据位置，例如 state、磁盘、store、Context Hub、sandbox |

这样设计的好处：

| 好处 | 说明 |
|------|------|
| 同一套工具适配不同环境 | 本地开发、云部署、长记忆、沙箱执行都可复用 |
| 可组合 | 用 `CompositeBackend` 把不同路径路由到不同 backend |
| 可隔离 | 用权限、namespace、sandbox 限制读写边界 |
| 可持久化 | 用 `StoreBackend` 或 `ContextHubBackend` 跨 thread 保存文件 |
| 可扩展 | 自定义 backend 可接入数据库、对象存储、远程文件系统 |

多模态补充：

1. `read_file` 对图片文件有原生支持。
2. 文档明确提到 `.png`、`.jpg`、`.jpeg`、`.gif`、`.webp` 可跨 backend 作为多模态 content blocks 返回。
3. 是否能理解图片仍取决于底层模型是否支持对应多模态输入。

---

## 内置 Backend 总览

| Backend | 主要用途 | 持久化范围 | 是否本地磁盘 | 是否有 shell 执行 | 安全备注 |
|---------|----------|------------|--------------|-------------------|----------|
| `StateBackend` | 默认 scratchpad、thread 内文件 | 单 thread，依赖 checkpoint | 否 | 否 | 最安全的默认选择 |
| `FilesystemBackend` | 访问本地目录 | 本地磁盘持久 | 是 | 否 | 直接读写真实文件，需谨慎 |
| `LocalShellBackend` | 本地文件 + 本机 shell | 本地磁盘持久 | 是 | 是 | 无隔离，风险最高之一 |
| `StoreBackend` | LangGraph store 中的长期文件 | 跨 thread 持久 | 否 | 否 | 适合长记忆和云部署 |
| `ContextHubBackend` | LangSmith Context Hub repo | Hub repo 持久 | 否 | 否 | 文件变更以 Hub commit 持久化 |
| `CompositeBackend` | 路由不同路径到不同 backend | 取决于各 route | 混合 | 取决于 route | 推荐用于复杂生产场景 |
| Sandbox backend | 隔离环境中的文件和命令 | 取决于 provider | 隔离环境 | 是 | 生产执行命令时优先考虑 |
| Custom backend | 接入数据库、对象存储等 | 自定义 | 自定义 | 可选 | 需实现协议和策略 |

选型口诀：

```text
默认 scratchpad -> StateBackend
本地项目文件 -> FilesystemBackend + virtual_mode=True
本机命令执行 -> LocalShellBackend，仅限可信开发环境
跨 thread 长期记忆 -> StoreBackend
LangSmith 原生 repo 持久化 -> ContextHubBackend
多存储分层 -> CompositeBackend
生产 shell/code 执行 -> Sandbox backend
外部存储系统 -> Custom backend
```

---

## StateBackend

`StateBackend` 是 Deep Agents 的默认 backend。

最小示例：

```python
from deepagents import create_deep_agent
from deepagents.backends import StateBackend

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
)

# 等价于显式指定：
agent2 = create_deep_agent(
    model="openai:gpt-5.5",
    backend=StateBackend(),
)
```

### 工作方式

| 点 | 说明 |
|----|------|
| 存储位置 | LangGraph agent state |
| 持久化方式 | 同一个 thread 中通过 checkpoint 跨 turn 保留 |
| 跨 thread 共享 | 不共享 |
| 默认行为 | `create_deep_agent` 不传 backend 时自动使用 |
| supervisor / subagents | supervisor 和 subagents 共享同一个 state backend |

### 适合场景

| 场景 | 说明 |
|------|------|
| Agent scratchpad | 保存临时计划、中间结果、草稿 |
| 大工具结果 offloading | 大输出被写入文件后可分段读取 |
| 单 thread 内连续任务 | 同一对话中跨 turn 保留文件 |
| 默认安全选择 | 不直接接触本地真实文件系统 |

### 注意事项

1. `StateBackend` 设计为在 graph run 中使用。
2. 在 graph 外直接调用 backend 方法，例如 `state_backend.upload_files(...)`，不会立即生效，直到 graph 执行为止。
3. Subagent 写入的文件会留在 LangGraph agent state 中，subagent 完成后 supervisor 和其他 subagents 仍能读取。
4. 如果需要跨 thread 持久化，不应只用 `StateBackend`，应使用 `StoreBackend`、`ContextHubBackend` 或 `CompositeBackend`。

---

## FilesystemBackend

`FilesystemBackend` 让 agent 读写真实本地磁盘中某个 `root_dir` 下的文件。

示例：

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    backend=FilesystemBackend(root_dir="", virtual_mode=True),
)
```

### 工作方式

| 点 | 说明 |
|----|------|
| 存储位置 | 本地真实文件系统 |
| 根目录 | `root_dir` |
| 路径限制 | 推荐 `virtual_mode=True` |
| 搜索能力 | 可使用安全路径解析，尽可能防止不安全 symlink traversal；可使用 ripgrep 加速 `grep` |
| 持久化 | 写入真实磁盘，通常永久存在 |

### `virtual_mode=True`

文档强调：如果使用 `root_dir` 做路径限制，应始终开启：

```python
FilesystemBackend(root_dir="/path/to/project", virtual_mode=True)
```

原因：

| 配置 | 结果 |
|------|------|
| `virtual_mode=True` | 对路径做虚拟化与归一化，阻止 `..`、`~` 和 root 外绝对路径 |
| `virtual_mode=False` | 即使设置了 `root_dir`，也不构成可靠安全边界 |

### 适合场景

| 场景 | 说明 |
|------|------|
| 本地开发 CLI | coding assistant、开发工具 |
| CI sandbox | 有明确工作目录与密钥隔离 |
| 挂载持久卷 | 容器或作业环境中访问指定目录 |
| 项目文件修改 | 让 agent 对真实项目文件读写 |

### 不适合场景

| 场景 | 原因 |
|------|------|
| Web server | 用户输入不可信，直接文件读写风险高 |
| 多租户 API | 容易发生越权读取或写入 |
| 生产服务直接挂载敏感目录 | 可能读取 secrets 或破坏文件 |

### 安全风险

| 风险 | 说明 |
|------|------|
| 读取敏感文件 | `.env`、API key、credentials 等 |
| 永久修改文件 | 写入和 edit 是真实磁盘操作 |
| 与网络工具组合泄密 | 例如读取 secrets 后通过网络工具外发 |
| 路径越界 | 未开启 `virtual_mode=True` 时更危险 |

### 推荐防护

1. 在本地或 CI 等受控环境使用。
2. 始终使用 `virtual_mode=True`。
3. 排除 secrets 和敏感目录。
4. 对敏感写入配合 Human-in-the-loop 审批。
5. 生产中需要文件交互时优先考虑 sandbox backend。
6. 多数场景下把 `FilesystemBackend` 包进 `CompositeBackend`，只把项目路径路由到本地磁盘。

### 为什么推荐用 Composite 包裹

Deep Agents 会自动写入内部数据：

| 内部路径 | 内容 |
|----------|------|
| `/large_tool_results/` | 被 offload 的大型工具结果 |
| `/conversation_history/` | 会话历史保存 |

如果直接用 `FilesystemBackend`，这些内部文件也会写入 `root_dir`，和项目文件混在一起。

推荐：

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, FilesystemBackend

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/": FilesystemBackend(
                root_dir="/path/to/project",
                virtual_mode=True,
            ),
        },
    ),
)
```

这样：

| 路径 | Backend |
|------|---------|
| `/workspace/...` | 真实项目目录 |
| `/large_tool_results/...` | `StateBackend` |
| `/conversation_history/...` | `StateBackend` |
| 其他内部文件 | `StateBackend` |

---

## LocalShellBackend

`LocalShellBackend` 在 `FilesystemBackend` 的基础上增加了 `execute` 工具，允许 agent 在宿主机直接执行 shell 命令。

示例：

```python
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    backend=LocalShellBackend(
        root_dir="../langchain-docs",
        virtual_mode=True,
        env={"PATH": "/usr/bin:/bin"},
    ),
)
```

### 工作方式

| 点 | 说明 |
|----|------|
| 基础能力 | 继承文件读写能力 |
| 额外能力 | 提供 `execute` shell 工具 |
| 执行方式 | 使用 `subprocess.run(shell=True)` 在本机执行 |
| 隔离性 | 无 sandbox 隔离 |
| 工作目录 | 使用 `root_dir` 作为命令工作目录 |
| 系统访问 | 命令仍可访问系统其他路径 |
| 默认 timeout | 120 秒 |
| 默认输出上限 | `max_output_bytes=100000` |
| 环境变量 | 支持 `env` 和 `inherit_env` |

### 适合场景

| 场景 | 说明 |
|------|------|
| 本地 coding assistant | 你信任 agent 且环境可控 |
| 个人开发环境 | 快速运行测试、脚本、构建 |
| CI/CD | 需要严格 secret 管理和隔离策略 |

### 不适合场景

| 场景 | 原因 |
|------|------|
| 生产 Web 服务 | 可执行任意命令，风险极高 |
| 多租户系统 | 可能越权读写和执行 |
| 处理不可信用户输入 | 容易被 prompt injection 诱导执行危险命令 |
| 共享主机 | 可能影响其他用户或服务 |

### 安全风险

| 风险 | 说明 |
|------|------|
| 任意命令执行 | agent 可用当前用户权限执行 shell |
| 读取 secrets | 包括 `.env`、凭据文件、SSH key 等 |
| 永久破坏 | 删除文件、修改系统状态、安装依赖等 |
| 资源耗尽 | CPU、内存、磁盘无强隔离 |
| `virtual_mode=True` 不足以保护 shell | shell 命令可访问 root_dir 外路径 |

关键结论：

```text
LocalShellBackend 不是 sandbox。
virtual_mode=True 对文件工具有帮助，但不能限制 shell 命令的系统访问。
```

生产环境如果需要执行代码或 shell，应优先使用 sandbox backend。

---

## StoreBackend

`StoreBackend` 把 Deep Agents 的文件存入 LangGraph `BaseStore`，适合跨 thread 的长期持久化。

示例：

```python
from deepagents import create_deep_agent
from deepagents.backends import StoreBackend
from langgraph.store.memory import InMemoryStore

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    backend=StoreBackend(
        namespace=lambda rt: (rt.server_info.user.identity,),
    ),
    store=InMemoryStore(),
)
```

在 LangSmith Deployment 中：

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    backend=StoreBackend(
        namespace=lambda rt: (rt.server_info.user.identity,),
    ),
)
```

平台会自动 provision store，因此可以省略 `store`。

### 工作方式

| 点 | 说明 |
|----|------|
| 存储位置 | LangGraph `BaseStore` |
| 持久化范围 | 可跨 thread |
| 本地开发 | 可用 `InMemoryStore` |
| 云部署 | LangSmith Deployment 自动提供 store |
| 关键配置 | `namespace` |

### 适合场景

| 场景 | 说明 |
|------|------|
| 长期记忆 | 用户偏好、长期项目资料、跨会话状态 |
| 多 thread 持久化 | 不随单个 conversation 消失 |
| LangSmith 部署 | 平台自动提供 store |
| 统一 durable storage | Redis、Postgres、云 store 等 BaseStore 实现 |

### Namespace Factories

`namespace` 决定 `StoreBackend` 在 store 中读写哪个命名空间。它接收 LangGraph `Runtime`，返回字符串 tuple。

类型：

```python
NamespaceFactory = Callable[[Runtime], tuple[str, ...]]
```

`Runtime` 中常用信息：

| 字段 | 说明 |
|------|------|
| `rt.context` | 用户传入的 runtime context，例如 `user_id` |
| `rt.server_info` | LangGraph Server 元数据，例如 assistant ID、graph ID、authenticated user |
| `rt.execution_info` | 执行信息，例如 thread ID、run ID、checkpoint ID |

常见 namespace 模式：

```python
from deepagents.backends import StoreBackend

# 每个用户独立存储
per_user = StoreBackend(
    namespace=lambda rt: (rt.server_info.user.identity,),
)

# 同一个 assistant 共享存储
per_assistant = StoreBackend(
    namespace=lambda rt: (rt.server_info.assistant_id,),
)

# 每个 thread 独立存储
per_thread = StoreBackend(
    namespace=lambda rt: (rt.execution_info.thread_id,),
)

# 用户 + thread 双重隔离
per_user_thread = StoreBackend(
    namespace=lambda rt: (
        rt.server_info.user.identity,
        rt.execution_info.thread_id,
    ),
)
```

### Namespace 安全与约束

| 点 | 说明 |
|----|------|
| 多用户生产环境 | 必须显式设置 namespace，避免数据共享 |
| 默认 legacy 行为 | 未设置时可能按 assistant ID 共享，同 assistant 用户会共享存储 |
| 字符限制 | 只能包含字母数字、连字符、下划线、点、`@`、`+`、冒号、波浪号 |
| 禁止通配符 | `*`、`?` 被拒绝，避免 glob injection |
| 未来要求 | 文档提示 `namespace` 将成为必需参数 |

---

## ContextHubBackend

`ContextHubBackend` 把 agent 的文件系统存储在 LangSmith Context Hub repo 中。

示例：

```python
from deepagents import create_deep_agent
from deepagents.backends import ContextHubBackend

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    backend=ContextHubBackend("my-agent"),
)
```

repo identifier 支持：

```text
"name"
"owner/name"
```

使用前需要设置：

```bash
export LANGSMITH_API_KEY=...
```

### Repo 结构

Context Hub 中有两类 repo：

| Repo 类型 | 作用 |
|-----------|------|
| Agent repo | 保存 agent 顶层指令和配置，例如 `AGENTS.md`、`tools.json` |
| Skill repo | 保存可复用 skill，例如 `SKILL.md` 和相关资源 |

当使用：

```python
ContextHubBackend("my-agent")
```

Backend 会把 agent repo 挂载到文件系统 root。被链接的 skill repos 会出现在：

```text
/skills/
```

这种设计让：

1. 每个 agent 有自己的顶层上下文。
2. Skill 可独立版本化。
3. Skill 可被多个 agent 复用。
4. 文件变更可通过 Hub commit history 追踪。

### 工作方式

| 点 | 说明 |
|----|------|
| 首次读取 | 懒加载拉取 Hub repo tree |
| 读取缓存 | 之后从内存 cache 服务 |
| 写入 / edit | 以 Hub commit 形式持久化 |
| 并发控制 | 使用 optimistic `parent_commit` |
| repo 不存在 | 首次 pull 视为空；首次成功写入可创建 repo |
| 冲突 | 如果其他 writer 先推进 repo，旧 parent commit 写入可能失败，需要 re-pull 并重试 |
| 文件上传 | `upload_files()` 接受 UTF-8 文本；非 UTF-8 文件按 path 返回 `invalid_path` |

### 适合场景

| 场景 | 说明 |
|------|------|
| LangSmith 原生持久化 | 不想单独配置 LangGraph `BaseStore` |
| 需要文件历史 | 每次变更成为 Hub commit |
| Agent context repo 化 | 指令、工具配置、skills 统一管理 |
| Skill 复用 | 多 agent 链接同一 skill repo |

---

## CompositeBackend

`CompositeBackend` 是路径路由器，可以把不同路径前缀交给不同 backend。

示例：默认使用 `StateBackend`，`/memories/` 使用 `StoreBackend`。

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(
                namespace=lambda _rt: ("memories",),
            ),
        },
    ),
    store=InMemoryStore(),
)
```

### 工作方式

| 点 | 说明 |
|----|------|
| 默认 backend | 未匹配 route 的路径走 `default` |
| 路由规则 | 按 path prefix 匹配 |
| 长前缀优先 | `"/memories/projects/"` 可覆盖 `"/memories/"` |
| 列表和搜索 | `ls`、`glob`、`grep` 聚合各 backend 结果 |
| 路径展示 | 保留原始路径前缀 |

### 适合场景

| 场景 | 示例 |
|------|------|
| 线程内临时文件 + 跨线程记忆 | default `StateBackend`，`/memories/` -> `StoreBackend` |
| 项目文件 + 内部临时文件隔离 | `/workspace/` -> `FilesystemBackend`，default -> `StateBackend` |
| 多知识源统一挂载 | `/docs/`、`/policies/`、`/memories/` 分别路由 |
| 不同权限边界 | 某些路径只读、某些路径可写 |

### 推荐模式

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, FilesystemBackend, StoreBackend

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/": FilesystemBackend(
                root_dir="/path/to/project",
                virtual_mode=True,
            ),
            "/memories/": StoreBackend(
                namespace=lambda rt: (rt.server_info.user.identity,),
            ),
        },
    ),
)
```

路径结果：

| 路径 | Backend | 持久化 |
|------|---------|--------|
| `/workspace/src/app.py` | 本地项目目录 | 本地磁盘 |
| `/memories/preferences.md` | LangGraph store | 跨 thread |
| `/large_tool_results/...` | State | 当前 thread |
| `/conversation_history/...` | State | 当前 thread |

为什么建议 default 用 `StateBackend`：

1. Deep Agents 内部 offloaded tool results 会写到 default backend。
2. Conversation history 也会写到 default backend。
3. 如果 default 是 `FilesystemBackend` 或 `StoreBackend`，内部文件可能污染真实项目目录或长期存储。
4. 用 `StateBackend` 可以让这些内部文件保持 thread-scoped 和 ephemeral。

---

## Sandbox Backends

Backends 文档中把 sandbox 列为快速可用选项之一。Sandbox backend 提供：

| 能力 | 说明 |
|------|------|
| 文件系统工具 | `ls/read/write/edit/glob/grep` |
| `execute` 工具 | 在隔离环境运行 shell 命令 |
| 隔离执行 | 避免命令直接作用于宿主机 |

可选 provider 包括：

```text
LangSmith, AgentCore, Daytona, Deno, E2B, Modal, Runloop, local VFS
```

与 `LocalShellBackend` 的核心区别：

| 对比 | Sandbox backend | LocalShellBackend |
|------|-----------------|-------------------|
| 命令执行位置 | 隔离环境 | 宿主机 |
| 生产适用性 | 更适合生产代码执行 | 不推荐生产 |
| 文件影响范围 | sandbox 内 | 本机真实文件系统 |
| 资源限制 | 取决于 sandbox provider | 本机无强隔离 |
| 安全边界 | 明确隔离 | 无 sandbox 隔离 |

结论：

```text
需要让 agent 执行代码或 shell 时，生产环境优先选 sandbox backend，而不是 LocalShellBackend。
```

---

## 如何指定 Backend

在 `create_deep_agent` 中传入 backend 实例：

```python
from deepagents import create_deep_agent
from deepagents.backends import StateBackend

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    backend=StateBackend(),
)
```

规则：

| 规则 | 说明 |
|------|------|
| 传入 backend instance | 推荐方式 |
| backend 必须实现 `BackendProtocol` | 例如 `StateBackend()`、`FilesystemBackend(...)` |
| 不传 backend | 默认使用 `StateBackend()` |
| 使用 `StoreBackend` | 需要传 `store=` 或由平台自动 provision |

---

## 路径路由与存储分层

路径路由常用于：

1. `/memories/*` 跨 thread 持久化。
2. `/workspace/*` 映射到本地项目目录。
3. 其他路径保持 thread-scoped。

示例：

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, FilesystemBackend

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": FilesystemBackend(
                root_dir="/deepagents/myagent",
                virtual_mode=True,
            ),
        },
    ),
)
```

行为：

| 路径 | 结果 |
|------|------|
| `/workspace/plan.md` | 走 `StateBackend` |
| `/memories/agent.md` | 写入 `/deepagents/myagent` 下的 `FilesystemBackend` |
| `ls` / `glob` / `grep` | 聚合不同 backend 的结果，并显示原始路径前缀 |

路由注意事项：

| 注意 | 说明 |
|------|------|
| 长前缀优先 | 更具体的 route 会覆盖更宽泛的 route |
| StoreBackend 需要 store | 通过 `create_deep_agent(..., store=...)` 传入，或由平台 provision |
| 默认 backend 影响内部文件 | offloaded tool results 和 conversation history 写到 default |
| 生产建议 | default 多用 `StateBackend`，显式 route 长期或真实文件路径 |

---

## 自定义 Backend

如果需要连接数据库、对象存储、远程文件系统、自研知识库等，可以实现自定义 backend。

### 需要实现的协议

自定义 backend 应继承并实现 `BackendProtocol`。

核心方法：

| Method | 作用 |
|--------|------|
| `ls(path)` | 列出目录文件 |
| `read(file_path, offset, limit)` | 读取文件，可分页 |
| `write(file_path, content)` | 创建或写入文件 |
| `edit(file_path, old_string, new_string, replace_all)` | 查找替换 |
| `glob(pattern, path)` | 按 pattern 匹配路径 |
| `grep(pattern, path, glob)` | 搜索文件内容 |

如果还要支持 `execute` 工具，需要实现：

```text
SandboxBackendProtocol
```

它在 `BackendProtocol` 基础上增加 `execute` 方法。

### 错误处理原则

文档明确要求：

```text
失败时返回结构化 result 的 error 字段，不要抛异常。
```

例如：

```python
ReadResult(error="Error: File '/x' not found")
```

这样模型和工具层能收到可解释错误，并尝试恢复。

### S3 风格 Backend 示例骨架

```python
from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)


class S3Backend(BackendProtocol):
    def __init__(self, bucket: str, prefix: str = ""):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")

    def _key(self, path: str) -> str:
        return f"{self.prefix}{path}"

    def ls(self, path: str) -> LsResult:
        ...

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        ...

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        ...

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        ...

    def write(self, file_path: str, content: str) -> WriteResult:
        ...

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        ...
```

### 自定义 Backend 适合场景

| 场景 | 示例 |
|------|------|
| 对象存储 | S3、GCS、OSS |
| 数据库文件视图 | 把数据库记录映射成文件 |
| 远程文件系统 | SFTP、WebDAV、自研文档系统 |
| 企业知识库 | `/docs/`、`/policies/` 挂载内部知识 |
| 审计存储 | 所有写入都有审计记录 |

---

## Permissions：路径权限控制

Deep Agents 支持用 permissions 对内置文件系统工具做声明式读写控制。权限检查发生在调用 backend 之前。

示例：

```python
from deepagents import create_deep_agent, FilesystemPermission
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
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

这个例子表示：

| 路径 | Backend | 权限 |
|------|---------|------|
| `/memories/**` | 用户 namespace 的 StoreBackend | 未显式禁止 |
| `/policies/**` | org namespace 的 StoreBackend | 禁止写入 |
| 其他路径 | StateBackend | 默认规则 |

适合用 permissions 控制：

| 控制目标 | 示例 |
|----------|------|
| 只读目录 | `/docs/**` 禁止 write/edit |
| 禁止修改策略文件 | `/policies/**` deny write |
| 限制 subagent 权限 | 子智能体只能读某些路径 |
| 防止内部文件被改 | 禁止写 `/conversation_history/**` |

注意：

1. Permissions 只控制 Deep Agents 内置文件系统工具。
2. 如果外部 MCP tool 或自定义 tool 自己读写文件，需在工具内部另做权限控制。
3. `LocalShellBackend` 的 shell 命令不受普通文件工具路径权限完整约束，因为命令本身可访问系统路径。

---

## Policy Hooks：自定义策略控制

如果路径 allow/deny 不够，可以通过 subclass 或 wrapper 在 backend 层加自定义策略。

适合场景：

| 场景 | 示例 |
|------|------|
| 审计日志 | 每次 write/edit 记录操作者和内容摘要 |
| 内容检查 | 禁止写入 secrets、PII、恶意脚本 |
| 速率限制 | 限制单位时间写入次数 |
| 企业规则 | 某些前缀必须只读或必须审批 |
| 动态策略 | 根据 runtime context 判断权限 |

### Subclass 示例

```python
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import WriteResult, EditResult


class GuardedBackend(FilesystemBackend):
    def __init__(self, *, deny_prefixes: list[str], **kwargs):
        super().__init__(**kwargs)
        self.deny_prefixes = [
            p if p.endswith("/") else p + "/"
            for p in deny_prefixes
        ]

    def write(self, file_path: str, content: str) -> WriteResult:
        if any(file_path.startswith(p) for p in self.deny_prefixes):
            return WriteResult(error=f"Writes are not allowed under {file_path}")
        return super().write(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        if any(file_path.startswith(p) for p in self.deny_prefixes):
            return EditResult(error=f"Edits are not allowed under {file_path}")
        return super().edit(file_path, old_string, new_string, replace_all)
```

### Wrapper 示例

Wrapper 可以包裹任意 backend：

```python
from deepagents.backends.protocol import (
    BackendProtocol,
    WriteResult,
    EditResult,
    LsResult,
    ReadResult,
    GrepResult,
    GlobResult,
)


class PolicyWrapper(BackendProtocol):
    def __init__(
        self,
        inner: BackendProtocol,
        deny_prefixes: list[str] | None = None,
    ):
        self.inner = inner
        self.deny_prefixes = [
            p if p.endswith("/") else p + "/"
            for p in (deny_prefixes or [])
        ]

    def _deny(self, path: str) -> bool:
        return any(path.startswith(p) for p in self.deny_prefixes)

    def ls(self, path: str) -> LsResult:
        return self.inner.ls(path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return self.inner.read(file_path, offset=offset, limit=limit)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
    ) -> GrepResult:
        return self.inner.grep(pattern, path, glob)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        return self.inner.glob(pattern, path)

    def write(self, file_path: str, content: str) -> WriteResult:
        if self._deny(file_path):
            return WriteResult(error=f"Writes are not allowed under {file_path}")
        return self.inner.write(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        if self._deny(file_path):
            return EditResult(error=f"Edits are not allowed under {file_path}")
        return self.inner.edit(file_path, old_string, new_string, replace_all)
```

Subclass 与 wrapper 的选择：

| 方式 | 适合场景 |
|------|----------|
| Subclass | 针对某个具体 backend 增强，例如 `FilesystemBackend` |
| Wrapper | 对任意 backend 统一加策略，例如审计、过滤、动态权限 |

---

## 迁移：从 Backend Factory 到 Backend Instance

文档指出：从 `deepagents` 0.5.0 开始，backend factory pattern 已 deprecated。

旧写法：

```python
backend=lambda rt: StateBackend(rt)
```

新写法：

```python
backend=StateBackend()
```

### 主要变化

| 旧写法 | 新写法 |
|--------|--------|
| `backend=lambda rt: StateBackend(rt)` | `backend=StateBackend()` |
| `backend=lambda rt: StoreBackend(rt)` | `backend=StoreBackend()` |
| `backend=lambda rt: CompositeBackend(default=StateBackend(rt), ...)` | `backend=CompositeBackend(default=StateBackend(), ...)` |
| `StateBackend(runtime)` | `StateBackend()` |
| `StoreBackend(runtime)` | `StoreBackend()` 或 `StoreBackend(namespace=..., store=...)` |

原因：

1. Backend 现在通过 LangGraph 的 `get_config()`、`get_store()`、`get_runtime()` 等 helper 内部解析 runtime context。
2. 不再需要由外部 factory 把 runtime 传入 backend 构造函数。
3. 工具层也不再需要通过 `Command(update=...)` 包装写入结果。

### 迁移示例

旧写法：

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    backend=lambda rt: CompositeBackend(
        default=StateBackend(rt),
        routes={
            "/memories/": StoreBackend(
                rt,
                namespace=lambda rt: (rt.server_info.user.identity,),
            )
        },
    ),
)
```

新写法：

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(
                namespace=lambda rt: (rt.server_info.user.identity,),
            )
        },
    ),
)
```

### 从 `BackendContext` 迁移

在 `deepagents>=0.5.2` 中，namespace factory 接收 LangGraph `Runtime`，不再接收 `BackendContext` wrapper。

旧写法：

```python
StoreBackend(
    namespace=lambda ctx: (ctx.runtime.context.user_id,),
)
```

新写法：

```python
StoreBackend(
    namespace=lambda rt: (rt.server_info.user.identity,),
)
```

注意：

1. `.runtime` accessor 仍可能兼容，但会发出 deprecation warning。
2. `.state` 没有直接替代；namespace 应来自稳定只读信息。
3. 不建议从可变 state 派生 namespace，因为 state 会随步骤变化，可能导致数据写到不一致的 key 下。
4. 旧 accessor 文档提示会在 `deepagents>=0.7` 移除。

---

## Backend Protocol 速查

自定义 backend 必须实现 `BackendProtocol`。

### Required Methods

| Method | 返回 | 要求 |
|--------|------|------|
| `ls(path: str)` | `LsResult` | 返回目录项；建议按 path 排序 |
| `read(file_path: str, offset=0, limit=2000)` | `ReadResult` | 返回文件数据；文件不存在时返回 error |
| `grep(pattern, path=None, glob=None)` | `GrepResult` | 返回匹配项；失败时返回 error |
| `glob(pattern, path=None)` | `GlobResult` | 返回匹配文件；无结果返回空列表 |
| `write(file_path, content)` | `WriteResult` | create-only；冲突时返回 error |
| `edit(file_path, old_string, new_string, replace_all=False)` | `EditResult` | 默认要求 `old_string` 唯一；找不到返回 error |

### Supporting Types

| 类型 | 字段 |
|------|------|
| `LsResult` | `error`, `entries` |
| `ReadResult` | `error`, `file_data` |
| `GrepResult` | `error`, `matches` |
| `GlobResult` | `error`, `matches` |
| `WriteResult` | `error`, `path`, `files_update` |
| `EditResult` | `error`, `path`, `files_update`, `occurrences` |
| `FileInfo` | `path` 必填；可选 `is_dir`, `size`, `modified_at` |
| `GrepMatch` | `path`, `line`, `text` |
| `FileData` | `content`, `encoding`, `created_at`, `modified_at` |

### 实现注意事项

1. 所有失败都应通过 `error` 字段表达，不要抛异常。
2. `read` 应支持 `offset` 和 `limit`，便于分页读取。
3. `edit` 默认要确保 `old_string` 唯一，除非 `replace_all=True`。
4. `grep` 应返回结构化 `path/line/text`，方便模型定位。
5. `FileData.encoding` 可为 `"utf-8"` 或 `"base64"`。
6. 外部 backend 通常不需要设置 `files_update`。

---

## 常见架构模式

### 模式一：默认线程内 scratchpad

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
)
```

| 项 | 配置 |
|----|------|
| Backend | `StateBackend()` |
| 文件范围 | 当前 thread |
| 跨 thread | 不共享 |
| 适合 | 临时文件、草稿、offloaded results |

### 模式二：本地项目工作区

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/": FilesystemBackend(
                root_dir="/path/to/project",
                virtual_mode=True,
            ),
        },
    ),
)
```

| 项 | 配置 |
|----|------|
| 项目文件 | `/workspace/` -> `FilesystemBackend` |
| 内部文件 | default -> `StateBackend` |
| 安全重点 | `virtual_mode=True`，排除 secrets |
| 适合 | 本地 coding assistant |

### 模式三：长期记忆

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
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

| 项 | 配置 |
|----|------|
| 临时文件 | `StateBackend` |
| 长期记忆 | `/memories/` -> `StoreBackend` |
| 隔离 | per-user namespace |
| 适合 | 用户偏好、跨会话知识 |

### 模式四：Context Hub Agent Repo

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    backend=ContextHubBackend("my-agent"),
)
```

| 项 | 配置 |
|----|------|
| 存储 | LangSmith Context Hub repo |
| 版本历史 | Hub commits |
| Skills | linked skill repos under `/skills/` |
| 适合 | LangSmith 原生上下文管理 |

### 模式五：生产代码执行

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    backend=sandbox,
)
```

| 项 | 配置 |
|----|------|
| 文件 | sandbox 内 |
| 命令 | sandbox `execute` |
| 安全边界 | 隔离环境 |
| 适合 | 生产或不可信输入下的代码执行 |

### 模式六：本地快速 shell 开发

```python
agent = create_deep_agent(
    model="google_genai:gemini-3.5-flash",
    backend=LocalShellBackend(
        root_dir=".",
        virtual_mode=True,
        env={"PATH": "/usr/bin:/bin"},
    ),
)
```

| 项 | 配置 |
|----|------|
| 文件 | 本地真实文件 |
| 命令 | 本机 shell |
| 隔离 | 无 |
| 适合 | 个人可信开发环境 |

---

## 安全边界与选型建议

### Backend 安全级别粗略排序

从更保守到更危险：

```text
StateBackend
-> StoreBackend / ContextHubBackend
-> Sandbox backend
-> FilesystemBackend
-> LocalShellBackend
```

说明：

| Backend | 安全边界 |
|---------|----------|
| `StateBackend` | 不直接接触真实文件系统，scope 为 thread |
| `StoreBackend` | 取决于 namespace 和 store 权限 |
| `ContextHubBackend` | 取决于 LangSmith repo 权限和 commit 流程 |
| Sandbox | 取决于 provider 隔离能力 |
| `FilesystemBackend` | 直接读写真机文件 |
| `LocalShellBackend` | 直接读写文件并执行本机命令 |

### 关键安全结论

1. `FilesystemBackend` 不是 sandbox。
2. `LocalShellBackend` 不是 sandbox。
3. `virtual_mode=True` 只帮助文件路径限制，不能限制 shell 命令访问系统。
4. MCP tools 也不是 sandbox，MCP server 的权限取决于 server 自身运行环境。
5. 生产代码执行优先使用 sandbox backend。
6. 多用户长期存储必须设置 namespace。
7. 路径权限只控制 Deep Agents 内置 filesystem tools，不自动约束外部工具。

### 生产建议

| 需求 | 推荐 |
|------|------|
| 只需要临时文件 | `StateBackend` |
| 需要长期记忆 | `CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend(...)})` |
| 需要访问项目文件 | `CompositeBackend` + `FilesystemBackend(virtual_mode=True)` |
| 需要执行代码 | Sandbox backend |
| 需要本机 shell | 只限可信本地开发，配 HITL |
| 多租户 | StoreBackend namespace 按 user/tenant 隔离 |
| 企业策略 | permissions + backend wrapper |

---

## 排查清单

### 文件不可见

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| 写了文件但下一轮看不到 | 没有 checkpoint 或 thread 不一致 | 检查 thread/config/checkpointer |
| 跨 thread 看不到文件 | 使用了 `StateBackend` | 改用 `StoreBackend` 或 `ContextHubBackend` |
| `/memories/` 没持久化 | route 没配置或 store 未传入 | 检查 `CompositeBackend.routes` 和 `store=` |
| 本地目录读不到 | `root_dir` 不正确 | 使用绝对路径或确认工作目录 |

### 权限与安全

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| Agent 能读到不该读的文件 | `FilesystemBackend` 暴露路径过宽 | 缩小 `root_dir`，开启 `virtual_mode=True`，加 permissions |
| shell 能访问 root_dir 外路径 | 使用 `LocalShellBackend` | 这是预期风险；改用 sandbox |
| 多用户看到同一份 memory | Store namespace 共享 | 按 user/tenant 设置 namespace |
| deny 规则不生效 | 工具不是内置 filesystem tool | 在自定义 tool/MCP server 内做权限控制 |

### Composite 路由

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| 文件写到错误 backend | route prefix 不匹配 | 检查路径是否以 `/prefix/` 开头 |
| 更具体 route 没生效 | prefix 设置不完整 | 确认长前缀和尾部 `/` |
| 内部文件污染项目目录 | default backend 是 `FilesystemBackend` | default 改为 `StateBackend` |
| `StoreBackend` 报错 | 没有 store 或 namespace | 传 `store=`，设置 namespace |

### 自定义 Backend

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| 工具调用崩溃 | backend 抛异常 | 返回 result.error，不抛异常 |
| `edit` 替换错误 | 未检查 old_string 唯一性 | 按协议实现唯一性检查 |
| `read_file` 分页异常 | 未实现 offset/limit | 补齐分页逻辑 |
| `grep` 结果模型不好用 | 返回非结构化文本 | 返回 `path/line/text` |

---

## 快速参考

### Backend 选型

| 需求 | 推荐 Backend |
|------|--------------|
| 默认临时文件 | `StateBackend` |
| 本地真实文件 | `FilesystemBackend(root_dir=..., virtual_mode=True)` |
| 本机 shell | `LocalShellBackend`，仅开发环境 |
| 跨 thread 长期文件 | `StoreBackend` |
| LangSmith Context Hub | `ContextHubBackend` |
| 多路径混合存储 | `CompositeBackend` |
| 生产命令执行 | Sandbox backend |
| 接入 S3/DB/远程系统 | Custom backend |

### 路径规划

| 路径 | 推荐用途 | 推荐 Backend |
|------|----------|--------------|
| `/workspace/` | 项目文件 | `FilesystemBackend` |
| `/memories/` | 长期记忆 | `StoreBackend` |
| `/docs/` | 只读文档 | `StoreBackend` / custom backend / Context Hub |
| `/policies/` | 组织策略 | `StoreBackend` + deny write |
| `/large_tool_results/` | 自动 offload 大工具结果 | `StateBackend` |
| `/conversation_history/` | 会话历史 | `StateBackend` |
| `/skills/` | Context Hub linked skills | `ContextHubBackend` |

### 安全速查

| 配置 | 是否 sandbox | 备注 |
|------|--------------|------|
| `StateBackend` | 否 | 但不访问真实文件系统 |
| `FilesystemBackend` | 否 | 真实磁盘读写；开启 `virtual_mode=True` |
| `LocalShellBackend` | 否 | 本机命令执行，无隔离 |
| `StoreBackend` | 否 | 依赖 namespace 隔离 |
| `ContextHubBackend` | 否 | 依赖 LangSmith repo 权限 |
| Sandbox backend | 是 | 用于隔离执行 |
| MCP tools | 否 | 权限取决于 MCP server |

### 一句话总结

```text
Backend 决定 Deep Agents 的文件系统工具实际读写哪里；
CompositeBackend 用路径把临时文件、项目文件和长期记忆分层；
FilesystemBackend 与 LocalShellBackend 都不是 sandbox；
生产中涉及 shell/code 执行，应优先使用 sandbox backend。
```

---

## 资料来源

- Deep Agents Backends 文档：附件 `a53fb4f2-d8e5-4dab-bb17-36e1c7c20737/pasted-text.txt`。
