# LangGraph Checkpointers 详细指南

> 基于 LangGraph 官方 Checkpointers 文档整理。本文聚焦 checkpointer 如何把 graph state 保存为 checkpoints，如何支撑 human-in-the-loop、conversation memory、time travel、fault tolerance，以及如何选择、配置或实现自定义 checkpointer。

## 目录

1. [整体理解](#整体理解)
2. [Checkpointer 是什么](#checkpointer-是什么)
3. [为什么需要 Checkpointers](#为什么需要-checkpointers)
4. [Threads](#threads)
5. [Checkpoints](#checkpoints)
6. [Super-steps 与 Pending Writes](#super-steps-与-pending-writes)
7. [Checkpoint Namespace](#checkpoint-namespace)
8. [Get State](#get-state)
9. [StateSnapshot 字段](#statesnapshot-字段)
10. [Get State History](#get-state-history)
11. [查找特定 Checkpoint](#查找特定-checkpoint)
12. [Replay](#replay)
13. [Update State](#update-state)
14. [Durability Modes](#durability-modes)
15. [Optimize Checkpoint Storage](#optimize-checkpoint-storage)
16. [Checkpointer Libraries](#checkpointer-libraries)
17. [Checkpointer Interface](#checkpointer-interface)
18. [Serializer 与 Encryption](#serializer-与-encryption)
19. [自定义 Checkpointer](#自定义-checkpointer)
20. [Row Key 与 Index 设计](#row-key-与-index-设计)
21. [Delta Channel Support](#delta-channel-support)
22. [Conformance Suite](#conformance-suite)
23. [生产实践建议](#生产实践建议)
24. [故障排查](#故障排查)
25. [快速参考](#快速参考)
26. [资料来源](#资料来源)

---

## 整体理解

Checkpointer 是 LangGraph persistence 的核心组件之一，它会把 graph state 按步骤保存为 checkpoints。

```text
Graph execution
  -> super-step 0
      -> checkpoint
  -> super-step 1
      -> checkpoint
  -> super-step 2
      -> checkpoint

所有 checkpoints 按 thread_id 组织。
```

一句话：

```text
Checkpointer 让 LangGraph 的执行过程可以被暂停、恢复、回放、调试和容错。
```

它支撑：

| 能力 | 作用 |
|------|------|
| Human-in-the-loop | 暂停 graph，让人检查/审批 state，再恢复 |
| Memory | 同一 thread 的多轮交互保留上下文 |
| Time travel | 回到历史 checkpoint 重新执行或 fork |
| Fault tolerance | 失败后从最近成功 checkpoint 恢复 |
| Pending writes | 并行 super-step 中已成功节点不必重跑 |

---

## Checkpointer 是什么

官方定义：

```text
LangGraph checkpointers save graph state as checkpoints at each step,
enabling persistence, human-in-the-loop, and fault-tolerant execution.
```

编译时配置：

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
```

调用时必须传 `thread_id`：

```python
config = {"configurable": {"thread_id": "1"}}
graph.invoke(input_state, config)
```

核心对象关系：

```text
thread_id
  ├─ checkpoint_id A
  │   ├─ state snapshot
  │   └─ pending writes
  ├─ checkpoint_id B
  │   ├─ state snapshot
  │   └─ pending writes
  └─ checkpoint_id C
      ├─ state snapshot
      └─ pending writes
```

---

## 为什么需要 Checkpointers

### Human-in-the-loop

HITL 需要：

1. 人能查看当前 graph state。
2. graph 能在 interrupt 后暂停。
3. 人类做出审批/修改后，graph 能从暂停点恢复。

这些都依赖 checkpointer。

### Memory

同一个 thread 中的多轮交互可以共享之前 state：

```text
thread_id = "conversation-123"
turn 1 -> 保存 messages
turn 2 -> 读取 turn 1 的 messages 后继续
turn 3 -> 继续累积
```

### Time travel

Checkpoints 让你可以：

| 能力 | 说明 |
|------|------|
| inspect | 查看历史 state |
| replay | 从历史 checkpoint 重新执行后续步骤 |
| fork | 从某个 checkpoint 创建另一条执行轨迹 |

### Fault tolerance

如果节点失败：

```text
不用从头跑整个 graph；
可以从最近成功 checkpoint 恢复。
```

### Pending writes

在同一个 super-step 中，如果多个节点并行执行：

```text
node A 成功
node B 失败
```

LangGraph 会保存 node A 的 pending writes。恢复时 node A 不必重跑。

---

## Threads

Thread 是 checkpoints 的组织单位。

```text
thread = 一串 graph runs 的累积状态
```

调用时配置：

```python
{"configurable": {"thread_id": "1"}}
```

`thread_id` 的作用：

| 用途 | 说明 |
|------|------|
| 保存 checkpoint | checkpointer 用它作为主键的一部分 |
| 读取历史 state | `graph.get_state(config)` |
| 恢复 interrupt | 从该 thread 加载暂停状态 |
| 多轮 memory | 后续 run 写入同一 thread |

没有 `thread_id` 的后果：

```text
checkpointer 不知道把 state 存到哪里；
也不知道 resume 时从哪里加载。
```

实践建议：

| 场景 | thread_id |
|------|-----------|
| 聊天会话 | conversation UUID |
| 工单处理 | ticket/job UUID |
| 后台任务 | task/run UUID |
| 人工审批流 | approval workflow ID |

不要把所有用户会话都塞进同一个 thread。长期用户记忆应使用 Store，而不是 thread state。

---

## Checkpoints

Checkpoint 是 thread 在某个时间点的 graph state snapshot。

它包含：

| 内容 | 说明 |
|------|------|
| channel values | state 中各 channel 的值 |
| next nodes | 下一步要执行的节点 |
| config | `thread_id`、`checkpoint_ns`、`checkpoint_id` |
| metadata | source、writes、step 等执行元数据 |
| parent config | 上一个 checkpoint |
| tasks | 当前 step 的任务、错误、interrupts、subgraph state |

一个顺序图：

```text
START -> node_a -> node_b -> END
```

会有 4 个 checkpoint：

| 顺序 | checkpoint 内容 |
|------|-----------------|
| 1 | empty checkpoint，next = START |
| 2 | 输入 state，next = node_a |
| 3 | node_a 输出后，next = node_b |
| 4 | node_b 输出后，next = empty |

如果 state channel 有 reducer，例如：

```python
bar: Annotated[list[str], add]
```

那么 `bar` 会累积多个节点输出，而不是后写覆盖前写。

---

## Super-steps 与 Pending Writes

Super-step 是 graph 的一个执行 tick：同一个 super-step 中被调度的节点会执行，可能并行。

LangGraph 在 super-step 边界保存完整 checkpoint。

此外，它还保存 node/task 级别 writes：

```text
checkpoint_writes table:
  checkpoint_id
  task_id
  channel
  value
```

这就是 pending writes 的基础。

为什么重要：

| 场景 | 没有 pending writes | 有 pending writes |
|------|---------------------|-------------------|
| 并行节点中一个失败 | 所有并行节点可能重跑 | 成功节点的输出保留 |
| 外部 API 有副作用 | 可能重复调用 | 降低重复调用风险 |
| 大模型调用昂贵 | 成本重复 | 成功结果复用 |

注意：

```text
task writes 不是完整 StateSnapshot。
time travel 仍然从 super-step 边界的完整 checkpoint 恢复。
```

---

## Checkpoint Namespace

每个 checkpoint 有 `checkpoint_ns` 字段，用来标识它属于 root graph 还是 subgraph。

| `checkpoint_ns` | 含义 |
|-----------------|------|
| `""` | root graph |
| `"node_name:uuid"` | 某个作为节点调用的 subgraph |
| `"outer:uuid|inner:uuid"` | 嵌套 subgraphs |

在 node 中读取：

```python
from langchain_core.runnables import RunnableConfig

def my_node(state: State, config: RunnableConfig):
    checkpoint_ns = config["configurable"]["checkpoint_ns"]
```

使用场景：

| 场景 | 作用 |
|------|------|
| 调试 subgraph state | 区分父图和子图 checkpoint |
| 多层 subgraph | 通过 namespace path 定位 |
| 自定义 checkpointer | 作为 row key 的一部分 |

---

## Get State

读取最新 state：

```python
config = {"configurable": {"thread_id": "1"}}
snapshot = graph.get_state(config)
```

读取指定 checkpoint：

```python
config = {
    "configurable": {
        "thread_id": "1",
        "checkpoint_id": "1ef663ba-28fe-6528-8002-5a559208592c",
    }
}
snapshot = graph.get_state(config)
```

必须指定 thread identifier。

如果指定 `checkpoint_id`：

```text
读取该 thread 下的特定 checkpoint。
```

如果不指定：

```text
读取该 thread 最新 checkpoint。
```

---

## StateSnapshot 字段

`graph.get_state()` 返回 `StateSnapshot`。

| Field | Type | Description |
|-------|------|-------------|
| `values` | `dict` | 当前 checkpoint 的 state channel values |
| `next` | `tuple[str, ...]` | 下一步要执行的节点，空 tuple 表示 graph 完成 |
| `config` | `dict` | 包含 `thread_id`、`checkpoint_ns`、`checkpoint_id` |
| `metadata` | `dict` | 包含 `source`、`writes`、`step` |
| `created_at` | `str` | checkpoint 创建时间 |
| `parent_config` | `dict \| None` | 前一个 checkpoint config |
| `tasks` | `tuple[PregelTask, ...]` | 当前 step 任务，含 error、interrupts、subgraph state 等 |

`metadata["source"]` 常见值：

| Source | 含义 |
|--------|------|
| `"input"` | 初始输入 |
| `"loop"` | graph 正常执行产生 |
| `"update"` | `update_state` 人工更新产生 |

---

## Get State History

读取 thread 的完整历史：

```python
config = {"configurable": {"thread_id": "1"}}
history = list(graph.get_state_history(config))
```

特点：

```text
返回 StateSnapshot 列表；
通常最新 checkpoint 在前。
```

可以用于：

| 目标 | 做法 |
|------|------|
| 查看历史执行路径 | 遍历 `history` |
| 找某个 step | 看 `snapshot.metadata["step"]` |
| 找 interrupt 点 | 检查 `snapshot.tasks[*].interrupts` |
| 找某节点执行前 | 看 `snapshot.next` |
| 找人工更新 | `metadata["source"] == "update"` |

---

## 查找特定 Checkpoint

示例：

```python
history = list(graph.get_state_history(config))

# Find the checkpoint before a specific node executed
before_node_b = next(s for s in history if s.next == ("node_b",))

# Find a checkpoint by step number
step_2 = next(s for s in history if s.metadata["step"] == 2)

# Find checkpoints created by update_state
forks = [s for s in history if s.metadata["source"] == "update"]

# Find the checkpoint where an interrupt occurred
interrupted = next(
    s for s in history
    if s.tasks and any(t.interrupts for t in s.tasks)
)
```

排查技巧：

| 你要找 | 看哪个字段 |
|--------|------------|
| 当前卡在哪 | `snapshot.next` |
| 哪个节点写了 state | `snapshot.metadata["writes"]` |
| 是否有人为更新 | `snapshot.metadata["source"]` |
| 是否 interrupt | `snapshot.tasks[].interrupts` |
| 子图状态 | `snapshot.tasks[].state`，需 `subgraphs=True` |

---

## Replay

Replay 是从历史 checkpoint 重新执行后续步骤。

语义：

```text
checkpoint 之前的节点：跳过，直接使用保存结果
checkpoint 之后的节点：重新执行
```

会重新执行：

| 类型 | 说明 |
|------|------|
| LLM calls | 会重新调用模型 |
| API requests | 会重新调用外部 API |
| interrupts | 会重新触发 |
| tools | 如果在 checkpoint 之后，也会重跑 |

适合：

| 场景 | 用途 |
|------|------|
| 调试 | 从某一步复现问题 |
| time travel | 比较不同后续路径 |
| fork | 基于旧状态探索替代执行 |

注意：

```text
Replay 可能重复外部副作用。
对发送邮件、扣款、写数据库等 action 节点要有幂等设计。
```

---

## Update State

`update_state` 可以编辑 graph state。

特点：

| 特点 | 说明 |
|------|------|
| 创建新 checkpoint | 不修改原 checkpoint |
| 走 reducer | 有 reducer 的 channel 会累积 |
| 可指定 `as_node` | 控制这次更新被视为哪个节点写入 |
| 可影响下一步 | `as_node` 会影响后续执行路径 |

用途：

| 场景 | 示例 |
|------|------|
| 人工修正 | 修改错误分类结果 |
| fork 实验 | 从历史状态创建新路径 |
| 调试 | 注入某个节点输出 |
| HITL | 审批后更新草稿或工具参数 |

注意：

```text
update_state 产生 metadata.source = "update" 的 checkpoint。
```

---

## Durability Modes

LangGraph 支持三种 durability mode，在执行时指定：

```python
graph.stream(
    {"input": "test"},
    durability="sync",
)
```

| Mode | 保存时机 | 优点 | 风险 |
|------|----------|------|------|
| `"exit"` | graph 退出时保存，包括成功、错误或 interrupt | 性能最好 | 中途进程崩溃无法恢复 |
| `"async"` | 下一步执行时异步保存 | 性能和可靠性平衡 | crash 时可能丢最新写入 |
| `"sync"` | 下一步开始前同步保存 | 最可靠 | 性能开销较高 |

选择建议：

| 场景 | 推荐 |
|------|------|
| 普通开发 | 默认或 `async` |
| 长任务但可接受中途丢失 | `exit` |
| HITL / 关键业务 | `sync` |
| 外部副作用前后 | `sync` + 幂等设计 |
| 高吞吐 agent | `async` |

---

## Optimize Checkpoint Storage

默认情况下，checkpoint 会在每个 super-step 写入所有 state channel 的完整值。

对于长线程，尤其是累计型 channel，如 messages：

```text
checkpoint 1: [m1]
checkpoint 2: [m1, m2]
checkpoint 3: [m1, m2, m3]
...
```

这会导致存储增长明显。

优化方式：

| 方式 | 说明 |
|------|------|
| pruning | 定期删除旧 checkpoints |
| retention policy | 保留最近 N 天或 N 个 checkpoint |
| state 瘦身 | 不把大对象放进 state |
| `DeltaChannel` | 对 append-heavy channel 只保存增量 |

`DeltaChannel`：

```text
不在 checkpoint blob 中保存完整累计值；
只保存增量或快照，通过 ancestor writes 重建。
```

注意：

```text
DeltaChannel 需要 langgraph>=1.2，目前是 beta，API 可能变化。
```

---

## Checkpointer Libraries

官方 checkpointer 实现：

| 包 | 实现 | 适合场景 |
|----|------|----------|
| `langgraph-checkpoint` | `BaseCheckpointSaver`、`SerializerProtocol`、`InMemorySaver` | 基础接口、实验 |
| `langgraph-checkpoint-sqlite` | `SqliteSaver`、`AsyncSqliteSaver` | 本地开发、文件型存储 |
| `langgraph-checkpoint-postgres` | `PostgresSaver`、`AsyncPostgresSaver` | 生产环境 |
| `langchain-azure-cosmosdb` | `CosmosDBSaverSync`、`CosmosDBSaver` | Azure 生产环境 |

选择：

| 环境 | 推荐 |
|------|------|
| 单元测试 | `InMemorySaver` |
| 本地长期调试 | `SqliteSaver` |
| Python async graph | `InMemorySaver` 或 async saver |
| 生产 | `PostgresSaver` / `AsyncPostgresSaver` |
| Azure | Cosmos DB saver |

---

## Checkpointer Interface

所有 checkpointer 都遵循 `BaseCheckpointSaver`。

基础方法：

| 方法 | 说明 |
|------|------|
| `.put` / `.aput` | 存储 checkpoint 及 config、metadata |
| `.put_writes` / `.aput_writes` | 存储 pending writes |
| `.get_tuple` / `.aget_tuple` | 根据 config 获取 checkpoint tuple |
| `.list` / `.alist` | 列出匹配 config/filter 的 checkpoints |
| `.delete_thread` / `.adelete_thread` | 删除整个 thread 的 checkpoints 和 writes |

这些方法分别支撑：

| Runtime 行为 | 依赖方法 |
|--------------|----------|
| 保存 checkpoint | `put` |
| pending writes | `put_writes` |
| `graph.get_state()` | `get_tuple` |
| `graph.get_state_history()` | `list` |
| 删除 thread | `delete_thread` |

异步执行时会使用 async 方法：

```text
ainvoke / astream / abatch -> aput / aget_tuple / alist ...
```

---

## Serializer 与 Encryption

### Serializer

Checkpointer 保存 state 时需要序列化 channel values。

默认 serializer：

```text
JsonPlusSerializer
```

它支持：

| 类型 | 说明 |
|------|------|
| LangChain / LangGraph primitives | message、tool call 等 |
| datetimes | 时间对象 |
| enums | 枚举 |
| Pydantic v2 models | 结构化对象 |
| dataclasses | 数据类 |
| numpy arrays | 数组 |
| `_DeltaSnapshot` | DeltaChannel 快照 blob |

pickle fallback：

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

graph.compile(
    checkpointer=InMemorySaver(
        serde=JsonPlusSerializer(pickle_fallback=True)
    )
)
```

谨慎使用 pickle：

```text
pickle 不适合不可信数据；
生产中优先使用可 JSON/msgpack 序列化的 state。
```

### Encryption

可以用 `EncryptedSerializer` 加密持久化 state。

SQLite：

```python
import sqlite3
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

serde = EncryptedSerializer.from_pycryptodome_aes()
checkpointer = SqliteSaver(sqlite3.connect("checkpoint.db"), serde=serde)
```

Postgres：

```python
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.postgres import PostgresSaver

serde = EncryptedSerializer.from_pycryptodome_aes()
checkpointer = PostgresSaver.from_conn_string("postgresql://...", serde=serde)
checkpointer.setup()
```

默认读取：

```text
LANGGRAPH_AES_KEY
```

在 LangSmith 中，只要提供 `LANGGRAPH_AES_KEY`，加密会自动启用。

---

## 自定义 Checkpointer

自定义 checkpointer 需要实现 `BaseCheckpointSaver`。

底层通常有两张逻辑表：

| 表 | 内容 |
|----|------|
| checkpoints table | 每个 super-step 一行，存 serialized graph state 和 parent checkpoint |
| writes table | 每个 node output 一行，存 task/channel/value |

基础 async contract：

```python
class MyCheckpointer(BaseCheckpointSaver):
    async def aput(self, config, checkpoint, metadata, new_versions):
        ...

    async def aput_writes(self, config, writes, task_id, task_path=""):
        ...

    async def aget_tuple(self, config):
        ...

    async def alist(self, config, *, filter=None, before=None, limit=None):
        ...
        yield

    async def adelete_thread(self, thread_id: str):
        ...
```

### `put` / `aput`

要求：

1. 用 `self.serde.dumps_typed(checkpoint)` 序列化 checkpoint。
2. 完整保存 metadata，不要丢未知字段。
3. 保存 parent checkpoint ID。
4. 返回包含新 `checkpoint_id` 的 config。

关键字段：

```text
thread_id
checkpoint_ns
checkpoint_id
parent_checkpoint_id
type
checkpoint blob
metadata
```

### `put_writes` / `aput_writes`

要求：

1. 保存当前 super-step 中某个 task 的 writes。
2. 通过 `(thread_id, checkpoint_ns, checkpoint_id)` 关联 checkpoint。
3. 使用 `WRITES_IDX_MAP` 处理特殊 channel。

特殊 channel 如：

```text
__error__
__interrupt__
```

会映射到保留的负 index，避免和普通 writes 冲突。

### `get_tuple` / `aget_tuple`

必须支持两种路径：

| config | 行为 |
|--------|------|
| 没有 `checkpoint_id` | 返回 thread + namespace 的最新 checkpoint |
| 有 `checkpoint_id` | 返回指定 checkpoint |

指定 ID 查询非常关键：

```text
time travel 需要它；
DeltaChannel 重建状态也依赖它。
```

如果指定 ID 查询错误，DeltaChannel 可能静默重建为空。

### `list` / `alist`

返回某个 thread 的 checkpoints，通常 newest first。

需要支持：

| 参数 | 说明 |
|------|------|
| `before` | 只返回早于该 checkpoint 的记录 |
| `limit` | 限制数量 |
| `filter` | 元数据过滤 |

### `delete_thread` / `adelete_thread`

必须删除：

1. checkpoint rows
2. write rows

否则会残留 pending writes 或占用存储。

---

## Row Key 与 Index 设计

推荐 SQL schema：

```sql
CREATE TABLE checkpoints (
    thread_id          TEXT NOT NULL,
    checkpoint_ns      TEXT NOT NULL DEFAULT '',
    checkpoint_id      TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type               TEXT,
    checkpoint         BYTEA,
    metadata           JSONB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE writes (
    thread_id     TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    task_path     TEXT NOT NULL DEFAULT '',
    idx           INTEGER NOT NULL,
    channel       TEXT NOT NULL,
    type          TEXT,
    value         BYTEA,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx)
);
```

关键原则：

| 需求 | 设计 |
|------|------|
| 快速获取最新 checkpoint | `(thread_id, checkpoint_ns)` + `checkpoint_id DESC` |
| 快速按 ID 获取 checkpoint | 主键包含 `checkpoint_id` |
| 支持 subgraph | 主键包含 `checkpoint_ns` |
| 支持 pending writes | writes 表包含 checkpoint 和 task 维度 |

非 SQL backend 也要满足：

```text
按 (thread_id, checkpoint_ns, checkpoint_id) 直接读取应接近 O(1)。
不要依赖扫描整个 thread 再过滤 checkpoint_id。
```

---

## Delta Channel Support

`DeltaChannel` 是 beta 功能，用于减少 append-heavy channel 的 checkpoint 存储。

传统方式：

```text
checkpoint 1: messages = [m1]
checkpoint 2: messages = [m1, m2]
checkpoint 3: messages = [m1, m2, m3]
```

DeltaChannel：

```text
checkpoint blob 中不保存完整 messages；
通过 ancestor writes + 最近的 _DeltaSnapshot 重建。
```

Runtime 重建 delta channel 需要：

| 数据 | 说明 |
|------|------|
| `writes` | ancestor chain 中该 channel 的所有 writes，oldest first |
| `seed` | 最近 ancestor 中的 `_DeltaSnapshot`，可选 |

默认实现会沿 parent checkpoint 往回走：

```text
get_tuple(head)
  -> parent_config
  -> get_tuple(parent)
  -> collect pending_writes
  -> 找 seed
  -> 继续直到 root 或找到 seed
```

关键依赖：

```text
get_tuple(cursor) 总是带具体 checkpoint_id。
所以自定义 checkpointer 的 by-id lookup 必须正确。
```

性能优化：

| 默认方式 | 优化方式 |
|----------|----------|
| 每个 ancestor 一次 `get_tuple` | 两次查询取 ancestor chain 和 writes |
| 简单但可能慢 | 后端支持查询时建议 override |

### DeltaChannel pruning 注意

DeltaChannel 的状态不完全包含在单个 checkpoint 中，可能依赖 ancestor writes。

如果实现 prune/delete：

| 安全策略 | 说明 |
|----------|------|
| walk before pruning | 对保留 checkpoint 标记其依赖 writes |
| force snapshot before pruning | 在保留 checkpoint 写入 `_DeltaSnapshot` |
| skip pruning | 不确定时先不要 prune delta-channel threads |

### Copy thread 注意

复制 thread 时：

```text
不能只复制 head checkpoint；
必须复制完整 ancestor chain，至少到每个 delta channel 的 _DeltaSnapshot。
```

否则复制后的 thread 可能重建为空状态。

---

## Conformance Suite

自定义 checkpointer 应使用官方 conformance suite 验证。

安装：

```bash
pip install langgraph-checkpoint-conformance
```

示例：

```python
import asyncio
from langgraph.checkpoint.conformance import checkpointer_test, validate

@checkpointer_test(name="MyCheckpointer")
async def my_checkpointer():
    async with MyCheckpointer.create() as saver:
        yield saver

async def main():
    report = await validate(my_checkpointer)
    report.print_report()
    if not report.passed_all_base():
        raise RuntimeError("Checkpointer failed conformance suite")

asyncio.run(main())
```

测试覆盖：

| 范围 | 说明 |
|------|------|
| base methods | `put`、`put_writes`、`get_tuple`、`list`、`delete_thread` |
| async behavior | async saver contract |
| extended capabilities | 自动检测并测试 |
| delta channel history | 如果实现则验证 |

建议：

```text
把 conformance suite 放进 CI。
自定义 checkpointer 没通过之前不要上线。
```

---

## 生产实践建议

1. 生产不要用 `InMemorySaver`，它不跨进程重启。
2. 对 async graph 使用 async checkpointer。
3. `thread_id` 保持短且稳定，Postgres 场景建议小于 255 字符。
4. state 中避免存大对象和不可序列化对象。
5. 高敏感数据启用 `EncryptedSerializer`。
6. 外部副作用节点结合 `durability="sync"` 和幂等设计。
7. 长线程要规划 checkpoint pruning 或 DeltaChannel。
8. 使用 LangSmith 观察 checkpointed state 和 resume 行为。
9. 自定义 checkpointer 必须支持 by-id lookup。
10. 自定义 checkpointer 不要丢 metadata 中未知字段。
11. 子图场景必须正确处理 `checkpoint_ns`。
12. 实现 pruning/copy_thread 时特别小心 DeltaChannel 依赖链。

---

## 故障排查

| 问题 | 可能原因 | 处理方式 |
|------|----------|----------|
| interrupt 后无法恢复 | 没有 checkpointer 或没有 thread_id | compile 配 checkpointer，调用传 `configurable.thread_id` |
| 多轮对话没有记忆 | 每轮用了不同 thread_id 或没配置 reducer | 固定 conversation thread_id，检查 `messages` reducer |
| 进程重启后 state 消失 | 使用 `InMemorySaver` | 换 SQLite/Postgres/CosmosDB |
| Postgres thread_id 报错 | thread_id 过长 | 用 UUID/hash，控制 255 字符以内 |
| replay 重复调用外部 API | checkpoint 后节点会重执行 | 对 action 节点做幂等，或选择合适 checkpoint |
| state 越来越大 | 累积 channel 反复完整保存 | prune、瘦身 state、考虑 DeltaChannel |
| 自定义 checkpointer time travel 失败 | `get_tuple` 指定 checkpoint_id 路径不正确 | 建立 `(thread_id, checkpoint_ns, checkpoint_id)` 直接索引 |
| DeltaChannel 重建为空 | ancestor by-id lookup 失败或 writes 被删 | 修复 get_tuple，保留依赖 writes |
| 子图 checkpoint 混乱 | 未处理 `checkpoint_ns` | row key 加入 `checkpoint_ns` |
| 加密未生效 | 未设置 serializer 或缺少 AES key | 设置 `EncryptedSerializer` 和 `LANGGRAPH_AES_KEY` |

---

## 快速参考

### 何时需要 Checkpointer

| 需求 | 是否需要 |
|------|----------|
| HITL interrupt/resume | 必须 |
| 多轮 conversation memory | 必须 |
| time travel / replay | 必须 |
| fault tolerance | 必须 |
| 单次无状态调用 | 可选 |
| 跨 thread 长期记忆 | 用 Store，不是 Checkpointer |

### 常用 API

| API | 说明 |
|-----|------|
| `workflow.compile(checkpointer=...)` | 启用 checkpointing |
| `configurable.thread_id` | 指定 thread |
| `graph.get_state(config)` | 读取最新或指定 checkpoint |
| `graph.get_state_history(config)` | 读取 checkpoint history |
| `graph.update_state(config, values)` | 创建 state update checkpoint |
| `durability="exit"` | 结束时保存 |
| `durability="async"` | 异步保存 |
| `durability="sync"` | 同步保存 |

### StateSnapshot

| 字段 | 说明 |
|------|------|
| `values` | state values |
| `next` | 下一步节点 |
| `config` | thread/checkpoint config |
| `metadata` | source/writes/step |
| `created_at` | 创建时间 |
| `parent_config` | 父 checkpoint |
| `tasks` | task/error/interrupt/subgraph state |

### Checkpointer 选择

| 场景 | 实现 |
|------|------|
| 测试 | `InMemorySaver` |
| 本地持久化 | `SqliteSaver` |
| 生产 | `PostgresSaver` |
| 生产 async | `AsyncPostgresSaver` |
| Azure | Cosmos DB saver |
| 自定义存储 | 实现 `BaseCheckpointSaver` 并跑 conformance suite |

### 自定义 Checkpointer 必须支持

| 能力 | 关键点 |
|------|--------|
| 保存 checkpoint | `put/aput` |
| 保存 writes | `put_writes/aput_writes` |
| 最新 checkpoint 查询 | no checkpoint_id path |
| 指定 checkpoint 查询 | by checkpoint_id path |
| history 列表 | newest first、支持 before/limit |
| 删除 thread | 同时删 checkpoints 和 writes |
| 序列化 | 使用 `self.serde` |
| subgraph | row key 包含 `checkpoint_ns` |

---

## 资料来源

- LangGraph Checkpointers 官方文档：<https://docs.langchain.com/oss/python/langgraph/checkpointers>
