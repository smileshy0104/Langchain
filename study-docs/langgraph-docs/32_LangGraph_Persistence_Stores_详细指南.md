# LangGraph Persistence 与 Stores 详细指南

> 基于 LangGraph 官方 Persistence、Stores 文档整理，并补充 Checkpointers 关键概念。本文聚焦 LangGraph 持久化体系：如何用 checkpointer 保存线程级短期状态，如何用 store 保存跨线程长期记忆，以及二者如何一起支撑 conversation continuity、human-in-the-loop、failure recovery、time travel 和长期用户记忆。

## 目录

1. [整体理解](#整体理解)
2. [Persistence 是什么](#persistence-是什么)
3. [Checkpointer 与 Store 的区别](#checkpointer-与-store-的区别)
4. [Quickstart](#quickstart)
5. [Checkpointers 核心概念](#checkpointers-核心概念)
6. [Threads](#threads)
7. [Checkpoints 与 Super-steps](#checkpoints-与-super-steps)
8. [StateSnapshot](#statesnapshot)
9. [读取、回放与更新 State](#读取回放与更新-state)
10. [Durability Modes](#durability-modes)
11. [Checkpointer 实现与生产选择](#checkpointer-实现与生产选择)
12. [Stores 是什么](#stores-是什么)
13. [Store 基本用法](#store-基本用法)
14. [Namespace 设计](#namespace-设计)
15. [Listing 与分页](#listing-与分页)
16. [Semantic Search](#semantic-search)
17. [在 LangGraph 中使用 Store](#在-langgraph-中使用-store)
18. [自定义 Store](#自定义-store)
19. [Checkpointer + Store 组合模式](#checkpointer--store-组合模式)
20. [常见问题与排障](#常见问题与排障)
21. [最佳实践](#最佳实践)
22. [快速参考](#快速参考)
23. [资料来源](#资料来源)

---

## 整体理解

LangGraph 的 Persistence 不是单一功能，而是两套互补机制：

```text
Persistence
├─ Checkpointer
│  └─ 保存某个 thread 的 graph state checkpoints
│     用于短期、线程级记忆和恢复
└─ Store
   └─ 保存应用定义的 key-value 数据
      用于长期、跨线程记忆
```

一句话：

```text
Checkpointer 让一次对话/任务不断片；
Store 让 Agent 跨对话记住用户、事实和偏好。
```

典型组合：

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

checkpointer = InMemorySaver()
store = InMemoryStore()

graph = builder.compile(checkpointer=checkpointer, store=store)
```

运行时：

```text
thread_id -> 找到当前 conversation / task 的 checkpoints
user_id   -> 找到该用户跨 threads 的 long-term memories
```

---

## Persistence 是什么

Persistence 让 LangGraph 应用在单次 graph run 之外保留信息。

它解决的问题：

| 问题 | Persistence 的作用 |
|------|--------------------|
| 多轮对话如何接上上文 | checkpointer 用 `thread_id` 保存对话状态 |
| Human-in-the-loop 暂停后如何恢复 | checkpointer 保存 interrupt 前状态 |
| 节点失败后如何不从头重跑 | checkpoint 和 pending writes 让成功步骤可复用 |
| 如何调试历史执行路径 | checkpoint history 支持 state inspection 和 time travel |
| 如何跨线程记住用户偏好 | store 以 namespace/key/value 保存长期记忆 |
| 子图/多 Agent 如何共享长期信息 | store 提供跨 graph/thread 的共享数据层 |

官方总结：

| 机制 | 记忆类型 |
|------|----------|
| Checkpointers | short-term, thread-scoped memory |
| Stores | long-term, cross-thread memory |

---

## Checkpointer 与 Store 的区别

| 维度 | Checkpointer | Store |
|------|--------------|-------|
| 保存内容 | Graph state snapshots | Application-defined key-value data |
| 范围 | 单个 thread | 跨 threads |
| 记忆类型 | 短期、线程级记忆 | 长期、跨线程记忆 |
| 典型用途 | conversation continuity、HITL、time travel、fault tolerance | user preferences、facts、shared knowledge |
| 访问方式 | graph config 中传 `thread_id` | node 或 application code 中读写 |
| 数据形态 | `StateSnapshot` / checkpoint tuple | namespace + key + dict value |
| 生命周期 | 跟一次会话/任务强相关 | 跟用户、组织、应用域强相关 |
| 官方指南 | Checkpointers | Stores |

直觉判断：

```text
“这个信息是否属于当前线程的执行状态？” -> Checkpointer
“这个信息是否未来其他线程也要用？” -> Store
```

示例：

| 数据 | 应放哪里 | 原因 |
|------|----------|------|
| 当前 conversation messages | Checkpointer | 属于当前 thread state |
| 当前工具调用结果 | Checkpointer | 用于恢复当前执行 |
| interrupt 前的草稿 | Checkpointer | resume 需要 |
| 用户喜欢披萨 | Store | 跨多次对话复用 |
| 用户所在组织 ID | Context 或 Store | run-scoped 可放 context，长期可放 store |
| 全局业务知识 | Store | 多线程共享 |

---

## Quickstart

同时配置 checkpointer 和 store：

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

checkpointer = InMemorySaver()
store = InMemoryStore()

graph = builder.compile(checkpointer=checkpointer, store=store)

result = graph.invoke(
    {"messages": [{"role": "user", "content": "Hi, my name is Bob."}]},
    {"configurable": {"thread_id": "thread-1"}},
)
```

如果使用 LangSmith Agent Server：

```text
Agent Server 会自动处理 persistence infrastructure。
通常不需要手动配置 checkpointers 或 stores。
```

注意：

| 实现 | 适合 |
|------|------|
| `InMemorySaver` / `MemorySaver` | 本地开发、测试 |
| `InMemoryStore` | 本地开发、测试 |
| `PostgresSaver` / `AsyncPostgresSaver` | 生产 checkpoint |
| `SqliteSaver` | 本地文件型开发 |
| `PostgresStore` / `MongoDBStore` / `RedisStore` | 生产 store |

---

## Checkpointers 核心概念

Checkpointer 会在每个 super-step 保存 graph state snapshot，并按 thread 组织。

它是以下能力的基础：

| 能力 | 为什么需要 checkpointer |
|------|--------------------------|
| Human-in-the-loop | 人类要查看 state，graph 要在审批后恢复 |
| Memory | 同一 thread 的后续输入要继承之前状态 |
| Time travel | 需要回看、回放或 fork 历史 checkpoint |
| Fault tolerance | 节点失败后从最近成功 checkpoint 恢复 |
| Pending writes | 并行 super-step 中成功节点的输出不必重算 |

编译：

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

没有 `thread_id`，checkpointer 无法知道 state 应该保存到哪个 thread，也无法在 interrupt 后恢复。

---

## Threads

Thread 是 checkpointer 组织 checkpoint 的基本单位。

```text
thread_id = 一条 conversation / 一个 long-running task / 一个可恢复执行上下文
```

示例：

```python
config = {"configurable": {"thread_id": "customer_123"}}
```

Thread 中包含：

| 内容 | 说明 |
|------|------|
| 当前 state | 最新 checkpoint 的 state values |
| 历史 checkpoints | 每个 super-step 的快照 |
| pending writes | super-step 内已成功节点的写入 |
| interrupts | 暂停点和恢复所需信息 |

命名建议：

| 场景 | thread_id 建议 |
|------|----------------|
| 会话 | conversation UUID |
| 工单 | ticket ID 或 ticket UUID |
| 后台任务 | job ID |
| 用户多会话 | 每次新会话一个新 thread，而不是直接用 user_id |

注意：

```text
user_id 通常适合做 store namespace；
thread_id 通常适合做 checkpointer key。
不要把所有用户长期记忆都塞进同一个 thread state。
```

---

## Checkpoints 与 Super-steps

Checkpoint 是某个 thread 在某个时间点的 graph state snapshot。

LangGraph 在每个 super-step 边界创建 checkpoint。

Super-step 可以理解为 graph 的一个执行 tick：

```text
START -> A -> B -> END

会产生：
1. 初始 empty checkpoint，next = START
2. 输入 checkpoint，next = A
3. A 执行后 checkpoint，next = B
4. B 执行后 checkpoint，next = ()
```

在一个 super-step 中，多个节点可能并行执行。LangGraph 不只保存完整 checkpoint，还会保存 node/task 级别的 writes。

这支持 pending writes：

```text
同一个 super-step 中：
  node A 成功
  node B 失败

恢复时：
  node A 的输出已经 durable
  只需要重新执行失败或未完成的部分
```

Checkpoint namespace：

| `checkpoint_ns` | 含义 |
|-----------------|------|
| `""` | root graph checkpoint |
| `"node_name:uuid"` | 某个 subgraph 的 checkpoint |
| `"outer:uuid|inner:uuid"` | nested subgraphs checkpoint |

在节点里读取：

```python
def my_node(state: State, config: RunnableConfig):
    checkpoint_ns = config["configurable"]["checkpoint_ns"]
```

---

## StateSnapshot

`graph.get_state(config)` 返回 `StateSnapshot`。

核心字段：

| Field | 说明 |
|-------|------|
| `values` | 当前 checkpoint 的 state channel values |
| `next` | 接下来要执行的节点名，空 tuple 表示完成 |
| `config` | 包含 `thread_id`、`checkpoint_ns`、`checkpoint_id` |
| `metadata` | 执行元数据，如 source、writes、step |
| `created_at` | checkpoint 创建时间 |
| `parent_config` | 上一个 checkpoint 的 config |
| `tasks` | 当前 step 的任务信息，包含 error、interrupts、subgraph state 等 |

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

---

## 读取、回放与更新 State

### Get state history

获取整个 thread 的历史：

```python
config = {"configurable": {"thread_id": "1"}}
history = list(graph.get_state_history(config))
```

历史通常按最新 checkpoint 在前排列。

查找特定 checkpoint：

```python
history = list(graph.get_state_history(config))

before_node_b = next(s for s in history if s.next == ("node_b",))
step_2 = next(s for s in history if s.metadata["step"] == 2)
forks = [s for s in history if s.metadata["source"] == "update"]
interrupted = next(
    s for s in history
    if s.tasks and any(t.interrupts for t in s.tasks)
)
```

### Replay

Replay 是从历史 checkpoint 重新执行后续步骤。

```text
checkpoint 前的节点：跳过，直接使用保存结果
checkpoint 后的节点：重新执行，包括 LLM/API/interrupt
```

适合：

| 场景 | 用途 |
|------|------|
| 调试 | 从某一步重新跑 |
| time travel | 比较不同路径 |
| fork | 从旧 state 创建新轨迹 |

### Update state

`update_state` 会创建一个新 checkpoint，不会修改原 checkpoint。

```text
update_state = 以人工/外部方式写入 state
它和节点更新一样会经过 reducer
```

可选 `as_node` 用于控制这次更新被视为来自哪个节点，从而影响下一步执行。

---

## Durability Modes

执行 graph 时可以设置 durability：

```python
graph.stream(
    {"input": "test"},
    durability="sync",
)
```

三种模式：

| Mode | 持久化时机 | 优点 | 风险/成本 |
|------|------------|------|-----------|
| `"exit"` | graph 结束、报错或 interrupt 时保存 | 性能最好 | 中途进程崩溃无法恢复中间状态 |
| `"async"` | 下一步执行时异步保存 | 性能和可靠性平衡 | crash 时可能丢少量最新 checkpoint |
| `"sync"` | 下一步开始前同步保存 | 最可靠 | 性能开销更高 |

选择建议：

| 场景 | 推荐 |
|------|------|
| 本地实验 | `async` 或默认 |
| 高吞吐低风险任务 | `exit` |
| HITL / 金融 / 外部副作用前 | `sync` |
| 长任务且希望性能平衡 | `async` |

---

## Checkpointer 实现与生产选择

官方 checkpointer 生态：

| 包 | 实现 | 适合 |
|----|------|------|
| `langgraph-checkpoint` | base interface + `InMemorySaver` | 实验、测试 |
| `langgraph-checkpoint-sqlite` | `SqliteSaver` / `AsyncSqliteSaver` | 本地文件型开发 |
| `langgraph-checkpoint-postgres` | `PostgresSaver` / `AsyncPostgresSaver` | 生产 |
| `langchain-azure-cosmosdb` | Cosmos DB saver | Azure 生产环境 |

Checkpointer interface：

| 方法 | 说明 |
|------|------|
| `.put` / `.aput` | 保存 checkpoint |
| `.put_writes` / `.aput_writes` | 保存 pending writes |
| `.get_tuple` / `.aget_tuple` | 读取 checkpoint tuple |
| `.list` / `.alist` | 列出 checkpoints |

Serializer：

| Serializer | 说明 |
|------------|------|
| `JsonPlusSerializer` | 默认 serializer，支持 LangChain/LangGraph primitives、datetime、enum 等 |
| `pickle_fallback` | 对 msgpack/JSON 不支持的对象 fallback 到 pickle |

生产提醒：

```text
MemorySaver / InMemorySaver 不会跨进程重启保留数据。
生产应使用 Postgres / CosmosDB 等持久化 checkpointer。
```

---

## Stores 是什么

Store 提供跨 thread 的长期记忆。

官方定义：

```text
Stores let agents persist information across threads,
including user preferences, accumulated knowledge, and facts
that should survive beyond a single conversation.
```

与 checkpointer 不同：

```text
Checkpointer 保存 graph state；
Store 保存应用定义的数据。
```

适合存：

| 类型 | 示例 |
|------|------|
| 用户偏好 | 喜欢披萨、偏好中文、喜欢简洁回答 |
| 事实记忆 | 用户公司、角色、项目背景 |
| 跨会话知识 | 长期任务资料、业务规则 |
| shared knowledge | 团队共享设置、组织级配置 |

不适合存：

| 类型 | 原因 |
|------|------|
| 当前节点中间状态 | 应放 checkpointer state |
| 可从 thread state 推导的临时字符串 | 不需要长期保存 |
| 不可 JSON 序列化 Python 对象 | store value 应是普通 dict |

---

## Store 基本用法

创建：

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
```

Store 的基本模型：

```text
namespace: tuple[str, ...]
key:       str
value:     dict
```

示例 namespace：

```python
user_id = "1"
namespace_for_memory = (user_id, "memories")
```

写入 memory：

```python
import uuid

memory_id = str(uuid.uuid4())
memory = {"food_preference": "I like pizza"}

store.put(namespace_for_memory, memory_id, memory)
```

读取：

```python
memories = store.search(namespace_for_memory)
latest = memories[-1].dict()
```

Item 字段：

| 字段 | 说明 |
|------|------|
| `value` | 保存的 dict value |
| `key` | namespace 内唯一 key |
| `namespace` | 该 item 所在 namespace |
| `created_at` | 创建时间 |
| `updated_at` | 更新时间 |
| `score` | 语义搜索时可能出现的相似度分数 |

注意：`namespace` 在 Python 类型中是 `tuple[str, ...]`，但转 JSON 后可能表现为 list。

---

## Namespace 设计

Namespace 是 Store 的组织方式，类型是 tuple。

常见设计：

| Namespace | 含义 |
|-----------|------|
| `(user_id, "memories")` | 某个用户的长期记忆 |
| `(user_id, "preferences")` | 某个用户偏好 |
| `(org_id, "shared_knowledge")` | 组织级共享知识 |
| `(tenant_id, user_id, "memories")` | 多租户用户记忆 |
| `("global", "policies")` | 全局策略 |

设计原则：

1. namespace 应体现访问边界。
2. user memory 不要和 thread_id 绑定。
3. 多租户系统应把 tenant/org 放在 namespace 前缀。
4. 同类数据放在稳定后缀中，例如 `"memories"`、`"preferences"`。
5. 需要批量枚举时，让 namespace 前缀可预测。

Prefix matching：

```text
store.search(("alice",))
```

会匹配：

```text
("alice", "memories")
("alice", "preferences")
...
```

如果只想查单层，传完整 namespace 或在客户端过滤 `item.namespace`。

---

## Listing 与分页

列出 namespace 下的 items：

```python
items = store.search(("alice", "memories"), limit=100)
```

行为注意：

| 行为 | 说明 |
|------|------|
| `namespace_prefix` 是前缀匹配 | `("alice",)` 会返回其所有子 namespace |
| 超过 `limit` 会静默截断 | 没有 overflow signal |
| 默认排序依赖 backend | 不同 store 实现可能不一样 |

不同实现排序：

| Store | 默认排序 |
|-------|----------|
| `InMemoryStore` | insertion order，最新插入通常在最后 |
| `PostgresStore` / `AsyncPostgresStore` | `updated_at` descending，最新更新在前 |

分页：

```python
page_size = 50
offset = 0

while True:
    page = store.search(("alice", "memories"), limit=page_size, offset=offset)
    if not page:
        break
    for item in page:
        pass
    offset += page_size
```

列出 namespaces：

```python
namespaces = store.list_namespaces(prefix=("alice",), max_depth=2)
```

---

## Semantic Search

Store 支持语义搜索。需要配置 embedding model：

```python
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore

store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "dims": 1536,
        "fields": ["food_preference", "$"],
    }
)
```

语义搜索：

```python
memories = store.search(
    namespace_for_memory,
    query="What does the user like to eat?",
    limit=3,
)
```

控制哪些字段被 embedding：

```python
store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {
        "food_preference": "I love Italian cuisine",
        "context": "Discussing dinner plans",
    },
    index=["food_preference"],
)
```

不 embedding，但仍可精确检索：

```python
store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {"system_info": "Last updated: 2024-01-01"},
    index=False,
)
```

`fields` 说明：

| 字段 | 含义 |
|------|------|
| `"food_preference"` | 只 embedding value 中的该字段 |
| `"$"` | embedding 整个 value |
| `index=False` | 不写入向量索引 |

---

## 在 LangGraph 中使用 Store

Store 和 checkpointer 通常一起使用：

```python
from dataclasses import dataclass
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState

@dataclass
class Context:
    user_id: str

checkpointer = InMemorySaver()
store = InMemoryStore()

builder = StateGraph(MessagesState, context_schema=Context)
graph = builder.compile(checkpointer=checkpointer, store=store)
```

调用时：

```python
config = {"configurable": {"thread_id": "1"}}

for update in graph.stream(
    {"messages": [{"role": "user", "content": "hi"}]},
    config,
    stream_mode="updates",
    context=Context(user_id="1"),
):
    print(update)
```

在 node 中通过 `Runtime` 访问 store 和 context：

```python
from dataclasses import dataclass
from langgraph.runtime import Runtime

@dataclass
class Context:
    user_id: str

async def update_memory(state: MessagesState, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")

    memory_id = str(uuid.uuid4())
    memory = "User likes pizza"

    await runtime.store.aput(namespace, memory_id, {"memory": memory})
```

在模型调用前搜索长期记忆：

```python
async def call_model(state: MessagesState, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")

    memories = await runtime.store.asearch(
        namespace,
        query=state["messages"][-1].content,
        limit=3,
    )
    info = "\n".join([d.value["memory"] for d in memories])

    # Use info in model prompt
```

跨 thread 复用：

```text
thread_id = "1", user_id = "1" -> 写入 memories
thread_id = "2", user_id = "1" -> 仍可读取同一用户 memories
```

这正是 store 的核心价值。

---

## 自定义 Store

如果内置 backend 不满足需求，可以继承 `BaseStore`。

必需 async 方法：

| Method | 说明 |
|--------|------|
| `aput(namespace, key, value, index=None)` | 写入或覆盖 item |
| `aget(namespace, key)` | 按 key 读取，缺失返回 `None` |
| `adelete(namespace, key)` | 删除 item |
| `asearch(namespace_prefix, query=None, filter=None, limit=10, offset=0)` | 查找 items，可支持语义 query |
| `alist_namespaces(prefix=None, suffix=None, max_depth=None, limit=100, offset=0)` | 列出 namespaces |

Sync 对应方法可选但推荐：

```text
put / get / delete / search / list_namespaces
```

### SQL schema 示例

```sql
CREATE TABLE store_items (
    namespace   TEXT[] NOT NULL,
    key         TEXT NOT NULL,
    value       JSONB NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (namespace, key)
);

CREATE INDEX ON store_items USING gin(namespace);
```

### Serialization

Store values 应是普通 JSON-serializable dict：

```text
推荐：dict / list / str / number / bool / null
避免：raw Python objects、DataFrame、open file handle、自定义类实例
```

### Semantic search support

如果 backend 支持向量搜索：

1. 接受 `query: str | None`。
2. query 不为空时 embedding query。
3. 用 cosine similarity 或 backend 等价能力排序。
4. 结果 item 包含 `score`。

如果不支持：

```text
当传入 query 时 raise NotImplementedError。
```

### Testing

官方建议以 `InMemoryStore` 为参考实现做对照测试：

```python
async def test_put_and_get(store, reference):
    ns = ("test", "ns")
    for s in [store, reference]:
        await s.aput(ns, "k1", {"val": 1})
        item = await s.aget(ns, "k1")
        assert item is not None
        assert item.value == {"val": 1}
```

---

## Checkpointer + Store 组合模式

### 多轮对话 + 长期记忆

```text
checkpointer:
  thread_id = conversation_123
  保存当前对话 messages / tool results / interrupts

store:
  namespace = (user_id, "memories")
  保存用户偏好、事实、长期背景
```

### HITL 审批

```text
checkpointer:
  暂停在 interrupt 节点
  保存 draft、tool call、current state

store:
  可保存审批偏好或用户长期策略
```

### 子图共享数据

```text
subgraph checkpoint_ns:
  管理子图自己的 checkpoint namespace

store:
  用于 parent graph 和 subgraph 共享长期数据
```

### Agent 记忆写入

```text
1. call_model 前从 store 搜索 relevant memories
2. LLM 根据当前消息和 memories 回答
3. update_memory 节点抽取值得长期保存的信息
4. 写入 store
5. checkpointer 保存当前 thread 的执行状态
```

---

## 常见问题与排障

### PostgresSaver: `thread_id` too long

原因：

```text
PostgresSaver / AsyncPostgresSaver 的 thread_id 列长度有限。
```

修复：

```python
import uuid

config = {"configurable": {"thread_id": str(uuid.uuid4())[:255]}}
```

建议：

```text
thread_id 控制在 255 字符以内。
需要确定性 ID 时，用 hash。
```

### MemorySaver 不跨重启持久化

原因：

```text
MemorySaver / InMemorySaver 数据存在 RAM 中。
进程重启后数据丢失。
```

修复：

| 环境 | 推荐 |
|------|------|
| 本地开发 | `SqliteSaver` |
| 生产 | `PostgresSaver` |

### Checkpoints 无限增长

原因：

```text
长对话会不断产生 checkpoints。
```

修复：

1. 定期 prune old checkpoints。
2. 设置 retention policy。
3. 对 append-heavy channel 评估 `DeltaChannel`。
4. 不要把大对象无节制放进 state。

### Parent graph 看不到 subgraph state

原因：

```text
subgraph 有自己的 checkpoint namespace。
```

修复：

1. 需要跨 graph 共享的数据写入 Store。
2. 或配置 subgraph 写入 parent checkpoint。
3. 调试时关注 `checkpoint_ns`。

### Store 搜索结果顺序不一致

原因：

```text
不同 store backend 默认排序不同。
```

修复：

```text
如果顺序重要，在客户端按 item.updated_at 或 score 排序。
```

### 语义搜索没有结果

可能原因：

| 原因 | 处理 |
|------|------|
| store 未配置 embedding index | 配置 `index.embed`、`dims`、`fields` |
| put 时 `index=False` | 该 item 不可语义搜索 |
| fields 配错 | 确认需要搜索的字段被 embedding |
| backend 不支持 query | 实现语义搜索或避免传 query |

---

## 最佳实践

1. 用 `thread_id` 管当前执行，用 `user_id/org_id` 管长期记忆 namespace。
2. 开发用 `InMemorySaver` / `InMemoryStore`，生产换持久化 backend。
3. 不要把跨线程长期记忆塞进 checkpoint state。
4. 不要把 thread 临时执行状态塞进 store。
5. Store value 用 JSON-serializable dict。
6. Namespace 设计要体现租户、用户、数据类型。
7. 需要语义检索时，只 embedding 有语义价值的字段。
8. 对 checkpoints 配 retention/pruning 策略。
9. HITL 场景必须配置 checkpointer。
10. 外部副作用节点结合 durability 和幂等设计。
11. 用 `get_state_history` 调试执行轨迹和 interrupt 点。
12. Agent Server 场景优先利用平台自动 persistence。

---

## 快速参考

### Persistence 选择

| 需求 | 使用 |
|------|------|
| 对话上下文连续 | Checkpointer |
| interrupt 后恢复 | Checkpointer |
| time travel / replay | Checkpointer |
| 节点失败恢复 | Checkpointer |
| 用户偏好长期保存 | Store |
| 跨 thread 共享事实 | Store |
| 语义搜索长期记忆 | Store + embedding index |
| 子图/父图共享长期数据 | Store |

### Checkpointer API

| API | 说明 |
|-----|------|
| `compile(checkpointer=...)` | 启用 checkpoint |
| `configurable.thread_id` | 指定 thread |
| `graph.get_state(config)` | 读取最新或指定 checkpoint |
| `graph.get_state_history(config)` | 读取历史 checkpoints |
| `graph.update_state(config, values)` | 创建 state update checkpoint |
| `durability="exit"` | 结束时保存 |
| `durability="async"` | 异步保存 |
| `durability="sync"` | 同步保存 |

### Store API

| API | 说明 |
|-----|------|
| `store.put(namespace, key, value, index=None)` | 写入 item |
| `store.get(namespace, key)` | 按 key 读取 |
| `store.delete(namespace, key)` | 删除 |
| `store.search(namespace_prefix, query=None, filter=None, limit=10, offset=0)` | 搜索/列出 items |
| `store.list_namespaces(prefix=..., max_depth=...)` | 列出 namespaces |
| `runtime.store.aput(...)` | node 中异步写 store |
| `runtime.store.asearch(...)` | node 中异步搜索 store |

### Item 字段

| 字段 | 说明 |
|------|------|
| `value` | 存储的 dict |
| `key` | namespace 内唯一 key |
| `namespace` | tuple/list 形式 namespace |
| `created_at` | 创建时间 |
| `updated_at` | 更新时间 |
| `score` | 语义搜索相关度 |

---

## 资料来源

- LangGraph Persistence 官方文档：<https://docs.langchain.com/oss/python/langgraph/persistence>
- LangGraph Stores 官方文档：<https://docs.langchain.com/oss/python/langgraph/stores>
- LangGraph Checkpointers 官方文档：<https://docs.langchain.com/oss/python/langgraph/checkpointers>
