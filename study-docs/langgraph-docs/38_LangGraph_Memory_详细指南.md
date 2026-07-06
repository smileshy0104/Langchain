# LangGraph Memory 详细指南

> 基于 LangGraph 官方 Memory 文档整理。本文聚焦 LangGraph 中两类记忆：基于 checkpointer 的短期记忆，以及基于 Store 的长期记忆；同时总结生产环境持久化、子图记忆、语义搜索、消息裁剪/删除/总结、checkpoint 管理和数据库迁移。

## 目录

1. [整体理解](#整体理解)
2. [Memory 是什么](#memory-是什么)
3. [短期记忆与长期记忆](#短期记忆与长期记忆)
4. [Add Short-term Memory](#add-short-term-memory)
5. [短期记忆生产配置](#短期记忆生产配置)
6. [Postgres Checkpointer](#postgres-checkpointer)
7. [MongoDB Checkpointer](#mongodb-checkpointer)
8. [Redis Checkpointer](#redis-checkpointer)
9. [Oracle Checkpointer](#oracle-checkpointer)
10. [Short-term Memory in Subgraphs](#short-term-memory-in-subgraphs)
11. [Add Long-term Memory](#add-long-term-memory)
12. [Store 与 Namespace](#store-与-namespace)
13. [Access Store Inside Nodes](#access-store-inside-nodes)
14. [长期记忆生产配置](#长期记忆生产配置)
15. [Postgres Store](#postgres-store)
16. [Redis Store](#redis-store)
17. [Oracle Store](#oracle-store)
18. [Semantic Search](#semantic-search)
19. [Manage Short-term Memory](#manage-short-term-memory)
20. [Trim Messages](#trim-messages)
21. [Delete Messages](#delete-messages)
22. [Summarize Messages](#summarize-messages)
23. [Manage Checkpoints](#manage-checkpoints)
24. [View Thread State](#view-thread-state)
25. [View Thread History](#view-thread-history)
26. [Delete Thread Checkpoints](#delete-thread-checkpoints)
27. [Database Management](#database-management)
28. [最佳实践](#最佳实践)
29. [故障排查](#故障排查)
30. [快速参考](#快速参考)
31. [资料来源](#资料来源)

---

## 整体理解

AI 应用需要 memory 来跨多轮交互共享上下文。

LangGraph 中主要有两类 memory：

| 类型 | 作用 | 生命周期 | 核心组件 |
|------|------|----------|----------|
| Short-term memory | 保存某个 thread 内的对话和 graph state | 同一个 `thread_id` 内 | Checkpointer |
| Long-term memory | 保存跨 thread、跨会话的用户或应用数据 | 跨多个 thread | Store |

一句话：

```text
短期记忆回答“这个会话刚刚聊了什么”；
长期记忆回答“这个用户长期偏好和资料是什么”。
```

典型架构：

```text
Graph State
  -> messages / summary / context
  -> checkpointer persists per thread
  -> short-term memory

Store
  -> namespace + key + value
  -> optional semantic index
  -> long-term memory
```

---

## Memory 是什么

Memory 让 agent 能在多次调用中保留上下文。

没有 memory 时：

```text
第 1 轮：Hi, I am Bob.
第 2 轮：What's my name?
模型可能不知道 Bob，因为第 1 轮状态没有被保存。
```

有短期记忆时：

```text
同一个 thread_id 下，messages 会保存在 graph state 中；
下一轮调用会加载之前的 messages。
```

有长期记忆时：

```text
即使换了新的 thread_id，也能通过 user_id 从 Store 中检索用户资料。
```

---

## 短期记忆与长期记忆

| 维度 | Short-term Memory | Long-term Memory |
|------|-------------------|------------------|
| 保存位置 | graph state / checkpoints | Store |
| 作用范围 | thread-level | user-level / app-level |
| 典型键 | `thread_id` | namespace + key |
| 典型内容 | messages、summary、当前 workflow state | 用户偏好、档案、历史事实、业务知识 |
| 生命周期 | 一个会话或一个工作流实例 | 多会话长期存在 |
| 读取方式 | checkpointer 自动加载 state | 节点内通过 `runtime.store.search/get` |
| 写入方式 | 节点 return state update | `store.put/aput` |
| 适合问题 | “刚才说了什么？” | “这个用户长期喜欢什么？” |

两者经常一起使用：

```text
checkpointer 保存当前对话；
store 保存跨会话用户记忆；
node 调用模型前同时读取 state.messages 和 store memories。
```

---

## Add Short-term Memory

短期记忆本质上是 thread-level persistence。

最小示例：

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph

checkpointer = InMemorySaver()

builder = StateGraph(...)
graph = builder.compile(checkpointer=checkpointer)

graph.invoke(
    {"messages": [{"role": "user", "content": "hi! i am Bob"}]},
    {"configurable": {"thread_id": "1"}},
)
```

关键点：

| 配置 | 说明 |
|------|------|
| `checkpointer` | 保存 graph state |
| `thread_id` | 指向同一个对话/工作流实例 |
| `MessagesState` | 常用于保存多轮 messages |
| `stream_events(..., version="v3")` | 可观察 state snapshots 和 message stream |

第二次用同一个 `thread_id` 调用：

```python
graph.invoke(
    {"messages": [{"role": "user", "content": "what's my name?"}]},
    {"configurable": {"thread_id": "1"}},
)
```

graph 会从 checkpointer 读取之前的 state，因此模型能看到前面的 messages。

---

## 短期记忆生产配置

`InMemorySaver` 适合 demo、测试和本地实验。

生产环境应使用数据库支持的 checkpointer：

| Checkpointer | 包 | 适合 |
|--------------|----|------|
| PostgresSaver | `langgraph-checkpoint-postgres` | 通用生产关系型存储 |
| MongoDBSaver | `langgraph-checkpoint-mongodb` | 已使用 MongoDB 的应用 |
| RedisSaver | `langgraph-checkpoint-redis` | 低延迟会话状态、缓存型场景 |
| OracleSaver | `langgraph-oracledb` | Oracle AI Database / 企业数据库环境 |

生产注意事项：

```text
首次使用数据库型 checkpointer 通常需要调用 setup() 执行迁移。
建议把迁移作为部署步骤，而不是每次请求时运行。
```

---

## Postgres Checkpointer

安装：

```bash
pip install -U "psycopg[binary,pool]" langgraph langgraph-checkpoint-postgres
```

同步示例：

```python
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import MessagesState, START, StateGraph

model = init_chat_model(model="claude-haiku-4-5-20251001")

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # checkpointer.setup()

    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_edge(START, "call_model")

    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "1"}}

    stream = graph.stream_events(
        {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
        config,
        version="v3",
    )

    for snapshot in stream.values:
        print(snapshot)
```

异步版本使用：

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
```

并通过：

```python
async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # await checkpointer.setup()
```

---

## MongoDB Checkpointer

安装：

```bash
pip install -U pymongo langgraph langgraph-checkpoint-mongodb
```

同步示例：

```python
from langgraph.checkpoint.mongodb import MongoDBSaver

MONGODB_URI = "localhost:27017"

with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
```

异步版本：

```python
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

async with AsyncMongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
```

适合：

| 场景 | 说明 |
|------|------|
| 应用已有 MongoDB | 减少基础设施复杂度 |
| 对话状态与文档模型接近 | 存储结构自然 |
| 希望 checkpoint 和业务文档共用数据库 | 运维更简单 |

---

## Redis Checkpointer

安装：

```bash
pip install -U langgraph langgraph-checkpoint-redis
```

同步示例：

```python
from langgraph.checkpoint.redis import RedisSaver

DB_URI = "redis://localhost:6379"

with RedisSaver.from_conn_string(DB_URI) as checkpointer:
    # checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
```

异步版本：

```python
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

async with AsyncRedisSaver.from_conn_string(DB_URI) as checkpointer:
    # await checkpointer.asetup()
    graph = builder.compile(checkpointer=checkpointer)
```

注意：

```text
Redis checkpointer 首次使用也需要 setup/asetup。
```

---

## Oracle Checkpointer

安装：

```bash
pip install -U langgraph langgraph-oracledb
```

同步示例：

```python
from langgraph_oracledb.checkpoint.oracle import OracleSaver

DB_URI = "user/password@localhost:1521/FREEPDB1"

with OracleSaver.from_conn_string(DB_URI) as checkpointer:
    # checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
```

异步版本：

```python
from langgraph_oracledb.checkpoint.oracle import AsyncOracleSaver

async with AsyncOracleSaver.from_conn_string(DB_URI) as checkpointer:
    # await checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
```

适合 Oracle AI Database 或 OCI Autonomous Database 场景。

---

## Short-term Memory in Subgraphs

如果 graph 包含 subgraphs，通常只需要在父图 compile 时传入 checkpointer。

LangGraph 会自动把 checkpointer 传播给子图：

```python
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph


class State(TypedDict):
    foo: str


def subgraph_node_1(state: State):
    return {"foo": state["foo"] + "bar"}


subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()

builder = StateGraph(State)
builder.add_node("node_1", subgraph)
builder.add_edge(START, "node_1")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

如果需要子图自己的 checkpoint history：

```python
subgraph = subgraph_builder.compile(checkpointer=True)
```

区别：

| 配置 | 行为 |
|------|------|
| 子图不传 checkpointer | 继承父图 checkpointer |
| `compile(checkpointer=True)` | 子图拥有自己的 checkpoint history |
| 父图 compile 传 checkpointer | 父图和默认子图都可持久化 |

这会影响 interrupt、stateful continuation 和 time travel 的粒度。

---

## Add Long-term Memory

长期记忆用于存储跨会话的用户级或应用级数据。

最小示例：

```python
from langgraph.graph import StateGraph
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

builder = StateGraph(...)
graph = builder.compile(store=store)
```

长期记忆保存的内容通常包括：

| 类型 | 示例 |
|------|------|
| 用户偏好 | dark mode、语言、写作风格 |
| 用户资料 | 姓名、职业、所在地 |
| 长期事实 | 业务规则、项目背景 |
| 历史偏好 | 喜欢的格式、常用工具 |
| 应用级数据 | 可被多个 thread 共享的知识 |

和短期记忆的最大区别：

```text
长期记忆不是自动塞进 prompt 的；
你需要在节点中显式 search/get，然后把相关内容放进模型输入。
```

---

## Store 与 Namespace

Store 使用 namespace + key 组织数据。

常见写法：

```python
namespace = (user_id, "memories")
```

或者：

```python
namespace = ("memories", user_id)
```

写入：

```python
store.put(namespace, "memory-id", {"data": "User prefers dark mode"})
```

读取/搜索：

```python
memories = store.search(namespace, query="user preference", limit=3)
```

异步版本：

```python
await store.aput(namespace, "memory-id", {"data": "User prefers dark mode"})
memories = await store.asearch(namespace, query="user preference", limit=3)
```

namespace 设计建议：

| 需求 | namespace 示例 |
|------|----------------|
| 用户私有记忆 | `("memories", user_id)` |
| 组织级知识 | `("org", org_id, "knowledge")` |
| 项目级资料 | `("project", project_id, "facts")` |
| 应用配置 | `("app", "settings")` |
| 多租户隔离 | `(tenant_id, "users", user_id, "memories")` |

---

## Access Store Inside Nodes

graph compile 时传入 `store` 后，LangGraph 会把 store 注入到节点运行时。

推荐通过 `Runtime` 对象访问：

```python
from dataclasses import dataclass
import uuid

from langgraph.graph import MessagesState, START, StateGraph
from langgraph.runtime import Runtime


@dataclass
class Context:
    user_id: str


async def call_model(
    state: MessagesState,
    runtime: Runtime[Context],
):
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")

    memories = await runtime.store.asearch(
        namespace,
        query=state["messages"][-1].content,
        limit=3,
    )

    info = "\n".join([d.value["data"] for d in memories])

    await runtime.store.aput(
        namespace,
        str(uuid.uuid4()),
        {"data": "User prefers dark mode"},
    )

    # Use info in model call
    ...
```

创建 graph：

```python
builder = StateGraph(MessagesState, context_schema=Context)
builder.add_node(call_model)
builder.add_edge(START, "call_model")
graph = builder.compile(store=store)
```

调用时传入 context：

```python
graph.invoke(
    {"messages": [{"role": "user", "content": "hi"}]},
    {"configurable": {"thread_id": "1"}},
    context=Context(user_id="1"),
)
```

关键点：

| 对象 | 作用 |
|------|------|
| `context_schema` | 定义运行时上下文结构 |
| `runtime.context` | 读取当前调用的 user_id 等上下文 |
| `runtime.store` | 访问长期记忆 store |
| `thread_id` | 仍然用于短期记忆 |
| `user_id` | 常用于长期记忆 namespace |

---

## 长期记忆生产配置

生产环境建议使用数据库支持的 Store：

| Store | 包 | 说明 |
|-------|----|------|
| PostgresStore | `langgraph-checkpoint-postgres` | 通用持久 store |
| RedisStore | `langgraph-checkpoint-redis` | 低延迟 store |
| OracleStore | `langgraph-oracledb` | 支持 Oracle AI Vector Search |
| MongoDB Store | 文档中占位提及 | 适合 MongoDB 生态 |

长期记忆通常和短期记忆一起配置：

```python
graph = builder.compile(
    checkpointer=checkpointer,
    store=store,
)
```

这样：

```text
checkpointer 保存 thread 内 messages；
store 保存跨 thread 的 user memories。
```

---

## Postgres Store

安装：

```bash
pip install -U "psycopg[binary,pool]" langgraph langgraph-checkpoint-postgres
```

同步示例：

```python
from dataclasses import dataclass
import uuid

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.postgres import PostgresStore


model = init_chat_model(model="claude-haiku-4-5-20251001")


@dataclass
class Context:
    user_id: str


def call_model(state: MessagesState, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    namespace = ("memories", user_id)

    memories = runtime.store.search(
        namespace,
        query=str(state["messages"][-1].content),
    )
    info = "\n".join([d.value["data"] for d in memories])

    last_message = state["messages"][-1]
    if "remember" in last_message.content.lower():
        runtime.store.put(
            namespace,
            str(uuid.uuid4()),
            {"data": "User name is Bob"},
        )

    response = model.invoke(
        [{"role": "system", "content": f"User info: {info}"}]
        + state["messages"]
    )
    return {"messages": response}


DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"

with (
    PostgresStore.from_conn_string(DB_URI) as store,
    PostgresSaver.from_conn_string(DB_URI) as checkpointer,
):
    # store.setup()
    # checkpointer.setup()

    builder = StateGraph(MessagesState, context_schema=Context)
    builder.add_node(call_model)
    builder.add_edge(START, "call_model")

    graph = builder.compile(
        checkpointer=checkpointer,
        store=store,
    )
```

跨 thread 使用同一个 `user_id`：

```python
stream = graph.stream_events(
    {"messages": [{"role": "user", "content": "Hi! Remember: my name is Bob"}]},
    {"configurable": {"thread_id": "1"}},
    version="v3",
    context=Context(user_id="1"),
)

stream = graph.stream_events(
    {"messages": [{"role": "user", "content": "what is my name?"}]},
    {"configurable": {"thread_id": "2"}},
    version="v3",
    context=Context(user_id="1"),
)
```

这里两个不同 thread 都能读取同一个用户的长期记忆。

---

## Redis Store

安装：

```bash
pip install -U langgraph langgraph-checkpoint-redis
```

同步示例：

```python
from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.redis import RedisStore

DB_URI = "redis://localhost:6379"

with (
    RedisStore.from_conn_string(DB_URI) as store,
    RedisSaver.from_conn_string(DB_URI) as checkpointer,
):
    store.setup()
    checkpointer.setup()

    graph = builder.compile(
        checkpointer=checkpointer,
        store=store,
    )
```

异步版本：

```python
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.store.redis.aio import AsyncRedisStore

async with (
    AsyncRedisStore.from_conn_string(DB_URI) as store,
    AsyncRedisSaver.from_conn_string(DB_URI) as checkpointer,
):
    # await store.setup()
    # await checkpointer.asetup()
    graph = builder.compile(checkpointer=checkpointer, store=store)
```

---

## Oracle Store

安装：

```bash
pip install -U langgraph langgraph-oracledb langchain-openai
```

Oracle Store 支持结合 Oracle AI Vector Search 做语义搜索。

同步示例：

```python
from langchain.embeddings import init_embeddings
from langgraph_oracledb.checkpoint.oracle import OracleSaver
from langgraph_oracledb.store.oracle import OracleStore

embeddings = init_embeddings("openai:text-embedding-3-small")

DB_URI = "user/password@localhost:1521/FREEPDB1"

with (
    OracleStore.from_conn_string(
        DB_URI,
        index={"embed": embeddings, "dims": 1536},
    ) as store,
    OracleSaver.from_conn_string(DB_URI) as checkpointer,
):
    store.setup()
    checkpointer.setup()

    graph = builder.compile(
        checkpointer=checkpointer,
        store=store,
    )
```

异步版本：

```python
from langgraph_oracledb.checkpoint.oracle import AsyncOracleSaver
from langgraph_oracledb.store.oracle import AsyncOracleStore

async with (
    AsyncOracleStore.from_conn_string(
        DB_URI,
        index={"embed": embeddings, "dims": 1536},
    ) as store,
    AsyncOracleSaver.from_conn_string(DB_URI) as checkpointer,
):
    await store.setup()
    await checkpointer.setup()
```

---

## Semantic Search

长期记忆可以启用语义搜索，让 agent 根据用户当前输入检索语义相关的记忆。

配置方式：

```python
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore

embeddings = init_embeddings("openai:text-embedding-3-small")

store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
    }
)
```

写入记忆：

```python
store.put(("user_123", "memories"), "1", {"text": "I love pizza"})
store.put(("user_123", "memories"), "2", {"text": "I am a plumber"})
```

搜索：

```python
items = store.search(
    ("user_123", "memories"),
    query="I'm hungry",
    limit=1,
)
```

语义搜索可以把 `"I'm hungry"` 匹配到 `"I love pizza"`。

在节点中使用：

```python
async def chat(state: MessagesState, runtime: Runtime):
    items = await runtime.store.asearch(
        ("user_123", "memories"),
        query=state["messages"][-1].content,
        limit=2,
    )

    memories = "\n".join(item.value["text"] for item in items)
    memories = f"## Memories of user\n{memories}" if memories else ""

    response = await model.ainvoke(
        [
            {
                "role": "system",
                "content": f"You are a helpful assistant.\n{memories}",
            },
            *state["messages"],
        ]
    )
    return {"messages": [response]}
```

---

## Manage Short-term Memory

开启短期记忆后，长对话可能超过模型上下文窗口。

常见处理方式：

| 方法 | 说明 | 是否永久修改 state |
|------|------|--------------------|
| Trim messages | 调用模型前临时裁剪消息 | 否 |
| Delete messages | 从 graph state 删除部分消息 | 是 |
| Summarize messages | 把早期消息总结成摘要 | 是，保留 summary |
| Manage checkpoints | 查看、回放或删除 checkpoint | 取决于操作 |
| Custom strategy | 自定义过滤、压缩、归档 | 取决于实现 |

选择建议：

| 目标 | 推荐 |
|------|------|
| 只想控制本次 prompt token | Trim |
| 想真正减少 state 里的 messages | Delete |
| 想保留早期信息但减少 token | Summarize |
| 想查看历史状态或 debug | Manage checkpoints |

---

## Trim Messages

Trim messages 是在调用 LLM 前裁剪 message history。

它不一定删除 graph state 中的 messages，而是给模型一个裁剪后的输入。

示例：

```python
from langchain_core.messages.utils import (
    count_tokens_approximately,
    trim_messages,
)


def call_model(state: MessagesState):
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=128,
        start_on="human",
        end_on=("human", "tool"),
    )

    response = model.invoke(messages)
    return {"messages": [response]}
```

参数理解：

| 参数 | 说明 |
|------|------|
| `strategy="last"` | 保留最后一段消息 |
| `token_counter` | token 计数函数 |
| `max_tokens` | 保留的最大 token 数 |
| `start_on="human"` | 裁剪后尽量从 human message 开始 |
| `end_on=("human", "tool")` | 保持 provider 接受的消息边界 |

注意：

```text
Trim messages 可能丢掉早期事实。
如果需要保留早期信息，可以结合 summary 或 long-term store。
```

---

## Delete Messages

Delete messages 是从 graph state 中永久删除消息。

需要使用带 `add_messages` reducer 的 state key，例如 `MessagesState`。

删除指定消息：

```python
from langchain.messages import RemoveMessage


def delete_messages(state):
    messages = state["messages"]

    if len(messages) > 2:
        return {
            "messages": [
                RemoveMessage(id=m.id)
                for m in messages[:2]
            ]
        }
```

删除全部消息：

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES


def delete_messages(state):
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES)
        ]
    }
```

完整流程通常是：

```python
builder = StateGraph(MessagesState)
builder.add_sequence([call_model, delete_messages])
builder.add_edge(START, "call_model")
```

重要警告：

```text
删除消息后，要确保剩余 message history 对模型 provider 仍然合法。
```

常见约束：

| Provider 约束 | 风险 |
|---------------|------|
| 有些模型要求消息从 user/human 开始 | 删除后可能以 assistant 开头 |
| assistant tool call 后必须跟对应 tool result | 删除 tool result 会导致请求非法 |
| 多模态消息结构要完整 | 删除部分 content block 可能破坏输入 |

---

## Summarize Messages

Trim 或 Delete 会丢失信息。

Summarize messages 的思路是：

```text
把早期 conversation 压缩成 summary；
state 中保留 summary + 最近几条 messages。
```

可以扩展 `MessagesState`：

```python
from langgraph.graph import MessagesState


class State(MessagesState):
    summary: str
```

手写总结节点：

```python
def summarize_conversation(state: State):
    summary = state.get("summary", "")

    if summary:
        summary_message = (
            f"This is a summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)

    delete_messages = [
        RemoveMessage(id=m.id)
        for m in state["messages"][:-2]
    ]

    return {
        "summary": response.content,
        "messages": delete_messages,
    }
```

也可以使用 `langmem.short_term.SummarizationNode`：

```python
from typing import TypedDict

from langchain.messages import AnyMessage
from langchain_core.messages.utils import count_tokens_approximately
from langmem.short_term import RunningSummary, SummarizationNode


class State(MessagesState):
    context: dict[str, RunningSummary]


class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, RunningSummary]


summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=256,
    max_tokens_before_summary=256,
    max_summary_tokens=128,
)


def call_model(state: LLMInputState):
    response = model.invoke(state["summarized_messages"])
    return {"messages": [response]}


builder = StateGraph(State)
builder.add_node("summarize", summarization_node)
builder.add_node(call_model)
builder.add_edge(START, "summarize")
builder.add_edge("summarize", "call_model")
graph = builder.compile(checkpointer=checkpointer)
```

总结策略适合：

| 场景 | 说明 |
|------|------|
| 长对话客服 | 保留历史意图和关键信息 |
| Agent task | 保留计划、结果、约束 |
| 写作协作 | 保留用户偏好和文稿演变 |
| 项目助手 | 保留上下文但控制 token |

---

## Manage Checkpoints

checkpointer 中保存的是短期记忆的基础数据。

可以管理：

| 操作 | API |
|------|-----|
| 查看当前 thread state | `graph.get_state(config)` |
| 查看指定 checkpoint | `graph.get_state(config_with_checkpoint_id)` |
| 查看历史 | `graph.get_state_history(config)` |
| 直接读 checkpointer | `checkpointer.get_tuple(config)` |
| 列出 checkpoint tuples | `checkpointer.list(config)` |
| 删除 thread 全部 checkpoints | `checkpointer.delete_thread(thread_id)` |

这些能力也和 time travel、debug、interrupt 恢复有关。

---

## View Thread State

通过 Graph API 查看最新 state：

```python
config = {
    "configurable": {
        "thread_id": "1",
        # Optional:
        # "checkpoint_id": "1f029ca3-1f5b-6704-8004-820c16b69a5a",
    }
}

snapshot = graph.get_state(config)
```

返回值通常是 `StateSnapshot`，包含：

| 字段 | 说明 |
|------|------|
| `values` | 当前 state values，例如 messages |
| `next` | 下一步要执行的节点 |
| `config` | thread/checkpoint config |
| `metadata` | 写入来源、step、thread_id 等 |
| `created_at` | checkpoint 创建时间 |
| `parent_config` | 父 checkpoint |
| `tasks` | 待执行任务 |
| `interrupts` | pending interrupts |

也可以直接使用 Checkpointer API：

```python
checkpoint_tuple = checkpointer.get_tuple(config)
```

---

## View Thread History

通过 Graph API 查看历史：

```python
config = {
    "configurable": {
        "thread_id": "1"
    }
}

history = list(graph.get_state_history(config))
```

通过 Checkpointer API：

```python
tuples = list(checkpointer.list(config))
```

历史 checkpoint 常用于：

| 场景 | 说明 |
|------|------|
| Debug | 看每一步 state 如何变化 |
| Time travel | 找到要 replay/fork 的 checkpoint |
| 审计 | 查看某个 thread 的执行轨迹 |
| 错误恢复 | 找到失败前的状态 |

注意：

```text
history 通常按时间倒序返回。
```

---

## Delete Thread Checkpoints

删除某个 thread 的全部 checkpoints：

```python
thread_id = "1"
checkpointer.delete_thread(thread_id)
```

适合：

| 场景 | 说明 |
|------|------|
| 清理测试数据 | 删除本地实验 thread |
| 用户删除会话 | 移除对应 thread state |
| 合规要求 | 清除短期对话记录 |
| 重置任务 | 让同一个 thread_id 重新开始 |

注意：

```text
delete_thread 只删除 checkpointer 中该 thread 的短期记忆；
不会自动删除 Store 中的长期记忆，除非你另外实现对应清理。
```

---

## Database Management

使用数据库支持的 persistence 实现时，通常需要运行 migrations 创建 schema。

官方说明：

```text
大多数数据库特定库会在 checkpointer 或 store 实例上提供 setup() 方法。
但具体方法名和用法要以对应 BaseCheckpointSaver / BaseStore 实现为准。
```

建议：

| 做法 | 说明 |
|------|------|
| 部署时运行 migrations | 更可控，避免请求路径中建表 |
| 服务启动时确保 setup | 小型应用可以接受 |
| 避免每次请求都 setup | 降低延迟和竞争风险 |
| checkpointer 和 store 分别 setup | 两者 schema 不一定相同 |

示例：

```python
with (
    PostgresStore.from_conn_string(DB_URI) as store,
    PostgresSaver.from_conn_string(DB_URI) as checkpointer,
):
    store.setup()
    checkpointer.setup()
```

异步：

```python
async with (
    AsyncPostgresStore.from_conn_string(DB_URI) as store,
    AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
):
    await store.setup()
    await checkpointer.setup()
```

---

## 最佳实践

1. 明确区分 `thread_id` 和 `user_id`。

```text
thread_id 用于短期记忆；
user_id 用于长期记忆 namespace。
```

2. 生产环境不要依赖 `InMemorySaver` 或 `InMemoryStore`。

```text
进程重启后内存数据会丢失。
```

3. 长期记忆写入要有明确触发条件。

```text
不要把所有消息都写进长期记忆；
只保存稳定、有复用价值的信息。
```

4. Store namespace 要支持多租户隔离。

```python
namespace = (tenant_id, "users", user_id, "memories")
```

5. 调用模型前检索长期记忆，并显式放入 system prompt。

```text
Store 不会自动影响模型输入。
```

6. 长对话结合 trim、delete、summary。

```text
trim 控制输入 token；
delete 控制 state 增长；
summary 保留早期信息。
```

7. 删除消息时保持 provider message 格式合法。

```text
尤其注意 tool call 和 tool result 必须配对。
```

8. 需要语义检索时配置 embedding index。

```python
InMemoryStore(index={"embed": embeddings, "dims": 1536})
```

9. 数据库 schema migration 放到部署流程。

```text
checkpointer.setup() / store.setup() 不建议每个请求都跑。
```

10. 使用 `get_state_history` 辅助 debug 和 time travel。

```python
list(graph.get_state_history(config))
```

---

## 故障排查

| 问题 | 常见原因 | 处理方式 |
|------|----------|----------|
| 第二轮对话忘记第一轮 | `thread_id` 不同或未配置 checkpointer | 使用同一个 `thread_id`，compile 时传 checkpointer |
| 进程重启后记忆丢失 | 使用 `InMemorySaver` / `InMemoryStore` | 换成数据库支持的 checkpointer/store |
| 长期记忆跨会话读不到 | namespace 或 `user_id` 不一致 | 统一 namespace 设计，通过 context 传 user_id |
| Store 中有记忆但模型没用 | 未把检索结果放进 prompt | 在 node 中 search 后构造 system message |
| `search` 语义不准 | 未启用 embedding index 或 query 不合适 | 配置 `index`，调整 query 和 memory schema |
| 数据库报表不存在 | 未运行 setup/migration | 首次使用前执行 `setup()` |
| 删除消息后模型报错 | 剩余 messages 不符合 provider 要求 | 保证 human 开头、tool call/result 成对 |
| token 仍然超限 | 只保存了 state，未裁剪 LLM 输入 | 使用 `trim_messages` 或 summary |
| summary 丢失关键信息 | 摘要 prompt 或 token 太小 | 增大 summary token，保留最近原文 |
| 子图没有记忆 | 父图未传 checkpointer 或子图配置不符合预期 | 父图 compile 传 checkpointer，必要时子图 `checkpointer=True` |
| 删除 thread 后长期记忆仍在 | `delete_thread` 只删 checkpoints | 另外删除 Store namespace 中的数据 |

---

## 快速参考

### 短期记忆

```python
from langgraph.checkpoint.memory import InMemorySaver

graph = builder.compile(checkpointer=InMemorySaver())

graph.invoke(
    {"messages": [{"role": "user", "content": "hi"}]},
    {"configurable": {"thread_id": "1"}},
)
```

### 长期记忆

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
graph = builder.compile(store=store)
```

### 同时配置短期和长期记忆

```python
graph = builder.compile(
    checkpointer=checkpointer,
    store=store,
)
```

### Runtime 中访问 Store

```python
async def node(state: MessagesState, runtime: Runtime[Context]):
    namespace = ("memories", runtime.context.user_id)

    items = await runtime.store.asearch(
        namespace,
        query=state["messages"][-1].content,
        limit=3,
    )

    await runtime.store.aput(
        namespace,
        "id",
        {"data": "User prefers dark mode"},
    )
```

### 启用语义搜索

```python
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
    }
)
```

### Trim messages

```python
messages = trim_messages(
    state["messages"],
    strategy="last",
    token_counter=count_tokens_approximately,
    max_tokens=128,
    start_on="human",
    end_on=("human", "tool"),
)
```

### Delete messages

```python
return {
    "messages": [
        RemoveMessage(id=m.id)
        for m in state["messages"][:2]
    ]
}
```

### Delete all messages

```python
return {
    "messages": [
        RemoveMessage(id=REMOVE_ALL_MESSAGES)
    ]
}
```

### 查看当前 state

```python
snapshot = graph.get_state(
    {"configurable": {"thread_id": "1"}}
)
```

### 查看历史

```python
history = list(graph.get_state_history(
    {"configurable": {"thread_id": "1"}}
))
```

### 删除 thread checkpoints

```python
checkpointer.delete_thread("1")
```

### 核心 API

| API | 作用 |
|-----|------|
| `builder.compile(checkpointer=...)` | 开启短期记忆 |
| `builder.compile(store=...)` | 开启长期记忆 |
| `runtime.store.search/asearch` | 检索长期记忆 |
| `runtime.store.put/aput` | 写入长期记忆 |
| `graph.get_state(config)` | 查看 thread 当前 state |
| `graph.get_state_history(config)` | 查看 checkpoint 历史 |
| `checkpointer.delete_thread(thread_id)` | 删除 thread checkpoints |

---

## 资料来源

- LangGraph Memory 官方文档：<https://docs.langchain.com/oss/python/langgraph/add-memory>
- LangGraph Persistence 官方文档：<https://docs.langchain.com/oss/python/langgraph/persistence>
- LangGraph Checkpointers 官方文档：<https://docs.langchain.com/oss/python/langgraph/checkpointers>
- LangGraph Stores 官方文档：<https://docs.langchain.com/oss/python/langgraph/persistence#memory-store>
- LangGraph Subgraphs 官方文档：<https://docs.langchain.com/oss/python/langgraph/use-subgraphs>
