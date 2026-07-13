# LangGraph Persistence 与 Stores 参考案例

本目录基于文档：

`study-docs/langgraph-docs/32_LangGraph_Persistence_Stores_详细指南.md`

提供几个不依赖真实 LLM 的最小参考案例，用于理解 LangGraph 中 `checkpointer`、`thread_id`、`Store`、`namespace`、checkpoint history、replay/fork 等概念。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 案例列表

| 文件 | 主题 | 说明 |
|------|------|------|
| `01_short_term_checkpointer.py` | 短期记忆 | 用 `InMemorySaver` 和 `thread_id` 保存单个会话状态 |
| `02_thread_isolation.py` | Thread 隔离 | 同一个 graph 中不同 `thread_id` 互不影响 |
| `03_long_term_store.py` | 长期记忆 | 用 `InMemoryStore`、`namespace`、`key/value` 保存跨会话信息 |
| `04_checkpointer_store_together.py` | 组合模式 | `checkpointer` 管当前 thread，`store` 管跨 thread 用户记忆 |
| `05_state_history_replay_fork.py` | 历史与分叉 | 用 `get_state_history` 找 checkpoint，并用 `update_state` 创建 fork |

## 运行方式

```bash
python 01_short_term_checkpointer.py
python 02_thread_isolation.py
python 03_long_term_store.py
python 04_checkpointer_store_together.py
python 05_state_history_replay_fork.py
```

## 核心概念速记

| 概念 | 含义 |
|------|------|
| `checkpointer` | 保存 graph state checkpoints，用于短期、线程级记忆 |
| `thread_id` | 标识一条 conversation/task，是读取 checkpoint 的游标 |
| `Store` | 保存应用定义的长期 key-value 数据 |
| `namespace` | Store 中的数据分组，常用于用户、组织、项目隔离 |
| `checkpoint_id` | 某个历史 checkpoint 的唯一标识 |
| `state.next` | 从该 checkpoint 继续时，下一步要执行的节点 |
| `update_state` | 基于某个 checkpoint 写入新 state，创建分支 |

这些示例使用内存实现，适合学习和测试。生产环境应改用 Postgres、Redis、MongoDB、Oracle 等持久化 backend。
