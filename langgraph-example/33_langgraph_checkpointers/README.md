# LangGraph Checkpointers 参考案例

本目录基于文档：

`study-docs/langgraph-docs/33_LangGraph_Checkpointers_详细指南.md`

提供几个不依赖真实 LLM 的最小案例，用于理解 LangGraph Checkpointer 如何保存 checkpoints，并支撑 thread 级记忆、state inspection、history、replay、fork、interrupt/resume 和 durability modes。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 案例列表

| 文件 | 主题 | 说明 |
|------|------|------|
| `01_basic_thread_checkpointer.py` | 基础 checkpointer | 使用 `InMemorySaver` 和 `thread_id` 保存 thread state |
| `02_get_state_and_history.py` | StateSnapshot | 使用 `get_state` / `get_state_history` 查看 checkpoints |
| `03_replay_from_checkpoint.py` | Replay | 从历史 checkpoint 重新执行后续节点 |
| `04_update_state_fork.py` | Fork | 用 `update_state` 从历史 checkpoint 创建新分支 |
| `05_interrupt_resume.py` | HITL | 用 `interrupt()` 暂停，再用 `Command(resume=...)` 恢复 |
| `06_durability_modes.py` | Durability | 演示 `sync` / `async` / `exit` 三种 durability 调用方式 |

## 运行方式

```bash
python 01_basic_thread_checkpointer.py
python 02_get_state_and_history.py
python 03_replay_from_checkpoint.py
python 04_update_state_fork.py
python 05_interrupt_resume.py
python 06_durability_modes.py
```

## 核心概念速记

| 概念 | 含义 |
|------|------|
| `checkpointer` | 在 super-step 边界保存 graph state |
| `thread_id` | 一条 conversation/task 的 checkpoint 分组键 |
| `checkpoint_id` | 某个历史 checkpoint 的唯一 ID |
| `StateSnapshot.values` | 当前 checkpoint 中的 state values |
| `StateSnapshot.next` | 从该 checkpoint 继续时下一步要执行的节点 |
| `StateSnapshot.metadata` | checkpoint 来源、writes、step 等元数据 |
| `get_state(config)` | 读取最新或指定 checkpoint |
| `get_state_history(config)` | 读取 thread 的 checkpoint 历史 |
| `invoke(None, checkpoint.config)` | 从历史 checkpoint replay |
| `update_state(config, values)` | 创建一个 state update checkpoint，常用于 fork |
| `Command(resume=...)` | 恢复 `interrupt()` 暂停的 graph |

这些示例使用内存 checkpointer，适合学习。生产环境应改用 SQLite、Postgres、Redis、MongoDB、Oracle 等持久化后端。
