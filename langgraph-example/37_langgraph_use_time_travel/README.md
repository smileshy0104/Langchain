# LangGraph Use Time Travel 参考案例

本目录基于文档：

`study-docs/langgraph-docs/37_LangGraph_Use_Time_Travel_详细指南.md`

提供几个不依赖真实 LLM 的参考案例，用于理解 LangGraph Time Travel 中的 checkpoint history、Replay、Fork、`update_state`、`as_node`、interrupt 重新触发、多 interrupt 中间分叉和 subgraph checkpoint 粒度。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 案例列表

| 文件 | 主题 | 说明 |
|------|------|------|
| `01_state_history_basics.py` | Checkpoint history | 查看 `get_state_history`、`state.next`、`checkpoint_id` |
| `02_replay_from_checkpoint.py` | Replay | 从某个历史 checkpoint 重新执行后续节点 |
| `03_fork_with_update_state.py` | Fork | 用 `update_state` 修改历史 state 并创建新分支 |
| `04_update_state_as_node.py` | `as_node` | 显式指定 update 来自哪个节点，从其 successor 继续 |
| `05_interrupt_time_travel.py` | Interrupts | Time travel 后 `interrupt()` 会重新触发 |
| `06_multiple_interrupts_between.py` | Multiple interrupts | 从两个 interrupt 中间 fork，只重新询问后一个问题 |
| `07_subgraph_checkpoint_granularity.py` | Subgraphs | 对比默认继承 checkpointer 与 `checkpointer=True` 的粒度 |

## 运行方式

```bash
python 01_state_history_basics.py
python 02_replay_from_checkpoint.py
python 03_fork_with_update_state.py
python 04_update_state_as_node.py
python 05_interrupt_time_travel.py
python 06_multiple_interrupts_between.py
python 07_subgraph_checkpoint_granularity.py
```

## 核心概念速记

| 概念 | 含义 |
|------|------|
| `get_state_history(config)` | 获取某个 thread 的 checkpoint 历史，通常 newest first |
| `state.next` | 从这个 checkpoint 继续时下一步要执行的节点 |
| Replay | `graph.invoke(None, checkpoint.config)`，从历史 checkpoint 重新跑后续节点 |
| Fork | `graph.update_state(...)` 创建一个带修改 state 的新 checkpoint |
| `as_node` | 指定这次 state update 被视为哪个节点的输出 |
| interrupt + time travel | checkpoint 后的 `interrupt()` 会重新触发 |
| subgraph checkpoint 粒度 | 默认父图级；`compile(checkpointer=True)` 支持子图内部 checkpoint |

重要提醒：

- Replay 不是读缓存，checkpoint 后的节点会重新执行。
- Fork 不会回滚原 thread，而是创建新 checkpoint 分支。
- 最终 checkpoint 没有 `next` 节点，从它 replay 是 no-op。
- 与 interrupt 结合时，UI 要准备好重新收集 resume 值。
