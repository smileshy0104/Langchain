# LangGraph Subgraphs 参考案例

本目录基于文档：

`study-docs/langgraph-docs/39_LangGraph_Subgraphs_详细指南.md`

提供几个不依赖真实 LLM 的参考案例，用于理解 LangGraph Subgraphs 中的两种接入方式、父子 state 通信、多层嵌套、persistence 模式、子图状态查看、namespace 隔离和 Event Streaming。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 案例列表

| 文件 | 主题 | 说明 |
|------|------|------|
| `01_call_subgraph_inside_node.py` | Inside a node | 父子 state schema 不同时，用 wrapper 手动做输入输出映射 |
| `02_add_subgraph_as_node.py` | As a node | 父子共享 state key 时，直接把 compiled subgraph 添加为节点 |
| `03_nested_subgraphs.py` | 多层 Subgraphs | parent -> child -> grandchild，每层显式转换接口 |
| `04_persistence_modes.py` | Persistence 三模式 | 对比 per-invocation、per-thread、stateless 子图记忆行为 |
| `05_interrupt_and_state_inspection.py` | Interrupt + State Inspection | 子图中断会暂停顶层 graph，并可查看当前子图 state |
| `06_namespace_isolation.py` | Namespace Isolation | 用唯一 node name 区分多个 subgraph 的执行 namespace |
| `07_stream_subgraph_outputs.py` | Event Streaming | 用 `stream.subgraphs` 和 raw events 观察子图输出 |

## 运行方式

```bash
python 01_call_subgraph_inside_node.py
python 02_add_subgraph_as_node.py
python 03_nested_subgraphs.py
python 04_persistence_modes.py
python 05_interrupt_and_state_inspection.py
python 06_namespace_isolation.py
python 07_stream_subgraph_outputs.py
```

## 核心概念速记

| 概念 | 含义 |
|------|------|
| Subgraph | 被另一个 graph 当作节点使用的 graph |
| Inside a node | 在父节点函数里调用 `subgraph.invoke(...)`，适合 schema 不同或需要适配 |
| As a node | `builder.add_node("name", subgraph)`，适合父子共享 state key |
| 共享 key | 父图和子图都能读写的 state channel |
| 私有 key | 子图内部使用，父图不会自动接收 |
| Per-invocation | 默认模式；每次子图调用 fresh，但单次调用可继承父图 checkpointer |
| Per-thread | `compile(checkpointer=True)`；同一个 thread 内子图 state 可跨调用累积 |
| Stateless | `compile(checkpointer=False)`；无 checkpoint、无 interrupt/durable execution |
| Namespace | Event Streaming / checkpoint 中标识嵌套 graph 路径的名字 |
| `get_state(..., subgraphs=True)` | 查看可静态发现的子图状态，常用于 interrupt 暂停时调试 |

这些示例使用内存实现，适合学习和测试。生产环境应根据需要选择持久化 checkpointer，并避免同一个 per-thread subgraph 被并行调用造成 namespace 冲突。
