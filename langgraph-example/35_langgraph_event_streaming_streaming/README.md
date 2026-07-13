# LangGraph Event Streaming 与 Streaming 参考案例

本目录基于文档：

`study-docs/langgraph-docs/35_LangGraph_Event_Streaming_Streaming_详细指南.md`

提供几个不依赖真实 LLM 的参考案例，用于理解两套流式接口：

- `graph.stream_events(..., version="v3")`：应用层 typed projections。
- `graph.stream(..., version="v2")`：底层 runtime stream modes / `StreamPart`。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 案例列表

| 文件 | 主题 | 说明 |
|------|------|------|
| `01_event_streaming_typed_projections.py` | Typed projections | 使用 `stream.values`、`stream.output`、raw events 理解 projection 层 |
| `02_streaming_modes_values_updates.py` | Streaming v2 modes | 对比 `updates`、`values` 和多 mode 的 `StreamPart` |
| `03_custom_data_stream_writer.py` | Custom data | 用 `get_stream_writer()` 从节点发出业务进度 |
| `04_subgraph_outputs.py` | Subgraph streaming | Event Streaming 的 `stream.subgraphs` 与 Streaming 的 `subgraphs=True` |
| `05_interrupt_stream_events.py` | Interrupt resume | 用 `stream.interrupted`、`stream.interrupts` 和 `Command(resume=...)` 处理 HITL |
| `06_raw_protocol_events_vs_projections.py` | Raw vs projection | 对比直接遍历 protocol events 和读取 typed projections |

## 运行方式

```bash
python 01_event_streaming_typed_projections.py
python 02_streaming_modes_values_updates.py
python 03_custom_data_stream_writer.py
python 04_subgraph_outputs.py
python 05_interrupt_stream_events.py
python 06_raw_protocol_events_vs_projections.py
```

## 核心概念速记

| 概念 | 含义 |
|------|------|
| Event Streaming | `stream_events(..., version="v3")`，面向应用层消费 |
| Typed projections | 从 raw events 投影出的结构化视图，如 `stream.values`、`stream.output` |
| Streaming | `stream(..., version="v2")`，面向底层 runtime modes |
| StreamPart | v2 中统一 chunk 结构，通常包含 `type`、`ns`、`data` |
| `stream.values` | graph state snapshots |
| `stream.output` | graph 最终输出 |
| `stream.subgraphs` | nested graph executions |
| `stream.interrupts` | human-in-the-loop interrupt payloads |
| `get_stream_writer()` | 在节点中写入 custom stream data |

这些示例使用普通 `StateGraph` 和内存 checkpointer，适合学习 streaming 机制。涉及真实 chat model token 的 `stream.messages` 需要接入真实 LLM，本目录用普通节点示例重点演示 state/custom/subgraph/interrupt。
