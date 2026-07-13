# LangGraph Interrupts 参考案例

本目录基于文档：

`study-docs/langgraph-docs/36_LangGraph_Interrupts_详细指南.md`

提供几个不依赖真实 LLM 的参考案例，用于理解 LangGraph 中动态 `interrupt()`、`Command(resume=...)`、多 interrupt、审批/编辑、工具调用审核、人工输入校验和静态断点调试。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 案例列表

| 文件 | 主题 | 说明 |
|------|------|------|
| `01_basic_interrupt_resume.py` | 基础动态 interrupt | 用 `stream_events` 暂停，并用 `Command(resume=...)` 恢复 |
| `02_invoke_interrupt_api.py` | invoke API | 使用 `graph.invoke()` 读取 `__interrupt__` 并恢复 |
| `03_multiple_interrupts_resume_map.py` | 多 interrupt | 并行分支同时暂停，用 interrupt id map 一次性恢复 |
| `04_approve_and_edit_state.py` | 审批与编辑 | 审批动作、人工编辑 state 后继续 |
| `05_tool_like_interrupt_review.py` | 工具调用审核 | 在 tool-like 函数里调用 `interrupt()`，允许批准/修改/取消 |
| `06_validate_input_conditional_loop.py` | 输入校验 | 每次节点调用只 `interrupt()` 一次，用 conditional edge 重新提问 |
| `07_static_breakpoints.py` | 静态断点 | `interrupt_before` / `interrupt_after` 用于调试 |

## 运行方式

```bash
python 01_basic_interrupt_resume.py
python 02_invoke_interrupt_api.py
python 03_multiple_interrupts_resume_map.py
python 04_approve_and_edit_state.py
python 05_tool_like_interrupt_review.py
python 06_validate_input_conditional_loop.py
python 07_static_breakpoints.py
```

## 核心概念速记

| 概念 | 含义 |
|------|------|
| `interrupt(payload)` | 在节点中动态暂停 graph，并把 payload 返回给调用方 |
| `Command(resume=value)` | 恢复暂停的 graph，`value` 会成为 `interrupt()` 返回值 |
| `thread_id` | 恢复游标，resume 必须使用同一个 `thread_id` |
| `stream.interrupted` | Event Streaming 中判断本轮是否暂停 |
| `stream.interrupts` | Event Streaming 中读取 pending interrupt payloads |
| `__interrupt__` | `invoke()` API 中的 interrupt 返回位置 |
| 多 interrupt resume map | `Command(resume={interrupt.id: answer})` |
| 静态断点 | `interrupt_before` / `interrupt_after`，用于调试而不是生产 HITL |

重要规则：

- 恢复时节点会从开头重新执行，不是从 `interrupt()` 那一行继续。
- 不要用裸 `try/except Exception` 包住 `interrupt()`。
- 同一节点内多个 `interrupt()` 的顺序必须稳定。
- `interrupt()` payload 应保持 JSON-serializable。
- `interrupt()` 前的不可逆副作用必须幂等，或移到 `interrupt()` 后。
