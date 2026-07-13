"""
案例 6：Graceful Shutdown 模式

目标：
- 展示 RunControl / GraphDrained 的合作式停机模板。
- request_drain() 请求 graph 在 superstep 边界停止。
- 后续可使用同一个 config 通过 graph.invoke(None, config) 恢复。

对应文档概念：
- Graceful Shutdown
- RunControl
- GraphDrained

注意：
- 这是停机模式模板。实际 SIGTERM hook 通常由进程管理器触发。
"""

from __future__ import annotations

import signal
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphDrained
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import RunControl, Runtime


class JobState(TypedDict):
    step: str
    status: str


def step_a(state: JobState) -> dict:
    return {"step": "a", "status": "step_a_done"}


def step_b(state: JobState, runtime: Runtime) -> dict:
    if runtime.drain_requested:
        return {
            "step": "b",
            "status": f"skipped_due_to_drain:{runtime.drain_reason}",
        }
    return {"step": "b", "status": "step_b_done"}


def build_graph():
    builder = StateGraph(JobState)
    builder.add_node("step_a", step_a)
    builder.add_node("step_b", step_b)
    builder.add_edge(START, "step_a")
    builder.add_edge("step_a", "step_b")
    builder.add_edge("step_b", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "drain-demo"}}
    control = RunControl()

    signal.signal(
        signal.SIGTERM,
        lambda *_: control.request_drain("sigterm"),
    )

    control.request_drain("demo-drain-request")

    try:
        result = graph.invoke(
            {"step": "", "status": "started"},
            config,
            control=control,
        )
        print("graph finished naturally:", result)
    except GraphDrained as exc:
        print(f"graph drained safely: {exc.reason}")
        print("resume later with:")
        print("graph.invoke(None, config)")


if __name__ == "__main__":
    main()
