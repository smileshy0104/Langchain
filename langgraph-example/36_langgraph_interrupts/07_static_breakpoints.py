"""
案例 7：静态断点调试

目标：
- 使用 interrupt_before / interrupt_after 设置静态断点。
- 用 graph.invoke(None, config) 从断点继续。
- 区分 static breakpoint 和业务里的 dynamic interrupt()。

对应文档概念：
- Debugging with Static Interrupts
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class DebugState(TypedDict):
    log: Annotated[list[str], operator.add]


def node_a(state: DebugState) -> dict:
    return {"log": ["node_a"]}


def node_b(state: DebugState) -> dict:
    return {"log": ["node_b"]}


def build_graph():
    builder = StateGraph(DebugState)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)

    return builder.compile(
        checkpointer=InMemorySaver(),
        interrupt_before=["node_b"],
    )


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "static-breakpoint"}}

    first = graph.invoke({"log": []}, config)
    print("paused before node_b:", first)
    print("state:", graph.get_state(config).values)

    resumed = graph.invoke(None, config)
    print("resumed:", resumed)


if __name__ == "__main__":
    main()
