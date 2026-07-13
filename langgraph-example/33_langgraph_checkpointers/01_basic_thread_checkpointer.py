"""
案例 1：基础 checkpointer 与 thread_id

目标：
- 使用 InMemorySaver 启用 checkpointing。
- 用 thread_id 把多次 graph 调用串成同一个 thread。
- 展示同一 thread 的 state 会累积。

对应文档概念：
- Checkpointer
- Threads
- Checkpoints
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class CounterState(TypedDict):
    events: Annotated[list[str], operator.add]
    count: int


def count_node(state: CounterState) -> dict:
    previous_count = state.get("count", 0)
    next_count = previous_count + 1
    return {
        "count": next_count,
        "events": [f"count_node ran, count={next_count}"],
    }


def build_graph():
    builder = StateGraph(CounterState)
    builder.add_node("count", count_node)
    builder.add_edge(START, "count")
    builder.add_edge("count", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "counter-thread"}}

    first = graph.invoke({"events": [], "count": 0}, config)
    second = graph.invoke({"events": ["manual event before second run"]}, config)

    print("第一次运行:", first)
    print("第二次运行:", second)

    snapshot = graph.get_state(config)
    print("\n最新 checkpoint values:")
    print(snapshot.values)
    print("checkpoint_id:", snapshot.config["configurable"]["checkpoint_id"])


if __name__ == "__main__":
    main()
