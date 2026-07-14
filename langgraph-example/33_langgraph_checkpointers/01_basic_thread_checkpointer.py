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
    # Annotated + operator.add 表示多次写入 events 时追加，而不是覆盖。
    events: Annotated[list[str], operator.add]
    count: int


def count_node(state: CounterState) -> dict:
    # checkpointer 会在同一个 thread_id 下恢复上一次 checkpoint 的 state，
    # 因此第二次调用时 previous_count 会接着第一次的结果继续增长。
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
    # compile 时传入 checkpointer 后，graph 才会把每个 super-step 的 state 保存为 checkpoint。
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    # thread_id 是读取和写入 checkpoint 的分组键；同一个 thread_id 代表同一条会话/任务线。
    config = {"configurable": {"thread_id": "counter-thread"}}

    first = graph.invoke({"events": [], "count": 0}, config)
    # 第二次调用没有传 count，但会从同一个 thread 的最新 checkpoint 中恢复 count。
    second = graph.invoke({"events": ["manual event before second run"]}, config)

    print("第一次运行:", first)
    print("第二次运行:", second)

    snapshot = graph.get_state(config)
    print("\n最新 checkpoint values:")
    print(snapshot.values)
    # checkpoint_id 可用于定位某个历史 checkpoint，后续 replay/fork 会用到这个能力。
    print("checkpoint_id:", snapshot.config["configurable"]["checkpoint_id"])


if __name__ == "__main__":
    main()
