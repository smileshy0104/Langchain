"""
案例 7：Manage checkpoints

目标：
- 查看当前 thread state。
- 查看 checkpoint history。
- 直接通过 checkpointer 列出 checkpoint tuples。
- 删除某个 thread 的全部 checkpoints。

对应文档概念：
- Manage Checkpoints
- View Thread State
- View Thread History
- Delete Thread Checkpoints
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class CounterState(TypedDict):
    log: Annotated[list[str], operator.add]
    count: int


def increment(state: CounterState) -> dict:
    next_count = state.get("count", 0) + 1
    return {"count": next_count, "log": [f"increment -> {next_count}"]}


def build_graph():
    checkpointer = InMemorySaver()
    builder = StateGraph(CounterState)
    builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    return builder.compile(checkpointer=checkpointer), checkpointer


def main() -> None:
    graph, checkpointer = build_graph()
    config = {"configurable": {"thread_id": "checkpoint-admin"}}

    graph.invoke({"log": [], "count": 0}, config)
    graph.invoke({"log": [], "count": 10}, config)

    snapshot = graph.get_state(config)
    print("current values:", snapshot.values)
    print("current next:", snapshot.next)

    print("\ngraph.get_state_history:")
    for item in graph.get_state_history(config):
        checkpoint_id = item.config["configurable"]["checkpoint_id"]
        print("-", checkpoint_id, item.values)

    print("\ncheckpointer.list:")
    for checkpoint_tuple in checkpointer.list(config):
        checkpoint_id = checkpoint_tuple.config["configurable"]["checkpoint_id"]
        print("-", checkpoint_id)

    checkpointer.delete_thread("checkpoint-admin")
    remaining = list(checkpointer.list(config))
    print("\nafter delete_thread:", remaining)


if __name__ == "__main__":
    main()
