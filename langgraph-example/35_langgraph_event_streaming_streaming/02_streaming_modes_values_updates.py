"""
案例 2：Streaming API v2 stream modes

目标：
- 使用 graph.stream(..., version="v2")。
- 对比 stream_mode="updates" 与 stream_mode="values"。
- 使用多个 stream modes，读取统一 StreamPart。

对应文档概念：
- Streaming Quickstart
- Stream Output Format v2
- values / updates
- Multiple Modes
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class CounterState(TypedDict):
    count: int
    log: Annotated[list[str], operator.add]


def increment(state: CounterState) -> dict:
    new_count = state["count"] + 1
    return {"count": new_count, "log": [f"increment -> {new_count}"]}


def double(state: CounterState) -> dict:
    new_count = state["count"] * 2
    return {"count": new_count, "log": [f"double -> {new_count}"]}


def build_graph():
    builder = StateGraph(CounterState)
    builder.add_node("increment", increment)
    builder.add_node("double", double)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", "double")
    builder.add_edge("double", END)
    return builder.compile()


def print_chunks(title: str, chunks) -> None:
    print(f"\n{title}")
    for chunk in chunks:
        print(chunk)


def main() -> None:
    graph = build_graph()
    inputs = {"count": 1, "log": []}

    print_chunks(
        "updates mode: 每个节点的 state delta",
        graph.stream(inputs, stream_mode="updates", version="v2"),
    )

    print_chunks(
        "values mode: 每步后的完整 state",
        graph.stream(inputs, stream_mode="values", version="v2"),
    )

    print_chunks(
        "multiple modes: 统一 StreamPart，读 chunk['type'] / chunk['data']",
        graph.stream(inputs, stream_mode=["updates", "values"], version="v2"),
    )


if __name__ == "__main__":
    main()
