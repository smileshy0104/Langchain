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
    # count 是普通字段，每次节点返回的新值会覆盖旧值。
    count: int
    # log 使用追加 reducer，用来保留每个节点的执行日志。
    log: Annotated[list[str], operator.add]


# 第一步：把 count 加 1，并返回本节点的 state delta。
def increment(state: CounterState) -> dict:
    new_count = state["count"] + 1
    return {"count": new_count, "log": [f"increment -> {new_count}"]}


# 第二步：把当前 count 乘 2，并继续追加日志。
def double(state: CounterState) -> dict:
    new_count = state["count"] * 2
    return {"count": new_count, "log": [f"double -> {new_count}"]}


# 构建两步计数 graph，用来对比不同 stream_mode 的输出形态。
def build_graph():
    builder = StateGraph(CounterState)
    builder.add_node("increment", increment)
    builder.add_node("double", double)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", "double")
    builder.add_edge("double", END)
    return builder.compile()


# 简单打印工具：把每种 streaming 模式返回的 chunk 原样输出。
def print_chunks(title: str, chunks) -> None:
    print(f"\n{title}")
    for chunk in chunks:
        print(chunk)


# 主函数演示 v2 Streaming API 的 updates、values 和 multiple modes。
def main() -> None:
    graph = build_graph()
    inputs = {"count": 1, "log": []}

    print_chunks(
        "updates mode: 每个节点的 state delta",
        # updates 只输出“每个节点本次写入了什么”，例如 {"increment": {...}}。
        graph.stream(inputs, stream_mode="updates", version="v2"),
    )

    print_chunks(
        "values mode: 每步后的完整 state",
        # values 输出每个 step 后完整 state，包含 input 初始快照和每次更新后的快照。
        graph.stream(inputs, stream_mode="values", version="v2"),
    )

    print_chunks(
        "multiple modes: 统一 StreamPart，读 chunk['type'] / chunk['data']",
        # 多个 mode 同时开启时，每个 chunk 都是统一格式：type 标识模式，data 存放对应数据。
        graph.stream(inputs, stream_mode=["updates", "values"], version="v2"),
    )


if __name__ == "__main__":
    main()
