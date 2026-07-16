"""
案例 4：Subgraph outputs

目标：
- 创建父图 + 子图。
- Event Streaming 中使用 stream.subgraphs typed projection。
- Streaming v2 中使用 subgraphs=True 和 chunk["ns"]。

对应文档概念：
- Stream Subgraphs
- Subgraph Outputs
- stream.subgraphs
- subgraphs=True
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    # text 在父图和子图节点中逐步拼接，用来观察数据流经了哪些节点。
    text: str
    # log 使用追加 reducer，父图和子图的日志会累积在同一个 state 字段里。
    log: Annotated[list[str], operator.add]


# 子图第一个节点：追加 child:a 日志并改写 text。
def child_step_a(state: State) -> dict:
    return {"log": ["child:a"], "text": state["text"] + " -> child_a"}


# 子图第二个节点：追加 child:b 日志并继续改写 text。
def child_step_b(state: State) -> dict:
    return {"log": ["child:b"], "text": state["text"] + " -> child_b"}


# 父图节点：在进入子图前先写入父图自己的状态更新。
def parent_step(state: State) -> dict:
    return {"log": ["parent:start"], "text": state["text"] + " -> parent"}


# 构建父图 + 子图：子图作为父图中的一个节点 child_graph。
def build_graph():
    # 先编译子图：child_a -> child_b。
    child_builder = StateGraph(State)
    child_builder.add_node("child_a", child_step_a)
    child_builder.add_node("child_b", child_step_b)
    child_builder.add_edge(START, "child_a")
    child_builder.add_edge("child_a", "child_b")
    child_builder.add_edge("child_b", END)
    child_graph = child_builder.compile()

    # 再把已编译的 child_graph 注册为父图中的一个节点。
    parent_builder = StateGraph(State)
    parent_builder.add_node("parent_step", parent_step)
    parent_builder.add_node("child_graph", child_graph)
    parent_builder.add_edge(START, "parent_step")
    parent_builder.add_edge("parent_step", "child_graph")
    parent_builder.add_edge("child_graph", END)
    return parent_builder.compile()


# 主函数演示两种读取子图输出的方式。
def main() -> None:
    graph = build_graph()
    inputs = {"text": "root", "log": []}

    print("Event Streaming stream.subgraphs:")
    event_stream = graph.stream_events(inputs, version="v3")
    # stream.subgraphs 是 typed projection：每个 subgraph 对象都有 graph_name/path/values 等结构化字段。
    for subgraph in event_stream.subgraphs:
        print("subgraph:", subgraph.graph_name, subgraph.path)
        # subgraph.values 只展示该子图内部产生的 state snapshots。
        for snapshot in subgraph.values:
            print("  value:", snapshot)

    print("\nStreaming v2 with subgraphs=True:")
    for chunk in graph.stream(
        inputs,
        stream_mode="updates",
        subgraphs=True,
        version="v2",
    ):
        # subgraphs=True 后，chunk["ns"] 会标识事件来自父图根命名空间还是某个子图路径。
        print("ns=", chunk["ns"], "type=", chunk["type"], "data=", chunk["data"])


if __name__ == "__main__":
    main()
