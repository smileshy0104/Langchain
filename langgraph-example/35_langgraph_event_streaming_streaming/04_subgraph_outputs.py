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
    text: str
    log: Annotated[list[str], operator.add]


def child_step_a(state: State) -> dict:
    return {"log": ["child:a"], "text": state["text"] + " -> child_a"}


def child_step_b(state: State) -> dict:
    return {"log": ["child:b"], "text": state["text"] + " -> child_b"}


def parent_step(state: State) -> dict:
    return {"log": ["parent:start"], "text": state["text"] + " -> parent"}


def build_graph():
    child_builder = StateGraph(State)
    child_builder.add_node("child_a", child_step_a)
    child_builder.add_node("child_b", child_step_b)
    child_builder.add_edge(START, "child_a")
    child_builder.add_edge("child_a", "child_b")
    child_builder.add_edge("child_b", END)
    child_graph = child_builder.compile()

    parent_builder = StateGraph(State)
    parent_builder.add_node("parent_step", parent_step)
    parent_builder.add_node("child_graph", child_graph)
    parent_builder.add_edge(START, "parent_step")
    parent_builder.add_edge("parent_step", "child_graph")
    parent_builder.add_edge("child_graph", END)
    return parent_builder.compile()


def main() -> None:
    graph = build_graph()
    inputs = {"text": "root", "log": []}

    print("Event Streaming stream.subgraphs:")
    event_stream = graph.stream_events(inputs, version="v3")
    for subgraph in event_stream.subgraphs:
        print("subgraph:", subgraph.graph_name, subgraph.path)
        for snapshot in subgraph.values:
            print("  value:", snapshot)

    print("\nStreaming v2 with subgraphs=True:")
    for chunk in graph.stream(
        inputs,
        stream_mode="updates",
        subgraphs=True,
        version="v2",
    ):
        print("ns=", chunk["ns"], "type=", chunk["type"], "data=", chunk["data"])


if __name__ == "__main__":
    main()
