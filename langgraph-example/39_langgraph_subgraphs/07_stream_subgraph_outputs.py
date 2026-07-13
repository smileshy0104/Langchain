"""
案例 7：Stream subgraph outputs

目标：
- 使用 stream.subgraphs typed projection 观察子图输出。
- 同时展示 raw protocol events 中的 namespace。
- 理解 typed projection 适合应用层，raw events 适合底层日志。

对应文档概念：
- Stream Subgraph Outputs
- stream.subgraphs
- Raw Protocol Events
"""

from __future__ import annotations

import warnings
from typing import TypedDict

from langchain_core._api.beta_decorator import LangChainBetaWarning

from langgraph.graph import END, START, StateGraph


warnings.filterwarnings("ignore", category=LangChainBetaWarning)


class State(TypedDict):
    text: str
    trace: str


def child_a(state: State) -> dict:
    return {
        "text": state["text"] + " -> child_a",
        "trace": f"{state['trace']} -> child:a",
    }


def child_b(state: State) -> dict:
    return {
        "text": state["text"] + " -> child_b",
        "trace": f"{state['trace']} -> child:b",
    }


def parent_start(state: State) -> dict:
    return {
        "text": state["text"] + " -> parent",
        "trace": f"{state['trace']} -> parent:start",
    }


def build_graph():
    child_builder = StateGraph(State)
    child_builder.add_node("child_a", child_a)
    child_builder.add_node("child_b", child_b)
    child_builder.add_edge(START, "child_a")
    child_builder.add_edge("child_a", "child_b")
    child_builder.add_edge("child_b", END)
    child_graph = child_builder.compile()

    parent_builder = StateGraph(State)
    parent_builder.add_node("parent_start", parent_start)
    parent_builder.add_node("child_graph", child_graph)
    parent_builder.add_edge(START, "parent_start")
    parent_builder.add_edge("parent_start", "child_graph")
    parent_builder.add_edge("child_graph", END)
    return parent_builder.compile()


def main() -> None:
    graph = build_graph()
    inputs = {"text": "root", "trace": "start"}

    print("typed projection:")
    event_stream = graph.stream_events(inputs, version="v3")
    if hasattr(event_stream, "subgraphs"):
        for subgraph in event_stream.subgraphs:
            print("subgraph:", subgraph.graph_name, subgraph.path)
            for snapshot in subgraph.values:
                print("  value:", snapshot)
    else:
        print("stream.subgraphs is not available in this installed version")

    print("\nraw protocol values:")
    for event in graph.stream_events(inputs, version="v3"):
        if event.get("method") == "values":
            print(event["params"]["namespace"], event["params"]["data"])


if __name__ == "__main__":
    main()
