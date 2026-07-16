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


# 避免 beta API 警告干扰示例输出。
warnings.filterwarnings("ignore", category=LangChainBetaWarning)


class State(TypedDict):
    # text 是共享业务字段，会被父图和子图连续修改。
    text: str
    # trace 用于观察执行路径。
    trace: str


def child_a(state: State) -> dict:
    """子图第一个节点。"""

    return {
        "text": state["text"] + " -> child_a",
        "trace": f"{state['trace']} -> child:a",
    }


def child_b(state: State) -> dict:
    """子图第二个节点。"""

    return {
        "text": state["text"] + " -> child_b",
        "trace": f"{state['trace']} -> child:b",
    }


def parent_start(state: State) -> dict:
    """父图起始节点：先修改共享状态，再进入子图。"""

    return {
        "text": state["text"] + " -> parent",
        "trace": f"{state['trace']} -> parent:start",
    }


def build_graph():
    """构建父图 + 子图，并演示对子图输出的流式观察。"""

    # 子图内部包含两个连续节点。
    child_builder = StateGraph(State)
    child_builder.add_node("child_a", child_a)
    child_builder.add_node("child_b", child_b)
    child_builder.add_edge(START, "child_a")
    child_builder.add_edge("child_a", "child_b")
    child_builder.add_edge("child_b", END)
    child_graph = child_builder.compile()

    # 父图先执行 parent_start，再把 child_graph 当作一个节点执行。
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
    # stream_events(version="v3") 在新版本中可能提供 typed projection，如 subgraphs。
    event_stream = graph.stream_events(inputs, version="v3")
    if hasattr(event_stream, "subgraphs"):
        # typed projection 更适合应用层代码：按子图对象访问 graph_name/path/values。
        for subgraph in event_stream.subgraphs:
            print("subgraph:", subgraph.graph_name, subgraph.path)
            for snapshot in subgraph.values:
                print("  value:", snapshot)
    else:
        print("stream.subgraphs is not available in this installed version")

    print("\nraw protocol values:")
    # raw events 更接近底层协议，适合调试 namespace 和事件结构。
    for event in graph.stream_events(inputs, version="v3"):
        if event.get("method") == "values":
            print(event["params"]["namespace"], event["params"]["data"])


if __name__ == "__main__":
    main()
