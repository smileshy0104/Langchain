"""
案例 6：Namespace isolation with named subgraphs

目标：
- 构建 fruit_agent 和 veggie_agent 两个子图。
- 通过不同父图 node name 获得稳定 namespace。
- 用 raw stream events 观察 namespace 路径。

对应文档概念：
- Namespace Isolation
- 多个 per-thread subgraphs
- 通过 add_node 添加的 subgraph 自动基于节点名获得 namespace
"""

from __future__ import annotations

import warnings
from typing import TypedDict

from langchain_core._api.beta_decorator import LangChainBetaWarning

from langgraph.graph import END, START, StateGraph


warnings.filterwarnings("ignore", category=LangChainBetaWarning)


class State(TypedDict):
    topic: str
    trace: str


def fruit_step(state: State) -> dict:
    return {"trace": f"{state['trace']} -> fruit_agent({state['topic']})"}


def veggie_step(state: State) -> dict:
    return {"trace": f"{state['trace']} -> veggie_agent({state['topic']})"}


def build_named_subgraph(name: str, step):
    builder = StateGraph(State)
    builder.add_node(name, step)
    builder.add_edge(START, name)
    builder.add_edge(name, END)
    return builder.compile(checkpointer=True)


def build_graph():
    fruit_agent = build_named_subgraph("fruit_step", fruit_step)
    veggie_agent = build_named_subgraph("veggie_step", veggie_step)

    builder = StateGraph(State)
    builder.add_node("fruit_agent", fruit_agent)
    builder.add_node("veggie_agent", veggie_agent)
    builder.add_edge(START, "fruit_agent")
    builder.add_edge("fruit_agent", "veggie_agent")
    builder.add_edge("veggie_agent", END)
    return builder.compile()


def main() -> None:
    graph = build_graph()
    result = graph.invoke({"topic": "inventory", "trace": "start"})
    print("result:", result)

    print("\nraw value namespaces:")
    for event in graph.stream_events({"topic": "inventory", "trace": "start"}, version="v3"):
        if event.get("method") == "values":
            namespace = event["params"]["namespace"]
            data = event["params"]["data"]
            print(namespace, data)


if __name__ == "__main__":
    main()
