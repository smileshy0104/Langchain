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


# stream_events 在部分版本中仍标记为 beta；示例中屏蔽 beta 警告，保持输出简洁。
warnings.filterwarnings("ignore", category=LangChainBetaWarning)


class State(TypedDict):
    # 两个子图共享同一套状态结构。
    topic: str
    # 通过 trace 观察 fruit_agent 和 veggie_agent 的执行顺序。
    trace: str


def fruit_step(state: State) -> dict:
    """fruit_agent 子图中的业务节点。"""

    return {"trace": f"{state['trace']} -> fruit_agent({state['topic']})"}


def veggie_step(state: State) -> dict:
    """veggie_agent 子图中的业务节点。"""

    return {"trace": f"{state['trace']} -> veggie_agent({state['topic']})"}


def build_named_subgraph(name: str, step):
    """构建一个只有单个节点的子图，节点名由调用方传入。"""

    builder = StateGraph(State)
    builder.add_node(name, step)
    builder.add_edge(START, name)
    builder.add_edge(name, END)

    # checkpointer=True 让子图拥有 per-thread checkpoint namespace，便于观察命名空间隔离。
    return builder.compile(checkpointer=True)


def build_graph():
    """构建包含两个命名子图节点的父图。"""

    fruit_agent = build_named_subgraph("fruit_step", fruit_step)
    veggie_agent = build_named_subgraph("veggie_step", veggie_step)

    builder = StateGraph(State)
    # 父图节点名 fruit_agent/veggie_agent 会成为 stream/checkpoint namespace 路径的一部分。
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
    # raw stream events 会暴露 namespace，可以看到事件来自顶层图还是某个子图路径。
    for event in graph.stream_events({"topic": "inventory", "trace": "start"}, version="v3"):
        if event.get("method") == "values":
            namespace = event["params"]["namespace"]
            data = event["params"]["data"]
            print(namespace, data)


if __name__ == "__main__":
    main()
