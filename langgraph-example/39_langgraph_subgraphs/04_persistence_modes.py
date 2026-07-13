"""
案例 4：Subgraph persistence modes

目标：
- Per-invocation：默认模式，每次子图调用都从 fresh private state 开始。
- Per-thread：checkpointer=True，同一个 thread 内子图 private state 会累积。
- Stateless：checkpointer=False，无子图 checkpoint。

对应文档概念：
- Subgraph Persistence
- Per-invocation
- Per-thread
- Stateless
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class ParentState(TypedDict):
    query: str
    result: str


class ExpertState(TypedDict):
    query: str
    result: str
    private_notes: Annotated[list[str], operator.add]


def expert_node(state: ExpertState) -> dict:
    previous_notes = state.get("private_notes", [])
    return {
        "result": f"previous_private_notes={len(previous_notes)}",
        "private_notes": [state["query"]],
    }


def build_expert_graph(mode: Literal["per-invocation", "per-thread", "stateless"]):
    builder = StateGraph(ExpertState)
    builder.add_node("expert_node", expert_node)
    builder.add_edge(START, "expert_node")
    builder.add_edge("expert_node", END)

    if mode == "per-thread":
        return builder.compile(checkpointer=True)
    if mode == "stateless":
        return builder.compile(checkpointer=False)
    return builder.compile()


def build_parent_graph(mode: Literal["per-invocation", "per-thread", "stateless"]):
    expert = build_expert_graph(mode)
    builder = StateGraph(ParentState)
    builder.add_node("expert", expert)
    builder.add_edge(START, "expert")
    builder.add_edge("expert", END)
    return builder.compile(checkpointer=InMemorySaver())


def run_mode(mode: Literal["per-invocation", "per-thread", "stateless"]) -> None:
    graph = build_parent_graph(mode)
    config = {"configurable": {"thread_id": f"thread-{mode}"}}

    first = graph.invoke({"query": "apples", "result": ""}, config)
    second = graph.invoke({"query": "bananas", "result": ""}, config)

    print(f"{mode}:")
    print("  first :", first["result"])
    print("  second:", second["result"])


def main() -> None:
    run_mode("per-invocation")
    run_mode("per-thread")
    run_mode("stateless")


if __name__ == "__main__":
    main()
