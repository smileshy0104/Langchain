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
    # 父图输入查询。
    query: str
    # 子图写回给父图的结果。
    result: str


class ExpertState(TypedDict):
    # query/result 与父图共享，因此子图可以读父图 query、写父图 result。
    query: str
    result: str
    # private_notes 是子图私有状态，用来观察不同 persistence mode 下是否会累积。
    private_notes: Annotated[list[str], operator.add]


def expert_node(state: ExpertState) -> dict:
    """专家子图节点：报告已有私有笔记数量，并把本次 query 追加为私有笔记。"""

    previous_notes = state.get("private_notes", [])
    return {
        "result": f"previous_private_notes={len(previous_notes)}",
        "private_notes": [state["query"]],
    }


def build_expert_graph(mode: Literal["per-invocation", "per-thread", "stateless"]):
    """根据 mode 构建不同持久化行为的子图。"""

    builder = StateGraph(ExpertState)
    builder.add_node("expert_node", expert_node)
    builder.add_edge(START, "expert_node")
    builder.add_edge("expert_node", END)

    if mode == "per-thread":
        # 子图拥有按 thread 维度保存的私有 checkpoint，因此 private_notes 会跨调用累积。
        return builder.compile(checkpointer=True)
    if mode == "stateless":
        # 显式关闭子图 checkpoint：子图不保存自己的私有状态。
        return builder.compile(checkpointer=False)
    # 默认 per-invocation：每次父图调用子图时，子图私有状态都是新的。
    return builder.compile()


def build_parent_graph(mode: Literal["per-invocation", "per-thread", "stateless"]):
    """构建父图，并把 expert 子图作为一个节点。"""

    expert = build_expert_graph(mode)
    builder = StateGraph(ParentState)
    builder.add_node("expert", expert)
    builder.add_edge(START, "expert")
    builder.add_edge("expert", END)

    # 父图启用 checkpointer 后，同一个 thread_id 下可以连续调用两次进行对比。
    return builder.compile(checkpointer=InMemorySaver())


def run_mode(mode: Literal["per-invocation", "per-thread", "stateless"]) -> None:
    """连续调用两次同一 mode 的图，观察第二次是否看到第一次的子图私有状态。"""

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
