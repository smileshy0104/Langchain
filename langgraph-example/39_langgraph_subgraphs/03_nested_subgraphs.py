"""
案例 3：Nested subgraphs

目标：
- parent 调用 child。
- child 再调用 grandchild。
- 每层 state schema 不同，每层都显式做输入输出转换。

对应文档概念：
- 多层 Subgraphs
- 父图 key 不会自动进入子图
- 子图 key 不会自动回到父图
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class GrandchildState(TypedDict):
    grandchild_text: str


class ChildState(TypedDict):
    child_text: str


class ParentState(TypedDict):
    parent_text: str


def grandchild_node(state: GrandchildState) -> dict:
    return {"grandchild_text": f"grandchild({state['grandchild_text']})"}


def build_grandchild_graph():
    builder = StateGraph(GrandchildState)
    builder.add_node("grandchild_node", grandchild_node)
    builder.add_edge(START, "grandchild_node")
    builder.add_edge("grandchild_node", END)
    return builder.compile()


GRANDCHILD = build_grandchild_graph()


def child_calls_grandchild(state: ChildState) -> dict:
    output = GRANDCHILD.invoke(
        {"grandchild_text": f"mapped from child: {state['child_text']}"}
    )
    return {"child_text": f"child({output['grandchild_text']})"}


def build_child_graph():
    builder = StateGraph(ChildState)
    builder.add_node("child_calls_grandchild", child_calls_grandchild)
    builder.add_edge(START, "child_calls_grandchild")
    builder.add_edge("child_calls_grandchild", END)
    return builder.compile()


CHILD = build_child_graph()


def parent_calls_child(state: ParentState) -> dict:
    output = CHILD.invoke({"child_text": f"mapped from parent: {state['parent_text']}"})
    return {"parent_text": f"parent({output['child_text']})"}


def build_graph():
    builder = StateGraph(ParentState)
    builder.add_node("parent_calls_child", parent_calls_child)
    builder.add_edge(START, "parent_calls_child")
    builder.add_edge("parent_calls_child", END)
    return builder.compile()


def main() -> None:
    graph = build_graph()
    print(graph.invoke({"parent_text": "root"}))


if __name__ == "__main__":
    main()
