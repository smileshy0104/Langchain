"""
案例 2：Add a subgraph as a node

目标：
- 父图和子图共享 foo/log。
- 子图有自己的私有 key：bar。
- 父图直接 add_node("child_graph", subgraph)，不写 wrapper。

对应文档概念：
- Add a Subgraph as a Node
- 共享 State Schema
- 子图私有 key 不会自动暴露给父图
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class ParentState(TypedDict):
    foo: str
    trace: str


class ChildState(TypedDict):
    foo: str
    bar: str
    trace: str


def parent_prepare(state: ParentState) -> dict:
    return {
        "foo": f"parent({state['foo']})",
        "trace": f"{state['trace']} -> parent:prepare",
    }


def child_private_step(state: ChildState) -> dict:
    return {
        "bar": "child-private",
        "trace": f"{state['trace']} -> child:set_bar",
    }


def child_shared_step(state: ChildState) -> dict:
    return {
        "foo": f"{state['foo']} -> {state['bar']}",
        "trace": f"{state['trace']} -> child:update_shared_foo",
    }


def build_child_graph():
    builder = StateGraph(ChildState)
    builder.add_node("child_private_step", child_private_step)
    builder.add_node("child_shared_step", child_shared_step)
    builder.add_edge(START, "child_private_step")
    builder.add_edge("child_private_step", "child_shared_step")
    builder.add_edge("child_shared_step", END)
    return builder.compile()


def build_graph():
    child_graph = build_child_graph()

    builder = StateGraph(ParentState)
    builder.add_node("parent_prepare", parent_prepare)
    builder.add_node("child_graph", child_graph)
    builder.add_edge(START, "parent_prepare")
    builder.add_edge("parent_prepare", "child_graph")
    builder.add_edge("child_graph", END)
    return builder.compile()


def main() -> None:
    graph = build_graph()
    result = graph.invoke({"foo": "root", "trace": "start"})
    print(result)
    print("Parent output has shared keys only:", list(result.keys()))


if __name__ == "__main__":
    main()
