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
    # 最内层 grandchild 图只认识 grandchild_text。
    grandchild_text: str


class ChildState(TypedDict):
    # 中间层 child 图只认识 child_text。
    child_text: str


class ParentState(TypedDict):
    # 最外层 parent 图只认识 parent_text。
    parent_text: str


def grandchild_node(state: GrandchildState) -> dict:
    """grandchild 图的业务节点：包装自己的文本。"""

    return {"grandchild_text": f"grandchild({state['grandchild_text']})"}


def build_grandchild_graph():
    """构建最内层子图。"""

    builder = StateGraph(GrandchildState)
    builder.add_node("grandchild_node", grandchild_node)
    builder.add_edge(START, "grandchild_node")
    builder.add_edge("grandchild_node", END)
    return builder.compile()


# 编译后的 grandchild 图会被 child 节点调用。
GRANDCHILD = build_grandchild_graph()


def child_calls_grandchild(state: ChildState) -> dict:
    """child 图中的 wrapper 节点：把 ChildState 映射给 GrandchildState。"""

    output = GRANDCHILD.invoke(
        {"grandchild_text": f"mapped from child: {state['child_text']}"}
    )
    # 再把 grandchild 输出映射回 child_text。
    return {"child_text": f"child({output['grandchild_text']})"}


def build_child_graph():
    """构建中间层 child 图，内部会调用 grandchild 图。"""

    builder = StateGraph(ChildState)
    builder.add_node("child_calls_grandchild", child_calls_grandchild)
    builder.add_edge(START, "child_calls_grandchild")
    builder.add_edge("child_calls_grandchild", END)
    return builder.compile()


# 编译后的 child 图会被 parent 节点调用。
CHILD = build_child_graph()


def parent_calls_child(state: ParentState) -> dict:
    """parent 图中的 wrapper 节点：把 ParentState 映射给 ChildState。"""

    output = CHILD.invoke({"child_text": f"mapped from parent: {state['parent_text']}"})
    # 子图私有字段不会自动回到父图，必须显式写入 parent_text。
    return {"parent_text": f"parent({output['child_text']})"}


def build_graph():
    """构建最外层 parent 图。"""

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
