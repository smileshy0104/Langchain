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
    # 父图公开字段：子图如果也声明同名字段，就能读写它。
    foo: str
    # 用字符串记录父/子图执行轨迹。
    trace: str


class ChildState(TypedDict):
    # foo/trace 与父图同名，是父子图共享字段。
    foo: str
    # bar 是子图私有字段，不在 ParentState 中，因此不会出现在最终父图输出里。
    bar: str
    trace: str


def parent_prepare(state: ParentState) -> dict:
    """父图预处理节点：先修改共享字段 foo/trace。"""

    return {
        "foo": f"parent({state['foo']})",
        "trace": f"{state['trace']} -> parent:prepare",
    }


def child_private_step(state: ChildState) -> dict:
    """子图内部节点：写入子图私有 key bar。"""

    return {
        "bar": "child-private",
        "trace": f"{state['trace']} -> child:set_bar",
    }


def child_shared_step(state: ChildState) -> dict:
    """子图内部节点：用私有 bar 更新共享 key foo。"""

    return {
        "foo": f"{state['foo']} -> {state['bar']}",
        "trace": f"{state['trace']} -> child:update_shared_foo",
    }


def build_child_graph():
    """构建子图；子图可以有比父图更多的状态字段。"""

    builder = StateGraph(ChildState)
    builder.add_node("child_private_step", child_private_step)
    builder.add_node("child_shared_step", child_shared_step)
    builder.add_edge(START, "child_private_step")
    builder.add_edge("child_private_step", "child_shared_step")
    builder.add_edge("child_shared_step", END)
    return builder.compile()


def build_graph():
    """把已编译的 child_graph 直接作为父图节点加入。"""

    child_graph = build_child_graph()

    builder = StateGraph(ParentState)
    builder.add_node("parent_prepare", parent_prepare)
    # 当父子图有共享 key 时，可以直接 add_node("child_graph", child_graph)，无需 wrapper。
    builder.add_node("child_graph", child_graph)
    builder.add_edge(START, "parent_prepare")
    builder.add_edge("parent_prepare", "child_graph")
    builder.add_edge("child_graph", END)
    return builder.compile()


def main() -> None:
    graph = build_graph()
    result = graph.invoke({"foo": "root", "trace": "start"})
    print(result)
    # 可以看到 bar 没有出现在父图输出中，因为 ParentState 没有声明这个字段。
    print("Parent output has shared keys only:", list(result.keys()))


if __name__ == "__main__":
    main()
