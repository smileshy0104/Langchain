"""
案例 1：Call a subgraph inside a node

目标：
- 父图和子图 state schema 不同。
- 父节点 wrapper 手动把 parent state 转成 subgraph input。
- 子图 output 再被 wrapper 转回 parent state。

对应文档概念：
- Call a Subgraph Inside a Node
- 不同 State Schema
- Parent -> Subgraph / Subgraph -> Parent 映射
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class ParentState(TypedDict):
    # 父图只关心 foo，以及父图层面的执行日志。
    foo: str
    # 使用 reducer 累加父图和子图返回的日志，便于观察调用顺序。
    log: Annotated[list[str], operator.add]


class SubgraphState(TypedDict):
    # 子图自己的主输入/输出字段；父图中没有 bar。
    bar: str
    # 子图内部临时字段；父图中也没有 baz。
    baz: str
    # 子图内部日志，同样用 reducer 累加。
    log: Annotated[list[str], operator.add]


def subgraph_step_1(state: SubgraphState) -> dict:
    """子图第一步：写入子图私有中间值 baz。"""

    return {"baz": " + baz", "log": ["subgraph:step_1"]}


def subgraph_step_2(state: SubgraphState) -> dict:
    """子图第二步：把 bar 和 baz 合并成新的 bar。"""

    return {
        "bar": f"{state['bar']}{state['baz']}",
        "log": ["subgraph:step_2"],
    }


def build_subgraph():
    """构建独立子图：START -> subgraph_step_1 -> subgraph_step_2 -> END。"""

    builder = StateGraph(SubgraphState)
    builder.add_node("subgraph_step_1", subgraph_step_1)
    builder.add_node("subgraph_step_2", subgraph_step_2)
    builder.add_edge(START, "subgraph_step_1")
    builder.add_edge("subgraph_step_1", "subgraph_step_2")
    builder.add_edge("subgraph_step_2", END)
    return builder.compile()


# 示例中子图可复用，因此提前编译成全局对象。
SUBGRAPH = build_subgraph()


def call_subgraph(state: ParentState) -> dict:
    """父图 wrapper 节点：负责父状态和子图状态之间的显式映射。"""

    # 父图和子图 schema 不同，不能直接把 ParentState 传给子图，需要手动转换。
    subgraph_input = {
        "bar": f"mapped from parent foo: {state['foo']}",
        "baz": "",
        "log": [],
    }

    # 在普通节点内部调用子图，子图对父图来说只是一次函数调用。
    subgraph_output = SUBGRAPH.invoke(subgraph_input)

    # 子图输出也不会自动写回父图，需要 wrapper 明确决定哪些字段回到 ParentState。
    return {
        "foo": subgraph_output["bar"],
        "log": ["parent:called_subgraph", *subgraph_output["log"]],
    }


def build_graph():
    """构建父图，父图只有一个负责调用子图的 wrapper 节点。"""

    builder = StateGraph(ParentState)
    builder.add_node("call_subgraph", call_subgraph)
    builder.add_edge(START, "call_subgraph")
    builder.add_edge("call_subgraph", END)
    return builder.compile()


def main() -> None:
    graph = build_graph()
    result = graph.invoke({"foo": "hello", "log": []})
    print(result)


if __name__ == "__main__":
    main()
