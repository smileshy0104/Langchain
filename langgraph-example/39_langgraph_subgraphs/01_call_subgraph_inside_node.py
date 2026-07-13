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
    foo: str
    log: Annotated[list[str], operator.add]


class SubgraphState(TypedDict):
    bar: str
    baz: str
    log: Annotated[list[str], operator.add]


def subgraph_step_1(state: SubgraphState) -> dict:
    return {"baz": " + baz", "log": ["subgraph:step_1"]}


def subgraph_step_2(state: SubgraphState) -> dict:
    return {
        "bar": f"{state['bar']}{state['baz']}",
        "log": ["subgraph:step_2"],
    }


def build_subgraph():
    builder = StateGraph(SubgraphState)
    builder.add_node("subgraph_step_1", subgraph_step_1)
    builder.add_node("subgraph_step_2", subgraph_step_2)
    builder.add_edge(START, "subgraph_step_1")
    builder.add_edge("subgraph_step_1", "subgraph_step_2")
    builder.add_edge("subgraph_step_2", END)
    return builder.compile()


SUBGRAPH = build_subgraph()


def call_subgraph(state: ParentState) -> dict:
    subgraph_input = {
        "bar": f"mapped from parent foo: {state['foo']}",
        "baz": "",
        "log": [],
    }
    subgraph_output = SUBGRAPH.invoke(subgraph_input)
    return {
        "foo": subgraph_output["bar"],
        "log": ["parent:called_subgraph", *subgraph_output["log"]],
    }


def build_graph():
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
