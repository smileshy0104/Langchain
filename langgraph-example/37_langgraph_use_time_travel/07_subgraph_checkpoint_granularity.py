"""
案例 7：Subgraph checkpoint 粒度

目标：
- 展示默认 subgraph 继承父图 checkpointer 时，父图只看到 subgraph_node 级别。
- 展示 subgraph.compile(checkpointer=True) 时，可以通过 get_state(..., subgraphs=True)
  取得子图内部 checkpoint config。

对应文档概念：
- Time Travel 与 Subgraphs
- Inherited Checkpointer
- Subgraph Own Checkpointer
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    value: Annotated[list[str], operator.add]


def step_a(state: State) -> dict:
    answer = interrupt("step_a input?")
    return {"value": [f"a:{answer}"]}


def step_b(state: State) -> dict:
    answer = interrupt("step_b input?")
    return {"value": [f"b:{answer}"]}


def build_parent_graph(*, subgraph_own_checkpointer: bool):
    sub_builder = StateGraph(State)
    sub_builder.add_node("step_a", step_a)
    sub_builder.add_node("step_b", step_b)
    sub_builder.add_edge(START, "step_a")
    sub_builder.add_edge("step_a", "step_b")
    sub_builder.add_edge("step_b", END)

    if subgraph_own_checkpointer:
        subgraph = sub_builder.compile(checkpointer=True)
    else:
        subgraph = sub_builder.compile()

    parent_builder = StateGraph(State)
    parent_builder.add_node("subgraph_node", subgraph)
    parent_builder.add_edge(START, "subgraph_node")
    parent_builder.add_edge("subgraph_node", END)
    return parent_builder.compile(checkpointer=InMemorySaver())


def run_default_subgraph() -> None:
    print("Default subgraph: inherited parent checkpointer")
    graph = build_parent_graph(subgraph_own_checkpointer=False)
    config = {"configurable": {"thread_id": "subgraph-default"}}

    graph.invoke({"value": []}, config)
    graph.invoke(Command(resume="Alice"), config)
    graph.invoke(Command(resume="30"), config)

    history = list(graph.get_state_history(config))
    print("parent-level next values:")
    for snapshot in history:
        print("-", snapshot.next)
    print("Default mode cannot time travel between step_a and step_b from parent history.")


def run_own_checkpoint_subgraph() -> None:
    print("\nSubgraph checkpointer=True: own checkpoint history")
    graph = build_parent_graph(subgraph_own_checkpointer=True)
    config = {"configurable": {"thread_id": "subgraph-own"}}

    graph.invoke({"value": []}, config)
    graph.invoke(Command(resume="Alice"), config)

    parent_state = graph.get_state(config, subgraphs=True)
    sub_config = parent_state.tasks[0].state.config
    print("subgraph checkpoint config:", sub_config)

    fork_config = graph.update_state(sub_config, {"value": ["forked-inside-subgraph"]})
    fork_result = graph.invoke(None, fork_config)
    print("forked from subgraph checkpoint:", fork_result)


def main() -> None:
    run_default_subgraph()
    run_own_checkpoint_subgraph()


if __name__ == "__main__":
    main()
