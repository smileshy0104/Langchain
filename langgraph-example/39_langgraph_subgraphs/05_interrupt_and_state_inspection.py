"""
案例 5：Subgraph interrupt and state inspection

目标：
- 子图内部调用 interrupt()。
- 顶层 graph 暂停。
- 使用 get_state(config, subgraphs=True) 查看暂停的子图 state。
- 顶层通过 Command(resume=...) 恢复执行。

对应文档概念：
- Interrupt
- View Subgraph State
- Per-invocation State Inspection
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    answer: str


def ask_inside_subgraph(state: State) -> dict:
    value = interrupt("Provide value for subgraph:")
    return {"answer": f"subgraph received: {value}"}


def build_subgraph():
    builder = StateGraph(State)
    builder.add_node("ask_inside_subgraph", ask_inside_subgraph)
    builder.add_edge(START, "ask_inside_subgraph")
    builder.add_edge("ask_inside_subgraph", END)
    return builder.compile()


def build_graph():
    subgraph = build_subgraph()
    builder = StateGraph(State)
    builder.add_node("subgraph_node", subgraph)
    builder.add_edge(START, "subgraph_node")
    builder.add_edge("subgraph_node", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "subgraph-interrupt"}}

    paused = graph.invoke({"answer": ""}, config)
    print("paused:", paused)

    parent_state = graph.get_state(config, subgraphs=True)
    subgraph_state = parent_state.tasks[0].state
    print("subgraph next:", subgraph_state.next)
    print("subgraph values:", subgraph_state.values)

    resumed = graph.invoke(Command(resume="approved"), config)
    print("resumed:", resumed)


if __name__ == "__main__":
    main()
