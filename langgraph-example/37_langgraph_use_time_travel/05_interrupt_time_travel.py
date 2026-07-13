"""
案例 5：Time travel 与 interrupts

目标：
- 先完成一次 interrupt。
- 从 interrupt 节点前 replay，interrupt 会再次触发。
- 从 interrupt 节点前 fork，interrupt 也会再次触发。

对应文档概念：
- Time Travel 与 Interrupts
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class GreetingState(TypedDict):
    value: Annotated[list[str], operator.add]


def ask_human(state: GreetingState) -> dict:
    answer = interrupt("What is your name?")
    return {"value": [f"Hello, {answer}!"]}


def final_step(state: GreetingState) -> dict:
    return {"value": ["Done"]}


def build_graph():
    builder = StateGraph(GreetingState)
    builder.add_node("ask_human", ask_human)
    builder.add_node("final_step", final_step)
    builder.add_edge(START, "ask_human")
    builder.add_edge("ask_human", "final_step")
    builder.add_edge("final_step", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "interrupt-time-travel"}}

    print("First run pauses:")
    graph.invoke({"value": []}, config)
    completed = graph.invoke(Command(resume="Alice"), config)
    print("completed:", completed)

    history = list(graph.get_state_history(config))
    before_ask = [s for s in history if s.next == ("ask_human",)][-1]

    print("\nReplay from before ask_human pauses again:")
    replay_result = graph.invoke(None, before_ask.config)
    print(replay_result)

    print("\nFork from before ask_human also pauses:")
    fork_config = graph.update_state(before_ask.config, {"value": ["forked"]})
    fork_result = graph.invoke(None, fork_config)
    print(fork_result)

    resumed_fork = graph.invoke(Command(resume="Bob"), fork_config)
    print("resumed fork:", resumed_fork)


if __name__ == "__main__":
    main()
