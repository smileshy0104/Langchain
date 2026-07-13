"""
案例 6：从多个 interrupts 中间 fork

目标：
- ask_name -> ask_age -> final。
- 完成两个问题后，从 ask_name 和 ask_age 中间的 checkpoint fork。
- 保留名字回答，只重新询问年龄。

对应文档概念：
- Multiple Interrupts
- between checkpoint
- s.next == ("ask_age",)
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class FormState(TypedDict):
    value: Annotated[list[str], operator.add]


def ask_name(state: FormState) -> dict:
    name = interrupt("What is your name?")
    return {"value": [f"name:{name}"]}


def ask_age(state: FormState) -> dict:
    age = interrupt("How old are you?")
    return {"value": [f"age:{age}"]}


def final(state: FormState) -> dict:
    return {"value": ["final"]}


def build_graph():
    builder = StateGraph(FormState)
    builder.add_node("ask_name", ask_name)
    builder.add_node("ask_age", ask_age)
    builder.add_node("final", final)
    builder.add_edge(START, "ask_name")
    builder.add_edge("ask_name", "ask_age")
    builder.add_edge("ask_age", "final")
    builder.add_edge("final", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "multi-interrupt-time-travel"}}

    graph.invoke({"value": []}, config)
    graph.invoke(Command(resume="Alice"), config)
    completed = graph.invoke(Command(resume=30), config)
    print("completed:", completed)

    history = list(graph.get_state_history(config))
    matches = [s for s in history if s.next == ("ask_age",)]
    between = matches[0]

    print("\nbetween means: ask_name completed, ask_age not started")
    print("between values:", between.values)

    fork_config = graph.update_state(between.config, {"value": ["modified"]})
    fork_result = graph.invoke(None, fork_config)
    print("fork pauses at ask_age again:", fork_result)

    resumed = graph.invoke(Command(resume=18), fork_config)
    print("resumed fork:", resumed)


if __name__ == "__main__":
    main()
