"""
案例 2：invoke API 中处理 interrupt

目标：
- 不使用 Event Streaming，直接 graph.invoke()。
- interrupt payload 会出现在 result["__interrupt__"]。
- 用 Command(resume=...) 恢复。

对应文档概念：
- invoke API 中处理 Interrupt
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class FormState(TypedDict):
    name: str | None
    greeting: str


def ask_name(state: FormState) -> dict:
    name = interrupt("What is your name?")
    return {"name": name}


def greet(state: FormState) -> dict:
    return {"greeting": f"Hello, {state['name']}!"}


def build_graph():
    builder = StateGraph(FormState)
    builder.add_node("ask_name", ask_name)
    builder.add_node("greet", greet)
    builder.add_edge(START, "ask_name")
    builder.add_edge("ask_name", "greet")
    builder.add_edge("greet", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "invoke-interrupt"}}

    result = graph.invoke({"name": None, "greeting": ""}, config)
    print("first invoke result:", result)

    if "__interrupt__" in result:
        print("interrupt payload:", result["__interrupt__"])

    resumed = graph.invoke(Command(resume="Alice"), config)
    print("resumed result:", resumed)


if __name__ == "__main__":
    main()
