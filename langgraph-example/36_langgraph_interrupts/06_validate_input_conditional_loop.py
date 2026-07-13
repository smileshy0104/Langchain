"""
案例 6：人工输入校验

目标：
- 每次节点调用只调用一次 interrupt()。
- 输入无效时，把新的 pending_question 写回 state。
- 用 conditional edge 回到同一个节点继续提问。

对应文档概念：
- Validating Human Input
- 避免 while True + interrupt()
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class FormState(TypedDict):
    age: int | None
    pending_question: str | None


def collect_age(state: FormState) -> dict:
    question = state.get("pending_question") or "What is your age?"
    answer = interrupt(question)

    if isinstance(answer, int) and answer > 0:
        return {"age": answer, "pending_question": None}

    return {
        "pending_question": (
            f"'{answer}' is not a valid age. Please enter a positive number."
        )
    }


def route(state: FormState) -> Literal["collect_age", "__end__"]:
    return END if state.get("age") is not None else "collect_age"


def build_graph():
    builder = StateGraph(FormState)
    builder.add_node("collect_age", collect_age)
    builder.add_edge(START, "collect_age")
    builder.add_conditional_edges("collect_age", route)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "validate-age"}}

    first = graph.stream_events(
        {"age": None, "pending_question": None},
        config=config,
        version="v3",
    )
    _ = first.output
    print("first prompt:", first.interrupts)

    retry = graph.stream_events(Command(resume="thirty"), config=config, version="v3")
    _ = retry.output
    print("retry prompt:", retry.interrupts)

    final = graph.stream_events(Command(resume=30), config=config, version="v3")
    print("final:", final.output)


if __name__ == "__main__":
    main()
