"""
案例 4：审批与编辑 state

目标：
- 先让人工审批是否继续。
- 再让人工编辑草稿内容。
- resume payload 分别成为两个 interrupt() 的返回值。

对应文档概念：
- Approve or Reject
- Review and Edit State
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ReviewState(TypedDict):
    draft: str
    approved: bool | None
    status: str


def approval_node(state: ReviewState) -> Command[Literal["review", "cancel"]]:
    approved = interrupt(
        {
            "type": "approval",
            "question": "Approve editing this draft?",
            "draft": state["draft"],
        }
    )
    return Command(
        update={"approved": approved},
        goto="review" if approved else "cancel",
    )


def review_node(state: ReviewState) -> dict:
    edited = interrupt(
        {
            "type": "edit",
            "instruction": "Edit this draft",
            "current_value": state["draft"],
        }
    )
    return {"draft": edited, "status": "edited"}


def cancel_node(state: ReviewState) -> dict:
    return {"status": "cancelled"}


def build_graph():
    builder = StateGraph(ReviewState)
    builder.add_node("approval", approval_node)
    builder.add_node("review", review_node)
    builder.add_node("cancel", cancel_node)
    builder.add_edge(START, "approval")
    builder.add_edge("review", END)
    builder.add_edge("cancel", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "approve-edit"}}

    first = graph.stream_events(
        {"draft": "Initial draft", "approved": None, "status": "pending"},
        config=config,
        version="v3",
    )
    _ = first.output
    print("approval interrupt:", first.interrupts)

    second = graph.stream_events(Command(resume=True), config=config, version="v3")
    _ = second.output
    print("edit interrupt:", second.interrupts)

    final = graph.stream_events(
        Command(resume="Improved draft after review"),
        config=config,
        version="v3",
    )
    print("final:", final.output)


if __name__ == "__main__":
    main()
