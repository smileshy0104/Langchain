"""
案例 1：基础动态 interrupt + resume

目标：
- 在节点里调用 interrupt() 暂停 graph。
- 用 Event Streaming 的 stream.interrupted / stream.interrupts 读取暂停信息。
- 用 Command(resume=...) 恢复。

对应文档概念：
- Pause using interrupt
- Resume interrupts
- Event Streaming 中处理 Interrupt
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ApprovalState(TypedDict):
    action: str
    approved: bool | None
    result: str


def approval_node(state: ApprovalState) -> dict:
    approved = interrupt(
        {
            "type": "approval",
            "question": "Approve this action?",
            "action": state["action"],
        }
    )
    return {"approved": approved}


def execute_node(state: ApprovalState) -> dict:
    if state["approved"]:
        return {"result": f"executed: {state['action']}"}
    return {"result": f"cancelled: {state['action']}"}


def build_graph():
    builder = StateGraph(ApprovalState)
    builder.add_node("approval", approval_node)
    builder.add_node("execute", execute_node)
    builder.add_edge(START, "approval")
    builder.add_edge("approval", "execute")
    builder.add_edge("execute", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "approval-basic"}}

    stream = graph.stream_events(
        {"action": "transfer $500", "approved": None, "result": ""},
        config=config,
        version="v3",
    )

    _ = stream.output

    if stream.interrupted:
        print("paused with interrupts:")
        for item in stream.interrupts:
            print("-", item)

    resumed = graph.stream_events(
        Command(resume=True),
        config=config,
        version="v3",
    )
    print("final:", resumed.output)


if __name__ == "__main__":
    main()
