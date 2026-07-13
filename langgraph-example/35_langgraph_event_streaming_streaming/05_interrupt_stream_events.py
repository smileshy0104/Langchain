"""
案例 5：Interrupt resume with Event Streaming

目标：
- 在节点中使用 interrupt() 暂停 graph。
- 用 stream.interrupted 和 stream.interrupts 获取 HITL payload。
- 用 Command(resume=...) 恢复运行。

对应文档概念：
- Interrupt Resume
- stream.interrupts
- stream.interrupted
- Command(resume=...)
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
            "question": "approve action?",
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
    config = {"configurable": {"thread_id": "approval-stream"}}

    stream = graph.stream_events(
        {"action": "send email", "approved": None, "result": ""},
        config=config,
        version="v3",
    )

    _ = stream.output

    if stream.interrupted:
        print("interrupt payloads:", stream.interrupts)

    resumed = graph.stream_events(
        Command(resume=True),
        config=config,
        version="v3",
    )
    print("final output:", resumed.output)


if __name__ == "__main__":
    main()
