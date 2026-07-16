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
    # action 是需要人工审批的业务动作。
    action: str
    # approved 在首次运行时为 None；resume 后会被 interrupt() 的返回值填充。
    approved: bool | None
    # result 保存最终执行或取消的结果。
    result: str


# 审批节点：调用 interrupt() 后 graph 会暂停，并把 payload 返回给调用方。
def approval_node(state: ApprovalState) -> dict:
    approved = interrupt(
        {
            "type": "approval",
            "question": "Approve this action?",
            "action": state["action"],
        }
    )
    # 恢复时，Command(resume=...) 中的值会作为 interrupt() 的返回值。
    return {"approved": approved}


# 执行节点：只有审批通过才执行 action。
def execute_node(state: ApprovalState) -> dict:
    if state["approved"]:
        return {"result": f"executed: {state['action']}"}
    return {"result": f"cancelled: {state['action']}"}


# 构建 graph；动态 interrupt/resume 必须配合 checkpointer 保存暂停点。
def build_graph():
    builder = StateGraph(ApprovalState)
    builder.add_node("approval", approval_node)
    builder.add_node("execute", execute_node)
    builder.add_edge(START, "approval")
    builder.add_edge("approval", "execute")
    builder.add_edge("execute", END)
    return builder.compile(checkpointer=InMemorySaver())


# 主函数演示 Event Streaming 下的暂停和恢复。
def main() -> None:
    graph = build_graph()

    # thread_id 用于定位暂停时写入的 checkpoint；恢复时必须复用同一个 config。
    config = {"configurable": {"thread_id": "approval-basic"}}

    stream = graph.stream_events(
        {"action": "transfer $500", "approved": None, "result": ""},
        config=config,
        version="v3",
    )

    # 读取 output 会驱动 graph 执行，直到完成或遇到 interrupt。
    _ = stream.output

    # interrupted/interupts 是 Event Streaming 的 typed projection，便于 UI 展示暂停信息。
    if stream.interrupted:
        print("paused with interrupts:")
        for item in stream.interrupts:
            print("-", item)

    # 传入 True 表示人工批准；该值会回到 approval_node 中 interrupt() 的返回值。
    resumed = graph.stream_events(
        Command(resume=True),
        config=config,
        version="v3",
    )
    print("final:", resumed.output)


if __name__ == "__main__":
    main()
