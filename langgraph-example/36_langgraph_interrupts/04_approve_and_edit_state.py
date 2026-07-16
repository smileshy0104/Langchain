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
    # draft 是待审批和编辑的草稿内容。
    draft: str
    # approved 记录人工是否允许进入编辑流程。
    approved: bool | None
    # status 标识当前流程状态。
    status: str


# 审批节点：先暂停等待人工 approve/reject，再根据结果动态路由。
def approval_node(state: ReviewState) -> Command[Literal["review", "cancel"]]:
    approved = interrupt(
        {
            "type": "approval",
            "question": "Approve editing this draft?",
            "draft": state["draft"],
        }
    )
    # Command 可同时更新 state 并指定下一步 goto。
    return Command(
        update={"approved": approved},
        goto="review" if approved else "cancel",
    )


# 编辑节点：再次暂停，让人工返回编辑后的草稿内容。
def review_node(state: ReviewState) -> dict:
    edited = interrupt(
        {
            "type": "edit",
            "instruction": "Edit this draft",
            "current_value": state["draft"],
        }
    )
    return {"draft": edited, "status": "edited"}


# 取消节点：审批未通过时进入该分支。
def cancel_node(state: ReviewState) -> dict:
    return {"status": "cancelled"}


# 构建审批 -> 编辑/取消 的分支 graph。
def build_graph():
    builder = StateGraph(ReviewState)
    builder.add_node("approval", approval_node)
    builder.add_node("review", review_node)
    builder.add_node("cancel", cancel_node)
    builder.add_edge(START, "approval")
    builder.add_edge("review", END)
    builder.add_edge("cancel", END)
    return builder.compile(checkpointer=InMemorySaver())


# 主函数演示同一 thread 中连续处理两个 interrupt。
def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "approve-edit"}}

    # 第一次运行停在 approval_node。
    first = graph.stream_events(
        {"draft": "Initial draft", "approved": None, "status": "pending"},
        config=config,
        version="v3",
    )
    _ = first.output
    print("approval interrupt:", first.interrupts)

    # 恢复 approval_node，并传入 True 让流程进入 review_node；review_node 会再次 interrupt。
    second = graph.stream_events(Command(resume=True), config=config, version="v3")
    _ = second.output
    print("edit interrupt:", second.interrupts)

    # 恢复 review_node，并传入编辑后的文本作为新的 draft。
    final = graph.stream_events(
        Command(resume="Improved draft after review"),
        config=config,
        version="v3",
    )
    print("final:", final.output)


if __name__ == "__main__":
    main()
