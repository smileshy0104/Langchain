"""
案例 5：interrupt 与 resume

目标：
- 在节点中调用 interrupt() 暂停 graph。
- checkpointer 保存暂停状态。
- 用 Command(resume=...) 恢复同一个 thread。

对应文档概念：
- Human-in-the-loop
- interrupt
- thread_id
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
            "question": "是否批准这个动作？",
            "action": state["action"],
        }
    )
    return {"approved": approved}


def execute_node(state: ApprovalState) -> dict:
    if state["approved"]:
        result = f"已执行：{state['action']}"
    else:
        result = f"已取消：{state['action']}"
    return {"result": result}


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
    config = {"configurable": {"thread_id": "approval-thread"}}

    print("第一次调用：graph 会在 approval_node 中暂停")
    interrupted_result = graph.invoke(
        {
            "action": "transfer $500",
            "approved": None,
            "result": "",
        },
        config,
    )
    print(interrupted_result)

    snapshot = graph.get_state(config)
    print("\n暂停后的 StateSnapshot:")
    print("values:", snapshot.values)
    print("next:", snapshot.next)
    print("interrupts:", snapshot.interrupts)

    print("\n恢复：批准该动作")
    resumed = graph.invoke(Command(resume=True), config)
    print(resumed)


if __name__ == "__main__":
    main()
