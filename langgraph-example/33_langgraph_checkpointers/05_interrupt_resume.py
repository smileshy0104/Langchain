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
    # interrupt 会把 payload 返回给调用方，并暂停当前 graph；
    # 当前执行位置和 state 会依赖 checkpointer 保存下来。
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
    # 没有 checkpointer 时，interrupt 暂停后就无法可靠恢复同一个 thread。
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    # resume 必须使用同一个 thread_id，LangGraph 才能找到之前暂停的 checkpoint。
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
    # next 指向暂停后仍待完成的节点；interrupts 中保存等待外部输入的信息。
    print("next:", snapshot.next)
    print("interrupts:", snapshot.interrupts)

    print("\n恢复：批准该动作")
    # Command(resume=True) 会把 True 作为 interrupt() 的返回值送回 approval_node。
    resumed = graph.invoke(Command(resume=True), config)
    print(resumed)


if __name__ == "__main__":
    main()
