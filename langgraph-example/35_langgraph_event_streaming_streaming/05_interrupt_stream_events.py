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

import warnings
from typing import TypedDict

from langchain_core._api.beta_decorator import LangChainBetaWarning

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


# 当前 LangGraph 版本中 v3 streaming protocol 仍标记为 beta；示例隐藏该警告，避免干扰教学输出。
warnings.filterwarnings("ignore", category=LangChainBetaWarning)


class ApprovalState(TypedDict):
    # action 是等待人工审批的动作。
    action: str
    # approved 在中断前为 None，恢复时由 Command(resume=...) 的值填入。
    approved: bool | None
    # result 保存最终执行或取消的结果。
    result: str


# 审批节点：调用 interrupt 暂停 graph，并把 HITL payload 暴露给调用方。
def approval_node(state: ApprovalState) -> dict:
    approved = interrupt(
        {
            "question": "approve action?",
            "action": state["action"],
        }
    )
    # graph resume 后，interrupt(...) 的返回值就是 Command(resume=...) 传入的数据。
    return {"approved": approved}


# 执行节点：根据人工审批结果决定是否执行动作。
def execute_node(state: ApprovalState) -> dict:
    if state["approved"]:
        return {"result": f"executed: {state['action']}"}
    return {"result": f"cancelled: {state['action']}"}


# 构建带 checkpointer 的 graph；interrupt/resume 必须依赖 checkpoint 保存暂停点。
def build_graph():
    builder = StateGraph(ApprovalState)
    builder.add_node("approval", approval_node)
    builder.add_node("execute", execute_node)
    builder.add_edge(START, "approval")
    builder.add_edge("approval", "execute")
    builder.add_edge("execute", END)
    return builder.compile(checkpointer=InMemorySaver())


# 主函数演示：通过 Event Streaming 获取 interrupt 信息，再恢复执行。
def main() -> None:
    graph = build_graph()

    # thread_id 用于在恢复时找到同一个暂停的 graph run。
    config = {"configurable": {"thread_id": "approval-stream"}}

    stream = graph.stream_events(
        {"action": "send email", "approved": None, "result": ""},
        config=config,
        version="v3",
    )

    # 消费 output 会驱动 graph 执行到中断点；中断时 output 通常不是最终业务结果。
    _ = stream.output

    # stream.interrupted 是 typed projection 提供的布尔标志。
    # stream.interrupts 包含 interrupt(...) 中传出的 payload，供 UI 展示给用户。
    if stream.interrupted:
        print("interrupt payloads:", stream.interrupts)

    # 使用同一个 config + Command(resume=True) 从 checkpoint 中恢复运行。
    resumed = graph.stream_events(
        Command(resume=True),
        config=config,
        version="v3",
    )
    print("final output:", resumed.output)


if __name__ == "__main__":
    main()
