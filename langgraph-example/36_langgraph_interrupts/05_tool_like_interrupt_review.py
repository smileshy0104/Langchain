"""
案例 5：工具调用审核

目标：
- 在 tool-like 函数内部调用 interrupt()。
- 人工可以 approve、edit 或 cancel。
- 真正的副作用放在 interrupt() 之后。

对应文档概念：
- Interrupts in tools
- Interrupt 前的副作用必须幂等
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class EmailState(TypedDict):
    # 邮件收件人、主题、正文来自 graph state。
    to: str
    subject: str
    body: str
    # result 保存工具调用的最终结果。
    result: str


# 模拟一个 tool-like 函数：在真正发送邮件前先 interrupt 让人审核。
def send_email_tool_like(to: str, subject: str, body: str) -> str:
    review = interrupt(
        {
            "action": "send_email",
            "to": to,
            "subject": subject,
            "body": body,
            "message": "Approve sending this email?",
        }
    )

    # 人工可以通过 resume payload 决定 approve/edit/cancel。
    if review.get("action") != "approve":
        return "Email cancelled by user"

    # 允许人工在批准时修改邮件字段；未提供则使用原值。
    final_to = review.get("to", to)
    final_subject = review.get("subject", subject)
    final_body = review.get("body", body)

    # 真正发送邮件的副作用应放在 interrupt() 之后。
    # 重要原则：interrupt() 之前的代码在恢复时可能重放，因此副作用必须幂等或延后执行。
    return f"Email sent to {final_to}: {final_subject} / {final_body}"


# graph 节点包装 tool-like 函数，把工具返回值写回 state。
def email_node(state: EmailState) -> dict:
    result = send_email_tool_like(
        state["to"],
        state["subject"],
        state["body"],
    )
    return {"result": result}


# 构建单节点 graph，重点演示 tool 内部 interrupt。
def build_graph():
    builder = StateGraph(EmailState)
    builder.add_node("email", email_node)
    builder.add_edge(START, "email")
    builder.add_edge("email", END)
    return builder.compile(checkpointer=InMemorySaver())


# 主函数演示审核并修改工具调用参数。
def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "email-review"}}

    stream = graph.stream_events(
        {
            "to": "alice@example.com",
            "subject": "Meeting",
            "body": "Let's meet tomorrow.",
            "result": "",
        },
        config=config,
        version="v3",
    )
    _ = stream.output
    print("tool review interrupt:", stream.interrupts)

    # resume payload 是一个 dict：批准发送，同时覆盖 subject。
    resumed = graph.stream_events(
        Command(
            resume={
                "action": "approve",
                "subject": "Updated meeting subject",
            }
        ),
        config=config,
        version="v3",
    )
    print("final:", resumed.output)


if __name__ == "__main__":
    main()
