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
    to: str
    subject: str
    body: str
    result: str


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

    if review.get("action") != "approve":
        return "Email cancelled by user"

    final_to = review.get("to", to)
    final_subject = review.get("subject", subject)
    final_body = review.get("body", body)

    # 真正发送邮件的副作用应放在 interrupt() 之后。
    return f"Email sent to {final_to}: {final_subject} / {final_body}"


def email_node(state: EmailState) -> dict:
    result = send_email_tool_like(
        state["to"],
        state["subject"],
        state["body"],
    )
    return {"result": result}


def build_graph():
    builder = StateGraph(EmailState)
    builder.add_node("email", email_node)
    builder.add_edge(START, "email")
    builder.add_edge("email", END)
    return builder.compile(checkpointer=InMemorySaver())


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
