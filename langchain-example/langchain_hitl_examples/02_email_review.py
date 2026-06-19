"""
案例 2：邮件发送审核系统。

模型可调用发送邮件和安排会议工具；真正发送前由
HumanInTheLoopMiddleware 暂停，人工可批准、修改或拒绝。
"""

from __future__ import annotations

import argparse
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import ToolCall
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from hitl_utils import build_decisions, get_interrupts, get_last_message_text, print_interrupt
from hitl_utils import resume_with_decisions
from model_config import create_configured_model


@tool
def send_email(
    to: list[str],
    subject: str,
    body: str,
    attachments: list[str] | None = None,
) -> str:
    """Send an email after human approval."""

    attachments = attachments or []
    allowed_domains = ["company.com", "trusted-partner.com"]
    for email in to:
        domain = email.split("@")[-1]
        if domain not in allowed_domains:
            return f"错误：不允许发送到 {domain}。"

    attachment_text = f"，附件：{', '.join(attachments)}" if attachments else ""
    return f"已发送邮件至 {', '.join(to)}，主题：{subject}{attachment_text}"


@tool
def schedule_meeting(
    attendees: list[str],
    title: str,
    start_time: str,
    duration_minutes: int,
) -> str:
    """Schedule a meeting after human approval."""

    return (
        f"已安排会议：{title}，开始时间：{start_time}，"
        f"时长：{duration_minutes} 分钟，参与者：{len(attendees)} 人。"
    )


def describe_email_review(tool_call: ToolCall, state: Any, runtime: Any) -> str:
    args = tool_call["args"]
    body_preview = str(args.get("body", ""))[:100]
    return (
        "邮件审核:\n"
        f"收件人: {', '.join(args['to'])}\n"
        f"主题: {args['subject']}\n"
        f"正文预览: {body_preview}"
    )


def create_email_agent() -> Any:
    return create_agent(
        model=create_configured_model(),
        tools=[send_email, schedule_meeting],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_email": {
                        "allowed_decisions": ["approve", "edit", "reject"],
                        "description": describe_email_review,
                    },
                    "schedule_meeting": {
                        "allowed_decisions": ["approve", "edit"],
                        "description": "会议安排审核：请确认参会人、主题和时间。",
                    },
                }
            )
        ],
        checkpointer=InMemorySaver(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HITL 案例 2：邮件发送审核系统")
    parser.add_argument(
        "--request",
        default=(
            "请给 alice@trusted-partner.com 发送一封邮件，主题为项目同步，"
            "正文说明明天上午 10 点开同步会。"
        ),
    )
    parser.add_argument(
        "--decision",
        choices=["approve", "edit", "reject"],
        default="approve",
        help="人工审批决策。schedule_meeting 不允许 reject。",
    )
    parser.add_argument("--edited-to", help="decision=edit 时替换收件人，多个邮箱用逗号分隔。")
    parser.add_argument("--edited-subject", help="decision=edit 时替换邮件主题。")
    parser.add_argument("--thread-id", default="hitl-email-session-001")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = create_email_agent()
    config = {"configurable": {"thread_id": args.thread_id}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": args.request}]},
        config=config,
    )

    interrupts = get_interrupts(result)
    while interrupts:
        interrupt = interrupts[0]
        print_interrupt(interrupt)

        edited_args: dict[str, Any] = {}
        if args.edited_to:
            edited_args["to"] = [email.strip() for email in args.edited_to.split(",") if email.strip()]
        if args.edited_subject:
            edited_args["subject"] = args.edited_subject

        decisions = build_decisions(
            interrupt,
            args.decision,
            edited_args=edited_args,
            reject_message="人工审核拒绝发送该邮件。",
        )
        print("\n本次提交的人工决策:")
        print(decisions)

        result = resume_with_decisions(agent, config, decisions)
        interrupts = get_interrupts(result)

    print("\n最终回复:")
    print(get_last_message_text(result))


if __name__ == "__main__":
    main()
