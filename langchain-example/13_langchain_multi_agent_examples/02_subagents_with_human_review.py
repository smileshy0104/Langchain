"""
Subagents 添加人工审核：Supervisor 调用子 Agent 前先暂停审批。

人工审核只放在 Supervisor 层：
- schedule_event：审批是否允许调用日程子 Agent
- manage_email：审批是否允许调用邮件子 Agent
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
def create_calendar_event(
    title: str,
    start_time: str,
    end_time: str,
    attendees: list[str],
    location: str = "",
) -> str:
    """Create a calendar event. Use ISO datetime strings for start and end."""

    location_text = f"，地点：{location}" if location else ""
    return (
        f"事件已创建：{title}，从 {start_time} 到 {end_time}，"
        f"参会人：{', '.join(attendees)}{location_text}"
    )


@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,
    duration_minutes: int,
) -> list[str]:
    """Check available time slots for attendees on a given date."""

    return ["09:00", "14:00", "16:00"]


@tool
def send_email(
    to: list[str],
    subject: str,
    body: str,
    cc: list[str] | None = None,
) -> str:
    """Send an email through a mock email API."""

    cc = cc or []
    cc_text = f"，抄送：{', '.join(cc)}" if cc else ""
    return f"邮件已发送至 {', '.join(to)}，主题：{subject}{cc_text}"


CALENDAR_PROMPT = (
    "你是一个日程安排助手。"
    "将自然语言的日程请求解析为正确的 ISO 日期时间格式。"
    "需要时使用 get_available_time_slots 检查可用性。"
    "使用 create_calendar_event 安排事件。"
)

EMAIL_PROMPT = (
    "你是一个邮件助手。"
    "根据自然语言请求撰写简洁、专业的邮件。"
    "使用 send_email 发送邮件。"
)

SUPERVISOR_PROMPT = (
    "你是一个个人助理 Supervisor。"
    "你可以通过 schedule_event 调用日程子 Agent，通过 manage_email 调用邮件子 Agent。"
    "当请求涉及多个操作时，先安排日程，再发送邮件，最后汇总结果。"
)


def last_message_text(result: dict[str, Any]) -> str:
    return str(result["messages"][-1].content)


def describe_subagent_call(tool_call: ToolCall, state: Any, runtime: Any) -> str:
    args = tool_call["args"]
    return (
        "子 Agent 调用审核:\n"
        f"目标工具: {tool_call['name']}\n"
        f"转交请求: {args.get('request', '')}\n"
        "请确认该子任务可以继续执行。"
    )


def create_subagents() -> tuple[Any, Any]:
    calendar_agent = create_agent(
        model=create_configured_model(),
        tools=[create_calendar_event, get_available_time_slots],
        system_prompt=CALENDAR_PROMPT,
    )
    email_agent = create_agent(
        model=create_configured_model(),
        tools=[send_email],
        system_prompt=EMAIL_PROMPT,
    )
    return calendar_agent, email_agent


def create_supervisor_agent() -> Any:
    calendar_agent, email_agent = create_subagents()

    @tool
    def schedule_event(request: str) -> str:
        """Use the calendar subagent to schedule events from natural language."""

        result = calendar_agent.invoke(
            {"messages": [{"role": "user", "content": request}]}
        )
        return last_message_text(result)

    @tool
    def manage_email(request: str) -> str:
        """Use the email subagent to draft and send email from natural language."""

        result = email_agent.invoke(
            {"messages": [{"role": "user", "content": request}]}
        )
        return last_message_text(result)

    return create_agent(
        model=create_configured_model(),
        tools=[schedule_event, manage_email],
        system_prompt=SUPERVISOR_PROMPT,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "schedule_event": {
                        "allowed_decisions": ["approve", "edit", "reject"],
                        "description": describe_subagent_call,
                    },
                    "manage_email": {
                        "allowed_decisions": ["approve", "edit", "reject"],
                        "description": describe_subagent_call,
                    },
                }
            )
        ],
        checkpointer=InMemorySaver(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi Agent 案例：Subagents 添加人工审核")
    parser.add_argument(
        "--request",
        default=(
            "安排下周二下午2点和设计团队开1小时的会，"
            "然后给他们发一封提醒邮件，让他们审核新的原型设计。"
        ),
    )
    parser.add_argument(
        "--decision",
        choices=["approve", "edit", "reject"],
        default="approve",
        help="人工审核决策。",
    )
    parser.add_argument("--edited-request", help="decision=edit 时替换转交给子 Agent 的请求。")
    parser.add_argument("--thread-id", default="multi-agent-subagents-hitl-001")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    supervisor_agent = create_supervisor_agent()
    config = {"configurable": {"thread_id": args.thread_id}}

    result = supervisor_agent.invoke(
        {"messages": [{"role": "user", "content": args.request}]},
        config=config,
    )

    interrupts = get_interrupts(result)
    while interrupts:
        interrupt = interrupts[0]
        print_interrupt(interrupt)
        decisions = build_decisions(
            interrupt,
            args.decision,
            edited_request=args.edited_request,
            reject_message="人工审核拒绝调用该子 Agent。",
        )
        print("\n本次提交的人工决策:")
        print(decisions)

        result = resume_with_decisions(supervisor_agent, config, decisions)
        interrupts = get_interrupts(result)

    print("\n最终回复:")
    print(get_last_message_text(result))


if __name__ == "__main__":
    main()
