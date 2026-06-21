"""
Handoffs：单 Agent + Middleware 的轻量交接模式。

这个版本适合顺序客服流程：同一个 Agent 保留连续消息历史，
通过状态字段 current_step 切换一线客服和专家阶段的提示词与工具。
"""

from __future__ import annotations

import argparse
from typing import Any, Literal

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.tools import ToolRuntime
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from typing_extensions import NotRequired

from model_config import create_configured_model


def last_non_empty_message_text(result: dict[str, Any]) -> str:
    """Return the most recent non-empty message content."""

    for message in reversed(result.get("messages", [])):
        content = getattr(message, "content", "")
        if content:
            return str(content)
    return ""


class SupportHandoffState(AgentState):
    """Single-agent handoff state."""

    current_step: NotRequired[Literal["triage", "specialist"]]


@tool
def transfer_to_specialist(runtime: ToolRuntime) -> Command:
    """Transfer the conversation from triage to the technical specialist."""

    return Command(
        update={
            "current_step": "specialist",
            "messages": [
                ToolMessage(
                    content="已转接到高级技术专家。",
                    tool_call_id=runtime.tool_call_id or "",
                )
            ],
        }
    )


@tool
def resolve_issue(solution: str) -> str:
    """Provide the final specialist solution."""

    return f"专家解决方案：{solution}"


@wrap_model_call(state_schema=SupportHandoffState, tools=[transfer_to_specialist, resolve_issue])
def apply_step_config(
    request: ModelRequest,
    handler: Any,
) -> ModelResponse:
    """Switch prompt and available tools according to current_step."""

    step = request.state.get("current_step", "triage")
    if step == "triage":
        request = request.override(
            system_message=SystemMessage(
                content=(
                    "你是一线客服。先判断用户问题是否需要高级专家。"
                    "如果用户的问题涉及设备损坏、复杂故障或保修判断，"
                    "必须调用 transfer_to_specialist。"
                )
            ),
            tools=[transfer_to_specialist],
        )
    else:
        request = request.override(
            system_message=SystemMessage(
                content=(
                    "你是高级技术专家。直接分析问题，给出清晰解决方案。"
                    "需要结束时调用 resolve_issue。"
                )
            ),
            tools=[resolve_issue],
        )

    return handler(request)


def create_support_agent() -> Any:
    return create_agent(
        model=create_configured_model(),
        tools=[transfer_to_specialist, resolve_issue],
        state_schema=SupportHandoffState,
        middleware=[apply_step_config],
        checkpointer=InMemorySaver(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Handoffs 案例：单 Agent + Middleware")
    parser.add_argument(
        "--request",
        default="我的手机屏幕摔碎了，而且还在保修期内，请问怎么处理？",
    )
    parser.add_argument("--thread-id", default="handoff-single-agent-001")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = create_support_agent()
    config = {"configurable": {"thread_id": args.thread_id}}
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": args.request}],
            "current_step": "triage",
        },
        config=config,
    )

    print(last_non_empty_message_text(result))
    print(f"\n当前阶段: {result.get('current_step', 'triage')}")


if __name__ == "__main__":
    main()
