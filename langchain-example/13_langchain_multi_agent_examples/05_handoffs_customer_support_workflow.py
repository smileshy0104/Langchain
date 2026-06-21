"""
Handoffs：客户支持多步骤工作流。

保修收集 -> 问题分类 -> 解决方案专家。每一步由当前 Agent
通过 Command(update=...) 更新状态，把流程交接给下一步。
"""

from __future__ import annotations

import argparse
from typing import Any, Literal

from langchain.agents import AgentState, create_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
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


def agent_visible_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Hide previous workflow tool-call artifacts before invoking the next agent."""

    visible: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if isinstance(message, AIMessage) and message.tool_calls:
            continue
        content = getattr(message, "content", "")
        if isinstance(message, HumanMessage) or content:
            visible.append(message)
    return visible


def agent_input_state(state: SupportWorkflowState) -> dict[str, Any]:
    return {
        **state,
        "messages": agent_visible_messages(list(state.get("messages", []))),
    }


class SupportWorkflowState(AgentState):
    """State for the support handoff workflow."""

    current_step: NotRequired[
        Literal["warranty_collector", "issue_classifier", "resolution_specialist"]
    ]
    warranty_status: NotRequired[Literal["in_warranty", "out_of_warranty"]]
    issue_type: NotRequired[Literal["hardware", "software"]]


@tool
def record_warranty_status(
    status: Literal["in_warranty", "out_of_warranty"],
    runtime: ToolRuntime,
) -> Command:
    """Record warranty status and transfer to issue classification."""

    return Command(
        goto="issue_classifier",
        update={
            "messages": [
                ToolMessage(
                    content=f"保修状态已记录：{status}",
                    tool_call_id=runtime.tool_call_id or "",
                )
            ],
            "warranty_status": status,
            "current_step": "issue_classifier",
        },
        graph=Command.PARENT,
    )


@tool
def record_issue_type(
    issue_type: Literal["hardware", "software"],
    runtime: ToolRuntime,
) -> Command:
    """Record issue type and transfer to the resolution specialist."""

    return Command(
        goto="resolution_specialist",
        update={
            "messages": [
                ToolMessage(
                    content=f"问题类型已记录：{issue_type}",
                    tool_call_id=runtime.tool_call_id or "",
                )
            ],
            "issue_type": issue_type,
            "current_step": "resolution_specialist",
        },
        graph=Command.PARENT,
    )


@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the support case to a human agent."""

    return f"正在转接到人工客服，原因：{reason}"


@tool
def provide_solution(solution: str) -> str:
    """Provide a final support solution."""

    return f"解决方案：{solution}"


def create_support_workflow() -> Any:
    warranty_collector = create_agent(
        model=create_configured_model(),
        tools=[record_warranty_status],
        system_prompt=(
            "你是保修信息收集专员。根据用户描述判断是否在保修期内。"
            "如果用户说仍在保修期内，调用 record_warranty_status(status='in_warranty')；"
            "否则调用 record_warranty_status(status='out_of_warranty')。"
        ),
    )
    issue_classifier = create_agent(
        model=create_configured_model(),
        tools=[record_issue_type],
        system_prompt=(
            "你是问题分类专家。根据用户描述判断是硬件还是软件问题。"
            "屏幕碎裂、电池损坏、设备进水属于 hardware；"
            "登录失败、配置错误、应用崩溃通常属于 software。"
            "判断后调用 record_issue_type。"
        ),
    )
    resolution_specialist = create_agent(
        model=create_configured_model(),
        tools=[provide_solution, escalate_to_human],
        system_prompt=(
            "你是解决方案专家。根据保修状态和问题类型给出处理方案。"
            "如果硬件问题且在保修期内，建议预约检测并说明是否可能免费维修；"
            "如果复杂或需要人工确认，调用 escalate_to_human。"
        ),
    )

    def call_warranty_collector(state: SupportWorkflowState) -> dict[str, Any]:
        return warranty_collector.invoke(agent_input_state(state))

    def call_issue_classifier(state: SupportWorkflowState) -> dict[str, Any]:
        return issue_classifier.invoke(agent_input_state(state))

    def call_resolution_specialist(state: SupportWorkflowState) -> dict[str, Any]:
        return resolution_specialist.invoke(agent_input_state(state))

    def route_by_step(
        state: SupportWorkflowState,
    ) -> Literal["warranty_collector", "issue_classifier", "resolution_specialist", "__end__"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and not last.tool_calls:
            return "__end__"
        if state.get("current_step") == "resolution_specialist" and isinstance(last, ToolMessage):
            return "__end__"
        return state.get("current_step") or "warranty_collector"

    builder = StateGraph(SupportWorkflowState)
    builder.add_node("warranty_collector", call_warranty_collector)
    builder.add_node("issue_classifier", call_issue_classifier)
    builder.add_node("resolution_specialist", call_resolution_specialist)
    builder.add_conditional_edges(START, lambda _: "warranty_collector", ["warranty_collector"])
    builder.add_conditional_edges(
        "warranty_collector",
        route_by_step,
        ["issue_classifier", "resolution_specialist", END],
    )
    builder.add_conditional_edges(
        "issue_classifier",
        route_by_step,
        ["resolution_specialist", END],
    )
    builder.add_conditional_edges(
        "resolution_specialist",
        route_by_step,
        ["resolution_specialist", END],
    )
    return builder.compile()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Handoffs 案例：客户支持多步骤工作流")
    parser.add_argument(
        "--request",
        default="我的手机屏幕摔碎了，设备还在保修期内，请问可以免费维修吗？",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workflow = create_support_workflow()
    result = workflow.invoke(
        {
            "messages": [HumanMessage(content=args.request)],
            "current_step": "warranty_collector",
        }
    )

    print(last_non_empty_message_text(result))
    print(f"\n保修状态: {result.get('warranty_status', 'unknown')}")
    print(f"问题类型: {result.get('issue_type', 'unknown')}")


if __name__ == "__main__":
    main()
