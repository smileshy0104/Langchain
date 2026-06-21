"""
实战案例 1：完整的多领域客服系统。

账单 Agent 与技术支持 Agent 是两个独立节点，通过 handoff 工具
在父图中交接控制权，并使用 InMemorySaver 保留同一 thread 的状态。
"""

from __future__ import annotations

import argparse
from typing import Any, Literal

from langchain.agents import AgentState, create_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import NotRequired

from model_config import create_configured_model


class CustomerServiceState(AgentState):
    """Shared state for the customer service handoff graph."""

    active_agent: NotRequired[Literal["billing_agent", "technical_agent"]]
    customer_id: NotRequired[str]
    issue_category: NotRequired[str]
    handoff_context: NotRequired[str]


def last_non_empty_message_text(result: dict[str, Any]) -> str:
    for message in reversed(result.get("messages", [])):
        content = getattr(message, "content", "")
        if content:
            return str(content)
    return ""


def agent_visible_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Avoid passing stale tool-call artifacts between independent agents."""

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


def agent_input_state(state: CustomerServiceState) -> dict[str, Any]:
    messages = agent_visible_messages(list(state.get("messages", [])))
    if handoff_context := state.get("handoff_context"):
        messages.append(HumanMessage(content=f"交接上下文：{handoff_context}"))
    return {**state, "messages": messages}


@tool
def transfer_to_billing(runtime: ToolRuntime) -> Command:
    """Transfer to the billing specialist."""

    return Command(
        goto="billing_agent",
        update={
            "active_agent": "billing_agent",
            "issue_category": "billing",
            "handoff_context": "用户的问题涉及账单、发票、付款状态或退款。请账单专家直接处理。",
            "messages": [
                ToolMessage(
                    content="已转接到账单专家。",
                    tool_call_id=runtime.tool_call_id or "",
                )
            ],
        },
        graph=Command.PARENT,
    )


@tool
def transfer_to_technical(runtime: ToolRuntime) -> Command:
    """Transfer to technical support."""

    return Command(
        goto="technical_agent",
        update={
            "active_agent": "technical_agent",
            "issue_category": "technical",
            "handoff_context": "用户的问题涉及设备、故障、网络或产品使用。请技术支持直接处理。",
            "messages": [
                ToolMessage(
                    content="已转接到技术支持。",
                    tool_call_id=runtime.tool_call_id or "",
                )
            ],
        },
        graph=Command.PARENT,
    )


@tool
def check_invoice(invoice_id: str) -> str:
    """Look up invoice details."""

    return f"发票 {invoice_id}: 金额 299.00 美元，状态已支付，账期 2026-05。"


@tool
def request_refund(invoice_id: str, reason: str) -> str:
    """Create a refund review request."""

    return f"已为发票 {invoice_id} 创建退款审核请求，原因：{reason}。"


@tool
def troubleshoot_device(device: str, issue: str) -> str:
    """Troubleshoot a device issue."""

    return (
        f"对于 {device} 的 {issue} 问题：请先重启设备，检查网络连接，"
        "再升级到最新固件；如果仍失败，请收集错误码并联系人工支持。"
    )


@tool
def create_support_ticket(summary: str, priority: Literal["normal", "high"] = "normal") -> str:
    """Create a technical support ticket."""

    return f"已创建技术支持工单，优先级：{priority}，摘要：{summary}"


def create_customer_service_graph() -> Any:
    billing_agent = create_agent(
        model=create_configured_model(),
        tools=[check_invoice, request_refund, transfer_to_technical],
        system_prompt=(
            "你是账单专家，负责发票查询、付款状态、退款和账单解释。"
            "如果用户询问设备故障、网络或产品使用问题，调用 transfer_to_technical。"
            "如果交接上下文说明这是账单问题，请直接处理，不要再转接。"
        ),
    )
    technical_agent = create_agent(
        model=create_configured_model(),
        tools=[troubleshoot_device, create_support_ticket, transfer_to_billing],
        system_prompt=(
            "你是技术支持专家，负责设备故障、网络和产品使用排查。"
            "如果用户询问发票、付款、账单或退款，调用 transfer_to_billing。"
            "如果交接上下文说明这是技术问题，请直接处理，不要再转接。"
        ),
    )

    def call_billing_agent(state: CustomerServiceState) -> dict[str, Any]:
        return billing_agent.invoke(agent_input_state(state))

    def call_technical_agent(state: CustomerServiceState) -> dict[str, Any]:
        return technical_agent.invoke(agent_input_state(state))

    def route_initial(
        state: CustomerServiceState,
    ) -> Literal["billing_agent", "technical_agent"]:
        return state.get("active_agent") or "billing_agent"

    def route_after_agent(
        state: CustomerServiceState,
    ) -> Literal["billing_agent", "technical_agent", "__end__"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and not last.tool_calls:
            return "__end__"
        return state.get("active_agent") or "billing_agent"

    builder = StateGraph(CustomerServiceState)
    builder.add_node("billing_agent", call_billing_agent)
    builder.add_node("technical_agent", call_technical_agent)
    builder.add_conditional_edges(START, route_initial, ["billing_agent", "technical_agent"])
    builder.add_conditional_edges(
        "billing_agent",
        route_after_agent,
        ["billing_agent", "technical_agent", END],
    )
    builder.add_conditional_edges(
        "technical_agent",
        route_after_agent,
        ["billing_agent", "technical_agent", END],
    )
    return builder.compile(checkpointer=InMemorySaver())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实战案例 1：完整多领域客服系统")
    parser.add_argument(
        "--request",
        default="我的发票 #12345 金额是多少？",
    )
    parser.add_argument("--customer-id", default="customer-123")
    parser.add_argument(
        "--active-agent",
        choices=["billing_agent", "technical_agent"],
        default="billing_agent",
    )
    parser.add_argument("--thread-id", default="practical-customer-service-001")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = create_customer_service_graph()
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=args.request)],
            "customer_id": args.customer_id,
            "active_agent": args.active_agent,
            "issue_category": "billing" if args.active_agent == "billing_agent" else "technical",
        },
        config={
            "configurable": {"thread_id": args.thread_id},
            "recursion_limit": 8,
        },
    )

    print(last_non_empty_message_text(result))
    print(f"\n客户 ID: {result.get('customer_id', args.customer_id)}")
    print(f"最终活跃 Agent: {result.get('active_agent', args.active_agent)}")
    print(f"问题分类: {result.get('issue_category', 'unknown')}")


if __name__ == "__main__":
    main()
