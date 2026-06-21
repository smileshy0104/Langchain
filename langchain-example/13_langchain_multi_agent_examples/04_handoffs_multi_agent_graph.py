"""
Handoffs：多个 Agent 子图之间交接控制权。

销售 Agent 和支持 Agent 是两个独立节点。当前 Agent 可以通过
Command(goto=..., graph=Command.PARENT) 把父图控制权交给另一个 Agent。
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
    """Hide handoff tool-call artifacts before invoking the next agent."""

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


def agent_input_state(state: "MultiAgentHandoffState") -> dict[str, Any]:
    messages = agent_visible_messages(list(state.get("messages", [])))
    handoff_context = state.get("handoff_context")
    if handoff_context:
        messages.append(HumanMessage(content=f"交接上下文：{handoff_context}"))
    return {
        **state,
        "messages": messages,
    }


class MultiAgentHandoffState(AgentState):
    """Shared state for sales/support handoffs."""

    active_agent: NotRequired[Literal["sales_agent", "support_agent"]]
    handoff_context: NotRequired[str]


def last_ai_message(runtime: ToolRuntime) -> AIMessage:
    return next(
        message
        for message in reversed(runtime.state["messages"])
        if isinstance(message, AIMessage)
    )


@tool
def transfer_to_sales(runtime: ToolRuntime) -> Command:
    """Transfer to the sales agent for pricing, plans, or purchasing questions."""

    return Command(
        goto="sales_agent",
        update={
            "active_agent": "sales_agent",
            "handoff_context": "用户需要了解套餐、定价或购买流程。请销售专家直接处理这些商业问题。",
            "messages": [
                ToolMessage(
                    content="已从技术支持转接到销售专家。",
                    tool_call_id=runtime.tool_call_id or "",
                ),
            ],
        },
        graph=Command.PARENT,
    )


@tool
def transfer_to_support(runtime: ToolRuntime) -> Command:
    """Transfer to technical support for troubleshooting or product usage issues."""

    return Command(
        goto="support_agent",
        update={
            "active_agent": "support_agent",
            "handoff_context": "用户需要技术排查或产品使用支持。请技术支持直接处理故障问题，不要因为原始请求里提到套餐就转回销售。",
            "messages": [
                ToolMessage(
                    content="已从销售专家转接到技术支持。",
                    tool_call_id=runtime.tool_call_id or "",
                ),
            ],
        },
        graph=Command.PARENT,
    )


@tool
def explain_pricing(plan: str) -> str:
    """Explain pricing for a product plan."""

    prices = {
        "free": "免费版：0 元，适合个人试用。",
        "pro": "Pro 版：299 元/月，包含团队协作和优先支持。",
        "enterprise": "企业版：按席位报价，包含 SLA、私有化选项和专属支持。",
    }
    return prices.get(plan.lower(), f"{plan} 套餐需要销售顾问进一步确认。")


@tool
def troubleshoot_product(product: str, issue: str) -> str:
    """Troubleshoot a product usage issue."""

    return f"针对 {product} 的 {issue} 问题：请先重启设备，检查网络，再更新到最新版本。"


def create_handoff_graph() -> Any:
    sales_agent = create_agent(
        model=create_configured_model(),
        tools=[explain_pricing, transfer_to_support],
        system_prompt=(
            "你是销售专家，负责产品套餐、定价和购买流程。"
            "如果用户询问故障、使用问题或维修问题，必须调用 transfer_to_support。"
            "如果交接上下文说明这是销售问题，请直接回答，不要再次转接。"
        ),
    )
    support_agent = create_agent(
        model=create_configured_model(),
        tools=[troubleshoot_product, transfer_to_sales],
        system_prompt=(
            "你是技术支持专家，负责产品使用、故障排查和维修建议。"
            "如果用户询问定价、套餐或购买流程，必须调用 transfer_to_sales。"
            "如果交接上下文说明这是技术问题，请直接排查，不要再次转接。"
        ),
    )

    def call_sales_agent(state: MultiAgentHandoffState) -> dict[str, Any]:
        return sales_agent.invoke(agent_input_state(state))

    def call_support_agent(state: MultiAgentHandoffState) -> dict[str, Any]:
        return support_agent.invoke(agent_input_state(state))

    def route_initial(
        state: MultiAgentHandoffState,
    ) -> Literal["sales_agent", "support_agent"]:
        return state.get("active_agent") or "sales_agent"

    def route_after_agent(
        state: MultiAgentHandoffState,
    ) -> Literal["sales_agent", "support_agent", "__end__"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and not last.tool_calls:
            return "__end__"
        return state.get("active_agent") or "sales_agent"

    builder = StateGraph(MultiAgentHandoffState)
    builder.add_node("sales_agent", call_sales_agent)
    builder.add_node("support_agent", call_support_agent)
    builder.add_conditional_edges(START, route_initial, ["sales_agent", "support_agent"])
    builder.add_conditional_edges(
        "sales_agent",
        route_after_agent,
        ["sales_agent", "support_agent", END],
    )
    builder.add_conditional_edges(
        "support_agent",
        route_after_agent,
        ["sales_agent", "support_agent", END],
    )
    return builder.compile()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Handoffs 案例：多 Agent 子图交接")
    parser.add_argument(
        "--request",
        default="我本来在了解 Pro 套餐，但现在设备一直无法联网，请转技术支持帮我排查。",
    )
    parser.add_argument(
        "--active-agent",
        choices=["sales_agent", "support_agent"],
        default="sales_agent",
        help="初始活跃 Agent。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = create_handoff_graph()
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=args.request)],
            "active_agent": args.active_agent,
        },
        config={"recursion_limit": 8},
    )

    print(last_non_empty_message_text(result))
    print(f"\n最终活跃 Agent: {result.get('active_agent', args.active_agent)}")


if __name__ == "__main__":
    main()
