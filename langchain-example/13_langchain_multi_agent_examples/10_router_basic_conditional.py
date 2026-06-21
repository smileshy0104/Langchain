"""
Router：条件边单路由。

根据用户请求把消息路由到 research / code / writing 三个专家 Agent 中的一个。
这是 Router 模式最小可运行形态：先分类，再分发。
"""

from __future__ import annotations

import argparse
from typing import Any, Literal

from langchain.agents import AgentState, create_agent
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from model_config import create_configured_model


RouteName = Literal["research", "code", "writing"]


def last_message_text(result: dict[str, Any]) -> str:
    return str(result["messages"][-1].content)


def classify_query(state: AgentState) -> RouteName:
    """Classify the latest user query into one expert route."""

    query = str(state["messages"][-1].content).lower()
    if any(keyword in query for keyword in ["code", "programming", "function", "bug", "python", "代码", "函数", "报错"]):
        return "code"
    if any(keyword in query for keyword in ["write", "edit", "improve", "grammar", "文案", "润色", "写作", "修改"]):
        return "writing"
    return "research"


def create_router_graph() -> Any:
    research_agent = create_agent(
        model=create_configured_model(),
        tools=[],
        system_prompt="你是研究专家，提供结构化背景信息、事实分析和可执行建议。",
    )
    code_agent = create_agent(
        model=create_configured_model(),
        tools=[],
        system_prompt="你是编程专家，帮助解决代码、算法、调试和工程实现问题。",
    )
    writing_agent = create_agent(
        model=create_configured_model(),
        tools=[],
        system_prompt="你是写作专家，帮助改进文本质量、结构、语气和表达。",
    )

    def call_research(state: AgentState) -> dict[str, Any]:
        return research_agent.invoke(state)

    def call_code(state: AgentState) -> dict[str, Any]:
        return code_agent.invoke(state)

    def call_writing(state: AgentState) -> dict[str, Any]:
        return writing_agent.invoke(state)

    builder = StateGraph(AgentState)
    builder.add_node("research", call_research)
    builder.add_node("code", call_code)
    builder.add_node("writing", call_writing)
    builder.add_conditional_edges(START, classify_query, ["research", "code", "writing"])

    for node in ["research", "code", "writing"]:
        builder.add_edge(node, END)

    return builder.compile()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Router 案例：条件边单路由")
    parser.add_argument(
        "--request",
        default="如何用 Python 实现快速排序？",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initial_state = {"messages": [HumanMessage(content=args.request)]}
    route = classify_query(initial_state)  # type: ignore[arg-type]
    graph = create_router_graph()
    result = graph.invoke(initial_state)
    print(f"路由目标: {route}")
    print(last_message_text(result))


if __name__ == "__main__":
    main()
