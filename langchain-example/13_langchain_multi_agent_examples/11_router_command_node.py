"""
Router：显式 router 节点 + Command(goto=...)。

和条件边路由不同，本示例把 router 做成一个普通节点，由节点返回
Command(goto="目标节点", update={...})，便于记录路由决策。
"""

from __future__ import annotations

import argparse
from typing import Any, Literal

from langchain.agents import AgentState, create_agent
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import NotRequired

from model_config import create_configured_model


RouteName = Literal["research", "code", "writing"]


class RouterState(AgentState):
    route: NotRequired[RouteName]


def last_message_text(result: dict[str, Any]) -> str:
    return str(result["messages"][-1].content)


def classify_query(state: RouterState) -> RouteName:
    query = str(state["messages"][-1].content).lower()
    if any(keyword in query for keyword in ["api", "code", "python", "bug", "函数", "接口", "代码"]):
        return "code"
    if any(keyword in query for keyword in ["write", "rewrite", "grammar", "文案", "润色", "邮件"]):
        return "writing"
    return "research"


def create_command_router_graph() -> Any:
    research_agent = create_agent(
        model=create_configured_model(),
        tools=[],
        system_prompt="你是研究专家。回答时给出背景、关键结论和下一步建议。",
    )
    code_agent = create_agent(
        model=create_configured_model(),
        tools=[],
        system_prompt="你是编程专家。回答时优先给出代码、解释和注意事项。",
    )
    writing_agent = create_agent(
        model=create_configured_model(),
        tools=[],
        system_prompt="你是写作专家。回答时优化结构、语气和表达。",
    )

    def router_node(state: RouterState) -> Command:
        route = classify_query(state)
        return Command(goto=route, update={"route": route})

    builder = StateGraph(RouterState)
    builder.add_node("router", router_node)
    builder.add_node("research", lambda state: research_agent.invoke(state))
    builder.add_node("code", lambda state: code_agent.invoke(state))
    builder.add_node("writing", lambda state: writing_agent.invoke(state))
    builder.add_edge(START, "router")

    for node in ["research", "code", "writing"]:
        builder.add_edge(node, END)

    return builder.compile()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Router 案例：显式 Command 路由节点")
    parser.add_argument(
        "--request",
        default="帮我润色一封发给客户的项目延期说明邮件。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = create_command_router_graph()
    result = graph.invoke({"messages": [{"role": "user", "content": args.request}]})
    print(f"路由目标: {result.get('route', 'unknown')}")
    print(last_message_text(result))


if __name__ == "__main__":
    main()
