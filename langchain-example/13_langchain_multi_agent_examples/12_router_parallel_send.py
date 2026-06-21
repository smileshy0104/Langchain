"""
Router：并行 Send fan-out。

同一个查询并行发送给 GitHub / Notion / Slack 三个知识源 Agent，
等待所有分支完成后由 summarize 节点综合答案。
"""

from __future__ import annotations

import argparse
from typing import Any

from langchain.agents import AgentState, create_agent
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import NotRequired

from model_config import create_configured_model


class ParallelRouterState(AgentState):
    github_result: NotRequired[str]
    notion_result: NotRequired[str]
    slack_result: NotRequired[str]
    answer: NotRequired[str]


def classify_and_route(state: ParallelRouterState) -> list[Send]:
    """Fan out the same query to independent knowledge-source agents."""

    messages = state["messages"]
    return [
        Send("github", {"messages": messages}),
        Send("notion", {"messages": messages}),
        Send("slack", {"messages": messages}),
    ]


def create_parallel_router() -> Any:
    github_agent = create_agent(
        model=create_configured_model(),
        tools=[],
        system_prompt=(
            "你是 GitHub 知识源 Agent。"
            "基于模拟的代码仓库、README、issue、PR 和提交历史回答。"
            "请给出与用户问题相关的代码侧注意事项。"
        ),
    )
    notion_agent = create_agent(
        model=create_configured_model(),
        tools=[],
        system_prompt=(
            "你是 Notion 知识源 Agent。"
            "基于模拟的产品文档、项目计划、会议记录和内部规范回答。"
            "请给出与用户问题相关的产品和流程注意事项。"
        ),
    )
    slack_agent = create_agent(
        model=create_configured_model(),
        tools=[],
        system_prompt=(
            "你是 Slack 知识源 Agent。"
            "基于模拟的团队讨论、上线通知、故障沟通和临时决策回答。"
            "请给出与用户问题相关的团队协作注意事项。"
        ),
    )
    summarizer = create_agent(
        model=create_configured_model(),
        tools=[],
        system_prompt="你是汇总专家。整合多个知识源结果，消除重复，给出最终中文答案。",
    )

    def call_github(state: ParallelRouterState) -> dict[str, str]:
        result = github_agent.invoke(state)
        return {"github_result": str(result["messages"][-1].content)}

    def call_notion(state: ParallelRouterState) -> dict[str, str]:
        result = notion_agent.invoke(state)
        return {"notion_result": str(result["messages"][-1].content)}

    def call_slack(state: ParallelRouterState) -> dict[str, str]:
        result = slack_agent.invoke(state)
        return {"slack_result": str(result["messages"][-1].content)}

    def summarize_results(state: ParallelRouterState) -> dict[str, Any]:
        query = str(state["messages"][-1].content)
        prompt = f"""请综合以下知识源结果回答用户问题。

用户问题:
{query}

GitHub:
{state.get('github_result', '无结果')}

Notion:
{state.get('notion_result', '无结果')}

Slack:
{state.get('slack_result', '无结果')}
"""
        result = summarizer.invoke({"messages": [{"role": "user", "content": prompt}]})
        answer = str(result["messages"][-1].content)
        return {"answer": answer, "messages": [AIMessage(content=answer)]}

    builder = StateGraph(ParallelRouterState)
    builder.add_node("github", call_github)
    builder.add_node("notion", call_notion)
    builder.add_node("slack", call_slack)
    builder.add_node("summarize", summarize_results)
    builder.add_conditional_edges(START, classify_and_route, ["github", "notion", "slack"])
    builder.add_edge(["github", "notion", "slack"], "summarize")
    builder.add_edge("summarize", END)
    return builder.compile()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Router 案例：并行 Send fan-out")
    parser.add_argument(
        "--request",
        default="我们准备上线新的账单导出功能，需要确认代码、文档和团队讨论里有哪些注意事项。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = create_parallel_router()
    result = graph.invoke({"messages": [{"role": "user", "content": args.request}]})
    print(result.get("answer", result["messages"][-1].content))


if __name__ == "__main__":
    main()
