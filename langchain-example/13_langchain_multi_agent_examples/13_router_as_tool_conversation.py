"""
Router：把 stateless router 包装成外层对话 Agent 的工具。

Router 本身保持一次性查询；多轮对话、语气和记忆由外层 Agent 负责。
"""

from __future__ import annotations

import argparse
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool

from model_config import create_configured_model


def ask_github(query: str) -> str:
    return (
        "GitHub 结果：需要检查相关模块的 README、导出任务实现、权限校验、"
        "失败重试逻辑，以及最近 PR 中是否修改了账单字段。"
    )


def ask_notion(query: str) -> str:
    return (
        "Notion 结果：产品方案要求支持 CSV/XLSX，导出内容需要包含发票号、"
        "账期、支付状态和客户 ID；上线前需要更新帮助文档。"
    )


def ask_slack(query: str) -> str:
    return (
        "Slack 结果：团队讨论提到大客户导出量可能较大，需要限流；"
        "客服团队希望上线前拿到常见问题回复模板。"
    )


@tool
def ask_knowledge_router(query: str) -> str:
    """Query GitHub, Notion, and Slack style knowledge sources, then synthesize results."""

    results = [ask_github(query), ask_notion(query), ask_slack(query)]
    return "\n".join(results)


def create_conversational_router_agent() -> Any:
    return create_agent(
        model=create_configured_model(),
        tools=[ask_knowledge_router],
        system_prompt=(
            "你是一个带长期对话语气的助手。"
            "当用户需要跨知识源查询、上线准备、项目背景或团队决策时，"
            "必须调用 ask_knowledge_router，然后用自然语言综合回答。"
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Router 案例：stateless router 包装成工具")
    parser.add_argument(
        "--request",
        default="帮我跨知识源确认账单导出功能上线前要注意什么。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = create_conversational_router_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": args.request}]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
