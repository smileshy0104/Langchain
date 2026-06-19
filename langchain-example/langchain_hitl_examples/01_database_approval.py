"""
案例 1：数据库操作审批系统。

本示例演示 HumanInTheLoopMiddleware 如何在模型准备调用 SQL 执行或
数据导出工具前暂停，让人工审批 approve / edit / reject 后再继续执行。
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
def execute_sql(query: str) -> str:
    """Execute a SQL query after human approval."""

    dangerous_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER"]
    query_upper = query.upper()
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            return f"错误：检测到危险关键字 {keyword}，查询未执行。"
    return f"已执行 SQL：{query}"


@tool
def export_data(table: str, format: str = "csv") -> str:
    """Export table data after human approval."""

    return f"已导出表 {table}，格式：{format}"


def describe_sql_approval(tool_call: ToolCall, state: Any, runtime: Any) -> str:
    args = tool_call["args"]
    return (
        "SQL 审批:\n"
        f"查询: {args['query']}\n"
        "请确认查询正确且不会泄露或破坏数据。"
    )


def create_db_agent() -> Any:
    return create_agent(
        model=create_configured_model(),
        tools=[execute_sql, export_data],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "execute_sql": {
                        "allowed_decisions": ["approve", "edit"],
                        "description": describe_sql_approval,
                    },
                    "export_data": {
                        "allowed_decisions": ["approve", "reject"],
                        "description": "数据导出审批：请确认导出表和格式符合数据安全要求。",
                    },
                }
            )
        ],
        checkpointer=InMemorySaver(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HITL 案例 1：数据库操作审批系统")
    parser.add_argument(
        "--request",
        default="请导出 users 表为 csv。",
        help="发送给 Agent 的用户请求。",
    )
    parser.add_argument(
        "--decision",
        choices=["approve", "edit", "reject"],
        default="approve",
        help="人工审批决策。execute_sql 不允许 reject；export_data 不允许 edit。",
    )
    parser.add_argument("--edited-query", help="decision=edit 时替换 SQL 查询。")
    parser.add_argument("--edited-table", help="decision=edit 时替换导出表名。")
    parser.add_argument("--thread-id", default="hitl-db-session-001")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = create_db_agent()
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
        if args.edited_query:
            edited_args["query"] = args.edited_query
        if args.edited_table:
            edited_args["table"] = args.edited_table

        decisions = build_decisions(
            interrupt,
            args.decision,
            edited_args=edited_args,
            reject_message="人工审核拒绝导出数据。",
        )
        print("\n本次提交的人工决策:")
        print(decisions)

        result = resume_with_decisions(agent, config, decisions)
        interrupts = get_interrupts(result)

    print("\n最终回复:")
    print(get_last_message_text(result))


if __name__ == "__main__":
    main()
