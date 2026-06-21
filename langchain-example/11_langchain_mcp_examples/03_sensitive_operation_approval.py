"""
案例 3：敏感操作审批。

本示例用 tool_interceptors 拦截 MCP 敏感工具：
- 未在 --approved-tools 中声明的敏感工具会被短路拦截
- 已批准的工具才会继续发送给 MCP Server 执行

文档中的 Command(interrupt=...) 更适合接入完整 Human-in-the-loop UI。
为了让脚本可直接运行，这里采用命令行批准列表演示同一审批边界。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from mcp.types import CallToolResult, TextContent

from model_config import create_configured_model


BASE_DIR = Path(__file__).parent
SENSITIVE_TOOLS = {"delete_file", "send_email", "make_payment"}


@dataclass
class ApprovalContext:
    approved_tools: set[str] = field(default_factory=set)


async def approval_interceptor(request: MCPToolCallRequest, handler):
    """Short-circuit sensitive MCP tools until the user approves them."""

    if request.name in SENSITIVE_TOOLS:
        approved = request.runtime.context.approved_tools
        if request.name not in approved:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            f"敏感工具 {request.name!r} 尚未审批，已阻止执行。"
                            f"参数：{request.args}"
                        ),
                    )
                ],
                isError=True,
            )

    return await handler(request)


async def run_agent(question: str, context: ApprovalContext, thread_id: str) -> str:
    client = MultiServerMCPClient(
        {
            "operations": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(BASE_DIR / "servers" / "operations_server.py")],
            }
        },
        tool_interceptors=[approval_interceptor],
    )

    tools = await client.get_tools()
    agent = create_agent(
        create_configured_model(),
        tools,
        context_schema=ApprovalContext,
    )

    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}},
        context=context,
    )
    return response["messages"][-1].content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCP 案例 3：敏感操作审批")
    parser.add_argument(
        "--approved-tools",
        nargs="*",
        default=[],
        choices=sorted(SENSITIVE_TOOLS),
        help="允许执行的敏感 MCP 工具",
    )
    parser.add_argument("--thread-id", default="mcp-approval-thread")
    parser.add_argument(
        "--question",
        default="Send an email to alice@example.com with subject Hello and body This is a test.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    context = ApprovalContext(approved_tools=set(args.approved_tools))
    print(await run_agent(args.question, context, args.thread_id))


if __name__ == "__main__":
    asyncio.run(main())
