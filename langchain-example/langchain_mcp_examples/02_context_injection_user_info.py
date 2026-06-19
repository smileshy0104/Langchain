"""
案例 2：基于上下文注入用户信息。

MCP Server 与 LangChain Runtime 是隔离的；tool_interceptors 用来桥接边界。
这里在调用 orders MCP 工具前，把 AppContext.user_id 注入工具参数。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

from model_config import create_configured_model


BASE_DIR = Path(__file__).parent


@dataclass
class AppContext:
    user_id: str
    api_key: str = "demo-api-key"

# 工具拦截器
async def inject_user_context(request: MCPToolCallRequest, handler):
    """Inject runtime context into MCP tool args before the server sees the call."""

    runtime = request.runtime
    request = request.override(
        args={
            **request.args,
            "user_id": runtime.context.user_id,
        },
        headers={
            "Authorization": f"Bearer {runtime.context.api_key}",
        },
    )
    return await handler(request)

# 运行
async def run_agent(question: str, context: AppContext, thread_id: str) -> str:
    client = MultiServerMCPClient(
        {
            "orders": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(BASE_DIR / "servers" / "orders_server.py")],
            }
        },
        tool_interceptors=[inject_user_context],
    )

    tools = await client.get_tools()
    agent = create_agent(
        create_configured_model(),
        tools,
        context_schema=AppContext,
    )

    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}},
        context=context,
    )
    return response["messages"][-1].content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCP 案例 2：上下文注入用户信息")
    parser.add_argument("--user-id", default="user_123")
    parser.add_argument("--api-key", default="demo-api-key")
    parser.add_argument("--thread-id", default="mcp-context-injection-thread")
    parser.add_argument("--question", default="Search my orders")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    context = AppContext(user_id=args.user_id, api_key=args.api_key)
    print(await run_agent(args.question, context, args.thread_id))


if __name__ == "__main__":
    asyncio.run(main())
