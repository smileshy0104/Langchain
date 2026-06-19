"""
案例 1：聚合数学与天气 MCP 服务。

本示例启动两个本地 stdio MCP Server：
- servers/math_server.py 提供 add / multiply
- servers/weather_server.py 提供 get_weather

LangChain 通过 MultiServerMCPClient 聚合两个 MCP Server 的工具，再交给 Agent 使用。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from model_config import create_configured_model


BASE_DIR = Path(__file__).parent


async def run_agent(question: str) -> str:
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(BASE_DIR / "servers" / "math_server.py")],
            },
            "weather": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(BASE_DIR / "servers" / "weather_server.py")],
            },
        }
    )

    tools = await client.get_tools()
    agent = create_agent(create_configured_model(), tools)

    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    return response["messages"][-1].content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCP 案例 1：聚合数学与天气服务")
    parser.add_argument(
        "--question",
        default="What is the weather in NYC? Also calculate (3 + 5) * 12.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    print(await run_agent(args.question))


if __name__ == "__main__":
    asyncio.run(main())
