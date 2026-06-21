"""Local Weather MCP server over stdio."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Weather")


@mcp.tool()
def get_weather(location: str) -> str:
    """Get weather for a location."""

    weather = {
        "NYC": "NYC is sunny, 24 C, light wind.",
        "New York": "New York is sunny, 24 C, light wind.",
        "北京": "北京晴，20 C，空气质量良好。",
        "上海": "上海多云，22 C，湿度 65%。",
    }
    return weather.get(location, f"{location} 天气晴朗，温度 23 C。")


if __name__ == "__main__":
    mcp.run(transport="stdio")
