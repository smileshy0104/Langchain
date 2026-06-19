"""Local MCP server exposing sensitive operations for approval demos."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Operations")


@mcp.tool()
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. This is a sensitive operation."""

    return f"已发送邮件给 {to}，主题：{subject}，正文长度：{len(body)}"


@mcp.tool()
def make_payment(payee: str, amount: float) -> str:
    """Make a payment. This is a sensitive operation."""

    return f"已向 {payee} 支付 {amount:.2f} 元。"


@mcp.tool()
def delete_file(file_path: str) -> str:
    """Delete a file. This is a sensitive operation."""

    return f"已模拟删除文件：{file_path}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
