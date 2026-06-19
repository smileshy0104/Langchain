"""Local Orders MCP server used by the context injection example."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Orders")


ORDERS = {
    "user_123": [
        {"order_id": "ORD-1001", "item": "LangChain 课程", "status": "已完成"},
        {"order_id": "ORD-1002", "item": "MCP 实战训练营", "status": "配送中"},
    ],
    "user_456": [
        {"order_id": "ORD-2001", "item": "企业支持服务", "status": "处理中"},
    ],
}


@mcp.tool()
def search_orders(query: str = "", user_id: str = "") -> str:
    """Search orders for the current user."""

    if not user_id:
        return "缺少 user_id，无法查询订单。"

    orders = ORDERS.get(user_id, [])
    if not orders:
        return f"用户 {user_id} 暂无订单记录。"

    filtered = [
        order
        for order in orders
        if not query
        or query.lower() in order["order_id"].lower()
        or query.lower() in order["item"].lower()
        or query.lower() in order["status"].lower()
    ]
    if not filtered:
        return f"用户 {user_id} 没有匹配 {query!r} 的订单。"

    lines = [
        f"- {order['order_id']}：{order['item']}，状态：{order['status']}"
        for order in filtered
    ]
    return f"用户 {user_id} 的订单：\n" + "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
