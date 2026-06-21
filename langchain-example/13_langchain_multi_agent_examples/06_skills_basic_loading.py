"""
Skills：基础技能加载模式。

Agent 一开始只知道可用技能目录；当用户请求进入某个领域时，
通过 load_skill 按需加载该领域的 prompt、schema 和业务规则。
"""

from __future__ import annotations

import argparse
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool

from model_config import create_configured_model


SKILLS = {
    "sales_analytics": {
        "name": "销售分析",
        "description": "分析销售数据、生成报告、识别趋势",
        "schema": (
            "orders(order_id, customer_id, order_date, status, total_amount)\n"
            "customers(customer_id, name, region, segment)\n"
            "revenues(order_id, revenue_amount, recognized_date)"
        ),
        "rules": [
            "使用标准 SQL 语法。",
            "日期使用 ISO 格式。",
            "货币金额保留两位小数。",
            "趋势分析默认按周聚合，除非用户指定粒度。",
        ],
    },
    "inventory_management": {
        "name": "库存管理",
        "description": "管理库存水平、补货建议、库存优化",
        "schema": (
            "products(product_id, sku, name, category)\n"
            "inventory(product_id, warehouse_id, quantity_on_hand, reorder_point)\n"
            "suppliers(supplier_id, name, lead_time_days)"
        ),
        "rules": [
            "补货建议需要考虑当前库存、补货点和供应商交期。",
            "低库存商品定义为 quantity_on_hand <= reorder_point。",
            "库存报告默认按仓库和品类分组。",
        ],
    },
}


@tool
def load_skill(skill_name: str) -> str:
    """Load detailed instructions for a specific skill."""

    if skill_name not in SKILLS:
        return f"错误：未知技能 '{skill_name}'。可用技能：{list(SKILLS.keys())}"

    skill = SKILLS[skill_name]
    rules = "\n".join(f"- {rule}" for rule in skill["rules"])
    return f"""技能: {skill['name']}

描述: {skill['description']}

数据库 Schema:
{skill['schema']}

业务规则:
{rules}
"""


def create_skill_agent() -> Any:
    return create_agent(
        model=create_configured_model(),
        tools=[load_skill],
        system_prompt=(
            "你是一个 SQL 分析助手。\n\n"
            "你可以访问以下技能：\n"
            "- sales_analytics: 销售分析\n"
            "- inventory_management: 库存管理\n\n"
            "当需要处理特定领域请求时，先使用 load_skill 加载该领域详细信息，"
            "再基于技能内容给出 SQL 草案和分析说明。"
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Skills 案例：基础技能加载")
    parser.add_argument(
        "--request",
        default="帮我分析上个月的销售额趋势，并给出可以使用的 SQL。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = create_skill_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": args.request}]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
