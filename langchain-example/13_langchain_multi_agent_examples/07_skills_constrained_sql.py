"""
Skills：强约束 SQL 生成。

write_sql_query 会检查 Agent state 中的 skills_loaded。
如果还没有加载对应技能，就拒绝验证 SQL，迫使 Agent 先调用 load_sql_skill。
"""

from __future__ import annotations

import argparse
from typing import Any, Literal

from langchain.agents import AgentState, create_agent
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command
from typing_extensions import NotRequired

from model_config import create_configured_model


SUPPORTED_SKILLS = ("sales_analytics", "inventory_management")

SQL_SKILLS = {
    "sales_analytics": {
        "name": "销售分析",
        "schema": (
            "orders(order_id, customer_id, order_date, status, total_amount)\n"
            "customers(customer_id, name, region, segment)\n"
            "revenues(order_id, revenue_amount, recognized_date)"
        ),
        "rules": "只统计 status='paid' 的订单；趋势分析默认按周聚合。",
    },
    "inventory_management": {
        "name": "库存管理",
        "schema": (
            "products(product_id, sku, name, category)\n"
            "inventory(product_id, warehouse_id, quantity_on_hand, reorder_point)\n"
            "suppliers(supplier_id, name, lead_time_days)"
        ),
        "rules": "低库存定义为 quantity_on_hand <= reorder_point；补货需考虑 lead_time_days。",
    },
}


class SkillAgentState(AgentState):
    """Agent state recording loaded skills."""

    skills_loaded: NotRequired[list[str]]


@tool
def load_sql_skill(skill_name: Literal["sales_analytics", "inventory_management"], runtime: ToolRuntime) -> Command:
    """Load SQL schema and rules for a supported business skill."""

    skill = SQL_SKILLS[skill_name]
    loaded = list(runtime.state.get("skills_loaded", []))
    if skill_name not in loaded:
        loaded.append(skill_name)

    content = f"""已加载技能: {skill['name']}

Schema:
{skill['schema']}

规则:
{skill['rules']}
"""
    return Command(
        update={
            "skills_loaded": loaded,
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=runtime.tool_call_id or "",
                )
            ],
        }
    )


@tool
def write_sql_query(
    query: str,
    vertical: Literal["sales_analytics", "inventory_management"],
    runtime: ToolRuntime,
) -> str:
    """Validate a SQL query after the matching skill has been loaded."""

    skills_loaded = runtime.state.get("skills_loaded", [])
    if vertical not in skills_loaded:
        return (
            f"错误：你必须先加载 '{vertical}' 技能以了解数据库 schema。"
            f"请调用 load_sql_skill('{vertical}')。"
        )

    return (
        f"SQL 查询已验证 ({vertical}):\n\n"
        f"```sql\n{query}\n```\n\n"
        "查询已验证，准备执行。"
    )


def create_constrained_sql_agent() -> Any:
    return create_agent(
        model=create_configured_model(),
        tools=[load_sql_skill, write_sql_query],
        state_schema=SkillAgentState,
        system_prompt=(
            "你是一个严格的 SQL 分析助手。\n"
            f"支持的技能只有：{', '.join(SUPPORTED_SKILLS)}。\n"
            "必须先调用 load_sql_skill 加载对应技能，再调用 write_sql_query 验证 SQL。"
            "不要在未验证前把 SQL 当作最终答案。"
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Skills 案例：强约束 SQL")
    parser.add_argument(
        "--request",
        default="生成一个 SQL，统计上个月每周已支付订单的销售额趋势。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = create_constrained_sql_agent()
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": args.request}],
            "skills_loaded": [],
        }
    )
    print(result["messages"][-1].content)
    print(f"\n已加载技能: {result.get('skills_loaded', [])}")


if __name__ == "__main__":
    main()
