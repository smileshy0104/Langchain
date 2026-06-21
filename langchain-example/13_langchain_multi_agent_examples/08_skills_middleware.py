"""
Skills：自定义中间件注入技能目录。

中间件把可用技能清单动态追加到 system message，Agent 仍然通过
load_skill 按需加载完整技能内容。
"""

from __future__ import annotations

import argparse
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from model_config import create_configured_model


SKILLS = {
    "sales_analytics": {
        "name": "销售分析",
        "description": "分析销售趋势、客群贡献和收入异常。",
        "details": "使用 orders、customers、revenues 表；优先输出 SQL 和指标解释。",
    },
    "inventory_management": {
        "name": "库存管理",
        "description": "识别低库存商品、补货优先级和供应商交期风险。",
        "details": "使用 products、inventory、suppliers 表；低库存以 reorder_point 为阈值。",
    },
    "customer_success": {
        "name": "客户成功",
        "description": "分析续费风险、支持工单和客户健康度。",
        "details": "使用 accounts、tickets、usage_events 表；重点关注活跃度下降和高优先级工单。",
    },
}


@tool
def load_skill(skill_name: str) -> str:
    """Load the full details of a named skill."""

    skill = SKILLS.get(skill_name)
    if not skill:
        return f"错误：未知技能 '{skill_name}'。可用技能：{list(SKILLS.keys())}"

    return (
        f"技能: {skill['name']}\n\n"
        f"描述: {skill['description']}\n\n"
        f"详细说明: {skill['details']}"
    )


class SkillCatalogMiddleware(AgentMiddleware):
    """Inject a compact skill catalog before every model call."""

    tools = [load_skill]

    def __init__(self, skills: dict[str, dict[str, str]]) -> None:
        super().__init__()
        self.skills_prompt = "\n".join(
            f"- {name}: {info['description']}" for name, info in skills.items()
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Any,
    ) -> ModelResponse:
        base_content = ""
        if request.system_message:
            base_content = str(request.system_message.content)

        addendum = (
            "\n\n## 可用技能\n"
            f"{self.skills_prompt}\n\n"
            "处理具体领域问题前，使用 load_skill 获取完整技能说明。"
        )
        modified_request = request.override(
            system_message=SystemMessage(content=base_content + addendum)
        )
        return handler(modified_request)


def create_middleware_skill_agent() -> Any:
    return create_agent(
        model=create_configured_model(),
        tools=[],
        middleware=[SkillCatalogMiddleware(SKILLS)],
        system_prompt="你是企业运营分析助手。根据用户问题选择并加载合适技能。",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Skills 案例：中间件注入技能目录")
    parser.add_argument(
        "--request",
        default="帮我判断哪些客户可能有续费风险，并说明需要什么数据。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = create_middleware_skill_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": args.request}]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
