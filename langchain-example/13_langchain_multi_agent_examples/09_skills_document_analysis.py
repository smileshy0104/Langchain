"""
Skills：智能文档分析系统。

Supervisor 根据文档类型加载财务或法律分析技能，再提取 mock 文档文本，
最后给出结构化分析结果。
"""

from __future__ import annotations

import argparse
from typing import Any, Literal

from langchain.agents import create_agent
from langchain_core.tools import tool

from model_config import create_configured_model


ANALYSIS_SKILLS = {
    "financial": {
        "description": "财务文档分析",
        "prompts": (
            "重点提取收入、成本、毛利率、现金流、同比/环比变化和异常项。"
            "输出应包含关键指标、风险提示和后续核查问题。"
        ),
    },
    "legal": {
        "description": "法律文档分析",
        "prompts": (
            "重点识别合同主体、付款条款、违约责任、自动续约、数据合规和终止条款。"
            "输出应包含关键条款、潜在风险和建议修改点。"
        ),
    },
}

MOCK_DOCUMENTS = {
    "quarterly_report.txt": (
        "Q1 revenue was 12.5M CNY, up 18% YoY. Gross margin decreased from 62% to 55%. "
        "Operating cash flow was negative 1.2M CNY due to delayed enterprise payments. "
        "Cloud infrastructure cost increased 31%."
    ),
    "service_contract.txt": (
        "The agreement renews automatically for one year unless either party gives 60 days notice. "
        "Payment is due within 15 days. Late payment incurs 1.5% monthly interest. "
        "The vendor may process customer usage data for service improvement."
    ),
}


@tool
def load_analysis_skill(domain: Literal["financial", "legal"]) -> str:
    """Load a domain-specific document analysis skill."""

    skill = ANALYSIS_SKILLS[domain]
    return f"""技能: {skill['description']}

分析指南:
{skill['prompts']}
"""


@tool
def extract_text(document_path: str) -> str:
    """Extract text from a mock document path."""

    if document_path not in MOCK_DOCUMENTS:
        return f"错误：找不到文档 {document_path}。可用文档：{list(MOCK_DOCUMENTS.keys())}"
    return MOCK_DOCUMENTS[document_path]


def create_document_analysis_agent() -> Any:
    return create_agent(
        model=create_configured_model(),
        tools=[load_analysis_skill, extract_text],
        system_prompt=(
            "你是文档分析主管。"
            "根据用户指定的文档类型加载 financial 或 legal 分析技能，"
            "再使用 extract_text 提取文档内容。"
            "最终用中文输出：摘要、关键发现、风险、建议。"
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Skills 案例：智能文档分析系统")
    parser.add_argument(
        "--request",
        default="请分析 quarterly_report.txt，这是一份财务报告。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = create_document_analysis_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": args.request}]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
