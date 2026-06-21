"""
实战案例 2：智能文档分析系统。

Supervisor + Skills 组合：主管 Agent 先加载财务/法律分析技能，
再提取 mock 文档内容，并按技能指南输出分析结论。
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
            "提取收入、成本、毛利率、现金流、应收账款、同比/环比变化。"
            "输出必须包含：摘要、关键指标、风险、建议、待核查问题。"
        ),
    },
    "legal": {
        "description": "法律文档分析",
        "prompts": (
            "识别合同主体、付款条款、违约责任、自动续约、数据处理、终止条款。"
            "输出必须包含：关键条款、风险等级、建议修改点、待确认问题。"
        ),
    },
}

MOCK_DOCUMENTS = {
    "quarterly_report.txt": (
        "Q1 revenue was 12.5M CNY, up 18% YoY. Gross margin decreased from 62% to 55%. "
        "Operating cash flow was negative 1.2M CNY due to delayed enterprise payments. "
        "Cloud infrastructure cost increased 31% and accounts receivable days increased to 74."
    ),
    "service_contract.txt": (
        "The agreement renews automatically for one year unless either party gives 60 days notice. "
        "Payment is due within 15 days. Late payment incurs 1.5% monthly interest. "
        "The vendor may process customer usage data for service improvement. "
        "Liability is capped at fees paid in the previous three months."
    ),
}


@tool
def load_analysis_skill(domain: Literal["financial", "legal"]) -> str:
    """Load a domain-specific analysis skill."""

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


@tool
def extract_financial_data(document_text: str) -> str:
    """Extract key financial facts from document text."""

    return (
        "提取的财务数据：收入 12.5M CNY，同比增长 18%；"
        "毛利率 55%，前期 62%；经营现金流 -1.2M CNY；"
        "云成本增长 31%；应收账款天数 74。"
    )


@tool
def calculate_ratios(revenue: float, gross_margin: float, operating_cash_flow: float) -> str:
    """Calculate simple financial ratios."""

    return (
        f"财务比率：毛利金额约 {revenue * gross_margin:.2f}M；"
        f"经营现金流/收入约 {operating_cash_flow / revenue:.2%}。"
    )


@tool
def extract_clauses(document_text: str) -> str:
    """Extract key legal clauses from document text."""

    return (
        "提取的法律条款：自动续约一年，需提前 60 天通知；"
        "付款期限 15 天；逾期月利息 1.5%；供应商可处理使用数据；"
        "责任上限为前三个月已付费用。"
    )


@tool
def identify_risks(clauses: str) -> str:
    """Identify legal risks from extracted clauses."""

    return (
        "风险识别：自动续约通知期较长；数据处理目的较宽泛；"
        "责任上限偏低；逾期利息需要确认是否符合适用法律。"
    )


def create_document_analysis_supervisor() -> Any:
    return create_agent(
        model=create_configured_model(),
        tools=[
            load_analysis_skill,
            extract_text,
            extract_financial_data,
            calculate_ratios,
            extract_clauses,
            identify_risks,
        ],
        system_prompt=(
            "你是文档分析主管。"
            "必须先根据文档类型调用 load_analysis_skill 加载相应技能，"
            "再调用 extract_text 提取文档内容。"
            "财务文档可继续调用 extract_financial_data 和 calculate_ratios；"
            "法律文档可继续调用 extract_clauses 和 identify_risks。"
            "最终用中文输出完整分析。"
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实战案例 2：智能文档分析系统")
    parser.add_argument(
        "--request",
        default="请分析 quarterly_report.txt，这是一份财务报告。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    supervisor = create_document_analysis_supervisor()
    result = supervisor.invoke({"messages": [{"role": "user", "content": args.request}]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
