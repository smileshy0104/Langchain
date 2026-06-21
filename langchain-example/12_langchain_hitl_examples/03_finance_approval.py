"""
案例 3：金融交易审批系统。

本示例展示对转账和发票审批工具做人工审核，并在应用层增加
大额转账二级审批校验。
"""

from __future__ import annotations

import argparse
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import ToolCall
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from hitl_utils import build_decisions, get_interrupts, get_last_message_text, print_interrupt
from hitl_utils import resume_with_decisions
from model_config import create_configured_model


@tool
def transfer_funds(
    from_account: str,
    to_account: str,
    amount: float,
    currency: str = "USD",
) -> str:
    """Transfer funds after human approval."""

    if amount > 100000:
        return "错误：单笔转账限额为 100,000。"
    return f"已从 {from_account} 转账 {amount:.2f} {currency} 到 {to_account}。"


@tool
def approve_invoice(invoice_id: str, approval_amount: float, notes: str = "") -> str:
    """Approve an invoice after human approval."""

    notes_text = f"，备注：{notes}" if notes else ""
    return f"已审批发票 {invoice_id}，金额：{approval_amount:.2f}{notes_text}。"


def describe_transfer_review(tool_call: ToolCall, state: Any, runtime: Any) -> str:
    args = tool_call["args"]
    return (
        "转账审批:\n"
        f"从: {args['from_account']}\n"
        f"到: {args['to_account']}\n"
        f"金额: {args['amount']} {args.get('currency', 'USD')}\n"
        "注意：该操作执行后不可撤销。"
    )


def describe_invoice_review(tool_call: ToolCall, state: Any, runtime: Any) -> str:
    args = tool_call["args"]
    return (
        "发票审批:\n"
        f"发票号: {args['invoice_id']}\n"
        f"审批金额: {args['approval_amount']}"
    )


def create_finance_agent() -> Any:
    return create_agent(
        model=create_configured_model(),
        tools=[transfer_funds, approve_invoice],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "transfer_funds": {
                        "allowed_decisions": ["approve", "reject"],
                        "description": describe_transfer_review,
                    },
                    "approve_invoice": {
                        "allowed_decisions": ["approve", "edit", "reject"],
                        "description": describe_invoice_review,
                    },
                }
            )
        ],
        checkpointer=InMemorySaver(),
    )


def has_high_risk_transfer(interrupt: Any) -> bool:
    for action in interrupt.value["action_requests"]:
        if action["name"] != "transfer_funds":
            continue
        args = action["args"]
        amount = float(args.get("amount", 0))
        to_account = str(args.get("to_account", ""))
        if amount > 50000 or to_account.startswith("EXTERNAL"):
            return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HITL 案例 3：金融交易审批系统")
    parser.add_argument(
        "--request",
        default="请从 ACCT-001 向 EXTERNAL-998 转账 75000 美元。",
    )
    parser.add_argument(
        "--decision",
        choices=["approve", "edit", "reject"],
        default="approve",
        help="人工审批决策。transfer_funds 不允许 edit。",
    )
    parser.add_argument("--edited-amount", type=float, help="decision=edit 时替换发票审批金额。")
    parser.add_argument(
        "--second-approver",
        help="大额或外部账户转账批准时需要二级审批人 ID。",
    )
    parser.add_argument("--thread-id", default="hitl-finance-session-001")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = create_finance_agent()
    config = {"configurable": {"thread_id": args.thread_id}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": args.request}]},
        config=config,
    )

    interrupts = get_interrupts(result)
    while interrupts:
        interrupt = interrupts[0]
        print_interrupt(interrupt)

        decision = args.decision
        if decision == "approve" and has_high_risk_transfer(interrupt) and not args.second_approver:
            print("\n检测到高风险转账：金额超过 50,000 或目标为外部账户。")
            print("未提供 --second-approver，本示例自动改为 reject。")
            decision = "reject"

        edited_args = {}
        if args.edited_amount is not None:
            edited_args["approval_amount"] = args.edited_amount

        decisions = build_decisions(
            interrupt,
            decision,
            edited_args=edited_args,
            reject_message="人工审核拒绝该金融交易。",
        )
        print("\n本次提交的人工决策:")
        print(decisions)
        if args.second_approver:
            print(f"二级审批人: {args.second_approver}")

        result = resume_with_decisions(agent, config, decisions)
        interrupts = get_interrupts(result)

    print("\n最终回复:")
    print(get_last_message_text(result))


if __name__ == "__main__":
    main()
