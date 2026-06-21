"""
案例 4：带日志和审计的 HITL。

本示例在恢复 Agent 前将待审批工具、参数和人工决策写入 JSONL 审计日志。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from hitl_utils import build_decisions, get_interrupts, get_last_message_text, print_interrupt
from hitl_utils import resume_with_decisions
from model_config import create_configured_model


@tool
def sensitive_operation(operation: str, target: str, reason: str = "") -> str:
    """Run a sensitive operation after human approval."""

    reason_text = f"，原因：{reason}" if reason else ""
    return f"已执行敏感操作：{operation}，目标：{target}{reason_text}。"


class AuditLogger:
    """Append-only JSONL audit logger for HITL decisions."""

    def __init__(self, log_file: Path) -> None:
        self.log_file = log_file

    def log_decision(self, interrupt: Any, decisions: list[dict[str, Any]], user_id: str) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "user_id": user_id,
            "actions": [
                {
                    "tool": action["name"],
                    "arguments": action["args"],
                    "description": action.get("description", ""),
                }
                for action in interrupt.value["action_requests"]
            ],
            "decisions": decisions,
        }

        with self.log_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def create_audited_agent() -> Any:
    return create_agent(
        model=create_configured_model(),
        tools=[sensitive_operation],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "sensitive_operation": {
                        "allowed_decisions": ["approve", "edit", "reject", "respond"],
                        "description": "敏感操作审批：请确认该操作已获得授权并可审计追踪。",
                    }
                }
            )
        ],
        checkpointer=InMemorySaver(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HITL 案例 4：带日志和审计的 HITL")
    parser.add_argument(
        "--request",
        default="请执行敏感操作 rotate_api_key，目标 production/payment-service，原因是例行密钥轮换。",
    )
    parser.add_argument(
        "--decision",
        choices=["approve", "edit", "reject", "respond"],
        default="approve",
    )
    parser.add_argument("--edited-operation", help="decision=edit 时替换 operation。")
    parser.add_argument("--edited-target", help="decision=edit 时替换 target。")
    parser.add_argument("--respond-message", help="decision=respond 时由人工代替工具返回的信息。")
    parser.add_argument("--user-id", default="reviewer-001")
    parser.add_argument("--audit-log", default="logs/hitl_audit.jsonl")
    parser.add_argument("--thread-id", default="hitl-audit-session-001")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent = create_audited_agent()
    audit = AuditLogger(Path(args.audit_log))
    config = {"configurable": {"thread_id": args.thread_id}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": args.request}]},
        config=config,
    )

    interrupts = get_interrupts(result)
    while interrupts:
        interrupt = interrupts[0]
        print_interrupt(interrupt)

        edited_args = {}
        if args.edited_operation:
            edited_args["operation"] = args.edited_operation
        if args.edited_target:
            edited_args["target"] = args.edited_target

        decisions = build_decisions(
            interrupt,
            args.decision,
            edited_args=edited_args,
            reject_message="人工审核拒绝该敏感操作。",
            respond_message=args.respond_message,
        )
        audit.log_decision(interrupt, decisions, args.user_id)
        print(f"\n审计日志已写入: {Path(args.audit_log).resolve()}")
        print("本次提交的人工决策:")
        print(decisions)

        result = resume_with_decisions(agent, config, decisions)
        interrupts = get_interrupts(result)

    print("\n最终回复:")
    print(get_last_message_text(result))


if __name__ == "__main__":
    main()
