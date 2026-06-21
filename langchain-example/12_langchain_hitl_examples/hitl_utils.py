"""Utilities shared by the Human-in-the-Loop examples."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Literal

from langgraph.types import Command


DecisionName = Literal["approve", "edit", "reject", "respond"]


def get_interrupts(result: dict[str, Any]) -> list[Any]:
    """Return LangGraph interrupts from a create_agent invoke result."""

    return list(result.get("__interrupt__", []))


def get_last_message_text(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    if not messages:
        return ""
    return str(getattr(messages[-1], "content", messages[-1]))


def print_interrupt(interrupt: Any) -> None:
    """Pretty-print the action requests and review policy."""

    value = interrupt.value
    print("\n" + "=" * 72)
    print("待人工审批")
    print("=" * 72)

    for index, action in enumerate(value["action_requests"], start=1):
        review_config = value["review_configs"][index - 1]
        print(f"\n[{index}] 工具: {action['name']}")
        print(f"允许决策: {', '.join(review_config['allowed_decisions'])}")
        description = action.get("description")
        if description:
            print(description)
        print("参数:")
        print(json.dumps(action["args"], ensure_ascii=False, indent=2))


def build_decisions(
    interrupt: Any,
    decision: DecisionName,
    *,
    edited_args: dict[str, Any] | None = None,
    reject_message: str | None = None,
    respond_message: str | None = None,
) -> list[dict[str, Any]]:
    """Build one decision per interrupted action.

    The examples intentionally use a single command-line decision for all pending
    actions so they are easy to run from a terminal.
    """

    decisions: list[dict[str, Any]] = []
    for action in interrupt.value["action_requests"]:
        if decision == "approve":
            decisions.append({"type": "approve"})
        elif decision == "edit":
            updated_args = deepcopy(action["args"])
            updated_args.update(edited_args or {})
            decisions.append(
                {
                    "type": "edit",
                    "edited_action": {
                        "name": action["name"],
                        "args": updated_args,
                    },
                }
            )
        elif decision == "reject":
            decisions.append(
                {
                    "type": "reject",
                    "message": reject_message or "人工审核拒绝执行该工具调用。",
                }
            )
        elif decision == "respond":
            decisions.append(
                {
                    "type": "respond",
                    "message": respond_message or "人工审核员已代替工具给出结果。",
                }
            )
    return decisions


def resume_with_decisions(
    agent: Any,
    config: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Resume a paused HITL agent with human decisions."""

    return agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config,
    )
