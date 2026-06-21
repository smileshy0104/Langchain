"""Small helpers for Human-in-the-Loop multi-agent examples."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Literal

from langgraph.types import Command


DecisionName = Literal["approve", "edit", "reject"]


def get_interrupts(result: dict[str, Any]) -> list[Any]:
    return list(result.get("__interrupt__", []))


def get_last_message_text(result: dict[str, Any]) -> str:
    messages = result.get("messages", [])
    if not messages:
        return ""
    return str(getattr(messages[-1], "content", messages[-1]))


def print_interrupt(interrupt: Any) -> None:
    value = interrupt.value
    print("\n" + "=" * 72)
    print("Supervisor 待人工审核")
    print("=" * 72)

    for index, action in enumerate(value["action_requests"], start=1):
        review_config = value["review_configs"][index - 1]
        print(f"\n[{index}] 子 Agent 工具: {action['name']}")
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
    edited_request: str | None = None,
    reject_message: str | None = None,
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for action in interrupt.value["action_requests"]:
        if decision == "approve":
            decisions.append({"type": "approve"})
        elif decision == "edit":
            updated_args = deepcopy(action["args"])
            if edited_request is not None:
                updated_args["request"] = edited_request
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
                    "message": reject_message or "人工审核拒绝调用该子 Agent。",
                }
            )
    return decisions


def resume_with_decisions(
    agent: Any,
    config: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    return agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config,
    )
