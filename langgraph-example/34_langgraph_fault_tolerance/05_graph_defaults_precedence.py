"""
案例 5：Graph defaults 与节点级覆盖

目标：
- 用 set_node_defaults 配置默认 retry_policy 和 error_handler。
- step_a 使用默认 handler。
- step_b 用节点级 error_handler 覆盖默认 handler。

对应文档概念：
- Graph Defaults
- Precedence
- Applicability
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy


class WorkflowState(TypedDict):
    which: Literal["a", "b"]
    status: str
    error: str


def choose_path(state: WorkflowState) -> Command[Literal["step_a", "step_b"]]:
    return Command(goto="step_a" if state["which"] == "a" else "step_b")


def step_a(state: WorkflowState) -> dict:
    raise ConnectionError("step_a failed")


def step_b(state: WorkflowState) -> dict:
    raise ConnectionError("step_b failed")


def default_error_handler(
    state: WorkflowState,
    error: NodeError,
) -> Command[Literal["finalize"]]:
    return Command(
        update={
            "status": "handled_by_default",
            "error": f"{error.node}: {error.error}",
        },
        goto="finalize",
    )


def custom_error_handler(
    state: WorkflowState,
    error: NodeError,
) -> Command[Literal["finalize"]]:
    return Command(
        update={
            "status": "handled_by_custom_step_b",
            "error": f"{error.node}: {error.error}",
        },
        goto="finalize",
    )


def finalize(state: WorkflowState) -> dict:
    return {"status": f"final:{state['status']}"}


def build_graph():
    builder = (
        StateGraph(WorkflowState)
        .set_node_defaults(
            retry_policy=RetryPolicy(
                max_attempts=1,
                retry_on=ConnectionError,
            ),
            error_handler=default_error_handler,
        )
        .add_node("choose_path", choose_path)
        .add_node("step_a", step_a)
        .add_node("step_b", step_b, error_handler=custom_error_handler)
        .add_node("finalize", finalize)
    )

    builder.add_edge(START, "choose_path")
    builder.add_edge("step_a", "finalize")
    builder.add_edge("step_b", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()

    result_a = graph.invoke(
        {"which": "a", "status": "start", "error": ""},
        {"configurable": {"thread_id": "defaults-a"}},
    )
    print("step_a uses default handler:", result_a)

    result_b = graph.invoke(
        {"which": "b", "status": "start", "error": ""},
        {"configurable": {"thread_id": "defaults-b"}},
    )
    print("step_b overrides handler:", result_b)


if __name__ == "__main__":
    main()
