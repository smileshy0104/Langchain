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
    # which 决定 choose_path 路由到 step_a 还是 step_b。
    which: Literal["a", "b"]
    # status 保存默认 handler 或自定义 handler 写入的处理结果。
    status: str
    # error 保存 NodeError 提供的错误上下文。
    error: str


# 使用 Command(goto=...) 动态选择下一步节点。
def choose_path(state: WorkflowState) -> Command[Literal["step_a", "step_b"]]:
    return Command(goto="step_a" if state["which"] == "a" else "step_b")


# step_a 故意失败；它没有节点级 handler，因此会使用 graph defaults。
def step_a(state: WorkflowState) -> dict:
    raise ConnectionError("step_a failed")


# step_b 也故意失败；它会用 add_node 上显式传入的自定义 handler。
def step_b(state: WorkflowState) -> dict:
    raise ConnectionError("step_b failed")


# 默认错误处理器：会被没有覆盖配置的节点使用。
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


# step_b 的节点级错误处理器：优先级高于 set_node_defaults 中的默认配置。
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


# 所有失败路径最终都进入 finalize，方便对比 handler 写入的 status。
def finalize(state: WorkflowState) -> dict:
    return {"status": f"final:{state['status']}"}


# 构建 graph，并演示 graph defaults 与节点级配置的优先级。
def build_graph():
    builder = (
        StateGraph(WorkflowState)
        .set_node_defaults(
            # 默认 retry_policy 适用于后续未单独覆盖 retry_policy 的节点。
            retry_policy=RetryPolicy(
                max_attempts=1,  # 不重试，只执行一次；失败后直接进入 error_handler。
                retry_on=ConnectionError,
            ),
            # 默认 error_handler 适用于 step_a 等没有节点级 handler 的节点。
            error_handler=default_error_handler,
        )
        .add_node("choose_path", choose_path)
        .add_node("step_a", step_a)
        # 节点级 error_handler 会覆盖 graph defaults 中的 error_handler。
        .add_node("step_b", step_b, error_handler=custom_error_handler)
        .add_node("finalize", finalize)
    )

    builder.add_edge(START, "choose_path")
    builder.add_edge("step_a", "finalize")
    builder.add_edge("step_b", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=InMemorySaver())


# 主函数分别走 step_a 和 step_b，观察不同 handler 的输出。
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
