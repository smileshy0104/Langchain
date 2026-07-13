"""
案例 4：Error handler + Command 路由

目标：
- charge_payment 节点重试耗尽后进入 error_handler。
- error_handler 读取 NodeError，写入补偿状态。
- 通过 Command(goto="finalize") 跳转到收尾节点。

对应文档概念：
- Error Handling
- NodeError
- Route with Command
- Saga compensation pattern
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy


class PaymentState(TypedDict):
    order_id: str
    status: str
    error: str


def reserve_inventory(state: PaymentState) -> dict:
    return {"status": f"inventory_reserved:{state['order_id']}"}


def charge_payment(state: PaymentState) -> dict:
    raise ConnectionError("payment gateway unavailable")


def payment_error_handler(
    state: PaymentState,
    error: NodeError,
) -> Command[Literal["finalize"]]:
    return Command(
        update={
            "status": "payment_failed_inventory_released",
            "error": f"{error.node}: {error.error}",
        },
        goto="finalize",
    )


def finalize(state: PaymentState) -> dict:
    return {"status": f"finalized:{state['status']}"}


def build_graph():
    builder = StateGraph(PaymentState)
    builder.add_node("reserve_inventory", reserve_inventory)
    builder.add_node(
        "charge_payment",
        charge_payment,
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_interval=0.1,
            retry_on=ConnectionError,
            jitter=False,
        ),
        error_handler=payment_error_handler,
    )
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "reserve_inventory")
    builder.add_edge("reserve_inventory", "charge_payment")
    builder.add_edge("charge_payment", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    result = graph.invoke(
        {"order_id": "order-001", "status": "created", "error": ""},
        {"configurable": {"thread_id": "payment-demo"}},
    )
    print(result)


if __name__ == "__main__":
    main()
