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
    # order_id 标识要处理的订单。
    order_id: str
    # status 保存订单流程状态，会被库存、支付、补偿和收尾节点逐步更新。
    status: str
    # error 保存失败节点和异常信息。
    error: str


# 第一步：预留库存，模拟 Saga 中已经完成的前置动作。
def reserve_inventory(state: PaymentState) -> dict:
    return {"status": f"inventory_reserved:{state['order_id']}"}


# 第二步：模拟支付网关不可用；该节点会先按 RetryPolicy 重试。
def charge_payment(state: PaymentState) -> dict:
    raise ConnectionError("payment gateway unavailable")


# 支付节点最终失败后进入该 error_handler，而不是让整个 graph 直接异常退出。
def payment_error_handler(
    state: PaymentState,
    error: NodeError,
) -> Command[Literal["finalize"]]:
    # NodeError 中包含失败节点名和原始异常；Command 可同时更新 state 并跳转。
    return Command(
        update={
            # 这里模拟补偿动作：支付失败后释放之前预留的库存。
            "status": "payment_failed_inventory_released",
            "error": f"{error.node}: {error.error}",
        },
        # 即使支付失败，也路由到 finalize 做统一收尾。
        goto="finalize",
    )


# 收尾节点：统一把流程状态标记为 finalized。
def finalize(state: PaymentState) -> dict:
    return {"status": f"finalized:{state['status']}"}


# 构建一个带重试和错误处理的支付流程 graph。
def build_graph():
    builder = StateGraph(PaymentState)
    builder.add_node("reserve_inventory", reserve_inventory)
    builder.add_node(
        "charge_payment",
        charge_payment,
        # 支付节点最多尝试 2 次；都失败后才会调用 error_handler。
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


# 主函数演示：支付失败不会中断整个流程，而是经过补偿后 finalize。
def main() -> None:
    graph = build_graph()
    result = graph.invoke(
        {"order_id": "order-001", "status": "created", "error": ""},
        {"configurable": {"thread_id": "payment-demo"}},
    )
    print(result)


if __name__ == "__main__":
    main()
