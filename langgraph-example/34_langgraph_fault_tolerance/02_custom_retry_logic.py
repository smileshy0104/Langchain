"""
案例 2：Custom Retry Logic

目标：
- 使用 retry_on callable 自定义哪些异常可以重试。
- TransientServiceError 可以重试。
- BusinessRuleError 不应该重试。

对应文档概念：
- Custom Retry Logic
- default_retry_on
- 不要重试确定性业务错误
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, default_retry_on


class TransientServiceError(Exception):
    """临时服务错误，适合 retry。"""


class BusinessRuleError(Exception):
    """确定性业务错误，不应该 retry。"""


class ServiceState(TypedDict):
    # mode 控制示例走“临时错误”还是“业务错误”分支。
    mode: Literal["transient", "business"]
    # result 保存成功恢复后的结果。
    result: str


# 自定义 retry_on：返回 True 表示该异常可重试，False 表示直接失败。
def custom_retry_on(exc: BaseException) -> bool:
    # 业务规则错误通常是确定性的，重试不会改变结果，因此不要重试。
    if isinstance(exc, BusinessRuleError):
        return False

    # 临时服务错误可能是 503、短暂断连等，适合重试。
    if isinstance(exc, TransientServiceError):
        return True

    # 对其它异常沿用 LangGraph 的默认重试判断逻辑。
    return default_retry_on(exc)


# 模拟服务调用：根据 mode 抛出不同类型错误。
def service_node(state: ServiceState, runtime: Runtime) -> dict:
    # node_attempt 可用来区分首次执行和重试执行。
    attempt = runtime.execution_info.node_attempt
    print(f"mode={state['mode']}, attempt={attempt}")

    # business 模式每次都会抛业务错误，custom_retry_on 会阻止重试。
    if state["mode"] == "business":
        raise BusinessRuleError("invalid order amount")

    # transient 模式第一次失败，第二次成功，用于验证 retry 生效。
    if attempt == 1:
        raise TransientServiceError("temporary 503")

    return {"result": "transient failure recovered"}


# 构建 graph，并把 custom_retry_on 注入 RetryPolicy。
def build_graph():
    builder = StateGraph(ServiceState)
    builder.add_node(
        "service",
        service_node,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_interval=0.1,
            retry_on=custom_retry_on,  # callable 可以根据异常类型精细控制重试。
            jitter=False,
        ),
    )
    builder.add_edge(START, "service")
    builder.add_edge("service", END)
    return builder.compile(checkpointer=InMemorySaver())


# 主函数分别演示“可重试临时错误”和“不可重试业务错误”。
def main() -> None:
    graph = build_graph()

    print("Transient error: should retry and succeed")
    ok = graph.invoke(
        {"mode": "transient", "result": ""},
        {"configurable": {"thread_id": "custom-retry-ok"}},
    )
    print(ok)

    print("\nBusiness error: should not retry")
    try:
        graph.invoke(
            {"mode": "business", "result": ""},
            {"configurable": {"thread_id": "custom-retry-fail"}},
        )
    except BusinessRuleError as exc:
        # 由于 custom_retry_on 返回 False，这里应只执行一次并直接抛出。
        print(f"caught expected business error: {exc}")


if __name__ == "__main__":
    main()
