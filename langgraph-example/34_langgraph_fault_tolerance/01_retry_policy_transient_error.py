"""
案例 1：RetryPolicy 处理临时错误

目标：
- 节点第一次抛出 ConnectionError。
- RetryPolicy 捕获后自动重试。
- 第二次执行成功。

对应文档概念：
- Retries
- RetryPolicy 参数
- Inspect Retry State
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy


class ApiState(TypedDict):
    result: str
    attempts_seen: int


def flaky_api_node(state: ApiState, runtime: Runtime) -> dict:
    attempt = runtime.execution_info.node_attempt
    print(f"flaky_api_node attempt={attempt}")

    if attempt == 1:
        raise ConnectionError("temporary network failure")

    return {
        "result": "api call succeeded after retry",
        "attempts_seen": attempt,
    }


def build_graph():
    builder = StateGraph(ApiState)
    builder.add_node(
        "call_api",
        flaky_api_node,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_interval=0.1,
            retry_on=ConnectionError,
            jitter=False,
        ),
    )
    builder.add_edge(START, "call_api")
    builder.add_edge("call_api", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "retry-demo"}}

    result = graph.invoke({"result": "", "attempts_seen": 0}, config)
    print(result)


if __name__ == "__main__":
    main()
