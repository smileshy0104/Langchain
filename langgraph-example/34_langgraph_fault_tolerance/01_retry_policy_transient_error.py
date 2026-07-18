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
    # result 保存 API 调用的最终结果。
    result: str
    # attempts_seen 记录节点最终看到的是第几次 attempt，便于确认是否发生了重试。
    attempts_seen: int


# 模拟一个“不稳定 API”：第一次调用失败，第二次调用成功。
def flaky_api_node(state: ApiState, runtime: Runtime) -> dict:
    # LangGraph 会在 Runtime.execution_info 中提供当前节点的执行上下文。
    # node_attempt 表示当前节点第几次尝试执行：首次为 1，重试时递增。
    attempt = runtime.execution_info.node_attempt # 默认首次指为1
    print(f"flaky_api_node attempt={attempt}")

    # 第一次 attempt 抛出临时网络错误；RetryPolicy 会判断该异常是否可重试。
    if attempt == 1:
        raise ConnectionError("temporary network failure")

    # 第二次 attempt 成功返回，状态会合并进 graph state 并写入 checkpoint。
    return {
        "result": "api call succeeded after retry",
        "attempts_seen": attempt,
    }


# 构建一个只有一个节点的 graph，重点演示节点级 retry_policy。
def build_graph():
    builder = StateGraph(ApiState)
    builder.add_node(
        "call_api",
        flaky_api_node,
        # RetryPolicy 配在节点上，只影响 call_api 这个节点。
        retry_policy=RetryPolicy(
            max_attempts=3,  # 最多尝试 3 次：1 次原始执行 + 最多 2 次重试。
            initial_interval=0.1,  # 第一次重试前等待 0.1 秒。
            retry_on=ConnectionError,  # 只有 ConnectionError 会触发重试。
            jitter=False,  # 关闭随机抖动，方便示例输出稳定。
        ),
    )
    builder.add_edge(START, "call_api")
    builder.add_edge("call_api", END)

    # checkpointer 保存成功后的 graph state；如果执行中断，也可用于恢复。
    return builder.compile(checkpointer=InMemorySaver())


# 主函数演示：第一次失败会被自动重试，最终 invoke 返回成功结果。
def main() -> None:
    graph = build_graph()

    # thread_id 用于标识本次运行所属的 checkpoint 线程。
    config = {"configurable": {"thread_id": "retry-demo"}}

    result = graph.invoke({"result": "", "attempts_seen": 0}, config)
    print(result)


if __name__ == "__main__":
    main()
