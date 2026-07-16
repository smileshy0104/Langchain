"""
案例 3：Async TimeoutPolicy

目标：
- async 节点配置 TimeoutPolicy(run_timeout=...)。
- 第一次 attempt 故意 sleep 超时。
- RetryPolicy 让第二次 attempt 成功。

对应文档概念：
- Timeouts
- Run Timeout
- NodeTimeoutError

注意：
- 文档说明 node timeouts only apply to async nodes。
"""

from __future__ import annotations

import asyncio
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, TimeoutPolicy


class TimeoutState(TypedDict):
    # result 保存最终完成结果。
    result: str
    # attempts_seen 用于展示第二次 attempt 才成功。
    attempts_seen: int


# async 节点用于演示 TimeoutPolicy；节点级 timeout 只作用于异步节点。
async def slow_then_fast_node(state: TimeoutState, runtime: Runtime) -> dict:
    # 当前节点执行尝试次数；超时后重试会进入下一次 attempt。
    attempt = runtime.execution_info.node_attempt
    print(f"slow_then_fast_node attempt={attempt}")

    if attempt == 1:
        # 第一次睡眠 0.5 秒，超过 run_timeout=0.1，会触发节点超时。
        await asyncio.sleep(0.5)
    else:
        # 第二次睡眠很短，在超时限制内完成。
        await asyncio.sleep(0.01)

    return {
        "result": "completed before timeout",
        "attempts_seen": attempt,
    }


# 构建同时带 TimeoutPolicy 和 RetryPolicy 的异步 graph。
def build_graph():
    builder = StateGraph(TimeoutState)
    builder.add_node(
        "slow_then_fast",
        slow_then_fast_node,
        # 单次节点运行超过 0.1 秒会被取消并视为失败。
        timeout=TimeoutPolicy(run_timeout=0.1),
        # 超时失败可由 RetryPolicy 触发重试；第二次会成功。
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_interval=0.1,
            jitter=False,
        ),
    )
    builder.add_edge(START, "slow_then_fast")
    builder.add_edge("slow_then_fast", END)
    return builder.compile(checkpointer=InMemorySaver())


# 主函数使用 ainvoke 运行异步 graph。
async def main() -> None:
    graph = build_graph()
    result = await graph.ainvoke(
        {"result": "", "attempts_seen": 0},
        {"configurable": {"thread_id": "timeout-demo"}},
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
