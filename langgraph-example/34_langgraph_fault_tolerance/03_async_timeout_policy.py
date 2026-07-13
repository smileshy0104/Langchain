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
    result: str
    attempts_seen: int


async def slow_then_fast_node(state: TimeoutState, runtime: Runtime) -> dict:
    attempt = runtime.execution_info.node_attempt
    print(f"slow_then_fast_node attempt={attempt}")

    if attempt == 1:
        await asyncio.sleep(0.5)
    else:
        await asyncio.sleep(0.01)

    return {
        "result": "completed before timeout",
        "attempts_seen": attempt,
    }


def build_graph():
    builder = StateGraph(TimeoutState)
    builder.add_node(
        "slow_then_fast",
        slow_then_fast_node,
        timeout=TimeoutPolicy(run_timeout=0.1),
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_interval=0.1,
            jitter=False,
        ),
    )
    builder.add_edge(START, "slow_then_fast")
    builder.add_edge("slow_then_fast", END)
    return builder.compile(checkpointer=InMemorySaver())


async def main() -> None:
    graph = build_graph()
    result = await graph.ainvoke(
        {"result": "", "attempts_seen": 0},
        {"configurable": {"thread_id": "timeout-demo"}},
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
