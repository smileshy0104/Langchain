"""
案例 6：Durability Modes

目标：
- 演示 invoke 时传入 durability 参数。
- 对比 sync / async / exit 的使用形式。

对应文档概念：
- Durability Modes

注意：
- 这个案例展示 API 使用方式。
- InMemorySaver 无法体现数据库写入时机差异；生产中差异主要体现在持久化后端。
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class JobState(TypedDict):
    # log 追加写入，方便观察节点执行记录；result 则保持普通覆盖语义。
    log: Annotated[list[str], operator.add]
    result: str


def work_node(state: JobState) -> dict:
    return {
        "log": ["work_node completed"],
        "result": "ok",
    }


def build_graph():
    builder = StateGraph(JobState)
    builder.add_node("work", work_node)
    builder.add_edge(START, "work")
    builder.add_edge("work", END)
    # durability 参数只有在 graph 配置了 checkpointer 时才有意义。
    return builder.compile(checkpointer=InMemorySaver())


def run_with_durability(mode: Literal["sync", "async", "exit"]) -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": f"durability-{mode}"}}

    # durability 控制 checkpoint 写入时机：
    # sync 更稳，async 更快，exit 写入最少但中途崩溃恢复能力最弱。
    result = graph.invoke(
        {"log": [], "result": ""},
        config,
        durability=mode,
    )

    # InMemorySaver 能展示 API 调用形式，但不能体现数据库落盘时机的差异。
    snapshot = graph.get_state(config)
    print(f"\nmode={mode}")
    print("result:", result)
    print("checkpoint_id:", snapshot.config["configurable"]["checkpoint_id"])


def main() -> None:
    print("sync: 每个 step 同步保存 checkpoint，恢复保障更强。")
    print("async: 异步保存 checkpoint，吞吐更好，但极端崩溃可能丢最近写入。")
    print("exit: graph 退出时保存，性能最好，但中途崩溃恢复能力最弱。")

    for mode in ("sync", "async", "exit"):
        run_with_durability(mode)


if __name__ == "__main__":
    main()
