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
    return builder.compile(checkpointer=InMemorySaver())


def run_with_durability(mode: Literal["sync", "async", "exit"]) -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": f"durability-{mode}"}}

    result = graph.invoke(
        {"log": [], "result": ""},
        config,
        durability=mode,
    )

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
