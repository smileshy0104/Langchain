"""
案例 7：Manage checkpoints

目标：
- 查看当前 thread state。
- 查看 checkpoint history。
- 直接通过 checkpointer 列出 checkpoint tuples。
- 删除某个 thread 的全部 checkpoints。

对应文档概念：
- Manage Checkpoints
- View Thread State
- View Thread History
- Delete Thread Checkpoints
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class CounterState(TypedDict):
    # log 使用 reducer 累加每次 increment 的执行记录。
    log: Annotated[list[str], operator.add]
    # count 是普通字段；每次节点返回的新 count 会覆盖旧值。
    count: int


def increment(state: CounterState) -> dict:
    """简单计数节点：读取当前 count，加 1 后写回。"""

    next_count = state.get("count", 0) + 1
    return {"count": next_count, "log": [f"increment -> {next_count}"]}


def build_graph():
    """构建图并把 checkpointer 一起返回，便于 main 中直接管理 checkpoints。"""

    # 单独保存 checkpointer 引用，后面可以调用 list/delete_thread 等管理接口。
    checkpointer = InMemorySaver()
    builder = StateGraph(CounterState)
    builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    return builder.compile(checkpointer=checkpointer), checkpointer


def main() -> None:
    graph, checkpointer = build_graph()
    config = {"configurable": {"thread_id": "checkpoint-admin"}}

    # 同一个 thread_id 执行两次，会产生多组 checkpoint 历史。
    graph.invoke({"log": [], "count": 0}, config)
    graph.invoke({"log": [], "count": 10}, config)

    # get_state 查看当前 thread 的最新状态快照。
    snapshot = graph.get_state(config)
    print("current values:", snapshot.values)
    print("current next:", snapshot.next)

    print("\ngraph.get_state_history:")
    # graph.get_state_history 是图级 API，返回带 values/next/metadata/config 的 StateSnapshot。
    for item in graph.get_state_history(config):
        checkpoint_id = item.config["configurable"]["checkpoint_id"]
        print("-", checkpoint_id, item.values)

    print("\ncheckpointer.list:")
    # checkpointer.list 是更底层的 checkpoint API，返回 checkpoint tuple。
    for checkpoint_tuple in checkpointer.list(config):
        checkpoint_id = checkpoint_tuple.config["configurable"]["checkpoint_id"]
        print("-", checkpoint_id)

    # 删除某个 thread_id 下的全部 checkpoints；之后该 thread 的短期记忆被清空。
    checkpointer.delete_thread("checkpoint-admin")
    remaining = list(checkpointer.list(config))
    print("\nafter delete_thread:", remaining)


if __name__ == "__main__":
    main()
