"""
案例 4：update_state 与 fork

目标：
- 从历史 checkpoint 创建一个修改过 state 的新 checkpoint。
- 用 fork_config 继续执行，探索另一条执行路径。
- 观察 update_state 不会删除原始历史。

对应文档概念：
- Update State
- metadata.source = "update"
- fork
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class DraftState(TypedDict):
    topic: str
    draft: str


def choose_topic(state: DraftState) -> dict:
    return {"topic": state.get("topic") or "Checkpointers"}


def write_draft(state: DraftState) -> dict:
    return {"draft": f"这是一段关于 {state['topic']} 的草稿。"}


def build_graph():
    builder = StateGraph(DraftState)
    builder.add_node("choose_topic", choose_topic)
    builder.add_node("write_draft", write_draft)
    builder.add_edge(START, "choose_topic")
    builder.add_edge("choose_topic", "write_draft")
    builder.add_edge("write_draft", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "fork-thread"}}

    original = graph.invoke({"topic": "Checkpointers", "draft": ""}, config)
    print("原始结果:", original)

    history = list(graph.get_state_history(config))
    before_write = next(
        snapshot
        for snapshot in history
        if snapshot.next == ("write_draft",)
    )

    fork_config = graph.update_state(
        before_write.config,
        values={"topic": "Checkpoint Fork"},
        as_node="choose_topic",
    )
    forked = graph.invoke(None, fork_config)
    print("\nFork 后结果:", forked)

    print("\n包含 update_state 的 checkpoints:")
    for snapshot in graph.get_state_history(config):
        if snapshot.metadata.get("source") == "update":
            print(snapshot.config)
            print(snapshot.values)


if __name__ == "__main__":
    main()
