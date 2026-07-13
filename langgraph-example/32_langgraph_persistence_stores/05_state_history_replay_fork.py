"""
案例 5：Checkpoint history、Replay 与 Fork

目标：
- 用 get_state_history 查看 thread 的历史 checkpoints。
- 用 state.next 找到某个节点执行前的 checkpoint。
- 用 invoke(None, checkpoint.config) 从旧 checkpoint replay。
- 用 update_state 基于旧 checkpoint 创建 fork。

对应文档概念：
- StateSnapshot
- checkpoint_id
- state.next
- time travel
- update_state
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class DraftState(TypedDict):
    topic: NotRequired[str]
    draft: NotRequired[str]


def choose_topic(state: DraftState) -> dict:
    return {"topic": state.get("topic", "LangGraph Persistence")}


def write_draft(state: DraftState) -> dict:
    topic = state["topic"]
    return {"draft": f"这是一段关于 {topic} 的草稿。"}


def build_graph():
    builder = StateGraph(DraftState)
    builder.add_node("choose_topic", choose_topic)
    builder.add_node("write_draft", write_draft)
    builder.add_edge(START, "choose_topic")
    builder.add_edge("choose_topic", "write_draft")
    builder.add_edge("write_draft", END)
    return builder.compile(checkpointer=InMemorySaver())


def print_history(graph, config) -> None:
    print("\nCheckpoint history：")
    for snapshot in graph.get_state_history(config):
        checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
        print(f"- next={snapshot.next}, checkpoint_id={checkpoint_id}")


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "draft-thread"}}

    print("第一次运行 graph")
    result = graph.invoke({}, config)
    print(result)

    history = list(graph.get_state_history(config))
    print_history(graph, config)

    before_write = next(
        snapshot
        for snapshot in history
        if snapshot.next == ("write_draft",)
    )

    print("\nReplay：从 write_draft 前的 checkpoint 继续")
    replay_result = graph.invoke(None, before_write.config)
    print(replay_result)

    print("\nFork：从 write_draft 前的 checkpoint 改写 topic，再继续")
    fork_config = graph.update_state(
        before_write.config,
        values={"topic": "LangGraph Stores"},
        as_node="choose_topic",
    )
    fork_result = graph.invoke(None, fork_config)
    print(fork_result)

    print("\n说明：")
    print("- Replay 不会重跑 choose_topic，但会重跑 write_draft。")
    print("- Fork 不会删除原历史，而是从旧 checkpoint 创建一个新分支。")


if __name__ == "__main__":
    main()
