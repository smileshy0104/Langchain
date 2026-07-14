"""
案例 3：Replay

目标：
- 从 get_state_history 中找到某个节点执行前的 checkpoint。
- 使用 graph.invoke(None, checkpoint.config) 从该 checkpoint replay。
- 观察 checkpoint 前的节点不会重跑，checkpoint 后的节点会重跑。

对应文档概念：
- 查找特定 Checkpoint
- Replay
- state.next
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


RUN_COUNTER = {"prepare": 0, "write": 0}


class DraftState(TypedDict):
    # 使用追加 reducer 可以直观看到 replay 后哪些节点重新执行过。
    log: Annotated[list[str], operator.add]
    topic: str
    draft: str


def prepare_topic(state: DraftState) -> dict:
    # 全局计数器只用于演示 replay 行为：如果节点重跑，计数会增加。
    RUN_COUNTER["prepare"] += 1
    topic = state.get("topic") or "checkpointers"
    return {
        "topic": topic,
        "log": [f"prepare_topic run #{RUN_COUNTER['prepare']}"],
    }


def write_draft(state: DraftState) -> dict:
    RUN_COUNTER["write"] += 1
    return {
        "draft": f"Draft #{RUN_COUNTER['write']} about {state['topic']}",
        "log": [f"write_draft run #{RUN_COUNTER['write']}"],
    }


def build_graph():
    builder = StateGraph(DraftState)
    builder.add_node("prepare_topic", prepare_topic)
    builder.add_node("write_draft", write_draft)
    builder.add_edge(START, "prepare_topic")
    builder.add_edge("prepare_topic", "write_draft")
    builder.add_edge("write_draft", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "replay-thread"}}

    first = graph.invoke({"log": [], "topic": "Persistence", "draft": ""}, config)
    print("第一次运行:", first)
    print("运行计数:", RUN_COUNTER)

    history = list(graph.get_state_history(config))
    # state.next 表示“从这个 checkpoint 继续时下一个要执行的节点”。
    # 这里筛出 write_draft 执行前的 checkpoint。
    before_write = next(
        snapshot
        for snapshot in history
        if snapshot.next == ("write_draft",)
    )

    # invoke(None, checkpoint.config) 表示从历史 checkpoint replay；
    # checkpoint 之前的节点不重跑，checkpoint 之后的节点会重新执行。
    replay = graph.invoke(None, before_write.config)
    print("\nReplay from checkpoint before write_draft:", replay)
    print("运行计数:", RUN_COUNTER)
    print("\n说明：prepare_topic 没有重跑，write_draft 重跑了。")


if __name__ == "__main__":
    main()
