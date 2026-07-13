"""
案例 2：get_state 与 get_state_history

目标：
- 查看最新 StateSnapshot。
- 查看 thread 的 checkpoint history。
- 理解 StateSnapshot 中 values、next、config、metadata、parent_config。

对应文档概念：
- Get State
- StateSnapshot 字段
- Get State History
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class WorkflowState(TypedDict):
    log: Annotated[list[str], operator.add]
    topic: str
    draft: str


def prepare_topic(state: WorkflowState) -> dict:
    topic = state.get("topic") or "checkpointers"
    return {"topic": topic, "log": [f"prepare_topic: {topic}"]}


def write_draft(state: WorkflowState) -> dict:
    draft = f"Draft about {state['topic']}"
    return {"draft": draft, "log": [f"write_draft: {draft}"]}


def build_graph():
    builder = StateGraph(WorkflowState)
    builder.add_node("prepare_topic", prepare_topic)
    builder.add_node("write_draft", write_draft)
    builder.add_edge(START, "prepare_topic")
    builder.add_edge("prepare_topic", "write_draft")
    builder.add_edge("write_draft", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "history-thread"}}

    graph.invoke({"log": [], "topic": "LangGraph", "draft": ""}, config)

    latest = graph.get_state(config)
    print("最新 StateSnapshot:")
    print("values:", latest.values)
    print("next:", latest.next)
    print("config:", latest.config)
    print("metadata:", latest.metadata)
    print("parent_config:", latest.parent_config)

    print("\nHistory newest first:")
    for index, snapshot in enumerate(graph.get_state_history(config), start=1):
        checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
        print(
            f"{index}. step={snapshot.metadata.get('step')}, "
            f"next={snapshot.next}, checkpoint_id={checkpoint_id}"
        )


if __name__ == "__main__":
    main()
