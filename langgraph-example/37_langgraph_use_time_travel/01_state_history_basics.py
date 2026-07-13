"""
案例 1：Checkpoint history 基础

目标：
- 运行一个多节点 graph。
- 使用 get_state_history(config) 查看历史 checkpoints。
- 理解 state.next 和 checkpoint_id。

对应文档概念：
- 核心依赖：Checkpoints
- get_state_history
- state.next
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class PipelineState(TypedDict):
    log: Annotated[list[str], operator.add]
    topic: str
    result: str


def plan(state: PipelineState) -> dict:
    return {"log": [f"plan:{state['topic']}"]}


def write(state: PipelineState) -> dict:
    return {"log": ["write"], "result": f"result for {state['topic']}"}


def build_graph():
    builder = StateGraph(PipelineState)
    builder.add_node("plan", plan)
    builder.add_node("write", write)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "write")
    builder.add_edge("write", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "history-basics"}}

    graph.invoke({"log": [], "topic": "time travel", "result": ""}, config)

    print("History is usually newest first:")
    for index, snapshot in enumerate(graph.get_state_history(config), start=1):
        checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
        print(
            f"{index}. next={snapshot.next}, "
            f"step={snapshot.metadata.get('step')}, "
            f"checkpoint_id={checkpoint_id}"
        )


if __name__ == "__main__":
    main()
