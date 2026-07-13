"""
案例 2：Replay from checkpoint

目标：
- 找到 write 节点执行前的 checkpoint。
- 使用 graph.invoke(None, before_write.config) replay。
- 观察 checkpoint 前的 plan 不重跑，write 会重跑。

对应文档概念：
- Replay
- Replay re-executes nodes
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


COUNTERS = {"plan": 0, "write": 0}


class DraftState(TypedDict):
    log: Annotated[list[str], operator.add]
    topic: str
    draft: str


def plan(state: DraftState) -> dict:
    COUNTERS["plan"] += 1
    return {"log": [f"plan run #{COUNTERS['plan']}"]}


def write(state: DraftState) -> dict:
    COUNTERS["write"] += 1
    return {
        "draft": f"draft #{COUNTERS['write']} about {state['topic']}",
        "log": [f"write run #{COUNTERS['write']}"],
    }


def build_graph():
    builder = StateGraph(DraftState)
    builder.add_node("plan", plan)
    builder.add_node("write", write)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "write")
    builder.add_edge("write", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "replay-demo"}}

    first = graph.invoke({"log": [], "topic": "checkpoints", "draft": ""}, config)
    print("first:", first)
    print("counters:", COUNTERS)

    history = list(graph.get_state_history(config))
    before_write = next(s for s in history if s.next == ("write",))

    replay = graph.invoke(None, before_write.config)
    print("\nreplay:", replay)
    print("counters:", COUNTERS)
    print("plan did not run again; write did.")


if __name__ == "__main__":
    main()
