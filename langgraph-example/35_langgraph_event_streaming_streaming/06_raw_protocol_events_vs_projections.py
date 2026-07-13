"""
案例 6：Raw protocol events vs typed projections

目标：
- 直接遍历 stream_events()，观察 event["method"] / event["params"]。
- 再用 typed projections 读取相同 graph 的 values/output。
- 理解 projection 层如何减少 raw event 分支解析。

对应文档概念：
- Raw Protocol Events
- Typed Projection 层是什么
- Event Streaming projections
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class PipelineState(TypedDict):
    topic: str
    events: Annotated[list[str], operator.add]
    done: bool


def step_one(state: PipelineState) -> dict:
    return {"events": [f"step_one:{state['topic']}"]}


def step_two(state: PipelineState) -> dict:
    return {"events": ["step_two:done"], "done": True}


def build_graph():
    builder = StateGraph(PipelineState)
    builder.add_node("step_one", step_one)
    builder.add_node("step_two", step_two)
    builder.add_edge(START, "step_one")
    builder.add_edge("step_one", "step_two")
    builder.add_edge("step_two", END)
    return builder.compile()


def main() -> None:
    graph = build_graph()
    inputs = {"topic": "projection layer", "events": [], "done": False}

    print("Raw protocol events:")
    raw_stream = graph.stream_events(inputs, version="v3")
    for event in raw_stream:
        method = event["method"]
        namespace = event["params"].get("namespace")
        data = event["params"].get("data")
        print(f"method={method}, namespace={namespace}, data={data}")

    print("\nTyped projections:")
    projected_stream = graph.stream_events(inputs, version="v3")
    for snapshot in projected_stream.values:
        print("value:", snapshot)
    print("output:", projected_stream.output)


if __name__ == "__main__":
    main()
