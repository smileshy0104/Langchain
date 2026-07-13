"""
案例 1：Event Streaming typed projections

目标：
- 使用 graph.stream_events(..., version="v3")。
- 读取 stream.values 观察 state snapshots。
- 读取 stream.output 获取最终结果。
- 直接遍历 stream 查看 raw protocol events。

对应文档概念：
- Event Streaming
- Typed Projection 层
- stream.values
- stream.output
- raw events still available
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class ReportState(TypedDict):
    topic: str
    steps: Annotated[list[str], operator.add]
    result: str


def plan_node(state: ReportState) -> dict:
    return {"steps": [f"plan:{state['topic']}"]}


def write_node(state: ReportState) -> dict:
    return {
        "steps": ["write:draft"],
        "result": f"Report about {state['topic']}",
    }


def build_graph():
    builder = StateGraph(ReportState)
    builder.add_node("plan", plan_node)
    builder.add_node("write", write_node)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "write")
    builder.add_edge("write", END)
    return builder.compile()


def main() -> None:
    graph = build_graph()

    stream = graph.stream_events(
        {"topic": "typed projections", "steps": [], "result": ""},
        version="v3",
    )

    print("stream.values projection:")
    for snapshot in stream.values:
        print(snapshot)

    print("\nstream.output:")
    print(stream.output)

    print("\nraw protocol events:")
    raw_stream = graph.stream_events(
        {"topic": "raw events", "steps": [], "result": ""},
        version="v3",
    )
    for event in raw_stream:
        print(event["method"], event["params"].get("namespace"), event["params"].get("data"))


if __name__ == "__main__":
    main()
