"""
案例 3：Custom data with get_stream_writer()

目标：
- 在节点中通过 get_stream_writer() 写入业务进度。
- 使用 graph.stream(..., stream_mode=["updates", "custom"], version="v2")。
- 从 chunk["type"] 区分 custom 数据和 state updates。

对应文档概念：
- Custom Data
- get_stream_writer()
- Multiple Modes
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph


class ExportState(TypedDict):
    filename: str
    status: str


def export_node(state: ExportState) -> dict:
    writer = get_stream_writer()

    writer({"phase": "validate", "message": f"checking {state['filename']}"})
    writer({"phase": "render", "message": "rendering report"})
    writer({"phase": "upload", "message": "upload complete"})

    return {"status": "exported"}


def build_graph():
    builder = StateGraph(ExportState)
    builder.add_node("export", export_node)
    builder.add_edge(START, "export")
    builder.add_edge("export", END)
    return builder.compile()


def main() -> None:
    graph = build_graph()

    for chunk in graph.stream(
        {"filename": "monthly-report.md", "status": "pending"},
        stream_mode=["updates", "custom"],
        version="v2",
    ):
        if chunk["type"] == "custom":
            print("custom progress:", chunk["data"])
        elif chunk["type"] == "updates":
            print("state update:", chunk["data"])


if __name__ == "__main__":
    main()
