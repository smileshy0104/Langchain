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
    # filename 是要导出的文件名。
    filename: str
    # status 保存导出任务的最终状态。
    status: str


# 导出节点：除了返回 state update，还通过 stream writer 主动推送业务进度。
def export_node(state: ExportState) -> dict:
    # get_stream_writer() 返回当前运行上下文中的自定义流写入器。
    # 写入的数据会以 stream_mode="custom" 的 chunk 形式发给调用方。
    writer = get_stream_writer()

    # 这些 writer(...) 调用不会改变 graph state，只是向外部发送进度事件。
    writer({"phase": "validate", "message": f"checking {state['filename']}"})
    writer({"phase": "render", "message": "rendering report"})
    writer({"phase": "upload", "message": "upload complete"})

    # return 的内容才会作为 state update 合并回 graph state。
    return {"status": "exported"}


# 构建单节点 graph，重点演示 custom streaming。
def build_graph():
    builder = StateGraph(ExportState)
    builder.add_node("export", export_node)
    builder.add_edge(START, "export")
    builder.add_edge("export", END)
    return builder.compile()


# 主函数同时订阅 custom 和 updates 两种 stream mode。
def main() -> None:
    graph = build_graph()

    for chunk in graph.stream(
        {"filename": "monthly-report.md", "status": "pending"},
        stream_mode=["updates", "custom"],
        version="v2",
    ):
        # custom chunk 来自 writer(...)，适合展示进度条、日志、token 等业务事件。
        if chunk["type"] == "custom":
            print("custom progress:", chunk["data"])
        # updates chunk 来自节点 return，表示 graph state 的增量更新。
        elif chunk["type"] == "updates":
            print("state update:", chunk["data"])


if __name__ == "__main__":
    main()
