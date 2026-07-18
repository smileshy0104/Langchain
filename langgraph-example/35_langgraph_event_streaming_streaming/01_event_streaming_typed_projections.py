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

import warnings
import operator
from typing import Annotated, TypedDict

from langchain_core._api.beta_decorator import LangChainBetaWarning

from langgraph.graph import END, START, StateGraph


# 当前 LangGraph 版本中 v3 streaming protocol 仍标记为 beta；示例隐藏该警告，避免干扰教学输出。
warnings.filterwarnings("ignore", category=LangChainBetaWarning)


class ReportState(TypedDict):
    # topic 是输入主题，后续节点会围绕它生成报告。
    topic: str
    # steps 使用追加 reducer：每个节点返回的步骤日志都会累积到同一个列表中。
    steps: Annotated[list[str], operator.add]
    # result 保存最终报告内容。
    result: str


# 第一个节点：生成计划步骤，只更新 steps。
def plan_node(state: ReportState) -> dict:
    return {"steps": [f"plan:{state['topic']}"]}


# 第二个节点：写报告草稿，同时追加步骤并写入最终 result。
def write_node(state: ReportState) -> dict:
    return {
        "steps": ["write:draft"],
        "result": f"Report about {state['topic']}",
    }


# 构建一个两步 graph，用来产生多次 state 变化，便于观察事件流。
def build_graph():
    builder = StateGraph(ReportState)
    builder.add_node("plan", plan_node)
    builder.add_node("write", write_node)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "write")
    builder.add_edge("write", END)
    return builder.compile()


# 主函数演示 Event Streaming v3 的 typed projections 与 raw events。
def main() -> None:
    graph = build_graph()

    # stream_events(version="v3") 返回 GraphRunStream。
    # 它既可以直接迭代 raw protocol events，也可以通过 values/output 等 projection 读取结构化结果。
    stream = graph.stream_events(
        {"topic": "typed projections", "steps": [], "result": ""},
        version="v3",
    )

    print("stream.values projection:")
    # stream.values 会把 raw event 解析成“每一步后的完整 state snapshot”。
    # 调用方无需自己判断 event["method"] 或 event["params"]。
    for snapshot in stream.values:
        print(snapshot)

    print("\nstream.output:")
    # stream.output 是本次 graph 运行完成后的最终 state。
    print(stream.output)

    print("\nraw protocol events:")
    # 同一个 API 仍然支持直接遍历 raw events，适合调试协议细节或做自定义解析。
    raw_stream = graph.stream_events(
        {"topic": "raw events", "steps": [], "result": ""},
        version="v3",
    )
    for event in raw_stream:
        # method 表示事件类型，params 中包含 namespace/data 等底层协议字段。
        print(event["method"], event["params"].get("namespace"), event["params"].get("data"))


if __name__ == "__main__":
    main()
