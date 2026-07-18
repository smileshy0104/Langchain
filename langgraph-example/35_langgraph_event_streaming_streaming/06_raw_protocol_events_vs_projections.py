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

import warnings
import operator
from typing import Annotated, TypedDict

from langchain_core._api.beta_decorator import LangChainBetaWarning

from langgraph.graph import END, START, StateGraph


# 当前 LangGraph 版本中 v3 streaming protocol 仍标记为 beta；示例隐藏该警告，避免干扰教学输出。
warnings.filterwarnings("ignore", category=LangChainBetaWarning)


class PipelineState(TypedDict):
    # topic 是输入主题。
    topic: str
    # events 使用追加 reducer，记录 pipeline 中每个节点产生的业务事件。
    events: Annotated[list[str], operator.add]
    # done 标识 pipeline 是否完成。
    done: bool


# pipeline 第一步：追加一个事件，不结束流程。
def step_one(state: PipelineState) -> dict:
    return {"events": [f"step_one:{state['topic']}"]}


# pipeline 第二步：追加完成事件，并把 done 置为 True。
def step_two(state: PipelineState) -> dict:
    return {"events": ["step_two:done"], "done": True}


# 构建两步 pipeline，用同一份 graph 对比 raw events 和 typed projections。
def build_graph():
    builder = StateGraph(PipelineState)
    builder.add_node("step_one", step_one)
    builder.add_node("step_two", step_two)
    builder.add_edge(START, "step_one")
    builder.add_edge("step_one", "step_two")
    builder.add_edge("step_two", END)
    return builder.compile()


# 主函数演示两种读取方式：底层 raw protocol 和上层 typed projections。
def main() -> None:
    graph = build_graph()
    inputs = {"topic": "projection layer", "events": [], "done": False}

    print("Raw protocol events:")
    raw_stream = graph.stream_events(inputs, version="v3")
    for event in raw_stream:
        # raw event 需要调用方自己解析 method/params，并根据事件类型处理 data。
        method = event["method"]
        namespace = event["params"].get("namespace")
        data = event["params"].get("data")
        print(f"method={method}, namespace={namespace}, data={data}")

    print("\nTyped projections:")
    projected_stream = graph.stream_events(inputs, version="v3")
    # values projection 直接给出每个 step 后的 state，隐藏 raw protocol 的分支解析。
    for snapshot in projected_stream.values:
        print("value:", snapshot)
    # output projection 直接给出最终 state。
    print("output:", projected_stream.output)


if __name__ == "__main__":
    main()
