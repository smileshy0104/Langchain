"""
案例 3：Handling multiple interrupts

目标：
- 两个并行分支同时调用 interrupt()。
- stream.interrupts 中会有多个 pending interrupts。
- 用 {interrupt.id: answer} 的 resume map 一次性恢复。

对应文档概念：
- Handling Multiple Interrupts
- Command(resume=resume_map)
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ParallelState(TypedDict):
    # 两个并行分支都会写 answers，因此用 operator.add 追加合并列表。
    answers: Annotated[list[str], operator.add]


# 并行分支 A：暂停并等待 question_a 的答案。
def ask_a(state: ParallelState) -> dict:
    answer = interrupt("question_a")
    return {"answers": [f"a:{answer}"]}


# 并行分支 B：暂停并等待 question_b 的答案。
def ask_b(state: ParallelState) -> dict:
    answer = interrupt("question_b")
    return {"answers": [f"b:{answer}"]}


# START 同时连到 ask_a 和 ask_b，形成两个并行 interrupt。
def build_graph():
    builder = StateGraph(ParallelState)
    builder.add_node("ask_a", ask_a)
    builder.add_node("ask_b", ask_b)
    builder.add_edge(START, "ask_a")
    builder.add_edge(START, "ask_b")
    builder.add_edge("ask_a", END)
    builder.add_edge("ask_b", END)
    return builder.compile(checkpointer=InMemorySaver())


# 主函数演示多个 pending interrupts 的恢复方式。
def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "multiple-interrupts"}}

    stream = graph.stream_events({"answers": []}, config=config, version="v3")
    _ = stream.output

    print("pending interrupts:")
    for item in stream.interrupts:
        # 每个 interrupt 都有唯一 id；并行恢复时必须用 id 区分答案属于哪个暂停点。
        print(f"- id={item.id}, value={item.value}")

    # resume_map 的 key 是 interrupt.id，value 是对应的人类答案。
    resume_map = {
        item.id: f"answer for {item.value}"
        for item in stream.interrupts
    }

    # 一次性恢复所有 pending interrupts。
    resumed = graph.stream_events(
        Command(resume=resume_map),
        config=config,
        version="v3",
    )
    print("final:", resumed.output)


if __name__ == "__main__":
    main()
