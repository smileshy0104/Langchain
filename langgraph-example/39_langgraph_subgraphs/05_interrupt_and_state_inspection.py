"""
案例 5：Subgraph interrupt and state inspection

目标：
- 子图内部调用 interrupt()。
- 顶层 graph 暂停。
- 使用 get_state(config, subgraphs=True) 查看暂停的子图 state。
- 顶层通过 Command(resume=...) 恢复执行。

对应文档概念：
- Interrupt
- View Subgraph State
- Per-invocation State Inspection
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    # 父图和子图共享 answer 字段；子图会把 interrupt 的回答写入这里。
    answer: str


def ask_inside_subgraph(state: State) -> dict:
    """子图内部节点：通过 interrupt 暂停，等待外部输入。"""

    # interrupt 会让整个顶层 graph 暂停，而不仅仅是子图暂停。
    value = interrupt("Provide value for subgraph:")
    return {"answer": f"subgraph received: {value}"}


def build_subgraph():
    """构建包含 interrupt 的子图。"""

    builder = StateGraph(State)
    builder.add_node("ask_inside_subgraph", ask_inside_subgraph)
    builder.add_edge(START, "ask_inside_subgraph")
    builder.add_edge("ask_inside_subgraph", END)
    return builder.compile()


def build_graph():
    """把子图作为父图节点，并在父图启用 checkpointer 支持暂停/恢复。"""

    subgraph = build_subgraph()
    builder = StateGraph(State)
    builder.add_node("subgraph_node", subgraph)
    builder.add_edge(START, "subgraph_node")
    builder.add_edge("subgraph_node", END)

    # interrupt/resume 需要 checkpointer 保存暂停点。
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "subgraph-interrupt"}}

    # 首次执行会进入子图，并在 ask_inside_subgraph 的 interrupt 处暂停。
    paused = graph.invoke({"answer": ""}, config)
    print("paused:", paused)

    # subgraphs=True 会把当前暂停任务中的子图状态也展开出来。
    parent_state = graph.get_state(config, subgraphs=True)
    # tasks[0] 是当前暂停的父图任务；其 state 是子图的 StateSnapshot。
    subgraph_state = parent_state.tasks[0].state
    print("subgraph next:", subgraph_state.next)
    print("subgraph values:", subgraph_state.values)

    # 顶层直接 resume，LangGraph 会把值路由回子图内部的 interrupt 调用点。
    resumed = graph.invoke(Command(resume="approved"), config)
    print("resumed:", resumed)


if __name__ == "__main__":
    main()
