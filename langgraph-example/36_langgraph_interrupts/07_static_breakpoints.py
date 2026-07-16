"""
案例 7：静态断点调试

目标：
- 使用 interrupt_before / interrupt_after 设置静态断点。
- 用 graph.invoke(None, config) 从断点继续。
- 区分 static breakpoint 和业务里的 dynamic interrupt()。

对应文档概念：
- Debugging with Static Interrupts
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class DebugState(TypedDict):
    # log 使用追加 reducer，记录节点执行顺序。
    log: Annotated[list[str], operator.add]


# 第一个节点：写入 node_a 日志。
def node_a(state: DebugState) -> dict:
    return {"log": ["node_a"]}


# 第二个节点：写入 node_b 日志。
def node_b(state: DebugState) -> dict:
    return {"log": ["node_b"]}


# 构建带静态断点的 graph。
def build_graph():
    builder = StateGraph(DebugState)
    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_edge(START, "node_a")
    builder.add_edge("node_a", "node_b")
    builder.add_edge("node_b", END)

    return builder.compile(
        checkpointer=InMemorySaver(),
        # 静态断点：每次执行到 node_b 之前暂停，适合调试或人工检查 state。
        # 它不同于节点里主动调用 interrupt() 的业务中断，不需要写 interrupt() 代码。
        interrupt_before=["node_b"],
    )


# 主函数演示：先停在 node_b 前，再从断点继续执行。
def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "static-breakpoint"}}

    # 第一次 invoke 执行 node_a 后，在 node_b 前暂停。
    first = graph.invoke({"log": []}, config)
    print("paused before node_b:", first)
    print("state:", graph.get_state(config).values)

    # invoke(None, config) 表示不提供新输入，从 checkpoint 中的暂停点继续。
    resumed = graph.invoke(None, config)
    print("resumed:", resumed)


if __name__ == "__main__":
    main()
