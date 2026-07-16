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


# 用计数器证明 replay 行为：checkpoint 之前的节点不会重跑，checkpoint 之后的节点会重新执行。
COUNTERS = {"plan": 0, "write": 0}


class DraftState(TypedDict):
    # reducer 会把每次节点输出的 log 列表累加起来，便于观察历史轨迹。
    log: Annotated[list[str], operator.add]
    # 写作主题。
    topic: str
    # write 节点生成的草稿。
    draft: str


def plan(state: DraftState) -> dict:
    """规划节点：只在首次运行时执行，replay 到 write 前不会再次执行。"""

    COUNTERS["plan"] += 1
    return {"log": [f"plan run #{COUNTERS['plan']}"]}


def write(state: DraftState) -> dict:
    """写作节点：从 write 前的 checkpoint replay 时会被重新执行。"""

    COUNTERS["write"] += 1
    return {
        "draft": f"draft #{COUNTERS['write']} about {state['topic']}",
        "log": [f"write run #{COUNTERS['write']}"],
    }


def build_graph():
    """构建 START -> plan -> write -> END 的线性图。"""

    builder = StateGraph(DraftState)
    builder.add_node("plan", plan)
    builder.add_node("write", write)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "write")
    builder.add_edge("write", END)

    # 保存 checkpoints 后，才能从历史快照 replay。
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "replay-demo"}}

    # 第一次完整运行，生成 plan/write 相关 checkpoints。
    first = graph.invoke({"log": [], "topic": "checkpoints", "draft": ""}, config)
    print("first:", first)
    print("counters:", COUNTERS)

    # 找到“下一步将执行 write”的快照，即 write 节点执行之前的 checkpoint。
    history = list(graph.get_state_history(config))
    before_write = next(s for s in history if s.next == ("write",))

    # 输入 None 表示不提供新的外部输入，直接从 before_write.config 指向的 checkpoint 继续执行。
    replay = graph.invoke(None, before_write.config)
    print("\nreplay:", replay)
    print("counters:", COUNTERS)
    print("plan did not run again; write did.")


if __name__ == "__main__":
    main()
