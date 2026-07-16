"""
案例 1：Checkpoint history 基础

目标：
- 运行一个多节点 graph。
- 使用 get_state_history(config) 查看历史 checkpoints。
- 理解 state.next 和 checkpoint_id。

对应文档概念：
- 核心依赖：Checkpoints
- get_state_history
- state.next
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class PipelineState(TypedDict):
    # log 使用 reducer：每个节点返回的 list[str] 会追加到已有 log 后面，方便观察执行顺序。
    log: Annotated[list[str], operator.add]
    # 用户输入的主题，会在 plan/write 两个节点之间传递。
    topic: str
    # 最终生成结果，由 write 节点写入。
    result: str


def plan(state: PipelineState) -> dict:
    """第一个业务节点：根据 topic 记录规划动作。"""

    return {"log": [f"plan:{state['topic']}"]}


def write(state: PipelineState) -> dict:
    """第二个业务节点：写入最终结果，并继续追加执行日志。"""

    return {"log": ["write"], "result": f"result for {state['topic']}"}


def build_graph():
    """构建一个线性图：START -> plan -> write -> END。"""

    builder = StateGraph(PipelineState)
    builder.add_node("plan", plan)
    builder.add_node("write", write)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "write")
    builder.add_edge("write", END)

    # Time travel 必须有 checkpointer；InMemorySaver 适合示例/测试，进程结束后数据会丢失。
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()

    # thread_id 用来标识一条会话/执行线程；同一 thread_id 下会保存连续的 checkpoints。
    config = {"configurable": {"thread_id": "history-basics"}}

    # invoke 执行图；每经过一个节点，checkpointer 都会记录新的状态快照。
    graph.invoke({"log": [], "topic": "time travel", "result": ""}, config)

    print("History is usually newest first:")
    # get_state_history 返回当前 thread 的历史快照，通常按“最新 -> 最旧”排序。
    for index, snapshot in enumerate(graph.get_state_history(config), start=1):
        # checkpoint_id 是具体快照的唯一标识，后续 replay/fork 都可以基于这个 config 继续。
        checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
        print(
            f"{index}. next={snapshot.next}, "  # next 表示从该快照继续时，下一步要执行哪些节点。
            f"step={snapshot.metadata.get('step')}, "  # metadata.step 是 LangGraph 内部的执行步序号。
            f"checkpoint_id={checkpoint_id}"
        )


if __name__ == "__main__":
    main()
