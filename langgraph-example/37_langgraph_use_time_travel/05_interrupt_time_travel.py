"""
案例 5：Time travel 与 interrupts

目标：
- 先完成一次 interrupt。
- 从 interrupt 节点前 replay，interrupt 会再次触发。
- 从 interrupt 节点前 fork，interrupt 也会再次触发。

对应文档概念：
- Time Travel 与 Interrupts
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class GreetingState(TypedDict):
    # 使用 reducer 累加每个节点输出，便于看到 replay/fork 后新增的轨迹。
    value: Annotated[list[str], operator.add]


def ask_human(state: GreetingState) -> dict:
    """需要人工输入的节点：interrupt 会暂停图，并把问题返回给调用方。"""

    # 第一次执行到这里会暂停；之后通过 Command(resume=...) 把答案传回该节点。
    answer = interrupt("What is your name?")
    return {"value": [f"Hello, {answer}!"]}


def final_step(state: GreetingState) -> dict:
    """interrupt 恢复后执行的收尾节点。"""

    return {"value": ["Done"]}


def build_graph():
    """构建 START -> ask_human -> final_step -> END。"""

    builder = StateGraph(GreetingState)
    builder.add_node("ask_human", ask_human)
    builder.add_node("final_step", final_step)
    builder.add_edge(START, "ask_human")
    builder.add_edge("ask_human", "final_step")
    builder.add_edge("final_step", END)

    # interrupt 必须配合 checkpointer，暂停点和恢复点都依赖 checkpoint。
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "interrupt-time-travel"}}

    print("First run pauses:")
    # 第一次执行会在 ask_human 处暂停，并返回 __interrupt__ 信息。
    graph.invoke({"value": []}, config)
    # Command(resume=...) 把人工回答送回上一次 interrupt 暂停的位置。
    completed = graph.invoke(Command(resume="Alice"), config)
    print("completed:", completed)

    history = list(graph.get_state_history(config))
    # 选择最早的“下一步是 ask_human”的快照：它位于 interrupt 节点执行之前。
    before_ask = [s for s in history if s.next == ("ask_human",)][-1]

    print("\nReplay from before ask_human pauses again:")
    # replay 到 interrupt 节点之前时，ask_human 会重新执行，因此再次暂停等待输入。
    replay_result = graph.invoke(None, before_ask.config)
    print(replay_result)

    print("\nFork from before ask_human also pauses:")
    # fork 时先在历史快照上写入额外 value，再继续执行；由于下一步仍是 ask_human，也会再次 interrupt。
    fork_config = graph.update_state(before_ask.config, {"value": ["forked"]})
    fork_result = graph.invoke(None, fork_config)
    print(fork_result)

    # 用新的答案恢复 fork 分支，得到与原始 Alice 分支不同的执行结果。
    resumed_fork = graph.invoke(Command(resume="Bob"), fork_config)
    print("resumed fork:", resumed_fork)


if __name__ == "__main__":
    main()
