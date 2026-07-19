"""
案例 6：从多个 interrupts 中间 fork

目标：
- ask_name -> ask_age -> final。
- 完成两个问题后，从 ask_name 和 ask_age 中间的 checkpoint fork。
- 保留名字回答，只重新询问年龄。

对应文档概念：
- Multiple Interrupts
- between checkpoint
- s.next == ("ask_age",)
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class FormState(TypedDict):
    # 表单填写轨迹；operator.add 会把 name/age/final 等片段顺序追加。
    value: Annotated[list[str], operator.add]


def ask_name(state: FormState) -> dict:
    """第一个 interrupt：询问姓名。"""

    name = interrupt("What is your name?")
    return {"value": [f"name:{name}"]}


def ask_age(state: FormState) -> dict:
    """第二个 interrupt：询问年龄。"""

    age = interrupt("How old are you?")
    return {"value": [f"age:{age}"]}


def final(state: FormState) -> dict:
    """所有问题完成后的收尾节点。"""

    return {"value": ["final"]}


def build_graph():
    """构建包含两个连续 interrupt 节点的线性图。"""

    builder = StateGraph(FormState)
    builder.add_node("ask_name", ask_name)
    builder.add_node("ask_age", ask_age)
    builder.add_node("final", final)
    builder.add_edge(START, "ask_name")
    builder.add_edge("ask_name", "ask_age")
    builder.add_edge("ask_age", "final")
    builder.add_edge("final", END)

    # 多次 interrupt/resume 都需要基于同一个 thread_id 的 checkpoints。
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "multi-interrupt-time-travel"}}

    # 第一次暂停在 ask_name。
    graph.invoke({"value": []}, config)
    # 恢复 ask_name 后会继续到 ask_age，并再次暂停。
    graph.invoke(Command(resume="Alice"), config)
    # 恢复 ask_age 后执行 final，完成整条原始分支。
    completed = graph.invoke(Command(resume=30), config)
    print("completed:", completed)

    history = list(graph.get_state_history(config))
    # s.next == ("ask_age",) 表示 ask_name 已完成，下一步将进入 ask_age。
    matches = [s for s in history if s.next == ("ask_age",)]
    between = matches[0]
    print("matches:",matches)

    print("\nbetween means: ask_name completed, ask_age not started")
    print("between values:", between.values)

    # 从两个 interrupt 中间 fork：保留 ask_name 的结果，再额外追加一条 modified。
    fork_config = graph.update_state(between.config, {"value": ["modified"]})
    # 继续 fork 分支会再次停在 ask_age，因为 ask_age 在该 checkpoint 之后尚未执行。
    fork_result = graph.invoke(None, fork_config)
    print("fork pauses at ask_age again:", fork_result)

    # 用新的年龄恢复 fork 分支；原始分支中的 age:30 不会被回滚或覆盖。
    resumed = graph.invoke(Command(resume=18), fork_config)
    print("resumed fork:", resumed)


if __name__ == "__main__":
    main()
