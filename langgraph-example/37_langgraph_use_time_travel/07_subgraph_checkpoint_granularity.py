"""
案例 7：Subgraph checkpoint 粒度

目标：
- 展示默认 subgraph 继承父图 checkpointer 时，父图只看到 subgraph_node 级别。
- 展示 subgraph.compile(checkpointer=True) 时，可以通过 get_state(..., subgraphs=True)
  取得子图内部 checkpoint config。

对应文档概念：
- Time Travel 与 Subgraphs
- Inherited Checkpointer
- Subgraph Own Checkpointer
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    # 父图和子图共用同一个状态结构；reducer 让子图每一步的结果都追加到 value。
    value: Annotated[list[str], operator.add]


def step_a(state: State) -> dict:
    """子图内部第一个 interrupt 节点。"""

    answer = interrupt("step_a input?")
    return {"value": [f"a:{answer}"]}


def step_b(state: State) -> dict:
    """子图内部第二个 interrupt 节点。"""

    answer = interrupt("step_b input?")
    return {"value": [f"b:{answer}"]}


def build_parent_graph(*, subgraph_own_checkpointer: bool):
    """构建父图，其中唯一业务节点是一个子图。"""

    # 子图内部有两个节点：step_a -> step_b。
    sub_builder = StateGraph(State)
    sub_builder.add_node("step_a", step_a)
    sub_builder.add_node("step_b", step_b)
    sub_builder.add_edge(START, "step_a")
    sub_builder.add_edge("step_a", "step_b")
    sub_builder.add_edge("step_b", END)

    if subgraph_own_checkpointer:
        # checkpointer=True 表示子图维护自己的 checkpoint，可暴露更细粒度的内部状态。
        subgraph = sub_builder.compile(checkpointer=True)
    else:
        # 默认不显式设置时，子图使用父图继承/调用上下文；父图历史通常只体现 subgraph_node 层级。
        subgraph = sub_builder.compile()

    # 父图把整个子图当作一个节点 subgraph_node。
    parent_builder = StateGraph(State)
    parent_builder.add_node("subgraph_node", subgraph)
    parent_builder.add_edge(START, "subgraph_node")
    parent_builder.add_edge("subgraph_node", END)
    return parent_builder.compile(checkpointer=InMemorySaver())


def run_default_subgraph() -> None:
    """演示默认子图 checkpoint 粒度：只能从父图看到 subgraph_node。"""

    print("Default subgraph: inherited parent checkpointer")
    # 父图默认继承子图 checkpointer，子图 checkpoint 只能被父图访问。
    # subgraph_own_checkpointer 是为了测试子图 checkpoint 粒度，可显式设置子图 checkpointer。
    graph = build_parent_graph(subgraph_own_checkpointer=False)
    config = {"configurable": {"thread_id": "subgraph-default"}}

    # 依次触发 step_a 暂停、恢复后触发 step_b 暂停、再恢复完成。
    graph.invoke({"value": []}, config)
    graph.invoke(Command(resume="Alice"), config)
    graph.invoke(Command(resume="30"), config)

    history = list(graph.get_state_history(config))
    print("parent-level next values:")
    for snapshot in history:
        # 默认场景中，父图历史关注的是父图节点层级，而不是 step_a/step_b 的每个内部 checkpoint。
        print("-", snapshot.next)
    print("Default mode cannot time travel between step_a and step_b from parent history.")


def run_own_checkpoint_subgraph() -> None:
    """演示子图拥有独立 checkpointer 后，如何拿到子图内部 config 并 fork。"""

    print("\nSubgraph checkpointer=True: own checkpoint history")
    graph = build_parent_graph(subgraph_own_checkpointer=True)
    config = {"configurable": {"thread_id": "subgraph-own"}}

    # 第一次停在 step_a；恢复后会进入 step_b 并停下，此时父图仍在 subgraph_node 任务中。
    graph.invoke({"value": []}, config)
    graph.invoke(Command(resume="Alice"), config)

    # subgraphs=True 会把正在运行的子图任务状态一起取出来。
    parent_state = graph.get_state(config, subgraphs=True)
    # tasks[0].state.config 是子图当前 checkpoint 的 config，可用于子图内部 time travel/fork。
    sub_config = parent_state.tasks[0].state.config
    print("subgraph checkpoint config:", sub_config)

    # 基于子图内部 checkpoint fork，并注入一条 value。
    fork_config = graph.update_state(sub_config, {"value": ["forked-inside-subgraph"]})
    # 继续执行时会从子图内部的当前位置继续（通常会再次停在下一处 interrupt）。
    fork_result = graph.invoke(None, fork_config)
    print("forked from subgraph checkpoint:", fork_result)


def main() -> None:
    run_default_subgraph()
    run_own_checkpoint_subgraph()


if __name__ == "__main__":
    main()
