"""
案例 3：Fork with update_state

目标：
- 从 write 节点前的 checkpoint fork。
- 修改 topic 后继续执行。
- 原始历史不被回滚。

对应文档概念：
- Fork
- update_state does not roll back a thread
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class JokeState(TypedDict):
    # 笑话主题；generate_topic 可以设置默认值，fork 时也会修改它。
    topic: str
    # 最终笑话文本，由 write_joke 生成。
    joke: str


def generate_topic(state: JokeState) -> dict:
    """生成/补全主题：如果调用方没传 topic，则使用 socks。"""

    return {"topic": state.get("topic") or "socks"}


def write_joke(state: JokeState) -> dict:
    """根据当前 topic 写笑话；fork 后会使用新的 topic 重新生成。"""

    return {"joke": f"Why do {state['topic']} disappear? They elope!"}


def build_graph():
    """构建 START -> generate_topic -> write_joke -> END。"""

    builder = StateGraph(JokeState)
    builder.add_node("generate_topic", generate_topic)
    builder.add_node("write_joke", write_joke)
    builder.add_edge(START, "generate_topic")
    builder.add_edge("generate_topic", "write_joke")
    builder.add_edge("write_joke", END)

    # checkpointer 会记录原始分支和 fork 后的新分支。
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "fork-demo"}}

    # 先完整运行一次，得到一条原始执行历史。
    original = graph.invoke({"topic": "socks", "joke": ""}, config)
    print("original:", original)

    history = list(graph.get_state_history(config))
    # 找到 write_joke 之前的 checkpoint：此时 topic 已确定，但 joke 还未生成。
    before_joke = next(s for s in history if s.next == ("write_joke",))

    # update_state 基于历史 checkpoint 写入新状态，会创建一个新的 checkpoint 分支，而不是回滚旧历史。
    fork_config = graph.update_state(
        before_joke.config,
        values={"topic": "chickens"},
    )
    # 从 fork_config 继续执行，下一步仍是 write_joke，但会读取被改成 chickens 的 topic。
    forked = graph.invoke(None, fork_config)
    print("forked:", forked)

    print("\nupdate checkpoints:")
    # metadata.source == "update" 的快照就是 update_state 创建的人工修改点。
    for snapshot in graph.get_state_history(config):
        if snapshot.metadata.get("source") == "update":
            print(snapshot.config, snapshot.values)


if __name__ == "__main__":
    main()
