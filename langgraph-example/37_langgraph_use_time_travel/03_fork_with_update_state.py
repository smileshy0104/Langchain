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
    topic: str
    joke: str


def generate_topic(state: JokeState) -> dict:
    return {"topic": state.get("topic") or "socks"}


def write_joke(state: JokeState) -> dict:
    return {"joke": f"Why do {state['topic']} disappear? They elope!"}


def build_graph():
    builder = StateGraph(JokeState)
    builder.add_node("generate_topic", generate_topic)
    builder.add_node("write_joke", write_joke)
    builder.add_edge(START, "generate_topic")
    builder.add_edge("generate_topic", "write_joke")
    builder.add_edge("write_joke", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "fork-demo"}}

    original = graph.invoke({"topic": "socks", "joke": ""}, config)
    print("original:", original)

    history = list(graph.get_state_history(config))
    before_joke = next(s for s in history if s.next == ("write_joke",))

    fork_config = graph.update_state(
        before_joke.config,
        values={"topic": "chickens"},
    )
    forked = graph.invoke(None, fork_config)
    print("forked:", forked)

    print("\nupdate checkpoints:")
    for snapshot in graph.get_state_history(config):
        if snapshot.metadata.get("source") == "update":
            print(snapshot.config, snapshot.values)


if __name__ == "__main__":
    main()
