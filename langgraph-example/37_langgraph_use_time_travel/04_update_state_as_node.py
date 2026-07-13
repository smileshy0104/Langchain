"""
案例 4：update_state 与 as_node

目标：
- 显式指定 as_node，让 LangGraph 把 update 视为某个节点的输出。
- 执行从该节点的 successors 继续。

对应文档概念：
- update_state 与 as_node
- Skipping nodes
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class EssayState(TypedDict):
    topic: str
    outline: str
    essay: str


def choose_topic(state: EssayState) -> dict:
    return {"topic": state.get("topic") or "LangGraph"}


def make_outline(state: EssayState) -> dict:
    return {"outline": f"Outline for {state['topic']}"}


def write_essay(state: EssayState) -> dict:
    return {"essay": f"Essay using: {state['outline']}"}


def build_graph():
    builder = StateGraph(EssayState)
    builder.add_node("choose_topic", choose_topic)
    builder.add_node("make_outline", make_outline)
    builder.add_node("write_essay", write_essay)
    builder.add_edge(START, "choose_topic")
    builder.add_edge("choose_topic", "make_outline")
    builder.add_edge("make_outline", "write_essay")
    builder.add_edge("write_essay", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "as-node-demo"}}

    graph.invoke({"topic": "Persistence", "outline": "", "essay": ""}, config)

    history = list(graph.get_state_history(config))
    before_outline = next(s for s in history if s.next == ("make_outline",))

    fork_config = graph.update_state(
        before_outline.config,
        values={"outline": "Manual outline injected by reviewer"},
        as_node="make_outline",
    )
    result = graph.invoke(None, fork_config)
    print(result)
    print("\nBecause as_node='make_outline', execution continues at write_essay.")


if __name__ == "__main__":
    main()
