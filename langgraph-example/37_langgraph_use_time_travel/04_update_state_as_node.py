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
    # 文章主题，由 choose_topic 确定。
    topic: str
    # 提纲，由 make_outline 生成；本例会用 update_state 手动注入。
    outline: str
    # 最终文章，由 write_essay 根据 outline 生成。
    essay: str


def choose_topic(state: EssayState) -> dict:
    """选择主题：如果没有输入则给一个默认主题。"""

    return {"topic": state.get("topic") or "LangGraph"}


def make_outline(state: EssayState) -> dict:
    """生成提纲；后面会演示如何跳过它，直接注入人工提纲。"""

    return {"outline": f"Outline for {state['topic']}"}


def write_essay(state: EssayState) -> dict:
    """根据当前 outline 写文章。"""

    return {"essay": f"Essay using: {state['outline']}"}


def build_graph():
    """构建三步写作图：选题 -> 提纲 -> 正文。"""

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

    # 第一次运行用于产生完整历史，后续从历史 checkpoint 中选择 fork 点。
    graph.invoke({"topic": "Persistence", "outline": "", "essay": ""}, config)

    history = list(graph.get_state_history(config))
    # 找到 make_outline 之前的快照。
    before_outline = next(s for s in history if s.next == ("make_outline",))

    # as_node="make_outline" 表示：把这次人工 update 当成 make_outline 节点的输出。
    # 因此继续执行时，会从 make_outline 的后继节点 write_essay 开始，而不是再次执行 make_outline。
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
