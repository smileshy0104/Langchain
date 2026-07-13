"""
案例 1：短期记忆与 checkpointer

目标：
- 使用 InMemorySaver 保存 thread 级 graph state。
- 使用同一个 thread_id 进行多轮调用。
- 观察第二轮调用能读到第一轮保存的 messages。

对应文档概念：
- Checkpointer
- thread_id
- short-term memory
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class ChatState(TypedDict):
    messages: Annotated[list[str], operator.add]
    answer: str


def assistant_node(state: ChatState) -> dict:
    messages = state.get("messages", [])
    latest_user = next(
        (message for message in reversed(messages) if message.startswith("user:")),
        "user:",
    )

    known_name = None
    for message in messages:
        if message.startswith("user: 我叫"):
            known_name = message.removeprefix("user: 我叫").strip(" 。.")

    if "我叫什么" in latest_user and known_name:
        answer = f"你叫 {known_name}。这是从同一个 thread 的历史消息里读到的。"
    elif known_name:
        answer = f"我记住了，你叫 {known_name}。"
    else:
        answer = "我还没有在这个 thread 中记住你的名字。"

    return {
        "messages": [f"assistant: {answer}"],
        "answer": answer,
    }


def build_graph():
    builder = StateGraph(ChatState)
    builder.add_node("assistant", assistant_node)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "conversation-1"}}

    print("第一轮：写入当前 thread 的短期记忆")
    first = graph.invoke(
        {"messages": ["user: 我叫 Bob。"]},
        config,
    )
    print(first["answer"])

    print("\n第二轮：复用同一个 thread_id，读取上一轮状态")
    second = graph.invoke(
        {"messages": ["user: 我叫什么名字？"]},
        config,
    )
    print(second["answer"])

    snapshot = graph.get_state(config)
    print("\n当前 checkpoint 中保存的 messages：")
    for message in snapshot.values["messages"]:
        print("-", message)


if __name__ == "__main__":
    main()
