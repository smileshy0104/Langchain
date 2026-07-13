"""
案例 2：thread_id 隔离

目标：
- 同一个 graph、同一个 checkpointer。
- 不同 thread_id 保存不同 conversation state。
- 演示 thread_id 是 checkpointer 读取和恢复 state 的游标。

对应文档概念：
- Threads
- thread_id
- Checkpointer 与 Store 的区别
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
    known_name = None
    for message in state.get("messages", []):
        if message.startswith("user: 我叫"):
            known_name = message.removeprefix("user: 我叫").strip(" 。.")

    answer = f"当前 thread 里记住的名字：{known_name or '未知'}"
    return {"messages": [f"assistant: {answer}"], "answer": answer}


def build_graph():
    builder = StateGraph(ChatState)
    builder.add_node("assistant", assistant_node)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()

    alice_thread = {"configurable": {"thread_id": "thread-alice"}}
    bob_thread = {"configurable": {"thread_id": "thread-bob"}}

    graph.invoke({"messages": ["user: 我叫 Alice。"]}, alice_thread)
    graph.invoke({"messages": ["user: 我叫 Bob。"]}, bob_thread)

    alice_result = graph.invoke({"messages": ["user: 我是谁？"]}, alice_thread)
    bob_result = graph.invoke({"messages": ["user: 我是谁？"]}, bob_thread)

    print("Alice thread:", alice_result["answer"])
    print("Bob thread:", bob_result["answer"])

    print("\n两个 thread 的 checkpoint state 是隔离的：")
    print("thread-alice messages:")
    for message in graph.get_state(alice_thread).values["messages"]:
        print("-", message)

    print("\nthread-bob messages:")
    for message in graph.get_state(bob_thread).values["messages"]:
        print("-", message)


if __name__ == "__main__":
    main()
