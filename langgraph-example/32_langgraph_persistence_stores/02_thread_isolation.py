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
    # operator.add 让每轮传入和节点返回的 messages 都追加到该 thread 的历史中。
    messages: Annotated[list[str], operator.add]
    answer: str


# 该节点只读取“当前 thread”的 messages，因此不同 thread_id 之间不会串数据。
def assistant_node(state: ChatState) -> dict:
    known_name = None

    # 遍历当前状态中的所有历史消息，查找本 thread 内保存的名字。
    for message in state.get("messages", []):
        if message.startswith("user: 我叫"):
            known_name = message.removeprefix("user: 我叫").strip(" 。.")

    answer = f"当前 thread 里记住的名字：{known_name or '未知'}"

    # 返回 assistant 消息，继续追加到当前 thread 的 checkpoint state。
    return {"messages": [f"assistant: {answer}"], "answer": answer}


# 同一个 graph 使用同一个 InMemorySaver，但保存状态时会按 thread_id 分桶。
def build_graph():
    builder = StateGraph(ChatState)
    builder.add_node("assistant", assistant_node)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)
    return builder.compile(checkpointer=InMemorySaver())


# 主函数演示：Alice 和 Bob 使用不同 thread_id，短期记忆互相隔离。
def main() -> None:
    graph = build_graph()

    # 两个配置只有 thread_id 不同；这会让 checkpointer 读取/写入不同的 checkpoint。
    alice_thread = {"configurable": {"thread_id": "thread-alice"}}
    bob_thread = {"configurable": {"thread_id": "thread-bob"}}

    # 分别在两个 thread 中写入不同名字。
    graph.invoke({"messages": ["user: 我叫 Alice。"]}, alice_thread)
    graph.invoke({"messages": ["user: 我叫 Bob。"]}, bob_thread)

    # 再次调用时，每个 thread 只能读到自己历史中保存的名字。
    alice_result = graph.invoke({"messages": ["user: 我是谁？"]}, alice_thread)
    bob_result = graph.invoke({"messages": ["user: 我是谁？"]}, bob_thread)

    print("Alice thread:", alice_result["answer"])
    print("Bob thread:", bob_result["answer"])

    # 对比两个 thread 的状态快照，验证 messages 分别保存在不同 checkpoint 中。
    print("\n两个 thread 的 checkpoint state 是隔离的：")
    print("thread-alice messages:")
    for message in graph.get_state(alice_thread).values["messages"]:
        print("-", message)

    print("\nthread-bob messages:")
    for message in graph.get_state(bob_thread).values["messages"]:
        print("-", message)


if __name__ == "__main__":
    main()
