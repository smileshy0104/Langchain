"""
案例 1：Short-term memory with checkpointer

目标：
- 使用 InMemorySaver 保存 thread 内 messages。
- 同一个 thread_id 的第二轮调用能看到第一轮消息。
- 不同 thread_id 互相隔离。

对应文档概念：
- Short-term memory
- checkpointer
- thread_id
- MessagesState
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph


def _text(message) -> str:
    return str(getattr(message, "content", message))


def memory_echo(state: MessagesState) -> dict:
    human_messages = [
        _text(message)
        for message in state["messages"]
        if getattr(message, "type", None) == "human"
    ]
    answer = f"我在这个 thread 中看到了 {len(human_messages)} 条用户消息：{human_messages}"
    return {"messages": [AIMessage(content=answer)]}


def build_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("memory_echo", memory_echo)
    builder.add_edge(START, "memory_echo")
    builder.add_edge("memory_echo", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    thread_a = {"configurable": {"thread_id": "conversation-a"}}
    thread_b = {"configurable": {"thread_id": "conversation-b"}}

    print("Thread A 第一轮：")
    print(graph.invoke({"messages": [HumanMessage(content="Hi, I am Bob.")]}, thread_a))

    print("\nThread A 第二轮：同一个 thread_id 会加载上一轮 messages")
    print(graph.invoke({"messages": [HumanMessage(content="What is my name?")]}, thread_a))

    print("\nThread B 第一轮：不同 thread_id 不共享短期记忆")
    print(graph.invoke({"messages": [HumanMessage(content="What is my name?")]}, thread_b))


if __name__ == "__main__":
    main()
