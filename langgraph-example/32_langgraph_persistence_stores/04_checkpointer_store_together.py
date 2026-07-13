"""
案例 4：Checkpointer + Store 组合模式

目标：
- checkpointer 保存当前 thread 的短期对话状态。
- store 保存跨 thread 的用户长期记忆。
- 演示同一个 user_id 在不同 thread 中共享长期记忆。

对应文档概念：
- Checkpointer + Store 组合模式
- thread_id vs user_id
- short-term memory vs long-term memory
"""

from __future__ import annotations

import operator
import uuid
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


@dataclass
class Context:
    user_id: str


class AssistantState(TypedDict):
    messages: Annotated[list[str], operator.add]
    answer: str


def assistant_node(state: AssistantState, runtime: Runtime[Context]) -> dict:
    latest_message = state["messages"][-1]
    namespace = ("users", runtime.context.user_id, "memories")

    if latest_message.startswith("user: remember "):
        memory_text = latest_message.removeprefix("user: remember ").strip()
        runtime.store.put(
            namespace,
            str(uuid.uuid4()),
            {"text": memory_text},
        )

    memories = runtime.store.search(namespace)
    long_term_memory = [item.value["text"] for item in memories]
    thread_message_count = len(state["messages"])

    answer = (
        f"当前 thread 消息数：{thread_message_count}；"
        f"该用户长期记忆：{long_term_memory or '暂无'}"
    )

    return {
        "messages": [f"assistant: {answer}"],
        "answer": answer,
    }


def build_graph():
    checkpointer = InMemorySaver()
    store = InMemoryStore()

    builder = StateGraph(AssistantState, context_schema=Context)
    builder.add_node("assistant", assistant_node)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)

    return builder.compile(checkpointer=checkpointer, store=store)


def main() -> None:
    graph = build_graph()
    context = Context(user_id="user-1")

    thread_a = {"configurable": {"thread_id": "conversation-a"}}
    thread_b = {"configurable": {"thread_id": "conversation-b"}}

    print("Thread A：写入长期记忆，同时保存当前 thread state")
    result_a = graph.invoke(
        {"messages": ["user: remember 喜欢 Python 和深色模式"]},
        thread_a,
        context=context,
    )
    print(result_a["answer"])

    print("\nThread B：不同 thread，但同一 user_id，可以读到 Store 长期记忆")
    result_b = graph.invoke(
        {"messages": ["user: 我有哪些长期偏好？"]},
        thread_b,
        context=context,
    )
    print(result_b["answer"])

    print("\n检查两个 thread 的短期 messages：")
    print("Thread A:", graph.get_state(thread_a).values["messages"])
    print("Thread B:", graph.get_state(thread_b).values["messages"])


if __name__ == "__main__":
    main()
