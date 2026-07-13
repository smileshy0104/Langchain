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
    # user_id 由调用方传入，用来决定长期记忆写到哪个 Store namespace。
    user_id: str


class AssistantState(TypedDict):
    # messages 归 checkpointer 管，是 thread 级短期记忆。
    messages: Annotated[list[str], operator.add]
    answer: str


# 该节点同时使用 state（短期记忆）和 runtime.store（长期记忆）。
def assistant_node(state: AssistantState, runtime: Runtime[Context]) -> dict:
    # 因为 messages 使用追加 reducer，本轮最新用户消息会出现在列表末尾。
    latest_message = state["messages"][-1]

    # Store 的 namespace 与 user_id 绑定，因此同一个用户跨 thread 共享长期记忆。
    namespace = ("users", runtime.context.user_id, "memories")

    if latest_message.startswith("user: remember "):
        # 将用户显式要求记住的信息写入长期 Store。
        memory_text = latest_message.removeprefix("user: remember ").strip()
        runtime.store.put(
            namespace,
            str(uuid.uuid4()),
            {"text": memory_text},
        )

    # 无论本轮是否写入，都读取当前用户的长期记忆，展示 Store 的跨 thread 效果。
    memories = runtime.store.search(namespace)
    long_term_memory = [item.value["text"] for item in memories]

    # 当前 thread 的消息数来自 checkpoint state，只统计本 thread 的短期对话。
    thread_message_count = len(state["messages"])

    answer = (
        f"当前 thread 消息数：{thread_message_count}；"
        f"该用户长期记忆：{long_term_memory or '暂无'}"
    )

    # assistant 回复会追加到当前 thread 的 messages；answer 供调用方直接读取。
    return {
        "messages": [f"assistant: {answer}"],
        "answer": answer,
    }


# 同时编译 checkpointer 和 store，形成“短期 + 长期”组合记忆模式。
def build_graph():
    checkpointer = InMemorySaver()
    store = InMemoryStore()

    builder = StateGraph(AssistantState, context_schema=Context)
    builder.add_node("assistant", assistant_node)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)

    # checkpointer 根据 thread_id 保存 graph state；store 根据 namespace 保存应用数据。
    return builder.compile(checkpointer=checkpointer, store=store)


# 主函数演示：不同 thread 的短期状态隔离，同一 user_id 的长期记忆共享。
def main() -> None:
    graph = build_graph()
    context = Context(user_id="user-1")

    # 两个 thread_id 表示两条独立会话，会得到两份不同的 checkpoint state。
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

    # get_state 验证 checkpointer 仍然按 thread_id 隔离短期 messages。
    print("\n检查两个 thread 的短期 messages：")
    print("Thread A:", graph.get_state(thread_a).values["messages"])
    print("Thread B:", graph.get_state(thread_b).values["messages"])


if __name__ == "__main__":
    main()
