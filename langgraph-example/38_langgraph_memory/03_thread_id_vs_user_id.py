"""
案例 3：thread_id vs user_id

目标：
- thread_id 用于短期记忆：区分不同会话。
- user_id 用于长期记忆：不同 thread 可共享同一个用户资料。
- checkpointer 和 store 可以同时配置。

对应文档概念：
- Short-term memory + Long-term memory
- thread_id vs user_id
- compile(checkpointer=..., store=...)
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


def assistant(state: AssistantState, runtime: Runtime[Context]) -> dict:
    latest = state["messages"][-1]
    namespace = ("users", runtime.context.user_id, "memories")

    if latest.startswith("remember:"):
        memory = latest.removeprefix("remember:").strip()
        runtime.store.put(namespace, str(uuid.uuid4()), {"text": memory})

    memories = [item.value["text"] for item in runtime.store.search(namespace)]
    answer = (
        f"thread messages={len(state['messages'])}; "
        f"user memories={memories or '暂无'}"
    )
    return {"messages": [f"assistant: {answer}"], "answer": answer}


def build_graph():
    builder = StateGraph(AssistantState, context_schema=Context)
    builder.add_node("assistant", assistant)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)
    return builder.compile(
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
    )


def main() -> None:
    graph = build_graph()
    context = Context(user_id="user-1")

    thread_a = {"configurable": {"thread_id": "thread-a"}}
    thread_b = {"configurable": {"thread_id": "thread-b"}}

    print("Thread A 写入长期记忆：")
    print(
        graph.invoke(
            {"messages": ["remember: 用户喜欢 Python 示例"]},
            thread_a,
            context=context,
        )["answer"]
    )

    print("\nThread B 是新会话，但同一个 user_id 能读取长期记忆：")
    print(
        graph.invoke(
            {"messages": ["请根据我的偏好回答"]},
            thread_b,
            context=context,
        )["answer"]
    )

    print("\n两个 thread 的短期 state 仍然不同：")
    print("thread-a:", graph.get_state(thread_a).values["messages"])
    print("thread-b:", graph.get_state(thread_b).values["messages"])


if __name__ == "__main__":
    main()
