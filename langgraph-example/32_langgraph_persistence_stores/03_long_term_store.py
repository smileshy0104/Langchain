"""
案例 3：长期记忆 Store

目标：
- 使用 InMemoryStore 保存跨 thread 的长期记忆。
- 使用 namespace 隔离不同用户的数据。
- 在 node 中通过 Runtime 访问 runtime.store。

对应文档概念：
- Store
- namespace
- key/value
- long-term memory
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


@dataclass
class Context:
    user_id: str


class MemoryState(TypedDict):
    message: str
    answer: str


def memory_node(state: MemoryState, runtime: Runtime[Context]) -> dict:
    namespace = ("users", runtime.context.user_id, "memories")
    message = state["message"]

    if message.startswith("remember:"):
        memory_text = message.removeprefix("remember:").strip()
        runtime.store.put(
            namespace,
            str(uuid.uuid4()),
            {"text": memory_text, "source": "user"},
        )
        answer = f"已写入长期记忆：{memory_text}"
    else:
        memories = runtime.store.search(namespace)
        memory_lines = [item.value["text"] for item in memories]
        answer = "该用户长期记忆：" + (", ".join(memory_lines) or "暂无")

    return {"answer": answer}


def build_graph(store: InMemoryStore):
    builder = StateGraph(MemoryState, context_schema=Context)
    builder.add_node("memory", memory_node)
    builder.add_edge(START, "memory")
    builder.add_edge("memory", END)
    return builder.compile(store=store)


def main() -> None:
    store = InMemoryStore()
    graph = build_graph(store)

    config_1 = {"configurable": {"thread_id": "thread-1"}}
    config_2 = {"configurable": {"thread_id": "thread-2"}}
    context = Context(user_id="user-1")

    print("第一条 thread：写入长期记忆")
    result = graph.invoke(
        {"message": "remember: 用户喜欢深色模式"},
        config_1,
        context=context,
    )
    print(result["answer"])

    print("\n第二条 thread：读取同一个 user_id 的长期记忆")
    result = graph.invoke(
        {"message": "show memories"},
        config_2,
        context=context,
    )
    print(result["answer"])

    print("\n换一个 user_id：namespace 不同，因此读不到 user-1 的记忆")
    result = graph.invoke(
        {"message": "show memories"},
        {"configurable": {"thread_id": "thread-3"}},
        context=Context(user_id="user-2"),
    )
    print(result["answer"])


if __name__ == "__main__":
    main()
