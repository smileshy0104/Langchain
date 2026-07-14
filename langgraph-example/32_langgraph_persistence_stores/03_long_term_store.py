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
from ftplib import print_line
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


@dataclass
class Context:
    # context 是运行时上下文，不属于 graph state；适合放 user_id、tenant_id 等外部身份信息。
    user_id: str


class MemoryState(TypedDict):
    # message 是本轮用户输入；本示例没有 checkpointer，因此不会自动保留多轮 state。
    message: str
    answer: str


# memory_node 通过 runtime.store 操作长期记忆，而不是通过 state 保存短期对话。
def memory_node(state: MemoryState, runtime: Runtime[Context]) -> dict:
    # namespace 用元组表达层级：所有 user 记忆 / 具体 user_id / memories 集合。
    # 相同 user_id 在不同 thread 中使用同一个 namespace，因此可以共享长期记忆。
    namespace = ("users", runtime.context.user_id, "memories")
    message = state["message"]

    if message.startswith("remember:"):
        # 将 remember: 后面的文本写入 Store。key 使用 uuid，避免覆盖已有记忆。
        memory_text = message.removeprefix("remember:").strip()
        runtime.store.put(
            namespace,
            str(uuid.uuid4()),
            {"text": memory_text, "source": "user"},
        )
        answer = f"已写入长期记忆：{memory_text}"
    else:
        # search 会在当前 namespace 下查找所有长期记忆项。
        memories = runtime.store.search(namespace)
        memory_lines = [item.value["text"] for item in memories]
        answer = "该用户长期记忆：" + (", ".join(memory_lines) or "暂无")

    return {"answer": answer}


# 编译 graph 时注入 store，节点才能通过 runtime.store 访问它。
def build_graph(store: InMemoryStore):
    builder = StateGraph(MemoryState, context_schema=Context)
    builder.add_node("memory", memory_node)
    builder.add_edge(START, "memory")
    builder.add_edge("memory", END)
    return builder.compile(store=store)


# 主函数演示：Store 按 user_id namespace 跨 thread 保存长期记忆。
def main() -> None:
    # InMemoryStore 仅用于本地学习；生产环境一般替换成持久化 Store backend。
    store = InMemoryStore()
    graph = build_graph(store)

    # thread_id 仍然可以传入 config，但本例没有 checkpointer，核心隔离维度是 user_id namespace。
    config_1 = {"configurable": {"thread_id": "thread-1"}}
    config_2 = {"configurable": {"thread_id": "thread-2"}}
    context = Context(user_id="user-1")

    print_line("长期记忆 Store")
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
