"""
案例 4：Access Store inside nodes

目标：
- 通过 context_schema 定义运行时上下文。
- 在 node 中用 runtime.context.user_id 定位用户。
- 在 node 中用 runtime.store.search/put 读写长期记忆。

对应文档概念：
- Runtime
- runtime.context
- runtime.store
- Access Store Inside Nodes
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore


@dataclass
class Context:
    user_id: str


def _content(message) -> str:
    return str(getattr(message, "content", message))


def profile_assistant(state: MessagesState, runtime: Runtime[Context]) -> dict:
    latest = _content(state["messages"][-1])
    namespace = ("memories", runtime.context.user_id)

    if latest.lower().startswith("remember "):
        memory_text = latest.removeprefix("remember ").strip()
        runtime.store.put(namespace, str(uuid.uuid4()), {"data": memory_text})

    memories = runtime.store.search(namespace, limit=3)
    memory_texts = [item.value["data"] for item in memories]

    answer = (
        "本次回答会显式使用 Store 检索到的长期记忆："
        f"{memory_texts or '暂无长期记忆'}"
    )
    return {"messages": [AIMessage(content=answer)]}


def build_graph():
    builder = StateGraph(MessagesState, context_schema=Context)
    builder.add_node("profile_assistant", profile_assistant)
    builder.add_edge(START, "profile_assistant")
    builder.add_edge("profile_assistant", END)
    return builder.compile(store=InMemoryStore())


def main() -> None:
    graph = build_graph()
    context = Context(user_id="user-1")

    graph.invoke(
        {"messages": [HumanMessage(content="remember 我喜欢简洁的 Markdown 总结")]},
        context=context,
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="我的输出偏好是什么？")]},
        context=context,
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
