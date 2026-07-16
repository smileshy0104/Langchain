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
    # 运行时上下文字段：用于决定当前节点应该访问哪个用户的长期记忆。
    user_id: str


def _content(message) -> str:
    """从 HumanMessage/AIMessage 或普通对象中取文本内容。"""

    return str(getattr(message, "content", message))


def profile_assistant(state: MessagesState, runtime: Runtime[Context]) -> dict:
    """在节点内部通过 runtime.store 读写长期记忆。"""

    latest = _content(state["messages"][-1])

    # namespace 通常包含业务域和 user_id，确保不同用户的长期记忆隔离。
    namespace = ("memories", runtime.context.user_id)

    # 简化示例：以 remember 开头的消息会被当作需要保存的长期记忆。
    if latest.lower().startswith("remember "):
        memory_text = latest.removeprefix("remember ").strip()
        runtime.store.put(namespace, str(uuid.uuid4()), {"data": memory_text})

    # 节点可以直接检索 Store；limit=3 表示最多取 3 条相关/最近记忆。
    memories = runtime.store.search(namespace, limit=3)
    memory_texts = [item.value["data"] for item in memories]

    answer = (
        "本次回答会显式使用 Store 检索到的长期记忆："
        f"{memory_texts or '暂无长期记忆'}"
    )
    # MessagesState 的 messages 会追加这条 AIMessage，作为本轮输出。
    return {"messages": [AIMessage(content=answer)]}


def build_graph():
    """构建可以访问 runtime.context 和 runtime.store 的图。"""

    builder = StateGraph(MessagesState, context_schema=Context)
    builder.add_node("profile_assistant", profile_assistant)
    builder.add_edge(START, "profile_assistant")
    builder.add_edge("profile_assistant", END)

    # 这里只配置 store，不配置 checkpointer：长期记忆可用，但不会跨轮保存 messages 短期历史。
    return builder.compile(store=InMemoryStore())


def main() -> None:
    graph = build_graph()
    context = Context(user_id="user-1")

    # 第一次调用写入长期记忆。
    graph.invoke(
        {"messages": [HumanMessage(content="remember 我喜欢简洁的 Markdown 总结")]},
        context=context,
    )
    # 第二次调用读取同一 user_id namespace 下的长期记忆。
    result = graph.invoke(
        {"messages": [HumanMessage(content="我的输出偏好是什么？")]},
        context=context,
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
