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
    # user_id 是运行时上下文，不属于 graph state；常用于定位用户级长期记忆。
    user_id: str


class AssistantState(TypedDict):
    # messages 是短期会话内容；operator.add 让同一 thread 的多轮消息累加。
    messages: Annotated[list[str], operator.add]
    # answer 保存本轮助手回复，方便 main 中直接打印。
    answer: str


def assistant(state: AssistantState, runtime: Runtime[Context]) -> dict:
    """同时演示短期记忆 thread_id 和长期记忆 user_id 的节点。"""

    latest = state["messages"][-1]

    # 长期记忆按 user_id 组织：同一个用户的不同 thread 可以共享这些数据。
    namespace = ("users", runtime.context.user_id, "memories")

    # 用 remember: 作为简单协议：本轮输入以该前缀开头时，把内容写入长期记忆。
    if latest.startswith("remember:"):
        memory = latest.removeprefix("remember:").strip()
        runtime.store.put(namespace, str(uuid.uuid4()), {"text": memory})

    # 每轮回答前都从 Store 中检索当前用户的长期记忆。
    memories = [item.value["text"] for item in runtime.store.search(namespace)]
    answer = (
        f"thread messages={len(state['messages'])}; "
        f"user memories={memories or '暂无'}"
    )

    # messages 追加助手输出；answer 则覆盖为本轮最终结果。
    return {"messages": [f"assistant: {answer}"], "answer": answer}


def build_graph():
    """构建同时拥有 checkpointer 和 store 的图。"""

    # context_schema 声明 runtime.context 的类型，节点即可安全访问 runtime.context.user_id。
    builder = StateGraph(AssistantState, context_schema=Context)
    builder.add_node("assistant", assistant)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)
    return builder.compile(
        # checkpointer 保存 thread 级短期 state。
        checkpointer=InMemorySaver(),
        # store 保存 user 级长期记忆。
        store=InMemoryStore(),
    )


def main() -> None:
    graph = build_graph()

    # 同一个 context/user_id 会共享长期记忆。
    context = Context(user_id="user-1")

    # 两个不同 thread_id 代表两个短期会话，messages 历史互相隔离。
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
