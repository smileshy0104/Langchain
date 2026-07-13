"""
案例 1：短期记忆与 checkpointer

目标：
- 使用 InMemorySaver 保存 thread 级 graph state。
- 使用同一个 thread_id 进行多轮调用。
- 观察第二轮调用能读到第一轮保存的 messages。

对应文档概念：
- Checkpointer
- thread_id
- short-term memory
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


# ChatState 描述 graph 在每次执行时会维护的状态结构。
class ChatState(TypedDict):
    # Annotated + operator.add 表示多次写入 messages 时采用“追加”语义，
    # 而不是用新列表覆盖旧列表。这正是对话历史能够累积的关键。
    messages: Annotated[list[str], operator.add]
    # answer 保存本轮 assistant_node 生成的最终回答，方便调用方读取。
    answer: str


# assistant_node 是图中的唯一业务节点：从历史 messages 中读取用户名字并生成回复。
def assistant_node(state: ChatState) -> dict:
    # checkpointer 恢复出的历史消息和本轮输入会先合并到 state["messages"] 中。
    messages = state.get("messages", [])

    # 从后往前找到最近一条用户消息，作为本轮需要处理的输入。
    latest_user = next(
        (message for message in reversed(messages) if message.startswith("user:")),
        "user:",
    )

    # 在整段 thread 历史中寻找“我叫...”这类自我介绍。
    # 这里用一个简单规则模拟真实 Agent 从短期记忆中抽取信息。
    known_name = None
    for message in messages:
        if message.startswith("user: 我叫"):
            known_name = message.removeprefix("user: 我叫").strip(" 。.")

    # 如果用户询问自己的名字，就优先使用当前 thread 中保存过的名字。
    if "我叫什么" in latest_user and known_name:
        answer = f"你叫 {known_name}。这是从同一个 thread 的历史消息里读到的。"
    elif known_name:
        answer = f"我记住了，你叫 {known_name}。"
    else:
        answer = "我还没有在这个 thread 中记住你的名字。"

    # 返回的 messages 会根据 ChatState 中的 reducer 追加到历史消息末尾，
    # answer 则覆盖为本轮回答。
    return {
        "messages": [f"assistant: {answer}"],
        "answer": answer,
    }


# 构建并编译 LangGraph。
def build_graph():
    builder = StateGraph(ChatState)
    builder.add_node("assistant", assistant_node)
    builder.add_edge(START, "assistant")
    builder.add_edge("assistant", END)

    # InMemorySaver 是最小化的内存 checkpointer：
    # 它按 thread_id 保存 checkpoint，进程结束后数据会丢失。
    return builder.compile(checkpointer=InMemorySaver())


# 主函数演示：同一个 thread_id 下的两次 invoke 会共享短期状态。
def main() -> None:
    graph = build_graph()

    # config.configurable.thread_id 是 checkpointer 定位会话状态的游标。
    config = {"configurable": {"thread_id": "conversation-1"}}

    print("第一轮：写入当前 thread 的短期记忆")
    first = graph.invoke(
        {"messages": ["user: 我叫 Bob。"]},
        config,
    )
    print(first["answer"])

    print("\n第二轮：复用同一个 thread_id，读取上一轮状态")
    second = graph.invoke(
        {"messages": ["user: 我叫什么名字？"]},
        config,
    )
    print(second["answer"])

    # get_state 可以查看当前 thread 最新 checkpoint 对应的状态快照。
    snapshot = graph.get_state(config)
    print("\n当前 checkpoint 中保存的 messages：")
    for message in snapshot.values["messages"]:
        print("-", message)


if __name__ == "__main__":
    main()
