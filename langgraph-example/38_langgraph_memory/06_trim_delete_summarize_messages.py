"""
案例 6：Trim / Delete / Summarize messages

目标：
- trim_messages：只裁剪传给模型的输入，不永久修改 state。
- RemoveMessage：从 MessagesState 中永久删除旧消息。
- summary：把早期消息压缩到 summary 字段，保留关键信息。

对应文档概念：
- Manage Short-term Memory
- Trim Messages
- Delete Messages
- Summarize Messages
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph


class State(MessagesState):
    summary: str


def _content(message: Any) -> str:
    return str(getattr(message, "content", message))


def summarize_then_delete(state: State) -> dict:
    messages = state["messages"]
    if len(messages) <= 4:
        return {}

    old_messages = messages[:-2]
    old_text = " | ".join(_content(message) for message in old_messages)
    previous_summary = state.get("summary", "")
    summary = f"{previous_summary} {old_text}".strip()

    return {
        "summary": summary,
        "messages": [RemoveMessage(id=message.id) for message in old_messages],
    }


def call_model_with_trimmed_input(state: State) -> dict:
    trimmed = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=80,
        start_on="human",
        end_on=("human", "ai"),
    )
    prompt_preview = [_content(message) for message in trimmed]
    answer = (
        f"summary={state.get('summary', '') or '无'}; "
        f"model_input={prompt_preview}"
    )
    return {"messages": [AIMessage(content=answer)]}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("summarize_then_delete", summarize_then_delete)
    builder.add_node("call_model", call_model_with_trimmed_input)
    builder.add_edge(START, "summarize_then_delete")
    builder.add_edge("summarize_then_delete", "call_model")
    builder.add_edge("call_model", END)
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "trim-delete-summary"}}

    initial_messages = [
        HumanMessage(content="第 1 轮：我叫 Bob", id="h1"),
        AIMessage(content="你好 Bob", id="a1"),
        HumanMessage(content="第 2 轮：我喜欢 Python", id="h2"),
        AIMessage(content="我会记住这个偏好", id="a2"),
        HumanMessage(content="第 3 轮：请给我案例", id="h3"),
    ]

    result = graph.invoke(
        {"messages": initial_messages, "summary": ""},
        config,
    )
    print("latest answer:", result["messages"][-1].content)

    snapshot = graph.get_state(config)
    print("\nstate summary:", snapshot.values.get("summary"))
    print("remaining messages:")
    for message in snapshot.values["messages"]:
        print("-", message.type, message.content)


if __name__ == "__main__":
    main()
