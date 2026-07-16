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
    # summary 是额外状态字段，用来保存被删除旧消息的摘要。
    summary: str


def _content(message: Any) -> str:
    """统一获取消息内容，方便拼接摘要和打印。"""

    return str(getattr(message, "content", message))


def summarize_then_delete(state: State) -> dict:
    """当消息过多时，把较早消息写入 summary，并用 RemoveMessage 删除原消息。"""

    messages = state["messages"]
    # 示例阈值：消息不超过 4 条时不做摘要/删除。
    if len(messages) <= 4:
        return {}

    # 保留最后 2 条消息，把更早的消息压缩进 summary。
    old_messages = messages[:-2]
    old_text = " | ".join(_content(message) for message in old_messages)
    previous_summary = state.get("summary", "")
    summary = f"{previous_summary} {old_text}".strip()

    return {
        "summary": summary,
        # RemoveMessage 会根据 message.id 从 MessagesState 中永久删除对应消息。
        "messages": [RemoveMessage(id=message.id) for message in old_messages],
    }


def call_model_with_trimmed_input(state: State) -> dict:
    """演示 trim_messages：仅裁剪模型输入，不改变 state 中保存的 messages。"""

    trimmed = trim_messages(
        state["messages"],
        strategy="last",  # 保留最后的消息。
        token_counter=count_tokens_approximately,  # 使用近似 token 计数器，避免依赖真实模型 tokenizer。
        max_tokens=80,
        start_on="human",  # 裁剪后的消息尽量从 human 开始。
        end_on=("human", "ai"),  # 裁剪后的消息可以以 human 或 ai 结束。
    )
    prompt_preview = [_content(message) for message in trimmed]
    answer = (
        f"summary={state.get('summary', '') or '无'}; "
        f"model_input={prompt_preview}"
    )
    return {"messages": [AIMessage(content=answer)]}


def build_graph():
    """构建先管理短期记忆、再调用模型的两节点图。"""

    builder = StateGraph(State)
    builder.add_node("summarize_then_delete", summarize_then_delete)
    builder.add_node("call_model", call_model_with_trimmed_input)
    builder.add_edge(START, "summarize_then_delete")
    builder.add_edge("summarize_then_delete", "call_model")
    builder.add_edge("call_model", END)

    # checkpointer 保存修改后的 messages 和 summary，方便后续轮次继续使用。
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "trim-delete-summary"}}

    # 显式设置 id，RemoveMessage 才能准确删除指定旧消息。
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

    # 查看 checkpointer 中最终保存的状态：旧消息已删除，关键信息进入 summary。
    snapshot = graph.get_state(config)
    print("\nstate summary:", snapshot.values.get("summary"))
    print("remaining messages:")
    for message in snapshot.values["messages"]:
        print("-", message.type, message.content)


if __name__ == "__main__":
    main()
