"""
案例 2：invoke API 中处理 interrupt

目标：
- 不使用 Event Streaming，直接 graph.invoke()。
- interrupt payload 会出现在 result["__interrupt__"]。
- 用 Command(resume=...) 恢复。

对应文档概念：
- invoke API 中处理 Interrupt
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class FormState(TypedDict):
    # name 首次运行时未知；恢复后由人工输入写入。
    name: str | None
    # greeting 保存最终问候语。
    greeting: str


# ask_name 节点通过 interrupt() 向用户提问。
def ask_name(state: FormState) -> dict:
    # 在 invoke API 中，interrupt payload 会被放进返回结果的 __interrupt__ 字段。
    name = interrupt("What is your name?")
    return {"name": name}


# greet 节点使用恢复后写入的 name 生成问候。
def greet(state: FormState) -> dict:
    return {"greeting": f"Hello, {state['name']}!"}


# 构建带 checkpointer 的 graph，用于保存中断点。
def build_graph():
    builder = StateGraph(FormState)
    builder.add_node("ask_name", ask_name)
    builder.add_node("greet", greet)
    builder.add_edge(START, "ask_name")
    builder.add_edge("ask_name", "greet")
    builder.add_edge("greet", END)
    return builder.compile(checkpointer=InMemorySaver())


# 主函数演示非 streaming 的 interrupt 处理方式。
def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "invoke-interrupt"}}

    # 第一次 invoke 会在 ask_name 中暂停，不会继续到 greet。
    result = graph.invoke({"name": None, "greeting": ""}, config)
    print("first invoke result:", result)

    # invoke 返回的 __interrupt__ 中包含需要展示给人的问题或 payload。
    if "__interrupt__" in result:
        print("interrupt payload:", result["__interrupt__"])

    # 使用 Command(resume="Alice") 恢复，"Alice" 成为 ask_name 中 interrupt() 的返回值。
    resumed = graph.invoke(Command(resume="Alice"), config)
    print("resumed result:", resumed)


if __name__ == "__main__":
    main()
