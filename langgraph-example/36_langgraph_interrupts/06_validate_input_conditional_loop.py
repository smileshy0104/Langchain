"""
案例 6：人工输入校验

目标：
- 每次节点调用只调用一次 interrupt()。
- 输入无效时，把新的 pending_question 写回 state。
- 用 conditional edge 回到同一个节点继续提问。

对应文档概念：
- Validating Human Input
- 避免 while True + interrupt()
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class FormState(TypedDict):
    # age 保存校验通过后的年龄。
    age: int | None
    # pending_question 保存下一次需要向用户展示的问题。
    pending_question: str | None


# 收集年龄：每次节点执行只 interrupt 一次，避免在 while True 中反复 interrupt。
def collect_age(state: FormState) -> dict:
    question = state.get("pending_question") or "What is your age?"
    # interrupt() 返回用户输入。
    answer = interrupt(question)

    # 输入有效时写入 age，并清空 pending_question。
    if isinstance(answer, int) and answer > 0:
        return {"age": answer, "pending_question": None}

    # 输入无效时不写 age，只更新下一次要提示的问题。
    return {
        "pending_question": (
            f"'{answer}' is not a valid age. Please enter a positive number."
        )
    }


# 条件路由：有 age 则结束，否则回到 collect_age 再次提问。
def route(state: FormState) -> Literal["collect_age", "__end__"]:
    return END if state.get("age") is not None else "collect_age"


# 构建带条件循环的 graph。
def build_graph():
    builder = StateGraph(FormState)
    builder.add_node("collect_age", collect_age)
    builder.add_edge(START, "collect_age")
    builder.add_conditional_edges("collect_age", route)
    return builder.compile(checkpointer=InMemorySaver())


# 主函数演示：第一次提问 -> 无效输入 -> 再次提问 -> 有效输入结束。
def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "validate-age"}}

    first = graph.stream_events(
        {"age": None, "pending_question": None},
        config=config,
        version="v3",
    )
    _ = first.output
    print("first prompt:", first.interrupts)

    # 恢复后传入无效答案，节点会写入 pending_question，并通过 conditional edge 回到自己。
    retry = graph.stream_events(Command(resume="thirty"), config=config, version="v3")
    _ = retry.output
    print("retry prompt:", retry.interrupts)

    # 再次恢复并传入有效整数，graph 结束。
    final = graph.stream_events(Command(resume=30), config=config, version="v3")
    print("final:", final.output)


if __name__ == "__main__":
    main()
