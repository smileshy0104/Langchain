"""
案例 5：Checkpoint history、Replay 与 Fork

目标：
- 用 get_state_history 查看 thread 的历史 checkpoints。
- 用 state.next 找到某个节点执行前的 checkpoint。
- 用 invoke(None, checkpoint.config) 从旧 checkpoint replay。
- 用 update_state 基于旧 checkpoint 创建 fork。

对应文档概念：
- StateSnapshot
- checkpoint_id
- state.next
- time travel
- update_state
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class DraftState(TypedDict):
    # NotRequired 表示初始输入可以为空，节点会逐步补齐这些字段。
    topic: NotRequired[str]
    draft: NotRequired[str]


# 第一个节点：确定写作主题。如果调用方未传入 topic，就使用默认主题。
def choose_topic(state: DraftState) -> dict:
    return {"topic": state.get("topic", "LangGraph Persistence")}


# 第二个节点：根据 topic 生成草稿。
def write_draft(state: DraftState) -> dict:
    topic = state["topic"]
    return {"draft": f"这是一段关于 {topic} 的草稿。"}


# 构建一个两步 graph，用于产生多个 checkpoint，便于演示 replay/fork。
def build_graph():
    builder = StateGraph(DraftState)
    builder.add_node("choose_topic", choose_topic)
    builder.add_node("write_draft", write_draft)
    builder.add_edge(START, "choose_topic")
    builder.add_edge("choose_topic", "write_draft")
    builder.add_edge("write_draft", END)

    # 只有带 checkpointer 的 graph 才能查询历史、回放和从旧 checkpoint 分叉。
    return builder.compile(checkpointer=InMemorySaver())


# 打印当前 thread 的所有历史快照，帮助观察 next 与 checkpoint_id。
def print_history(graph, config) -> None:
    print("\nCheckpoint history：")
    for snapshot in graph.get_state_history(config):
        # checkpoint_id 是历史快照的唯一标识；config 可直接用于从该点继续执行。
        checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]

        # snapshot.next 表示从该 checkpoint 继续时，下一个会执行的节点。
        print(f"- next={snapshot.next}, checkpoint_id={checkpoint_id}")


# 主函数演示 checkpoint 历史、从旧点 replay，以及基于旧点 fork。
def main() -> None:
    graph = build_graph()

    # 所有 checkpoint 都归属到同一个 thread_id 下，形成一条可查询的历史链。
    config = {"configurable": {"thread_id": "draft-thread"}}

    print("第一次运行 graph")
    result = graph.invoke({}, config)
    print(result)

    # get_state_history 返回从新到旧的 StateSnapshot 列表。
    history = list(graph.get_state_history(config))
    print_history(graph, config)

    # 找到“已经执行完 choose_topic，但还没执行 write_draft”的 checkpoint。
    # 从该点继续时，next 会指向 write_draft。
    before_write = next(
        snapshot
        for snapshot in history
        if snapshot.next == ("write_draft",)
    )

    print("\nReplay：从 write_draft 前的 checkpoint 继续")
    # invoke(None, checkpoint.config) 表示不提供新输入，直接从旧 checkpoint 继续执行。
    # 这里不会重跑 choose_topic，只会继续执行 next 中的 write_draft。
    replay_result = graph.invoke(None, before_write.config)
    print(replay_result)

    print("\nFork：从 write_draft 前的 checkpoint 改写 topic，再继续")
    # update_state 会基于旧 checkpoint 写入一份新状态，产生一个新的分支配置。
    # as_node 告诉 LangGraph 这次状态更新应当归因于哪个节点，影响后续路由/历史记录。
    fork_config = graph.update_state(
        before_write.config,
        values={"topic": "LangGraph Stores"},
        as_node="choose_topic",
    )

    # 用 fork_config 继续执行，得到与原历史不同的新 draft。
    fork_result = graph.invoke(None, fork_config)
    print(fork_result)

    print("\n说明：")
    print("- Replay 不会重跑 choose_topic，但会重跑 write_draft。")
    print("- Fork 不会删除原历史，而是从旧 checkpoint 创建一个新分支。")


if __name__ == "__main__":
    main()
