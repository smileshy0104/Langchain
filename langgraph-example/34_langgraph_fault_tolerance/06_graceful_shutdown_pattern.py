"""
案例 6：Graceful Shutdown 模式

目标：
- 展示 RunControl / GraphDrained 的合作式停机模板。
- request_drain() 请求 graph 在 superstep 边界停止。
- 后续可使用同一个 config 通过 graph.invoke(None, config) 恢复。

对应文档概念：
- Graceful Shutdown
- RunControl
- GraphDrained

注意：
- 这是停机模式模板。实际 SIGTERM hook 通常由进程管理器触发。
"""

from __future__ import annotations

import signal
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphDrained
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import RunControl, Runtime


class JobState(TypedDict):
    # step 记录当前执行到哪个步骤。
    step: str
    # status 记录任务状态或停机原因。
    status: str


# 第一步：正常完成，并把进度写入 state/checkpoint。
def step_a(state: JobState) -> dict:
    return {"step": "a", "status": "step_a_done"}


# 第二步：在执行前检查是否收到 drain 请求。
def step_b(state: JobState, runtime: Runtime) -> dict:
    # drain_requested 表示外部希望 graph 在安全边界停止。
    if runtime.drain_requested:
        return {
            "step": "b",
            "status": f"skipped_due_to_drain:{runtime.drain_reason}",
        }

    return {"step": "b", "status": "step_b_done"}


# 构建一个两步 graph；checkpointer 用于保存可恢复的进度。
def build_graph():
    builder = StateGraph(JobState)
    builder.add_node("step_a", step_a)
    builder.add_node("step_b", step_b)
    builder.add_edge(START, "step_a")
    builder.add_edge("step_a", "step_b")
    builder.add_edge("step_b", END)
    return builder.compile(checkpointer=InMemorySaver())


# 主函数演示合作式停机：收到 drain 请求后安全退出，未来可用同一 config 恢复。
def main() -> None:
    graph = build_graph()

    # thread_id 对应一条可恢复的执行线程；恢复时必须复用同一个 config。
    config = {"configurable": {"thread_id": "drain-demo"}}

    # RunControl 是外部控制对象，可被 signal handler 或其它控制面调用。
    control = RunControl()

    # 注册 SIGTERM 处理器：真实服务里通常在收到终止信号时请求 drain。
    signal.signal(
        signal.SIGTERM,
        lambda *_: control.request_drain("sigterm"),
    )

    # 示例中主动请求 drain，模拟服务即将下线。
    control.request_drain("demo-drain-request")

    try:
        result = graph.invoke(
            {"step": "", "status": "started"},
            config,
            control=control,
        )
        print("graph finished naturally:", result)
    except GraphDrained as exc:
        # 如果 graph 在安全边界抛出 GraphDrained，可稍后用 checkpoint 恢复。
        print(f"graph drained safely: {exc.reason}")
        print("resume later with:")
        print("graph.invoke(None, config)")


if __name__ == "__main__":
    main()
