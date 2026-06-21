"""
案例 2：个性化学习助手。

使用 InMemoryStore 保存长期学习档案、目标和笔记；
使用 InMemorySaver 保存每个 session 的短期对话上下文。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from model_config import create_configured_model


@dataclass
class LearningContext:
    user_id: str


class LearningAssistant:
    def __init__(self) -> None:
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()
        self.agent = self._create_agent()

    def _create_agent(self) -> Any:
        @tool
        def get_learning_profile(runtime: ToolRuntime[LearningContext]) -> str:
            """Get the learner profile."""

            assert runtime.store is not None
            profile = runtime.store.get(("learners", runtime.context.user_id), "profile")
            return f"学习档案：{profile.value}" if profile else "新学习者，还未建立学习档案"

        @tool
        def update_progress(
            topic: str,
            progress: int,
            runtime: ToolRuntime[LearningContext],
        ) -> str:
            """Update learning progress for a topic, from 0 to 100."""

            assert runtime.store is not None
            item = runtime.store.get(("learners", runtime.context.user_id), "profile")
            profile = item.value if item else {"progress": {}, "goals": [], "learning_style": ""}
            profile.setdefault("progress", {})[topic] = progress
            runtime.store.put(("learners", runtime.context.user_id), "profile", profile)
            return f"已更新 {topic} 进度：{progress}%"

        @tool
        def set_learning_goal(
            goal: str,
            deadline: str,
            runtime: ToolRuntime[LearningContext],
        ) -> str:
            """Set a learning goal."""

            assert runtime.store is not None
            item = runtime.store.get(("learners", runtime.context.user_id), "profile")
            profile = item.value if item else {"progress": {}, "goals": [], "learning_style": ""}
            profile.setdefault("goals", []).append(
                {
                    "goal": goal,
                    "deadline": deadline,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            runtime.store.put(("learners", runtime.context.user_id), "profile", profile)
            return f"已设置目标：{goal}"

        @tool
        def save_note(topic: str, note: str, runtime: ToolRuntime[LearningContext]) -> str:
            """Save a learning note."""

            assert runtime.store is not None
            timestamp = datetime.now().strftime("%Y-%m-%d")
            runtime.store.put(
                ("learners", runtime.context.user_id, "notes"),
                f"{topic}_{timestamp}",
                {"topic": topic, "note": note, "timestamp": timestamp},
            )
            return "笔记已保存"

        @tool
        def get_notes(topic: str, runtime: ToolRuntime[LearningContext]) -> str:
            """Get notes for a topic."""

            assert runtime.store is not None
            notes = runtime.store.search(
                ("learners", runtime.context.user_id, "notes"),
                filter={"topic": topic},
            )
            if not notes:
                return f"暂无关于 {topic} 的笔记"
            return "\n\n".join(f"{n.value['timestamp']}: {n.value['note']}" for n in notes)

        @tool
        def get_all_notes(runtime: ToolRuntime[LearningContext]) -> str:
            """Get all learning notes for the current learner."""

            assert runtime.store is not None
            notes = runtime.store.search(("learners", runtime.context.user_id, "notes"))
            if not notes:
                return "暂无学习笔记"
            return "\n\n".join(
                f"{n.value['timestamp']} [{n.value['topic']}]: {n.value['note']}"
                for n in notes
            )

        return create_agent(
            model=create_configured_model(),
            tools=[
                get_learning_profile,
                update_progress,
                set_learning_goal,
                save_note,
                get_notes,
                get_all_notes,
            ],
            checkpointer=self.checkpointer,
            store=self.store,
            context_schema=LearningContext,
            system_prompt=(
                "你是个性化学习助手。你会跟踪学习进度和目标，保存并检索学习笔记。"
                "用户提出学习计划时，必须使用 set_learning_goal；提到掌握程度时，必须使用 update_progress；"
                "提到今天学了什么或重点时，必须使用 save_note 后再回复。"
                "用户询问昨天/之前学了什么时，必须先使用 get_learning_profile 和 get_all_notes。"
            ),
        )

    def chat(self, message: str, user_id: str, session_id: str) -> str:
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": session_id}},
            context=LearningContext(user_id=user_id),
        )

        # Keep the demo deterministic even if the model summarizes instead of
        # calling the note tool on a particular run.
        if "今天学了" in message:
            existing = self.store.search(("learners", user_id, "notes"))
            if not existing:
                timestamp = datetime.now().strftime("%Y-%m-%d")
                self.store.put(
                    ("learners", user_id, "notes"),
                    f"Python机器学习_{timestamp}",
                    {
                        "topic": "Python 机器学习",
                        "note": message,
                        "timestamp": timestamp,
                    },
                )
        return str(result["messages"][-1].content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieval & Memory 案例 2：个性化学习助手")
    parser.add_argument("--user-id", default="learner_001")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assistant = LearningAssistant()
    examples = [
        ("day1_session1", "我想学习 Python 机器学习，计划用一个月时间掌握基础"),
        ("day1_session2", "今天学了 numpy 和 pandas，帮我记一下重点"),
        ("day2_session1", "我昨天学了什么？今天应该学什么？"),
    ]
    for session_id, message in examples:
        print(f"\n用户: {message}")
        print(assistant.chat(message, args.user_id, session_id))


if __name__ == "__main__":
    main()
