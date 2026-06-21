"""
案例 3：带记忆的 RAG 系统。

使用 Milvus 存储知识库文档，使用 InMemoryStore 记录用户偏好和反馈。
查询时先取用户偏好，再检索文档，最后让模型按偏好回答。
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
from retrieval_utils import create_milvus_vectorstore, format_docs


@dataclass
class UserContext:
    user_id: str


class MemoryEnhancedRAG:
    def __init__(self) -> None:
        self.store = InMemoryStore()
        self.vectorstore = create_milvus_vectorstore(
            collection_name="rm_memory_enhanced_rag_docs",
            texts=[
                "Python 是一种高级编程语言，语法简洁，适合 Web、自动化、数据分析和机器学习。",
                "机器学习是 AI 的一个分支，关注让模型从数据中学习规律并进行预测。",
                "TensorFlow 是 Google 开发的机器学习框架，适合生产部署和大规模训练。",
                "PyTorch 是 Meta 开发的深度学习框架，动态图易用，研究和原型开发体验好。",
                "RAG 通过检索外部知识增强生成模型，适合知识更新快或需要引用资料的应用。",
            ],
        )
        self.memory_agent = self._create_memory_agent()

    def _create_memory_agent(self) -> Any:
        @tool
        def get_user_preferences(runtime: ToolRuntime[UserContext]) -> str:
            """Get user answer preferences."""

            assert runtime.store is not None
            prefs = runtime.store.get(("users", runtime.context.user_id), "preferences")
            return str(prefs.value) if prefs else "暂无偏好设置"

        @tool
        def update_preference(key: str, value: str, runtime: ToolRuntime[UserContext]) -> str:
            """Update user answer preferences."""

            assert runtime.store is not None
            item = runtime.store.get(("users", runtime.context.user_id), "preferences")
            prefs = item.value if item else {}
            prefs[key] = value
            runtime.store.put(("users", runtime.context.user_id), "preferences", prefs)
            return f"已更新偏好：{key} = {value}"

        @tool
        def record_feedback(
            query: str,
            answer: str,
            rating: int,
            runtime: ToolRuntime[UserContext],
        ) -> str:
            """Record answer feedback."""

            assert runtime.store is not None
            runtime.store.put(
                ("users", runtime.context.user_id, "feedback"),
                datetime.now().isoformat(timespec="seconds"),
                {"query": query, "answer": answer, "rating": rating},
            )
            # 减少回答长度
            if rating <= 2:
                item = runtime.store.get(("users", runtime.context.user_id), "preferences")
                prefs = item.value if item else {}
                prefs.setdefault("avoid_styles", []).append("verbose")
                runtime.store.put(("users", runtime.context.user_id), "preferences", prefs)
            return "反馈已记录"

        return create_agent(
            model=create_configured_model(),
            tools=[get_user_preferences, update_preference, record_feedback],
            checkpointer=InMemorySaver(),
            store=self.store,
            context_schema=UserContext,
            system_prompt=(
                "你是记忆管理助手。用户表达回答风格、长度、偏好时，"
                "使用 update_preference 保存；用户评价答案时，使用 record_feedback。"
            ),
        )

    def query(self, question: str, user_id: str) -> str:
        prefs = self.store.get(("users", user_id), "preferences")
        user_preferences = str(prefs.value) if prefs else "标准回答风格"
        docs = self.vectorstore.similarity_search(question, k=4)
        context = format_docs(docs)
        model = create_configured_model()
        response = model.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "你是 RAG 问答助手。基于检索文档回答问题，并遵守用户偏好。"
                        "如果文档不足，说明限制。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"用户偏好：{user_preferences}\n\n"
                        f"相关文档：\n{context}\n\n"
                        f"问题：{question}"
                    ),
                },
            ]
        )
        return str(response.content)

    def chat(self, message: str, user_id: str, session_id: str) -> str:
        if "?" in message or "？" in message or "什么" in message or "区别" in message or "如何" in message:
            return self.query(message, user_id)

        result = self.memory_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": session_id}},
            context=UserContext(user_id=user_id),
        )
        return str(result["messages"][-1].content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieval & Memory 案例 3：带记忆的 RAG 系统")
    parser.add_argument("--user-id", default="user_001")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rag = MemoryEnhancedRAG()
    examples = [
        ("sess_001", "我喜欢简洁的回答"),
        ("sess_002", "什么是 TensorFlow？"),
        ("sess_003", "把回答长度改成详细版"),
        ("sess_004", "PyTorch 和 TensorFlow 有什么区别？"),
    ]
    for session_id, message in examples:
        print(f"\n用户: {message}")
        print(rag.chat(message, args.user_id, session_id))


if __name__ == "__main__":
    main()
