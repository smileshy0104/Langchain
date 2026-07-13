"""
案例 5：Semantic search index shape

目标：
- 展示 InMemoryStore 开启 semantic search 的配置形状。
- 使用轻量 fake embedding，避免真实 API key。
- 理解 query 会根据 embedding 检索语义相近的长期记忆。

对应文档概念：
- Semantic Search
- InMemoryStore(index={...})
- Store search(query=...)
"""

from __future__ import annotations

from langgraph.store.memory import InMemoryStore


class TinyEmbeddings:
    """一个教学用 embedding：只为演示接口形状，不用于真实检索质量。"""

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            1.0 if any(word in lowered for word in ["hungry", "pizza", "food"]) else 0.0,
            1.0 if any(word in lowered for word in ["python", "code", "example"]) else 0.0,
            1.0 if any(word in lowered for word in ["dark", "theme", "mode"]) else 0.0,
            float(len(text) % 7) / 7.0,
        ]


def main() -> None:
    store = InMemoryStore(
        index={
            "embed": TinyEmbeddings(),
            "dims": 4,
            "fields": ["text"],
        }
    )
    namespace = ("users", "user-1", "memories")

    store.put(namespace, "food", {"text": "User loves pizza"})
    store.put(namespace, "coding", {"text": "User wants Python examples"})
    store.put(namespace, "ui", {"text": "User prefers dark mode"})

    print("Query: I'm hungry")
    for item in store.search(namespace, query="I'm hungry", limit=2):
        print("-", item.key, item.value)

    print("\nQuery: show me code")
    for item in store.search(namespace, query="show me code", limit=2):
        print("-", item.key, item.value)


if __name__ == "__main__":
    main()
