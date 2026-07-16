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
        """InMemoryStore 会把 embedding 对象当作可调用对象使用。"""

        return self.embed_documents(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量把文档文本转换成向量。"""

        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """把查询文本转换成 4 维向量；每一维用关键词粗略模拟语义。"""

        lowered = text.lower()
        return [
            # 第 1 维：食物相关。
            1.0 if any(word in lowered for word in ["hungry", "pizza", "food"]) else 0.0,
            # 第 2 维：编程相关。
            1.0 if any(word in lowered for word in ["python", "code", "example"]) else 0.0,
            # 第 3 维：UI/主题相关。
            1.0 if any(word in lowered for word in ["dark", "theme", "mode"]) else 0.0,
            # 第 4 维：为了让向量不完全相同，加入一个无业务意义的长度特征。
            float(len(text) % 7) / 7.0,
        ]


def main() -> None:
    # index 配置说明：embed 是向量模型，dims 是向量维度，fields 指定 value 中参与索引的字段。
    store = InMemoryStore(
        index={
            "embed": TinyEmbeddings(),
            "dims": 4,
            "fields": ["text"],
        }
    )
    namespace = ("users", "user-1", "memories")

    # 写入 value 时，text 字段会被用于构建语义索引。
    store.put(namespace, "food", {"text": "User loves pizza"})
    store.put(namespace, "coding", {"text": "User wants Python examples"})
    store.put(namespace, "ui", {"text": "User prefers dark mode"})

    print("Query: I'm hungry")
    # search(query=...) 会先对 query 做 embedding，再返回向量相近的记忆。
    for item in store.search(namespace, query="I'm hungry", limit=2):
        print("-", item.key, item.value)

    print("\nQuery: show me code")
    for item in store.search(namespace, query="show me code", limit=2):
        print("-", item.key, item.value)


if __name__ == "__main__":
    main()
