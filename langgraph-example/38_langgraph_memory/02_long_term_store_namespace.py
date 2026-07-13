"""
案例 2：Long-term memory with Store namespace

目标：
- 使用 InMemoryStore 保存长期记忆。
- 理解 namespace + key + value 的组织方式。
- 展示 user / org / project 等 namespace 隔离。

对应文档概念：
- Long-term memory
- Store
- namespace
- key/value
"""

from __future__ import annotations

from langgraph.store.memory import InMemoryStore


def print_items(title: str, items) -> None:
    print(title)
    for item in items:
        print(f"- key={item.key}, value={item.value}")


def main() -> None:
    store = InMemoryStore()

    user_1_memories = ("users", "user-1", "memories")
    user_2_memories = ("users", "user-2", "memories")
    project_facts = ("projects", "langgraph-study", "facts")

    store.put(user_1_memories, "preference-language", {"text": "用户偏好中文回答"})
    store.put(user_1_memories, "preference-style", {"text": "用户喜欢代码案例"})
    store.put(user_2_memories, "preference-language", {"text": "User prefers English"})
    store.put(project_facts, "goal", {"text": "整理 LangGraph 学习案例"})

    print_items("user-1 memories:", store.search(user_1_memories))
    print_items("\nuser-2 memories:", store.search(user_2_memories))
    print_items("\nproject facts:", store.search(project_facts))

    one_item = store.get(user_1_memories, "preference-language")
    print("\nget one item:", one_item.value if one_item else None)


if __name__ == "__main__":
    main()
