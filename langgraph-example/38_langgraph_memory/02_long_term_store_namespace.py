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
    """打印 Store.search 返回的 item，便于观察 namespace 内的数据。"""

    print(title)
    for item in items:
        # item.key 是 namespace 内的唯一键；item.value 是写入的 JSON-like 数据。
        print(f"- key={item.key}, value={item.value}")


def main() -> None:
    # InMemoryStore 是进程内长期记忆示例实现；生产环境通常换成持久化后端。
    store = InMemoryStore()

    # namespace 是 tuple，可按业务维度组织数据；不同 namespace 之间互相隔离。
    user_1_memories = ("users", "user-1", "memories")
    user_2_memories = ("users", "user-2", "memories")
    project_facts = ("projects", "langgraph-study", "facts")

    # put(namespace, key, value)：在指定 namespace 中写入一条长期记忆。
    store.put(user_1_memories, "preference-language", {"text": "用户偏好中文回答"})
    store.put(user_1_memories, "preference-style", {"text": "用户喜欢代码案例"})
    store.put(user_2_memories, "preference-language", {"text": "User prefers English"})
    store.put(project_facts, "goal", {"text": "整理 LangGraph 学习案例"})

    # search(namespace) 只会检索该 namespace 下的数据，不会跨用户/跨项目泄漏。
    print_items("user-1 memories:", store.search(user_1_memories))
    print_items("\nuser-2 memories:", store.search(user_2_memories))
    print_items("\nproject facts:", store.search(project_facts))

    # get(namespace, key) 用于精确读取某一条长期记忆。
    one_item = store.get(user_1_memories, "preference-language")
    print("\nget one item:", one_item.value if one_item else None)


if __name__ == "__main__":
    main()
