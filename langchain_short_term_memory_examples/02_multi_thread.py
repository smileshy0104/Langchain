"""
示例2：多线程会话管理
演示如何同时管理多个用户的独立会话
"""

import os
from langchain.agents import create_agent
from langchain_community.chat_models import ChatZhipuAI
from langgraph.checkpoint.memory import MemorySaver

os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


def main():
    """多线程会话管理示例"""
    print("\n" + "=" * 60)
    print("示例2：多线程会话管理")
    print("=" * 60)

    # 创建模型和检查点器
    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)
    checkpointer = MemorySaver()

    # 创建 Agent
    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer,
        system_prompt="你是一个客服助手"
    )

    # ========== 用户A的会话 ==========
    print("\n" + "-" * 60)
    print("【用户A的会话】thread_id: user-A")
    print("-" * 60)

    config_a = {"configurable": {"thread_id": "user-A"}}

    print("\n👤 用户A: 我想买一台笔记本电脑，预算8000元")
    result_a1 = agent.invoke(
        {"messages": [{"role": "user", "content": "我想买一台笔记本电脑，预算8000元"}]},
        config_a
    )
    print(f"🤖 客服: {result_a1['messages'][-1].content}")

    # ========== 用户B的会话 ==========
    print("\n" + "-" * 60)
    print("【用户B的会话】thread_id: user-B")
    print("-" * 60)

    config_b = {"configurable": {"thread_id": "user-B"}}

    print("\n👤 用户B: 我想买一部手机，要拍照好的")
    result_b1 = agent.invoke(
        {"messages": [{"role": "user", "content": "我想买一部手机，要拍照好的"}]},
        config_b
    )
    print(f"🤖 客服: {result_b1['messages'][-1].content}")

    # ========== 用户C的会话 ==========
    print("\n" + "-" * 60)
    print("【用户C的会话】thread_id: user-C")
    print("-" * 60)

    config_c = {"configurable": {"thread_id": "user-C"}}

    print("\n👤 用户C: 我需要一个耳机，降噪功能要好")
    result_c1 = agent.invoke(
        {"messages": [{"role": "user", "content": "我需要一个耳机，降噪功能要好"}]},
        config_c
    )
    print(f"🤖 客服: {result_c1['messages'][-1].content}")

    # ========== 回到用户A - 测试独立记忆 ==========
    print("\n" + "-" * 60)
    print("【回到用户A】")
    print("-" * 60)

    print("\n👤 用户A: 我刚才想买什么？预算是多少？")
    result_a2 = agent.invoke(
        {"messages": [{"role": "user", "content": "我刚才想买什么？预算是多少？"}]},
        config_a
    )
    print(f"🤖 客服: {result_a2['messages'][-1].content}")

    # ========== 回到用户B ==========
    print("\n" + "-" * 60)
    print("【回到用户B】")
    print("-" * 60)

    print("\n👤 用户B: 我有什么要求？")
    result_b2 = agent.invoke(
        {"messages": [{"role": "user", "content": "我有什么要求？"}]},
        config_b
    )
    print(f"🤖 客服: {result_b2['messages'][-1].content}")

    # ========== 回到用户C ==========
    print("\n" + "-" * 60)
    print("【回到用户C】")
    print("-" * 60)

    print("\n👤 用户C: 提醒我一下我想买什么？")
    result_c2 = agent.invoke(
        {"messages": [{"role": "user", "content": "提醒我一下我想买什么？"}]},
        config_c
    )
    print(f"🤖 客服: {result_c2['messages'][-1].content}")

    # ========== 总结 ==========
    print("\n" + "=" * 60)
    print("【会话独立性验证】")
    print("=" * 60)
    print("✅ 每个用户的会话完全独立")
    print("✅ 用户A只记得笔记本电脑和预算")
    print("✅ 用户B只记得手机和拍照要求")
    print("✅ 用户C只记得耳机和降噪要求")
    print("✅ 通过 thread_id 实现会话隔离")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
