"""
示例1：基础短期记忆
演示最基本的短期记忆功能，使用 InMemorySaver
"""

import os
from langchain.agents import create_agent
from langchain_community.chat_models import ChatZhipuAI
from langgraph.checkpoint.memory import MemorySaver

# 设置 API Key
os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


def main():
    """基础短期记忆示例"""
    print("\n" + "=" * 60)
    print("示例1：基础短期记忆")
    print("=" * 60)

    # 1. 创建模型
    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)

    # 2. 创建检查点器（内存存储）
    checkpointer = MemorySaver()

    # 3. 创建 Agent，启用短期记忆
    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer,  # 关键：启用记忆功能
        system_prompt="你是一个友好的助手，能记住对话历史"
    )

    # 4. 配置会话 ID（thread_id）
    config = {"configurable": {"thread_id": "conversation-1"}}

    # 第一轮对话
    print("\n【第一轮对话】")
    print("👤 用户: 你好！我叫张三，我喜欢编程。")
    result1 = agent.invoke(
        {"messages": [{"role": "user", "content": "你好！我叫张三，我喜欢编程。"}]},
        config
    )
    print(f"🤖 助手: {result1['messages'][-1].content}")

    # 第二轮对话 - Agent 应该记得名字
    print("\n【第二轮对话】")
    print("👤 用户: 我叫什么名字？")
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
        config
    )
    print(f"🤖 助手: {result2['messages'][-1].content}")

    # 第三轮对话 - Agent 应该记得爱好
    print("\n【第三轮对话】")
    print("👤 用户: 我喜欢什么？")
    result3 = agent.invoke(
        {"messages": [{"role": "user", "content": "我喜欢什么？"}]},
        config
    )
    print(f"🤖 助手: {result3['messages'][-1].content}")

    # 第四轮对话 - 测试完整上下文记忆
    print("\n【第四轮对话】")
    print("👤 用户: 总结一下我刚才说的所有信息")
    result4 = agent.invoke(
        {"messages": [{"role": "user", "content": "总结一下我刚才说的所有信息"}]},
        config
    )
    print(f"🤖 助手: {result4['messages'][-1].content}")

    # 显示消息统计
    print("\n【统计信息】")
    print(f"总消息数: {len(result4['messages'])}")
    print(f"对话轮次: {len([m for m in result4['messages'] if m.type == 'human'])}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
