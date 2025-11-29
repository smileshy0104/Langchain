"""
示例3：消息修剪（Trim Messages）
演示如何自动修剪过长的对话历史，控制上下文窗口大小
"""

import os
from typing import Any
from langchain.agents import create_agent, AgentState
from langchain_community.chat_models import ChatZhipuAI
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 中间件：消息修剪 ====================

@before_model
def trim_messages_middleware(
    state: AgentState,
    runtime: Runtime
) -> dict[str, Any] | None:
    """
    在调用模型前修剪消息历史

    策略：
    - 最多保留 max_messages 条消息
    - 始终保留第一条消息（通常是系统提示）
    - 保留最近的消息
    """
    max_messages = 6  # 最多保留6条消息
    messages = state["messages"]

    if len(messages) <= max_messages:
        return None  # 不需要修剪

    print(f"\n✂️  触发修剪: {len(messages)} 条 -> {max_messages} 条")

    # 保留策略：第一条 + 最近的几条
    first_msg = messages[0]
    recent_messages = messages[-(max_messages - 1):]

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),  # 清空所有
            first_msg,                               # 添加第一条
            *recent_messages                         # 添加最近的
        ]
    }


# ==================== 主函数 ====================

def main():
    """消息修剪示例"""
    print("\n" + "=" * 60)
    print("示例3：消息修剪（Trim Messages）")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)
    checkpointer = MemorySaver()

    # 创建 Agent，添加修剪中间件
    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer,
        middleware=[trim_messages_middleware],  # 关键：添加修剪中间件
        system_prompt="你是一个助手，会自动管理对话历史长度"
    )

    config = {"configurable": {"thread_id": "trim-test"}}

    print("\n【策略说明】")
    print("- 最多保留 6 条消息")
    print("- 保留第 1 条（系统提示）")
    print("- 保留最近的 5 条")
    print("- 自动删除中间的旧消息")

    # 发送10轮对话，观察修剪过程
    for i in range(10):
        print(f"\n{'='*60}")
        print(f"第 {i+1} 轮对话")
        print(f"{'='*60}")

        print(f"👤 用户: 这是第 {i+1} 条消息")

        result = agent.invoke(
            {"messages": [{"role": "user", "content": f"这是第 {i+1} 条消息，请简短回复"}]},
            config
        )

        print(f"🤖 助手: {result['messages'][-1].content}")

        # 统计信息
        total_messages = len(result['messages'])
        human_count = len([m for m in result['messages'] if m.type == 'human'])
        ai_count = len([m for m in result['messages'] if m.type == 'ai'])

        print(f"\n📊 统计:")
        print(f"   总消息数: {total_messages}")
        print(f"   用户消息: {human_count}")
        print(f"   助手消息: {ai_count}")

        # 显示当前保留的用户消息内容
        user_messages = [m.content for m in result['messages'] if m.type == 'human']
        if user_messages:
            print(f"   保留的用户消息: {user_messages}")

    # 最终测试：Agent 是否记得最早的消息
    print(f"\n{'='*60}")
    print("【记忆测试】")
    print(f"{'='*60}")

    print("\n👤 用户: 我最开始说的第一条消息是什么？")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我最开始说的第一条消息是什么？"}]},
        config
    )
    print(f"🤖 助手: {result['messages'][-1].content}")

    print("\n💡 说明：由于修剪策略，Agent 可能记不住最早的消息（已被删除）")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
