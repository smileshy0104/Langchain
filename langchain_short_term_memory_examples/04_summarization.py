"""
示例4：消息摘要（Summarization）
演示如何使用 SummarizationMiddleware 自动总结对话历史
"""

import os
from langchain.agents import create_agent
from langchain_community.chat_models import ChatZhipuAI
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import MemorySaver

os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


def main():
    """消息摘要示例"""
    print("\n" + "=" * 60)
    print("示例4：消息摘要（Summarization）")
    print("=" * 60)

    # 创建模型
    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)
    checkpointer = MemorySaver()

    # 创建 Agent，添加摘要中间件
    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer,
        middleware=[
            SummarizationMiddleware(
                model=model,                    # 用于生成摘要的模型
                trigger={"messages": 6},        # 当消息数超过6条时触发摘要
                keep={"messages": 3},           # 摘要后保留最近3条消息
            )
        ],
        system_prompt="你是一个助手，能够自动总结对话历史"
    )

    config = {"configurable": {"thread_id": "summary-test"}}

    print("\n【摘要策略说明】")
    print("- 触发条件: 消息数 > 6 条")
    print("- 保留策略: 保留最近 3 条消息")
    print("- 其余消息会被自动总结")
    print("- 摘要会永久替换旧消息（持久化更新）")

    # 模拟长对话
    messages_to_send = [
        "你好！我叫李明。",
        "我今年25岁。",
        "我在北京工作。",
        "我是一名软件工程师。",
        "我喜欢Python编程。",
        "我最近在学习人工智能和大语言模型。",
        "我的兴趣爱好是看书和旅行。",
        "你能帮我总结一下我的基本信息吗？",
    ]

    for i, msg in enumerate(messages_to_send, 1):
        print(f"\n{'='*60}")
        print(f"第 {i} 轮对话")
        print(f"{'='*60}")

        print(f"👤 用户: {msg}")

        result = agent.invoke(
            {"messages": [{"role": "user", "content": msg}]},
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
        print(f"   AI 消息: {ai_count}")

        # 检查是否有摘要消息
        summary_msg = None
        for m in result['messages']:
            if hasattr(m, 'content') and '总结' in str(m.content):
                summary_msg = m
                break

        if summary_msg:
            print(f"   📝 发现摘要消息")

        # 如果触发了摘要，显示当前保留的消息
        if i >= 6:
            print(f"\n   💡 当前保留的消息类型:")
            for j, m in enumerate(result['messages'], 1):
                msg_type = m.type
                content_preview = str(m.content)[:50] + "..." if len(str(m.content)) > 50 else str(m.content)
                print(f"      {j}. [{msg_type}] {content_preview}")

    # 最终测试：验证摘要效果
    print(f"\n{'='*60}")
    print("【摘要效果验证】")
    print(f"{'='*60}")

    # 查询状态
    state = agent.get_state(config)
    print(f"\n📊 最终状态:")
    print(f"   总消息数: {len(state.values['messages'])}")

    # 尝试回忆早期信息
    print("\n👤 用户: 我一开始说我叫什么名字？我在哪里工作？")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我一开始说我叫什么名字？我在哪里工作？"}]},
        config
    )
    print(f"🤖 助手: {result['messages'][-1].content}")

    print("\n💡 说明：虽然早期消息被摘要，但 Agent 仍能从摘要中提取关键信息")


def example_with_custom_trigger():
    """使用自定义触发条件的示例"""
    print("\n" + "=" * 60)
    print("【高级示例】自定义触发条件")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)
    checkpointer = MemorySaver()

    # 使用 Token 数量作为触发条件
    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer,
        middleware=[
            SummarizationMiddleware(
                model=model,
                trigger={"tokens": 500},     # 当 Token 数超过 500 时触发
                keep={"tokens": 200},        # 保留约 200 Token
            )
        ],
        system_prompt="你是一个助手"
    )

    print("\n触发条件: Token 数 > 500")
    print("保留策略: 保留约 200 Token 的消息")


if __name__ == "__main__":
    try:
        main()
        # example_with_custom_trigger()  # 可选：运行高级示例
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
