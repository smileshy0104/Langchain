import os
import dotenv
import time
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatZhipuAI
dotenv.load_dotenv()

# 检查API密钥
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key or api_key == "your-zhipu-api-key-here":
    print("❌ 错误：请在.env文件中设置您的ZHIPUAI_API_KEY")
    exit(1)

def create_batch_messages() -> List[List]:
    """创建批处理消息列表"""
    system_prompt = "你是一位专业的科技知识专家，请用简洁明了的语言解释技术概念。"

    topics = [
        "机器学习",
        "AIGC（人工智能生成内容）",
        "大模型技术"
    ]

    messages_list = []
    for topic in topics:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"请帮我介绍一下什么是{topic}")
        ]
        messages_list.append(messages)

    return messages_list

def run_batch_processing():
    """执行批处理"""
    print("🚀 开始GLM批处理演示")
    print("=" * 50)

    # 初始化GLM大模型
    chat_model = ChatZhipuAI(
        model="glm-4.6",
        temperature=0.7,
        api_key=api_key
    )

    # 创建批处理消息
    messages_list = create_batch_messages()

    print(f"📝 准备处理 {len(messages_list)} 个问题:")
    for i, messages in enumerate(messages_list, 1):
        human_msg = messages[1]
        print(f"  {i}. {human_msg.content}")

    print(f"\n⏳ 开始批量处理...")
    start_time = time.time()

    try:
        # 执行批处理
        responses = chat_model.batch(messages_list)

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"✅ 批处理完成，耗时: {processing_time:.2f} 秒")
        print("=" * 50)

        # 显示结果
        for i, response in enumerate(responses, 1):
            print(f"🤖 问题 {i} 的回答:")
            print(f"{response.content}")
            print("-" * 30)

    except Exception as e:
        print(f"❌ 批处理失败: {e}")

def run_individual_processing():
    """单独处理对比"""
    print("\n🔄 单独处理对比:")
    print("=" * 50)

    chat_model = ChatZhipuAI(
        model="glm-4.6",
        temperature=0.7,
        api_key=api_key
    )

    messages_list = create_batch_messages()

    print(f"⏳ 开始单独处理 {len(messages_list)} 个问题...")
    start_time = time.time()

    try:
        responses = []
        for messages in messages_list:
            response = chat_model.invoke(messages)
            responses.append(response)

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"✅ 单独处理完成，耗时: {processing_time:.2f} 秒")

    except Exception as e:
        print(f"❌ 单独处理失败: {e}")

if __name__ == "__main__":
    # 运行批处理
    run_batch_processing()

    # 运行单独处理对比
    run_individual_processing()

    print("\n🎉 演示完成！")
    print("💡 批处理可以提高API调用效率，减少总体响应时间。")