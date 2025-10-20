#!/usr/bin/env python3
"""
GLM-4.6 + LangChain ConversationMemory 示例 (现代语法)
演示各种对话记忆类型的使用方法和最佳实践
包括 ConversationBufferWindowMemory、ConversationTokenBufferMemory、
ConversationSummaryMemory、ConversationSummaryBufferMemory
"""

import os
import dotenv
from typing import Dict, Any
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.memory import (
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory
)
from langchain.chains import LLMChain

# 加载环境变量 - 从项目根目录加载.env文件
dotenv.load_dotenv(dotenv_path="../.env")

# 检查API密钥
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key or api_key == "your-zhipu-api-key-here":
    print("❌ 错误：请在.env文件中设置您的ZHIPUAI_API_KEY")
    exit(1)

# 初始化GLM模型
def get_glm_model(temperature: float = 0.7):
    """获取GLM模型实例"""
    return ChatZhipuAI(
        model="glm-4.6",
        temperature=temperature,
        api_key=api_key
    )

def conversation_buffer_window_memory_example():
    """ConversationBufferWindowMemory 示例 - 保持固定对话轮数"""
    print("=" * 60)
    print("🪟 ConversationBufferWindowMemory 示例")
    print("=" * 60)

    model = get_glm_model()

    # 创建窗口记忆 - 只保留最近3轮对话
    memory = ConversationBufferWindowMemory(
        k=3,  # 保留最近3轮对话
        return_messages=True,
        memory_key="chat_history"
    )

    # 创建提示词模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的AI助手，能够记住最近的对话内容。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # 由于memory需要传统链，我们创建一个包装函数
    def run_with_memory(user_input: str) -> str:
        # 格式化消息
        messages = prompt.format_messages(
            input=user_input,
            chat_history=memory.chat_memory.messages
        )

        # 调用模型
        response = model.invoke(messages)

        # 保存对话到记忆
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(response.content)

        return response.content

    # 模拟对话
    conversations = [
        "你好，我叫小明",
        "我喜欢编程和人工智能",
        "今天天气怎么样？",
        "你能推荐一本Python书籍吗？",
        "我刚才说我叫什么名字？",  # 测试记忆
        "我提到我喜欢什么？"       # 测试记忆
    ]

    print("🚀 开始对话测试...")
    for i, user_input in enumerate(conversations, 1):
        print(f"\n👤 用户 [{i}]: {user_input}")

        response = run_with_memory(user_input)
        print(f"🤖 AI: {response}")

        # 显示当前记忆中的消息数量
        print(f"📝 记忆中消息数: {len(memory.chat_memory.messages)}")

    print(f"\n🎯 记忆中的最后{memory.k}轮对话:")
    for msg in memory.chat_memory.messages:
        msg_type = "用户" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {msg_type}: {msg.content[:50]}...")

def conversation_token_buffer_memory_example():
    """ConversationTokenBufferMemory 示例 - 基于token数量限制"""
    print("\n" + "=" * 60)
    print("🪙 ConversationTokenBufferMemory 示例")
    print("=" * 60)

    model = get_glm_model()

    # 创建token限制记忆 - 最多200个token
    memory = ConversationTokenBufferMemory(
        llm=model,
        max_token_limit=200,  # 最多200个token
        return_messages=True,
        memory_key="chat_history"
    )

    # 创建提示词模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个技术专家助手，对话记忆基于token数量限制。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    def run_with_memory(user_input: str) -> str:
        messages = prompt.format_messages(
            input=user_input,
            chat_history=memory.chat_memory.messages
        )

        response = model.invoke(messages)

        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message(response.content)

        return response.content

    # 模拟较长的对话
    conversations = [
        "请详细解释什么是机器学习？",
        "机器学习有哪些主要的分类？请逐一介绍监督学习、无监督学习和强化学习的特点和应用场景。",
        "深度学习和机器学习有什么区别？神经网络是如何工作的？",
        "请推荐一些学习机器学习的入门资源，包括书籍、在线课程和实践项目。",
        "我想知道我的第一个问题是什么？"  # 测试记忆
    ]

    print("🚀 开始对话测试...")
    for i, user_input in enumerate(conversations, 1):
        print(f"\n👤 用户 [{i}]: {user_input}")

        response = run_with_memory(user_input)
        print(f"🤖 AI: {response[:100]}...")

        # 估算token数量
        total_chars = sum(len(msg.content) for msg in memory.chat_memory.messages)
        estimated_tokens = total_chars // 4  # 粗略估算
        print(f"📊 记忆中字符数: {total_chars}, 估算token数: {estimated_tokens}")

def conversation_summary_memory_example():
    """ConversationSummaryMemory 示例 - 对话摘要记忆"""
    print("\n" + "=" * 60)
    print("📋 ConversationSummaryMemory 示例")
    print("=" * 60)

    model = get_glm_model()

    # 创建摘要记忆
    memory = ConversationSummaryMemory(
        llm=model,
        return_messages=True,
        memory_key="chat_history"
    )

    # 创建提示词模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的学习助手，能够记住并总结对话历史。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    def run_with_memory(user_input: str) -> str:
        # 添加当前消息到记忆
        memory.save_context(
            {"input": user_input},
            {"output": "我理解了您的问题，正在为您解答..."}
        )

        messages = prompt.format_messages(
            input=user_input,
            chat_history=[SystemMessage(content=memory.buffer)]
        )

        response = model.invoke(messages)

        # 更新输出到记忆
        memory.chat_memory.add_ai_message(response.content)

        return response.content

    # 模拟学习相关的对话
    conversations = [
        "我想学习Python编程，应该从哪里开始？",
        "Python有哪些主要的应用领域？",
        "学习Python需要什么基础知识？",
        "你能推荐一些Python学习资源吗？",
        "我应该如何制定学习计划？",
        "请总结一下我们的对话内容"  # 测试摘要
    ]

    print("🚀 开始对话测试...")
    for i, user_input in enumerate(conversations, 1):
        print(f"\n👤 用户 [{i}]: {user_input}")

        response = run_with_memory(user_input)
        print(f"🤖 AI: {response[:100]}...")

        print(f"📝 当前摘要长度: {len(memory.buffer)} 字符")

    print(f"\n🎯 完整对话摘要:")
    print(f"{memory.buffer}")

def conversation_summary_buffer_memory_example():
    """ConversationSummaryBufferMemory 示例 - 混合摘要和缓冲记忆"""
    print("\n" + "=" * 60)
    print("🔄 ConversationSummaryBufferMemory 示例")
    print("=" * 60)

    model = get_glm_model()

    # 创建混合记忆 - 摘要 + 最近2条消息
    memory = ConversationSummaryBufferMemory(
        llm=model,
        max_token_limit=300,  # 最大token限制
        return_messages=True,
        memory_key="chat_history"
    )

    # 创建提示词模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个项目管理助手，使用混合记忆策略管理对话历史。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    def run_with_memory(user_input: str) -> str:
        messages = prompt.format_messages(
            input=user_input,
            chat_history=memory.buffer + memory.chat_memory.messages
        )

        response = model.invoke(messages)

        memory.save_context(
            {"input": user_input},
            {"output": response.content}
        )

        return response.content

    # 模拟项目管理对话
    conversations = [
        "我需要管理一个软件开发项目，有什么建议？",
        "如何制定项目计划和时间线？",
        "团队协作工具推荐哪些？",
        "如何进行有效的进度跟踪？",
        "项目风险管理需要注意什么？",
        "我之前问的第一个问题是什么？"  # 测试记忆
    ]

    print("🚀 开始对话测试...")
    for i, user_input in enumerate(conversations, 1):
        print(f"\n👤 用户 [{i}]: {user_input}")

        response = run_with_memory(user_input)
        print(f"🤖 AI: {response[:100]}...")

        # 显示记忆状态
        summary_len = len(memory.buffer) if memory.buffer else 0
        recent_msgs = len(memory.chat_memory.messages)
        print(f"📊 摘要长度: {summary_len}, 最近消息数: {recent_msgs}")

    print(f"\n🎯 记忆状态:")
    print(f"📋 摘要: {memory.buffer[:200]}...")
    print(f"💬 最近消息:")
    for msg in memory.chat_memory.messages[-2:]:  # 显示最后2条消息
        msg_type = "用户" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {msg_type}: {msg.content[:50]}...")

def memory_comparison():
    """记忆类型比较"""
    print("\n" + "=" * 60)
    print("⚖️ 对话记忆类型比较")
    print("=" * 60)

    print("""
📊 四种对话记忆类型对比:

🪟 ConversationBufferWindowMemory:
✅ 优点:
   - 保持固定数量的对话轮数
   - 简单直观的滑动窗口机制
   - 适合短期对话场景

❌ 缺点:
   - 可能丢失重要的早期对话
   - 不考虑内容重要性

🪙 ConversationTokenBufferMemory:
✅ 优点:
   - 基于token数量精确控制
   - 考虑消息长度差异
   - 适合有严格token限制的场景

❌ 缺点:
   - 可能截断重要信息
   - token计算可能有误差

📋 ConversationSummaryMemory:
✅ 优点:
   - 保持对话的完整摘要
   - 节省存储空间
   - 适合长期对话

❌ 缺点:
   - 可能丢失细节信息
   - 摘要质量依赖模型能力

🔄 ConversationSummaryBufferMemory:
✅ 优点:
   - 平衡摘要和详细记录
   - 保持最近的完整对话
   - 适合复杂的长期对话

❌ 缺点:
   - 配置相对复杂
   - 需要调优参数

🎯 使用建议:
1. 短期对话 → ConversationBufferWindowMemory
2. 严格的token限制 → ConversationTokenBufferMemory
3. 长期对话且需要摘要 → ConversationSummaryMemory
4. 需要平衡的场景 → ConversationSummaryBufferMemory
    """)

def modern_syntax_memory_example():
    """现代语法记忆示例 - 使用 RunnablePassthrough"""
    print("\n" + "=" * 60)
    print("🚀 现代语法记忆示例")
    print("=" * 60)

    model = get_glm_model()

    # 创建简单的窗口记忆
    memory = ConversationBufferWindowMemory(
        k=2,
        return_messages=True,
        memory_key="chat_history"
    )

    # 现代语法的处理链
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个使用现代语法的AI助手。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    chain = prompt | model | StrOutputParser()

    # 创建包装函数来集成记忆
    def create_memory_chain():
        def run_chain(inputs: Dict[str, Any]) -> str:
            # 获取历史消息
            chat_history = memory.chat_memory.messages

            # 准备输入
            chain_inputs = {
                "input": inputs["input"],
                "chat_history": chat_history
            }

            # 运行链
            response = chain.invoke(chain_inputs)

            # 保存到记忆
            memory.save_context(
                {"input": inputs["input"]},
                {"output": response}
            )

            return response

        return run_chain

    # 创建带记忆的链
    memory_chain = create_memory_chain()

    # 测试对话
    print("🚀 使用现代语法测试对话记忆...")

    test_inputs = [
        {"input": "你好，我想学习LangChain"},
        {"input": "LangChain有哪些主要功能？"},
        {"input": "什么是Runnable？"},
        {"input": "我刚才第一个问题是什么？"}
    ]

    for i, inputs in enumerate(test_inputs, 1):
        print(f"\n👤 用户 [{i}]: {inputs['input']}")
        response = memory_chain(inputs)
        print(f"🤖 AI: {response[:100]}...")
        print(f"📝 记忆中消息数: {len(memory.chat_memory.messages)}")

def main():
    """主函数：运行所有示例"""
    print("🚀 GLM-4.6 + LangChain ConversationMemory 详细使用示例")
    print("=" * 80)

    try:
        # 运行各种示例
        conversation_buffer_window_memory_example()
        conversation_token_buffer_memory_example()
        conversation_summary_memory_example()
        conversation_summary_buffer_memory_example()
        memory_comparison()
        modern_syntax_memory_example()

        print("\n🎉 所有示例运行完成！")
        print("\n📚 更多信息请参考：")
        print("- LangChain官方文档: https://python.langchain.com/")
        print("- 记忆组件指南: https://python.langchain.com/docs/modules/memory/")

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")

if __name__ == "__main__":
    main()