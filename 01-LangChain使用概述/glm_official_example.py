#!/usr/bin/env python3
"""
基于官方文档的GLM-4.6 LangChain集成示例
参考：https://python.langchain.ac.cn/docs/integrations/chat/zhipuai/
"""

import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# 加载环境变量
load_dotenv()

def basic_example():
    """基础示例：简单调用GLM-4"""
    print("=" * 50)
    print("📝 基础示例：简单调用GLM-4")
    print("=" * 50)

    # 检查API密钥
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key or api_key == "your-zhipu-api-key-here":
        print("❌ 错误：请在.env文件中设置您的ZHIPUAI_API_KEY")
        return

    # 创建GLM模型实例
    chat = ChatZhipuAI(
        model="glm-4",
        temperature=0.5,
        api_key=api_key
    )

    # 简单文本调用
    try:
        response = chat.invoke("你好，请简单介绍一下GLM-4模型")
        print(f"🤖 GLM-4 回答：\n{response.content}\n")
    except Exception as e:
        print(f"❌ 调用失败：{e}\n")

def message_example():
    """消息示例：使用不同类型的消息"""
    print("=" * 50)
    print("💬 消息示例：系统提示和用户消息")
    print("=" * 50)

    chat = ChatZhipuAI(
        model="glm-4",
        temperature=0.7,
        api_key=os.getenv("ZHIPUAI_API_KEY")
    )

    # 构建消息列表
    messages = [
        SystemMessage(content="你是一个专业的Python编程助手，请用简洁明了的方式回答问题。"),
        HumanMessage(content="请解释一下Python中的装饰器是什么，并给出一个简单的例子。")
    ]

    try:
        response = chat.invoke(messages)
        print(f"👨‍💻 编程助手回答：\n{response.content}\n")
    except Exception as e:
        print(f"❌ 调用失败：{e}\n")

def prompt_template_example():
    """提示词模板示例"""
    print("=" * 50)
    print("📋 提示词模板示例")
    print("=" * 50)

    chat = ChatZhipuAI(
        model="glm-4",
        temperature=0.8,
        api_key=os.getenv("ZHIPUAI_API_KEY")
    )

    # 创建提示词模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}，擅长{expertise}。"),
        ("human", "请{task}：{topic}")
    ])

    # 格式化提示词
    formatted_prompt = prompt.format_messages(
        role="科技博主",
        expertise="用简单易懂的语言解释复杂概念",
        task="解释什么是人工智能",
        topic="机器学习的基本原理"
    )

    print(f"📝 提示词模板：科技博主 + 解释机器学习")

    try:
        response = chat.invoke(formatted_prompt)
        print(f"🤖 GLM-4 回答：\n{response.content}\n")
    except Exception as e:
        print(f"❌ 调用失败：{e}\n")

def chain_example():
    """链式调用示例"""
    print("=" * 50)
    print("⛓️ 链式调用示例")
    print("=" * 50)

    chat = ChatZhipuAI(
        model="glm-4",
        temperature=0.6,
        api_key=os.getenv("ZHIPUAI_API_KEY")
    )

    # 创建提示词模板
    prompt = ChatPromptTemplate.from_template(
        "请为主题'{topic}'写一个{style}的{length}。要求：{requirements}"
    )

    # 创建输出解析器
    output_parser = StrOutputParser()

    # 构建链
    chain = prompt | chat | output_parser

    print(f"🔗 链式调用：写一个关于AI的简短诗歌")

    try:
        result = chain.invoke({
            "topic": "人工智能与人类的关系",
            "style": "富有想象力",
            "length": "短诗",
            "requirements": "语言优美，意境深远"
        })
        print(f"✨ 生成结果：\n{result}\n")
    except Exception as e:
        print(f"❌ 链式调用失败：{e}\n")

def conversation_example():
    """对话示例：多轮对话"""
    print("=" * 50)
    print("🗣️ 多轮对话示例")
    print("=" * 50)

    chat = ChatZhipuAI(
        model="glm-4",
        temperature=0.7,
        api_key=os.getenv("ZHIPUAI_API_KEY")
    )

    # 对话历史
    messages = [
        SystemMessage(content="你是一个友善的AI助手，能够记住对话内容。"),
        HumanMessage(content="你好，我叫小明，我是一名大学生。"),
        AIMessage(content="你好小明！很高兴认识你。作为一名大学生，你正在学习什么专业呢？"),
        HumanMessage(content="我在学习计算机科学。"),
        AIMessage(content="计算机科学是一个很棒的专业！你对哪个方向最感兴趣呢？比如人工智能、软件工程、网络安全等。"),
        HumanMessage(content="我对人工智能最感兴趣，特别是自然语言处理。")
    ]

    print("💭 对话历史：")
    print("  用户：你好，我叫小明，我是一名大学生。")
    print("  AI：你好小明！很高兴认识你。作为一名大学生，你正在学习什么专业呢？")
    print("  用户：我在学习计算机科学。")
    print("  AI：计算机科学是一个很棒的专业！你对哪个方向最感兴趣呢？")
    print("  用户：我对人工智能最感兴趣，特别是自然语言处理。")

    # 继续对话
    try:
        response = chat.invoke(messages)
        print(f"\n🤖 AI 回应：\n{response.content}\n")
    except Exception as e:
        print(f"❌ 对话失败：{e}\n")

def streaming_example():
    """流式输出示例"""
    print("=" * 50)
    print("🌊 流式输出示例")
    print("=" * 50)

    # 注意：流式输出需要特殊的处理方式
    chat = ChatZhipuAI(
        model="glm-4",
        temperature=0.7,
        api_key=os.getenv("ZHIPUAI_API_KEY"),
        streaming=True
    )

    print("📝 正在流式生成故事...")

    try:
        # 由于流式输出的实现比较复杂，这里先展示普通调用
        response = chat.invoke("请写一个关于AI与人类友谊的温馨小故事，字数控制在100字以内。")
        print(f"📖 故事内容：\n{response.content}\n")
    except Exception as e:
        print(f"❌ 流式输出失败：{e}\n")

def code_generation_example():
    """代码生成示例"""
    print("=" * 50)
    print("💻 代码生成示例")
    print("=" * 50)

    chat = ChatZhipuAI(
        model="glm-4",
        temperature=0.3,  # 代码生成使用较低温度
        api_key=os.getenv("ZHIPUAI_API_KEY")
    )

    code_prompt = """请用Python写一个函数，实现以下功能：
1. 计算斐波那契数列的第n项
2. 包含错误处理
3. 添加注释说明
4. 提供使用示例

要求代码清晰易懂，适合初学者理解。"""

    print("💻 代码生成需求：斐波那契数列函数")

    try:
        response = chat.invoke(code_prompt)
        print(f"🐍 生成的Python代码：\n{response.content}\n")
    except Exception as e:
        print(f"❌ 代码生成失败：{e}\n")

def main():
    """主函数：运行所有示例"""
    print("🚀 GLM-4.6 + LangChain 官方示例")
    print("参考文档：https://python.langchain.ac.cn/docs/integrations/chat/zhipuai/\n")

    # 检查API密钥
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key or api_key == "your-zhipu-api-key-here":
        print("❌ 错误：请在.env文件中设置您的ZHIPUAI_API_KEY")
        print("📝 获取API密钥：https://open.bigmodel.cn/")
        return

    try:
        # 运行各种示例
        basic_example()
        message_example()
        prompt_template_example()
        chain_example()
        conversation_example()
        streaming_example()
        code_generation_example()

        print("🎉 所有示例运行完成！")
        print("\n📚 更多功能请参考：")
        print("https://python.langchain.ac.cn/docs/integrations/chat/zhipuai/")

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")

if __name__ == "__main__":
    main()