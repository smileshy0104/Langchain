#!/usr/bin/env python3
"""
GLM-4.6 + LangChain PromptTemplate 详细使用示例
演示各种提示词模板的使用方法和最佳实践
"""

import os
import dotenv
from typing import List, Dict, Any
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser

# 加载环境变量 - 从项目根目录加载.env文件
dotenv.load_dotenv(dotenv_path="../../.env")

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

def basic_template_example():
    """基础模板示例"""
    print("=" * 60)
    print("📝 基础模板示例")
    print("=" * 60)

    model = get_glm_model()

    # 1. 简单字符串模板
    print("\n1️⃣ 简单字符串模板")
    prompt_template = PromptTemplate(
        input_variables=["topic", "style"],
        template="请写一篇关于{topic}的{style}风格的文章。"
    )

    # 格式化提示词
    prompt = prompt_template.format(topic="人工智能", style="科技")
    print(f"📋 生成的提示词: {prompt}")

    # 调用模型
    response = model.invoke(prompt)
    print(f"🤖 模型回复: {response.content[:200]}...\n")

    # 2. 聊天模板
    print("2️⃣ 聊天模板")
    chat_prompt = ChatPromptTemplate.from_template(
        "你是一个{role}。请用{tone}的语调回答：{question}"
    )

    formatted_prompt = chat_prompt.format_messages(
        role="专业程序员",
        tone="友好耐心",
        question="什么是装饰器？"
    )

    response = model.invoke(formatted_prompt)
    print(f"🤖 模型回复: {response.content[:200]}...\n")

def multi_variable_template():
    """多变量模板示例"""
    print("=" * 60)
    print("🔧 多变量模板示例")
    print("=" * 60)

    model = get_glm_model()

    # 复杂的多变量模板
    complex_template = """
任务描述：{task}
目标受众：{audience}
内容类型：{content_type}
字数要求：{word_count}
特殊要求：{requirements}

请根据以上要求生成内容。
    """.strip()

    prompt_template = PromptTemplate(
        input_variables=["task", "audience", "content_type", "word_count", "requirements"],
        template=complex_template
    )

    # 填充变量
    prompt = prompt_template.format(
        task="写一篇技术博客",
        audience="编程初学者",
        content_type="教程",
        word_count="800-1000字",
        requirements="包含代码示例，语言通俗易懂，结构清晰"
    )

    print(f"📋 生成的提示词:\n{prompt}\n")

    response = model.invoke(prompt)
    print(f"🤖 模型回复: {response.content[:300]}...\n")

def chat_message_template():
    """聊天消息模板示例"""
    print("=" * 60)
    print("💬 聊天消息模板示例")
    print("=" * 60)

    model = get_glm_model()

    # 1. 多角色聊天模板
    print("1️⃣ 多角色聊天模板")
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}，专门{specialty}。"),
        ("human", "你好！我需要{help_type}。"),
        ("ai", "您好！我是专业的{role}，很高兴为您服务。请问您具体需要什么{help_type}？"),
        ("human", "{specific_question}")
    ])

    messages = chat_prompt.format_messages(
        role="数据科学家",
        specialty="数据分析和机器学习",
        help_type="数据分析建议",
        specific_question="我有一个销售数据集，应该如何开始分析？"
    )

    print("📋 生成的对话模板:")
    for i, msg in enumerate(messages, 1):
        print(f"  {i}. [{msg.__class__.__name__}]: {msg.content[:100]}...")

    response = model.invoke(messages)
    print(f"\n🤖 模型回复: {response.content[:300]}...\n")

    # 2. 使用 MessagesPlaceholder
    print("2️⃣ 动态消息占位符")
    dynamic_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的{role}。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{current_question}")
    ])

    # 模拟对话历史
    history = [
        HumanMessage(content="什么是Python？"),
        AIMessage(content="Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名。"),
        HumanMessage(content="Python适合做什么？")
    ]

    messages = dynamic_prompt.format_messages(
        role="Python专家",
        history=history,
        current_question="请推荐一些Python学习的资源。"
    )

    response = model.invoke(messages)
    print(f"🤖 模型回复: {response.content[:300]}...\n")

def few_shot_template():
    """少样本学习模板示例"""
    print("=" * 60)
    print("🎯 少样本学习模板示例")
    print("=" * 60)

    model = get_glm_model()

    # 1. 文本格式少样本
    print("1️⃣ 文本格式少样本学习")
    examples = [
        {
            "question": "苹果是什么？",
            "answer": "苹果是一种水果，富含维生素，口感清脆。"
        },
        {
            "question": "香蕉是什么？",
            "answer": "香蕉是一种热带水果，富含钾元素，口感软糯。"
        },
        {
            "question": "橙子是什么？",
            "answer": "橙子是一种柑橘类水果，富含维生素C，口感酸甜。"
        }
    ]

    example_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="问题：{question}\n回答：{answer}"
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="请根据以下示例回答问题：",
        suffix="问题：{input}\n回答：",
        input_variables=["input"]
    )

    prompt = few_shot_prompt.format(input="草莓是什么？")
    print(f"📋 生成的少样本提示词:\n{prompt}")

    response = model.invoke(prompt)
    print(f"🤖 模型回复: {response.content}\n")

    # 2. 聊天格式少样本
    print("2️⃣ 聊天格式少样本学习")
    chat_examples = [
        {
            "input": "解释什么是递归",
            "output": "递归是一种编程技术，函数直接或间接调用自身来解决问题。"
        },
        {
            "input": "解释什么是面向对象",
            "output": "面向对象是一种编程范式，通过对象和类来组织代码。"
        }
    ]

    chat_example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ])

    few_shot_chat_prompt = FewShotChatMessagePromptTemplate(
        examples=chat_examples,
        example_prompt=chat_example_prompt
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个编程老师，请用简洁的语言解释编程概念。"),
        few_shot_chat_prompt,
        ("human", "{input}")
    ])

    messages = final_prompt.format_messages(input="解释什么是API")
    response = model.invoke(messages)
    print(f"🤖 模型回复: {response.content}\n")

def output_parser_template():
    """输出解析器模板示例"""
    print("=" * 60)
    print("🔍 输出解析器模板示例")
    print("=" * 60)

    model = get_glm_model()

    # 1. 列表输出解析
    print("1️⃣ 列表输出解析")
    list_parser = CommaSeparatedListOutputParser()

    list_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个知识助手，请提供相关的关键词列表。"),
        ("human", "请列出5个与{topic}相关的关键词。"),
        ("system", "{format_instructions}")
    ])

    chain = list_prompt | model | list_parser

    result = chain.invoke({
        "topic": "机器学习",
        "format_instructions": list_parser.get_format_instructions()
    })

    print(f"📋 解析结果: {result}")
    print(f"📊 结果类型: {type(result)}\n")

    # 2. 字符串输出解析
    print("2️⃣ 字符串输出解析")
    string_parser = StrOutputParser()

    string_prompt = ChatPromptTemplate.from_template(
        "请写一个关于{subject}的{length}，要求{style}。"
    )

    chain = string_prompt | model | string_parser

    result = chain.invoke({
        "subject": "编程",
        "length": "四行诗",
        "style": "富有诗意"
    })

    print(f"📋 解析结果: {result}")
    print(f"📊 结果类型: {type(result)}\n")

def conditional_template():
    """条件模板示例"""
    print("=" * 60)
    print("🔄 条件模板示例")
    print("=" * 60)

    model = get_glm_model()

    def create_conditional_prompt(user_level: str, topic: str) -> ChatPromptTemplate:
        """根据用户水平创建不同的提示词模板"""

        if user_level == "beginner":
            system_msg = "你是一个编程初学者导师，请用最简单易懂的语言解释概念，避免专业术语。"
        elif user_level == "intermediate":
            system_msg = "你是一个编程进阶导师，可以适当使用专业术语，但需要解释清楚。"
        elif user_level == "advanced":
            system_msg = "你是一个编程专家，可以进行深入的技术讨论，假设用户有扎实的基础。"
        else:
            system_msg = "你是一个编程导师，请根据问题的复杂程度调整解释深度。"

        return ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", f"请解释{topic}")
        ])

    # 测试不同水平
    levels = ["beginner", "intermediate", "advanced"]
    topic = "什么是异步编程？"

    for level in levels:
        print(f"📚 {level.upper()} 水平:")
        prompt = create_conditional_prompt(level, topic)
        messages = prompt.format_messages()

        response = model.invoke(messages)
        print(f"🤖 回答: {response.content[:200]}...\n")

def template_pipeline():
    """模板管道示例"""
    print("=" * 60)
    print("🔗 模板管道示例")
    print("=" * 60)

    model = get_glm_model()

    # 创建一个处理管道
    def create_content_pipeline(topic: str, content_type: str, audience: str):
        """创建内容生成管道"""

        # 第一步：内容规划
        planning_prompt = ChatPromptTemplate.from_template(
            """
            任务：为{audience}规划一篇关于{topic}的{content_type}

            请提供：
            1. 主要要点（3-5个）
            2. 内容结构
            3. 注意事项

            请用简洁的要点形式回答。
            """
        )

        # 第二步：内容生成
        generation_prompt = ChatPromptTemplate.from_template(
            """
            基于以下规划，为{audience}写一篇关于{topic}的{content_type}：

            规划要点：
            {outline}

            要求：
            - 语言适合目标受众
            - 结构清晰
            - 内容准确
            - 长度适中
            """
        )

        # 第三步：内容优化
        optimization_prompt = ChatPromptTemplate.from_template(
            """
            请优化以下内容，使其更加完善：

            原始内容：
            {content}

            优化要求：
            - 检查并修正错误
            - 改善语言表达
            - 确保逻辑清晰
            - 适当增加实例
            """
        )

        return planning_prompt, generation_prompt, optimization_prompt

    # 执行管道
    topic = "机器学习基础"
    content_type = "入门教程"
    audience = "编程初学者"

    planning_prompt, generation_prompt, optimization_prompt = create_content_pipeline(
        topic, content_type, audience
    )

    print("🔧 步骤1: 内容规划")
    planning_chain = planning_prompt | model | StrOutputParser()
    outline = planning_chain.invoke({
        "topic": topic,
        "content_type": content_type,
        "audience": audience
    })
    print(f"📋 规划结果:\n{outline}\n")

    print("✍️ 步骤2: 内容生成")
    generation_chain = generation_prompt | model | StrOutputParser()
    content = generation_chain.invoke({
        "topic": topic,
        "content_type": content_type,
        "audience": audience,
        "outline": outline
    })
    print(f"📝 生成内容:\n{content[:400]}...\n")

    print("⚡ 步骤3: 内容优化")
    optimization_chain = optimization_prompt | model | StrOutputParser()
    optimized_content = optimization_chain.invoke({"content": content})
    print(f"🌟 优化内容:\n{optimized_content[:400]}...\n")

def best_practices():
    """最佳实践示例"""
    print("=" * 60)
    print("💡 PromptTemplate 最佳实践")
    print("=" * 60)

    print("""
1. 🎯 明确性原则
   - 清晰定义输入变量
   - 使用描述性的变量名
   - 提供具体的格式要求

2. 🔧 模块化设计
   - 将复杂提示词分解为多个简单模板
   - 使用组合而非继承
   - 保持模板的可重用性

3. 🛡️ 安全性考虑
   - 验证输入参数
   - 设置合理的默认值
   - 避免提示词注入攻击

4. 📊 性能优化
   - 缓存常用的模板
   - 预编译复杂的模板
   - 使用批量处理

5. 🧪 测试策略
   - 为模板编写单元测试
   - 测试边界情况
   - 验证输出格式
    """)

def main():
    """主函数：运行所有示例"""
    print("🚀 GLM-4.6 + LangChain PromptTemplate 详细使用示例")
    print("=" * 80)

    try:
        # 运行各种示例
        basic_template_example()
        multi_variable_template()
        chat_message_template()
        few_shot_template()
        output_parser_template()
        conditional_template()
        template_pipeline()
        best_practices()

        print("🎉 所有示例运行完成！")
        print("\n📚 更多信息请参考：")
        print("- LangChain官方文档: https://python.langchain.com/")
        print("- 提示词工程指南: https://www.promptingguide.ai/")

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")

if __name__ == "__main__":
    main()