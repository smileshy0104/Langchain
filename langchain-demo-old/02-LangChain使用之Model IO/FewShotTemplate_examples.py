#!/usr/bin/env python3
"""
GLM-4.6 + LangChain 少样本学习模板详细使用示例
演示 FewShotPromptTemplate 和 FewShotChatMessagePromptTemplate 的使用方法
"""

import os
import dotenv
from typing import List, Dict, Any
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import (
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
    ChatPromptTemplate
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

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

def basic_few_shot_prompt():
    """基础 FewShotPromptTemplate 示例"""
    print("=" * 60)
    print("📝 基础 FewShotPromptTemplate 示例")
    print("=" * 60)

    model = get_glm_model()

    # 1. 创建示例
    examples = [
        {
            "question": "什么是人工智能？",
            "answer": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"
        },
        {
            "question": "什么是机器学习？",
            "answer": "机器学习是人工智能的一个子集，使计算机能够从数据中学习并改进性能，而无需明确编程。"
        },
        {
            "question": "什么是深度学习？",
            "answer": "深度学习是机器学习的一个子集，使用多层神经网络来模拟人脑的工作方式，处理复杂的模式。"
        }
    ]

    # 2. 创建示例提示模板
    example_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="问题：{question}\n回答：{answer}"
    )

    # 3. 创建少样本提示模板
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="请根据以下示例回答问题，保持回答的简洁性和准确性：",
        suffix="问题：{input}\n回答：",
        input_variables=["input"],
        example_separator="\n\n"
    )

    # 4. 格式化提示词
    new_question = "什么是自然语言处理？"
    formatted_prompt = few_shot_prompt.format(input=new_question)

    print("📋 生成的少样本提示词：")
    print(formatted_prompt)
    print("-" * 40)

    # 5. 调用模型
    response = model.invoke(formatted_prompt)
    print(f"🤖 GLM-4.6 回答：")
    print(f"{response.content}\n")

def translation_few_shot():
    """翻译任务少样本示例"""
    print("=" * 60)
    print("🌍 翻译任务 FewShotPromptTemplate 示例")
    print("=" * 60)

    model = get_glm_model()

    # 翻译示例
    translation_examples = [
        {
            "chinese": "今天天气很好。",
            "english": "The weather is nice today."
        },
        {
            "chinese": "我喜欢学习编程。",
            "english": "I like learning programming."
        },
        {
            "chinese": "这本书很有趣。",
            "english": "This book is very interesting."
        }
    ]

    # 翻译提示模板
    translation_prompt = PromptTemplate(
        input_variables=["chinese", "english"],
        template="中文：{chinese}\n英文：{english}"
    )

    # 创建少样本模板
    few_shot_translation = FewShotPromptTemplate(
        examples=translation_examples,
        example_prompt=translation_prompt,
        prefix="请根据以下示例将中文翻译成英文：",
        suffix="中文：{input}\n英文：",
        input_variables=["input"]
    )

    # 测试翻译
    test_sentence = "人工智能正在改变世界。"
    formatted_prompt = few_shot_translation.format(input=test_sentence)

    print("📋 翻译任务提示词：")
    print(formatted_prompt)
    print("-" * 40)

    response = model.invoke(formatted_prompt)
    print(f"🤖 GLM-4.6 翻译结果：")
    print(f"{response.content}\n")

def code_generation_few_shot():
    """代码生成少样本示例"""
    print("=" * 60)
    print("💻 代码生成 FewShotPromptTemplate 示例")
    print("=" * 60)

    model = get_glm_model(temperature=0.3)  # 代码生成使用较低温度

    # 代码示例
    code_examples = [
        {
            "description": "计算两个数字的和",
            "code": "def add(a, b):\n    return a + b"
        },
        {
            "description": "计算数字的平方",
            "code": "def square(x):\n    return x ** 2"
        },
        {
            "description": "判断数字是否为偶数",
            "code": "def is_even(n):\n    return n % 2 == 0"
        }
    ]

    # 代码提示模板
    code_prompt = PromptTemplate(
        input_variables=["description", "code"],
        template="功能描述：{description}\n代码实现：\n{code}"
    )

    # 创建少样本模板
    few_shot_code = FewShotPromptTemplate(
        examples=code_examples,
        example_prompt=code_prompt,
        prefix="请根据以下示例编写Python函数：",
        suffix="功能描述：{input}\n代码实现：",
        input_variables=["input"]
    )

    # 测试代码生成
    function_description = "计算列表的平均值"
    formatted_prompt = few_shot_code.format(input=function_description)

    print("📋 代码生成提示词：")
    print(formatted_prompt)
    print("-" * 40)

    response = model.invoke(formatted_prompt)
    print(f"🤖 GLM-4.6 生成的代码：")
    print(f"{response.content}\n")

def basic_few_shot_chat():
    """基础 FewShotChatMessagePromptTemplate 示例"""
    print("=" * 60)
    print("💬 基础 FewShotChatMessagePromptTemplate 示例")
    print("=" * 60)

    model = get_glm_model()

    # 1. 创建对话示例
    chat_examples = [
        {
            "input": "解释什么是递归",
            "output": "递归是一种编程技术，函数直接或间接调用自身来解决问题。就像俄罗斯套娃，每个娃娃里面都有一个更小的娃娃。"
        },
        {
            "input": "解释什么是面向对象编程",
            "output": "面向对象编程是一种编程范式，它将数据和操作数据的方法组织在对象中。就像现实世界中的汽车，它有属性（颜色、品牌）和行为（加速、刹车）。"
        },
        {
            "input": "解释什么是API",
            "output": "API是应用程序接口，它定义了不同软件组件之间如何交互。就像餐厅的服务员，他负责将你的订单传达给厨房，并将食物带给你。"
        }
    ]

    # 2. 创建示例提示模板
    chat_example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ])

    # 3. 创建少样本聊天提示模板
    few_shot_chat_prompt = FewShotChatMessagePromptTemplate(
        examples=chat_examples,
        example_prompt=chat_example_prompt
    )

    # 4. 创建最终提示模板
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个编程老师，擅长用生动的比喻来解释复杂的编程概念。请保持回答简洁明了。"),
        few_shot_chat_prompt,
        ("human", "{input}")
    ])

    # 5. 格式化消息
    messages = final_prompt.format_messages(input="解释什么是数据库索引")

    print("📋 生成的聊天消息：")
    for i, msg in enumerate(messages, 1):
        print(f"  {i}. [{msg.__class__.__name__}]: {msg.content}")
        if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
            print(f"     额外信息: {msg.additional_kwargs}")

    print("-" * 40)

    # 6. 调用模型
    response = model.invoke(messages)
    print(f"🤖 GLM-4.6 回答：")
    print(f"{response.content}\n")

def sentiment_analysis_chat():
    """情感分析聊天模板示例"""
    print("=" * 60)
    print("😊 情感分析 FewShotChatMessagePromptTemplate 示例")
    print("=" * 60)

    model = get_glm_model(temperature=0.2)  # 情感分析使用较低温度

    # 情感分析示例
    sentiment_examples = [
        {
            "text": "今天真是太棒了！我完成了所有的任务。",
            "sentiment": "正面"
        },
        {
            "text": "这个产品质量很差，完全不值得购买。",
            "sentiment": "负面"
        },
        {
            "text": "这部电影还可以，没有特别的惊喜。",
            "sentiment": "中性"
        }
    ]

    # 情感分析提示模板
    sentiment_example_prompt = ChatPromptTemplate.from_messages([
        ("human", "请分析以下文本的情感：{text}"),
        ("ai", "情感分析结果：{sentiment}")
    ])

    # 创建少样本模板
    few_shot_sentiment = FewShotChatMessagePromptTemplate(
        examples=sentiment_examples,
        example_prompt=sentiment_example_prompt
    )

    # 最终提示模板
    final_sentiment_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个情感分析专家，请准确判断文本的情感倾向：正面、负面或中性。"),
        few_shot_sentiment,
        ("human", "请分析以下文本的情感：{input}")
    ])

    # 创建处理链
    chain = final_sentiment_prompt | model | StrOutputParser()

    # 测试情感分析
    test_texts = [
        "虽然遇到了一些困难，但最终还是解决了问题。",
        "服务态度恶劣，完全不会再来这里了。",
        "这次的产品体验一般般，没什么特别的感觉。"
    ]

    for text in test_texts:
        print(f"📝 分析文本：{text}")
        result = chain.invoke({"input": text})
        print(f"🎯 分析结果：{result}\n")

def dynamic_few_shot():
    """动态选择示例的少样本模板"""
    print("=" * 60)
    print("🔄 动态选择示例的 FewShotPromptTemplate")
    print("=" * 60)

    model = get_glm_model()

    # 数学题示例库
    math_examples = [
        {
            "difficulty": "easy",
            "question": "2 + 2 = ?",
            "answer": "4"
        },
        {
            "difficulty": "easy",
            "question": "5 × 3 = ?",
            "answer": "15"
        },
        {
            "difficulty": "medium",
            "question": "12 ÷ 4 = ?",
            "answer": "3"
        },
        {
            "difficulty": "medium",
            "question": "7² = ?",
            "answer": "49"
        },
        {
            "difficulty": "hard",
            "question": "√144 = ?",
            "answer": "12"
        }
    ]

    def select_examples_by_difficulty(examples: List[Dict], difficulty: str, max_examples: int = 2) -> List[Dict]:
        """根据难度选择示例"""
        filtered = [ex for ex in examples if ex["difficulty"] == difficulty]
        return filtered[:max_examples]

    # 基础提示模板
    math_prompt = PromptTemplate(
        input_variables=["question", "answer"],
        template="题目：{question}\n答案：{answer}"
    )

    def create_dynamic_few_shot(difficulty: str):
        """创建动态少样本模板"""
        selected_examples = select_examples_by_difficulty(math_examples, difficulty)

        return FewShotPromptTemplate(
            examples=selected_examples,
            example_prompt=math_prompt,
            prefix="请根据以下示例计算数学题：",
            suffix="题目：{input}\n答案：",
            input_variables=["input"]
        )

    # 测试不同难度
    difficulties = ["easy", "medium", "hard"]
    test_questions = {
        "easy": "3 + 6 = ?",
        "medium": "8 × 7 = ?",
        "hard": "2³ = ?"
    }

    for difficulty in difficulties:
        print(f"🎯 难度级别：{difficulty.upper()}")
        dynamic_prompt = create_dynamic_few_shot(difficulty)

        formatted_prompt = dynamic_prompt.format(input=test_questions[difficulty])
        print(f"📋 生成的提示词：\n{formatted_prompt}")

        response = model.invoke(formatted_prompt)
        print(f"🤖 GLM-4.6 回答：{response.content}")
        print("-" * 40)

def few_shot_chain_example():
    """少样本模板链式调用示例"""
    print("=" * 60)
    print("⛓️ 少样本模板链式调用示例")
    print("=" * 60)

    model = get_glm_model()

    # 1. 创建分类示例
    classification_examples = [
        {
            "text": "苹果、香蕉、橙子",
            "category": "水果"
        },
        {
            "text": "汽车、火车、飞机",
            "category": "交通工具"
        },
        {
            "text": "狗、猫、兔子",
            "category": "动物"
        }
    ]

    # 2. 分类提示模板
    classification_prompt = PromptTemplate(
        input_variables=["text", "category"],
        template="文本：{text}\n分类：{category}"
    )

    # 3. 创建少样本模板
    few_shot_classification = FewShotPromptTemplate(
        examples=classification_examples,
        example_prompt=classification_prompt,
        prefix="请将以下文本分类到合适的类别：",
        suffix="文本：{input}\n分类：",
        input_variables=["input"]
    )

    # 4. 创建处理链
    chain = few_shot_classification | model | StrOutputParser()

    # 5. 批量测试
    test_items = [
        "西红柿、黄瓜、白菜",
        "自行车、地铁、公交车",
        "狮子、老虎、大象"
    ]

    print("🔄 批量分类测试：")
    for item in test_items:
        result = chain.invoke({"input": item})
        print(f"📝 {item} → {result}")

    print()

def best_practices():
    """少样本模板最佳实践"""
    print("=" * 60)
    print("💡 少样本模板最佳实践")
    print("=" * 60)

    print("""
✅ 推荐做法:
1. 选择高质量、代表性的示例
2. 示例数量通常在3-10个之间
3. 保持示例格式的一致性
4. 根据任务难度调整示例复杂度
5. 使用动态选择来优化示例相关性

❌ 避免做法:
1. 使用过多或过少的示例
2. 选择低质量或有偏见的示例
3. 示例之间格式不一致
4. 忽略示例的代表性

🎯 示例选择策略:
- 简单任务：3-5个示例
- 复杂任务：5-10个示例
- 根据输入特征动态选择
- 覆盖不同的输入模式

📊 性能优化:
- 缓存常用的少样本模板
- 预计算示例的选择
- 使用链式调用提高效率
    """)

def comparison_summary():
    """两种模板的对比总结"""
    print("=" * 60)
    print("📊 FewShotPromptTemplate vs FewShotChatMessagePromptTemplate")
    print("=" * 60)

    comparison_data = [
        ["特性", "FewShotPromptTemplate", "FewShotChatMessagePromptTemplate"],
        ["格式", "纯文本格式", "消息格式"],
        ["适用场景", "简单文本生成任务", "对话和聊天任务"],
        ["灵活性", "较低", "较高"],
        ["消息类型", "不区分", "区分用户/助手消息"],
        ["复杂度", "简单", "中等"],
        ["控制力", "完全控制格式", "遵循聊天格式"],
        ["最佳用途", "文本补全、翻译", "对话系统、问答"]
    ]

    print("📋 功能对比：")
    for row in comparison_data:
        print(f"  {row[0]:<15} | {row[1]:<25} | {row[2]:<25}")

    print("\n💡 选择建议：")
    print("  📝 使用 FewShotPromptTemplate 当你需要：")
    print("     - 简单的文本生成")
    print("     - 翻译任务")
    print("     - 代码生成")
    print("     - 格式化输出")

    print("\n  💬 使用 FewShotChatMessagePromptTemplate 当你需要：")
    print("     - 对话系统")
    print("     - 问答任务")
    print("     - 情感分析")
    print("     - 复杂的推理任务")

def main():
    """主函数：运行所有示例"""
    print("🚀 GLM-4.6 + 少样本学习模板详细使用示例")
    print("=" * 80)

    try:
        # 运行各种示例
        basic_few_shot_prompt()
        translation_few_shot()
        code_generation_few_shot()
        basic_few_shot_chat()
        sentiment_analysis_chat()
        dynamic_few_shot()
        few_shot_chain_example()
        best_practices()
        comparison_summary()

        print("🎉 所有示例运行完成！")
        print("\n📚 更多信息请参考：")
        print("- LangChain官方文档: https://python.langchain.com/")
        print("- 少样本学习指南: https://www.promptingguide.ai/techniques/fewshot")

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")

if __name__ == "__main__":
    main()