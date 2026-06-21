#!/usr/bin/env python3
"""
GLM-4.6 + LangChain v1.0 Runnable 链式调用示例
演示使用 LangChain v1.0 的 Runnable 和管道语法进行链式调用
已弃用 LLMChain 和 SequentialChain，使用 prompt | model | output_parser 语法
"""

import os
import dotenv
from typing import Dict, Any
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

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


# ========== 使用 Runnable 的链式调用示例 ==========

def simple_sequential_chain_example():
    """Simple Sequential Chain 简单示例 - 使用现代语法"""
    print("=" * 60)
    print("🔗 Simple Sequential Chain 简单示例 (v1.0)")
    print("=" * 60)

    model = get_glm_model()

    # 第一步：生成故事主题
    story_prompt = PromptTemplate.from_template(
        "请为一个儿童故事生成一个有趣的主题，主题关于{topic}。"
    ) | model | StrOutputParser()

    # 第二步：基于主题写故事
    write_prompt = PromptTemplate.from_template(
        "基于以下主题：{story_theme}\n请写一个简短的儿童故事，适合5-8岁的孩子。"
    ) | model | StrOutputParser()

    # 第三步：为故事添加寓意
    moral_prompt = PromptTemplate.from_template(
        "基于以下故事：{story}\n请为这个故事写一个简单的寓意总结。"
    ) | model | StrOutputParser()

    # 使用现代语法创建顺序执行的链
    print("🚀 使用现代管道语法:")

    # 完整的管道 - 顺序执行
    full_chain = (
        {
            "story_theme": story_prompt,
            "topic": RunnablePassthrough()
        }
        | RunnablePassthrough.assign(
            story=lambda x: write_prompt.invoke({"story_theme": x["story_theme"]})
        )
        | RunnablePassthrough.assign(
            moral=lambda x: moral_prompt.invoke({"story": x["story"]})
        )
    )

    # 运行链
    print("🚀 开始运行现代语法的链...")
    result = full_chain.invoke("友谊")

    print(f"\n🎉 最终结果:")
    print(f"📖 故事主题: {result['story_theme']}")
    print(f"📝 故事内容: {result['story'][:200]}...")
    print(f"💡 故事寓意: {result['moral']}")


def sequential_chain_example():
    """Sequential Chain 复杂示例 - 使用现代语法"""
    print("\n" + "=" * 60)
    print("🔗 Sequential Chain 复杂示例 (v1.0)")
    print("=" * 60)

    model = get_glm_model()

    # 步骤1：生成大纲
    outline_chain = (
        PromptTemplate.from_template(
            "为{audience}写一篇关于{topic}的文章大纲。文章风格：{style}。"
        ) | model | StrOutputParser()
    )

    # 步骤2：根据大纲生成内容
    content_chain = (
        PromptTemplate.from_template(
            "基于以下大纲：{outline}\n请为{audience}写一篇关于{topic}的详细文章。文章风格：{style}，字数约{word_count}字。"
        ) | model | StrOutputParser()
    )

    # 步骤3：生成标题
    title_chain = (
        PromptTemplate.from_template(
            "基于以下文章内容：{content}\n请生成一个吸引人的标题和简短摘要。"
        ) | model | StrOutputParser()
    )

    # 步骤4：生成关键词
    keywords_chain = (
        PromptTemplate.from_template(
            "基于以下内容：{content}\n请提取5个相关的关键词。"
        ) | model | StrOutputParser()
    )

    # 使用现代语法的并行和顺序执行
    overall_chain = (
        {
            "outline": lambda x: outline_chain.invoke({
                "audience": x["audience"],
                "topic": x["topic"],
                "style": x["style"]
            })
        }
        | RunnablePassthrough.assign(
            content=lambda x: content_chain.invoke({
                "outline": x["outline"],
                "audience": x["audience"],
                "topic": x["topic"],
                "style": x["style"],
                "word_count": x["word_count"]
            })
        )
        | RunnablePassthrough.assign(
            title_and_summary=lambda x: title_chain.invoke({"content": x["content"]})
        )
        | RunnablePassthrough.assign(
            keywords=lambda x: keywords_chain.invoke({"content": x["content"]})
        )
    )

    # 运行链
    print("🚀 开始运行现代语法的链...")
    result = overall_chain.invoke({
        "topic": "人工智能的未来",
        "audience": "科技爱好者",
        "style": "专业但不失通俗",
        "word_count": "800"
    })

    print(f"\n🎉 最终结果:")
    print(f"📋 大纲: {result['outline'][:200]}...")
    print(f"📝 内容: {result['content'][:300]}...")
    print(f"🏷️ 标题和摘要: {result['title_and_summary']}")
    print(f"🔑 关键词: {result['keywords']}")


def practical_content_creation_chain():
    """实用内容创建链 - 博客文章生成器 (现代语法)"""
    print("\n" + "=" * 60)
    print("📝 实用内容创建链 - 博客文章生成器 (v1.0)")
    print("=" * 60)

    model = get_glm_model()

    # 步骤1：市场研究
    research_chain = (
        PromptTemplate.from_template(
            "针对主题'{topic}'，进行简单的市场分析。包括：1)目标受众 2)竞争情况 3)内容机会"
        ) | model | StrOutputParser()
    )

    # 步骤2：内容规划
    planning_chain = (
        PromptTemplate.from_template(
            "基于研究：{research}\n为'{topic}'制定内容计划：1)核心观点 2)文章结构 3)关键信息点"
        ) | model | StrOutputParser()
    )

    # 步骤3：草稿撰写
    draft_chain = (
        PromptTemplate.from_template(
            "基于计划：{plan}\n撰写关于'{topic}'的博客草稿，要求：{requirements}"
        ) | model | StrOutputParser()
    )

    # 步骤4：内容优化
    optimize_chain = (
        PromptTemplate.from_template(
            "优化以下草稿：{draft}\n改进要求：1)增强可读性 2)添加CTA 3)SEO优化建议"
        ) | model | StrOutputParser()
    )

    # 步骤5：社交媒体推广
    social_chain = (
        PromptTemplate.from_template(
            "基于优化内容：{optimized_content}\n创作3条社交媒体推广文案（Twitter、LinkedIn、Facebook）"
        ) | model | StrOutputParser()
    )

    # 使用现代语法创建完整的内容创建链
    content_creation_chain = (
        {
            "research": lambda x: research_chain.invoke({"topic": x["topic"]})
        }
        | RunnablePassthrough.assign(
            plan=lambda x: planning_chain.invoke({
                "research": x["research"],
                "topic": x["topic"]
            })
        )
        | RunnablePassthrough.assign(
            draft=lambda x: draft_chain.invoke({
                "plan": x["plan"],
                "topic": x["topic"],
                "requirements": x["requirements"]
            })
        )
        | RunnablePassthrough.assign(
            optimized_content=lambda x: optimize_chain.invoke({"draft": x["draft"]})
        )
        | RunnablePassthrough.assign(
            social_posts=lambda x: social_chain.invoke({"optimized_content": x["optimized_content"]})
        )
    )

    # 运行内容创建链
    print("🚀 开始运行现代语法的内容创建链...")
    result = content_creation_chain.invoke({
        "topic": "远程工作的效率提升技巧",
        "requirements": "实用性强，包含具体案例，适合职场人士阅读，长度1000字左右"
    })

    print(f"\n🎉 内容创建完成!")
    print(f"🔍 市场研究: {result['research'][:200]}...")
    print(f"📋 内容计划: {result['plan'][:200]}...")
    print(f"📝 优化后内容: {result['optimized_content'][:300]}...")
    print(f"📱 社交媒体文案: {result['social_posts']}")


def translation_chain_example():
    """翻译链示例 - 现代语法的实际应用"""
    print("\n" + "=" * 60)
    print("🌐 翻译链示例 (v1.0)")
    print("=" * 60)

    model = get_glm_model()

    # 第一步：检测语言
    detect_chain = (
        PromptTemplate.from_template(
            "检测以下文本的语言：{text}\n只回答语言名称，如：中文、英文、日文等。"
        ) | model | StrOutputParser()
    )

    # 第二步：翻译成英文
    translate_chain = (
        PromptTemplate.from_template(
            "将以下{detected_language}文本翻译成自然流畅的英文：{original_text}\n保持原文的意思和语气。"
        ) | model | StrOutputParser()
    )

    # 第三步：生成翻译摘要
    summary_chain = (
        PromptTemplate.from_template(
            "基于原文：{original_text}\n和译文：{translated_text}\n请分析翻译质量，并给出改进建议。"
        ) | model | StrOutputParser()
    )

    # 使用现代语法创建翻译链
    translation_chain = (
        {
            "original_text": RunnablePassthrough(),
            "detected_language": lambda x: detect_chain.invoke({"text": x})
        }
        | RunnablePassthrough.assign(
            translated_text=lambda x: translate_chain.invoke({
                "detected_language": x["detected_language"],
                "original_text": x["original_text"]
            })
        )
        | RunnablePassthrough.assign(
            translation_summary=lambda x: summary_chain.invoke({
                "original_text": x["original_text"],
                "translated_text": x["translated_text"]
            })
        )
    )

    # 测试翻译
    test_text = "人工智能正在改变我们的生活方式，从智能手机到自动驾驶汽车，AI技术无处不在。"

    print("🚀 开始运行现代语法的翻译链...")
    result = translation_chain.invoke(test_text)

    print(f"\n🎉 翻译分析结果:")
    print(f"🌍 检测到的语言: {result['detected_language']}")
    print(f"📝 原文: {result['original_text']}")
    print(f"🔄 译文: {result['translated_text'][:200]}...")
    print(f"📊 翻译分析: {result['translation_summary']}")


def parallel_processing_example():
    """并行处理示例 - 使用 RunnableParallel"""
    print("\n" + "=" * 60)
    print("⚡ 并行处理示例 - RunnableParallel (v1.0)")
    print("=" * 60)

    model = get_glm_model()

    # 三个独立的分析任务
    sentiment_chain = (
        PromptTemplate.from_template(
            "分析以下文本的情感倾向：{text}\n请从积极、中性、消极中选择一个。"
        ) | model | StrOutputParser()
    )

    summary_chain = (
        PromptTemplate.from_template(
            "请总结以下文本的核心观点：{text}\n"
        ) | model | StrOutputParser()
    )

    keywords_chain = (
        PromptTemplate.from_template(
            "从以下文本中提取3个关键词：{text}\n"
        ) | model | StrOutputParser()
    )

    # 使用 RunnableParallel 并行处理
    parallel_chain = RunnableParallel(
        sentiment=sentiment_chain,
        summary=summary_chain,
        keywords=keywords_chain
    )

    test_text = "人工智能技术的发展为我们的生活带来了巨大的变化。它不仅提高了工作效率，还创造了新的可能性。"

    print("🚀 开始并行处理...")
    result = parallel_chain.invoke({"text": test_text})

    print(f"\n🎉 并行处理结果:")
    print(f"😊 情感分析: {result['sentiment']}")
    print(f"📝 文本摘要: {result['summary']}")
    print(f"🔑 关键词: {result['keywords']}")


def chain_comparison():
    """链类型比较和最佳实践 - 现代语法 vs 传统语法"""
    print("\n" + "=" * 60)
    print("⚖️ 链类型比较和最佳实践 (v1.0)")
    print("=" * 60)

    print("""
📊 传统 LLMChain/SequentialChain vs 现代 Runnable 语法:

🔗 传统 LLMChain 语法 (已移除):
❌ 缺点:
   - LLMChain 在 LangChain 0.1.17 中已弃用
   - 在 LangChain 1.0 中已完全移除
   - SequentialChain 也被移除
   - 配置复杂，需要明确指定输出键
   - 不够灵活

✅ 现代 Runnable 语法 (v1.0 推荐):
✅ 优点:
   - 使用管道操作符 | 进行链式组合
   - 更简洁直观的语法
   - 使用 RunnablePassthrough 和 RunnableParallel 提供灵活性
   - 更好的类型支持和错误处理
   - 符合函数式编程理念
   - 支持并行处理

🎯 迁移建议:
1. 替换 LLMChain → prompt | model | output_parser
2. 使用 RunnablePassthrough 替代复杂的手动数据传递
3. 使用 RunnableParallel 进行并行处理
4. 采用现代的管道语法提高代码可读性

🔧 代码示例对比:

传统语法 (已移除):
   from langchain.chains import LLMChain
   chain = LLMChain(llm=model, prompt=prompt, output_key="result")

现代语法 (v1.0):
   chain = prompt | model | StrOutputParser()
   result = chain.invoke({"input": "..."})

顺序执行:
   chain = (
       prompt1 | model1 | output_parser1
       | RunnablePassthrough.assign(
           next_step=lambda x: prompt2 | model2 | output_parser2
       )
   )

并行处理:
   parallel_chain = RunnableParallel(
       task1=prompt1 | model1 | output_parser1,
       task2=prompt2 | model2 | output_parser2
   )
    """)


def error_handling_example():
    """错误处理示例 - 现代语法"""
    print("\n" + "=" * 60)
    print("⚠️ 错误处理示例 (v1.0)")
    print("=" * 60)

    model = get_glm_model()

    try:
        # 创建一个可能失败的链 - 使用现代语法
        risky_chain = (
            PromptTemplate.from_template(
                "请{action}关于{topic}的内容。"
            ) | model | StrOutputParser()
        )

        print("🚀 运行可能失败的任务...")
        result = risky_chain.invoke({
            "action": "生成",
            "topic": "一个不存在的概念"
        })

        print(f"✅ 任务成功完成: {result[:100]}...")

    except Exception as e:
        print(f"❌ 链执行出错: {e}")
        print("💡 建议：检查输入参数和提示词模板")


def main():
    """主函数：运行所有示例"""
    print("🚀 GLM-4.6 + LangChain v1.0 Runnable 链式调用详细示例")
    print("=" * 80)
    print("""
✨ LangChain v1.0 主要变化:
1. 移除了 LLMChain 和 SequentialChain
2. 全面采用 Runnable 架构
3. 使用管道操作符 | 进行链式调用
4. 支持并行处理 (RunnableParallel)
5. 更简洁、更灵活的语法
    """)
    print()

    try:
        # 运行各种示例
        simple_sequential_chain_example()
        sequential_chain_example()
        practical_content_creation_chain()
        translation_chain_example()
        parallel_processing_example()
        chain_comparison()
        error_handling_example()

        print("\n🎉 所有示例运行完成！")
        print("\n📚 更多信息请参考：")
        print("- LangChain v1.0 文档: https://python.langchain.com/")
        print("- Runnable API: https://python.langchain.com/docs/concepts/runnables/")

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
