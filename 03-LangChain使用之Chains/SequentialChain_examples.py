#!/usr/bin/env python3
"""
GLM-4.6 + LangChain SequentialChain 和 SimpleSequentialChain 示例
演示链式调用的使用方法和最佳实践
"""

import os
import dotenv
from typing import Dict, Any
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

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

def simple_sequential_chain_example():
    """SimpleSequentialChain 简单示例"""
    print("=" * 60)
    print("🔗 SimpleSequentialChain 简单示例")
    print("=" * 60)

    model = get_glm_model()

    # 第一步：生成故事主题
    story_prompt = PromptTemplate.from_template(
        "请为一个儿童故事生成一个有趣的主题，主题关于{topic}。"
    )
    story_chain = LLMChain(llm=model, prompt=story_prompt)

    # 第二步：基于主题写故事
    write_prompt = PromptTemplate.from_template(
        "基于以下主题：{story_theme}\n请写一个简短的儿童故事，适合5-8岁的孩子。"
    )
    write_chain = LLMChain(llm=model, prompt=write_prompt)

    # 第三步：为故事添加寓意
    moral_prompt = PromptTemplate.from_template(
        "基于以下故事：{story}\n请为这个故事写一个简单的寓意总结。"
    )
    moral_chain = LLMChain(llm=model, prompt=moral_prompt)

    # 创建SimpleSequentialChain
    overall_chain = SimpleSequentialChain(
        chains=[story_chain, write_chain, moral_chain],
        verbose=True
    )

    # 运行链
    print("🚀 开始运行SimpleSequentialChain...")
    result = overall_chain.run("友谊")

    print(f"\n🎉 最终结果:")
    print(f"{result}")

def sequential_chain_example():
    """SequentialChain 复杂示例 - 多输入多输出"""
    print("\n" + "=" * 60)
    print("🔗 SequentialChain 复杂示例")
    print("=" * 60)

    model = get_glm_model()

    # 步骤1：生成大纲
    outline_prompt = PromptTemplate.from_template(
        "为{audience}写一篇关于{topic}的文章大纲。文章风格：{style}。"
    )
    outline_chain = LLMChain(
        llm=model,
        prompt=outline_prompt,
        output_key="outline"
    )

    # 步骤2：根据大纲生成内容
    content_prompt = PromptTemplate.from_template(
        "基于以下大纲：{outline}\n请为{audience}写一篇关于{topic}的详细文章。文章风格：{style}，字数约{word_count}字。"
    )
    content_chain = LLMChain(
        llm=model,
        prompt=content_prompt,
        output_key="content"
    )

    # 步骤3：生成标题
    title_prompt = PromptTemplate.from_template(
        "基于以下文章内容：{content}\n请生成一个吸引人的标题和简短摘要。"
    )
    title_chain = LLMChain(
        llm=model,
        prompt=title_prompt,
        output_key="title_and_summary"
    )

    # 步骤4：生成关键词
    keywords_prompt = PromptTemplate.from_template(
        "基于以下内容：{content}\n请提取5个相关的关键词。"
    )
    keywords_chain = LLMChain(
        llm=model,
        prompt=keywords_prompt,
        output_key="keywords"
    )

    # 创建SequentialChain
    overall_chain = SequentialChain(
        chains=[outline_chain, content_chain, title_chain, keywords_chain],
        input_variables=["topic", "audience", "style", "word_count"],
        output_variables=["outline", "content", "title_and_summary", "keywords"],
        verbose=True
    )

    # 运行链
    print("🚀 开始运行SequentialChain...")
    result = overall_chain({
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
    """实用内容创建链 - 博客文章生成器"""
    print("\n" + "=" * 60)
    print("📝 实用内容创建链 - 博客文章生成器")
    print("=" * 60)

    model = get_glm_model()

    # 步骤1：市场研究
    research_prompt = PromptTemplate.from_template(
        "针对主题'{topic}'，进行简单的市场分析。包括：1)目标受众 2)竞争情况 3)内容机会"
    )
    research_chain = LLMChain(
        llm=model,
        prompt=research_prompt,
        output_key="research"
    )

    # 步骤2：内容规划
    planning_prompt = PromptTemplate.from_template(
        "基于研究：{research}\n为'{topic}'制定内容计划：1)核心观点 2)文章结构 3)关键信息点"
    )
    planning_chain = LLMChain(
        llm=model,
        prompt=planning_prompt,
        output_key="plan"
    )

    # 步骤3：草稿撰写
    draft_prompt = PromptTemplate.from_template(
        "基于计划：{plan}\n撰写关于'{topic}'的博客草稿，要求：{requirements}"
    )
    draft_chain = LLMChain(
        llm=model,
        prompt=draft_prompt,
        output_key="draft"
    )

    # 步骤4：内容优化
    optimize_prompt = PromptTemplate.from_template(
        "优化以下草稿：{draft}\n改进要求：1)增强可读性 2)添加CTA 3)SEO优化建议"
    )
    optimize_chain = LLMChain(
        llm=model,
        prompt=optimize_prompt,
        output_key="optimized_content"
    )

    # 步骤5：社交媒体推广
    social_prompt = PromptTemplate.from_template(
        "基于优化内容：{optimized_content}\n创作3条社交媒体推广文案（Twitter、LinkedIn、Facebook）"
    )
    social_chain = LLMChain(
        llm=model,
        prompt=social_prompt,
        output_key="social_posts"
    )

    # 创建完整的内容创建链
    content_creation_chain = SequentialChain(
        chains=[research_chain, planning_chain, draft_chain, optimize_chain, social_chain],
        input_variables=["topic", "requirements"],
        output_variables=["research", "plan", "draft", "optimized_content", "social_posts"],
        verbose=True
    )

    # 运行内容创建链
    print("🚀 开始运行内容创建链...")
    result = content_creation_chain({
        "topic": "远程工作的效率提升技巧",
        "requirements": "实用性强，包含具体案例，适合职场人士阅读，长度1000字左右"
    })

    print(f"\n🎉 内容创建完成!")
    print(f"🔍 市场研究: {result['research'][:200]}...")
    print(f"📋 内容计划: {result['plan'][:200]}...")
    print(f"📝 优化后内容: {result['optimized_content'][:300]}...")
    print(f"📱 社交媒体文案: {result['social_posts']}")

def translation_chain_example():
    """翻译链示例 - SimpleSequentialChain的实际应用"""
    print("\n" + "=" * 60)
    print("🌐 翻译链示例")
    print("=" * 60)

    model = get_glm_model()

    # 第一步：检测语言
    detect_prompt = PromptTemplate.from_template(
        "检测以下文本的语言：{text}\n只回答语言名称，如：中文、英文、日文等。"
    )
    detect_chain = LLMChain(llm=model, prompt=detect_prompt)

    # 第二步：翻译成英文
    translate_prompt = PromptTemplate.from_template(
        "将以下{detected_language}文本翻译成自然流畅的英文：{original_text}\n保持原文的意思和语气。"
    )
    translate_chain = LLMChain(llm=model, prompt=translate_prompt)

    # 第三步：生成翻译摘要
    summary_prompt = PromptTemplate.from_template(
        "基于原文：{original_text}\n和译文：{translated_text}\n请分析翻译质量，并给出改进建议。"
    )
    summary_chain = LLMChain(llm=model, prompt=summary_prompt)

    # 创建翻译链
    translation_chain = SimpleSequentialChain(
        chains=[detect_chain, translate_chain, summary_chain],
        verbose=True
    )

    # 测试翻译
    test_text = "人工智能正在改变我们的生活方式，从智能手机到自动驾驶汽车，AI技术无处不在。"

    print("🚀 开始运行翻译链...")
    result = translation_chain.run(test_text)

    print(f"\n🎉 翻译分析结果:")
    print(f"{result}")

def chain_comparison():
    """链类型比较和最佳实践"""
    print("\n" + "=" * 60)
    print("⚖️ 链类型比较和最佳实践")
    print("=" * 60)

    print("""
📊 SimpleSequentialChain vs SequentialChain:

🔗 SimpleSequentialChain:
✅ 优点:
   - 简单易用，适合线性任务
   - 自动传递前一个链的输出
   - 配置简单

❌ 缺点:
   - 只能处理单一输入输出
   - 无法访问中间结果
   - 不够灵活

🔗 SequentialChain:
✅ 优点:
   - 支持多输入多输出
   - 可以访问所有中间结果
   - 更加灵活和强大
   - 支持复杂的依赖关系

❌ 缺点:
   - 配置相对复杂
   - 需要明确指定输入输出变量

🎯 使用建议:
1. 简单线性任务 → SimpleSequentialChain
2. 复杂多步骤任务 → SequentialChain
3. 需要访问中间结果 → SequentialChain
4. 快速原型开发 → SimpleSequentialChain
    """)

def error_handling_example():
    """错误处理示例"""
    print("\n" + "=" * 60)
    print("⚠️ 错误处理示例")
    print("=" * 60)

    model = get_glm_model()

    try:
        # 创建一个可能失败的链
        risky_prompt = PromptTemplate.from_template(
            "请{action}关于{topic}的内容。"
        )

        risky_chain = LLMChain(llm=model, prompt=risky_prompt)

        # 使用SimpleSequentialChain包装
        safe_chain = SimpleSequentialChain(
            chains=[risky_chain],
            verbose=True
        )

        print("🚀 运行可能失败的任务...")
        result = safe_chain.run({
            "action": "生成",
            "topic": "一个不存在的概念"
        })

        print(f"✅ 任务成功完成: {result[:100]}...")

    except Exception as e:
        print(f"❌ 链执行出错: {e}")
        print("💡 建议：检查输入参数和提示词模板")

def main():
    """主函数：运行所有示例"""
    print("🚀 GLM-4.6 + LangChain SequentialChain 详细使用示例")
    print("=" * 80)

    try:
        # 运行各种示例
        simple_sequential_chain_example()
        sequential_chain_example()
        practical_content_creation_chain()
        translation_chain_example()
        chain_comparison()
        error_handling_example()

        print("\n🎉 所有示例运行完成！")
        print("\n📚 更多信息请参考：")
        print("- LangChain官方文档: https://python.langchain.com/")
        print("- 链式调用指南: https://python.langchain.com/docs/modules/chains/")

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")

if __name__ == "__main__":
    main()