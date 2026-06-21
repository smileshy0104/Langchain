import os
import dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

# 加载环境变量 - 从项目根目录加载.env文件
dotenv.load_dotenv(dotenv_path="../../.env")

# 检查API密钥
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key or api_key == "your-zhipu-api-key-here":
    print("❌ 错误：请在.env文件中设置您的ZHIPUAI_API_KEY")
    exit(1)

# 初始化GLM模型
llm = ChatZhipuAI(
    model="glm-4.6",
    temperature=0.7,
    api_key=api_key
)

def prompt_template_demo():
    """PromptTemplate 基础演示"""
    print("🚀 GLM-4.6 + PromptTemplate 基础演示")
    print("=" * 50)

    # 1. 基础模板使用
    print("\n1️⃣ 基础模板示例")
    prompt_template = PromptTemplate.from_template(
        template="请评价{product}的优缺点，包括{aspect1}和{aspect2}。"
    )

    # 使用format方法
    prompt = prompt_template.format(product="笔记本电脑", aspect1="性能", aspect2="电池续航")
    print(f"📋 生成的提示词: {prompt}")
    print(f"📊 提示词类型: {type(prompt)}")

    # 2. 使用invoke方法 (推荐)
    print("\n2️⃣ 使用invoke方法")
    prompt_dict = prompt_template.invoke({
        "product": "智能手机",
        "aspect1": "拍照质量",
        "aspect2": "系统流畅度"
    })
    print(f"📋 生成的提示词: {prompt_dict}")
    print(f"📊 提示词类型: {type(prompt_dict)}")

    # 3. 创建处理链
    print("\n3️⃣ 创建处理链")
    chain = prompt_template | llm | StrOutputParser()

    result = chain.invoke({
        "product": "平板电脑",
        "aspect1": "屏幕显示",
        "aspect2": "便携性"
    })
    print(f"🤖 GLM-4.6 回答:")
    print(f"{result}\n")

def chat_prompt_template_demo():
    """ChatPromptTemplate 演示"""
    print("💬 ChatPromptTemplate 演示")
    print("=" * 50)

    # 创建聊天模板
    chat_prompt = ChatPromptTemplate.from_template(
        "你是一个专业的产品评测专家。请{action}：{topic}，重点关注{aspects}。"
    )

    # 格式化聊天提示词
    messages = chat_prompt.format_messages(
        action="评价",
        topic="无线耳机",
        aspects="音质、续航、舒适度"
    )

    print("📋 生成的聊天消息:")
    for i, msg in enumerate(messages, 1):
        print(f"  {i}. [{msg.__class__.__name__}]: {msg.content}")

    # 调用模型
    response = llm.invoke(messages)
    print(f"\n🤖 GLM-4.6 回答:")
    print(f"{response.content}\n")

def advanced_template_usage():
    """高级模板使用示例"""
    print("🔧 高级模板使用示例")
    print("=" * 50)

    # 1. 复杂多变量模板
    print("1️⃣ 复杂多变量模板")
    complex_template = PromptTemplate(
        template="""
产品名称: {product}
评测维度: {aspects}
目标用户: {target_audience}
评测风格: {style}

请为{target_audience}写一份关于{product}的{style}评测报告，重点关注：{aspects}。
        """.strip(),
        input_variables=["product", "aspects", "target_audience", "style"]
    )

    # 创建处理链
    chain = complex_template | llm | StrOutputParser()

    result = chain.invoke({
        "product": "机械键盘",
        "aspects": "手感、声音、耐用性",
        "target_audience": "游戏玩家",
        "style": "专业详细"
    })

    print(f"🤖 GLM-4.6 生成的评测报告:")
    print(f"{result[:300]}...\n")

    # 2. 模板组合
    print("2️⃣ 模板组合示例")

    # 系统提示模板
    system_template = PromptTemplate.from_template(
        "你是一个{role}，专门{specialty}。"
    )

    # 用户提示模板
    user_template = PromptTemplate.from_template(
        "请{action}：{topic}"
    )

    # 组合成聊天模板
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", user_template)
    ])

    # 创建链
    chain = chat_prompt | llm | StrOutputParser()

    result = chain.invoke({
        "role": "科技评测博主",
        "specialty": "数码产品评测",
        "action": "评测智能手表",
        "topic": "Apple Watch Series 9"
    })

    print(f"🤖 GLM-4.6 的回答:")
    print(f"{result[:300]}...\n")

def best_practices():
    """最佳实践示例"""
    print("💡 PromptTemplate 最佳实践")
    print("=" * 50)

    print("""
✅ 推荐做法:
1. 使用描述性的变量名
2. 提供清晰的模板结构
3. 使用invoke方法而非format方法
4. 创建处理链以提高代码可读性
5. 为复杂模板添加注释

❌ 避免做法:
1. 使用模糊的变量名
2. 创建过于复杂的单一模板
3. 硬编码变量值
4. 忽略错误处理

🔧 代码示例:
    """)

    # 好的示例
    print("好的示例:")
    good_template = PromptTemplate.from_template(
        "请为{target_audience}解释{concept}，使用{language_style}的语言。"
    )

    chain = good_template | llm | StrOutputParser()
    result = chain.invoke({
        "target_audience": "初学者",
        "concept": "什么是API",
        "language_style": "简单易懂"
    })
    print(f"结果: {result[:100]}...")

if __name__ == "__main__":
    """主函数：运行所有演示"""
    try:
        prompt_template_demo()
        chat_prompt_template_demo()
        advanced_template_usage()
        best_practices()

        print("🎉 PromptTemplate 演示完成！")

    except Exception as e:
        print(f"❌ 运行出错: {e}")