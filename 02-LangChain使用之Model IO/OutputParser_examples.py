#!/usr/bin/env python3
"""
GLM-4.6 + LangChain 输出解析器详细使用示例
演示 StrOutputParser、JsonOutputParser、XMLOutputParser 的使用方法
"""

import os
import dotenv
import json
import xml.etree.ElementTree as ET
from typing import List
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    XMLOutputParser,
    PydanticOutputParser,
    CommaSeparatedListOutputParser
)
from langchain_core.pydantic_v1 import BaseModel, Field

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

def str_output_parser_basic():
    """StrOutputParser 基础示例"""
    print("=" * 60)
    print("📝 StrOutputParser 基础示例")
    print("=" * 60)

    model = get_glm_model()
    parser = StrOutputParser()

    # 1. 简单文本处理
    print("1️⃣ 简单文本处理")
    prompt = ChatPromptTemplate.from_template(
        "请用一句话概括{topic}的核心概念。"
    )

    # 创建处理链
    chain = prompt | model | parser

    result = chain.invoke({"topic": "机器学习"})
    print(f"📋 输入: 机器学习")
    print(f"🤖 输出: {result}")
    print(f"📊 输出类型: {type(result)}")
    print()

    # 2. 多步骤处理
    print("2️⃣ 多步骤文本处理")
    complex_prompt = ChatPromptTemplate.from_template(
        """
        任务：处理以下文本

        文本：{text}
        要求：
        1. 提取关键信息
        2. 总结要点
        3. 用简洁的语言重新表述
        """
    )

    complex_chain = complex_prompt | model | parser

    result = complex_chain.invoke({
        "text": "人工智能（AI）是计算机科学的一个分支，它致力于创建能够执行通常需要人类智能的任务的系统。AI包括机器学习、深度学习、自然语言处理等多个子领域。"
    })

    print(f"📋 原始文本: 人工智能相关描述")
    print(f"🤖 处理结果: {result}\n")

def str_output_parser_advanced():
    """StrOutputParser 高级示例"""
    print("=" * 60)
    print("🔧 StrOutputParser 高级示例")
    print("=" * 60)

    model = get_glm_model()
    parser = StrOutputParser()

    # 1. 批量处理
    print("1️⃣ 批量文本处理")
    batch_prompt = ChatPromptTemplate.from_template(
        "请将以下内容总结为{length}的摘要：{content}"
    )

    batch_chain = batch_prompt | model | parser

    contents = [
        ("Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名。", "一句话"),
        ("机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习并做出预测。", "50字以内"),
        ("深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的工作方式。", "简洁的要点")
    ]

    for content, length in contents:
        result = batch_chain.invoke({"content": content, "length": length})
        print(f"📋 {length}摘要: {result}")

    print()

    # 2. 格式化输出
    print("2️⃣ 格式化文本输出")
    format_prompt = ChatPromptTemplate.from_template(
        """
        请按照以下格式输出信息：

        标题：{title}
        作者：{author}
        类型：{genre}
        简介：{description}
        """
    )

    format_chain = format_prompt | model | parser

    result = format_chain.invoke({
        "title": "三体",
        "author": "刘慈欣",
        "genre": "科幻小说",
        "description": "描述地球文明与三体文明的首次接触"
    })

    print(f"📋 格式化输出:\n{result}\n")

def json_output_parser_basic():
    """JsonOutputParser 基础示例"""
    print("=" * 60)
    print("📋 JsonOutputParser 基础示例")
    print("=" * 60)

    model = get_glm_model(temperature=0.3)  # JSON输出使用较低温度
    parser = JsonOutputParser()

    # 1. 简单JSON输出
    print("1️⃣ 简单JSON输出")
    json_prompt = ChatPromptTemplate.from_template(
        """
        请将以下信息转换为JSON格式：

        姓名：{name}
        年龄：{age}
        职业：{occupation}

        输出格式：{format_instructions}
        """
    )

    json_chain = json_prompt | model | parser

    result = json_chain.invoke({
        "name": "张三",
        "age": "28",
        "occupation": "软件工程师",
        "format_instructions": parser.get_format_instructions()
    })

    print(f"📋 输入信息: 张三, 28岁, 软件工程师")
    print(f"🤖 JSON输出: {json.dumps(result, ensure_ascii=False, indent=2)}")
    print(f"📊 输出类型: {type(result)}")
    print()

    # 2. 复杂JSON结构
    print("2️⃣ 复杂JSON结构")
    complex_json_prompt = ChatPromptTemplate.from_template(
        """
        请创建一个产品信息的JSON对象，包含以下信息：

        产品名称：{product_name}
        价格：{price}
        特性：{features}
        评分：{rating}

        输出格式：{format_instructions}
        """
    )

    complex_json_chain = complex_json_prompt | model | parser

    result = complex_json_chain.invoke({
        "product_name": "智能手机",
        "price": "2999元",
        "features": "5G网络, 高像素摄像头, 快速充电",
        "rating": "4.5/5.0",
        "format_instructions": parser.get_format_instructions()
    })

    print(f"📋 产品信息JSON:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()

def json_output_parser_advanced():
    """JsonOutputParser 高级示例"""
    print("=" * 60)
    print("🔧 JsonOutputParser 高级示例")
    print("=" * 60)

    model = get_glm_model(temperature=0.2)
    parser = JsonOutputParser()

    # 1. 数组输出
    print("1️⃣ JSON数组输出")
    array_prompt = ChatPromptTemplate.from_template(
        """
        请创建一个包含{category}的JSON数组，每个项目包含名称和描述。

        示例格式：
        [
            {{"name": "项目1", "description": "描述1"}},
            {{"name": "项目2", "description": "描述2"}}
        ]

        输出格式：{format_instructions}
        """
    )

    array_chain = array_prompt | model | parser

    result = array_chain.invoke({
        "category": "编程语言",
        "format_instructions": parser.get_format_instructions()
    })

    print(f"📋 编程语言数组:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()

    # 2. 嵌套JSON
    print("2️⃣ 嵌套JSON结构")
    nested_prompt = ChatPromptTemplate.from_template(
        """
        请创建一个公司信息的嵌套JSON结构，包含基本信息、部门和员工。

        输出格式：{format_instructions}
        """
    )

    nested_chain = nested_prompt | model | parser

    result = nested_chain.invoke({
        "format_instructions": parser.get_format_instructions()
    })

    print(f"📋 公司信息嵌套JSON:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()

def pydantic_output_parser_example():
    """PydanticOutputParser 示例"""
    print("=" * 60)
    print("🏗️ PydanticOutputParser 示例")
    print("=" * 60)

    # 定义Pydantic模型
    class Product(BaseModel):
        name: str = Field(description="产品名称")
        price: float = Field(description="产品价格")
        category: str = Field(description="产品类别")
        features: List[str] = Field(description="产品特性列表")
        in_stock: bool = Field(description="是否有库存")

    model = get_glm_model(temperature=0.3)
    parser = PydanticOutputParser(pydantic_object=Product)

    print("1️⃣ 结构化数据提取")
    product_prompt = ChatPromptTemplate.from_template(
        """
        请从以下文本中提取产品信息，并按照指定格式输出：

        文本：{text}

        输出格式：{format_instructions}
        """
    )

    product_chain = product_prompt | model | parser

    text = "iPhone 15 Pro售价8999元，属于智能手机类别，具有A17芯片、钛合金材质、48MP相机等特性，目前有现货。"

    try:
        result = product_chain.invoke({
            "text": text,
            "format_instructions": parser.get_format_instructions()
        })

        print(f"📋 输入文本: {text}")
        print(f"🤖 结构化输出:")
        print(f"  产品名称: {result.name}")
        print(f"  价格: {result.price}")
        print(f"  类别: {result.category}")
        print(f"  特性: {', '.join(result.features)}")
        print(f"  有库存: {result.in_stock}")
        print(f"📊 输出类型: {type(result)}")

    except Exception as e:
        print(f"❌ 解析失败: {e}")

    print()

def xml_output_parser_basic():
    """XMLOutputParser 基础示例"""
    print("=" * 60)
    print("📄 XMLOutputParser 基础示例")
    print("=" * 60)

    model = get_glm_model(temperature=0.3)
    parser = XMLOutputParser()

    # 1. 简单XML输出
    print("1️⃣ 简单XML输出")
    xml_prompt = ChatPromptTemplate.from_template(
        """
        请将以下信息转换为XML格式：

        标题：{title}
        作者：{author}
        内容：{content}

        输出格式：{format_instructions}
        """
    )

    xml_chain = xml_prompt | model | parser

    result = xml_chain.invoke({
        "title": "Python入门",
        "author": "程序员",
        "content": "Python是一种简单易学的编程语言。",
        "format_instructions": parser.get_format_instructions()
    })

    print(f"📋 输入信息: Python入门, 程序员, Python简介")
    print(f"🤖 XML输出:")
    print(result)
    print(f"📊 输出类型: {type(result)}")
    print()

    # 2. 复杂XML结构
    print("2️⃣ 复杂XML结构")
    complex_xml_prompt = ChatPromptTemplate.from_template(
        """
        请创建一个包含书籍信息的XML结构：

        书名：{title}
        作者：{author}
        ISBN：{isbn}
        价格：{price}
        章节：{chapters}

        输出格式：{format_instructions}
        """
    )

    complex_xml_chain = complex_xml_prompt | model | parser

    result = complex_xml_chain.invoke({
        "title": "深度学习",
        "author": "Ian Goodfellow",
        "isbn": "978-0262035613",
        "price": "128元",
        "chapters": "数学基础, 机器学习基础, 深度前馈网络",
        "format_instructions": parser.get_format_instructions()
    })

    print(f"📋 书籍信息XML:")
    print(result)
    print()

def xml_output_parser_advanced():
    """XMLOutputParser 高级示例"""
    print("=" * 60)
    print("🔧 XMLOutputParser 高级示例")
    print("=" * 60)

    model = get_glm_model(temperature=0.3)
    parser = XMLOutputParser()

    # 1. 嵌套XML结构
    print("1️⃣ 嵌套XML结构")
    nested_xml_prompt = ChatPromptTemplate.from_template(
        """
        请创建一个包含订单信息的嵌套XML结构，包含订单信息和多个商品项。

        订单号：{order_id}
        客户：{customer}
        日期：{date}
        商品项：{items}

        输出格式：{format_instructions}
        """
    )

    nested_xml_chain = nested_xml_prompt | model | parser

    result = nested_xml_chain.invoke({
        "order_id": "ORD-2024-001",
        "customer": "张三",
        "date": "2024-01-15",
        "items": "笔记本电脑(5999元), 鼠标(99元), 键盘(299元)",
        "format_instructions": parser.get_format_instructions()
    })

    print(f"📋 订单信息XML:")
    print(result)

    # 2. XML解析验证
    print("\n2️⃣ XML解析验证")
    try:
        # 验证XML是否有效
        root = ET.fromstring(result)
        print("✅ XML格式验证成功")
        print(f"📊 根元素: {root.tag}")

        # 遍历子元素
        for child in root:
            print(f"  - {child.tag}: {child.text}")

    except ET.ParseError as e:
        print(f"❌ XML解析失败: {e}")

    print()

def comma_separated_parser_example():
    """CommaSeparatedListOutputParser 示例"""
    print("=" * 60)
    print("📝 CommaSeparatedListOutputParser 示例")
    print("=" * 60)

    model = get_glm_model()
    parser = CommaSeparatedListOutputParser()

    # 列表输出示例
    list_prompt = ChatPromptTemplate.from_template(
        """
        请列出5个与{topic}相关的关键词。

        输出格式：{format_instructions}
        """
    )

    list_chain = list_prompt | model | parser

    result = list_chain.invoke({
        "topic": "人工智能",
        "format_instructions": parser.get_format_instructions()
    })

    print(f"📋 主题: 人工智能")
    print(f"🤖 关键词列表: {result}")
    print(f"📊 输出类型: {type(result)}")
    print(f"📊 列表长度: {len(result)}")
    print()

def parser_comparison():
    """解析器对比示例"""
    print("=" * 60)
    print("📊 输出解析器对比")
    print("=" * 60)

    model = get_glm_model(temperature=0.3)

    # 同一个输入，不同解析器
    input_text = "请介绍Python编程语言的主要特性。"

    parsers = [
        ("StrOutputParser", StrOutputParser()),
        ("JsonOutputParser", JsonOutputParser()),
        ("XMLOutputParser", XMLOutputParser())
    ]

    for parser_name, parser in parsers:
        print(f"🔍 {parser_name} 测试:")

        try:
            if parser_name == "StrOutputParser":
                prompt = ChatPromptTemplate.from_template("{text}")
                chain = prompt | model | parser
                result = chain.invoke({"text": input_text})

            elif parser_name == "JsonOutputParser":
                prompt = ChatPromptTemplate.from_template(
                    "请将以下回答转换为JSON格式，包含'topic'和'features'字段：{text}\n{format_instructions}"
                )
                chain = prompt | model | parser
                result = chain.invoke({
                    "text": input_text,
                    "format_instructions": parser.get_format_instructions()
                })

            elif parser_name == "XMLOutputParser":
                prompt = ChatPromptTemplate.from_template(
                    "请将以下回答转换为XML格式：{text}\n{format_instructions}"
                )
                chain = prompt | model | parser
                result = chain.invoke({
                    "text": input_text,
                    "format_instructions": parser.get_format_instructions()
                })

            print(f"  输出类型: {type(result)}")
            print(f"  输出预览: {str(result)[:100]}...")

        except Exception as e:
            print(f"  ❌ 解析失败: {e}")

        print()

def best_practices():
    """输出解析器最佳实践"""
    print("=" * 60)
    print("💡 输出解析器最佳实践")
    print("=" * 60)

    print("""
✅ 推荐做法:
1. 根据任务需求选择合适的解析器
2. 为结构化输出设置较低的temperature
3. 提供清晰的格式说明
4. 使用try-catch处理解析错误
5. 验证解析结果的正确性

🎯 解析器选择指南:

📝 StrOutputParser:
- 适用: 简单文本生成、摘要、翻译
- 优点: 简单直接，兼容性好
- 缺点: 无结构化，需要后处理

📋 JsonOutputParser:
- 适用: 数据提取、结构化信息、API集成
- 优点: 结构化，易解析，广泛支持
- 缺点: 对模型输出要求高

📄 XMLOutputParser:
- 适用: 文档处理、配置文件、数据交换
- 优点: 自描述性好，层次结构清晰
- 缺点: 冗长，解析复杂

🏗️ PydanticOutputParser:
- 适用: 类型安全的数据结构、API接口
- 优点: 类型检查、数据验证
- 缺点: 需要预定义schema

🚀 性能优化建议:
1. 缓存常用的解析器实例
2. 批量处理时复用解析器
3. 对复杂输出使用流式解析
4. 监控解析成功率
    """)

def error_handling_example():
    """错误处理示例"""
    print("=" * 60)
    print("⚠️ 错误处理示例")
    print("=" * 60)

    model = get_glm_model(temperature=0.1)
    json_parser = JsonOutputParser()

    # 模拟错误的JSON输出
    error_prompt = ChatPromptTemplate.from_template(
        "请输出一个无效的JSON格式：故意制造错误"
    )

    error_chain = error_prompt | model

    print("🔍 错误处理测试:")
    try:
        raw_output = error_chain.invoke({})
        print(f"📋 原始输出: {raw_output}")

        # 尝试解析
        parsed_output = json_parser.parse(raw_output.content)
        print(f"✅ 解析成功: {parsed_output}")

    except Exception as e:
        print(f"❌ 解析失败: {e}")
        print("💡 错误处理建议:")
        print("  1. 检查模型输出的格式")
        print("  2. 降低temperature参数")
        print("  3. 提供更详细的格式说明")
        print("  4. 使用重试机制")

    print()

def main():
    """主函数：运行所有示例"""
    print("🚀 GLM-4.6 + 输出解析器详细使用示例")
    print("=" * 80)

    try:
        # 运行各种示例
        str_output_parser_basic()
        str_output_parser_advanced()
        json_output_parser_basic()
        json_output_parser_advanced()
        pydantic_output_parser_example()
        xml_output_parser_basic()
        xml_output_parser_advanced()
        comma_separated_parser_example()
        parser_comparison()
        best_practices()
        error_handling_example()

        print("🎉 所有示例运行完成！")
        print("\n📚 更多信息请参考：")
        print("- LangChain输出解析器文档: https://python.langchain.com/docs/modules/model_io/output_parsers/")

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")

if __name__ == "__main__":
    main()