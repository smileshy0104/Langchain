#!/usr/bin/env python3
"""
测试 Plan-and-Solve 修复后的版本
"""

from utils import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

# 定义计划结构
class Plan(BaseModel):
    steps: List[str] = Field(description="步骤列表")

def test_simple_prompt():
    """测试简单提示词格式"""
    print("🧪 测试1: 简单提示词格式（应该工作）\n")

    llm = get_llm(provider="zhipuai", model="glm-4.6", temperature=0.3)

    # 使用 from_template（简单格式）
    prompt = ChatPromptTemplate.from_template("你是AI助手。\n\n问题: {question}\n\n请回答:")

    chain = prompt | llm

    try:
        result = chain.invoke({"question": "1+1等于几？"})
        print(f"✅ 成功！")
        print(f"响应: {result.content[:100]}\n")
    except Exception as e:
        print(f"❌ 失败: {e}\n")


def test_messages_format():
    """测试消息格式"""
    print("🧪 测试2: from_messages 格式（可能失败）\n")

    llm = get_llm(provider="zhipuai", model="glm-4.6", temperature=0.3)

    # 使用 from_messages（可能有问题）
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是AI助手。"),
        ("human", "问题: {question}")
    ])

    chain = prompt | llm

    try:
        result = chain.invoke({"question": "1+1等于几？"})
        print(f"✅ 成功！")
        print(f"响应: {result.content[:100]}\n")
    except Exception as e:
        print(f"❌ 失败: {e}\n")


def test_json_output():
    """测试 JSON 输出解析"""
    print("🧪 测试3: JSON 输出解析\n")

    llm = get_llm(provider="zhipuai", model="glm-4.6", temperature=0.3)
    parser = JsonOutputParser(pydantic_object=Plan)

    prompt = ChatPromptTemplate.from_template("""你是规划专家。

{format_instructions}

问题: {question}

请输出 JSON 格式的计划:""")

    chain = prompt.partial(
        format_instructions=parser.get_format_instructions()
    ) | llm | parser

    try:
        result = chain.invoke({
            "question": "如何做一道番茄炒蛋？"
        })
        print(f"✅ 成功！")
        print(f"计划: {result}\n")
    except Exception as e:
        print(f"❌ 失败: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("="*70)
    print("🔍 Plan-and-Solve 提示词格式测试")
    print("="*70)
    print()

    test_simple_prompt()
    test_messages_format()
    test_json_output()

    print("="*70)
    print("✨ 测试完成")
    print("="*70)
