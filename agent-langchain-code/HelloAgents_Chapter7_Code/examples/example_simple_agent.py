"""
示例 1: SimpleAgent 基础用法
展示如何使用 SimpleAgent 进行简单对话和工具调用
"""

import os
import sys

# 设置 API Key
os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import setup_llm
from agents.simple_agent_langchain import SimpleAgent
from tools.calculator_tool import CalculatorTool
from tools.search_tool import MockSearchTool

def main():
    print("🚀 SimpleAgent 示例")
    print("=" * 60)

    # 1. 初始化 LLM
    try:
        llm = setup_llm(model="glm-4-flash")
    except Exception as e:
        print(f"❌ LLM 初始化失败: {e}")
        print("请检查 .env 文件是否配置了 ZHIPUAI_API_KEY")
        return

    # 2. 基础对话示例（无工具）
    print("\n📝 示例 A: 基础对话（无工具）")
    print("-" * 40)
    
    agent = SimpleAgent(
        name="聊天助手",
        llm=llm,
        enable_tool_calling=False
    )
    
    question = "请用一句话介绍什么是 Python。"
    print(f"用户: {question}")
    response = agent.run(question)
    print(f"助手: {response}")

    # 3. 工具调用示例
    print("\n📝 示例 B: 工具调用")
    print("-" * 40)
    
    # 准备工具
    calculator = CalculatorTool()
    search = MockSearchTool()
    
    agent_with_tools = SimpleAgent(
        name="工具助手",
        llm=llm,
        tools=[calculator, search],
        enable_tool_calling=True
    )
    
    # 测试数学计算
    question = "计算 123 * 45 + 678"
    print(f"\n用户: {question}")
    response = agent_with_tools.run(question)
    print(f"助手: {response}")
    
    # 测试搜索
    question = "LangChain 是什么？"
    print(f"\n用户: {question}")
    response = agent_with_tools.run(question)
    print(f"助手: {response}")

    print("\n✅ 示例运行完成")

if __name__ == "__main__":
    main()