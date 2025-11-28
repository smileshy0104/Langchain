"""
示例 2: ReActAgent 用法
展示 ReAct Agent 如何通过推理和行动解决复杂问题
"""

import os
import sys
from dotenv import load_dotenv

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import setup_llm
from agents.react_agent_langchain import ReActAgent
from tools.calculator_tool import CalculatorTool
from tools.search_tool import MockSearchTool

# 加载环境变量
load_dotenv()

def main():
    print("🚀 ReActAgent 示例")
    print("=" * 60)

    # 1. 初始化
    try:
        llm = setup_llm(model="glm-4-flash")
    except Exception as e:
        print(f"❌ LLM 初始化失败: {e}")
        return

    calculator = CalculatorTool()
    search = MockSearchTool()
    
    agent = ReActAgent(
        name="ReAct助手",
        llm=llm,
        tools=[calculator, search],
        max_steps=10
    )

    # 2. 复杂任务测试
    # 这个任务需要：
    # 1. 搜索北京的天气 (MOCK_SEARCH_DB 中有 "北京" 和 "天气")
    # 2. 假设需要根据温度计算穿衣指数（模拟计算需求）
    
    task = "查询北京的信息，并计算 365 * 24 是多少，最后告诉我北京适合旅游吗？"
    
    print(f"\n📝 任务: {task}")
    print("-" * 40)
    
    response = agent.run(task)
    
    print(f"\n✅ 最终答案:\n{response}")

    # 3. 另一个测试
    task2 = "先搜索'机器学习'的定义，然后告诉我它和'人工智能'的关系。"
    print(f"\n\n📝 任务 2: {task2}")
    print("-" * 40)
    
    response2 = agent.run(task2)
    print(f"\n✅ 最终答案:\n{response2}")

if __name__ == "__main__":
    main()