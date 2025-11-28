"""
示例 4: PlanAndSolveAgent 用法
展示如何使用计划与执行 Agent 解决复杂的多步骤问题
"""

import os
import sys
from dotenv import load_dotenv

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import setup_llm
from agents.plan_solve_agent_langchain import PlanAndSolveAgent

# 加载环境变量
load_dotenv()

def main():
    print("🚀 PlanAndSolveAgent 示例")
    print("=" * 60)

    # 1. 初始化
    try:
        llm = setup_llm(model="glm-4-flash")
    except Exception as e:
        print(f"❌ LLM 初始化失败: {e}")
        return

    agent = PlanAndSolveAgent(
        name="策划助手",
        llm=llm
    )

    # 2. 复杂任务测试
    # 这是一个经典的需要分步骤解决的数学问题
    task = """
一个水果店周一卖了15个苹果。
周二卖的苹果数量是周一的2倍。
周三卖的苹果比周二少5个。
请问这三天总共卖了多少个苹果？
"""
    
    print(f"\n📝 复杂任务: {task}")
    print("-" * 40)
    
    final_answer = agent.run(task)
    
    print(f"\n✅ 最终答案:\n{final_answer}")

    # 3. 另一个任务：旅行计划
    task2 = "帮我制定一个去云南大理的三天旅游计划，包括交通、住宿建议和每天的行程。"
    print(f"\n\n📝 任务 2: {task2}")
    print("-" * 40)
    
    final_answer2 = agent.run(task2)
    print(f"\n✅ 最终计划:\n{final_answer2}")

if __name__ == "__main__":
    main()