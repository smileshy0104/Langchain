"""
示例 3: ReflectionAgent 用法
展示如何使用自我反思 Agent 提升写作质量
"""

import os
import sys

# 设置 API Key
os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import setup_llm
from agents.reflection_agent_langchain import ReflectionAgent

def main():
    print("🚀 ReflectionAgent 示例")
    print("=" * 60)

    # 1. 初始化
    try:
        llm = setup_llm(model="glm-4-flash")
    except Exception as e:
        print(f"❌ LLM 初始化失败: {e}")
        return

    agent = ReflectionAgent(
        name="写作助手",
        llm=llm,
        max_reflections=2  # 反思 2 轮
    )

    # 2. 写作任务测试
    task = """
写一篇关于"为什么即使有AI，学习编程仍然很重要"的短文。
要求：
1. 观点鲜明，逻辑清晰
2. 包含具体的例子
3. 篇幅 150 字左右
4. 风格积极向上
"""
    
    print(f"\n📝 写作任务: {task}")
    print("-" * 40)
    
    final_article = agent.run(task)
    
    print(f"\n✅ 最终成品:\n")
    print(final_article)

if __name__ == "__main__":
    main()