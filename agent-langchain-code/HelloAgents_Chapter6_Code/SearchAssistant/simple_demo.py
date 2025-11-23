#!/usr/bin/env python3
"""
智能搜索助手 - 简化演示版本

这是一个快速演示版本，使用简单的查询避免长时间运行。

特点：
- 使用模拟搜索（不需要真实搜索API）
- 快速验证 LangGraph 工作流
- 展示状态图的多节点协作

适合：
- 快速测试和演示
- 理解 LangGraph 状态机
- 验证环境配置

如需集成真实搜索，请参考 search_assistant_langgraph.py
"""

import os
import sys

# 添加 Chapter4 目录到路径以导入工具模块
chapter4_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "HelloAgents_Chapter4_Code")
sys.path.insert(0, os.path.abspath(chapter4_path))

from search_assistant_langgraph import SearchAssistant


def demo_basic_search():
    """演示1: 基础搜索（推荐用于测试）"""
    print("="*80)
    print("📌 快速演示: 基础信息搜索")
    print("="*80)
    print("\n使用模拟搜索数据库，快速展示搜索流程。\n")

    assistant = SearchAssistant(use_memory=False, debug=True)

    queries = [
        "什么是 LangChain？",
        "Python 有什么特点？",
    ]

    for query in queries:
        print(f"\n{'='*80}")
        print(f"用户查询: {query}")
        print(f"{'='*80}\n")

        answer = assistant.search(query)

        print(f"\n💡 最终答案:")
        print(f"{answer}")
        print(f"\n{'-'*80}")

    print("\n✅ 演示完成！可以看到完整的三步流程：")
    print("   1. 🤔 理解查询并生成搜索关键词")
    print("   2. 🔎 执行搜索")
    print("   3. 💡 基于搜索结果生成答案")


def demo_conversation():
    """演示2: 多轮对话（带记忆）"""
    print("\n" + "="*80)
    print("📌 快速演示: 多轮对话")
    print("="*80)
    print("\n展示对话记忆功能，助手能记住上下文。\n")

    assistant = SearchAssistant(use_memory=True, debug=False)

    conversation = [
        "什么是智谱AI？",
        "它的主要产品是什么？",  # 测试上下文理解
    ]

    thread_id = "demo_conversation"

    for i, user_input in enumerate(conversation, 1):
        print(f"\n--- 第 {i} 轮对话 ---")
        print(f"👤 用户: {user_input}")

        response = assistant.chat(user_input, thread_id=thread_id)

        print(f"🤖 助手: {response}")

    print("\n✅ 演示完成！可以看到助手理解了上下文（\"它\" 指代 \"智谱AI\"）")


def demo_technical_query():
    """演示3: 技术问题查询"""
    print("\n" + "="*80)
    print("📌 快速演示: 技术问题查询")
    print("="*80)

    assistant = SearchAssistant(temperature=0.5, debug=True)

    query = "LangGraph 的主要特点是什么？"

    print(f"\n查询: {query}\n")
    answer = assistant.search(query)

    print(f"\n完整答案:\n{answer}")


def main():
    """主函数：运行演示"""
    print("\n" + "="*80)
    print("🚀 LangGraph 智能搜索助手 - 快速演示")
    print("="*80)
    print("\n这些演示使用模拟搜索，快速展示 LangGraph 工作流。")
    print("每个演示通常在30秒内完成。\n")

    # 检查 API 密钥
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key or api_key.startswith("your-"):
        print("❌ 错误：请在 .env 文件中设置您的 ZHIPUAI_API_KEY")
        print("📝 获取 API 密钥：https://open.bigmodel.cn/")
        return

    try:
        # 演示1：基础搜索（推荐用于首次测试）
        demo_basic_search()

        # 可选：取消注释以运行更多演示
        # demo_conversation()
        # demo_technical_query()

        print("\n" + "="*80)
        print("✅ 所有演示完成！")
        print("="*80)
        print("\n💡 关于 LangGraph 工作流:")
        print("   - 状态图: START → 理解查询 → 搜索 → 生成答案 → END")
        print("   - 每个节点都是独立的函数")
        print("   - 状态在节点间传递和更新")
        print("   - 支持记忆功能（多轮对话）")
        print("\n💡 集成真实搜索:")
        print("   - 在 search_information_node 中替换为真实搜索API")
        print("   - 支持: Tavily, SerpAPI, Google Search 等")
        print("   - 参考 search_assistant_langgraph.py 中的注释")
        print("\n💡 完整功能:")
        print("   - 参考 search_assistant_langgraph.py")
        print("   - 查看 README.md 了解更多示例")
        print("="*80 + "\n")

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序。")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
        print("\n常见问题:")
        print("1. 超时错误：检查网络连接，或任务太复杂")
        print("2. API 错误：检查 API 密钥是否正确，是否有足够的配额")
        print("3. 导入错误：确保已运行 quick_test.py 验证依赖")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
