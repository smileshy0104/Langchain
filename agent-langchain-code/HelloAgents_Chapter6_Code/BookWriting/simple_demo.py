#!/usr/bin/env python3
"""
角色扮演协作系统 - 简化演示版本

这是一个快速演示版本，使用简短的任务避免长时间运行和超时问题。

适合：
- 快速测试和演示
- 理解角色扮演协作机制
- 验证环境配置

如需完整功能，请参考 role_playing_langchain.py
"""

import os
import sys

# 添加 Chapter4 目录到路径以导入工具模块
chapter4_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "HelloAgents_Chapter4_Code")
sys.path.insert(0, os.path.abspath(chapter4_path))

from role_playing_langchain import RolePlayingSession


def demo_outline_creation():
    """演示1: 快速大纲创作（推荐用于测试）"""
    print("="*80)
    print("📌 快速演示: Python 教程大纲创作")
    print("="*80)
    print("\n这是一个简化的演示，只需要2-3轮对话即可完成。\n")

    task = """创建一个 Python 入门教程的大纲。

要求：
1. 包含3个核心章节
2. 每个章节2-3个要点
3. 目标读者：完全没有编程经验的初学者
4. 完成后明确说"大纲完成"

注意：只需要大纲框架，不需要详细内容。"""

    session = RolePlayingSession(
        assistant_role="Python讲师",
        user_role="教学设计师",
        task=task,
        temperature=0.6,
        max_turns=3,  # 只需要3轮对话
        debug=True
    )

    print("\n提示: 这个演示通常在1-2分钟内完成。\n")

    conversation = session.run()

    print(f"\n✅ 完成！共进行了 {len(conversation)} 轮对话")


def demo_technical_review():
    """演示2: 技术方案评审"""
    print("\n" + "="*80)
    print("📌 快速演示: 技术方案评审")
    print("="*80)

    task = """评审以下技术方案：

方案：使用 Redis 作为分布式锁实现微服务的并发控制。

请从以下角度评审：
1. 可行性
2. 潜在风险
3. 一个改进建议

完成后说"评审完成"。"""

    session = RolePlayingSession(
        assistant_role="架构师",
        user_role="开发工程师",
        task=task,
        temperature=0.4,
        max_turns=2,  # 2轮对话足够
        debug=True
    )

    conversation = session.run()
    print(f"\n✅ 完成！共进行了 {len(conversation)} 轮对话")


def demo_simple_qa():
    """演示3: 简单问答协作"""
    print("\n" + "="*80)
    print("📌 快速演示: 学习辅导问答")
    print("="*80)

    task = """请帮我理解：什么是 LangChain 的 LCEL？

要求：
1. 用1-2句话解释概念
2. 给出一个简单的代码示例
3. 说明它的主要优势

回答后说"解答完成"。"""

    session = RolePlayingSession(
        assistant_role="LangChain专家",
        user_role="学习者",
        task=task,
        temperature=0.5,
        max_turns=2,
        debug=True
    )

    conversation = session.run()
    print(f"\n✅ 完成！共进行了 {len(conversation)} 轮对话")


def main():
    """主函数：运行演示"""
    print("\n" + "="*80)
    print("🚀 角色扮演协作系统 - 快速演示")
    print("="*80)
    print("\n这些演示使用简短任务，避免长时间运行和超时问题。")
    print("每个演示通常在1-2分钟内完成。\n")

    # 检查 API 密钥
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key or api_key.startswith("your-"):
        print("❌ 错误：请在 .env 文件中设置您的 ZHIPUAI_API_KEY")
        print("📝 获取 API 密钥：https://open.bigmodel.cn/")
        return

    try:
        # 演示1：大纲创作（推荐用于首次测试）
        demo_outline_creation()

        # 可选：取消注释以运行更多演示
        # demo_technical_review()
        # demo_simple_qa()

        print("\n" + "="*80)
        print("✅ 演示完成！")
        print("="*80)
        print("\n💡 提示：")
        print("  - 查看生成的对话历史文件了解详细过程")
        print("  - 修改 max_turns 参数控制对话轮次")
        print("  - 修改 temperature 参数调整创造性")
        print("  - 完整功能请参考 role_playing_langchain.py")
        print("="*80 + "\n")

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序。")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
        print("\n常见问题:")
        print("1. 超时错误：任务可能太复杂，尝试简化任务描述或减少 max_turns")
        print("2. API 错误：检查 API 密钥是否正确，是否有足够的配额")
        print("3. 网络错误：检查网络连接")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
