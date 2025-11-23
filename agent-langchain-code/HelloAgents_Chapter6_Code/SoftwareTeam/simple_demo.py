#!/usr/bin/env python3
"""
软件开发团队协作 - 简化演示版本

这是一个快速演示版本，使用简单的任务避免长时间运行。

适合：
- 快速测试和演示
- 理解团队协作流程
- 验证环境配置

如需完整功能，请参考 software_team_langchain.py
"""

import os
import sys

# 添加 Chapter4 目录到路径以导入工具模块
chapter4_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "HelloAgents_Chapter4_Code")
sys.path.insert(0, os.path.abspath(chapter4_path))

from software_team_langchain import SoftwareTeamAgent


def demo_simple_function():
    """演示1: 简单函数开发（推荐用于测试）"""
    print("="*80)
    print("📌 快速演示: 简单函数开发")
    print("="*80)
    print("\n这是一个简化的演示，开发一个简单的Python函数。\n")

    team = SoftwareTeamAgent(temperature=0.3, debug=True)

    task = """开发一个 Python 函数，计算斐波那契数列的第 n 项。

要求：
1. 函数签名：def fibonacci(n: int) -> int
2. 包含文档字符串
3. 简单实现即可（不要求最优算法）
4. 添加基本的输入验证"""

    print("提示: 这个演示通常在1-2分钟内完成。\n")

    results = team.run(task)

    print("\n" + "="*80)
    print("📊 协作结果摘要")
    print("="*80)
    print(f"\n✅ 产品经理完成需求分析")
    print(f"✅ 工程师完成代码实现")
    print(f"✅ 审查员完成代码审查")
    print("\n查看上方输出了解详细过程。")
    print("="*80)


def demo_data_processing():
    """演示2: 数据处理脚本"""
    print("\n" + "="*80)
    print("📌 快速演示: 数据处理脚本")
    print("="*80)

    team = SoftwareTeamAgent(temperature=0.2, debug=True)

    task = """编写一个 Python 函数，读取 CSV 文件并统计数值列的平均值。

要求：
1. 使用 pandas 库
2. 函数签名：def calculate_average(file_path: str, column_name: str) -> float
3. 包含错误处理（文件不存在、列不存在等）
4. 添加文档字符串"""

    results = team.run(task)
    print(f"\n✅ 团队协作完成！")


def demo_api_client():
    """演示3: API 客户端"""
    print("\n" + "="*80)
    print("📌 快速演示: API 客户端函数")
    print("="*80)

    team = SoftwareTeamAgent(temperature=0.3, debug=True)

    task = """开发一个函数，调用 JSONPlaceholder API 获取用户信息。

要求：
1. API 端点：https://jsonplaceholder.typicode.com/users/{id}
2. 函数签名：def get_user(user_id: int) -> dict
3. 使用 requests 库
4. 错误处理：网络错误、用户不存在等
5. 返回用户的 name 和 email"""

    results = team.run(task)
    print(f"\n✅ 团队协作完成！")


def main():
    """主函数：运行演示"""
    print("\n" + "="*80)
    print("🚀 软件开发团队协作系统 - 快速演示")
    print("="*80)
    print("\n这些演示使用简单任务，每个通常在1-2分钟内完成。\n")

    # 检查 API 密钥
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key or api_key.startswith("your-"):
        print("❌ 错误：请在 .env 文件中设置您的 ZHIPUAI_API_KEY")
        print("📝 获取 API 密钥：https://open.bigmodel.cn/")
        return

    try:
        # 演示1：简单函数（推荐用于首次测试）
        demo_simple_function()

        # 可选：取消注释以运行更多演示
        # demo_data_processing()
        # demo_api_client()

        print("\n" + "="*80)
        print("✅ 演示完成！")
        print("="*80)
        print("\n💡 提示：")
        print("  - 每个角色的输出已在上方显示")
        print("  - 产品经理: 需求分析和技术规划")
        print("  - 工程师: 代码实现")
        print("  - 代码审查员: 质量检查和改进建议")
        print("\n  - 完整功能请参考 software_team_langchain.py")
        print("  - 多轮迭代请查看 MultiRoundCollaboration 类")
        print("="*80 + "\n")

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序。")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
        print("\n常见问题:")
        print("1. 超时错误：任务可能太复杂，尝试简化任务描述")
        print("2. API 错误：检查 API 密钥是否正确，是否有足够的配额")
        print("3. 网络错误：检查网络连接")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
