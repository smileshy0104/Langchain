#!/usr/bin/env python3
"""
快速运行所有示例的脚本
"""

import os
import sys
import subprocess


def print_header(title):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def run_example(script_name, description):
    """运行单个示例"""
    print_header(description)
    print(f"正在运行: {script_name}\n")

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=False,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            print(f"\n✅ {script_name} 运行成功")
        else:
            print(f"\n❌ {script_name} 运行失败")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"\n⏱️  {script_name} 运行超时（超过120秒）")
        return False
    except Exception as e:
        print(f"\n❌ 运行 {script_name} 时出错: {str(e)}")
        return False


def main():
    """主函数"""
    print_header("LangChain 短期记忆示例 - 全部运行")

    # 检查环境变量
    if not os.getenv("ZHIPUAI_API_KEY"):
        print("⚠️  警告: 未设置 ZHIPUAI_API_KEY 环境变量")
        print("请先设置: export ZHIPUAI_API_KEY='your-api-key'\n")
        response = input("是否继续运行？(y/n): ")
        if response.lower() != 'y':
            print("已取消")
            return

    # 所有示例
    examples = [
        ("01_basic_memory.py", "示例1：基础短期记忆"),
        ("02_multi_thread.py", "示例2：多线程会话管理"),
        ("03_trim_messages.py", "示例3：消息修剪"),
        ("04_summarization.py", "示例4：消息摘要"),
        ("05_custom_state.py", "示例5：自定义状态"),
        ("06_tool_state_access.py", "示例6：工具读写状态"),
    ]

    # 交互式选择
    print("请选择运行模式：")
    print("  1. 运行所有示例")
    print("  2. 选择单个示例")
    print("  3. 退出")

    choice = input("\n请输入选项（1-3）: ").strip()

    if choice == "1":
        # 运行所有示例
        results = []
        for script, description in examples:
            success = run_example(script, description)
            results.append((script, success))

            if success:
                input("\n按 Enter 继续下一个示例...")

        # 总结
        print_header("运行总结")
        for script, success in results:
            status = "✅ 成功" if success else "❌ 失败"
            print(f"{status} - {script}")

    elif choice == "2":
        # 选择单个示例
        print("\n可用示例：")
        for i, (script, description) in enumerate(examples, 1):
            print(f"  {i}. {description} ({script})")

        try:
            example_choice = int(input("\n请选择示例（1-6）: ").strip())
            if 1 <= example_choice <= len(examples):
                script, description = examples[example_choice - 1]
                run_example(script, description)
            else:
                print("无效的选择")
        except ValueError:
            print("请输入有效的数字")

    elif choice == "3":
        print("已退出")
    else:
        print("无效的选择")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已中断")
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
