"""
运行所有习题的测试脚本
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_header(title: str):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_exercise_02():
    """运行习题2: 多模型支持"""
    print_header("习题 2: 多模型支持")

    try:
        import exercise_02_new_model_provider as ex2

        # 只测试智谱AI (其他需要额外配置)
        ex2.test_zhipuai()

        print("\n✅ 习题2 测试完成")
    except Exception as e:
        print(f"\n❌ 习题2 测试失败: {e}")


def run_exercise_04():
    """运行习题4: 自定义工具"""
    print_header("习题 4: 自定义工具开发")

    try:
        import exercise_04_custom_tools as ex4

        # 运行部分演示
        ex4.demo_file_tools()
        ex4.demo_json_tools()
        ex4.demo_datetime_tools()
        ex4.demo_text_tools()

        print("\n✅ 习题4 测试完成")
    except Exception as e:
        print(f"\n❌ 习题4 测试失败: {e}")
        import traceback
        traceback.print_exc()


def run_exercise_05():
    """运行习题5: 插件系统"""
    print_header("习题 5: 插件系统架构")

    try:
        import exercise_05_plugin_system as ex5

        # 运行演示
        ex5.demo_basic_plugin_system()
        ex5.demo_plugin_lifecycle()
        ex5.demo_plugin_dependency()

        print("\n✅ 习题5 测试完成")
    except Exception as e:
        print(f"\n❌ 习题5 测试失败: {e}")
        import traceback
        traceback.print_exc()


def show_exercise_01():
    """显示习题1信息"""
    print_header("习题 1: 框架设计理念分析")
    print("""
这是一道思考题,请阅读:
  📄 exercises/exercise_01_framework_analysis.md

主要内容:
  ✅ "万物皆工具"设计理念的优点
  ❌ "万物皆工具"设计理念的缺点
  🎯 综合评价和选择建议
  💡 折中方案设计
    """)


def show_exercise_03():
    """显示习题3信息"""
    print_header("习题 3: Agent 实现对比")
    print("""
这是一道分析题,请阅读:
  📄 exercises/exercise_03_agent_comparison.md

主要内容:
  🤖 四种 Agent 架构详细对比
  📊 性能指标和适用场景
  🎯 选择决策树
  💼 实际应用案例
    """)


def main():
    """主函数"""
    print("=" * 70)
    print("  Hello Agents 第七章 - 习题测试套件")
    print("=" * 70)

    print("""
本测试套件包含以下习题:

  📚 习题1: 框架设计理念分析 (阅读)
  💻 习题2: 多模型支持 (代码)
  📊 习题3: Agent 实现对比 (阅读)
  🔧 习题4: 自定义工具开发 (代码)
  🔌 习题5: 插件系统架构 (代码)

开始运行测试...
    """)

    # 显示阅读类习题
    show_exercise_01()
    show_exercise_03()

    # 运行代码类习题
    run_exercise_02()
    run_exercise_04()
    run_exercise_05()

    # 总结
    print("\n" + "=" * 70)
    print("  测试总结")
    print("=" * 70)
    print("""
✅ 完成情况:
  - 习题1: 请阅读 exercise_01_framework_analysis.md
  - 习题2: 多模型支持测试完成
  - 习题3: 请阅读 exercise_03_agent_comparison.md
  - 习题4: 自定义工具测试完成
  - 习题5: 插件系统测试完成

💡 下一步:
  1. 仔细阅读习题1和习题3的分析文档
  2. 尝试修改和扩展习题2、4、5的代码
  3. 思考每道习题后面的扩展问题
  4. 结合实际项目应用所学知识

📚 学习建议:
  - 动手实践比只看文档更重要
  - 尝试创建自己的工具和插件
  - 对比不同框架的设计思路
  - 思考如何应用到实际项目中
    """)

    print("\n" + "=" * 70)
    print("  所有测试完成! 🎉")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
