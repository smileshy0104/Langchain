#!/usr/bin/env python3
"""
快速测试脚本 - 验证所有 Chapter 6 示例是否正常工作

使用方法:
    python quick_test.py

要求:
    - 设置 ZHIPUAI_API_KEY 环境变量
    - 安装必要的依赖包
"""

import os
import sys


def test_imports():
    """测试所有必要的导入"""
    print("="*80)
    print("🧪 测试 1: 验证依赖导入")
    print("="*80)

    try:
        import langchain
        print("✅ langchain")
    except ImportError as e:
        print(f"❌ langchain: {e}")
        return False

    try:
        from langchain_core.messages import HumanMessage
        print("✅ langchain_core")
    except ImportError as e:
        print(f"❌ langchain_core: {e}")
        return False

    try:
        from langgraph.graph import StateGraph
        print("✅ langgraph")
    except ImportError as e:
        print(f"❌ langgraph: {e}")
        return False

    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv")
    except ImportError as e:
        print(f"❌ python-dotenv: {e}")
        return False

    # 测试导入 Chapter4 工具
    chapter4_path = os.path.join(os.path.dirname(__file__), "..", "HelloAgents_Chapter4_Code")
    sys.path.insert(0, os.path.abspath(chapter4_path))

    try:
        from utils import get_llm
        print("✅ utils.get_llm (from Chapter 4)")
    except ImportError as e:
        print(f"❌ utils.get_llm: {e}")
        return False

    print("\n✅ 所有依赖导入成功！\n")
    return True


def test_api_key():
    """测试 API 密钥配置"""
    print("="*80)
    print("🧪 测试 2: 验证 API 密钥配置")
    print("="*80)

    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("ZHIPUAI_API_KEY")

    if not api_key or api_key.startswith("your-"):
        print("❌ ZHIPUAI_API_KEY 未配置或无效")
        print("\n配置方法:")
        print("1. 创建 .env 文件")
        print("2. 添加: ZHIPUAI_API_KEY=your-actual-api-key")
        print("3. 获取 API 密钥: https://open.bigmodel.cn/")
        return False

    print(f"✅ ZHIPUAI_API_KEY 已配置: {api_key[:10]}...")
    print("\n✅ API 密钥配置正确！\n")
    return True


def test_software_team():
    """测试软件开发团队示例"""
    print("="*80)
    print("🧪 测试 3: 软件开发团队协作")
    print("="*80)

    try:
        # 添加 SoftwareTeam 目录到路径
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SoftwareTeam"))
        from software_team_langchain import SoftwareTeamAgent

        print("✅ 成功导入 SoftwareTeamAgent")

        # 创建实例（不运行，仅测试初始化）
        team = SoftwareTeamAgent(debug=False)
        print("✅ 成功创建 SoftwareTeamAgent 实例")

        print("\n✅ 软件团队模块测试通过！\n")
        return True

    except Exception as e:
        print(f"❌ 软件团队测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_role_playing():
    """测试角色扮演示例"""
    print("="*80)
    print("🧪 测试 4: 角色扮演协作系统")
    print("="*80)

    try:
        # 添加 BookWriting 目录到路径
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "BookWriting"))
        from role_playing_langchain import RolePlayingSession

        print("✅ 成功导入 RolePlayingSession")

        # 创建实例（不运行，仅测试初始化）
        session = RolePlayingSession(
            assistant_role="测试专家",
            user_role="测试执行者",
            task="这是一个测试任务",
            debug=False
        )
        print("✅ 成功创建 RolePlayingSession 实例")

        print("\n✅ 角色扮演模块测试通过！\n")
        return True

    except Exception as e:
        print(f"❌ 角色扮演测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_search_assistant():
    """测试智能搜索助手示例"""
    print("="*80)
    print("🧪 测试 5: LangGraph 智能搜索助手")
    print("="*80)

    try:
        # 添加 SearchAssistant 目录到路径
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SearchAssistant"))
        from search_assistant_langgraph import SearchAssistant

        print("✅ 成功导入 SearchAssistant")

        # 创建实例（不运行，仅测试初始化）
        assistant = SearchAssistant(use_memory=False, debug=False)
        print("✅ 成功创建 SearchAssistant 实例")

        print("\n✅ 搜索助手模块测试通过！\n")
        return True

    except Exception as e:
        print(f"❌ 搜索助手测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("🚀 Hello-Agents Chapter 6 - LangChain v1.0 实现快速测试")
    print("="*80 + "\n")

    results = []

    # 测试 1: 依赖导入
    results.append(("依赖导入", test_imports()))

    # 测试 2: API 密钥
    results.append(("API 密钥", test_api_key()))

    # 测试 3-5: 各个模块
    results.append(("软件团队", test_software_team()))
    results.append(("角色扮演", test_role_playing()))
    results.append(("搜索助手", test_search_assistant()))

    # 汇总结果
    print("="*80)
    print("📊 测试结果汇总")
    print("="*80)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:12} - {status}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n" + "="*80)
        print("🎉 所有测试通过！您可以开始使用 Chapter 6 示例了。")
        print("="*80)
        print("\n运行示例:")
        print("  python SoftwareTeam/software_team_langchain.py")
        print("  python BookWriting/role_playing_langchain.py")
        print("  python SearchAssistant/search_assistant_langgraph.py")
        print("\n查看文档:")
        print("  cat README.md")
        print("="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("⚠️ 部分测试失败，请检查上述错误信息。")
        print("="*80 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
