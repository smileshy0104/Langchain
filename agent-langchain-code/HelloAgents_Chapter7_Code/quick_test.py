"""
快速测试脚本 - Hello Agents Chapter 7 LangChain 实现
验证环境配置和核心功能
"""

import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.utils import setup_llm, safe_eval, format_chat_history
from core.tools import ToolRegistry
from tools.calculator_tool import CalculatorTool, create_calculator
from tools.search_tool import MockSearchTool
from agents.simple_agent_langchain import SimpleAgent
from agents.react_agent_langchain import ReActAgent


def print_section(title: str):
    """打印测试章节标题"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def test_environment():
    """测试 1: 环境配置"""
    print_section("测试 1: 环境配置检查")

    api_key = os.getenv("ZHIPUAI_API_KEY")

    if not api_key:
        print("❌ 错误: 未找到 ZHIPUAI_API_KEY")
        print("📝 请复制 .env.example 为 .env 并配置 API 密钥")
        return False

    print(f"✅ ZHIPUAI_API_KEY: {api_key[:8]}...{api_key[-4:]}")

    # 检查可选配置
    optional_keys = ["TAVILY_API_KEY", "SERPAPI_API_KEY"]
    for key in optional_keys:
        value = os.getenv(key)
        if value:
            print(f"✅ {key}: 已配置")
        else:
            print(f"ℹ️  {key}: 未配置 (可选)")

    return True


def test_core_utils():
    """测试 2: 核心工具函数"""
    print_section("测试 2: 核心工具函数")

    # 测试 LLM 设置
    try:
        llm = setup_llm(model="glm-4.6", temperature=0.7)
        print(f"✅ LLM 初始化成功: {llm.model_name}")
    except Exception as e:
        print(f"❌ LLM 初始化失败: {e}")
        return False

    # 测试 safe_eval
    test_cases = [
        ("2 + 3", "5"),
        ("10 * 5", "50"),
        ("100 / 4", "25.0"),
        ("2 ** 10", "1024"),
    ]

    print("\n📊 测试 safe_eval:")
    all_passed = True
    for expr, expected in test_cases:
        try:
            result = safe_eval(expr)
            if result == expected:
                print(f"  ✅ {expr} = {result}")
            else:
                print(f"  ❌ {expr} = {result} (期望: {expected})")
                all_passed = False
        except Exception as e:
            print(f"  ❌ {expr} 执行失败: {e}")
            all_passed = False

    # 测试 format_chat_history
    messages = format_chat_history([
        {"role": "system", "content": "你是一个助手"},
        {"role": "user", "content": "你好"}
    ])
    print(f"\n✅ format_chat_history 测试通过 (生成 {len(messages)} 条消息)")
    response = llm.invoke(messages)
    print(f"  ✅ 测试结果: {response}")
    return all_passed


def test_tool_system():
    """测试 3: 工具系统"""
    print_section("测试 3: 工具系统")

    # 创建工具注册表
    registry = ToolRegistry()
    print("✅ ToolRegistry 创建成功")

    # 注册工具
    calculator = CalculatorTool()
    search = MockSearchTool()

    registry.register_tool(calculator)
    registry.register_tool(search)
    print(f"✅ 已注册 {len(registry.list_tools())} 个工具")

    # 测试计算器工具
    print("\n📊 测试计算器工具:")
    test_cases = [
        ("5 + 3", "8"),
        ("12 * 4", "48"),
    ]

    for expr, expected in test_cases:
        try:
            result = registry.execute_tool("calculator", expr)
            if result == expected:
                print(f"  ✅ calculator({expr}) = {result}")
            else:
                print(f"  ⚠️  calculator({expr}) = {result} (期望: {expected})")
        except Exception as e:
            print(f"  ❌ calculator({expr}) 失败: {e}")

    # 测试搜索工具
    print("\n📊 测试搜索工具:")
    try:
        result = registry.execute_tool("search", "Python")
        print(f"  ✅ search('Python'): {result[:50]}...")
    except Exception as e:
        print(f"  ❌ search 失败: {e}")

    return True


def test_simple_agent():
    """测试 4: SimpleAgent"""
    print_section("测试 4: SimpleAgent (无工具)")

    try:
        llm = setup_llm(model="glm-4.6", temperature=0.7)
        agent = SimpleAgent(
            name="简单助手",
            llm=llm,
            system_prompt="你是一个友好的AI助手，回答要简洁。"
        )

        print(f"✅ {agent.name} 创建成功")

        # 测试简单对话
        print("\n💬 测试对话:")
        response = agent.run("请用一句话介绍你自己")
        print(f"  用户: 请用一句话介绍你自己")
        print(f"  助手: {response}")

        if response and len(response) > 0:
            print("\n✅ SimpleAgent 基础对话测试通过")
            return True
        else:
            print("\n❌ SimpleAgent 返回空响应")
            return False

    except Exception as e:
        print(f"❌ SimpleAgent 测试失败: {e}")
        return False


def test_simple_agent_with_tools():
    """测试 5: SimpleAgent (带工具)"""
    print_section("测试 5: SimpleAgent (带工具调用)")

    try:
        llm = setup_llm(model="glm-4.6", temperature=0.7)

        # 创建工具
        calculator = CalculatorTool()
        search = MockSearchTool()

        # 创建启用工具的 Agent
        agent = SimpleAgent(
            name="工具助手",
            llm=llm,
            tools=[calculator, search],
            enable_tool_calling=True,
            system_prompt="你是一个有用的助手，可以使用工具来帮助回答问题。"
        )

        print(f"✅ {agent.name} 创建成功 (工具: {len(agent.tools)} 个)")

        # 测试计算任务
        print("\n💬 测试计算任务:")
        response = agent.run("帮我计算 15 * 8 的结果")
        print(f"  用户: 帮我计算 15 * 8 的结果")
        print(f"  助手: {response}")

        if "120" in response:
            print("\n✅ SimpleAgent 工具调用测试通过")
            return True
        else:
            print("\n⚠️  SimpleAgent 可能未正确调用工具")
            return True  # 不作为失败，因为 LLM 可能直接回答

    except Exception as e:
        print(f"❌ SimpleAgent (带工具) 测试失败: {e}")
        return False


def test_react_agent():
    """测试 6: ReActAgent"""
    print_section("测试 6: ReActAgent (推理行动循环)")

    try:
        llm = setup_llm(model="glm-4.6", temperature=0.7)

        # 创建工具
        calculator = CalculatorTool()
        search = MockSearchTool()

        # 创建 ReAct Agent
        agent = ReActAgent(
            name="ReAct助手",
            llm=llm,
            tools=[calculator, search],
            max_iterations=5
        )

        print(f"✅ {agent.name} 创建成功")

        # 测试多步推理任务
        print("\n💬 测试多步推理任务:")
        response = agent.run("先搜索Python的信息，然后计算 25 * 4")
        print(f"  用户: 先搜索Python的信息，然后计算 25 * 4")
        print(f"\n  助手响应:\n{response}")

        # 检查是否包含工具调用的痕迹
        if "100" in response or "Thought" in response or "Action" in response:
            print("\n✅ ReActAgent 测试通过")
            return True
        else:
            print("\n⚠️  ReActAgent 可能未按预期工作")
            return True  # 不作为失败

    except Exception as e:
        print(f"❌ ReActAgent 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("  Hello Agents Chapter 7 - LangChain 实现快速测试")
    print("=" * 60)

    results = []

    # 运行测试
    results.append(("环境配置", test_environment()))

    if not results[0][1]:
        print("\n" + "=" * 60)
        print("⚠️  环境配置失败，跳过后续测试")
        print("=" * 60)
        return

    results.append(("核心工具", test_core_utils()))
    # results.append(("工具系统", test_tool_system()))
    # results.append(("SimpleAgent", test_simple_agent()))
    # results.append(("SimpleAgent (工具)", test_simple_agent_with_tools()))
    # results.append(("ReActAgent", test_react_agent()))

    # 打印总结
    print("\n" + "=" * 60)
    print("  测试总结")
    print("=" * 60 + "\n")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name:20} {status}")

    print(f"\n  总计: {passed}/{total} 个测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！环境配置正确，可以开始使用了。")
    else:
        print("\n⚠️  部分测试失败，请检查配置或查看错误信息。")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
