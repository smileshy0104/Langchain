"""
测试环境配置脚本
检查所有依赖是否正确安装，API Key 是否配置
"""

import sys
import os


def test_imports():
    """测试必要的包是否安装"""
    print("=" * 50)
    print("测试 Python 包导入...")
    print("=" * 50)

    packages = {
        "langchain": "langchain",
        "langchain_core": "langchain-core",
        "langchain_community": "langchain-community",
        "langgraph": "langgraph",
        "zhipuai": "zhipuai",
        "pydantic": "pydantic",
    }

    missing = []
    for module_name, package_name in packages.items():
        try:
            __import__(module_name)
            print(f"✅ {package_name:25} 已安装")
        except ImportError:
            print(f"❌ {package_name:25} 未安装")
            missing.append(package_name)

    if missing:
        print(f"\n缺少以下包，请运行:")
        print(f"pip install {' '.join(missing)}")
        return False

    return True


def test_api_key():
    """测试 API Key 是否配置"""
    print("\n" + "=" * 50)
    print("测试 API Key 配置...")
    print("=" * 50)

    api_key = os.getenv("ZHIPUAI_API_KEY")

    if not api_key:
        print("❌ ZHIPUAI_API_KEY 环境变量未设置")
        print("\n请运行以下命令设置 API Key:")
        print("export ZHIPUAI_API_KEY='your-api-key-here'")
        return False

    if api_key == "your-api-key-here":
        print("⚠️  检测到默认 API Key，请替换为真实的 API Key")
        return False

    print(f"✅ ZHIPUAI_API_KEY 已设置")
    print(f"   Key 前缀: {api_key[:10]}...")
    return True


def test_langchain_agent():
    """测试创建简单的 Agent"""
    print("\n" + "=" * 50)
    print("测试创建 LangChain Agent...")
    print("=" * 50)

    try:
        from langchain.agents import create_agent
        from langchain_community.chat_models import ChatZhipuAI
        from langchain_core.tools import tool

        @tool
        def test_tool(query: str) -> str:
            """测试工具"""
            return f"测试结果: {query}"

        model = ChatZhipuAI(
            model="glm-4-flash",
            temperature=0.5,
        )

        agent = create_agent(
            model=model,
            tools=[test_tool],
            system_prompt="你是一个测试助手"
        )

        print("✅ Agent 创建成功")
        return True

    except Exception as e:
        print(f"❌ Agent 创建失败: {str(e)}")
        return False


def test_simple_invoke():
    """测试简单的 Agent 调用"""
    print("\n" + "=" * 50)
    print("测试 Agent 调用...")
    print("=" * 50)

    try:
        from langchain.agents import create_agent
        from langchain_community.chat_models import ChatZhipuAI
        from langchain_core.tools import tool

        @tool
        def echo(text: str) -> str:
            """回显工具"""
            return f"Echo: {text}"

        model = ChatZhipuAI(
            model="glm-4-flash",
            temperature=0.5,
        )

        agent = create_agent(
            model=model,
            tools=[echo],
            system_prompt="你是一个测试助手，收到消息后直接回复'测试成功'"
        )

        print("正在调用 Agent...")
        result = agent.invoke({
            "messages": [{"role": "user", "content": "测试"}]
        })

        response = result['messages'][-1].content
        print(f"✅ Agent 调用成功")
        print(f"   响应: {response[:100]}...")
        return True

    except Exception as e:
        print(f"❌ Agent 调用失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 50)
    print("LangChain Agents 环境测试")
    print("=" * 50 + "\n")

    results = []

    # 1. 测试包导入
    results.append(("包导入", test_imports()))

    # 2. 测试 API Key
    results.append(("API Key", test_api_key()))

    # 如果基础测试通过，继续测试 Agent
    if all(r[1] for r in results):
        results.append(("Agent 创建", test_langchain_agent()))

        # 如果 Agent 创建成功，测试调用
        if results[-1][1]:
            results.append(("Agent 调用", test_simple_invoke()))

    # 输出总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)

    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name:20} {status}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n🎉 所有测试通过！环境配置正确。")
        print("\n你可以运行以下示例:")
        print("  python 01_basic_agent.py")
        print("  python 02_middleware_examples.py")
        print("  python 03_memory_management.py")
        print("  python 04_structured_output.py")
        print("  python 05_human_in_the_loop.py")
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息。")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
