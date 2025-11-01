#!/usr/bin/env python3
"""
Anthropic 工具调用 Agent 示例（LangChain v1.0 语法）。

演示如何使用 Anthropic Claude 与 LangChain v1.0 的 create_agent
构建一个可以调用工具的对话式 Agent。

主要变化（v1.0）：
- 使用新的 create_agent API（基于 langgraph）
- AgentExecutor 已被移除
- 返回 CompiledStateGraph 对象
- 使用内置的工具调用机制
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, Tool, tool
from langchain.agents import create_agent

# 从项目根目录加载 .env
dotenv.load_dotenv(dotenv_path="../.env")


def _require_env_var(name: str) -> str:
    """确保必需的环境变量存在。"""
    value = os.getenv(name)
    if not value or value.startswith("your-"):
        raise EnvironmentError(
            f"未检测到有效的 {name}，请在项目根目录的 .env 中配置后重试。"
        )
    return value


def _normalize_tools(
    tools: Sequence[BaseTool],
) -> list[BaseTool]:
    """确保工具是 BaseTool 实例列表。"""
    normalized: list[BaseTool] = []
    for item in tools:
        if isinstance(item, BaseTool):
            normalized.append(item)
        else:
            raise TypeError(f"工具必须是 BaseTool 实例，得到: {type(item)}")
    return normalized


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气预报（示例函数）。"""
    import random
    conditions = ["晴天", "多云", "小雨", "阴天"]
    temp = random.randint(10, 30)
    condition = random.choice(conditions)
    return f"{city}今天天气：{condition}，温度 {temp}°C"


@tool
def calculate(expression: str) -> str:
    """执行数学计算。

    Args:
        expression: 数学表达式，例如 '2 + 3 * 4'
    """
    try:
        # 安全的方式计算（仅用于演示）
        allowed_chars = set('0123456789+-*/.() ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"计算结果: {expression} = {result}"
        else:
            return "错误：表达式包含无效字符"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def get_time(timezone: str = "Asia/Shanghai") -> str:
    """获取当前时间。

    Args:
        timezone: 时区，默认为亚洲/上海
    """
    from datetime import datetime
    return f"当前时间（{timezone}）：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


def create_anthropic_agent(
    model: str = "claude-3-5-sonnet-20241022",
    tools: Sequence[BaseTool] | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.2,
) -> Any:
    """
    创建一个 Anthropic Agent（v1.0 版本）。

    Args:
        model: Anthropic 模型名称
        tools: 工具列表
        system_prompt: 系统提示词
        temperature: 温度参数

    Returns:
        CompiledStateGraph 对象，可以直接调用
    """
    api_key = _require_env_var("ANTHROPIC_API_KEY")

    # 如果没有提供工具，使用默认工具
    if tools is None:
        tools = [get_weather, calculate, get_time]

    normalized_tools = _normalize_tools(tools)

    # 创建模型
    llm = ChatAnthropic(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )

    # 系统提示词
    if system_prompt is None:
        system_prompt = """你是一个有用的AI助手，能够调用工具来回答用户的问题。

可用工具：
{tools}

当需要使用工具时：
1. 分析用户的问题
2. 选择合适的工具
3. 调用工具获取信息
4. 基于工具返回结果回答用户

请始终保持友好、专业和准确的回答。"""

    # 使用新的 create_agent API
    agent = create_agent(
        model=llm,
        tools=normalized_tools,
        system_prompt=system_prompt,
        debug=True,  # 启用调试模式以便查看执行过程
    )

    return agent


def _as_messages(payloads: list[dict[str, Any]]) -> list[BaseMessage]:
    """将 {role, content} 形式的消息转换为 LangChain 所需的消息对象。"""
    role_map = {
        "system": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }
    messages: list[BaseMessage] = []
    for payload in payloads:
        role = payload.get("role")
        content = payload.get("content", "")
        if role not in role_map:
            raise ValueError(f"不支持的消息角色: {role!r}")
        messages.append(role_map[role](content=content))
    return messages


def run_basic_demo() -> None:
    """运行基础演示：简单问答"""
    print("=" * 70)
    print("🧑‍💼 Anthropic Agent 基础演示（v1.0）")
    print("=" * 70)

    agent = create_anthropic_agent(
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        temperature=0.0,
    )

    print("\n💬 测试对话：")
    messages = [
        {"role": "user", "content": "你好！你能做什么？"}
    ]

    result = agent.invoke({"messages": _as_messages(messages)})

    print(f"\n🤖 Agent 回答：\n{result['messages'][-1].content}\n")


def run_weather_demo() -> None:
    """运行天气查询演示：使用 get_weather 工具"""
    print("=" * 70)
    print("🌤️ 天气查询演示")
    print("=" * 70)

    agent = create_anthropic_agent(
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        tools=[get_weather],
        temperature=0.0,
    )

    messages = [
        {
            "role": "user",
            "content": "请帮我查一下旧金山的天气怎么样？"
        }
    ]

    print("\n💬 用户：查询旧金山天气")
    print("\n🔍 Agent 正在调用工具...")
    print()

    result = agent.invoke({"messages": _as_messages(messages)})

    print(f"\n🤖 Agent 回答：\n{result['messages'][-1].content}\n")


def run_calculator_demo() -> None:
    """运行计算演示：使用 calculate 工具"""
    print("=" * 70)
    print("🧮 数学计算演示")
    print("=" * 70)

    agent = create_anthropic_agent(
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        tools=[calculate],
        temperature=0.0,
    )

    test_expressions = [
        "请计算 15 * 23 + 7",
        "计算 (100 + 50) / 3",
        "15的平方是多少？"
    ]

    for expr in test_expressions:
        print(f"\n💬 用户：{expr}")
        messages = [{"role": "user", "content": expr}]

        print("\n🔍 Agent 正在思考...")
        result = agent.invoke({"messages": _as_messages(messages)})

        print(f"\n🤖 Agent 回答：\n{result['messages'][-1].content}\n")


def run_time_demo() -> None:
    """运行时间查询演示：使用 get_time 工具"""
    print("=" * 70)
    print("🕐 时间查询演示")
    print("=" * 70)

    agent = create_anthropic_agent(
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        tools=[get_time],
        temperature=0.0,
    )

    messages = [
        {"role": "user", "content": "现在是什么时间？"}
    ]

    print("\n💬 用户：查询当前时间")
    print("\n🔍 Agent 正在获取时间...")
    print()

    result = agent.invoke({"messages": _as_messages(messages)})

    print(f"\n🤖 Agent 回答：\n{result['messages'][-1].content}\n")


def run_multi_tool_demo() -> None:
    """运行多工具演示：同时使用多个工具"""
    print("=" * 70)
    print("🛠️ 多工具组合演示")
    print("=" * 70)

    agent = create_anthropic_agent(
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        tools=[get_weather, calculate, get_time],
        temperature=0.0,
    )

    messages = [
        {
            "role": "user",
            "content": "请帮我查一下北京的天气，然后计算 123 * 456，最后告诉我当前时间。"
        }
    ]

    print("\n💬 用户：多工具组合请求")
    print("\n🔍 Agent 正在处理复杂请求...")
    print()

    result = agent.invoke({"messages": _as_messages(messages)})

    print(f"\n🤖 Agent 完整回答：\n{result['messages'][-1].content}\n")


def run_conversation_demo() -> None:
    """运行对话演示：多轮对话"""
    print("=" * 70)
    print("💭 多轮对话演示")
    print("=" * 70)

    agent = create_anthropic_agent(
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        tools=[get_weather, calculate],
        temperature=0.0,
    )

    conversation = [
        {"role": "user", "content": "我想知道上海的天气"},
        {"role": "user", "content": "上海和北京哪个更热？"},
        {"role": "user", "content": "请计算一下 15 + 27 的结果"},
    ]

    messages: list[BaseMessage] = []
    for i, msg in enumerate(conversation, 1):
        print(f"\n💬 轮次 {i} - 用户：{msg['content']}")
        messages.append(HumanMessage(content=msg["content"]))

        print("\n🔍 Agent 正在思考...")
        result = agent.invoke({"messages": messages})

        # 更新消息历史
        messages = result["messages"]
        response = messages[-1]
        print(f"\n🤖 Agent 回答：\n{response.content}\n")


def compare_v1_v2():
    """对比 v0.3 和 v1.0 的差异"""
    print("\n" + "=" * 70)
    print("📊 LangChain Agent v0.3 vs v1.0 对比")
    print("=" * 70)

    print("""
🔴 v0.3.x（已弃用）：
   from langchain.agents import AgentExecutor, create_tool_calling_agent

   agent_runnable = create_tool_calling_agent(llm, tools, prompt)
   executor = AgentExecutor(agent=agent_runnable, tools=tools)
   result = executor.invoke({"messages": [...]})

🟢 v1.0.x（当前）：
   from langchain.agents import create_agent
   from langchain_anthropic import ChatAnthropic

   agent = create_agent(model=llm, tools=tools, system_prompt=...)
   result = agent.invoke({"messages": [...]})

主要变化：
✅ 不再需要 AgentExecutor
✅ 不再使用 create_tool_calling_agent
✅ 使用新的 create_agent API
✅ 基于 langgraph 的状态管理
✅ 更简洁的 API 设计

⚠️  重要提示：
   - v1.0 的 create_agent 返回 CompiledStateGraph（基于 langgraph）
   - AgentExecutor 在 v1.0 中已被移除
   - 工具调用机制已重构
    """)


def main() -> None:
    """主函数：运行所有演示"""
    print("🚀 Anthropic Claude + LangChain v1.0 Agent 示例")
    print("=" * 80)

    # 检查 API 密钥
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key.startswith("your-"):
        print("❌ 错误：请在 .env 文件中设置您的 ANTHROPIC_API_KEY")
        print("📝 获取 API 密钥：https://console.anthropic.com/")
        return

    print("""
✨ LangChain v1.0 Agent 新特性：
1. 基于 langgraph 的新架构
2. 简化的 create_agent API
3. 内置状态管理
4. 更灵活的工具调用
    """)

    try:
        # 运行各种演示
        run_basic_demo()
        compare_v1_v2()
        run_weather_demo()
        run_calculator_demo()
        run_time_demo()
        run_multi_tool_demo()
        run_conversation_demo()

        print("\n" + "=" * 70)
        print("🎉 所有演示运行完成！")
        print("=" * 70)
        print("\n📚 更多信息请参考：")
        print("- LangChain v1.0 Agent 文档: https://python.langchain.com/docs/concepts/agents/")
        print("- LangGraph 文档: https://langchain-ai.github.io/langgraph/")
        print("- Anthropic 集成: https://python.langchain.com/docs/integrations/chat/anthropic/")

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序。")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
