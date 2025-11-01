#!/usr/bin/env python3
"""
GLM-4.6 工具调用 Agent 示例（LangChain v1.0 语法）。

演示如何使用 GLM-4.6 与 LangChain v1.0 的 create_agent
构建一个可以调用工具的对话式 Agent。

主要变化（v1.0）：
- 使用新的 create_agent API（基于 langgraph）
- AgentExecutor 已被移除
- 返回 CompiledStateGraph 对象
- 使用内置的工具调用机制
- 支持 GLM-4.6 智谱AI模型
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
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
    """获取指定城市的天气预报（示例函数）。

    Args:
        city: 城市名称，例如 '北京'、'上海'、'深圳'
    """
    import random
    conditions = ["晴天", "多云", "小雨", "阴天", "雾霾"]
    temp = random.randint(10, 30)
    condition = random.choice(conditions)
    return f"{city}今天天气：{condition}，温度 {temp}°C"


@tool
def calculate(expression: str) -> str:
    """执行数学计算。

    Args:
        expression: 数学表达式，例如 '2 + 3 * 4'，'100 / (5 + 5)'
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
        timezone: 时区，默认为亚洲/上海（Asia/Shanghai）
                  可选值：Asia/Shanghai, Asia/Tokyo, America/New_York, Europe/London
    """
    from datetime import datetime
    try:
        import pytz as _pytz
        tz = _pytz.timezone(timezone)
        time_str = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
        return f"当前时间（{timezone}）：{time_str}"
    except ImportError:
        # 如果没有 pytz，使用本地时间
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"当前时间（本地）：{time_str}"


@tool
def get_news(topic: str = "科技") -> str:
    """获取指定主题的模拟新闻（示例函数）。

    Args:
        topic: 新闻主题，例如 '科技'、'体育'、'娱乐'、'财经'
    """
    news_items = {
        "科技": [
            "人工智能技术在医疗领域取得重大突破",
            "5G网络建设加速，6G研发已启动",
            "量子计算原型机发布，计算能力提升千倍"
        ],
        "体育": [
            "北京冬奥会筹备工作进展顺利",
            "中国足球青训体系改革启动",
            "马拉松赛事报名人数创历史新高"
        ],
        "娱乐": [
            "国产电影票房创新高",
            "虚拟偶像技术日趋成熟",
            "流媒体平台内容竞争激烈"
        ],
        "财经": [
            "新能源汽车销量持续增长",
            "数字货币试点范围扩大",
            "跨境电商政策利好频出"
        ]
    }

    import random
    topic_news = news_items.get(topic, news_items["科技"])
    selected_news = random.choice(topic_news)
    return f"【{topic}新闻】{selected_news}"


@tool
def translate_text(text: str, target_language: str = "英文") -> str:
    """翻译文本到指定语言（模拟翻译功能）。

    Args:
        text: 要翻译的文本
        target_language: 目标语言，例如 '英文'、'日文'、'韩文'
    """
    # 这是一个模拟翻译函数，实际应用中需要调用真实的翻译API
    translations = {
        ("中文", "英文"): {
            "你好": "Hello",
            "谢谢": "Thank you",
            "再见": "Goodbye",
            "我爱你": "I love you"
        },
        ("中文", "日文"): {
            "你好": "こんにちは",
            "谢谢": "ありがとうございます",
            "再见": "さようなら"
        }
    }

    key = ("中文", target_language)
    if key in translations and text in translations[key]:
        return f"翻译结果（中文 → {target_language}）：{translations[key][text]}"
    else:
        return f"翻译结果（中文 → {target_language}）：【模拟翻译】{text}（实际应用中需要调用真实翻译API）"


def create_glm_agent(
    model: str = "glm-4.6",
    tools: Sequence[BaseTool] | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.7,
) -> Any:
    """
    创建一个 GLM Agent（v1.0 版本）。

    Args:
        model: GLM 模型名称，默认为 "glm-4.6"
        tools: 工具列表
        system_prompt: 系统提示词
        temperature: 温度参数（0.0-1.0，越低越精确）

    Returns:
        CompiledStateGraph 对象，可以直接调用
    """
    api_key = _require_env_var("ZHIPUAI_API_KEY")

    # 如果没有提供工具，使用默认工具
    if tools is None:
        tools = [get_weather, calculate, get_time, get_news]

    normalized_tools = _normalize_tools(tools)

    # 创建 GLM 模型
    llm = ChatZhipuAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )

    # 系统提示词
    if system_prompt is None:
        system_prompt = """你是一个强大的AI助手，名为GLM，具有调用各种工具的能力。

可用工具：
{tools}

工具使用指南：
1. 当用户问天气相关问题时，使用 get_weather 工具
2. 当需要进行数学计算时，使用 calculate 工具
3. 当需要查询时间时，使用 get_time 工具
4. 当需要获取新闻时，使用 get_news 工具
5. 当需要翻译文本时，使用 translate_text 工具

请遵循以下原则：
- 始终保持友好、专业和准确的回答
- 当需要使用工具时，明确说明你要调用哪个工具
- 基于工具返回的结果给出最终答案
- 如果工具调用失败，请尝试其他方法帮助用户
- 对于复杂问题，可以组合使用多个工具

记住：你是一个智能助手，工具是你的超能力！"""

    # 使用新的 create_agent API
    agent = create_agent(
        model=llm,
        tools=normalized_tools,
        system_prompt=system_prompt,
        debug=False,  # 启用调试模式以便查看执行过程
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
    print("🧑‍💼 GLM Agent 基础演示（v1.0）")
    print("=" * 70)

    agent = create_glm_agent(
        model="glm-4.6",
        temperature=0.3,
    )

    print("\n💬 测试对话：")
    messages = [
        {"role": "user", "content": "你好！你是谁？有什么能力？"}
    ]

    result = agent.invoke({"messages": _as_messages(messages)})

    print(f"\n🤖 GLM 回答：\n{result['messages'][-1].content}\n")


def run_weather_demo() -> None:
    """运行天气查询演示：使用 get_weather 工具"""
    print("=" * 70)
    print("🌤️ 天气查询演示")
    print("=" * 70)

    agent = create_glm_agent(
        model="glm-4.6",
        tools=[get_weather],
        temperature=0.3,
    )

    messages = [
        {
            "role": "user",
            "content": "请帮我查一下厦门的天气怎么样？"
        }
    ]

    print("\n💬 用户：查询厦门天气")
    print("\n🔍 Agent 正在调用工具...")
    print()

    result = agent.invoke({"messages": _as_messages(messages)})

    print(f"\n🤖 GLM 回答：\n{result['messages'][-1].content}\n")


def run_calculator_demo() -> None:
    """运行计算演示：使用 calculate 工具"""
    print("=" * 70)
    print("🧮 数学计算演示")
    print("=" * 70)

    agent = create_glm_agent(
        model="glm-4.6",
        tools=[calculate],
        temperature=0.1,
    )

    test_expressions = [
        "请计算 15 * 23 + 7",
        "计算 (100 + 50) / 3",
        "15的平方是多少？"
    ]

    for expr in test_expressions:
        print(f"\n💬 用户：{expr}")
        messages = [{"role": "user", "content": expr}]

        print("\n🔍 Agent 正在计算...")
        result = agent.invoke({"messages": _as_messages(messages)})

        print(f"\n🤖 GLM 回答：\n{result['messages'][-1].content}\n")


def run_time_demo() -> None:
    """运行时间查询演示：使用 get_time 工具"""
    print("=" * 70)
    print("🕐 时间查询演示")
    print("=" * 70)

    agent = create_glm_agent(
        model="glm-4.6",
        tools=[get_time],
        temperature=0.3,
    )

    messages = [
        {"role": "user", "content": "现在是什么时间？"}
    ]

    print("\n💬 用户：查询当前时间")
    print("\n🔍 Agent 正在获取时间...")
    print()

    result = agent.invoke({"messages": _as_messages(messages)})

    print(f"\n🤖 GLM 回答：\n{result['messages'][-1].content}\n")


def run_news_demo() -> None:
    """运行新闻查询演示：使用 get_news 工具"""
    print("=" * 70)
    print("📰 新闻查询演示")
    print("=" * 70)

    agent = create_glm_agent(
        model="glm-4.6",
        tools=[get_news],
        temperature=0.5,
    )

    messages = [
        {"role": "user", "content": "请给我讲一条科技新闻"}
    ]

    print("\n💬 用户：获取科技新闻")
    print("\n🔍 Agent 正在获取新闻...")
    print()

    result = agent.invoke({"messages": _as_messages(messages)})

    print(f"\n🤖 GLM 回答：\n{result['messages'][-1].content}\n")


def run_translate_demo() -> None:
    """运行翻译演示：使用 translate_text 工具"""
    print("=" * 70)
    print("🌐 翻译演示")
    print("=" * 70)

    agent = create_glm_agent(
        model="glm-4.6",
        tools=[translate_text],
        temperature=0.3,
    )

    messages = [
        {"role": "user", "content": "请把'你好'翻译成英文"}
    ]

    print("\n💬 用户：翻译'你好'到英文")
    print("\n🔍 Agent 正在翻译...")
    print()

    result = agent.invoke({"messages": _as_messages(messages)})

    print(f"\n🤖 GLM 回答：\n{result['messages'][-1].content}\n")


def run_multi_tool_demo() -> None:
    """运行多工具演示：同时使用多个工具"""
    print("=" * 70)
    print("🛠️ 多工具组合演示")
    print("=" * 70)

    agent = create_glm_agent(
        model="glm-4.6",
        tools=[get_weather, calculate, get_time],
        temperature=0.3,
    )

    messages = [
        {
            "role": "user",
            "content": "请帮我查一下上海的天气，然后计算 123 * 456，最后告诉我当前时间。"
        }
    ]

    print("\n💬 用户：多工具组合请求")
    print("\n🔍 Agent 正在处理复杂请求...")
    print()

    result = agent.invoke({"messages": _as_messages(messages)})

    print(f"\n🤖 GLM 完整回答：\n{result['messages'][-1].content}\n")


def run_conversation_demo() -> None:
    """运行对话演示：多轮对话"""
    print("=" * 70)
    print("💭 多轮对话演示")
    print("=" * 70)

    agent = create_glm_agent(
        model="glm-4.6",
        tools=[get_weather, calculate, get_news],
        temperature=0.5,
    )

    conversation = [
        {"role": "user", "content": "我想了解一下科技新闻"},
        {"role": "user", "content": "能帮我计算一下 50 * 80 吗？"},
        {"role": "user", "content": "谢谢，我想问一个数学问题，20的平方是多少？"},
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
        print(f"\n🤖 GLM 回答：\n{response.content}\n")


def compare_models():
    """对比不同模型的特性"""
    print("\n" + "=" * 70)
    print("📊 GLM-4.6 vs Anthropic Claude 特性对比")
    print("=" * 70)

    print("""
🟢 GLM-4.6 (智谱AI):
   ✅ 支持中文优化
   ✅ 适合中文对话
   ✅ 工具调用能力强
   ✅ 成本相对较低
   ✅ 国内访问速度更快
   ✅ 生态集成度高

🔵 Anthropic Claude:
   ✅ 英文对话优秀
   ✅ 长文本处理能力强
   ✅ 安全性高
   ✅ 逻辑推理能力强
   ✅ 国际化程度高

📋 共同特性（v1.0）：
   ✅ 基于 langgraph 架构
   ✅ 统一的 create_agent API
   ✅ 内置状态管理
   ✅ 工具调用框架统一
   ✅ 消息格式兼容

⚠️  重要提示：
   - 两个模型都使用相同的 LangChain v1.0 API
   - 主要区别在于模型参数和API配置
   - 建议根据使用场景选择合适模型
    """)


def main() -> None:
    """主函数：运行所有演示"""
    print("🚀 GLM-4.6 + LangChain v1.0 Agent 示例")
    print("=" * 80)

    # 检查 API 密钥
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key or api_key.startswith("your-"):
        print("❌ 错误：请在 .env 文件中设置您的 ZHIPUAI_API_KEY")
        print("📝 获取 API 密钥：https://open.bigmodel.cn/")
        return

    print("""
✨ GLM-4.6 + LangChain v1.0 Agent 新特性：
1. 基于 langgraph 的新架构
2. 简化的 create_agent API
3. 内置状态管理
4. 更灵活的工具调用
5. 中文优化，适合国内用户
    """)

    try:
        # 运行各种演示
        # run_basic_demo()
        # compare_models()
        # run_weather_demo()
        # run_calculator_demo()
        run_time_demo()
        # run_news_demo()
        # run_translate_demo()
        # run_multi_tool_demo()
        # run_conversation_demo()

        print("\n" + "=" * 70)
        print("🎉 所有演示运行完成！")
        print("=" * 70)
        print("\n📚 更多信息请参考：")
        print("- LangChain v1.0 Agent 文档: https://python.langchain.com/docs/concepts/agents/")
        print("- LangGraph 文档: https://langchain-ai.github.io/langgraph/")
        print("- 智谱AI开放平台: https://open.bigmodel.cn/")
        print("- GLM-4.6 API 文档: https://open.bigmodel.cn/dev/api")

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序。")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
