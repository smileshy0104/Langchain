"""
示例2：Agent 中使用 Structured Output
演示如何在 LangChain Agent 中使用结构化输出
"""

import os
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from langchain_community.chat_models import ChatZhipuAI
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver

os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 示例 2.1: 基础 Agent 结构化输出 ====================

class Weather(BaseModel):
    """天气信息"""
    temperature: float = Field(description="温度（摄氏度）")
    condition: str = Field(description="天气状况（晴/阴/雨/雪等）")
    humidity: int = Field(description="湿度（百分比）", ge=0, le=100)
    wind_speed: float = Field(description="风速（公里/小时）")


@tool
def get_weather_tool(city: str) -> str:
    """
    获取指定城市的天气信息

    Args:
        city: 城市名称
    """
    # 模拟天气数据
    weather_db = {
        "北京": "北京今天晴天，温度25度，湿度45%，风速15公里/小时",
        "上海": "上海今天多云，温度28度，湿度60%，风速10公里/小时",
        "广州": "广州今天雨天，温度30度，湿度80%，风速20公里/小时",
    }
    return weather_db.get(city, f"{city}的天气信息暂无")


def example_01_basic_agent():
    """示例 2.1: 基础 Agent 结构化输出"""
    print("\n" + "=" * 60)
    print("示例 2.1: 基础 Agent 结构化输出")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)

    # 使用 ToolStrategy（适用于所有支持工具调用的模型）
    agent = create_agent(
        model=model,
        tools=[get_weather_tool],
        response_format=ToolStrategy(Weather)  # 使用 ToolStrategy
    )

    print("\n👤 用户: 北京的天气怎么样？")
    result = agent.invoke({
        "messages": [{"role": "user", "content": "北京的天气怎么样？"}]
    })

    # 结构化响应在 'structured_response' 键中
    weather = result["structured_response"]

    print(f"\n🤖 结构化天气信息:")
    print(f"   类型: {type(weather)}")
    print(f"   温度: {weather.temperature}°C")
    print(f"   状况: {weather.condition}")
    print(f"   湿度: {weather.humidity}%")
    print(f"   风速: {weather.wind_speed} km/h")

    print(f"\n📊 完整消息历史:")
    for i, msg in enumerate(result['messages'], 1):
        print(f"   {i}. [{msg.type}] {str(msg.content)[:80]}...")


# ==================== 示例 2.2: 复杂查询 ====================

class ResearchResult(BaseModel):
    """研究结果"""
    topic: str = Field(description="研究主题")
    summary: str = Field(description="研究摘要")
    key_findings: List[str] = Field(description="关键发现列表")
    sources: List[str] = Field(description="信息来源")
    confidence: float = Field(description="结果可信度（0-1）", ge=0, le=1)


@tool
def search_tool(query: str) -> str:
    """
    搜索工具（模拟）

    Args:
        query: 搜索查询
    """
    # 模拟搜索结果
    return f"""
    关于 {query} 的搜索结果:
    1. Python 是一种高级编程语言，以简洁易读著称
    2. Python 广泛用于数据科学、机器学习、Web开发等领域
    3. Python 拥有丰富的第三方库生态系统
    来源: Python 官方文档, Stack Overflow, GitHub
    """


def example_02_complex_query():
    """示例 2.2: 复杂查询与结构化输出"""
    print("\n" + "=" * 60)
    print("示例 2.2: 复杂查询与结构化输出")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)

    agent = create_agent(
        model=model,
        tools=[search_tool],
        response_format=ToolStrategy(ResearchResult)
    )

    print("\n👤 用户: 研究一下 Python 编程语言")
    result = agent.invoke({
        "messages": [{"role": "user", "content": "帮我研究一下 Python 编程语言"}]
    })

    research = result["structured_response"]

    print(f"\n🤖 研究结果:")
    print(f"   主题: {research.topic}")
    print(f"   摘要: {research.summary}")
    print(f"\n🔍 关键发现:")
    for i, finding in enumerate(research.key_findings, 1):
        print(f"   {i}. {finding}")
    print(f"\n📚 信息来源:")
    for i, source in enumerate(research.sources, 1):
        print(f"   {i}. {source}")
    print(f"\n💯 可信度: {research.confidence * 100:.0f}%")


# ==================== 示例 2.3: 多工具协作 ====================

class TaskAnalysis(BaseModel):
    """任务分析结果"""
    task_description: str = Field(description="任务描述")
    steps: List[str] = Field(description="执行步骤")
    tools_used: List[str] = Field(description="使用的工具列表")
    estimated_time: str = Field(description="预估时间")
    status: str = Field(description="执行状态")


@tool
def calculate_tool(expression: str) -> str:
    """
    计算数学表达式

    Args:
        expression: 数学表达式
    """
    try:
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def file_info_tool(filename: str) -> str:
    """
    获取文件信息（模拟）

    Args:
        filename: 文件名
    """
    return f"文件 {filename} 的信息: 大小 1.5MB, 创建时间 2024-01-01"


def example_03_multi_tool():
    """示例 2.3: 多工具协作"""
    print("\n" + "=" * 60)
    print("示例 2.3: 多工具协作")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)

    agent = create_agent(
        model=model,
        tools=[calculate_tool, file_info_tool, search_tool],
        response_format=ToolStrategy(TaskAnalysis)
    )

    print("\n👤 用户: 计算 123 * 456，然后查找 data.csv 的信息")
    result = agent.invoke({
        "messages": [{"role": "user", "content": "帮我计算 123 * 456，然后查找 data.csv 的信息"}]
    })

    analysis = result["structured_response"]

    print(f"\n🤖 任务分析:")
    print(f"   描述: {analysis.task_description}")
    print(f"\n📋 执行步骤:")
    for i, step in enumerate(analysis.steps, 1):
        print(f"   {i}. {step}")
    print(f"\n🔧 使用的工具:")
    for tool in analysis.tools_used:
        print(f"   - {tool}")
    print(f"\n⏱️  预估时间: {analysis.estimated_time}")
    print(f"\n✅ 状态: {analysis.status}")


# ==================== 示例 2.4: 带记忆的 Agent ====================

class ConversationSummary(BaseModel):
    """对话摘要"""
    topics_discussed: List[str] = Field(description="讨论的主题列表")
    key_points: List[str] = Field(description="关键要点")
    user_intent: str = Field(description="用户意图")
    next_steps: Optional[List[str]] = Field(None, description="建议的下一步")


def example_04_agent_with_memory():
    """示例 2.4: 带记忆的 Agent"""
    print("\n" + "=" * 60)
    print("示例 2.4: 带记忆的 Agent")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[search_tool],
        checkpointer=checkpointer,
        response_format=ToolStrategy(ConversationSummary)
    )

    config = {"configurable": {"thread_id": "conversation-1"}}

    # 第一轮对话
    print("\n👤 用户: 我想学习机器学习")
    result1 = agent.invoke(
        {"messages": [{"role": "user", "content": "我想学习机器学习"}]},
        config
    )

    summary1 = result1["structured_response"]
    print(f"\n🤖 第一轮摘要:")
    print(f"   主题: {', '.join(summary1.topics_discussed)}")
    print(f"   意图: {summary1.user_intent}")

    # 第二轮对话
    print("\n\n👤 用户: 从哪里开始比较好？")
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "从哪里开始比较好？"}]},
        config
    )

    summary2 = result2["structured_response"]
    print(f"\n🤖 第二轮摘要:")
    print(f"   主题: {', '.join(summary2.topics_discussed)}")
    print(f"   关键要点:")
    for point in summary2.key_points:
        print(f"   - {point}")
    if summary2.next_steps:
        print(f"   建议:")
        for step in summary2.next_steps:
            print(f"   - {step}")


# ==================== 示例 2.5: 错误处理 ====================

def example_05_error_handling():
    """示例 2.5: 错误处理"""
    print("\n" + "=" * 60)
    print("示例 2.5: 错误处理")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)

    # 测试不同的错误处理策略
    strategies = [
        ("raise", "抛出异常"),
        ("return_none", "返回 None"),
    ]

    for handle_errors, description in strategies:
        print(f"\n--- 策略: {description} ---")

        agent = create_agent(
            model=model,
            tools=[get_weather_tool],
            response_format=ToolStrategy(
                Weather,
                handle_errors=handle_errors  # 设置错误处理策略
            )
        )

        try:
            result = agent.invoke({
                "messages": [{"role": "user", "content": "北京天气"}]
            })

            response = result.get("structured_response")
            if response is None:
                print(f"   ⚠️  返回值为 None（模型可能没有返回结构化数据）")
            else:
                print(f"   ✅ 成功获取结构化响应")
                print(f"   温度: {response.temperature}°C")

        except Exception as e:
            print(f"   ❌ 捕获异常: {type(e).__name__}")
            print(f"   错误信息: {str(e)[:100]}...")


# ==================== 主函数 ====================

def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("LangChain Structured Output - Agent 用法")
    print("=" * 60)

    examples = [
        ("基础 Agent 结构化输出", example_01_basic_agent),
        ("复杂查询", example_02_complex_query),
        ("多工具协作", example_03_multi_tool),
        ("带记忆的 Agent", example_04_agent_with_memory),
        ("错误处理", example_05_error_handling),
    ]

    for i, (name, func) in enumerate(examples, 1):
        print(f"\n{'='*60}")
        print(f"运行示例 {i}/{len(examples)}: {name}")
        print(f"{'='*60}")
        try:
            func()
        except Exception as e:
            print(f"\n❌ 错误: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已终止")
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
