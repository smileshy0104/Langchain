"""
LangChain Agents 基础示例
演示基本的 Agent 创建、工具定义和系统提示
使用 GLM 模型
"""

from langchain.agents import create_agent
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import os

# 设置 API Key
os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 1. 基础工具定义 ====================

@tool
def get_weather(location: str) -> str:
    """获取指定位置的天气信息

    Args:
        location: 位置名称,如"北京"、"上海"

    Returns:
        天气信息字符串
    """
    # 模拟天气数据
    weather_data = {
        "北京": "晴朗,温度 20°C,湿度 45%",
        "上海": "多云,温度 22°C,湿度 65%",
        "广州": "小雨,温度 25°C,湿度 80%",
    }
    return weather_data.get(location, f"{location} 的天气是晴朗的,温度 18°C")


@tool
def calculate(expression: str) -> float:
    """
    计算数学表达式

    Args:
        expression: 要计算的数学表达式,如 "2 + 2"

    Returns:
        计算结果

    Examples:
        >>> calculate("10 * 5")
        50.0
    """
    try:
        result = eval(expression)
        return float(result)
    except Exception as e:
        return f"计算错误: {str(e)}"


# ==================== 2. 带参数验证的工具 ====================

class SearchInput(BaseModel):
    """搜索输入参数"""
    query: str = Field(description="搜索查询词")
    limit: int = Field(default=10, ge=1, le=50, description="返回结果数量")


@tool(args_schema=SearchInput)
def search_web(query: str, limit: int = 10) -> str:
    """
    在网络上搜索信息

    Args:
        query: 搜索关键词
        limit: 最多返回多少条结果

    Returns:
        搜索结果摘要
    """
    # 模拟搜索结果
    results = [
        f"结果 {i+1}: 关于 '{query}' 的内容..."
        for i in range(min(limit, 3))
    ]
    return f"找到 {limit} 条关于 '{query}' 的结果:\n" + "\n".join(results)


# ==================== 3. 创建基础 Agent ====================

def basic_agent_example():
    """基础 Agent 示例"""
    print("=" * 50)
    print("基础 Agent 示例")
    print("=" * 50)

    # 创建 GLM 模型
    model = ChatZhipuAI(
        model="glm-4-plus",
        temperature=0.7,
    )

    # 创建 Agent
    agent = create_agent(
        model=model,
        tools=[get_weather, calculate],
        system_prompt="你是一个有帮助的 AI 助手,可以查询天气和进行计算"
    )

    # 执行查询
    result = agent.invoke({
        "messages": [{"role": "user", "content": "北京天气如何？"}]
    })

    print("\n问题: 北京天气如何？")
    print(f"回答: {result['messages'][-1].content}")


# ==================== 4. 结构化系统提示 ====================

STRUCTURED_SYSTEM_PROMPT = """
# 角色定义
你是一个专业的信息查询助手。

# 核心能力
- 天气查询
- 数学计算
- 网络搜索

# 工作流程
1. 理解用户需求
2. 选择合适的工具
3. 执行工具调用
4. 返回清晰的结果

# 输出格式
- 使用简洁的语言
- 数据准确无误
- 结论清晰明了

# 限制
- 不要编造数据
- 不确定时明确说明
- 超出能力范围时说明无法完成
"""


def structured_prompt_agent():
    """使用结构化提示词的 Agent"""
    print("\n" + "=" * 50)
    print("结构化提示词 Agent 示例")
    print("=" * 50)

    model = ChatZhipuAI(
        model="glm-4-plus",
        temperature=0.5,
    )

    agent = create_agent(
        model=model,
        tools=[get_weather, calculate, search_web],
        system_prompt=STRUCTURED_SYSTEM_PROMPT
    )

    # 测试多个查询
    queries = [
        "上海的天气怎么样？",
        "帮我计算 123 * 456",
        "搜索 Python 教程,返回 5 条结果"
    ]

    for query in queries:
        result = agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })
        print(f"\n问题: {query}")
        print(f"回答: {result['messages'][-1].content}")


# ==================== 5. 多工具组合使用 ====================

def multi_tool_agent():
    """多工具组合使用示例"""
    print("\n" + "=" * 50)
    print("多工具组合使用示例")
    print("=" * 50)

    model = ChatZhipuAI(
        model="glm-4-plus",
        temperature=0.7,
    )

    agent = create_agent(
        model=model,
        tools=[get_weather, calculate, search_web],
        system_prompt="你是一个智能助手,可以查天气、做计算、搜索信息"
    )

    # 复杂查询,需要使用多个工具
    query = "北京今天的天气如何？如果温度是20度,转换成华氏度是多少？"

    result = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })

    print(f"\n问题: {query}")
    print(f"回答: {result['messages'][-1].content}")

    # 打印执行过程中的所有消息
    print("\n执行过程:")
    for i, msg in enumerate(result['messages']):
        print(f"\n步骤 {i+1}: {msg.__class__.__name__}")
        if hasattr(msg, 'content') and msg.content:
            print(f"内容: {msg.content[:200]}...")


if __name__ == "__main__":
    # 运行所有示例
    try:
        basic_agent_example()
        structured_prompt_agent()
        multi_tool_agent()
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("\n请确保已设置 ZHIPUAI_API_KEY 环境变量")
