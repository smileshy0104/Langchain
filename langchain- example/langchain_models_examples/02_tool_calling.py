"""
LangChain Models - 工具调用示例
演示基本工具调用、完整流程、强制调用、并行调用等
使用 GLM 模型
"""

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
import os

os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 工具定义 ====================

@tool
def get_weather(location: str) -> str:
    """获取指定位置的天气信息。

    Args:
        location: 城市名称，例如 '北京' 或 '上海'
    """
    # 模拟天气数据
    weather_map = {
        "北京": "晴朗，温度 22°C，湿度 45%",
        "上海": "多云，温度 25°C，湿度 70%",
        "深圳": "小雨，温度 28°C，湿度 85%"
    }
    return weather_map.get(location, f"{location}的天气: 晴朗，温度 20°C")


@tool
def calculate(expression: str) -> float:
    """计算数学表达式。

    Args:
        expression: 要计算的数学表达式，例如 '2 + 2'
    """
    try:
        return eval(expression)
    except:
        return "计算错误"


@tool
def search_database(query: str) -> str:
    """在数据库中搜索信息。"""
    return f"找到 5 条关于 '{query}' 的记录"


# ==================== 1. 基本工具调用 ====================

def basic_tool_calling():
    """基本工具调用示例"""
    print("=" * 50)
    print("基本工具调用示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6")
    model_with_tools = model.bind_tools([get_weather, calculate])

    # 调用模型
    response = model_with_tools.invoke([
        HumanMessage(content="北京的天气怎么样?")
    ])

    # 检查是否有工具调用
    if response.tool_calls:
        print("\n模型决定调用工具:")
        for tool_call in response.tool_calls:
            print(f"  - 工具: {tool_call['name']}")
            print(f"  - 参数: {tool_call['args']}")
    else:
        print(f"\n直接回答: {response.content}")


# ==================== 2. 完整工具调用流程 ====================

def complete_tool_calling_flow():
    """完整工具调用流程示例"""
    print("\n" + "=" * 50)
    print("完整工具调用流程示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6")
    model_with_tools = model.bind_tools([search_database])

    # 1. 用户问题
    messages = [HumanMessage(content="查找所有关于 Python 的记录")]

    # 2. 模型决定调用工具
    response = model_with_tools.invoke(messages)

    print("\n步骤 1: 用户提问")
    print("问题: 查找所有关于 Python 的记录")

    # 3. 执行工具调用
    if response.tool_calls:
        print("\n步骤 2: 模型决定调用工具")
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"  工具: {tool_name}")
            print(f"  参数: {tool_args}")

            # 执行工具
            tool_result = search_database.invoke(tool_args)

            print(f"\n步骤 3: 工具执行结果")
            print(f"  {tool_result}")

            # 4. 将工具结果添加到消息历史
            messages.append(response)
            messages.append(ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"]
            ))

        # 5. 模型使用工具结果生成最终回答
        final_response = model.invoke(messages)
        print(f"\n步骤 4: 最终回答")
        print(f"  {final_response.content}")


# ==================== 3. 并行工具调用 ====================

def parallel_tool_calling():
    """并行工具调用示例"""
    print("\n" + "=" * 50)
    print("并行工具调用示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6")
    model_with_tools = model.bind_tools([get_weather, calculate])

    # 模型可能同时调用多个工具
    response = model_with_tools.invoke([
        HumanMessage(content="告诉我北京的天气，并计算 100 * 50")
    ])

    print("\n用户问题: 告诉我北京的天气，并计算 100 * 50")

    # 处理多个工具调用
    if response.tool_calls:
        print(f"\n模型同时调用了 {len(response.tool_calls)} 个工具:")
        for i, tool_call in enumerate(response.tool_calls, 1):
            print(f"\n工具调用 {i}:")
            print(f"  工具名: {tool_call['name']}")
            print(f"  参数: {tool_call['args']}")


# ==================== 4. 禁用并行工具调用 ====================

def sequential_tool_calling():
    """顺序工具调用示例"""
    print("\n" + "=" * 50)
    print("顺序工具调用示例 (禁用并行)")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6")

    # 禁用并行工具调用，强制模型一次只调用一个工具
    model_sequential = model.bind_tools(
        [get_weather, calculate],
        parallel_tool_calls=False  # 禁用并行调用
    )

    response = model_sequential.invoke([
        HumanMessage(content="告诉我上海的天气，并计算 25 * 4")
    ])

    print("\n问题: 告诉我上海的天气，并计算 25 * 4")
    print(f"工具调用数量: {len(response.tool_calls) if response.tool_calls else 0}")


# ==================== 5. 强制工具调用 ====================

def forced_tool_calling():
    """强制工具调用示例"""
    print("\n" + "=" * 50)
    print("强制工具调用示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6")

    # 注意: ChatZhipuAI 不支持强制使用特定工具
    # 只能使用 auto 模式，让模型自动决定
    model_with_tools = model.bind_tools([calculate])

    response = model_with_tools.invoke([
        HumanMessage(content="50 加 50 等于多少?")
    ])

    print("\n问题: 50 加 50 等于多少?")
    print("注意: ChatZhipuAI 不支持强制工具调用，使用 auto 模式")
    if response.tool_calls:
        print(f"模型选择调用工具: {response.tool_calls[0]['name']}")
        print(f"参数: {response.tool_calls[0]['args']}")


# ==================== 6. 工具调用决策 ====================

def tool_choice_modes():
    """工具调用决策模式示例"""
    print("\n" + "=" * 50)
    print("工具调用决策模式")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6")

    # 模式 1: auto - 自动决定是否调用工具 (默认)
    model_auto = model.bind_tools([get_weather], tool_choice="auto")

    # 注意: ChatZhipuAI 目前只支持 "auto" 模式
    # 模式 2: any - 必须调用某个工具 (ChatZhipuAI 不支持)
    # model_any = model.bind_tools([get_weather, calculate], tool_choice="any")

    # 模式 3: 指定工具名 - 强制使用特定工具 (ChatZhipuAI 不支持)
    # model_specific = model.bind_tools([get_weather], tool_choice="get_weather")

    print("\nChatZhipuAI 支持的工具调用模式:")
    print("  1. auto - 自动决定 (默认) ✓")
    print("\n注意: ChatZhipuAI 目前只支持 'auto' 模式")
    print("  - 'any' (必须使用某个工具) - 不支持")
    print("  - 指定工具名 (强制使用特定工具) - 不支持")

    # 测试 auto 模式
    response = model_auto.invoke([HumanMessage(content="北京的天气怎么样?")])
    if response.tool_calls:
        print(f"\nauto 模式测试: 模型自动选择调用 {response.tool_calls[0]['name']} 工具")


if __name__ == "__main__":
    try:
        # basic_tool_calling()
        # complete_tool_calling_flow()
        # parallel_tool_calling()
        # sequential_tool_calling()
        # forced_tool_calling()
        tool_choice_modes()

        print("\n" + "=" * 50)
        print("所有工具调用示例完成!")
        print("=" * 50)

    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
        import traceback
        traceback.print_exc()
