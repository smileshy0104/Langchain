#!/usr/bin/env python3
"""
自定义工具模块

定义了各种工具供智能体使用:
- 搜索工具: 网页搜索（需要 SerpAPI）
- 计算器: 数学表达式计算
- 时间查询: 获取当前时间
- 天气查询: 模拟天气查询
"""

import os
from langchain_core.tools import tool


@tool
def search(query: str) -> str:
    """网页搜索引擎工具。当你需要查询实时信息、事实或不在知识库中的内容时使用。

    Args:
        query: 搜索查询内容，例如 '华为最新手机'、'北京今天天气'

    Returns:
        搜索结果摘要
    """
    print(f"🔍 正在搜索: {query}")

    try:
        from serpapi import SerpApiClient

        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key or api_key.startswith("your-"):
            return (
                "⚠️ 搜索功能未配置。请设置 SERPAPI_API_KEY 环境变量。\n"
                "获取密钥: https://serpapi.com/"
            )

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",  # 国家代码：中国
            "hl": "zh-cn",  # 语言代码：简体中文
        }

        client = SerpApiClient(params)
        results = client.get_dict()

        # 智能提取结果（优先级从高到低）
        # 1. 答案框列表
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])

        # 2. 答案框
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]

        # 3. 知识图谱
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]

        # 4. 有机搜索结果（前3个）
        if "organic_results" in results and results["organic_results"]:
            snippets = []
            for i, res in enumerate(results["organic_results"][:3], 1):
                title = res.get("title", "")
                snippet = res.get("snippet", "")
                snippets.append(f"[{i}] {title}\n{snippet}")
            return "\n\n".join(snippets)

        return f"未找到关于 '{query}' 的相关信息"

    except ImportError:
        return (
            "⚠️ 搜索功能需要安装 google-search-results 包。\n"
            "安装命令: pip install google-search-results"
        )
    except Exception as e:
        return f"搜索时发生错误: {str(e)}"


@tool
def calculator(expression: str) -> str:
    """执行数学计算。支持基本运算和复杂表达式。

    Args:
        expression: 数学表达式，例如 '2 + 3 * 4', '(100 + 50) / 3', '15 ** 2'

    Returns:
        计算结果

    Examples:
        >>> calculator("2 + 3")
        "计算结果: 2 + 3 = 5"
        >>> calculator("15 ** 2")
        "计算结果: 15 ** 2 = 225"
    """
    print(f"🧮 正在计算: {expression}")

    try:
        # 安全检查：只允许数字和运算符
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars or c == '*' for c in expression):
            return f"错误：表达式包含无效字符。只允许数字和运算符 (+ - * / ** ( ))"

        # 计算结果
        result = eval(expression)
        return f"计算结果: {expression} = {result}"

    except ZeroDivisionError:
        return f"错误：除数不能为零"
    except SyntaxError:
        return f"错误：表达式语法错误"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def get_time(timezone: str = "Asia/Shanghai") -> str:
    """获取当前时间。

    Args:
        timezone: 时区，默认为亚洲/上海（Asia/Shanghai）
            常用时区:
            - Asia/Shanghai (中国)
            - Asia/Tokyo (日本)
            - America/New_York (美国东部)
            - Europe/London (英国)
            - UTC (协调世界时)

    Returns:
        当前时间字符串
    """
    print(f"🕐 正在获取时间: {timezone}")

    from datetime import datetime

    try:
        # 尝试使用 pytz
        import pytz
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        return f"当前时间（{timezone}）: {time_str}"

    except ImportError:
        # 如果没有 pytz，使用本地时间
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"当前时间（本地）: {time_str}\n提示: 安装 pytz 可支持时区功能"

    except Exception as e:
        return f"获取时间时发生错误: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气预报（模拟数据）。

    注意: 这是一个演示工具，返回模拟的天气数据。
    实际应用中应接入真实的天气 API（如和风天气、OpenWeatherMap等）。

    Args:
        city: 城市名称，例如 '北京'、'上海'、'深圳'、'厦门'

    Returns:
        天气信息字符串
    """
    print(f"🌤️ 正在查询天气: {city}")

    import random

    # 模拟天气数据
    conditions = [
        "晴天",
        "多云",
        "小雨",
        "阴天",
        "雾霾",
        "大风",
        "雷阵雨"
    ]

    temp = random.randint(10, 30)
    condition = random.choice(conditions)
    humidity = random.randint(40, 80)
    wind_speed = random.randint(1, 15)

    return f"""
{city}今天天气：
- 天气状况: {condition}
- 温度: {temp}°C
- 湿度: {humidity}%
- 风速: {wind_speed} km/h

⚠️ 注意: 这是模拟数据，仅供演示使用。
""".strip()


@tool
def python_repl(code: str) -> str:
    """执行 Python 代码并返回结果。

    ⚠️ 警告: 这是一个危险的工具，仅用于演示。
    生产环境中应使用沙箱环境执行代码。

    Args:
        code: 要执行的 Python 代码

    Returns:
        执行结果或错误信息
    """
    print(f"🐍 正在执行 Python 代码...")

    try:
        # 创建隔离的命名空间
        namespace = {}

        # 执行代码
        exec(code, namespace)

        # 获取结果（如果有的话）
        if 'result' in namespace:
            return f"执行成功，结果: {namespace['result']}"
        else:
            return "代码执行成功（无返回值）"

    except Exception as e:
        return f"执行错误: {type(e).__name__}: {str(e)}"


# 工具测试代码
if __name__ == "__main__":
    print("🧪 测试自定义工具\n")

    # 1. 测试计算器
    print("1️⃣ 测试计算器:")
    result = calculator.invoke("15 * 23 + 7")
    print(f"   {result}\n")

    # 2. 测试天气查询
    print("2️⃣ 测试天气查询:")
    result = get_weather.invoke("厦门")
    print(f"   {result}\n")

    # 3. 测试时间查询
    print("3️⃣ 测试时间查询:")
    result = get_time.invoke("Asia/Shanghai")
    print(f"   {result}\n")

    # 4. 测试搜索（如果配置了 API）
    print("4️⃣ 测试搜索:")
    result = search.invoke("LangChain")
    print(f"   {result}\n")

    # 5. 测试 Python REPL
    print("5️⃣ 测试 Python REPL:")
    result = python_repl.invoke("result = 2 ** 10")
    print(f"   {result}\n")

    print("✨ 所有工具测试完成！")

    # 打印工具列表
    print("\n📋 可用工具列表:")
    all_tools = [search, calculator, get_time, get_weather, python_repl]
    for tool in all_tools:
        print(f"   - {tool.name}: {tool.description}")
