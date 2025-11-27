"""
搜索工具
提供模拟搜索功能（实际项目中可接入真实搜索 API）
"""

from langchain_core.tools import BaseTool, tool
from typing import Dict


# 模拟搜索数据库
MOCK_SEARCH_DB = {
    "python": "Python是一种高级编程语言，由Guido van Rossum创建。它以简洁易读的语法著称，广泛应用于Web开发、数据分析、人工智能等领域。",
    "机器学习": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习并改进性能，而无需明确编程。常见算法包括神经网络、决策树、支持向量机等。",
    "langchain": "LangChain是一个用于开发由语言模型驱动的应用程序的框架。它提供了工具、组件和接口，简化了创建LLM应用程序的过程。",
    "北京": "北京是中华人民共和国的首都，也是中国的政治、文化和国际交往中心。北京有着3000多年的建城史和800多年的建都史。",
    "天气": "今天天气晴朗，温度适中，适合户外活动。明天可能有小雨，建议携带雨具出行。",
    "人工智能": "人工智能(AI)是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。",
}


class MockSearchTool(BaseTool):
    """
    模拟搜索工具 - 继承 BaseTool 方式
    用于演示，返回预定义的搜索结果
    """
    name: str = "search"
    description: str = "搜索互联网信息，返回相关内容"

    def _run(self, query: str) -> str:
        """
        执行搜索

        Args:
            query: 搜索查询

        Returns:
            搜索结果
        """
        query_lower = query.lower()

        # 查找匹配的结果
        for key, value in MOCK_SEARCH_DB.items():
            if key in query_lower or query_lower in key:
                return f"搜索结果: {value}"

        return f"未找到关于 '{query}' 的相关信息。请尝试其他关键词。"

    async def _arun(self, query: str) -> str:
        """异步执行"""
        return self._run(query)


@tool
def create_mock_search(query: str) -> str:
    """
    模拟搜索工具 - 装饰器方式

    搜索互联网信息并返回相关内容。这是一个模拟实现，返回预定义的结果。

    Args:
        query: 搜索查询关键词

    Returns:
        搜索结果字符串
    """
    query_lower = query.lower()

    for key, value in MOCK_SEARCH_DB.items():
        if key in query_lower or query_lower in key:
            return f"搜索结果: {value}"

    return f"未找到关于 '{query}' 的相关信息。"


@tool
def get_current_time() -> str:
    """
    获取当前时间

    Returns:
        当前时间字符串
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_weather(city: str) -> str:
    """
    获取城市天气信息（模拟）

    Args:
        city: 城市名称

    Returns:
        天气信息
    """
    weather_data = {
        "北京": "晴朗，温度 15-25°C，空气质量良好",
        "上海": "多云，温度 18-28°C，有轻微雾霾",
        "广州": "小雨，温度 22-30°C，湿度较大",
        "深圳": "晴转多云，温度 20-28°C，空气清新",
    }

    return weather_data.get(city, f"{city}的天气信息暂时无法获取")
