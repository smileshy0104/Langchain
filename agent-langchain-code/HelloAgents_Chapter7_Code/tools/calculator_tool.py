"""
计算器工具
使用 LangChain 的 BaseTool 和 @tool 装饰器实现
"""

from langchain_core.tools import BaseTool, tool
from pydantic import Field
from core.utils import safe_eval


class CalculatorTool(BaseTool):
    """
    计算器工具 - 继承 BaseTool 方式
    支持基本数学运算和常用函数
    """
    name: str = "calculator"
    description: str = "执行数学计算，支持基本运算(+,-,*,/)和常用函数(sqrt,sin,cos等)"

    def _run(self, expression: str) -> str:
        """
        执行计算

        Args:
            expression: 数学表达式

        Returns:
            计算结果
        """
        return safe_eval(expression)

    async def _arun(self, expression: str) -> str:
        """异步执行（当前实现与同步相同）"""
        return self._run(expression)


@tool
def create_calculator(expression: str) -> str:
    """
    计算器工具 - 装饰器方式

    执行数学计算，支持：
    - 基本运算: +, -, *, /
    - 常用函数: sqrt(开方), sin, cos, tan, log, exp, abs
    - 常数: pi, e

    Args:
        expression: 要计算的数学表达式，如 "2 + 3 * 4" 或 "sqrt(16)"

    Returns:
        计算结果字符串
    """
    return safe_eval(expression)


@tool
def simple_add(a: float, b: float) -> str:
    """
    简单加法工具

    Args:
        a: 第一个数
        b: 第二个数

    Returns:
        a + b 的结果
    """
    return str(a + b)


@tool
def simple_multiply(a: float, b: float) -> str:
    """
    简单乘法工具

    Args:
        a: 第一个数
        b: 第二个数

    Returns:
        a * b 的结果
    """
    return str(a * b)
