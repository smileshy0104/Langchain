"""
工具系统
基于 LangChain 的工具定义和管理
"""

from typing import List, Dict, Any, Optional, Callable
from langchain_core.tools import BaseTool, tool
from pydantic import Field


class ToolRegistry:
    """
    工具注册表
    管理和执行工具的中心类
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._tool_descriptions: Dict[str, str] = {}

    def register_tool(self, tool_instance: BaseTool):
        """
        注册一个工具实例

        Args:
            tool_instance: BaseTool 实例
        """
        name = tool_instance.name
        self._tools[name] = tool_instance
        self._tool_descriptions[name] = tool_instance.description
        print(f"✅ 工具 '{name}' 已注册")

    def register_function(
        self,
        name: str,
        func: Callable,
        description: str
    ):
        """
        直接注册函数为工具（简便方式）

        Args:
            name: 工具名称
            func: 工具函数
            description: 工具描述
        """
        # 使用 LangChain 的 @tool 装饰器
        tool_instance = tool(name=name, description=description)(func)
        self.register_tool(tool_instance)

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        获取工具实例

        Args:
            name: 工具名称

        Returns:
            工具实例或 None
        """
        return self._tools.get(name)

    def get_all_tools(self) -> List[BaseTool]:
        """
        获取所有已注册的工具

        Returns:
            工具列表
        """
        return list(self._tools.values())

    def execute_tool(self, tool_name: str, tool_input: Any) -> str:
        """
        执行指定工具

        Args:
            tool_name: 工具名称
            tool_input: 工具输入

        Returns:
            工具执行结果
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return f"❌ 错误: 未找到工具 '{tool_name}'"

        try:
            # LangChain 工具统一使用 invoke 方法
            result = tool.invoke(tool_input)
            return str(result)
        except Exception as e:
            return f"❌ 工具执行失败: {str(e)}"

    def get_tools_description(self) -> str:
        """
        获取所有工具的格式化描述

        Returns:
            工具描述字符串
        """
        if not self._tool_descriptions:
            return "暂无可用工具"

        descriptions = []
        for name, desc in self._tool_descriptions.items():
            descriptions.append(f"- {name}: {desc}")

        return "\n".join(descriptions)

    def list_tools(self) -> List[str]:
        """
        列出所有工具名称

        Returns:
            工具名称列表
        """
        return list(self._tools.keys())

    def unregister_tool(self, name: str) -> bool:
        """
        注销工具

        Args:
            name: 工具名称

        Returns:
            是否成功注销
        """
        if name in self._tools:
            del self._tools[name]
            del self._tool_descriptions[name]
            print(f"✅ 工具 '{name}' 已注销")
            return True
        return False

    def clear(self):
        """清空所有已注册的工具"""
        self._tools.clear()
        self._tool_descriptions.clear()
        print("✅ 已清空所有工具")

    def __len__(self) -> int:
        """返回已注册工具的数量"""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """检查工具是否已注册"""
        return name in self._tools


# ==================== 示例工具定义 ====================

class CalculatorToolExample(BaseTool):
    """
    示例：计算器工具
    展示如何继承 BaseTool 创建工具
    """
    name: str = "calculator"
    description: str = "执行数学计算，支持基本运算和常用函数"

    def _run(self, expression: str) -> str:
        """
        执行计算

        Args:
            expression: 数学表达式

        Returns:
            计算结果
        """
        from .utils import safe_eval
        return safe_eval(expression)

    async def _arun(self, expression: str) -> str:
        """异步执行（当前实现与同步相同）"""
        return self._run(expression)


# ==================== 装饰器方式定义工具 ====================

@tool
def simple_calculator(expression: str) -> str:
    """
    简单计算器工具（装饰器方式）

    Args:
        expression: 要计算的数学表达式，如 "2 + 3 * 4"

    Returns:
        计算结果
    """
    from .utils import safe_eval
    return safe_eval(expression)


@tool
def text_length(text: str) -> str:
    """
    计算文本长度

    Args:
        text: 输入文本

    Returns:
        文本长度
    """
    return f"文本长度: {len(text)} 个字符"


@tool
def reverse_text(text: str) -> str:
    """
    反转文本

    Args:
        text: 输入文本

    Returns:
        反转后的文本
    """
    return text[::-1]


# ==================== 工具创建辅助函数 ====================

def create_tool_from_function(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None
) -> BaseTool:
    """
    从普通函数创建工具

    Args:
        func: 函数
        name: 工具名称（默认使用函数名）
        description: 工具描述（默认使用函数文档字符串）

    Returns:
        BaseTool 实例
    """
    tool_name = name or func.__name__
    tool_description = description or (func.__doc__ or "无描述")

    return tool(name=tool_name, description=tool_description)(func)


def create_simple_registry() -> ToolRegistry:
    """
    创建一个带有基础工具的注册表

    Returns:
        包含基础工具的 ToolRegistry
    """
    registry = ToolRegistry()

    # 注册基础工具
    registry.register_tool(CalculatorToolExample())
    registry.register_tool(simple_calculator)
    registry.register_tool(text_length)
    registry.register_tool(reverse_text)

    return registry
