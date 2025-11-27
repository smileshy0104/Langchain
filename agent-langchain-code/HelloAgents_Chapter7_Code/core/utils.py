"""
工具函数
提供 LLM 设置、消息格式化等通用功能
"""

import os
from typing import List, Dict, Any, Optional
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def setup_llm(
    model: str = "glm-4-plus",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs
) -> ChatZhipuAI:
    """
    设置和初始化 LLM

    Args:
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大 token 数
        **kwargs: 其他参数

    Returns:
        ChatZhipuAI 实例
    """
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError("未设置 ZHIPUAI_API_KEY 环境变量")

    return ChatZhipuAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        zhipuai_api_key=api_key,
        **kwargs
    )


def format_chat_history(messages: List[Dict[str, str]]) -> List[Any]:
    """
    将字典格式的消息转换为 LangChain 消息对象

    Args:
        messages: 字典格式的消息列表，如 [{"role": "user", "content": "hi"}]

    Returns:
        LangChain 消息对象列表
    """
    langchain_messages = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            langchain_messages.append(SystemMessage(content=content))
        elif role == "assistant" or role == "ai":
            langchain_messages.append(AIMessage(content=content))
        else:  # user or human
            langchain_messages.append(HumanMessage(content=content))

    return langchain_messages


def extract_text_from_response(response: Any) -> str:
    """
    从 LLM 响应中提取文本内容

    Args:
        response: LLM 响应对象

    Returns:
        提取的文本内容
    """
    if hasattr(response, 'content'):
        return response.content
    elif isinstance(response, str):
        return response
    elif isinstance(response, dict):
        return response.get('content', str(response))
    else:
        return str(response)


def truncate_messages(
    messages: List[Dict[str, str]],
    max_length: int = 10
) -> List[Dict[str, str]]:
    """
    截断消息历史，保留最近的消息

    Args:
        messages: 消息列表
        max_length: 保留的最大消息数

    Returns:
        截断后的消息列表
    """
    if len(messages) <= max_length:
        return messages

    # 保留系统消息（如果有）
    system_messages = [m for m in messages if m.get("role") == "system"]
    other_messages = [m for m in messages if m.get("role") != "system"]

    # 保留最近的消息
    truncated = other_messages[-max_length:]

    # 系统消息放在最前面
    return system_messages + truncated


def safe_eval(expression: str) -> str:
    """
    安全地计算数学表达式

    Args:
        expression: 数学表达式字符串

    Returns:
        计算结果或错误信息
    """
    import ast
    import operator
    import math

    # 支持的运算符
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    # 支持的函数
    functions = {
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'exp': math.exp,
        'abs': abs,
        'pi': math.pi,
        'e': math.e,
    }

    def _eval_node(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            return functions.get(node.id, 0)
        elif isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = _eval_node(node.operand)
            return operators[type(node.op)](operand)
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name in functions:
                args = [_eval_node(arg) for arg in node.args]
                return functions[func_name](*args)
        else:
            raise ValueError(f"不支持的节点类型: {type(node)}")

    try:
        tree = ast.parse(expression, mode='eval')
        result = _eval_node(tree.body)
        return str(result)
    except Exception as e:
        return f"计算错误: {str(e)}"


def format_tool_output(tool_name: str, tool_input: Any, tool_output: Any) -> str:
    """
    格式化工具输出信息

    Args:
        tool_name: 工具名称
        tool_input: 工具输入
        tool_output: 工具输出

    Returns:
        格式化的字符串
    """
    return f"""
🔧 工具调用: {tool_name}
📥 输入: {tool_input}
📤 输出: {tool_output}
"""
