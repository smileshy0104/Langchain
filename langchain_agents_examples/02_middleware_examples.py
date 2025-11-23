"""
LangChain Agents 中间件示例
演示动态模型选择、工具错误处理、动态提示词等
使用 GLM 模型
"""

from langchain.agents import create_agent
from langchain_community.chat_models import ChatZhipuAI
from langchain.agents.middleware import (
    wrap_model_call,
    wrap_tool_call,
    dynamic_prompt,
    before_model,
    after_model
)
from langchain_core.tools import tool, ToolException
from langchain_core.messages import ToolMessage, AIMessage
from typing import Callable
import os
import time

# 设置 API Key
os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 1. 工具定义 ====================

@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"搜索结果: {query}"


@tool
def get_weather(location: str) -> str:
    """获取天气信息"""
    # 模拟可能失败的API调用
    if location == "火星":
        raise Exception("无法获取火星天气数据")
    return f"{location} 的天气是晴朗的,温度 22°C"


@tool
def calculate(expression: str) -> float:
    """计算数学表达式"""
    try:
        return eval(expression)
    except Exception as e:
        raise ToolException(f"计算错误: {str(e)}")


@tool
def risky_operation(param: str) -> str:
    """可能失败的操作"""
    if "error" in param.lower():
        raise Exception("操作失败: 参数包含错误关键字")
    return f"成功处理: {param}"


# ==================== 2. 工具错误处理中间件 ====================

@wrap_tool_call
def handle_tool_errors(request, handler):
    """
    统一处理工具执行错误
    
    该装饰器用于捕获工具执行过程中的异常，并返回友好的错误消息给模型。
    
    参数:
        request: 工具调用请求对象，包含tool_call等信息
        handler: 工具处理函数
        
    返回:
        工具执行结果或ToolMessage错误消息
    """
    try:
        return handler(request)
    except Exception as e:
        print(f"⚠️  工具 {request.tool_call['name']} 执行失败: {str(e)}")
        # 返回友好的错误消息给模型
        return ToolMessage(
            content=f"工具执行失败: 请检查输入参数。详情: {str(e)}",
            tool_call_id=request.tool_call["id"]
        )


def tool_error_handling_example():
    """工具错误处理示例"""
    print("=" * 50)
    print("工具错误处理示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)

    agent = create_agent(
        model=model,
        tools=[get_weather, calculate, risky_operation],
        middleware=[handle_tool_errors],
        system_prompt="你是一个智能助手,可以查询天气、计算和执行操作"
    )

    # 测试会导致错误的查询
    test_queries = [
        "火星的天气如何？",  # 会导致天气API错误
        "计算 10 / 0",  # 会导致计算错误
        "执行 error 操作"  # 会导致操作失败
    ]

    for query in test_queries:
        print(f"\n问题: {query}")
        try:
            result = agent.invoke({
                "messages": [{"role": "user", "content": query}]
            })
            print(f"回答: {result['messages'][-1].content}")
        except Exception as e:
            print(f"异常: {str(e)}")


# ==================== 3. 动态模型选择中间件 ====================

@wrap_model_call
def dynamic_model_selection(request, handler):
    """根据对话复杂度选择模型"""
    message_count = len(request.messages)

    print(f"\n📊 当前消息数: {message_count}")

    # 根据消息数量选择模型
    if message_count > 5:
        print("✨ 使用高级模型 (glm-4.6)")
        model = ChatZhipuAI(model="glm-4.6", temperature=0.7)
    else:
        print("⚡ 使用快速模型 (glm-4-flash)")
        model = ChatZhipuAI(model="glm-4-flash", temperature=0.5)

    # 使用override方法替换模型
    request = request.override(model=model)
    return handler(request)


def dynamic_model_example():
    """动态模型选择示例"""
    print("\n" + "=" * 50)
    print("动态模型选择示例")
    print("=" * 50)

    # 初始使用基础模型
    basic_model = ChatZhipuAI(model="glm-4-flash", temperature=0.5)

    agent = create_agent(
        model=basic_model,
        tools=[search, get_weather],
        middleware=[dynamic_model_selection],
        system_prompt="你是一个智能助手"
    )

    # 短对话
    print("\n--- 短对话测试 ---")
    result = agent.invoke({
        "messages": [{"role": "user", "content": "你好"}]
    })
    print(f"回答: {result['messages'][-1].content}")


# ==================== 4. 动态提示词中间件 ====================

@dynamic_prompt
def context_aware_prompt(request):
    """基于上下文生成动态提示"""
    message_count = len(request.messages)

    base = "你是一个有帮助的助手。"

    # 长对话时要求简洁
    if message_count > 10:
        base += "\n这是一个长对话 - 请保持回答简洁。"

    # 检查是否有工具调用
    has_tool_calls = any(
        hasattr(msg, 'tool_calls') and msg.tool_calls
        for msg in request.messages
    )
    if has_tool_calls:
        base += "\n你已经使用了工具,请基于工具结果给出准确回答。"

    return base


def dynamic_prompt_example():
    """动态提示词示例"""
    print("\n" + "=" * 50)
    print("动态提示词示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)

    agent = create_agent(
        model=model,
        tools=[search, get_weather],
        middleware=[context_aware_prompt]
    )

    # 测试查询
    result = agent.invoke({
        "messages": [{"role": "user", "content": "北京天气如何？"}]
    })

    print(f"\n问题: 北京天气如何？")
    print(f"回答: {result['messages'][-1].content}")


# ==================== 5. before_model 钩子 ====================

@before_model
def log_before_model(state, runtime):
    """记录模型调用前的状态"""
    print(f"\n🔍 准备调用模型,当前有 {len(state['messages'])} 条消息")
    return None  # 不修改状态


def before_model_example():
    """before_model 钩子示例"""
    print("\n" + "=" * 50)
    print("before_model 钩子示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[log_before_model],
        system_prompt="你是一个智能助手"
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": "搜索 Python 教程"}]
    })

    print(f"\n回答: {result['messages'][-1].content}")


# ==================== 6. after_model 钩子 ====================

@after_model(can_jump_to=["end"])
def validate_output(state, runtime):
    """验证模型输出并应用内容过滤"""
    last_message = state["messages"][-1]

    # 检查是否包含禁止内容
    forbidden_words = ["禁止", "敏感"]
    if any(word in last_message.content for word in forbidden_words):
        print("\n⚠️  检测到禁止内容,提前结束")
        return {
            "messages": [AIMessage(content="抱歉,我无法回应该请求。")],
            "jump_to": "end"  # 提前结束 Agent
        }

    return None


def after_model_example():
    """after_model 钩子示例"""
    print("\n" + "=" * 50)
    print("after_model 钩子示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)

    agent = create_agent(
        model=model,
        tools=[search],
        middleware=[validate_output],
        system_prompt="你是一个智能助手"
    )

    # 正常查询
    result = agent.invoke({
        "messages": [{"role": "user", "content": "搜索天气信息"}]
    })
    print(f"\n问题: 搜索天气信息")
    print(f"回答: {result['messages'][-1].content}")


# ==================== 7. 工具执行日志中间件 ====================

@wrap_tool_call
def log_tool_execution(request, handler):
    """记录工具执行"""
    tool_name = request.tool_call["name"]
    print(f"\n🔧 执行工具: {tool_name}")
    print(f"   参数: {request.tool_call.get('args', {})}")

    start_time = time.time()
    result = handler(request)
    elapsed = time.time() - start_time

    print(f"✅ 工具 {tool_name} 完成,耗时: {elapsed:.2f}s")
    return result


def tool_logging_example():
    """工具执行日志示例"""
    print("\n" + "=" * 50)
    print("工具执行日志示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)

    agent = create_agent(
        model=model,
        tools=[search, get_weather, calculate],
        middleware=[log_tool_execution],
        system_prompt="你是一个智能助手,可以搜索、查天气、计算"
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": "北京天气如何？然后计算 100 * 50"}]
    })

    print(f"\n最终回答: {result['messages'][-1].content}")


# ==================== 8. 组合多个中间件 ====================

def combined_middleware_example():
    """组合多个中间件示例"""
    print("\n" + "=" * 50)
    print("组合多个中间件示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)

    agent = create_agent(
        model=model,
        tools=[search, get_weather, calculate],
        middleware=[
            log_before_model,      # 记录调用前状态
            context_aware_prompt,  # 动态提示词
            handle_tool_errors,    # 工具错误处理
            log_tool_execution,    # 工具执行日志
        ],
        system_prompt="你是一个智能助手"
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": "查询上海天气,并计算 25 * 4"}]
    })

    print(f"\n最终回答: {result['messages'][-1].content}")


if __name__ == "__main__":
    try:
        # tool_error_handling_example()
        # dynamic_model_example()
        # dynamic_prompt_example()
        # before_model_example()
        # after_model_example()
        # tool_logging_example()
        combined_middleware_example()
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
