"""
LangChain 短期记忆（Short-Term Memory）完整示例
基于官方文档：https://docs.langchain.com/oss/python/langchain/short-term-memory

包含以下功能：
1. 基础短期记忆（InMemorySaver）
2. 多线程会话管理
3. 消息修剪（Trim Messages）
4. 消息删除（Remove Messages）
5. 消息摘要（Summarization）
6. 自定义状态（Custom State）
7. 工具中读写状态
8. 动态提示词
9. 状态查询与管理

使用 GLM 模型
"""

import os
from typing import Any
from pydantic import BaseModel, Field

from langchain.agents import create_agent, AgentState
from langchain_community.chat_models import ChatZhipuAI
from langchain.agents.middleware import (
    before_model,
    after_model,
    dynamic_prompt,
    SummarizationMiddleware
)
from langchain.tools import tool, ToolRuntime
from langchain.messages import RemoveMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig

# 设置 API Key
os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 1. 基础短期记忆 ====================

def example_01_basic_memory():
    """示例1：基础短期记忆 - 使用 InMemorySaver"""
    print("\n" + "=" * 60)
    print("示例1：基础短期记忆")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.5-air", temperature=0.5)
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer,
        system_prompt="你是一个友好的助手，能记住对话历史"
    )

    config = {"configurable": {"thread_id": "conversation-1"}}

    # 第一轮对话
    print("\n👤 用户: 你好！我叫张三，我喜欢编程。")
    result1 = agent.invoke(
        {"messages": [{"role": "user", "content": "你好！我叫张三，我喜欢编程。"}]},
        config
    )
    print(f"🤖 助手: {result1['messages'][-1].content}")

    # 第二轮对话 - 测试记忆
    print("\n👤 用户: 我叫什么名字？")
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
        config
    )
    print(f"🤖 助手: {result2['messages'][-1].content}")

    # 第三轮对话
    print("\n👤 用户: 我喜欢什么？")
    result3 = agent.invoke(
        {"messages": [{"role": "user", "content": "我喜欢什么？"}]},
        config
    )
    print(f"🤖 助手: {result3['messages'][-1].content}")


# ==================== 2. 多线程会话管理 ====================

def example_02_multi_thread():
    """示例2：多线程会话管理 - 每个用户独立的对话"""
    print("\n" + "=" * 60)
    print("示例2：多线程会话管理")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.5-air", temperature=0.5)
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer,
        system_prompt="你是一个客服助手"
    )

    # 用户A的会话
    print("\n--- 用户A的会话（thread_id: user-A） ---")
    config_a = {"configurable": {"thread_id": "user-A"}}

    print("👤 用户A: 我想买一台笔记本电脑")
    agent.invoke(
        {"messages": [{"role": "user", "content": "我想买一台笔记本电脑"}]},
        config_a
    )

    # 用户B的会话
    print("\n--- 用户B的会话（thread_id: user-B） ---")
    config_b = {"configurable": {"thread_id": "user-B"}}

    print("👤 用户B: 我想买一部手机")
    agent.invoke(
        {"messages": [{"role": "user", "content": "我想买一部手机"}]},
        config_b
    )

    # 回到用户A - 测试独立记忆
    print("\n--- 回到用户A ---")
    print("👤 用户A: 我刚才想买什么？")
    result_a = agent.invoke(
        {"messages": [{"role": "user", "content": "我刚才想买什么？"}]},
        config_a
    )
    print(f"🤖 助手: {result_a['messages'][-1].content}")

    # 回到用户B
    print("\n--- 回到用户B ---")
    print("👤 用户B: 我刚才想买什么？")
    result_b = agent.invoke(
        {"messages": [{"role": "user", "content": "我刚才想买什么？"}]},
        config_b
    )
    print(f"🤖 助手: {result_b['messages'][-1].content}")


# ==================== 3. 消息修剪（Trim Messages） ====================

@before_model
def trim_messages_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """保留最后几条消息以适应上下文窗口"""
    max_messages = 6  # 最多保留6条消息
    messages = state["messages"]

    if len(messages) <= max_messages:
        return None  # 不需要修剪

    print(f"\n✂️  修剪消息: {len(messages)} -> {max_messages} 条")

    # 保留第一条（通常是系统消息）和最后几条
    first_msg = messages[0] if messages else []
    recent_messages = messages[-(max_messages-1):]

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            first_msg,
            *recent_messages
        ]
    }


def example_03_trim_messages():
    """示例3：消息修剪 - 自动修剪过长的对话历史"""
    print("\n" + "=" * 60)
    print("示例3：消息修剪")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.5-air", temperature=0.5)
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer,
        middleware=[trim_messages_middleware],
        system_prompt="你是一个助手"
    )

    config = {"configurable": {"thread_id": "trim-test"}}

    # 发送多轮对话，观察修剪过程
    for i in range(10):
        print(f"\n--- 第 {i+1} 轮对话 ---")
        result = agent.invoke(
            {"messages": [{"role": "user", "content": f"这是第 {i+1} 条消息"}]},
            config
        )
        print(f"👤 用户: 这是第 {i+1} 条消息")
        print(f"📊 当前消息数: {len(result['messages'])}")


# ==================== 4. 消息删除（Remove Messages） ====================

@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """删除最早的两条消息"""
    messages = state["messages"]

    if len(messages) > 4:
        print(f"\n🗑️  删除最早的2条消息")
        # 删除最早的两条消息（跳过系统消息）
        to_delete = messages[1:3]  # 删除第2和第3条
        return {"messages": [RemoveMessage(id=m.id) for m in to_delete]}

    return None


def example_04_delete_messages():
    """示例4：消息删除 - 删除特定消息"""
    print("\n" + "=" * 60)
    print("示例4：消息删除")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.5-air", temperature=0.5)
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer,
        middleware=[delete_old_messages],
        system_prompt="你是一个简洁的助手"
    )

    config = {"configurable": {"thread_id": "delete-test"}}

    # 发送多轮对话
    for i in range(8):
        result = agent.invoke(
            {"messages": [{"role": "user", "content": f"消息 {i+1}"}]},
            config
        )
        print(f"\n第 {i+1} 轮 - 消息数: {len(result['messages'])}")


# ==================== 5. 消息摘要（Summarization） ====================

def example_05_summarization():
    """示例5：消息摘要 - 自动总结对话历史"""
    print("\n" + "=" * 60)
    print("示例5：消息摘要")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.5-air", temperature=0.5)
    checkpointer = MemorySaver()

    # 使用 SummarizationMiddleware
    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer,
        middleware=[
            SummarizationMiddleware(
                model=model,  # 使用相同模型进行摘要
                max_tokens_before_summary=500,  # 达到500个token时触发摘要
                messages_to_keep=5,             # 保留最近 5 条消息
            )
        ],
        system_prompt="你是一个助手"
    )

    config = {"configurable": {"thread_id": "summary-test"}}

    # 模拟长对话
    messages_to_send = [
        "你好！我叫李明。",
        "我今年25岁。",
        "我在北京工作。",
        "我是一名软件工程师。",
        "我喜欢Python编程。",
        "我最近在学习AI。",
        "你能帮我总结一下我的信息吗？",
    ]

    for i, msg in enumerate(messages_to_send, 1):
        print(f"\n--- 第 {i} 轮对话 ---")
        print(f"👤 用户: {msg}")
        result = agent.invoke(
            {"messages": [{"role": "user", "content": msg}]},
            config
        )
        print(f"🤖 助手: {result['messages'][-1].content}")
        print(f"📊 消息数: {len(result['messages'])}")


# ==================== 6. 自定义状态（Custom State） ====================

class UserPreferencesState(AgentState):
    """扩展的状态：包含用户偏好"""
    user_id: str = ""
    preferences: dict = {}
    session_count: int = 0


def example_06_custom_state():
    """示例6：自定义状态 - 扩展 AgentState"""
    print("\n" + "=" * 60)
    print("示例6：自定义状态")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.5-air", temperature=0.5)
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[],
        state_schema=UserPreferencesState,
        checkpointer=checkpointer,
        system_prompt="你是一个个性化助手"
    )

    config = {"configurable": {"thread_id": "custom-state-test"}}

    # 第一次调用 - 传入自定义状态
    print("\n第1次会话:")
    result1 = agent.invoke(
        {
            "messages": [{"role": "user", "content": "你好"}],
            "user_id": "user_123",
            "preferences": {"theme": "dark", "language": "zh-CN"},
            "session_count": 1
        },
        config
    )
    print(f"用户ID: {result1.get('user_id', 'N/A')}")
    print(f"偏好: {result1.get('preferences', {})}")
    print(f"会话计数: {result1.get('session_count', 0)}")

    # 第二次调用 - 状态被保留
    print("\n第2次会话:")
    result2 = agent.invoke(
        {
            "messages": [{"role": "user", "content": "我的用户ID是什么？"}],
            "session_count": 2
        },
        config
    )
    print(f"🤖 助手: {result2['messages'][-1].content}")


# ==================== 7. 工具中读取状态 ====================

class UserInfoState(AgentState):
    """包含用户信息的状态"""
    user_id: str = ""
    user_name: str = ""


@tool
def get_user_profile(runtime: ToolRuntime) -> str:
    """查询用户资料（从状态中读取 user_id）"""
    user_id = runtime.state.get("user_id", "unknown")

    # 模拟数据库查询
    user_db = {
        "user_123": {"name": "张三", "age": 28, "city": "北京"},
        "user_456": {"name": "李四", "age": 32, "city": "上海"},
    }

    user = user_db.get(user_id)
    if user:
        return f"用户 {user_id}: {user['name']}, {user['age']}岁, 来自{user['city']}"
    else:
        return f"未找到用户 {user_id}"


def example_07_tool_read_state():
    """示例7：工具中读取状态"""
    print("\n" + "=" * 60)
    print("示例7：工具中读取状态")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.5-air", temperature=0.5)

    agent = create_agent(
        model=model,
        tools=[get_user_profile],
        state_schema=UserInfoState,
        system_prompt="你是一个助手，可以查询用户信息"
    )

    print("\n👤 用户: 查询我的资料")
    result = agent.invoke({
        "messages": [{"role": "user", "content": "查询我的资料"}],
        "user_id": "user_123"
    })
    print(f"🤖 助手: {result['messages'][-1].content}")


# ==================== 8. 工具中写入状态 ====================

class UserContextState(AgentState):
    """包含用户上下文的状态"""
    user_name: str = ""
    last_action: str = ""


@tool
def update_user_name(
    user_id: str,
    runtime: ToolRuntime
) -> Command:
    """查询并更新用户名到状态

    Args:
        user_id: 用户ID
    """
    # 模拟查询
    name_db = {
        "user_123": "张三",
        "user_456": "李四"
    }
    user_name = name_db.get(user_id, "未知用户")

    # 返回 Command 更新状态
    return Command(update={
        "user_name": user_name,
        "last_action": "update_user_name",
        "messages": [
            ToolMessage(
                f"已更新用户名: {user_name}",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })


@tool
def greet_user(
    runtime: ToolRuntime
) -> str:
    """问候用户（从状态读取用户名）"""
    user_name = runtime.state.get("user_name", "")

    if not user_name:
        return "请先更新用户信息"

    return f"你好，{user_name}！"


def example_08_tool_write_state():
    """示例8：工具中写入状态"""
    print("\n" + "=" * 60)
    print("示例8：工具中写入状态")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.5-air", temperature=0.5)
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[update_user_name, greet_user],
        state_schema=UserContextState,
        checkpointer=checkpointer,
        system_prompt="你是一个助手，可以更新和使用用户信息"
    )

    config = {"configurable": {"thread_id": "write-state-test"}}

    print("\n👤 用户: 帮我更新用户ID为 user_123 的用户名，然后问候我")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "帮我更新用户ID为 user_123 的用户名，然后问候我"}]},
        config
    )
    print(f"🤖 助手: {result['messages'][-1].content}")


# ==================== 9. 动态提示词 ====================

class GreetingContext(BaseModel):
    """问候上下文"""
    user_name: str
    time_of_day: str


@dynamic_prompt
def create_dynamic_system_prompt(request) -> str:
    """根据上下文动态生成系统提示词"""
    context = request.runtime.context
    # GreetingContext 是 Pydantic 模型，使用属性访问而不是 .get()
    user_name = getattr(context, "user_name", "用户")
    time_of_day = getattr(context, "time_of_day", "")

    greeting = {
        "morning": "早上好",
        "afternoon": "下午好",
        "evening": "晚上好"
    }.get(time_of_day, "你好")

    return f"{greeting}，{user_name}！我是你的AI助手。"


def example_09_dynamic_prompt():
    """示例9：动态提示词"""
    print("\n" + "=" * 60)
    print("示例9：动态提示词")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.5-air", temperature=0.5)

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[create_dynamic_system_prompt],
        context_schema=GreetingContext
    )

    # 早上问候
    print("\n--- 早上 ---")
    result1 = agent.invoke(
        {"messages": [{"role": "user", "content": "你好"}]},
        context=GreetingContext(user_name="张三", time_of_day="morning")
    )
    print(f"🤖 助手: {result1['messages'][-1].content}")

    # 晚上问候
    print("\n--- 晚上 ---")
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "你好"}]},
        context=GreetingContext(user_name="李四", time_of_day="evening")
    )
    print(f"🤖 助手: {result2['messages'][-1].content}")


# ==================== 10. 状态查询与管理 ====================

def example_10_state_management():
    """示例10：状态查询与管理"""
    print("\n" + "=" * 60)
    print("示例10：状态查询与管理")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.5-air", temperature=0.5)
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer,
        system_prompt="你是一个助手"
    )

    config = {"configurable": {"thread_id": "state-mgmt-test"}}

    # 发送几条消息
    print("\n发送消息...")
    for i in range(3):
        agent.invoke(
            {"messages": [{"role": "user", "content": f"消息 {i+1}"}]},
            config
        )
        print(f"  ✓ 已发送: 消息 {i+1}")

    # 查询当前状态
    print("\n查询状态:")
    state = agent.get_state(config)
    print(f"  消息数量: {len(state.values['messages'])}")
    print(f"  最新消息: {state.values['messages'][-1].content}")

    # 查看所有消息
    print("\n所有消息:")
    for i, msg in enumerate(state.values['messages'], 1):
        print(f"  {i}. [{msg.type}] {msg.content[:50]}...")


# ==================== 主函数 ====================

def main():
    """运行所有示例"""
    examples = [
        ("基础短期记忆", example_01_basic_memory),
        ("多线程会话管理", example_02_multi_thread),
        ("消息修剪", example_03_trim_messages),
        ("消息删除", example_04_delete_messages),
        ("消息摘要", example_05_summarization),
        ("自定义状态", example_06_custom_state),
        ("工具读取状态", example_07_tool_read_state),
        ("工具写入状态", example_08_tool_write_state),
        ("动态提示词", example_09_dynamic_prompt),
        ("状态管理", example_10_state_management),
    ]

    print("\n" + "=" * 60)
    print("LangChain 短期记忆完整示例")
    print("=" * 60)

    for i, (name, func) in enumerate(examples, 1):
        print(f"\n{i}. {name}")

    print("\n请选择要运行的示例（输入数字，0运行全部）:")
    choice = input(">>> ").strip()

    if choice == "0":
        for name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"\n❌ 错误: {str(e)}")
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        try:
            examples[int(choice) - 1][1]()
        except Exception as e:
            print(f"\n❌ 错误: {str(e)}")
    else:
        print("无效的选择")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已终止")
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        print("\n请确保已设置 ZHIPUAI_API_KEY 环境变量")
