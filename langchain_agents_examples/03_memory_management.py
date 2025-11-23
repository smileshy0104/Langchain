"""
LangChain Agents 记忆管理示例
演示短期记忆、消息修剪、自定义状态等
使用 GLM 模型
"""

from langchain.agents import create_agent, AgentState
from langchain_community.chat_models import ChatZhipuAI
from langchain.agents.middleware import before_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
import os

# 设置 API Key
os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 1. 工具定义 ====================

@tool
def get_user_info(user_id: str) -> str:
    """获取用户信息"""
    # 模拟用户数据
    users = {
        "user_123": {"name": "张三", "age": 28, "city": "北京"},
        "user_456": {"name": "李四", "age": 32, "city": "上海"},
    }
    user = users.get(user_id, {"name": "未知", "age": 0, "city": "未知"})
    return f"用户 {user_id}: {user['name']}, {user['age']}岁, 来自{user['city']}"


@tool
def save_preference(key: str, value: str) -> str:
    """保存用户偏好"""
    return f"已保存偏好: {key} = {value}"


# ==================== 2. 基础短期记忆 ====================

def basic_memory_example():
    """基础短期记忆示例"""
    print("=" * 50)
    print("基础短期记忆示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4-plus", temperature=0.5)

    # 创建检查点器
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer,  # 启用记忆
        system_prompt="你是一个有帮助的助手,可以记住对话历史"
    )

    # 第一轮对话
    print("\n第一轮对话:")
    result1 = agent.invoke(
        {"messages": [{"role": "user", "content": "你好！我叫张三。"}]},
        {"configurable": {"thread_id": "conversation-1"}}
    )
    print(f"用户: 你好！我叫张三。")
    print(f"助手: {result1['messages'][-1].content}")

    # 第二轮对话 - Agent 记得之前的内容
    print("\n第二轮对话:")
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
        {"configurable": {"thread_id": "conversation-1"}}
    )
    print(f"用户: 我叫什么名字？")
    print(f"助手: {result2['messages'][-1].content}")

    # 第三轮对话
    print("\n第三轮对话:")
    result3 = agent.invoke(
        {"messages": [{"role": "user", "content": "我刚才说了什么？"}]},
        {"configurable": {"thread_id": "conversation-1"}}
    )
    print(f"用户: 我刚才说了什么？")
    print(f"助手: {result3['messages'][-1].content}")


# ==================== 3. 多会话管理 ====================

def multi_session_example():
    """多会话管理示例"""
    print("\n" + "=" * 50)
    print("多会话管理示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4-plus", temperature=0.5)
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer,
        system_prompt="你是一个客服助手"
    )

    # 会话1: 用户A
    print("\n--- 会话1: 用户A ---")
    agent.invoke(
        {"messages": [{"role": "user", "content": "我想买一台笔记本电脑"}]},
        {"configurable": {"thread_id": "user-A"}}
    )
    print("用户A: 我想买一台笔记本电脑")

    # 会话2: 用户B
    print("\n--- 会话2: 用户B ---")
    agent.invoke(
        {"messages": [{"role": "user", "content": "我想买一部手机"}]},
        {"configurable": {"thread_id": "user-B"}}
    )
    print("用户B: 我想买一部手机")

    # 回到会话1
    print("\n--- 回到会话1: 用户A ---")
    result_a = agent.invoke(
        {"messages": [{"role": "user", "content": "我刚才想买什么？"}]},
        {"configurable": {"thread_id": "user-A"}}
    )
    print("用户A: 我刚才想买什么？")
    print(f"助手: {result_a['messages'][-1].content}")

    # 回到会话2
    print("\n--- 回到会话2: 用户B ---")
    result_b = agent.invoke(
        {"messages": [{"role": "user", "content": "我刚才想买什么？"}]},
        {"configurable": {"thread_id": "user-B"}}
    )
    print("用户B: 我刚才想买什么？")
    print(f"助手: {result_b['messages'][-1].content}")


# ==================== 4. 自定义状态模式 ====================

class CustomAgentState(AgentState):
    """扩展的 Agent 状态"""
    user_id: str = ""  # 用户 ID
    preferences: dict = {}  # 用户偏好


def custom_state_example():
    """自定义状态示例"""
    print("\n" + "=" * 50)
    print("自定义状态示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4-plus", temperature=0.5)
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[get_user_info],
        state_schema=CustomAgentState,
        checkpointer=checkpointer,
        system_prompt="你是一个个性化助手,可以记住用户信息和偏好"
    )

    # 传入自定义状态
    print("\n第一轮对话:")
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "推荐一部电影"}],
            "user_id": "user_123",
            "preferences": {"genre": "科幻", "rating": ">8.0"}
        },
        {"configurable": {"thread_id": "session-1"}}
    )
    print("用户: 推荐一部电影")
    print(f"用户ID: user_123")
    print(f"偏好: 科幻, 评分>8.0")
    print(f"助手: {result['messages'][-1].content}")


# ==================== 5. 消息修剪 ====================

@before_model
async def trim_long_conversations(state, runtime):
    """修剪过长的对话历史"""
    max_messages = 10  # 最多保留10条消息

    if len(state["messages"]) > max_messages:
        print(f"\n✂️  修剪消息历史: {len(state['messages'])} -> {max_messages}")
        # 保留最后的消息
        trimmed = state["messages"][-max_messages:]
        return {"messages": trimmed}

    return None


def message_trimming_example():
    """消息修剪示例"""
    print("\n" + "=" * 50)
    print("消息修剪示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4-plus", temperature=0.5)
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer,
        middleware=[trim_long_conversations],
        system_prompt="你是一个助手"
    )

    # 发送多轮对话
    thread_id = "trim-test"
    for i in range(15):
        print(f"\n第 {i+1} 轮对话")
        result = agent.invoke(
            {"messages": [{"role": "user", "content": f"这是第 {i+1} 条消息"}]},
            {"configurable": {"thread_id": thread_id}}
        )
        print(f"用户: 这是第 {i+1} 条消息")
        print(f"当前消息总数: {len(result['messages'])}")


# ==================== 6. 状态访问中间件 ====================

@before_model
def check_authentication(state, runtime):
    """检查用户是否已认证"""
    is_authenticated = state.get("authenticated", False)

    if not is_authenticated:
        print("\n⚠️  用户未认证")
        # 在实际应用中,这里可以抛出异常或返回错误消息
        # raise ValueError("用户未认证,请先登录")

    return None


class AuthState(AgentState):
    """带认证状态的 Agent 状态"""
    authenticated: bool = False
    user_role: str = "guest"


def state_access_example():
    """状态访问示例"""
    print("\n" + "=" * 50)
    print("状态访问示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4-plus", temperature=0.5)
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[],
        state_schema=AuthState,
        checkpointer=checkpointer,
        middleware=[check_authentication],
        system_prompt="你是一个需要认证的助手"
    )

    # 未认证状态
    print("\n未认证访问:")
    agent.invoke(
        {
            "messages": [{"role": "user", "content": "你好"}],
            "authenticated": False,
            "user_role": "guest"
        },
        {"configurable": {"thread_id": "auth-test"}}
    )

    # 已认证状态
    print("\n已认证访问:")
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "你好"}],
            "authenticated": True,
            "user_role": "admin"
        },
        {"configurable": {"thread_id": "auth-test-2"}}
    )
    print(f"助手: {result['messages'][-1].content}")


# ==================== 7. 查看和重置状态 ====================

def state_management_example():
    """状态管理示例"""
    print("\n" + "=" * 50)
    print("状态管理示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4-plus", temperature=0.5)
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer,
        system_prompt="你是一个助手"
    )

    thread_id = "state-test"

    # 发送几条消息
    print("\n发送消息:")
    for i in range(3):
        agent.invoke(
            {"messages": [{"role": "user", "content": f"消息 {i+1}"}]},
            {"configurable": {"thread_id": thread_id}}
        )
        print(f"  发送: 消息 {i+1}")

    # 查看当前状态
    print("\n查看状态:")
    state = agent.get_state({"configurable": {"thread_id": thread_id}})
    print(f"  消息数量: {len(state.values['messages'])}")
    print(f"  最后一条消息: {state.values['messages'][-1].content[:50]}...")


if __name__ == "__main__":
    try:
        basic_memory_example()
        multi_session_example()
        custom_state_example()
        message_trimming_example()
        state_access_example()
        state_management_example()
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
