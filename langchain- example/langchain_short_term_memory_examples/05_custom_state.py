"""
示例5：自定义状态（Custom State）
演示如何扩展 AgentState 添加自定义业务字段
"""

import os
from langchain.agents import create_agent, AgentState
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 自定义状态类 ====================

class UserProfileState(AgentState):
    """
    扩展的 Agent 状态

    除了默认的 messages 字段，还包含：
    - user_id: 用户ID
    - user_name: 用户名
    - preferences: 用户偏好设置
    - session_count: 会话计数
    - last_action: 最后一次操作
    """
    user_id: str = ""
    user_name: str = ""
    preferences: dict = {}
    session_count: int = 0
    last_action: str = ""


# ==================== 工具定义 ====================

@tool
def get_user_info(user_id: str) -> str:
    """根据用户ID获取用户信息"""
    # 模拟用户数据库
    user_db = {
        "user_123": {
            "name": "张三",
            "age": 28,
            "city": "北京",
            "job": "软件工程师"
        },
        "user_456": {
            "name": "李四",
            "age": 32,
            "city": "上海",
            "job": "产品经理"
        },
        "user_789": {
            "name": "王五",
            "age": 25,
            "city": "深圳",
            "job": "UI设计师"
        }
    }

    user = user_db.get(user_id)
    if user:
        return f"用户信息：{user['name']}, {user['age']}岁, 来自{user['city']}, 职业是{user['job']}"
    else:
        return f"未找到用户 {user_id}"


# ==================== 主函数 ====================

def main():
    """自定义状态示例"""
    print("\n" + "=" * 60)
    print("示例5：自定义状态（Custom State）")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)
    checkpointer = MemorySaver()

    # 创建 Agent，使用自定义状态
    agent = create_agent(
        model=model,
        tools=[get_user_info],
        state_schema=UserProfileState,  # 关键：使用自定义状态
        checkpointer=checkpointer,
        system_prompt="你是一个个性化助手，能记住用户信息和偏好"
    )

    config = {"configurable": {"thread_id": "custom-state-test"}}

    print("\n【自定义状态字段】")
    print("- user_id: 用户ID")
    print("- user_name: 用户名")
    print("- preferences: 用户偏好（主题、语言等）")
    print("- session_count: 会话计数")
    print("- last_action: 最后操作")

    # ========== 第一次会话 ==========
    print(f"\n{'='*60}")
    print("第1次会话 - 设置用户信息")
    print(f"{'='*60}")

    result1 = agent.invoke(
        {
            "messages": [{"role": "user", "content": "你好，我是新用户"}],
            "user_id": "user_123",
            "user_name": "张三",
            "preferences": {
                "theme": "dark",
                "language": "zh-CN",
                "notification": True
            },
            "session_count": 1,
            "last_action": "login"
        },
        config
    )

    print(f"🤖 助手: {result1['messages'][-1].content}")

    print(f"\n📊 状态信息:")
    print(f"   用户ID: {result1.get('user_id', 'N/A')}")
    print(f"   用户名: {result1.get('user_name', 'N/A')}")
    print(f"   偏好设置: {result1.get('preferences', {})}")
    print(f"   会话计数: {result1.get('session_count', 0)}")
    print(f"   最后操作: {result1.get('last_action', 'N/A')}")

    # ========== 第二次会话 - 状态被保留 ==========
    print(f"\n{'='*60}")
    print("第2次会话 - 状态自动保留")
    print(f"{'='*60}")

    result2 = agent.invoke(
        {
            "messages": [{"role": "user", "content": "查询我的基本信息"}],
            "session_count": 2,
            "last_action": "query_info"
        },
        config
    )

    print(f"🤖 助手: {result2['messages'][-1].content}")

    print(f"\n📊 状态信息:")
    print(f"   用户ID: {result2.get('user_id', 'N/A')} (自动保留)")
    print(f"   用户名: {result2.get('user_name', 'N/A')} (自动保留)")
    print(f"   会话计数: {result2.get('session_count', 0)} (已更新)")
    print(f"   最后操作: {result2.get('last_action', 'N/A')} (已更新)")

    # ========== 第三次会话 - 修改偏好 ==========
    print(f"\n{'='*60}")
    print("第3次会话 - 修改用户偏好")
    print(f"{'='*60}")

    result3 = agent.invoke(
        {
            "messages": [{"role": "user", "content": "我想切换到浅色主题"}],
            "preferences": {
                "theme": "light",  # 修改主题
                "language": "zh-CN",
                "notification": True
            },
            "session_count": 3,
            "last_action": "update_theme"
        },
        config
    )

    print(f"🤖 助手: {result3['messages'][-1].content}")

    print(f"\n📊 状态信息:")
    print(f"   偏好设置: {result3.get('preferences', {})} (已更新)")
    print(f"   会话计数: {result3.get('session_count', 0)}")

    # ========== 查询完整状态 ==========
    print(f"\n{'='*60}")
    print("【查询完整状态】")
    print(f"{'='*60}")

    state = agent.get_state(config)
    print(f"\n完整状态:")
    print(f"   消息数: {len(state.values.get('messages', []))}")
    print(f"   用户ID: {state.values.get('user_id', 'N/A')}")
    print(f"   用户名: {state.values.get('user_name', 'N/A')}")
    print(f"   偏好: {state.values.get('preferences', {})}")
    print(f"   会话数: {state.values.get('session_count', 0)}")
    print(f"   最后操作: {state.values.get('last_action', 'N/A')}")


def example_multiple_users():
    """多用户使用自定义状态"""
    print("\n" + "=" * 60)
    print("【高级示例】多用户自定义状态")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[],
        state_schema=UserProfileState,
        checkpointer=checkpointer,
        system_prompt="你是一个助手"
    )

    # 用户A
    config_a = {"configurable": {"thread_id": "user-A"}}
    agent.invoke({
        "messages": [{"role": "user", "content": "你好"}],
        "user_id": "user_123",
        "user_name": "张三",
        "preferences": {"theme": "dark"}
    }, config_a)

    # 用户B
    config_b = {"configurable": {"thread_id": "user-B"}}
    agent.invoke({
        "messages": [{"role": "user", "content": "你好"}],
        "user_id": "user_456",
        "user_name": "李四",
        "preferences": {"theme": "light"}
    }, config_b)

    print("\n✅ 每个用户都有独立的自定义状态")


if __name__ == "__main__":
    try:
        main()
        # example_multiple_users()  # 可选：运行多用户示例
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
