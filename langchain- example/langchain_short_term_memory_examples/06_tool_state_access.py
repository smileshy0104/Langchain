"""
示例6：工具中读写状态
演示如何在工具函数中访问和修改会话状态
"""

import os
from langchain.agents import create_agent, AgentState
from langchain_community.chat_models import ChatZhipuAI
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 自定义状态 ====================

class UserContextState(AgentState):
    """包含用户上下文的状态"""
    user_id: str = ""
    user_name: str = ""
    user_points: int = 0
    last_purchase: str = ""


# ==================== 工具：读取状态 ====================

@tool
def get_user_profile(runtime: ToolRuntime) -> str:
    """
    查询用户资料（从状态中读取 user_id）

    这个工具展示了如何在工具中读取状态
    """
    # 从状态中读取 user_id
    user_id = runtime.state.get("user_id", "unknown")
    user_name = runtime.state.get("user_name", "未知用户")
    user_points = runtime.state.get("user_points", 0)

    # 模拟数据库查询
    user_db = {
        "user_123": {"email": "zhangsan@example.com", "vip_level": "Gold"},
        "user_456": {"email": "lisi@example.com", "vip_level": "Silver"},
    }

    user_detail = user_db.get(user_id, {"email": "N/A", "vip_level": "Normal"})

    return f"""用户资料:
- ID: {user_id}
- 姓名: {user_name}
- 积分: {user_points}
- 邮箱: {user_detail['email']}
- VIP等级: {user_detail['vip_level']}"""


# ==================== 工具：写入状态 ====================

@tool
def update_user_info(
    user_id: str,
    runtime: ToolRuntime
) -> Command:
    """
    更新用户信息到状态

    这个工具展示了如何在工具中写入状态

    Args:
        user_id: 用户ID
    """
    # 模拟从数据库查询用户信息
    user_db = {
        "user_123": {"name": "张三", "points": 1500},
        "user_456": {"name": "李四", "points": 800},
    }

    user = user_db.get(user_id, {"name": "未知", "points": 0})

    # 返回 Command 更新状态
    return Command(update={
        "user_id": user_id,
        "user_name": user['name'],
        "user_points": user['points'],
        "messages": [
            ToolMessage(
                f"✅ 已更新用户信息: {user['name']} ({user['points']} 积分)",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })


@tool
def add_points(
    points: int,
    runtime: ToolRuntime
) -> Command:
    """
    增加用户积分

    Args:
        points: 要增加的积分数
    """
    current_points = runtime.state.get("user_points", 0)
    new_points = current_points + points

    return Command(update={
        "user_points": new_points,
        "messages": [
            ToolMessage(
                f"✅ 已添加 {points} 积分，当前总积分: {new_points}",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })


@tool
def record_purchase(
    item_name: str,
    runtime: ToolRuntime
) -> Command:
    """
    记录购买记录

    Args:
        item_name: 商品名称
    """
    return Command(update={
        "last_purchase": item_name,
        "messages": [
            ToolMessage(
                f"✅ 已记录购买: {item_name}",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })


# ==================== 主函数 ====================

def main():
    """工具状态访问示例"""
    print("\n" + "=" * 60)
    print("示例6：工具中读写状态")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)
    checkpointer = MemorySaver()

    # 创建 Agent
    agent = create_agent(
        model=model,
        tools=[
            get_user_profile,    # 读取状态
            update_user_info,    # 写入状态
            add_points,          # 修改状态
            record_purchase      # 记录状态
        ],
        state_schema=UserContextState,
        checkpointer=checkpointer,
        system_prompt="你是一个购物助手，可以查询和更新用户信息"
    )

    config = {"configurable": {"thread_id": "tool-state-test"}}

    # ========== 示例1：初始化用户信息 ==========
    print(f"\n{'='*60}")
    print("【示例1】工具写入状态 - 初始化用户")
    print(f"{'='*60}")

    print("\n👤 用户: 初始化用户ID为 user_123 的账户信息")
    # 调用update_user_info工具
    result1 = agent.invoke(
        {"messages": [{"role": "user", "content": "帮我初始化用户ID为 user_123 的账户信息"}]},
        config=config
    )
    print(f"🤖 助手: {result1['messages'][-1].content}")

    # ========== 示例2：查询用户信息（从状态读取） ==========
    print(f"\n{'='*60}")
    print("【示例2】工具读取状态 - 查询资料")
    print(f"{'='*60}")

    print("\n👤 用户: 查询我的资料")
    # 调用get_user_profile工具
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "查询我的详细资料"}]},
        config=config
    )
    print(f"🤖 助手: {result2['messages'][-1].content}")

    # ========== 示例3：增加积分 ==========
    print(f"\n{'='*60}")
    print("【示例3】工具修改状态 - 增加积分")
    print(f"{'='*60}")

    print("\n👤 用户: 给我增加200积分")
    # 调用add_points工具
    result3 = agent.invoke(
        {"messages": [{"role": "user", "content": "给我增加200积分"}]},
        config=config
    )
    print(f"🤖 助手: {result3['messages'][-1].content}")

    # ========== 示例4：记录购买 ==========
    print(f"\n{'='*60}")
    print("【示例4】工具记录状态 - 购买记录")
    print(f"{'='*60}")

    print("\n👤 用户: 我购买了一台iPhone 15")
    # 调用record_purchase工具
    result4 = agent.invoke(
        {"messages": [{"role": "user", "content": "我购买了一台iPhone 15"}]},
        config=config
    )
    print(f"🤖 助手: {result4['messages'][-1].content}")

    # ========== 查看最终状态 ==========
    print(f"\n{'='*60}")
    print("【最终状态】")
    print(f"{'='*60}")

    state = agent.get_state(config)
    print(f"\n📊 完整状态信息:")
    print(f"   用户ID: {state.values.get('user_id', 'N/A')}")
    print(f"   用户名: {state.values.get('user_name', 'N/A')}")
    print(f"   积分: {state.values.get('user_points', 0)}")
    print(f"   最后购买: {state.values.get('last_purchase', 'N/A')}")

    # ========== 验证状态持久化 ==========
    print(f"\n{'='*60}")
    print("【验证状态持久化】")
    print(f"{'='*60}")

    print("\n👤 用户: 我的积分是多少？上次买了什么？")
    result5 = agent.invoke(
        {"messages": [{"role": "user", "content": "我的积分是多少？上次买了什么？"}]},
        config=config
    )
    print(f"🤖 助手: {result5['messages'][-1].content}")

    print("\n💡 说明：工具修改的状态会被持久化，下次查询时仍然有效")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
