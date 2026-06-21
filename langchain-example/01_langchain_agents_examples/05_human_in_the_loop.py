"""
LangChain Agents 人机协作(Human-in-the-Loop)示例
演示需要人工审批的操作、决策流程等
使用 GLM 模型 + LangGraph interrupt()

⚠️ 注意: ChatZhipuAI 与 create_agent 不兼容
本文件使用 LangGraph 的 StateGraph 和 interrupt() 实现 Human-in-the-Loop
"""

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from typing import Literal
import os

# 设置 API Key
os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 1. 定义敏感操作工具 ====================

@tool
def delete_file(path: str) -> str:
    """删除文件

    Args:
        path: 文件路径

    Returns:
        删除结果
    """
    # 模拟删除操作
    return f"已删除文件: {path}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件

    Args:
        to: 收件人
        subject: 邮件主题
        body: 邮件正文

    Returns:
        发送结果
    """
    # 模拟发送邮件
    return f"已发送邮件至 {to}, 主题: {subject}"


@tool
def transfer_money(from_account: str, to_account: str, amount: float) -> str:
    """转账

    Args:
        from_account: 源账户
        to_account: 目标账户
        amount: 金额

    Returns:
        转账结果
    """
    # 模拟转账操作
    return f"已从 {from_account} 转账 {amount} 元到 {to_account}"


@tool
def search_info(query: str) -> str:
    """搜索信息(无需审批的安全操作)

    Args:
        query: 搜索查询

    Returns:
        搜索结果
    """
    return f"搜索结果: 关于 '{query}' 的信息..."


# ==================== 2. 基础人机协作示例 ====================

def basic_hitl_example():
    """基础人机协作示例"""
    print("=" * 50)
    print("基础人机协作示例 - 文件删除审批")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4-plus", temperature=0.5)

    # HITL 需要 checkpointer
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[delete_file, search_info],
        middleware=[
            human_in_the_loop_middleware(
                interrupt_on={
                    "delete_file": True,  # 删除文件需要审批
                }
            )
        ],
        checkpointer=checkpointer,
        system_prompt="你是一个文件管理助手,可以搜索和删除文件"
    )

    thread_id = "hitl-test-1"

    # 1. 启动 Agent
    print("\n用户请求: 删除 report.pdf 文件")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "删除 report.pdf 文件"}]},
        {"configurable": {"thread_id": thread_id}}
    )

    # 2. 检查是否有中断请求
    state = agent.get_state({"configurable": {"thread_id": thread_id}})
    if "hitl_request" in state.values:
        interrupt_request = state.values["hitl_request"]
        print(f"\n⚠️  检测到需要审批的操作:")

        for action in interrupt_request.action_requests:
            print(f"  工具: {action.tool_call['name']}")
            print(f"  参数: {action.tool_call['args']}")

        # 3. 模拟人工决策 - 批准
        print("\n人工决策: 批准操作")
        response = HITLResponse(
            decisions=[Decision(type="approve")]
        )

        # 4. 恢复执行
        result = agent.invoke(
            Command(resume=response),
            {"configurable": {"thread_id": thread_id}}
        )

        print(f"\n执行结果: {result['messages'][-1].content}")
    else:
        print("\n无需审批,直接执行")
        print(f"结果: {result['messages'][-1].content}")


# ==================== 3. 编辑操作示例 ====================

def edit_decision_example():
    """编辑决策示例"""
    print("\n" + "=" * 50)
    print("编辑决策示例 - 修改邮件内容")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4-plus", temperature=0.5)
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[send_email],
        middleware=[
            human_in_the_loop_middleware(
                interrupt_on={
                    "send_email": True,
                }
            )
        ],
        checkpointer=checkpointer,
        system_prompt="你是一个邮件助手"
    )

    thread_id = "hitl-test-2"

    # 1. 启动 Agent
    print("\n用户请求: 发送项目报告邮件给老板")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "发送项目报告邮件给 boss@company.com"}]},
        {"configurable": {"thread_id": thread_id}}
    )

    # 2. 获取中断请求
    state = agent.get_state({"configurable": {"thread_id": thread_id}})
    if "hitl_request" in state.values:
        interrupt_request = state.values["hitl_request"]
        action = interrupt_request.action_requests[0]

        print(f"\n📧 邮件草稿:")
        print(f"  收件人: {action.tool_call['args']['to']}")
        print(f"  主题: {action.tool_call['args']['subject']}")
        print(f"  正文: {action.tool_call['args']['body']}")

        # 3. 人工编辑 - 修改收件人和主题
        print("\n✏️  人工编辑: 修改收件人和主题")
        response = HITLResponse(
            decisions=[
                Decision(
                    type="edit",
                    tool_call={
                        "name": "send_email",
                        "args": {
                            "to": "manager@company.com",  # 修改收件人
                            "subject": "【重要】项目进度报告",  # 修改主题
                            "body": action.tool_call['args']['body']
                        }
                    }
                )
            ]
        )

        # 4. 恢复执行
        result = agent.invoke(
            Command(resume=response),
            {"configurable": {"thread_id": thread_id}}
        )

        print(f"\n执行结果: {result['messages'][-1].content}")


# ==================== 4. 拒绝操作示例 ====================

def reject_decision_example():
    """拒绝决策示例"""
    print("\n" + "=" * 50)
    print("拒绝决策示例 - 拒绝转账")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4-plus", temperature=0.5)
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[transfer_money],
        middleware=[
            human_in_the_loop_middleware(
                interrupt_on={
                    "transfer_money": True,
                }
            )
        ],
        checkpointer=checkpointer,
        system_prompt="你是一个银行助手,可以帮助转账"
    )

    thread_id = "hitl-test-3"

    # 1. 启动 Agent
    print("\n用户请求: 转账 10000 元")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "从账户A转 10000 元到账户B"}]},
        {"configurable": {"thread_id": thread_id}}
    )

    # 2. 获取中断请求
    state = agent.get_state({"configurable": {"thread_id": thread_id}})
    if "hitl_request" in state.values:
        interrupt_request = state.values["hitl_request"]
        action = interrupt_request.action_requests[0]

        print(f"\n💰 转账请求:")
        print(f"  源账户: {action.tool_call['args']['from_account']}")
        print(f"  目标账户: {action.tool_call['args']['to_account']}")
        print(f"  金额: {action.tool_call['args']['amount']} 元")

        # 3. 人工决策 - 拒绝
        print("\n❌ 人工决策: 拒绝转账 (金额过大)")
        response = HITLResponse(
            decisions=[
                Decision(
                    type="reject",
                    explanation="金额过大,需要额外的审批流程。请联系财务部门。"
                )
            ]
        )

        # 4. 恢复执行
        result = agent.invoke(
            Command(resume=response),
            {"configurable": {"thread_id": thread_id}}
        )

        print(f"\n执行结果: {result['messages'][-1].content}")


# ==================== 5. 多工具调用审批 ====================

def multi_tool_approval_example():
    """多工具调用审批示例"""
    print("\n" + "=" * 50)
    print("多工具调用审批示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4-plus", temperature=0.5)
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[send_email, delete_file, search_info],
        middleware=[
            human_in_the_loop_middleware(
                interrupt_on={
                    "send_email": True,
                    "delete_file": True,
                }
            )
        ],
        checkpointer=checkpointer,
        system_prompt="你是一个办公助手"
    )

    thread_id = "hitl-test-4"

    # 1. 启动 Agent - 可能触发多个工具调用
    print("\n用户请求: 发送报告邮件并删除旧文件")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "发送周报给 manager@company.com,然后删除 old_report.pdf"}]},
        {"configurable": {"thread_id": thread_id}}
    )

    # 2. 获取中断请求
    state = agent.get_state({"configurable": {"thread_id": thread_id}})
    if "hitl_request" in state.values:
        interrupt_request = state.values["hitl_request"]

        print(f"\n检测到 {len(interrupt_request.action_requests)} 个需要审批的操作:")

        # 显示所有待审批操作
        for i, action in enumerate(interrupt_request.action_requests, 1):
            print(f"\n操作 {i}:")
            print(f"  工具: {action.tool_call['name']}")
            print(f"  参数: {action.tool_call['args']}")

        # 3. 为每个操作提供决策
        print("\n人工决策:")
        print("  操作1 (send_email): 批准")
        print("  操作2 (delete_file): 拒绝")

        response = HITLResponse(
            decisions=[
                Decision(type="approve"),  # 批准发送邮件
                Decision(
                    type="reject",
                    explanation="请先备份文件再删除"
                )  # 拒绝删除文件
            ]
        )

        # 4. 恢复执行
        result = agent.invoke(
            Command(resume=response),
            {"configurable": {"thread_id": thread_id}}
        )

        print(f"\n执行结果: {result['messages'][-1].content}")


# ==================== 6. 选择性审批 ====================

def selective_approval_example():
    """选择性审批示例 - 只审批特定工具"""
    print("\n" + "=" * 50)
    print("选择性审批示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4-plus", temperature=0.5)
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[search_info, delete_file, send_email],
        middleware=[
            human_in_the_loop_middleware(
                interrupt_on={
                    "delete_file": True,  # 只有删除文件需要审批
                    # send_email 和 search_info 不需要审批
                }
            )
        ],
        checkpointer=checkpointer,
        system_prompt="你是一个智能助手"
    )

    thread_id = "hitl-test-5"

    # 测试1: 搜索信息 (不需要审批)
    print("\n测试1: 搜索信息 (不需要审批)")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "搜索 Python 教程"}]},
        {"configurable": {"thread_id": thread_id + "-1"}}
    )
    print(f"结果: {result['messages'][-1].content}")

    # 测试2: 删除文件 (需要审批)
    print("\n测试2: 删除文件 (需要审批)")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "删除 test.txt"}]},
        {"configurable": {"thread_id": thread_id + "-2"}}
    )

    state = agent.get_state({"configurable": {"thread_id": thread_id + "-2"}})
    if "hitl_request" in state.values:
        print("⚠️  需要人工审批")
        # 批准操作
        response = HITLResponse(decisions=[Decision(type="approve")])
        result = agent.invoke(
            Command(resume=response),
            {"configurable": {"thread_id": thread_id + "-2"}}
        )
        print(f"批准后结果: {result['messages'][-1].content}")


# ==================== 7. 自定义审批决策 ====================

def custom_approval_logic():
    """自定义审批逻辑示例"""
    print("\n" + "=" * 50)
    print("自定义审批逻辑示例")
    print("=" * 50)

    model = ChatZhipuAI(model="glm-4-plus", temperature=0.5)
    checkpointer = MemorySaver()

    agent = create_agent(
        model=model,
        tools=[send_email],
        middleware=[
            human_in_the_loop_middleware(
                interrupt_on={
                    "send_email": {
                        "allowed_decisions": ["approve", "reject"]  # 不允许编辑
                    }
                }
            )
        ],
        checkpointer=checkpointer,
        system_prompt="你是一个邮件助手"
    )

    thread_id = "hitl-test-6"

    print("\n用户请求: 发送邮件")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "发送通知邮件给 team@company.com"}]},
        {"configurable": {"thread_id": thread_id}}
    )

    state = agent.get_state({"configurable": {"thread_id": thread_id}})
    if "hitl_request" in state.values:
        print("\n✅ 只能批准或拒绝,不能编辑")

        # 批准
        response = HITLResponse(decisions=[Decision(type="approve")])
        result = agent.invoke(
            Command(resume=response),
            {"configurable": {"thread_id": thread_id}}
        )
        print(f"结果: {result['messages'][-1].content}")


if __name__ == "__main__":
    try:
        basic_hitl_example()
        # edit_decision_example()
        # reject_decision_example()
        # multi_tool_approval_example()
        # selective_approval_example()
        # custom_approval_logic()

        print("\n" + "=" * 50)
        print("所有人机协作示例完成!")
        print("=" * 50)
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
        import traceback
        traceback.print_exc()
