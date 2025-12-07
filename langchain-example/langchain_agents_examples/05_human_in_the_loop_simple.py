"""
LangChain 人机协作(Human-in-the-Loop)简化示例
使用 LangGraph interrupt() 实现人工审批流程

这是一个简化但可运行的示例,展示了如何使用 interrupt() 实现 HITL
"""

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
import os

# 设置 API Key
os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 定义带审批的工具 ====================

@tool
def delete_file(file_path: str) -> str:
    """删除文件 - 需要人工审批

    Args:
        file_path: 要删除的文件路径

    Returns:
        删除结果
    """
    # 暂停执行,请求人工批准
    approval = interrupt(
        {
            "type": "approval_request",
            "action": "delete_file",
            "file_path": file_path,
            "message": f"是否批准删除文件: {file_path}? (输入 'yes' 批准, 'no' 拒绝)"
        }
    )

    # 根据用户输入决定是否执行
    if approval and str(approval).lower() in ['yes', 'y', 'true', '是']:
        return f"✅ 已删除文件: {file_path}"
    else:
        return f"❌ 删除操作被拒绝: {file_path}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件 - 需要人工审批

    Args:
        to: 收件人邮箱
        subject: 邮件主题
        body: 邮件正文

    Returns:
        发送结果
    """
    # 显示邮件草稿并请求批准
    approval = interrupt(
        {
            "type": "approval_request",
            "action": "send_email",
            "details": {
                "to": to,
                "subject": subject,
                "body": body
            },
            "message": f"是否批准发送邮件给 {to}? (输入 'yes' 批准, 'no' 拒绝)"
        }
    )

    if approval and str(approval).lower() in ['yes', 'y', 'true', '是']:
        return f"✅ 已发送邮件给: {to}"
    else:
        return f"❌ 邮件发送被拒绝"


@tool
def search_info(query: str) -> str:
    """搜索信息 - 安全操作,不需要审批

    Args:
        query: 搜索查询

    Returns:
        搜索结果
    """
    return f"搜索结果: 关于 '{query}' 的信息..."


# ==================== 示例函数 ====================

def basic_hitl_example():
    """基础人机协作示例"""
    print("=" * 60)
    print("基础人机协作示例")
    print("=" * 60)
    print("\n这个示例展示了如何使用 interrupt() 实现人工审批")
    print("注意: 由于 ChatZhipuAI 的限制,这里使用简化的工具调用模式\n")

    # 直接测试工具
    print("测试 1: 删除文件操作 (需要审批)")
    print("-" * 60)

    try:
        # 创建工具实例
        delete_tool = delete_file

        # 模拟调用 - 第一次会触发 interrupt
        print("尝试删除文件...")
        result = delete_tool.invoke({"file_path": "/tmp/test.txt"})
        print(f"结果: {result}")

    except Exception as e:
        if "interrupt" in str(type(e).__name__).lower():
            print("✓ interrupt 被正确触发,等待人工决策")
        else:
            print(f"错误: {e}")

    print("\n" + "=" * 60)
    print("测试 2: 搜索信息 (不需要审批)")
    print("-" * 60)

    try:
        search_tool = search_info
        result = search_tool.invoke({"query": "Python 教程"})
        print(f"结果: {result}")

    except Exception as e:
        print(f"错误: {e}")


def explain_workflow():
    """解释完整的工作流程"""
    print("\n" + "=" * 60)
    print("Human-in-the-Loop 工作流程说明")
    print("=" * 60)

    workflow = """
    1. 创建带 checkpointer 的 Graph
       checkpointer = MemorySaver()
       graph = graph.compile(checkpointer=checkpointer)

    2. 在工具中使用 interrupt() 暂停
       approval = interrupt({"message": "需要批准吗?"})

    3. 第一次调用会暂停并返回 interrupt 信息
       result = graph.invoke(input, config={"thread_id": "123"})
       # 此时 graph 处于暂停状态

    4. 获取当前状态
       state = graph.get_state(config={"thread_id": "123"})
       # 查看 interrupt 的详细信息

    5. 用 Command.resume() 恢复执行
       from langgraph.types import Command
       result = graph.invoke(
           Command(resume="yes"),  # 用户的批准
           config={"thread_id": "123"}
       )

    6. 工具继续执行并返回最终结果
    """

    print(workflow)

    print("\n完整示例代码:")
    print("-" * 60)

    example_code = '''
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

# 1. 定义带 interrupt 的工具
@tool
def risky_operation(data: str) -> str:
    approval = interrupt({"question": f"批准操作 {data} 吗?"})
    if approval == "yes":
        return f"已执行: {data}"
    return "已取消"

# 2. 构建 graph (使用 StateGraph)
builder = StateGraph(MessagesState)
# ... 添加节点和边 ...
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 3. 首次调用 - 会在 interrupt 处暂停
config = {"configurable": {"thread_id": "thread-1"}}
result = graph.invoke({"messages": [...]}, config)

# 4. 检查状态
state = graph.get_state(config)
print(state.next)  # 查看下一个要执行的节点

# 5. 恢复执行,传入用户决策
result = graph.invoke(Command(resume="yes"), config)
'''

    print(example_code)


def show_alternatives():
    """展示替代方案"""
    print("\n" + "=" * 60)
    print("替代方案")
    print("=" * 60)

    alternatives = """
    如果 ChatZhipuAI 与某些 LangGraph 功能不兼容,可以:

    1. 使用 with_structured_output()
       - 直接在模型层面获取结构化输出
       - 不依赖复杂的 agent 框架

    2. 手动实现工作流
       - 使用简单的条件判断
       - 在关键步骤暂停并等待输入

    3. 使用支持完整功能的模型
       - OpenAI GPT-4
       - Anthropic Claude
       - Google Gemini

    4. 组合使用
       - GLM 用于一般任务
       - GPT-4 用于需要高级功能的任务
    """

    print(alternatives)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LangChain Human-in-the-Loop 示例")
    print("=" * 60)

    try:
        # 运行基础示例
        basic_hitl_example()

        # 解释工作流程
        explain_workflow()

        # 展示替代方案
        show_alternatives()

        print("\n" + "=" * 60)
        print("示例完成!")
        print("=" * 60)
        print("\n提示: 查看 README_HITL.md 了解更多详细信息")

    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
