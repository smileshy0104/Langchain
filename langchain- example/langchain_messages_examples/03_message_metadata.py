"""
LangChain Messages - 消息元数据示例
演示 tool_calls, usage_metadata, response_metadata, additional_kwargs 等
使用 GLM 模型
"""

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage
)
from langchain_core.tools import tool
from typing import Dict, Any
import json
import os

os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 定义工具 ====================

@tool
def get_weather(city: str) -> str:
    """获取天气信息

    Args:
        city: 城市名称

    Returns:
        天气信息字符串
    """
    return f"{city}今天晴朗,温度 22°C"


@tool
def calculate(expression: str) -> float:
    """计算数学表达式

    Args:
        expression: 数学表达式

    Returns:
        计算结果
    """
    return eval(expression)


# ==================== 1. Tool Calls 基础 ====================

def tool_calls_basic():
    """Tool Calls 基础示例"""
    print("=" * 60)
    print("Tool Calls 基础示例")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6", temperature=0)
    model_with_tools = model.bind_tools([get_weather])

    response = model_with_tools.invoke([
        HumanMessage(content="北京天气怎么样？")
    ])

    print(f"\nAI 响应类型: {type(response).__name__}")
    print(f"响应内容: {response.content}")

    if response.tool_calls:
        print(f"\nTool Calls 数量: {len(response.tool_calls)}")
        for i, tool_call in enumerate(response.tool_calls, 1):
            print(f"\nTool Call {i}:")
            print(f"  名称: {tool_call['name']}")
            print(f"  参数: {tool_call['args']}")
            print(f"  ID: {tool_call.get('id', 'N/A')}")
    else:
        print("\n未调用工具")


# ==================== 2. Tool Call 完整流程 ====================

def tool_call_complete_flow():
    """Tool Call 完整流程示例"""
    print("\n" + "=" * 60)
    print("Tool Call 完整流程示例")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6", temperature=0)
    model_with_tools = model.bind_tools([get_weather])

    # 第一步: 用户请求
    messages = [HumanMessage(content="上海天气如何？")]
    print("\n第一步: 用户请求")
    print(f"  {messages[0].content}")

    # 第二步: AI 决定调用工具
    ai_response = model_with_tools.invoke(messages)
    messages.append(ai_response)

    print("\n第二步: AI 决定调用工具")
    if ai_response.tool_calls:
        tool_call = ai_response.tool_calls[0]
        print(f"  工具: {tool_call['name']}")
        print(f"  参数: {tool_call['args']}")

        # 第三步: 执行工具
        tool_result = get_weather.invoke(tool_call['args'])

        print("\n第三步: 执行工具")
        print(f"  结果: {tool_result}")

        # 第四步: 返回工具结果
        tool_message = ToolMessage(
            content=tool_result,
            tool_call_id=tool_call.get('id', 'call_001'),
            name=tool_call['name']
        )
        messages.append(tool_message)

        print("\n第四步: 返回工具结果")
        print(f"  Tool Message: {tool_message.content}")

        # 第五步: AI 生成最终回复
        final_response = model.invoke(messages)

        print("\n第五步: AI 最终回复")
        print(f"  {final_response.content}")


# ==================== 3. Usage Metadata ====================

def usage_metadata_example():
    """Usage Metadata 示例"""
    print("\n" + "=" * 60)
    print("Usage Metadata 示例")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)

    response = model.invoke([
        HumanMessage(content="写一首关于春天的短诗")
    ])

    print(f"\nAI 回复:\n{response.content}")

    # 检查 usage metadata
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        print("\nUsage Metadata:")
        usage = response.usage_metadata
        print(f"  输入 Tokens: {usage.get('input_tokens', 'N/A')}")
        print(f"  输出 Tokens: {usage.get('output_tokens', 'N/A')}")
        print(f"  总 Tokens: {usage.get('total_tokens', 'N/A')}")
    else:
        print("\n未提供 Usage Metadata")


# ==================== 4. Response Metadata ====================

def response_metadata_example():
    """Response Metadata 示例"""
    print("\n" + "=" * 60)
    print("Response Metadata 示例")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.7)

    response = model.invoke([
        HumanMessage(content="你好")
    ])

    print(f"\nAI 回复: {response.content}")

    # 检查 response metadata
    if hasattr(response, 'response_metadata') and response.response_metadata:
        print("\nResponse Metadata:")
        metadata = response.response_metadata
        print(f"  模型: {metadata}")
        # 常见的元数据字段
        for key in ['model_name', 'finish_reason', 'system_fingerprint']:
            if key in metadata:
                print(f"  {key}: {metadata[key]}")

        # 显示所有元数据
        print("\n完整 Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    else:
        print("\n未提供 Response Metadata")


# ==================== 5. Additional Kwargs ====================

def additional_kwargs_example():
    """Additional Kwargs 示例"""
    print("\n" + "=" * 60)
    print("Additional Kwargs 示例")
    print("=" * 60)

    # 创建带有 additional_kwargs 的消息
    message = HumanMessage(
        content="测试消息",
        additional_kwargs={
            "user_id": "user_123",
            "session_id": "session_456",
            "metadata": {
                "source": "web",
                "timestamp": "2025-01-01T12:00:00"
            }
        }
    )

    print("\nAdditional Kwargs:")
    print(json.dumps(message.additional_kwargs, indent=2, ensure_ascii=False))

    # AI 响应也可能包含 additional_kwargs
    model = ChatZhipuAI(model="glm-4.6")
    response = model.invoke([message])
    print(f"\nAI 回复: {response}")
    print(f"\nAI 回复: {response.content}")

    if response.additional_kwargs:
        print("\nAI 响应的 Additional Kwargs:")
        print(json.dumps(response.additional_kwargs, indent=2, ensure_ascii=False))


# ==================== 6. Message ID ====================

def message_id_example():
    """Message ID 示例"""
    print("\n" + "=" * 60)
    print("Message ID 示例")
    print("=" * 60)

    # 手动指定 ID
    msg1 = HumanMessage(content="消息1", id="msg-001")
    msg2 = HumanMessage(content="消息2", id="msg-002")

    print("\n手动指定 ID:")
    print(f"  消息1 ID: {msg1.id}")
    print(f"  消息2 ID: {msg2.id}")

    # 自动生成 ID
    msg3 = HumanMessage(content="消息3")
    msg4 = HumanMessage(content="消息4")

    print("\n自动生成 ID:")
    print(f"  消息3 ID: {msg3.id}")
    print(f"  消息4 ID: {msg4.id}")

    # ID 用于消息跟踪和引用
    print("\n使用场景:")
    print("  - 消息跟踪和调试")
    print("  - 引用特定消息")
    print("  - 删除特定消息 (RemoveMessage)")


# ==================== 7. Message Name ====================

def message_name_example():
    """Message Name 示例"""
    print("\n" + "=" * 60)
    print("Message Name 示例")
    print("=" * 60)

    # 多用户场景
    messages = [
        HumanMessage(content="大家好！", name="张三"),
        HumanMessage(content="你好张三！", name="李四"),
        HumanMessage(content="大家好！", name="王五"),
    ]

    print("\n多用户对话:")
    for msg in messages:
        print(f"  {msg.name}: {msg.content}")

    # 在 AI 响应中使用
    ai_msg = AIMessage(
        content="大家好！我是 AI 助手",
        name="assistant"
    )

    print(f"\n  {ai_msg.name}: {ai_msg.content}")


# ==================== 8. ToolMessage 详解 ====================

def tool_message_details():
    """ToolMessage 详解"""
    print("\n" + "=" * 60)
    print("ToolMessage 详解")
    print("=" * 60)

    # 创建 ToolMessage
    tool_msg = ToolMessage(
        content="天气数据: 晴朗, 22°C",
        tool_call_id="call_abc123",
        name="get_weather",
        additional_kwargs={
            "execution_time": "0.5s",
            "cache_hit": False
        }
    )

    print("\nToolMessage 属性:")
    print(f"  content: {tool_msg.content}")
    print(f"  tool_call_id: {tool_msg.tool_call_id}")
    print(f"  name: {tool_msg.name}")
    print(f"  type: {tool_msg.type}")

    print("\nAdditional Kwargs:")
    print(json.dumps(tool_msg.additional_kwargs, indent=2))

    # ToolMessage 在对话流中的位置
    print("\n对话流中的位置:")
    print("  1. HumanMessage: 用户请求")
    print("  2. AIMessage (with tool_calls): AI 决定调用工具")
    print("  3. ToolMessage: 工具执行结果")
    print("  4. AIMessage: AI 最终回复")


# ==================== 9. 自定义元数据 ====================

def custom_metadata_example():
    """自定义元数据示例"""
    print("\n" + "=" * 60)
    print("自定义元数据示例")
    print("=" * 60)

    def create_tracked_message(content: str, user_info: Dict[str, Any]) -> HumanMessage:
        """创建带跟踪信息的消息"""
        return HumanMessage(
            content=content,
            name=user_info.get("name"),
            additional_kwargs={
                "user_id": user_info.get("id"),
                "department": user_info.get("department"),
                "priority": user_info.get("priority", "normal"),
                "tags": user_info.get("tags", [])
            }
        )

    # 创建带元数据的消息
    user_info = {
        "id": "emp_001",
        "name": "张三",
        "department": "研发部",
        "priority": "high",
        "tags": ["urgent", "technical"]
    }

    message = create_tracked_message("需要技术支持", user_info)

    print("\n消息元数据:")
    print(f"  用户: {message.name}")
    print(f"  内容: {message.content}")
    print(f"\n自定义字段:")
    for key, value in message.additional_kwargs.items():
        print(f"  {key}: {value}")


# ==================== 10. 元数据过滤和查询 ====================

def metadata_filtering():
    """元数据过滤和查询示例"""
    print("\n" + "=" * 60)
    print("元数据过滤和查询示例")
    print("=" * 60)

    # 创建消息列表
    messages = [
        HumanMessage(
            content="紧急问题",
            name="张三",
            additional_kwargs={"priority": "high", "department": "研发"}
        ),
        HumanMessage(
            content="一般咨询",
            name="李四",
            additional_kwargs={"priority": "normal", "department": "销售"}
        ),
        HumanMessage(
            content="重要事项",
            name="王五",
            additional_kwargs={"priority": "high", "department": "研发"}
        ),
    ]

    print("\n所有消息:")
    for i, msg in enumerate(messages, 1):
        kwargs = msg.additional_kwargs
        print(f"  {i}. {msg.name} ({kwargs['department']}): {msg.content} [{kwargs['priority']}]")

    # 过滤高优先级消息
    high_priority = [
        msg for msg in messages
        if msg.additional_kwargs.get("priority") == "high"
    ]

    print(f"\n高优先级消息 ({len(high_priority)}):")
    for msg in high_priority:
        print(f"  - {msg.name}: {msg.content}")

    # 过滤研发部消息
    rd_messages = [
        msg for msg in messages
        if msg.additional_kwargs.get("department") == "研发"
    ]

    print(f"\n研发部消息 ({len(rd_messages)}):")
    for msg in rd_messages:
        print(f"  - {msg.name}: {msg.content}")


# ==================== 11. 元数据继承和传播 ====================

def metadata_propagation():
    """元数据继承和传播示例"""
    print("\n" + "=" * 60)
    print("元数据继承和传播示例")
    print("=" * 60)

    # 原始消息
    original = HumanMessage(
        content="原始消息",
        name="用户A",
        id="msg-001",
        additional_kwargs={"session_id": "session_123", "lang": "zh"}
    )

    # 创建副本并添加新元数据
    modified = original.model_copy(
        update={
            "content": "修改后的消息",
            "additional_kwargs": {
                **original.additional_kwargs,
                "modified": True,
                "timestamp": "2025-01-01"
            }
        }
    )

    print("\n原始消息:")
    print(f"  content: {original.content}")
    print(f"  additional_kwargs: {original.additional_kwargs}")

    print("\n修改后消息:")
    print(f"  content: {modified.content}")
    print(f"  additional_kwargs: {modified.additional_kwargs}")


# ==================== 12. 元数据最佳实践 ====================

def metadata_best_practices():
    """元数据最佳实践"""
    print("\n" + "=" * 60)
    print("元数据最佳实践")
    print("=" * 60)

    print("\n1. 使用标准字段")
    print("   - name: 消息发送者名称")
    print("   - id: 唯一标识符")
    print("   - type: 消息类型 (自动设置)")

    print("\n2. Additional Kwargs 使用场景")
    print("   - 用户信息: user_id, session_id")
    print("   - 业务元数据: priority, category, tags")
    print("   - 技术元数据: source, version, trace_id")

    print("\n3. Tool Calls 注意事项")
    print("   - 检查 tool_calls 是否存在")
    print("   - 保存 tool_call_id 用于关联")
    print("   - 正确构建 ToolMessage")

    print("\n4. 性能考虑")
    print("   - 不要在元数据中存储大量数据")
    print("   - 使用引用而非嵌入完整数据")
    print("   - 定期清理不需要的元数据")

    print("\n5. 隐私和安全")
    print("   - 不要在元数据中存储敏感信息")
    print("   - 注意日志记录的元数据内容")
    print("   - 遵守数据保护法规")


if __name__ == "__main__":
    try:
        # tool_calls_basic()
        # tool_call_complete_flow()
        # usage_metadata_example()
        # response_metadata_example()
        # additional_kwargs_example()
        # message_id_example()
        # message_name_example()
        # tool_message_details()
        # custom_metadata_example()
        # metadata_filtering()
        # metadata_propagation()
        metadata_best_practices()

        print("\n" + "=" * 60)
        print("所有消息元数据示例完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
        import traceback
        traceback.print_exc()
