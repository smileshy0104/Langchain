"""
LangChain Messages - 基础消息类型示例
演示 HumanMessage, AIMessage, SystemMessage 的基本用法
使用 GLM 模型
"""

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage
)
from typing import List
import os

os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 1. 基础 HumanMessage ====================

def basic_human_message():
    """基础 HumanMessage 示例"""
    print("=" * 60)
    print("基础 HumanMessage 示例")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.7)

    # 方式 1: 直接传入字符串内容
    message1 = HumanMessage(content="你好，请介绍一下自己")

    # 方式 2: 使用命名参数
    message2 = HumanMessage(
        content="什么是机器学习？",
        name="user"  # 可选: 指定消息发送者名称
    )

    # 发送消息
    response = model.invoke([message1])

    print(f"\n发送消息: {message1.content}")
    print(f"AI 回复: {response.content}")

    print(f"\n消息类型: {message1.type}")
    print(f"消息角色: {type(message1).__name__}")


# ==================== 2. 基础 AIMessage ====================

def basic_ai_message():
    """基础 AIMessage 示例"""
    print("\n" + "=" * 60)
    print("基础 AIMessage 示例")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6")

    # 创建对话历史
    messages = [
        HumanMessage(content="1+1等于多少？"),
        AIMessage(content="1+1等于2"),
        HumanMessage(content="那2+2呢？")
    ]

    response = model.invoke(messages)

    print("\n对话历史:")
    for msg in messages:
        print(f"  {type(msg).__name__}: {msg.content}")

    print(f"\n新的 AI 回复: {response.content}")
    print(f"回复类型: {type(response).__name__}")


# ==================== 3. SystemMessage 系统提示 ====================

def system_message_example():
    """SystemMessage 系统提示示例"""
    print("\n" + "=" * 60)
    print("SystemMessage 系统提示示例")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.3)

    # 使用系统消息设定角色
    messages = [
        SystemMessage(content="你是一个专业的 Python 编程助手，回答要简洁、准确。"),
        HumanMessage(content="如何读取文件？")
    ]

    response = model.invoke(messages)

    print("\n系统提示: " + messages[0].content)
    print(f"用户问题: {messages[1].content}")
    print(f"\nAI 回复:\n{response.content}")


# ==================== 4. 多轮对话示例 ====================

def multi_turn_conversation():
    """多轮对话示例"""
    print("\n" + "=" * 60)
    print("多轮对话示例")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.7)

    # 初始化对话历史
    conversation: List[BaseMessage] = [
        SystemMessage(content="你是一个友好的聊天助手")
    ]

    # 模拟多轮对话
    user_inputs = [
        "你好！",
        "今天天气怎么样？",
        "推荐一部电影吧"
    ]

    print("\n开始对话:")
    print("-" * 60)

    for user_input in user_inputs:
        # 添加用户消息
        conversation.append(HumanMessage(content=user_input))

        # 获取 AI 回复
        response = model.invoke(conversation)

        # 添加 AI 回复到历史
        conversation.append(AIMessage(content=response.content))

        # 打印对话
        print(f"\n用户: {user_input}")
        print(f"AI: {response.content}")


# ==================== 5. 消息属性访问 ====================

def message_properties():
    """消息属性访问示例"""
    print("\n" + "=" * 60)
    print("消息属性访问示例")
    print("=" * 60)

    # 创建各类消息
    human_msg = HumanMessage(
        content="测试消息",
        name="张三",
        id="msg-001"
    )

    ai_msg = AIMessage(
        content="这是回复",
        name="assistant"
    )

    system_msg = SystemMessage(
        content="系统提示",
        id="sys-001"
    )

    print("\nHumanMessage 属性:")
    print(f"  content: {human_msg.content}")
    print(f"  type: {human_msg.type}")
    print(f"  name: {human_msg.name}")
    print(f"  id: {human_msg.id}")

    print("\nAIMessage 属性:")
    print(f"  content: {ai_msg.content}")
    print(f"  type: {ai_msg.type}")
    print(f"  name: {ai_msg.name}")

    print("\nSystemMessage 属性:")
    print(f"  content: {system_msg.content}")
    print(f"  type: {system_msg.type}")
    print(f"  id: {system_msg.id}")


# ==================== 6. 消息拷贝和修改 ====================

def message_copy_and_modify():
    """消息拷贝和修改示例"""
    print("\n" + "=" * 60)
    print("消息拷贝和修改示例")
    print("=" * 60)

    original = HumanMessage(content="原始消息", name="用户A")

    # 使用 model_copy() 创建副本并修改
    modified = original.model_copy(
        update={"content": "修改后的消息", "name": "用户B"}
    )

    print(f"\n原始消息:")
    print(f"  content: {original.content}")
    print(f"  name: {original.name}")

    print(f"\n修改后消息:")
    print(f"  content: {modified.content}")
    print(f"  name: {modified.name}")

    print(f"\n原始消息未改变: {original.content}")


# ==================== 7. 消息序列化 ====================

def message_serialization():
    """消息序列化示例"""
    print("\n" + "=" * 60)
    print("消息序列化示例")
    print("=" * 60)

    message = HumanMessage(
        content="测试序列化",
        name="user",
        id="msg-123"
    )

    # 转换为字典
    msg_dict = message.model_dump()

    print("\n消息转字典:")
    print(f"  {msg_dict}")

    # 从字典重建消息
    reconstructed = HumanMessage(**msg_dict)

    print(f"\n重建的消息:")
    print(f"  content: {reconstructed.content}")
    print(f"  type: {reconstructed.type}")
    print(f"  name: {reconstructed.name}")


# ==================== 8. 消息列表操作 ====================

def message_list_operations():
    """消息列表操作示例"""
    print("\n" + "=" * 60)
    print("消息列表操作示例")
    print("=" * 60)

    messages: List[BaseMessage] = [
        SystemMessage(content="你是一个助手"),
        HumanMessage(content="问题1"),
        AIMessage(content="回答1"),
        HumanMessage(content="问题2"),
        AIMessage(content="回答2")
    ]

    print("\n消息总数:", len(messages))

    print("\n所有消息:")
    for i, msg in enumerate(messages, 1):
        print(f"  {i}. {type(msg).__name__}: {msg.content}")

    # 过滤特定类型
    human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    print(f"\n用户消息数量: {len(human_messages)}")

    ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
    print(f"AI 消息数量: {len(ai_messages)}")

    # 获取最后 N 条消息
    last_3 = messages[-3:]
    print("\n最后 3 条消息:")
    for msg in last_3:
        print(f"  {type(msg).__name__}: {msg.content}")


# ==================== 9. 条件消息构建 ====================

def conditional_message_building():
    """条件消息构建示例"""
    print("\n" + "=" * 60)
    print("条件消息构建示例")
    print("=" * 60)

    def build_messages(include_system: bool = True, context: str = None) -> List[BaseMessage]:
        """根据条件构建消息列表"""
        messages: List[BaseMessage] = []

        # 可选的系统消息
        if include_system:
            messages.append(SystemMessage(content="你是一个专业助手"))

        # 可选的上下文
        if context:
            messages.append(HumanMessage(content=f"背景信息: {context}"))

        # 主要问题
        messages.append(HumanMessage(content="请帮我解决问题"))

        return messages

    # 测试不同配置
    config1 = build_messages(include_system=True, context="项目开发")
    config2 = build_messages(include_system=False, context=None)

    print("\n配置 1 (包含系统消息和上下文):")
    for msg in config1:
        print(f"  {type(msg).__name__}: {msg.content[:30]}...")

    print("\n配置 2 (仅主要问题):")
    for msg in config2:
        print(f"  {type(msg).__name__}: {msg.content}")


# ==================== 10. 消息内容格式化 ====================

def message_content_formatting():
    """消息内容格式化示例"""
    print("\n" + "=" * 60)
    print("消息内容格式化示例")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.5)

    # 使用不同格式的内容
    messages = [
        HumanMessage(content="请用列表形式说明 Python 的优点"),
    ]

    response = model.invoke(messages)

    print("\n用户问题: " + messages[0].content)
    print(f"\nAI 回复:\n{response.content}")

    # 使用 Markdown 格式
    messages2 = [
        HumanMessage(content="请用 Markdown 表格形式比较 Python 和 Java"),
    ]

    response2 = model.invoke(messages2)
    print("\n" + "-" * 60)
    print("\n用户问题: " + messages2[0].content)
    print(f"\nAI 回复:\n{response2.content}")


if __name__ == "__main__":
    try:
        basic_human_message()
        basic_ai_message()
        system_message_example()
        multi_turn_conversation()
        message_properties()
        message_copy_and_modify()
        message_serialization()
        message_list_operations()
        conditional_message_building()
        message_content_formatting()

        print("\n" + "=" * 60)
        print("所有基础消息示例完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
        import traceback
        traceback.print_exc()
