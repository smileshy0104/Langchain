"""
LangChain Messages - 消息操作示例
演示 add_messages, trim_messages, RemoveMessage 等操作
使用 GLM 模型
"""

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    RemoveMessage,
    BaseMessage,
    trim_messages
)
from langgraph.graph import MessagesState
from typing import List, Sequence
import os

os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 1. add_messages 基础 ====================

def add_messages_basic():
    """add_messages 基础示例"""
    print("=" * 60)
    print("add_messages 基础示例")
    print("=" * 60)

    from langgraph.graph.message import add_messages

    # 初始消息列表
    existing = [
        SystemMessage(content="你是一个助手", id="sys-1"),
        HumanMessage(content="你好", id="msg-1"),
        AIMessage(content="你好！有什么可以帮您？", id="msg-2")
    ]

    # 添加新消息
    new = [
        HumanMessage(content="今天天气如何？", id="msg-3")
    ]

    # 使用 add_messages 合并
    result = add_messages(existing, new)

    print("\n原有消息数量:", len(existing))
    print("新消息数量:", len(new))
    print("合并后数量:", len(result))

    print("\n合并后的消息:")
    for i, msg in enumerate(result, 1):
        print(f"  {i}. [{msg.id}] {type(msg).__name__}: {msg.content}")


# ==================== 2. add_messages 更新现有消息 ====================

def add_messages_update():
    """add_messages 更新现有消息示例"""
    print("\n" + "=" * 60)
    print("add_messages 更新现有消息示例")
    print("=" * 60)

    from langgraph.graph.message import add_messages

    # 初始消息
    existing = [
        HumanMessage(content="原始内容", id="msg-1"),
        AIMessage(content="原始回复", id="msg-2")
    ]

    # 更新消息 (相同 ID)
    updates = [
        HumanMessage(content="更新后的内容", id="msg-1")  # 相同 ID
    ]

    result = add_messages(existing, updates)

    print("\n更新前:")
    for msg in existing:
        print(f"  [{msg.id}] {msg.content}")

    print("\n更新后:")
    for msg in result:
        print(f"  [{msg.id}] {msg.content}")

    print("\n说明: 相同 ID 的消息会被更新")


# ==================== 3. RemoveMessage 删除消息 ====================

def remove_message_example():
    """RemoveMessage 删除消息示例"""
    print("\n" + "=" * 60)
    print("RemoveMessage 删除消息示例")
    print("=" * 60)

    from langgraph.graph.message import add_messages

    # 初始消息
    messages = [
        SystemMessage(content="系统提示", id="sys-1"),
        HumanMessage(content="消息1", id="msg-1"),
        AIMessage(content="回复1", id="msg-2"),
        HumanMessage(content="消息2", id="msg-3"),
        AIMessage(content="回复2", id="msg-4")
    ]

    print("\n删除前的消息:")
    for msg in messages:
        print(f"  [{msg.id}] {type(msg).__name__}: {msg.content}")

    # 删除特定消息
    deletions = [
        RemoveMessage(id="msg-1"),  # 删除 msg-1
        RemoveMessage(id="msg-2")   # 删除 msg-2
    ]

    result = add_messages(messages, deletions)

    print(f"\n删除后的消息 (删除了 {len(deletions)} 条):")
    for msg in result:
        print(f"  [{msg.id}] {type(msg).__name__}: {msg.content}")


# ==================== 4. 批量删除消息 ====================

def batch_remove_messages():
    """批量删除消息示例"""
    print("\n" + "=" * 60)
    print("批量删除消息示例")
    print("=" * 60)

    from langgraph.graph.message import add_messages

    messages = [
        SystemMessage(content="系统提示", id="sys-1"),
        HumanMessage(content="问题1", id="q-1"),
        AIMessage(content="回答1", id="a-1"),
        HumanMessage(content="问题2", id="q-2"),
        AIMessage(content="回答2", id="a-2"),
        HumanMessage(content="问题3", id="q-3"),
        AIMessage(content="回答3", id="a-3")
    ]

    print(f"\n原始消息数量: {len(messages)}")

    # 删除所有 AI 回复
    ai_ids = [msg.id for msg in messages if isinstance(msg, AIMessage)]
    deletions = [RemoveMessage(id=msg_id) for msg_id in ai_ids]

    result = add_messages(messages, deletions)

    print(f"删除 {len(deletions)} 条 AI 消息后: {len(result)} 条")

    print("\n剩余消息:")
    for msg in result:
        print(f"  [{msg.id}] {type(msg).__name__}: {msg.content}")


# ==================== 5. trim_messages 基础 ====================

def trim_messages_basic():
    """trim_messages 基础示例"""
    print("\n" + "=" * 60)
    print("trim_messages 基础示例")
    print("=" * 60)

    messages = [
        SystemMessage(content="你是一个助手"),
        HumanMessage(content="消息1"),
        AIMessage(content="回复1"),
        HumanMessage(content="消息2"),
        AIMessage(content="回复2"),
        HumanMessage(content="消息3"),
        AIMessage(content="回复3")
    ]

    print(f"\n原始消息数量: {len(messages)}")

    # 保留最后 4 条消息
    trimmed = trim_messages(
        messages,
        max_tokens=4,  # 保留 4 条
        strategy="last",
        token_counter=len  # 使用消息数量作为 token 计数
    )

    print(f"修剪后消息数量: {len(trimmed)}")

    print("\n修剪后的消息:")
    for i, msg in enumerate(trimmed, 1):
        print(f"  {i}. {type(msg).__name__}: {msg.content}")


# ==================== 6. trim_messages 保留系统消息 ====================

def trim_messages_keep_system():
    """trim_messages 保留系统消息示例"""
    print("\n" + "=" * 60)
    print("trim_messages 保留系统消息示例")
    print("=" * 60)

    messages = [
        SystemMessage(content="重要: 你是专业助手"),
        HumanMessage(content="问题1"),
        AIMessage(content="回答1"),
        HumanMessage(content="问题2"),
        AIMessage(content="回答2"),
        HumanMessage(content="问题3"),
        AIMessage(content="回答3")
    ]

    print(f"\n原始消息: {len(messages)} 条")

    # 修剪但保留系统消息
    trimmed = trim_messages(
        messages,
        max_tokens=3,  # 只保留 3 条
        strategy="last",
        token_counter=len,
        include_system=True  # 始终保留系统消息
    )

    print(f"修剪后: {len(trimmed)} 条")

    print("\n修剪后的消息:")
    for msg in trimmed:
        msg_type = type(msg).__name__
        indicator = " ← 保留" if isinstance(msg, SystemMessage) else ""
        print(f"  {msg_type}: {msg.content}{indicator}")


# ==================== 7. 按 Token 数量修剪 ====================

def trim_by_token_count():
    """按 Token 数量修剪示例"""
    print("\n" + "=" * 60)
    print("按 Token 数量修剪示例")
    print("=" * 60)

    def simple_token_counter(messages: List[BaseMessage]) -> int:
        """简单的 token 计数器 (示例用)"""
        # 使用 split() 函数计算词数，空格分割
        return sum(len(msg.content.split()) for msg in messages)

    messages = [
        SystemMessage(content="你是助手"),
        HumanMessage(content="这是一个很长的问题 包含很多词语"),
        AIMessage(content="这是一个简短回复"),
        HumanMessage(content="又一个问题"),
        AIMessage(content="又一个回复 稍微长一点")
    ]

    print("\n每条消息的词数:")
    for i, msg in enumerate(messages, 1):
        word_count = len(msg.content.split()) # 词数
        print(f"  {i}. {type(msg).__name__}: {word_count} 词")

    total_words = simple_token_counter(messages)
    print(f"\n总词数: {total_words}")

    # 限制为 10 个词以内
    trimmed = trim_messages(
        messages,
        max_tokens=10,
        strategy="last",
        token_counter=simple_token_counter
    )

    trimmed_words = simple_token_counter(trimmed)
    print(f"\n修剪后词数: {trimmed_words}")
    print(f"修剪后消息数: {len(trimmed)}")


# ==================== 8. 消息窗口滑动 ====================

def sliding_window_messages():
    """消息窗口滑动示例"""
    print("\n" + "=" * 60)
    print("消息窗口滑动示例")
    print("=" * 60)

    # 模拟长对话
    all_messages = [SystemMessage(content="你是助手")]

    for i in range(1, 11):
        all_messages.append(HumanMessage(content=f"问题{i}"))
        all_messages.append(AIMessage(content=f"回答{i}"))

    print(f"\n完整对话: {len(all_messages)} 条消息")

    # 维护固定窗口大小
    window_size = 5

    def get_recent_window(messages: List[BaseMessage], size: int) -> List[BaseMessage]:
        """获取最近的 N 条消息,保留系统消息"""
        system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
        other_msgs = [msg for msg in messages if not isinstance(msg, SystemMessage)]

        recent = other_msgs[-size:]
        return system_msgs + recent

    window = get_recent_window(all_messages, window_size)

    print(f"窗口大小: {window_size} (不含系统消息)")
    print(f"实际消息数: {len(window)} (含系统消息)")

    print("\n窗口内容:")
    for msg in window:
        print(f"  {type(msg).__name__}: {msg.content}")


# ==================== 9. 消息摘要和压缩 ====================

def message_summarization():
    """消息摘要和压缩示例"""
    print("\n" + "=" * 60)
    print("消息摘要和压缩示例")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.3)

    # 长对话历史
    long_history = [
        SystemMessage(content="你是助手"),
        HumanMessage(content="今天天气怎么样？"),
        AIMessage(content="今天天气晴朗,温度适宜。"),
        HumanMessage(content="那明天呢？"),
        AIMessage(content="明天可能会有小雨。"),
        HumanMessage(content="我应该带伞吗？"),
        AIMessage(content="建议带伞,以防万一。")
    ]

    print(f"\n原始对话: {len(long_history)} 条消息")

    # 生成摘要
    summary_prompt = "请用一句话总结以下对话:\n\n"
    for msg in long_history[1:]:  # 跳过系统消息
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        summary_prompt += f"{role}: {msg.content}\n"

    summary_response = model.invoke([HumanMessage(content=summary_prompt)])
    summary = summary_response.content

    print(f"\n对话摘要: {summary}")

    # 用摘要替换历史
    compressed = [
        SystemMessage(content="你是助手"),
        SystemMessage(content=f"之前的对话摘要: {summary}"),
        HumanMessage(content="那后天呢？")  # 新问题
    ]

    print(f"\n压缩后: {len(compressed)} 条消息")


# ==================== 10. 消息去重 ====================

def message_deduplication():
    """消息去重示例"""
    print("\n" + "=" * 60)
    print("消息去重示例")
    print("=" * 60)

    messages = [
        HumanMessage(content="你好", id="msg-1"),
        AIMessage(content="你好！", id="msg-2"),
        HumanMessage(content="你好", id="msg-3"),  # 重复内容
        AIMessage(content="有什么可以帮您？", id="msg-4"),
        HumanMessage(content="你好", id="msg-5")  # 再次重复
    ]

    print("\n原始消息:")
    for i, msg in enumerate(messages, 1):
        print(f"  {i}. {type(msg).__name__}: {msg.content}")

    # 基于内容去重
    def deduplicate_by_content(msgs: List[BaseMessage]) -> List[BaseMessage]:
        """按内容去重"""
        seen = set()
        result = []
        for msg in msgs:
            key = (type(msg).__name__, msg.content)
            if key not in seen:
                seen.add(key)
                result.append(msg)
        return result

    deduped = deduplicate_by_content(messages)

    print(f"\n去重后: {len(deduped)} 条")
    for i, msg in enumerate(deduped, 1):
        print(f"  {i}. {type(msg).__name__}: {msg.content}")


# ==================== 11. 消息过滤 ====================

def message_filtering():
    """消息过滤示例"""
    print("\n" + "=" * 60)
    print("消息过滤示例")
    print("=" * 60)

    messages = [
        SystemMessage(content="你是助手"),
        HumanMessage(content="正常问题1"),
        AIMessage(content="正常回答1"),
        HumanMessage(content="[DEBUG] 测试信息"),
        AIMessage(content="这是调试回复"),
        HumanMessage(content="正常问题2"),
        AIMessage(content="正常回答2")
    ]

    print(f"\n原始消息: {len(messages)} 条")

    # 过滤调试信息
    def filter_debug_messages(msgs: List[BaseMessage]) -> List[BaseMessage]:
        """过滤包含 [DEBUG] 的消息"""
        return [
            msg for msg in msgs
            if not msg.content.startswith("[DEBUG]")
        ]

    filtered = filter_debug_messages(messages)

    print(f"过滤后: {len(filtered)} 条")

    print("\n过滤后的消息:")
    for msg in filtered:
        print(f"  {type(msg).__name__}: {msg.content}")


# ==================== 12. 消息操作最佳实践 ====================

def message_operations_best_practices():
    """消息操作最佳实践"""
    print("\n" + "=" * 60)
    print("消息操作最佳实践")
    print("=" * 60)

    print("\n1. 使用 add_messages")
    print("   ✓ 合并消息列表")
    print("   ✓ 更新现有消息 (相同 ID)")
    print("   ✓ 配合 RemoveMessage 删除")

    print("\n2. 使用 trim_messages")
    print("   ✓ 控制上下文长度")
    print("   ✓ 保留系统消息")
    print("   ✓ 使用合适的 token 计数器")

    print("\n3. 使用 RemoveMessage")
    print("   ✓ 精确删除特定消息")
    print("   ✓ 批量删除")
    print("   ✓ 需要消息有 ID")

    print("\n4. 消息管理策略")
    print("   - 滑动窗口: 保留最近 N 条")
    print("   - 摘要压缩: 总结旧对话")
    print("   - 选择性保留: 保留重要消息")

    print("\n5. 性能优化")
    print("   - 及时删除不需要的消息")
    print("   - 避免过长的上下文")
    print("   - 使用高效的 token 计数器")

    print("\n6. 注意事项")
    print("   - 始终保留必要的系统消息")
    print("   - 保持对话连贯性")
    print("   - 注意消息顺序")


if __name__ == "__main__":
    try:
        # add_messages_basic()
        # add_messages_update()
        # remove_message_example()
        # batch_remove_messages()
        # trim_messages_basic()
        # trim_messages_keep_system()
        # trim_by_token_count()
        # sliding_window_messages()
        # message_summarization()
        # message_deduplication()
        # message_filtering()
        message_operations_best_practices()

        print("\n" + "=" * 60)
        print("所有消息操作示例完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
        import traceback
        traceback.print_exc()
