#!/usr/bin/env python3
"""
GLM-4.6 + LangChain v1.0 对话记忆示例
演示使用LangChain v1.0新API进行对话记忆管理
使用Runnable和messages手动管理对话历史
"""

import os
import dotenv
from typing import List, Dict, Any
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 加载环境变量 - 从项目根目录加载.env文件
dotenv.load_dotenv(dotenv_path="../.env")

# 检查API密钥
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key or api_key == "your-zhipu-api-key-here":
    print("❌ 错误：请在.env文件中设置您的ZHIPUAI_API_KEY")
    exit(1)

# 初始化GLM模型
def get_glm_model(temperature: float = 0.7):
    """获取GLM模型实例"""
    return ChatZhipuAI(
        model="glm-4.6",
        temperature=temperature,
        api_key=api_key
    )


# ========== 记忆管理工具类 ==========
class ConversationBufferWindowMemory:
    """窗口记忆 - 保持固定轮数的对话历史"""

    def __init__(self, k: int = 3):
        self.k = k  # 保留的对话轮数
        self.messages: List[Dict] = []  # 存储对话消息

    def add_message(self, role: str, content: str):
        """添加消息到记忆"""
        self.messages.append({"role": role, "content": content})

        # 如果超过k轮，删除最早的消息（保留system消息）
        if len(self.messages) > self.k * 2:  # *2因为每次对话包含user和assistant
            # 找到第一个非system消息并删除
            non_system_count = sum(1 for msg in self.messages if msg["role"] != "system")
            if non_system_count > self.k * 2:
                # 删除第二个消息开始（索引1，保留system）
                self.messages.pop(1)

    def get_formatted_messages(self, system_prompt: str = "") -> List[Dict]:
        """获取格式化后的消息列表"""
        result = []

        # 添加系统提示
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})
        elif self.messages and self.messages[0]["role"] != "system":
            # 如果没有system消息但有其他消息，添加默认system消息
            result.append({"role": "system", "content": "你是一个友好的AI助手。"})

        # 添加对话历史（限制在k轮）
        recent_messages = self.messages[-self.k * 2:] if len(self.messages) > self.k * 2 else self.messages
        result.extend(recent_messages)

        return result

    def clear(self):
        """清空记忆"""
        self.messages = []


class ConversationTokenBufferMemory:
    """Token限制记忆 - 基于token数量限制对话历史"""

    def __init__(self, max_token_limit: int = 200):
        self.max_token_limit = max_token_limit
        self.messages: List[Dict] = []

    def _estimate_tokens(self, text: str) -> int:
        """粗略估算token数量（中文约4字符=1token）"""
        return len(text) // 4

    def add_message(self, role: str, content: str):
        """添加消息并检查token限制"""
        self.messages.append({"role": role, "content": content})

        # 检查总token数量
        total_tokens = sum(self._estimate_tokens(msg["content"]) for msg in self.messages)

        # 如果超出限制，删除最早的消息
        while total_tokens > self.max_token_limit and len(self.messages) > 1:
            # 保留system消息
            if self.messages[0]["role"] == "system":
                self.messages.pop(1)
            else:
                self.messages.pop(0)

            total_tokens = sum(self._estimate_tokens(msg["content"]) for msg in self.messages)

    def get_formatted_messages(self) -> List[Dict]:
        """获取格式化后的消息列表"""
        return self.messages.copy()

    def clear(self):
        """清空记忆"""
        self.messages = []


class ConversationSummaryMemory:
    """摘要记忆 - 自动总结对话历史"""

    def __init__(self, llm):
        self.llm = llm
        self.summary = ""  # 存储对话摘要
        self.recent_messages: List[Dict] = []  # 存储最近几轮对话

    def add_message(self, role: str, content: str):
        """添加消息并更新摘要"""
        self.recent_messages.append({"role": role, "content": content})

        # 只保留最近2轮对话（节省token）
        if len(self.recent_messages) > 4:  # 2轮对话 = 4条消息
            self.recent_messages.pop(0)

        # 更新摘要
        self._update_summary()

    def _update_summary(self):
        """使用LLM生成对话摘要"""
        if len(self.recent_messages) < 4:  # 至少2轮对话才生成摘要
            return

        summary_prompt = f"""
请将以下对话总结成简短的中文摘要（不超过100字），保留关键信息：

{chr(10).join([f"{'用户' if msg['role']=='user' else '助手'}: {msg['content']}" for msg in self.recent_messages])}

摘要：
"""
        try:
            response = self.llm.invoke([HumanMessage(content=summary_prompt)])
            self.summary = response.content
        except Exception:
            pass  # 摘要生成失败则跳过

    def get_formatted_messages(self) -> List[Dict]:
        """获取格式化后的消息列表"""
        messages = []

        # 添加系统消息
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"对话摘要：{self.summary}"
            })
        else:
            messages.append({
                "role": "system",
                "content": "你是一个专业的AI助手。"
            })

        messages.extend(self.recent_messages)
        return messages

    def clear(self):
        """清空记忆"""
        self.summary = ""
        self.recent_messages = []


# ========== 示例函数 ==========

def conversation_buffer_window_memory_example():
    """窗口记忆示例 - 保持固定对话轮数"""
    print("=" * 60)
    print("🪟 ConversationBufferWindowMemory 示例 (v1.0)")
    print("=" * 60)

    model = get_glm_model()

    # 创建窗口记忆 - 只保留最近3轮对话
    memory = ConversationBufferWindowMemory(k=3)

    system_prompt = "你是一个友好的AI助手，能够记住最近的对话内容。"

    # 模拟对话
    conversations = [
        "你好，我叫小明",
        "我喜欢编程和人工智能",
        "今天天气怎么样？",
        "你能推荐一本Python书籍吗？",
        "我刚才说我叫什么名字？",  # 测试记忆
        "我提到我喜欢什么？"       # 测试记忆
    ]

    print("🚀 开始对话测试...")
    for i, user_input in enumerate(conversations, 1):
        print(f"\n👤 用户 [{i}]: {user_input}")

        # 获取格式化消息
        messages = memory.get_formatted_messages(system_prompt)
        messages.append({"role": "user", "content": user_input})

        # 调用模型
        response = model.invoke(messages)
        print(f"🤖 AI: {response.content}")

        # 更新记忆
        memory.add_message("user", user_input)
        memory.add_message("assistant", response.content)

        # 显示当前记忆状态
        print(f"📝 记忆中消息数: {len(memory.messages)}")


def conversation_token_buffer_memory_example():
    """Token限制记忆示例 - 基于token数量限制"""
    print("\n" + "=" * 60)
    print("🪙 ConversationTokenBufferMemory 示例 (v1.0)")
    print("=" * 60)

    model = get_glm_model()

    # 创建token限制记忆 - 最多200个token
    memory = ConversationTokenBufferMemory(max_token_limit=200)

    # 模拟较长的对话
    conversations = [
        "请详细解释什么是机器学习？",
        "机器学习有哪些主要的分类？请逐一介绍监督学习、无监督学习和强化学习的特点和应用场景。",
        "深度学习和机器学习有什么区别？神经网络是如何工作的？",
        "请推荐一些学习机器学习的入门资源，包括书籍、在线课程和实践项目。",
        "我想知道我的第一个问题是什么？"  # 测试记忆
    ]

    print("🚀 开始对话测试...")
    for i, user_input in enumerate(conversations, 1):
        print(f"\n👤 用户 [{i}]: {user_input[:30]}...")

        # 获取格式化消息
        messages = memory.get_formatted_messages()
        messages.append({"role": "user", "content": user_input})

        # 调用模型
        response = model.invoke(messages)
        print(f"🤖 AI: {response.content[:50]}...")

        # 更新记忆（会自动处理token限制）
        memory.add_message("user", user_input)
        memory.add_message("assistant", response.content)

        # 估算token数量
        total_chars = sum(len(msg["content"]) for msg in memory.messages)
        print(f"📊 记忆中字符数: {total_chars}, 估算token数: {total_chars // 4}")


def conversation_summary_memory_example():
    """摘要记忆示例 - 对话摘要记忆"""
    print("\n" + "=" * 60)
    print("📋 ConversationSummaryMemory 示例 (v1.0)")
    print("=" * 60)

    model = get_glm_model()

    # 创建摘要记忆
    memory = ConversationSummaryMemory(llm=model)

    # 模拟多轮对话
    conversations = [
        "我想学习Python编程",
        "我没有任何编程经验，应该从哪里开始？",
        "请推荐一些入门书籍和在线资源",
        "我应该先学Python 2还是Python 3？",
        "根据之前的对话，我应该怎么开始？"
    ]

    print("🚀 开始对话测试...")
    for i, user_input in enumerate(conversations, 1):
        print(f"\n👤 用户 [{i}]: {user_input}")

        # 获取格式化消息（包含摘要）
        messages = memory.get_formatted_messages()
        messages.append({"role": "user", "content": user_input})

        # 调用模型
        response = model.invoke(messages)
        print(f"🤖 AI: {response.content}")

        # 更新记忆和摘要
        memory.add_message("user", user_input)
        memory.add_message("assistant", response.content)

        # 显示摘要
        if memory.summary:
            print(f"📝 当前摘要: {memory.summary}")


def conversation_summary_buffer_memory_example():
    """摘要缓冲记忆示例 - 组合摘要和缓冲记忆"""
    print("\n" + "=" * 60)
    print("📊 ConversationSummaryBufferMemory 示例 (v1.0)")
    print("=" * 60)

    model = get_glm_model()

    # 创建组合记忆
    summary_memory = ConversationSummaryMemory(llm=model)
    buffer_memory = ConversationBufferWindowMemory(k=2)

    # 模拟长时间对话
    conversations = [
        "我想了解人工智能的发展历史",
        "1950年代有哪些重要事件？",
        "图灵测试是什么？",
        "为什么1956年被称为AI的诞生年？",
        "专家系统在1980年代有什么突破？",
        "深度学习革命是从什么时候开始的？",
        "根据我们讨论的历史，AI的未来趋势是什么？",
        "我第一个问题是关于什么的？"  # 测试记忆
    ]

    print("🚀 开始长时间对话测试...")
    for i, user_input in enumerate(conversations, 1):
        print(f"\n👤 用户 [{i}]: {user_input[:30]}...")

        # 获取摘要作为系统提示
        messages = summary_memory.get_formatted_messages()
        # 添加最近的缓冲消息
        recent_messages = buffer_memory.get_formatted_messages()
        # 合并消息（避免重复system消息）
        if recent_messages and recent_messages[0]["role"] != "system":
            messages.extend(recent_messages)
        elif recent_messages:
            messages.extend(recent_messages[1:])  # 跳过重复的system消息

        messages.append({"role": "user", "content": user_input})

        # 调用模型
        response = model.invoke(messages)
        print(f"🤖 AI: {response.content[:50]}...")

        # 更新两种记忆
        summary_memory.add_message("user", user_input)
        summary_memory.add_message("assistant", response.content)
        buffer_memory.add_message("user", user_input)
        buffer_memory.add_message("assistant", response.content)

        # 显示摘要
        if summary_memory.summary:
            print(f"📝 摘要: {summary_memory.summary[:50]}...")


# ========== 主函数 ==========

if __name__ == "__main__":
    print("""
    🎉 LangChain v1.0 对话记忆示例
    ============

    新特性：
    1. 使用Runnable和messages API
    2. 手动管理对话历史
    3. 支持窗口记忆、Token限制记忆和摘要记忆

    """)
    print()

    # 运行所有示例
    conversation_buffer_window_memory_example()
    conversation_token_buffer_memory_example()
    conversation_summary_memory_example()
    conversation_summary_buffer_memory_example()

    print("\n" + "=" * 60)
    print("✅ 所有示例运行完成！")
    print("=" * 60)
