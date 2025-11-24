"""
LangChain Messages - 消息历史管理示例
演示对话历史存储、检索、管理等
使用 GLM 模型
"""

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
    trim_messages
)
from langchain_core.chat_history import InMemoryChatMessageHistory
from typing import List, Dict
import json
import os

os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY", "your-api-key-here")


# ==================== 1. 基础消息历史 ====================

def basic_message_history():
    """基础消息历史示例"""
    print("=" * 60)
    print("基础消息历史示例")
    print("=" * 60)

    # 创建内存历史
    history = InMemoryChatMessageHistory()

    # 添加消息
    history.add_user_message("你好")
    history.add_ai_message("你好！有什么可以帮您？")
    history.add_user_message("今天天气如何？")
    history.add_ai_message("今天天气晴朗。")

    # 获取所有消息
    messages = history.messages

    print(f"\n历史消息数量: {len(messages)}")
    print("\n对话历史:")
    for i, msg in enumerate(messages, 1):
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {i}. {role}: {msg.content}")


# ==================== 2. 使用历史进行对话 ====================

def conversation_with_history():
    """使用历史进行对话示例"""
    print("\n" + "=" * 60)
    print("使用历史进行对话示例")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.7)
    history = InMemoryChatMessageHistory()

    # 添加系统消息
    history.add_message(SystemMessage(content="你是一个友好的助手"))

    # 模拟多轮对话
    conversations = [
        "你好",
        "我叫张三",
        "我的名字是什么？"  # 测试记忆
    ]

    print("\n对话过程:")
    print("-" * 60)

    for user_input in conversations:
        # 添加用户消息
        history.add_user_message(user_input)

        # 使用历史获取回复
        response = model.invoke(history.messages)

        # 添加 AI 回复
        history.add_ai_message(response.content)

        print(f"\n用户: {user_input}")
        print(f"AI: {response.content}")


# ==================== 3. 会话管理 ====================

def session_management():
    """会话管理示例"""
    print("\n" + "=" * 60)
    print("会话管理示例")
    print("=" * 60)

    class SessionManager:
        """简单的会话管理器"""

        def __init__(self):
            self.sessions: Dict[str, InMemoryChatMessageHistory] = {}

        def get_session(self, session_id: str) -> InMemoryChatMessageHistory:
            """获取或创建会话"""
            if session_id not in self.sessions:
                self.sessions[session_id] = InMemoryChatMessageHistory()
            return self.sessions[session_id]

        def list_sessions(self) -> List[str]:
            """列出所有会话"""
            return list(self.sessions.keys())

        def clear_session(self, session_id: str):
            """清除会话"""
            if session_id in self.sessions:
                del self.sessions[session_id]

    # 使用会话管理器
    manager = SessionManager()

    # 会话 1
    session1 = manager.get_session("user-001")
    session1.add_user_message("我是用户1")
    session1.add_ai_message("你好,用户1")

    # 会话 2
    session2 = manager.get_session("user-002")
    session2.add_user_message("我是用户2")
    session2.add_ai_message("你好,用户2")

    print("\n活跃会话:")
    for session_id in manager.list_sessions():
        session = manager.get_session(session_id)
        print(f"  {session_id}: {len(session.messages)} 条消息")


# ==================== 4. 历史限制和修剪 ====================

def history_trimming():
    """历史限制和修剪示例"""
    print("\n" + "=" * 60)
    print("历史限制和修剪示例")
    print("=" * 60)

    class BoundedMessageHistory:
        """有界限的消息历史"""

        def __init__(self, max_messages: int = 10):
            self.history = InMemoryChatMessageHistory()
            self.max_messages = max_messages
            self.system_message = None

        def add_message(self, message: BaseMessage):
            """添加消息并自动修剪"""
            # 保存系统消息
            if isinstance(message, SystemMessage) and not self.system_message:
                self.system_message = message

            self.history.add_message(message)
            self._trim_if_needed()

        def _trim_if_needed(self):
            """如需要则修剪历史"""
            messages = self.history.messages
            system_msgs = [self.system_message] if self.system_message else []
            other_msgs = [msg for msg in messages if not isinstance(msg, SystemMessage)]

            if len(other_msgs) > self.max_messages:
                # 保留最近的消息
                kept = other_msgs[-self.max_messages:]
                self.history.clear()

                # 重新添加
                for msg in system_msgs + kept:
                    self.history.add_message(msg)

        @property
        def messages(self) -> List[BaseMessage]:
            return self.history.messages

    # 测试有界历史
    bounded = BoundedMessageHistory(max_messages=4)

    bounded.add_message(SystemMessage(content="你是助手"))

    for i in range(6):
        bounded.add_message(HumanMessage(content=f"问题{i+1}"))
        bounded.add_message(AIMessage(content=f"回答{i+1}"))

    print(f"\n添加了 6 轮对话 (12 条消息)")
    print(f"实际保留: {len(bounded.messages)} 条 (含系统消息)")

    print("\n保留的消息:")
    for msg in bounded.messages:
        print(f"  {type(msg).__name__}: {msg.content}")


# ==================== 5. 历史持久化 (JSON) ====================

def history_persistence_json():
    """历史持久化 JSON 示例"""
    print("\n" + "=" * 60)
    print("历史持久化 JSON 示例")
    print("=" * 60)

    def save_history_to_json(history: InMemoryChatMessageHistory, filepath: str):
        """保存历史到 JSON 文件"""
        messages_data = []
        for msg in history.messages:
            messages_data.append({
                "type": type(msg).__name__,
                "content": msg.content,
                "id": msg.id
            })

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(messages_data, f, ensure_ascii=False, indent=2)

    def load_history_from_json(filepath: str) -> InMemoryChatMessageHistory:
        """从 JSON 文件加载历史"""
        with open(filepath, 'r', encoding='utf-8') as f:
            messages_data = json.load(f)

        history = InMemoryChatMessageHistory()
        for data in messages_data:
            msg_type = data["type"]
            if msg_type == "HumanMessage":
                msg = HumanMessage(content=data["content"], id=data.get("id"))
            elif msg_type == "AIMessage":
                msg = AIMessage(content=data["content"], id=data.get("id"))
            elif msg_type == "SystemMessage":
                msg = SystemMessage(content=data["content"], id=data.get("id"))
            else:
                continue
            history.add_message(msg)

        return history

    # 创建并保存历史
    history = InMemoryChatMessageHistory()
    history.add_message(SystemMessage(content="你是助手"))
    history.add_user_message("你好")
    history.add_ai_message("你好！")

    filepath = "/tmp/chat_history.json"
    save_history_to_json(history, filepath)
    print(f"\n历史已保存到: {filepath}")

    # 加载历史
    loaded_history = load_history_from_json(filepath)
    print(f"加载了 {len(loaded_history.messages)} 条消息")

    print("\n加载的消息:")
    for msg in loaded_history.messages:
        print(f"  {type(msg).__name__}: {msg.content}")


# ==================== 6. 历史摘要 ====================

def history_summarization():
    """历史摘要示例"""
    print("\n" + "=" * 60)
    print("历史摘要示例")
    print("=" * 60)

    model = ChatZhipuAI(model="glm-4.6", temperature=0.3)

    class SummarizingHistory:
        """带摘要的历史"""

        def __init__(self, model, summarize_after: int = 10):
            self.model = model
            self.summarize_after = summarize_after
            self.history = InMemoryChatMessageHistory()
            self.summary = None

        def add_message(self, message: BaseMessage):
            """添加消息"""
            self.history.add_message(message)

            # 定期生成摘要
            if len(self.history.messages) >= self.summarize_after:
                self._create_summary()

        def _create_summary(self):
            """创建历史摘要"""
            messages = self.history.messages

            # 构建摘要提示
            summary_prompt = "简要总结以下对话(一句话):\n\n"
            for msg in messages:
                if not isinstance(msg, SystemMessage):
                    role = "用户" if isinstance(msg, HumanMessage) else "AI"
                    summary_prompt += f"{role}: {msg.content}\n"

            response = self.model.invoke([HumanMessage(content=summary_prompt)])
            self.summary = response.content

            # 清除旧历史,保留摘要
            self.history.clear()
            if self.summary:
                self.history.add_message(
                    SystemMessage(content=f"之前的对话摘要: {self.summary}")
                )

        def get_messages(self) -> List[BaseMessage]:
            """获取消息(含摘要)"""
            return self.history.messages

    # 测试摘要历史
    sum_history = SummarizingHistory(model, summarize_after=6)

    print("\n添加对话...")
    for i in range(4):
        sum_history.add_message(HumanMessage(content=f"问题{i+1}"))
        sum_history.add_message(AIMessage(content=f"回答{i+1}"))

    if sum_history.summary:
        print(f"\n生成的摘要: {sum_history.summary}")

    print(f"\n当前消息数: {len(sum_history.get_messages())}")


# ==================== 7. 历史搜索 ====================

def history_search():
    """历史搜索示例"""
    print("\n" + "=" * 60)
    print("历史搜索示例")
    print("=" * 60)

    history = InMemoryChatMessageHistory()

    # 添加测试数据
    conversations = [
        ("Python 是什么？", "Python 是一种编程语言"),
        ("如何学习 Python？", "可以从官方教程开始"),
        ("JavaScript 和 Python 的区别", "JavaScript 主要用于 Web 开发"),
        ("推荐一本 Python 书", "《Python 编程:从入门到实践》")
    ]

    for user, ai in conversations:
        history.add_user_message(user)
        history.add_ai_message(ai)

    print(f"\n历史中有 {len(history.messages)} 条消息")

    # 搜索功能
    def search_history(history: InMemoryChatMessageHistory, keyword: str) -> List[BaseMessage]:
        """搜索包含关键词的消息"""
        return [
            msg for msg in history.messages
            if keyword.lower() in msg.content.lower()
        ]

    # 搜索 "Python"
    keyword = "Python"
    results = search_history(history, keyword)

    print(f"\n搜索 '{keyword}' 的结果 ({len(results)} 条):")
    for msg in results:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        print(f"  {role}: {msg.content}")


# ==================== 8. 历史分支 ====================

def history_branching():
    """历史分支示例"""
    print("\n" + "=" * 60)
    print("历史分支示例")
    print("=" * 60)

    class BranchingHistory:
        """支持分支的历史"""

        def __init__(self):
            self.branches: Dict[str, InMemoryChatMessageHistory] = {
                "main": InMemoryChatMessageHistory()
            }
            self.current_branch = "main"

        def create_branch(self, branch_name: str, from_branch: str = None):
            """创建新分支"""
            if from_branch is None:
                from_branch = self.current_branch

            # 复制消息
            source = self.branches[from_branch]
            new_branch = InMemoryChatMessageHistory()

            for msg in source.messages:
                new_branch.add_message(msg)

            self.branches[branch_name] = new_branch

        def switch_branch(self, branch_name: str):
            """切换分支"""
            if branch_name in self.branches:
                self.current_branch = branch_name

        def add_message(self, message: BaseMessage):
            """添加消息到当前分支"""
            self.branches[self.current_branch].add_message(message)

        @property
        def messages(self) -> List[BaseMessage]:
            """获取当前分支的消息"""
            return self.branches[self.current_branch].messages

    # 测试分支
    branching = BranchingHistory()

    # 主分支
    branching.add_message(HumanMessage(content="问题1"))
    branching.add_message(AIMessage(content="回答1"))

    print("\n主分支消息数:", len(branching.messages))

    # 创建分支
    branching.create_branch("experiment")
    branching.switch_branch("experiment")

    branching.add_message(HumanMessage(content="实验性问题"))
    branching.add_message(AIMessage(content="实验性回答"))

    print("实验分支消息数:", len(branching.messages))

    # 切回主分支
    branching.switch_branch("main")
    print("切回主分支,消息数:", len(branching.messages))


# ==================== 9. 历史统计 ====================

def history_statistics():
    """历史统计示例"""
    print("\n" + "=" * 60)
    print("历史统计示例")
    print("=" * 60)

    history = InMemoryChatMessageHistory()

    # 添加测试数据
    for i in range(5):
        history.add_user_message(f"这是第 {i+1} 个问题")
        history.add_ai_message(f"这是第 {i+1} 个回答,内容稍微长一些")

    messages = history.messages

    # 统计
    user_count = sum(1 for msg in messages if isinstance(msg, HumanMessage))
    ai_count = sum(1 for msg in messages if isinstance(msg, AIMessage))

    user_chars = sum(len(msg.content) for msg in messages if isinstance(msg, HumanMessage))
    ai_chars = sum(len(msg.content) for msg in messages if isinstance(msg, AIMessage))

    print("\n历史统计:")
    print(f"  总消息数: {len(messages)}")
    print(f"  用户消息: {user_count} 条")
    print(f"  AI 消息: {ai_count} 条")
    print(f"  用户字符数: {user_chars}")
    print(f"  AI 字符数: {ai_chars}")
    print(f"  平均用户消息长度: {user_chars/user_count:.1f}")
    print(f"  平均 AI 消息长度: {ai_chars/ai_count:.1f}")


# ==================== 10. 历史最佳实践 ====================

def history_best_practices():
    """历史管理最佳实践"""
    print("\n" + "=" * 60)
    print("历史管理最佳实践")
    print("=" * 60)

    print("\n1. 选择合适的存储")
    print("   - InMemoryChatMessageHistory: 简单场景")
    print("   - FileChatMessageHistory: 需要持久化")
    print("   - RedisChatMessageHistory: 分布式应用")

    print("\n2. 控制历史长度")
    print("   ✓ 使用滑动窗口")
    print("   ✓ 定期生成摘要")
    print("   ✓ 删除不重要的消息")

    print("\n3. 会话管理")
    print("   ✓ 为每个用户/会话分配 ID")
    print("   ✓ 设置会话过期时间")
    print("   ✓ 定期清理过期会话")

    print("\n4. 性能优化")
    print("   - 异步加载历史")
    print("   - 使用索引加速搜索")
    print("   - 缓存常用数据")

    print("\n5. 隐私和安全")
    print("   - 不记录敏感信息")
    print("   - 加密存储")
    print("   - 遵守数据保留政策")

    print("\n6. 监控和维护")
    print("   - 记录历史大小")
    print("   - 监控内存使用")
    print("   - 定期备份重要数据")


if __name__ == "__main__":
    try:
        basic_message_history()
        conversation_with_history()
        session_management()
        history_trimming()
        history_persistence_json()
        history_summarization()
        history_search()
        history_branching()
        history_statistics()
        history_best_practices()

        print("\n" + "=" * 60)
        print("所有消息历史管理示例完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("请确保已设置 ZHIPUAI_API_KEY 环境变量")
        import traceback
        traceback.print_exc()
