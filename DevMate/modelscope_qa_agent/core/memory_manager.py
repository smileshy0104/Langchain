"""对话记忆管理模块

该模块负责管理多轮对话的历史记录，实现以下功能:
1. 滑动窗口机制: 保留最近 N 轮完整对话
2. 早期对话摘要: 将超过窗口的早期对话压缩为摘要
3. Token 限制: 控制对话历史的 Token 消耗
4. 对话上下文提取: 为 LLM 提供优化的上下文信息

设计参考:
- spec.md:14 - 滑动窗口 + 摘要策略
- spec.md:161 - FR-003: 保留最近10轮完整对话
- research.md:55-78 - trim_messages 工具函数
- data-model.md:42 - conversation_summary 字段

Author: Claude Code
Created: 2025-12-01
"""

from typing import Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, trim_messages
from langchain_core.language_models import BaseChatModel


class MemoryManager:
    """对话记忆管理器

    实现滑动窗口 + 摘要策略的对话历史管理:
    - 保留最近 N 轮完整对话（默认10轮）
    - 将更早的对话压缩为摘要文本
    - 控制 Token 消耗在合理范围内
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        max_turns: int = 10,
        max_tokens: int = 4000,
        include_system: bool = True
    ):
        """初始化对话记忆管理器

        Args:
            llm: 用于生成摘要的 LLM 实例（可选）
            max_turns: 滑动窗口大小（保留的完整对话轮次）
            max_tokens: 对话历史的最大 Token 限制
            include_system: 是否始终保留系统消息
        """
        self.llm = llm
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.include_system = include_system

    def trim_conversation(
        self,
        messages: list[BaseMessage],
        strategy: str = "last"
    ) -> list[BaseMessage]:
        """修剪对话历史，保留最近 N 轮对话

        实现滑动窗口机制，保留最近的完整对话轮次。

        Args:
            messages: 原始消息列表
            strategy: 修剪策略
                - "last": 保留最新消息（默认）
                - "first": 保留最早消息（通常用于保留系统 Prompt）

        Returns:
            修剪后的消息列表

        Examples:
            >>> manager = MemoryManager(max_turns=3)
            >>> messages = [
            ...     SystemMessage(content="你是助手"),
            ...     HumanMessage(content="问题1"),
            ...     AIMessage(content="回答1"),
            ...     HumanMessage(content="问题2"),
            ...     AIMessage(content="回答2"),
            ...     HumanMessage(content="问题3"),
            ...     AIMessage(content="回答3"),
            ...     HumanMessage(content="问题4"),
            ...     AIMessage(content="回答4"),
            ... ]
            >>> trimmed = manager.trim_conversation(messages)
            >>> # 保留 SystemMessage + 最近3轮对话（6条消息）
            >>> len(trimmed) == 7  # 1 system + 3*2 turns
            True

        Reference:
            - research.md:69-78 - trim_messages 使用示例
        """
        if not messages:
            return []

        # 分离系统消息和其他消息
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]

        # 计算要保留的消息数量（每轮对话2条消息）
        max_messages = self.max_turns * 2

        # 根据策略修剪非系统消息
        if strategy == "last":
            # 保留最新的 N 轮对话
            if len(non_system_messages) > max_messages:
                trimmed_non_system = non_system_messages[-max_messages:]
            else:
                trimmed_non_system = non_system_messages
        else:  # "first"
            # 保留最早的 N 轮对话
            if len(non_system_messages) > max_messages:
                trimmed_non_system = non_system_messages[:max_messages]
            else:
                trimmed_non_system = non_system_messages

        # 重新组合：系统消息 + 修剪后的对话
        if self.include_system:
            result = system_messages + trimmed_non_system
        else:
            result = trimmed_non_system

        return result

    def summarize_early_messages(
        self,
        messages: list[BaseMessage],
        current_summary: Optional[str] = None
    ) -> Optional[str]:
        """将早期对话压缩为摘要文本

        当对话超过滑动窗口时，将窗口外的早期消息压缩为简洁摘要。
        摘要内容包括:
        - 讨论的主要问题
        - 已尝试的解决方案
        - 当前的进展状态

        Args:
            messages: 需要摘要的消息列表（通常是窗口外的早期消息）
            current_summary: 已有的摘要（用于增量更新）

        Returns:
            生成的摘要文本，如果 LLM 未配置或消息为空则返回 None

        Examples:
            >>> from langchain_community.chat_models import ChatTongyi
            >>> llm = ChatTongyi(model="qwen-plus")
            >>> manager = MemoryManager(llm=llm)
            >>> early_messages = [
            ...     HumanMessage(content="模型加载失败"),
            ...     AIMessage(content="请检查 CUDA 版本"),
            ...     HumanMessage(content="CUDA 11.7"),
            ...     AIMessage(content="建议升级到 PyTorch 2.0"),
            ... ]
            >>> summary = manager.summarize_early_messages(early_messages)
            >>> "CUDA" in summary and "PyTorch" in summary
            True

        Reference:
            - data-model.md:42 - conversation_summary 字段
            - spec.md:161 - 更早轮次压缩为摘要
        """
        if not self.llm:
            # LLM 未配置，无法生成摘要
            return current_summary

        if not messages:
            # 没有需要摘要的消息
            return current_summary

        # 过滤出非系统消息
        non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]
        if not non_system_messages:
            return current_summary

        # 构建摘要 Prompt
        conversation_text = self._format_messages_for_summary(non_system_messages)

        if current_summary:
            # 增量更新现有摘要
            prompt = f"""已有对话摘要:
{current_summary}

新增对话内容:
{conversation_text}

请更新摘要,保留重要信息:
1. 讨论的主要问题
2. 已尝试的解决方案
3. 当前的进展状态

要求: 摘要应简洁明了,不超过200字。"""
        else:
            # 首次生成摘要
            prompt = f"""请为以下对话生成简洁摘要:

{conversation_text}

摘要应包含:
1. 讨论的主要问题
2. 已尝试的解决方案
3. 当前的进展状态

要求: 摘要应简洁明了,不超过200字。"""

        try:
            # 调用 LLM 生成摘要
            response = self.llm.invoke([HumanMessage(content=prompt)])
            summary = response.content.strip()
            return summary
        except Exception as e:
            # 生成摘要失败，返回现有摘要
            print(f"生成对话摘要失败: {e}")
            return current_summary

    def _format_messages_for_summary(self, messages: list[BaseMessage]) -> str:
        """将消息列表格式化为可读文本

        Args:
            messages: 消息列表

        Returns:
            格式化的对话文本
        """
        formatted_lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "用户"
            elif isinstance(msg, AIMessage):
                role = "Agent"
            else:
                role = "系统"

            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            formatted_lines.append(f"{role}: {content}")

        return "\n".join(formatted_lines)

    def get_conversation_window(
        self,
        messages: list[BaseMessage],
        summary: Optional[str] = None
    ) -> list[BaseMessage]:
        """获取优化的对话窗口（摘要 + 最近对话）

        将早期对话摘要和最近的完整对话合并，作为 LLM 的上下文输入。

        Args:
            messages: 完整消息历史
            summary: 早期对话摘要（可选）

        Returns:
            优化后的消息列表，包含:
            - 系统消息（如果有）
            - 摘要消息（如果有）
            - 最近 N 轮完整对话

        Examples:
            >>> manager = MemoryManager(max_turns=2)
            >>> messages = [
            ...     SystemMessage(content="你是助手"),
            ...     HumanMessage(content="问题1"),
            ...     AIMessage(content="回答1"),
            ...     HumanMessage(content="问题2"),
            ...     AIMessage(content="回答2"),
            ...     HumanMessage(content="问题3"),
            ...     AIMessage(content="回答3"),
            ... ]
            >>> summary = "早期讨论了问题1和问题2"
            >>> window = manager.get_conversation_window(messages, summary)
            >>> # 包含: SystemMessage + 摘要 + 最近2轮对话
            >>> len(window) == 6  # 1 system + 1 summary + 2*2 turns
            True
        """
        # 修剪整个消息列表（会自动处理系统消息）
        trimmed_messages = self.trim_conversation(messages)

        # 如果没有摘要，直接返回修剪后的消息
        if not summary:
            return trimmed_messages

        # 如果有摘要，插入摘要消息
        # 分离系统消息和其他消息
        system_messages = [m for m in trimmed_messages if isinstance(m, SystemMessage)]
        non_system_messages = [m for m in trimmed_messages if not isinstance(m, SystemMessage)]

        # 构建最终窗口
        result = []

        # 1. 添加系统消息
        result.extend(system_messages)

        # 2. 添加摘要
        summary_message = SystemMessage(
            content=f"早期对话摘要:\n{summary}\n\n以下是最近的对话历史:"
        )
        result.append(summary_message)

        # 3. 添加最近对话
        result.extend(non_system_messages)

        return result

    def should_generate_summary(self, messages: list[BaseMessage]) -> bool:
        """判断是否需要生成对话摘要

        当非系统消息数量超过滑动窗口大小时，需要生成摘要。

        Args:
            messages: 消息列表

        Returns:
            是否需要生成摘要

        Examples:
            >>> manager = MemoryManager(max_turns=3)
            >>> messages = [HumanMessage(content=f"问题{i}") for i in range(10)]
            >>> manager.should_generate_summary(messages)
            True
            >>> short_messages = [HumanMessage(content="问题1")]
            >>> manager.should_generate_summary(short_messages)
            False
        """
        non_system_count = sum(1 for m in messages if not isinstance(m, SystemMessage))
        # 每轮对话 2 条消息，超过窗口时需要摘要
        return non_system_count > (self.max_turns * 2)

    def get_early_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """获取需要摘要的早期消息（窗口外的消息）

        Args:
            messages: 完整消息列表

        Returns:
            窗口外的早期消息列表

        Examples:
            >>> manager = MemoryManager(max_turns=2)
            >>> messages = [
            ...     HumanMessage(content="问题1"),
            ...     AIMessage(content="回答1"),
            ...     HumanMessage(content="问题2"),
            ...     AIMessage(content="回答2"),
            ...     HumanMessage(content="问题3"),
            ...     AIMessage(content="回答3"),
            ... ]
            >>> early = manager.get_early_messages(messages)
            >>> len(early) == 2  # 第1轮对话（2条消息）
            True
        """
        non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]

        # 计算窗口内的消息数量
        window_size = self.max_turns * 2

        if len(non_system_messages) <= window_size:
            # 没有超出窗口的消息
            return []

        # 返回窗口外的早期消息
        early_count = len(non_system_messages) - window_size
        return non_system_messages[:early_count]

    def get_statistics(self, messages: list[BaseMessage]) -> dict:
        """获取对话历史统计信息

        Args:
            messages: 消息列表

        Returns:
            统计信息字典，包含:
            - total_messages: 总消息数
            - system_messages: 系统消息数
            - user_messages: 用户消息数
            - ai_messages: AI 消息数
            - turn_count: 对话轮次（用户消息数）
            - needs_summary: 是否需要摘要
        """
        system_count = sum(1 for m in messages if isinstance(m, SystemMessage))
        user_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        ai_count = sum(1 for m in messages if isinstance(m, AIMessage))

        return {
            "total_messages": len(messages),
            "system_messages": system_count,
            "user_messages": user_count,
            "ai_messages": ai_count,
            "turn_count": user_count,
            "needs_summary": self.should_generate_summary(messages),
            "max_turns": self.max_turns,
            "max_tokens": self.max_tokens
        }


# ============================================================================
# 便捷函数
# ============================================================================

def create_memory_manager(
    llm: Optional[BaseChatModel] = None,
    max_turns: int = 10,
    max_tokens: int = 4000
) -> MemoryManager:
    """创建默认配置的 MemoryManager 实例

    Args:
        llm: LLM 实例（用于生成摘要）
        max_turns: 滑动窗口大小（默认10轮）
        max_tokens: Token 限制（默认4000）

    Returns:
        配置好的 MemoryManager 实例

    Examples:
        >>> manager = create_memory_manager()
        >>> manager.max_turns
        10
        >>> manager.max_tokens
        4000
    """
    return MemoryManager(
        llm=llm,
        max_turns=max_turns,
        max_tokens=max_tokens,
        include_system=True
    )
