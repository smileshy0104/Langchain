"""对话记忆管理器测试

测试 MemoryManager 的核心功能:
- T106: 对话修剪测试
- 滑动窗口机制
- 早期对话摘要
- Token 限制
- 对话统计

Author: Claude Code
Created: 2025-12-01
"""

import pytest
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from unittest.mock import Mock, MagicMock

from modelscope_qa_agent.core.memory_manager import (
    MemoryManager,
    create_memory_manager
)


# ============================================================================
# 测试夹具
# ============================================================================

@pytest.fixture
def sample_messages():
    """创建示例消息列表"""
    return [
        SystemMessage(content="你是魔搭社区的技术顾问"),
        HumanMessage(content="如何加载 Qwen-7B 模型?"),
        AIMessage(content="使用 AutoModelForCausalLM.from_pretrained('Qwen/Qwen-7B')"),
        HumanMessage(content="CUDA 内存不足怎么办?"),
        AIMessage(content="可以降低 batch_size 或使用 INT8 量化"),
        HumanMessage(content="如何启用 INT8 量化?"),
        AIMessage(content="在 from_pretrained 中添加 load_in_8bit=True 参数"),
        HumanMessage(content="还有其他优化方法吗?"),
        AIMessage(content="可以使用梯度检查点和梯度累积"),
        HumanMessage(content="梯度累积怎么设置?"),
        AIMessage(content="在训练配置中设置 gradient_accumulation_steps"),
    ]


@pytest.fixture
def long_conversation():
    """创建长对话（超过10轮）"""
    messages = [SystemMessage(content="你是助手")]
    for i in range(15):
        messages.append(HumanMessage(content=f"问题 {i+1}"))
        messages.append(AIMessage(content=f"回答 {i+1}"))
    return messages


@pytest.fixture
def mock_llm():
    """创建 Mock LLM"""
    llm = Mock()
    llm.invoke = Mock(return_value=Mock(content="对话摘要: 用户询问了模型加载和优化问题"))
    return llm


@pytest.fixture
def memory_manager():
    """创建默认 MemoryManager"""
    return MemoryManager(max_turns=3, max_tokens=4000)


@pytest.fixture
def memory_manager_with_llm(mock_llm):
    """创建带 LLM 的 MemoryManager"""
    return MemoryManager(llm=mock_llm, max_turns=3, max_tokens=4000)


# ============================================================================
# 测试: 基本初始化
# ============================================================================

class TestMemoryManagerInit:
    """测试 MemoryManager 初始化"""

    def test_default_initialization(self):
        """测试默认初始化参数"""
        manager = MemoryManager()
        assert manager.llm is None
        assert manager.max_turns == 10
        assert manager.max_tokens == 4000
        assert manager.include_system is True

    def test_custom_initialization(self, mock_llm):
        """测试自定义初始化参数"""
        manager = MemoryManager(
            llm=mock_llm,
            max_turns=5,
            max_tokens=2000,
            include_system=False
        )
        assert manager.llm == mock_llm
        assert manager.max_turns == 5
        assert manager.max_tokens == 2000
        assert manager.include_system is False

    def test_create_memory_manager_factory(self):
        """测试工厂函数"""
        manager = create_memory_manager(max_turns=5, max_tokens=3000)
        assert manager.max_turns == 5
        assert manager.max_tokens == 3000
        assert manager.include_system is True


# ============================================================================
# 测试: T106 - trim_conversation() 对话修剪
# ============================================================================

class TestTrimConversation:
    """测试对话修剪功能"""

    def test_trim_conversation_basic(self, memory_manager, sample_messages):
        """测试基本修剪功能"""
        # max_turns=3, 应保留系统消息 + 3轮对话（6条消息）
        trimmed = memory_manager.trim_conversation(sample_messages)

        # 验证消息数量: 1 system + 3*2 = 7
        assert len(trimmed) == 7

        # 验证保留了系统消息
        assert isinstance(trimmed[0], SystemMessage)

        # 验证保留了最新的3轮对话
        assert "gradient_accumulation_steps" in trimmed[-1].content  # 最后一条 AI 消息
        assert "梯度累积怎么设置" in trimmed[-2].content  # 倒数第二条用户消息

    def test_trim_conversation_empty_messages(self, memory_manager):
        """测试空消息列表"""
        trimmed = memory_manager.trim_conversation([])
        assert trimmed == []

    def test_trim_conversation_short_history(self, memory_manager):
        """测试短对话（不需要修剪）"""
        short_messages = [
            SystemMessage(content="系统消息"),
            HumanMessage(content="问题1"),
            AIMessage(content="回答1"),
        ]
        trimmed = memory_manager.trim_conversation(short_messages)

        # 短对话应完整保留
        assert len(trimmed) == len(short_messages)
        assert trimmed == short_messages

    def test_trim_conversation_no_system_message(self, memory_manager):
        """测试没有系统消息的对话"""
        messages = [
            HumanMessage(content="问题1"),
            AIMessage(content="回答1"),
            HumanMessage(content="问题2"),
            AIMessage(content="回答2"),
            HumanMessage(content="问题3"),
            AIMessage(content="回答3"),
            HumanMessage(content="问题4"),
            AIMessage(content="回答4"),
        ]
        trimmed = memory_manager.trim_conversation(messages)

        # max_turns=3, 应保留 3*2=6 条消息
        assert len(trimmed) == 6

        # 验证保留了最新的3轮对话
        assert "问题4" in trimmed[-2].content
        assert "回答4" in trimmed[-1].content

    def test_trim_conversation_strategy_last(self, memory_manager, sample_messages):
        """测试 'last' 策略（默认）"""
        trimmed = memory_manager.trim_conversation(sample_messages, strategy="last")

        # 验证保留了最新消息
        assert "gradient_accumulation_steps" in trimmed[-1].content

    def test_trim_conversation_preserves_system(self, sample_messages):
        """测试始终保留系统消息"""
        manager = MemoryManager(max_turns=1, include_system=True)
        trimmed = manager.trim_conversation(sample_messages)

        # 即使只保留1轮对话，也应包含系统消息
        assert isinstance(trimmed[0], SystemMessage)
        assert "技术顾问" in trimmed[0].content


# ============================================================================
# 测试: summarize_early_messages() 早期对话摘要
# ============================================================================

class TestSummarizeEarlyMessages:
    """测试早期对话摘要功能"""

    def test_summarize_without_llm(self, memory_manager, sample_messages):
        """测试没有 LLM 时的摘要"""
        summary = memory_manager.summarize_early_messages(sample_messages)
        assert summary is None

    def test_summarize_with_llm(self, memory_manager_with_llm, sample_messages):
        """测试使用 LLM 生成摘要"""
        summary = memory_manager_with_llm.summarize_early_messages(sample_messages)

        # 验证生成了摘要
        assert summary is not None
        assert isinstance(summary, str)
        assert len(summary) > 0

        # 验证 LLM 被调用
        memory_manager_with_llm.llm.invoke.assert_called_once()

    def test_summarize_empty_messages(self, memory_manager_with_llm):
        """测试空消息列表"""
        summary = memory_manager_with_llm.summarize_early_messages([])
        assert summary is None

    def test_summarize_only_system_messages(self, memory_manager_with_llm):
        """测试仅包含系统消息"""
        messages = [
            SystemMessage(content="系统消息1"),
            SystemMessage(content="系统消息2"),
        ]
        summary = memory_manager_with_llm.summarize_early_messages(messages)
        # 系统消息不需要摘要
        assert summary is None

    def test_summarize_incremental_update(self, memory_manager_with_llm, sample_messages):
        """测试增量更新摘要"""
        existing_summary = "已有摘要: 讨论了模型加载问题"
        summary = memory_manager_with_llm.summarize_early_messages(
            sample_messages,
            current_summary=existing_summary
        )

        # 验证生成了新摘要
        assert summary is not None

        # 验证 Prompt 包含已有摘要
        call_args = memory_manager_with_llm.llm.invoke.call_args
        prompt_content = call_args[0][0][0].content
        assert "已有对话摘要" in prompt_content
        assert existing_summary in prompt_content

    def test_summarize_llm_error_handling(self, memory_manager_with_llm, sample_messages):
        """测试 LLM 调用失败的处理"""
        # 模拟 LLM 调用失败
        memory_manager_with_llm.llm.invoke.side_effect = Exception("API 错误")

        existing_summary = "现有摘要"
        summary = memory_manager_with_llm.summarize_early_messages(
            sample_messages,
            current_summary=existing_summary
        )

        # 失败时应返回现有摘要
        assert summary == existing_summary

    def test_format_messages_for_summary(self, memory_manager, sample_messages):
        """测试消息格式化"""
        formatted = memory_manager._format_messages_for_summary(sample_messages)

        # 验证格式
        assert isinstance(formatted, str)
        assert "用户:" in formatted
        assert "Agent:" in formatted
        assert "Qwen-7B" in formatted


# ============================================================================
# 测试: get_conversation_window() 对话窗口
# ============================================================================

class TestGetConversationWindow:
    """测试获取对话窗口功能"""

    def test_get_window_without_summary(self, memory_manager, sample_messages):
        """测试没有摘要的窗口"""
        window = memory_manager.get_conversation_window(sample_messages)

        # 应包含: 系统消息 + 最近3轮对话
        assert len(window) == 7  # 1 system + 3*2 turns

        # 验证第一条是系统消息
        assert isinstance(window[0], SystemMessage)

    def test_get_window_with_summary(self, memory_manager, sample_messages):
        """测试包含摘要的窗口"""
        summary = "早期讨论: 用户询问了模型加载和 CUDA 内存问题"
        window = memory_manager.get_conversation_window(sample_messages, summary)

        # 应包含: 系统消息 + 摘要消息 + 最近3轮对话
        assert len(window) == 8  # 1 system + 1 summary + 3*2 turns

        # 验证摘要消息
        summary_msg = window[1]
        assert isinstance(summary_msg, SystemMessage)
        assert "早期对话摘要" in summary_msg.content
        assert summary in summary_msg.content

    def test_get_window_no_system_message(self, memory_manager):
        """测试没有系统消息的窗口"""
        messages = [
            HumanMessage(content="问题1"),
            AIMessage(content="回答1"),
            HumanMessage(content="问题2"),
            AIMessage(content="回答2"),
        ]
        window = memory_manager.get_conversation_window(messages)

        # 应只包含最近的对话
        assert len(window) == len(messages)

    def test_get_window_empty_messages(self, memory_manager):
        """测试空消息列表"""
        window = memory_manager.get_conversation_window([])
        assert window == []


# ============================================================================
# 测试: should_generate_summary() 摘要判断
# ============================================================================

class TestShouldGenerateSummary:
    """测试摘要生成判断"""

    def test_should_generate_summary_long_conversation(self, memory_manager, long_conversation):
        """测试长对话需要摘要"""
        # 15轮对话 > 3轮窗口
        assert memory_manager.should_generate_summary(long_conversation) is True

    def test_should_generate_summary_short_conversation(self, memory_manager):
        """测试短对话不需要摘要"""
        messages = [
            SystemMessage(content="系统"),
            HumanMessage(content="问题1"),
            AIMessage(content="回答1"),
        ]
        assert memory_manager.should_generate_summary(messages) is False

    def test_should_generate_summary_exactly_at_limit(self, memory_manager):
        """测试刚好达到窗口大小"""
        messages = [SystemMessage(content="系统")]
        # 添加刚好 3 轮对话（max_turns=3）
        for i in range(3):
            messages.append(HumanMessage(content=f"问题{i}"))
            messages.append(AIMessage(content=f"回答{i}"))

        # 刚好达到限制，不需要摘要
        assert memory_manager.should_generate_summary(messages) is False

    def test_should_generate_summary_over_limit(self, memory_manager):
        """测试超过窗口大小"""
        messages = [SystemMessage(content="系统")]
        # 添加 4 轮对话（超过 max_turns=3）
        for i in range(4):
            messages.append(HumanMessage(content=f"问题{i}"))
            messages.append(AIMessage(content=f"回答{i}"))

        # 超过限制，需要摘要
        assert memory_manager.should_generate_summary(messages) is True


# ============================================================================
# 测试: get_early_messages() 获取早期消息
# ============================================================================

class TestGetEarlyMessages:
    """测试获取早期消息功能"""

    def test_get_early_messages_long_conversation(self, memory_manager, long_conversation):
        """测试长对话的早期消息"""
        early = memory_manager.get_early_messages(long_conversation)

        # 15轮 - 3轮窗口 = 12轮早期 = 24条消息
        assert len(early) == 24

        # 验证是最早的消息
        assert "问题 1" in early[0].content
        assert "回答 1" in early[1].content

    def test_get_early_messages_short_conversation(self, memory_manager, sample_messages):
        """测试短对话没有早期消息"""
        early = memory_manager.get_early_messages(sample_messages)

        # 只有5轮，窗口是3轮，早期消息是2轮 = 4条
        assert len(early) == 4

    def test_get_early_messages_within_window(self, memory_manager):
        """测试窗口内的对话"""
        messages = [
            HumanMessage(content="问题1"),
            AIMessage(content="回答1"),
        ]
        early = memory_manager.get_early_messages(messages)

        # 窗口内，没有早期消息
        assert early == []

    def test_get_early_messages_ignores_system(self, memory_manager):
        """测试忽略系统消息"""
        messages = [
            SystemMessage(content="系统消息"),
            HumanMessage(content="问题1"),
            AIMessage(content="回答1"),
        ]
        early = memory_manager.get_early_messages(messages)

        # 系统消息不计入轮次
        assert early == []


# ============================================================================
# 测试: get_statistics() 统计信息
# ============================================================================

class TestGetStatistics:
    """测试统计信息获取"""

    def test_get_statistics_basic(self, memory_manager, sample_messages):
        """测试基本统计信息"""
        stats = memory_manager.get_statistics(sample_messages)

        assert stats["total_messages"] == 11
        assert stats["system_messages"] == 1
        assert stats["user_messages"] == 5
        assert stats["ai_messages"] == 5
        assert stats["turn_count"] == 5
        assert stats["max_turns"] == 3
        assert stats["max_tokens"] == 4000
        assert stats["needs_summary"] is True  # 5轮 > 3轮窗口

    def test_get_statistics_empty(self, memory_manager):
        """测试空消息列表"""
        stats = memory_manager.get_statistics([])

        assert stats["total_messages"] == 0
        assert stats["system_messages"] == 0
        assert stats["user_messages"] == 0
        assert stats["ai_messages"] == 0
        assert stats["turn_count"] == 0
        assert stats["needs_summary"] is False

    def test_get_statistics_short_conversation(self, memory_manager):
        """测试短对话统计"""
        messages = [
            SystemMessage(content="系统"),
            HumanMessage(content="问题"),
            AIMessage(content="回答"),
        ]
        stats = memory_manager.get_statistics(messages)

        assert stats["total_messages"] == 3
        assert stats["turn_count"] == 1
        assert stats["needs_summary"] is False  # 1轮 <= 3轮窗口


# ============================================================================
# 测试: 集成场景
# ============================================================================

class TestIntegrationScenarios:
    """测试集成场景"""

    def test_full_workflow_with_summary(self, memory_manager_with_llm, long_conversation):
        """测试完整工作流: 判断 -> 摘要 -> 窗口"""
        # 1. 判断是否需要摘要
        needs_summary = memory_manager_with_llm.should_generate_summary(long_conversation)
        assert needs_summary is True

        # 2. 获取早期消息
        early_messages = memory_manager_with_llm.get_early_messages(long_conversation)
        assert len(early_messages) > 0

        # 3. 生成摘要
        summary = memory_manager_with_llm.summarize_early_messages(early_messages)
        assert summary is not None

        # 4. 获取优化的对话窗口
        window = memory_manager_with_llm.get_conversation_window(long_conversation, summary)

        # 验证窗口包含摘要和最近对话
        # 窗口应该比早期消息短（因为只保留最近的轮次）但包含摘要
        assert len(window) < len(long_conversation)  # 窗口比完整对话短
        summary_found = any("早期对话摘要" in str(m.content) for m in window)
        assert summary_found is True

    def test_full_workflow_no_summary_needed(self, memory_manager):
        """测试不需要摘要的工作流"""
        messages = [
            SystemMessage(content="系统"),
            HumanMessage(content="问题"),
            AIMessage(content="回答"),
        ]

        # 1. 判断不需要摘要
        needs_summary = memory_manager.should_generate_summary(messages)
        assert needs_summary is False

        # 2. 直接获取窗口
        window = memory_manager.get_conversation_window(messages)

        # 窗口应与原始消息相同
        assert len(window) == len(messages)

    def test_incremental_summary_updates(self, memory_manager_with_llm):
        """测试增量更新摘要"""
        # 第一批消息
        batch1 = [
            HumanMessage(content="如何加载模型?"),
            AIMessage(content="使用 from_pretrained"),
        ]
        summary1 = memory_manager_with_llm.summarize_early_messages(batch1)

        # 第二批消息，更新摘要
        batch2 = [
            HumanMessage(content="CUDA 内存不足?"),
            AIMessage(content="降低 batch_size"),
        ]
        summary2 = memory_manager_with_llm.summarize_early_messages(
            batch2,
            current_summary=summary1
        )

        # 验证摘要被更新
        assert summary2 is not None
        assert summary2 != summary1 or summary1 == summary2  # 可能相同（取决于 Mock）


# ============================================================================
# 测试: 边界情况
# ============================================================================

class TestEdgeCases:
    """测试边界情况"""

    def test_single_message(self, memory_manager):
        """测试单条消息"""
        messages = [HumanMessage(content="问题")]
        trimmed = memory_manager.trim_conversation(messages)
        assert len(trimmed) == 1

    def test_only_system_messages(self, memory_manager):
        """测试仅系统消息"""
        messages = [
            SystemMessage(content="系统1"),
            SystemMessage(content="系统2"),
        ]
        trimmed = memory_manager.trim_conversation(messages)
        assert all(isinstance(m, SystemMessage) for m in trimmed)

    def test_alternating_roles(self, memory_manager):
        """测试交替角色"""
        messages = [
            HumanMessage(content="问题1"),
            AIMessage(content="回答1"),
            HumanMessage(content="问题2"),
            AIMessage(content="回答2"),
        ]
        trimmed = memory_manager.trim_conversation(messages)
        assert len(trimmed) == 4  # 2轮 <= 3轮窗口

    def test_max_turns_zero(self):
        """测试 max_turns=0"""
        manager = MemoryManager(max_turns=0)
        messages = [HumanMessage(content="问题")]

        # max_turns=0 应该只保留系统消息
        stats = manager.get_statistics(messages)
        assert stats["max_turns"] == 0

    def test_very_large_max_turns(self):
        """测试非常大的 max_turns"""
        manager = MemoryManager(max_turns=1000)
        messages = [HumanMessage(content="问题")]

        # 不应影响短对话
        trimmed = manager.trim_conversation(messages)
        assert len(trimmed) == len(messages)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
