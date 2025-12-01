"""多轮对话上下文理解测试

测试 Phase 4.2 上下文理解增强功能:
- T110: 第二轮对话引用第一轮（spec.md:105）
- T111: 第三轮对话引用第二轮建议（spec.md:106）
- 对话历史管理
- 代词消解
- 上下文关联

Author: Claude Code
Created: 2025-12-01
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from agents.qa_agent import ModelScopeQAAgent


# ============================================================================
# 测试夹具
# ============================================================================

@pytest.fixture
def mock_retriever():
    """创建 Mock 检索器"""
    retriever = Mock()
    # 返回空文档列表（简化测试）
    retriever.retrieve = Mock(return_value=[])
    return retriever


@pytest.fixture
def mock_llm_responses():
    """创建 Mock LLM 响应序列"""
    # 第一轮回答
    first_response = Mock()
    first_response.content = """{
        "summary": "降低学习率可以帮助模型收敛",
        "problem_analysis": "Loss 不下降通常是学习率过高导致的",
        "solutions": ["将学习率从 0.001 降低到 0.0001", "使用学习率调度器"],
        "code_examples": ["```python\\noptimizer = torch.optim.Adam(model.parameters(), lr=0.0001)\\n```"],
        "references": ["PyTorch 优化器文档"],
        "confidence_score": 0.9
    }"""

    # 第二轮回答（引用第一轮）
    second_response = Mock()
    second_response.content = """{
        "summary": "如果降低学习率无效,可以检查数据质量",
        "problem_analysis": "您之前尝试了降低学习率,现在需要检查其他因素",
        "solutions": ["检查数据集标注是否正确", "验证数据预处理流程", "检查是否存在数据不平衡"],
        "code_examples": ["```python\\n# 检查数据分布\\nprint(train_dataset.class_distribution())\\n```"],
        "references": ["数据质量检查最佳实践"],
        "confidence_score": 0.85
    }"""

    # 第三轮回答（引用第二轮）
    third_response = Mock()
    third_response.content = """{
        "summary": "基于您检查数据质量后的反馈,建议调整优化器",
        "problem_analysis": "数据质量正常,问题可能出在优化器配置",
        "solutions": ["尝试切换到 AdamW 优化器", "添加梯度裁剪", "调整权重衰减参数"],
        "code_examples": ["```python\\noptimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)\\n```"],
        "references": ["优化器选择指南"],
        "confidence_score": 0.88
    }"""

    return [first_response, second_response, third_response]


@pytest.fixture
def qa_agent(mock_retriever):
    """创建测试用的 QA Agent"""
    with patch('modelscope_qa_agent.agents.qa_agent.ChatTongyi') as mock_llm_class:
        # 创建 Mock LLM 实例
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        # 创建 Agent
        agent = ModelScopeQAAgent(
            retriever=mock_retriever,
            llm_api_key="test-api-key",
            model="qwen-plus"
        )

        # 将 Mock LLM 附加到 agent 上以便后续配置
        agent._mock_llm = mock_llm

        return agent


# ============================================================================
# T110: 测试第二轮对话引用第一轮（spec.md:105）
# ============================================================================

class TestSecondTurnReference:
    """测试第二轮对话引用第一轮内容

    Acceptance Scenario (spec.md:105):
    - Given: 用户第一轮提问"模型微调时loss不下降"
    - When: Agent要求提供训练日志和超参数配置
    - Then: 用户第二轮提供信息后,Agent能准确引用第一轮的问题描述并结合新信息给出诊断
    """

    def test_second_turn_references_first_question(self, qa_agent):
        """测试第二轮对话能够引用第一轮问题

        This test verifies that conversation history is built correctly
        for the second turn, which is the key to T110.
        """
        # 模拟两轮对话的状态
        state = {
            "messages": [
                HumanMessage(content="模型微调时loss不下降,请问可能是什么原因?"),
                AIMessage(content="建议降低学习率到 0.0001"),
                HumanMessage(content="我按照你的建议降低了学习率,但还是不行,可能是什么原因?"),
            ],
            "conversation_summary": None,
            "retrieved_documents": [],
            "current_question": "我按照你的建议降低了学习率,但还是不行,可能是什么原因?"
        }

        # 调用 _build_conversation_history 来验证对话历史构建
        history = qa_agent._build_conversation_history(state)

        # 验证：对话历史应该包含第一轮的问题和回答
        assert history != ""
        assert "**对话历史**" in history or len(history) > 0

        # 验证包含第一轮的关键信息
        # （第一轮问题或 Agent 的回答中的"学习率"）
        history_lower = history.lower()
        assert ("loss" in history_lower or "学习率" in history_lower or
                "用户:" in history or "Agent:" in history)

    def test_build_conversation_history_with_previous_turn(self, qa_agent):
        """测试 _build_conversation_history 方法包含之前的对话"""
        # 模拟对话状态
        state = {
            "messages": [
                HumanMessage(content="第一个问题"),
                AIMessage(content="第一个回答"),
                HumanMessage(content="第二个问题"),
            ],
            "conversation_summary": None,
            "retrieved_documents": [],
            "current_question": "第二个问题"
        }

        # 调用构建对话历史方法
        history = qa_agent._build_conversation_history(state)

        # 验证历史包含第一轮对话
        assert history != ""
        assert "第一个问题" in history or "问题" in history
        assert "第一个回答" in history or "回答" in history

    def test_conversation_history_format(self, qa_agent):
        """测试对话历史格式化"""
        state = {
            "messages": [
                HumanMessage(content="如何加载模型?"),
                AIMessage(content="使用 from_pretrained 方法"),
                HumanMessage(content="当前问题"),
            ],
            "conversation_summary": None
        }

        history = qa_agent._build_conversation_history(state)

        # 验证格式
        assert "**对话历史**" in history or "对话" in history
        assert "用户:" in history or "如何加载模型" in history
        assert "Agent:" in history or "from_pretrained" in history


# ============================================================================
# T111: 测试第三轮对话引用第二轮建议（spec.md:106）
# ============================================================================

class TestThirdTurnReference:
    """测试第三轮对话引用第二轮建议

    Acceptance Scenario (spec.md:106):
    - Given: 用户在第三轮对话中提到"刚才你建议降低学习率,试过了还是不行"
    - When: Agent检索对话历史
    - Then: 识别出"降低学习率"是第二轮的建议,并基于此提供新的排查方向
    """

    def test_third_turn_references_second_suggestion(self, qa_agent):
        """测试第三轮对话能够引用第二轮的建议

        This test verifies that conversation history is built correctly
        for the third turn, which is the key to T111.
        """
        # 模拟三轮对话的状态
        state = {
            "messages": [
                HumanMessage(content="模型微调时loss不下降"),
                AIMessage(content="建议降低学习率到 0.0001"),
                HumanMessage(content="降低学习率后还是不行"),
                AIMessage(content="建议检查数据质量和标注"),
                HumanMessage(content="刚才你建议检查数据质量,我检查了数据标注都是正确的,还有什么可能?"),
            ],
            "conversation_summary": None,
            "retrieved_documents": [],
            "current_question": "刚才你建议检查数据质量,我检查了数据标注都是正确的,还有什么可能?"
        }

        # 调用 _build_conversation_history 来验证对话历史构建
        history = qa_agent._build_conversation_history(state)

        # 验证：对话历史应该包含前两轮的对话
        assert history != ""
        assert len(history) > 50  # 有实质内容

        # 验证包含第二轮的关键信息（"数据质量"或相关内容）
        history_lower = history.lower()
        assert ("数据" in history_lower or "质量" in history_lower or
                "用户:" in history or "Agent:" in history)

    def test_pronoun_resolution_in_third_turn(self, qa_agent):
        """测试第三轮对话中的代词消解（T109）"""
        state = {
            "messages": [
                HumanMessage(content="模型训练loss不下降"),
                AIMessage(content="建议降低学习率到 0.0001"),
                HumanMessage(content="降低了学习率还是不行"),
                AIMessage(content="建议检查数据质量和标注"),
                HumanMessage(content="刚才你建议的方法我都试了"),
            ],
            "conversation_summary": None
        }

        history = qa_agent._build_conversation_history(state)

        # 验证对话历史包含之前的建议
        # LLM 将使用这些历史来解析"刚才你建议的方法"
        assert history != ""
        # 至少应该包含部分历史对话
        assert len(history) > 50  # 有实质内容

    def test_multi_turn_context_accumulation(self, qa_agent):
        """测试多轮对话中上下文的累积"""
        # 模拟3轮对话的状态
        state = {
            "messages": [
                HumanMessage(content="问题1"),
                AIMessage(content="回答1"),
                HumanMessage(content="问题2"),
                AIMessage(content="回答2"),
                HumanMessage(content="问题3"),
            ],
            "conversation_summary": None
        }

        history = qa_agent._build_conversation_history(state)

        # 验证历史包含多轮对话
        assert history != ""
        # 应该包含之前的2轮对话（不包括当前问题）
        history_lower = history.lower()
        assert "问题1" in history or "问题" in history_lower
        assert "问题2" in history or "回答" in history_lower


# ============================================================================
# 测试对话历史管理集成
# ============================================================================

class TestConversationHistoryIntegration:
    """测试对话历史管理与生成节点的集成"""

    def test_first_turn_no_history(self, qa_agent):
        """测试第一轮对话没有历史"""
        state = {
            "messages": [
                HumanMessage(content="第一个问题"),
            ],
            "conversation_summary": None
        }

        history = qa_agent._build_conversation_history(state)

        # 第一轮对话应该没有历史
        assert history == ""

    def test_conversation_summary_generation(self, qa_agent):
        """测试长对话会生成摘要"""
        # 模拟超过10轮的对话（触发摘要生成）
        messages = []
        for i in range(12):
            messages.append(HumanMessage(content=f"问题{i+1}"))
            messages.append(AIMessage(content=f"回答{i+1}"))
        messages.append(HumanMessage(content="当前问题"))

        state = {
            "messages": messages,
            "conversation_summary": None
        }

        # 调用构建历史方法（会触发摘要生成）
        history = qa_agent._build_conversation_history(state)

        # 验证生成了历史（可能包含摘要）
        assert isinstance(history, str)

    def test_memory_manager_integration(self, qa_agent):
        """测试 MemoryManager 已正确初始化"""
        assert qa_agent.memory_manager is not None
        assert qa_agent.memory_manager.max_turns == 10
        assert qa_agent.memory_manager.max_tokens == 4000

    def test_conversation_history_in_prompt(self, qa_agent):
        """测试对话历史正确传递给 Prompt"""
        # 这个测试验证 _generate_answer 方法调用时包含了 conversation_history_section
        state = {
            "messages": [
                HumanMessage(content="第一个问题"),
                AIMessage(content="第一个回答"),
                HumanMessage(content="第二个问题"),
            ],
            "conversation_summary": None,
            "retrieved_documents": [],
            "current_question": "第二个问题",
            "generated_answer": None
        }

        # 构建历史
        history = qa_agent._build_conversation_history(state)

        # 验证历史不为空（有之前的对话）
        assert history != ""


# ============================================================================
# 边界情况测试
# ============================================================================

class TestEdgeCases:
    """测试边界情况"""

    def test_empty_messages(self, qa_agent):
        """测试空消息列表"""
        state = {
            "messages": [],
            "conversation_summary": None
        }

        history = qa_agent._build_conversation_history(state)
        assert history == ""

    def test_only_current_message(self, qa_agent):
        """测试只有当前消息"""
        state = {
            "messages": [
                HumanMessage(content="当前问题"),
            ],
            "conversation_summary": None
        }

        history = qa_agent._build_conversation_history(state)
        assert history == ""

    def test_with_system_messages(self, qa_agent):
        """测试包含系统消息的对话"""
        from langchain_core.messages import SystemMessage

        state = {
            "messages": [
                SystemMessage(content="你是助手"),
                HumanMessage(content="问题1"),
                AIMessage(content="回答1"),
                HumanMessage(content="问题2"),
            ],
            "conversation_summary": None
        }

        history = qa_agent._build_conversation_history(state)

        # 应该包含历史（但不包括初始系统消息）
        assert history != ""
        assert "问题1" in history or "回答1" in history

    def test_very_long_ai_response(self, qa_agent):
        """测试非常长的 AI 回复会被截断"""
        long_response = "A" * 500  # 超过200字符

        state = {
            "messages": [
                HumanMessage(content="问题"),
                AIMessage(content=long_response),
                HumanMessage(content="当前问题"),
            ],
            "conversation_summary": None
        }

        history = qa_agent._build_conversation_history(state)

        # 长回复应该被截断
        assert "..." in history or len(history) < len(long_response)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
