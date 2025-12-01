"""对话进度评估测试

测试 Phase 4.4 对话进度评估功能:
- T117: assess_progress() 评估问题解决进度
- T118: 主动总结已尝试方法和排除的可能性
- T119: 测试场景：对话超过5轮主动总结（对应 spec.md:108）
- T120: 建议是否转向其他排查路径或人工支持

Author: Claude Code
Created: 2025-12-01
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from tools.progress_assessment_tool import ProgressAssessmentTool, ProgressAssessment
from agents.qa_agent import ModelScopeQAAgent


# ============================================================================
# 测试夹具
# ============================================================================

@pytest.fixture
def mock_llm():
    """创建 Mock LLM"""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content="问题尚未解决。已尝试降低学习率、调整batch_size。"
                "建议检查数据质量和模型架构。继续排查。"
    )
    return llm


@pytest.fixture
def progress_tool():
    """创建测试用的进度评估工具"""
    with patch('tools.progress_assessment_tool.ChatTongyi') as mock_tongyi:
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = MagicMock(
            content="问题尚未解决。已尝试降低���习率。建议检查数据质量。"
        )
        mock_tongyi.return_value = mock_llm_instance

        tool = ProgressAssessmentTool(
            llm_api_key="test-key-12345",
            turn_threshold=5
        )
        return tool


@pytest.fixture
def mock_retriever():
    """创建 Mock 检索器"""
    retriever = Mock()
    retriever.retrieve = Mock(return_value=[])
    return retriever


@pytest.fixture
def qa_agent(mock_retriever):
    """创建测试用的 QA Agent"""
    with patch('agents.qa_agent.ChatTongyi') as mock_tongyi, \
         patch('agents.qa_agent.ClarificationTool') as mock_clarif, \
         patch('agents.qa_agent.ProgressAssessmentTool') as mock_progress:

        # 配置 Mock LLM
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = MagicMock(
            content='{"problem_analysis": "测试", "solutions": ["方案1"], '
                    '"code_examples": [], "configuration_notes": [], "warnings": [], '
                    '"references": [], "confidence_score": 0.9}'
        )
        mock_tongyi.return_value = mock_llm_instance

        # 配置 Mock ClarificationTool
        mock_clarif_instance = MagicMock()
        mock_clarif_result = MagicMock()
        mock_clarif_result.needs_clarification = False
        mock_clarif_result.clarification_questions = []
        mock_clarif_instance.check_and_clarify.return_value = mock_clarif_result
        mock_clarif.return_value = mock_clarif_instance

        # 配置 Mock ProgressAssessmentTool
        mock_progress_instance = MagicMock()
        mock_progress_instance.turn_threshold = 5
        mock_progress_instance.should_assess.return_value = False
        mock_progress.return_value = mock_progress_instance

        # 创建 Agent
        agent = ModelScopeQAAgent(
            retriever=mock_retriever,
            llm_api_key="test-key-12345"
        )

        return agent


# ============================================================================
# T117: assess_progress() 评估功能测试
# ============================================================================

class TestProgressAssessmentFunction:
    """测试 assess_progress() 核心评估功能"""

    def test_should_assess_threshold(self, progress_tool):
        """测试轮次阈值判断"""
        # 低于阈值
        assert not progress_tool.should_assess(3)
        assert not progress_tool.should_assess(4)

        # 等于阈值
        assert progress_tool.should_assess(5)

        # 高于阈值
        assert progress_tool.should_assess(6)
        assert progress_tool.should_assess(10)

    def test_assess_progress_basic(self, progress_tool):
        """测试基本进度评估"""
        messages = [
            HumanMessage(content="模型训练loss不下降"),
            AIMessage(content="建议降低学习率到0.0001"),
            HumanMessage(content="降低了还是不行"),
            AIMessage(content="建议调整batch_size到16"),
            HumanMessage(content="还是没效果"),
        ]

        assessment = progress_tool.assess_progress(
            messages=messages,
            turn_count=5,
            current_question="还是没效果"
        )

        # 验证基本字段
        assert isinstance(assessment, ProgressAssessment)
        assert assessment.turn_count == 5
        assert isinstance(assessment.problem_resolved, bool)
        assert 0 <= assessment.confidence_score <= 1
        assert isinstance(assessment.attempted_solutions, list)
        assert assessment.recommendation in ["continue", "pivot", "escalate"]

    def test_assess_progress_low_turn_count(self, progress_tool):
        """测试低轮次评估"""
        messages = [
            HumanMessage(content="问题"),
            AIMessage(content="回答"),
        ]

        assessment = progress_tool.assess_progress(
            messages=messages,
            turn_count=3,
            current_question="问题"
        )

        # 低轮次应该建议继续
        assert assessment.recommendation == "continue"
        assert not assessment.needs_human_support

    def test_assess_progress_high_turn_count(self, progress_tool):
        """测试高轮次评估（应建议升级）"""
        messages = [HumanMessage(content=f"问题{i}") for i in range(20)]

        assessment = progress_tool.assess_progress(
            messages=messages,
            turn_count=10,
            current_question="还是不行"
        )

        # 高轮次应该建议升级
        assert assessment.recommendation == "escalate"
        assert assessment.needs_human_support
        assert assessment.confidence_score < 0.5


# ============================================================================
# T118: 主动总结测试
# ============================================================================

class TestActiveSummarization:
    """测试主动总结已尝试方法和排除可能性"""

    def test_summarize_attempted_solutions(self, progress_tool):
        """测试总结已尝试的方案"""
        messages = [
            HumanMessage(content="模型loss不下降"),
            AIMessage(content="建议降低学习率"),
            HumanMessage(content="降低了"),
            AIMessage(content="建议调整batch_size"),
            HumanMessage(content="调整了"),
            AIMessage(content="建议检查数据质量"),
        ]

        assessment = progress_tool.assess_progress(messages, turn_count=6)

        # 应该提取到尝试的方案
        assert len(assessment.attempted_solutions) > 0
        # 至少应该有某些方案被识别
        assert any(assessment.attempted_solutions)

    def test_identify_excluded_causes(self, progress_tool):
        """测试识别已排除的可能性"""
        messages = [
            HumanMessage(content="模型报错"),
            AIMessage(content="不是学习率问题，已经调整过"),
            HumanMessage(content="还有其他可能吗"),
            AIMessage(content="显存足够，不是显存问题"),
        ]

        assessment = progress_tool.assess_progress(messages, turn_count=6)

        # 应该能识别排除的原因
        assert isinstance(assessment.excluded_causes, list)

    def test_suggest_remaining_options(self, progress_tool):
        """测试建议剩余可尝试选项"""
        messages = [
            HumanMessage(content="问题"),
            AIMessage(content="方案1"),
            HumanMessage(content="试过了"),
        ]

        assessment = progress_tool.assess_progress(messages, turn_count=6)

        # 应该提供剩余选项
        assert len(assessment.remaining_options) > 0
        assert all(isinstance(opt, str) for opt in assessment.remaining_options)

    def test_format_assessment_summary(self, progress_tool):
        """测试格式化评估摘要"""
        assessment = ProgressAssessment(
            turn_count=6,
            problem_resolved=False,
            confidence_score=0.4,
            attempted_solutions=["方案1", "方案2"],
            excluded_causes=["原因1"],
            remaining_options=["选项1", "选项2"],
            recommendation="pivot",
            recommendation_reason="建议转向其他角度",
            next_steps=["步骤1", "步骤2"],
            needs_human_support=False
        )

        summary = progress_tool.format_assessment_summary(assessment)

        # 验证摘要格式
        assert "对话进度评估报告" in summary
        assert "第 6 轮" in summary
        assert "方案1" in summary
        assert "方案2" in summary
        assert "原因1" in summary
        assert "选项1" in summary
        assert "步骤1" in summary


# ============================================================================
# T119: 对话超过5轮主动总结测试（spec.md:108）
# ============================================================================

class TestMultiTurnActiveSummary:
    """测试对话超过5轮时的主动总结功能"""

    def test_trigger_assessment_at_threshold(self, qa_agent):
        """测试在第5轮触发评估"""
        # 模拟第5轮对话
        state = {
            "messages": [HumanMessage(content=f"问题{i}") for i in range(11)],  # 5轮对话 = 10条消息 + 当前
            "turn_count": 5,
            "current_question": "第5个问题",
            "retrieved_documents": [],
            "generated_answer": {
                "problem_analysis": "分析",
                "solutions": ["方案1"],
                "confidence_score": 0.9
            }
        }

        # Mock progress_tool 应该被触发
        with patch.object(qa_agent.progress_tool, 'should_assess', return_value=True) as mock_should:
            with patch.object(qa_agent.progress_tool, 'assess_progress') as mock_assess:
                mock_assessment = ProgressAssessment(
                    turn_count=5,
                    problem_resolved=False,
                    confidence_score=0.5,
                    attempted_solutions=["方案A"],
                    excluded_causes=["原因X"],
                    remaining_options=["选项Y"],
                    recommendation="pivot",
                    recommendation_reason="转向其他角度",
                    next_steps=["步骤1"],
                    needs_human_support=False
                )
                mock_assess.return_value = mock_assessment

                # 调用 _generate_answer
                result_state = qa_agent._generate_answer(state)

                # 验证 should_assess 被调用
                mock_should.assert_called_once_with(5)

                # 验证 assess_progress 被调用
                mock_assess.assert_called_once()

    def test_no_assessment_below_threshold(self, qa_agent):
        """测试低于阈值时不触发评估"""
        state = {
            "messages": [HumanMessage(content="问题1")],
            "turn_count": 3,
            "current_question": "问题1",
            "retrieved_documents": [],
            "generated_answer": {
                "problem_analysis": "分析",
                "solutions": ["方案1"],
                "confidence_score": 0.9
            }
        }

        with patch.object(qa_agent.progress_tool, 'assess_progress') as mock_assess:
            qa_agent._generate_answer(state)

            # 不应该调用评估
            mock_assess.assert_not_called()

    def test_assessment_added_to_solution(self, qa_agent):
        """测试评估结果被添加到答案中"""
        state = {
            "messages": [HumanMessage(content=f"问题{i}") for i in range(11)],
            "turn_count": 6,
            "current_question": "问题",
            "retrieved_documents": [],
            "generated_answer": {
                "problem_analysis": "分析",
                "solutions": ["原始方案"],
                "confidence_score": 0.9
            }
        }

        with patch.object(qa_agent.progress_tool, 'should_assess', return_value=True):
            with patch.object(qa_agent.progress_tool, 'assess_progress') as mock_assess:
                mock_assessment = ProgressAssessment(
                    turn_count=6,
                    problem_resolved=False,
                    confidence_score=0.4,
                    attempted_solutions=["方案A", "方案B"],
                    excluded_causes=[],
                    remaining_options=[],
                    recommendation="pivot",
                    recommendation_reason="建议转向其他角度",
                    next_steps=[],
                    needs_human_support=False
                )
                mock_assess.return_value = mock_assessment

                result_state = qa_agent._generate_answer(state)

                # 验证评估信息被添加到解决方案中
                solutions = result_state["generated_answer"]["solutions"]
                assert len(solutions) > 0
                assert "对话进度总结" in solutions[0]
                assert "已尝试" in solutions[0]


# ============================================================================
# T120: 建议转向其他路径或人工支持测试
# ============================================================================

class TestRecommendations:
    """测试后续行动建议"""

    def test_recommend_continue(self, progress_tool):
        """测试建议继续当前路径"""
        messages = [
            HumanMessage(content="问题"),
            AIMessage(content="方案"),
        ]

        assessment = progress_tool.assess_progress(messages, turn_count=3)

        # 低轮次应该建议继续
        assert assessment.recommendation == "continue"
        assert not assessment.needs_human_support
        assert len(assessment.next_steps) > 0

    def test_recommend_pivot(self, progress_tool):
        """测试建议转向其他排查角度"""
        messages = [
            HumanMessage(content="问题"),
            AIMessage(content="方案"),
        ] * 3  # 6条消息 = 3轮

        assessment = progress_tool.assess_progress(messages, turn_count=6)

        # 6轮应该建议转向
        assert assessment.recommendation == "pivot"
        assert not assessment.needs_human_support
        assert len(assessment.remaining_options) > 0

    def test_recommend_escalate(self, progress_tool):
        """测试建议人工支持"""
        messages = []
        for idx in range(5):
            messages.append(HumanMessage(content=f"问题{idx}"))
            messages.append(AIMessage(content=f"方案{idx}"))

        assessment = progress_tool.assess_progress(messages, turn_count=9)

        # 9轮应该建议升级
        assert assessment.recommendation == "escalate"
        assert assessment.needs_human_support
        assert "人工" in assessment.recommendation_reason or "支持" in assessment.recommendation_reason

    def test_next_steps_provided(self, progress_tool):
        """测试提供下一步建议"""
        messages = [
            HumanMessage(content="问题"),
            AIMessage(content="方案"),
        ]

        assessment = progress_tool.assess_progress(messages, turn_count=6)

        # 应该提供下一步建议
        assert len(assessment.next_steps) > 0
        assert all(isinstance(step, str) for step in assessment.next_steps)
        assert all(len(step) > 0 for step in assessment.next_steps)

    def test_recommendation_reason_provided(self, progress_tool):
        """测试提供建议理由"""
        messages = [HumanMessage(content="问题")]

        assessment = progress_tool.assess_progress(messages, turn_count=5)

        # 应该提供理由
        assert assessment.recommendation_reason
        assert len(assessment.recommendation_reason) > 0


# ============================================================================
# 集成测试
# ============================================================================

class TestProgressAssessmentIntegration:
    """测试进度评估的完整集成"""

    def test_full_assessment_workflow(self, progress_tool):
        """测试完整的评估工作流"""
        # 模拟多轮对话
        messages = [
            HumanMessage(content="模型训练loss不下降，怎么办？"),
            AIMessage(content="建议降低学习率到0.0001"),
            HumanMessage(content="降低了学习率，还是不行"),
            AIMessage(content="建议调整batch_size到16"),
            HumanMessage(content="调整了batch_size，依然没效果"),
            AIMessage(content="建议检查数据质量和标注准确性"),
            HumanMessage(content="数据检查过了，没问题"),
        ]

        # 执行评估
        assessment = progress_tool.assess_progress(
            messages=messages,
            turn_count=7,
            current_question="数据检查过了，没问题"
        )

        # 验证完整评估
        assert assessment.turn_count == 7
        assert assessment.recommendation in ["continue", "pivot", "escalate"]
        assert len(assessment.attempted_solutions) >= 0
        assert len(assessment.next_steps) > 0

        # 格式化摘要
        summary = progress_tool.format_assessment_summary(assessment)
        assert len(summary) > 0
        assert "对话进度评估报告" in summary

    def test_assessment_failure_fallback(self, progress_tool):
        """测试评估失败时的降级处理"""
        # Mock LLM 调用失败
        with patch.object(progress_tool.llm, 'invoke', side_effect=Exception("LLM Error")):
            messages = [HumanMessage(content="问题")]

            assessment = progress_tool.assess_progress(messages, turn_count=6)

            # 应该返回降级评估
            assert isinstance(assessment, ProgressAssessment)
            assert assessment.turn_count == 6
            assert assessment.recommendation in ["continue", "pivot", "escalate"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
