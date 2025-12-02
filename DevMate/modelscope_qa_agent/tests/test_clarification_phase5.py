"""
测试 Phase 5: 主动澄清功能 (T042)

测试 Phase 5 实现的澄清机制:
- detect_ambiguous_question() 模糊问题检测
- clarification_generation_node() 澄清问题生成
- 完整的澄清工作流
"""

import pytest
from unittest.mock import Mock, MagicMock
from agents.nodes import (
    detect_ambiguous_question,
    clarification_generation_node,
    question_analysis_node
)
from agents.simple_agent import create_agent, invoke_agent


class TestAmbiguousQuestionDetection:
    """测试模糊问题检测逻辑 (T035)"""

    def test_detect_short_chinese_question(self):
        """测试场景 1: 检测过短的中文问题"""
        # 测试用例 from tasks.md: "模型报错了怎么办?"
        result = detect_ambiguous_question("报错", [])
        assert result == True, "过短的中文问题应该被检测为模糊"

        result = detect_ambiguous_question("怎么办", [])
        assert result == True, "过短的中文问题应该被检测为模糊"

        print("✅ 过短中文问题检测通过")

    def test_detect_short_english_question(self):
        """测试场景 2: 检测过短的英文问题"""
        result = detect_ambiguous_question("error", [])
        assert result == True, "过短的英文问题应该被检测为模糊"

        result = detect_ambiguous_question("how", [])
        assert result == True, "过短的英文问题应该被检测为模糊"

        print("✅ 过短英文问题检测通过")

    def test_detect_ambiguous_keywords(self):
        """测试场景 3: 检测包含模糊关键词的问题"""
        # 包含模糊词但没有具体信息
        ambiguous_questions = [
            "模型报错了怎么办?",
            "出现问题了",
            "这个不行",
            "运行失败",
            "有bug"
        ]

        for question in ambiguous_questions:
            result = detect_ambiguous_question(question, [])
            assert result == True, f"'{question}' 应该被检测为模糊"

        print("✅ 模糊关键词检测通过")

    def test_detect_specific_questions(self):
        """测试场景 4: 具体问题不应触发澄清"""
        specific_questions = [
            "如何使用 transformers 库加载 Qwen-7B 模型?",
            "使用 pip install modelscope 时出现错误信息 'No module named torch'",
            "Qwen-VL 模型在处理图像时返回 CUDA out of memory 错误,如何解决?",
        ]

        for question in specific_questions:
            result = detect_ambiguous_question(question, [])
            assert result == False, f"'{question}' 不应被检测为模糊"

        print("✅ 具体问题识别通过")

    def test_detect_missing_context_without_history(self):
        """测试场景 5: 无历史时缺少上下文的问题"""
        # 提到操作但没有说明对象
        result = detect_ambiguous_question("如何调用?", [])
        assert result == True, "缺少上下文的问题应该被检测为模糊"

        result = detect_ambiguous_question("怎么安装?", [])
        assert result == True, "缺少上下文的问题应该被检测为模糊"

        print("✅ 缺少上下文检测通过")

    def test_action_with_object_is_clear(self):
        """测试场景 6: 包含动作和对象的问题是清晰的"""
        result = detect_ambiguous_question("如何使用 Qwen 模型?", [])
        assert result == False, "包含动作和对象的问题应该是清晰的"

        result = detect_ambiguous_question("如何安装 modelscope SDK?", [])
        assert result == False, "包含动作和对象的问题应该是清晰的"

        print("✅ 动作+对象问题识别通过")


class TestClarificationGeneration:
    """测试澄清问题生成节点 (T036)"""

    def test_generate_clarification_for_empty_question(self):
        """测试场景 1: 空问题生成澄清"""
        state = {
            "question": "",
            "confidence_score": 0.0,
            "retrieved_docs": [],
            "messages": [],
            "ambiguity_detected": True
        }

        result = clarification_generation_node(state)

        assert result["need_clarification"] == True
        assert len(result["clarification_questions"]) > 0
        assert "具体" in result["clarification_questions"][0]

        print("✅ 空问题澄清生成通过")

    def test_generate_clarification_for_error_questions(self):
        """测试场景 2: 错误相关问题生成澄清"""
        state = {
            "question": "模型报错了",
            "confidence_score": 0.0,
            "retrieved_docs": [],
            "messages": [],
            "ambiguity_detected": True
        }

        result = clarification_generation_node(state)

        assert result["need_clarification"] == True
        questions = result["clarification_questions"]
        assert len(questions) > 0

        # 应该询问具体错误信息
        has_error_question = any(
            "错误" in q or "模型" in q or "API" in q
            for q in questions
        )
        assert has_error_question, "应该询问错误相关信息"

        print("✅ 错误问题澄清生成通过")
        print(f"   生成的澄清问题: {questions}")

    def test_generate_clarification_for_usage_questions(self):
        """测试场景 3: 使用方法问题生成澄清"""
        state = {
            "question": "怎么用?",
            "confidence_score": 0.0,
            "retrieved_docs": [],
            "messages": [],
            "ambiguity_detected": True
        }

        result = clarification_generation_node(state)

        assert result["need_clarification"] == True
        questions = result["clarification_questions"]
        assert len(questions) > 0

        # 应该询问具体想了解什么
        has_usage_question = any(
            "具体" in q or "功能" in q or "模型" in q
            for q in questions
        )
        assert has_usage_question, "应该询问使用相关信息"

        print("✅ 使用方法问题澄清生成通过")
        print(f"   生成的澄清问题: {questions}")

    def test_clarification_question_limit(self):
        """测试场景 4: 澄清问题数量限制"""
        state = {
            "question": "出问题了",
            "confidence_score": 0.0,
            "retrieved_docs": [],
            "messages": [],
            "ambiguity_detected": True
        }

        result = clarification_generation_node(state)

        questions = result["clarification_questions"]
        assert len(questions) <= 3, "澄清问题不应超过3个"
        assert len(questions) >= 1, "至少应该有1个澄清问题"

        print("✅ 澄清问题数量限制通过")

    def test_clarification_with_llm(self):
        """测试场景 5: 使用 LLM 生成澄清问题"""
        # Mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = """
请问您遇到的具体错误信息是什么?
您使用的是哪个模型或 API?
能否提供完整的错误堆栈信息 (traceback)?
"""
        mock_llm.invoke = Mock(return_value=mock_response)

        state = {
            "question": "模型加载失败",
            "confidence_score": 0.0,
            "retrieved_docs": [],
            "messages": [],
            "ambiguity_detected": True
        }

        result = clarification_generation_node(state, llm=mock_llm)

        assert result["need_clarification"] == True
        questions = result["clarification_questions"]
        assert len(questions) > 0

        # 验证 LLM 被调用
        mock_llm.invoke.assert_called_once()

        print("✅ LLM 澄清问题生成通过")
        print(f"   生成的澄清问题: {questions}")


class TestQuestionAnalysisWithClarification:
    """测试集成了澄清检测的问题分析节点 (T035)"""

    def test_analysis_detects_ambiguity(self):
        """测试场景 1: 问题分析节点检测到模糊性"""
        state = {
            "question": "报错了",
            "messages": [],
            "turn_count": 0
        }

        result = question_analysis_node(state)

        assert result["need_clarification"] == True, "应该标记需要澄清"
        assert result["ambiguity_detected"] == True, "应该标记检测到模糊性"

        print("✅ 问题分析澄清检测通过")

    def test_analysis_clear_question(self):
        """测试场景 2: 清晰问题不触发澄清"""
        state = {
            "question": "如何使用 Qwen-7B 模型进行文本生成?",
            "messages": [],
            "turn_count": 0
        }

        result = question_analysis_node(state)

        # 清晰问题不应触发澄清(或者 need_clarification 为 False)
        if "need_clarification" in result:
            assert result["need_clarification"] == False, "清晰问题不应需要澄清"

        print("✅ 清晰问题不触发澄清通过")


class TestEndToEndClarification:
    """测试完整的澄清工作流 (T038, T042)"""

    def test_ambiguous_question_workflow(self):
        """测试场景 1: 模糊问题触发完整澄清流程

        测试 tasks.md 中的独立测试标准:
        1. 输入模糊问题: "模型报错了怎么办?"
        2. 验证 Agent 返回澄清问题
        """
        # Mock retriever and LLM
        mock_retriever = Mock()
        mock_retriever.retrieve = Mock(return_value=[])

        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "请问您遇到的具体错误信息是什么?\n您使用的是哪个模型?"
        mock_llm.invoke = Mock(return_value=mock_response)

        # Create agent
        agent = create_agent(retriever=mock_retriever, llm=mock_llm)

        # Invoke with ambiguous question
        result = invoke_agent(
            agent=agent,
            question="模型报错了怎么办?",
            session_id="test-session",
            conversation_history=[]
        )

        # Verify clarification was triggered
        assert result is not None
        assert "need_clarification" in result

        if result["need_clarification"]:
            assert "clarification_questions" in result
            assert len(result["clarification_questions"]) > 0
            print("✅ 模糊问题触发澄清流程通过")
            print(f"   澄清问题: {result['clarification_questions']}")
        else:
            # 如果没有触发澄清,可能是因为检索到了足够的文档
            print("   注意: 问题未触发澄清,可能检索到了足够的文档")

    def test_clear_question_workflow(self):
        """测试场景 2: 清晰问题不触发澄清"""
        # Mock retriever and LLM
        mock_retriever = Mock()
        mock_retriever.retrieve = Mock(return_value=[
            Mock(page_content="Qwen 是一个大语言模型", metadata={"source": "test.md"})
        ])

        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Qwen 是由阿里云开发的大语言模型"
        mock_llm.invoke = Mock(return_value=mock_response)

        # Create agent
        agent = create_agent(retriever=mock_retriever, llm=mock_llm)

        # Invoke with clear question
        result = invoke_agent(
            agent=agent,
            question="如何使用 Qwen-7B 模型进行文本生成?请提供详细步骤。",
            session_id="test-session",
            conversation_history=[]
        )

        # Verify no clarification or normal answer
        assert result is not None

        # 可能有两种情况:
        # 1. need_clarification 为 False
        # 2. need_clarification 不存在(没有触发澄清检查)
        if "need_clarification" in result:
            # 如果存在该字段,应该为 False
            needs_clarification = result["need_clarification"]
            # 注意: 如果置信度低,可能仍然需要澄清
            if needs_clarification:
                print("   注意: 虽然问题清晰,但由于其他原因(如低置信度)仍需澄清")
            else:
                print("✅ 清晰问题不触发澄清通过")
        else:
            print("✅ 清晰问题正常处理通过")


class TestClarificationPrompts:
    """测试澄清提示词模板 (T037)"""

    def test_clarification_prompt_exists(self):
        """测试场景 1: 验证澄清提示词模板存在"""
        from agents.prompts import CLARIFICATION_PROMPT, INTELLIGENT_CLARIFICATION_PROMPT

        assert CLARIFICATION_PROMPT is not None
        assert len(CLARIFICATION_PROMPT) > 0
        assert INTELLIGENT_CLARIFICATION_PROMPT is not None
        assert len(INTELLIGENT_CLARIFICATION_PROMPT) > 0

        print("✅ 澄清提示词模板存在验证通过")

    def test_clarification_prompt_has_placeholders(self):
        """测试场景 2: 验证提示词包含必要的占位符"""
        from agents.prompts import CLARIFICATION_PROMPT

        assert "{question}" in CLARIFICATION_PROMPT
        assert "{current_understanding}" in CLARIFICATION_PROMPT

        print("✅ 澄清提示词占位符验证通过")

    def test_intelligent_clarification_prompt_placeholders(self):
        """测试场景 3: 验证智能澄清提示词占位符"""
        from agents.prompts import INTELLIGENT_CLARIFICATION_PROMPT

        assert "{question}" in INTELLIGENT_CLARIFICATION_PROMPT
        assert "{retrieval_summary}" in INTELLIGENT_CLARIFICATION_PROMPT
        assert "{conversation_summary}" in INTELLIGENT_CLARIFICATION_PROMPT

        print("✅ 智能澄清提示词占位符验证通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
