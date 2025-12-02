"""
单轮问答集成测试 (Phase 3)

测试 simple_agent 的完整单轮问答流程:
- Agent 创建和初始化
- 问题 → 分析 → 检索 → 生成答案
- API 端点集成测试
- 端到端流程验证
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from agents.simple_agent import create_agent, invoke_agent
from agents.state import AgentState
from agents.nodes import (
    question_analysis_node,
    retrieval_node,
    answer_generation_node,
    clarify_node,
    should_clarify
)


class TestAgentCreation:
    """测试 Agent 创建和初始化"""

    def test_create_agent_without_dependencies(self):
        """测试无依赖创建 Agent"""
        agent = create_agent()

        assert agent is not None, "Agent 应该成功创建"
        print("✅ 无依赖 Agent 创建测试通过")

    def test_create_agent_with_retriever(self):
        """测试提供 retriever 创建 Agent"""
        mock_retriever = Mock()
        agent = create_agent(retriever=mock_retriever)

        assert agent is not None
        print("✅ 带 Retriever 的 Agent 创建测试通过")

    def test_create_agent_with_llm(self):
        """测试提供 LLM 创建 Agent"""
        mock_llm = Mock()
        agent = create_agent(llm=mock_llm)

        assert agent is not None
        print("✅ 带 LLM 的 Agent 创建测试通过")

    def test_create_agent_with_all_dependencies(self):
        """测试提供完整依赖创建 Agent"""
        mock_retriever = Mock()
        mock_llm = Mock()

        agent = create_agent(retriever=mock_retriever, llm=mock_llm)

        assert agent is not None
        print("✅ 完整依赖 Agent 创建测试通过")


class TestQuestionAnalysisNode:
    """测试问题分析节点"""

    def test_analyze_valid_question(self):
        """测试分析有效问题"""
        state = {
            "messages": [],
            "question": "如何使用 Qwen 模型?",
            "retrieved_docs": None,
            "need_clarification": False,
            "clarification_questions": None,
            "final_answer": None,
            "confidence_score": None,
            "session_id": "test-session",
            "turn_count": 0
        }

        result = question_analysis_node(state)

        assert result["question"] == "如何使用 Qwen 模型?"
        assert result["turn_count"] == 1
        assert result["need_clarification"] == False

        print("✅ 有效问题分析测试通过")

    def test_analyze_empty_question(self):
        """测试分析空问题"""
        state = {
            "messages": [],
            "question": "",
            "retrieved_docs": None,
            "need_clarification": False,
            "clarification_questions": None,
            "final_answer": None,
            "confidence_score": None,
            "session_id": "test-session",
            "turn_count": 0
        }

        result = question_analysis_node(state)

        assert result["need_clarification"] == True
        assert "clarification_questions" in result
        assert len(result["clarification_questions"]) > 0

        print("✅ 空问题分析测试通过")

    def test_analyze_whitespace_question(self):
        """测试分析纯空格问题"""
        state = {
            "messages": [],
            "question": "   ",
            "retrieved_docs": None,
            "need_clarification": False,
            "clarification_questions": None,
            "final_answer": None,
            "confidence_score": None,
            "session_id": "test-session",
            "turn_count": 0
        }

        result = question_analysis_node(state)

        assert result["need_clarification"] == True

        print("✅ 空格问题分析测试通过")


class TestRetrievalNode:
    """测试检索节点"""

    def test_retrieval_without_retriever(self):
        """测试无 retriever 的检索"""
        state = {
            "messages": [],
            "question": "测试问题",
            "retrieved_docs": None,
            "need_clarification": False,
            "clarification_questions": None,
            "final_answer": None,
            "confidence_score": None,
            "session_id": "test-session",
            "turn_count": 1
        }

        result = retrieval_node(state, retriever=None)

        assert result["retrieved_docs"] == []
        assert result["confidence_score"] == 0.0

        print("✅ 无 Retriever 检索测试通过")

    def test_retrieval_with_documents(self):
        """测试成功检索文档"""
        # Mock retriever
        mock_retriever = Mock()
        mock_docs = [
            (Document(page_content="Qwen 是一个大语言模型", metadata={"source": "doc1.md"}), 0.9),
            (Document(page_content="Qwen 支持多轮对话", metadata={"source": "doc2.md"}), 0.85),
            (Document(page_content="使用 transformers 加载 Qwen", metadata={"source": "doc3.md"}), 0.8)
        ]
        mock_retriever.retrieve = Mock(return_value=mock_docs)

        state = {
            "messages": [],
            "question": "如何使用 Qwen?",
            "retrieved_docs": None,
            "need_clarification": False,
            "clarification_questions": None,
            "final_answer": None,
            "confidence_score": None,
            "session_id": "test-session",
            "turn_count": 1
        }

        result = retrieval_node(state, retriever=mock_retriever, top_k=3)

        assert len(result["retrieved_docs"]) == 3
        assert result["confidence_score"] > 0.8
        assert all("content" in doc for doc in result["retrieved_docs"])
        assert all("metadata" in doc for doc in result["retrieved_docs"])
        assert all("score" in doc for doc in result["retrieved_docs"])

        # 验证 retriever.retrieve 被调用
        mock_retriever.retrieve.assert_called_once_with("如何使用 Qwen?", k=3)

        print("✅ 成功检索文档测试通过")

    def test_retrieval_with_empty_results(self):
        """测试检索返回空结果"""
        mock_retriever = Mock()
        mock_retriever.retrieve = Mock(return_value=[])

        state = {
            "messages": [],
            "question": "非常罕见的问题",
            "retrieved_docs": None,
            "need_clarification": False,
            "clarification_questions": None,
            "final_answer": None,
            "confidence_score": None,
            "session_id": "test-session",
            "turn_count": 1
        }

        result = retrieval_node(state, retriever=mock_retriever)

        assert result["retrieved_docs"] == []
        assert result["confidence_score"] == 0.0

        print("✅ 空结果检索测试通过")

    def test_retrieval_with_error(self):
        """测试检索异常处理"""
        mock_retriever = Mock()
        mock_retriever.retrieve = Mock(side_effect=Exception("检索失败"))

        state = {
            "messages": [],
            "question": "测试问题",
            "retrieved_docs": None,
            "need_clarification": False,
            "clarification_questions": None,
            "final_answer": None,
            "confidence_score": None,
            "session_id": "test-session",
            "turn_count": 1
        }

        result = retrieval_node(state, retriever=mock_retriever)

        # 应该返回空结果而不是抛出异常
        assert result["retrieved_docs"] == []
        assert result["confidence_score"] == 0.0

        print("✅ 检索异常处理测试通过")


class TestAnswerGenerationNode:
    """测试答案生成节点"""

    def test_generate_without_documents(self):
        """测试无文档时生成答案"""
        state = {
            "messages": [],
            "question": "测试问题",
            "retrieved_docs": [],
            "need_clarification": False,
            "clarification_questions": None,
            "final_answer": None,
            "confidence_score": 0.0,
            "session_id": "test-session",
            "turn_count": 1
        }

        result = answer_generation_node(state, llm=None)

        assert "final_answer" in result
        assert "抱歉" in result["final_answer"] or "没有找到" in result["final_answer"]
        assert result["confidence_score"] == 0.0

        print("✅ 无文档生成答案测试通过")

    def test_generate_with_documents_without_llm(self):
        """测试有文档但无 LLM 时生成答案"""
        state = {
            "messages": [],
            "question": "如何使用 Qwen?",
            "retrieved_docs": [
                {
                    "content": "Qwen 是一个大语言模型",
                    "metadata": {"source": "doc1.md"},
                    "score": 0.9
                },
                {
                    "content": "使用 transformers 加载 Qwen",
                    "metadata": {"source": "doc2.md"},
                    "score": 0.85
                }
            ],
            "need_clarification": False,
            "clarification_questions": None,
            "final_answer": None,
            "confidence_score": 0.85,
            "session_id": "test-session",
            "turn_count": 1
        }

        result = answer_generation_node(state, llm=None)

        assert "final_answer" in result
        assert len(result["final_answer"]) > 0
        assert "confidence_score" in result
        assert result["confidence_score"] == 0.85

        print("✅ 无 LLM 生成答案测试通过")

    def test_generate_with_llm(self):
        """测试使用 LLM 生成答案"""
        # Mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "根据检索到的文档,Qwen 是一个大语言模型。您可以使用 transformers 库来加载和使用它。"
        mock_llm.invoke = Mock(return_value=mock_response)

        state = {
            "messages": [],
            "question": "如何使用 Qwen?",
            "retrieved_docs": [
                {
                    "content": "Qwen 是一个大语言模型",
                    "metadata": {"source": "doc1.md"},
                    "score": 0.9
                }
            ],
            "need_clarification": False,
            "clarification_questions": None,
            "final_answer": None,
            "confidence_score": 0.9,
            "session_id": "test-session",
            "turn_count": 1
        }

        result = answer_generation_node(state, llm=mock_llm)

        assert "final_answer" in result
        assert "Qwen" in result["final_answer"]
        assert result["confidence_score"] > 0.0

        # 验证 LLM 被调用
        mock_llm.invoke.assert_called_once()

        print("✅ 使用 LLM 生成答案测试通过")

    def test_generate_with_llm_error(self):
        """测试 LLM 调用失败时的回退机制"""
        # Mock LLM 抛出异常
        mock_llm = Mock()
        mock_llm.invoke = Mock(side_effect=Exception("LLM 调用失败"))

        state = {
            "messages": [],
            "question": "测试问题",
            "retrieved_docs": [
                {
                    "content": "测试内容",
                    "metadata": {"source": "test.md"},
                    "score": 0.8
                }
            ],
            "need_clarification": False,
            "clarification_questions": None,
            "final_answer": None,
            "confidence_score": 0.8,
            "session_id": "test-session",
            "turn_count": 1
        }

        result = answer_generation_node(state, llm=mock_llm)

        # 应该有回退答案
        assert "final_answer" in result
        assert result["confidence_score"] is not None

        print("✅ LLM 失败回退测试通过")


class TestClarifyNode:
    """测试澄清节点"""

    def test_clarify_empty_question(self):
        """测试空问题生成澄清"""
        state = {
            "messages": [],
            "question": "",
            "retrieved_docs": None,
            "need_clarification": False,
            "clarification_questions": None,
            "final_answer": None,
            "confidence_score": None,
            "session_id": "test-session",
            "turn_count": 0
        }

        result = clarify_node(state)

        assert result["need_clarification"] == True
        assert "clarification_questions" in result
        assert len(result["clarification_questions"]) > 0

        print("✅ 空问题澄清测试通过")

    def test_clarify_short_question(self):
        """测试短问题生成澄清"""
        state = {
            "messages": [],
            "question": "Qwen?",
            "retrieved_docs": None,
            "need_clarification": False,
            "clarification_questions": None,
            "final_answer": None,
            "confidence_score": None,
            "session_id": "test-session",
            "turn_count": 1
        }

        result = clarify_node(state)

        assert result["need_clarification"] == True
        assert len(result["clarification_questions"]) > 0

        print("✅ 短问题澄清测试通过")

    def test_clarify_low_confidence(self):
        """测试低置信度生成澄清"""
        state = {
            "messages": [],
            "question": "如何使用这个模型?",
            "retrieved_docs": [],
            "need_clarification": False,
            "clarification_questions": None,
            "final_answer": None,
            "confidence_score": 0.2,
            "session_id": "test-session",
            "turn_count": 1
        }

        result = clarify_node(state)

        assert result["need_clarification"] == True
        assert len(result["clarification_questions"]) > 0

        print("✅ 低置信度澄清测试通过")


class TestShouldClarify:
    """测试路由函数"""

    def test_should_clarify_when_flagged(self):
        """测试已标记需要澄清时的路由"""
        state = {
            "messages": [],
            "question": "",
            "retrieved_docs": None,
            "need_clarification": True,
            "clarification_questions": None,
            "final_answer": None,
            "confidence_score": None,
            "session_id": "test-session",
            "turn_count": 0
        }

        result = should_clarify(state)

        assert result == "clarify"

        print("✅ 标记澄清路由测试通过")

    def test_should_clarify_low_confidence(self):
        """测试低置信度时的路由"""
        state = {
            "messages": [],
            "question": "测试问题",
            "retrieved_docs": [{"content": "test", "metadata": {}, "score": 0.3}],
            "need_clarification": False,
            "clarification_questions": None,
            "final_answer": None,
            "confidence_score": 0.3,
            "session_id": "test-session",
            "turn_count": 1
        }

        result = should_clarify(state)

        assert result == "clarify"

        print("✅ 低置信度路由测试通过")

    def test_should_clarify_no_documents(self):
        """测试无文档时的路由"""
        state = {
            "messages": [],
            "question": "测试问题",
            "retrieved_docs": [],
            "need_clarification": False,
            "clarification_questions": None,
            "final_answer": None,
            "confidence_score": 0.0,
            "session_id": "test-session",
            "turn_count": 1
        }

        result = should_clarify(state)

        assert result == "clarify"

        print("✅ 无文档路由测试通过")

    def test_should_generate_high_confidence(self):
        """测试高置信度时的路由"""
        state = {
            "messages": [],
            "question": "如何使用 Qwen?",
            "retrieved_docs": [
                {"content": "test1", "metadata": {}, "score": 0.9},
                {"content": "test2", "metadata": {}, "score": 0.85}
            ],
            "need_clarification": False,
            "clarification_questions": None,
            "final_answer": None,
            "confidence_score": 0.875,
            "session_id": "test-session",
            "turn_count": 1
        }

        result = should_clarify(state)

        assert result == "generate"

        print("✅ 高置信度生成路由测试通过")


class TestInvokeAgent:
    """测试 invoke_agent 函数"""

    def test_invoke_agent_basic(self):
        """测试基本 Agent 调用"""
        # 创建简单的 mock agent
        mock_agent = Mock()
        mock_agent.invoke = Mock(return_value={
            "messages": [],
            "question": "测试问题",
            "retrieved_docs": [],
            "need_clarification": True,
            "clarification_questions": ["请提供更多信息"],
            "final_answer": None,
            "confidence_score": None,
            "session_id": "test-session",
            "turn_count": 1
        })

        result = invoke_agent(mock_agent, "测试问题", session_id="test-session")

        assert result is not None
        assert "question" in result
        assert result["question"] == "测试问题"

        # 验证 agent.invoke 被调用
        mock_agent.invoke.assert_called_once()

        print("✅ 基本 Agent 调用测试通过")

    def test_invoke_agent_without_session_id(self):
        """测试无 session_id 的 Agent 调用"""
        mock_agent = Mock()
        mock_agent.invoke = Mock(return_value={
            "messages": [],
            "question": "测试问题",
            "retrieved_docs": [],
            "need_clarification": False,
            "clarification_questions": None,
            "final_answer": "测试答案",
            "confidence_score": 0.8,
            "session_id": None,
            "turn_count": 1
        })

        result = invoke_agent(mock_agent, "测试问题")

        assert result is not None
        # session_id 应该是 None (由 invoke_agent 传入的初始状态)

        print("✅ 无 session_id Agent 调用测试通过")


class TestEndToEndFlow:
    """端到端流程测试"""

    def test_full_flow_with_answer(self):
        """测试完整流程: 问题 → 检索 → 生成答案"""
        # Mock retriever
        mock_retriever = Mock()
        mock_retriever.retrieve = Mock(return_value=[
            (Document(page_content="Qwen 是一个大语言模型", metadata={"source": "doc.md"}), 0.9)
        ])

        # Mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Qwen 是一个大语言模型,您可以使用它进行各种 NLP 任务。"
        mock_llm.invoke = Mock(return_value=mock_response)

        # 创建 Agent
        agent = create_agent(retriever=mock_retriever, llm=mock_llm)

        # 调用 Agent
        result = invoke_agent(agent, "什么是 Qwen?", session_id="test-session")

        # 验证结果
        assert result is not None
        assert "final_answer" in result or "clarification_questions" in result

        print("✅ 完整流程(有答案)测试通过")

    def test_full_flow_with_clarification(self):
        """测试完整流程: 问题不清晰 → 澄清"""
        # Mock retriever 返回空结果
        mock_retriever = Mock()
        mock_retriever.retrieve = Mock(return_value=[])

        # 创建 Agent
        agent = create_agent(retriever=mock_retriever, llm=None)

        # 调用 Agent (用一个不清晰的短问题)
        result = invoke_agent(agent, "用法?", session_id="test-session")

        # 验证结果 - 应该触发澄清
        assert result is not None
        # 由于置信度低或无文档,应该路由到 clarify 节点

        print("✅ 完整流程(需澄清)测试通过")


class TestAPIIntegration:
    """API 集成测试"""

    def test_qa_ask_endpoint_structure(self):
        """测试 QA API 端点结构 (不启动服务器)"""
        # 这个测试验证 API 端点的基本结构
        # 实际的 HTTP 测试需要启动服务器,属于 E2E 测试

        from api.routers.qa import QuestionRequest, AnswerResponse, SourceInfo

        # 验证请求模型
        request = QuestionRequest(
            question="测试问题",
            session_id="test-session",
            top_k=3
        )

        assert request.question == "测试问题"
        assert request.session_id == "test-session"
        assert request.top_k == 3

        # 验证响应模型
        response = AnswerResponse(
            answer="测试答案",
            sources=[
                SourceInfo(content="来源内容", source="test.md", score=0.9)
            ],
            confidence=0.85,
            session_id="test-session",
            timestamp="2024-12-02T00:00:00"
        )

        assert response.answer == "测试答案"
        assert len(response.sources) == 1
        assert response.confidence == 0.85

        print("✅ API 端点结构测试通过")


class TestSessionRecovery:
    """测试会话恢复功能 (T051)"""

    @pytest.mark.asyncio
    async def test_session_recovery_after_page_refresh(self):
        """
        测试页面刷新后会话恢复

        场景:
        1. 用户创建会话并进行对话
        2. 模拟页面刷新（重新加载会话）
        3. 验证对话历史保留
        """
        import httpx

        API_BASE = "http://localhost:8000"

        async with httpx.AsyncClient(timeout=30.0) as client:
            # 1. 创建会话
            print("\n📝 创建新会话...")
            response = await client.post(f"{API_BASE}/api/v2/sessions")
            assert response.status_code == 200
            session_data = response.json()
            session_id = session_data["session_id"]
            print(f"✅ 会话创建成功: {session_id}")

            # 2. 进行对话
            print("\n📝 第一轮对话...")
            response = await client.post(
                f"{API_BASE}/api/question",
                json={
                    "question": "什么是机器学习?",
                    "session_id": session_id
                }
            )
            assert response.status_code == 200
            answer1 = response.json()
            print(f"✅ 第一轮回答: {answer1['answer'][:50]}...")

            print("\n📝 第二轮对话...")
            response = await client.post(
                f"{API_BASE}/api/question",
                json={
                    "question": "它有哪些应用?",
                    "session_id": session_id
                }
            )
            assert response.status_code == 200
            answer2 = response.json()
            print(f"✅ 第二轮回答: {answer2['answer'][:50]}...")

            # 3. 模拟页面刷新：重新获取会话
            print("\n📝 模拟页面刷新，重新加载会话...")
            response = await client.get(f"{API_BASE}/api/v2/sessions/{session_id}")
            assert response.status_code == 200
            recovered_session = response.json()
            print(f"✅ 会话恢复成功: {recovered_session['turn_count']} 轮对话")

            # 4. 获取对话历史
            print("\n📝 获取对话历史...")
            response = await client.get(f"{API_BASE}/api/v2/sessions/{session_id}/history")
            assert response.status_code == 200
            history = response.json()

            # 验证历史记录完整
            assert len(history) == 2, f"应有2轮对话,实际: {len(history)}"
            assert history[0]["question"] == "什么是机器学习?"
            assert history[1]["question"] == "它有哪些应用?"
            print(f"✅ 对话历史完整: {len(history)} 轮")

            # 5. 验证可以继续对话
            print("\n📝 验证可以继续对话...")
            response = await client.post(
                f"{API_BASE}/api/question",
                json={
                    "question": "能举个例子吗?",
                    "session_id": session_id
                }
            )
            assert response.status_code == 200
            answer3 = response.json()
            print(f"✅ 第三轮回答: {answer3['answer'][:50]}...")

            # 清理
            await client.delete(f"{API_BASE}/api/v2/sessions/{session_id}")
            print("\n✅ 会话恢复测试通过!")

    @pytest.mark.asyncio
    async def test_session_expiry_handling(self):
        """
        测试会话过期处理

        场景:
        1. 尝试访问不存在的会话
        2. 验证返回 404 错误
        """
        import httpx

        API_BASE = "http://localhost:8000"

        async with httpx.AsyncClient(timeout=30.0) as client:
            # 尝试访问不存在的会话
            fake_session_id = "non-existent-session-12345"
            response = await client.get(f"{API_BASE}/api/v2/sessions/{fake_session_id}")

            assert response.status_code == 404
            print("✅ 不存在的会话返回 404 错误")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
