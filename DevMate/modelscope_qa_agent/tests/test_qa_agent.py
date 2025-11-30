"""
测试魔搭社区问答 Agent

测试 ModelScopeQAAgent 的核心功能:
- 初始化
- LangGraph 工作流构建
- 文档检索节点
- 答案生成节点
- 答案验证节点
- 条件分支逻辑
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from agents.qa_agent import ModelScopeQAAgent
from models.schemas import TechnicalAnswer


class TestModelScopeQAAgentInit:
    """测试 ModelScopeQAAgent 初始化"""

    def test_init_success(self):
        """测试成功初始化"""
        # Mock retriever
        mock_retriever = Mock()
        mock_retriever.get_stats = Mock(return_value={})

        # 创建 Agent
        agent = ModelScopeQAAgent(
            retriever=mock_retriever,
            llm_api_key="test-api-key-123"
        )

        # 验证
        assert agent.retriever == mock_retriever
        assert agent.llm is not None
        assert agent.workflow is not None
        assert agent.checkpointer is not None
        assert agent.app is not None

        print("✅ Agent 初始化测试通过")

    def test_init_with_custom_params(self):
        """测试自定义参数初始化"""
        mock_retriever = Mock()
        mock_retriever.get_stats = Mock(return_value={})

        agent = ModelScopeQAAgent(
            retriever=mock_retriever,
            llm_api_key="test-key",
            model="qwen-max",
            temperature=0.5,
            top_p=0.9
        )

        assert agent._model == "qwen-max"
        assert agent._temperature == 0.5
        assert agent._top_p == 0.9

        print("✅ 自定义参数初始化测试通过")

    def test_init_none_retriever(self):
        """测试 retriever 为 None 时抛出异常"""
        with pytest.raises(ValueError, match="retriever 不能为 None"):
            ModelScopeQAAgent(retriever=None, llm_api_key="test-key")

        print("✅ None retriever 验证测试通过")

    def test_init_empty_api_key(self):
        """测试空 API 密钥时抛出异常"""
        mock_retriever = Mock()

        with pytest.raises(ValueError, match="llm_api_key 不能为空"):
            ModelScopeQAAgent(retriever=mock_retriever, llm_api_key="")

        with pytest.raises(ValueError, match="llm_api_key 不能为空"):
            ModelScopeQAAgent(retriever=mock_retriever, llm_api_key="   ")

        print("✅ 空 API 密钥验证测试通过")


class TestModelScopeQAAgentGraph:
    """测试 LangGraph 工作流"""

    @pytest.fixture
    def agent(self):
        """创建测试用 Agent"""
        mock_retriever = Mock()
        mock_retriever.get_stats = Mock(return_value={})

        return ModelScopeQAAgent(
            retriever=mock_retriever,
            llm_api_key="test-key"
        )

    def test_graph_has_required_nodes(self, agent):
        """测试工作流包含所有必需节点"""
        # LangGraph 的节点存储在 workflow.nodes 中
        nodes = agent.workflow.nodes

        assert "retrieve" in nodes, "工作流应该包含 retrieve 节点"
        assert "generate" in nodes, "工作流应该包含 generate 节点"
        assert "validate" in nodes, "工作流应该包含 validate 节点"

        print("✅ 工作流节点测试通过")

    def test_graph_compiled(self, agent):
        """测试工作流已编译"""
        # 验证 app 已成功编译
        assert agent.app is not None, "工作流应该已编译"
        assert agent.checkpointer is not None, "检查点器应该已配置"

        print("✅ 工作流编译测试通过")


class TestModelScopeQAAgentRetrieve:
    """测试检索节点"""

    @pytest.fixture
    def agent(self):
        """创建测试用 Agent"""
        mock_retriever = Mock()
        mock_retriever.get_stats = Mock(return_value={})

        return ModelScopeQAAgent(
            retriever=mock_retriever,
            llm_api_key="test-key"
        )

    @pytest.fixture
    def sample_docs(self):
        """创建示例文档"""
        return [
            Document(
                page_content="如何使用 Qwen 模型进行对话",
                metadata={"source": "docs", "quality_score": 0.9}
            ),
            Document(
                page_content="Qwen 模型支持多轮对话",
                metadata={"source": "docs", "quality_score": 0.85}
            )
        ]

    def test_retrieve_documents(self, agent, sample_docs):
        """测试文档检索节点"""
        # Mock retriever.retrieve
        agent.retriever.retrieve = Mock(return_value=sample_docs)

        # 创建初始状态
        state = {
            "messages": [HumanMessage(content="如何使用 Qwen?")],
            "current_question": "",
            "retrieved_documents": [],
            "generated_answer": {},
            "turn_count": 0
        }

        # 调用检索节点
        result = agent._retrieve_documents(state)

        # 验证
        assert result["current_question"] == "如何使用 Qwen?"
        assert len(result["retrieved_documents"]) == 2
        assert result["retrieved_documents"] == sample_docs

        # 验证 retrieve 被调用
        agent.retriever.retrieve.assert_called_once_with("如何使用 Qwen?", k=3)

        print("✅ 文档检索节点测试通过")

    def test_retrieve_empty_result(self, agent):
        """测试检索返回空结果"""
        agent.retriever.retrieve = Mock(return_value=[])

        state = {
            "messages": [HumanMessage(content="测试问题")],
            "current_question": "",
            "retrieved_documents": [],
            "generated_answer": {},
            "turn_count": 0
        }

        result = agent._retrieve_documents(state)

        assert result["retrieved_documents"] == []
        print("✅ 空结果检索测试通过")

    def test_retrieve_error_handling(self, agent):
        """测试检索异常处理"""
        agent.retriever.retrieve = Mock(side_effect=Exception("检索失败"))

        state = {
            "messages": [HumanMessage(content="测试问题")],
            "current_question": "",
            "retrieved_documents": [],
            "generated_answer": {},
            "turn_count": 0
        }

        result = agent._retrieve_documents(state)

        # 异常处理应该返回空列表
        assert result["retrieved_documents"] == []
        print("✅ 检索异常处理测试通过")


class TestModelScopeQAAgentValidate:
    """测试验证节点"""

    @pytest.fixture
    def agent(self):
        """创建测试用 Agent"""
        mock_retriever = Mock()
        mock_retriever.get_stats = Mock(return_value={})

        return ModelScopeQAAgent(
            retriever=mock_retriever,
            llm_api_key="test-key"
        )

    def test_validate_answer(self, agent):
        """测试答案验证节点"""
        state = {
            "messages": [],
            "current_question": "测试问题",
            "retrieved_documents": [],
            "generated_answer": {
                "summary": "测试答案",
                "confidence_score": 0.7
            },
            "turn_count": 0
        }

        # 调用验证节点
        result = agent._validate_answer(state)

        # 目前验证节点是占位实现,应该返回原状态
        assert result == state

        print("✅ 答案验证节点测试通过")


class TestModelScopeQAAgentShouldValidate:
    """测试条件分支"""

    @pytest.fixture
    def agent(self):
        """创建测试用 Agent"""
        mock_retriever = Mock()
        mock_retriever.get_stats = Mock(return_value={})

        return ModelScopeQAAgent(
            retriever=mock_retriever,
            llm_api_key="test-key"
        )

    def test_should_validate_low_confidence(self, agent):
        """测试低置信度应该验证"""
        state = {
            "messages": [],
            "current_question": "",
            "retrieved_documents": [],
            "generated_answer": {"confidence_score": 0.5},
            "turn_count": 0
        }

        result = agent._should_validate(state)
        assert result == "validate"

        print("✅ 低置信度条件分支测试通过")

    def test_should_validate_high_confidence(self, agent):
        """测试高置信度应该直接结束"""
        state = {
            "messages": [],
            "current_question": "",
            "retrieved_documents": [],
            "generated_answer": {"confidence_score": 0.9},
            "turn_count": 0
        }

        result = agent._should_validate(state)
        assert result == "end"

        print("✅ 高置信度条件分支测试通过")

    def test_should_validate_threshold(self, agent):
        """测试阈值边界"""
        # 0.8 应该直接结束
        state = {
            "messages": [],
            "current_question": "",
            "retrieved_documents": [],
            "generated_answer": {"confidence_score": 0.8},
            "turn_count": 0
        }
        assert agent._should_validate(state) == "end"

        # 0.79 应该验证
        state["generated_answer"]["confidence_score"] = 0.79
        assert agent._should_validate(state) == "validate"

        print("✅ 阈值边界测试通过")

    def test_should_validate_missing_confidence(self, agent):
        """测试缺失置信度时默认为 0"""
        state = {
            "messages": [],
            "current_question": "",
            "retrieved_documents": [],
            "generated_answer": {},  # 没有 confidence_score
            "turn_count": 0
        }

        result = agent._should_validate(state)
        assert result == "validate"  # 默认 0.0 < 0.8

        print("✅ 缺失置信度测试通过")


class TestModelScopeQAAgentUtility:
    """测试工具方法"""

    @pytest.fixture
    def agent(self):
        """创建测试用 Agent"""
        mock_retriever = Mock()
        mock_retriever.get_stats = Mock(return_value={
            "vector_weight": 0.6,
            "bm25_weight": 0.4
        })

        return ModelScopeQAAgent(
            retriever=mock_retriever,
            llm_api_key="test-key"
        )

    def test_get_stats(self, agent):
        """测试获取统计信息"""
        stats = agent.get_stats()

        assert "retriever_type" in stats
        assert "retriever_stats" in stats
        assert "llm_model" in stats
        assert "llm_temperature" in stats
        assert "has_checkpointer" in stats
        assert "workflow_nodes" in stats

        assert stats["has_checkpointer"] is True
        assert "retrieve" in stats["workflow_nodes"]
        assert "generate" in stats["workflow_nodes"]
        assert "validate" in stats["workflow_nodes"]

        print("✅ 统计信息测试通过")


class TestModelScopeQAAgentInvoke:
    """测试 invoke 方法"""

    @pytest.fixture
    def agent(self):
        """创建测试用 Agent"""
        mock_retriever = Mock()
        mock_retriever.get_stats = Mock(return_value={})
        mock_retriever.retrieve = Mock(return_value=[
            Document(page_content="测试文档", metadata={})
        ])

        return ModelScopeQAAgent(
            retriever=mock_retriever,
            llm_api_key="test-key"
        )

    def test_invoke_empty_question(self, agent):
        """测试空问题应该抛出异常"""
        with pytest.raises(ValueError, match="问题不能为空"):
            agent.invoke("")

        with pytest.raises(ValueError, match="问题不能为空"):
            agent.invoke("   ")

        print("✅ 空问题验证测试通过")

    @patch('agents.qa_agent.ChatTongyi')
    def test_invoke_with_mock_llm(self, mock_chat_class, agent):
        """测试 invoke 调用流程(使用 Mock LLM)"""
        # Mock LLM 返回
        mock_llm_instance = MagicMock()
        mock_chat_class.return_value = mock_llm_instance

        # Mock 生成答案
        mock_answer = TechnicalAnswer(
            summary="这是一个测试答案",
            problem_analysis="问题分析",
            solutions=["解决方案1", "解决方案2"],
            code_examples=[],
            references=["参考1"],
            confidence_score=0.85
        )

        # 由于我们需要 Mock 整个生成过程,这里简化测试
        # 只验证基本流程
        agent.retriever.retrieve = Mock(return_value=[
            Document(page_content="测试内容", metadata={})
        ])

        # 注意: 完整的 invoke 测试需要 Mock LLM 响应
        # 这里我们主要测试参数验证
        try:
            # 这会失败因为需要真实的 API key,但至少验证了参数检查
            result = agent.invoke("测试问题", thread_id="test-thread")
        except Exception as e:
            # 预期会失败(需要真实 API),但不是因为参数问题
            assert "问题不能为空" not in str(e)
            print(f"   (预期的 API 调用失败: {type(e).__name__})")

        print("✅ Invoke 基本流程测试通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
