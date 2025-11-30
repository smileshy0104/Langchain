"""
数据模型单元测试

测试所有 Pydantic 数据模型的验证逻辑、序列化和反序列化功能。
"""

import pytest
from datetime import datetime
from pydantic import ValidationError
from models.schemas import (
    # Models
    KnowledgeEntry, TechnicalAnswer, DialogueSession,
    MessageRecord, UserFeedback, QuestionCategory,
    UserQuery,
    # Enums
    SourceType, DocumentType, ChunkBoundary, SessionStatus,
    MessageRole, ContentType, ResolutionStatus, MainCategory
)


class TestKnowledgeEntry:
    """测试 KnowledgeEntry 模型"""

    def test_create_valid_knowledge_entry(self):
        """测试创建有效的知识库条目"""
        entry = KnowledgeEntry(
            entry_id="test-123",
            title="Test Document",
            content="This is test content",
            content_summary="Test summary",
            source_type=SourceType.OFFICIAL_DOCS,
            source_url="https://example.com",
            document_type=DocumentType.TUTORIAL,
            chunk_boundary=ChunkBoundary.SECTION,
            tags=["test", "example"],
            question_categories=["model_usage"],
            embedding_vector=[0.1] * 1536,
            quality_score=0.95,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        assert entry.entry_id == "test-123"
        assert entry.source_type == SourceType.OFFICIAL_DOCS
        assert len(entry.embedding_vector) == 1536
        assert 0 <= entry.quality_score <= 1

    def test_invalid_quality_score(self):
        """测试无效的质量评分"""
        with pytest.raises(ValueError):
            KnowledgeEntry(
                entry_id="test-123",
                title="Test",
                content="Content",
                content_summary="Summary",
                source_type=SourceType.OFFICIAL_DOCS,
                source_url="https://example.com",
                document_type=DocumentType.TUTORIAL,
                chunk_boundary=ChunkBoundary.SECTION,
                embedding_vector=[0.1] * 1536,
                quality_score=1.5,  # 无效: 超过 1.0
                created_at=datetime.now(),
                last_updated=datetime.now()
            )

    def test_serialization(self):
        """测试序列化"""
        entry = KnowledgeEntry(
            entry_id="test-123",
            title="Test",
            content="Content",
            content_summary="Summary",
            source_type=SourceType.OFFICIAL_DOCS,
            source_url="https://example.com",
            document_type=DocumentType.TUTORIAL,
            chunk_boundary=ChunkBoundary.SECTION,
            embedding_vector=[0.1] * 1536,
            quality_score=0.95,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        # 序列化为 dict
        data = entry.model_dump()
        assert isinstance(data, dict)
        assert data["entry_id"] == "test-123"

        # 序列化为 JSON
        json_str = entry.model_dump_json()
        assert isinstance(json_str, str)


class TestTechnicalAnswer:
    """测试 TechnicalAnswer 模型"""

    def test_create_valid_answer(self):
        """测试创建有效的技术回答"""
        answer = TechnicalAnswer(
            problem_analysis="User has CUDA memory error",
            solutions=["Solution 1", "Solution 2"],
            code_examples=["```python\nprint('test')\n```"],
            configuration_notes=["Note 1"],
            warnings=["Warning 1"],
            references=["Ref 1"],
            confidence_score=0.92
        )

        assert len(answer.solutions) >= 1
        assert 0 <= answer.confidence_score <= 1

    def test_empty_solutions_validation(self):
        """测试空解决方案列表验证"""
        with pytest.raises(ValidationError):
            TechnicalAnswer(
                problem_analysis="Problem",
                solutions=[],  # 无效: 空列表
                confidence_score=0.9
            )

    def test_invalid_code_format(self):
        """测试无效的代码格式"""
        with pytest.raises(ValueError, match="代码示例必须使用 Markdown 代码块格式"):
            TechnicalAnswer(
                problem_analysis="Problem",
                solutions=["Solution"],
                code_examples=["print('test')"],  # 无效: 缺少 ```
                confidence_score=0.9
            )

    def test_valid_code_format(self):
        """测试有效的代码格式"""
        answer = TechnicalAnswer(
            problem_analysis="Problem",
            solutions=["Solution"],
            code_examples=[
                "```python\nprint('test')\n```",
                "```bash\nls -la\n```"
            ],
            confidence_score=0.9
        )

        assert len(answer.code_examples) == 2


class TestDialogueSession:
    """测试 DialogueSession 模型"""

    def test_create_valid_session(self):
        """测试创建有效的对话会话"""
        session = DialogueSession(
            session_id="thread-123",
            user_id="user-456",
            created_at=datetime.now(),
            last_updated=datetime.now(),
            turn_count=5,
            total_tokens=3500,
            status=SessionStatus.ACTIVE
        )

        assert session.session_id == "thread-123"
        assert session.turn_count == 5
        assert session.status == SessionStatus.ACTIVE

    def test_default_values(self):
        """测试默认值"""
        session = DialogueSession(
            session_id="thread-123",
            user_id="user-456",
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        assert session.turn_count == 0
        assert session.total_tokens == 0
        assert session.status == SessionStatus.ACTIVE
        assert session.summary is None

    def test_feedback_score_range(self):
        """测试反馈评分范围"""
        session = DialogueSession(
            session_id="thread-123",
            user_id="user-456",
            created_at=datetime.now(),
            last_updated=datetime.now(),
            feedback_score=4.5
        )

        assert 1 <= session.feedback_score <= 5

    def test_invalid_feedback_score(self):
        """测试无效的反馈评分"""
        with pytest.raises(ValueError):
            DialogueSession(
                session_id="thread-123",
                user_id="user-456",
                created_at=datetime.now(),
                last_updated=datetime.now(),
                feedback_score=6.0  # 无效: 超过 5
            )


class TestMessageRecord:
    """测试 MessageRecord 模型"""

    def test_create_user_message(self):
        """测试创建用户消息"""
        message = MessageRecord(
            message_id="msg-123",
            session_id="thread-456",
            role=MessageRole.USER,
            content_type=ContentType.TEXT,
            content="How to use Qwen model?",
            timestamp=datetime.now()
        )

        assert message.role == MessageRole.USER
        assert message.content_type == ContentType.TEXT
        assert len(message.retrieved_doc_ids) == 0

    def test_create_assistant_message(self):
        """测试创建 Assistant 消息"""
        message = MessageRecord(
            message_id="msg-124",
            session_id="thread-456",
            role=MessageRole.ASSISTANT,
            content_type=ContentType.TEXT,
            content="Here's how to use Qwen...",
            timestamp=datetime.now(),
            retrieved_doc_ids=["doc-1", "doc-2"],
            confidence_score=0.95
        )

        assert message.role == MessageRole.ASSISTANT
        assert len(message.retrieved_doc_ids) == 2
        assert message.confidence_score == 0.95

    def test_default_values(self):
        """测试默认值"""
        message = MessageRecord(
            message_id="msg-125",
            session_id="thread-456",
            role=MessageRole.USER,
            content_type=ContentType.TEXT,
            content="Test",
            timestamp=datetime.now()
        )

        assert message.token_count == 0
        assert message.retrieved_doc_ids == []
        assert message.tool_calls == []
        assert message.confidence_score is None


class TestUserFeedback:
    """测试 UserFeedback 模型"""

    def test_create_valid_feedback(self):
        """测试创建有效的用户反馈"""
        feedback = UserFeedback(
            feedback_id="fb-123",
            message_id="msg-456",
            session_id="thread-789",
            helpful_score=5,
            resolution_status=ResolutionStatus.RESOLVED,
            comment="Very helpful!",
            submitted_at=datetime.now()
        )

        assert feedback.helpful_score == 5
        assert feedback.resolution_status == ResolutionStatus.RESOLVED

    def test_helpful_score_range(self):
        """测试评分范围"""
        # 有效评分
        feedback = UserFeedback(
            feedback_id="fb-123",
            message_id="msg-456",
            session_id="thread-789",
            helpful_score=3,
            resolution_status=ResolutionStatus.PARTIALLY_RESOLVED,
            submitted_at=datetime.now()
        )

        assert 1 <= feedback.helpful_score <= 5

    def test_invalid_helpful_score(self):
        """测试无效评分"""
        with pytest.raises(ValueError):
            UserFeedback(
                feedback_id="fb-123",
                message_id="msg-456",
                session_id="thread-789",
                helpful_score=6,  # 无效: 超过 5
                resolution_status=ResolutionStatus.RESOLVED,
                submitted_at=datetime.now()
            )

    def test_default_comment(self):
        """测试默认评论"""
        feedback = UserFeedback(
            feedback_id="fb-123",
            message_id="msg-456",
            session_id="thread-789",
            helpful_score=4,
            resolution_status=ResolutionStatus.RESOLVED,
            submitted_at=datetime.now()
        )

        assert feedback.comment == ""


class TestQuestionCategory:
    """测试 QuestionCategory 模型"""

    def test_create_valid_category(self):
        """测试创建有效的问题分类"""
        category = QuestionCategory(
            main_category=MainCategory.TECHNICAL,
            sub_category="debugging",
            confidence=0.95,
            keywords=["CUDA", "error", "memory"]
        )

        assert category.main_category == MainCategory.TECHNICAL
        assert category.confidence == 0.95
        assert len(category.keywords) == 3

    def test_default_values(self):
        """测试默认值"""
        category = QuestionCategory(
            main_category=MainCategory.MODEL_USAGE,
            confidence=0.9
        )

        assert category.sub_category is None
        assert category.keywords == []

    def test_confidence_range(self):
        """测试置信度范围"""
        category = QuestionCategory(
            main_category=MainCategory.PLATFORM,
            confidence=0.75
        )

        assert 0 <= category.confidence <= 1


class TestUserQuery:
    """测试 UserQuery 输入验证"""

    def test_create_valid_query(self):
        """测试创建有效的用户查询"""
        query = UserQuery(
            question="How to use Qwen model?",
            thread_id="thread-123"
        )

        assert query.question == "How to use Qwen model?"
        assert query.include_code is True
        assert query.max_results == 3

    def test_question_length_validation(self):
        """测试问题长度验证"""
        # 问题太短 (小于3个字符)
        with pytest.raises(ValidationError):
            UserQuery(question="Hi")

        # 问题为空
        with pytest.raises(ValueError, match="问题不能为空"):
            UserQuery(question="   ")

    def test_question_trimming(self):
        """测试问题空格修剪"""
        query = UserQuery(question="  How to use Qwen model?  ")
        assert query.question == "How to use Qwen model?"

    def test_max_results_range(self):
        """测试最大结果数范围"""
        query = UserQuery(
            question="How to use Qwen model?",
            max_results=5
        )

        assert 1 <= query.max_results <= 10

    def test_invalid_max_results(self):
        """测试无效的最大结果数"""
        with pytest.raises(ValueError):
            UserQuery(
                question="How to use Qwen model?",
                max_results=15  # 无效: 超过 10
            )

    def test_default_values(self):
        """测试默认值"""
        query = UserQuery(question="How to use Qwen model?")

        assert query.thread_id is None
        assert query.include_code is True
        assert query.max_results == 3


class TestEnumValues:
    """测试所有枚举类型"""

    def test_source_type_enum(self):
        """测试 SourceType 枚举"""
        assert SourceType.OFFICIAL_DOCS == "official_docs"
        assert SourceType.GITHUB_DOCS == "github_docs"
        assert SourceType.QA_DATASET == "qa_dataset"

    def test_document_type_enum(self):
        """测试 DocumentType 枚举"""
        assert DocumentType.TUTORIAL == "tutorial"
        assert DocumentType.API_DOC == "api_doc"
        assert DocumentType.FAQ == "faq"

    def test_session_status_enum(self):
        """测试 SessionStatus 枚举"""
        assert SessionStatus.ACTIVE == "active"
        assert SessionStatus.COMPLETED == "completed"
        assert SessionStatus.ABANDONED == "abandoned"

    def test_message_role_enum(self):
        """测试 MessageRole 枚举"""
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.SYSTEM == "system"

    def test_resolution_status_enum(self):
        """测试 ResolutionStatus 枚举"""
        assert ResolutionStatus.RESOLVED == "resolved"
        assert ResolutionStatus.PARTIALLY_RESOLVED == "partially_resolved"
        assert ResolutionStatus.UNRESOLVED == "unresolved"

    def test_main_category_enum(self):
        """测试 MainCategory 枚举"""
        assert MainCategory.MODEL_USAGE == "model_usage"
        assert MainCategory.TECHNICAL == "technical"
        assert MainCategory.PLATFORM == "platform"
        assert MainCategory.PROJECT == "project"
