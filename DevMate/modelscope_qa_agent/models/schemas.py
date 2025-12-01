"""
数据模型定义

定义所有 Pydantic 数据模型，用于数据验证和序列化。
包括知识库条目、技术回答、对话会话、消息记录、用户反馈和问题分类等模型。
"""

from typing import TypedDict, Annotated, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document


# ============================================================================
# 枚举类型定义
# ============================================================================

class SourceType(str, Enum):
    """文档来源类型"""
    OFFICIAL_DOCS = "official_docs"  # 官方文档
    GITHUB_DOCS = "github_docs"  # GitHub 文档
    QA_DATASET = "qa_dataset"  # 真实问答
    MODEL_HUB = "model_hub"  # 模型库
    DATASET_HUB = "dataset_hub"  # 数据集库
    LEARN_CENTER = "learn_center"  # 研习社


class DocumentType(str, Enum):
    """文档类型"""
    TUTORIAL = "tutorial"  # 教程
    API_DOC = "api_doc"  # API 文档
    FAQ = "faq"  # 常见问题
    EXAMPLE = "example"  # 示例代码
    GUIDE = "guide"  # 指南


class ChunkBoundary(str, Enum):
    """分块边界类型"""
    PARAGRAPH = "paragraph"  # 段落
    SECTION = "section"  # 章节
    CODE_BLOCK = "code_block"  # 代码块
    MIXED = "mixed"  # 混合


class SessionStatus(str, Enum):
    """会话状态"""
    ACTIVE = "active"  # 进行中
    COMPLETED = "completed"  # 已完成
    ABANDONED = "abandoned"  # 已放弃


class MessageRole(str, Enum):
    """消息角色"""
    USER = "user"  # 用户
    ASSISTANT = "assistant"  # Agent
    SYSTEM = "system"  # 系统


class ContentType(str, Enum):
    """内容类型"""
    TEXT = "text"  # 纯文本
    CODE = "code"  # 代码片段
    IMAGE = "image"  # 图片


class ResolutionStatus(str, Enum):
    """问题解决状态"""
    RESOLVED = "resolved"  # 已解决
    PARTIALLY_RESOLVED = "partially_resolved"  # 部分解决
    UNRESOLVED = "unresolved"  # 未解决


class MainCategory(str, Enum):
    """主分类"""
    MODEL_USAGE = "model_usage"  # 模型使用
    TECHNICAL = "technical"  # 技术问题
    PLATFORM = "platform"  # 平台功能
    PROJECT = "project"  # 项目指导


class TechnicalSubCategory(str, Enum):
    """技术问题子分类"""
    ENVIRONMENT = "environment"  # 环境搭建
    DEBUGGING = "debugging"  # 代码调试
    OPTIMIZATION = "optimization"  # 性能优化
    DEPLOYMENT = "deployment"  # 部署


# ============================================================================
# T028: KnowledgeEntry - 知识库条目模型
# ============================================================================

class KnowledgeEntry(BaseModel):
    """知识库条目模型（存储在 Milvus）

    用于存储和管理魔搭社区文档的分块内容,包括元数据、向量和质量评分。
    """

    # 唯一标识
    entry_id: str = Field(description="条目 UUID")

    # 内容
    title: str = Field(description="文档标题")
    content: str = Field(description="文档内容（已分块）")
    content_summary: str = Field(description="内容摘要（用于展示）")

    # 元数据
    source_type: SourceType
    source_url: str = Field(description="原始文档 URL")
    document_type: DocumentType
    chunk_boundary: ChunkBoundary

    # 分类标签
    tags: list[str] = Field(default_factory=list, description="标签列表")
    question_categories: list[str] = Field(
        default_factory=list,
        description="适用的问题分类（模型使用/技术调试/平台功能/项目指导）"
    )

    # 向量和评分
    embedding_vector: list[float] = Field(description="通义千问 Embedding 向量（1536维）")
    quality_score: float = Field(ge=0, le=1, description="质量评分（0-1）")

    # 时间戳
    created_at: datetime
    last_updated: datetime

    model_config = {
        "json_schema_extra": {
            "example": {
                "entry_id": "uuid-12345",
                "title": "Qwen 模型快速上手",
                "content": "# Qwen 模型快速上手\n\n## 安装\n```python\npip install modelscope\n```",
                "content_summary": "介绍如何安装和使用 Qwen 模型",
                "source_type": "official_docs",
                "source_url": "https://modelscope.cn/docs/qwen",
                "document_type": "tutorial",
                "chunk_boundary": "section",
                "tags": ["qwen", "安装", "快速上手"],
                "question_categories": ["model_usage"],
                "embedding_vector": [0.1] * 1536,  # 示例向量
                "quality_score": 0.95,
                "created_at": "2025-11-30T10:00:00Z",
                "last_updated": "2025-11-30T10:00:00Z"
            }
        }
    }


# ============================================================================
# T029: TechnicalAnswer - 技术回答模型
# ============================================================================

class TechnicalAnswer(BaseModel):
    """技术问答响应格式（符合 FR-005 结构化输出要求）

    用于规范 Agent 生成的技术回答格式,确保包含问题分析、解决方案、
    代码示例、配置说明、注意事项和参考来源。
    """

    # 问题分析
    problem_analysis: str = Field(
        description="问题分析,说明用户遇到的问题类型和核心痛点"
    )

    # 解决方案列表
    solutions: list[str] = Field(
        min_length=1,
        description="解决方案步骤列表,每个方案应独立完整"
    )

    # 代码示例
    code_examples: list[str] = Field(
        default_factory=list,
        description="可运行的代码示例,使用 Markdown 代码块格式"
    )

    # 参数配置说明
    configuration_notes: list[str] = Field(
        default_factory=list,
        description="参数配置说明和最佳实践"
    )

    # 注意事项
    warnings: list[str] = Field(
        default_factory=list,
        description="注意事项和常见错误"
    )

    # 参考来源
    references: list[str] = Field(
        default_factory=list,
        description="信息来源（官方文档章节或真实问答案例）"
    )

    # 置信度评分
    confidence_score: float = Field(
        ge=0, le=1,
        description="答案置信度（基于检索文档相关性）"
    )

    @field_validator("solutions")
    @classmethod
    def validate_solutions(cls, v: list[str]) -> list[str]:
        """确保至少有一个解决方案"""
        if not v:
            raise ValueError("必须提供至少一种解决方案")
        return v

    @field_validator("code_examples")
    @classmethod
    def validate_code_format(cls, v: list[str]) -> list[str]:
        """验证代码示例格式"""
        for code in v:
            if not code.strip().startswith("```"):
                raise ValueError("代码示例必须使用 Markdown 代码块格式")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "problem_analysis": "用户遇到 CUDA 内存不足错误,导致模型无法加载",
                "solutions": [
                    "1. 降低 batch_size: 将批次大小从 32 降低到 16 或 8",
                    "2. 使用模型量化: 启用 INT8 量化减少显存占用",
                    "3. 启用梯度检查点: 使用 gradient_checkpointing 节省显存"
                ],
                "code_examples": [
                    "```python\n# 方案1: 降低 batch_size\nfrom modelscope import AutoModelForCausalLM\n\nmodel = AutoModelForCausalLM.from_pretrained(\n    'qwen/Qwen-7B',\n    device_map='auto'\n)\n\n# 使用更小的 batch_size\noutputs = model.generate(inputs, batch_size=8)\n```"
                ],
                "configuration_notes": [
                    "batch_size 建议从 32 开始,逐步降低直到不报错",
                    "INT8 量化会略微降低精度,但可节省约 50% 显存"
                ],
                "warnings": [
                    "降低 batch_size 会增加训练时间",
                    "梯度检查点会增加约 20% 计算开销"
                ],
                "references": [
                    "官方文档 - 模型加载与优化 > 显存优化策略",
                    "真实问答 - ModelScope Issue #1234"
                ],
                "confidence_score": 0.92
            }
        }
    }


# ============================================================================
# T030: DialogueSession - 对话会话模型
# ============================================================================

class DialogueSession(BaseModel):
    """对话会话元数据（存储在数据库）

    记录用户与 Agent 的对话会话信息,包括会话状态、统计信息和用户反馈。
    """

    # 唯一标识
    session_id: str = Field(description="会话 UUID（对应 LangGraph thread_id）")
    user_id: str = Field(description="用户 ID")

    # 时间戳
    created_at: datetime
    last_updated: datetime

    # 统计信息
    turn_count: int = Field(default=0, description="对话轮次计数")
    total_tokens: int = Field(default=0, description="累计消耗 token 数")

    # 会话状态
    status: SessionStatus = Field(default=SessionStatus.ACTIVE)

    # 对话摘要（超过10轮后生成）
    summary: Optional[str] = Field(default=None, description="早期对话摘要")

    # 用户满意度反馈
    feedback_score: Optional[float] = Field(default=None, ge=1, le=5, description="1-5星评分")
    is_resolved: Optional[bool] = Field(default=None, description="问题是否解决")

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "thread-abc123",
                "user_id": "user-456",
                "created_at": "2025-11-30T10:00:00Z",
                "last_updated": "2025-11-30T10:15:00Z",
                "turn_count": 5,
                "total_tokens": 3500,
                "status": "active",
                "summary": None,
                "feedback_score": 4.5,
                "is_resolved": True
            }
        }
    }


# ============================================================================
# T031: MessageRecord - 消息记录模型
# ============================================================================

class MessageRecord(BaseModel):
    """消息记录（存储在数据库）

    记录每一条用户和 Agent 的消息,包括内容、元数据和引用信息。
    """

    # 唯一标识
    message_id: str = Field(description="消息 UUID")
    session_id: str = Field(description="所属会话 ID")

    # 消息内容
    role: MessageRole
    content_type: ContentType
    content: str = Field(description="消息内容")

    # 元数据
    timestamp: datetime
    token_count: int = Field(default=0, description="token 数量")

    # Agent 特有字段（仅 assistant 角色）
    retrieved_doc_ids: list[str] = Field(
        default_factory=list,
        description="引用的知识库条目 ID"
    )
    tool_calls: list[str] = Field(
        default_factory=list,
        description="调用的工具列表"
    )
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0, le=1,
        description="答案置信度（仅 assistant）"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "message_id": "msg-xyz789",
                "session_id": "thread-abc123",
                "role": "user",
                "content_type": "text",
                "content": "如何在魔搭社区使用 Qwen 模型?",
                "timestamp": "2025-11-30T10:00:00Z",
                "token_count": 15,
                "retrieved_doc_ids": [],
                "tool_calls": []
            }
        }
    }


# ============================================================================
# T032: UserFeedback - 用户反馈模型
# ============================================================================

class UserFeedback(BaseModel):
    """用户反馈记录（符合 FR-016 要求）

    收集用户对 Agent 回答的反馈,包括有帮助度评分和问题解决状态。
    """

    # 唯一标识
    feedback_id: str = Field(description="反馈 UUID")
    message_id: str = Field(description="对应的 Agent 回答消息 ID")
    session_id: str = Field(description="会话 ID")

    # 反馈内容
    helpful_score: int = Field(ge=1, le=5, description="有帮助度评分（1-5星）")
    resolution_status: ResolutionStatus
    comment: str = Field(default="", description="用户评论")

    # 时间戳
    submitted_at: datetime

    model_config = {
        "json_schema_extra": {
            "example": {
                "feedback_id": "fb-qwe456",
                "message_id": "msg-xyz789",
                "session_id": "thread-abc123",
                "helpful_score": 5,
                "resolution_status": "resolved",
                "comment": "代码示例非常清晰,问题完美解决!",
                "submitted_at": "2025-11-30T10:05:00Z"
            }
        }
    }


# ============================================================================
# T033: QuestionCategory - 问题分类模型
# ============================================================================

class QuestionCategory(BaseModel):
    """问题分类结果

    用于记录问题的自动分类结果,帮助路由到合适的处理流程。
    """

    main_category: MainCategory
    sub_category: Optional[str] = Field(default=None, description="子分类（如果有）")
    confidence: float = Field(ge=0, le=1, description="分类置信度")
    keywords: list[str] = Field(default_factory=list, description="提取的关键词")

    model_config = {
        "json_schema_extra": {
            "example": {
                "main_category": "technical",
                "sub_category": "debugging",
                "confidence": 0.95,
                "keywords": ["CUDA", "内存不足", "模型加载", "错误"]
            }
        }
    }


# ============================================================================
# T034: ConversationState - LangGraph 状态模型
# ============================================================================

class ConversationState(TypedDict):
    """Agent 核心对话状态

    LangGraph 工作流使用的状态模型,包含消息历史、检索结果、
    问题分类和生成的答案等信息。
    """

    # 消息历史（自动管理添加）
    messages: Annotated[list[BaseMessage], add_messages]

    # 检索到的文档
    retrieved_documents: list[Document]

    # 当前用户问题
    current_question: str

    # 改写后的问题（用于检索优化）
    rewritten_question: Optional[str]

    # 问题分类结果
    question_category: Optional[str]  # "model_usage", "technical", "platform", "project"

    # 生成的答案
    generated_answer: Optional[dict]  # TechnicalAnswer schema

    # 对话元数据
    thread_id: str  # 会话 ID
    turn_count: int  # 对话轮次计数
    last_updated: str  # ISO 格式时间戳

    # 早期对话摘要（滑动窗口之外的内容）
    conversation_summary: Optional[str]

    # 用户反馈
    user_feedback: Optional[dict]  # {"helpful": bool, "resolved": bool, "comment": str}

    # 澄清相关 (Phase 3.6: 主动澄清机制)
    needs_clarification: bool  # 是否需要澄清
    clarification_questions: list[str]  # 澄清问题列表

    # 错误信息（如果有）
    error: Optional[str]


# ============================================================================
# 辅助模型
# ============================================================================

class UserQuery(BaseModel):
    """用户查询输入验证"""

    question: str = Field(min_length=3, max_length=1000)
    thread_id: Optional[str] = Field(default=None, max_length=100)
    include_code: bool = Field(default=True, description="是否包含代码示例")
    max_results: int = Field(default=3, ge=1, le=10, description="最大返回结果数")

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """验证问题格式"""
        v = v.strip()
        if not v:
            raise ValueError("问题不能为空")
        if len(v.split()) < 2:
            raise ValueError("问题太短,至少包含2个词")
        return v
