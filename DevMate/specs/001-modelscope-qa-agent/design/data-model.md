# Data Model: 魔搭社区智能答疑 Agent

**Feature**: 001-modelscope-qa-agent
**Created**: 2025-11-30
**Based on**: LangChain v1.0 + Milvus + Qwen

## 1. LangGraph 状态模型

### 1.1 ConversationState (对话状态)

```python
from typing import TypedDict, Annotated, Optional
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

class ConversationState(TypedDict):
    """Agent 核心对话状态"""

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

    # 错误信息（如果有）
    error: Optional[str]
```

**字段说明**:
- `messages`: 使用 `add_messages` reducer 自动合并新消息
- `retrieved_documents`: 存储检索结果,供答案生成使用
- `rewritten_question`: 经过 LLM 改写的优化查询
- `conversation_summary`: 当消息超过10轮时,早期消息压缩为摘要

---

## 2. Pydantic 数据模型

### 2.1 知识库文档 (Knowledge Entry)

```python
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

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

class KnowledgeEntry(BaseModel):
    """知识库条目模型（存储在 Milvus）"""

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
    tags: list[str] = Field(default=[], description="标签列表")
    question_categories: list[str] = Field(
        default=[],
        description="适用的问题分类（模型使用/技术调试/平台功能/项目指导）"
    )

    # 向量和评分
    embedding_vector: list[float] = Field(description="通义千问 Embedding 向量（1536维）")
    quality_score: float = Field(ge=0, le=1, description="质量评分（0-1）")

    # 时间戳
    created_at: datetime
    last_updated: datetime

    class Config:
        json_schema_extra = {
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
                "quality_score": 0.95,
                "created_at": "2025-11-30T10:00:00Z",
                "last_updated": "2025-11-30T10:00:00Z"
            }
        }
```

---

### 2.2 技术回答 (Technical Answer)

```python
from pydantic import BaseModel, Field, validator

class TechnicalAnswer(BaseModel):
    """技术问答响应格式（符合 FR-005 结构化输出要求）"""

    # 问题分析
    problem_analysis: str = Field(
        description="问题分析,说明用户遇到的问题类型和核心痛点"
    )

    # 解决方案列表
    solutions: list[str] = Field(
        min_items=1,
        description="解决方案步骤列表,每个方案应独立完整"
    )

    # 代码示例
    code_examples: list[str] = Field(
        default=[],
        description="可运行的代码示例,使用 Markdown 代码块格式"
    )

    # 参数配置说明
    configuration_notes: list[str] = Field(
        default=[],
        description="参数配置说明和最佳实践"
    )

    # 注意事项
    warnings: list[str] = Field(
        default=[],
        description="注意事项和常见错误"
    )

    # 参考来源
    references: list[str] = Field(
        default=[],
        description="信息来源（官方文档章节或真实问答案例）"
    )

    # 置信度评分
    confidence_score: float = Field(
        ge=0, le=1,
        description="答案置信度（基于检索文档相关性）"
    )

    @validator("solutions")
    def validate_solutions(cls, v):
        """确保至少有一个解决方案"""
        if not v:
            raise ValueError("必须提供至少一种解决方案")
        return v

    @validator("code_examples")
    def validate_code_format(cls, v):
        """验证代码示例格式"""
        for code in v:
            if not code.strip().startswith("```"):
                raise ValueError("代码示例必须使用 Markdown 代码块格式")
        return v

    class Config:
        json_schema_extra = {
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
```

---

### 2.3 对话会话 (Dialogue Session)

```python
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class SessionStatus(str, Enum):
    """会话状态"""
    ACTIVE = "active"  # 进行中
    COMPLETED = "completed"  # 已完成
    ABANDONED = "abandoned"  # 已放弃

class DialogueSession(BaseModel):
    """对话会话元数据（存储在数据库）"""

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

    class Config:
        json_schema_extra = {
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
```

---

### 2.4 消息记录 (Message Record)

```python
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

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

class MessageRecord(BaseModel):
    """消息记录（存储在数据库）"""

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
        default=[],
        description="引用的知识库条目 ID"
    )
    tool_calls: list[str] = Field(
        default=[],
        description="调用的工具列表"
    )
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0, le=1,
        description="答案置信度（仅 assistant）"
    )

    class Config:
        json_schema_extra = {
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
```

---

### 2.5 用户反馈 (User Feedback)

```python
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class ResolutionStatus(str, Enum):
    """问题解决状态"""
    RESOLVED = "resolved"  # 已解决
    PARTIALLY_RESOLVED = "partially_resolved"  # 部分解决
    UNRESOLVED = "unresolved"  # 未解决

class UserFeedback(BaseModel):
    """用户反馈记录（符合 FR-016 要求）"""

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

    class Config:
        json_schema_extra = {
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
```

---

### 2.6 问题分类 (Question Category)

```python
from pydantic import BaseModel, Field
from enum import Enum

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

class QuestionCategory(BaseModel):
    """问题分类结果"""

    main_category: MainCategory
    sub_category: Optional[str] = Field(default=None, description="子分类（如果有）")
    confidence: float = Field(ge=0, le=1, description="分类置信度")
    keywords: list[str] = Field(default=[], description="提取的关键词")

    class Config:
        json_schema_extra = {
            "example": {
                "main_category": "technical",
                "sub_category": "debugging",
                "confidence": 0.95,
                "keywords": ["CUDA", "内存不足", "模型加载", "错误"]
            }
        }
```

---

## 3. Milvus Collection Schema

### 3.1 Collection 定义

```python
from pymilvus import CollectionSchema, FieldSchema, DataType

# 定义 Schema
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
    FieldSchema(name="content_summary", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="source_url", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="document_type", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="chunk_boundary", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="tags", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=20, max_length=100),
    FieldSchema(name="question_categories", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=10, max_length=100),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),  # 通义千问 Embedding 维度
    FieldSchema(name="quality_score", dtype=DataType.FLOAT),
    FieldSchema(name="created_at", dtype=DataType.INT64),  # Unix timestamp
    FieldSchema(name="last_updated", dtype=DataType.INT64)
]

schema = CollectionSchema(
    fields=fields,
    description="ModelScope Q&A Knowledge Base",
    enable_dynamic_field=True
)
```

### 3.2 索引配置

```python
from pymilvus import Collection

# 创建 Collection
collection = Collection(name="modelscope_docs", schema=schema)

# 向量索引（IVF_FLAT 适合中等规模数据）
index_params = {
    "metric_type": "IP",  # 内积（适合归一化向量）
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}
collection.create_index(field_name="embedding", index_params=index_params)

# 标量字段索引（加速过滤）
collection.create_index(field_name="source_type")
collection.create_index(field_name="document_type")
collection.create_index(field_name="quality_score")
```

---

## 4. 数据关系图

```
┌─────────────────────────────────────────────────────────────┐
│                    DialogueSession                          │
│  - session_id (PK)                                          │
│  - user_id                                                  │
│  - turn_count                                               │
│  - status                                                   │
└─────────────────────────────────────────────────────────────┘
                        │
                        │ 1:N
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    MessageRecord                            │
│  - message_id (PK)                                          │
│  - session_id (FK)                                          │
│  - role                                                     │
│  - content                                                  │
│  - retrieved_doc_ids ──────┐                                │
└─────────────────────────────┼───────────────────────────────┘
                              │
                              │ N:M
                              ▼
┌─────────────────────────────────────────────────────────────┐
│             KnowledgeEntry (Milvus)                         │
│  - entry_id (PK)                                            │
│  - title                                                    │
│  - content                                                  │
│  - embedding_vector [1536]                                  │
│  - quality_score                                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    UserFeedback                             │
│  - feedback_id (PK)                                         │
│  - message_id (FK)                                          │
│  - helpful_score                                            │
│  - resolution_status                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 状态转换图

### 5.1 对话会话状态转换

```
        [创建会话]
            │
            ▼
       ┌─────────┐
       │ ACTIVE  │ ◄──┐ 用户继续对话
       └─────────┘    │
            │         │
            │  ┌──────┴────────┐
            │  │               │
            ▼  ▼               │
       用户主动结束      超时无响应
            │               │
            ▼               ▼
     ┌────────────┐   ┌──────────┐
     │ COMPLETED  │   │ ABANDONED│
     └────────────┘   └──────────┘
```

### 5.2 LangGraph 节点状态流转

```
  [用户输入]
      │
      ▼
 ┌───────────┐
 │ 问题分类  │ ─── (分类失败) ──► [请求澄清]
 └───────────┘
      │ (分类成功)
      ▼
 ┌───────────┐
 │ 问题改写  │ ─── (可选) ──────► [原问题检索]
 └───────────┘
      │
      ▼
 ┌───────────┐
 │ 混合检索  │ ─── (结果不足) ──► [扩展查询]
 └───────────┘
      │ (结果充分)
      ▼
 ┌───────────┐
 │ 答案生成  │ ─── (置信度低) ──► [人工审核]
 └───────────┘
      │ (置信度高)
      ▼
 ┌───────────┐
 │ 返回答案  │
 └───────────┘
      │
      ▼
 ┌───────────┐
 │ 收集反馈  │
 └───────────┘
```

---

## 6. 数据验证规则

### 6.1 输入验证

```python
from pydantic import validator, Field

class UserQuery(BaseModel):
    """用户查询输入验证"""

    question: str = Field(min_length=3, max_length=1000)
    thread_id: Optional[str] = Field(default=None, max_length=100)
    include_code: bool = Field(default=True, description="是否包含代码示例")
    max_results: int = Field(default=3, ge=1, le=10, description="最大返回结果数")

    @validator("question")
    def validate_question(cls, v):
        """验证问题格式"""
        if v.strip() == "":
            raise ValueError("问题不能为空")
        if len(v.split()) < 2:
            raise ValueError("问题太短,至少包含2个词")
        return v.strip()
```

---

## 7. 数据持久化策略

### 7.1 存储分层

| 数据类型 | 存储系统 | 理由 |
|---------|---------|------|
| 对话状态 | LangGraph Checkpointer (SQLite/PostgreSQL) | 原生支持,便于恢复 |
| 知识库文档 | Milvus | 高效向量检索 |
| 会话元数据 | PostgreSQL/MySQL | 结构化数据,便于查询统计 |
| 消息历史 | PostgreSQL/MySQL | 关系型数据,支持复杂查询 |
| 用户反馈 | PostgreSQL/MySQL | 需要关联查询和分析 |
| 缓存数据 | Redis | 高速读写 |

---

**数据模型版本**: 1.0
**最后更新**: 2025-11-30
**审核状态**: 待审核
