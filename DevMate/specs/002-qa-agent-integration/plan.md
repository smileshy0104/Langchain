# Implementation Plan: 魔搭社区智能答疑 Agent 完整集成

**Feature**: 002-qa-agent-integration
**Branch**: `002-qa-agent-integration`
**Created**: 2025-12-01
**Status**: Planning
**Spec**: [spec.md](spec.md)

---

## 执行摘要

基于 Feature 001 已完成的所有核心组件(配置系统、向量数据库、文档处理、混合检索、Web 前端等),构建完整的智能答疑 Agent 系统,实现:
1. **Agent 编排层**: 统一的 Agent 控制器,编排单轮问答、多轮对话、主动澄清等核心能力
2. **会话管理**: 基于 Redis 的会话持久化和多用户隔离
3. **前后端集成**: 完整的多轮对话 UI 和实时状态提示
4. **部署工具**: 一键启动脚本和健康检查工具
5. **知识库管理**: 文档统计和删除功能

**核心技术栈**:
- **Agent 框架**: LangGraph (状态管理 + 多轮对话)
- **会话存储**: Redis (会话持久化 + TTL 自动过期)
- **流式输出**: Server-Sent Events (SSE)
- **前端框架**: 原生 JavaScript (保持轻量)
- **现有组件**: 复用 Feature 001 所有已实现的模块

---

## Technical Context

**Language/Version**: Python 3.11 (与 Feature 001 一致)
**Primary Dependencies**:
- LangGraph 0.1+ (Agent 编排和状态管理)
- Redis 7.0+ (会话存储)
- redis-py 5.0+ (Python Redis 客户端)
- FastAPI 0.123+ (已有,增强 SSE 支持)
- 复用 Feature 001 所有依赖 (LangChain, Milvus, VolcEngine 等)

**Storage**:
- Redis: 会话数据 (Session metadata + ConversationTurn history)
- Milvus: 向量数据库 (已有,Feature 001)
- MinIO: 文档存储 (已有,Feature 001)

**Testing**:
- pytest (已有测试框架)
- pytest-asyncio (异步测试)
- locust (压力测试,用于并发场景)

**Target Platform**: Linux server / macOS 开发环境
**Project Type**: Web application (Backend + Frontend)
**Performance Goals**:
- 单轮问答响应时间 ≤ 30s (P50), ≤ 60s (P95)
- 支持 ≥ 10 并发用户
- 会话恢复时间 ≤ 500ms

**Constraints**:
- 单机环境 (8 核 CPU, 16GB 内存)
- Redis 内存占用 ≤ 2GB (通过 TTL 和摘要策略控制)
- 前端应用大小 ≤ 500KB (未压缩)

**Scale/Scope**:
- 预计 10-50 并发用户
- 每个会话平均 5-10 轮对话
- 会话保留时间 24 小时 (可配置)

---

## Constitution Check

*GATE: 必须在 Phase 0 研究前通过,在 Phase 1 设计后重新检查*

### 简洁性原则 (Simplicity)

✅ **PASS**: Agent 设计遵循最小必要复杂度
- 优先使用 LangGraph 的内置状态管理,避免自定义状态机
- 会话存储使用 Redis 简单数据结构 (Hash + List),不引入 ORM
- 流式输出优先使用 SSE (单向通信),避免 WebSocket 的双向复杂性
- 前端使用原生 JavaScript,不引入 React/Vue 等框架

### 可复用性原则 (Reusability)

✅ **PASS**: 100% 复用 Feature 001 已有组件
- 配置系统 (`config.config_loader`)
- 向量存储 (`core.vector_store`)
- 文档处理 (`core.document_processor`)
- 混合检索 (`retrieval.hybrid_retrieval`)
- 文档上传 (`services.document_upload_service`)
- 现有 FastAPI 后端 (`api.main`)

### 可测试性原则 (Testability)

✅ **PASS**: 各模块可独立测试
- Agent 逻辑: 单元测试验证决策树 (单轮/多轮/澄清)
- 会话管理: 单元测试验证 CRUD 和过期逻辑
- 前后端集成: 集成测试验证完整流程
- 并发隔离: 压力测试验证多用户场景

### 可观测性原则 (Observability)

✅ **PASS**: 结构化日志和健康检查
- 所有 Agent 决策记录日志 (问题分类、澄清触发、检索结果)
- Redis 连接状态监控
- 会话统计指标 (活跃会话数、平均对话轮数)
- 健康检查脚本输出诊断报告

### 无违规项

本 Feature 设计完全符合项目架构原则,无需复杂度豁免。

---

## Project Structure

### Documentation (this feature)

```text
specs/002-qa-agent-integration/
├── spec.md              # Feature 规范 (已完成)
├── plan.md              # 本文件 - 实施计划
├── research.md          # Phase 0 输出 - 技术研究
├── data-model.md        # Phase 1 输出 - 数据模型
├── contracts/           # Phase 1 输出 - API 合约
│   ├── agent-api.yaml   # Agent API 定义
│   └── session-api.yaml # 会话管理 API 定义
└── tasks.md             # Phase 2 输出 (由 /speckit.tasks 生成)
```

### Source Code (repository root)

**注**: Feature 002 在 Feature 001 的基础上扩展,主要新增 Agent 模块和会话管理

```text
modelscope_qa_agent/
├── agents/                    # 新增: Agent 控制器模块
│   ├── __init__.py
│   ├── qa_agent.py           # 主 Agent 控制器 (LangGraph)
│   ├── state.py              # Agent 状态定义
│   ├── nodes.py              # LangGraph 节点 (问答/澄清/检索)
│   └── prompts.py            # Agent Prompt 模板
│
├── services/                  # 扩展: 新增会话管理服务
│   ├── document_upload_service.py  # 已有 (Feature 001)
│   └── session_manager.py    # 新增: 会话管理服务 (Redis)
│
├── api/                       # 扩展: 增强 FastAPI 后端
│   ├── main.py               # 已有,需扩展 (增加会话 API, SSE 端点)
│   ├── routers/              # 新增: API 路由模块化
│   │   ├── __init__.py
│   │   ├── qa.py             # 问答相关 API
│   │   ├── session.py        # 会话管理 API
│   │   └── admin.py          # 管理功能 API
│   └── static/
│       └── index.html        # 已有,需扩展 (多轮对话 UI)
│
├── scripts/                   # 新增: 部署和运维脚本
│   ├── setup.sh              # 环境初始化
│   ├── start.sh              # 一键启动
│   ├── status.sh             # 健康检查
│   └── docker-compose.yml    # Docker 服务编排
│
├── tests/                     # 扩展: 新增测试
│   ├── test_agent.py         # Agent 逻辑测试
│   ├── test_session_manager.py  # 会话管理测试
│   ├── test_integration.py   # 端到端集成测试
│   └── test_concurrent.py    # 并发测试
│
├── config/                    # 已有 (Feature 001)
│   ├── config.yaml           # 需扩展: 增加 agent 和 session 配置段
│   └── config_loader.py      # 已有
│
└── core/                      # 已有 (Feature 001,无需修改)
    ├── embeddings.py
    ├── vector_store.py
    └── document_processor.py
```

**结构决策**: 采用 Web application 结构,在 Feature 001 基础上扩展。新增模块清晰隔离:
- `agents/`: Agent 逻辑独立模块
- `services/session_manager.py`: 会话管理服务
- `api/routers/`: 模块化路由,便于维护
- `scripts/`: 运维工具独立目录

---

## Complexity Tracking

无违规项,无需填写。

---

## Phase 0: 研究与设计

**目标**: 解决所有技术不确定性,形成清晰的实施方案

### 研究任务

1. **LangGraph Agent 架构研究**
   - **问题**: 如何使用 LangGraph 实现单轮/多轮/澄清的统一 Agent?
   - **输出**: Agent 状态图设计、节点定义、状态转移逻辑
   - **参考**: LangGraph 官方文档 - `StateGraph` 和 `MemorySaver`

2. **Redis 会话存储方案研究**
   - **问题**: Redis 数据结构选择 (Hash vs List vs String)?
   - **输出**: 会话元数据存储结构、对话历史存储结构、TTL 策略
   - **参考**: Redis 最佳实践、内存优化策略

3. **SSE 流式输出实现研究**
   - **问题**: FastAPI 如何实现 SSE?如何处理网络中断?
   - **输出**: SSE 端点实现代码、错误恢复机制
   - **参考**: FastAPI StreamingResponse、SSE 规范

4. **对话历史摘要策略研究**
   - **问题**: 如何压缩早期对话为摘要?使用 LLM 还是规则?
   - **输出**: 摘要算法选择、触发条件 (如 > N 轮)
   - **参考**: LangChain ConversationSummaryMemory

5. **主动澄清决策逻辑研究**
   - **问题**: 如何判断问题是否需要澄清?如何生成澄清问题?
   - **输出**: 澄清触发条件 (置信度、实体识别)、Prompt 模板
   - **参考**: 实体识别 (NER)、问题完整性评估

### 输出文档

📄 **research.md** (约 3000 字)

章节结构:
```markdown
# Technical Research: 魔搭社区智能答疑 Agent 集成

## 1. LangGraph Agent 架构设计
- 状态定义 (AgentState)
- 节点实现 (问答节点、澄清节点、检索节点)
- 状态转移图
- 代码示例

## 2. Redis 会话存储方案
- 数据结构设计 (Hash + List)
- TTL 策略 (24 小时默认)
- 内存优化 (摘要压缩)
- 代码示例

## 3. SSE 流式输出实现
- FastAPI StreamingResponse
- 错误处理和重连
- 代码示例

## 4. 对话摘要策略
- 摘要算法 (LLM vs 规则)
- 触发条件
- Prompt 模板

## 5. 主动澄清逻辑
- 决策树 (什么时候澄清)
- 澄清问题生成 Prompt
- 代码示例
```

---

## Phase 1: 数据模型与合约设计

**目标**: 定义所有数据实体和 API 合约

### 1.1 数据模型设计

📄 **data-model.md** (约 2000 字)

核心实体:

#### Entity 1: Session (会话)

```python
@dataclass
class Session:
    session_id: str          # UUID, 主键
    user_id: Optional[str]   # 用户 ID (可选,当前版本未实现用户认证)
    created_at: datetime     # 创建时间
    last_active_at: datetime # 最后活跃时间
    metadata: Dict[str, Any] # 元数据 (如 user_agent, ip 等)

    # Redis 存储结构
    # Key: session:{session_id}
    # Type: Hash
    # Fields: {user_id, created_at, last_active_at, metadata}
    # TTL: 24 hours (可配置)
```

**状态转移**:
```
[Created] → [Active] → [Expired/Deleted]
```

**验证规则**:
- session_id 必须是有效的 UUID v4
- created_at ≤ last_active_at
- TTL 范围: 1 小时 - 7 天

#### Entity 2: ConversationTurn (对话轮次)

```python
@dataclass
class ConversationTurn:
    turn_id: int             # 自增 ID (从 1 开始)
    session_id: str          # 外键 → Session
    role: Literal["user", "assistant", "system"]
    content: str             # 内容 (用户问题 or Agent 回答)
    timestamp: datetime      # 时间戳
    sources: Optional[List[Source]]  # 来源引用 (仅 assistant 角色)
    is_clarification: bool   # 是否为澄清问题 (仅 assistant 角色)

    # Redis 存储结构
    # Key: conversation:{session_id}
    # Type: List
    # Value: JSON serialized ConversationTurn
```

**关系**:
- 一个 Session 包含多个 ConversationTurn
- 按 turn_id 顺序存储 (List RPUSH)

**验证规则**:
- role 必须是 "user" | "assistant" | "system"
- sources 仅在 role="assistant" 时有效
- is_clarification 仅在 role="assistant" 时有效

#### Entity 3: AgentState (Agent 状态)

```python
@dataclass
class AgentState:
    session_id: str                    # 外键 → Session
    current_question: str              # 当前问题
    context_summary: Optional[str]     # 上下文摘要 (超过 N 轮后生成)
    clarification_pending: bool        # 是否等待澄清回答
    retrieval_cache: Optional[List]    # 最近检索结果缓存 (避免重复检索)

    # Redis 存储结构
    # Key: agent_state:{session_id}
    # Type: Hash
    # TTL: 与 session 相同
```

**状态转移**:
```
[Initial] → [Retrieving] → [Generating] → [Answered]
                      ↓
                 [Clarifying] → [Waiting for Clarification] → [Retrieving]
```

#### Entity 4: Source (来源引用)

```python
@dataclass
class Source:
    document_id: str         # Milvus 文档 ID
    title: str               # 文档标题
    content_snippet: str     # 内容片段 (前 200 字符)
    source_url: Optional[str]  # 来源 URL
    relevance_score: float   # 相关度评分 (0-1)
```

### 1.2 API 合约设计

📄 **contracts/agent-api.yaml** (OpenAPI 规范)

```yaml
openapi: 3.0.0
info:
  title: 魔搭社区智能答疑 Agent API
  version: 2.0.0

paths:
  /api/v2/sessions:
    post:
      summary: 创建新会话
      responses:
        '201':
          description: 会话创建成功
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Session'

    get:
      summary: 获取用户的会话列表
      parameters:
        - name: limit
          in: query
          schema:
            type: integer
            default: 10
      responses:
        '200':
          description: 会话列表
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/Session'

  /api/v2/sessions/{session_id}:
    get:
      summary: 获取会话详情 (包含对话历史)
      parameters:
        - name: session_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: 会话详情
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SessionDetail'

    delete:
      summary: 删除会话
      parameters:
        - name: session_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '204':
          description: 删除成功

  /api/v2/qa/ask:
    post:
      summary: 发起问答 (支持单轮和多轮)
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                session_id:
                  type: string
                question:
                  type: string
                top_k:
                  type: integer
                  default: 3
      responses:
        '200':
          description: 回答成功
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AgentResponse'

  /api/v2/qa/stream:
    post:
      summary: 流式问答 (SSE)
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                session_id:
                  type: string
                question:
                  type: string
      responses:
        '200':
          description: SSE 流
          content:
            text/event-stream:
              schema:
                type: string

components:
  schemas:
    Session:
      type: object
      properties:
        session_id:
          type: string
        created_at:
          type: string
          format: date-time
        last_active_at:
          type: string
          format: date-time

    SessionDetail:
      allOf:
        - $ref: '#/components/schemas/Session'
        - type: object
          properties:
            conversation_history:
              type: array
              items:
                $ref: '#/components/schemas/ConversationTurn'

    ConversationTurn:
      type: object
      properties:
        turn_id:
          type: integer
        role:
          type: string
          enum: [user, assistant, system]
        content:
          type: string
        timestamp:
          type: string
          format: date-time
        sources:
          type: array
          items:
            $ref: '#/components/schemas/Source'
        is_clarification:
          type: boolean

    AgentResponse:
      type: object
      properties:
        answer:
          type: string
        sources:
          type: array
          items:
            $ref: '#/components/schemas/Source'
        is_clarification:
          type: boolean
        confidence:
          type: number
          format: float

    Source:
      type: object
      properties:
        document_id:
          type: string
        title:
          type: string
        content_snippet:
          type: string
        source_url:
          type: string
        relevance_score:
          type: number
          format: float
```

📄 **contracts/admin-api.yaml** (知识库管理 API)

```yaml
openapi: 3.0.0
info:
  title: 知识库管理 API
  version: 1.0.0

paths:
  /api/v2/admin/knowledge-base/stats:
    get:
      summary: 获取知识库统计信息
      responses:
        '200':
          description: 统计信息
          content:
            application/json:
              schema:
                type: object
                properties:
                  total_documents:
                    type: integer
                  total_vectors:
                    type: integer
                  storage_size_mb:
                    type: number
                  last_updated:
                    type: string
                    format: date-time
                  document_types:
                    type: object
                    additionalProperties:
                      type: integer

  /api/v2/admin/knowledge-base/documents/{document_id}:
    delete:
      summary: 删除文档 (包括向量和原文件)
      parameters:
        - name: document_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '204':
          description: 删除成功
```

### 1.3 快速开始指南

📄 **quickstart.md**

```markdown
# Quick Start: 魔搭社区智能答疑 Agent

## 前置要求

- Python 3.11+
- Docker 和 Docker Compose
- 8GB+ RAM
- VolcEngine API Key (豆包 Embedding + Chat)

## 一键启动

### 1. 环境初始化

bash
cd modelscope_qa_agent
./scripts/setup.sh


该脚本会:
- 检查 Python 版本和 Docker
- 安装所有依赖
- 启动 Docker 服务 (Milvus, MinIO, Redis)
- 初始化配置文件

### 2. 配置 API Key

编辑 `config/config.yaml`:

yaml
ai:
  provider: "volcengine"
  api_key: "YOUR_VOLCENGINE_API_KEY"  # 替换为您的密钥


### 3. 启动应用

bash
./scripts/start.sh


访问: http://localhost:8000

### 4. 上传文档

通过 Web 界面上传知识库文档,或使用 API:

bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@your_document.pdf" \
  -F "category=technical"


### 5. 开始问答

在 Web 界面输入问题,或使用 API:

bash
curl -X POST http://localhost:8000/api/v2/qa/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "如何使用魔搭社区的 Qwen 模型?",
    "session_id": "auto"
  }'


## 健康检查

bash
./scripts/status.sh


输出示例:
```
✅ Milvus: Running (localhost:19530)
✅ MinIO: Running (localhost:9000)
✅ Redis: Running (localhost:6379)
✅ FastAPI: Running (localhost:8000)
```

## 故障排查

见 [WEB_FRONTEND_GUIDE.md](../../modelscope_qa_agent/WEB_FRONTEND_GUIDE.md)
```

---

## Phase 2: 核心实现 (6个子阶段)

**注**: Phase 2 的详细任务将由 `/speckit.tasks` 命令生成到 `tasks.md`

### 2.1 Agent 核心逻辑 (Priority: P1)

**目标**: 实现 LangGraph Agent 控制器

**关键文件**:
- `agents/qa_agent.py`: 主 Agent 类
- `agents/state.py`: AgentState 定义
- `agents/nodes.py`: LangGraph 节点实现
- `agents/prompts.py`: Prompt 模板

**核心逻辑**:
```python
# agents/state.py
from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: List[BaseMessage]        # 对话历史
    question: str                       # 当前问题
    retrieved_docs: Optional[List]      # 检索结果
    need_clarification: bool            # 是否需要澄清
    clarification_questions: Optional[List[str]]  # 澄清问题
    final_answer: Optional[str]         # 最终答案

# agents/qa_agent.py
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.nodes import (
    question_analysis_node,
    retrieval_node,
    clarification_node,
    answer_generation_node
)

def create_agent() -> StateGraph:
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("analyze", question_analysis_node)
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("clarify", clarification_node)
    workflow.add_node("answer", answer_generation_node)

    # 定义边
    workflow.set_entry_point("analyze")

    workflow.add_conditional_edges(
        "analyze",
        lambda state: "clarify" if state["need_clarification"] else "retrieve"
    )

    workflow.add_edge("clarify", END)  # 澄清后等待用户回答
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("answer", END)

    return workflow.compile()
```

**验收标准**:
- ✅ 单轮问答正确响应
- ✅ 多轮对话能引用上下文
- ✅ 主动澄清能识别信息不足的问题

### 2.2 会话管理 (Priority: P1)

**目标**: 实现基于 Redis 的会话存储

**关键文件**:
- `services/session_manager.py`

**核心逻辑**:
```python
# services/session_manager.py
import redis
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Optional
from dataclasses import dataclass, asdict

@dataclass
class Session:
    session_id: str
    created_at: str
    last_active_at: str
    metadata: dict

class SessionManager:
    def __init__(self, redis_client: redis.Redis, ttl_hours: int = 24):
        self.redis = redis_client
        self.ttl = ttl_hours * 3600  # 转换为秒

    def create_session(self, metadata: dict = None) -> Session:
        session_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        session = Session(
            session_id=session_id,
            created_at=now,
            last_active_at=now,
            metadata=metadata or {}
        )

        # 存储到 Redis
        key = f"session:{session_id}"
        self.redis.hset(key, mapping=asdict(session))
        self.redis.expire(key, self.ttl)

        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        key = f"session:{session_id}"
        data = self.redis.hgetall(key)

        if not data:
            return None

        return Session(**{k.decode(): v.decode() for k, v in data.items()})

    def add_turn(self, session_id: str, turn: ConversationTurn):
        # 添加对话轮次到 List
        key = f"conversation:{session_id}"
        self.redis.rpush(key, json.dumps(asdict(turn)))
        self.redis.expire(key, self.ttl)

        # 更新 session 的 last_active_at
        self.redis.hset(
            f"session:{session_id}",
            "last_active_at",
            datetime.utcnow().isoformat()
        )

    def get_conversation_history(self, session_id: str) -> List[ConversationTurn]:
        key = f"conversation:{session_id}"
        turns = self.redis.lrange(key, 0, -1)

        return [
            ConversationTurn(**json.loads(turn))
            for turn in turns
        ]

    def delete_session(self, session_id: str):
        self.redis.delete(f"session:{session_id}")
        self.redis.delete(f"conversation:{session_id}")
        self.redis.delete(f"agent_state:{session_id}")
```

**验收标准**:
- ✅ 会话创建和恢复
- ✅ 对话历史持久化
- ✅ TTL 自动过期
- ✅ 多用户会话隔离

### 2.3 前后端集成 (Priority: P1)

**目标**: 扩展 Web 前端支持多轮对话

**修改文件**:
- `api/main.py`: 增加会话 API 和 SSE 端点
- `api/static/index.html`: 增加多轮对话 UI

**新增路由**:
```python
# api/routers/session.py
from fastapi import APIRouter, HTTPException
from services.session_manager import SessionManager
import redis

router = APIRouter(prefix="/api/v2/sessions", tags=["sessions"])

@router.post("/")
async def create_session():
    session = session_manager.create_session()
    return session

@router.get("/{session_id}")
async def get_session(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    history = session_manager.get_conversation_history(session_id)
    return {
        "session": session,
        "conversation_history": history
    }

@router.delete("/{session_id}")
async def delete_session(session_id: str):
    session_manager.delete_session(session_id)
    return {"status": "deleted"}

# api/routers/qa.py
@router.post("/ask")
async def ask_question(request: QuestionRequest):
    # 获取会话
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # 加载对话历史
    history = session_manager.get_conversation_history(request.session_id)

    # 调用 Agent
    agent = create_agent()
    result = agent.invoke({
        "messages": history,
        "question": request.question
    })

    # 保存对话轮次
    session_manager.add_turn(request.session_id, ConversationTurn(
        role="user",
        content=request.question,
        timestamp=datetime.utcnow()
    ))

    session_manager.add_turn(request.session_id, ConversationTurn(
        role="assistant",
        content=result["final_answer"],
        timestamp=datetime.utcnow(),
        sources=result.get("sources", []),
        is_clarification=result.get("need_clarification", False)
    ))

    return {
        "answer": result["final_answer"],
        "sources": result.get("sources", []),
        "is_clarification": result.get("need_clarification", False)
    }
```

**前端更新** (index.html):
```javascript
// 新增: 会话管理
let currentSessionId = null;

async function initSession() {
    const response = await fetch('/api/v2/sessions', {
        method: 'POST'
    });
    const session = await response.json();
    currentSessionId = session.session_id;
    loadConversationHistory();
}

async function loadConversationHistory() {
    if (!currentSessionId) return;

    const response = await fetch(`/api/v2/sessions/${currentSessionId}`);
    const data = await response.json();

    // 渲染对话历史
    displayConversationHistory(data.conversation_history);
}

async function askQuestion() {
    const question = questionInput.value;

    // 显示用户问题
    appendMessage('user', question);

    // 调用 API
    const response = await fetch('/api/v2/qa/ask', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            session_id: currentSessionId,
            question: question
        })
    });

    const result = await response.json();

    // 显示 Agent 回答
    appendMessage('assistant', result.answer, result.sources, result.is_clarification);
}

function appendMessage(role, content, sources, isClarification) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    // 添加内容
    messageDiv.innerHTML = `<p>${content}</p>`;

    // 如果是澄清问题,高亮显示
    if (isClarification) {
        messageDiv.classList.add('clarification');
    }

    // 添加来源
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';
        sourcesDiv.innerHTML = '<strong>来源:</strong><ul>' +
            sources.map(s => `<li><a href="${s.source_url}">${s.title}</a></li>`).join('') +
            '</ul>';
        messageDiv.appendChild(sourcesDiv);
    }

    chatContainer.appendChild(messageDiv);
}

// 页面加载时初始化会话
window.onload = initSession;
```

**验收标准**:
- ✅ 多轮对话 UI 正常显示
- ✅ 会话切换功能可用
- ✅ 澄清问题高亮显示
- ✅ 来源引用正确展示

### 2.4 流式输出 (Priority: P2)

**目标**: 实现 SSE 流式输出

**新增端点**:
```python
# api/routers/qa.py
from fastapi.responses import StreamingResponse

@router.post("/stream")
async def stream_answer(request: QuestionRequest):
    async def event_stream():
        # 调用 Agent (流式)
        agent = create_agent()

        async for chunk in agent.astream({
            "messages": history,
            "question": request.question
        }):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

**前端 SSE 客户端**:
```javascript
async function askQuestionStream() {
    const question = questionInput.value;

    const eventSource = new EventSource('/api/v2/qa/stream', {
        method: 'POST',
        body: JSON.stringify({
            session_id: currentSessionId,
            question: question
        })
    });

    let answerText = '';

    eventSource.onmessage = (event) => {
        if (event.data === '[DONE]') {
            eventSource.close();
            return;
        }

        const data = JSON.parse(event.data);
        answerText += data.chunk;
        updateAnswer(answerText);  // 逐字更新
    };

    eventSource.onerror = () => {
        eventSource.close();
        showError('连接中断,请重试');
    };
}
```

**验收标准**:
- ✅ 流式输出逐字显示
- ✅ 网络中断后能恢复
- ✅ 完成后正确关闭连接

### 2.5 部署与运维 (Priority: P2)

**目标**: 编写一键启动和健康检查脚本

**scripts/setup.sh**:
```bash
#!/bin/bash
set -e

echo "=== 魔搭社区智能答疑 Agent - 环境初始化 ==="

# 检查 Python 版本
python_version=$(python3 --version | awk '{print $2}')
echo "Python 版本: $python_version"

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装"
    exit 1
fi

# 启动 Docker 服务
echo "启动 Docker 服务..."
docker-compose up -d

# 等待服务就绪
echo "等待服务启动..."
sleep 10

# 安装 Python 依赖
echo "安装 Python 依赖..."
pip install -r requirements.txt

# 初始化配置文件
if [ ! -f "config/config.yaml" ]; then
    cp config/config.yaml.example config/config.yaml
    echo "⚠️  请编辑 config/config.yaml 填写 API Key"
fi

echo "✅ 环境初始化完成!"
```

**scripts/start.sh**:
```bash
#!/bin/bash
set -e

echo "=== 启动魔搭社区智能答疑 Agent ==="

# 检查 Docker 服务
./scripts/status.sh

# 启动应用
echo "启动 FastAPI 应用..."
cd modelscope_qa_agent
uvicorn api.main:app --host 0.0.0.0 --port 8000

echo "✅ 应用已启动: http://localhost:8000"
```

**scripts/status.sh**:
```bash
#!/bin/bash

echo "=== 系统健康检查 ==="

# 检查 Milvus
if nc -z localhost 19530; then
    echo "✅ Milvus: Running (localhost:19530)"
else
    echo "❌ Milvus: Not Running"
fi

# 检查 MinIO
if nc -z localhost 9000; then
    echo "✅ MinIO: Running (localhost:9000)"
else
    echo "❌ MinIO: Not Running"
fi

# 检查 Redis
if nc -z localhost 6379; then
    echo "✅ Redis: Running (localhost:6379)"
else
    echo "❌ Redis: Not Running"
fi

# 检查 FastAPI
if nc -z localhost 8000; then
    echo "✅ FastAPI: Running (localhost:8000)"
else
    echo "❌ FastAPI: Not Running"
fi
```

**验收标准**:
- ✅ 一键启动所有服务
- ✅ 健康检查准确显示状态
- ✅ 错误提示清晰友好

### 2.6 知识库管理 (Priority: P3)

**目标**: 实现知识库统计和文档删除

**新增 API**:
```python
# api/routers/admin.py
@router.get("/knowledge-base/stats")
async def get_kb_stats():
    vector_store = doc_service.vector_store
    collection = vector_store.collection

    # 统计信息
    stats = {
        "total_documents": collection.num_entities,
        "total_vectors": collection.num_entities,
        "storage_size_mb": collection.num_entities * 2560 * 4 / (1024 * 1024),  # 估算
        "last_updated": datetime.utcnow().isoformat(),
        "document_types": {}  # TODO: 按类型统计
    }

    return stats

@router.delete("/knowledge-base/documents/{document_id}")
async def delete_document(document_id: str):
    # 从 Milvus 删除向量
    vector_store.delete([document_id])

    # 从 MinIO 删除原文件 (TODO: 需要文档 ID 到文件路径的映射)

    return {"status": "deleted", "document_id": document_id}
```

**前端展示**:
```javascript
async function loadKBStats() {
    const response = await fetch('/api/v2/admin/knowledge-base/stats');
    const stats = await response.json();

    document.getElementById('total-docs').textContent = stats.total_documents;
    document.getElementById('total-vectors').textContent = stats.total_vectors;
    document.getElementById('storage-size').textContent = stats.storage_size_mb.toFixed(2) + ' MB';
}
```

**验收标准**:
- ✅ 统计信息正确显示
- ✅ 文档删除功能可用
- ✅ 删除后同步清理所有数据

---

## Testing Strategy

### Unit Tests (pytest)

```python
# tests/test_agent.py
def test_single_turn_qa():
    agent = create_agent()
    result = agent.invoke({
        "question": "什么是模型微调?",
        "messages": []
    })

    assert result["final_answer"] is not None
    assert "微调" in result["final_answer"]
    assert result["need_clarification"] == False

def test_clarification_trigger():
    agent = create_agent()
    result = agent.invoke({
        "question": "模型报错了",  # 信息不足
        "messages": []
    })

    assert result["need_clarification"] == True
    assert len(result["clarification_questions"]) > 0

# tests/test_session_manager.py
def test_session_crud():
    session = session_manager.create_session()
    assert session.session_id is not None

    retrieved = session_manager.get_session(session.session_id)
    assert retrieved.session_id == session.session_id

    session_manager.delete_session(session.session_id)
    assert session_manager.get_session(session.session_id) is None

def test_conversation_history():
    session = session_manager.create_session()

    turn1 = ConversationTurn(role="user", content="问题1", timestamp=datetime.utcnow())
    session_manager.add_turn(session.session_id, turn1)

    turn2 = ConversationTurn(role="assistant", content="回答1", timestamp=datetime.utcnow())
    session_manager.add_turn(session.session_id, turn2)

    history = session_manager.get_conversation_history(session.session_id)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"
```

### Integration Tests

```python
# tests/test_integration.py
def test_end_to_end_qa():
    # 1. 创建会话
    response = client.post("/api/v2/sessions")
    session_id = response.json()["session_id"]

    # 2. 发起问答
    response = client.post("/api/v2/qa/ask", json={
        "session_id": session_id,
        "question": "如何使用 Qwen 模型?"
    })

    assert response.status_code == 200
    result = response.json()
    assert "answer" in result
    assert len(result["sources"]) > 0

    # 3. 多轮对话
    response = client.post("/api/v2/qa/ask", json={
        "session_id": session_id,
        "question": "它支持哪些任务?"  # 上下文引用
    })

    assert response.status_code == 200
    result = response.json()
    assert "Qwen" in result["answer"] or "模型" in result["answer"]

def test_multi_user_isolation():
    # 创建两个会话
    session1 = client.post("/api/v2/sessions").json()["session_id"]
    session2 = client.post("/api/v2/sessions").json()["session_id"]

    # 用户1提问
    client.post("/api/v2/qa/ask", json={
        "session_id": session1,
        "question": "问题A"
    })

    # 用户2提问
    client.post("/api/v2/qa/ask", json={
        "session_id": session2,
        "question": "问题B"
    })

    # 验证隔离
    history1 = client.get(f"/api/v2/sessions/{session1}").json()["conversation_history"]
    history2 = client.get(f"/api/v2/sessions/{session2}").json()["conversation_history"]

    assert history1[0]["content"] == "问题A"
    assert history2[0]["content"] == "问题B"
```

### System Tests

```bash
# tests/system/test_deployment.sh
#!/bin/bash

# 测试一键启动
./scripts/setup.sh
./scripts/start.sh &

sleep 10

# 测试健康检查
./scripts/status.sh | grep "✅"

# 测试 API 可用性
curl -f http://localhost:8000/api/health

# 清理
pkill -f "uvicorn api.main"
docker-compose down
```

### Performance Tests (locust)

```python
# tests/test_concurrent.py
from locust import HttpUser, task, between

class QAUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        # 创建会话
        response = self.client.post("/api/v2/sessions")
        self.session_id = response.json()["session_id"]

    @task
    def ask_question(self):
        self.client.post("/api/v2/qa/ask", json={
            "session_id": self.session_id,
            "question": "如何使用魔搭社区?"
        })

# 运行: locust -f tests/test_concurrent.py --users 10 --spawn-rate 2
```

---

## Risks and Mitigations

### Risk 1: LangGraph 学习曲线

**风险**: 团队不熟悉 LangGraph,可能导致开发延期

**缓解**:
- Phase 0 研究阶段深入学习 LangGraph 文档和示例
- 先实现简化版 Agent (基于 LangChain ConversationChain)
- 核心功能完成后再迁移到 LangGraph

### Risk 2: Redis 内存占用过高

**风险**: 大量会话导致 Redis 内存溢出

**缓解**:
- 严格配置 TTL (默认 24 小时)
- 对话历史超过 10 轮后强制摘要
- 监控 Redis 内存使用,设置 `maxmemory` 和 `allkeys-lru` 淘汰策略
- 定期清理过期会话 (Cron 任务)

### Risk 3: 多轮对话上下文理解不准确

**风险**: Agent 无法正确理解上下文引用 (如"它"、"刚才提到的")

**缓解**:
- 使用 LLM 重写问题,将上下文引用展开为完整问题
- 在 Prompt 中明确指示 LLM 利用对话历史
- 收集测试案例,持续优化 Prompt 工程
- 实现"重新生成"功能,允许用户反馈不准确的回答

### Risk 4: 流式输出实现复杂度高

**风险**: SSE 实现可能遇到浏览器兼容性或网络问题

**缓解**:
- 第一版不实现流式输出,使用传统请求-响应模式
- 流式输出作为增强功能在 v1.1 版本实现
- 提供降级方案 (检测 SSE 不可用时自动切换)

---

## Success Criteria Checklist

- [ ] **SC-001**: 单轮问答响应时间 ≤ 30s (P50), ≤ 60s (P95)
- [ ] **SC-002**: 支持 ≥ 10 并发用户,每轮回答 ≤ 5s (P90)
- [ ] **SC-003**: 多轮对话准确率 ≥ 85%
- [ ] **SC-004**: 主动澄清准确率 ≥ 80%
- [ ] **SC-005**: 会话恢复成功率 100%
- [ ] **SC-006**: 系统启动成功率 ≥ 95%
- [ ] **SC-007**: 首次部署成功率 ≥ 90%
- [ ] **SC-008**: 文档上传成功率 ≥ 98%
- [ ] **SC-009**: 用户满意度 ≥ 4.0/5.0 (后期收集)

---

## Next Steps

1. **Phase 0 研究**: 执行研究任务,输出 `research.md`
2. **Phase 1 设计**: 完善数据模型和 API 合约,输出 `data-model.md` 和 `contracts/`
3. **Phase 2 任务生成**: 运行 `/speckit.tasks` 生成详细任务清单 (`tasks.md`)
4. **Phase 3 实施**: 按照 tasks.md 执行开发任务

---

**计划完成日期**: 2025-12-01
**预计实施周期**: 2-3 周 (取决于团队规模)
