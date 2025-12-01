# Feature Specification: 魔搭社区智能答疑 Agent 完整集成

**Feature Branch**: `002-qa-agent-integration`
**Created**: 2025-12-01
**Status**: Draft
**Input**: User description: "目前大部分的DevMate/specs/001-modelscope-qa-agent,都已经完成,但是这些完成的代码没有能够被正常的使用,希望可以配合前端页面完成"魔搭社区智能答疑 Agent""

## 背景说明

在 Feature 001 (modelscope-qa-agent) 中,我们已经完成了以下核心组件的开发:

### 已完成的后端组件 (Phase 1-12)
1. **配置系统** (Phase 1): YAML 配置加载、多 AI 服务提供商支持
2. **Embedding 服务** (Phase 2): VolcEngine/DashScope/OpenAI 向量化
3. **向量数据库** (Phase 3): Milvus 集成、Collection 管理、索引配置
4. **文档加载器** (Phase 4): 多格式文档解析 (PDF/DOCX/TXT/MD/JSON/XML/HTML/RTF)
5. **文档清洗** (Phase 5): 文本规范化、特殊字符处理、格式标准化
6. **文档分块** (Phase 6): 语义分块、章节分块、代码块分块
7. **质量评分** (Phase 7): 文档质量自动评估
8. **存储管理** (Phase 8): MinIO/本地存储、文件上传下载
9. **混合检索** (Phase 9): 向量检索 + BM25 检索、加权融合
10. **单轮问答** (Phase 10): 基本 RAG 问答、上下文增强
11. **文档上传服务** (Phase 11): 完整文档处理流程
12. **Web 前端** (Phase 12): FastAPI 后端 + HTML/CSS/JS 前端

### 当前问题

虽然所有核心组件都已开发完成并通过测试,但系统作为一个**完整的智能答疑 Agent** 还无法被最终用户正常使用,主要问题包括:

1. **缺少完整的 Agent 编排逻辑**: 单轮问答、多轮对话、主动澄清等功能分散在不同模块,没有统一的 Agent 控制器进行编排
2. **缺少对话记忆管理**: 虽然设计了滑动窗口+摘要策略,但未实现对话历史的持久化和管理
3. **缺少会话状态管理**: 多个用户并发使用时,没有会话隔离和状态管理机制
4. **前后端集成不完整**: Web 前端提供了基础 UI,但仅支持单次问答,不支持多轮对话和澄清交互
5. **缺少部署和运维工具**: 没有统一的启动脚本、健康检查、日志管理等生产环境必需功能

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 完整的单次技术问答流程 (Priority: P1)

用户通过 Web 界面提出一个技术问题,系统能够完整执行:文档检索 → 上下文构建 → LLM 生成答案 → 展示答案和来源,全流程无需人工干预。

**Why this priority**: 这是 Agent 的核心功能,是系统可用性的基础。只有这个流程完全打通,Agent 才能提供实际价值。

**Independent Test**:
1. 启动系统(一键启动脚本)
2. 在 Web 界面输入问题 "如何使用魔搭社区的 Qwen 模型?"
3. 系统返回准确答案,包含代码示例和文档链接
4. 整个过程在 30 秒内完成

**Acceptance Scenarios**:

1. **Given** 用户访问 Web 界面, **When** 输入问题并点击提交, **Then** 系统显示加载状态,并在 30 秒内返回包含答案文本、置信度评分和 2-3 个来源链接的完整回答
2. **Given** 用户提出的问题知识库中没有相关信息, **When** 检索完成但置信度低于阈值, **Then** 系统诚实告知"抱歉,我在知识库中未找到相关信息",并建议用户改写问题或联系人工客服
3. **Given** 用户提问后系统正在处理, **When** 用户刷新页面或重新访问, **Then** 问题处理状态不丢失,用户能看到之前的提问和回答历史
4. **Given** 向量数据库为空(首次部署), **When** 用户提问, **Then** 系统友好提示"知识库尚未初始化,请先上传文档"

---

### User Story 2 - 多轮对话与上下文理解 (Priority: P1)

用户与 Agent 进行多轮对话,系统能够记住之前的对话内容,理解上下文引用(如"刚才你说的方法"),并基于对话历史提供连贯的回答。

**Why this priority**: 多轮对话是复杂问题排查的必要能力,也是区分智能 Agent 与简单搜索系统的关键特征。这直接影响用户体验和问题解决率。

**Independent Test**:
1. 第一轮提问:"什么是模型微调?"
2. 第二轮提问:"有哪些常见的微调方法?"
3. 第三轮提问:"刚才提到的 LoRA 方法具体怎么用?"
4. 验证第三轮回答正确引用了第二轮提到的 "LoRA"

**Acceptance Scenarios**:

1. **Given** 用户第一轮问"Qwen 模型支持哪些任务", **When** 第二轮追问"它的最大上下文长度是多少", **Then** 系统理解"它"指代 Qwen 模型,并基于第一轮上下文给出准确答案
2. **Given** 用户在第三轮提到"你刚才推荐的方法试过了不行", **When** Agent 检索对话历史, **Then** 识别出第二轮的推荐内容,并提供替代方案
3. **Given** 对话持续超过 10 轮(超过上下文窗口), **When** 系统应用滑动窗口策略, **Then** 保留最近 5 轮完整对话,更早的对话压缩为摘要,确保上下文不丢失关键信息
4. **Given** 用户在对话中提供多种类型信息(文本描述、代码片段、配置参数), **When** Agent 整合信息, **Then** 构建完整的问题上下文并给出综合建议

---

### User Story 3 - 主动澄清与引导 (Priority: P2)

当用户问题不清晰或信息不足时,Agent 能够主动识别缺失信息,并提出 2-3 个针对性的澄清问题,引导用户提供必要的上下文。

**Why this priority**: 主动澄清能显著提高首次回答的准确率,减少用户反复提问的次数,提升用户满意度。虽然重要,但相比基础问答功能属于增强特性。

**Independent Test**:
1. 用户提问:"模型报错了怎么办?"(信息不足)
2. Agent 识别缺失信息并询问:"请问您遇到的具体错误信息是什么?使用的是哪个模型?运行环境如何?"
3. 用户补充信息后,Agent 给出针对性解决方案

**Acceptance Scenarios**:

1. **Given** 用户提问"模型加载失败", **When** Agent 分析问题发现缺少关键信息, **Then** 主动询问:"1) 您使用的是哪个模型? 2) 具体的错误信息是什么? 3) Python 和相关库的版本是?"
2. **Given** 用户回答了澄清问题, **When** Agent 获得补充信息, **Then** 结合原始问题和澄清信息,给出完整准确的解决方案,并在回答中引用用户提供的具体信息
3. **Given** 用户对澄清问题不回答或回答不完整, **When** Agent 检测到仍然缺乏关键信息, **Then** 基于现有信息给出通用建议,并标注"如提供更多信息(如错误日志),可获得更精准的解决方案"
4. **Given** 用户问题本身已经很清晰和完整, **When** Agent 评估不需要澄清, **Then** 直接给出答案,不进行不必要的追问,避免影响用户体验

---

### User Story 4 - 会话管理与多用户支持 (Priority: P2)

系统支持多个用户同时使用,每个用户的对话会话独立隔离,用户可以查看历史会话,继续之前的对话,或开始新会话。

**Why this priority**: 会话管理是生产环境的必要功能,确保多用户并发场景下的数据隔离和用户体验。但相比核心问答能力,属于基础设施层面的支持功能。

**Independent Test**:
1. 用户 A 和用户 B 同时访问系统
2. 用户 A 提问并得到回答
3. 用户 B 提问,验证回答不受用户 A 的对话影响
4. 用户 A 刷新页面,验证对话历史仍然存在

**Acceptance Scenarios**:

1. **Given** 用户首次访问系统, **When** 发起第一个问题, **Then** 系统自动创建新会话 ID(如 UUID),并在后续交互中保持会话状态
2. **Given** 用户 A 和用户 B 同时使用系统, **When** 两人分别进行多轮对话, **Then** 各自的对话历史、上下文记忆完全隔离,互不干扰
3. **Given** 用户在会话中进行了 5 轮对话后关闭浏览器, **When** 24 小时内重新访问(通过会话 ID 或 Cookie), **Then** 能够恢复之前的对话历史,继续之前的话题
4. **Given** 用户点击"新建会话"按钮, **When** 系统创建新会话, **Then** 之前的对话历史不再影响新会话,但旧会话仍可通过会话列表访问

---

### User Story 5 - 系统部署与运维 (Priority: P3)

运维人员或开发者能够通过简单的命令一键启动所有依赖服务(Milvus/MinIO/Redis)和应用服务,查看系统健康状态,访问日志,进行故障排查。

**Why this priority**: 部署和运维是系统可用性的基础保障,但对于功能开发优先级较低。可以在核心功能完成后逐步完善。

**Independent Test**:
1. 在全新环境执行 `./start.sh`
2. 系统自动启动所有依赖服务和应用
3. 访问 http://localhost:8000 验证服务可用
4. 执行 `./status.sh` 查看所有服务状态

**Acceptance Scenarios**:

1. **Given** 在新环境首次部署, **When** 执行 `./scripts/setup.sh`, **Then** 脚本自动检查依赖(Python 版本、Docker)、初始化配置文件、创建数据目录、启动 Docker 容器(Milvus/MinIO/Redis)
2. **Given** 依赖服务已启动, **When** 执行 `./scripts/start.sh`, **Then** 启动 FastAPI 应用,输出启动日志,并在所有服务就绪后显示访问地址
3. **Given** 系统正在运行, **When** 执行 `./scripts/status.sh`, **Then** 显示所有服务的健康状态(Running/Stopped)、资源占用(CPU/内存)、端口监听情况
4. **Given** 某个依赖服务(如 Milvus)未启动, **When** 应用启动时连接失败, **Then** 显示友好的错误提示和排查建议(如"Milvus 服务未启动,请执行 docker-compose up -d")

---

### User Story 6 - 知识库管理 (Priority: P3)

管理员能够通过 Web 界面批量上传文档,查看知识库统计信息(文档数量、向量数量、存储大小),删除或更新已有文档。

**Why this priority**: 知识库管理是内容维护的必要功能,但对于 MVP 阶段,可以先支持基础的文档上传,管理功能可以后续迭代。

**Independent Test**:
1. 访问管理界面(如 http://localhost:8000/admin)
2. 批量上传 10 个文档
3. 查看知识库统计信息
4. 删除某个文档,验证向量数据库中对应记录被删除

**Acceptance Scenarios**:

1. **Given** 管理员访问知识库管理页面, **When** 上传多个文档(拖拽或批量选择), **Then** 系统并行处理文档(加载→清洗→分块→向量化→存储),显示每个文档的处理进度和结果
2. **Given** 文档上传完成, **When** 查看知识库统计, **Then** 显示总文档数、总向量数、存储占用、最近更新时间,并按文档类型/来源分组展示统计信息
3. **Given** 知识库中存在某个文档, **When** 管理员点击删除按钮, **Then** 系统从 MinIO 删除原文件,从 Milvus 删除对应向量,并更新统计信息
4. **Given** 需要更新某个文档, **When** 管理员重新上传同名文件, **Then** 系统先删除旧版本的所有向量,再处理新文档并存储,确保知识库不包含过期信息

---

### Edge Cases

- **系统资源不足**: 当并发用户数过多导致 LLM API 限流或向量检索超时,系统应排队处理请求,并显示等待状态和预计时间
- **知识库为空**: 首次部署未上传文档时,用户提问应收到友好提示而非系统错误
- **会话过期**: 超过 24 小时的旧会话应自动清理(可配置),用户访问旧会话 ID 时提示已过期
- **文档上传失败**: 网络中断或文件格式异常导致上传失败时,应显示具体错误原因,允许用户重试
- **向量数据库连接断开**: 运行时 Milvus 连接丢失时,应自动重连(最多 3 次),并在无法恢复时降级为仅基于 BM25 的检索
- **对话历史过长**: 超过配置的最大对话轮数(如 20 轮)时,应强制应用摘要策略,避免上下文溢出

## Requirements *(mandatory)*

### Functional Requirements

#### 核心 Agent 能力

- **FR-001**: 系统 MUST 实现统一的 Agent 控制器,编排单轮问答、多轮对话、主动澄清等核心能力,并管理对话状态和上下文
- **FR-002**: 系统 MUST 支持多轮对话,每个会话保持对话历史(最近 N 轮完整记录 + 早期对话摘要),并在回答时理解上下文引用
- **FR-003**: 系统 MUST 实现主动澄清机制,当检测到用户问题信息不足时(关键实体缺失、问题模糊),自动生成 2-3 个澄清问题
- **FR-004**: 系统 MUST 对每个回答计算置信度评分,当置信度低于阈值(如 0.7)时,明确告知用户"信息可能不准确"或"未找到相关信息"

#### 会话与状态管理

- **FR-005**: 系统 MUST 为每个用户会话分配唯一 ID,实现会话隔离,确保多用户并发时对话历史和状态互不干扰
- **FR-006**: 系统 MUST 支持会话持久化,将对话历史存储到 Redis,用户刷新页面或短时间离开后能恢复对话
- **FR-007**: 系统 MUST 支持会话管理,用户可以创建新会话、查看历史会话列表(最近 10 个)、切换或删除会话
- **FR-008**: 系统 MUST 实现会话过期策略,超过配置时间(默认 24 小时)的非活跃会话自动清理,释放存储空间

#### 前后端集成

- **FR-009**: Web 前端 MUST 支持多轮对话 UI,显示完整对话历史(用户问题 + Agent 回答 + 来源引用),并区分普通回答和澄清问题
- **FR-010**: Web 前端 MUST 实现会话管理 UI,包括新建会话按钮、会话列表侧边栏、会话切换和删除功能
- **FR-011**: Web 前端 MUST 显示实时状态提示,包括"正在检索知识库"、"正在生成答案"、"等待 API 响应"等,提升用户体验
- **FR-012**: 系统 MUST 提供 WebSocket 或 SSE(Server-Sent Events)支持,实现流式输出答案(逐字显示),而非等待完整答案后一次性返回

#### 部署与运维

- **FR-013**: 系统 MUST 提供一键启动脚本 `scripts/start.sh`,自动启动所有依赖服务(Milvus/MinIO/Redis)和应用服务
- **FR-014**: 系统 MUST 提供健康检查脚本 `scripts/status.sh`,检查所有服务状态、端口监听、资源占用,并输出诊断报告
- **FR-015**: 系统 MUST 实现结构化日志,包括用户请求日志、检索日志、LLM 调用日志、错误日志,并支持日志级别配置
- **FR-016**: 系统 MUST 提供配置验证工具,在启动前检查 config.yaml 的有效性(必填字段、格式正确性、服务连通性)

#### 知识库管理

- **FR-017**: 系统 MUST 提供知识库管理 API,支持查询统计信息(文档数、向量数、存储占用、按类型分组统计)
- **FR-018**: 系统 MUST 支持按文档 ID 或元数据删除向量,并同步清理 MinIO 存储的原文件
- **FR-019**: Web 前端 MUST 提供知识库统计展示,包括总文档数、总向量数、文档类型分布(饼图)、最近上传记录(列表)

### Key Entities

- **Session(会话)**:
  - 属性: session_id(UUID)、user_id(可选)、created_at、last_active_at、conversation_history(对话历史数组)、metadata(会话元数据)
  - 关系: 一个会话包含多条对话记录(ConversationTurn)

- **ConversationTurn(对话轮次)**:
  - 属性: turn_id(自增 ID)、session_id(外键)、role(user/assistant/system)、content(内容)、timestamp、sources(来源引用数组,仅 assistant 角色)、is_clarification(是否为澄清问题,仅 assistant 角色)
  - 关系: 属于某个 Session

- **AgentState(Agent 状态)**:
  - 属性: session_id(外键)、current_question(当前问题)、context_summary(上下文摘要)、clarification_pending(是否等待澄清回答)、retrieval_results(最近检索结果缓存)
  - 关系: 每个 Session 对应一个 AgentState

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 用户从访问系统到得到第一个问题的答案,全流程耗时 ≤ 30 秒(P50)、≤ 60 秒(P95)
- **SC-002**: 支持至少 10 个并发用户同时进行多轮对话,每轮回答时间 ≤ 5 秒(P90)
- **SC-003**: 多轮对话准确率 ≥ 85%,即连续 3 轮对话中,Agent 能正确理解上下文引用并给出相关回答的比例
- **SC-004**: 主动澄清准确率 ≥ 80%,即当问题信息不足时,Agent 提出的澄清问题与真实缺失信息匹配的比例
- **SC-005**: 会话恢复成功率 100%,即用户刷新页面或 24 小时内重新访问,对话历史能完整恢复
- **SC-006**: 系统启动成功率 ≥ 95%,即执行 `./scripts/start.sh` 后,所有服务正常启动并通过健康检查的比例
- **SC-007**: 首次部署成功率 ≥ 90%,即在标准环境(Ubuntu 20.04 + Docker)下,执行 `./scripts/setup.sh` 后能成功部署的比例
- **SC-008**: 知识库文档上传成功率 ≥ 98%,即文档上传后成功处理并存储到向量数据库的比例(排除格式错误等用户侧问题)
- **SC-009**: 用户满意度 ≥ 4.0/5.0,通过对话结束后的评分收集(可选功能,后期实现)

## Implementation Constraints

### Technical Constraints

- **TC-001**: 系统 MUST 支持在单机环境(8 核 CPU、16GB 内存)下稳定运行,满足至少 10 并发用户
- **TC-002**: 对话历史存储 MUST 使用 Redis,利用其 TTL 机制自动清理过期会话,避免内存无限增长
- **TC-003**: Agent 控制器 MUST 基于 LangGraph 或 LangChain Agent 框架实现,确保可维护性和可扩展性
- **TC-004**: 前端 MUST 保持轻量化,单页应用总大小 ≤ 500KB(未压缩),首屏加载时间 ≤ 2 秒

### Integration Constraints

- **IC-001**: 系统 MUST 复用 Feature 001 已实现的所有组件(配置、向量存储、文档处理、混合检索等),不允许重复开发
- **IC-002**: API 接口 MUST 保持向后兼容,Feature 001 的单轮问答 API(`POST /api/question`) 必须继续可用
- **IC-003**: 配置文件(config.yaml)MUST 保持结构稳定,新增配置项需使用独立的 section(如 `agent:`、`session:`)

## Out of Scope *(明确不在本 Feature 范围内)*

以下功能虽然在 Feature 001 规划中提及,但不在本次集成范围内,将在后续 Feature 中实现:

- **多模态输入**: 图片上传、错误截图识别(需视觉模型集成)
- **用户认证与权限**: 用户登录、角色管理、访问控制(需独立的身份认证系统)
- **高级 RAG 算法**: Self-RAG、LightRAG、HippoRAG(需评估和研究)
- **评分与反馈**: 用户对答案的点赞/点踩、反馈收集(需数据分析和模型优化流程)
- **多语言支持**: 英文问答、自动语言检测(需多语言模型和语料)
- **性能监控**: Prometheus 指标、Grafana 仪表盘(需监控基础设施)

## Dependencies

### External Dependencies

- LangGraph 或 LangChain Agent 框架(用于 Agent 编排)
- Redis Python 客户端(redis-py,用于会话存储)
- FastAPI WebSocket 或 SSE 支持(用于流式输出)

### Internal Dependencies

- Feature 001 的所有已完成组件:
  - config.config_loader(配置加载)
  - core.embeddings(向量化)
  - core.vector_store(向量数据库)
  - core.document_processor(文档处理)
  - services.document_upload_service(文档上传)
  - retrieval.hybrid_retrieval(混合检索)

## Implementation Notes

### Recommended Implementation Order

1. **Phase 1: Agent 核心逻辑** (P1)
   - 实现 AgentController 类,编排单轮问答、多轮对话、主动澄清
   - 实现对话历史管理(滑动窗口 + 摘要)
   - 单元测试验证 Agent 逻辑正确性

2. **Phase 2: 会话管理** (P1)
   - 实现基于 Redis 的会话存储
   - 实现会话 CRUD API
   - 实现会话过期和清理机制

3. **Phase 3: 前后端集成** (P1)
   - 更新前端支持多轮对话 UI
   - 实现会话管理 UI(新建、切换、删除)
   - 实现实时状态提示

4. **Phase 4: 流式输出** (P2)
   - 实现 WebSocket 或 SSE 接口
   - 前端实现流式接收和逐字显示
   - 测试网络异常恢复

5. **Phase 5: 部署与运维** (P2)
   - 编写启动脚本(setup.sh、start.sh、status.sh)
   - 实现健康检查接口
   - 配置结构化日志

6. **Phase 6: 知识库管理** (P3)
   - 实现知识库统计 API
   - 实现文档删除功能
   - 前端展示知识库统计

### Key Design Decisions

1. **Agent 框架选择**: 优先使用 LangGraph,因其对状态管理和多轮对话的原生支持;如 LangGraph 学习曲线过陡,可降级为 LangChain ConversationChain
2. **会话存储**: 使用 Redis Hash 存储会话元数据,使用 Redis List 存储对话历史,利用 EXPIRE 实现自动过期
3. **流式输出**: 优先使用 SSE(Server-Sent Events),因其实现简单且单向通信满足需求;WebSocket 作为备选方案
4. **前端状态管理**: 使用简单的 JavaScript 对象管理状态,避免引入 React/Vue 等框架增加复杂度

## Testing Strategy

### Unit Tests
- Agent 逻辑:单轮问答、多轮对话、主动澄清的决策逻辑
- 会话管理:会话创建、恢复、过期、清理

### Integration Tests
- 前后端集成:完整的用户交互流程(从提问到收到答案)
- 多轮对话:连续 3-5 轮对话的上下文正确性
- 并发测试:10 个并发会话的隔离性和正确性

### System Tests
- 端到端测试:从系统启动到完成一次完整问答
- 部署测试:在全新环境执行 setup.sh 和 start.sh
- 压力测试:50 并发用户持续 5 分钟的稳定性

### Acceptance Tests
- 每个 User Story 的 Acceptance Scenarios 必须全部通过
- 所有 Success Criteria 必须达标

## Risks and Mitigations

### Risk 1: LangGraph 学习曲线导致开发延期
**Mitigation**:
- 先实现简化版 Agent(基于 LangChain ConversationChain)
- 在核心功能完成后,逐步迁移到 LangGraph

### Risk 2: Redis 内存占用过高
**Mitigation**:
- 严格配置会话 TTL(默认 24 小时)
- 对话历史超过 N 轮后强制摘要,减少存储
- 监控 Redis 内存使用,设置 maxmemory 和 eviction 策略

### Risk 3: 流式输出实现复杂度高
**Mitigation**:
- 第一版可以不实现流式输出,使用传统的请求-响应模式
- 流式输出作为增强功能在后续迭代中实现

### Risk 4: 多轮对话上下文理解不准确
**Mitigation**:
- 使用 LLM 重写问题,将上下文引用展开为完整问题
- 在 Prompt 中明确指示 LLM 利用对话历史
- 收集测试案例,持续优化 Prompt 工程

## Documentation Requirements

- **用户手册**: 包含安装、配置、使用、故障排查的完整指南
- **API 文档**: 所有 REST API 和 WebSocket 接口的 OpenAPI 规范
- **架构文档**: 系统架构图、数据流图、会话状态机
- **运维手册**: 部署步骤、监控指标、日志分析、常见问题
