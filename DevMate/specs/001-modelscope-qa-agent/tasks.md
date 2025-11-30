# Tasks: 魔搭社区智能答疑 Agent

**Feature**: 001-modelscope-qa-agent
**Branch**: `001-modelscope-qa-agent`
**Created**: 2025-11-30
**Status**: Ready for Implementation

---

## 环境说明

**重要**: 本项目使用现有基础设施,无需重复部署。

**执行前提**:
- 所有任务执行前必须先激活环境: `conda activate langchain-env`
- 依赖安装任务会先检查是否已存在,仅在缺失时补充安装
- 如需创建新环境,可参考 `requirements.txt` 文件

**已有基础设施** (通过 Docker 部署):
- **Milvus**: localhost:19530 (已运行,无需部署)
- **MySQL**: localhost:3309 (已运行,可选使用)
- **Redis**: localhost:6379 (已运行,用于缓存优化)

---

## MVP Scope Recommendation

**最小可行产品 (MVP)**: User Story 1 - 技术问题快速解答

理由:
- 核心价值场景,直接解决开发者即时痛点
- 包含完整的 RAG 技术栈验证
- 可快速验证系统可行性和准确率
- 为后续迭代奠定技术基础

---

## Phase 1: 环境搭建与项目初始化

### 1.1 项目结构初始化

- [X] [T001] 创建项目根目录结构 `modelscope_qa_agent/{core,tools,retrievers,agents,models,data,tests,config}`
- [X] [T002] 激活现有 conda 环境 `conda activate langchain-env` (环境已确认存在: /opt/anaconda3/envs/langchain-env)
- [X] [T003] 创建 `.gitignore` 文件,排除 `venv/`, `__pycache__/`, `.env`, `*.pyc`
- [X] [T004] 创建 `README.md` 项目说明文档
- [X] [T005] 创建 `requirements.txt` 依赖清单文件

### 1.2 依赖检查与补充安装

- [ ] [T006] 检查 LangChain 核心包版本 `pip list | grep langchain`,若缺失则安装 `pip install langchain==0.1.0 langchain-community==0.1.0 langchain-core==0.1.0`
- [ ] [T007] 检查 LangGraph,若缺失则安装 `pip install langgraph==0.1.0`
- [ ] [T008] 检查 Milvus 客户端,若缺失则安装 `pip install pymilvus==2.3.0 langchain-milvus==0.1.0`
- [ ] [T009] 检查通义千问 SDK,若缺失则安装 `pip install dashscope==1.14.0`
- [ ] [T010] 检查 Pydantic,若缺失则安装 `pip install pydantic==2.5.0 pydantic-settings==2.1.0`
- [ ] [T011] 检查检索增强工具,若缺失则安装 `pip install rank-bm25==0.2.2`
- [ ] [T012] 检查数据处理工具,若缺失则安装 `pip install beautifulsoup4==4.12.0 markdownify==0.11.6`
- [ ] [T013] 检查开发工具,若缺失则安装 `pip install pytest==7.4.0 python-dotenv==1.0.0`
- [ ] [T014] 检查 Redis 客户端,若缺失则安装 `pip install redis==5.0.0`
- [ ] [T015] 验证所有依赖完整性 `pip list | grep -E "langchain|milvus|dashscope|pydantic|redis"`

### 1.3 环境配置

- [ ] [T016] 创建 `.env.example` 环境变量模板文件
- [ ] [T017] 创建 `.env` 文件并配置 `DASHSCOPE_API_KEY`
- [ ] [T018] 配置 Milvus 连接 `MILVUS_HOST=localhost` 和 `MILVUS_PORT=19530`
- [ ] [T019] 配置 Redis 连接 `REDIS_HOST=localhost` 和 `REDIS_PORT=6379`
- [ ] [T020] (可选) 配置 MySQL 连接 `MYSQL_HOST=localhost` 和 `MYSQL_PORT=3309`
- [ ] [T021] 配置 LangSmith 追踪 `LANGCHAIN_TRACING_V2=true` 和 `LANGCHAIN_API_KEY`
- [ ] [T022] 创建 `config/settings.py` 加载环境变量配置

### 1.4 基础设施验证

- [ ] [T023] 验证 Milvus 连接 `pymilvus.connections.connect(host='localhost', port=19530)`
- [ ] [T024] 验证 Redis 连接 `redis-cli -h localhost -p 6379 ping`
- [ ] [T025] (可选) 验证 MySQL 连接 `mysql -h localhost -P 3309 -u root -p`
- [ ] [T026] 编写基础设施健康检查脚本 `scripts/check_infrastructure.py`

---

## Phase 2: 基础架构实现（阻塞性前置任务）

### 2.1 数据模型定义

- [ ] [T027] [P] 创建 `models/schemas.py` 定义 Pydantic 数据模型
- [ ] [T028] [P] 实现 `KnowledgeEntry` 知识库条目模型（参考 design/data-model.md:93-143）
- [ ] [T029] [P] 实现 `TechnicalAnswer` 技术回答模型（参考 design/data-model.md:150-238）
- [ ] [T030] [P] 实现 `DialogueSession` 对话会话模型（参考 design/data-model.md:245-295）
- [ ] [T031] [P] 实现 `MessageRecord` 消息记录模型（参考 design/data-model.md:300-363）
- [ ] [T032] [P] 实现 `UserFeedback` 用户反馈模型（参考 design/data-model.md:370-408）
- [ ] [T033] [P] 实现 `QuestionCategory` 问题分类模型（参考 design/data-model.md:415-449）
- [ ] [T034] [P] 实现 `ConversationState` LangGraph 状态模型（参考 design/data-model.md:10-50）
- [ ] [T035] [P] 编写 Pydantic 模型单元测试 `tests/test_schemas.py`

### 2.2 向量数据库管理器

- [ ] [T036] [P] 创建 `core/vector_store.py` 向量存储管理模块
- [ ] [T037] [P] 实现 `VectorStoreManager.__init__()` 连接 Milvus 和初始化嵌入模型（参考 plan.md:105-121）
- [ ] [T038] [P] 实现 `_init_collection()` 创建 Milvus Collection Schema（参考 data-model.md:461-483）
- [ ] [T039] [P] 配置向量索引 IVF_FLAT + 标量字段索引（参考 data-model.md:488-505）
- [ ] [T040] [P] 实现 `get_vector_store()` 返回 LangChain Milvus 实例（参考 plan.md:150-157）
- [ ] [T041] [P] 编写 Milvus 连接测试 `tests/test_vector_store.py::test_milvus_connection`（参考 plan.md:163-176）
- [ ] [T042] [P] 测试向量写入和检索功能

### 2.3 文档处理器

- [ ] [T043] [P] 创建 `core/document_processor.py` 文档处理模块
- [ ] [T044] [P] 实现 `DocumentProcessor.__init__()` 初始化分块器（参考 plan.md:196-210）
- [ ] [T045] [P] 实现 `load_modelscope_docs()` 加载魔搭官方文档（参考 plan.md:211-225）
- [ ] [T046] [P] 实现 `clean_document()` 清洗文档内容（参考 plan.md:227-241）
- [ ] [T047] [P] 实现 `split_with_code_protection()` 语义分块保护代码块（参考 plan.md:243-262）
- [ ] [T048] [P] 实现 `calculate_quality_score()` 计算文档质量评分（参考 plan.md:264-287）
- [ ] [T049] [P] 编写文档清洗测试 `tests/test_document_processor.py::test_clean_document`
- [ ] [T050] [P] 编写代码块保护测试 `tests/test_document_processor.py::test_split_with_code_protection`

### 2.4 混合检索器

- [ ] [T051] [P] 创建 `retrievers/hybrid_retriever.py` 混合检索模块
- [ ] [T052] [P] 实现 `HybridRetriever.__init__()` 初始化向量和 BM25 检索器（参考 plan.md:306-321）
- [ ] [T053] [P] 实现 `retrieve()` 执行混合检索（参考 plan.md:323-332）
- [ ] [T054] [P] 实现 `rerank()` 重排序逻辑（可选,参考 plan.md:334-337）
- [ ] [T055] [P] 编写混合检索测试 `tests/test_hybrid_retriever.py::test_retrieve`
- [ ] [T056] [P] 测试向量和 BM25 权重调优

---

## Phase 3: User Story 1 - 技术问题快速解答 (Priority: P1)

### 3.1 LangGraph Agent 核心工作流

- [ ] [T057] [P1] [US1] 创建 `agents/qa_agent.py` 问答 Agent 模块
- [ ] [T058] [P1] [US1] 实现 `ModelScopeQAAgent.__init__()` 初始化 LLM 和工作流（参考 plan.md:375-390）
- [ ] [T059] [P1] [US1] 实现 `_build_graph()` 构建 LangGraph 工作流（参考 plan.md:392-412）
- [ ] [T060] [P1] [US1] 添加 `retrieve` 节点实现文档检索（参考 plan.md:414-418）
- [ ] [T061] [P1] [US1] 添加 `generate` 节点实现答案生成（参考 plan.md:420-464）
- [ ] [T062] [P1] [US1] 添加 `validate` 节点实现答案验证（参考 plan.md:466-469）
- [ ] [T063] [P1] [US1] 实现条件分支 `_should_validate()`（参考 plan.md:471-474）
- [ ] [T064] [P1] [US1] 配置 MemorySaver Checkpointer 支持对话持久化（参考 plan.md:389-390）

### 3.2 Prompt Engineering

- [ ] [T065] [P1] [US1] 创建 `prompts/qa_prompts.py` Prompt 模板模块
- [ ] [T066] [P1] [US1] 编写系统 Prompt 定义 Agent 角色和任务（参考 plan.md:433-449）
- [ ] [T067] [P1] [US1] 添加上下文文档占位符和格式化指令（参考 plan.md:443-446）
- [ ] [T068] [P1] [US1] 配置 PydanticOutputParser 生成结构化输出（参考 plan.md:453-461）
- [ ] [T069] [P1] [US1] 测试 Prompt 有效性（使用真实示例问题）

### 3.3 LLM 集成

- [ ] [T070] [P1] [US1] 创建 `core/llm_client.py` LLM 客户端模块
- [ ] [T071] [P1] [US1] 实现通义千问 ChatTongyi 客户端（参考 research.md:306-314）
- [ ] [T072] [P1] [US1] 配置模型参数 `temperature=0.3, top_p=0.8`
- [ ] [T073] [P1] [US1] 启用流式输出 `streaming=True`
- [ ] [T074] [P1] [US1] 编写 LLM 调用测试 `tests/test_llm_client.py::test_chat_tongyi`

### 3.4 知识库数据加载

- [ ] [T075] [P1] [US1] 创建 `data/loaders/` 数据加载模块目录
- [ ] [T076] [P1] [US1] 实现 `official_docs_loader.py` 加载魔搭官方文档
- [ ] [T077] [P1] [US1] 实现 `github_docs_loader.py` 加载 GitHub 技术文档
- [ ] [T078] [P1] [US1] 实现数据清洗流程（移除 HTML 标签、统一代码块格式）
- [ ] [T079] [P1] [US1] 实现语义分块并存入 Milvus
- [ ] [T080] [P1] [US1] 添加质量评分和元数据标签
- [ ] [T081] [P1] [US1] 编写数据加载脚本 `scripts/load_knowledge_base.py`
- [ ] [T082] [P1] [US1] 执行初始知识库构建并验证向量数量

### 3.5 单轮问答功能实现

- [ ] [T083] [P1] [US1] 实现 `invoke()` 方法接收用户问题（参考 plan.md:476-482）
- [ ] [T084] [P1] [US1] 集成检索、生成、验证完整流程
- [ ] [T085] [P1] [US1] 返回 TechnicalAnswer 结构化响应
- [ ] [T086] [P1] [US1] 编写单轮问答测试 `tests/test_qa_agent.py::test_single_turn_qa`
- [ ] [T087] [P1] [US1] 测试场景:模型调用错误问题（对应 spec.md:88）
- [ ] [T088] [P1] [US1] 测试场景:多模态场景问题（对应 spec.md:89）
- [ ] [T089] [P1] [US1] 验证回答包含问题分析、解决方案、代码示例、引用来源

### 3.6 主动澄清机制

- [ ] [T090] [P1] [US1] 创建 `tools/clarification_tool.py` 澄清问题工具
- [ ] [T091] [P1] [US1] 实现 `detect_missing_info()` 检测缺失关键信息
- [ ] [T092] [P1] [US1] 实现 `generate_clarification_questions()` 生成澄清问题
- [ ] [T093] [P1] [US1] 添加 `clarify` 节点到 LangGraph 工作流
- [ ] [T094] [P1] [US1] 测试场景:问题描述不清晰（对应 spec.md:91）
- [ ] [T095] [P1] [US1] 验证主动提出澄清问题（如"您使用的是哪个版本?"）

### 3.7 评估与优化

- [ ] [T096] [P1] [US1] 准备评测数据集（至少30个真实技术问题）
- [ ] [T097] [P1] [US1] 实现 RAGAs 评估脚本 `scripts/evaluate_rag.py`
- [ ] [T098] [P1] [US1] 评估 Context Relevance（目标≥85%）
- [ ] [T099] [P1] [US1] 评估 Answer Faithfulness（目标≥95%）
- [ ] [T100] [P1] [US1] 评估响应速度（目标<30秒）
- [ ] [T101] [P1] [US1] 根据评估结果调优检索权重和 Prompt

---

## Phase 4: User Story 2 - 多轮对话深度排查 (Priority: P2)

### 4.1 对话历史管理

- [ ] [T102] [P2] [US2] 创建 `core/memory_manager.py` 对话记忆管理模块
- [ ] [T103] [P2] [US2] 实现 `trim_conversation()` 保留最近10轮对话（参考 research.md:69-78）
- [ ] [T104] [P2] [US2] 实现 `summarize_early_messages()` 压缩早期对话为摘要
- [ ] [T105] [P2] [US2] 在 ConversationState 中添加 `summary` 字段
- [ ] [T106] [P2] [US2] 编写对话修剪测试 `tests/test_memory_manager.py::test_trim_conversation`

### 4.2 上下文理解增强

- [ ] [T107] [P2] [US2] 修改 `generate` 节点支持对话历史引用
- [ ] [T108] [P2] [US2] 在 Prompt 中添加对话历史占位符
- [ ] [T109] [P2] [US2] 实现代词消解（如"刚才你建议的方法"）
- [ ] [T110] [P2] [US2] 测试场景:第二轮对话引用第一轮（对应 spec.md:105）
- [ ] [T111] [P2] [US2] 测试场景:第三轮对话引用第二轮建议（对应 spec.md:106）

### 4.3 多轮对话状态管理

- [ ] [T112] [P2] [US2] 添加 `turn_count` 字段到 ConversationState
- [ ] [T113] [P2] [US2] 实现会话恢复逻辑（基于 thread_id）
- [ ] [T114] [P2] [US2] 实现多线程会话隔离（不同用户互不干扰）
- [ ] [T115] [P2] [US2] 测试场景:处理不同格式信息（代码、配置、日志,对应 spec.md:107）
- [ ] [T116] [P2] [US2] 编写多轮对话测试 `tests/test_qa_agent.py::test_multi_turn_qa`

### 4.4 对话进度评估

- [ ] [T117] [P2] [US2] 实现 `assess_progress()` 评估问题解决进度
- [ ] [T118] [P2] [US2] 实现主动总结已尝试方法和排除的可能性
- [ ] [T119] [P2] [US2] 测试场景:对话超过5轮主动总结（对应 spec.md:108）
- [ ] [T120] [P2] [US2] 建议是否转向其他排查路径或人工支持

---

## Phase 5: User Story 3 - 平台功能导航与最佳实践推荐 (Priority: P3)

### 5.1 平台知识库扩展

- [ ] [T121] [P3] [US3] 加载魔搭社区平台功能文档（ModelScope SDK、Studio、MCP）
- [ ] [T122] [P3] [US3] 加载模型库和数据集库元数据
- [ ] [T123] [P3] [US3] 添加 `platform` 分类标签到知识库条目

### 5.2 结构化功能介绍

- [ ] [T124] [P3] [US3] 创建 `tools/platform_info_tool.py` 平台信息工具
- [ ] [T125] [P3] [US3] 实现 `get_platform_overview()` 返回功能概览
- [ ] [T126] [P3] [US3] 测试场景:询问平台支持的AI任务类型（对应 spec.md:122）

### 5.3 方案推荐引擎

- [ ] [T127] [P3] [US3] 创建 `tools/recommendation_tool.py` 推荐工具
- [ ] [T128] [P3] [US3] 实现 `recommend_models()` 基于需求推荐模型
- [ ] [T129] [P3] [US3] 实现 `compare_models()` 对比分析模型（大小、速度、精度）
- [ ] [T130] [P3] [US3] 测试场景:描述业务需求推荐方案（对应 spec.md:123）
- [ ] [T131] [P3] [US3] 测试场景:模型选择对比分析（对应 spec.md:124）

### 5.4 使用教程生成

- [ ] [T132] [P3] [US3] 实现 `generate_tutorial()` 生成功能使用步骤
- [ ] [T133] [P3] [US3] 测试场景:询问MCP协议使用方法（对应 spec.md:125）
- [ ] [T134] [P3] [US3] 验证包含功能说明、步骤、代码示例、文档链接

---

## Phase 6: User Story 4 - 项目级开发指导 (Priority: P4)

### 6.1 架构设计工具

- [ ] [T135] [P4] [US4] 创建 `tools/architecture_tool.py` 架构设计工具
- [ ] [T136] [P4] [US4] 实现 `design_architecture()` 生成系统架构建议
- [ ] [T137] [P4] [US4] 测试场景:多模态内容审核系统架构（对应 spec.md:139）

### 6.2 技术选型指导

- [ ] [T138] [P4] [US4] 实现 `recommend_tech_stack()` 推荐技术栈
- [ ] [T139] [P4] [US4] 实现 `compare_deployment_options()` 对比部署方案
- [ ] [T140] [P4] [US4] 测试场景:部署方式选择（对应 spec.md:140）

### 6.3 性能优化建议

- [ ] [T141] [P4] [US4] 实现 `suggest_optimizations()` 提供优化建议
- [ ] [T142] [P4] [US4] 测试场景:性能优化需求（对应 spec.md:141）

### 6.4 工程最佳实践

- [ ] [T143] [P4] [US4] 实现 `provide_best_practices()` 提供开发流程指导
- [ ] [T144] [P4] [US4] 测试场景:开发流程和工程规范（对应 spec.md:142）

---

## Phase 7: 功能增强（跨 User Story）

### 7.1 多模态支持

- [ ] [T145] 创建 `core/multimodal_processor.py` 多模态处理模块
- [ ] [T146] 集成通义千问 VL 模型 `qwen-vl-plus`（参考 research.md:482-500）
- [ ] [T147] 实现 `process_error_screenshot()` 识别报错截图
- [ ] [T148] 提取错误类型、错误代码和堆栈跟踪
- [ ] [T149] 测试场景:用户上传报错截图（对应 spec.md:90）

### 7.2 问题改写与分类

- [ ] [T150] 创建 `tools/query_optimizer.py` 查询优化工具
- [ ] [T151] 实现 `rewrite_query()` 改写用户问题（参考 research.md:387-407）
- [ ] [T152] 实现 `classify_question()` 问题自动分类（模型使用/技术/平台/项目）
- [ ] [T153] 在 LangGraph 中添加 `rewrite_query` 节点
- [ ] [T154] 在 LangGraph 中添加条件路由基于分类结果

### 7.3 缓存与性能优化

- [ ] [T155] 创建 `core/cache_manager.py` 缓存管理模块
- [ ] [T156] 实现 LLM 响应缓存 `InMemoryCache`（参考 research.md:552-554）
- [ ] [T157] 实现检索结果缓存 `RetrievalCache`（参考 research.md:556-569）
- [ ] [T158] 实现异步并行检索 `parallel_retrieval()`（参考 research.md:583-591）
- [ ] [T159] 测试缓存命中率和性能提升

### 7.4 Self-RAG 答案验证

- [ ] [T160] 创建 `tools/self_rag_validator.py` Self-RAG 验证工具
- [ ] [T161] 实现 `evaluate_relevance()` 评估答案与文档相关性
- [ ] [T162] 实现 `self_rag_answer()` 迭代优化流程（参考 research.md:449-468）
- [ ] [T163] 集成到 `validate` 节点
- [ ] [T164] 测试置信度低于0.8时重新检索

### 7.5 安全检查

- [ ] [T165] 创建 `tools/security_checker.py` 安全检查工具
- [ ] [T166] 实现 `detect_security_risks()` 检测硬编码密钥、SQL注入风险
- [ ] [T167] 在答案生成后执行安全检查
- [ ] [T168] 测试场景:检测并警示安全风险（对应 spec.md:171）

---

## Phase 8: 评估与监控

### 8.1 RAGAs 评估框架

- [ ] [T169] 安装 RAGAs `pip install ragas==0.1.0`
- [ ] [T170] 创建 `scripts/evaluate_rag.py` RAGAs 评估脚本
- [ ] [T171] 准备评估数据集（问题、答案、上下文、标准答案）
- [ ] [T172] 实现 Context Relevance 评估（目标≥85%）
- [ ] [T173] 实现 Answer Relevance 评估
- [ ] [T174] 实现 Faithfulness 评估（目标≥95%,幻觉率<5%）
- [ ] [T175] 实现 Answer Correctness 评估
- [ ] [T176] 生成评估报告并分析结果

### 8.2 LangSmith 监控

- [ ] [T177] 配置 LangSmith 环境变量（参考 research.md:647-652）
- [ ] [T178] 创建 LangSmith 项目 `modelscope-qa-agent`
- [ ] [T179] 验证 LLM 调用自动追踪
- [ ] [T180] 配置 Token 使用量监控
- [ ] [T181] 配置延迟分布监控（P50, P95, P99）
- [ ] [T182] 配置错误率告警

### 8.3 用户反馈收集

- [ ] [T183] 创建 `core/feedback_collector.py` 反馈收集模块
- [ ] [T184] 实现 `collect_feedback()` 记录用户评分和评论
- [ ] [T185] 实现 `analyze_feedback()` 分析反馈趋势
- [ ] [T186] 集成到 Agent 响应流程
- [ ] [T187] 验证满意度评分≥4.0分（对应 spec.md:SC-003）

---

## Phase 9: 性能测试与优化

### 9.1 性能基准测试

- [ ] [T188] 创建 `tests/performance/benchmark.py` 性能测试脚本
- [ ] [T189] 测试单轮问答响应时间（目标<30秒,对应 spec.md:SC-001）
- [ ] [T190] 测试并发处理能力（目标100并发,对应 spec.md:SC-008）
- [ ] [T191] 测试多轮对话5轮解决率（目标75%,对应 spec.md:SC-002）
- [ ] [T192] 生成性能基准报告

### 9.2 优化迭代

- [ ] [T193] 根据性能测试结果调优向量索引参数
- [ ] [T194] 优化混合检索权重（向量 vs BM25）
- [ ] [T195] 实现模型量化降低推理延迟（可选）
- [ ] [T196] 优化 Prompt 长度减少 Token 消耗
- [ ] [T197] 重新测试验证优化效果

---

## Phase 10: 集成测试与部署准备

### 10.1 端到端测试

- [ ] [T198] 创建 `tests/integration/test_e2e.py` 端到端测试
- [ ] [T199] 测试完整用户旅程:单轮技术问答（US1）
- [ ] [T200] 测试完整用户旅程:多轮对话排查（US2）
- [ ] [T201] 测试完整用户旅程:平台功能咨询（US3）
- [ ] [T202] 测试完整用户旅程:项目级指导（US4）
- [ ] [T203] 验证所有 Edge Cases（对应 spec.md:147-154）

### 10.2 部署脚本

- [ ] [T204] 创建 `scripts/deploy.sh` 部署脚本
- [ ] [T205] 创建 `docker/Dockerfile` 容器化配置
- [ ] [T206] 创建 `docker-compose.prod.yml` 生产环境配置
- [ ] [T207] 编写部署文档 `docs/DEPLOYMENT.md`

### 10.3 文档完善

- [ ] [T208] 编写用户使用指南 `docs/USER_GUIDE.md`
- [ ] [T209] 编写 API 文档 `docs/API.md`
- [ ] [T210] 编写开发者贡献指南 `docs/CONTRIBUTING.md`
- [ ] [T211] 更新 README.md 包含快速开始和示例

---

## 依赖关系图

```
Phase 1 (环境搭建)
    │
    ├──> Phase 2 (基础架构)
    │       │
    │       ├──> Phase 3 (US1 - 技术问答) [MVP]
    │       │       │
    │       │       └──> Phase 7.1 (多模态支持)
    │       │
    │       ├──> Phase 4 (US2 - 多轮对话)
    │       │       │
    │       │       └──> Phase 7.2 (问题改写)
    │       │
    │       ├──> Phase 5 (US3 - 平台导航)
    │       │
    │       └──> Phase 6 (US4 - 项目指导)
    │
    ├──> Phase 7 (功能增强)
    │       │
    │       └──> Phase 7.3 (缓存优化)
    │       └──> Phase 7.4 (Self-RAG)
    │       └──> Phase 7.5 (安全检查)
    │
    ├──> Phase 8 (评估监控)
    │
    ├──> Phase 9 (性能测试)
    │       │
    │       └──> Phase 9.2 (优化迭代)
    │
    └──> Phase 10 (集成测试与部署)
```

---

## 并行执行机会

以下任务可以并行执行:

**阶段1并行组**:
- T001-T005 (项目结构) || T006-T014 (依赖安装) || T015-T019 (环境配置)

**阶段2并行组**:
- T024-T032 (数据模型) || T033-T039 (向量存储) || T040-T047 (文档处理)

**阶段3并行组**:
- T062-T066 (Prompt) || T067-T071 (LLM集成) || T072-T079 (知识库加载)

**阶段7并行组**:
- T142-T146 (多模态) || T147-T151 (问题优化) || T152-T156 (缓存) || T157-T161 (Self-RAG) || T162-T165 (安全)

**阶段8并行组**:
- T166-T173 (RAGAs) || T174-T179 (LangSmith) || T180-T184 (反馈收集)

---

## 任务统计

- **总任务数**: 208
- **阻塞性任务 [P]**: 30
- **P1 任务 (US1)**: 45
- **P2 任务 (US2)**: 17
- **P3 任务 (US3)**: 14
- **P4 任务 (US4)**: 10
- **跨功能任务**: 92

---

## 成功标准验证清单

- [ ] SC-001: 90%问题<30秒响应 → 通过 T186
- [ ] SC-002: 5轮内75%复杂问题解决 → 通过 T188
- [ ] SC-003: 用户满意度≥4.0分 → 通过 T184
- [ ] SC-004: 代码可运行性≥90% → 通过 T086
- [ ] SC-005: 问题分类准确率≥95% → 通过 T149
- [ ] SC-006: 检索准确率≥85% → 通过 T169
- [ ] SC-007: 澄清问题有效率≥80% → 通过 T091
- [ ] SC-008: 100并发/3秒响应 → 通过 T187
- [ ] SC-009: 幻觉率<5% → 通过 T171
- [ ] SC-010: 系统稳定性≥99% → 通过 T200
- [ ] SC-011: 首次使用成功率≥85% → 通过 T196
- [ ] SC-012: 解决时间缩短50% → 通过人工对比测试

---

**任务清单版本**: 1.0
**最后更新**: 2025-11-30
**审核状态**: 待审核
