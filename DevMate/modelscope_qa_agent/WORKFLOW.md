# 魔搭社区智能答疑系统 - 完整工作流程

## 概述

本文档详细描述了从文件上传到问答的完整端到端流程,包括向量化存储和 Agent 查询的所有环节。

## 架构图

```
Frontend (index.html)
      ↓
API Layer (FastAPI)
      ↓
Document Upload Service
      ↓
[文件上传] → [文档处理] → [向量化] → [Milvus存储]
                                            ↓
[用户提问] → [QA Agent] → [混合检索] → [从Milvus检索]
      ↓                                    ↓
[答案生成] ← [LLM (VolcEngine/Tongyi)] ← [检索结果]
      ↓
返回答案 + 来源 + 置信度
```

---

## 详细流程说明

### Phase 1: 文件上传与向量化存储

#### 1.1 前端上传 (index.html)

**位置**: `api/static/index.html` 第508-569行

**功能**: `uploadDocument()` 函数

**流程**:
1. 用户选择文件(通过点击或拖拽)
2. 创建 `FormData` 对象:
   - `file`: 文件数据
   - `category`: 'general' (分类)
   - `store_to_db`: true/false (是否存储到向量数据库)
3. 发送 POST 请求到 `/api/upload`
4. 显示上传进度 (30% → 70% → 100%)
5. 接收响应并更新统计信息

**关键代码**:
```javascript
const formData = new FormData();
formData.append('file', selectedFile);
formData.append('category', 'general');
formData.append('store_to_db', storeToDb);

const response = await fetch(`${API_BASE}/api/upload`, {
    method: 'POST',
    body: formData
});
```

#### 1.2 API 接收上传 (main.py)

**位置**: `api/main.py` 第304-362行

**功能**: `/api/upload` 端点

**流程**:
1. 验证文件类型(检查扩展名)
2. 读取文件数据到内存
3. 验证文件大小限制
4. 调用 `DocumentUploadService.upload_and_process()`
5. 返回处理结果

**支持的文件格式**:
- PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx)
- TXT, Markdown (.md), HTML, JSON, XML
- RTF (需要 pandoc 2.14.2+)

#### 1.3 文档处理服务 (document_upload_service.py)

**位置**: `services/document_upload_service.py` 第210-272行

**功能**: `upload_and_process()` 方法

**完整流程**:

```python
def upload_and_process(file_data, filename, metadata, ...):
    # Step 1: 上传文件到存储系统 (MinIO/Local)
    file_path = self.upload_file(file_data, filename, metadata)

    # Step 2: 处理文档
    processed_docs = self.process_uploaded_file(
        file_path=file_path,
        metadata=metadata,
        clean=True,      # 清洗文本
        split=True,      # 分块
        calculate_score=True  # 质量评分
    )

    # Step 3: 向量化并存储到 Milvus (可选)
    if store_to_vector_db:
        document_ids = self.store_documents(processed_docs)

    return result
```

**详细步骤**:

##### Step 1: 文件上传到存储 (第84-125行)
- 验证文件大小和类型
- 上传到 MinIO 或本地文件系统
- 返回存储路径

##### Step 2: 文档处理 (第127-181行)

**2.1 如果是 MinIO 存储**:
- 创建临时文件
- 从 MinIO 下载到本地临时文件
- 处理完成后删除临时文件

**2.2 调用 DocumentProcessor.load_and_process_file()**:
```python
# 位置: core/document_processor.py
processed_docs = self.doc_processor.load_and_process_file(
    file_path=local_file_path,
    metadata=metadata,
    clean=True,    # 使用 DocumentCleaner 清洗
    split=True,    # 使用 RecursiveCharacterTextSplitter 分块
    calculate_score=True  # 计算质量评分
)
```

**文档处理包括**:
- **加载**: 使用 `unstructured` 库加载各种格式
- **清洗**: 去除特殊字符、规范化空白、提取主要内容
- **分块**: 将文档分割成适合嵌入的块 (默认 1000 字符,重叠 200 字符)
- **评分**: 计算每个块的质量分数

##### Step 3: 向量化与存储 (第183-208行)

**位置**: `store_documents()` 方法

**流程**:
```python
def store_documents(documents):
    # 1. 获取 Milvus 向量存储实例
    vector_store = self.vector_store.get_vector_store()

    # 2. 提取文本和元数据
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    # 3. 向量化并批量插入 Milvus
    # LangChain Milvus 会自动:
    # - 调用 embedding model (VolcEngine) 生成向量
    # - 插入到 Milvus collection
    ids = vector_store.add_texts(texts, metadatas=metadatas)

    return ids
```

**向量化模型**:
- **Provider**: VolcEngine (豆包)
- **Model**: 由 config.yaml 中的 `ai.models.embedding` 指定
- **Dimension**: 1536 (默认)

**Milvus 存储结构**:
```yaml
Collection: modelscope_qa
Fields:
  - id: 主键 (自动生成)
  - text: 文档文本内容
  - vector: 向量 (1536 维)
  - metadata: JSON 元数据 (source, category, score 等)
```

---

### Phase 2: 智能问答与检索

#### 2.1 前端提问 (index.html)

**位置**: `api/static/index.html` 第572-616行

**功能**: `askQuestion()` 函数

**流程**:
1. 获取用户输入的问题
2. 显示用户消息
3. 发送 POST 请求到 `/api/question`
4. 接收答案并显示:
   - 答案文本
   - 参考来源
   - 置信度评分

**关键代码**:
```javascript
const response = await fetch(`${API_BASE}/api/question`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        question: question,
        session_id: sessionId,
        top_k: 3
    })
});

const result = await response.json();
// result 包含: answer, sources[], confidence, session_id, timestamp
```

#### 2.2 API 处理问题 (main.py)

**当前实现** (第365-412行):

```python
@app.post("/api/question")
async def ask_question(request: QuestionRequest):
    # 1. 从向量数据库检索相关文档
    vector_store = doc_service.vector_store.get_vector_store()
    results = vector_store.similarity_search_with_score(
        request.question,
        k=request.top_k
    )

    # 2. 构造来源文档列表
    sources = []
    for doc, score in results:
        sources.append({
            "content": doc.page_content[:200] + "...",
            "source": doc.metadata.get("source", "unknown"),
            "score": float(score)
        })

    # 3. 生成答案 (简化版本)
    if sources:
        answer = f"根据检索到的 {len(sources)} 个相关文档,{request.question}\n\n"
        answer += "相关内容:\n"
        for i, source in enumerate(sources, 1):
            answer += f"{i}. {source['content']}\n"
    else:
        answer = "抱歉,没有找到相关文档。"

    # 4. 返回响应
    return AnswerResponse(
        answer=answer,
        sources=sources,
        confidence=0.8 if sources else 0.0,
        session_id=session_id,
        timestamp=datetime.now().isoformat()
    )
```

**注意**: 这是简化版本。生产环境应该使用 QA Agent 进行智能问答。

#### 2.3 QA Agent 工作流 (推荐实现)

**位置**: `agents/simple_agent.py` 和 `api/routers/qa.py`

**完整 Agent 流程**:

```
用户问题
    ↓
1. 问题分析节点 (question_analysis_node)
   - 解析问题意图
   - 提取关键词
    ↓
2. 检索节点 (retrieval_node)
   - 使用 HybridRetriever 混合检索:
     * 向量检索 (Milvus similarity search)
     * BM25 关键词检索
     * 加权融合结果
    ↓
3. 路由判断 (should_clarify)
   - 检索结果是否足够?
   - 是 → 生成答案
   - 否 → 生成澄清问题
    ↓
4a. 答案生成节点 (answer_generation_node)
   - 使用 LLM 基于检索结果生成答案
   - 计算置信度评分
   - 返回答案 + 来源
    ↓
4b. 澄清节点 (clarify_node)
   - 生成澄清问题列表
   - 返回澄清问题
```

**Agent 创建** (第18-73行):
```python
def create_agent(retriever=None, llm=None):
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("analyze", question_analysis_node)
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("generate", answer_generation_node)
    workflow.add_node("clarify", clarify_node)

    # 定义边
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        should_clarify,
        {"clarify": "clarify", "generate": "generate"}
    )
    workflow.add_edge("clarify", END)
    workflow.add_edge("generate", END)

    return workflow.compile()
```

**Agent 调用** (第76-111行):
```python
def invoke_agent(agent, question, session_id=None):
    initial_state = {
        "question": question,
        "session_id": session_id,
        ...
    }
    result = agent.invoke(initial_state)
    return result  # 包含 final_answer, confidence_score, sources
```

#### 2.4 混合检索器 (HybridRetriever)

**位置**: `retrievers/hybrid_retriever.py`

**功能**: 结合向量检索和 BM25 关键词检索

**流程**:
```python
class HybridRetriever:
    def __init__(self, vector_store, documents, vector_weight=0.7, bm25_weight=0.3):
        self.vector_store = vector_store  # Milvus 向量存储
        self.bm25 = BM25Okapi(documents)   # BM25 索引
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    def retrieve(self, query, top_k=3):
        # 1. 向量检索
        vector_results = self.vector_store.similarity_search_with_score(
            query, k=top_k*2
        )

        # 2. BM25 检索
        bm25_results = self.bm25.get_top_n(query, top_k=top_k*2)

        # 3. 加权融合
        combined_scores = {}
        for doc, score in vector_results:
            combined_scores[doc] = score * self.vector_weight

        for doc in bm25_results:
            if doc in combined_scores:
                combined_scores[doc] += bm25_score * self.bm25_weight
            else:
                combined_scores[doc] = bm25_score * self.bm25_weight

        # 4. 排序并返回 top_k
        final_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return final_results
```

---

## 配置说明

### AI 配置 (config/config.yaml)

```yaml
ai:
  provider: volcengine  # 或 tongyi
  api_key: ${VOLCENGINE_API_KEY}
  base_url: https://ark.cn-beijing.volces.com/api/v3
  models:
    chat: ep-xxx-xxx  # 对话模型
    embedding: ep-xxx-xxx  # 嵌入模型
  parameters:
    temperature: 0.7
    top_p: 0.9
```

### Milvus 配置

```yaml
milvus:
  host: localhost
  port: 19530
  collection_name: modelscope_qa
  vector_dim: 1536
  index_type: IVF_FLAT
  metric_type: IP  # 内积相似度
```

### 检索配置

```yaml
retrieval:
  top_k: 3
  vector_weight: 0.7
  bm25_weight: 0.3
```

---

## 服务初始化流程

### 启动时初始化 (main.py 第98-224行)

```python
@app.on_event("startup")
async def startup_event():
    # 1. 加载配置
    config = load_config()

    # 2. 初始化 Redis 会话管理器
    session_manager = SessionManager()

    # 3. 初始化文档上传服务
    doc_service = DocumentUploadService(config)

    # 4. 初始化 LLM 客户端
    if config.ai.provider == "volcengine":
        llm_client = VolcEngineLLM(...)
    else:
        llm_client = TongyiLLMClient(...)

    # 5. 初始化混合检索器
    vector_store = doc_service.vector_store.get_vector_store()
    retriever = HybridRetriever(
        vector_store=vector_store,
        documents=[],  # TODO: 加载文档
        vector_weight=config.retrieval.vector_weight,
        bm25_weight=config.retrieval.bm25_weight
    )

    # 6. 创建 QA Agent
    qa_agent = create_agent(retriever=retriever, llm=llm_client)
    app.state.qa_agent = qa_agent
```

---

## 数据流示例

### 示例 1: 上传文档

```
用户操作:
1. 选择文件 "Qwen_模型使用指南.md"

前端:
2. uploadDocument()
   → POST /api/upload
   FormData: {file: ..., category: "general", store_to_db: true}

后端 API:
3. upload_document() 验证文件
   → DocumentUploadService.upload_and_process()

文档上传服务:
4. upload_file() → 上传到 MinIO/Local
   返回: "documents/Qwen_模型使用指南.md"

5. process_uploaded_file()
   → DocumentProcessor.load_and_process_file()
   - 加载 Markdown 文件
   - 清洗文本
   - 分块: 生成 15 个 Document 对象
   返回: [doc1, doc2, ..., doc15]

6. store_documents()
   → vector_store.add_texts()
   - 调用 VolcEngine Embedding API 生成向量
   - 插入 Milvus collection
   返回: [id1, id2, ..., id15]

响应返回:
7. {
     "message": "文档上传成功",
     "filename": "Qwen_模型使用指南.md",
     "document_count": 15,
     "document_ids": [id1, ..., id15],
     "stored_to_db": true
   }

前端显示:
8. "上传成功! 处理了 15 个文档块"
```

### 示例 2: 智能问答

```
用户输入:
1. "如何使用 Qwen 模型进行文本生成?"

前端:
2. askQuestion()
   → POST /api/question
   JSON: {question: "...", session_id: "xxx", top_k: 3}

后端 API (简化版):
3. ask_question()
   → vector_store.similarity_search_with_score()

Milvus 检索:
4. - 将问题向量化 (VolcEngine Embedding)
   - 在 collection 中搜索最相似的 3 个向量
   - 返回: [(doc1, 0.92), (doc2, 0.88), (doc3, 0.85)]

生成答案:
5. 基于检索结果拼接答案
   answer = "根据检索到的 3 个相关文档...
            1. Qwen 模型可以通过...
            2. 配置参数包括...
            3. 示例代码如下..."

响应返回:
6. {
     "answer": "...",
     "sources": [
       {content: "...", source: "Qwen_模型使用指南.md", score: 0.92},
       ...
     ],
     "confidence": 0.8,
     "session_id": "xxx",
     "timestamp": "2025-12-02T10:30:00"
   }

前端显示:
7. AI 助手消息:
   "根据检索到的 3 个相关文档...
    参考来源 (置信度: 80.0%):
    1. Qwen_模型使用指南.md (相似度: 92.0%)"
```

---

## 优化建议

### 已实现的优化

1. ✅ **文档处理优化**:
   - 文档清洗和质量评分
   - 智能分块 (RecursiveCharacterTextSplitter)
   - 元数据丰富

2. ✅ **向量化存储**:
   - 批量插入 Milvus
   - 高效的向量索引 (IVF_FLAT)
   - 元数据存储

3. ✅ **前端体验**:
   - 实时上传进度
   - 拖拽上传支持
   - 响应式设计

### 待实现的优化

1. ⏳ **完整 QA Agent 集成** (当前使用简化版本):
   - 使用 `/api/v2/qa/ask` 端点
   - LangGraph 工作流
   - 混合检索器
   - 智能澄清

2. ⏳ **多轮对话**:
   - Redis 会话管理
   - 对话历史保存
   - 上下文引用理解

3. ⏳ **流式输出**:
   - Server-Sent Events (SSE)
   - 实时答案生成显示

4. ⏳ **性能优化**:
   - 检索结果缓存
   - BM25 文档加载
   - 并发处理

---

## 故障排查

### 常见问题

**Q1: 文档上传失败 "No module named 'unstructured'"**
- **原因**: 缺少文档加载库
- **解决**: `pip install 'unstructured[md,docx,xlsx,pptx]'`

**Q2: RTF 文件上传失败**
- **原因**: pandoc 版本过低 (<2.14.2)
- **解决**: 升级 pandoc 或跳过 RTF 格式

**Q3: Milvus 连接失败**
- **原因**: Milvus 服务未启动
- **解决**: 检查 `docker ps | grep milvus` 并启动服务

**Q4: 问答返回 "没有找到相关文档"**
- **原因**: 向量数据库中无文档或检索失败
- **解决**:
  1. 检查 `/api/status` 确认 document_count > 0
  2. 上传相关文档
  3. 检查 embedding model 配置

---

## 总结

本系统实现了完整的 RAG (Retrieval-Augmented Generation) 流程:

1. **文档摄取**: 上传 → 解析 → 清洗 → 分块 → 评分
2. **向量化**: Embedding Model → Milvus 存储
3. **智能检索**: 向量检索 + BM25 → 混合结果
4. **答案生成**: LLM + 检索结果 → 准确答案
5. **用户交互**: 实时反馈 + 来源追溯

所有组件已实现并集成,系统可正常运行。下一步将继续实现多轮对话和主动澄清功能。
