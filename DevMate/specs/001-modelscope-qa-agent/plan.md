# Implementation Plan: 魔搭社区智能答疑 Agent

**Feature**: 001-modelscope-qa-agent
**Branch**: `001-modelscope-qa-agent`
**Created**: 2025-11-30
**Status**: Planning Complete
**Framework**: LangChain v1.0 + LangGraph + Milvus + Qwen

---

## 执行摘要

基于 LangChain v1.0 框架构建魔搭社区智能答疑 Agent,采用 RAG (Retrieval-Augmented Generation) 架构,结合 Milvus 向量数据库和通义千问大语言模型,实现准确、专业、可信的技术问答能力。

**核心技术栈**:
- **Agent 框架**: LangGraph (LangChain v1.0 推荐)
- **向量数据库**: Milvus 2.3+
- **嵌入模型**: 通义千问 Embedding (DashScope API)
- **大语言模型**: Qwen-2.5-72B-Instruct (通义千问 API)
- **检索策略**: 混合检索 (向量检索 + BM25)
- **对话管理**: MemorySaver Checkpointer + 滑动窗口摘要

---

## Phase 0: 研究与设计 ✅

**状态**: 已完成

**输出文档**:
- ✅ [research.md](design/research.md) - 技术研究报告
- ✅ [data-model.md](design/data-model.md) - 数据模型设计

**关键决策**:
1. 采用 LangGraph 构建 Agent 工作流
2. 使用 Milvus 作为向量数据库（符合规格要求）
3. 通义千问 Embedding + Qwen LLM（符合澄清结果）
4. 混合检索策略（向量 + BM25）
5. 滑动窗口 + 摘要的对话管理

---

## Phase 1: 核心架构实现

### 1.1 环境搭建与依赖安装

**目标**: 搭建完整的开发和运行环境

**任务清单**:
```bash
# 1. 创建项目目录结构
mkdir -p modelscope_qa_agent/{
  core,
  tools,
  retrievers,
  agents,
  models,
  data,
  tests,
  config
}

# 2. 安装核心依赖
pip install \
  langchain==0.1.0 \
  langchain-community==0.1.0 \
  langchain-core==0.1.0 \
  langgraph==0.1.0 \
  langchain-milvus==0.1.0 \
  pymilvus==2.3.0 \
  dashscope==1.14.0 \
  pydantic==2.5.0 \
  rank-bm25==0.2.2

# 3. 配置环境变量
cat > .env <<EOF
DASHSCOPE_API_KEY=your_qwen_api_key
MILVUS_HOST=localhost
MILVUS_PORT=19530
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
EOF
```

**验收标准**:
- ✅ 所有依赖包成功安装
- ✅ Milvus 服务正常运行（docker-compose up）
- ✅ 通义千问 API 连接测试通过
- ✅ LangSmith 监控配置成功

---

### 1.2 向量数据库初始化

**目标**: 搭建 Milvus 向量存储并定义 Schema

**实现代码**:
```python
# File: modelscope_qa_agent/core/vector_store.py

from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, connections
from langchain_milvus import Milvus
from langchain_community.embeddings import DashScopeEmbeddings
import os

class VectorStoreManager:
    """Milvus 向量存储管理器"""

    def __init__(self, host: str = "localhost", port: str = "19530"):
        # 连接 Milvus
        connections.connect(host=host, port=port)

        # 初始化嵌入模型
        self.embeddings = DashScopeEmbeddings(
            model="text-embedding-v2",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )

        # 创建或加载 Collection
        self.collection_name = "modelscope_docs"
        self._init_collection()

    def _init_collection(self):
        """初始化 Collection Schema"""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="content_summary", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="source_url", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="quality_score", dtype=DataType.FLOAT),
        ]

        schema = CollectionSchema(fields=fields, description="ModelScope Q&A KB")

        # 创建 Collection（如果不存在）
        if not Collection.exists(self.collection_name):
            collection = Collection(name=self.collection_name, schema=schema)

            # 创建索引
            index_params = {
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            collection.load()

    def get_vector_store(self) -> Milvus:
        """获取 LangChain Milvus 实例"""
        return Milvus(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_args={"host": "localhost", "port": "19530"}
        )
```

**测试脚本**:
```python
# File: tests/test_vector_store.py

def test_milvus_connection():
    """测试 Milvus 连接"""
    manager = VectorStoreManager()
    vector_store = manager.get_vector_store()

    # 测试写入
    test_doc = "这是一个测试文档,用于验证 Milvus 连接。"
    vector_store.add_texts([test_doc], metadatas=[{"source": "test"}])

    # 测试检索
    results = vector_store.similarity_search("测试文档", k=1)
    assert len(results) > 0
    assert "测试文档" in results[0].page_content
```

---

### 1.3 文档加载与分块处理

**目标**: 实现魔搭社区文档的抓取、清洗和语义分块

**实现代码**:
```python
# File: modelscope_qa_agent/core/document_processor.py

from langchain.document_loaders import WebBaseLoader, GitHubLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document
import re

class DocumentProcessor:
    """文档处理器（加载、清洗、分块）"""

    def __init__(self):
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "!", "?", " "]
        )

    def load_modelscope_docs(self) -> list[Document]:
        """加载魔搭社区官方文档"""
        urls = [
            "https://www.modelscope.cn/docs/overview",
            "https://www.modelscope.cn/docs/models",
            # ... 更多 URL
        ]

        documents = []
        for url in urls:
            loader = WebBaseLoader(url)
            docs = loader.load()
            documents.extend(docs)

        return documents

    def clean_document(self, doc: Document) -> Document:
        """清洗文档内容"""
        content = doc.page_content

        # 1. 移除多余空白
        content = re.sub(r'\n\s*\n', '\n\n', content)

        # 2. 移除 HTML 标签残留
        content = re.sub(r'<[^>]+>', '', content)

        # 3. 统一代码块格式
        content = re.sub(r'```(\w+)?\n(.*?)```', r'```\1\n\2\n```', content, flags=re.DOTALL)

        doc.page_content = content.strip()
        return doc

    def split_with_code_protection(self, doc: Document) -> list[Document]:
        """语义分块（保护代码块完整性）"""
        # 1. 先按 Markdown 标题分块
        header_chunks = self.markdown_splitter.split_text(doc.page_content)

        # 2. 对每个标题块进一步分块（保护代码块）
        final_chunks = []
        for chunk in header_chunks:
            # 检测代码块
            code_blocks = re.findall(r'```[\s\S]*?```', chunk.page_content)

            if code_blocks:
                # 有代码块 - 保持完整性
                final_chunks.append(chunk)
            else:
                # 无代码块 - 可以进一步拆分
                sub_chunks = self.text_splitter.split_documents([chunk])
                final_chunks.extend(sub_chunks)

        return final_chunks

    def calculate_quality_score(self, doc: Document) -> float:
        """计算文档质量评分（0-1）"""
        score = 0.0
        content = doc.page_content

        # 长度合理性（100-2000字符）
        length = len(content)
        if 100 < length < 2000:
            score += 0.25

        # 结构完整性（有标题）
        if re.search(r'#+\s', content):
            score += 0.25

        # 代码示例（技术文档必备）
        if '```' in content:
            score += 0.25

        # 来源可信度
        if doc.metadata.get("source_type") == "official_docs":
            score += 0.25

        return score
```

---

### 1.4 混合检索器实现

**目标**: 实现向量检索 + BM25 关键词检索的混合策略

**实现代码**:
```python
# File: modelscope_qa_agent/retrievers/hybrid_retriever.py

from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain_core.documents import Document
from langchain_milvus import Milvus

class HybridRetriever:
    """混合检索器（向量 + BM25）"""

    def __init__(self, vector_store: Milvus, documents: list[Document]):
        # 向量检索器
        self.vector_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )

        # BM25 关键词检索器
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 10

        # 混合检索器（加权融合）
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[0.6, 0.4]  # 向量60%, BM25 40%
        )

    def retrieve(self, query: str, k: int = 3) -> list[Document]:
        """执行混合检索"""
        # 初步检索（Top 10）
        results = self.ensemble_retriever.get_relevant_documents(query)

        # 重排序（可选 - 使用 Reranker 模型）
        # results = self.rerank(query, results)

        # 返回 Top K
        return results[:k]

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """重排序（可选实现）"""
        # TODO: 使用 Cross-encoder 模型进行精排
        return documents
```

---

### 1.5 LangGraph Agent 工作流

**目标**: 构建核心对话管理 Agent

**实现代码**:
```python
# File: modelscope_qa_agent/agents/qa_agent.py

from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.chat_models import ChatTongyi
from pydantic import BaseModel

# 状态定义
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_question: str
    retrieved_documents: list
    generated_answer: Optional[dict]
    turn_count: int

# 结构化输出模型
class TechnicalAnswer(BaseModel):
    problem_analysis: str
    solutions: list[str]
    code_examples: list[str]
    references: list[str]
    confidence_score: float

class ModelScopeQAAgent:
    """魔搭社区问答 Agent"""

    def __init__(self, retriever, llm_api_key: str):
        self.retriever = retriever
        self.llm = ChatTongyi(
            model="qwen-plus",
            temperature=0.3,
            dashscope_api_key=llm_api_key
        )

        # 构建工作流
        self.workflow = StateGraph(AgentState)
        self._build_graph()

        # 添加检查点器
        self.checkpointer = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    def _build_graph(self):
        """构建 LangGraph 工作流"""
        # 添加节点
        self.workflow.add_node("retrieve", self._retrieve_documents)
        self.workflow.add_node("generate", self._generate_answer)
        self.workflow.add_node("validate", self._validate_answer)

        # 设置入口
        self.workflow.set_entry_point("retrieve")

        # 添加边
        self.workflow.add_edge("retrieve", "generate")
        self.workflow.add_conditional_edges(
            "generate",
            self._should_validate,
            {
                "validate": "validate",
                "end": END
            }
        )
        self.workflow.add_edge("validate", END)

    def _retrieve_documents(self, state: AgentState) -> AgentState:
        """检索相关文档"""
        question = state["messages"][-1].content
        docs = self.retriever.retrieve(question, k=3)

        state["current_question"] = question
        state["retrieved_documents"] = docs
        return state

    def _generate_answer(self, state: AgentState) -> AgentState:
        """生成技术回答"""
        from langchain.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import PydanticOutputParser

        # 构建上下文
        context = "\n\n".join([doc.page_content for doc in state["retrieved_documents"]])

        # 系统提示词
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是魔搭社区的技术支持专家。

**任务**: 基于提供的文档上下文,回答用户的技术问题。

**要求**:
1. 回答必须基于文档内容,不得编造
2. 提供至少1种可执行的解决方案
3. 包含完整的代码示例（如果适用）
4. 引用信息来源

**上下文文档**:
{context}

**输出格式**: 请使用以下 JSON 格式:
{format_instructions}
"""),
            ("human", "{question}")
        ])

        # 解析器
        parser = PydanticOutputParser(pydantic_object=TechnicalAnswer)

        # 生成答案
        chain = prompt | self.llm | parser
        answer = chain.invoke({
            "context": context,
            "question": state["current_question"],
            "format_instructions": parser.get_format_instructions()
        })

        state["generated_answer"] = answer.dict()
        return state

    def _validate_answer(self, state: AgentState) -> AgentState:
        """验证答案质量"""
        # TODO: 实现答案验证逻辑（Self-RAG）
        return state

    def _should_validate(self, state: AgentState) -> str:
        """判断是否需要验证"""
        confidence = state["generated_answer"].get("confidence_score", 0)
        return "validate" if confidence < 0.8 else "end"

    def invoke(self, question: str, thread_id: str = "default") -> dict:
        """调用 Agent"""
        result = self.app.invoke(
            {"messages": [HumanMessage(content=question)], "turn_count": 0},
            {"configurable": {"thread_id": thread_id}}
        )
        return result["generated_answer"]
```

---

## Phase 2: 功能增强与优化

### 2.1 问题改写与分类

**目标**: 实现问题优化和自动分类

**实现要点**:
- 使用 LLM 进行问题改写（扩展关键词、补充上下文）
- 实现问题分类器（模型使用/技术问题/平台功能/项目指导）
- 基于分类结果路由到不同处理流程

### 2.2 多轮对话管理

**目标**: 实现滑动窗口 + 摘要策略

**实现要点**:
- 使用 `trim_messages` 保留最近10轮对话
- 超过10轮后,早期消息压缩为摘要
- 在状态中维护 `conversation_summary` 字段

### 2.3 多模态支持

**目标**: 处理用户上传的报错截图

**实现要点**:
- 使用 `qwen-vl-plus` 模型识别图片中的错误信息
- 提取错误类型、错误代码和堆栈跟踪
- 将提取的文本作为检索查询

### 2.4 缓存与性能优化

**目标**: 降低响应延迟和 API 成本

**实现要点**:
- LLM 响应缓存（InMemoryCache）
- 检索结果缓存（Redis）
- 异步处理（并行检索和问题改写）
- 批量向量化（减少 Embedding API 调用）

---

## Phase 3: 评估与监控

### 3.1 RAG 评估

**评估指标**:
- Context Relevance: 检索文档与问题的相关性
- Answer Relevance: 答案与问题的相关性
- Faithfulness: 答案与文档的一致性
- Answer Correctness: 答案的正确性

**工具**: RAGAs 评估框架

### 3.2 LangSmith 监控

**监控内容**:
- Token 使用量和成本
- 延迟分布（P50, P95, P99）
- 错误率和异常
- 用户反馈评分

---

## 技术债务与改进计划

### 短期（1-2周）
- [ ] 实现 Self-RAG 答案验证
- [ ] 添加 Reranker 模型精排
- [ ] 实现问题分解策略

### 中期（1-2月）
- [ ] 向量模型微调（基于魔搭语料）
- [ ] 探索 LightRAG/HippoRAG 算法
- [ ] 实现 A/B 测试框架

### 长期（3-6月）
- [ ] 本地模型部署（降低 API 依赖）
- [ ] 多模型混合策略
- [ ] 知识图谱增强（HippoRAG）

---

## 依赖清单

```txt
# requirements.txt
langchain==0.1.0
langchain-community==0.1.0
langchain-core==0.1.0
langgraph==0.1.0
langchain-milvus==0.1.0

# 向量数据库
pymilvus==2.3.0

# 嵌入和 LLM
dashscope==1.14.0

# 数据处理
pydantic==2.5.0
pydantic-settings==2.1.0

# 检索增强
rank-bm25==0.2.2
ragas==0.1.0

# 工具
beautifulsoup4==4.12.0
markdownify==0.11.6
python-dotenv==1.0.0

# 数据库
sqlalchemy==2.0.0
redis==5.0.0

# 监控
langsmith==0.1.0
```

---

## 成功标准验证

| 规格需求 | 实现方案 | 验证方法 |
|---------|---------|---------|
| FR-001 问题分类 | LLM Classifier + LangGraph Conditional Edges | 分类准确率 >95% |
| FR-002 知识库构建 | Milvus + 语义分块 | 文档覆盖率 100% |
| FR-003 对话管理 | MemorySaver + trim_messages | 10轮历史测试 |
| FR-007 混合检索 | EnsembleRetriever | 检索准确率 >85% |
| SC-001 响应速度 | 缓存 + 异步 | 平均<30秒 |
| SC-003 用户满意度 | 反馈收集 | 平均≥4.0分 |
| SC-006 检索准确率 | RAGAs 评估 | ≥85% |
| SC-009 幻觉率 | Self-RAG 验证 | <5% |

---

**计划状态**: ✅ Ready for Implementation
**最后更新**: 2025-11-30
**审核人**: Pending Review
