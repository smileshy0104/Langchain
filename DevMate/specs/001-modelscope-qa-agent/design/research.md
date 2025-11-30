# Research Document: 魔搭社区智能答疑 Agent

**Feature**: 001-modelscope-qa-agent
**Created**: 2025-11-30
**Status**: Completed

## 研究目标

基于 LangChain v1.0 框架和 Milvus 向量数据库,研究并确定魔搭社区智能答疑 Agent 的技术实现方案。

---

## 1. LangChain v1.0 核心组件研究

### 1.1 Agent 架构 (LangGraph)

**决策**: 采用 LangGraph 构建对话管理和工作流编排

**理由**:
- LangGraph 是 LangChain v1.0 推荐的 Agent 编排框架
- 支持复杂的状态管理和条件分支
- 原生支持人机协作 (Human-in-the-Loop)
- 内置 Checkpointer 机制支持对话持久化

**核心模式**:
```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    documents: list  # 检索到的文档
    user_feedback: Optional[dict]  # 用户反馈

# 构建图
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("generate", generate_answer)
workflow.add_conditional_edges("retrieve", should_rewrite)
workflow.set_entry_point("retrieve")

# 添加检查点器
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
```

**参考代码**: `/Users/yuyansong/AiProject/Langchain/langchain_agents_examples/`

---

### 1.2 对话记忆管理 (Short-term Memory)

**决策**: 采用滑动窗口 + 消息摘要策略

**理由**:
- 符合规格文档 FR-003 的要求
- LangChain 提供 `trim_messages` 工具函数
- 支持自定义状态扩展（添加摘要字段）
- 可通过 `Checkpointer` 实现多会话管理

**核心模式**:
```python
from langchain_core.messages import trim_messages
from langgraph.checkpoint.memory import MemorySaver

# 消息修剪配置
def trim_conversation(messages: list, max_tokens: int = 4000):
    """保留最近10轮完整对话,更早内容压缩为摘要"""
    return trim_messages(
        messages,
        max_tokens=max_tokens,
        strategy="last",  # 保留最新消息
        token_counter=len,  # 使用字符长度作为近似
        include_system=True,
        start_on="human"
    )

# 状态定义
class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str  # 早期对话摘要
    turn_count: int  # 对话轮次计数
```

**参考代码**: `/Users/yuyansong/AiProject/Langchain/langchain_short_term_memory_examples/`

---

### 1.3 工具定义 (Tools)

**决策**: 采用 `@tool` 装饰器 + Pydantic Schema 混合方式

**理由**:
- 简单工具使用 `@tool` 装饰器快速定义
- 复杂工具使用 `args_schema` 进行严格验证
- 支持同步和异步工具
- 内置错误处理机制 (ToolException)

**核心模式**:
```python
from langchain_core.tools import tool, ToolException
from pydantic import BaseModel, Field, validator

# 简单工具
@tool
def search_modelscope_docs(query: str, limit: int = 3) -> str:
    """搜索魔搭社区官方文档"""
    # 实现检索逻辑
    return "检索结果..."

# 复杂工具（带验证）
class QueryRewriteInput(BaseModel):
    original_query: str = Field(description="用户原始问题")
    context: list = Field(default=[], description="对话上下文")

    @validator("original_query")
    def validate_query(cls, v):
        if len(v.strip()) < 3:
            raise ValueError("查询太短,至少3个字符")
        return v.strip()

@tool(args_schema=QueryRewriteInput)
def rewrite_query(original_query: str, context: list) -> str:
    """改写用户问题以提升检索效果"""
    # 使用 LLM 改写问题
    return "改写后的问题..."
```

**参考代码**: `/Users/yuyansong/AiProject/Langchain/langchain_tools_examples/`

---

### 1.4 结构化输出 (Structured Output)

**决策**: 使用 Pydantic BaseModel 定义响应格式

**理由**:
- 符合 FR-005 要求的结构化技术回答
- 强制模型返回预定义格式
- 便于后续处理和展示
- 支持嵌套结构和联合类型

**核心模式**:
```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class TechnicalAnswer(BaseModel):
    """技术问答响应格式"""
    problem_analysis: str = Field(description="问题分析")
    solutions: list[str] = Field(description="解决方案列表,至少1种")
    code_examples: list[str] = Field(default=[], description="代码示例")
    references: list[str] = Field(default=[], description="参考文档来源")
    confidence_score: float = Field(ge=0, le=1, description="置信度评分")

# 使用
parser = PydanticOutputParser(pydantic_object=TechnicalAnswer)
chain = prompt | llm | parser
```

**参考代码**: `/Users/yuyansong/AiProject/Langchain/langchain_structured_output_examples/`

---

## 2. 向量数据库集成研究 (Milvus)

### 2.1 Milvus 与 LangChain 集成

**决策**: 使用 `langchain-milvus` 官方集成包

**安装**:
```bash
pip install langchain-milvus pymilvus
```

**核心模式**:
```python
from langchain_milvus import Milvus
from langchain_community.embeddings import DashScopeEmbeddings  # 通义千问

# 初始化嵌入模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 连接 Milvus
vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={
        "host": "localhost",
        "port": "19530"
    },
    collection_name="modelscope_docs",
    index_params={
        "metric_type": "IP",  # 内积（适合归一化向量）
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
)
```

**参考文档**: https://python.langchain.com/docs/integrations/vectorstores/milvus

---

### 2.2 混合检索策略

**决策**: 结合向量检索 + BM25 关键词检索

**理由**:
- 符合规格文档 FR-007 要求
- 向量检索擅长语义匹配
- BM25 擅长精确关键词匹配
- 两者互补,提升准确率

**核心模式**:
```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# 向量检索器
vector_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)

# BM25 关键词检索器
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10

# 混合检索（加权融合）
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.6, 0.4]  # 向量60%, BM25 40%
)
```

**参考文档**: https://python.langchain.com/docs/how_to/ensemble_retriever

---

### 2.3 文档分块策略

**决策**: 采用语义分块（基于段落、章节、代码块边界）

**理由**:
- 符合规格文档澄清会话结果
- 保持内容完整性和上下文连贯性
- 适合技术文档的结构化特点

**核心模式**:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

# 1. Markdown 文档按标题分块
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
)

# 2. 代码块识别和保护
def split_with_code_protection(text: str, chunk_size: int = 1000):
    """识别代码块并保持完整性"""
    code_block_pattern = r'```[\s\S]*?```'
    # 提取代码块
    code_blocks = re.findall(code_block_pattern, text)
    # 替换为占位符
    text_with_placeholders = re.sub(code_block_pattern, "<<<CODE_BLOCK>>>", text)

    # 分块非代码部分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "!", "?", ";", " ", ""]
    )
    chunks = text_splitter.split_text(text_with_placeholders)

    # 还原代码块
    # ... (实现占位符替换逻辑)
    return chunks
```

---

## 3. 大语言模型集成研究 (Qwen)

### 3.1 通义千问 API 集成

**决策**: 使用 Qwen-2.5-72B-Instruct 作为主模型

**理由**:
- 符合规格文档澄清结果
- 魔搭社区原生支持
- 中文和代码生成能力优秀
- 支持 Function Calling

**核心模式**:
```python
from langchain_community.chat_models import ChatTongyi

llm = ChatTongyi(
    model="qwen-plus",  # 或 qwen-turbo, qwen-max
    temperature=0.3,  # 降低随机性以提高准确性
    top_p=0.8,
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    streaming=True  # 支持流式输出
)
```

**替代方案**: 通过 ModelScope 平台调用
```python
from modelscope import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    'qwen/Qwen-2.5-72B-Instruct',
    device_map='auto',
    trust_remote_code=True
)
```

**参考文档**: https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-by-api

---

### 3.2 Prompt Engineering

**决策**: 使用 PromptTemplate + Few-shot Examples

**核心模式**:
```python
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# Few-shot 示例
examples = [
    {
        "question": "如何在魔搭社区使用Qwen模型?",
        "answer": {
            "problem_analysis": "用户需要了解Qwen模型的基本调用方法",
            "solutions": ["通过 ModelScope SDK 调用", "通过 API 调用"],
            "code_examples": ["from modelscope import ..."],
            "references": ["官方文档-模型调用指南"]
        }
    }
]

# 系统提示词
system_template = """你是魔搭社区的技术支持专家,负责回答开发者的技术问题。

**核心职责**:
1. 准确理解用户的技术问题
2. 从知识库中检索相关文档
3. 提供结构化的技术回答（包括问题分析、解决方案、代码示例）
4. 引用官方文档来源以增强可信度

**回答要求**:
- 回答必须基于检索到的文档内容,不得编造信息
- 提供至少1种可执行的解决方案
- 代码示例必须完整可运行
- 明确说明信息来源

**上下文文档**:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", "{question}")
])
```

---

## 4. RAG 优化策略研究

### 4.1 Query Rewriting (问题改写)

**决策**: 使用 LLM 进行问题扩展和改写

**实现方式**:
```python
from langchain.prompts import ChatPromptTemplate

rewrite_prompt = ChatPromptTemplate.from_template("""
你是一个专业的查询优化专家。

原始问题: {original_question}
对话历史: {conversation_history}

任务: 将原始问题改写为更适合检索的查询语句。

改写策略:
1. 补充上下文信息（基于对话历史）
2. 展开缩写和专业术语
3. 添加相关关键词
4. 保持语义完整性

改写后的查询:
""")

query_rewriter = rewrite_prompt | llm | StrOutputParser()
```

---

### 4.2 Re-ranking (重排序)

**决策**: 使用 Cross-encoder 模型进行精排

**实现方式**:
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

# 初始化 Reranker
compressor = CohereRerank(
    model="rerank-multilingual-v2.0",
    top_n=3  # 从10个初排结果中选择Top 3
)

# 包装原有检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever
)
```

---

### 4.3 Self-RAG (自我反思检索)

**决策**: 实现答案可信度评估机制

**核心流程**:
```
1. 生成初步答案
2. 评估答案与检索文档的相关性
3. 如果相关性低 → 重新检索或改写问题
4. 如果相关性高 → 返回答案
```

**实现方式**:
```python
def self_rag_answer(question: str, max_iterations: int = 3):
    """Self-RAG 答案生成流程"""
    for i in range(max_iterations):
        # 1. 检索文档
        docs = retriever.get_relevant_documents(question)

        # 2. 生成答案
        answer = generate_answer(question, docs)

        # 3. 评估可信度
        relevance_score = evaluate_relevance(question, docs, answer)

        if relevance_score > 0.8:
            return answer
        elif i < max_iterations - 1:
            # 改写问题重试
            question = rewrite_query(question, docs)

    return answer  # 达到最大迭代次数
```

---

## 5. 多模态支持研究

### 5.1 图片理解 (OCR + Vision Model)

**决策**: 使用通义千问 VL 模型处理报错截图

**实现方式**:
```python
from langchain_community.chat_models import ChatTongyi

vision_llm = ChatTongyi(
    model="qwen-vl-plus",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

def process_error_screenshot(image_path: str) -> str:
    """从错误截图中提取错误信息"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请识别这张截图中的错误信息,包括错误类型、错误代码和堆栈跟踪。"},
                {"type": "image_url", "image_url": {"url": f"file://{image_path}"}}
            ]
        }
    ]
    response = vision_llm.invoke(messages)
    return response.content
```

---

### 5.2 代码片段解析

**决策**: 使用正则表达式 + AST 解析

**实现方式**:
```python
import ast
import re

def extract_and_validate_code(text: str) -> dict:
    """提取并验证代码片段"""
    # 提取 Python 代码块
    code_pattern = r'```python\n(.*?)\n```'
    code_blocks = re.findall(code_pattern, text, re.DOTALL)

    results = []
    for code in code_blocks:
        try:
            # 语法检查
            ast.parse(code)
            results.append({
                "code": code,
                "valid": True,
                "error": None
            })
        except SyntaxError as e:
            results.append({
                "code": code,
                "valid": False,
                "error": str(e)
            })

    return results
```

---

## 6. 性能优化研究

### 6.1 缓存策略

**决策**: 实现多层缓存机制

**实现方式**:
```python
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache

# 1. LLM 响应缓存
set_llm_cache(InMemoryCache())

# 2. 检索结果缓存
class RetrievalCache:
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size

    def get(self, query: str) -> Optional[list]:
        return self.cache.get(query)

    def set(self, query: str, documents: list):
        if len(self.cache) >= self.max_size:
            # LRU 淘汰
            self.cache.pop(next(iter(self.cache)))
        self.cache[query] = documents
```

---

### 6.2 异步处理

**决策**: 对 I/O 密集操作使用异步

**实现方式**:
```python
import asyncio
from langchain.schema.runnable import RunnableParallel

# 并行检索和问题改写
async def parallel_retrieval(question: str):
    tasks = {
        "vector_docs": vector_retriever.ainvoke(question),
        "bm25_docs": bm25_retriever.ainvoke(question),
        "rewritten_query": query_rewriter.ainvoke(question)
    }
    results = await RunnableParallel(tasks).ainvoke({})
    return results
```

---

## 7. 评估与监控研究

### 7.1 RAG 评估指标

**决策**: 实现 RAGAs 评估框架

**核心指标**:
- **Context Relevance**: 检索文档与问题的相关性
- **Answer Relevance**: 答案与问题的相关性
- **Faithfulness**: 答案与检索文档的一致性
- **Answer Correctness**: 答案的正确性

**实现方式**:
```python
from ragas import evaluate
from ragas.metrics import (
    context_relevance,
    answer_relevance,
    faithfulness,
    answer_correctness
)

# 评估数据集
eval_dataset = {
    "question": ["如何使用Qwen模型?"],
    "answer": ["生成的答案"],
    "contexts": [["检索到的文档1", "文档2"]],
    "ground_truth": ["标准答案"]
}

# 执行评估
results = evaluate(
    eval_dataset,
    metrics=[
        context_relevance,
        answer_relevance,
        faithfulness,
        answer_correctness
    ]
)
```

**参考文档**: https://docs.ragas.io/

---

### 7.2 LangSmith 监控

**决策**: 使用 LangSmith 进行生产环境监控

**配置**:
```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "modelscope-qa-agent"

# 自动记录所有 LLM 调用、检索和 Agent 执行
```

**监控指标**:
- Token 使用量
- 延迟分布
- 错误率
- 用户反馈评分

---

## 8. 数据管理研究

### 8.1 知识库构建流程

**决策**: 采用批量 + 增量更新模式

**工作流**:
```python
from langchain.document_loaders import (
    WebBaseLoader,
    GitHubLoader,
    MarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_knowledge_base():
    """构建知识库"""
    # 1. 数据源加载
    loaders = [
        WebBaseLoader("https://www.modelscope.cn/docs/overview"),
        GitHubLoader(repo="modelscope/modelscope", file_filter=lambda x: x.endswith(".md"))
    ]

    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    # 2. 数据清洗
    documents = clean_documents(documents)

    # 3. 分块
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # 4. 添加元数据
    for chunk in chunks:
        chunk.metadata["source_type"] = "official_docs"
        chunk.metadata["last_updated"] = datetime.now().isoformat()

    # 5. 存入 Milvus
    vector_store.add_documents(chunks)
```

---

### 8.2 数据质量评分

**决策**: 实现自动化质量评分机制

**评分维度**:
```python
def calculate_quality_score(document: Document) -> float:
    """计算文档质量评分"""
    score = 0.0

    # 1. 长度检查（避免过短或过长）
    length = len(document.page_content)
    if 100 < length < 2000:
        score += 0.25

    # 2. 结构完整性（包含标题、代码块）
    if re.search(r'#+\s', document.page_content):  # 有标题
        score += 0.25

    # 3. 代码示例（技术文档必备）
    if '```' in document.page_content:
        score += 0.25

    # 4. 来源可信度
    if document.metadata.get("source_type") == "official_docs":
        score += 0.25

    return score
```

---

## 9. 技术栈总结

### 9.1 核心依赖

```python
# requirements.txt
langchain==0.1.0
langchain-community==0.1.0
langchain-core==0.1.0
langgraph==0.1.0
langchain-milvus==0.1.0

# 向量数据库
pymilvus==2.3.0
milvus==2.3.0

# 嵌入和 LLM
dashscope==1.14.0  # 通义千问 API

# 数据处理
pydantic==2.5.0
pydantic-settings==2.1.0

# 检索增强
rank-bm25==0.2.2
ragas==0.1.0  # RAG 评估

# 工具
beautifulsoup4==4.12.0  # 文档爬取
markdownify==0.11.6  # HTML 转 Markdown
```

---

### 9.2 架构组件映射

| 规格需求 | LangChain 组件 | 实现方式 |
|---------|---------------|---------|
| FR-001 问题分类 | LangGraph Conditional Edges | 条件路由节点 |
| FR-002 知识库构建 | Milvus + DashScopeEmbeddings | 向量存储 |
| FR-003 对话管理 | MemorySaver + trim_messages | Checkpointer + 消息修剪 |
| FR-004 多模态 | ChatTongyi (qwen-vl-plus) | Vision LLM |
| FR-005 结构化输出 | Pydantic + OutputParser | Schema 定义 |
| FR-006 代码生成 | ChatTongyi (qwen-plus) | LLM 生成 |
| FR-007 混合检索 | EnsembleRetriever | 向量 + BM25 |
| FR-008 来源引用 | Document metadata | 元数据追踪 |
| FR-009 问题澄清 | Human-in-the-Loop | interrupt 节点 |
| FR-017 问题改写 | LLM Chain | Prompt + LLM |
| FR-019 缓存 | InMemoryCache / SQLiteCache | LangChain 缓存 |
| FR-020 A/B测试 | LangSmith | 监控平台 |

---

## 10. 替代方案对比

### 10.1 向量模型对比

| 模型 | 维度 | 性能 | 成本 | 推荐场景 |
|------|------|------|------|---------|
| 通义千问 Embedding | 1536 | ⭐⭐⭐⭐⭐ | API 调用 | 推荐（原生支持） |
| BGE-large-zh-v1.5 | 1024 | ⭐⭐⭐⭐ | 本地部署 | 算力充足时 |
| M3E-base | 768 | ⭐⭐⭐ | 本地部署 | 资源受限时 |

**最终选择**: 通义千问 Embedding（符合规格澄清结果）

---

### 10.2 检索策略对比

| 策略 | 准确率 | 召回率 | 延迟 | 复杂度 |
|------|-------|-------|------|--------|
| 纯向量检索 | ⭐⭐⭐ | ⭐⭐⭐⭐ | 低 | 低 |
| 纯 BM25 | ⭐⭐⭐ | ⭐⭐⭐ | 极低 | 极低 |
| 混合检索 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 中 | 中 |
| Self-RAG | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 高 | 高 |

**最终选择**: 混合检索作为基础,Self-RAG 作为进阶优化

---

## 11. 风险与缓解措施

### 11.1 技术风险

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| API 限流 | 高 | 中 | 实现本地模型备份 + 限流队列 |
| 向量数据库故障 | 高 | 低 | 定期备份 + 副本部署 |
| 检索准确率不达标 | 中 | 中 | A/B 测试 + 迭代优化 |
| LLM 幻觉 | 高 | 中 | Self-RAG + 答案验证 |

---

### 11.2 性能风险

| 风险 | 目标 | 缓解措施 |
|------|------|---------|
| 响应延迟过高 | <3秒 | 缓存 + 异步 + 模型量化 |
| 并发处理能力不足 | 100+ | 异步架构 + 负载均衡 |
| Token 消耗过高 | 成本控制 | 消息修剪 + 缓存复用 |

---

## 12. 研究结论

### 12.1 技术可行性

✅ **LangChain v1.0 完全满足需求**:
- LangGraph 提供强大的 Agent 编排能力
- 原生支持 Milvus 向量数据库
- 完善的工具和记忆管理机制
- 丰富的社区生态和文档

### 12.2 核心技术栈

- **Agent 框架**: LangGraph + MemorySaver
- **向量数据库**: Milvus + 通义千问 Embedding
- **大语言模型**: Qwen-2.5-72B-Instruct (通义千问 API)
- **检索策略**: EnsembleRetriever (向量 + BM25)
- **监控评估**: LangSmith + RAGAs

### 12.3 下一步行动

1. Phase 1: 数据模型设计（Entity Schema）
2. Phase 1: API 契约定义（Tool Interfaces）
3. Phase 1: 快速启动指南（Quickstart）
4. Phase 2: 核心功能实现
5. Phase 2: 测试与优化

---

**研究完成日期**: 2025-11-30
**研究负责人**: AI Assistant
**审核状态**: 待审核
