# LangChain Retrieval & Memory（检索与记忆）详细指南

## 目录

1. [概述](#概述)
2. [检索基础](#检索基础)
3. [知识库构建](#知识库构建)
4. [RAG 架构](#rag-架构)
5. [向量存储](#向量存储)
6. [嵌入模型](#嵌入模型)
7. [长期记忆](#长期记忆)
8. [短期记忆](#短期记忆)
9. [Store API](#store-api)
10. [内存管理](#内存管理)
11. [实战案例](#实战案例)
12. [最佳实践](#最佳实践)
13. [快速参考](#快速参考)

---

## 概述

### 为什么需要检索与记忆

**大型语言模型（LLMs）的局限性：**

```
┌─────────────────────────────────────────────────────────────┐
│                  LLM 的两大限制                               │
│                                                             │
│  1. 有限上下文 (Finite Context)                              │
│     ├─ 无法一次性处理整个文档库                               │
│     ├─ 上下文窗口大小受限                                    │
│     └─ 长对话中信息丢失                                      │
│                                                             │
│  2. 静态知识 (Static Knowledge)                              │
│     ├─ 训练数据截止时间                                      │
│     ├─ 无法访问实时信息                                      │
│     └─ 无法记住用户偏好                                      │
│                                                             │
│  解决方案：检索 + 记忆                                        │
│  ├─ Retrieval：动态获取外部知识                              │
│  └─ Memory：跨会话保持信息                                   │
└─────────────────────────────────────────────────────────────┘
```

### Retrieval（检索）

**Retrieval** 在查询时获取相关的外部知识，这是 **RAG（检索增强生成）** 的基础。

| 功能 | 说明 | 应用场景 |
|------|------|----------|
| 知识检索 | 从文档库中获取相关信息 | 文档问答、知识库查询 |
| 语义搜索 | 基于含义而非关键词匹配 | 智能搜索、推荐系统 |
| 上下文增强 | 为 LLM 提供外部知识 | RAG、Agent 工具 |

### Memory（记忆）

**Memory** 系统让 Agent 能够记住之前的交互、从反馈中学习并适应用户偏好。

| 类型 | 范围 | 存储位置 | 持久性 |
|------|------|----------|--------|
| **短期记忆** | 单个会话 (thread) | Checkpointer | 跨调用保持 |
| **长期记忆** | 跨会话 | Store | 永久存储 |

---

## 检索基础

### 检索工作流

```
┌─────────────────────────────────────────────────────────────┐
│                    检索流程 (Retrieval)                       │
│                                                             │
│  索引阶段 (Indexing)                                         │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │
│  │ Sources │───→│ Loader  │───→│ Split   │───→│ Vector  │   │
│  │ 文档源  │    │ 加载器  │    │ 分块    │    │ Store   │   │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘   │
│                                     │                      │
│                                     ▼                      │
│                                ┌─────────┐                  │
│                                │Embedding│                  │
│                                │嵌入模型 │                  │
│                                └─────────┘                  │
│                                                             │
│  查询阶段 (Query)                                            │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐   │
│  │  Query  │───→│Embedding│───→│ Search  │───→│ Results │   │
│  │ 用户查询 │    │查询嵌入 │    │相似搜索 │    │检索结果 │   │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘   │
│                                                             │
│  生成阶段 (Generation)                                       │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                  │
│  │ Results │───→│   LLM   │───→│ Answer  │                  │
│  │ 检索结果 │    │生成回答 │    │ 最终答案 │                  │
│  └─────────┘    └─────────┘    └─────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 功能 | LangChain 实现 |
|------|------|----------------|
| **Document Loaders** | 从外部源加载数据 | `langchain.document_loaders` |
| **Text Splitters** | 将大文档切分为小块 | `langchain.text_splitter` |
| **Embedding Models** | 文本转换为向量 | `langchain.embeddings` |
| **Vector Stores** | 存储和搜索向量 | `langchain.vectorstores` |
| **Retrievers** | 检索接口 | `langchain_core.retrievers` |

---

## 知识库构建

### 基本构建流程

```python
from langchain_core.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

# 1. 加载文档
loader = TextLoader("path/to/document.txt")
documents = loader.load()

# 2. 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
splits = text_splitter.split_documents(documents)

# 3. 创建嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 4. 创建向量存储
vectorstore = InMemoryVectorStore.from_documents(
    documents=splits,
    embedding=embeddings,
)

# 5. 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 6. 检索相关文档
query = "什么是机器学习？"
results = retriever.invoke(query)

for doc in results:
    print(f"内容: {doc.page_content[:100]}...")
    print(f"元数据: {doc.metadata}")
    print("---")
```

### 文档加载器

```python
from langchain_core.document_loaders import (
    TextLoader,
    PyPDFLoader,
    DirectoryLoader,
    WebBaseLoader,
    JSONLoader,
)

# 文本文件
loader = TextLoader("file.txt")
docs = loader.load()

# PDF 文件
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# 目录加载
loader = DirectoryLoader(
    "path/to/dir",
    glob="**/*.txt",
    loader_cls=TextLoader,
)
docs = loader.load()

# 网页加载
loader = WebBaseLoader("https://example.com")
docs = loader.load()

# JSON 文件
loader = JSONLoader(
    "file.json",
    jq_schema=".[]",
    text_content=False,
)
docs = loader.load()
```

### 文本分割策略

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)

# 递归分割（推荐）
# 按段落、句子、单词递归尝试分割
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", "！", "?", " ", ""],
)

# 字符分割
character_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
)

# Token 分割（适用于 LLM）
token_splitter = TokenTextSplitter(
    chunk_size=500,  # token 数
    chunk_overlap=50,
    encoding_name="cl100k_base",  # OpenAI encoding
)

# 代码分割
from langchain_text_splitters import PythonCodeTextSplitter

code_splitter = PythonCodeTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# Markdown 分割
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ],
)
```

---

## RAG 架构

### 架构对比

| 架构 | 描述 | 控制力 | 灵活性 | 延迟 | 适用场景 |
|------|------|--------|--------|------|----------|
| **2-Step RAG** | 先检索后生成 | 高 | 低 | 快 | FAQ、文档问答 |
| **Agentic RAG** | Agent 自主决定何时检索 | 低 | 高 | 可变 | 复杂研究助手 |
| **Hybrid RAG** | 混合两者，带验证步骤 | 中 | 中 | 可变 | 高质量问答 |

### 2-Step RAG

最简单的 RAG 实现，检索和生成是固定的两个步骤。

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 1. 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 2. 创建提示词模板
prompt = ChatPromptTemplate.from_template("""
根据以下上下文回答问题：

上下文：
{context}

问题：{input}

答案：
""")

# 3. 创建 LLM
llm = ChatOpenAI(model="gpt-4o")

# 4. 创建文档链
combine_docs_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

# 5. 创建检索链
rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=combine_docs_chain
)

# 6. 执行查询
result = rag_chain.invoke({"input": "什么是机器学习？"})
print(result["answer"])
```

### Agentic RAG

Agent 自主决定何时检索、检索什么、如何使用检索结果。

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
import requests

# 1. 创建检索工具
@tool
def fetch_url(url: str) -> str:
    """获取网页内容"""
    response = requests.get(url, timeout=10.0)
    response.raise_for_status()
    return response.text

@tool
def search_knowledge(query: str) -> str:
    """搜索知识库"""
    results = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in results])

# 2. 创建 Agent
agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[fetch_url, search_knowledge],
    system_prompt="""你是一个研究助手。
使用 search_knowledge 搜索内部知识库。
使用 fetch_url 获取最新在线信息。
在回答问题之前，确保你引用了相关的信息来源。
""",
)

# 3. 执行查询
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "比较最新的 TensorFlow 和 PyTorch 性能差异"
    }]
})

print(result["messages"][-1].content)
```

### Hybrid RAG

结合两者的优点，添加验证和自纠正步骤。

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 1. 查询增强工具
@tool
def rewrite_query(query: str) -> str:
    """重写查询以提高检索质量"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_template("""
重写以下查询，使其更清晰、更具体：
原查询：{query}
重写后的查询：
""")
    result = llm.invoke(prompt.format(query=query))
    return result.content

# 2. 检索验证工具
@tool
def validate_retrieval(query: str, results: str) -> str:
    """验证检索结果是否相关"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_template("""
评估以下检索结果是否与查询相关：

查询：{query}
检索结果：{results}

评分（1-5）和说明：
""")
    result = llm.invoke(prompt.format(query=query, results=results))
    return result.content

# 3. 答案验证工具
@tool
def validate_answer(answer: str, context: str) -> str:
    """验证答案是否基于上下文"""
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_template("""
检查以下答案是否基于给定的上下文：

答案：{answer}
上下文：{context}

答案是否准确？是否包含上下文之外的编造信息？
""")
    result = llm.invoke(prompt.format(answer=answer, context=context))
    return result.content

# 4. 创建混合 RAG Agent
agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[rewrite_query, search_knowledge, validate_retrieval, validate_answer],
    system_prompt="""你是一个高质量的问答系统。

流程：
1. 对于复杂的查询，使用 rewrite_query 重写
2. 使用 search_knowledge 检索相关信息
3. 使用 validate_retrieval 验证检索质量
4. 如果质量不足，重新检索
5. 基于检索结果生成答案
6. 使用 validate_answer 验证答案准确性
""",
)
```

---

## 向量存储

### 向量存储类型

LangChain 支持 40+ 种向量存储：

| 类型 | 适用场景 | 特点 |
|------|----------|------|
| **InMemoryVectorStore** | 开发测试 | 简单、无需数据库 |
| **FAISS** | 本地生产 | 快速、高效 |
| **PGVector** | PostgreSQL 环境 | 与现有数据库集成 |
| **Pinecone** | 云端托管 | 托管服务、易扩展 |
| **Chroma** | 轻量级生产 | 简单、本地持久化 |
| **Qdrant** | 高性能需求 | 快速、过滤支持 |

### 基本用法

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 1. 创建向量存储
embeddings = OpenAIEmbeddings()

# 从文档创建
documents = [
    Document(page_content="Python 是一种编程语言"),
    Document(page_content="机器学习是 AI 的分支"),
]
vectorstore = InMemoryVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
)

# 从文本创建
texts = ["LangChain 是一个框架", "RAG 是检索增强生成"]
vectorstore = InMemoryVectorStore.from_texts(
    texts=texts,
    embedding=embeddings,
)

# 2. 相似度搜索
query = "什么是 Python？"
results = vectorstore.similarity_search(query, k=2)

for doc in results:
    print(doc.page_content)

# 3. 带分数的相似度搜索
results_with_scores = vectorstore.similarity_search_with_score(query, k=2)

for doc, score in results_with_scores:
    print(f"分数: {score:.4f}")
    print(f"内容: {doc.page_content}")
    print("---")

# 4. 最大边际相关性搜索 (MMR)
# 平衡相关性和多样性
results = vectorstore.max_marginal_relevance_search(
    query,
    k=4,
    fetch_k=10,  # 获取更多候选以提高多样性
)
```

### 高级检索

```python
# 1. 过滤搜索
from langchain_core.vectorstores import VectorStore

results = vectorstore.similarity_search(
    query="编程语言",
    k=3,
    filter={"source": "python_docs"},  # 元数据过滤
)

# 2. 自定义检索器
from langchain_core.retrievers import BaseRetriever

class CustomRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str):
        # 自定义检索逻辑
        results = vectorstore.similarity_search(query, k=5)

        # 添加后处理
        filtered = [r for r in results if self._is_relevant(r)]
        return filtered[:3]

    def _is_relevant(self, doc):
        # 自定义相关性判断
        return len(doc.page_content) > 50

# 3. 组合检索器
from langchain.retrievers import MergerRetriever

retriever1 = vectorstore.as_retriever(search_kwargs={"k": 3})
retriever2 = another_vectorstore.as_retriever(search_kwargs={"k": 3})

# 合并多个检索器
merged_retriever = MergerRetriever(
    retrievers=[retriever1, retriever2]
)
```

### FAISS 向量存储

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 1. 创建 FAISS 索引
embeddings = OpenAIEmbeddings()

documents = [
    Document(page_content="机器学习使用数据训练模型"),
    Document(page_content="深度学习是机器学习的子集"),
    Document(page_content="神经网络模仿人脑结构"),
]

# 2. 创建 FAISS 向量存储
vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embeddings,
)

# 3. 保存和加载
vectorstore.save_local("faiss_index")

# 加载
loaded_vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

# 4. 搜索
query = "什么是神经网络？"
results = vectorstore.similarity_search(query, k=2)

# 5. 添加新文档
new_docs = [Document(page_content="Transformer 是一种神经网络架构")]
vectorstore.add_documents(new_docs)

# 6. 删除文档
vectorstore.delete(ids=["doc_id_1", "doc_id_2"])
```

---

## 嵌入模型

### 嵌入模型选择

| 提供商 | 模型 | 维度 | 特点 |
|--------|------|------|------|
| **OpenAI** | text-embedding-3-small | 1536 | 快速、性价比高 |
| **OpenAI** | text-embedding-3-large | 3072 | 最高质量 |
| **Cohere** | embed-english-v3.0 | 1024 | 英文优化 |
| **HuggingFace** | all-mpnet-base-v2 | 768 | 开源免费 |

### 基本用法

```python
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# OpenAI
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    # dimensions=512,  # 可选：降维
)

# Cohere
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
)

# HuggingFace
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},  # 或 'cuda'
    encode_kwargs={'normalize_embeddings': True},
)

# 嵌入文本
text = "这是一段示例文本"
vector = embeddings.embed_query(text)

print(f"向量维度: {len(vector)}")
print(f"前 5 个值: {vector[:5]}")

# 批量嵌入
texts = ["文本 1", "文本 2", "文本 3"]
vectors = embeddings.embed_documents(texts)

print(f"嵌入数量: {len(vectors)}")
```

### 嵌入缓存

避免重复计算嵌入，提高性能：

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings

# 1. 创建底层嵌入模型
underlying_embeddings = OpenAIEmbeddings()

# 2. 创建缓存存储
store = LocalFileStore("./embeddings_cache")

# 3. 创建带缓存的嵌入
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=underlying_embeddings,
    document_embedding_store=store,
    query_embedding_store=store,  # 也可缓存查询
    namespace=underlying_embeddings.model,  # 避免模型间冲突
)

# 4. 使用
# 第一次调用：计算并缓存
vector1 = cached_embeddings.embed_query("示例文本")

# 第二次调用：从缓存读取
vector2 = cached_embeddings.embed_query("示例文本")

# 两个向量相同
assert vector1 == vector2
```

---

## 长期记忆

### Memory vs Retrieval

```
┌─────────────────────────────────────────────────────────────┐
│              Memory vs Retrieval 对比                         │
│                                                             │
│  Retrieval (检索)                                            │
│  ├─ 目的：获取外部知识                                       │
│  ├─ 数据源：文档库、知识库                                   │
│  ├─ 访问方式：语义搜索、关键词匹配                            │
│  └─ 持久性：永久存储                                         │
│                                                             │
│  Memory (记忆)                                               │
│  ├─ 目的：记住用户信息和交互历史                             │
│  ├─ 数据源：对话、用户行为                                   │
│  ├─ 访问方式：namespace + key                               │
│  └─ 持久性：跨会话保持                                       │
└─────────────────────────────────────────────────────────────┘
```

### Store 结构

LangGraph Store 使用 **namespace** 和 **key** 组织记忆：

```
Store 结构
│
├─ Namespace (类似文件夹)
│  ├─ ("users", "user_123")        # 用户特定记忆
│  ├─ ("preferences", "user_123")  # 用户偏好
│  └─ ("conversations", "thread_456")  # 对话记忆
│
├─ Key (类似文件名)
│  ├─ "profile"                    # 用户资料
│  ├─ "settings"                   # 设置
│  └─ "summary_2024-01-15"         # 对话摘要
│
└─ Value (JSON 数据)
   ├─ 基本类型
   ├─ 列表
   └─ 嵌套对象
```

### 基本用法

```python
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from dataclasses import dataclass

# 1. 创建 Store
store = InMemoryStore()

# 生产环境使用持久化 Store
# from langgraph.store.postgres import AsyncPostgresStore
# store = AsyncPostgresStore(conn_string="postgresql://...")

# 2. 定义 Context
@dataclass
class Context:
    user_id: str

# 3. 创建工具读取记忆
@tool
def get_user_preferences(runtime: ToolRuntime[Context]) -> str:
    """获取用户偏好设置"""
    # 从 runtime 访问 store
    user_preferences = runtime.store.get(
        ("preferences", runtime.context.user_id),
        "settings"
    )

    if user_preferences:
        return str(user_preferences.value)
    return "未找到用户偏好"

# 4. 创建工具写入记忆
from typing_extensions import TypedDict

class UserPreferences(TypedDict):
    language: str
    theme: str
    notifications: bool

@tool
def save_user_preferences(
    preferences: UserPreferences,
    runtime: ToolRuntime[Context]
) -> str:
    """保存用户偏好设置"""
    runtime.store.put(
        ("preferences", runtime.context.user_id),
        "settings",
        preferences
    )
    return "偏好设置已保存"

# 5. 创建带 Store 的 Agent
agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[get_user_preferences, save_user_preferences],
    store=store,  # 传入 store
    context_schema=Context,
)

# 6. 使用 Agent
result = agent.invoke(
    {"messages": [{
        "role": "user",
        "content": "把我的语言设置为中文"
    }]},
    context=Context(user_id="user_123")
)
```

### Store API

```python
from langgraph.store.memory import InMemoryStore
import json

store = InMemoryStore()

# 1. put - 存储数据
store.put(
    namespace=("users", "user_123"),
    key="profile",
    value={
        "name": "张三",
        "language": "zh-CN",
        "preferences": {
            "theme": "dark",
            "notifications": True
        }
    }
)

# 2. get - 获取单个数据
item = store.get(("users", "user_123"), "profile")
if item:
    print(f"值: {item.value}")
    print(f"创建时间: {item.created_at}")
    print(f"更新时间: {item.updated_at}")

# 3. search - 搜索数据
# 简单搜索
items = store.search(("users", "user_123"))

# 带过滤器搜索
items = store.search(
    ("users", "user_123"),
    filter={"language": "zh-CN"},
    limit=10
)

# 语义搜索（需要配置嵌入）
def embed(texts: list[str]) -> list[list[float]]:
    # 实现嵌入函数
    return [[0.1, 0.2] * len(texts)]

store_with_search = InMemoryStore(
    index={"embed": embed, "dims": 2}
)

items = store_with_search.search(
    ("users",),
    query="用户偏好设置"
)

# 4. delete - 删除数据
store.delete(("users", "user_123"), "profile")

# 批量删除
store.delete(("users", "user_123"))
```

---

## 短期记忆

### 短期记忆 vs 长期记忆

```
┌─────────────────────────────────────────────────────────────┐
│            短期记忆 vs 长期记忆                               │
│                                                             │
│  短期记忆 (Short-term Memory)                                │
│  ├─ 范围：单个 thread (会话)                                 │
│  ├─ 存储：Checkpointer                                      │
│  ├─ 内容：消息历史、会话状态                                 │
│  ├─ 访问：同 thread_id 内共享                               │
│  └─ 示例：对话上下文                                        │
│                                                             │
│  长期记忆 (Long-term Memory)                                 │
│  ├─ 范围：跨 thread、跨会话                                  │
│  ├─ 存储：Store                                             │
│  ├─ 内容：用户偏好、知识库                                   │
│  ├─ 访问：任何 thread、任何时间                             │
│  └─ 示例：用户档案、设置                                    │
└─────────────────────────────────────────────────────────────┘
```

### 配置短期记忆

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

# 1. 创建 Checkpointer
checkpointer = InMemorySaver()

# 生产环境
# from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
# checkpointer = AsyncPostgresSaver.from_conn_string("postgresql://...")

# 2. 创建带 Checkpointer 的 Agent
agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[...],
    checkpointer=checkpointer,  # 添加 checkpointer
)

# 3. 使用 thread_id 维护会话
thread_id = "user_session_123"

# 第一次调用
result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫张三"}]},
    config={"configurable": {"thread_id": thread_id}}
)

# 第二次调用 - Agent 记住了用户名
result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
    config={"configurable": {"thread_id": thread_id}}
)

# Agent 会回答"你叫张三"
```

### 查看会话历史

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

# ... 运行 Agent 后 ...

# 1. 获取当前状态
config = {"configurable": {"thread_id": "thread_123"}}
state = checkpointer.get(config)

print("当前状态:", state.values)

# 2. 列出所有检查点
for checkpoint in checkpointer.list(config):
    print(f"检查点 ID: {checkpoint.config['checkpoint_id']}")
    print(f"步骤: {checkpoint.metadata.get('step')}")
    print(f"时间: {checkpoint.metadata.get('created_at')}")
    print("---")

# 3. 获取历史消息
state = checkpointer.get(config)
messages = state.values.get("messages", [])

for msg in messages:
    print(f"{msg.type}: {msg.content}")
```

### 消息摘要

对于长对话，可以使用 SummarizationMiddleware 自动摘要：

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model=ChatOpenAI(model="gpt-4o-mini"),  # 用于摘要的模型
            trigger=("tokens", 6000),  # 触发条件：超过 6000 tokens
            keep=("messages", 20),  # 保留最近 20 条消息
        )
    ],
    checkpointer=InMemorySaver(),
)
```

---

## Store API

### 完整 API 参考

```python
from langgraph.store.memory import InMemoryStore
from typing import Any

store = InMemoryStore()

# ===== put =====
# 存储数据到指定 namespace 和 key
store.put(
    namespace: tuple[str, ...],  # 命名空间
    key: str,                    # 键
    value: dict | list | str | int | float | bool | None,  # 值
)

# ===== get =====
# 获取单个值
item = store.get(
    namespace: tuple[str, ...],
    key: str,
)
# 返回 StoreItem 对象
# - item.value: 存储的值
# - item.key: 键
# - item.namespace: 命名空间
# - item.created_at: 创建时间
# - item.updated_at: 更新时间

# ===== search =====
# 搜索匹配的项
items = store.search(
    namespace: tuple[str, ...],
    filter: dict[str, Any] | None = None,  # 过滤条件
    query: str | None = None,  # 语义搜索查询
    limit: int = 10,  # 返回数量限制
)
# 返回 StoreItem 列表

# ===== delete =====
# 删除单个项
store.delete(
    namespace: tuple[str, ...],
    key: str,
)

# 删除整个命名空间
store.delete(namespace: tuple[str, ...])
```

### Namespace 设计模式

```python
# 1. 用户中心设计
user_id = "user_123"

# 用户资料
store.put(("users", user_id), "profile", {...})
store.put(("users", user_id), "settings", {...})

# 用户对话摘要
store.put(("users", user_id, "summaries"), "2024-01-15", {...})

# 2. 应用中心设计
org_id = "org_abc"

# 组织设置
store.put(("orgs", org_id), "settings", {...})
store.put(("orgs", org_id, "members"), user_id, {...})

# 3. 层级设计
# ("memory", user_id, "conversations", date, key)
store.put(
    ("memory", "user_123", "conversations", "2024-01-15"),
    "summary",
    {...}
)

# 4. 类型中心设计
# 按数据类型组织，便于全局搜索
store.put(("profiles",), "user_123", {...})
store.put(("preferences",), "user_123", {...})
store.put(("history",), "user_123_2024-01-15", {...})
```

### 语义搜索配置

```python
from langgraph.store.memory import InMemoryStore
from langchain_openai import OpenAIEmbeddings

# 1. 创建嵌入函数
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def embed(texts: list[str]) -> list[list[float]]:
    return embeddings.embed_documents(texts)

# 2. 创建带索引的 Store
store = InMemoryStore(
    index={
        "embed": embed,
        "dims": 1536,  # text-embedding-3-small 的维度
    }
)

# 3. 存储数据（自动嵌入）
store.put(
    ("memories",),
    "memory_1",
    {
        "content": "用户喜欢简洁直接的回答风格",
        "category": "preference"
    }
)

# 4. 语义搜索
results = store.search(
    ("memories",),
    query="用户的沟通偏好",
    limit=5
)

for item in results:
    print(f"相关度: {item.score if hasattr(item, 'score') else 'N/A'}")
    print(f"内容: {item.value}")
```

---

## 内存管理

### 内存类型选择

```
┌─────────────────────────────────────────────────────────────┐
│              选择合适的内存类型                               │
│                                                             │
│  需求                                                     │
│    │                                                       │
│    ├─→ 对话上下文？ ──→ 短期记忆 (Checkpointer)             │
│    │                                                       │
│    ├─→ 用户偏好？ ──→ 长期记忆 (Store + namespace)          │
│    │                                                       │
│    ├─→ 知识库？ ──→ 检索 (VectorStore + Retriever)         │
│    │                                                       │
│    └─→ 临时状态？ ──→ Agent State                          │
│                                                             │
│  组合使用：                                                 │
│  ├─ 短期记忆：消息历史                                     │
│  ├─ 长期记忆：用户档案                                     │
│  └─ 检索：知识库查询                                       │
└─────────────────────────────────────────────────────────────┘
```

### 混合内存策略

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from dataclasses import dataclass
from typing_extensions import TypedDict

# 1. 三层内存架构
# - 短期：Checkpointer（消息历史）
# - 长期：Store（用户偏好）
# - 检索：VectorStore（知识库）

@dataclass
class Context:
    user_id: str

# 2. 长期记忆工具
@tool
def get_user_memory(runtime: ToolRuntime[Context]) -> str:
    """获取长期记忆中的用户信息"""
    memory = runtime.store.get(
        ("users", runtime.context.user_id),
        "memory"
    )
    if memory:
        # 构建用户画像
        return f"用户档案：{memory.value}"
    return "暂无用户记录"

@tool
def save_user_memory(
    key: str,
    value: str,
    runtime: ToolRuntime[Context]
) -> str:
    """保存信息到长期记忆"""
    # 获取现有记忆
    memory = runtime.store.get(
        ("users", runtime.context.user_id),
        "memory"
    )

    if memory:
        memory.value[key] = value
        runtime.store.put(
            ("users", runtime.context.user_id),
            "memory",
            memory.value
        )
    else:
        runtime.store.put(
            ("users", runtime.context.user_id),
            "memory",
            {key: value}
        )

    return f"已保存：{key} = {value}"

# 3. 检索工具
@tool
def search_knowledge(query: str) -> str:
    """从知识库检索信息"""
    results = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in results])

# 4. 创建 Agent（三层内存）
agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[get_user_memory, save_user_memory, search_knowledge],
    checkpointer=InMemorySaver(),  # 短期记忆
    store=InMemoryStore(),          # 长期记忆
    context_schema=Context,
    system_prompt="""你是一个智能助手，具有以下能力：

1. 使用 get_user_memory 查看用户档案和偏好
2. 使用 save_user_memory 保存重要信息
3. 使用 search_knowledge 查询知识库

记住用户的偏好和重要信息，提供个性化服务。
""",
)

# 5. 使用
thread_id = "session_123"
user_id = "user_456"

# 第一轮对话：保存用户信息
agent.invoke(
    {"messages": [{"role": "user", "content": "我是张三，喜欢 Python 编程"}]},
    config={"configurable": {"thread_id": thread_id}},
    context=Context(user_id=user_id)
)

# 第二轮对话：使用用户偏好（短期记忆）
agent.invoke(
    {"messages": [{"role": "user", "content": "给我推荐一些学习资源"}]},
    config={"configurable": {"thread_id": thread_id}},
    context=Context(user_id=user_id)
)

# 新会话：使用长期记忆
new_thread_id = "session_456"
agent.invoke(
    {"messages": [{"role": "user", "content": "我喜欢什么编程语言？"}]},
    config={"configurable": {"thread_id": new_thread_id}},
    context=Context(user_id=user_id)
)
# Agent 会从长期记忆中知道用户喜欢 Python
```

### 内存清理策略

```python
# 1. 短期记忆清理
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

# 删除整个会话
config = {"configurable": {"thread_id": "old_thread"}}
checkpointer.delete(config)

# 列出并清理旧会话
for checkpoint in checkpointer.list({"configurable": {}}):
    thread_id = checkpoint.config.get("configurable", {}).get("thread_id", "")
    if thread_id.startswith("temp_"):
        checkpointer.delete(checkpoint.config)

# 2. 长期记忆清理
from datetime import datetime, timedelta
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# 删除过期数据
cutoff_date = datetime.now() - timedelta(days=30)

for item in store.search(("temp",)):
    if item.created_at < cutoff_date:
        store.delete(item.namespace, item.key)

# 3. 检索索引更新
# 对于向量存储，定期更新索引
vectorstore.add_documents(new_documents)
```

---

## 实战案例

### 案例 1：智能客服系统

```python
from langchain.agents import create_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from dataclasses import dataclass
import asyncio

@dataclass
class CustomerContext:
    customer_id: str
    session_id: str

class CustomerServiceAgent:
    def __init__(self):
        # 1. 初始化存储
        self.checkpointer = AsyncPostgresSaver.from_conn_string(
            "postgresql://user:pass@localhost/customer_db"
        )
        self.store = AsyncPostgresStore.from_conn_string(
            "postgresql://user:pass@localhost/customer_db"
        )
        self.vectorstore = self._init_knowledge_base()

        # 2. 创建工具
        self.tools = self._init_tools()

        # 3. 创建 Agent
        self.agent = create_agent(
            model=ChatOpenAI(model="gpt-4o"),
            tools=self.tools,
            checkpointer=self.checkpointer,
            store=self.store,
            context_schema=CustomerContext,
            system_prompt="""你是专业的客服助手。

能力：
1. 查询客户历史记录和偏好
2. 搜索产品知识和常见问题
3. 记录客户问题和反馈
4. 提供个性化服务

始终保持友好、专业的态度。
""",
        )

    def _init_knowledge_base(self):
        """初始化知识库"""
        from langchain_openai import OpenAIEmbeddings

        # 产品文档
        product_docs = [
            "产品 A 是一款企业级软件，定价 $99/月",
            "产品 B 面向中小企业，定价 $49/月",
            # ... 更多文档
        ]

        embeddings = OpenAIEmbeddings()
        vectorstore = InMemoryVectorStore.from_texts(
            texts=product_docs,
            embedding=embeddings,
        )

        return vectorstore

    def _init_tools(self):
        """初始化工具"""

        @tool
        def get_customer_info(runtime: ToolRuntime[CustomerContext]) -> str:
            """获取客户信息"""
            customer_id = runtime.context.customer_id

            # 从长期记忆获取
            info = runtime.store.get(
                ("customers", customer_id),
                "profile"
            )

            if info:
                return f"客户信息：{info.value}"

            # 从数据库查询（模拟）
            return "客户：新用户"

        @tool
        def search_faq(query: str) -> str:
            """搜索常见问题"""
            results = self.vectorstore.similarity_search(query, k=3)
            return "\n\n".join([doc.page_content for doc in results])

        @tool
        def save_interaction(
            issue: str,
            resolution: str,
            runtime: ToolRuntime[CustomerContext]
        ) -> str:
            """保存交互记录"""
            customer_id = runtime.context.customer_id
            session_id = runtime.context.session_id

            runtime.store.put(
                ("customers", customer_id, "interactions"),
                session_id,
                {
                    "issue": issue,
                    "resolution": resolution,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            return "已保存交互记录"

        @tool
        def get_purchase_history(
            runtime: ToolRuntime[CustomerContext]
        ) -> str:
            """获取购买历史"""
            customer_id = runtime.context.customer_id

            history = runtime.store.get(
                ("customers", customer_id),
                "purchases"
            )

            if history:
                return f"购买历史：{history.value}"

            return "暂无购买记录"

        return [
            get_customer_info,
            search_faq,
            save_interaction,
            get_purchase_history,
        ]

    async def handle_message(
        self,
        message: str,
        customer_id: str,
        session_id: str
    ):
        """处理客户消息"""
        result = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": session_id}},
            context=CustomerContext(
                customer_id=customer_id,
                session_id=session_id
            )
        )

        return result["messages"][-1].content

# 使用
async def main():
    agent = CustomerServiceAgent()

    response = await agent.handle_message(
        message="你们的产品有哪些？",
        customer_id="cust_001",
        session_id="sess_001"
    )

    print(response)

asyncio.run(main())
```

### 案例 2：个性化学习助手

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from dataclasses import dataclass
from typing_extensions import TypedDict
from datetime import datetime

@dataclass
class LearningContext:
    user_id: str

class LearningAssistant:
    def __init__(self):
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()
        self.agent = self._create_agent()

    def _create_agent(self):
        """创建学习助手 Agent"""

        @tool
        def get_learning_profile(runtime: ToolRuntime[LearningContext]) -> str:
            """获取学习档案"""
            profile = runtime.store.get(
                ("learners", runtime.context.user_id),
                "profile"
            )

            if profile:
                return f"学习档案：{profile.value}"

            return "新学习者，还未建立学习档案"

        @tool
        def update_progress(
            topic: str,
            progress: int,  # 0-100
            runtime: ToolRuntime[LearningContext]
        ) -> str:
            """更新学习进度"""
            profile = runtime.store.get(
                ("learners", runtime.context.user_id),
                "profile"
            )

            if profile:
                profile.value["progress"][topic] = progress
                runtime.store.put(
                    ("learners", runtime.context.user_id),
                    "profile",
                    profile.value
                )
            else:
                runtime.store.put(
                    ("learners", runtime.context.user_id),
                    "profile",
                    {
                        "progress": {topic: progress},
                        "goals": [],
                        "learning_style": "",
                    }
                )

            return f"已更新 {topic} 进度：{progress}%"

        @tool
        def set_learning_goal(
            goal: str,
            deadline: str,
            runtime: ToolRuntime[LearningContext]
        ) -> str:
            """设置学习目标"""
            profile = runtime.store.get(
                ("learners", runtime.context.user_id),
                "profile"
            )

            new_goal = {
                "goal": goal,
                "deadline": deadline,
                "created_at": datetime.now().isoformat(),
            }

            if profile:
                if "goals" not in profile.value:
                    profile.value["goals"] = []
                profile.value["goals"].append(new_goal)
                runtime.store.put(
                    ("learners", runtime.context.user_id),
                    "profile",
                    profile.value
                )
            else:
                runtime.store.put(
                    ("learners", runtime.context.user_id),
                    "profile",
                    {"goals": [new_goal]}
                )

            return f"已设置目标：{goal}"

        @tool
        def save_note(
            topic: str,
            note: str,
            runtime: ToolRuntime[LearningContext]
        ) -> str:
            """保存学习笔记"""
            timestamp = datetime.now().strftime("%Y-%m-%d")
            runtime.store.put(
                ("learners", runtime.context.user_id, "notes"),
                f"{topic}_{timestamp}",
                {
                    "topic": topic,
                    "note": note,
                    "timestamp": timestamp,
                }
            )

            return "笔记已保存"

        @tool
        def get_notes(topic: str, runtime: ToolRuntime[LearningContext]) -> str:
            """获取特定主题的笔记"""
            notes = runtime.store.search(
                ("learners", runtime.context.user_id, "notes"),
                filter={"topic": topic},
            )

            if notes:
                return "\n\n".join([
                    f"{n.value['timestamp']}: {n.value['note']}"
                    for n in notes
                ])

            return f"暂无关于 {topic} 的笔记"

        # 创建 Agent
        agent = create_agent(
            model=ChatOpenAI(model="gpt-4o"),
            tools=[
                get_learning_profile,
                update_progress,
                set_learning_goal,
                save_note,
                get_notes,
            ],
            checkpointer=self.checkpointer,
            store=self.store,
            context_schema=LearningContext,
            system_prompt="""你是个性化学习助手。

能力：
1. 跟踪学习进度和目标
2. 保存和检索学习笔记
3. 提供个性化学习建议
4. 根据学习风格调整教学方式

使用 save_note 记录重要知识点。
使用 update_progress 跟踪每个主题的掌握程度。
使用 set_learning_goal 设定明确的学习目标。
""",
        )

        return agent

    def chat(self, message: str, user_id: str, session_id: str):
        """处理学习对话"""
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": session_id}},
            context=LearningContext(user_id=user_id)
        )

        return result["messages"][-1].content

# 使用示例
assistant = LearningAssistant()

# 第一天：开始学习
response1 = assistant.chat(
    message="我想学习 Python 机器学习，计划用一个月时间掌握基础",
    user_id="learner_001",
    session_id="day1_session1"
)
print(response1)
# Agent 会设置目标并创建学习档案

# 记录学习笔记
response2 = assistant.chat(
    message="今天学了 numpy 和 pandas，帮我记一下重点",
    user_id="learner_001",
    session_id="day1_session2"
)
print(response2)

# 第二天：继续学习
response3 = assistant.chat(
    message="我昨天学了什么？今天应该学什么？",
    user_id="learner_001",
    session_id="day2_session1"
)
print(response3)
# Agent 会从长期记忆中获取昨天学习的内容
```

### 案例 3：带记忆的 RAG 系统

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from dataclasses import dataclass
from typing_extensions import TypedDict

@dataclass
class UserContext:
    user_id: str

class MemoryEnhancedRAG:
    def __init__(self):
        # 1. 初始化向量存储
        embeddings = OpenAIEmbeddings()
        self.vectorstore = InMemoryVectorStore(
            embedding=embeddings
        )

        # 2. 创建基础 RAG 链
        self.rag_chain = self._create_rag_chain()

        # 3. 创建带记忆的 Agent
        self.memory_agent = self._create_memory_agent()

    def _create_rag_chain(self):
        """创建基础 RAG 链"""
        prompt = ChatPromptTemplate.from_template("""
基于以下上下文和用户偏好回答问题：

用户偏好：
{user_preferences}

相关文档：
{context}

问题：{input}

答案：
""")

        llm = ChatOpenAI(model="gpt-4o")
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

        combine_docs_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt
        )

        return create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=combine_docs_chain
        )

    def _create_memory_agent(self):
        """创建记忆管理 Agent"""

        @tool
        def get_user_preferences(runtime: ToolRuntime[UserContext]) -> str:
            """获取用户偏好"""
            prefs = runtime.store.get(
                ("users", runtime.context.user_id),
                "preferences"
            )

            return str(prefs.value) if prefs else "暂无偏好设置"

        @tool
        def update_preference(
            key: str,
            value: str,
            runtime: ToolRuntime[UserContext]
        ) -> str:
            """更新用户偏好"""
            prefs = runtime.store.get(
                ("users", runtime.context.user_id),
                "preferences"
            )

            if prefs:
                prefs.value[key] = value
                runtime.store.put(
                    ("users", runtime.context.user_id),
                    "preferences",
                    prefs.value
                )
            else:
                runtime.store.put(
                    ("users", runtime.context.user_id),
                    "preferences",
                    {key: value}
                )

            return f"已更新偏好：{key} = {value}"

        @tool
        def record_feedback(
            query: str,
            answer: str,
            rating: int,  # 1-5
            runtime: ToolRuntime[UserContext]
        ) -> str:
            """记录反馈"""
            runtime.store.put(
                ("users", runtime.context.user_id, "feedback"),
                f"{datetime.now().isoformat()}",
                {
                    "query": query,
                    "answer": answer,
                    "rating": rating,
                }
            )

            # 更新偏好（低评分答案的风格应避免）
            if rating <= 2:
                prefs = runtime.store.get(
                    ("users", runtime.context.user_id),
                    "preferences"
                )
                if prefs:
                    if "avoid_styles" not in prefs.value:
                        prefs.value["avoid_styles"] = []
                    prefs.value["avoid_styles"].append("verbose")
                    runtime.store.put(
                        ("users", runtime.context.user_id),
                        "preferences",
                        prefs.value
                    )

            return "反馈已记录"

        agent = create_agent(
            model=ChatOpenAI(model="gpt-4o-mini"),
            tools=[
                get_user_preferences,
                update_preference,
                record_feedback,
            ],
            checkpointer=InMemorySaver(),
            store=InMemoryStore(),
            context_schema=UserContext,
        )

        return agent

    def add_documents(self, documents: list[str]):
        """添加文档到知识库"""
        self.vectorstore.add_texts(documents)

    def query(
        self,
        question: str,
        user_id: str,
    ) -> str:
        """带用户偏好的查询"""
        # 1. 获取用户偏好
        from langgraph.store.memory import InMemoryStore
        store = InMemoryStore()
        prefs = store.get(("users", user_id), "preferences")

        user_preferences = str(prefs.value) if prefs else "标准回答风格"

        # 2. 执行 RAG
        result = self.rag_chain.invoke({
            "input": question,
            "user_preferences": user_preferences,
        })

        return result["answer"]

    def chat(
        self,
        message: str,
        user_id: str,
        session_id: str
    ) -> str:
        """完整对话（包括偏好管理）"""
        # 先尝试查询
        if "?" in message or "如何" in message or "什么" in message:
            answer = self.query(message, user_id)
            return answer

        # 其他消息由记忆 Agent 处理
        result = self.memory_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": session_id}},
            context=UserContext(user_id=user_id)
        )

        return result["messages"][-1].content

# 使用
rag_system = MemoryEnhancedRAG()

# 添加文档
rag_system.add_documents([
    "Python 是一种高级编程语言",
    "机器学习是 AI 的一个分支",
    "TensorFlow 是 Google 开发的机器学习框架",
    "PyTorch 是 Meta 开发的深度学习框架",
])

# 对话
response1 = rag_system.chat(
    message="我喜欢简洁的回答",
    user_id="user_001",
    session_id="sess_001"
)

response2 = rag_system.chat(
    message="什么是 TensorFlow？",
    user_id="user_001",
    session_id="sess_001"
)
# 回答会采用简洁风格

response3 = rag_system.chat(
    message="把回答长度改成详细版",
    user_id="user_001",
    session_id="sess_001"
)

response4 = rag_system.chat(
    message="PyTorch 和 TensorFlow 有什么区别？",
    user_id="user_001",
    session_id="sess_001"
)
# 回答会变得更详细
```

---

## 最佳实践

### 1. Namespace 组织

```python
# ✅ 好的做法：清晰的层次结构
("users", user_id, "profile")
("users", user_id, "preferences")
("users", user_id, "history", date)
("orgs", org_id, "settings")
("orgs", org_id, "members", member_id)

# ❌ 不好的做法：扁平结构
("user_profile_user_123",)
("user_preferences_user_123",)
```

### 2. 数据大小控制

```python
# ✅ 好的做法：分片存储
# 单个记忆不要太大
for i, chunk in enumerate(large_text_chunks):
    store.put(
        ("users", user_id, "documents"),
        f"doc_{doc_id}_chunk_{i}",
        {"content": chunk, "index": i}
    )

# ❌ 不好的做法：存储大文档
store.put(
    ("users", user_id),
    "large_doc",
    {"content": very_large_string}  # 可能超过限制
)
```

### 3. 向量存储选择

```python
# 开发环境
from langchain_core.vectorstores import InMemoryVectorStore
vectorstore = InMemoryVectorStore(embedding=embeddings)

# 生产环境 - 本地
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(docs, embeddings)

# 生产环境 - 云端
from langchain_pinecone import PineconeVectorStore
vectorstore = PineconeVectorStore.from_documents(
    docs,
    embeddings,
    index_name="my-index"
)
```

### 4. 检索优化

```python
# ✅ 好的做法：混合检索
def hybrid_retrieval(query: str):
    # 语义检索
    semantic_results = vectorstore.similarity_search(query, k=5)

    # 关键词检索
    keyword_results = keyword_search(query, k=5)

    # 融合结果
    return merge_and_rerank(semantic_results, keyword_results)

# ✅ 好的做法：重排序
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

### 5. 内存同步

```python
# ✅ 好的做法：短期记忆转长期
async def summarize_and_store(
    checkpointer,
    store,
    thread_id,
    user_id
):
    """将会话摘要存储到长期记忆"""
    # 获取会话历史
    state = checkpointer.get({"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])

    # 生成摘要
    summary = await generate_summary(messages)

    # 存储到长期记忆
    timestamp = datetime.now().strftime("%Y-%m-%d")
    store.put(
        ("users", user_id, "summaries"),
        timestamp,
        {"summary": summary, "message_count": len(messages)}
    )

    # 清理短期记忆（可选）
    # checkpointer.delete({"configurable": {"thread_id": thread_id}})
```

### 6. 错误处理

```python
# ✅ 好的做法：优雅降级
@tool
def get_user_memory(runtime: ToolRuntime[Context]) -> str:
    """获取用户记忆（带容错）"""
    try:
        memory = runtime.store.get(
            ("users", runtime.context.user_id),
            "memory"
        )
        return str(memory.value) if memory else "无记忆"
    except Exception as e:
        logger.error(f"获取记忆失败: {e}")
        return "记忆暂时不可用"

@tool
def search_knowledge(query: str) -> str:
    """搜索知识库（带容错）"""
    try:
        results = retriever.invoke(query)
        return "\n".join([doc.page_content for doc in results])
    except Exception as e:
        logger.error(f"检索失败: {e}")
        return "检索服务暂时不可用"
```

---

## 快速参考

### RAG 基本模式

```python
# 2-Step RAG
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=create_stuff_documents_chain(llm, prompt)
)

# Agentic RAG
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=[search_tool],
    system_prompt="使用 search_tool 检索信息"
)
```

### Store API

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# 存储
store.put(("ns",), "key", {"data": "value"})

# 获取
item = store.get(("ns",), "key")

# 搜索
items = store.search(("ns",), filter={"data": "value"})

# 删除
store.delete(("ns",), "key")
```

### 向量存储

```python
from langchain_core.vectorstores import InMemoryVectorStore

# 创建
vectorstore = InMemoryVectorStore.from_texts(texts, embeddings)

# 搜索
results = vectorstore.similarity_search(query, k=4)

# 作为检索器
retriever = vectorstore.as_retriever()
```

### Checkpointer

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

# 使用
agent = create_agent(
    model=llm,
    tools=[...],
    checkpointer=checkpointer,
)

# 调用
result = agent.invoke(
    {"messages": [...]},
    config={"configurable": {"thread_id": "thread_123"}}
)
```

---

## 总结

**Retrieval & Memory** 是构建智能 AI 应用的两大核心能力：

### Retrieval（检索）

1. **核心组件**
   - Document Loaders：数据加载
   - Text Splitters：文档分割
   - Embedding Models：向量嵌入
   - Vector Stores：向量存储
   - Retrievers：检索接口

2. **RAG 架构**
   - 2-Step RAG：简单、可控
   - Agentic RAG：灵活、智能
   - Hybrid RAG：平衡、质量高

### Memory（记忆）

1. **记忆类型**
   - 短期记忆：会话级，使用 Checkpointer
   - 长期记忆：跨会话，使用 Store

2. **Store 结构**
   - Namespace：组织记忆
   - Key：标识记忆
   - Value：存储数据

### 最佳实践

- 合理组织 Namespace
- 控制单个记忆大小
- 选择合适的向量存储
- 使用混合检索提高质量
- 定期将会话转为长期记忆
- 做好错误处理和降级

通过合理使用检索和记忆，你可以构建出既能访问外部知识、又能记住用户偏好的智能应用！
