# Hello Agents 第八章：为 Agent 添加记忆与检索能力(详细版)

> **本章核心思想**：让 Agent 能够"记住"对话历史、检索外部知识,从"健忘症患者"变成"博学多识"的智能助手。

---

## 📖 目录

- [1. 为什么 Agent 需要记忆?](#1-为什么-agent-需要记忆)
- [2. Memory 系统设计](#2-memory-系统设计)
- [3. RAG 检索增强生成](#3-rag-检索增强生成)
- [4. 向量数据库实战](#4-向量数据库实战)
- [5. Memory 作为工具集成](#5-memory-作为工具集成)
- [6. 本章总结](#6-本章总结)

---

## 1. 为什么 Agent 需要记忆?

### 🤔 没有记忆的 Agent 有什么问题?

想象一下,你每天和朋友聊天,但他总是忘记你们之前说过的话:

```
你:我叫小明,今年25岁
Agent:你好!
你:我刚才说了什么?
Agent:对不起,我不知道你在说什么
```

**问题一:无法维持连贯对话** 😵
- 每次对话都是"新朋友"
- 无法基于上下文回答
- 用户体验很差

**问题二:无法学习用户偏好** 🔄
- 不知道用户喜好
- 重复询问相同信息
- 无法个性化服务

**问题三:知识局限** 📦
- 只能回答训练数据中的内容
- 无法获取最新信息
- 专业领域知识不足

### 💡 记忆系统的价值

```
对话记忆(Short-term Memory)
    ↓
能记住最近的对话
    ↓
知识记忆(Long-term Memory)
    ↓
能检索相关知识
    ↓
个性化记忆(User Profile)
    ↓
能记住用户偏好
```

> 💭 **类比**:就像人的记忆系统,有短期记忆(对话上下文)和长期记忆(知识库)

---

## 2. Memory 系统设计

### 2.1 记忆的三种类型

#### 📝 短期记忆(Short-term Memory)

**定义**:当前对话会话的上下文

```python
class ShortTermMemory:
    """短期记忆:对话历史"""

    def __init__(self, max_messages=20):
        self.messages = []  # 存储消息
        self.max_messages = max_messages

    def add_message(self, role, content):
        """添加消息"""
        message = Message(role=role, content=content)
        self.messages.append(message)

        # 保持最大长度
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self):
        """获取对话历史"""
        return self.messages

    def clear(self):
        """清空记忆"""
        self.messages = []
```

**使用示例**:

```python
memory = ShortTermMemory(max_messages=10)

# 添加对话
memory.add_message("user", "我叫小明")
memory.add_message("assistant", "你好,小明!")
memory.add_message("user", "我刚才叫什么名字?")

# 获取上下文
messages = memory.get_messages()
# Agent 可以看到完整对话历史,知道用户叫小明
```

#### 🗄️ 长期记忆(Long-term Memory)

**定义**:持久化存储的知识和经验

```python
class LongTermMemory:
    """长期记忆:知识库"""

    def __init__(self, storage_path="memory.db"):
        self.storage_path = storage_path
        self.db = self._init_database()

    def save(self, key, value, metadata=None):
        """保存记忆"""
        memory = {
            "key": key,
            "value": value,
            "metadata": metadata or {},
            "timestamp": datetime.now()
        }
        self.db.insert(memory)

    def recall(self, query, limit=5):
        """检索记忆"""
        # 基于关键词检索
        results = self.db.search(query, limit=limit)
        return results

    def forget(self, key):
        """删除记忆"""
        self.db.delete(key)
```

**使用场景**:

```python
ltm = LongTermMemory()

# 保存用户信息
ltm.save(
    key="user_profile_xiaoming",
    value={
        "name": "小明",
        "age": 25,
        "interests": ["编程", "阅读"]
    },
    metadata={"type": "user_profile"}
)

# 后续对话中检索
profile = ltm.recall("小明的兴趣")
# 返回:["编程", "阅读"]
```

#### 👤 个性化记忆(User Profile)

**定义**:用户特定的偏好和习惯

```python
class UserProfile:
    """用户画像记忆"""

    def __init__(self, user_id):
        self.user_id = user_id
        self.preferences = {}  # 偏好
        self.history = []      # 交互历史
        self.context = {}      # 上下文信息

    def update_preference(self, key, value):
        """更新偏好"""
        self.preferences[key] = value

    def add_interaction(self, interaction):
        """记录交互"""
        self.history.append({
            "timestamp": datetime.now(),
            "content": interaction
        })

    def get_summary(self):
        """获取用户摘要"""
        return {
            "user_id": self.user_id,
            "preferences": self.preferences,
            "total_interactions": len(self.history)
        }
```

### 2.2 记忆管理策略

#### 🗑️ 策略一:滑动窗口(Sliding Window)

**适用场景**:对话场景

```python
class SlidingWindowMemory(ShortTermMemory):
    """滑动窗口记忆"""

    def add_message(self, role, content):
        """添加消息,自动淘汰旧消息"""
        super().add_message(role, content)

        # 只保留最近的 N 条
        if len(self.messages) > self.max_messages:
            removed = self.messages.pop(0)
            print(f"淘汰旧消息: {removed.content[:20]}...")
```

**可视化**:

```
时间线:
[msg1] [msg2] [msg3] [msg4] [msg5] ... [msg20]
                                         ↑
                                      新消息来了
[msg2] [msg3] [msg4] [msg5] ... [msg20] [msg21]
 删除第一条,保持窗口大小 = 20
```

#### ⭐ 策略二:重要性采样(Importance Sampling)

**适用场景**:需要保留关键信息

```python
class ImportanceMemory:
    """基于重要性的记忆管理"""

    def __init__(self, max_messages=20):
        self.messages = []
        self.max_messages = max_messages

    def calculate_importance(self, message):
        """计算消息重要性(0-1)"""
        importance = 0.5  # 基础分数

        # 因素1:长度(长消息可能更重要)
        if len(message.content) > 100:
            importance += 0.1

        # 因素2:包含关键词
        keywords = ["重要", "记住", "关键"]
        if any(kw in message.content for kw in keywords):
            importance += 0.2

        # 因素3:角色(系统消息更重要)
        if message.role == "system":
            importance += 0.2

        return min(importance, 1.0)

    def add_message(self, role, content):
        """添加消息并计算重要性"""
        message = Message(role=role, content=content)
        importance = self.calculate_importance(message)

        self.messages.append({
            "message": message,
            "importance": importance
        })

        # 超过限制时,删除最不重要的
        if len(self.messages) > self.max_messages:
            self.messages.sort(key=lambda x: x["importance"], reverse=True)
            removed = self.messages.pop()
            print(f"删除低重要性消息: {removed['message'].content[:20]}...")
```

#### 📊 策略三:摘要压缩(Summarization)

**适用场景**:长对话历史

```python
class SummarizationMemory:
    """摘要压缩记忆"""

    def __init__(self, llm, compress_threshold=20):
        self.llm = llm
        self.messages = []
        self.compress_threshold = compress_threshold
        self.summary = None

    def add_message(self, role, content):
        """添加消息,达到阈值时压缩"""
        self.messages.append(Message(role=role, content=content))

        if len(self.messages) >= self.compress_threshold:
            self._compress()

    def _compress(self):
        """压缩历史为摘要"""
        # 构建压缩提示词
        history_text = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in self.messages
        ])

        prompt = f"""
        请将以下对话历史压缩为简洁的摘要(200字以内):

        {history_text}

        摘要:
        """

        # 调用 LLM 生成摘要
        self.summary = self.llm.generate(prompt)

        # 清空旧消息,只保留摘要
        self.messages = []
        print(f"✅ 压缩完成,摘要: {self.summary[:50]}...")

    def get_context(self):
        """获取上下文(摘要 + 最近消息)"""
        context = []

        # 添加摘要
        if self.summary:
            context.append(Message(
                role="system",
                content=f"[对话摘要] {self.summary}"
            ))

        # 添加最近消息
        context.extend(self.messages)

        return context
```

**使用效果**:

```
原始对话(100条消息,5000 tokens)
        ↓ 压缩
摘要(200字,150 tokens) + 最近10条消息(500 tokens)
        ↓
总计:650 tokens(节省 87%)
```

### 2.3 Memory 统一接口

#### 🎯 设计 Memory 基类

```python
from abc import ABC, abstractmethod

class BaseMemory(ABC):
    """Memory 统一接口"""

    @abstractmethod
    def add(self, message: Message):
        """添加记忆"""
        pass

    @abstractmethod
    def get(self, query: Optional[str] = None, limit: int = 10) -> List[Message]:
        """获取记忆"""
        pass

    @abstractmethod
    def clear(self):
        """清空记忆"""
        pass

    @abstractmethod
    def save(self, path: str):
        """持久化保存"""
        pass

    @abstractmethod
    def load(self, path: str):
        """加载记忆"""
        pass
```

#### 📝 多种 Memory 实现

```python
# 1. 简单列表记忆
class ListMemory(BaseMemory):
    """基于列表的内存记忆"""
    def __init__(self):
        self.messages = []

    def add(self, message):
        self.messages.append(message)

    def get(self, query=None, limit=10):
        return self.messages[-limit:]

# 2. 向量检索记忆
class VectorMemory(BaseMemory):
    """基于向量的语义检索记忆"""
    def __init__(self, embedding_model):
        self.embeddings = []
        self.messages = []
        self.embedding_model = embedding_model

    def add(self, message):
        embedding = self.embedding_model.encode(message.content)
        self.embeddings.append(embedding)
        self.messages.append(message)

    def get(self, query, limit=10):
        # 语义相似度检索
        query_embedding = self.embedding_model.encode(query)
        similarities = cosine_similarity(query_embedding, self.embeddings)
        top_indices = similarities.argsort()[-limit:]
        return [self.messages[i] for i in top_indices]

# 3. 数据库记忆
class DatabaseMemory(BaseMemory):
    """基于数据库的持久化记忆"""
    def __init__(self, db_path="memory.db"):
        self.db = sqlite3.connect(db_path)
        self._init_table()

    def add(self, message):
        self.db.execute(
            "INSERT INTO messages (role, content, timestamp) VALUES (?, ?, ?)",
            (message.role, message.content, datetime.now())
        )
        self.db.commit()

    def get(self, query=None, limit=10):
        if query:
            # 关键词搜索
            results = self.db.execute(
                "SELECT * FROM messages WHERE content LIKE ? LIMIT ?",
                (f"%{query}%", limit)
            )
        else:
            # 获取最新
            results = self.db.execute(
                "SELECT * FROM messages ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
        return [Message(role=r[0], content=r[1]) for r in results]
```

---

## 3. RAG 检索增强生成

### 3.1 什么是 RAG?

#### 🎯 核心思想

```
用户问题
    ↓
检索相关文档(Retrieval)
    ↓
增强 LLM 输入(Augmentation)
    ↓
生成回答(Generation)
```

**传统方式 vs RAG**:

```python
# ❌ 传统方式:只依赖模型知识
response = llm.generate("什么是量子计算?")
# 可能回答不准确或过时

# ✅ RAG 方式:检索 + 生成
docs = retrieve("量子计算")  # 检索相关文档
prompt = f"""
基于以下文档回答问题:

{docs}

问题:什么是量子计算?
答案:
"""
response = llm.generate(prompt)
# 回答更准确、更新
```

### 3.2 RAG 系统架构

#### 🏗️ 完整流程

```
┌─────────────────┐
│  文档库        │
│  (PDFs/Docs)   │
└────────┬────────┘
         │
         ↓ 分块(Chunking)
┌─────────────────┐
│  文本块        │
│  (Chunks)      │
└────────┬────────┘
         │
         ↓ 向量化(Embedding)
┌─────────────────┐
│  向量数据库    │
│  (Vector DB)   │
└────────┬────────┘
         │
         ↓ 检索(Retrieval)
┌─────────────────┐
│  相关文档      │
│  (Top-K Docs)  │
└────────┬────────┘
         │
         ↓ 构建提示词
┌─────────────────┐
│  增强输入      │
│  (Prompt)      │
└────────┬────────┘
         │
         ↓ LLM 生成
┌─────────────────┐
│  最终答案      │
│  (Answer)      │
└─────────────────┘
```

#### 📝 简单 RAG 实现

```python
class SimpleRAG:
    """简单的 RAG 系统"""

    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.embedding_model = embedding_model
        self.documents = []       # 原始文档
        self.chunks = []          # 文本块
        self.embeddings = []      # 向量

    def add_documents(self, documents):
        """添加文档"""
        for doc in documents:
            # 1. 分块
            chunks = self._split_document(doc)

            # 2. 向量化
            for chunk in chunks:
                embedding = self.embedding_model.encode(chunk)
                self.chunks.append(chunk)
                self.embeddings.append(embedding)

        print(f"✅ 已索引 {len(self.chunks)} 个文本块")

    def _split_document(self, document, chunk_size=500):
        """将文档分割成小块"""
        chunks = []
        for i in range(0, len(document), chunk_size):
            chunk = document[i:i + chunk_size]
            chunks.append(chunk)
        return chunks

    def retrieve(self, query, top_k=3):
        """检索相关文档"""
        # 1. 查询向量化
        query_embedding = self.embedding_model.encode(query)

        # 2. 计算相似度
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((sim, i))

        # 3. 返回 Top-K
        similarities.sort(reverse=True)
        top_chunks = [self.chunks[i] for _, i in similarities[:top_k]]

        return top_chunks

    def query(self, question):
        """RAG 查询"""
        # 1. 检索相关文档
        relevant_docs = self.retrieve(question, top_k=3)

        # 2. 构建提示词
        context = "\n\n".join(relevant_docs)
        prompt = f"""
        基于以下文档回答问题。如果文档中没有相关信息,请说"我不知道"。

        文档:
        {context}

        问题:{question}

        答案:
        """

        # 3. 生成回答
        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "sources": relevant_docs[:2]  # 返回引用来源
        }
```

#### 💡 使用示例

```python
from hello_agents import HelloAgentsLLM
from sentence_transformers import SentenceTransformer

# 1. 初始化
llm = HelloAgentsLLM()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
rag = SimpleRAG(llm, embedding_model)

# 2. 添加文档
documents = [
    "量子计算是利用量子力学现象进行信息处理的计算方式...",
    "Python 是一种高级编程语言,由 Guido van Rossum 创建...",
    "机器学习是人工智能的一个分支..."
]
rag.add_documents(documents)

# 3. 查询
result = rag.query("什么是量子计算?")
print("答案:", result["answer"])
print("来源:", result["sources"])
```

### 3.3 RAG 优化技巧

#### ⚡ 优化一:改进分块策略

```python
class ImprovedChunker:
    """改进的文档分块器"""

    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap  # 重叠部分

    def split_with_overlap(self, text):
        """带重叠的分块"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            # 下一块的起点向后移动(chunk_size - overlap)
            start = start + self.chunk_size - self.overlap

        return chunks

    def split_by_sentence(self, text):
        """按句子边界分块"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk)
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk)

        return chunks
```

**为什么要重叠?**

```
原始文本:
"...量子计算利用叠加原理。叠加原理是量子力学的核心..."

无重叠分块:
块1: "...量子计算利用叠加原理。"
块2: "叠加原理是量子力学的核心..."
❌ 上下文割裂

带重叠分块:
块1: "...量子计算利用叠加原理。叠加原理是..."
块2: "叠加原理。叠加原理是量子力学的核心..."
✅ 保留上下文
```

#### 🎯 优化二:混合检索

```python
class HybridRetriever:
    """混合检索:向量检索 + 关键词检索"""

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.vector_index = []  # 向量索引
        self.bm25_index = None  # BM25 索引
        self.documents = []

    def build_index(self, documents):
        """构建双重索引"""
        from rank_bm25 import BM25Okapi

        self.documents = documents

        # 1. 构建向量索引
        for doc in documents:
            embedding = self.embedding_model.encode(doc)
            self.vector_index.append(embedding)

        # 2. 构建 BM25 索引
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25_index = BM25Okapi(tokenized_docs)

    def retrieve(self, query, top_k=5):
        """混合检索"""
        # 1. 向量检索分数
        query_embedding = self.embedding_model.encode(query)
        vector_scores = [
            cosine_similarity(query_embedding, doc_emb)
            for doc_emb in self.vector_index
        ]

        # 2. BM25 检索分数
        tokenized_query = query.split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)

        # 3. 归一化分数
        vector_scores = self._normalize(vector_scores)
        bm25_scores = self._normalize(bm25_scores)

        # 4. 融合分数(可调权重)
        alpha = 0.7  # 向量检索权重
        final_scores = [
            alpha * v + (1 - alpha) * b
            for v, b in zip(vector_scores, bm25_scores)
        ]

        # 5. 返回 Top-K
        top_indices = sorted(
            range(len(final_scores)),
            key=lambda i: final_scores[i],
            reverse=True
        )[:top_k]

        return [self.documents[i] for i in top_indices]

    def _normalize(self, scores):
        """归一化分数到 [0, 1]"""
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [0.5] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]
```

#### 🔄 优化三:重排序(Re-ranking)

```python
class ReRanker:
    """重排序器"""

    def __init__(self, cross_encoder_model):
        self.model = cross_encoder_model

    def rerank(self, query, documents, top_k=3):
        """对检索结果重排序"""
        # 1. 计算每个文档与查询的相关性分数
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)

        # 2. 排序
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        # 3. 返回 Top-K
        return [documents[i] for i in ranked_indices[:top_k]]
```

**完整流程**:

```
查询:什么是量子计算?
    ↓
初步检索(混合)→ 返回 20 个候选文档
    ↓
重排序(精确)→ 返回 Top 3 最相关文档
    ↓
生成答案
```

---

## 4. 向量数据库实战

### 4.1 为什么需要向量数据库?

#### 🤔 问题场景

```python
# ❌ 问题:在内存中存储大量向量
embeddings = []  # 100万个向量
documents = []   # 100万个文档

# 占用内存:100万 × 768维 × 4字节 ≈ 3GB
# 检索速度:O(n) 线性搜索,非常慢
```

#### ✅ 向量数据库的优势

1. **高效存储**:压缩算法,节省空间
2. **快速检索**:近似最近邻算法(ANN),亚线性复杂度
3. **可扩展性**:支持海量数据
4. **持久化**:数据不会丢失

### 4.2 Chroma 向量数据库

#### 🚀 快速开始

```python
import chromadb
from chromadb.utils import embedding_functions

# 1. 创建客户端
client = chromadb.Client()

# 2. 创建集合(类似数据库表)
collection = client.create_collection(
    name="my_knowledge_base",
    embedding_function=embedding_functions.DefaultEmbeddingFunction()
)

# 3. 添加文档
collection.add(
    documents=[
        "量子计算是利用量子力学现象进行信息处理的计算方式",
        "Python 是一种高级编程语言",
        "机器学习是人工智能的一个分支"
    ],
    metadatas=[
        {"source": "wiki", "topic": "physics"},
        {"source": "wiki", "topic": "programming"},
        {"source": "wiki", "topic": "AI"}
    ],
    ids=["doc1", "doc2", "doc3"]
)

# 4. 查询
results = collection.query(
    query_texts=["什么是量子计算?"],
    n_results=2
)

print("检索结果:", results['documents'][0])
print("相关性:", results['distances'][0])
```

#### 💡 集成到 RAG 系统

```python
class ChromaRAG:
    """基于 Chroma 的 RAG 系统"""

    def __init__(self, llm, collection_name="knowledge_base"):
        self.llm = llm
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(self, documents, metadatas=None):
        """添加文档到向量数据库"""
        ids = [f"doc_{i}" for i in range(len(documents))]

        self.collection.add(
            documents=documents,
            metadatas=metadatas or [{}] * len(documents),
            ids=ids
        )

        print(f"✅ 已添加 {len(documents)} 个文档")

    def query(self, question, top_k=3):
        """RAG 查询"""
        # 1. 检索相关文档
        results = self.collection.query(
            query_texts=[question],
            n_results=top_k
        )

        relevant_docs = results['documents'][0]

        # 2. 构建提示词
        context = "\n\n".join(relevant_docs)
        prompt = f"""
        基于以下文档回答问题:

        {context}

        问题:{question}
        答案:
        """

        # 3. 生成回答
        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "sources": relevant_docs,
            "metadata": results['metadatas'][0]
        }

    def delete(self, doc_id):
        """删除文档"""
        self.collection.delete(ids=[doc_id])

    def update(self, doc_id, new_document):
        """更新文档"""
        self.collection.update(
            ids=[doc_id],
            documents=[new_document]
        )
```

### 4.3 其他向量数据库对比

| 数据库 | 特点 | 适用场景 | 部署方式 |
|--------|------|----------|----------|
| **Chroma** | 轻量级、易用 | 开发原型、小规模 | 本地/嵌入式 |
| **Pinecone** | 云原生、高性能 | 生产环境、大规模 | 云服务 |
| **Qdrant** | 高性能、可扩展 | 中大型项目 | 本地/云端 |
| **Milvus** | 企业级、分布式 | 大规模生产 | 分布式集群 |
| **FAISS** | Meta 出品、极速 | 研究实验 | 纯内存/本地 |

---

## 5. Memory 作为工具集成

### 5.1 Memory Tool 设计

#### 🎯 核心思想

将 Memory 当作"工具",让 Agent 主动决定何时读写记忆

```python
from hello_agents import Tool

class MemoryTool(Tool):
    """记忆工具"""

    def __init__(self, memory_system):
        super().__init__(
            name="memory",
            description="保存和检索长期记忆"
        )
        self.memory = memory_system

    def get_parameters(self):
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作类型:save(保存)或 recall(检索)",
                required=True
            ),
            ToolParameter(
                name="content",
                type="string",
                description="要保存的内容或检索的查询",
                required=True
            )
        ]

    def run(self, parameters):
        """执行记忆操作"""
        action = parameters.get("action")
        content = parameters.get("content")

        if action == "save":
            # 保存记忆
            self.memory.save(content)
            return f"✅ 已保存记忆: {content[:50]}..."

        elif action == "recall":
            # 检索记忆
            results = self.memory.recall(content, limit=3)
            if results:
                return f"📚 检索到 {len(results)} 条记忆:\n" + "\n".join(results)
            else:
                return "❌ 未找到相关记忆"

        else:
            return f"❌ 未知操作: {action}"
```

#### 💡 使用示例

```python
from hello_agents import ReActAgent, ToolRegistry

# 1. 创建记忆系统
memory = LongTermMemory(storage_path="agent_memory.db")

# 2. 创建记忆工具
memory_tool = MemoryTool(memory)

# 3. 注册到 Agent
registry = ToolRegistry()
registry.register_tool(memory_tool)

agent = ReActAgent(
    name="记忆助手",
    llm=llm,
    tool_registry=registry
)

# 4. 对话中自动使用记忆
agent.run("请记住,我的生日是 1990年1月1日")
# Agent 内部会调用:memory_tool.run({"action": "save", "content": "用户生日: 1990-01-01"})

agent.run("我的生日是什么时候?")
# Agent 内部会调用:memory_tool.run({"action": "recall", "content": "用户生日"})
# 返回:您的生日是 1990年1月1日
```

### 5.2 RAG Tool 设计

```python
class RAGTool(Tool):
    """RAG 检索工具"""

    def __init__(self, rag_system):
        super().__init__(
            name="knowledge_search",
            description="从知识库中搜索相关信息"
        )
        self.rag = rag_system

    def get_parameters(self):
        return [
            ToolParameter(
                name="query",
                type="string",
                description="要搜索的问题或关键词",
                required=True
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description="返回结果数量(默认3)",
                required=False,
                default=3
            )
        ]

    def run(self, parameters):
        """执行知识检索"""
        query = parameters.get("query")
        top_k = parameters.get("top_k", 3)

        # 检索相关文档
        docs = self.rag.retrieve(query, top_k=top_k)

        if not docs:
            return "❌ 未找到相关信息"

        # 格式化返回
        result = f"📚 找到 {len(docs)} 条相关信息:\n\n"
        for i, doc in enumerate(docs, 1):
            result += f"{i}. {doc[:200]}...\n\n"

        return result
```

### 5.3 完整示例:Agent with Memory & RAG

```python
from hello_agents import ReActAgent, HelloAgentsLLM, ToolRegistry

# 1. 初始化组件
llm = HelloAgentsLLM()
short_term_memory = ShortTermMemory(max_messages=20)
long_term_memory = LongTermMemory(storage_path="memory.db")
rag_system = ChromaRAG(llm, collection_name="docs")

# 2. 添加知识到 RAG
rag_system.add_documents([
    "公司政策:员工可享受年假15天",
    "公司地址:北京市朝阳区xxx大厦",
    "工作时间:周一至周五 9:00-18:00"
])

# 3. 创建工具
memory_tool = MemoryTool(long_term_memory)
rag_tool = RAGTool(rag_system)

# 4. 创建 Agent
registry = ToolRegistry()
registry.register_tool(memory_tool)
registry.register_tool(rag_tool)

agent = ReActAgent(
    name="智能助手",
    llm=llm,
    tool_registry=registry,
    memory=short_term_memory  # 添加短期记忆
)

# 5. 多轮对话
print("=== 第1轮 ===")
response1 = agent.run("公司的年假政策是什么?")
# Agent 会调用 knowledge_search 工具检索
print(response1)

print("\n=== 第2轮 ===")
response2 = agent.run("帮我记住,我打算在7月休年假")
# Agent 会调用 memory 工具保存
print(response2)

print("\n=== 第3轮 ===")
response3 = agent.run("我之前说的休假计划是什么?")
# Agent 会从短期记忆中找到之前的对话
# 或调用 memory 工具检索长期记忆
print(response3)
```

**执行流程可视化**:

```
用户:公司的年假政策是什么?
    ↓
Agent 思考:需要查询公司政策
    ↓
调用 knowledge_search("年假政策")
    ↓
RAG 系统检索 → "员工可享受年假15天"
    ↓
Agent 回答:根据公司政策,员工可享受年假15天

用户:帮我记住,我打算在7月休年假
    ↓
Agent 思考:用户要保存信息
    ↓
调用 memory.save("用户计划7月休年假")
    ↓
确认:✅ 已保存

用户:我之前说的休假计划是什么?
    ↓
Agent 思考:需要检索之前的记录
    ↓
方案1:查看短期记忆(对话历史)
方案2:调用 memory.recall("休假计划")
    ↓
Agent 回答:您之前说打算在7月休年假
```

---

## 6. 本章总结

### 🎯 你学到了什么?

#### 1. 记忆系统三层架构

```
短期记忆(对话上下文)
   ├── 滑动窗口策略
   ├── 重要性采样
   └── 摘要压缩

长期记忆(知识存储)
   ├── 数据库持久化
   ├── 向量检索
   └── 关键词索引

个性化记忆(用户画像)
   ├── 偏好记录
   ├── 交互历史
   └── 上下文信息
```

#### 2. RAG 核心流程

```
文档 → 分块 → 向量化 → 存储
                      ↓
用户查询 → 向量化 → 检索 → Top-K文档
                           ↓
                    构建提示词 → LLM生成 → 答案
```

#### 3. 技术栈选择

| 组件 | 推荐方案 | 备选方案 |
|------|----------|----------|
| **Embedding** | OpenAI text-embedding-ada-002 | sentence-transformers |
| **向量数据库** | Chroma(开发), Pinecone(生产) | Qdrant, Milvus |
| **检索算法** | 混合检索(向量+BM25) | 纯向量检索 |
| **优化策略** | 重排序 | - |

### 📈 Memory vs RAG 对比

| 维度 | Memory(记忆) | RAG(检索) |
|-----|---------------|------------|
| **用途** | 记住对话历史和用户信息 | 检索外部知识 |
| **数据来源** | 对话过程中生成 | 预先准备的文档库 |
| **更新频率** | 实时更新 | 定期批量更新 |
| **存储方式** | 列表/数据库 | 向量数据库 |
| **检索方式** | 时间顺序/关键词 | 语义相似度 |
| **典型场景** | "记住我的偏好" | "查找相关文档" |

### 🚀 实战建议

#### 对于初学者

1. ✅ **从简单开始**
   ```python
   # 先实现一个简单的列表记忆
   memory = []
   memory.append("用户喜欢Python")
   ```

2. ✅ **理解核心概念**
   - 什么是向量?(数字表示的语义)
   - 什么是相似度?(向量之间的距离)
   - 什么是检索?(找到最相似的)

3. ✅ **跑通完整流程**
   - 使用 Chroma 做一个简单 RAG
   - 体验检索效果

#### 对于进阶者

1. ✅ **优化检索质量**
   - 调整分块大小和重叠
   - 尝试不同的 Embedding 模型
   - 实现混合检索

2. ✅ **提升系统性能**
   - 使用 GPU 加速 Embedding
   - 优化向量索引结构
   - 实现结果缓存

3. ✅ **处理实际问题**
   - 多语言文档处理
   - 大规模数据索引
   - 实时更新策略

#### 对于专业开发者

1. ✅ **生产级部署**
   - 使用 Pinecone/Qdrant 云服务
   - 实现分布式索引
   - 添加监控和日志

2. ✅ **高级优化**
   - 自训练 Embedding 模型
   - 实现 Hypothetical Document Embeddings (HyDE)
   - 多模态检索(文本+图片)

3. ✅ **系统集成**
   - 与现有系统集成
   - 构建知识管理平台
   - 实现智能客服系统

### 💡 常见问题 FAQ

#### Q1: 向量数据库和传统数据库有什么区别?

**传统数据库**:
```sql
SELECT * FROM docs WHERE title = '量子计算'
```
- 精确匹配
- 无法理解语义

**向量数据库**:
```python
search("什么是量子?")
# 能找到"量子计算"、"量子力学"等相关文档
```
- 语义相似
- 模糊匹配

#### Q2: 如何选择合适的 Embedding 模型?

```
小项目(<10万文档):
  → sentence-transformers/all-MiniLM-L6-v2
  → 快速、免费

中型项目(10-100万文档):
  → OpenAI text-embedding-ada-002
  → 质量好、成本可控

大型项目(>100万文档):
  → 自训练领域模型
  → 针对特定领域优化
```

#### Q3: RAG 检索效果不好怎么办?

**诊断步骤**:

1. **检查分块质量**
   ```python
   # 打印几个 chunk 看看
   for chunk in chunks[:5]:
       print(chunk)
       print("---")
   ```

2. **检查检索结果**
   ```python
   results = retrieve("查询词")
   print("是否包含相关信息?", results)
   ```

3. **调整参数**
   ```python
   # 增加检索数量
   results = retrieve(query, top_k=10)

   # 尝试不同的 chunk_size
   chunker = Chunker(chunk_size=300)  # 默认500
   ```

#### Q4: Memory 和 RAG 如何配合使用?

```python
# 短期记忆:当前对话
short_memory = [
    "用户:我想了解量子计算",
    "助手:好的,让我查找相关资料"
]

# 长期记忆:用户画像
long_memory = {
    "interests": ["物理", "编程"],
    "skill_level": "中级"
}

# RAG:知识库检索
rag_docs = retrieve("量子计算", user_profile=long_memory)

# 综合生成答案
context = {
    "recent_chat": short_memory,
    "user_profile": long_memory,
    "knowledge": rag_docs
}
answer = llm.generate_with_context(context)
```

### 🔗 相关资源

- **GitHub 仓库**: https://github.com/jjyaoao/helloagents
- **Chroma 文档**: https://docs.trychroma.com/
- **LangChain RAG 教程**: https://python.langchain.com/docs/use_cases/question_answering/
- **Sentence Transformers**: https://www.sbert.net/

---

## 📝 快速参考

### 安装依赖

```bash
# 基础包
pip install "hello-agents==0.1.1"

# 向量数据库
pip install chromadb

# Embedding 模型
pip install sentence-transformers

# 可选:BM25检索
pip install rank-bm25
```

### 最小 Memory 示例

```python
from hello_agents import ShortTermMemory

memory = ShortTermMemory(max_messages=10)
memory.add_message("user", "我叫小明")
memory.add_message("assistant", "你好,小明!")

print(memory.get_messages())
```

### 最小 RAG 示例

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_docs")

# 添加文档
collection.add(
    documents=["Python是编程语言", "机器学习是AI分支"],
    ids=["doc1", "doc2"]
)

# 查询
results = collection.query(
    query_texts=["什么是Python?"],
    n_results=1
)
print(results['documents'])
```

### 集成到 Agent

```python
from hello_agents import ReActAgent, ToolRegistry

# 创建 Memory 和 RAG 工具
memory_tool = MemoryTool(memory_system)
rag_tool = RAGTool(rag_system)

# 注册到 Agent
registry = ToolRegistry()
registry.register_tool(memory_tool)
registry.register_tool(rag_tool)

agent = ReActAgent(llm=llm, tool_registry=registry)
```

---

## 🎓 章节习题提示

1. **Memory 策略对比**:实现三种记忆管理策略,对比效果
2. **RAG 系统搭建**:从零搭建一个完整的 RAG 系统
3. **混合检索实践**:实现向量检索 + BM25 的混合检索
4. **向量数据库选型**:对比 Chroma、Pinecone、Qdrant 的性能
5. **实战项目**:构建一个带记忆的客服 Agent

---

## 📌 核心要点回顾

```
🧠 为什么需要记忆?
   → 保持对话连贯 + 学习用户偏好 + 扩展知识边界

📝 三种记忆类型
   → 短期(对话) + 长期(知识) + 个性化(画像)

🔍 RAG 核心流程
   → 分块 → 向量化 → 存储 → 检索 → 生成

🗄️ 向量数据库
   → 高效存储 + 快速检索 + 可扩展

🔧 工具化集成
   → Memory Tool + RAG Tool → Agent 主动使用
```

---

**下一章预告**:第九章将探讨上下文工程,学习如何优化提示词、管理 Token 消耗、实现高效的上下文策略!

**Happy Learning! 🚀**
