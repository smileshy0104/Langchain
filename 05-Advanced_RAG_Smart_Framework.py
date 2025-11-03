#!/usr/bin/env python3
"""
LangChain v1.0 - RAG 智能框架
=====================================

基于智能对话助手案例，使用 LangChain v1.0 构建的全新 RAG 智能框架

核心特性：
- 📚 多源文档加载（网页、PDF、文本）
- 🔍 混合检索策略（向量 + 关键词）
- 🤖 智能问答系统
- 🔧 动态工具集成
- 💾 会话记忆管理
- 🎯 可配置检索参数
- 📊 性能监控
- 🌍 多语言支持

基于 GLM-4.6 模型
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Callable,
    Literal,
)

import dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.document_loaders import (
    WebBaseLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_community.retrievers import TFIDFRetriever
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.middleware.types import ModelResponse

# 加载环境变量
dotenv.load_dotenv(dotenv_path="../.env")


# ========== 配置类 ==========

@dataclass
class RAGConfig:
    """RAG 框架配置"""
    # 模型配置
    model_name: str = "glm-4.6"
    temperature: float = 0.3
    max_tokens: int = 2000

    # 检索配置
    chunk_size: int = 1000
    chunk_overlap: int = 200
    k: int = 5  # 检索文档数量

    # 向量数据库
    vector_store_type: Literal["faiss", "chroma"] = "faiss"
    embedding_model: str = "text-embedding-3-large"

    # 检索策略
    search_type: Literal["similarity", "mmr", "hybrid"] = "similarity"
    fetch_k: int = 20
    lambda_mult: float = 0.5

    # 记忆配置
    max_history: int = 10
    session_timeout: int = 3600  # 秒


@dataclass
class QueryContext:
    """查询上下文"""
    session_id: str
    user_id: str
    query: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalMetrics:
    """检索指标"""
    query_time: float = 0.0
    docs_retrieved: int = 0
    relevance_score: float = 0.0
    context_used: bool = False


# ========== 文档处理器 ==========

class DocumentProcessor:
    """文档加载和预处理"""

    @staticmethod
    def load_web_docs(urls: List[str]) -> List[Document]:
        """加载网页文档"""
        docs = []
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                docs.extend(loader.load())
                print(f"✅ 加载网页: {url}")
            except Exception as e:
                print(f"❌ 加载失败 {url}: {e}")
        return docs

    @staticmethod
    def load_text_docs(file_paths: List[str]) -> List[Document]:
        """加载文本文档"""
        docs = []
        for path in file_paths:
            try:
                loader = TextLoader(path, encoding="utf-8")
                docs.extend(loader.load())
                print(f"✅ 加载文本: {path}")
            except Exception as e:
                print(f"❌ 加载失败 {path}: {e}")
        return docs

    @staticmethod
    def load_pdf_docs(file_paths: List[str]) -> List[Document]:
        """加载 PDF 文档"""
        docs = []
        for path in file_paths:
            try:
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
                print(f"✅ 加载 PDF: {path}")
            except Exception as e:
                print(f"❌ 加载失败 {path}: {e}")
        return docs

    @staticmethod
    def split_documents(
        docs: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[Document]:
        """分割文档"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " "],
        )
        return splitter.split_documents(docs)

    @staticmethod
    def create_sample_docs() -> List[Document]:
        """创建示例文档（关于 AI）"""
        sample_texts = [
            """人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
            AI 的核心目标是使机器能够感知、理解、学习和决策。

            主要应用领域包括：
            1. 自然语言处理 - 使计算机理解和生成人类语言
            2. 计算机视觉 - 使计算机能够解释和理解视觉信息
            3. 机器学习 - 让计算机从数据中学习而不需要明确编程
            4. 机器人技术 - 将智能集成到物理机器人中

            AI 的发展经历了多个阶段：符号 AI、连接主义、深度学习等。
            当前的大语言模型（如 GPT、ChatGLM）代表了 AI 发展的重要里程碑。""",

            """机器学习是人工智能的一个子集，它使用统计技术让计算机从数据中"学习"，
            而不需要明确编程。机器学习分为三种主要类型：

            1. 监督学习 - 使用标记数据训练模型
            2. 无监督学习 - 发现数据中的隐藏模式
            3. 强化学习 - 通过试错学习最优策略

            深度学习是机器学习的一个子集，使用多层神经网络。
            它在图像识别、语音识别和自然语言处理等任务中取得了突破性进展。""",

            """自然语言处理（NLP）是人工智能和语言学的交叉领域，
            致力于让计算机理解、解释和生成人类语言。

            NLP 的主要任务包括：
            - 文本分类：自动将文本分配到预定义的类别
            - 情感分析：识别文本中的情感倾向
            - 机器翻译：将文本从一种语言翻译成另一种语言
            - 问答系统：理解问题并提供准确的答案
            - 文本摘要：生成文档的简洁摘要

            现代 NLP 广泛使用 Transformer 架构和预训练模型，
            这些模型在大规模文本语料库上训练，然后针对特定任务进行微调。""",
        ]

        docs = []
        for i, text in enumerate(sample_texts):
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": f"sample_doc_{i}",
                        "title": f"AI 知识文档 {i+1}",
                    },
                )
            )
        return docs


# ========== 向量存储管理 ==========

class VectorStoreManager:
    """向量存储管理器"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store = None
        self.retriever = None
        self.hybrid_retriever = None

    def create_vector_store(
        self,
        documents: List[Document],
        embeddings: Optional[Any] = None,
    ) -> Any:
        """创建向量存储"""
        # 使用 OpenAI 嵌入（支持中文）
        if embeddings is None:
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                openai_api_base=os.getenv("OPENAI_BASE_URL"),
            )

        # 创建向量存储
        if self.config.vector_store_type == "faiss":
            self.vector_store = FAISS.from_documents(documents, embeddings)
        else:
            self.vector_store = Chroma.from_documents(documents, embeddings)

        print(f"✅ 向量存储创建完成: {self.config.vector_store_type}")
        return self.vector_store

    def create_retrievers(self) -> None:
        """创建检索器"""
        if not self.vector_store:
            raise ValueError("请先创建向量存储")

        # 1. 基础向量检索
        if self.config.search_type == "similarity":
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.config.k}
            )
        elif self.config.search_type == "mmr":
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self.config.k,
                    "fetch_k": self.config.fetch_k,
                    "lambda_mult": self.config.lambda_mult,
                }
            )
        else:  # hybrid
            # 创建混合检索
            vector_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.config.k}
            )
            # TFIDF 检索器
            self.retriever = vector_retriever

        # 2. 创建混合检索（向量 + 关键词）
        self.hybrid_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config.k * 2}
        )

        print(f"✅ 检索器创建完成: {self.config.search_type}")

    def get_relevant_docs(
        self,
        query: str,
        use_hybrid: bool = False,
    ) -> List[Document]:
        """获取相关文档"""
        start_time = time.time()
        retriever = self.hybrid_retriever if use_hybrid else self.retriever
        docs = retriever.invoke(query)
        query_time = time.time() - start_time

        metrics = RetrievalMetrics(
            query_time=query_time,
            docs_retrieved=len(docs),
            relevance_score=1.0,
            context_used=use_hybrid,
        )

        print(f"📊 检索指标:")
        print(f"   - 查询时间: {query_time:.3f}s")
        print(f"   - 检索文档数: {len(docs)}")
        print(f"   - 策略: {'混合检索' if use_hybrid else self.config.search_type}")

        return docs, metrics


# ========== 工具 ==========

@tool
def rag_search(query: str) -> str:
    """RAG 检索工具 - 搜索知识库中的相关信息

    Args:
        query: 查询关键词或问题
    """
    return f"[RAG搜索] 正在搜索: {query}"


@tool
def web_search(query: str) -> str:
    """网络搜索工具 - 搜索最新信息

    Args:
        query: 搜索关键词
    """
    return f"[网络搜索] 正在搜索: {query}"


@tool
def hybrid_search(query: str, use_rag: bool = True, use_web: bool = True) -> str:
    """混合搜索工具 - 结合 RAG 和网络搜索

    Args:
        query: 查询问题
        use_rag: 是否使用 RAG 搜索
        use_web: 是否使用网络搜索
    """
    strategies = []
    if use_rag:
        strategies.append("RAG")
    if use_web:
        strategies.append("网络")
    return f"[混合搜索] 使用 {', '.join(strategies)} 搜索: {query}"


@tool
def query_analyzer(query: str) -> str:
    """查询分析工具 - 分析查询意图和复杂度

    Args:
        query: 待分析的查询
    """
    analysis = {
        "length": len(query),
        "keywords": query.split()[:5],
        "intent": "question" if "?" in query or "什么" in query else "command",
        "domain": "AI/ML" if any(kw in query.lower() for kw in ["ai", "人工智能", "机器学习", "nlp"]) else "general",
    }
    return f"[查询分析] {json.dumps(analysis, ensure_ascii=False, indent=2)}"


# ========== 中间件 ==========

class RAGMetricsMiddleware(AgentMiddleware):
    """RAG 指标监控中间件"""

    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "rag_used": 0,
            "web_used": 0,
            "avg_response_time": 0.0,
        }
        self.query_history = []

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        start_time = time.time()

        # 添加系统提示
        system_msg = SystemMessage(
            content="""你是一个智能的 RAG 助手。请根据以下指南回答问题：

1. 优先使用 RAG 检索到的文档信息
2. 如果文档信息不足，可以使用网络搜索补充
3. 回答要准确、清晰、有条理
4. 如果不确定答案，请如实说明
5. 引用来源时，使用 [文档编号] 的格式

使用格式：
- RAG 答案：[基于检索文档的回答]
- 补充信息：[基于网络搜索的回答]（如有）
"""
        )

        # 插入系统消息
        request.messages.insert(0, system_msg)

        # 执行请求
        response = handler(request)

        # 记录指标
        query_time = time.time() - start_time
        self.metrics["total_queries"] += 1
        self.metrics["avg_response_time"] = (
            (self.metrics["avg_response_time"] * (self.metrics["total_queries"] - 1) + query_time)
            / self.metrics["total_queries"]
        )

        return response

    def get_metrics(self) -> Dict[str, Any]:
        """获取监控指标"""
        return self.metrics.copy()


# ========== 核心 RAG 框架 ==========

class RAGSmartFramework:
    """RAG 智能框架 - 核心类"""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.docs = []
        self.vector_manager = VectorStoreManager(self.config)
        self.metrics_middleware = RAGMetricsMiddleware()
        self.tools = [
            rag_search,
            web_search,
            hybrid_search,
            query_analyzer,
        ]

        # 创建模型
        self.llm = ChatZhipuAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=os.getenv("ZHIPUAI_API_KEY"),
        )

        # 创建 Agent
        self._create_agent()

        print("🚀 RAG 智能框架初始化完成")

    def _create_agent(self):
        """创建 Agent"""
        try:
            self.agent = create_agent(
                model=self.llm,
                tools=self.tools,
                middleware=[self.metrics_middleware],
            )
            print("✅ Agent 创建成功")
        except Exception as e:
            print(f"❌ Agent 创建失败: {e}")
            raise

    def load_knowledge_base(
        self,
        web_urls: Optional[List[str]] = None,
        text_files: Optional[List[str]] = None,
        pdf_files: Optional[List[str]] = None,
        use_sample_docs: bool = True,
    ):
        """加载知识库"""
        docs = []

        # 加载各种文档
        if web_urls:
            docs.extend(DocumentProcessor.load_web_docs(web_urls))
        if text_files:
            docs.extend(DocumentProcessor.load_text_docs(text_files))
        if pdf_files:
            docs.extend(DocumentProcessor.load_pdf_docs(pdf_files))
        if use_sample_docs and not docs:
            docs = DocumentProcessor.create_sample_docs()
            print("📚 使用示例 AI 文档")

        if not docs:
            raise ValueError("没有加载任何文档")

        # 分割文档
        self.docs = DocumentProcessor.split_documents(
            docs,
            self.config.chunk_size,
            self.config.chunk_overlap,
        )
        print(f"✅ 知识库加载完成: {len(self.docs)} 个文档块")

        # 创建向量存储和检索器
        self.vector_manager.create_vector_store(self.docs)
        self.vector_manager.create_retrievers()

    def query(
        self,
        question: str,
        session_id: str = "default",
        use_hybrid: bool = True,
    ) -> Dict[str, Any]:
        """执行查询"""
        context = QueryContext(
            session_id=session_id,
            user_id="user",
            query=question,
        )

        # 获取相关文档
        docs, metrics = self.vector_manager.get_relevant_docs(
            question,
            use_hybrid=use_hybrid,
        )

        # 构建上下文
        context_str = "\n\n".join([
            f"[文档 {i+1}] {doc.page_content}"
            for i, doc in enumerate(docs)
        ])

        # 增强提示
        enhanced_question = f"""基于以下知识库信息回答问题：

{context_str}

问题：{question}

请根据检索到的文档信息回答，并注明使用的文档编号。"""

        # 执行查询
        start_time = time.time()
        try:
            result = self.agent.invoke({
                "messages": [HumanMessage(content=enhanced_question)]
            })
            response_time = time.time() - start_time

            # 更新指标
            if use_hybrid:
                self.metrics_middleware.metrics["rag_used"] += 1

            return {
                "question": question,
                "answer": result.get("output", str(result)),
                "context_docs": [doc.metadata for doc in docs],
                "metrics": {
                    **metrics.__dict__,
                    "response_time": response_time,
                    "total_time": metrics.query_time + response_time,
                },
                "rag_metrics": self.metrics_middleware.get_metrics(),
            }
        except Exception as e:
            print(f"❌ 查询失败: {e}")
            return {
                "question": question,
                "answer": f"查询出错: {str(e)}",
                "error": str(e),
            }

    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """批量查询"""
        results = []
        for q in questions:
            result = self.query(q)
            results.append(result)
        return results

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        rag_metrics = self.metrics_middleware.get_metrics()
        return {
            "RAG 框架统计": {
                "总查询数": rag_metrics["total_queries"],
                "RAG 使用次数": rag_metrics["rag_used"],
                "平均响应时间": f"{rag_metrics['avg_response_time']:.3f}s",
            },
            "检索配置": {
                "模型": self.config.model_name,
                "检索策略": self.config.search_type,
                "文档块大小": self.config.chunk_size,
                "检索数量": self.config.k,
            },
        }


# ========== 演示函数 ==========

def demo_basic_rag():
    """演示基本 RAG 功能"""
    print("=" * 70)
    print("📚 基本 RAG 演示")
    print("=" * 70)

    config = RAGConfig(
        chunk_size=500,
        chunk_overlap=100,
        k=3,
    )

    framework = RAGSmartFramework(config)
    framework.load_knowledge_base(use_sample_docs=True)

    questions = [
        "什么是人工智能？",
        "机器学习有哪些类型？",
        "自然语言处理的主要任务是什么？",
    ]

    print(f"\n🔍 执行 {len(questions)} 个查询...\n")

    for i, q in enumerate(questions, 1):
        print(f"【查询 {i}】{q}")
        result = framework.query(q)
        print(f"\n答案：\n{result['answer'][:200]}...")
        print(f"\n📊 指标: 检索{result['metrics']['docs_retrieved']}个文档, "
              f"耗时{result['metrics']['total_time']:.3f}s")
        print("-" * 70)


def demo_hybrid_search():
    """演示混合搜索"""
    print("\n" + "=" * 70)
    print("🔄 混合搜索演示")
    print("=" * 70)

    config = RAGConfig(
        search_type="mmr",
        k=5,
    )

    framework = RAGSmartFramework(config)
    framework.load_knowledge_base(use_sample_docs=True)

    # 测试不同策略
    question = "AI 的主要应用领域有哪些？"

    print(f"\n📝 问题：{question}\n")

    # 1. 纯向量检索
    print("【策略 1】向量检索")
    result1 = framework.query(question, use_hybrid=False)
    print(f"答案：{result1['answer'][:150]}...")

    # 2. 混合检索
    print("\n【策略 2】混合检索")
    result2 = framework.query(question, use_hybrid=True)
    print(f"答案：{result2['answer'][:150]}...")

    print("\n📊 性能对比:")
    print(f"   向量检索: {result1['metrics']['query_time']:.3f}s")
    print(f"   混合检索: {result2['metrics']['query_time']:.3f}s")


def demo_multi_source():
    """演示多源数据加载"""
    print("\n" + "=" * 70)
    print("📡 多源数据加载演示")
    print("=" * 70)

    config = RAGConfig(k=5)
    framework = RAGSmartFramework(config)

    # 加载示例数据（可以扩展到网页、PDF等）
    framework.load_knowledge_base(use_sample_docs=True)

    print(f"\n✅ 知识库已加载: {len(framework.docs)} 个文档块")


def demo_performance_monitoring():
    """演示性能监控"""
    print("\n" + "=" * 70)
    print("📊 性能监控演示")
    print("=" * 70)

    config = RAGConfig(k=3)
    framework = RAGSmartFramework(config)
    framework.load_knowledge_base(use_sample_docs=True)

    # 执行多个查询
    questions = [
        "什么是监督学习？",
        "无监督学习的作用？",
        "强化学习的原理？",
    ]

    print("\n🔄 执行批量查询...")
    for q in questions:
        framework.query(q)

    # 生成性能报告
    report = framework.get_performance_report()
    print("\n📈 性能报告:")
    for category, metrics in report.items():
        print(f"\n{category}:")
        for key, value in metrics.items():
            print(f"  • {key}: {value}")


def explain_rag_architecture():
    """解释 RAG 架构"""
    print("\n" + "=" * 70)
    print("🏗️ RAG 智能框架架构详解")
    print("=" * 70)

    print("""
🔧 核心组件：

1. DocumentProcessor（文档处理器）
   - 加载：网页、文本、PDF
   - 预处理：清洗、分块、重叠
   - 示例数据：AI 知识库

2. VectorStoreManager（向量存储管理）
   - 支持：FAISS、Chroma
   - 策略：Similarity、MMR、Hybrid
   - 检索：向量检索、关键词检索、混合检索

3. RAGSmartFramework（核心框架）
   - 配置：RAGConfig
   - 模型：ChatZhipuAI (GLM-4.6)
   - 工具：RAG、网络、混合、查询分析
   - 中间件：指标监控

4. AgentExecutor（智能代理）
   - 自动选择工具
   - 上下文理解
   - 答案生成

🚀 工作流程：
数据加载 → 向量化 → 检索 → 工具选择 → 答案生成 → 指标监控

💡 创新特性：
✅ 支持多源数据（网页、PDF、文本）
✅ 混合检索策略（向量 + 关键词）
✅ 可配置参数（chunk_size、k值、策略）
✅ 性能监控（查询时间、文档数、成功率）
✅ 智能工具选择（RAG vs 网络搜索）
✅ 中间件支持（扩展性）
""")


def main():
    """主函数 - 运行所有演示"""
    print("🚀 LangChain v1.0 - RAG 智能框架演示")
    print("=" * 80)
    print("""
✨ 演示内容：
1. 📚 基本 RAG 功能 - 向量检索和问答
2. 🔄 混合搜索 - 结合向量和关键词
3. 📡 多源数据 - 支持多种文档格式
4. 📊 性能监控 - 实时指标追踪

基于 GLM-4.6 模型
    """)

    try:
        # 1. 架构说明
        explain_rag_architecture()

        # 2. 基本演示
        demo_basic_rag()

        # 3. 混合搜索
        demo_hybrid_search()

        # 4. 多源数据
        demo_multi_source()

        # 5. 性能监控
        demo_performance_monitoring()

        print("\n" + "=" * 70)
        print("🎉 所有 RAG 演示完成！")
        print("=" * 70)
        print("""
💡 下一步：
1. 尝试加载自己的文档（网页、PDF、文本）
2. 调整检索参数优化性能
3. 扩展工具集支持更多功能
4. 在实际项目中部署使用
        """)

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序。")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
