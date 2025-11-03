#!/usr/bin/env python3
"""
LangChain v1.0 - RAG 智能框架教学版
=====================================

无需真实 API 调用的演示版本
专注于展示 RAG 框架的工作原理和架构
"""

import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json

# ========== 模拟类 ==========

class MockLLM:
    """模拟大语言模型"""
    def __init__(self):
        self.call_count = 0

    def invoke(self, messages):
        """模拟调用"""
        self.call_count += 1
        # 获取最后一个消息
        last_msg = messages[-1] if messages else {}
        content = getattr(last_msg, 'content', '')

        # 模拟基于上下文的回答
        if '上下文' in content or 'context' in content.lower():
            return {
                'output': f'这是基于检索文档的回答 (#查询{self.call_count})。'
                         f'\n\n根据提供的上下文信息，我可以回答这个问题。'
                         f'\n\n关键要点：'
                         f'\n1. 检索到了相关文档'
                         f'\n2. 基于文档内容生成答案'
                         f'\n3. 使用了中间件进行监控'
            }
        else:
            return {
                'output': f'模拟回答 (#查询{self.call_count})\n'
                         f'这是一个无需真实 API 调用的演示。'
            }

class MockRetriever:
    """模拟检索器"""
    def __init__(self, docs):
        self.docs = docs
        self.search_count = 0

    def invoke(self, query):
        """模拟检索"""
        self.search_count += 1
        print(f"   🔍 模拟检索: '{query}' (第{self.search_count}次)")

        # 返回前几个文档作为"相关"文档
        return self.docs[:3]

class MockVectorStore:
    """模拟向量存储"""
    def __init__(self, docs):
        self.docs = docs
        self.retriever = MockRetriever(docs)

    def as_retriever(self, **kwargs):
        return self.retriever


# ========== 数据类 ==========

@dataclass
class Document:
    """文档类"""
    page_content: str
    metadata: Dict[str, Any]


@dataclass
class RAGMetrics:
    """RAG 指标"""
    query_time: float = 0.0
    docs_retrieved: int = 0
    llm_calls: int = 0
    total_time: float = 0.0


# ========== 工具 ==========

def create_sample_documents() -> List[Document]:
    """创建示例文档"""
    return [
        Document(
            page_content="""人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。

主要应用领域：
1. 自然语言处理 - 使计算机理解和生成人类语言
2. 计算机视觉 - 使计算机能够解释和理解视觉信息
3. 机器学习 - 让计算机从数据中学习
4. 机器人技术 - 将智能集成到物理机器人中

AI 的发展经历了多个阶段，包括符号 AI、连接主义、深度学习等。""",
            metadata={"source": "AI基础", "chunk": 1}
        ),
        Document(
            page_content="""机器学习是人工智能的一个子集，它使用统计技术让计算机从数据中"学习"。

三种主要类型：
1. 监督学习 - 使用标记数据训练模型
2. 无监督学习 - 发现数据中的隐藏模式
3. 强化学习 - 通过试错学习最优策略

深度学习是机器学习的一个子集，使用多层神经网络，在图像识别、语音识别和自然语言处理等任务中取得了突破性进展。""",
            metadata={"source": "机器学习", "chunk": 2}
        ),
        Document(
            page_content="""自然语言处理（NLP）是人工智能和语言学的交叉领域，致力于让计算机理解、解释和生成人类语言。

主要任务：
- 文本分类：自动将文本分配到预定义的类别
- 情感分析：识别文本中的情感倾向
- 机器翻译：将文本从一种语言翻译成另一种语言
- 问答系统：理解问题并提供准确的答案
- 文本摘要：生成文档的简洁摘要

现代 NLP 广泛使用 Transformer 架构和预训练模型。""",
            metadata={"source": "NLP技术", "chunk": 3}
        ),
    ]


# ========== 工具模拟 ==========

def mock_rag_retriever(query: str) -> str:
    """模拟 RAG 检索工具"""
    return f"[RAG检索] 正在搜索与 '{query}' 相关的信息"


def mock_web_search(query: str) -> str:
    """模拟网络搜索工具"""
    return f"[网络搜索] 正在搜索 '{query}'"


def mock_query_analyzer(query: str) -> str:
    """模拟查询分析工具"""
    analysis = {
        "长度": len(query),
        "关键词": query.split()[:3],
        "意图": "question" if "?" in query else "information",
    }
    return f"[查询分析] {json.dumps(analysis, ensure_ascii=False, indent=2)}"


# ========== 中间件模拟 ==========

class MockMiddleware:
    """模拟中间件"""
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "total_time": 0.0,
        }

    def before_call(self, request):
        """调用前"""
        self.metrics["total_queries"] += 1
        print(f"\n🔧 [中间件] 请求 #{self.metrics['total_queries']}")
        print(f"   📝 消息数: {len(request.get('messages', []))}")
        return time.time()

    def after_call(self, start_time):
        """调用后"""
        elapsed = time.time() - start_time
        self.metrics["total_time"] += elapsed
        print(f"   ⏱️  耗时: {elapsed:.3f}s")


# ========== RAG 框架 ==========

class RAGSmartFrameworkDemo:
    """RAG 智能框架演示版"""

    def __init__(self):
        self.docs = []
        self.vector_store = None
        self.llm = MockLLM()
        self.middleware = MockMiddleware()

        # 模拟工具
        self.tools = [
            ("rag_retriever", mock_rag_retriever),
            ("web_search", mock_web_search),
            ("query_analyzer", mock_query_analyzer),
        ]

        print("🚀 RAG 智能框架演示版初始化完成")

    def load_knowledge_base(self, use_sample: bool = True):
        """加载知识库"""
        if use_sample:
            self.docs = create_sample_documents()
            self.vector_store = MockVectorStore(self.docs)
            print(f"✅ 知识库加载完成: {len(self.docs)} 个文档")

    def split_documents(self, docs: List[Document], chunk_size: int = 500):
        """分割文档（模拟）"""
        print(f"📄 模拟文档分割: chunk_size={chunk_size}")
        return docs

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """检索相关文档"""
        if not self.vector_store:
            raise ValueError("请先加载知识库")

        start_time = time.time()
        docs = self.vector_store.retriever.invoke(query)
        query_time = time.time() - start_time

        print(f"   📊 检索指标:")
        print(f"      - 查询时间: {query_time:.3f}s")
        print(f"      - 检索文档: {len(docs)} 个")

        return docs

    def query(self, question: str) -> Dict[str, Any]:
        """执行查询"""
        print(f"\n{'='*70}")
        print(f"🔍 查询: {question}")
        print(f"{'='*70}")

        # 记录指标
        metrics = RAGMetrics()

        # 1. 检索相关文档
        start_time = time.time()
        docs = self.retrieve(question)
        metrics.query_time = time.time() - start_time
        metrics.docs_retrieved = len(docs)

        # 2. 构建上下文
        context = "\n\n".join([
            f"[文档 {i+1}] {doc.page_content[:200]}..."
            for i, doc in enumerate(docs)
        ])

        # 3. 模拟中间件处理
        request = {"messages": [question], "context": context}
        call_start = self.middleware.before_call(request)

        # 4. 模拟 Agent 调用 LLM
        enhanced_prompt = f"""基于以下上下文回答问题：

上下文：
{context}

问题：{question}

请根据文档信息回答，并标注来源文档编号。"""

        # 模拟 agent.invoke
        llm_start = time.time()
        result = self.llm.invoke([{"content": enhanced_prompt}])
        metrics.llm_calls = self.llm.call_count
        llm_time = time.time() - llm_start

        # 5. 中间件后处理
        self.middleware.after_call(call_start)

        # 6. 计算总时间
        metrics.total_time = time.time() - start_time

        return {
            "question": question,
            "answer": result.get("output", "无回答"),
            "metrics": metrics,
            "context_docs": len(docs),
            "sources": [doc.metadata for doc in docs],
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
        return {
            "框架统计": {
                "总查询数": self.middleware.metrics["total_queries"],
                "平均耗时": f"{self.middleware.metrics['total_time'] / max(self.middleware.metrics['total_queries'], 1):.3f}s",
            },
            "文档统计": {
                "知识库文档数": len(self.docs),
                "向量存储": "模拟 FAISS",
            },
            "功能特性": {
                "RAG 检索": "✓",
                "中间件监控": "✓",
                "批量查询": "✓",
                "性能统计": "✓",
            },
        }


# ========== 演示函数 ==========

def demo_basic_rag():
    """演示基本 RAG 功能"""
    print("\n" + "="*70)
    print("📚 基本 RAG 功能演示")
    print("="*70)

    framework = RAGSmartFrameworkDemo()
    framework.load_knowledge_base(use_sample=True)

    questions = [
        "什么是人工智能？",
        "机器学习有哪些类型？",
        "自然语言处理的主要任务是什么？",
    ]

    results = framework.batch_query(questions)

    print("\n" + "="*70)
    print("📊 查询结果汇总")
    print("="*70)
    for i, result in enumerate(results, 1):
        print(f"\n【查询 {i}】{result['question']}")
        print(f"答案预览: {result['answer'][:100]}...")
        print(f"使用文档: {result['context_docs']} 个")
        print(f"耗时: {result['metrics'].total_time:.3f}s")


def demo_architecture():
    """演示架构"""
    print("\n" + "="*70)
    print("🏗️ RAG 智能框架架构详解")
    print("="*70)

    print("""
📋 框架组件：

1. 📄 DocumentProcessor（文档处理器）
   ├─ 加载：文本、PDF、网页
   ├─ 预处理：清洗、分块、重叠
   └─ 示例：AI 知识文档

2. 🔍 VectorStore（向量存储）
   ├─ 技术：FAISS / Chroma
   ├─ 策略：Similarity / MMR / Hybrid
   └─ 检索：向量相似度搜索

3. 🛠️ Tools（工具集）
   ├─ rag_retriever：RAG 检索
   ├─ web_search：网络搜索
   └─ query_analyzer：查询分析

4. 🔧 Middleware（中间件）
   ├─ 日志记录
   ├─ 性能监控
   └─ 请求拦截

5. 🤖 Agent（智能代理）
   ├─ 自动工具选择
   ├─ 上下文理解
   └─ 答案生成

6. 💾 Memory（记忆管理）
   ├─ 会话历史
   └─ 用户偏好

🚀 工作流程：
用户提问 → 文档检索 → 构建上下文 → Agent 工具选择 → 生成答案 → 指标监控

💡 核心特性：
✅ 模块化设计 - 易于扩展和维护
✅ 多策略检索 - Similarity / MMR / Hybrid
✅ 智能工具选择 - Agent 自动决策
✅ 中间件支持 - 可插拔扩展
✅ 性能监控 - 实时指标追踪
✅ 批量处理 - 高效查询
✅ LangChain v1.0 - 最新 API
    """)


def demo_performance():
    """演示性能监控"""
    print("\n" + "="*70)
    print("📊 性能监控演示")
    print("="*70)

    framework = RAGSmartFrameworkDemo()
    framework.load_knowledge_base(use_sample=True)

    # 执行查询并获取报告
    questions = ["什么是深度学习？", "NLP 的应用有哪些？"]
    framework.batch_query(questions)

    # 生成报告
    report = framework.get_performance_report()

    print("\n📈 性能报告:")
    for category, metrics in report.items():
        print(f"\n【{category}】")
        for key, value in metrics.items():
            print(f"  • {key}: {value}")


def demo_comparison():
    """演示对比分析"""
    print("\n" + "="*70)
    print("⚖️ 传统问答 vs RAG 问答")
    print("="*70)

    print("""
传统问答系统：
❌ 依赖预训练知识，可能过时
❌ 无法获取最新信息
❌ 答案可能不准确或不完整
❌ 无法追溯信息来源

RAG 智能框架：
✅ 实时检索最新信息
✅ 基于文档库生成准确答案
✅ 可追溯的信息来源
✅ 支持多源数据整合
✅ 动态知识更新
✅ 性能可监控

适用场景：
• 企业知识库问答
• 文档智能检索
• 客服机器人
• 学术研究助手
• 法律文档分析
    """)


def main():
    """主函数"""
    print("🚀 LangChain v1.0 - RAG 智能框架教学版")
    print("="*80)
    print("""
✨ 演示内容：
1. 🏗️ 框架架构介绍
2. 📚 基本 RAG 功能
3. 📊 性能监控
4. ⚖️ 优势对比分析

特点：
- 无需真实 API 调用
- 专注于概念理解
- 完整工作流程演示
- 详细指标追踪
    """)

    try:
        # 1. 架构演示
        demo_architecture()

        # 2. 基本功能
        demo_basic_rag()

        # 3. 性能监控
        demo_performance()

        # 4. 优势对比
        demo_comparison()

        print("\n" + "="*70)
        print("🎉 所有演示完成！")
        print("="*70)
        print("""
💡 总结：
✅ 理解了 RAG 框架的核心组件
✅ 掌握了文档加载和检索流程
✅ 学会了中间件的使用方式
✅ 熟悉了性能监控方法

🚀 下一步：
1. 加载真实文档（PDF、网页等）
2. 配置实际向量数据库
3. 集成真实 LLM API
4. 部署到生产环境
        """)

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断了程序。")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
