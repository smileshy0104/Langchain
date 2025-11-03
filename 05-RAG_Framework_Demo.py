#!/usr/bin/env python3
"""
LangChain v1.0 - RAG 智能框架演示版
=====================================

简化版演示，专注于核心 RAG 功能
"""

import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any

import dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.tools import tool

# 加载环境变量
dotenv.load_dotenv(dotenv_path="../.env")


# ========== 工具 ==========

@tool
def rag_retriever(query: str) -> str:
    """RAG 检索工具"""
    return f"[RAG] 正在检索关于 '{query}' 的信息"


@tool
def web_search_tool(query: str) -> str:
    """网络搜索工具"""
    return f"[搜索] 正在搜索 '{query}'"


@tool
def knowledge_qa(question: str, context: str) -> str:
    """知识问答工具

    Args:
        question: 问题
        context: 上下文信息
    """
    return f"[QA] 基于以下上下文回答：\n{context}\n\n问题：{question}"


# ========== 中间件 ==========

class RAGLoggingMiddleware(AgentMiddleware):
    """RAG 日志中间件"""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse:
        print(f"\n🔍 [中间件] 处理请求")
        print(f"   消息数: {len(request.messages)}")

        # 记录开始时间
        start_time = time.time()

        # 执行请求
        response = handler(request)

        # 记录结束时间
        end_time = time.time()
        print(f"   耗时: {end_time - start_time:.3f}s")

        return response


# ========== 示例文档 ==========

def create_sample_documents() -> List[Document]:
    """创建示例文档"""
    texts = [
        """人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。

AI 的主要应用领域：
1. 自然语言处理 - 使计算机理解和生成人类语言
2. 计算机视觉 - 使计算机能够解释和理解视觉信息
3. 机器学习 - 让计算机从数据中学习
4. 机器人技术 - 将智能集成到物理机器人中

AI 的发展经历了多个阶段，包括符号 AI、连接主义、深度学习等。""",

        """机器学习是人工智能的一个子集，它使用统计技术让计算机从数据中"学习"。

机器学习的三种主要类型：
1. 监督学习 - 使用标记数据训练模型
2. 无监督学习 - 发现数据中的隐藏模式
3. 强化学习 - 通过试错学习最优策略

深度学习是机器学习的一个子集，使用多层神经网络，在图像识别、语音识别和自然语言处理等任务中取得了突破性进展。""",

        """自然语言处理（NLP）是人工智能和语言学的交叉领域，致力于让计算机理解、解释和生成人类语言。

NLP 的主要任务：
- 文本分类：自动将文本分配到预定义的类别
- 情感分析：识别文本中的情感倾向
- 机器翻译：将文本从一种语言翻译成另一种语言
- 问答系统：理解问题并提供准确的答案
- 文本摘要：生成文档的简洁摘要

现代 NLP 广泛使用 Transformer 架构和预训练模型。""",
    ]

    return [
        Document(page_content=text, metadata={"source": f"doc_{i}"})
        for i, text in enumerate(texts)
    ]


# ========== RAG 框架 ==========

class SimpleRAGFramework:
    """简化版 RAG 框架"""

    def __init__(self):
        # 工具
        self.tools = [rag_retriever, web_search_tool, knowledge_qa]

        # 模型
        self.llm = ChatZhipuAI(
            model="glm-4.6",
            temperature=0.3,
            api_key=os.getenv("ZHIPUAI_API_KEY"),
        )

        # 中间件
        self.middleware = RAGLoggingMiddleware()

        # 创建 Agent
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            middleware=[self.middleware],
        )

        # 文档存储
        self.docs = []
        self.vector_store = None

        print("✅ RAG 框架初始化完成")

    def load_documents(self, docs: List[Document]):
        """加载文档"""
        # 分割文档
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )
        self.docs = splitter.split_documents(docs)

        # 创建向量存储
        embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_BASE_URL"),
        )
        self.vector_store = FAISS.from_documents(self.docs, embeddings)

        print(f"✅ 文档加载完成: {len(self.docs)} 个文档块")

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """检索相关文档"""
        if not self.vector_store:
            raise ValueError("请先加载文档")

        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )
        return retriever.invoke(query)

    def query(self, question: str) -> Dict[str, Any]:
        """执行查询"""
        # 检索相关文档
        docs = self.retrieve(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 构建查询
        enhanced_prompt = f"""请根据以下上下文信息回答问题：

上下文：
{context}

问题：{question}

请基于上下文信息回答，并注明信息来源。"""

        # 执行查询
        result = self.agent.invoke({
            "messages": [
                SystemMessage(content="你是一个智能的AI助手，擅长基于文档回答问题。"),
                HumanMessage(content=enhanced_prompt),
            ]
        })

        return {
            "question": question,
            "answer": result.get("output", str(result)),
            "context_docs": len(docs),
            "sources": [doc.metadata for doc in docs],
        }


# ========== 演示函数 ==========

def demo_basic_rag():
    """演示基本 RAG 功能"""
    print("=" * 70)
    print("📚 RAG 智能框架演示")
    print("=" * 70)

    # 初始化框架
    framework = SimpleRAGFramework()

    # 加载示例文档
    docs = create_sample_documents()
    framework.load_documents(docs)

    # 测试查询
    questions = [
        "什么是人工智能？",
        "机器学习有哪些类型？",
        "自然语言处理的主要任务是什么？",
    ]

    print(f"\n🔍 执行 {len(questions)} 个查询...\n")

    for i, q in enumerate(questions, 1):
        print(f"【查询 {i}】{q}")
        result = framework.query(q)
        print(f"\n📝 答案：\n{result['answer']}")
        print(f"\n📊 使用了 {result['context_docs']} 个相关文档")
        print("-" * 70)


def explain_framework():
    """解释框架架构"""
    print("\n" + "=" * 70)
    print("🏗️ RAG 智能框架架构")
    print("=" * 70)

    print("""
🔧 核心组件：

1. DocumentProcessor（文档处理器）
   - 加载和分割文档
   - 生成向量表示

2. VectorStore（向量存储）
   - 使用 FAISS 进行高效检索
   - 支持相似度搜索

3. Tools（工具集）
   - RAG 检索工具
   - 网络搜索工具
   - 知识问答工具

4. Middleware（中间件）
   - 日志记录
   - 性能监控
   - 请求拦截

5. Agent（智能代理）
   - 自动选择工具
   - 整合多个信息源
   - 生成最终答案

🚀 工作流程：
1. 用户提问
2. 检索相关文档
3. 构建增强上下文
4. Agent 自动选择工具
5. 生成最终答案

💡 创新特性：
✅ 模块化设计
✅ 可扩展工具集
✅ 中间件支持
✅ LangChain v1.0 兼容
✅ 中文优化
    """)


def main():
    """主函数"""
    print("🚀 LangChain v1.0 - RAG 智能框架演示")
    print("=" * 80)

    try:
        # 架构说明
        explain_framework()

        # 基本演示
        demo_basic_rag()

        print("\n" + "=" * 70)
        print("🎉 演示完成！")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
