"""
Agent Graph Nodes
定义 LangGraph 中的节点函数
"""
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from agents.state import AgentState
from agents.prompts import (
    SYSTEM_PROMPT,
    ANSWER_GENERATION_PROMPT,
    CONFIDENCE_EVALUATION_PROMPT
)


def question_analysis_node(state: AgentState) -> Dict[str, Any]:
    """
    问题分析节点:分析用户问题并准备检索

    功能:
    - 提取问题的关键信息
    - 识别问题类型
    - 为检索做准备

    Args:
        state: 当前 Agent 状态

    Returns:
        更新后的状态字典
    """
    question = state["question"]

    # 简单的问题分析:提取关键词,验证问题有效性
    # 对于 Phase 3 单次问答,主要验证问题不为空
    if not question or not question.strip():
        return {
            "need_clarification": True,
            "clarification_questions": ["您的问题似乎是空的,请提出一个具体的技术问题。"]
        }

    # 更新状态:问题已分析,准备检索
    # 对话轮数+1
    current_turn = state.get("turn_count", 0)

    return {
        "question": question.strip(),
        "turn_count": current_turn + 1,
        "need_clarification": False
    }


def retrieval_node(state: AgentState, retriever=None, top_k: int = 3) -> Dict[str, Any]:
    """
    检索节点:从向量数据库检索相关文档

    功能:
    - 使用混合检索器(向量+BM25)检索文档
    - 评估检索结果的相关性
    - 计算初步置信度分数

    Args:
        state: 当前 Agent 状态
        retriever: HybridRetriever 实例(注入依赖)
        top_k: 返回的文档数量

    Returns:
        更新后的状态字典,包含检索到的文档和置信度分数
    """
    question = state["question"]

    if retriever is None:
        # 如果没有提供检索器,返回空结果
        return {
            "retrieved_docs": [],
            "confidence_score": 0.0
        }

    try:
        # 使用混合检索器检索文档
        # HybridRetriever.retrieve() 返回 List[Tuple[Document, float]]
        results = retriever.retrieve(question, k=top_k)

        # 转换为字典格式便于序列化
        retrieved_docs = []
        scores = []

        for doc, score in results:
            doc_dict = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            }
            retrieved_docs.append(doc_dict)
            scores.append(float(score))

        # 计算平均置信度分数
        if scores:
            avg_confidence = sum(scores) / len(scores)
        else:
            avg_confidence = 0.0

        return {
            "retrieved_docs": retrieved_docs,
            "confidence_score": avg_confidence
        }

    except Exception as e:
        # 检索失败时返回空结果
        print(f"检索失败: {e}")
        return {
            "retrieved_docs": [],
            "confidence_score": 0.0
        }


def answer_generation_node(state: AgentState, llm=None) -> Dict[str, Any]:
    """
    答案生成节点:基于检索结果生成答案

    功能:
    - 使用 LLM 基于检索文档生成答案
    - 包含来源引用
    - 评估答案置信度

    Args:
        state: 当前 Agent 状态
        llm: LLM 客户端实例(注入依赖)

    Returns:
        更新后的状态字典,包含最终答案
    """
    question = state["question"]
    retrieved_docs = state.get("retrieved_docs", [])

    # 如果没有检索到文档,返回默认答案
    if not retrieved_docs:
        return {
            "final_answer": f"抱歉,我在知识库中没有找到与 '{question}' 相关的信息。请尝试:\n1. 重新表述您的问题\n2. 上传相关的技术文档\n3. 提供更多上下文信息",
            "confidence_score": 0.0
        }

    # 构建上下文
    context = "\n\n".join([
        f"文档 {i+1}:\n{doc['content'][:500]}...\n来源: {doc['metadata'].get('source', 'unknown')}"
        for i, doc in enumerate(retrieved_docs[:3])
    ])

    # 如果没有 LLM,使用简单的模板答案
    if llm is None:
        answer = f"基于检索到的 {len(retrieved_docs)} 个相关文档:\n\n{context}\n\n这些文档应该可以帮助回答您的问题: {question}"
        confidence = state.get("confidence_score", 0.5)

        return {
            "final_answer": answer,
            "confidence_score": confidence
        }

    try:
        # 使用 LLM 生成答案
        prompt = ANSWER_GENERATION_PROMPT.format(
            question=question,
            context=context
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]

        # 调用 LLM
        response = llm.invoke(messages)
        answer = response.content

        # 评估置信度 (基于检索分数和答案长度)
        retrieval_confidence = state.get("confidence_score", 0.5)
        answer_length_score = min(len(answer) / 500, 1.0)  # 答案长度指标

        # 综合置信度
        final_confidence = (retrieval_confidence * 0.7 + answer_length_score * 0.3)

        return {
            "final_answer": answer,
            "confidence_score": final_confidence
        }

    except Exception as e:
        # LLM 调用失败,返回基于检索的简单答案
        print(f"LLM 生成失败: {e}")
        fallback_answer = f"基于检索结果:\n\n{context}"

        return {
            "final_answer": fallback_answer,
            "confidence_score": state.get("confidence_score", 0.3)
        }


def clarify_node(state: AgentState) -> Dict[str, Any]:
    """
    澄清节点:生成澄清问题

    功能:
    - 当问题不明确时生成澄清问题
    - 帮助用户提供更多信息

    Args:
        state: 当前 Agent 状态

    Returns:
        更新后的状态字典,包含澄清问题
    """
    question = state.get("question", "")
    confidence = state.get("confidence_score", 0.0)

    # 基于置信度和问题内容生成澄清问题
    clarification_questions = []

    if not question.strip():
        clarification_questions.append("请提出一个具体的技术问题。")
    elif len(question.split()) < 3:
        clarification_questions.append("您能提供更多关于问题的细节吗?")
    elif confidence < 0.3:
        clarification_questions.append("我没有找到相关信息。您能换个方式描述您的问题吗?")
    else:
        clarification_questions.append(f"关于 '{question[:30]}...',您具体想了解哪方面的内容?")

    return {
        "need_clarification": True,
        "clarification_questions": clarification_questions
    }


def should_clarify(state: AgentState) -> str:
    """
    路由函数:判断是否需要澄清

    基于置信度和检索结果决定是否需要澄清

    Args:
        state: 当前 Agent 状态

    Returns:
        下一个节点名称 ("clarify" 或 "generate")
    """
    # 检查是否已经标记需要澄清
    if state.get("need_clarification", False):
        return "clarify"

    # 基于置信度决定
    confidence = state.get("confidence_score", 0.0)
    threshold = 0.4  # 置信度阈值

    # 检查检索结果
    retrieved_docs = state.get("retrieved_docs", [])

    if not retrieved_docs or confidence < threshold:
        return "clarify"

    return "generate"
