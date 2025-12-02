"""
Agent Graph Nodes
定义 LangGraph 中的节点函数
"""
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from agents.state import AgentState
from agents.prompts import (
    SYSTEM_PROMPT,
    ANSWER_GENERATION_PROMPT,
    CONFIDENCE_EVALUATION_PROMPT
)


def summarize_conversation_history(
    messages: List,
    llm=None,
    max_turns: int = 10
) -> Optional[str]:
    """
    对话历史摘要函数 (T029)

    当对话轮数超过 max_turns 时,对历史对话进行摘要
    保留最近的几轮完整对话,对更早的对话进行压缩

    Args:
        messages: 对话消息列表
        llm: LLM 客户端实例
        max_turns: 触发摘要的最大轮数

    Returns:
        摘要文本,如果不需要摘要则返回 None
    """
    if len(messages) <= max_turns * 2:  # 每轮包含用户和助手消息
        return None

    # 保留最近的 3 轮对话
    recent_messages = messages[-(3 * 2):]
    # 对更早的对话进行摘要
    history_to_summarize = messages[:-(3 * 2)]

    if not history_to_summarize:
        return None

    # 如果没有 LLM,使用简单的文本摘要
    if llm is None:
        summary_parts = []
        for i in range(0, len(history_to_summarize), 2):
            if i + 1 < len(history_to_summarize):
                q = history_to_summarize[i].content[:50]
                a = history_to_summarize[i + 1].content[:50]
                summary_parts.append(f"Q: {q}... A: {a}...")

        return "历史对话摘要:\n" + "\n".join(summary_parts[:5])

    try:
        # 使用 LLM 生成智能摘要
        history_text = "\n\n".join([
            f"{'用户' if i % 2 == 0 else '助手'}: {msg.content}"
            for i, msg in enumerate(history_to_summarize)
        ])

        summarization_prompt = f"""请对以下对话历史进行简洁的摘要,提取关键信息和讨论的主题:

{history_text}

摘要 (保持简洁,50-100字):"""

        messages_to_send = [
            SystemMessage(content="你是一个对话摘要助手,擅长提取对话的关键信息。"),
            HumanMessage(content=summarization_prompt)
        ]

        response = llm.invoke(messages_to_send)
        return f"历史对话摘要: {response.content}"

    except Exception as e:
        print(f"对话摘要生成失败: {e}")
        return None


def parse_context_references(question: str, conversation_history: List) -> str:
    """
    解析上下文引用函数 (T030)

    识别问题中的上下文引用 (如 "刚才提到的", "那个", "它", "这个方法" 等)
    并将其替换或补充为完整的表述

    Args:
        question: 用户当前问题
        conversation_history: 对话历史消息列表

    Returns:
        增强后的问题文本
    """
    # 上下文引用关键词
    context_keywords = [
        "刚才", "刚刚", "之前", "上面", "前面",
        "那个", "这个", "它", "他", "她",
        "上述", "上文", "刚提到", "你说的",
        "刚才提到的", "之前说的", "前面提到的"
    ]

    # 检查是否包含上下文引用
    has_context_ref = any(keyword in question for keyword in context_keywords)

    if not has_context_ref or not conversation_history:
        return question

    # 提取最近的对话内容作为上下文
    # 获取最近的 2-3 轮对话
    recent_turns = []
    for i in range(len(conversation_history) - 1, max(len(conversation_history) - 6, -1), -1):
        msg = conversation_history[i]
        recent_turns.insert(0, msg.content[:100])  # 限制长度

    if not recent_turns:
        return question

    # 构建增强的问题
    # 在问题前添加上下文信息
    context_info = " | ".join(recent_turns[-2:])  # 最近 2 条消息

    # 如果问题很短且包含代词,添加上下文
    if len(question.split()) < 10:
        enhanced_question = f"[上下文: {context_info}] {question}"
    else:
        # 问题较长,仅在必要时添加简短上下文
        enhanced_question = f"{question} (参考之前讨论的: {recent_turns[-1][:50]}...)"

    return enhanced_question


def question_analysis_node(state: AgentState, llm=None) -> Dict[str, Any]:
    """
    问题分析节点:分析用户问题并准备检索

    功能:
    - 提取问题的关键信息
    - 识别问题类型
    - 解析上下文引用 (T030)
    - 为检索做准备

    Args:
        state: 当前 Agent 状态
        llm: LLM 客户端实例 (用于智能摘要)

    Returns:
        更新后的状态字典
    """
    question = state["question"]

    # 简单的问题分析:提取关键词,验证问题有效性
    if not question or not question.strip():
        return {
            "need_clarification": True,
            "clarification_questions": ["您的问题似乎是空的,请提出一个具体的技术问题。"]
        }

    # 获取对话历史
    conversation_history = state.get("messages", [])
    current_turn = state.get("turn_count", 0)

    # T030: 解析上下文引用
    # 如果问题包含代词或引用词,增强问题文本
    enhanced_question = parse_context_references(question.strip(), conversation_history)

    # T029: 对话历史摘要 (如果超过 10 轮)
    summary = None
    if len(conversation_history) > 10 * 2:  # 每轮包含用户和助手消息
        summary = summarize_conversation_history(
            messages=conversation_history,
            llm=llm,
            max_turns=10
        )

    # 准备更新的状态
    update = {
        "question": enhanced_question,
        "turn_count": current_turn + 1,
        "need_clarification": False
    }

    # 如果生成了摘要,将其添加到消息历史的开头
    if summary:
        update["conversation_summary"] = summary

    return update


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
