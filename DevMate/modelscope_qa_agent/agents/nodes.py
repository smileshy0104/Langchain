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


def detect_ambiguous_question(question: str, conversation_history: List) -> bool:
    """
    检测问题是否模糊需要澄清 (T035)

    检测标准:
    - 问题过短 (< 3个词且不是常见简短问题)
    - 包含模糊词汇 (报错、问题、怎么办等)
    - 缺少关键信息 (模型名、具体场景等)
    - 多个可能的解释路径

    Args:
        question: 用户问题
        conversation_history: 对话历史

    Returns:
        是否需要澄清
    """
    question_lower = question.lower()

    # 1. 检查问题长度 (中文按字符,英文按词)
    chinese_chars = sum(1 for c in question if '\u4e00' <= c <= '\u9fff')
    if chinese_chars > 0:
        # 中文问题: 少于5个汉字视为过短
        # 但如果包含具体名称(大写字母/数字)或常见问题模式,则不算过短
        common_patterns = ["是什么", "为什么", "怎么用", "如何"]
        has_common_pattern = any(word in question for word in common_patterns)
        has_specific_name = any(c.isupper() or c.isdigit() or c in ['-', '_'] for c in question)

        if chinese_chars < 5 and not has_common_pattern and not has_specific_name:
            return True
    else:
        # 英文问题: 少于3个词视为过短
        if len(question.split()) < 3:
            return True

    # 2. 检查是否包含模糊词汇
    # 只检测真正模糊的词汇，排除可能在正常问题中出现的词
    ambiguous_keywords = [
        "报错了", "怎么办", "不行", "出问题", "有问题", "问题了",
        "不对", "不好用", "用不了", "这个", "那个", "失败"
    ]
    if any(keyword in question_lower for keyword in ambiguous_keywords):
        # 包含模糊词但没有具体信息
        # 真正的具体信息应该是:错误消息、具体型号等
        specific_info_keywords = [
            "错误信息", "报错信息", "具体", "详细",
            "traceback", "error:", "exception:",
            "'", '"',  # 包含引号说明有具体错误信息
            "no module", "cannot", "failed to",  # 具体错误模式
            "qwen", "modelscope", "transformers",  # 具体产品名
            "-", "v1", "v2", "版本"  # 版本号相关
        ]
        has_specific_info = any(keyword in question_lower for keyword in specific_info_keywords)

        # 如果问题包含模糊词但不够长也没有具体信息,需要澄清
        # 中文: < 12个汉字, 英文: < 10个词
        if chinese_chars > 0:
            if chinese_chars < 2 and not has_specific_info:
                return True
        else:
            if len(question.split()) < 2 and not has_specific_info:
                return True

    # 3. 检查是否缺少关键上下文 (仅当没有对话历史时)
    if len(conversation_history) == 0:
        # 问题提到"使用"、"调用"但没有说明是什么
        action_keywords = ["使用", "调用", "运行", "执行", "安装"]
        if any(keyword in question for keyword in action_keywords):
            # 检查是否有明确的对象(包括具体的名称,不仅是分类词)
            # 具体名称:包含大写字母、数字、连字符、下划线等
            has_specific_name = any(c.isupper() or c.isdigit() or c in ['-', '_'] for c in question)

            # 或包含常见的技术对象词
            object_keywords = ["模型", "api", "sdk", "库", "包", "工具", "方法", "函数"]
            has_object_keyword = any(keyword in question_lower for keyword in object_keywords)

            # 如果既没有具体名称也没有对象关键词,才认为缺少上下文
            if not has_specific_name and not has_object_keyword:
                return True

    return False


def question_analysis_node(state: AgentState, llm=None) -> Dict[str, Any]:
    """
    问题分析节点:分析用户问题并准备检索

    功能:
    - 提取问题的关键信息
    - 识别问题类型
    - 解析上下文引用 (T030)
    - 检测是否需要澄清 (T035)
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

    # T035: 检测是否需要澄清
    needs_clarification = detect_ambiguous_question(question.strip(), conversation_history)

    if needs_clarification:
        # 标记需要澄清,稍后由 clarification_generation_node 生成具体问题
        return {
            "question": enhanced_question,
            "turn_count": current_turn + 1,
            "need_clarification": True,
            "ambiguity_detected": True  # 标记检测到模糊性
        }

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
        # HybridRetriever.retrieve() 返回 List[Document]
        results = retriever.retrieve(question, k=top_k)

        # 转换为字典格式便于序列化
        retrieved_docs = []

        for doc in results:
            doc_dict = {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            retrieved_docs.append(doc_dict)

        # 基于检索结果数量计算置信度
        if retrieved_docs:
            avg_confidence = min(len(retrieved_docs) / top_k, 1.0)
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


def clarification_generation_node(state: AgentState, llm=None) -> Dict[str, Any]:
    """
    澄清问题生成节点 (T036 - 增强版)

    功能:
    - 基于问题类型智能生成澄清问题
    - 根据检索结果和对话历史生成针对性问题
    - 使用 LLM 生成自然的澄清问题

    Args:
        state: 当前 Agent 状态
        llm: LLM 客户端实例

    Returns:
        更新后的状态字典,包含澄清问题
    """
    question = state.get("question", "")
    confidence = state.get("confidence_score", 0.0)
    retrieved_docs = state.get("retrieved_docs", [])
    conversation_history = state.get("messages", [])
    ambiguity_detected = state.get("ambiguity_detected", False)

    clarification_questions = []

    # 1. 空问题处理
    if not question.strip():
        clarification_questions.append("请提出一个具体的技术问题。")
        return {
            "need_clarification": True,
            "clarification_questions": clarification_questions
        }

    # 2. 基于模糊性类型生成澄清问题
    question_lower = question.lower()

    # 检测具体的模糊类型
    if "报错" in question or "错误" in question or "bug" in question_lower:
        # 错误相关问题
        clarification_questions.extend([
            "请问您遇到的具体错误信息是什么?",
            "您使用的是哪个模型或 API?",
            "能否提供完整的错误堆栈信息 (traceback)?"
        ])
    elif "怎么办" in question or "怎么用" in question or "如何" in question:
        # 操作指导类问题
        if "模型" not in question and "api" not in question_lower:
            clarification_questions.extend([
                "请问您想了解哪个具体的模型或功能?",
                "您的使用场景是什么?"
            ])
        else:
            clarification_questions.append("您具体想了解使用的哪个方面?(安装、配置、调用、参数设置等)")
    elif len(question.split()) < 5 or len([c for c in question if '\u4e00' <= c <= '\u9fff']) < 5:
        # 问题过短
        clarification_questions.extend([
            f"关于 '{question}',您能提供更多细节吗?",
            "您具体想了解哪方面的内容?"
        ])
    elif confidence < 0.3 and len(retrieved_docs) == 0:
        # 检索无结果
        clarification_questions.extend([
            "我在知识库中没有找到相关信息。",
            "您能换个方式描述您的问题吗?或者提供更多关键词?"
        ])
    elif confidence < 0.5 and len(retrieved_docs) > 0:
        # 检索结果不确定
        doc_sources = [doc.get("metadata", {}).get("source", "") for doc in retrieved_docs[:2]]
        clarification_questions.append(
            f"我找到了一些相关信息,但不确定是否完全符合您的需求。您是想了解关于 {', '.join(filter(None, doc_sources))} 的内容吗?"
        )
    else:
        # 通用澄清
        clarification_questions.append(f"关于 '{question[:30]}...',您具体想了解哪方面的内容?")

    # 3. 使用 LLM 生成更自然的澄清问题 (如果可用)
    if llm and ambiguity_detected:
        try:
            from agents.prompts import CLARIFICATION_PROMPT
            from langchain_core.messages import HumanMessage, SystemMessage

            # 构建上下文
            context = f"问题: {question}\n"
            if retrieved_docs:
                context += f"检索到 {len(retrieved_docs)} 个相关文档\n"
            if conversation_history:
                context += f"对话历史: {len(conversation_history) // 2} 轮\n"

            prompt = CLARIFICATION_PROMPT.format(
                question=question,
                current_understanding=context
            )

            messages = [
                SystemMessage(content="你是一个智能助手,擅长通过提问帮助用户澄清需求。"),
                HumanMessage(content=prompt)
            ]

            response = llm.invoke(messages)
            llm_questions = response.content.strip().split('\n')
            # 提取生成的问题 (过滤空行和编号)
            llm_questions = [
                q.strip('0123456789. -').strip()
                for q in llm_questions
                if q.strip() and '?' in q or '吗' in q or '呢' in q
            ]

            if llm_questions:
                # 使用 LLM 生成的问题,限制数量
                clarification_questions = llm_questions[:3]
        except Exception as e:
            print(f"LLM 澄清问题生成失败: {e},使用规则生成的问题")
            # 失败时使用规则生成的问题

    # 限制澄清问题数量 (最多3个)
    clarification_questions = clarification_questions[:3]

    return {
        "need_clarification": True,
        "clarification_questions": clarification_questions
    }


def clarify_node(state: AgentState, llm=None) -> Dict[str, Any]:
    """
    澄清节点:生成澄清问题 (兼容旧版本,调用新的 clarification_generation_node)

    Args:
        state: 当前 Agent 状态
        llm: LLM 客户端实例

    Returns:
        更新后的状态字典,包含澄清问题
    """
    return clarification_generation_node(state, llm)


def should_clarify(state: AgentState) -> str:
    """
    路由函数:判断是否需要澄清

    基于置信度和检索结果决定是否需要澄清

    Args:
        state: 当前 Agent 状态

    Returns:
        下一个节点名称 ("clarify" 或 "generate")
    """
    # 只在明确检测到模糊问题时才澄清
    # 其他情况（如知识库没有相关文档）应该尝试生成答案，让LLM根据通用知识回答
    if state.get("need_clarification", False):
        return "clarify"

    # 对于其他所有情况，都尝试生成答案
    # 即使没有检索到文档，LLM也可能根据通用知识给出有用的回答
    # 或者明确告诉用户知识库中没有相关信息
    return "generate"
