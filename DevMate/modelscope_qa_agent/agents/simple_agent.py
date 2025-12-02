"""
Simple QA Agent Factory
创建简单的单次问答 Agent (Phase 3)
"""
from typing import Optional
from functools import partial
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.nodes import (
    question_analysis_node,
    retrieval_node,
    answer_generation_node,
    clarify_node,
    should_clarify
)


def create_agent(retriever=None, llm=None):
    """
    创建单次问答 Agent

    构建 LangGraph 工作流:
    问题分析 → 检索 → 路由(澄清/生成) → 答案生成

    Args:
        retriever: HybridRetriever 实例,用于文档检索
        llm: LLM 客户端实例,用于答案生成

    Returns:
        CompiledGraph: 编译后的 Agent 工作流

    Example:
        >>> agent = create_agent(retriever=my_retriever, llm=my_llm)
        >>> result = agent.invoke({"question": "如何使用 Qwen 模型?"})
        >>> print(result["final_answer"])
    """
    # 创建状态图
    workflow = StateGraph(AgentState)

    # 使用 partial 绑定 retriever 和 llm 参数到节点函数
    question_analysis_node_with_deps = partial(question_analysis_node, llm=llm)
    retrieval_node_with_deps = partial(retrieval_node, retriever=retriever)
    answer_generation_node_with_deps = partial(answer_generation_node, llm=llm)

    # 添加节点
    workflow.add_node("analyze", question_analysis_node_with_deps)
    workflow.add_node("retrieve", retrieval_node_with_deps)
    workflow.add_node("generate", answer_generation_node_with_deps)
    workflow.add_node("clarify", clarify_node)

    # 设置入口点
    workflow.set_entry_point("analyze")

    # 定义边(状态转移)
    workflow.add_edge("analyze", "retrieve")

    # 条件路由:根据检索结果决定是否需要澄清
    workflow.add_conditional_edges(
        "retrieve",
        should_clarify,
        {
            "clarify": "clarify",
            "generate": "generate"
        }
    )

    # 澄清和生成都是终点
    workflow.add_edge("clarify", END)
    workflow.add_edge("generate", END)

    # 编译工作流
    app = workflow.compile()

    return app


def invoke_agent(
    agent,
    question: str,
    session_id: Optional[str] = None,
    conversation_history: Optional[list] = None
):
    """
    调用 Agent 进行问答 (支持多轮对话)

    便捷函数,用于调用编译后的 Agent

    Args:
        agent: 编译后的 Agent 工作流
        question: 用户问题
        session_id: 会话ID (可选)
        conversation_history: 对话历史消息列表 (用于多轮对话)

    Returns:
        dict: Agent 执行结果,包含 final_answer, confidence_score 等

    Example:
        >>> agent = create_agent(retriever=my_retriever, llm=my_llm)
        >>> result = invoke_agent(agent, "如何使用 Qwen 模型?")
        >>> print(result["final_answer"])
    """
    # 初始化状态
    initial_state = {
        "messages": conversation_history or [],
        "question": question,
        "retrieved_docs": None,
        "need_clarification": False,
        "clarification_questions": None,
        "final_answer": None,
        "confidence_score": None,
        "session_id": session_id,
        "turn_count": len(conversation_history) // 2 if conversation_history else 0,
        "conversation_summary": None
    }

    # 调用 Agent
    result = agent.invoke(initial_state)

    return result
