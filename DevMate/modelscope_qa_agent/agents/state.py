"""
Agent State Definition for LangGraph
定义 Agent 的状态结构
"""
from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import BaseMessage
from operator import add


class AgentState(TypedDict):
    """
    Agent 状态定义
    用于 LangGraph 管理对话流程
    """
    # 对话历史消息列表
    messages: Annotated[List[BaseMessage], add]

    # 当前用户问题
    question: str

    # 检索到的文档列表
    retrieved_docs: Optional[List[dict]]

    # 是否需要澄清
    need_clarification: bool

    # 澄清问题列表
    clarification_questions: Optional[List[str]]

    # 最终答案
    final_answer: Optional[str]

    # 置信度分数
    confidence_score: Optional[float]

    # 会话ID
    session_id: Optional[str]

    # 当前对话轮数
    turn_count: int
