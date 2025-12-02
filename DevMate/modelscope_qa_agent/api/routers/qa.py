"""
Q&A Router
问答相关的 API 端点
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import uuid

router = APIRouter(prefix="/api/v2/qa", tags=["Q&A"])


class QuestionRequest(BaseModel):
    """问题请求模型"""
    question: str
    session_id: Optional[str] = None
    top_k: int = 3


class SourceInfo(BaseModel):
    """来源信息模型"""
    content: str
    source: str
    score: float


class AnswerResponse(BaseModel):
    """答案响应模型"""
    answer: str
    sources: List[SourceInfo]
    confidence: float
    session_id: str
    timestamp: str


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest, app_request: Request):
    """
    提交问题并获取答案

    功能流程:
    1. 验证问题有效性
    2. 调用 QA Agent 进行问答
    3. 返回答案、来源和置信度

    Args:
        request: 问题请求
        app_request: FastAPI Request 对象,用于访问 app state

    Returns:
        答案响应,包含答案、来源文档和置信度
    """
    # 从 app state 获取 qa_agent
    qa_agent = getattr(app_request.app.state, 'qa_agent', None)

    if qa_agent is None:
        raise HTTPException(
            status_code=503,
            detail="QA Agent 未初始化,请检查服务配置"
        )

    # 验证问题
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="问题不能为空"
        )

    try:
        # 生成或使用提供的 session_id
        session_id = request.session_id or str(uuid.uuid4())

        # 调用 Agent
        from agents.simple_agent import invoke_agent
        result = invoke_agent(
            agent=qa_agent,
            question=request.question.strip(),
            session_id=session_id
        )

        # 提取结果
        final_answer = result.get("final_answer")
        confidence_score = result.get("confidence_score", 0.0)
        retrieved_docs = result.get("retrieved_docs", [])

        # 检查是否需要澄清
        if result.get("need_clarification") or final_answer is None:
            # 如果需要澄清,返回澄清问题作为答案
            clarification_questions = result.get("clarification_questions", [])
            if clarification_questions:
                final_answer = "需要更多信息来回答您的问题:\n" + "\n".join(
                    f"{i+1}. {q}" for i, q in enumerate(clarification_questions)
                )
            else:
                final_answer = "抱歉,我需要更多信息才能回答您的问题。请提供更多详细信息。"
            confidence_score = 0.0

        # 确保final_answer不为None
        if final_answer is None:
            final_answer = "抱歉,无法生成答案"

        # 构建来源信息
        sources = []
        if retrieved_docs:
            for doc in retrieved_docs[:request.top_k]:
                source_info = SourceInfo(
                    content=doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", ""),
                    source=doc.get("metadata", {}).get("source", "unknown"),
                    score=doc.get("score", 0.0)
                )
                sources.append(source_info)

        # 返回响应
        return AnswerResponse(
            answer=final_answer,
            sources=sources,
            confidence=confidence_score,
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        # 记录错误并返回友好的错误信息
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"问答处理失败: {str(e)}"
        )


@router.post("/ask/stream")
async def ask_question_stream(request: QuestionRequest):
    """
    提交问题并以流式方式获取答案 (SSE)

    Args:
        request: 问题请求

    Returns:
        Server-Sent Events 流
    """
    # TODO: Phase 4 实现流式输出
    raise HTTPException(status_code=501, detail="流式输出功能将在 Phase 4 实现")
