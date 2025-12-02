"""
Session Router
会话管理相关的 API 端点
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

router = APIRouter(prefix="/api/v2/sessions", tags=["Session"])


class SessionCreateRequest(BaseModel):
    """创建会话请求模型"""
    user_id: Optional[str] = None


class SessionInfo(BaseModel):
    """会话信息模型"""
    session_id: str
    user_id: str
    created_at: str
    last_active: str
    turn_count: int


class ConversationTurnInfo(BaseModel):
    """对话轮次信息模型"""
    question: str
    answer: str
    timestamp: str
    confidence: Optional[float] = None


class SessionDeleteResponse(BaseModel):
    """删除会话响应模型"""
    success: bool
    message: str


@router.get("", response_model=List[SessionInfo], summary="列出会话")
async def list_sessions(
    user_id: Optional[str] = None,
    app_request: Request = None
):
    """
    列出会话 (T043)

    Args:
        user_id: 可选的用户ID过滤，如果不提供则返回所有会话
        app_request: FastAPI Request 对象,用于访问 session_manager

    Returns:
        会话信息列表，按最后活跃时间倒序排序
    """
    try:
        # 从 app state 获取 session_manager
        session_manager = getattr(app_request.app.state, 'session_manager', None)

        if session_manager is None:
            raise HTTPException(
                status_code=503,
                detail="会话管理器未初始化,请检查服务配置"
            )

        # 获取会话列表
        sessions_data = session_manager.list_sessions(user_id=user_id)

        # 转换为响应格式
        sessions = []
        for session_data in sessions_data:
            sessions.append(
                SessionInfo(
                    session_id=session_data["session_id"],
                    user_id=session_data["user_id"],
                    created_at=session_data["created_at"],
                    last_active=session_data["last_active"],
                    turn_count=int(session_data.get("turn_count", 0))
                )
            )

        return sessions

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"列出会话失败: {str(e)}"
        )


@router.post("", response_model=SessionInfo, summary="创建新会话")
async def create_session(
    request: SessionCreateRequest = SessionCreateRequest(),
    app_request: Request = None
):
    """
    创建新会话 (T031)

    Args:
        request: 创建会话请求,可选指定 user_id
        app_request: FastAPI Request 对象,用于访问 session_manager

    Returns:
        会话信息,包含 session_id, created_at 等
    """
    try:
        # 从 app state 获取 session_manager
        session_manager = getattr(app_request.app.state, 'session_manager', None)

        if session_manager is None:
            raise HTTPException(
                status_code=503,
                detail="会话管理器未初始化,请检查服务配置"
            )

        # 创建会话
        session_id = session_manager.create_session(user_id=request.user_id)

        # 获取会话信息
        session_data = session_manager.get_session(session_id)

        if session_data is None:
            raise HTTPException(
                status_code=500,
                detail="会话创建失败,无法获取会话数据"
            )

        # 返回会话信息
        return SessionInfo(
            session_id=session_data["session_id"],
            user_id=session_data["user_id"],
            created_at=session_data["created_at"],
            last_active=session_data["last_active"],
            turn_count=int(session_data.get("turn_count", 0))
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"创建会话失败: {str(e)}"
        )


@router.get("/{session_id}", response_model=SessionInfo, summary="获取会话信息")
async def get_session(session_id: str, app_request: Request):
    """
    获取会话信息 (T031)

    Args:
        session_id: 会话ID
        app_request: FastAPI Request 对象

    Returns:
        会话信息
    """
    try:
        # 从 app state 获取 session_manager
        session_manager = getattr(app_request.app.state, 'session_manager', None)

        if session_manager is None:
            raise HTTPException(
                status_code=503,
                detail="会话管理器未初始化"
            )

        # 获取会话信息
        session_data = session_manager.get_session(session_id)

        if session_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"会话 {session_id} 不存在或已过期"
            )

        return SessionInfo(
            session_id=session_data["session_id"],
            user_id=session_data["user_id"],
            created_at=session_data["created_at"],
            last_active=session_data["last_active"],
            turn_count=int(session_data.get("turn_count", 0))
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"获取会话信息失败: {str(e)}"
        )


@router.get("/{session_id}/history", response_model=List[ConversationTurnInfo], summary="获取对话历史")
async def get_session_history(
    session_id: str,
    limit: Optional[int] = None,
    app_request: Request = None
):
    """
    获取会话对话历史 (T031)

    Args:
        session_id: 会话ID
        limit: 限制返回的轮数,None 表示返回全部
        app_request: FastAPI Request 对象

    Returns:
        对话历史列表
    """
    try:
        # 从 app state 获取 session_manager
        session_manager = getattr(app_request.app.state, 'session_manager', None)

        if session_manager is None:
            raise HTTPException(
                status_code=503,
                detail="会话管理器未初始化"
            )

        # 获取对话历史
        history = session_manager.get_conversation_history(session_id, limit=limit)

        # 转换为响应格式
        conversation_turns = []
        for turn in history:
            conversation_turns.append(
                ConversationTurnInfo(
                    question=turn.question,
                    answer=turn.answer,
                    timestamp=turn.timestamp.isoformat(),
                    confidence=turn.confidence
                )
            )

        return conversation_turns

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"获取对话历史失败: {str(e)}"
        )


@router.delete("/{session_id}", response_model=SessionDeleteResponse, summary="删除会话")
async def delete_session(session_id: str, app_request: Request):
    """
    删除会话 (T031)

    Args:
        session_id: 会话ID
        app_request: FastAPI Request 对象

    Returns:
        删除结果
    """
    try:
        # 从 app state 获取 session_manager
        session_manager = getattr(app_request.app.state, 'session_manager', None)

        if session_manager is None:
            raise HTTPException(
                status_code=503,
                detail="会话管理器未初始化"
            )

        # 删除会话
        success = session_manager.delete_session(session_id)

        if not success:
            return SessionDeleteResponse(
                success=False,
                message=f"会话 {session_id} 不存在或已被删除"
            )

        return SessionDeleteResponse(
            success=True,
            message=f"会话 {session_id} 已成功删除"
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"删除会话失败: {str(e)}"
        )
