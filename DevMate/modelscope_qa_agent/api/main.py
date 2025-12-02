"""
FastAPI 主应用

提供魔搭社区智能答疑系统的 Web API 接口:
- 文档上传和管理
- 智能问答
- 系统状态监控
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import os

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

from config.config_loader import load_config
from services.document_upload_service import DocumentUploadService
from services.session_manager import SessionManager
from agents.simple_agent import create_agent
from core.llm_client import TongyiLLMClient
from api.routers import qa_router, session_router, admin_router

# 初始化 FastAPI 应用
app = FastAPI(
    title="魔搭社区智能答疑系统",
    description="基于 RAG 的智能问答系统,支持文档上传和自然语言问答",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录
static_dir = project_root / "api" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 注册路由
app.include_router(qa_router)
app.include_router(session_router)
app.include_router(admin_router)

# 全局服务实例
doc_service: Optional[DocumentUploadService] = None
session_manager: Optional[SessionManager] = None
qa_agent: Optional[Any] = None
llm_client: Optional[Any] = None


# ========== Pydantic Models ==========

class QuestionRequest(BaseModel):
    """问答请求模型"""
    question: str = Field(..., description="用户问题", min_length=1)
    session_id: Optional[str] = Field(None, description="会话ID")
    top_k: int = Field(3, description="检索文档数量", ge=1, le=10)


class AnswerResponse(BaseModel):
    """问答响应模型"""
    answer: str = Field(..., description="回答内容")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="来源文档")
    confidence: float = Field(..., description="置信度", ge=0.0, le=1.0)
    session_id: str = Field(..., description="会话ID")
    timestamp: str = Field(..., description="响应时间")


class SystemStatusResponse(BaseModel):
    """系统状态响应模型"""
    status: str = Field(..., description="系统状态: online/offline")
    milvus_connected: bool = Field(..., description="Milvus 连接状态")
    document_count: int = Field(..., description="文档数量")
    vector_dim: int = Field(..., description="向量维度")
    storage_type: str = Field(..., description="存储类型")
    ai_provider: str = Field(..., description="AI 服务提供商")
    timestamp: str = Field(..., description="查询时间")


# ========== Startup & Shutdown Events ==========

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化服务"""
    global doc_service, session_manager, qa_agent, llm_client

    print("\n" + "=" * 70)
    print("启动魔搭社区智能答疑系统 API 服务")
    print("=" * 70)

    try:
        # 加载配置
        config = load_config()
        print(f"✅ 配置加载成功")
        print(f"   - AI Provider: {config.ai.provider}")
        print(f"   - Storage Type: {config.storage.type}")
        print(f"   - Milvus: {config.milvus.host}:{config.milvus.port}")
        print(f"   - Redis: {config.redis.host}:{config.redis.port}")

        # 初始化 Redis 会话管理器
        try:
            session_manager = SessionManager()
            if session_manager.ping():
                print(f"✅ Redis 连接成功")
                print(f"   - TTL: {config.session.ttl}秒")
                print(f"   - 最大会话数/用户: {config.session.max_sessions_per_user}")
            else:
                print(f"⚠️  Redis 连接失败,会话功能将不可用")
                session_manager = None
        except Exception as e:
            print(f"⚠️  Redis 初始化失败: {e}")
            session_manager = None

        # 初始化文档上传服务
        doc_service = DocumentUploadService(config)
        print(f"✅ 文档上传服务初始化成功")

        # 初始化 LLM 客户端
        try:
            # 根据配置创建对应的 LLM 客户端
            if config.ai.provider == "volcengine":
                # 使用豆包 (VolcEngine) - 需要导入相应的客户端
                from langchain_community.chat_models import ChatTongyi
                # 创建一个简单的 wrapper
                class VolcEngineLLM:
                    def __init__(self, api_key, base_url, model_name, temperature, top_p):
                        # VolcEngine 使用兼容 OpenAI 的 API
                        from langchain_openai import ChatOpenAI
                        self.llm = ChatOpenAI(
                            api_key=api_key,
                            base_url=base_url,
                            model=model_name,
                            temperature=temperature,
                            top_p=top_p
                        )

                    def invoke(self, messages):
                        return self.llm.invoke(messages)

                llm_client = VolcEngineLLM(
                    api_key=config.ai.api_key,
                    base_url=config.ai.base_url,
                    model_name=config.ai.models["chat"],
                    temperature=config.ai.parameters["temperature"],
                    top_p=config.ai.parameters["top_p"]
                )
            else:
                # 使用通义千问
                from core.llm_client import TongyiLLMClient
                llm_client = TongyiLLMClient(
                    api_key=config.ai.api_key,
                    model_name=config.ai.models["chat"],
                    temperature=config.ai.parameters["temperature"],
                    top_p=config.ai.parameters["top_p"]
                )
            print(f"✅ LLM 客户端初始化成功 (提供商: {config.ai.provider}, 模型: {config.ai.models['chat']})")
        except Exception as e:
            print(f"⚠️  LLM 客户端初始化失败: {e}")
            import traceback
            traceback.print_exc()
            llm_client = None

        # 初始化 QA Agent
        try:
            # 获取检索器
            vector_store = doc_service.vector_store.get_vector_store()

            # 加载文档用于 BM25 (简化版本:使用向量存储中的文档)
            # 注意:生产环境应该维护一个专门的文档集合
            try:
                from retrievers.hybrid_retriever import HybridRetriever
                # 暂时使用空列表,实际应该从向量库加载
                retriever = HybridRetriever(
                    vector_store=vector_store,
                    documents=[],  # TODO: 加载文档
                    vector_weight=config.retrieval.vector_weight,
                    bm25_weight=config.retrieval.bm25_weight,
                    top_k=config.retrieval.top_k
                )
                print(f"✅ 混合检索器初始化成功")
            except Exception as e:
                print(f"⚠️  混合检索器初始化失败: {e}, 使用向量检索")
                retriever = None

            # 创建 Agent
            qa_agent = create_agent(retriever=retriever, llm=llm_client)
            # 将 agent 存储到 app.state 中
            app.state.qa_agent = qa_agent
            print(f"✅ QA Agent 初始化成功")
        except Exception as e:
            print(f"⚠️  QA Agent 初始化失败: {e}")
            qa_agent = None
            app.state.qa_agent = None

    except Exception as e:
        print(f"❌ 服务初始化失败: {e}")
        import traceback
        traceback.print_exc()
        # 不阻止应用启动,但服务将不可用

    print("=" * 70)
    print("API 服务已启动")
    print("=" * 70)
    print(f"访问地址: http://localhost:8000")
    print(f"API 文档: http://localhost:8000/docs")
    print(f"前端页面: http://localhost:8000/")
    print("=" * 70 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    print("\n关闭魔搭社区智能答疑系统 API 服务...")

    if doc_service:
        try:
            doc_service.vector_store.close()
            print("✅ 向量存储连接已关闭")
        except Exception as e:
            print(f"⚠️  关闭向量存储时出错: {e}")

    if session_manager:
        try:
            session_manager.redis_client.close()
            print("✅ Redis 连接已关闭")
        except Exception as e:
            print(f"⚠️  关闭 Redis 连接时出错: {e}")

    print("API 服务已关闭\n")


# ========== API Endpoints ==========

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """返回前端页面"""
    html_file = project_root / "api" / "static" / "index.html"

    if html_file.exists():
        return FileResponse(html_file)
    else:
        return HTMLResponse(content="""
        <html>
            <head><title>魔搭社区智能答疑系统</title></head>
            <body>
                <h1>魔搭社区智能答疑系统</h1>
                <p>前端页面未找到,请运行安装脚本创建前端文件。</p>
                <p>API 文档: <a href="/docs">/docs</a></p>
            </body>
        </html>
        """, status_code=200)


@app.get("/api/health", tags=["系统"])
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/status", response_model=SystemStatusResponse, tags=["系统"])
async def get_system_status():
    """获取系统状态"""
    if not doc_service:
        raise HTTPException(status_code=503, detail="服务未初始化")

    try:
        # 获取向量存储统计信息
        stats = doc_service.vector_store.get_collection_stats()

        config = doc_service.config

        return SystemStatusResponse(
            status="online",
            milvus_connected=stats["is_loaded"],
            document_count=stats["num_entities"],
            vector_dim=config.milvus.vector_dim,
            storage_type=config.storage.type,
            ai_provider=config.ai.provider,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取系统状态失败: {str(e)}")


@app.post("/api/upload", tags=["文档管理"])
async def upload_document(
    file: UploadFile = File(..., description="上传的文档文件"),
    category: str = Form("general", description="文档分类"),
    store_to_db: bool = Form(True, description="是否存储到向量数据库")
):
    """上传文档并处理"""
    if not doc_service:
        raise HTTPException(status_code=503, detail="服务未初始化")

    try:
        # 验证文件类型
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in doc_service.storage_manager.config.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {file_ext}"
            )

        # 读取文件数据
        file_data = await file.read()
        file_size = len(file_data)

        # 验证文件大小
        is_valid, error_msg = doc_service.storage_manager.validate_file(
            file.filename,
            file_size
        )
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        # 上传和处理文件
        from io import BytesIO
        file_stream = BytesIO(file_data)

        result = doc_service.upload_and_process(
            file_data=file_stream,
            filename=file.filename,
            metadata={"category": category, "source": "web_upload"},
            clean=True,
            split=True,
            calculate_score=True,
            store_to_vector_db=store_to_db
        )

        return {
            "message": "文档上传成功",
            "filename": file.filename,
            "file_size": file_size,
            "document_count": result["document_count"],
            "stored_to_db": store_to_db,
            "document_ids": result.get("document_ids", []) if store_to_db else [],
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")


@app.post("/api/question", response_model=AnswerResponse, tags=["问答"])
async def ask_question(request: QuestionRequest):
    """提交问题并获取答案 - 使用完整 QA Agent"""
    # 检查 QA Agent 是否已初始化
    qa_agent = getattr(app.state, 'qa_agent', None)

    if qa_agent is None:
        raise HTTPException(
            status_code=503,
            detail="QA Agent 未初始化,请检查服务配置"
        )

    try:
        # 生成或使用提供的 session_id
        import uuid
        session_id = request.session_id or str(uuid.uuid4())

        # 调用完整的 QA Agent
        from agents.simple_agent import invoke_agent

        result = invoke_agent(
            agent=qa_agent,
            question=request.question.strip(),
            session_id=session_id
        )

        # 提取 Agent 返回的结果
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

        # 确保 final_answer 不为 None
        if final_answer is None:
            final_answer = "抱歉,无法生成答案"

        # 构建来源信息
        sources = []
        if retrieved_docs:
            for doc in retrieved_docs[:request.top_k]:
                # 处理不同的文档格式
                if isinstance(doc, dict):
                    content = doc.get("content", "")
                    metadata = doc.get("metadata", {})
                    score = doc.get("score", 0.0)
                else:
                    # 如果是 LangChain Document 对象
                    content = getattr(doc, 'page_content', str(doc))
                    metadata = getattr(doc, 'metadata', {})
                    score = metadata.get('score', 0.0)

                source_info = {
                    "content": content[:200] + "..." if len(content) > 200 else content,
                    "source": metadata.get("source", "unknown"),
                    "score": float(score)
                }
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


@app.get("/api/documents", tags=["文档管理"])
async def list_documents(
    limit: int = 10,
    offset: int = 0
):
    """列出已上传的文档"""
    if not doc_service:
        raise HTTPException(status_code=503, detail="服务未初始化")

    try:
        stats = doc_service.vector_store.get_collection_stats()

        return {
            "total": stats["num_entities"],
            "limit": limit,
            "offset": offset,
            "documents": [],  # TODO: 实现文档列表查询
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


# ========== Main ==========

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
