"""
API Routers Package
模块化路由包
"""
from fastapi import APIRouter

# 导出所有路由器以便主应用注册
from api.routers.qa import router as qa_router
from api.routers.session import router as session_router
from api.routers.admin import router as admin_router

__all__ = ["qa_router", "session_router", "admin_router"]
