"""
存储模块

提供统一的文件存储接口,支持多种存储后端:
- Local: 本地文件系统
- MinIO: MinIO 对象存储
- OSS: 阿里云对象存储 (待实现)
- S3: AWS S3 对象存储 (待实现)
"""

from storage.storage_manager import (
    StorageManager,
    StorageBackend,
    LocalStorage,
    MinIOStorage
)

__all__ = [
    "StorageManager",
    "StorageBackend",
    "LocalStorage",
    "MinIOStorage"
]
