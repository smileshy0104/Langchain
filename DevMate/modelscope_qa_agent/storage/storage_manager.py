"""
存储管理器

支持多种存储后端:
- Local: 本地文件系统
- MinIO: MinIO 对象存储
- OSS: 阿里云对象存储 (待实现)
- S3: AWS S3 对象存储 (待实现)
"""

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, BinaryIO, Dict, Any
from datetime import datetime
import hashlib

try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False

from config.config_loader import StorageConfig, MinIOStorageConfig, LocalStorageConfig


class StorageBackend(ABC):
    """存储后端抽象基类"""

    @abstractmethod
    def upload_file(
        self,
        file_data: BinaryIO,
        filename: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """上传文件

        Args:
            file_data: 文件数据流
            filename: 文件名
            content_type: 内容类型 (MIME type)
            metadata: 元数据

        Returns:
            str: 文件存储路径或 URL
        """
        pass

    @abstractmethod
    def download_file(self, file_path: str, destination: str) -> None:
        """下载文件

        Args:
            file_path: 存储中的文件路径
            destination: 本地目标路径
        """
        pass

    @abstractmethod
    def delete_file(self, file_path: str) -> bool:
        """删除文件

        Args:
            file_path: 存储中的文件路径

        Returns:
            bool: 是否删除成功
        """
        pass

    @abstractmethod
    def file_exists(self, file_path: str) -> bool:
        """检查文件是否存在

        Args:
            file_path: 存储中的文件路径

        Returns:
            bool: 文件是否存在
        """
        pass

    @abstractmethod
    def get_file_url(self, file_path: str, expires: int = 3600) -> str:
        """获取文件访问 URL

        Args:
            file_path: 存储中的文件路径
            expires: URL 过期时间 (秒)

        Returns:
            str: 文件访问 URL
        """
        pass


class LocalStorage(StorageBackend):
    """本地文件系统存储"""

    def __init__(self, config: LocalStorageConfig):
        """初始化本地存储

        Args:
            config: 本地存储配置
        """
        self.upload_path = Path(config.upload_path)
        self.upload_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ 本地存储初始化成功: {self.upload_path}")

    def upload_file(
        self,
        file_data: BinaryIO,
        filename: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """上传文件到本地存储

        Args:
            file_data: 文件数据流
            filename: 文件名
            content_type: 内容类型 (忽略)
            metadata: 元数据 (忽略)

        Returns:
            str: 文件存储路径
        """
        # 生成唯一文件名 (使用时间戳 + 哈希)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        safe_filename = f"{timestamp}_{file_hash}_{filename}"

        # 保存文件
        file_path = self.upload_path / safe_filename

        with open(file_path, 'wb') as f:
            shutil.copyfileobj(file_data, f)

        return str(file_path)

    def download_file(self, file_path: str, destination: str) -> None:
        """下载文件 (实际上是复制)

        Args:
            file_path: 源文件路径
            destination: 目标文件路径
        """
        shutil.copy2(file_path, destination)

    def delete_file(self, file_path: str) -> bool:
        """删除文件

        Args:
            file_path: 文件路径

        Returns:
            bool: 是否删除成功
        """
        try:
            Path(file_path).unlink(missing_ok=True)
            return True
        except Exception as e:
            print(f"⚠️  删除文件失败: {e}")
            return False

    def file_exists(self, file_path: str) -> bool:
        """检查文件是否存在

        Args:
            file_path: 文件路径

        Returns:
            bool: 文件是否存在
        """
        return Path(file_path).exists()

    def get_file_url(self, file_path: str, expires: int = 3600) -> str:
        """获取文件路径 (本地存储返回绝对路径)

        Args:
            file_path: 文件路径
            expires: 过期时间 (忽略)

        Returns:
            str: 文件绝对路径
        """
        return str(Path(file_path).absolute())


class MinIOStorage(StorageBackend):
    """MinIO 对象存储"""

    def __init__(self, config: MinIOStorageConfig):
        """初始化 MinIO 存储

        Args:
            config: MinIO 配置

        Raises:
            ImportError: minio 包未安装
            Exception: 连接失败
        """
        if not MINIO_AVAILABLE:
            raise ImportError(
                "MinIO 支持需要安装 minio 包:\n"
                "  pip install minio"
            )

        self.endpoint = config.endpoint
        self.bucket_name = config.bucket_name
        self.use_ssl = config.use_ssl

        # 初始化 MinIO 客户端
        self.client = Minio(
            endpoint=config.endpoint,
            access_key=config.access_key,
            secret_key=config.secret_key,
            secure=config.use_ssl
        )

        # 确保存储桶存在
        self._ensure_bucket_exists()

        print(f"✅ MinIO 存储初始化成功: {self.endpoint}/{self.bucket_name}")

    def _ensure_bucket_exists(self):
        """确保存储桶存在,不存在则创建"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                print(f"✅ 创建存储桶: {self.bucket_name}")
            else:
                print(f"✅ 存储桶已存在: {self.bucket_name}")
        except S3Error as e:
            raise Exception(f"检查/创建存储桶失败: {e}")

    def upload_file(
        self,
        file_data: BinaryIO,
        filename: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """上传文件到 MinIO

        Args:
            file_data: 文件数据流
            filename: 文件名
            content_type: 内容类型 (MIME type)
            metadata: 元数据

        Returns:
            str: 对象名称
        """
        # 生成唯一对象名 (使用日期分组)
        date_prefix = datetime.now().strftime("%Y/%m/%d")
        timestamp = datetime.now().strftime("%H%M%S")
        file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        object_name = f"{date_prefix}/{timestamp}_{file_hash}_{filename}"

        try:
            # 获取文件大小
            file_data.seek(0, 2)  # 移动到文件末尾
            file_size = file_data.tell()
            file_data.seek(0)  # 回到文件开头

            # 上传文件
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=file_data,
                length=file_size,
                content_type=content_type or "application/octet-stream",
                metadata=metadata
            )

            print(f"✅ 文件上传成功: {object_name}")
            return object_name

        except S3Error as e:
            raise Exception(f"MinIO 上传失败: {e}")

    def download_file(self, file_path: str, destination: str) -> None:
        """从 MinIO 下载文件

        Args:
            file_path: MinIO 中的对象名称
            destination: 本地目标路径
        """
        try:
            self.client.fget_object(
                bucket_name=self.bucket_name,
                object_name=file_path,
                file_path=destination
            )
            print(f"✅ 文件下载成功: {destination}")
        except S3Error as e:
            raise Exception(f"MinIO 下载失败: {e}")

    def delete_file(self, file_path: str) -> bool:
        """从 MinIO 删除文件

        Args:
            file_path: MinIO 中的对象名称

        Returns:
            bool: 是否删除成功
        """
        try:
            self.client.remove_object(
                bucket_name=self.bucket_name,
                object_name=file_path
            )
            print(f"✅ 文件删除成功: {file_path}")
            return True
        except S3Error as e:
            print(f"⚠️  MinIO 删除失败: {e}")
            return False

    def file_exists(self, file_path: str) -> bool:
        """检查 MinIO 中文件是否存在

        Args:
            file_path: MinIO 中的对象名称

        Returns:
            bool: 文件是否存在
        """
        try:
            self.client.stat_object(
                bucket_name=self.bucket_name,
                object_name=file_path
            )
            return True
        except S3Error:
            return False

    def get_file_url(self, file_path: str, expires: int = 3600) -> str:
        """获取文件预签名 URL

        Args:
            file_path: MinIO 中的对象名称
            expires: URL 过期时间 (秒)

        Returns:
            str: 预签名 URL
        """
        try:
            from datetime import timedelta
            url = self.client.presigned_get_object(
                bucket_name=self.bucket_name,
                object_name=file_path,
                expires=timedelta(seconds=expires)
            )
            return url
        except S3Error as e:
            raise Exception(f"生成预签名 URL 失败: {e}")


class StorageManager:
    """统一存储管理器

    根据配置自动选择存储后端 (Local, MinIO, OSS, S3)。
    """

    def __init__(self, config: StorageConfig):
        """初始化存储管理器

        Args:
            config: 存储配置

        Raises:
            ValueError: 不支持的存储类型
        """
        self.config = config
        self.backend: StorageBackend

        # 根据配置选择存储后端
        if config.type == "local":
            if not config.local:
                raise ValueError("local 存储需要配置 storage.local")
            self.backend = LocalStorage(config.local)

        elif config.type == "minio":
            if not config.minio:
                raise ValueError("minio 存储需要配置 storage.minio")
            self.backend = MinIOStorage(config.minio)

        elif config.type == "oss":
            raise NotImplementedError("OSS 存储后端待实现")

        elif config.type == "s3":
            raise NotImplementedError("S3 存储后端待实现")

        else:
            raise ValueError(f"不支持的存储类型: {config.type}")

    def upload_file(
        self,
        file_data: BinaryIO,
        filename: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """上传文件

        Args:
            file_data: 文件数据流
            filename: 文件名
            content_type: 内容类型
            metadata: 元数据

        Returns:
            str: 文件存储路径或对象名称
        """
        return self.backend.upload_file(file_data, filename, content_type, metadata)

    def download_file(self, file_path: str, destination: str) -> None:
        """下载文件"""
        self.backend.download_file(file_path, destination)

    def delete_file(self, file_path: str) -> bool:
        """删除文件"""
        return self.backend.delete_file(file_path)

    def file_exists(self, file_path: str) -> bool:
        """检查文件是否存在"""
        return self.backend.file_exists(file_path)

    def get_file_url(self, file_path: str, expires: int = 3600) -> str:
        """获取文件访问 URL"""
        return self.backend.get_file_url(file_path, expires)

    def validate_file(self, filename: str, file_size: int) -> tuple[bool, str]:
        """验证文件是否符合要求

        Args:
            filename: 文件名
            file_size: 文件大小 (字节)

        Returns:
            tuple[bool, str]: (是否有效, 错误信息)
        """
        # 检查文件扩展名
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.config.allowed_extensions:
            return False, f"不支持的文件类型: {file_ext}"

        # 检查文件大小
        max_size = self.config.parse_max_file_size()
        if file_size > max_size:
            return False, f"文件过大: {file_size} 字节 (最大: {max_size} 字节)"

        return True, ""


# 示例用法
if __name__ == "__main__":
    print("=" * 70)
    print("存储管理器测试")
    print("=" * 70)

    from config.config_loader import load_config
    from io import BytesIO

    try:
        # 加载配置
        config = load_config("config.yaml")
        storage_config = config.storage

        print(f"\n存储类型: {storage_config.type}")
        print(f"最大文件大小: {storage_config.max_file_size}")
        print(f"允许的扩展名: {', '.join(storage_config.allowed_extensions)}\n")

        # 初始化存储管理器
        storage_manager = StorageManager(storage_config)

        # 测试文件验证
        print("\n文件验证测试:")
        print("-" * 70)

        test_files = [
            ("document.pdf", 1024 * 1024),  # 1MB
            ("document.exe", 1024),         # 不支持的类型
            ("huge.pdf", 200 * 1024 * 1024) # 超大文件
        ]

        for filename, size in test_files:
            is_valid, error = storage_manager.validate_file(filename, size)
            status = "✅" if is_valid else "❌"
            print(f"{status} {filename} ({size} 字节): {error or '有效'}")

        # 测试文件上传 (仅创建测试数据,不实际上传)
        print("\n文件上传测试 (模拟):")
        print("-" * 70)
        print("创建测试文件数据...")

        test_content = b"This is a test document for storage manager."
        test_file = BytesIO(test_content)

        print(f"准备上传: test_document.txt ({len(test_content)} 字节)")
        print("(实际上传需要存储服务运行)")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
