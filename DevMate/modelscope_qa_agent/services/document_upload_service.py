"""
文档上传服务

提供完整的文档上传、处理和存储流程:
1. 文件上传到存储系统 (MinIO/Local)
2. 文档加载和解析
3. 文档清洗和分块
4. 质量评分
5. 向量化并存储到 Milvus
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, BinaryIO
from io import BytesIO

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config_loader import load_config, Config
from storage.storage_manager import StorageManager
from core.document_processor import DocumentProcessor
from core.vector_store import VectorStoreManager
from langchain_core.documents import Document


class DocumentUploadService:
    """文档上传服务

    提供完整的文档上传和处理流程。

    Attributes:
        config: 配置实例
        storage_manager: 存储管理器
        doc_processor: 文档处理器
        vector_store: 向量存储管理器
    """

    def __init__(self, config: Optional[Config] = None):
        """初始化文档上传服务

        Args:
            config: 配置实例 (可选,不提供则自动加载)
        """
        # 加载配置
        if config is None:
            try:
                from config.config_loader import get_config
                self.config = get_config()
            except RuntimeError:
                self.config = load_config()
        else:
            self.config = config

        print("=" * 70)
        print("初始化文档上传服务")
        print("=" * 70)

        # 1. 初始化存储管理器
        print("\n1. 初始化存储管理器")
        print("-" * 70)
        self.storage_manager = StorageManager(self.config.storage)

        # 2. 初始化文档处理器
        print("\n2. 初始化文档处理器")
        print("-" * 70)
        self.doc_processor = DocumentProcessor(
            chunk_size=self.config.agent.context_window_size,
            chunk_overlap=200
        )

        # 3. 初始化向量存储管理器
        print("\n3. 初始化向量存储管理器")
        print("-" * 70)
        self.vector_store = VectorStoreManager(
            use_new_config=True
        )

        print("\n" + "=" * 70)
        print("✅ 文档上传服务初始化完成")
        print("=" * 70)

    def upload_file(
        self,
        file_data: BinaryIO,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """上传文件到存储系统

        Args:
            file_data: 文件数据流
            filename: 文件名
            metadata: 文件元数据

        Returns:
            str: 存储路径或对象名称

        Raises:
            ValueError: 文件验证失败
        """
        # 获取文件大小
        file_data.seek(0, 2)
        file_size = file_data.tell()
        file_data.seek(0)

        # 验证文件
        is_valid, error_msg = self.storage_manager.validate_file(filename, file_size)
        if not is_valid:
            raise ValueError(f"文件验证失败: {error_msg}")

        # 上传文件
        print(f"\n📤 上传文件: {filename} ({file_size} 字节)")

        file_path = self.storage_manager.upload_file(
            file_data=file_data,
            filename=filename,
            content_type=None,  # 自动检测
            metadata=metadata
        )

        print(f"✅ 文件上传成功: {file_path}")

        return file_path

    def process_uploaded_file(
        self,
        file_path: str,
        metadata: Optional[Dict] = None,
        clean: bool = True,
        split: bool = True,
        calculate_score: bool = True
    ) -> List[Document]:
        """处理上传的文件

        Args:
            file_path: 文件路径 (MinIO对象路径或本地路径)
            metadata: 自定义元数据
            clean: 是否清洗
            split: 是否分块
            calculate_score: 是否评分

        Returns:
            List[Document]: 处理后的文档列表
        """
        print(f"\n🔧 处理文件: {file_path}")

        # 如果是 MinIO 存储,需要先下载到临时目录
        local_file_path = file_path
        temp_file = None

        if self.config.storage.type == "minio":
            import tempfile
            import os

            # 创建临时文件
            suffix = Path(file_path).suffix
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file.close()
            local_file_path = temp_file.name

            print(f"📥 从 MinIO 下载文件到临时目录: {local_file_path}")
            self.storage_manager.download_file(file_path, local_file_path)

        try:
            # 处理本地文件
            processed_docs = self.doc_processor.load_and_process_file(
                file_path=local_file_path,
                metadata=metadata,
                clean=clean,
                split=split,
                calculate_score=calculate_score
            )
        finally:
            # 清理临时文件
            if temp_file and os.path.exists(local_file_path):
                os.unlink(local_file_path)
                print(f"🗑️  删除临时文件: {local_file_path}")

        return processed_docs

    def store_documents(self, documents: List[Document]) -> List[str]:
        """将文档存储到向量数据库

        Args:
            documents: 文档列表

        Returns:
            List[str]: 文档 ID 列表
        """
        print(f"\n💾 存储文档到向量数据库...")
        print(f"   - 文档数量: {len(documents)}")

        # 获取 LangChain Milvus 实例
        vector_store = self.vector_store.get_vector_store()

        # 提取文本和元数据
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # 批量插入
        ids = vector_store.add_texts(texts, metadatas=metadatas)

        print(f"✅ 文档存储成功!")
        print(f"   - 已存储文档数: {len(ids)}")

        return ids

    def upload_and_process(
        self,
        file_data: BinaryIO,
        filename: str,
        metadata: Optional[Dict] = None,
        clean: bool = True,
        split: bool = True,
        calculate_score: bool = True,
        store_to_vector_db: bool = True
    ) -> Dict[str, Any]:
        """一站式文档上传和处理

        Args:
            file_data: 文件数据流
            filename: 文件名
            metadata: 文件元数据
            clean: 是否清洗
            split: 是否分块
            calculate_score: 是否评分
            store_to_vector_db: 是否存储到向量数据库

        Returns:
            Dict[str, Any]: 处理结果
                - file_path: 文件存储路径
                - document_count: 文档数量
                - document_ids: 文档 ID 列表 (如果存储到向量数据库)
        """
        print("\n" + "=" * 70)
        print("开始文档上传和处理流程")
        print("=" * 70)

        # 1. 上传文件
        file_path = self.upload_file(file_data, filename, metadata)

        # 2. 处理文档
        processed_docs = self.process_uploaded_file(
            file_path=file_path,
            metadata=metadata,
            clean=clean,
            split=split,
            calculate_score=calculate_score
        )

        result = {
            "file_path": file_path,
            "document_count": len(processed_docs),
            "documents": processed_docs
        }

        # 3. 存储到向量数据库 (可选)
        if store_to_vector_db:
            document_ids = self.store_documents(processed_docs)
            result["document_ids"] = document_ids

        print("\n" + "=" * 70)
        print("✅ 文档上传和处理流程完成!")
        print("=" * 70)
        print(f"   - 文件路径: {file_path}")
        print(f"   - 文档块数: {len(processed_docs)}")
        if store_to_vector_db:
            print(f"   - 已存储到向量数据库")

        return result

    def process_local_file(
        self,
        local_file_path: Union[str, Path],
        metadata: Optional[Dict] = None,
        clean: bool = True,
        split: bool = True,
        calculate_score: bool = True,
        store_to_vector_db: bool = True
    ) -> Dict[str, Any]:
        """处理本地文件 (不上传到存储系统)

        直接从本地文件系统加载和处理文档。

        Args:
            local_file_path: 本地文件路径
            metadata: 自定义元数据
            clean: 是否清洗
            split: 是否分块
            calculate_score: 是否评分
            store_to_vector_db: 是否存储到向量数据库

        Returns:
            Dict[str, Any]: 处理结果
        """
        print("\n" + "=" * 70)
        print("处理本地文件")
        print("=" * 70)

        # 1. 处理文档
        processed_docs = self.process_uploaded_file(
            file_path=str(local_file_path),
            metadata=metadata,
            clean=clean,
            split=split,
            calculate_score=calculate_score
        )

        result = {
            "file_path": str(local_file_path),
            "document_count": len(processed_docs),
            "documents": processed_docs
        }

        # 2. 存储到向量数据库 (可选)
        if store_to_vector_db:
            document_ids = self.store_documents(processed_docs)
            result["document_ids"] = document_ids

        print("\n" + "=" * 70)
        print("✅ 本地文件处理完成!")
        print("=" * 70)

        return result


# 示例用法
if __name__ == "__main__":
    print("=" * 70)
    print("文档上传服务测试")
    print("=" * 70)

    try:
        # 初始化服务
        service = DocumentUploadService()

        # 测试本地文件处理
        test_file = project_root / "test_document.txt"

        if test_file.exists():
            print("\n测试本地文件处理:")
            print("-" * 70)

            result = service.process_local_file(
                local_file_path=test_file,
                metadata={"category": "test", "source": "local"},
                clean=True,
                split=True,
                calculate_score=True,
                store_to_vector_db=False  # 暂不存储到向量数据库
            )

            print(f"\n处理结果:")
            print(f"   - 文件: {result['file_path']}")
            print(f"   - 文档块数: {result['document_count']}")

        else:
            print(f"\n⚠️  测试文件不存在: {test_file}")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
