"""
向量数据库管理器

负责 Milvus 向量数据库的连接、Collection 创建、索引配置和 LangChain 集成。
提供统一的向量存储接口,支持文档插入、更新、删除和相似度检索。
支持多种 Embedding 服务: VolcEngine, DashScope, OpenAI
"""

from typing import Optional
import os
from pymilvus import (
    CollectionSchema, FieldSchema, DataType,
    Collection, connections, utility
)
from langchain_milvus import Milvus
from langchain_core.embeddings import Embeddings

# 支持旧配置和新配置
try:
    from config.settings import settings as old_settings
    OLD_CONFIG_AVAILABLE = True
except ImportError:
    OLD_CONFIG_AVAILABLE = False

try:
    from config.config_loader import load_config, get_config
    from core.embeddings import get_embeddings
    NEW_CONFIG_AVAILABLE = True
except ImportError:
    NEW_CONFIG_AVAILABLE = False


class VectorStoreManager:
    """Milvus 向量存储管理器

    管理 Milvus 向量数据库的连接和 Collection,提供 LangChain 兼容的向量存储接口。

    Features:
        - 自动创建和管理 Milvus Collection
        - 配置向量索引和标量字段索引
        - 支持多种 Embedding 服务 (VolcEngine, DashScope, OpenAI)
        - 提供 LangChain Milvus 实例

    Attributes:
        embeddings: Embedding 模型实例
        collection_name: Milvus collection 名称
        collection: Milvus Collection 对象
        connection_alias: Milvus 连接别名
        vector_dim: 向量维度
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: str = "modelscope_docs",
        connection_alias: str = "default",
        embeddings: Optional[Embeddings] = None,
        vector_dim: Optional[int] = None,
        use_new_config: bool = True
    ):
        """初始化向量存储管理器

        Args:
            host: Milvus 服务器地址,默认从配置读取
            port: Milvus 服务器端口,默认从配置读取
            collection_name: Collection 名称
            connection_alias: 连接别名,用于管理多个连接
            embeddings: Embeddings 实例 (可选,不提供则从配置创建)
            vector_dim: 向量维度 (可选,不提供则从配置读取)
            use_new_config: 是否使用新配置系统 (YAML)

        Raises:
            ConnectionError: 无法连接到 Milvus 服务器
            ValueError: 配置参数无效
        """
        self.collection_name = collection_name
        self.connection_alias = connection_alias
        self.use_new_config = use_new_config

        # 加载配置
        if use_new_config and NEW_CONFIG_AVAILABLE:
            # 使用新的 YAML 配置
            try:
                config = get_config()
            except RuntimeError:
                config = load_config()

            self.host = host or config.milvus.host
            self.port = port or config.milvus.port
            self.vector_dim = vector_dim or config.milvus.vector_dim

            # 获取 Embeddings
            if embeddings is None:
                self.embeddings = get_embeddings(config.ai)
            else:
                self.embeddings = embeddings

        elif OLD_CONFIG_AVAILABLE:
            # 使用旧的 .env 配置
            self.host = host or old_settings.milvus_host
            self.port = port or old_settings.milvus_port
            self.vector_dim = vector_dim or 1536  # DashScope 默认维度

            # 获取 Embeddings
            if embeddings is None:
                from langchain_community.embeddings import DashScopeEmbeddings
                self.embeddings = DashScopeEmbeddings(
                    model="text-embedding-v2",
                    dashscope_api_key=old_settings.dashscope_api_key
                )
            else:
                self.embeddings = embeddings

        else:
            raise RuntimeError(
                "无法加载配置。请确保以下之一可用:\n"
                "1. 新配置系统: config.yaml 和 config_loader.py\n"
                "2. 旧配置系统: .env 和 settings.py"
            )

        # 连接 Milvus
        self._connect_milvus()

        # 创建或加载 Collection
        self._init_collection()

    def _connect_milvus(self):
        """连接到 Milvus 服务器

        Raises:
            ConnectionError: 连接失败
        """
        try:
            # 如果连接已存在,先断开
            existing_connections = [c[0] for c in connections.list_connections()]
            if self.connection_alias in existing_connections:
                connections.disconnect(self.connection_alias)

            # 建立新连接
            connections.connect(
                alias=self.connection_alias,
                host=self.host,
                port=str(self.port)
            )

            print(f"✅ 成功连接到 Milvus: {self.host}:{self.port}")

        except Exception as e:
            raise ConnectionError(
                f"无法连接到 Milvus 服务器 {self.host}:{self.port}: {e}"
            ) from e

    def _init_collection(self):
        """初始化 Milvus Collection Schema

        创建完整的 Collection Schema,包含所有元数据字段。
        如果 Collection 已存在,则直接加载。

        Schema 字段（参考 design/data-model.md）:
            - id: 主键,VARCHAR(100)
            - title: 文档标题,VARCHAR(500)
            - content: 文档内容,VARCHAR(10000)
            - content_summary: 内容摘要,VARCHAR(1000)
            - source_type: 来源类型,VARCHAR(50)
            - source_url: 来源URL,VARCHAR(500)
            - document_type: 文档类型,VARCHAR(50)
            - chunk_boundary: 分块边界类型,VARCHAR(50)
            - tags: 标签数组,ARRAY<VARCHAR>
            - question_categories: 问题分类数组,ARRAY<VARCHAR>
            - embedding: 向量,FLOAT_VECTOR(1536)
            - quality_score: 质量评分,FLOAT
            - created_at: 创建时间,INT64 (Unix timestamp)
            - last_updated: 更新时间,INT64 (Unix timestamp)
        """
        # 检查 Collection 是否已存在
        if utility.has_collection(self.collection_name, using=self.connection_alias):
            self.collection = Collection(
                name=self.collection_name,
                using=self.connection_alias
            )
            print(f"✅ 加载已存在的 Collection: {self.collection_name}")

            # 确保 Collection 已加载到内存
            try:
                # 尝试加载 Collection (如果已加载,会静默忽略)
                self.collection.load()
                print(f"✅ Collection 已加载到内存")
            except Exception as e:
                # Collection 可能已经加载,忽略错误
                print(f"ℹ️  Collection 加载状态: {e}")
            return

        # 定义完整的 Schema
        fields = [
            # 主键 (使用 INT64 + auto_id 以兼容 LangChain)
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
                description="条目唯一标识符"
            ),

            # 文本字段
            FieldSchema(
                name="title",
                dtype=DataType.VARCHAR,
                max_length=500,
                nullable=True,  # 可选字段
                description="文档标题"
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=10000,
                description="文档内容（已分块）"
            ),
            FieldSchema(
                name="content_summary",
                dtype=DataType.VARCHAR,
                max_length=1000,
                nullable=True,  # 可选字段
                description="内容摘要"
            ),

            # 元数据字段
            FieldSchema(
                name="source_type",
                dtype=DataType.VARCHAR,
                max_length=50,
                nullable=True,  # 可选字段
                description="来源类型"
            ),
            FieldSchema(
                name="source_url",
                dtype=DataType.VARCHAR,
                max_length=500,
                nullable=True,  # 可选字段
                description="来源URL"
            ),
            FieldSchema(
                name="document_type",
                dtype=DataType.VARCHAR,
                max_length=50,
                nullable=True,  # 可选字段
                description="文档类型"
            ),
            FieldSchema(
                name="chunk_boundary",
                dtype=DataType.VARCHAR,
                max_length=50,
                nullable=True,  # 可选字段
                description="分块边界类型"
            ),

            # 数组字段
            FieldSchema(
                name="tags",
                dtype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=20,
                max_length=100,
                nullable=True,  # 可选字段
                description="标签列表"
            ),
            FieldSchema(
                name="question_categories",
                dtype=DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=10,
                max_length=100,
                nullable=True,  # 可选字段
                description="问题分类列表"
            ),

            # 向量字段
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.vector_dim,  # 根据配置的模型设置维度
                description="文档向量"
            ),

            # 数值字段
            FieldSchema(
                name="quality_score",
                dtype=DataType.FLOAT,
                nullable=True,  # 可选字段
                description="质量评分(0-1)"
            ),

            # 时间戳字段
            FieldSchema(
                name="created_at",
                dtype=DataType.INT64,
                nullable=True,  # 可选字段
                description="创建时间（Unix timestamp）"
            ),
            FieldSchema(
                name="last_updated",
                dtype=DataType.INT64,
                nullable=True,  # 可选字段
                description="最后更新时间（Unix timestamp）"
            ),
        ]

        # 创建 Schema
        schema = CollectionSchema(
            fields=fields,
            description="ModelScope Q&A Knowledge Base",
            enable_dynamic_field=True  # 允许动态字段
        )

        # 创建 Collection
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            using=self.connection_alias
        )

        print(f"✅ 创建新 Collection: {self.collection_name}")

        # 配置索引
        self._create_indexes()

        # 加载到内存
        self.collection.load()
        print(f"✅ Collection 已加载到内存")

    def _create_indexes(self):
        """配置向量索引和标量字段索引

        索引配置（参考 design/data-model.md:488-505）:
            - 向量索引: IVF_FLAT,适合中等规模数据
            - 标量索引: source_type, document_type, quality_score
        """
        # 向量索引参数
        vector_index_params = {
            "metric_type": "IP",  # 内积（适合归一化向量）
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }

        # 创建向量索引
        self.collection.create_index(
            field_name="embedding",
            index_params=vector_index_params
        )
        print("✅ 向量索引创建成功 (IVF_FLAT, IP)")

        # 创建标量字段索引（加速过滤查询）
        scalar_fields = ["source_type", "document_type", "quality_score"]
        for field in scalar_fields:
            try:
                self.collection.create_index(field_name=field)
                print(f"✅ 标量索引创建成功: {field}")
            except Exception as e:
                print(f"⚠️  标量索引创建失败 {field}: {e}")

    def get_vector_store(self) -> Milvus:
        """获取 LangChain Milvus 实例

        Returns:
            Milvus: LangChain 兼容的向量存储实例

        Example:
            >>> manager = VectorStoreManager()
            >>> vector_store = manager.get_vector_store()
            >>> vector_store.add_texts(["文档内容"], metadatas=[{"source": "test"}])
        """
        return Milvus(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_args={
                "host": self.host,
                "port": str(self.port),
                "alias": self.connection_alias
            },
            # 指定主键和文本字段
            primary_field="id",
            text_field="content",
            vector_field="embedding"
        )

    def get_collection_stats(self) -> dict:
        """获取 Collection 统计信息

        Returns:
            dict: 包含文档数量、索引状态等信息
        """
        # 使用 utility.load_state() 检查加载状态
        from pymilvus.client.types import LoadState
        load_state = utility.load_state(
            self.collection_name,
            using=self.connection_alias
        )

        stats = {
            "collection_name": self.collection_name,
            "num_entities": self.collection.num_entities,
            "is_loaded": load_state == LoadState.Loaded,
            "schema": {
                "fields": [f.name for f in self.collection.schema.fields],
                "description": self.collection.schema.description
            }
        }
        return stats

    def close(self):
        """关闭 Milvus 连接

        释放资源并断开连接。
        """
        try:
            # 释放 Collection
            if hasattr(self, 'collection') and self.collection:
                self.collection.release()
                print(f"✅ Collection {self.collection_name} 已释放")

            # 断开连接
            existing_connections = [c[0] for c in connections.list_connections()]
            if self.connection_alias in existing_connections:
                connections.disconnect(self.connection_alias)
                print(f"✅ Milvus 连接已断开")
        except Exception as e:
            print(f"⚠️  关闭连接时出错: {e}")

    def __enter__(self):
        """支持 context manager"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持 context manager"""
        self.close()
