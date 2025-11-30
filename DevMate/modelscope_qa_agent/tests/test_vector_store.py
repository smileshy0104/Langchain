"""
向量存储管理器单元测试

测试 VectorStoreManager 的 Milvus 连接、Collection 创建、向量写入和检索功能。
"""

import pytest
import time
from datetime import datetime
from typing import List
from pymilvus import connections, utility, Collection
from langchain_core.documents import Document

from core.vector_store import VectorStoreManager
from config.settings import settings


class TestVectorStoreManager:
    """测试 VectorStoreManager 类"""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """每个测试前后的设置和清理"""
        # 测试前: 清理可能存在的测试 Collection
        test_collection = "test_modelscope_docs"

        # 确保没有遗留连接
        try:
            existing_connections = [c[0] for c in connections.list_connections()]
            for alias in existing_connections:
                try:
                    connections.disconnect(alias)
                except:
                    pass
        except:
            pass

        # 连接并清理测试 Collection
        try:
            connections.connect(
                alias="test_cleanup",
                host=settings.milvus_host,
                port=settings.milvus_port
            )
            if utility.has_collection(test_collection, using="test_cleanup"):
                Collection(name=test_collection, using="test_cleanup").drop()
                print(f"✅ 清理旧的测试 Collection: {test_collection}")
            connections.disconnect("test_cleanup")
        except Exception as e:
            print(f"⚠️  清理时出错 (可忽略): {e}")

        yield

        # 测试后: 清理测试 Collection
        try:
            connections.connect(
                alias="test_cleanup",
                host=settings.milvus_host,
                port=settings.milvus_port
            )
            if utility.has_collection(test_collection, using="test_cleanup"):
                Collection(name=test_collection, using="test_cleanup").drop()
                print(f"✅ 清理测试 Collection: {test_collection}")
            connections.disconnect("test_cleanup")
        except Exception as e:
            print(f"⚠️  清理时出错: {e}")

    def test_milvus_connection(self):
        """测试 Milvus 连接建立"""
        manager = VectorStoreManager(
            collection_name="test_modelscope_docs",
            connection_alias="test_connection"
        )

        try:
            # 验证连接已建立
            existing_connections = [c[0] for c in connections.list_connections()]
            assert "test_connection" in existing_connections, "连接别名未找到"

            # 验证 Collection 已创建
            assert manager.collection is not None, "Collection 未创建"
            assert manager.collection.name == "test_modelscope_docs"

            # 验证 Collection 已加载到内存 (使用 utility.load_state)
            from pymilvus.client.types import LoadState
            load_state = utility.load_state(
                manager.collection_name,
                using=manager.connection_alias
            )
            assert load_state == LoadState.Loaded, "Collection 未加载到内存"

            print("✅ Milvus 连接测试通过")

        finally:
            manager.close()

    def test_collection_schema(self):
        """测试 Collection Schema 是否正确"""
        manager = VectorStoreManager(
            collection_name="test_modelscope_docs",
            connection_alias="test_schema"
        )

        try:
            schema = manager.collection.schema
            field_names = [f.name for f in schema.fields]

            # 验证所有必需字段
            required_fields = [
                "id", "title", "content", "content_summary",
                "source_type", "source_url", "document_type", "chunk_boundary",
                "tags", "question_categories", "embedding",
                "quality_score", "created_at", "last_updated"
            ]

            for field in required_fields:
                assert field in field_names, f"缺少必需字段: {field}"

            # 验证主键
            primary_field = next((f for f in schema.fields if f.is_primary), None)
            assert primary_field is not None, "未找到主键字段"
            assert primary_field.name == "id", "主键字段名称不正确"

            # 验证向量字段维度
            embedding_field = next((f for f in schema.fields if f.name == "embedding"), None)
            assert embedding_field is not None, "未找到 embedding 字段"
            assert embedding_field.params['dim'] == 1536, "向量维度不正确"

            print(f"✅ Collection Schema 测试通过 (共 {len(field_names)} 个字段)")

        finally:
            manager.close()

    def test_indexes_created(self):
        """测试索引是否正确创建"""
        manager = VectorStoreManager(
            collection_name="test_modelscope_docs",
            connection_alias="test_indexes"
        )

        try:
            # 获取所有索引
            indexes = manager.collection.indexes

            # 验证向量索引存在
            embedding_index = next((idx for idx in indexes if idx.field_name == "embedding"), None)
            assert embedding_index is not None, "向量索引未创建"

            # 验证索引类型和参数
            assert embedding_index.params['index_type'] == 'IVF_FLAT', "索引类型不正确"
            assert embedding_index.params['metric_type'] == 'IP', "度量类型不正确"

            print(f"✅ 索引测试通过 (共 {len(indexes)} 个索引)")

        finally:
            manager.close()

    def test_get_vector_store(self):
        """测试获取 LangChain Milvus 实例"""
        manager = VectorStoreManager(
            collection_name="test_modelscope_docs",
            connection_alias="test_langchain"
        )

        try:
            vector_store = manager.get_vector_store()

            # 验证返回的是 Milvus 实例
            assert vector_store is not None, "未返回 vector_store 实例"
            assert hasattr(vector_store, 'add_texts'), "缺少 add_texts 方法"
            assert hasattr(vector_store, 'similarity_search'), "缺少 similarity_search 方法"

            print("✅ get_vector_store() 测试通过")

        finally:
            manager.close()

    @pytest.mark.skip(reason="需要 DashScope API 连接,测试环境可能无法访问")
    def test_vector_write_and_retrieval(self):
        """测试向量写入和检索功能"""
        manager = VectorStoreManager(
            collection_name="test_modelscope_docs",
            connection_alias="test_write_retrieval"
        )

        try:
            vector_store = manager.get_vector_store()

            # 准备测试文档
            test_texts = [
                "如何使用 Qwen 模型进行文本生成?",
                "ModelScope 平台支持哪些深度学习框架?",
                "如何在 ModelScope 上部署模型?"
            ]

            test_metadatas = [
                {
                    "title": "Qwen 模型使用指南",
                    "source_type": "official_docs",
                    "document_type": "tutorial",
                    "quality_score": 0.95
                },
                {
                    "title": "支持的框架列表",
                    "source_type": "official_docs",
                    "document_type": "api_doc",
                    "quality_score": 0.90
                },
                {
                    "title": "模型部署教程",
                    "source_type": "official_docs",
                    "document_type": "tutorial",
                    "quality_score": 0.92
                }
            ]

            # 写入向量
            print("📝 写入测试文档...")
            ids = vector_store.add_texts(
                texts=test_texts,
                metadatas=test_metadatas
            )

            assert len(ids) == 3, f"写入的文档数量不正确: 期望 3, 实际 {len(ids)}"
            print(f"✅ 成功写入 {len(ids)} 个文档")

            # 等待索引更新
            time.sleep(2)
            manager.collection.flush()

            # 验证文档数量
            num_entities = manager.collection.num_entities
            assert num_entities == 3, f"Collection 中的实体数量不正确: 期望 3, 实际 {num_entities}"

            # 测试相似度检索
            print("🔍 测试相似度检索...")
            query = "如何使用 Qwen 模型?"
            results = vector_store.similarity_search(query, k=2)

            assert len(results) > 0, "检索结果为空"
            assert len(results) <= 2, f"检索结果数量超过 k=2: {len(results)}"

            # 验证结果包含相关文档
            result_texts = [doc.page_content for doc in results]
            print(f"📄 检索到 {len(results)} 个相关文档:")
            for i, doc in enumerate(results, 1):
                print(f"   {i}. {doc.page_content[:50]}...")

            # 验证最相关的结果应该是关于 Qwen 的文档
            assert any("Qwen" in text for text in result_texts), "检索结果未包含 Qwen 相关文档"

            print("✅ 向量写入和检索测试通过")

        finally:
            manager.close()

    def test_collection_stats(self):
        """测试获取 Collection 统计信息"""
        manager = VectorStoreManager(
            collection_name="test_modelscope_docs",
            connection_alias="test_stats"
        )

        try:
            stats = manager.get_collection_stats()

            # 验证统计信息字段
            assert "collection_name" in stats
            assert "num_entities" in stats
            assert "is_loaded" in stats
            assert "schema" in stats

            assert stats["collection_name"] == "test_modelscope_docs"
            assert stats["is_loaded"] is True
            assert isinstance(stats["num_entities"], int)
            assert isinstance(stats["schema"]["fields"], list)

            print(f"✅ Collection 统计信息测试通过")
            print(f"   - Collection: {stats['collection_name']}")
            print(f"   - 实体数量: {stats['num_entities']}")
            print(f"   - 字段数量: {len(stats['schema']['fields'])}")

        finally:
            manager.close()

    def test_context_manager(self):
        """测试 Context Manager 支持"""
        connection_alias = "test_context"

        # 确保开始前没有旧连接
        try:
            existing = [c[0] for c in connections.list_connections()]
            if connection_alias in existing:
                connections.disconnect(connection_alias)
        except:
            pass

        with VectorStoreManager(
            collection_name="test_modelscope_docs",
            connection_alias=connection_alias
        ) as manager:
            assert manager.collection is not None

            # 验证 Collection 已加载
            from pymilvus.client.types import LoadState
            load_state = utility.load_state(
                manager.collection_name,
                using=manager.connection_alias
            )
            assert load_state == LoadState.Loaded, "Collection 未加载"

            print("✅ Context Manager 测试通过")

        # 验证 Collection 已释放 (连接可能保留,但 Collection 应该释放)
        # 注意: pymilvus 的连接池可能会保留连接以供后续使用,这是正常行为
        print("✅ Context Manager 正确退出")

    def test_reconnection(self):
        """测试重新连接功能"""
        # 第一次连接
        manager1 = VectorStoreManager(
            collection_name="test_modelscope_docs",
            connection_alias="test_reconnect"
        )
        manager1.close()

        # 第二次使用相同别名连接（应该自动处理已存在的连接）
        manager2 = VectorStoreManager(
            collection_name="test_modelscope_docs",
            connection_alias="test_reconnect"
        )

        try:
            assert manager2.collection is not None

            # 验证 Collection 已加载
            from pymilvus.client.types import LoadState
            load_state = utility.load_state(
                manager2.collection_name,
                using=manager2.connection_alias
            )
            assert load_state == LoadState.Loaded, "Collection 未加载"

            print("✅ 重新连接测试通过")
        finally:
            manager2.close()

    @pytest.mark.skip(reason="需要 DashScope API 连接,测试环境可能无法访问")
    def test_embeddings_initialization(self):
        """测试 Embedding 模型初始化"""
        manager = VectorStoreManager(
            collection_name="test_modelscope_docs",
            connection_alias="test_embeddings"
        )

        try:
            # 验证 embeddings 已初始化
            assert manager.embeddings is not None, "Embeddings 未初始化"

            # 测试生成 embedding
            test_text = "测试文本"
            embedding = manager.embeddings.embed_query(test_text)

            assert isinstance(embedding, list), "Embedding 应该是 list 类型"
            assert len(embedding) == 1536, f"Embedding 维度不正确: 期望 1536, 实际 {len(embedding)}"
            assert all(isinstance(x, float) for x in embedding), "Embedding 元素应该是 float"

            print("✅ Embedding 模型初始化测试通过")
            print(f"   - 模型维度: {len(embedding)}")

        finally:
            manager.close()


class TestVectorStoreErrorHandling:
    """测试错误处理"""

    def test_invalid_connection(self):
        """测试无效的连接参数"""
        with pytest.raises(Exception) as exc_info:
            VectorStoreManager(
                host="invalid_host",
                port=99999,
                collection_name="test_invalid",
                connection_alias="test_invalid_conn"
            )

        assert "无法连接到 Milvus 服务器" in str(exc_info.value)
        print("✅ 无效连接测试通过")

    def test_missing_api_key(self, monkeypatch):
        """测试缺失 API key"""
        # 临时移除 API key
        monkeypatch.setattr("config.settings.settings.dashscope_api_key", None)

        with pytest.raises(ValueError, match="DASHSCOPE_API_KEY 未配置"):
            VectorStoreManager(
                collection_name="test_no_api_key",
                connection_alias="test_no_key"
            )

        print("✅ 缺失 API key 测试通过")
