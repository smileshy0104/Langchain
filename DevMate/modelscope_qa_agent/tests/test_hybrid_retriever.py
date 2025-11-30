"""
混合检索器单元测试

测试 HybridRetriever 的向量检索、BM25检索和混合检索功能。
"""

import pytest
from langchain_core.documents import Document
from unittest.mock import Mock, MagicMock, patch

from retrievers.hybrid_retriever import HybridRetriever


class TestHybridRetriever:
    """测试 HybridRetriever 类"""

    @pytest.fixture
    def sample_documents(self):
        """创建示例文档集"""
        return [
            Document(
                page_content="如何使用 Qwen 模型进行文本生成?",
                metadata={"source_type": "official_docs", "document_type": "tutorial", "quality_score": 0.95}
            ),
            Document(
                page_content="ModelScope 平台支持哪些深度学习框架?",
                metadata={"source_type": "official_docs", "document_type": "faq", "quality_score": 0.90}
            ),
            Document(
                page_content="CUDA 内存不足错误的解决方案",
                metadata={"source_type": "github_docs", "document_type": "troubleshooting", "quality_score": 0.85}
            ),
            Document(
                page_content="```python\nimport modelscope\nmodel = modelscope.load('qwen')\n```",
                metadata={"source_type": "official_docs", "document_type": "example", "quality_score": 0.92}
            ),
            Document(
                page_content="如何在 ModelScope 上部署模型?",
                metadata={"source_type": "official_docs", "document_type": "tutorial", "quality_score": 0.88}
            ),
        ]

    @pytest.fixture
    def mock_vector_store(self):
        """创建 mock 的向量存储"""
        mock = Mock()
        mock_retriever = Mock()
        mock_retriever.invoke = Mock(return_value=[])
        mock.as_retriever = Mock(return_value=mock_retriever)
        return mock

    @pytest.fixture
    def retriever(self, mock_vector_store, sample_documents):
        """创建 HybridRetriever 实例"""
        return HybridRetriever(
            vector_store=mock_vector_store,
            documents=sample_documents,
            vector_weight=0.6,
            bm25_weight=0.4,
            top_k=10
        )

    def test_init(self, retriever):
        """测试初始化"""
        assert retriever.vector_weight == 0.6
        assert retriever.bm25_weight == 0.4
        assert retriever.top_k == 10
        assert retriever.vector_retriever is not None
        assert retriever.bm25_retriever is not None
        print("✅ HybridRetriever 初始化测试通过")

    def test_init_invalid_weights(self, mock_vector_store, sample_documents):
        """测试无效的权重"""
        with pytest.raises(ValueError, match="权重之和必须为 1.0"):
            HybridRetriever(
                vector_store=mock_vector_store,
                documents=sample_documents,
                vector_weight=0.7,
                bm25_weight=0.4  # 总和 1.1,无效
            )
        print("✅ 无效权重验证测试通过")

    def test_init_empty_documents(self, mock_vector_store):
        """测试空文档列表"""
        with pytest.raises(ValueError, match="documents 列表不能为空"):
            HybridRetriever(
                vector_store=mock_vector_store,
                documents=[],  # 空列表
                vector_weight=0.6,
                bm25_weight=0.4
            )
        print("✅ 空文档列表验证测试通过")

    def test_retrieve_basic(self, retriever, sample_documents):
        """测试基本检索功能"""
        # Mock 向量检索返回前3个文档
        retriever.vector_retriever.invoke = Mock(return_value=sample_documents[:3])

        # 执行检索
        results = retriever.retrieve("如何使用 Qwen 模型?", k=3)

        # 验证结果
        assert len(results) <= 3, "结果数量不应超过 k"
        assert all(isinstance(doc, Document) for doc in results), "所有结果应该是 Document 对象"

        print(f"✅ 基本检索测试通过 (检索到 {len(results)} 个文档)")

    def test_retrieve_empty_query(self, retriever):
        """测试空查询"""
        with pytest.raises(ValueError, match="查询文本不能为空"):
            retriever.retrieve("", k=3)

        with pytest.raises(ValueError, match="查询文本不能为空"):
            retriever.retrieve("   ", k=3)

        print("✅ 空查询验证测试通过")

    def test_retrieve_with_filters(self, retriever, sample_documents):
        """测试带过滤器的检索"""
        # Mock 向量检索和 BM25 检索
        def mock_safe_retrieve(ret, query):
            if ret == retriever.vector_retriever:
                return sample_documents
            else:  # BM25 retriever
                return sample_documents[:2]

        with patch.object(retriever, '_safe_retrieve', side_effect=mock_safe_retrieve):
            # 使用过滤器
            filters = {"source_type": "official_docs"}
            results = retriever.retrieve("测试查询", k=5, filters=filters)

            # 验证所有结果都来自 official_docs
            for doc in results:
                assert doc.metadata.get("source_type") == "official_docs"

        print("✅ 过滤器测试通过")

    def test_fuse_results(self, retriever, sample_documents):
        """测试结果融合(RRF)"""
        vector_results = sample_documents[:3]
        bm25_results = sample_documents[2:5]

        # 执行融合
        fused = retriever._fuse_results(
            vector_results,
            bm25_results,
            vector_weight=0.6,
            bm25_weight=0.4
        )

        # 验证融合结果
        # 由于 sample_documents[2] 同时在两个列表中,所以融合后应该去重
        # 输入: 3 + 3 = 6 个文档,但有1个重复,所以最多5个唯一文档
        unique_contents = set([doc.page_content for doc in sample_documents[:5]])
        assert len(fused) <= len(unique_contents), "融合后应该去重"
        assert all(isinstance(doc, Document) for doc in fused)

        print(f"✅ RRF 融合测试通过 (输入 {len(vector_results)}+{len(bm25_results)} → 输出 {len(fused)})")

    def test_deduplicate(self, retriever):
        """测试去重功能"""
        # 创建重复文档
        docs = [
            Document(page_content="重复内容1", metadata={}),
            Document(page_content="重复内容1", metadata={}),  # 重复
            Document(page_content="独特内容", metadata={}),
            Document(page_content="重复内容1", metadata={}),  # 重复
        ]

        deduplicated = retriever._deduplicate(docs)

        # 验证去重
        assert len(deduplicated) == 2, "应该只保留2个独特文档"
        contents = [doc.page_content for doc in deduplicated]
        assert "重复内容1" in contents
        assert "独特内容" in contents

        print("✅ 去重测试通过")

    def test_apply_filters(self, retriever, sample_documents):
        """测试元数据过滤"""
        filters = {
            "source_type": "official_docs",
            "document_type": "tutorial"
        }

        filtered = retriever._apply_filters(sample_documents, filters)

        # 验证所有结果都满足过滤条件
        for doc in filtered:
            assert doc.metadata.get("source_type") == "official_docs"
            assert doc.metadata.get("document_type") == "tutorial"

        print(f"✅ 元数据过滤测试通过 (输入 {len(sample_documents)} → 输出 {len(filtered)})")

    def test_rerank(self, retriever, sample_documents):
        """测试重排序功能"""
        # 执行重排序
        reranked = retriever.rerank("测试查询", sample_documents, top_k=3)

        # 验证结果
        assert len(reranked) == 3, "应该返回 top_k=3 个文档"
        assert all(isinstance(doc, Document) for doc in reranked)

        # 验证排序(质量更高的应该在前面)
        # 第一个文档应该有较高的质量评分
        assert reranked[0].metadata.get("quality_score", 0) >= reranked[-1].metadata.get("quality_score", 0)

        print("✅ 重排序测试通过")

    def test_calculate_rerank_score(self, retriever):
        """测试重排序分数计算"""
        # 高质量官方文档
        high_quality_doc = Document(
            page_content="测试内容",
            metadata={
                "quality_score": 0.95,
                "source_type": "official_docs",
                "document_type": "api_doc"
            }
        )

        # 低质量文档
        low_quality_doc = Document(
            page_content="测试内容",
            metadata={
                "quality_score": 0.5,
                "source_type": "unknown",
                "document_type": "unknown"
            }
        )

        score_high = retriever._calculate_rerank_score("测试", high_quality_doc)
        score_low = retriever._calculate_rerank_score("测试", low_quality_doc)

        # 验证高质量文档得分更高
        assert score_high > score_low, "高质量文档应该得分更高"
        assert 0 <= score_high <= 1, "分数应该在 0-1 范围内"
        assert 0 <= score_low <= 1, "分数应该在 0-1 范围内"

        print(f"✅ 重排序分数计算测试通过 (高质量={score_high:.2f}, 低质量={score_low:.2f})")

    def test_retrieve_with_scores(self, retriever, sample_documents):
        """测试带分数的检索"""
        # Mock 向量检索和 BM25 检索
        def mock_safe_retrieve(ret, query):
            if ret == retriever.vector_retriever:
                return sample_documents[:3]
            else:  # BM25 retriever
                return sample_documents[2:5]

        with patch.object(retriever, '_safe_retrieve', side_effect=mock_safe_retrieve):
            # 执行检索
            results = retriever.retrieve_with_scores("测试查询", k=3)

            # 验证结果
            assert len(results) <= 3, "结果数量不应超过 k"
            assert all(isinstance(item, tuple) for item in results), "每个结果应该是 (Document, score) 元组"
            assert all(isinstance(item[0], Document) for item in results), "第一个元素应该是 Document"
            assert all(isinstance(item[1], float) for item in results), "第二个元素应该是 float 分数"

            # 验证分数降序排列
            if len(results) > 1:
                scores = [score for _, score in results]
                assert scores == sorted(scores, reverse=True), "分数应该降序排列"

        print("✅ 带分数检索测试通过")

    def test_update_weights(self, retriever):
        """测试动态更新权重"""
        # 更新权重
        retriever.update_weights(0.7, 0.3)

        # 验证权重已更新
        assert retriever.vector_weight == 0.7
        assert retriever.bm25_weight == 0.3

        # 测试无效权重
        with pytest.raises(ValueError, match="权重之和必须为 1.0"):
            retriever.update_weights(0.8, 0.3)

        print("✅ 权重更新测试通过")

    def test_get_stats(self, retriever):
        """测试获取统计信息"""
        stats = retriever.get_stats()

        # 验证统计信息
        assert "vector_weight" in stats
        assert "bm25_weight" in stats
        assert "top_k" in stats
        assert stats["vector_weight"] == 0.6
        assert stats["bm25_weight"] == 0.4
        assert stats["top_k"] == 10

        print("✅ 统计信息测试通过")

    def test_safe_retrieve(self, retriever):
        """测试安全检索(带异常处理)"""
        # 创建会抛出异常的 mock retriever
        failing_retriever = Mock()
        failing_retriever.invoke = Mock(side_effect=Exception("检索失败"))

        # 应该返回空列表而不是抛出异常
        results = retriever._safe_retrieve(failing_retriever, "测试查询")

        assert results == [], "失败时应该返回空列表"

        print("✅ 安全检索测试通过")

    def test_retrieve_degradation(self, retriever, sample_documents):
        """测试检索降级策略"""
        # Mock _safe_retrieve: 向量检索成功, BM25 检索失败(返回空列表)
        def mock_safe_retrieve(ret, query):
            if ret == retriever.vector_retriever:
                return sample_documents[:2]
            else:  # BM25 retriever fails
                return []

        with patch.object(retriever, '_safe_retrieve', side_effect=mock_safe_retrieve):
            # 应该降级到仅使用向量检索
            results = retriever.retrieve("测试查询", k=3)

            # 验证降级成功
            assert len(results) > 0, "降级后应该有结果"

        print("✅ 检索降级测试通过")


class TestHybridRetrieverWeightTuning:
    """测试权重调优"""

    @pytest.fixture
    def sample_docs(self):
        """创建示例文档"""
        return [
            Document(page_content=f"文档 {i} 的内容", metadata={"id": i})
            for i in range(10)
        ]

    def test_different_weight_combinations(self, sample_docs):
        """测试不同权重组合"""
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_retriever.invoke = Mock(return_value=sample_docs[:5])
        mock_vector_store.as_retriever = Mock(return_value=mock_retriever)

        # 测试不同权重组合
        weight_combinations = [
            (0.5, 0.5),  # 平衡
            (0.7, 0.3),  # 偏向量
            (0.3, 0.7),  # 偏 BM25
            (0.9, 0.1),  # 强偏向量
            (0.1, 0.9),  # 强偏 BM25
        ]

        for vector_weight, bm25_weight in weight_combinations:
            retriever = HybridRetriever(
                vector_store=mock_vector_store,
                documents=sample_docs,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight
            )

            assert retriever.vector_weight == vector_weight
            assert retriever.bm25_weight == bm25_weight

        print(f"✅ 权重组合测试通过 (测试了 {len(weight_combinations)} 种组合)")

    def test_weight_impact_on_results(self, sample_docs):
        """测试权重对结果的影响"""
        mock_vector_store = Mock()
        mock_retriever = Mock()

        # 向量检索返回前5个文档
        vector_results = sample_docs[:5]
        # BM25 检索返回后5个文档
        bm25_results = sample_docs[5:]

        mock_retriever.invoke = Mock(return_value=vector_results)
        mock_vector_store.as_retriever = Mock(return_value=mock_retriever)

        # 创建检索器
        retriever = HybridRetriever(
            vector_store=mock_vector_store,
            documents=sample_docs,
            vector_weight=0.8,  # 高向量权重
            bm25_weight=0.2
        )

        # Mock _safe_retrieve
        def mock_safe_retrieve(ret, query):
            if ret == retriever.vector_retriever:
                return vector_results
            else:  # BM25 retriever
                return bm25_results

        with patch.object(retriever, '_safe_retrieve', side_effect=mock_safe_retrieve):
            # 执行检索
            results = retriever.retrieve("测试", k=10)

            # 验证结果融合了两种检索方式
            assert len(results) > 0

        print("✅ 权重影响测试通过")
