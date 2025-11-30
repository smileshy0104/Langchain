"""
混合检索器

实现向量检索和 BM25 关键词检索的混合策略。
通过加权融合两种检索方式,提高检索的准确性和召回率。
"""

from typing import List, Optional, Tuple, Dict
from collections import defaultdict
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_community.retrievers import BM25Retriever


class HybridRetriever:
    """混合检索器(向量 + BM25)

    结合语义向量检索和关键词检索,通过加权融合提升检索效果。

    Features:
        - 向量语义检索(Milvus)
        - BM25 关键词检索
        - 加权融合策略(可配置)
        - 结果重排序(可选)

    Attributes:
        vector_retriever: Milvus 向量检索器
        bm25_retriever: BM25 关键词检索器
        ensemble_retriever: LangChain 混合检索器
        vector_weight: 向量检索权重
        bm25_weight: BM25 检索权重
    """

    def __init__(
        self,
        vector_store: Milvus,
        documents: List[Document],
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        top_k: int = 10
    ):
        """初始化混合检索器

        Args:
            vector_store: Milvus 向量存储实例
            documents: BM25 检索器使用的文档列表
            vector_weight: 向量检索权重(默认 0.6)
            bm25_weight: BM25 检索权重(默认 0.4)
            top_k: 初步检索数量(默认 10)

        Raises:
            ValueError: 如果权重之和不为 1.0

        Example:
            >>> from core.vector_store import VectorStoreManager
            >>> manager = VectorStoreManager()
            >>> vector_store = manager.get_vector_store()
            >>> retriever = HybridRetriever(vector_store, documents)
        """
        # 验证权重
        if not abs(vector_weight + bm25_weight - 1.0) < 1e-6:
            raise ValueError(
                f"权重之和必须为 1.0, 当前: vector_weight={vector_weight}, "
                f"bm25_weight={bm25_weight}, sum={vector_weight + bm25_weight}"
            )

        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.top_k = top_k

        # 初始化向量检索器
        self.vector_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )

        # 初始化 BM25 关键词检索器
        if not documents:
            raise ValueError("documents 列表不能为空,BM25 检索器需要文档集合")

        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = top_k

        print(f"✅ HybridRetriever 初始化成功")
        print(f"   - 向量权重: {vector_weight}")
        print(f"   - BM25 权重: {bm25_weight}")
        print(f"   - Top-K: {top_k}")
        print(f"   - BM25 文档数: {len(documents)}")

    def retrieve(
        self,
        query: str,
        k: int = 3,
        filters: Optional[dict] = None
    ) -> List[Document]:
        """执行混合检索

        使用向量检索和 BM25 检索的加权组合检索相关文档。

        Args:
            query: 查询文本
            k: 返回的文档数量
            filters: 元数据过滤条件(可选)

        Returns:
            List[Document]: 检索到的文档列表(按相关性排序)

        Example:
            >>> results = retriever.retrieve("如何使用 Qwen 模型?", k=3)
            >>> print(f"检索到 {len(results)} 个相关文档")
        """
        if not query or not query.strip():
            raise ValueError("查询文本不能为空")

        # 执行混合检索(自定义加权融合)
        try:
            # 1. 分别获取向量检索和 BM25 检索结果
            vector_results = self._safe_retrieve(self.vector_retriever, query)
            bm25_results = self._safe_retrieve(self.bm25_retriever, query)

            # 2. 使用 RRF (Reciprocal Rank Fusion) 融合结果
            results = self._fuse_results(
                vector_results,
                bm25_results,
                self.vector_weight,
                self.bm25_weight
            )

        except Exception as e:
            print(f"⚠️  混合检索失败: {e}")
            # 降级策略:仅使用向量检索
            try:
                results = self._safe_retrieve(self.vector_retriever, query)
                print("   使用向量检索降级策略")
            except Exception as vector_error:
                print(f"⚠️  向量检索也失败: {vector_error}")
                return []

        # 应用元数据过滤(如果提供)
        if filters:
            results = self._apply_filters(results, filters)

        # 去重(基于内容相似度)
        results = self._deduplicate(results)

        # 返回 Top-K 结果
        return results[:k]

    def retrieve_with_scores(
        self,
        query: str,
        k: int = 3
    ) -> List[Tuple[Document, float]]:
        """执行混合检索并返回分数

        Args:
            query: 查询文本
            k: 返回的文档数量

        Returns:
            List[Tuple[Document, float]]: (文档, 相关性分数) 元组列表

        Note:
            由于 EnsembleRetriever 不直接提供分数,
            这里使用启发式方法估算分数(基于排名)
        """
        results = self.retrieve(query, k=k * 2)  # 获取更多结果以便评分

        # 使用排名倒数作为分数(Reciprocal Rank Fusion)
        scored_results = []
        for rank, doc in enumerate(results, start=1):
            # RRF 分数: 1 / (rank + 60)
            score = 1.0 / (rank + 60)
            scored_results.append((doc, score))

        return scored_results[:k]

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Document]:
        """重排序检索结果

        使用简单的启发式重排序策略。
        未来可以集成 Cross-encoder 模型进行精排。

        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            top_k: 返回的文档数量(None 表示全部)

        Returns:
            List[Document]: 重排序后的文档列表

        Reranking Strategy:
            1. 质量评分(quality_score): 权重 40%
            2. 文档类型优先级(document_type): 权重 30%
            3. 来源可信度(source_type): 权重 30%
        """
        if not documents:
            return []

        # 计算每个文档的重排序分数
        scored_docs = []
        for doc in documents:
            score = self._calculate_rerank_score(query, doc)
            scored_docs.append((doc, score))

        # 按分数降序排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 返回排序后的文档
        reranked = [doc for doc, score in scored_docs]

        if top_k is not None:
            return reranked[:top_k]
        return reranked

    def _calculate_rerank_score(self, query: str, doc: Document) -> float:
        """计算文档的重排序分数

        Args:
            query: 查询文本
            doc: 文档

        Returns:
            float: 重排序分数(0-1)
        """
        score = 0.0
        metadata = doc.metadata

        # 1. 质量评分(40%)
        quality_score = metadata.get("quality_score", 0.5)
        score += quality_score * 0.4

        # 2. 文档类型优先级(30%)
        doc_type = metadata.get("document_type", "unknown")
        type_scores = {
            "api_doc": 1.0,      # API 文档最优先
            "tutorial": 0.9,     # 教程次之
            "faq": 0.8,          # FAQ
            "troubleshooting": 0.85,  # 故障排查
            "example": 0.7,      # 示例
            "unknown": 0.5
        }
        score += type_scores.get(doc_type, 0.5) * 0.3

        # 3. 来源可信度(30%)
        source_type = metadata.get("source_type", "unknown")
        source_scores = {
            "official_docs": 1.0,
            "github_docs": 0.8,
            "qa_dataset": 0.6,
            "unknown": 0.3
        }
        score += source_scores.get(source_type, 0.3) * 0.3

        return score

    def _apply_filters(
        self,
        documents: List[Document],
        filters: dict
    ) -> List[Document]:
        """应用元数据过滤

        Args:
            documents: 文档列表
            filters: 过滤条件字典

        Returns:
            List[Document]: 过滤后的文档列表

        Example:
            >>> filters = {"source_type": "official_docs", "document_type": "tutorial"}
            >>> filtered = retriever._apply_filters(docs, filters)
        """
        filtered = []
        for doc in documents:
            match = True
            for key, value in filters.items():
                if doc.metadata.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append(doc)

        return filtered

    def _deduplicate(self, documents: List[Document]) -> List[Document]:
        """去除重复文档

        基于文档内容的相似度去重。
        如果两个文档内容相同或高度相似,只保留第一个。

        Args:
            documents: 文档列表

        Returns:
            List[Document]: 去重后的文档列表
        """
        if not documents:
            return []

        deduplicated = []
        seen_contents = set()

        for doc in documents:
            content_hash = hash(doc.page_content.strip())

            if content_hash not in seen_contents:
                deduplicated.append(doc)
                seen_contents.add(content_hash)

        if len(deduplicated) < len(documents):
            print(f"   去重: {len(documents)} → {len(deduplicated)} 个文档")

        return deduplicated

    def _safe_retrieve(self, retriever, query: str) -> List[Document]:
        """安全地执行检索(带异常处理)

        Args:
            retriever: 检索器实例
            query: 查询文本

        Returns:
            List[Document]: 检索结果,失败时返回空列表
        """
        try:
            return retriever.invoke(query)
        except Exception as e:
            print(f"⚠️  检索器 {type(retriever).__name__} 失败: {e}")
            return []

    def _fuse_results(
        self,
        vector_results: List[Document],
        bm25_results: List[Document],
        vector_weight: float,
        bm25_weight: float
    ) -> List[Document]:
        """使用 Reciprocal Rank Fusion (RRF) 融合检索结果

        RRF 公式: score(d) = Σ weight_i / (k + rank_i(d))
        其中 k=60 是常数,rank_i(d) 是文档 d 在第 i 个检索器中的排名

        Args:
            vector_results: 向量检索结果
            bm25_results: BM25 检索结果
            vector_weight: 向量检索权重
            bm25_weight: BM25 检索权重

        Returns:
            List[Document]: 融合后的文档列表(按 RRF 分数降序)
        """
        k = 60  # RRF 常数
        doc_scores = defaultdict(float)
        doc_objects = {}  # 存储文档对象

        # 计算向量检索的 RRF 分数
        for rank, doc in enumerate(vector_results, start=1):
            doc_hash = hash(doc.page_content)
            doc_scores[doc_hash] += vector_weight / (k + rank)
            doc_objects[doc_hash] = doc

        # 计算 BM25 检索的 RRF 分数
        for rank, doc in enumerate(bm25_results, start=1):
            doc_hash = hash(doc.page_content)
            doc_scores[doc_hash] += bm25_weight / (k + rank)
            if doc_hash not in doc_objects:
                doc_objects[doc_hash] = doc

        # 按分数降序排序
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 返回排序后的文档列表
        return [doc_objects[doc_hash] for doc_hash, score in sorted_docs]

    def update_weights(self, vector_weight: float, bm25_weight: float):
        """动态更新检索权重

        允许在运行时调整向量检索和 BM25 检索的权重比例。

        Args:
            vector_weight: 新的向量检索权重
            bm25_weight: 新的 BM25 检索权重

        Raises:
            ValueError: 如果权重之和不为 1.0

        Example:
            >>> retriever.update_weights(0.7, 0.3)  # 增加向量检索权重
        """
        if not abs(vector_weight + bm25_weight - 1.0) < 1e-6:
            raise ValueError(
                f"权重之和必须为 1.0, 当前: {vector_weight + bm25_weight}"
            )

        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        print(f"✅ 权重已更新: 向量={vector_weight}, BM25={bm25_weight}")

    def get_stats(self) -> dict:
        """获取检索器统计信息

        Returns:
            dict: 统计信息字典
        """
        return {
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
            "top_k": self.top_k,
            "bm25_doc_count": len(self.bm25_retriever.docs) if hasattr(self.bm25_retriever, 'docs') else 0
        }
