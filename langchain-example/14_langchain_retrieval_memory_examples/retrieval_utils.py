"""Utilities for Milvus-backed retrieval examples."""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from typing import Iterable

from langchain_core.embeddings import Embeddings

# Milvus settings
DEFAULT_MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
EMBEDDING_DIM = int(os.getenv("LOCAL_HASH_EMBEDDING_DIM", "128"))
MILVUS_ALIAS = os.getenv("MILVUS_ALIAS", "default")


@dataclass
class RetrievedDocument:
    page_content: str
    metadata: dict

# Milvus vector store
class SimpleMilvusVectorStore:
    """Minimal Milvus-backed vector store for the runnable examples."""

    def __init__(
        self,
        *,
        collection_name: str,
        texts: list[str],
        metadatas: list[dict] | None = None,
        drop_old: bool = True,
    ) -> None:
        from pymilvus import DataType, MilvusClient

        self.collection_name = collection_name
        self.embeddings = HashEmbeddings()
        self.client = MilvusClient(uri=DEFAULT_MILVUS_URI)

        # Create collection
        if drop_old and self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)

        # Create collection schema
        if not self.client.has_collection(collection_name):
            schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.embeddings.dim)
            schema.add_field("text", DataType.VARCHAR, max_length=4096)
            schema.add_field("metadata", DataType.JSON)
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )

        self.add_texts(texts, metadatas=metadatas)

    # 增加文档
    def add_texts(self, texts: Iterable[str], metadatas: list[dict] | None = None) -> None:
        text_list = list(texts)
        if not text_list:
            return
        metadata_list = metadatas or [{} for _ in text_list]
        vectors = self.embeddings.embed_documents(text_list)
        data = [
            {"vector": vector, "text": text, "metadata": metadata}
            for text, vector, metadata in zip(text_list, vectors, metadata_list)
        ]
        self.client.insert(collection_name=self.collection_name, data=data)

    # 相似搜索
    def similarity_search(self, query: str, k: int = 4) -> list[RetrievedDocument]:
        query_vector = self.embeddings.embed_query(query)
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field="vector",
            limit=k,
            output_fields=["text", "metadata"],
            search_params={"metric_type": "COSINE"},
        )
        docs: list[RetrievedDocument] = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            docs.append(
                RetrievedDocument(
                    page_content=entity.get("text", ""),
                    metadata=entity.get("metadata", {}),
                )
            )
        return docs

# Embedding model
class HashEmbeddings(Embeddings):
    """Small deterministic embedding model for local Milvus examples.

    It avoids an external embedding API while still producing stable vectors for
    similarity search. This is for runnable demos, not production quality search.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

# 创建 Milvus 向量存储
def create_milvus_vectorstore(
    *,
    collection_name: str,
    texts: Iterable[str],
    metadatas: list[dict] | None = None,
    drop_old: bool = True,
):
    """Create a Milvus vector store from texts using local hash embeddings."""

    return SimpleMilvusVectorStore(
        texts=list(texts),
        metadatas=metadatas,
        collection_name=collection_name,
        drop_old=drop_old,
    )

# 格式化文档
def format_docs(docs) -> str:
    if not docs:
        return "未检索到相关文档。"
    return "\n\n".join(f"- {doc.page_content}" for doc in docs)
