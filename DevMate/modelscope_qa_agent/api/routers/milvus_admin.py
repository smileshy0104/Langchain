"""
Milvus 管理路由
提供 Milvus 向量数据库的可视化管理功能
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pymilvus import connections, Collection, utility
import logging

router = APIRouter(prefix="/api/milvus", tags=["Milvus Admin"])
logger = logging.getLogger(__name__)


class CollectionInfo(BaseModel):
    """Collection 信息模型"""
    name: str
    description: str
    num_entities: int
    num_partitions: int
    schema: Dict[str, Any]
    loaded: bool


class SearchRequest(BaseModel):
    """向量搜索请求"""
    collection_name: str
    query_text: str
    top_k: int = 10


@router.get("/collections", response_model=List[str])
async def list_collections():
    """
    获取所有 Collection 列表

    Returns:
        Collection 名称列表
    """
    try:
        collections = utility.list_collections()
        return collections
    except Exception as e:
        logger.error(f"获取 Collection 列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取 Collection 列表失败: {str(e)}")


@router.get("/collections/{collection_name}", response_model=CollectionInfo)
async def get_collection_info(collection_name: str):
    """
    获取指定 Collection 的详细信息

    Args:
        collection_name: Collection 名称

    Returns:
        Collection 详细信息
    """
    try:
        # 检查 Collection 是否存在
        if not utility.has_collection(collection_name):
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' 不存在")

        # 获取 Collection
        collection = Collection(collection_name)

        # 获取统计信息
        collection.flush()
        num_entities = collection.num_entities

        # 获取 Schema 信息
        schema = collection.schema
        schema_dict = {
            "description": schema.description,
            "fields": [
                {
                    "name": field.name,
                    "type": str(field.dtype),
                    "description": field.description,
                    "is_primary": field.is_primary,
                    "auto_id": field.auto_id if hasattr(field, 'auto_id') else False,
                }
                for field in schema.fields
            ]
        }

        # 获取分区信息
        partitions = collection.partitions

        # 检查是否已加载
        loaded = utility.load_state(collection_name)

        return CollectionInfo(
            name=collection_name,
            description=schema.description,
            num_entities=num_entities,
            num_partitions=len(partitions),
            schema=schema_dict,
            loaded=str(loaded) == "Loaded"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取 Collection 信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取 Collection 信息失败: {str(e)}")


@router.get("/collections/{collection_name}/stats")
async def get_collection_stats(collection_name: str):
    """
    获取 Collection 的统计信息

    Args:
        collection_name: Collection 名称

    Returns:
        统计信息
    """
    try:
        if not utility.has_collection(collection_name):
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' 不存在")

        collection = Collection(collection_name)
        collection.flush()

        stats = {
            "total_entities": collection.num_entities,
            "partitions": [
                {
                    "name": partition.name,
                    "num_entities": partition.num_entities
                }
                for partition in collection.partitions
            ],
            "indexes": []
        }

        # 获取索引信息
        try:
            for field in collection.schema.fields:
                if field.dtype in [101, 102]:  # FloatVector, BinaryVector
                    index_info = collection.index(field.name)
                    if index_info:
                        stats["indexes"].append({
                            "field": field.name,
                            "index_type": index_info.params.get("index_type", "N/A"),
                            "metric_type": index_info.params.get("metric_type", "N/A"),
                        })
        except Exception as e:
            logger.warning(f"获取索引信息失败: {e}")

        return stats

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


@router.get("/collections/{collection_name}/sample")
async def get_sample_data(
    collection_name: str,
    limit: int = 10,
    offset: int = 0
):
    """
    获取 Collection 的样本数据

    Args:
        collection_name: Collection 名称
        limit: 返回数量
        offset: 偏移量

    Returns:
        样本数据
    """
    try:
        if not utility.has_collection(collection_name):
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' 不存在")

        collection = Collection(collection_name)

        # 确保 Collection 已加载
        if str(utility.load_state(collection_name)) != "Loaded":
            collection.load()

        # 查询数据
        # 使用简单的查询获取样本
        expr = f"id >= {offset}"
        output_fields = [field.name for field in collection.schema.fields if not field.dtype in [101, 102]]

        results = collection.query(
            expr=expr,
            output_fields=output_fields,
            limit=limit
        )

        return {
            "total": collection.num_entities,
            "offset": offset,
            "limit": limit,
            "data": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取样本数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取样本数据失败: {str(e)}")


@router.post("/collections/{collection_name}/load")
async def load_collection(collection_name: str):
    """
    加载 Collection 到内存

    Args:
        collection_name: Collection 名称

    Returns:
        操作结果
    """
    try:
        if not utility.has_collection(collection_name):
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' 不存在")

        collection = Collection(collection_name)
        collection.load()

        return {"message": f"Collection '{collection_name}' 已成功加载到内存"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"加载 Collection 失败: {e}")
        raise HTTPException(status_code=500, detail=f"加载 Collection 失败: {str(e)}")


@router.post("/collections/{collection_name}/release")
async def release_collection(collection_name: str):
    """
    从内存中释放 Collection

    Args:
        collection_name: Collection 名称

    Returns:
        操作结果
    """
    try:
        if not utility.has_collection(collection_name):
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' 不存在")

        collection = Collection(collection_name)
        collection.release()

        return {"message": f"Collection '{collection_name}' 已从内存中释放"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"释放 Collection 失败: {e}")
        raise HTTPException(status_code=500, detail=f"释放 Collection 失败: {str(e)}")


@router.get("/status")
async def get_milvus_status():
    """
    获取 Milvus 服务状态

    Returns:
        Milvus 服务状态
    """
    try:
        # 获取连接信息
        conn_info = connections.get_connection_addr("default")

        # 获取所有 Collection
        collections = utility.list_collections()

        total_entities = 0
        for coll_name in collections:
            try:
                coll = Collection(coll_name)
                coll.flush()
                total_entities += coll.num_entities
            except Exception as e:
                logger.warning(f"获取 Collection {coll_name} 实体数失败: {e}")

        return {
            "connected": True,
            "host": conn_info.get("host", "unknown"),
            "port": conn_info.get("port", "unknown"),
            "total_collections": len(collections),
            "total_entities": total_entities,
            "collections": collections
        }

    except Exception as e:
        logger.error(f"获取 Milvus 状态失败: {e}")
        return {
            "connected": False,
            "error": str(e)
        }
