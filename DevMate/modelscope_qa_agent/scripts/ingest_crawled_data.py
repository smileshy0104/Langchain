#!/usr/bin/env python3
"""
将爬取的数据导入向量数据库

使用方法:
    python scripts/ingest_crawled_data.py --input data/processed/all_documents.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from core.vector_store import VectorStoreManager
from config.config_loader import load_config, get_config


def load_processed_documents(input_file: str) -> List[Document]:
    """
    加载处理后的文档

    Args:
        input_file: JSONL文件路径

    Returns:
        Document列表
    """
    print(f"📂 加载文档: {input_file}")

    documents = []
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"❌ 文件不存在: {input_file}")
        return []

    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())

                content = data.get('content', '')
                metadata = data.get('metadata', {})

                if content:
                    doc = Document(
                        page_content=content,
                        metadata=metadata
                    )
                    documents.append(doc)

                if i % 100 == 0:
                    print(f"   已加载 {i} 个文档...")

            except Exception as e:
                print(f"⚠️  第 {i} 行解析失败: {e}")
                continue

    print(f"✅ 共加载 {len(documents)} 个文档")
    return documents


def ingest_documents(documents: List[Document], batch_size: int = 50):
    """
    导入文档到向量数据库

    Args:
        documents: 文档列表
        batch_size: 批处理大小
    """
    # 加载配置
    config = load_config()

    print("\n" + "=" * 70)
    print("开始导入文档到向量数据库")
    print(f"集合名称: {get_config('milvus.collection_name')}")
    print(f"文档数量: {len(documents)}")
    print(f"批处理大小: {batch_size}")
    print("=" * 70)

    try:
        # 初始化向量存储
        print("\n📊 初始化向量存储...")
        vector_store_manager = VectorStoreManager()
        vector_store = vector_store_manager.get_vector_store()

        # 批量导入
        total_batches = (len(documents) + batch_size - 1) // batch_size
        print(f"\n📥 开始批量导入 (共 {total_batches} 批)...")

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1

            print(f"\n批次 {batch_num}/{total_batches}: 导入 {len(batch)} 个文档...")

            try:
                # 使用add_documents方法
                vector_store.add_documents(batch)
                print(f"✅ 批次 {batch_num} 导入成功")

            except Exception as e:
                print(f"❌ 批次 {batch_num} 导入失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 获取统计信息
        print("\n" + "=" * 70)
        print("📊 导入统计")
        print("=" * 70)

        # 尝试获取集合统计
        try:
            from pymilvus import connections, Collection
            connections.connect(
                alias="default",
                uri=get_config('milvus.uri')
            )
            collection = Collection(get_config('milvus.collection_name'))
            collection.load()
            stats = collection.num_entities
            print(f"✅ 向量库中当前文档总数: {stats}")
        except Exception as e:
            print(f"⚠️  无法获取统计信息: {e}")

        print("\n✅ 导入完成!")

    except Exception as e:
        print(f"\n❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description="将爬取的数据导入向量数据库")

    parser.add_argument(
        "--input",
        default="data/processed/all_documents.jsonl",
        help="输入文件路径(JSONL格式)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="批处理大小"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅加载文档但不导入(用于测试)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("爬取数据导入向量数据库")
    print("=" * 70)

    # 加载文档
    documents = load_processed_documents(args.input)

    if not documents:
        print("❌ 没有文档可以导入")
        return

    # 显示文档统计
    print("\n📊 文档统计:")
    source_types = {}
    for doc in documents:
        source_type = doc.metadata.get('source_type', 'unknown')
        source_types[source_type] = source_types.get(source_type, 0) + 1

    for source_type, count in sorted(source_types.items()):
        print(f"   {source_type}: {count} 个文档")

    if args.dry_run:
        print("\n⚠️  Dry-run模式，不执行实际导入")
        return

    # 确认
    try:
        response = input("\n是否继续导入? (y/n): ")
        if response.lower() != 'y':
            print("❌ 用户取消导入")
            return
    except KeyboardInterrupt:
        print("\n❌ 用户取消导入")
        return

    # 导入文档
    ingest_documents(documents, args.batch_size)


if __name__ == "__main__":
    main()
