"""
测试向量检索功能
"""
import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config_loader import load_config
from core.embeddings import VolcEngineEmbeddings
from pymilvus import Collection, connections

def test_retrieval():
    print("=" * 70)
    print("测试向量检索功能")
    print("=" * 70)

    # 1. 加载配置
    print("\n1. 加载配置...")
    config = load_config()
    print(f"✅ 配置加载成功")
    print(f"   - Milvus: {config.milvus.host}:{config.milvus.port}")
    print(f"   - Collection: {config.milvus.collection_name}")

    # 2. 连接 Milvus
    print("\n2. 连接 Milvus...")
    connections.connect(
        alias="default",
        host=config.milvus.host,
        port=config.milvus.port
    )
    collection = Collection(config.milvus.collection_name)
    collection.load()
    print(f"✅ 成功连接并加载 Collection")
    print(f"   - 总实体数: {collection.num_entities}")

    # 3. 初始化 Embeddings
    print("\n3. 初始化 Embeddings...")
    embeddings = VolcEngineEmbeddings(
        api_key=config.ai.api_key,
        base_url=config.ai.base_url,
        model=config.ai.models['embedding']
    )
    print(f"✅ Embeddings 初始化成功")

    # 4. 测试向量化
    print("\n4. 测试向量化...")
    test_text = "Chain"
    vector = embeddings.embed_query(test_text)
    print(f"✅ 向量化成功")
    print(f"   - 文本: {test_text}")
    print(f"   - 向量维度: {len(vector)}")

    # 5. 直接使用 Milvus 检索
    print("\n5. 使用 Milvus 直接检索...")
    search_params = {
        "metric_type": "IP",
        "params": {"nprobe": 10}
    }

    results = collection.search(
        data=[vector],
        anns_field="embedding",
        param=search_params,
        limit=5,
        output_fields=["title", "content", "source_type"]
    )

    print(f"✅ 检索完成,找到 {len(results[0])} 条结果:")
    for i, hit in enumerate(results[0]):
        print(f"\n   结果 {i+1}:")
        print(f"   - ID: {hit.id}")
        print(f"   - 距离: {hit.distance:.4f}")
        print(f"   - 标题: {hit.entity.get('title', 'N/A')}")
        print(f"   - 内容片段: {hit.entity.get('content', '')[:100]}...")
        print(f"   - 来源类型: {hit.entity.get('source_type', 'N/A')}")

    # 6. 测试更具体的问题
    print("\n\n6. 测试更具体的检索...")
    test_questions = [
        "LangChain Chain 是什么",
        "如何使用 Chain",
        "SimpleSequentialChain 的作用"
    ]

    for question in test_questions:
        print(f"\n   问题: {question}")
        q_vector = embeddings.embed_query(question)
        results = collection.search(
            data=[q_vector],
            anns_field="embedding",
            param=search_params,
            limit=3,
            output_fields=["title", "content"]
        )

        if len(results[0]) > 0:
            print(f"   ✅ 找到 {len(results[0])} 条相关结果")
            print(f"   最相关: {results[0][0].entity.get('content', '')[:80]}...")
        else:
            print(f"   ❌ 未找到相关结果")

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)

if __name__ == "__main__":
    test_retrieval()
