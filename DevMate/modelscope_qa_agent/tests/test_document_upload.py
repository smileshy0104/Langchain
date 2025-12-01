"""
文档上传功能测试

测试文件上传、处理和存储的完整流程。
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.document_processor import DocumentProcessor


def test_file_upload_and_processing():
    """测试文件上传和处理流程"""
    print("=" * 70)
    print("文档上传功能测试")
    print("=" * 70)

    # 初始化文档处理器
    print("\n1. 初始化 DocumentProcessor")
    print("-" * 70)
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)

    # 测试文件路径
    test_file = project_root / "test_document.txt"

    if not test_file.exists():
        print(f"❌ 测试文件不存在: {test_file}")
        return

    # 测试加载文件
    print("\n2. 加载测试文件")
    print("-" * 70)

    try:
        documents = processor.load_uploaded_file(test_file)
        print(f"\n✅ 文件加载成功!")
        print(f"   - 文档数量: {len(documents)}")
        print(f"   - 总字符数: {sum(len(doc.page_content) for doc in documents)}")

        # 显示第一个文档的元数据
        if documents:
            print(f"\n第一个文档元数据:")
            for key, value in documents[0].metadata.items():
                print(f"   - {key}: {value}")

    except Exception as e:
        print(f"❌ 文件加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 测试完整处理流程
    print("\n3. 测试完整处理流程 (加载 + 清洗 + 分块 + 评分)")
    print("-" * 70)

    try:
        processed_docs = processor.load_and_process_file(
            test_file,
            metadata={"category": "test", "source": "upload"},
            clean=True,
            split=True,
            calculate_score=True
        )

        print(f"\n✅ 文档处理成功!")
        print(f"   - 处理后文档块数量: {len(processed_docs)}")

        # 显示前 3 个文档块的信息
        print(f"\n前 3 个文档块:")
        for i, doc in enumerate(processed_docs[:3]):
            print(f"\n   块 {i+1}:")
            print(f"   - 内容长度: {len(doc.page_content)} 字符")
            print(f"   - 质量评分: {doc.metadata.get('quality_score', 'N/A')}")
            print(f"   - 内容预览: {doc.page_content[:100]}...")

    except Exception as e:
        print(f"❌ 文档处理失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 70)
    print("✅ 文档上传功能测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    test_file_upload_and_processing()
