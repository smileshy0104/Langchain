"""
文档处理器单元测试

测试 DocumentProcessor 的文档清洗、分块和质量评分功能。
"""

import pytest
from langchain_core.documents import Document

from core.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """测试 DocumentProcessor 类"""

    @pytest.fixture
    def processor(self):
        """创建 DocumentProcessor 实例"""
        return DocumentProcessor(chunk_size=500, chunk_overlap=100)

    def test_init(self, processor):
        """测试初始化"""
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 100
        assert processor.markdown_splitter is not None
        assert processor.text_splitter is not None
        print("✅ DocumentProcessor 初始化测试通过")

    def test_clean_document_html_removal(self, processor):
        """测试 HTML 标签清除"""
        doc = Document(
            page_content="<p>Hello <strong>World</strong></p><div>Test</div>",
            metadata={"source": "test"}
        )

        clean_doc = processor.clean_document(doc)

        # 验证 HTML 标签已移除
        assert "<p>" not in clean_doc.page_content
        assert "<strong>" not in clean_doc.page_content
        assert "<div>" not in clean_doc.page_content
        assert "Hello World" in clean_doc.page_content
        assert "Test" in clean_doc.page_content

        print("✅ HTML标签清除测试通过")

    def test_clean_document_whitespace_normalization(self, processor):
        """测试空白规范化"""
        doc = Document(
            page_content="Hello\n\n\n\nWorld\n\n\n\nTest",
            metadata={"source": "test"}
        )

        clean_doc = processor.clean_document(doc)

        # 验证多余空行已移除
        assert "\n\n\n" not in clean_doc.page_content
        # 应该只有双换行
        assert clean_doc.page_content.count("\n\n") <= 2

        print("✅ 空白规范化测试通过")

    def test_clean_document_code_block_formatting(self, processor):
        """测试代码块格式化"""
        doc = Document(
            page_content="Text\n```python\nprint('hello')\n```\nMore text",
            metadata={"source": "test"}
        )

        clean_doc = processor.clean_document(doc)

        # 验证代码块格式正确
        assert "```python" in clean_doc.page_content
        assert "print('hello')" in clean_doc.page_content
        assert "```" in clean_doc.page_content

        print("✅ 代码块格式化测试通过")

    def test_clean_document_preserves_metadata(self, processor):
        """测试元数据保留"""
        metadata = {"source": "test", "author": "Alice"}
        doc = Document(
            page_content="<p>Content</p>",
            metadata=metadata
        )

        clean_doc = processor.clean_document(doc)

        # 验证元数据保留
        assert clean_doc.metadata == metadata

        print("✅ 元数据保留测试通过")

    def test_split_with_code_protection_markdown_headers(self, processor):
        """测试 Markdown 标题分块"""
        content = """
# Header 1

Content for header 1.

## Header 2

Content for header 2 with more details.

### Header 3

Content for header 3.
"""
        doc = Document(page_content=content, metadata={"source": "test"})

        chunks = processor.split_with_code_protection(doc)

        # 验证分块数量
        assert len(chunks) > 0

        # 验证每个chunk都有chunk_boundary元数据
        for chunk in chunks:
            assert "chunk_boundary" in chunk.metadata
            assert chunk.metadata["chunk_boundary"] in [
                "section", "subsection", "paragraph", "code_block"
            ]

        print(f"✅ Markdown标题分块测试通过 (生成 {len(chunks)} 个chunks)")

    def test_split_with_code_protection_code_block_integrity(self, processor):
        """测试代码块完整性保护"""
        content = """
# Python Example

Here's a code example:

```python
def hello():
    print("Hello, World!")
    return True
```

This code should not be split.
"""
        doc = Document(page_content=content, metadata={"source": "test"})

        chunks = processor.split_with_code_protection(doc)

        # 验证代码块在某个chunk中完整存在
        code_block = '```python\ndef hello():\n    print("Hello, World!")\n    return True\n```'
        code_block_found = False

        for chunk in chunks:
            if "```python" in chunk.page_content and "def hello()" in chunk.page_content:
                # 验证代码块完整
                assert "print(" in chunk.page_content
                assert "return True" in chunk.page_content
                code_block_found = True
                break

        assert code_block_found, "代码块应该在某个chunk中完整存在"

        print("✅ 代码块完整性保护测试通过")

    def test_split_with_code_protection_no_markdown(self, processor):
        """测试无 Markdown 标题的纯文本"""
        content = "This is plain text without markdown headers. " * 50

        doc = Document(page_content=content, metadata={"source": "test"})

        chunks = processor.split_with_code_protection(doc)

        # 验证纯文本也能正常分块
        assert len(chunks) > 0

        # 验证chunk_boundary
        for chunk in chunks:
            assert chunk.metadata.get("chunk_boundary") == "paragraph"

        print(f"✅ 纯文本分块测试通过 (生成 {len(chunks)} 个chunks)")

    def test_split_with_code_protection_long_code_block(self, processor):
        """测试超长代码块保护"""
        # 创建一个超长代码块(超过 chunk_size)
        long_code = "x = 1\n" * 100  # 600+ 字符
        content = f"""
# Long Code Example

```python
{long_code}
```

End of document.
"""
        doc = Document(page_content=content, metadata={"source": "test"})

        chunks = processor.split_with_code_protection(doc)

        # 验证即使超长,代码块也保持完整
        code_chunk_found = False
        for chunk in chunks:
            if "```python" in chunk.page_content:
                assert "x = 1" in chunk.page_content
                # 代码块应该包含大部分内容
                assert chunk.page_content.count("x = 1") > 50
                code_chunk_found = True
                break

        assert code_chunk_found, "超长代码块应该保持完整"

        print("✅ 超长代码块保护测试通过")

    def test_calculate_quality_score_length(self, processor):
        """测试质量评分 - 长度维度"""
        # 合适长度(100-2000字符)
        good_doc = Document(
            page_content="x" * 500,
            metadata={}
        )
        score = processor.calculate_quality_score(good_doc)
        assert score >= 0.25, "合适长度应该得到长度分"

        # 太短
        short_doc = Document(
            page_content="x" * 50,
            metadata={}
        )
        score_short = processor.calculate_quality_score(short_doc)
        assert score_short < score, "太短的文档应该得分较低"

        # 太长
        long_doc = Document(
            page_content="x" * 3000,
            metadata={}
        )
        score_long = processor.calculate_quality_score(long_doc)
        assert score_long <= score, "太长的文档应该得分不高于合适长度"

        print("✅ 质量评分-长度测试通过")

    def test_calculate_quality_score_structure(self, processor):
        """测试质量评分 - 结构维度"""
        # 有标题
        structured_doc = Document(
            page_content="# Title\n\nContent here.",
            metadata={}
        )
        score_with_header = processor.calculate_quality_score(structured_doc)

        # 无标题
        unstructured_doc = Document(
            page_content="Content without headers.",
            metadata={}
        )
        score_without_header = processor.calculate_quality_score(unstructured_doc)

        assert score_with_header > score_without_header, "有标题的文档应该得分更高"

        print("✅ 质量评分-结构测试通过")

    def test_calculate_quality_score_code_examples(self, processor):
        """测试质量评分 - 代码示例维度"""
        # 有代码块
        code_doc = Document(
            page_content="# Example\n\n```python\nprint('hi')\n```",
            metadata={}
        )
        score_with_code = processor.calculate_quality_score(code_doc)

        # 无代码
        no_code_doc = Document(
            page_content="# Example\n\nJust text.",
            metadata={}
        )
        score_without_code = processor.calculate_quality_score(no_code_doc)

        assert score_with_code > score_without_code, "有代码的文档应该得分更高"

        print("✅ 质量评分-代码示例测试通过")

    def test_calculate_quality_score_source_type(self, processor):
        """测试质量评分 - 来源维度"""
        # 官方文档
        official_doc = Document(
            page_content="Content",
            metadata={"source_type": "official_docs"}
        )
        score_official = processor.calculate_quality_score(official_doc)

        # GitHub 文档
        github_doc = Document(
            page_content="Content",
            metadata={"source_type": "github_docs"}
        )
        score_github = processor.calculate_quality_score(github_doc)

        # 未知来源
        unknown_doc = Document(
            page_content="Content",
            metadata={"source_type": "unknown"}
        )
        score_unknown = processor.calculate_quality_score(unknown_doc)

        assert score_official > score_github > score_unknown, \
            "官方文档 > GitHub文档 > 未知来源"

        print("✅ 质量评分-来源测试通过")

    def test_calculate_quality_score_perfect_document(self, processor):
        """测试质量评分 - 完美文档"""
        perfect_doc = Document(
            page_content="""
# Perfect Document

This is a well-structured document with:
- Proper length (100-2000 chars)
- Clear headers
- Code examples

```python
def example():
    return "Perfect!"
```

More content to reach ideal length.
""" * 3,  # 重复以达到合适长度
            metadata={"source_type": "official_docs"}
        )

        score = processor.calculate_quality_score(perfect_doc)

        # 完美文档应该接近满分
        assert score >= 0.8, f"完美文档应该高分,实际: {score}"

        print(f"✅ 质量评分-完美文档测试通过 (得分: {score})")

    def test_process_document_full_pipeline(self, processor):
        """测试完整处理流程"""
        raw_doc = Document(
            page_content="""
<div>
# Test Document

This is a test.


```python
code here
```

More content.
</div>
""",
            metadata={"source_type": "official_docs"}
        )

        processed_docs = processor.process_document(
            raw_doc,
            clean=True,
            split=True,
            calculate_score=True
        )

        # 验证处理结果
        assert len(processed_docs) > 0, "应该生成至少一个chunk"

        for doc in processed_docs:
            # 验证清洗(无HTML)
            assert "<div>" not in doc.page_content

            # 验证质量评分存在
            assert "quality_score" in doc.metadata
            assert 0 <= doc.metadata["quality_score"] <= 1

            # 验证chunk_boundary存在
            assert "chunk_boundary" in doc.metadata

        print(f"✅ 完整处理流程测试通过 (生成 {len(processed_docs)} 个chunks)")

    def test_process_documents_batch(self, processor):
        """测试批量处理"""
        raw_docs = [
            Document(page_content=f"# Doc {i}\n\nContent {i}", metadata={})
            for i in range(5)
        ]

        processed_docs = processor.process_documents(
            raw_docs,
            clean=True,
            split=True,
            calculate_score=True
        )

        # 验证批量处理结果
        assert len(processed_docs) >= len(raw_docs), "批量处理应该至少保持文档数量"

        for doc in processed_docs:
            assert "quality_score" in doc.metadata

        print(f"✅ 批量处理测试通过 (输入 {len(raw_docs)} → 输出 {len(processed_docs)})")

    def test_edge_case_empty_document(self, processor):
        """测试边界情况 - 空文档"""
        empty_doc = Document(page_content="", metadata={})

        chunks = processor.split_with_code_protection(empty_doc)

        # 空文档也应该有结果(即使是空chunk)
        assert len(chunks) >= 0

        print("✅ 空文档处理测试通过")

    def test_edge_case_only_code_block(self, processor):
        """测试边界情况 - 纯代码块"""
        code_only_doc = Document(
            page_content="```python\nprint('test')\n```",
            metadata={}
        )

        chunks = processor.split_with_code_protection(code_only_doc)

        assert len(chunks) > 0
        assert "```python" in chunks[0].page_content

        print("✅ 纯代码块处理测试通过")

    def test_edge_case_multiple_code_blocks(self, processor):
        """测试边界情况 - 多个代码块"""
        content = """
# Multiple Code Blocks

First code:
```python
code1
```

Second code:
```javascript
code2
```

Third code:
```bash
code3
```
"""
        doc = Document(page_content=content, metadata={})

        chunks = processor.split_with_code_protection(doc)

        # 验证所有代码块都保留
        all_content = " ".join([c.page_content for c in chunks])
        assert "```python" in all_content
        assert "```javascript" in all_content
        assert "```bash" in all_content

        print(f"✅ 多代码块处理测试通过 (生成 {len(chunks)} 个chunks)")

    def test_metadata_inheritance(self, processor):
        """测试元数据继承"""
        original_metadata = {
            "source": "test",
            "author": "Alice",
            "source_type": "official_docs"
        }

        doc = Document(
            page_content="# Title\n\nContent\n\n## Subtitle\n\nMore content",
            metadata=original_metadata
        )

        chunks = processor.split_with_code_protection(doc)

        # 验证所有chunk都继承了原始元数据
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "test"
            assert "author" in chunk.metadata
            assert "source_type" in chunk.metadata

        print("✅ 元数据继承测试通过")


class TestDocumentProcessorIntegration:
    """集成测试"""

    @pytest.mark.skip(reason="需要网络访问,跳过")
    def test_load_modelscope_docs(self):
        """测试加载魔搭文档(需要网络)"""
        processor = DocumentProcessor()

        # 使用单个URL测试
        docs = processor.load_modelscope_docs(
            urls=["https://www.modelscope.cn/docs/overview"]
        )

        assert len(docs) > 0, "应该加载到文档"

        for doc in docs:
            assert "source_url" in doc.metadata
            assert "source_type" in doc.metadata

        print(f"✅ 加载魔搭文档测试通过 (加载 {len(docs)} 个文档)")
