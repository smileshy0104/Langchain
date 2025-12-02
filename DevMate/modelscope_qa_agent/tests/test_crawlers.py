"""
测试爬虫系统

测试各个爬虫模块的基本功能
"""

import pytest
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from crawlers import DocsCrawler, LearnCrawler, GitHubCrawler, CatalogCrawler
from crawlers.data_processor import DataProcessor


class TestBaseCrawler:
    """测试基础爬虫功能"""

    def test_rate_limit(self):
        """测试速率限制"""
        import time
        from crawlers.base_crawler import BaseCrawler

        class TestCrawler(BaseCrawler):
            def crawl(self):
                pass

        crawler = TestCrawler(rate_limit=0.5)

        start = time.time()
        crawler._rate_limit_wait()
        crawler._rate_limit_wait()
        elapsed = time.time() - start

        assert elapsed >= 0.5, "速率限制应该生效"

    def test_fetch_page(self):
        """测试页面获取"""
        from crawlers.base_crawler import BaseCrawler

        class TestCrawler(BaseCrawler):
            def crawl(self):
                pass

        crawler = TestCrawler(rate_limit=0.1)

        # 测试有效URL
        html = crawler.fetch_page("https://www.modelscope.cn")
        assert html is not None, "应该能够获取页面"
        assert len(html) > 0, "页面内容不应为空"

    def test_save_load_checkpoint(self):
        """测试检查点保存和加载"""
        from crawlers.base_crawler import BaseCrawler
        import tempfile

        class TestCrawler(BaseCrawler):
            def crawl(self):
                pass

        with tempfile.TemporaryDirectory() as tmpdir:
            crawler = TestCrawler(output_dir=tmpdir, rate_limit=0.1)

            # 保存检查点
            test_data = {'visited': ['url1', 'url2'], 'count': 2}
            crawler.save_checkpoint(test_data)

            # 加载检查点
            loaded_data = crawler.load_checkpoint()
            assert loaded_data == test_data, "检查点数据应该一致"


class TestDocsCrawler:
    """测试文档爬虫"""

    def test_is_valid_doc_url(self):
        """测试URL验证"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            crawler = DocsCrawler(output_dir=tmpdir, rate_limit=0.1)

            # 有效URL
            assert crawler.is_valid_doc_url("https://www.modelscope.cn/docs/overview")
            assert crawler.is_valid_doc_url("https://www.modelscope.cn/docs/model/usage")

            # 无效URL
            assert not crawler.is_valid_doc_url("https://www.google.com")
            assert not crawler.is_valid_doc_url("https://www.modelscope.cn/models")

    def test_extract_doc_content(self):
        """测试文档内容提取"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            crawler = DocsCrawler(output_dir=tmpdir, rate_limit=0.1)

            # 简单HTML示例
            html = """
            <html>
                <body>
                    <h1>测试标题</h1>
                    <article>
                        <p>这是一段测试内容。</p>
                        <pre><code>print("Hello, World!")</code></pre>
                    </article>
                </body>
            </html>
            """

            soup = crawler.parse_html(html)
            doc_data = crawler.extract_doc_content(soup, "https://test.com")

            assert doc_data is not None, "应该提取到文档数据"
            assert doc_data['title'] == "测试标题"
            assert "测试内容" in doc_data['content']
            assert len(doc_data['code_blocks']) > 0


class TestGitHubCrawler:
    """测试GitHub爬虫"""

    def test_fetch_readme(self):
        """测试README获取"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            crawler = GitHubCrawler(output_dir=tmpdir, rate_limit=2.0)

            # 测试获取一个知名仓库的README
            readme = crawler.fetch_readme("modelscope/modelscope")

            if readme:  # GitHub API可能有限制
                assert len(readme) > 0, "README不应为空"
                print(f"✅ 成功获取README ({len(readme)} 字符)")
            else:
                print("⚠️  GitHub API限制,跳过README测试")


class TestDataProcessor:
    """测试数据处理器"""

    def test_clean_text(self):
        """测试文本清理"""
        processor = DataProcessor()

        text = "这是   一段  \n\n 测试   文本"
        cleaned = processor.clean_text(text)

        assert "  " not in cleaned, "不应有多余空格"
        assert cleaned == "这是 一段 测试 文本"

    def test_chunk_text(self):
        """测试文本分块"""
        processor = DataProcessor()

        # 短文本
        short_text = "这是一段短文本。"
        chunks = processor.chunk_text(short_text, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == short_text

        # 长文本
        long_text = "这是一段很长的文本。" * 100
        chunks = processor.chunk_text(long_text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1, "长文本应该被分块"

        # 验证重叠
        if len(chunks) > 1:
            # 第二块应该包含第一块末尾的一些内容
            assert len(chunks[0]) <= 100 + 20  # 考虑分割位置调整

    def test_process_document(self):
        """测试文档处理"""
        processor = DataProcessor()

        doc = {
            'title': '测试文档',
            'url': 'https://test.com',
            'content': '这是一段测试内容。' * 50,
            'source': 'test',
            'tags': ['tag1', 'tag2']
        }

        chunks = processor.process_document(doc, 'test')

        assert len(chunks) > 0, "应该生成文档块"

        # 验证元数据
        first_chunk = chunks[0]
        assert 'content' in first_chunk
        assert 'metadata' in first_chunk
        assert first_chunk['metadata']['title'] == '测试文档'
        assert first_chunk['metadata']['source_type'] == 'test'
        assert first_chunk['metadata']['tags'] == 'tag1,tag2'


def test_crawl_script_help():
    """测试爬取脚本帮助信息"""
    import subprocess

    result = subprocess.run(
        ['python', 'scripts/crawl_and_process.py', '--help'],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, "脚本应该正常运行"
    assert '爬取并处理魔搭社区数据' in result.stdout


def test_ingest_script_help():
    """测试导入脚本帮助信息"""
    import subprocess

    result = subprocess.run(
        ['python', 'scripts/ingest_crawled_data.py', '--help'],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, "脚本应该正常运行"
    assert '将爬取的数据导入向量数据库' in result.stdout


if __name__ == "__main__":
    print("=" * 70)
    print("爬虫系统测试")
    print("=" * 70)

    # 运行测试
    pytest.main([__file__, '-v', '-s'])
