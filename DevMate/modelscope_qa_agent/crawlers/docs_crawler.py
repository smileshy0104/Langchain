"""
ModelScope Documentation Crawler

爬取魔搭社区官方文档: https://www.modelscope.cn/docs/overview
"""

from typing import List, Dict, Set, Optional
from urllib.parse import urljoin, urlparse
from .base_crawler import BaseCrawler


class DocsCrawler(BaseCrawler):
    """魔搭文档爬虫"""

    def __init__(self, output_dir: str = "data/crawled/docs", rate_limit: float = 1.5):
        super().__init__(output_dir, rate_limit)
        self.base_url = "https://www.modelscope.cn/docs/"
        self.start_url = "https://www.modelscope.cn/docs/overview"
        self.visited_urls: Set[str] = set()
        self.max_depth = 3

    def is_valid_doc_url(self, url: str) -> bool:
        """
        判断是否为有效的文档URL

        Args:
            url: URL

        Returns:
            是否有效
        """
        parsed = urlparse(url)
        return (
            parsed.netloc == "www.modelscope.cn" and
            parsed.path.startswith("/docs/") and
            url not in self.visited_urls
        )

    def extract_doc_links(self, soup, current_url: str) -> List[str]:
        """
        提取文档链接

        Args:
            soup: BeautifulSoup对象
            current_url: 当前URL

        Returns:
            链接列表
        """
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(current_url, href)

            # 移除URL中的fragment (#部分)
            full_url = full_url.split('#')[0]

            if self.is_valid_doc_url(full_url):
                links.append(full_url)

        return list(set(links))  # 去重

    def extract_doc_content(self, soup, url: str) -> Optional[Dict]:
        """
        提取文档内容

        Args:
            soup: BeautifulSoup对象
            url: 文档URL

        Returns:
            文档数据字典
        """
        try:
            # 查找文档标题
            title = None
            title_elem = soup.find('h1')
            if title_elem:
                title = title_elem.get_text(strip=True)

            # 查找主要内容区域
            content_elem = soup.find('article') or soup.find('main') or soup.find('div', class_='content')

            if not content_elem:
                print(f"⚠️  未找到内容区域: {url}")
                return None

            # 提取文本内容
            paragraphs = []
            for elem in content_elem.find_all(['p', 'h2', 'h3', 'h4', 'pre', 'code', 'li']):
                text = elem.get_text(strip=True)
                if text:
                    paragraphs.append(text)

            content = '\n\n'.join(paragraphs)

            if not content:
                print(f"⚠️  内容为空: {url}")
                return None

            # 提取代码块
            code_blocks = []
            for code in content_elem.find_all('pre'):
                code_text = code.get_text(strip=True)
                if code_text:
                    code_blocks.append(code_text)

            return {
                'url': url,
                'title': title or 'Untitled',
                'content': content,
                'code_blocks': code_blocks,
                'source': 'modelscope_docs',
                'type': 'documentation'
            }

        except Exception as e:
            print(f"❌ 解析失败: {url}")
            print(f"   错误: {e}")
            return None

    def crawl_page(self, url: str, depth: int = 0) -> List[Dict]:
        """
        递归爬取页面

        Args:
            url: 页面URL
            depth: 当前深度

        Returns:
            爬取的文档列表
        """
        if depth > self.max_depth or url in self.visited_urls:
            return []

        self.visited_urls.add(url)
        print(f"📄 爬取 (深度={depth}): {url}")

        html = self.fetch_page(url)
        if not html:
            return []

        soup = self.parse_html(html)
        documents = []

        # 提取当前页面内容
        doc_data = self.extract_doc_content(soup, url)
        if doc_data:
            documents.append(doc_data)
            # 保存单个文档
            filename = f"doc_{len(self.visited_urls)}.json"
            self.save_json(doc_data, filename)

        # 递归爬取链接
        if depth < self.max_depth:
            links = self.extract_doc_links(soup, url)
            print(f"   发现 {len(links)} 个链接")

            for link in links:
                documents.extend(self.crawl_page(link, depth + 1))

        return documents

    def crawl(self) -> List[Dict]:
        """
        执行爬取

        Returns:
            爬取的文档列表
        """
        print("=" * 70)
        print("开始爬取魔搭社区官方文档")
        print(f"起始URL: {self.start_url}")
        print(f"最大深度: {self.max_depth}")
        print("=" * 70)

        # 加载检查点
        checkpoint = self.load_checkpoint()
        if checkpoint.get('visited_urls'):
            self.visited_urls = set(checkpoint['visited_urls'])
            print(f"📌 从检查点恢复，已爬取 {len(self.visited_urls)} 个页面")

        try:
            documents = self.crawl_page(self.start_url)

            # 保存汇总数据
            summary = {
                'total_documents': len(documents),
                'visited_urls': list(self.visited_urls),
                'metadata': self.get_metadata()
            }
            self.save_json(summary, 'summary.json')

            print("\n" + "=" * 70)
            print(f"✅ 爬取完成! 共 {len(documents)} 个文档")
            print(f"   访问了 {len(self.visited_urls)} 个URL")
            print("=" * 70)

            return documents

        except KeyboardInterrupt:
            print("\n⚠️  用户中断，保存检查点...")
            self.save_checkpoint({'visited_urls': list(self.visited_urls)})
            raise

        except Exception as e:
            print(f"\n❌ 爬取失败: {e}")
            self.save_checkpoint({'visited_urls': list(self.visited_urls)})
            raise
