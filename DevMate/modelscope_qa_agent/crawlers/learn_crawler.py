"""
ModelScope Learn Section Crawler

爬取魔搭研习社: https://modelscope.cn/learn
"""

from typing import List, Dict, Optional
from urllib.parse import urljoin
from .base_crawler import BaseCrawler


class LearnCrawler(BaseCrawler):
    """魔搭研习社爬虫"""

    def __init__(self, output_dir: str = "data/crawled/learn", rate_limit: float = 1.5):
        super().__init__(output_dir, rate_limit)
        self.base_url = "https://modelscope.cn/learn"
        self.visited_articles = set()

    def extract_article_links(self, soup) -> List[str]:
        """
        提取文章链接

        Args:
            soup: BeautifulSoup对象

        Returns:
            文章链接列表
        """
        links = []

        # 查找文章链接 (根据实际页面结构调整选择器)
        for a in soup.find_all('a', href=True):
            href = a['href']
            if '/learn/' in href or '/article/' in href:
                full_url = urljoin(self.base_url, href)
                if full_url not in self.visited_articles:
                    links.append(full_url)

        return list(set(links))

    def extract_article_content(self, soup, url: str) -> Optional[Dict]:
        """
        提取文章内容

        Args:
            soup: BeautifulSoup对象
            url: 文章URL

        Returns:
            文章数据字典
        """
        try:
            # 提取标题
            title = None
            title_elem = soup.find('h1') or soup.find('title')
            if title_elem:
                title = title_elem.get_text(strip=True)

            # 提取作者
            author = None
            author_elem = soup.find(class_='author') or soup.find('meta', attrs={'name': 'author'})
            if author_elem:
                author = author_elem.get('content') if author_elem.name == 'meta' else author_elem.get_text(strip=True)

            # 提取发布日期
            date = None
            date_elem = soup.find(class_='date') or soup.find('time')
            if date_elem:
                date = date_elem.get('datetime') or date_elem.get_text(strip=True)

            # 提取正文内容
            content_elem = (
                soup.find('article') or
                soup.find(class_='content') or
                soup.find(class_='article-body') or
                soup.find('main')
            )

            if not content_elem:
                print(f"⚠️  未找到内容区域: {url}")
                return None

            # 提取段落
            paragraphs = []
            for elem in content_elem.find_all(['p', 'h2', 'h3', 'h4', 'li']):
                text = elem.get_text(strip=True)
                if text and len(text) > 10:  # 过滤太短的文本
                    paragraphs.append(text)

            content = '\n\n'.join(paragraphs)

            if not content or len(content) < 100:
                print(f"⚠️  内容过短或为空: {url}")
                return None

            # 提取代码块
            code_blocks = []
            for code in content_elem.find_all(['pre', 'code']):
                code_text = code.get_text(strip=True)
                if code_text and len(code_text) > 20:
                    code_blocks.append(code_text)

            # 提取标签
            tags = []
            for tag_elem in soup.find_all(class_='tag'):
                tag = tag_elem.get_text(strip=True)
                if tag:
                    tags.append(tag)

            return {
                'url': url,
                'title': title or 'Untitled',
                'author': author,
                'date': date,
                'content': content,
                'code_blocks': code_blocks,
                'tags': tags,
                'source': 'modelscope_learn',
                'type': 'article'
            }

        except Exception as e:
            print(f"❌ 解析文章失败: {url}")
            print(f"   错误: {e}")
            return None

    def crawl(self) -> List[Dict]:
        """
        执行爬取

        Returns:
            爬取的文章列表
        """
        print("=" * 70)
        print("开始爬取魔搭研习社")
        print(f"起始URL: {self.base_url}")
        print("=" * 70)

        articles = []

        # 加载检查点
        checkpoint = self.load_checkpoint()
        if checkpoint.get('visited_articles'):
            self.visited_articles = set(checkpoint['visited_articles'])
            print(f"📌 从检查点恢复，已爬取 {len(self.visited_articles)} 篇文章")

        try:
            # 获取主页
            print(f"📄 获取研习社主页...")
            html = self.fetch_page(self.base_url)
            if not html:
                print("❌ 无法获取研习社主页")
                return []

            soup = self.parse_html(html)

            # 提取文章链接
            article_links = self.extract_article_links(soup)
            print(f"   发现 {len(article_links)} 篇文章")

            # 爬取每篇文章
            for i, url in enumerate(article_links, 1):
                if url in self.visited_articles:
                    continue

                print(f"\n📄 [{i}/{len(article_links)}] 爬取文章: {url}")

                html = self.fetch_page(url)
                if not html:
                    continue

                soup = self.parse_html(html)
                article_data = self.extract_article_content(soup, url)

                if article_data:
                    articles.append(article_data)
                    # 保存单篇文章
                    filename = f"article_{len(articles)}.json"
                    self.save_json(article_data, filename)
                    self.visited_articles.add(url)

                # 每10篇保存一次检查点
                if len(articles) % 10 == 0:
                    self.save_checkpoint({'visited_articles': list(self.visited_articles)})

            # 保存汇总
            summary = {
                'total_articles': len(articles),
                'visited_articles': list(self.visited_articles),
                'metadata': self.get_metadata()
            }
            self.save_json(summary, 'summary.json')

            print("\n" + "=" * 70)
            print(f"✅ 爬取完成! 共 {len(articles)} 篇文章")
            print("=" * 70)

            return articles

        except KeyboardInterrupt:
            print("\n⚠️  用户中断，保存检查点...")
            self.save_checkpoint({'visited_articles': list(self.visited_articles)})
            raise

        except Exception as e:
            print(f"\n❌ 爬取失败: {e}")
            self.save_checkpoint({'visited_articles': list(self.visited_articles)})
            raise
