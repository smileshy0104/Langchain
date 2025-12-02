"""
ModelScope Catalog Crawler

爬取模型库、数据集、创空间等资源目录
"""

from typing import List, Dict, Optional
from .base_crawler import BaseCrawler


class CatalogCrawler(BaseCrawler):
    """资源目录爬虫"""

    def __init__(self, output_dir: str = "data/crawled/catalog", rate_limit: float = 1.5):
        super().__init__(output_dir, rate_limit)
        self.catalogs = {
            'models': 'https://modelscope.cn/models',
            'datasets': 'https://modelscope.cn/datasets',
            'studios': 'https://modelscope.cn/studios',
            'mcp': 'https://www.modelscope.cn/mcp',
            'aigc': 'https://www.modelscope.cn/aigc',
        }

    def extract_catalog_items(self, soup, catalog_type: str) -> List[Dict]:
        """
        提取目录项

        Args:
            soup: BeautifulSoup对象
            catalog_type: 目录类型 (models/datasets/studios等)

        Returns:
            目录项列表
        """
        items = []

        # 查找卡片元素 (根据实际页面结构调整)
        # 这里提供通用的选择器，实际使用时需要根据页面结构调整
        card_selectors = [
            'div.card',
            'div.item',
            'div.model-card',
            'div.dataset-card',
            'article',
            'li.list-item'
        ]

        cards = []
        for selector in card_selectors:
            found = soup.select(selector)
            if found:
                cards = found
                break

        print(f"   找到 {len(cards)} 个卡片元素")

        for card in cards[:50]:  # 限制每个目录最多50项
            try:
                # 提取标题
                title = None
                title_elem = card.find(['h2', 'h3', 'h4', 'a'])
                if title_elem:
                    title = title_elem.get_text(strip=True)

                # 提取链接
                url = None
                link_elem = card.find('a', href=True)
                if link_elem:
                    from urllib.parse import urljoin
                    url = urljoin(self.catalogs[catalog_type], link_elem['href'])

                # 提取描述
                description = None
                desc_elem = card.find(['p', 'div'], class_=lambda x: x and ('desc' in x or 'description' in x))
                if desc_elem:
                    description = desc_elem.get_text(strip=True)

                # 提取标签
                tags = []
                tag_elems = card.find_all(class_=lambda x: x and 'tag' in x)
                for tag_elem in tag_elems:
                    tag = tag_elem.get_text(strip=True)
                    if tag:
                        tags.append(tag)

                if title:
                    items.append({
                        'title': title,
                        'url': url,
                        'description': description or '',
                        'tags': tags,
                        'catalog_type': catalog_type,
                        'source': f'modelscope_{catalog_type}',
                        'type': 'catalog_item'
                    })

            except Exception as e:
                continue

        return items

    def crawl_catalog(self, catalog_type: str, url: str) -> List[Dict]:
        """
        爬取单个目录

        Args:
            catalog_type: 目录类型
            url: 目录URL

        Returns:
            目录项列表
        """
        print(f"\n📚 爬取 {catalog_type} 目录...")
        print(f"   URL: {url}")

        html = self.fetch_page(url)
        if not html:
            print(f"❌ 无法获取 {catalog_type} 目录")
            return []

        soup = self.parse_html(html)
        items = self.extract_catalog_items(soup, catalog_type)

        print(f"✅ 从 {catalog_type} 提取了 {len(items)} 项")

        # 保存目录数据
        if items:
            filename = f"{catalog_type}_catalog.json"
            self.save_json({
                'catalog_type': catalog_type,
                'url': url,
                'total_items': len(items),
                'items': items
            }, filename)

            # 为每个目录项保存Markdown
            for i, item in enumerate(items, 1):
                md_content = self.convert_to_markdown(item)
                md_filename = f"{catalog_type}_{i}"
                self.save_markdown(md_content, md_filename)

        return items

    def crawl(self) -> List[Dict]:
        """
        执行爬取

        Returns:
            爬取的目录项列表
        """
        print("=" * 70)
        print("开始爬取魔搭社区资源目录")
        print("=" * 70)

        all_items = []

        try:
            for catalog_type, url in self.catalogs.items():
                items = self.crawl_catalog(catalog_type, url)
                all_items.extend(items)

            # 保存汇总
            summary = {
                'total_items': len(all_items),
                'by_catalog': {
                    catalog: len([i for i in all_items if i['catalog_type'] == catalog])
                    for catalog in self.catalogs.keys()
                },
                'metadata': self.get_metadata()
            }
            self.save_json(summary, 'summary.json')

            print("\n" + "=" * 70)
            print(f"✅ 爬取完成! 共 {len(all_items)} 个目录项")
            for catalog, count in summary['by_catalog'].items():
                print(f"   {catalog}: {count} 项")
            print("=" * 70)

            return all_items

        except Exception as e:
            print(f"\n❌ 爬取失败: {e}")
            import traceback
            traceback.print_exc()
            raise
