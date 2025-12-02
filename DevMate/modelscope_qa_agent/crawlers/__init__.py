"""
ModelScope Data Crawlers

爬取魔搭社区官方指定的数据资源
"""

from .base_crawler import BaseCrawler
from .docs_crawler import DocsCrawler
from .learn_crawler import LearnCrawler
from .github_crawler import GitHubCrawler
from .catalog_crawler import CatalogCrawler

__all__ = [
    'BaseCrawler',
    'DocsCrawler',
    'LearnCrawler',
    'GitHubCrawler',
    'CatalogCrawler',
]
