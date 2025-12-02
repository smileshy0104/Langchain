"""
Base Crawler Class

所有爬虫的基类，提供通用功能
"""

import os
import json
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from datetime import datetime


class BaseCrawler(ABC):
    """爬虫基类"""

    def __init__(self, output_dir: str = "data/crawled", rate_limit: float = 1.0):
        """
        初始化爬虫

        Args:
            output_dir: 输出目录
            rate_limit: 请求间隔(秒)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.last_request_time = 0

        # 请求头
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        }

        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _rate_limit_wait(self):
        """速率限制等待"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def fetch_page(self, url: str, max_retries: int = 3) -> Optional[str]:
        """
        获取页面内容

        Args:
            url: 页面URL
            max_retries: 最大重试次数

        Returns:
            页面HTML内容，失败返回None
        """
        self._rate_limit_wait()

        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                print(f"❌ 请求失败 (尝试 {attempt + 1}/{max_retries}): {url}")
                print(f"   错误: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    return None

        return None

    def parse_html(self, html: str) -> BeautifulSoup:
        """
        解析HTML

        Args:
            html: HTML内容

        Returns:
            BeautifulSoup对象
        """
        return BeautifulSoup(html, 'html.parser')

    def save_json(self, data: Dict, filename: str):
        """
        保存JSON数据

        Args:
            data: 要保存的数据
            filename: 文件名
        """
        filepath = self.output_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ 已保存: {filepath}")

    def save_text(self, content: str, filename: str):
        """
        保存文本数据

        Args:
            content: 文本内容
            filename: 文件名
        """
        filepath = self.output_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✅ 已保存: {filepath}")

    def load_checkpoint(self, checkpoint_file: str = "checkpoint.json") -> Dict:
        """
        加载检查点

        Args:
            checkpoint_file: 检查点文件名

        Returns:
            检查点数据
        """
        filepath = self.output_dir / checkpoint_file
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_checkpoint(self, data: Dict, checkpoint_file: str = "checkpoint.json"):
        """
        保存检查点

        Args:
            data: 检查点数据
            checkpoint_file: 检查点文件名
        """
        filepath = self.output_dir / checkpoint_file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @abstractmethod
    def crawl(self) -> List[Dict]:
        """
        执行爬取

        Returns:
            爬取的数据列表
        """
        pass

    def get_metadata(self) -> Dict:
        """
        获取爬虫元数据

        Returns:
            元数据字典
        """
        return {
            'crawler_name': self.__class__.__name__,
            'output_dir': str(self.output_dir),
            'timestamp': datetime.now().isoformat(),
        }
