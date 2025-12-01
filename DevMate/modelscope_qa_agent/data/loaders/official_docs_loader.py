"""
魔搭社区官方文档加载器

从魔搭社区官网加载技术文档，支持多种格式：
- HTML 页面
- Markdown 文档
- API 文档

核心功能：
- 网页爬取和解析
- HTML 清洗和提取
- 元数据提取（标题、URL、更新时间等）
"""

import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader, RecursiveUrlLoader


class OfficialDocsLoader:
    """魔搭社区官方文档加载器

    支持从魔搭官网加载文档，自动提取和清洗内容。

    Attributes:
        base_url: 基础 URL
        max_depth: 最大爬取深度
        exclude_patterns: 排除的 URL 模式
        include_patterns: 包含的 URL 模式
        verbose: 是否输出详细日志

    Example:
        >>> loader = OfficialDocsLoader(
        ...     base_url="https://modelscope.cn/docs",
        ...     max_depth=2
        ... )
        >>> documents = loader.load()
        >>> print(f"加载了 {len(documents)} 个文档")
    """

    def __init__(
        self,
        base_url: str = "https://modelscope.cn/docs",
        max_depth: int = 2,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        verbose: bool = False
    ):
        """初始化官方文档加载器

        Args:
            base_url: 基础 URL (默认魔搭文档站)
            max_depth: 递归爬取的最大深度
            exclude_patterns: 排除的 URL 正则模式列表
            include_patterns: 包含的 URL 正则模式列表
            verbose: 是否输出详细日志
        """
        self.base_url = base_url
        self.max_depth = max_depth
        self.verbose = verbose

        # 默认排除模式
        self.exclude_patterns = exclude_patterns or [
            r".*\/login.*",
            r".*\/register.*",
            r".*\/logout.*",
            r".*\.pdf$",
            r".*\.zip$",
            r".*\.tar\.gz$"
        ]

        # 默认包含模式
        self.include_patterns = include_patterns or [
            r".*\/docs\/.*",
            r".*\/guide\/.*",
            r".*\/tutorial\/.*",
            r".*\/api\/.*"
        ]

        if verbose:
            print(f"✅ 官方文档加载器初始化成功")
            print(f"   - 基础 URL: {base_url}")
            print(f"   - 最大深度: {max_depth}")

    def _should_include_url(self, url: str) -> bool:
        """判断 URL 是否应该被包含

        Args:
            url: 待检查的 URL

        Returns:
            bool: 是否包含此 URL
        """
        # 检查排除模式
        for pattern in self.exclude_patterns:
            if re.match(pattern, url):
                return False

        # 检查包含模式
        for pattern in self.include_patterns:
            if re.match(pattern, url):
                return True

        return False

    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """从页面提取元数据

        Args:
            soup: BeautifulSoup 解析对象
            url: 页面 URL

        Returns:
            Dict[str, Any]: 元数据字典
        """
        metadata = {
            "source": url,
            "source_type": "official_docs",
            "url": url
        }

        # 提取标题
        title_tag = soup.find("title") or soup.find("h1")
        if title_tag:
            metadata["title"] = title_tag.get_text().strip()

        # 提取描述
        desc_tag = soup.find("meta", {"name": "description"})
        if desc_tag and desc_tag.get("content"):
            metadata["description"] = desc_tag.get("content").strip()

        # 提取关键词
        keywords_tag = soup.find("meta", {"name": "keywords"})
        if keywords_tag and keywords_tag.get("content"):
            metadata["keywords"] = keywords_tag.get("content").strip()

        # 提取作者
        author_tag = soup.find("meta", {"name": "author"})
        if author_tag and author_tag.get("content"):
            metadata["author"] = author_tag.get("content").strip()

        return metadata

    def _clean_html_content(self, soup: BeautifulSoup) -> str:
        """清洗 HTML 内容

        Args:
            soup: BeautifulSoup 解析对象

        Returns:
            str: 清洗后的文本内容
        """
        # 移除脚本和样式
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # 提取主要内容区域
        main_content = (
            soup.find("main") or
            soup.find("article") or
            soup.find("div", {"class": ["content", "main-content", "doc-content"]}) or
            soup.find("body")
        )

        if not main_content:
            return ""

        # 获取文本
        text = main_content.get_text(separator="\n", strip=True)

        # 清理多余空行
        text = re.sub(r'\n\s*\n', '\n\n', text)

        return text.strip()

    def load(self) -> List[Document]:
        """加载官方文档

        Returns:
            List[Document]: 文档列表

        Example:
            >>> loader = OfficialDocsLoader()
            >>> documents = loader.load()
            >>> print(f"加载了 {len(documents)} 个文档")
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"开始加载官方文档")
            print(f"{'='*70}\n")

        documents = []

        try:
            # 使用 RecursiveUrlLoader 递归加载
            loader = RecursiveUrlLoader(
                url=self.base_url,
                max_depth=self.max_depth,
                extractor=lambda x: BeautifulSoup(x, "html.parser").get_text(),
                prevent_outside=True,
                use_async=False,
                timeout=30,
                check_response_status=True
            )

            raw_documents = loader.load()

            if self.verbose:
                print(f"✅ 成功加载 {len(raw_documents)} 个原始文档")

            # 处理每个文档
            for doc in raw_documents:
                url = doc.metadata.get("source", "")

                # 过滤 URL
                if not self._should_include_url(url):
                    if self.verbose:
                        print(f"⏭️  跳过: {url}")
                    continue

                # 重新解析以提取更多元数据
                try:
                    response = requests.get(url, timeout=10)
                    soup = BeautifulSoup(response.content, "html.parser")

                    # 提取元数据
                    metadata = self._extract_metadata(soup, url)

                    # 清洗内容
                    cleaned_content = self._clean_html_content(soup)

                    if cleaned_content:
                        # 创建新文档
                        new_doc = Document(
                            page_content=cleaned_content,
                            metadata=metadata
                        )
                        documents.append(new_doc)

                        if self.verbose:
                            print(f"✅ 处理成功: {metadata.get('title', url)}")
                    else:
                        if self.verbose:
                            print(f"⚠️  内容为空: {url}")

                except Exception as e:
                    if self.verbose:
                        print(f"❌ 处理失败: {url} - {str(e)}")
                    continue

            if self.verbose:
                print(f"\n{'='*70}")
                print(f"✅ 官方文档加载完成")
                print(f"{'='*70}")
                print(f"总计加载: {len(documents)} 个有效文档")
                print(f"{'='*70}\n")

            return documents

        except Exception as e:
            if self.verbose:
                print(f"❌ 加载失败: {str(e)}")
            raise

    def load_from_urls(self, urls: List[str]) -> List[Document]:
        """从指定 URL 列表加载文档

        Args:
            urls: URL 列表

        Returns:
            List[Document]: 文档列表

        Example:
            >>> loader = OfficialDocsLoader()
            >>> urls = [
            ...     "https://modelscope.cn/docs/intro",
            ...     "https://modelscope.cn/docs/models"
            ... ]
            >>> documents = loader.load_from_urls(urls)
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"从指定 URL 加载文档")
            print(f"{'='*70}")
            print(f"URL 数量: {len(urls)}")
            print(f"{'='*70}\n")

        documents = []

        for url in urls:
            try:
                # 加载页面
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.content, "html.parser")

                # 提取元数据
                metadata = self._extract_metadata(soup, url)

                # 清洗内容
                cleaned_content = self._clean_html_content(soup)

                if cleaned_content:
                    # 创建文档
                    doc = Document(
                        page_content=cleaned_content,
                        metadata=metadata
                    )
                    documents.append(doc)

                    if self.verbose:
                        print(f"✅ 加载成功: {metadata.get('title', url)}")
                else:
                    if self.verbose:
                        print(f"⚠️  内容为空: {url}")

            except Exception as e:
                if self.verbose:
                    print(f"❌ 加载失败: {url} - {str(e)}")
                continue

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"✅ 加载完成: {len(documents)} / {len(urls)} 个文档")
            print(f"{'='*70}\n")

        return documents

    def get_stats(self) -> Dict[str, Any]:
        """获取加载器统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "loader_type": "OfficialDocsLoader",
            "base_url": self.base_url,
            "max_depth": self.max_depth,
            "num_exclude_patterns": len(self.exclude_patterns),
            "num_include_patterns": len(self.include_patterns)
        }


# 便捷函数

def load_official_docs(
    base_url: str = "https://modelscope.cn/docs",
    max_depth: int = 2,
    verbose: bool = False
) -> List[Document]:
    """加载魔搭官方文档的便捷函数

    Args:
        base_url: 基础 URL
        max_depth: 最大深度
        verbose: 详细输出

    Returns:
        List[Document]: 文档列表

    Example:
        >>> documents = load_official_docs(verbose=True)
        >>> print(f"加载了 {len(documents)} 个文档")
    """
    loader = OfficialDocsLoader(
        base_url=base_url,
        max_depth=max_depth,
        verbose=verbose
    )
    return loader.load()


# 示例用法
if __name__ == "__main__":
    print("=" * 70)
    print("官方文档加载器示例")
    print("=" * 70)

    # 示例 1: 基本用法
    print("\n示例 1: 创建加载器")
    print("-" * 70)

    loader = OfficialDocsLoader(
        base_url="https://modelscope.cn/docs",
        max_depth=1,
        verbose=True
    )

    # 获取统计信息
    stats = loader.get_stats()
    print("\n加载器统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 示例 2: 从指定 URL 加载
    print("\n示例 2: 从指定 URL 加载")
    print("-" * 70)

    # 注意：这里使用示例 URL，实际使用时需要替换为真实的魔搭文档 URL
    example_urls = [
        "https://modelscope.cn/docs/intro",
        "https://modelscope.cn/docs/models"
    ]

    print(f"准备从 {len(example_urls)} 个 URL 加载文档")
    print("(示例模式，实际使用时需要真实 URL)")

    print("\n" + "=" * 70)
    print("✅ 示例执行完成")
    print("=" * 70)
