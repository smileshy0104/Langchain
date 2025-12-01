"""
GitHub 技术文档加载器

从 GitHub 仓库加载技术文档，支持多种格式：
- Markdown 文件 (.md)
- reStructuredText 文件 (.rst)
- Jupyter Notebook (.ipynb)
- 代码注释和 README

核心功能：
- GitHub API 集成
- 仓库文件树遍历
- Markdown/RST 解析
- 元数据提取（作者、更新时间、Star 数等）
"""

import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    GitHubIssuesLoader,
    GithubFileLoader
)


class GitHubDocsLoader:
    """GitHub 技术文档加载器

    从 GitHub 仓库加载技术文档，支持多种文件格式。

    Attributes:
        repo_owner: 仓库所有者
        repo_name: 仓库名称
        github_token: GitHub API Token (可选，提高 rate limit)
        file_patterns: 要加载的文件模式列表
        exclude_patterns: 排除的文件模式列表
        branch: 分支名称
        verbose: 是否输出详细日志

    Example:
        >>> loader = GitHubDocsLoader(
        ...     repo_owner="modelscope",
        ...     repo_name="modelscope",
        ...     github_token="your-token"
        ... )
        >>> documents = loader.load()
        >>> print(f"加载了 {len(documents)} 个文档")
    """

    def __init__(
        self,
        repo_owner: str,
        repo_name: str,
        github_token: Optional[str] = None,
        file_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        branch: str = "main",
        verbose: bool = False
    ):
        """初始化 GitHub 文档加载器

        Args:
            repo_owner: 仓库所有者
            repo_name: 仓库名称
            github_token: GitHub Personal Access Token (可选)
            file_patterns: 文件匹配模式 (默认: .md, .rst, README)
            exclude_patterns: 排除的文件模式
            branch: 分支名称 (默认: main)
            verbose: 是否输出详细日志
        """
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.branch = branch
        self.verbose = verbose

        # 默认文件模式
        self.file_patterns = file_patterns or [
            r".*\.md$",
            r".*\.rst$",
            r".*README.*",
            r".*CONTRIBUTING.*",
            r".*CHANGELOG.*",
            r".*docs/.*",
            r".*doc/.*"
        ]

        # 默认排除模式
        self.exclude_patterns = exclude_patterns or [
            r".*node_modules.*",
            r".*\.git.*",
            r".*__pycache__.*",
            r".*\.pytest_cache.*",
            r".*\.venv.*",
            r".*venv.*",
            r".*build.*",
            r".*dist.*"
        ]

        # GitHub API 基础 URL
        self.api_base = "https://api.github.com"
        self.repo_url = f"https://github.com/{repo_owner}/{repo_name}"

        if verbose:
            print(f"✅ GitHub 文档加载器初始化成功")
            print(f"   - 仓库: {repo_owner}/{repo_name}")
            print(f"   - 分支: {branch}")
            print(f"   - Token: {'已配置' if self.github_token else '未配置'}")

    def _should_include_file(self, file_path: str) -> bool:
        """判断文件是否应该被包含

        Args:
            file_path: 文件路径

        Returns:
            bool: 是否包含此文件
        """
        # 检查排除模式
        for pattern in self.exclude_patterns:
            if re.match(pattern, file_path):
                return False

        # 检查包含模式
        for pattern in self.file_patterns:
            if re.match(pattern, file_path):
                return True

        return False

    def _get_api_headers(self) -> Dict[str, str]:
        """获取 API 请求头

        Returns:
            Dict[str, str]: 请求头字典
        """
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }

        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"

        return headers

    def _get_repo_info(self) -> Dict[str, Any]:
        """获取仓库信息

        Returns:
            Dict[str, Any]: 仓库信息字典
        """
        url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}"

        try:
            response = requests.get(url, headers=self._get_api_headers(), timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if self.verbose:
                print(f"⚠️  获取仓库信息失败: {str(e)}")
            return {}

    def _get_file_tree(self, path: str = "") -> List[Dict[str, Any]]:
        """获取文件树

        Args:
            path: 路径 (空字符串表示根目录)

        Returns:
            List[Dict[str, Any]]: 文件列表
        """
        url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/contents/{path}"

        params = {"ref": self.branch}

        try:
            response = requests.get(
                url,
                headers=self._get_api_headers(),
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if self.verbose:
                print(f"⚠️  获取文件树失败 ({path}): {str(e)}")
            return []

    def _get_file_content(self, file_path: str) -> Optional[str]:
        """获取文件内容

        Args:
            file_path: 文件路径

        Returns:
            Optional[str]: 文件内容
        """
        url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/contents/{file_path}"

        params = {"ref": self.branch}

        try:
            response = requests.get(
                url,
                headers=self._get_api_headers(),
                params=params,
                timeout=10
            )
            response.raise_for_status()

            file_data = response.json()

            # GitHub API 返回 base64 编码的内容
            import base64
            content = base64.b64decode(file_data["content"]).decode("utf-8")

            return content

        except Exception as e:
            if self.verbose:
                print(f"⚠️  获取文件内容失败 ({file_path}): {str(e)}")
            return None

    def _extract_file_metadata(
        self,
        file_path: str,
        file_info: Dict[str, Any],
        repo_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """提取文件元数据

        Args:
            file_path: 文件路径
            file_info: 文件信息
            repo_info: 仓库信息

        Returns:
            Dict[str, Any]: 元数据字典
        """
        metadata = {
            "source": f"{self.repo_url}/blob/{self.branch}/{file_path}",
            "source_type": "github_docs",
            "repo_owner": self.repo_owner,
            "repo_name": self.repo_name,
            "branch": self.branch,
            "file_path": file_path,
            "url": file_info.get("html_url", "")
        }

        # 仓库元数据
        if repo_info:
            metadata.update({
                "repo_description": repo_info.get("description", ""),
                "repo_stars": repo_info.get("stargazers_count", 0),
                "repo_forks": repo_info.get("forks_count", 0),
                "repo_language": repo_info.get("language", ""),
                "repo_topics": repo_info.get("topics", [])
            })

        # 文件元数据
        metadata.update({
            "file_name": file_info.get("name", ""),
            "file_size": file_info.get("size", 0),
            "file_sha": file_info.get("sha", "")
        })

        return metadata

    def _collect_files_recursive(
        self,
        path: str = "",
        max_depth: int = 10,
        current_depth: int = 0
    ) -> List[Dict[str, Any]]:
        """递归收集文件

        Args:
            path: 当前路径
            max_depth: 最大深度
            current_depth: 当前深度

        Returns:
            List[Dict[str, Any]]: 文件信息列表
        """
        if current_depth >= max_depth:
            return []

        files = []
        items = self._get_file_tree(path)

        for item in items:
            item_path = item.get("path", "")
            item_type = item.get("type", "")

            if item_type == "file":
                # 检查是否包含此文件
                if self._should_include_file(item_path):
                    files.append(item)
                    if self.verbose:
                        print(f"  📄 发现文件: {item_path}")

            elif item_type == "dir":
                # 递归处理目录
                if self.verbose:
                    print(f"  📁 进入目录: {item_path}")

                subfiles = self._collect_files_recursive(
                    item_path,
                    max_depth,
                    current_depth + 1
                )
                files.extend(subfiles)

        return files

    def load(self, max_depth: int = 10) -> List[Document]:
        """加载 GitHub 文档

        Args:
            max_depth: 最大递归深度

        Returns:
            List[Document]: 文档列表

        Example:
            >>> loader = GitHubDocsLoader("modelscope", "modelscope")
            >>> documents = loader.load()
            >>> print(f"加载了 {len(documents)} 个文档")
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"开始加载 GitHub 文档")
            print(f"{'='*70}")
            print(f"仓库: {self.repo_owner}/{self.repo_name}")
            print(f"分支: {self.branch}")
            print(f"{'='*70}\n")

        documents = []

        try:
            # 获取仓库信息
            repo_info = self._get_repo_info()

            if self.verbose and repo_info:
                print(f"✅ 仓库信息:")
                print(f"   - 描述: {repo_info.get('description', 'N/A')}")
                print(f"   - Stars: {repo_info.get('stargazers_count', 0)}")
                print(f"   - 语言: {repo_info.get('language', 'N/A')}")
                print()

            # 收集文件列表
            if self.verbose:
                print("🔍 扫描文件树...\n")

            files = self._collect_files_recursive(max_depth=max_depth)

            if self.verbose:
                print(f"\n✅ 发现 {len(files)} 个匹配文件\n")
                print("📥 开始下载文件内容...\n")

            # 加载每个文件
            for file_info in files:
                file_path = file_info.get("path", "")

                try:
                    # 获取文件内容
                    content = self._get_file_content(file_path)

                    if content:
                        # 提取元数据
                        metadata = self._extract_file_metadata(
                            file_path,
                            file_info,
                            repo_info
                        )

                        # 创建文档
                        doc = Document(
                            page_content=content,
                            metadata=metadata
                        )
                        documents.append(doc)

                        if self.verbose:
                            print(f"✅ 加载成功: {file_path} ({len(content)} 字符)")
                    else:
                        if self.verbose:
                            print(f"⚠️  内容为空: {file_path}")

                except Exception as e:
                    if self.verbose:
                        print(f"❌ 加载失败: {file_path} - {str(e)}")
                    continue

            if self.verbose:
                print(f"\n{'='*70}")
                print(f"✅ GitHub 文档加载完成")
                print(f"{'='*70}")
                print(f"总计加载: {len(documents)} 个文档")
                print(f"{'='*70}\n")

            return documents

        except Exception as e:
            if self.verbose:
                print(f"❌ 加载失败: {str(e)}")
            raise

    def load_specific_files(self, file_paths: List[str]) -> List[Document]:
        """加载指定文件

        Args:
            file_paths: 文件路径列表

        Returns:
            List[Document]: 文档列表

        Example:
            >>> loader = GitHubDocsLoader("modelscope", "modelscope")
            >>> files = ["README.md", "docs/intro.md"]
            >>> documents = loader.load_specific_files(files)
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"加载指定文件")
            print(f"{'='*70}")
            print(f"文件数量: {len(file_paths)}")
            print(f"{'='*70}\n")

        documents = []
        repo_info = self._get_repo_info()

        for file_path in file_paths:
            try:
                # 获取文件内容
                content = self._get_file_content(file_path)

                if content:
                    # 获取文件信息
                    file_info = {
                        "path": file_path,
                        "name": os.path.basename(file_path),
                        "html_url": f"{self.repo_url}/blob/{self.branch}/{file_path}"
                    }

                    # 提取元数据
                    metadata = self._extract_file_metadata(
                        file_path,
                        file_info,
                        repo_info
                    )

                    # 创建文档
                    doc = Document(
                        page_content=content,
                        metadata=metadata
                    )
                    documents.append(doc)

                    if self.verbose:
                        print(f"✅ 加载成功: {file_path}")
                else:
                    if self.verbose:
                        print(f"⚠️  内容为空: {file_path}")

            except Exception as e:
                if self.verbose:
                    print(f"❌ 加载失败: {file_path} - {str(e)}")
                continue

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"✅ 加载完成: {len(documents)} / {len(file_paths)} 个文件")
            print(f"{'='*70}\n")

        return documents

    def get_stats(self) -> Dict[str, Any]:
        """获取加载器统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "loader_type": "GitHubDocsLoader",
            "repo_owner": self.repo_owner,
            "repo_name": self.repo_name,
            "branch": self.branch,
            "num_file_patterns": len(self.file_patterns),
            "num_exclude_patterns": len(self.exclude_patterns),
            "has_token": self.github_token is not None
        }


# 便捷函数

def load_github_docs(
    repo_owner: str,
    repo_name: str,
    github_token: Optional[str] = None,
    branch: str = "main",
    max_depth: int = 10,
    verbose: bool = False
) -> List[Document]:
    """加载 GitHub 文档的便捷函数

    Args:
        repo_owner: 仓库所有者
        repo_name: 仓库名称
        github_token: GitHub Token
        branch: 分支名称
        max_depth: 最大深度
        verbose: 详细输出

    Returns:
        List[Document]: 文档列表

    Example:
        >>> documents = load_github_docs(
        ...     "modelscope",
        ...     "modelscope",
        ...     verbose=True
        ... )
        >>> print(f"加载了 {len(documents)} 个文档")
    """
    loader = GitHubDocsLoader(
        repo_owner=repo_owner,
        repo_name=repo_name,
        github_token=github_token,
        branch=branch,
        verbose=verbose
    )
    return loader.load(max_depth=max_depth)


# 示例用法
if __name__ == "__main__":
    print("=" * 70)
    print("GitHub 文档加载器示例")
    print("=" * 70)

    # 示例 1: 基本用法
    print("\n示例 1: 创建加载器")
    print("-" * 70)

    loader = GitHubDocsLoader(
        repo_owner="modelscope",
        repo_name="modelscope",
        branch="main",
        verbose=True
    )

    # 获取统计信息
    stats = loader.get_stats()
    print("\n加载器统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 示例 2: 加载指定文件
    print("\n示例 2: 加载指定文件")
    print("-" * 70)

    example_files = [
        "README.md",
        "docs/intro.md"
    ]

    print(f"准备加载 {len(example_files)} 个文件")
    print("(示例模式，实际使用时需要真实文件路径)")

    print("\n" + "=" * 70)
    print("✅ 示例执行完成")
    print("=" * 70)
