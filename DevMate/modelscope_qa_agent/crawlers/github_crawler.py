"""
GitHub Repositories Crawler

爬取魔搭社区GitHub仓库: https://github.com/modelscope
"""

from typing import List, Dict, Optional
import os
from .base_crawler import BaseCrawler


class GitHubCrawler(BaseCrawler):
    """GitHub仓库爬虫"""

    def __init__(self, output_dir: str = "data/crawled/github", rate_limit: float = 2.0):
        super().__init__(output_dir, rate_limit)
        self.org_name = "modelscope"
        self.api_base = "https://api.github.com"
        self.github_token = os.getenv("GITHUB_TOKEN")  # 可选: 提高API限制

        if self.github_token:
            self.session.headers.update({
                'Authorization': f'token {self.github_token}'
            })

    def fetch_repositories(self) -> List[Dict]:
        """
        获取组织的所有仓库

        Returns:
            仓库列表
        """
        repos = []
        page = 1
        per_page = 100

        print(f"📦 获取 {self.org_name} 组织的仓库...")

        while True:
            url = f"{self.api_base}/orgs/{self.org_name}/repos?page={page}&per_page={per_page}"
            self._rate_limit_wait()

            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                page_repos = response.json()

                if not page_repos:
                    break

                repos.extend(page_repos)
                print(f"   页 {page}: 获取了 {len(page_repos)} 个仓库")
                page += 1

            except Exception as e:
                print(f"❌ 获取仓库列表失败: {e}")
                break

        print(f"✅ 共找到 {len(repos)} 个仓库")
        return repos

    def fetch_readme(self, repo_full_name: str) -> Optional[str]:
        """
        获取仓库README

        Args:
            repo_full_name: 仓库全名 (如 modelscope/modelscope)

        Returns:
            README内容
        """
        url = f"{self.api_base}/repos/{repo_full_name}/readme"
        self._rate_limit_wait()

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            # README内容是base64编码的
            import base64
            content = base64.b64decode(data['content']).decode('utf-8')
            return content

        except Exception as e:
            print(f"   ⚠️  无法获取README: {e}")
            return None

    def extract_repo_info(self, repo: Dict) -> Dict:
        """
        提取仓库信息

        Args:
            repo: GitHub API返回的仓库数据

        Returns:
            格式化的仓库信息
        """
        readme = self.fetch_readme(repo['full_name'])

        return {
            'name': repo['name'],
            'full_name': repo['full_name'],
            'description': repo.get('description', ''),
            'url': repo['html_url'],
            'stars': repo['stargazers_count'],
            'forks': repo['forks_count'],
            'language': repo.get('language', ''),
            'topics': repo.get('topics', []),
            'created_at': repo['created_at'],
            'updated_at': repo['updated_at'],
            'readme': readme,
            'source': 'github',
            'type': 'repository'
        }

    def crawl(self) -> List[Dict]:
        """
        执行爬取

        Returns:
            爬取的仓库信息列表
        """
        print("=" * 70)
        print(f"开始爬取 GitHub {self.org_name} 组织仓库")
        print("=" * 70)

        try:
            # 获取所有仓库
            repos = self.fetch_repositories()

            # 提取详细信息
            repo_data_list = []
            for i, repo in enumerate(repos, 1):
                print(f"\n📦 [{i}/{len(repos)}] 处理仓库: {repo['name']}")
                print(f"   ⭐ {repo['stargazers_count']} stars | 🍴 {repo['forks_count']} forks")

                repo_data = self.extract_repo_info(repo)
                repo_data_list.append(repo_data)

                # 保存JSON格式
                filename = f"repo_{repo['name']}.json"
                self.save_json(repo_data, filename)

                # 保存Markdown格式
                md_content = self.convert_to_markdown(repo_data)
                md_filename = f"repo_{repo['name']}"
                self.save_markdown(md_content, md_filename)

            # 保存汇总
            summary = {
                'total_repositories': len(repo_data_list),
                'repositories': [{'name': r['name'], 'stars': r['stars']} for r in repo_data_list],
                'metadata': self.get_metadata()
            }
            self.save_json(summary, 'summary.json')

            print("\n" + "=" * 70)
            print(f"✅ 爬取完成! 共 {len(repo_data_list)} 个仓库")
            print("=" * 70)

            return repo_data_list

        except Exception as e:
            print(f"\n❌ 爬取失败: {e}")
            import traceback
            traceback.print_exc()
            raise
