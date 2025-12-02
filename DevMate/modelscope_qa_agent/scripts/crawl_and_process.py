#!/usr/bin/env python3
"""
爬取并处理魔搭社区数据

使用方法:
    python scripts/crawl_and_process.py --all
    python scripts/crawl_and_process.py --docs --learn
    python scripts/crawl_and_process.py --process-only
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from crawlers import DocsCrawler, LearnCrawler, GitHubCrawler, CatalogCrawler
from crawlers.data_processor import DataProcessor


def crawl_docs(output_dir: str = "data/crawled"):
    """爬取官方文档"""
    print("\n" + "🌐" * 35)
    crawler = DocsCrawler(output_dir=f"{output_dir}/docs")
    crawler.crawl()


def crawl_learn(output_dir: str = "data/crawled"):
    """爬取研习社"""
    print("\n" + "🌐" * 35)
    crawler = LearnCrawler(output_dir=f"{output_dir}/learn")
    crawler.crawl()


def crawl_github(output_dir: str = "data/crawled"):
    """爬取GitHub仓库"""
    print("\n" + "🌐" * 35)
    crawler = GitHubCrawler(output_dir=f"{output_dir}/github")
    crawler.crawl()


def crawl_catalog(output_dir: str = "data/crawled"):
    """爬取资源目录"""
    print("\n" + "🌐" * 35)
    crawler = CatalogCrawler(output_dir=f"{output_dir}/catalog")
    crawler.crawl()


def process_data(crawled_dir: str = "data/crawled", processed_dir: str = "data/processed"):
    """处理数据"""
    print("\n" + "⚙️" * 35)
    processor = DataProcessor(crawled_data_dir=crawled_dir, processed_data_dir=processed_dir)
    processor.process_all()
    processor.export_for_ingestion()


def main():
    parser = argparse.ArgumentParser(description="爬取并处理魔搭社区数据")

    # 爬取选项
    parser.add_argument("--all", action="store_true", help="爬取所有数据源")
    parser.add_argument("--docs", action="store_true", help="爬取官方文档")
    parser.add_argument("--learn", action="store_true", help="爬取研习社")
    parser.add_argument("--github", action="store_true", help="爬取GitHub仓库")
    parser.add_argument("--catalog", action="store_true", help="爬取资源目录")

    # 处理选项
    parser.add_argument("--process", action="store_true", help="处理爬取的数据")
    parser.add_argument("--process-only", action="store_true", help="只处理数据(不爬取)")

    # 目录选项
    parser.add_argument("--output-dir", default="data/crawled", help="爬取数据输出目录")
    parser.add_argument("--processed-dir", default="data/processed", help="处理后数据输出目录")

    args = parser.parse_args()

    # 如果没有指定任何选项，显示帮助
    if not any([args.all, args.docs, args.learn, args.github, args.catalog, args.process_only]):
        parser.print_help()
        return

    print("=" * 70)
    print("魔搭社区数据爬取与处理")
    print("=" * 70)

    # 爬取阶段
    if not args.process_only:
        if args.all or args.docs:
            try:
                crawl_docs(args.output_dir)
            except KeyboardInterrupt:
                print("\n⚠️  用户中断 docs 爬取")
            except Exception as e:
                print(f"\n❌ docs 爬取失败: {e}")

        if args.all or args.learn:
            try:
                crawl_learn(args.output_dir)
            except KeyboardInterrupt:
                print("\n⚠️  用户中断 learn 爬取")
            except Exception as e:
                print(f"\n❌ learn 爬取失败: {e}")

        if args.all or args.github:
            try:
                crawl_github(args.output_dir)
            except KeyboardInterrupt:
                print("\n⚠️  用户中断 github 爬取")
            except Exception as e:
                print(f"\n❌ github 爬取失败: {e}")

        if args.all or args.catalog:
            try:
                crawl_catalog(args.output_dir)
            except KeyboardInterrupt:
                print("\n⚠️  用户中断 catalog 爬取")
            except Exception as e:
                print(f"\n❌ catalog 爬取失败: {e}")

    # 处理阶段
    if args.process or args.process_only or args.all:
        try:
            process_data(args.output_dir, args.processed_dir)
        except Exception as e:
            print(f"\n❌ 数据处理失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("✅ 完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
