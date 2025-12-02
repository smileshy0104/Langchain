#!/usr/bin/env python3
"""
演示爬取单个页面并生成Markdown

使用方法:
    python scripts/demo_crawl_single.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from crawlers import DocsCrawler


def demo_crawl():
    """演示爬取单个文档页面"""
    print("=" * 70)
    print("演示爬取魔搭社区页面并生成Markdown")
    print("=" * 70)

    # 创建爬虫
    crawler = DocsCrawler(output_dir="data/demo_crawl", rate_limit=2.0)
    crawler.max_depth = 0  # 只爬取一个页面

    print("\n正在爬取魔搭社区首页...")
    print("URL: https://www.modelscope.cn/docs/overview")

    try:
        # 爬取页面
        documents = crawler.crawl()

        print("\n" + "=" * 70)
        print("✅ 爬取完成!")
        print("=" * 70)

        if documents:
            print(f"\n📊 爬取了 {len(documents)} 个文档")

            # 检查生成的文件
            print("\n📁 生成的文件:")

            # JSON文件
            json_files = list(Path("data/demo_crawl").glob("*.json"))
            if json_files:
                print(f"\nJSON格式 ({len(json_files)} 个文件):")
                for f in json_files:
                    if f.name != "checkpoint.json":
                        size = f.stat().st_size
                        print(f"   ✓ {f.name} ({size} 字节)")

            # Markdown文件
            md_dir = Path("data/demo_crawl/markdown")
            if md_dir.exists():
                md_files = list(md_dir.glob("*.md"))
                if md_files:
                    print(f"\nMarkdown格式 ({len(md_files)} 个文件):")
                    for f in md_files:
                        size = f.stat().st_size
                        print(f"   ✓ {f.name} ({size} 字节)")

                    # 显示第一个Markdown文件的内容
                    if md_files:
                        print(f"\n📄 {md_files[0].name} 内容预览:")
                        print("-" * 70)
                        with open(md_files[0], 'r', encoding='utf-8') as f:
                            content = f.read()
                            lines = content.split('\n')[:30]
                            print('\n'.join(lines))
                            if len(content.split('\n')) > 30:
                                print("\n... (内容较长，仅显示前30行)")
                        print("-" * 70)

                        print(f"\n💡 完整文件路径:")
                        print(f"   JSON: {Path('data/demo_crawl').absolute()}")
                        print(f"   Markdown: {md_dir.absolute()}")
        else:
            print("\n⚠️  未爬取到文档")

    except Exception as e:
        print(f"\n❌ 爬取失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_crawl()
