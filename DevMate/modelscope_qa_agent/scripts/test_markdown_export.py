#!/usr/bin/env python3
"""
测试Markdown导出功能

使用方法:
    python scripts/test_markdown_export.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from crawlers.base_crawler import BaseCrawler


class TestCrawler(BaseCrawler):
    """测试爬虫"""

    def crawl(self):
        pass


def test_markdown_conversion():
    """测试Markdown转换功能"""
    print("=" * 70)
    print("测试Markdown导出功能")
    print("=" * 70)

    # 创建测试爬虫
    crawler = TestCrawler(output_dir="data/test_markdown", rate_limit=0.1)

    # 测试1: 文档类型
    print("\n📝 测试1: 文档类型数据")
    doc_data = {
        'title': '魔搭社区快速入门',
        'url': 'https://www.modelscope.cn/docs/quickstart',
        'content': '这是一个快速入门教程。\n\n第一步：安装依赖。\n第二步：配置环境。\n第三步：运行示例。',
        'code_blocks': [
            'pip install modelscope',
            'from modelscope import Model\nmodel = Model()'
        ],
        'source': 'modelscope_docs',
        'type': 'documentation'
    }

    md_content = crawler.convert_to_markdown(doc_data)
    crawler.save_markdown(md_content, 'test_doc')
    print("✅ 文档类型测试完成")

    # 测试2: 文章类型
    print("\n📝 测试2: 文章类型数据")
    article_data = {
        'title': '如何使用ModelScope进行模型训练',
        'url': 'https://modelscope.cn/learn/article/123',
        'author': '张三',
        'date': '2025-12-02',
        'content': '本文介绍如何使用ModelScope平台进行模型训练。\n\n## 准备工作\n\n首先需要注册账号...',
        'code_blocks': [
            'modelscope train --config train.yaml'
        ],
        'tags': ['机器学习', '模型训练', 'ModelScope'],
        'source': 'modelscope_learn',
        'type': 'article'
    }

    md_content = crawler.convert_to_markdown(article_data)
    crawler.save_markdown(md_content, 'test_article')
    print("✅ 文章类型测试完成")

    # 测试3: GitHub仓库类型
    print("\n📝 测试3: GitHub仓库类型数据")
    repo_data = {
        'name': 'modelscope',
        'full_name': 'modelscope/modelscope',
        'title': 'modelscope',
        'description': 'ModelScope: bring the notion of Model-as-a-Service to life.',
        'url': 'https://github.com/modelscope/modelscope',
        'stars': 5678,
        'forks': 1234,
        'language': 'Python',
        'topics': ['machine-learning', 'deep-learning', 'nlp'],
        'readme': '# ModelScope\n\nModelScope是一个开源的模型即服务平台...\n\n## 快速开始\n\n```bash\npip install modelscope\n```',
        'source': 'github',
        'type': 'repository'
    }

    md_content = crawler.convert_to_markdown(repo_data)
    crawler.save_markdown(md_content, 'test_repo')
    print("✅ GitHub仓库类型测试完成")

    # 测试4: 目录项类型
    print("\n📝 测试4: 目录项类型数据")
    catalog_data = {
        'title': 'Qwen2.5-72B-Instruct',
        'url': 'https://modelscope.cn/models/qwen/Qwen2.5-72B-Instruct',
        'description': '通义千问2.5是阿里云研发的大语言模型，在多个基准测试中取得优异成绩。',
        'tags': ['NLP', '大语言模型', '中文'],
        'catalog_type': 'models',
        'source': 'modelscope_models',
        'type': 'catalog_item'
    }

    md_content = crawler.convert_to_markdown(catalog_data)
    crawler.save_markdown(md_content, 'test_catalog')
    print("✅ 目录项类型测试完成")

    # 显示结果
    print("\n" + "=" * 70)
    print("📊 测试结果")
    print("=" * 70)

    markdown_dir = Path("data/test_markdown/markdown")
    if markdown_dir.exists():
        md_files = list(markdown_dir.glob("*.md"))
        print(f"✅ 生成了 {len(md_files)} 个Markdown文件:")
        for md_file in md_files:
            size = md_file.stat().st_size
            print(f"   - {md_file.name} ({size} 字节)")

        # 显示第一个文件的内容预览
        if md_files:
            print(f"\n📄 {md_files[0].name} 预览:")
            print("-" * 70)
            with open(md_files[0], 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')[:20]  # 只显示前20行
                print('\n'.join(lines))
                if len(content.split('\n')) > 20:
                    print("...")
            print("-" * 70)
    else:
        print("❌ 未找到Markdown文件")

    print("\n✅ 所有测试完成!")
    print(f"📁 Markdown文件保存在: {markdown_dir}")


if __name__ == "__main__":
    test_markdown_conversion()
