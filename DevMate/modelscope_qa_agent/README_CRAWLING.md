# 魔搭社区数据爬取系统

## 概述

根据需求文档 (`need.md`) 中"4. 官方指定数据资源"的要求,本系统实现了完整的数据爬取、处理和导入流程。

## 快速开始

### 1. 爬取所有数据源

```bash
# 激活环境
conda activate langchain-env

# 爬取所有数据源并处理
python scripts/crawl_and_process.py --all --process
```

这将爬取:
- 魔搭社区官方文档 (https://www.modelscope.cn/docs/overview)
- 研习社 (https://modelscope.cn/learn)
- GitHub仓库 (https://github.com/modelscope)
- 模型/数据集/创空间等目录

### 2. 导入到向量数据库

```bash
# 导入处理后的数据
python scripts/ingest_crawled_data.py --input data/processed/all_documents.jsonl
```

## 目录结构

```
crawlers/                        # 爬虫模块
├── __init__.py
├── base_crawler.py             # 基类
├── docs_crawler.py             # 文档爬虫
├── learn_crawler.py            # 研习社爬虫
├── github_crawler.py           # GitHub爬虫
├── catalog_crawler.py          # 目录爬虫
└── data_processor.py           # 数据处理

scripts/
├── crawl_and_process.py        # 爬取和处理脚本
└── ingest_crawled_data.py      # 数据导入脚本

data/
├── crawled/                    # 原始数据
│   ├── docs/
│   ├── learn/
│   ├── github/
│   └── catalog/
└── processed/                  # 处理后数据
    └── all_documents.jsonl

docs/
└── DATA_CRAWLING.md            # 详细文档
```

## 使用指南

### 爬取特定数据源

```bash
# 只爬取官方文档
python scripts/crawl_and_process.py --docs --process

# 爬取研习社和GitHub
python scripts/crawl_and_process.py --learn --github --process

# 只爬取目录
python scripts/crawl_and_process.py --catalog --process
```

### 只处理已爬取的数据

```bash
python scripts/crawl_and_process.py --process-only
```

### 自定义导入参数

```bash
# 调整批处理大小
python scripts/ingest_crawled_data.py --batch-size 100

# 测试加载(不实际导入)
python scripts/ingest_crawled_data.py --dry-run
```

## 功能特点

### 1. 多源爬取
- ✅ 官方文档 (递归爬取,深度3层)
- ✅ 研习社文章 (含作者、标签、代码块)
- ✅ GitHub仓库 (README、元数据、stars)
- ✅ 资源目录 (models/datasets/studios/mcp/aigc)

### 2. 数据处理
- ✅ 文本清理和标准化
- ✅ 智能分块 (chunk_size=800, overlap=150)
- ✅ 元数据提取和增强
- ✅ JSONL格式导出

### 3. 断点续爬
- ✅ 自动保存检查点
- ✅ 中断后可继续
- ✅ 避免重复爬取

### 4. 速率限制
- ✅ 遵守网站规则
- ✅ 可配置请求间隔
- ✅ 自动重试机制

## 配置说明

### 爬虫速率限制

在爬虫初始化时配置:

```python
from crawlers import DocsCrawler

crawler = DocsCrawler(
    output_dir="data/crawled/docs",
    rate_limit=2.0  # 2秒/请求
)
```

### GitHub Token (可选)

为提高API速率限制:

```bash
export GITHUB_TOKEN="your_github_token_here"
```

### 文本分块参数

在 `crawlers/data_processor.py` 中:

```python
chunks = processor.chunk_text(
    text,
    chunk_size=800,      # 与系统配置一致
    chunk_overlap=150     # 块重叠
)
```

## 数据统计

爬取完成后,查看统计信息:

```bash
# 查看汇总
cat data/crawled/docs/summary.json
cat data/crawled/learn/summary.json
cat data/crawled/github/summary.json
cat data/crawled/catalog/summary.json

# 查看处理后统计
cat data/processed/summary.json
```

## 故障排除

### 问题1: 请求被封禁

**症状**: 大量403/429错误

**解决**:
```bash
# 增加速率限制
# 在crawler初始化时设置 rate_limit=3.0
```

### 问题2: GitHub API限制

**症状**: GitHub爬取失败

**解决**:
```bash
# 设置GitHub Token
export GITHUB_TOKEN="your_token"
```

### 问题3: 内存不足

**症状**: 处理大数据时内存溢出

**解决**:
```bash
# 分批处理
python scripts/crawl_and_process.py --docs --process
python scripts/crawl_and_process.py --learn --process
# 然后分别导入
```

### 问题4: 导入失败

**症状**: 向量库导入报错

**解决**:
```bash
# 确保Milvus运行正常
docker ps | grep milvus

# 检查配置
cat config/settings.yaml

# 减小批处理大小
python scripts/ingest_crawled_data.py --batch-size 20
```

## 测试

```bash
# 运行爬虫测试
pytest tests/test_crawlers.py -v

# 测试单个功能
pytest tests/test_crawlers.py::TestDocsCrawler -v
```

## 最佳实践

### 1. 定期更新

建议每周更新一次:

```bash
# 完整更新流程
./update_data.sh  # 见下方脚本
```

创建 `update_data.sh`:

```bash
#!/bin/bash
set -e

echo "开始更新数据..."

# 爬取所有数据源
python scripts/crawl_and_process.py --all --process

# 导入向量库
python scripts/ingest_crawled_data.py --input data/processed/all_documents.jsonl

# 备份
tar -czf backup_$(date +%Y%m%d).tar.gz data/crawled/

echo "更新完成!"
```

### 2. 增量更新

对于频繁更新的源(如研习社):

```bash
# 每天更新learn
python scripts/crawl_and_process.py --learn --process
python scripts/ingest_crawled_data.py --input data/processed/learn_processed.json
```

### 3. 监控数据质量

```bash
# 检查文档数量
python -c "
import json
with open('data/processed/summary.json') as f:
    data = json.load(f)
    print(f'Total chunks: {data[\"total_chunks\"]}')
    for source, count in data['by_source'].items():
        print(f'  {source}: {count}')
"
```

## 相关文档

- 📘 [详细爬取文档](docs/DATA_CRAWLING.md) - 完整的技术文档
- 📋 [需求文档](../need.md) - 竞赛需求
- 🏗️ [项目README](README.md) - 项目总览

## 数据来源

根据竞赛需求文档 (`need.md`) 第4节:

| 数据源 | URL | 状态 |
|-------|-----|-----|
| 官方文档 | https://www.modelscope.cn/docs/overview | ✅ 已实现 |
| 研习社 | https://modelscope.cn/learn | ✅ 已实现 |
| GitHub | https://github.com/modelscope | ✅ 已实现 |
| 模型库 | https://modelscope.cn/models | ✅ 已实现 |
| 数据集 | https://modelscope.cn/datasets | ✅ 已实现 |
| 创空间 | https://modelscope.cn/studios | ✅ 已实现 |
| MCP | https://www.modelscope.cn/mcp | ✅ 已实现 |
| AIGC | https://www.modelscope.cn/aigc | ✅ 已实现 |

## 注意事项

- ⚠️ 遵守robots.txt规则
- ⚠️ 合理设置速率限制
- ⚠️ 爬取的内容仅用于比赛和学习
- ⚠️ 定期备份原始数据
- ⚠️ 监控爬取质量和完整性

## 支持

如有问题,请:
1. 查看详细文档: `docs/DATA_CRAWLING.md`
2. 运行测试: `pytest tests/test_crawlers.py -v`
3. 检查日志输出

---

**最后更新**: 2025-12-02
**作者**: DevMate Team
