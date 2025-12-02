# 数据爬取与处理系统

本文档说明如何爬取魔搭社区官方数据资源并导入到向量数据库。

## 📋 目录

- [数据来源](#数据来源)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [详细使用](#详细使用)
- [配置说明](#配置说明)
- [故障排除](#故障排除)

---

## 数据来源

根据竞赛需求文档(`need.md`)，系统爬取以下官方数据资源:

1. **魔搭社区官方文档** - https://www.modelscope.cn/docs/overview
2. **研习社** - https://modelscope.cn/learn
3. **GitHub仓库** - https://github.com/modelscope
4. **模型库** - https://modelscope.cn/models
5. **数据集** - https://modelscope.cn/datasets
6. **创空间** - https://modelscope.cn/studios
7. **MCP** - https://www.modelscope.cn/mcp
8. **AIGC** - https://www.modelscope.cn/aigc

---

## 系统架构

### 模块组成

```
crawlers/
├── __init__.py              # 模块初始化
├── base_crawler.py          # 爬虫基类
├── docs_crawler.py          # 官方文档爬虫
├── learn_crawler.py         # 研习社爬虫
├── github_crawler.py        # GitHub爬虫
├── catalog_crawler.py       # 资源目录爬虫
└── data_processor.py        # 数据处理器

scripts/
├── crawl_and_process.py     # 爬取和处理脚本
└── ingest_crawled_data.py   # 数据导入脚本

data/
├── crawled/                 # 原始爬取数据
│   ├── docs/
│   ├── learn/
│   ├── github/
│   └── catalog/
└── processed/               # 处理后数据
    ├── docs_processed.json
    ├── learn_processed.json
    ├── github_processed.json
    ├── catalog_processed.json
    └── all_documents.jsonl
```

### 数据流程

```
1. 爬取阶段
   各个爬虫 → 原始JSON文件 → data/crawled/

2. 处理阶段
   data/crawled/ → DataProcessor → data/processed/

3. 导入阶段
   data/processed/all_documents.jsonl → MilvusVectorStore
```

---

## 快速开始

### 1. 安装依赖

```bash
conda activate langchain-env
pip install requests beautifulsoup4 pymilvus
```

### 2. 爬取所有数据并导入

```bash
# 爬取所有数据源并处理
python scripts/crawl_and_process.py --all --process

# 导入到向量数据库
python scripts/ingest_crawled_data.py
```

---

## 详细使用

### 爬取特定数据源

```bash
# 只爬取官方文档
python scripts/crawl_and_process.py --docs

# 爬取研习社和GitHub
python scripts/crawl_and_process.py --learn --github

# 爬取资源目录
python scripts/crawl_and_process.py --catalog
```

### 处理已爬取的数据

```bash
# 只处理数据(不爬取)
python scripts/crawl_and_process.py --process-only
```

### 自定义输出目录

```bash
# 指定输出目录
python scripts/crawl_and_process.py --all \
    --output-dir my_data/raw \
    --processed-dir my_data/processed
```

### 数据导入选项

```bash
# 指定输入文件
python scripts/ingest_crawled_data.py --input data/processed/all_documents.jsonl

# 调整批处理大小
python scripts/ingest_crawled_data.py --batch-size 100

# 仅测试加载(不实际导入)
python scripts/ingest_crawled_data.py --dry-run
```

---

## 配置说明

### 爬虫配置

#### 速率限制

每个爬虫都有速率限制以避免被封禁:

- **DocsCrawler**: 1.5秒/请求
- **LearnCrawler**: 1.5秒/请求
- **GitHubCrawler**: 2.0秒/请求
- **CatalogCrawler**: 1.5秒/请求

可在初始化时调整:

```python
crawler = DocsCrawler(rate_limit=2.0)  # 2秒/请求
```

#### GitHub Token (可选)

为提高GitHub API速率限制:

```bash
export GITHUB_TOKEN="your_github_token_here"
```

### 数据处理配置

#### 文本分块参数

在 `data_processor.py` 中:

```python
def chunk_text(self, text: str, chunk_size: int = 800, chunk_overlap: int = 150):
    # chunk_size: 每块文本大小
    # chunk_overlap: 块之间重叠大小
```

这与系统配置(`config/settings.yaml`)保持一致。

---

## 爬虫详细说明

### 1. DocsCrawler - 官方文档爬虫

**特点:**
- 递归爬取,最大深度为3
- 自动发现文档链接
- 提取标题、正文、代码块
- 支持断点续爬

**数据结构:**
```json
{
  "url": "https://www.modelscope.cn/docs/...",
  "title": "文档标题",
  "content": "文档内容...",
  "code_blocks": ["代码块1", "代码块2"],
  "source": "modelscope_docs",
  "type": "documentation"
}
```

### 2. LearnCrawler - 研习社爬虫

**特点:**
- 爬取研习社文章
- 提取作者、日期、标签
- 过滤过短内容
- 支持断点续爬

**数据结构:**
```json
{
  "url": "https://modelscope.cn/learn/...",
  "title": "文章标题",
  "author": "作者",
  "date": "发布日期",
  "content": "文章内容...",
  "code_blocks": ["..."],
  "tags": ["标签1", "标签2"],
  "source": "modelscope_learn",
  "type": "article"
}
```

### 3. GitHubCrawler - GitHub仓库爬虫

**特点:**
- 使用GitHub API
- 爬取组织所有仓库
- 获取README内容
- 提取仓库元数据(stars, forks等)

**数据结构:**
```json
{
  "name": "modelscope",
  "full_name": "modelscope/modelscope",
  "description": "仓库描述",
  "url": "https://github.com/modelscope/modelscope",
  "stars": 1234,
  "forks": 567,
  "language": "Python",
  "topics": ["ml", "ai"],
  "readme": "README内容...",
  "source": "github",
  "type": "repository"
}
```

### 4. CatalogCrawler - 资源目录爬虫

**特点:**
- 爬取多个资源目录
- 提取资源卡片信息
- 每个目录限制50项(可调整)

**支持目录:**
- models (模型库)
- datasets (数据集)
- studios (创空间)
- mcp (MCP)
- aigc (AIGC)

**数据结构:**
```json
{
  "title": "资源名称",
  "url": "资源URL",
  "description": "资源描述",
  "tags": ["标签1", "标签2"],
  "catalog_type": "models",
  "source": "modelscope_models",
  "type": "catalog_item"
}
```

---

## 数据处理流程

### 1. 文本清理

- 移除多余空白字符
- 标准化文本格式

### 2. 文本分块

- 默认块大小: 800字符
- 默认重叠: 150字符
- 智能分割(尝试在句号、段落处分割)

### 3. 元数据提取

每个文档块包含:

```json
{
  "content": "文档块内容",
  "metadata": {
    "source_type": "docs",
    "title": "文档标题",
    "url": "原始URL",
    "original_source": "modelscope_docs",
    "chunk_id": 0,
    "total_chunks": 5,
    "author": "作者(如果有)",
    "tags": "标签1,标签2",
    "stars": 1234
  }
}
```

### 4. 导出格式

最终导出为JSONL格式 (`all_documents.jsonl`):
- 每行一个JSON对象
- 方便流式处理
- 易于增量导入

---

## 向量数据库导入

### 导入流程

1. **加载文档**: 从JSONL文件加载
2. **转换格式**: 转为LangChain Document对象
3. **批量导入**: 分批导入向量库(默认50个/批)
4. **验证**: 检查导入统计

### 性能优化

- **批处理**: 减少网络往返次数
- **错误处理**: 单批失败不影响其他批
- **统计信息**: 实时显示导入进度

### 查看导入结果

```bash
# 启动服务器后访问
curl http://localhost:8000/api/status

# 或通过Web界面查看文档统计
```

---

## 故障排除

### 问题1: 爬取被封禁

**症状**: 大量请求失败,返回403/429错误

**解决方案**:
1. 增加速率限制: `rate_limit=3.0` (3秒/请求)
2. 使用代理(如需要)
3. 分多次爬取

### 问题2: GitHub API限制

**症状**: GitHub爬取失败,提示rate limit

**解决方案**:
```bash
# 设置GitHub Token
export GITHUB_TOKEN="your_token"

# 未认证: 60次/小时
# 已认证: 5000次/小时
```

### 问题3: 内存不足

**症状**: 处理大量数据时内存溢出

**解决方案**:
1. 分批处理:
```bash
# 分别爬取和处理
python scripts/crawl_and_process.py --docs --process
python scripts/crawl_and_process.py --learn --process
# ...
```

2. 减小批处理大小:
```bash
python scripts/ingest_crawled_data.py --batch-size 20
```

### 问题4: 断点续爬

**症状**: 爬取中断

**解决方案**:
爬虫自动保存检查点到 `checkpoint.json`。
重新运行命令即可从断点继续。

### 问题5: 页面结构变化

**症状**: 提取不到内容

**解决方案**:
检查并更新爬虫的选择器:
```python
# 在对应的爬虫文件中调整
content_elem = soup.find('article') or soup.find('main')
```

---

## 最佳实践

### 1. 定期更新数据

建议每周或每月重新爬取:
```bash
# 完整更新流程
python scripts/crawl_and_process.py --all --process
python scripts/ingest_crawled_data.py
```

### 2. 增量更新

对于频繁更新的源(如研习社):
```bash
# 只爬取learn并导入
python scripts/crawl_and_process.py --learn --process
python scripts/ingest_crawled_data.py --input data/processed/learn_processed.json
```

### 3. 监控数据质量

定期检查:
- 文档数量统计
- 内容完整性
- 元数据准确性

### 4. 备份原始数据

```bash
# 备份爬取的原始数据
tar -czf backup_$(date +%Y%m%d).tar.gz data/crawled/
```

---

## 扩展开发

### 添加新的爬虫

1. 继承 `BaseCrawler`:

```python
from crawlers.base_crawler import BaseCrawler

class MyCrawler(BaseCrawler):
    def __init__(self, output_dir: str = "data/crawled/my_source"):
        super().__init__(output_dir)

    def crawl(self) -> List[Dict]:
        # 实现爬取逻辑
        pass
```

2. 在 `__init__.py` 中注册:

```python
from .my_crawler import MyCrawler

__all__ = [..., 'MyCrawler']
```

3. 在脚本中使用:

```python
def crawl_my_source(output_dir: str = "data/crawled"):
    crawler = MyCrawler(output_dir=f"{output_dir}/my_source")
    crawler.crawl()
```

---

## 许可与注意事项

- **遵守robots.txt**: 爬虫尊重网站的robots.txt规则
- **速率限制**: 避免对服务器造成过大压力
- **版权**: 爬取的内容仅用于比赛和学习目的
- **更新频率**: 建议合理安排爬取频率

---

## 相关文档

- [need.md](../../need.md) - 竞赛需求文档
- [README.md](../README.md) - 项目总览
- [API文档](../api/README.md) - API说明

---

**最后更新**: 2025-12-02
