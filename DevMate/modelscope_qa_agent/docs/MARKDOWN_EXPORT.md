# Markdown导出功能

## 概述

爬虫系统现在支持将爬取的数据自动转换为Markdown格式并保存到单独的文件夹中,方便阅读、编��和版本管理。

## 功能特点

### 1. 自动Markdown转换

- ✅ 所有爬取的数据自动转换为Markdown格式
- ✅ 保留原始JSON格式和Markdown格式两份
- ✅ Markdown文件保存在独立的`markdown/`子目录中

### 2. 目录结构

```
data/crawled/
├── docs/
│   ├── doc_1.json          # JSON格式
│   ├── doc_2.json
│   └── markdown/           # Markdown文件夹
│       ├── doc_1.md        # Markdown格式
│       └── doc_2.md
├── learn/
│   ├── article_1.json
│   └── markdown/
│       └── article_1.md
├── github/
│   ├── repo_modelscope.json
│   └── markdown/
│       └── repo_modelscope.md
└── catalog/
    ├── models_catalog.json
    └── markdown/
        ├── models_1.md
        └── models_2.md
```

### 3. Markdown格式

每个Markdown文件包含:

#### 文档类型 (docs)
```markdown
# 文档标题
---
**URL**: https://www.modelscope.cn/docs/...
**来源**: modelscope_docs
---

## 内容

文档正文内容...

## 代码示例

### 示例 1
\`\`\`
代码块内容
\`\`\`
```

#### 文章类型 (learn)
```markdown
# 文章标题
---
**URL**: https://modelscope.cn/learn/...
**作者**: 作者名称
**日期**: 2025-12-02
**来源**: modelscope_learn
**标签**: 标签1, 标签2
---

## 内容

文章正文...

## 代码示例
...
```

#### GitHub仓库类型 (github)
```markdown
# 仓库名称
---
**URL**: https://github.com/modelscope/...
**来源**: github
**语言**: Python
**Stars**: ⭐ 5678
**Forks**: 🍴 1234
---

## 描述

仓库描述...

## README

README内容...
```

#### 目录项类型 (catalog)
```markdown
# 资源名称
---
**URL**: https://modelscope.cn/models/...
**来源**: modelscope_models
**标签**: 标签1, 标签2
---

## 描述

资源描述...
```

## 使用方法

### 1. 爬取数据(自动生成Markdown)

```bash
# 爬取所有数据源
python scripts/crawl_and_process.py --all --process

# 爬取特定数据源
python scripts/crawl_and_process.py --docs --process
```

爬取完成后,Markdown文件会自动保存在对应的`markdown/`子目录中。

### 2. 查看Markdown文件

```bash
# 查看docs的Markdown文件
ls data/crawled/docs/markdown/

# 查看learn的Markdown文件
ls data/crawled/learn/markdown/

# 查看github的Markdown文件
ls data/crawled/github/markdown/

# 查看catalog的Markdown文件
ls data/crawled/catalog/markdown/
```

### 3. 测试Markdown导出功能

```bash
# 运行测试脚本
python scripts/test_markdown_export.py

# 查看测试结果
cat data/test_markdown/markdown/test_doc.md
```

## 技术实现

### BaseCrawler新增方法

#### save_markdown()
```python
def save_markdown(self, content: str, filename: str):
    """
    保存Markdown文档

    Args:
        content: Markdown内容
        filename: 文件名 (自动添加.md后缀)
    """
```

#### convert_to_markdown()
```python
def convert_to_markdown(self, data: Dict) -> str:
    """
    将数据转换为Markdown格式

    Args:
        data: 文档数据字典

    Returns:
        Markdown格式文本
    """
```

### 各爬虫集成

所有爬虫 (DocsCrawler, LearnCrawler, GitHubCrawler, CatalogCrawler) 都已集成Markdown导出功能:

```python
# 保存JSON格式
self.save_json(doc_data, filename)

# 保存Markdown格式
md_content = self.convert_to_markdown(doc_data)
self.save_markdown(md_content, md_filename)
```

## Markdown文件优势

### 1. 可读性强
- 格式清晰,易于阅读
- 支持语法高亮
- 可在GitHub、GitLab等平台直接预览

### 2. 易于编辑
- 纯文本格式
- 可用任何文本编辑器编辑
- 支持版本控制 (Git)

### 3. 便于分享
- 可直接分享给团队成员
- 可生成HTML、PDF等格式
- 可集成到文档系统

### 4. 利于检索
- 纯文本,易于全文搜索
- 可用grep、ripgrep等工具快速检索
- 支持IDE内搜索

## 使用场景

### 1. 数据审查
```bash
# 查看爬取的文档质量
cat data/crawled/docs/markdown/doc_1.md

# 批量检查
for f in data/crawled/docs/markdown/*.md; do
    echo "=== $f ==="
    head -20 "$f"
done
```

### 2. 内容编辑
```bash
# 使用编辑器打开Markdown文件
vim data/crawled/docs/markdown/doc_1.md

# 或使用VS Code
code data/crawled/docs/markdown/
```

### 3. 版本管理
```bash
# 将Markdown文件加入Git
git add data/crawled/*/markdown/
git commit -m "Add crawled markdown documents"
```

### 4. 文档生成
```bash
# 使用pandoc转换为其他格式
pandoc doc_1.md -o doc_1.pdf
pandoc doc_1.md -o doc_1.html
```

### 5. 全文搜索
```bash
# 使用grep搜索关键词
grep -r "模型训练" data/crawled/*/markdown/

# 使用ripgrep (更快)
rg "模型训练" data/crawled/*/markdown/
```

## 配置选项

### 自定义Markdown输出目录

如果需要自定义Markdown输出目录,可以修改爬虫初始化:

```python
# 在 base_crawler.py 中
self.markdown_dir = self.output_dir / "markdown"  # 默认
# 可以改为:
self.markdown_dir = Path("custom/markdown/path")
```

### 自定义Markdown格式

如果需要自定义Markdown格式,可以覆盖`convert_to_markdown()`方法:

```python
class CustomCrawler(BaseCrawler):
    def convert_to_markdown(self, data: Dict) -> str:
        # 自定义Markdown格式
        md = f"# {data['title']}\n\n"
        md += f"{data['content']}\n"
        return md
```

## 数据统计

### 查看Markdown文件统计

```bash
# 统计Markdown文件数量
find data/crawled/*/markdown -name "*.md" | wc -l

# 统计总字数
find data/crawled/*/markdown -name "*.md" -exec wc -w {} + | tail -1

# 按目录统计
for dir in data/crawled/*/markdown; do
    count=$(find "$dir" -name "*.md" | wc -l)
    echo "$(basename $(dirname $dir)): $count 个文件"
done
```

## 注意事项

### 1. 存储空间
- Markdown文件会额外占用存储空间
- 通常比JSON文件略小
- 可以选择只保留其中一种格式

### 2. 文件编码
- 所有Markdown文件使用UTF-8编码
- 确保编辑器支持UTF-8

### 3. 特殊字符
- Markdown中的特殊字符会被正确处理
- 代码块使用围栏代码块(```)包裹

### 4. 文件名
- 文件名自动生成,避免冲突
- 可能包含序号或仓库名

## 示例

### 完整工作流程

```bash
# 1. 爬取数据
python scripts/crawl_and_process.py --docs --process

# 2. 查看生成的Markdown文件
ls -lh data/crawled/docs/markdown/

# 3. 查看第一个文档
cat data/crawled/docs/markdown/doc_1.md

# 4. 搜索特定内容
grep -n "快速入门" data/crawled/docs/markdown/*.md

# 5. 统计文档数量
echo "共生成 $(find data/crawled/docs/markdown -name '*.md' | wc -l) 个Markdown文档"
```

## 相关文档

- [爬取系统文档](DATA_CRAWLING.md) - 完整的爬取系统文档
- [README_CRAWLING.md](../README_CRAWLING.md) - 快速开始指南
- [need.md](../../need.md) - 竞赛需求文档

---

**最后更新**: 2025-12-02
