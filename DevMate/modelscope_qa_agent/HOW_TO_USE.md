# 如何使用 Markdown 导出功能

## 🎯 快速开始（3步）

### 步骤1: 进入项目目录

```bash
cd /Users/yuyansong/AiProject/Langchain/DevMate/modelscope_qa_agent
```

### 步骤2: 运行启动脚本

```bash
./START_CRAWLING.sh
```

### 步骤3: 选择爬取选项

```
1) 爬取所有数据源 (docs + learn + github + catalog)
2) 只爬取官方文档 (docs)
3) 只爬取研习社 (learn)
4) 只爬取GitHub仓库 (github)
5) 只爬取资源目录 (catalog)
6) 运行测试 (查看Markdown导出示例)  ← 推荐先试这个
```

## 📝 示例：运行测试

```bash
cd /Users/yuyansong/AiProject/Langchain/DevMate/modelscope_qa_agent
./START_CRAWLING.sh
# 输入: 6
```

输出：
```
✅ 生成了 4 个Markdown文件:
   - test_repo.md (355 字节)
   - test_doc.md (377 字节)
   - test_article.md (420 字节)
   - test_catalog.md (279 字节)
```

## 📁 查看生成的文件

```bash
# 查看测试生成的Markdown文件
ls data/test_markdown/markdown/

# 查看文件内容
cat data/test_markdown/markdown/test_doc.md
cat data/test_markdown/markdown/test_article.md
cat data/test_markdown/markdown/test_repo.md
```

## 🚀 实际爬取数据

### 爬取所有数据源

```bash
cd /Users/yuyansong/AiProject/Langchain/DevMate/modelscope_qa_agent
./START_CRAWLING.sh
# 输入: 1
```

### 只爬取官方文档

```bash
cd /Users/yuyansong/AiProject/Langchain/DevMate/modelscope_qa_agent
./START_CRAWLING.sh
# 输入: 2
```

### 或者直接使用命令行

```bash
cd /Users/yuyansong/AiProject/Langchain/DevMate/modelscope_qa_agent
conda activate langchain-env

# 爬取所有数据源
python scripts/crawl_and_process.py --all --process

# 爬取特定数据源
python scripts/crawl_and_process.py --docs --process
python scripts/crawl_and_process.py --learn --process
python scripts/crawl_and_process.py --github --process
python scripts/crawl_and_process.py --catalog --process
```

## 📂 Markdown文件保存位置

爬取完成后，Markdown文件会保存在：

```
data/crawled/
├── docs/markdown/          ← 官方文档的Markdown
├── learn/markdown/         ← 研习社文章的Markdown
├── github/markdown/        ← GitHub仓库的Markdown
└── catalog/markdown/       ← 资源目录的Markdown
```

## 🔍 查看和搜索

### 查看Markdown文件

```bash
cd /Users/yuyansong/AiProject/Langchain/DevMate/modelscope_qa_agent

# 列出所有Markdown文件
ls data/crawled/docs/markdown/
ls data/crawled/learn/markdown/
ls data/crawled/github/markdown/
ls data/crawled/catalog/markdown/

# 查看特定文件
cat data/crawled/docs/markdown/doc_1.md
```

### 搜索内容

```bash
cd /Users/yuyansong/AiProject/Langchain/DevMate/modelscope_qa_agent

# 搜索关键词
grep -r "模型训练" data/crawled/*/markdown/

# 搜索并显示行号
grep -rn "ModelScope" data/crawled/*/markdown/

# 统计Markdown文件数量
find data/crawled/*/markdown -name "*.md" | wc -l
```

## 💡 Markdown格式说明

每个Markdown文件包含：

### 文档类型
```markdown
# 标题
---
**URL**: 原始网址
**来源**: modelscope_docs
---

## 内容
正文内容...

## 代码示例
代码块...
```

### 文章类型
```markdown
# 标题
---
**URL**: 原始网址
**作者**: 作者名
**日期**: 2025-12-02
**标签**: 标签1, 标签2
---

## 内容
正文内容...
```

### GitHub仓库
```markdown
# 仓库名
---
**URL**: GitHub地址
**语言**: Python
**Stars**: ⭐ 数量
**Forks**: 🍴 数量
---

## 描述
仓库描述...

## README
README内容...
```

## ❓ 常见问题

### Q: 脚本提示找不到文件？
**A**: 确保在正确的目录：
```bash
cd /Users/yuyansong/AiProject/Langchain/DevMate/modelscope_qa_agent
```

### Q: 如何只生成Markdown不生成JSON？
**A**: 当前两种格式都会生成。如需只保留Markdown：
```bash
# 爬取后删除JSON文件（保留summary.json）
find data/crawled -name "*.json" -not -name "summary.json" -not -name "checkpoint.json" -delete
```

### Q: Markdown文件可以编辑吗？
**A**: 可以！用任何文本编辑器：
```bash
vim data/crawled/docs/markdown/doc_1.md
code data/crawled/docs/markdown/  # VS Code
open data/crawled/docs/markdown/  # macOS Finder
```

### Q: 如何将Markdown转为PDF？
**A**: 使用pandoc：
```bash
# 安装pandoc
brew install pandoc

# 转换为PDF
pandoc data/crawled/docs/markdown/doc_1.md -o doc_1.pdf
```

## 📚 更多文档

- **QUICK_START.md** - 详细的快速开始指南
- **MARKDOWN_FEATURE_SUMMARY.md** - 功能实现总结
- **docs/MARKDOWN_EXPORT.md** - 技术文档
- **README_CRAWLING.md** - 完整的爬虫系统指南

---

**最简单的开始方式：**

```bash
cd /Users/yuyansong/AiProject/Langchain/DevMate/modelscope_qa_agent
./START_CRAWLING.sh
# 选择 6 运行测试
```

就这么简单！ 🎉
