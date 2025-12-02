# 🚀 快速开始 - Markdown导出功能

## ✨ 新功能亮点

所有爬取的数据现在会**自动转换为Markdown格式**并保存在独立的`markdown/`文件夹中！

## 📦 三种启动方式

### 方式1: 使用快速启动脚本 (推荐)

```bash
./START_CRAWLING.sh
```

然后根据菜单选择：
- `1` - 爬取所有数据源
- `2` - 只爬取官方文档
- `3` - 只爬取研习社
- `4` - 只爬取GitHub仓库
- `5` - 只爬取资源目录
- `6` - 运行测试查看示例

### 方式2: 直接运行Python脚本

```bash
# 激活环境
conda activate langchain-env

# 爬取所有数据源
python scripts/crawl_and_process.py --all --process

# 或爬取特定数据源
python scripts/crawl_and_process.py --docs --process
python scripts/crawl_and_process.py --learn --process
python scripts/crawl_and_process.py --github --process
python scripts/crawl_and_process.py --catalog --process
```

### 方式3: 测试功能

```bash
# 运行测试查看Markdown导出示例
python scripts/test_markdown_export.py

# 查看生成的测试文件
ls data/test_markdown/markdown/
cat data/test_markdown/markdown/test_doc.md
```

## 📁 生成的文件结构

爬取后会生成如下目录结构：

```
data/crawled/
├── docs/
│   ├── doc_1.json              # JSON格式(供程序使用)
│   ├── doc_2.json
│   └── markdown/               # ✨ Markdown格式(供人阅读)
│       ├── doc_1.md
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

## 📝 Markdown文件格式

### 官方文档示例

```markdown
# 魔搭社区快速入门
---
**URL**: https://www.modelscope.cn/docs/quickstart
**来源**: modelscope_docs
---

## 内容

文档正文内容...

## 代码示例

### 示例 1
\`\`\`
pip install modelscope
\`\`\`
```

### 研习社文章示例

```markdown
# 如何使用ModelScope进行模型训练
---
**URL**: https://modelscope.cn/learn/article/123
**作者**: 张三
**日期**: 2025-12-02
**标签**: 机器学习, 模型训练
---

## 内容

文章正文...
```

### GitHub仓库示例

```markdown
# modelscope
---
**URL**: https://github.com/modelscope/modelscope
**语言**: Python
**Stars**: ⭐ 5678
**Forks**: 🍴 1234
---

## 描述

仓库描述...

## README

README内容...
```

## 🔍 常用操作

### 查看生成的Markdown文件

```bash
# 列出所有Markdown文件
ls data/crawled/docs/markdown/
ls data/crawled/learn/markdown/
ls data/crawled/github/markdown/
ls data/crawled/catalog/markdown/

# 查看文件内容
cat data/crawled/docs/markdown/doc_1.md
```

### 搜索Markdown内容

```bash
# 使用grep搜索
grep -r "模型训练" data/crawled/*/markdown/

# 使用ripgrep (更快)
rg "模型训练" data/crawled/*/markdown/

# 搜索特定文件类型
find data/crawled/*/markdown -name "*.md" -exec grep -l "ModelScope" {} \;
```

### 统计信息

```bash
# 统计Markdown文件数量
find data/crawled/*/markdown -name "*.md" | wc -l

# 统计总字数
find data/crawled/*/markdown -name "*.md" -exec wc -w {} + | tail -1

# 按目录统计
for dir in data/crawled/*/markdown; do
    count=$(find "$dir" -name "*.md" 2>/dev/null | wc -l)
    echo "$(basename $(dirname $dir)): $count 个文件"
done
```

### 导出为其他格式

```bash
# 使用pandoc转换为PDF
pandoc data/crawled/docs/markdown/doc_1.md -o doc_1.pdf

# 转换为HTML
pandoc data/crawled/docs/markdown/doc_1.md -o doc_1.html

# 批量转换
for f in data/crawled/docs/markdown/*.md; do
    pandoc "$f" -o "${f%.md}.html"
done
```

## 💡 使用场景

### 1. 快速浏览内容
```bash
# 在终端中直接查看
cat data/crawled/docs/markdown/doc_1.md

# 使用less分页查看
less data/crawled/docs/markdown/doc_1.md

# 使用bat高亮显示
bat data/crawled/docs/markdown/doc_1.md
```

### 2. 编辑修正
```bash
# 使用编辑器打开
vim data/crawled/docs/markdown/doc_1.md
code data/crawled/docs/markdown/  # VS Code
```

### 3. 版本控制
```bash
# 将Markdown文件加入Git
git add data/crawled/*/markdown/
git commit -m "Add crawled markdown documents"
```

### 4. 团队协作
- 分享Markdown文件给团队成员查看
- 在GitHub/GitLab上直接预览
- 转换为PDF分发

## 📊 示例输出

运行测试后的输出：

```bash
$ python scripts/test_markdown_export.py

======================================================================
测试Markdown导出功能
======================================================================

📝 测试1: 文档类型数据
📝 已保存Markdown: data/test_markdown/markdown/test_doc.md
✅ 文档类型测试完成

📝 测试2: 文章类型数据
📝 已保存Markdown: data/test_markdown/markdown/test_article.md
✅ 文章类型测试完成

📝 测试3: GitHub仓库类型数据
📝 已保存Markdown: data/test_markdown/markdown/test_repo.md
✅ GitHub仓库类型测试完成

📝 测试4: 目录项类型数据
📝 已保存Markdown: data/test_markdown/markdown/test_catalog.md
✅ 目录项类型测试完成

======================================================================
📊 测试结果
======================================================================
✅ 生成了 4 个Markdown文件:
   - test_repo.md (355 字节)
   - test_doc.md (377 字节)
   - test_article.md (420 字节)
   - test_catalog.md (279 字节)

✅ 所有测试完成!
📁 Markdown文件保存在: data/test_markdown/markdown
```

## ❓ 常见问题

### Q1: Markdown文件在哪里？
**A**: 在每个数据源目录下的`markdown/`子文件夹中：
- `data/crawled/docs/markdown/`
- `data/crawled/learn/markdown/`
- `data/crawled/github/markdown/`
- `data/crawled/catalog/markdown/`

### Q2: 如何只生成Markdown不生成JSON？
**A**: 当前两种格式都会生成。如果只需要Markdown，可以在爬取后删除JSON文件：
```bash
find data/crawled -name "*.json" -not -name "summary.json" -delete
```

### Q3: Markdown文件可以编辑吗？
**A**: 可以！Markdown是纯文本格式，可以用任何文本编辑器编辑。

### Q4: 如何分享Markdown文件？
**A**:
- 直接分享.md文件
- 转换为PDF: `pandoc file.md -o file.pdf`
- 上传到GitHub/GitLab查看
- 复制粘贴到其他Markdown编辑器

### Q5: 占用多少存储空间？
**A**: Markdown文件通常比JSON文件略小，因为格式更紧凑。具体大小取决于内容长度。

## 📚 相关文档

- [MARKDOWN_EXPORT.md](docs/MARKDOWN_EXPORT.md) - Markdown导出功能详细文档
- [README_CRAWLING.md](README_CRAWLING.md) - 爬虫系统完整指南
- [DATA_CRAWLING.md](docs/DATA_CRAWLING.md) - 技术文档

## 🎉 开始使用

最简单的开始方式：

```bash
# 1. 运行快速启动脚本
./START_CRAWLING.sh

# 2. 选择选项6运行测试

# 3. 查看生成的示例Markdown文件
cat data/test_markdown/markdown/test_doc.md
```

就是这么简单！🚀

---

**最后更新**: 2025-12-02
