#!/bin/bash
# 快速启动爬虫脚本

set -e

# 确保在正确的目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "魔搭社区数据爬取系统 - 快速启动"
echo "========================================================================"
echo ""
echo "📁 工作目录: $SCRIPT_DIR"
echo ""

# 激活conda环境
echo "📦 激活conda环境..."
source ~/.bash_profile
conda activate langchain-env

echo "✅ 环境已激活: langchain-env"
echo ""

# 显示菜单
echo "请选择爬取选项:"
echo ""
echo "1) 爬取所有数据源 (docs + learn + github + catalog)"
echo "2) 只爬取官方文档 (docs)"
echo "3) 只爬取研习社 (learn)"
echo "4) 只爬取GitHub仓库 (github)"
echo "5) 只爬取资源目录 (catalog)"
echo "6) 运行测试 (查看Markdown导出示例)"
echo "0) 退出"
echo ""

read -p "请输入选项 [0-6]: " choice

case $choice in
    1)
        echo ""
        echo "🌐 开始爬取所有数据源..."
        echo "========================================================================"
        python scripts/crawl_and_process.py --all --process
        echo ""
        echo "✅ 爬取完成!"
        echo "📁 Markdown文件保存在:"
        echo "   - data/crawled/docs/markdown/"
        echo "   - data/crawled/learn/markdown/"
        echo "   - data/crawled/github/markdown/"
        echo "   - data/crawled/catalog/markdown/"
        ;;
    2)
        echo ""
        echo "📄 开始爬取官方文档..."
        echo "========================================================================"
        python scripts/crawl_and_process.py --docs --process
        echo ""
        echo "✅ 爬取完成!"
        echo "📁 Markdown文件保存在: data/crawled/docs/markdown/"
        ;;
    3)
        echo ""
        echo "📝 开始爬取研习社..."
        echo "========================================================================"
        python scripts/crawl_and_process.py --learn --process
        echo ""
        echo "✅ 爬取完成!"
        echo "📁 Markdown文件保存在: data/crawled/learn/markdown/"
        ;;
    4)
        echo ""
        echo "📦 开始爬取GitHub仓库..."
        echo "========================================================================"
        python scripts/crawl_and_process.py --github --process
        echo ""
        echo "✅ 爬取完成!"
        echo "📁 Markdown文件保存在: data/crawled/github/markdown/"
        ;;
    5)
        echo ""
        echo "📚 开始爬取资源目录..."
        echo "========================================================================"
        python scripts/crawl_and_process.py --catalog --process
        echo ""
        echo "✅ 爬取完成!"
        echo "📁 Markdown文件保存在: data/crawled/catalog/markdown/"
        ;;
    6)
        echo ""
        echo "🧪 运行Markdown导出测试..."
        echo "========================================================================"
        python scripts/test_markdown_export.py
        echo ""
        echo "📁 测试文件保存在: data/test_markdown/markdown/"
        echo ""
        read -p "是否查看测试生成的Markdown文件? (y/n): " view
        if [[ "$view" == "y" || "$view" == "Y" ]]; then
            echo ""
            ls -lh data/test_markdown/markdown/
            echo ""
            echo "📄 示例文件内容:"
            echo "========================================================================"
            cat data/test_markdown/markdown/test_doc.md
            echo "========================================================================"
        fi
        ;;
    0)
        echo ""
        echo "👋 再见!"
        exit 0
        ;;
    *)
        echo ""
        echo "❌ 无效选项: $choice"
        exit 1
        ;;
esac

echo ""
echo "========================================================================"
echo "💡 提示:"
echo "   - 查看Markdown文件: ls data/crawled/*/markdown/"
echo "   - 查看文件内容: cat data/crawled/docs/markdown/doc_1.md"
echo "   - 搜索内容: grep -r '关键词' data/crawled/*/markdown/"
echo "========================================================================"
