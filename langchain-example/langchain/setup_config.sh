#!/bin/bash

# LangChain 配置设置脚本

echo "================================"
echo "LangChain 配置设置向导"
echo "================================"

# 检查 .env.example 是否存在
if [ ! -f ".env.example" ]; then
    echo "❌ 错误: .env.example 文件不存在"
    exit 1
fi

# 检查 .env 是否已存在
if [ -f ".env" ]; then
    echo ""
    echo "⚠️  .env 文件已存在"
    read -p "是否覆盖? (y/N): " overwrite
    if [ "$overwrite" != "y" ] && [ "$overwrite" != "Y" ]; then
        echo "取消操作"
        exit 0
    fi
fi

# 复制示例文件
echo ""
echo "📝 创建 .env 文件..."
cp .env.example .env

# 提示用户输入 API Key
echo ""
echo "请输入您的 API Key:"
echo ""

# 智谱 AI
read -p "智谱 AI API Key (按回车跳过): " zhipuai_key
if [ ! -z "$zhipuai_key" ]; then
    # macOS 使用 sed -i ''
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/ZHIPUAI_API_KEY=.*/ZHIPUAI_API_KEY=$zhipuai_key/" .env
    else
        sed -i "s/ZHIPUAI_API_KEY=.*/ZHIPUAI_API_KEY=$zhipuai_key/" .env
    fi
    echo "✅ 已设置智谱 AI API Key"
fi

# OpenAI
read -p "OpenAI API Key (按回车跳过): " openai_key
if [ ! -z "$openai_key" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$openai_key/" .env
    else
        sed -i "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$openai_key/" .env
    fi
    echo "✅ 已设置 OpenAI API Key"
fi

# Anthropic
read -p "Anthropic API Key (按回车跳过): " anthropic_key
if [ ! -z "$anthropic_key" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/ANTHROPIC_API_KEY=.*/ANTHROPIC_API_KEY=$anthropic_key/" .env
    else
        sed -i "s/ANTHROPIC_API_KEY=.*/ANTHROPIC_API_KEY=$anthropic_key/" .env
    fi
    echo "✅ 已设置 Anthropic API Key"
fi

echo ""
echo "================================"
echo "✅ 配置完成！"
echo "================================"
echo ""
echo "配置文件位置: $(pwd)/.env"
echo ""
echo "下一步:"
echo "1. 查看配置: cat .env"
echo "2. 测试配置: python config_loader.py"
echo "3. 运行示例: jupyter notebook langchain01.ipynb"
echo ""
