# LangChain 示例项目

## 📋 项目说明

这是一个 LangChain 示例项目，演示如何使用配置文件管理 API Key 和模型配置。

## 🚀 快速开始

### 1. 安装依赖

```bash
# 激活 conda 环境
conda activate langchain-env

# 安装必要的包
pip install python-dotenv
```

### 2. 配置 API Key

#### 方法 1: 复制示例文件（推荐）

```bash
# 复制 .env.example 为 .env
cp .env.example .env

# 编辑 .env 文件，填写您的 API Key
vim .env  # 或使用其他编辑器
```

#### 方法 2: 手动创建 .env 文件

创建 `.env` 文件并添加以下内容：

```bash
# 智谱 AI API Key
ZHIPUAI_API_KEY=your-actual-api-key-here

# 模型配置（可选）
ZHIPUAI_MODEL=glm-4-flash
ZHIPUAI_TEMPERATURE=0.7
```

### 3. 运行示例

#### 在 Jupyter Notebook 中运行

```bash
jupyter notebook langchain01.ipynb
```

#### 在 Python 脚本中运行

```bash
python -c "from config_loader import load_config; config = load_config(); print(config.get_model_config('zhipuai'))"
```

## 📁 文件说明

```
langchain/
├── .env.example          # 环境变量示例文件
├── .env                  # 实际配置文件（不会提交到 git）
├── config_loader.py      # 配置加载器
├── langchain01.ipynb     # Jupyter Notebook 示例
└── README.md             # 本文件
```

## 🔑 配置项说明

### 必需配置

| 配置项 | 说明 | 示例 |
|--------|------|------|
| `ZHIPUAI_API_KEY` | 智谱 AI 的 API Key | `your-api-key-here` |

### 可选配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `ZHIPUAI_MODEL` | 使用的模型名称 | `glm-4-flash` |
| `ZHIPUAI_TEMPERATURE` | 模型温度参数 | `0.7` |
| `DEFAULT_PROVIDER` | 默认提供商 | `zhipuai` |
| `REQUEST_TIMEOUT` | 请求超时时间（秒） | `30` |
| `RETRY_TIMES` | 重试次数 | `3` |

## 💡 使用示例

### 1. 基础使用

```python
from config_loader import load_config

# 加载配置
config = load_config()

# 获取智谱 AI 配置
zhipuai_config = config.get_model_config('zhipuai')

# 使用配置创建模型
from langchain_community.chat_models import ChatZhipuAI
model = ChatZhipuAI(**zhipuai_config)
```

### 2. 便捷函数

```python
from config_loader import get_zhipuai_config

# 直接获取配置
config = get_zhipuai_config()
print(config)
# 输出: {'api_key': 'xxx', 'model': 'glm-4-flash', 'temperature': 0.7}
```

### 3. 获取单个配置项

```python
from config_loader import load_config

config = load_config()

# 获取 API Key
api_key = config.get_api_key('zhipuai')

# 获取通用设置
timeout = config.get_setting('REQUEST_TIMEOUT', '30')
```

## 🔒 安全建议

1. **不要提交 .env 文件到 Git**
   - `.env` 文件已在 `.gitignore` 中
   - 只提交 `.env.example` 作为模板

2. **定期更换 API Key**
   - 建议定期更换 API Key
   - 不要在代码中硬编码 API Key

3. **使用环境变量**
   - 生产环境建议使用系统环境变量
   - 开发环境使用 `.env` 文件

## ❓ 常见问题

### Q1: 提示 ".env 文件不存在"

**A:** 请按照以下步骤操作：

```bash
# 1. 复制示例文件
cp .env.example .env

# 2. 编辑 .env 文件
vim .env

# 3. 填写您的 API Key
ZHIPUAI_API_KEY=your-actual-api-key
```

### Q2: 提示 "API Key 未配置"

**A:** 检查 `.env` 文件中的 API Key 是否正确：

```bash
# 查看 .env 文件内容
cat .env

# 确保 API Key 不是默认值
ZHIPUAI_API_KEY=your-actual-api-key  # ❌ 错误
ZHIPUAI_API_KEY=abc123...xyz         # ✅ 正确
```

### Q3: 如何添加其他提供商的配置？

**A:** 在 `.env` 文件中添加相应的配置：

```bash
# OpenAI
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-4o-mini

# Anthropic
ANTHROPIC_API_KEY=your-anthropic-key
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

## 📚 相关文档

- [LangChain 官方文档](https://python.langchain.com/)
- [智谱 AI 文档](https://open.bigmodel.cn/)
- [python-dotenv 文档](https://github.com/theskumar/python-dotenv)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License
