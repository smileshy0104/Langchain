# 魔搭社区智能答疑 Agent

基于 LangChain v1.0 + Milvus + Qwen 构建的魔搭社区智能技术支持 Agent。

## 项目概述

本项目旨在为魔搭社区开发者提供智能化的技术问答服务，采用 RAG (Retrieval-Augmented Generation) 架构，结合向量数据库和大语言模型，实现准确、专业、可信的技术支持。

## 核心功能

- **技术问题快速解答**: 单轮问答，提供问题分析、解决方案和代码示例
- **多轮对话深度排查**: 支持复杂问题的连续对话和上下文理解
- **平台功能导航**: 介绍魔搭社区功能、推荐模型和数据集
- **项目级开发指导**: 提供架构设计、技术选型和最佳实践建议

## 技术栈

- **Agent 框架**: LangGraph (LangChain v1.0)
- **向量数据库**: Milvus 2.3+
- **嵌入模型**: 通义千问 Embedding (DashScope API)
- **大语言模型**: Qwen-2.5-72B-Instruct
- **检索策略**: 混合检索 (向量检索 + BM25)
- **对话管理**: MemorySaver Checkpointer + 滑动窗口摘要
- **缓存优化**: Redis

## 项目结构

```
modelscope_qa_agent/
├── agents/          # Agent 工作流定义
├── core/            # 核心模块 (LLM, 向量存储, 文档处理)
├── models/          # Pydantic 数据模型
├── tools/           # Agent 工具集
├── retrievers/      # 检索器实现
├── prompts/         # Prompt 模板
├── data/            # 数据加载和存储
│   └── loaders/     # 数据加载器
├── scripts/         # 工具脚本
├── tests/           # 单元测试和集成测试
└── config/          # 配置管理
```

## 快速开始

### 1. 环境准备

```bash
# 激活 conda 环境
conda activate langchain-env

# 安装依赖
pip install -r requirements.txt
```

### 2. 环境配置

复制 `.env.example` 到 `.env` 并配置必要的 API 密钥:

```bash
cp .env.example .env
```

编辑 `.env` 文件:

```env
# 通义千问 API
DASHSCOPE_API_KEY=your_dashscope_api_key

# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Redis 配置
REDIS_HOST=localhost
REDIS_PORT=6379

# LangSmith (可选)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
```

### 3. 基础设施验证

```bash
# 检查基础设施连接
python scripts/check_infrastructure.py
```

### 4. 运行 Agent

```python
from agents.qa_agent import ModelScopeQAAgent

# 初始化 Agent
agent = ModelScopeQAAgent()

# 单轮问答
response = agent.invoke("如何在魔搭社区使用 Qwen 模型?")
print(response)

# 多轮对话
thread_id = "user-123"
agent.invoke("模型加载时出现 CUDA 内存不足", thread_id=thread_id)
agent.invoke("batch_size 是 32", thread_id=thread_id)
```

## 开发指南

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_qa_agent.py::test_single_turn_qa

# 生成覆盖率报告
pytest --cov=modelscope_qa_agent tests/
```

### 代码规范

- 遵循 PEP 8 编码规范
- 使用 type hints
- 编写清晰的 docstrings
- 保持函数简洁 (单一职责原则)

### 性能目标

- 单轮问答响应时间 < 30 秒
- 检索准确率 ≥ 85%
- 幻觉率 < 5%
- 用户满意度 ≥ 4.0/5.0

## 评估与监控

### RAG 评估

使用 RAGAs 框架评估检索和生成质量:

```bash
python scripts/evaluate_rag.py
```

### LangSmith 监控

配置 LangSmith 后，所有 LLM 调用会自动追踪:

- Token 使用量
- 延迟分布
- 错误率
- 用户反馈评分

## 贡献指南

欢迎贡献! 请遵循以下步骤:

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

[待定]

## 联系方式

如有问题或建议，请提交 Issue。
