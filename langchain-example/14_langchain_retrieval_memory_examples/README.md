# LangChain Retrieval & Memory 实战案例

本目录对应 `langchain-docs/LangChain_Retrieval_Memory_详细指南.md` 的实战案例。

示例使用：

- `Milvus`：本地 Docker 向量数据库，默认连接 `http://localhost:19530`
- `HashEmbeddings`：本地确定性 embedding，用于演示，不调用外部 embedding API
- `InMemoryStore`：长期记忆
- `InMemorySaver`：短期对话状态
- 自定义 OpenAI 兼容聊天模型：默认读取 `../langchain_context_engineering_examples/.env`

## 环境变量

```bash
LANGCHAIN_MODEL_PROVIDER=custom
CUSTOM_MODEL_BASE_URL=https://ai-api.kkidc.com
CUSTOM_MODEL_API_KEY=你的 API Key
LANGCHAIN_DEFAULT_MODEL=gpt-5.5
MILVUS_URI=http://localhost:19530
```

## 运行

如果当前环境还没有 Milvus Python 依赖：

```bash
python -m pip install -r requirements.txt
```

确认本地 Docker Milvus 已启动：

```bash
docker ps --format '{{.Names}}\t{{.Ports}}' | grep milvus
```

```bash
cd /Users/aiyer/Applications/GolandProjects/Langchain/langchain-example/langchain_retrieval_memory_examples
conda activate langchain-env

python 01_customer_service_system.py
python 02_personalized_learning_assistant.py
python 03_memory_enhanced_rag.py
```

## 文件说明

- `01_customer_service_system.py`：智能客服系统，结合客户长期记忆、购买历史和 Milvus FAQ 检索。
- `02_personalized_learning_assistant.py`：个性化学习助手，保存学习目标、进度和笔记。
- `03_memory_enhanced_rag.py`：带记忆的 RAG，按用户偏好检索和生成回答。
- `model_config.py`：统一模型配置。
- `retrieval_utils.py`：本地 hash embedding 与 Milvus 初始化工具。
