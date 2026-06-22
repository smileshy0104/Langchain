# LangSmith Evaluation 示例

本目录对应 LangSmith Evaluation quickstart，演示如何用 Python SDK 创建/复用数据集、定义 target function、定义 code evaluator，并通过 `evaluate()` 运行一次离线评测实验。默认 target 会使用本目录的 `model_config.py` 调用真实 LangChain 大模型；设置 `LANGSMITH_DEMO_TARGET=stub` 可只跑本地链路验证。

## 核心概念

- `Dataset`: 测试输入和期望输出集合。
- `Target function`: 被评测的应用逻辑，接收一条样本的 `inputs`，返回应用输出。
- `Evaluator`: 评分函数，用 `outputs` 和 `reference_outputs` 判断 target 的表现。
- `Experiment`: 一次 target 在 dataset 上的评测结果。

## 安装依赖

```bash
python -m pip install -r requirements.txt
```

或直接安装:

```bash
python -m pip install -U langsmith langchain langchain-openai python-dotenv
```

## 配置环境变量

本目录已提供本地 `.env` 文件，脚本会自动读取。`.env` 已被 `.gitignore` 忽略，避免误提交密钥。

你也可以在终端里覆盖这些变量:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
export LANGSMITH_API_KEY="<your-langsmith-api-key>"
export LANGSMITH_DEMO_TARGET="langchain"

export LANGCHAIN_MODEL_PROVIDER=custom
export CUSTOM_MODEL_BASE_URL=https://ai-api.kkidc.com
export CUSTOM_MODEL_API_KEY="<your-custom-model-api-key>"
export LANGCHAIN_DEFAULT_MODEL=gpt-5.5
```

`LANGSMITH_DEMO_TARGET=langchain` 会真实调用本目录 `model_config.py` 配置的大模型；`LANGSMITH_DEMO_TARGET=stub` 不会调用模型。

## 运行

```bash
cd /Users/aiyer/Applications/GolandProjects/Langchain/langchain-example/15_langsmith_evaluation_examples
python 01_langsmith_evaluation_demo.py
```

真实调用模型:

```bash
LANGSMITH_DEMO_TARGET=langchain python 01_langsmith_evaluation_demo.py
```

脚本会:

1. 创建或复用名为 `Test_Demo` 的 LangSmith 数据集。
2. 如果数据集为空，导入两条示例样本。
3. 默认运行 `langchain_target()`，使用本目录 `model_config.py` 的 `create_configured_model()`；设置 `LANGSMITH_DEMO_TARGET=stub` 时运行 `local_stub_target()`。
4. 使用 `custom_evaluator()`、`exact_match()` 和 `contains_required_answer()` 评测最终回答。
5. 创建前缀为 `Test_Demo Real Evaluation` 的 experiment。

## 关于评分结果

- `exact_match=false`: 真实 LLM 回答通常不会和 reference answer 字符串完全一致，这是正常的。
- `custom_evaluator=true`: 只评估最终回答，不评估 `rag_trace` 或检索块；有参考答案时使用轻量字符重合率检查。
- `contains_required_answer=true`: 只检查回答是否包含 reference 里要求出现的关键词，更适合这个入门 demo。
- 如果使用 `langchain_openai.ChatOpenAI` 遇到 `TypeError: 'NoneType' object is not iterable`，可在对应 `ChatOpenAI(...)` 初始化处设置 `use_responses_api=False`，避免走 Responses API 解析路径。

## Token 统计

- `LANGSMITH_DEMO_TARGET=stub` 不调用模型，因此没有真实 token usage。
- `LANGSMITH_DEMO_TARGET=langchain` 会从 LangChain AIMessage 的 `usage_metadata` 或 `response_metadata.token_usage` 中提取 token，并写入顶层 `usage_metadata`。
- 如果 OpenAI 兼容代理没有返回 usage 字段，脚本会打印提示，此时 LangSmith 无法得到真实 token 统计。

## 文件说明

- `01_langsmith_evaluation_demo.py`: 完整的 LangSmith evaluation 调用 demo。
- `model_config.py`: 本目录独立模型配置，风格参考 Retrieval & Memory 示例。
- `requirements.txt`: 最小依赖。
