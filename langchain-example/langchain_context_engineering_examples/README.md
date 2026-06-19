# LangChain Context Engineering 示例集

本目录用于存放 `langchain-docs/LangChain_Context_Engineering_详细指南.md` 对应的可运行案例。

## 目录

- [01_customer_service_system.py](01_customer_service_system.py) - 案例 1：智能客服系统
- [02_adaptive_learning_assistant.py](02_adaptive_learning_assistant.py) - 案例 2：自适应学习助手

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置模型

先配置环境变量，或在本目录创建 `.env`：

```bash
LANGCHAIN_MODEL_PROVIDER=zhipuai
LANGCHAIN_DEFAULT_MODEL=glm-4-flash
LANGCHAIN_ENTERPRISE_MODEL=glm-4-flash
LANGCHAIN_DEFAULT_TEMPERATURE=0.7
LANGCHAIN_ENTERPRISE_TEMPERATURE=0.2
ZHIPUAI_API_KEY=your-zhipuai-api-key
```

也可以用 `init_chat_model` 支持的其他模型标识，例如：

```bash
LANGCHAIN_MODEL_PROVIDER=openai
LANGCHAIN_DEFAULT_MODEL=openai:gpt-4o-mini
LANGCHAIN_ENTERPRISE_MODEL=openai:gpt-4o
OPENAI_API_KEY=your-openai-api-key
```

如果你的模型服务兼容 OpenAI Chat Completions，可以使用自定义供应商：

```bash
LANGCHAIN_MODEL_PROVIDER=custom
LANGCHAIN_DEFAULT_MODEL=gpt-5.5
LANGCHAIN_ENTERPRISE_MODEL=gpt-5.5
CUSTOM_MODEL_API_KEY=your-api-key
CUSTOM_MODEL_BASE_URL=https://ai-api.kkidc.com
```

脚本会在自定义供应商 URL 未以 `/v1` 结尾时自动补成 `https://ai-api.kkidc.com/v1`。

### 3. 运行案例

```bash
python 01_customer_service_system.py \
  --customer-id user003 \
  --customer-name 王五 \
  --subscription-tier enterprise \
  --question "请查一下我最近的订单，并帮我为延迟配送创建工单。"
```

运行案例 2：

```bash
python 02_adaptive_learning_assistant.py \
  --learner-id learner001 \
  --learner-name 小林 \
  --expertise-level beginner \
  --learning-style visual \
  --question "请用适合我的方式讲解 LangChain 的 Tool Context，并给我一个小练习。"
```

## 案例说明

### 01_customer_service_system.py

覆盖的上下文工程能力：

- `CustomerContext`：定义 Runtime Context，包含客户 ID、姓名、套餐和语言。
- `customer_service_prompt`：使用 `@dynamic_prompt` 根据客户上下文生成系统提示词。
- `tier_based_model`：使用 `@wrap_model_call` 按套餐选择模型配置。
- `support_output_format`：涉及工单时动态启用 `TicketSummary` 结构化输出。
- `get_customer_orders` / `create_support_ticket`：工具通过 `ToolRuntime[CustomerContext]` 读取客户上下文。
- `InMemorySaver` / `InMemoryStore`：分别承载 thread 内短期记忆和跨步骤工单存储。

### 02_adaptive_learning_assistant.py

覆盖的上下文工程能力：

- `LearnerContext`：定义 Runtime Context，包含学习者 ID、姓名、水平、学习风格和语言。
- `adaptive_learning_prompt`：使用 `@dynamic_prompt` 根据学习者画像调整讲解方式。
- `save_learning_progress`：工具通过 `ToolRuntime[LearnerContext]` 把学习进度写入 Store。
- `get_learning_history`：工具从 Store 读取同一学习者的历史掌握度。
- `InMemorySaver` / `InMemoryStore`：分别承载 thread 内短期记忆和跨步骤学习记录。
