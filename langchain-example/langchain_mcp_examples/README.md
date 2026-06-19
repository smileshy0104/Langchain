# LangChain MCP 示例集

本目录用于存放 `langchain-docs/LangChain_MCP_详细指南.md` 中“实战案例”的可运行版本。

## 目录

- [01_aggregate_math_weather.py](01_aggregate_math_weather.py) - 案例 1：聚合数学与天气服务
- [02_context_injection_user_info.py](02_context_injection_user_info.py) - 案例 2：基于上下文注入用户信息
- [03_sensitive_operation_approval.py](03_sensitive_operation_approval.py) - 案例 3：敏感操作审批
- [servers/math_server.py](servers/math_server.py) - 本地数学 MCP Server
- [servers/weather_server.py](servers/weather_server.py) - 本地天气 MCP Server
- [servers/orders_server.py](servers/orders_server.py) - 本地订单 MCP Server
- [servers/operations_server.py](servers/operations_server.py) - 本地敏感操作 MCP Server

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置模型

可在本目录创建 `.env`，复用前面示例的自定义供应商配置：

```bash
LANGCHAIN_MODEL_PROVIDER=custom
LANGCHAIN_DEFAULT_MODEL=gpt-5.5
LANGCHAIN_DEFAULT_TEMPERATURE=0.7
CUSTOM_MODEL_API_KEY=your-api-key
CUSTOM_MODEL_BASE_URL=https://ai-api.kkidc.com
```

脚本会在自定义供应商 URL 未以 `/v1` 结尾时自动补成 `https://ai-api.kkidc.com/v1`。

## 运行案例

### 案例 1：聚合数学与天气服务

```bash
python 01_aggregate_math_weather.py \
  --question "What is the weather in NYC? Also calculate (3 + 5) * 12."
```

### 案例 2：基于上下文注入用户信息

```bash
python 02_context_injection_user_info.py \
  --user-id user_123 \
  --question "Search my orders"
```

### 案例 3：敏感操作审批

未批准时，敏感工具会被拦截：

```bash
python 03_sensitive_operation_approval.py
```

批准 `send_email` 后再运行：

```bash
python 03_sensitive_operation_approval.py --approved-tools send_email
```

## 说明

- 示例使用本地 `stdio` MCP Server，运行客户端脚本时会自动启动对应 server 进程。
- 案例 2 展示 `tool_interceptors` 如何把 LangChain `runtime.context` 注入 MCP 工具参数。
- 案例 3 为了命令行可运行，使用 `--approved-tools` 模拟人工审批；接入 UI 时可替换为 LangGraph interrupt 流程。
