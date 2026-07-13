# LangGraph Fault Tolerance 参考案例

本目录基于文档：

`study-docs/langgraph-docs/34_LangGraph_Fault_Tolerance_详细指南.md`

提供几个不依赖真实 LLM 的参考案例，用于理解 LangGraph Fault Tolerance 中的 retries、custom retry、timeouts、error handlers、graph defaults 和 graceful shutdown。

## 安装依赖

```bash
pip install -r requirements.txt
```

> 注意：per-node `timeout` 和 node-level `error_handler` 需要 `langgraph>=1.2`。

## 案例列表

| 文件 | 主题 | 说明 |
|------|------|------|
| `01_retry_policy_transient_error.py` | RetryPolicy | 节点第一次失败、第二次成功，展示自动重试 |
| `02_custom_retry_logic.py` | Custom retry | 用 `retry_on` callable 区分 transient 和 business error |
| `03_async_timeout_policy.py` | TimeoutPolicy | async 节点配置 `run_timeout`，超时后进入 retry |
| `04_error_handler_command_route.py` | Error handler | retry 耗尽后用 handler 返回 `Command` 做补偿路由 |
| `05_graph_defaults_precedence.py` | Graph defaults | 用 `set_node_defaults()` 配默认 retry/handler，并展示节点级覆盖 |
| `06_graceful_shutdown_pattern.py` | Graceful shutdown | 使用 `RunControl` / `GraphDrained` 的合作式停机模板 |

## 运行方式

```bash
python 01_retry_policy_transient_error.py
python 02_custom_retry_logic.py
python 03_async_timeout_policy.py
python 04_error_handler_command_route.py
python 05_graph_defaults_precedence.py
python 06_graceful_shutdown_pattern.py
```

## 核心概念速记

| 概念 | 含义 |
|------|------|
| `RetryPolicy` | 节点失败后按策略重试 |
| `retry_on` | 判断哪些异常应该重试 |
| `TimeoutPolicy` | 限制单次 node attempt 的运行时间或空闲时间 |
| `NodeTimeoutError` | 节点超时时抛出的错误，默认可重试 |
| `error_handler` | retries 耗尽后执行的补偿逻辑 |
| `NodeError` | handler 中可读取失败节点和原始异常 |
| `Command(update=..., goto=...)` | handler 中同时更新 state 并指定下一跳 |
| `set_node_defaults()` | 为多个节点配置默认 retry/timeout/handler |
| `RunControl.request_drain()` | 请求 graph 在 superstep 边界安全停止 |

这些案例使用内存 checkpointer，适合学习 fault tolerance 机制。生产环境需要结合持久化 checkpointer、幂等副作用、重试预算和可观测性。
