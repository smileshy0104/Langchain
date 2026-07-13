# LangGraph Memory 参考案例

本目录基于文档：

`study-docs/langgraph-docs/38_LangGraph_Memory_详细指南.md`

提供几个不依赖真实 LLM 的参考案例，用于理解 LangGraph Memory 中的短期记忆、长期记忆、`thread_id` 与 `user_id` 区分、Store namespace、语义搜索、消息裁剪/删除/总结，以及 checkpoint 管理。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 案例列表

| 文件 | 主题 | 说明 |
|------|------|------|
| `01_short_term_memory_checkpointer.py` | 短期记忆 | 用 `InMemorySaver` 和同一个 `thread_id` 保留多轮消息 |
| `02_long_term_store_namespace.py` | 长期记忆 | 用 `InMemoryStore`、namespace、key/value 保存跨会话资料 |
| `03_thread_id_vs_user_id.py` | 组合模式 | `thread_id` 管短期状态，`user_id` 管长期 Store namespace |
| `04_access_store_inside_node.py` | Runtime Store | 在 node 中通过 `runtime.context` 和 `runtime.store` 读写长期记忆 |
| `05_semantic_search_index_shape.py` | 语义搜索 | 用轻量 fake embedding 展示 Store semantic search 配置形状 |
| `06_trim_delete_summarize_messages.py` | 短期记忆管理 | 对比 trim、delete、summary 三种处理长消息历史的方式 |
| `07_manage_checkpoints.py` | Checkpoint 管理 | 查看当前 state、历史 checkpoint，并删除 thread checkpoints |

## 运行方式

```bash
python 01_short_term_memory_checkpointer.py
python 02_long_term_store_namespace.py
python 03_thread_id_vs_user_id.py
python 04_access_store_inside_node.py
python 05_semantic_search_index_shape.py
python 06_trim_delete_summarize_messages.py
python 07_manage_checkpoints.py
```

## 核心概念速记

| 概念 | 含义 |
|------|------|
| Short-term memory | 当前 thread 内的 graph state / messages，由 checkpointer 保存 |
| Long-term memory | 跨 thread 的用户或应用资料，由 Store 保存 |
| `thread_id` | 定位一个会话或工作流实例，用于读取 checkpoints |
| `user_id` | 定位一个用户，常用于构造长期记忆 namespace |
| `Store namespace` | 长期记忆的数据分组，适合做用户、租户、组织、项目隔离 |
| `runtime.store` | node 内访问长期记忆 Store 的入口 |
| `trim_messages` | 只裁剪传给模型的输入，不一定修改 graph state |
| `RemoveMessage` | 从 `MessagesState` 中永久删除消息 |
| Summary | 把早期消息压缩到摘要字段，保留信息但减少上下文长度 |
| `delete_thread` | 删除某个 thread 的短期 checkpoints，不会自动删除 Store 长期记忆 |

这些示例使用内存实现，适合学习和测试。生产环境应使用 Postgres、Redis、MongoDB、Oracle 等持久化实现，并把 `setup()` / migration 放到部署或启动流程中。
