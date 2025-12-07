# LangChain Streaming 示例集合

> 基于官方文档：https://docs.langchain.com/oss/python/langchain/streaming
>
> 使用 GLM-4.5-air 模型实现

## 📖 项目简介

本项目提供完整的 LangChain Streaming（流式传输）功能示例代码，涵盖 Model、Agent 和 LangGraph 的流式传输用法。

**核心优势**：
- 🚀 实时反馈 - 用户立即看到进展
- ⚡ 改善感知性能 - 流式传输让应用感觉更快
- 📊 进度可视化 - 显示中间步骤和状态
- 🎯 更好的 UX - 特别适合长响应

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

## 🗂️ 项目结构

```
langchain_streaming_examples/
├── 01_basic_model_streaming.py     # Model 流式传输（6个示例）
├── 02_agent_streaming.py           # Agent 流式传输（5个示例）
├── 03_langgraph_streaming.py       # LangGraph 流式传输（5个示例）
├── 04_real_world_examples.py       # 实际应用场景（5个场景）
├── README.md                         # 本文件
├── QUICK_REFERENCE.md               # 快速参考
└── requirements.txt                 # 依赖列表
```

## 📚 示例说明

### 1️⃣ Model 流式传输 ([01_basic_model_streaming.py](01_basic_model_streaming.py))

**示例**：
- ✅ 基础 Token 流式输出
- ✅ 累积消息块
- ✅ 流式输出带元数据
- ✅ 实时打字效果
- ✅ 流式 vs 非流式对比
- ✅ 处理流式中断

**运行**：
```bash
python 01_basic_model_streaming.py
```

### 2️⃣ Agent 流式传输 ([02_agent_streaming.py](02_agent_streaming.py))

**示例**：
- ✅ Agent 基础流式传输
- ✅ 流式工具调用
- ✅ 多步骤流式传输
- ✅ Stream Modes (values/updates)
- ✅ 自动流式传输

### 3️⃣ LangGraph 流式传输 ([03_langgraph_streaming.py](03_langgraph_streaming.py))

**示例**：
- ✅ Values Mode - 完整状态
- ✅ Updates Mode - 增量更新
- ✅ Messages Mode - Token 流
- ✅ Custom Mode - 自定义数据
- ✅ Debug Mode - 调试信息

### 4️⃣ 实际应用 ([04_real_world_examples.py](04_real_world_examples.py))

**场景**：
- 📝 实时内容生成
- 💬 聊天机器人
- 📊 数据分析报告
- 🔍 搜索助手
- 📖 文档总结

## 🚀 快速开始

1. **设置 API Key**：
```bash
export ZHIPUAI_API_KEY="your-api-key"
```

2. **运行示例**：
```bash
python 01_basic_model_streaming.py
```

## 💡 核心概念

### Stream Modes

| 模式 | 描述 | 用途 |
|------|------|------|
| `values` | 每步后的完整状态 | 查看完整的图状态 |
| `updates` | 每步的状态更新 | 只看变化部分 |
| `messages` | LLM token 流 + 元数据 | 流式显示 LLM 输出 |
| `custom` | 自定义用户数据 | 进度更新、日志等 |
| `debug` | 详细的执行信息 | 调试和故障排除 |

### 基础用法

```python
from langchain_community.chat_models import ChatZhipuAI

model = ChatZhipuAI(model="glm-4.5-air")

# 流式输出
for chunk in model.stream("你好"):
    print(chunk.content, end="", flush=True)
```

## 📖 相关文档

- [LangChain 官方文档](https://docs.langchain.com/oss/python/langchain/streaming)
- [LangGraph Streaming](https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/)

---

**版本**: v1.0
**创建日期**: 2024-11-30
