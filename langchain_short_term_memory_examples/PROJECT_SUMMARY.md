# 📦 LangChain 短期记忆示例项目总览

## 🎯 项目概述

这是一个完整的 LangChain 短期记忆（Short-Term Memory）学习和实践项目，包含从基础到高级的所有核心功能示例。

**基于**: LangChain v0.3+ / LangGraph v0.2+  
**模型**: 智谱 AI GLM-4.6  
**文档**: https://docs.langchain.com/oss/python/langchain/short-term-memory

## 📂 项目结构

```
langchain_short_term_memory_examples/
│
├── 📄 README.md                    # 项目介绍和快速开始
├── 📘 LEARNING_GUIDE.md            # 详细学习指南（4天计划）
├── 📋 QUICK_REFERENCE.md           # 快速参考卡片
├── 📋 PROJECT_SUMMARY.md           # 本文件
│
├── 🐍 代码示例（6个）
│   ├── 01_basic_memory.py          # 基础短期记忆
│   ├── 02_multi_thread.py          # 多线程会话管理
│   ├── 03_trim_messages.py         # 消息修剪
│   ├── 04_summarization.py         # 消息摘要
│   ├── 05_custom_state.py          # 自定义状态
│   └── 06_tool_state_access.py     # 工具读写状态
│
├── 🛠️ 工具脚本
│   └── run_all_examples.py         # 批量运行示例
│
└── 📦 配置文件
    ├── requirements.txt            # Python 依赖
    └── .gitignore                  # Git 忽略规则
```

## ✨ 核心功能

### 1. 基础功能
- ✅ 启用短期记忆（InMemorySaver）
- ✅ 多用户会话隔离（Thread ID）
- ✅ 对话历史管理
- ✅ 状态查询和更新

### 2. 消息管理
- ✅ 消息修剪（Trim Messages）
- ✅ 消息删除（Remove Messages）
- ✅ 消息摘要（Summarization）
- ✅ 中间件机制

### 3. 高级功能
- ✅ 自定义状态（Custom State）
- ✅ 工具状态读取（Tool Runtime）
- ✅ 工具状态写入（Command Update）
- ✅ 动态提示词

### 4. 生产特性
- ✅ PostgreSQL 持久化
- ✅ 错误处理示例
- ✅ 性能优化建议
- ✅ 最佳实践指南

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆或下载项目
cd langchain_short_term_memory_examples

# 安装依赖
pip install -r requirements.txt

# 设置 API Key
export ZHIPUAI_API_KEY="your-api-key"
```

### 2. 运行示例

**方式1：单独运行**
```bash
python 01_basic_memory.py
python 02_multi_thread.py
# ...
```

**方式2：批量运行**
```bash
python run_all_examples.py
# 然后选择：
# 1 - 运行所有示例
# 2 - 选择单个示例
# 3 - 退出
```

### 3. 学习路径

1. **第1天**：阅读 README.md + 运行示例 1-2
2. **第2天**：运行示例 3-4，学习消息管理
3. **第3天**：运行示例 5-6，掌握高级功能
4. **第4天**：查看 LEARNING_GUIDE.md，完成实战项目

## 📚 文档说明

### README.md
- 项目简介和安装
- 每个示例的详细说明
- 常见问题解答
- 快速参考代码

### LEARNING_GUIDE.md
- 4天学习计划
- 每天的学习目标
- 实验和练习题
- 进阶方向建议

### QUICK_REFERENCE.md
- 常用代码片段
- API 快速查找
- 配置选项速查
- 调试技巧

## 🎓 学习目标

完成本项目后，你将能够：

- ✅ 理解短期记忆的核心概念
- ✅ 实现多用户对话系统
- ✅ 设计高效的消息管理策略
- ✅ 自定义 Agent 状态结构
- ✅ 集成工具和状态管理
- ✅ 优化性能和控制成本
- ✅ 部署到生产环境

## 📊 示例对比

| 示例 | 代码行数 | 难度 | 时间 | 依赖示例 |
|------|---------|------|------|----------|
| 01_basic_memory | ~90 | ⭐ | 30分钟 | 无 |
| 02_multi_thread | ~120 | ⭐⭐ | 30分钟 | 示例1 |
| 03_trim_messages | ~150 | ⭐⭐⭐ | 45分钟 | 示例1 |
| 04_summarization | ~180 | ⭐⭐⭐⭐ | 60分钟 | 示例3 |
| 05_custom_state | ~200 | ⭐⭐⭐ | 45分钟 | 示例1 |
| 06_tool_state_access | ~230 | ⭐⭐⭐⭐ | 60分钟 | 示例5 |

## 🔧 技术栈

- **核心**: LangChain, LangGraph
- **LLM**: 智谱 AI GLM-4.6
- **存储**: MemorySaver (开发) / PostgresSaver (生产)
- **Python**: 3.9+
- **依赖**: 见 requirements.txt

## 💡 使用建议

### 对于初学者
1. 按顺序学习示例 1-6
2. 先理解概念，再运行代码
3. 完成 LEARNING_GUIDE.md 中的练习

### 对于进阶者
1. 重点关注示例 4-6
2. 研究中间件机制
3. 尝试自定义状态结构
4. 优化性能和成本

### 对于生产使用
1. 替换为 PostgresSaver
2. 添加完整错误处理
3. 实现监控和日志
4. 参考最佳实践

## 🐛 故障排除

### 常见问题

**问题1**: 记忆不工作  
**解决**: 确保添加了 `checkpointer` 和 `thread_id`

**问题2**: API Key 错误  
**解决**: `export ZHIPUAI_API_KEY="your-key"`

**问题3**: 依赖安装失败  
**解决**: 使用虚拟环境 `python -m venv venv`

**更多问题**: 查看 README.md 的"常见问题"部分

## 📈 项目统计

- **示例代码**: 6 个完整示例
- **代码行数**: ~1000+ 行（含注释）
- **文档页数**: 3 个详细文档
- **学习时间**: 建议 4 天
- **难度级别**: 入门到进阶

## 🔗 相关资源

- **官方文档**: https://docs.langchain.com/oss/python/langchain/short-term-memory
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **智谱 AI**: https://open.bigmodel.cn/
- **GitHub**: https://github.com/langchain-ai/langchain

## 🤝 贡献

欢迎提交：
- Bug 报告
- 功能建议
- 文档改进
- 新示例代码

## 📄 许可

MIT License

---

**版本**: 1.0.0  
**创建日期**: 2024-11-29  
**维护者**: LangChain 学习者社区  

**Happy Learning! 🚀**
